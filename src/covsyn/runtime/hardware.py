# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detect the performance cores and choose how many worker processes to start (Task 2).

``os.cpu_count()`` counts efficiency cores and hardware threads, so a pool sized by it puts
slow cores and shared threads to work (finding D3). This counts the cores worth a worker:

- macOS: ``hw.perflevel0.physicalcpu``, or ``hw.physicalcpu`` on a Mac without performance
  levels (Intel);
- Linux with a hybrid Intel processor: the CPUs in ``/sys/devices/cpu_core/cpus`` collapsed to
  unique (package, core) pairs, since the file lists both threads of each P-core;
- other Linux: unique (package, core) pairs under ``/sys/devices/system/cpu/cpu*/topology``.

On Linux the count is limited to the CPUs this process may run on and to the cgroup quota.
When a read fails, for example because a sandbox denies ``sysctl``, it falls back to
``os.cpu_count()`` and logs a warning. The result is always between 1 and ``os.cpu_count()``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

ENV_VARIABLE = "COVSYN_WORKERS"
# Share of the available memory the workers may take, leaving room for the parent and others.
MEMORY_SHARE = 0.8
# Peak memory per worker of the checklist bootstrap (scripts/launch_phase_d.sh, B86, E98).
# The other stages get theirs from the peak worker memory that Task 5 records.
CHECKLIST_GB_PER_WORKER = 4.9


class InsufficientMemory(RuntimeError):
    """Not even one worker fits in the available memory."""


def _run_sysctl(key: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=5
    ).stdout


def _affinity() -> Callable[[], set[int]] | None:
    getter = getattr(os, "sched_getaffinity", None)
    return (lambda: set(getter(0))) if getter else None


def _available_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 1e9


@dataclass(frozen=True)
class Machine:
    """Where the counts are read from; the defaults read this machine, tests pass recordings."""

    platform: str = sys.platform
    sysctl: Callable[[str], str] = _run_sysctl
    sys_root: Path = Path("/sys")
    proc_root: Path = Path("/proc")
    cpu_count: Callable[[], int | None] = os.cpu_count
    affinity: Callable[[], set[int]] | None = field(default_factory=_affinity)


@dataclass(frozen=True)
class CoreCount:
    """A core count and how it was found."""

    count: int
    method: str


def parse_cpu_list(text: str) -> set[int]:
    """CPUs in a Linux CPU list such as ``0-3,8,10-11``.

    Raises:
        ValueError: The text is not a CPU list.
    """
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            if text.strip():
                raise ValueError(f"empty item in CPU list {text!r}")
            continue
        first, _, last = part.partition("-")
        low, high = int(first), int(last or first)
        if low < 0 or high < low:
            raise ValueError(f"bad range {part!r} in CPU list {text!r}")
        cpus.update(range(low, high + 1))
    return cpus


def _positive(text: str, source: str) -> int:
    value = int(text.strip())
    if value < 1:
        raise ValueError(f"{source} gave {value}")
    return value


def _mac_cores(machine: Machine) -> CoreCount:
    try:
        return CoreCount(
            _positive(machine.sysctl("hw.perflevel0.physicalcpu"), "hw.perflevel0.physicalcpu"),
            "sysctl hw.perflevel0.physicalcpu",
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return CoreCount(
            _positive(machine.sysctl("hw.physicalcpu"), "hw.physicalcpu"), "sysctl hw.physicalcpu"
        )


def _physical_cores(root: Path, cpus: set[int]) -> int:
    pairs = set()
    for cpu in cpus:
        topology = root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
        pairs.add(
            (
                int((topology / "physical_package_id").read_text()),
                int((topology / "core_id").read_text()),
            )
        )
    if not pairs:
        raise ValueError("no CPU left to count")
    return len(pairs)


def _quota(path: Path) -> int | None:
    """CPUs a cgroup v2 cpu.max allows; None for no limit or a file it cannot read."""
    try:
        quota, period = path.read_text().split()
        if quota == "max":
            return None
        # A quota below one CPU still lets one worker run, slowly.
        return max(1, int(quota) // int(period))
    except (OSError, ValueError) as error:
        log.warning("ignoring the CPU quota in %s (%s)", path, error)
        return None


def _cgroup_limit(machine: Machine) -> int | None:
    """The tightest cgroup v2 CPU quota from the process's own cgroup up to the root.

    A systemd slice or a container sets cpu.max in the process's cgroup, not the root's, so
    every level of the path /proc/self/cgroup names is read. cgroup v1 is not read.
    """
    base = machine.sys_root / "fs" / "cgroup"
    levels = [base]
    try:
        for line in (machine.proc_root / "self" / "cgroup").read_text().splitlines():
            if not line.startswith("0::"):
                continue
            parts = Path(line[3:].strip()).parts[1:]  # without the leading "/"
            # ".." means the cgroup lies outside this namespace, which /sys cannot reach.
            if ".." in parts:
                break
            levels += [base.joinpath(*parts[: depth + 1]) for depth in range(len(parts))]
    except OSError:
        pass  # no /proc: only the root's quota applies
    quotas = [_quota(level / "cpu.max") for level in levels if (level / "cpu.max").exists()]
    return min((q for q in quotas if q is not None), default=None)


def _linux_cores(machine: Machine) -> CoreCount:
    root = machine.sys_root
    p_cores = root / "devices" / "cpu_core" / "cpus"
    if p_cores.exists():
        cpus, method = parse_cpu_list(p_cores.read_text()), "P-cores in /sys/devices/cpu_core"
    else:
        cpus = {
            int(path.name[3:])
            for path in (root / "devices" / "system" / "cpu").glob("cpu*")
            if path.name[3:].isdigit()
        }
        method = "physical cores in /sys topology"
    if machine.affinity is not None:
        allowed = machine.affinity()
        if p_cores.exists() and not cpus & allowed:
            # Pinned to E-cores only: those are the cores this process can use. Pinned to some
            # P-cores and some E-cores, only the P-cores count, as on an unpinned machine.
            cpus, method = set(allowed), "allowed cores in /sys topology (no P-core allowed)"
        else:
            cpus &= allowed
    count = _physical_cores(root, cpus)
    limit = _cgroup_limit(machine)
    if limit is not None and limit < count:
        return CoreCount(limit, f"{method}, limited by the cgroup quota")
    return CoreCount(count, method)


def performance_core_count(machine: Machine | None = None) -> CoreCount:
    """The number of cores worth a worker on ``machine`` (this one by default)."""
    machine = machine or Machine()
    cpus = machine.cpu_count() or 1
    try:
        if machine.platform == "darwin":
            found = _mac_cores(machine)
        elif machine.platform.startswith("linux"):
            found = _linux_cores(machine)
        else:
            raise OSError(f"no core detection for platform {machine.platform!r}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # The CPUs this process may use and the cgroup quota still bound the fallback.
        allowed = cpus
        if machine.affinity is not None:
            try:
                allowed = min(allowed, len(machine.affinity()))
            except OSError:
                pass
        if machine.platform.startswith("linux"):
            allowed = min(allowed, _cgroup_limit(machine) or allowed)
        allowed = max(1, allowed)
        log.warning(
            "could not detect the performance cores (%s); falling back to os.cpu_count() = %d%s",
            error,
            cpus,
            f", limited by the affinity or the cgroup quota to {allowed}" if allowed < cpus else "",
        )
        return CoreCount(allowed, "os.cpu_count() fallback")
    return CoreCount(max(1, min(found.count, cpus)), found.method)


def default_workers(
    memory_per_worker_gb: float | None, cores: int | None = None, available_gb: float | None = None
) -> int:
    """The smaller of the core count and the workers that fit in 80% of the available memory.

    Args:
        memory_per_worker_gb: Peak memory of one worker; None for a stage whose peak is not
            recorded yet, which is then limited by cores only.
        cores: The core count; detected when None.
        available_gb: The available memory; read when None.

    Raises:
        InsufficientMemory: Not even one worker fits.
        ValueError: ``memory_per_worker_gb`` is not a positive number.
    """
    if cores is None:
        cores = performance_core_count().count
    if memory_per_worker_gb is None:
        return cores
    fitting, available_gb = _fitting(memory_per_worker_gb, available_gb)
    if fitting < 1:
        raise InsufficientMemory(
            f"one worker needs {memory_per_worker_gb} GB, but only {available_gb:.1f} GB is "
            f"available, of which CovSyn uses {MEMORY_SHARE:.0%}; free memory, or set "
            f"--workers or {ENV_VARIABLE} to start anyway"
        )
    return min(cores, fitting)


def _fitting(
    memory_per_worker_gb: float, available_gb: float | None, share: float = MEMORY_SHARE
) -> tuple[int, float]:
    """Workers that fit in ``share`` of the available memory, and the available memory read."""
    if not memory_per_worker_gb > 0 or not math.isfinite(memory_per_worker_gb):
        raise ValueError(
            f"memory per worker must be a positive number of GB, got {memory_per_worker_gb}"
        )
    if available_gb is None:
        available_gb = _available_gb()
    return math.floor(share * available_gb / memory_per_worker_gb), available_gb


def _capped(
    workers: int, source: str, memory_per_worker_gb: float | None, available_gb: float | None
) -> int:
    """An override, capped at what the available memory holds for a memory-bound stage (E98).

    One COVSYN_WORKERS sets every step of the chain, so a count meant for the optimiser's cores
    must not start that many 4.9 GB checklist workers. The cap is all of the available memory,
    not the 80% the default keeps free: an explicit count such as E98's 8 is the user's choice
    and is lowered only where it cannot fit. At least one worker starts, since the refusal of
    default_workers tells the user to override in order to start anyway.
    """
    if memory_per_worker_gb is None:
        return workers
    fitting, available_gb = _fitting(memory_per_worker_gb, available_gb, share=1.0)
    capped = max(1, fitting)
    if workers > capped:
        log.warning(
            "%s asks for %d workers of %s GB, but the available %.1f GB holds %d; starting %d",
            source,
            workers,
            memory_per_worker_gb,
            available_gb,
            fitting,
            capped,
        )
        return capped
    return workers


def _override(value: str | int, source: str) -> int:
    try:
        workers = int(value) if isinstance(value, int) else int(value.strip())
    except ValueError:
        workers = 0
    if workers < 1:
        raise ValueError(f"{source} must be a positive whole number of workers, got {value!r}")
    return workers


def resolve_workers(
    cli: int | None,
    env: Mapping[str, str],
    memory_per_worker_gb: float | None,
    cores: int | None = None,
    available_gb: float | None = None,
) -> int:
    """The worker count: ``--workers``, else ``COVSYN_WORKERS``, else :func:`default_workers`.

    An empty ``COVSYN_WORKERS`` counts as unset, because launch_phase_d.sh forwards unset
    variables into the chain's tmux session as empty strings (B86). An unusable one falls back
    to the default with a warning, as B86 decided for VERIFY_WORKERS: the chain reads it hours
    into a run, where a typo must neither stop the run nor mean zero workers. ``--workers`` is
    read when the command starts, so an unusable one stops it at once.

    For a memory-bound stage (``memory_per_worker_gb`` given) an override above what fits in
    memory is capped there, with a warning; an override below it is taken as given.

    Raises:
        ValueError: ``--workers`` is not a positive whole number.
    """
    if cli is not None:
        return _capped(_override(cli, "--workers"), "--workers", memory_per_worker_gb, available_gb)
    # Spelled out, not ENV_VARIABLE, so test_b86's scan of environment reads finds it.
    requested = env.get("COVSYN_WORKERS", "")
    if requested.strip():
        try:
            workers = _override(requested, ENV_VARIABLE)
        except ValueError as error:
            log.warning("%s; using the default instead", error)
        else:
            return _capped(workers, ENV_VARIABLE, memory_per_worker_gb, available_gb)
    return default_workers(memory_per_worker_gb, cores, available_gb)


def positive_int(text: str) -> int:
    """An argparse type for a worker count, so a bad ``--workers`` fails at parsing."""
    try:
        return _override(text, "--workers")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def add_workers_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--workers`` override every entry point shares."""
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help=f"worker processes; overrides {ENV_VARIABLE} and the default, "
        "which is the performance cores limited by memory",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """``covsyn-cores``: print the detected cores, how they were found, and the default workers."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--memory-per-worker-gb",
        type=float,
        default=CHECKLIST_GB_PER_WORKER,
        help="peak memory of one worker (default: the checklist bootstrap, "
        f"{CHECKLIST_GB_PER_WORKER} GB)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    found = performance_core_count()
    print(f"performance cores: {found.count}")
    print(f"method:            {found.method}")
    print(f"os.cpu_count():    {os.cpu_count()}")
    available = _available_gb()
    print(f"available memory:  {available:.1f} GB")
    try:
        workers = default_workers(args.memory_per_worker_gb, found.count, available)
        print(f"default workers:   {workers} (at {args.memory_per_worker_gb} GB per worker)")
    except InsufficientMemory as error:
        print(f"default workers:   none, {error}")


if __name__ == "__main__":
    main()
