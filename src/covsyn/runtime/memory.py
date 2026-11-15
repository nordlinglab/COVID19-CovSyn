# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Memory measurements for the timings tables and run.log (Task 5).

Current resident memory (RSS) and the memory the system has available come from ``psutil``;
a process's peak RSS comes from ``getrusage``, whose ``ru_maxrss`` is in bytes on macOS and
in KiB on Linux. A peak covers the process's whole life and cannot be reset, which is why
every pipeline step runs in a fresh process with a fresh pool.
"""

from __future__ import annotations

import contextlib
import resource
import sys
import threading
from collections.abc import Callable

import psutil

STEP_MODULE = "covsyn.runtime.measure"  # what a pipeline step runs under (Task 5)
LOW_MEMORY_FRACTION = 0.10  # warn below this share of the total (plan, Task 5)
WARNING_ATTEMPTS = 3  # a warning that fails to send is tried on this many samples
MB = 1e6


def megabytes(n: float | None) -> float | None:
    """Bytes as MB (10^6) to 0.1 MB; None, a reading that never succeeded, stays None."""
    return None if n is None else round(n / MB, 1)


def ru_maxrss_bytes(value: int, platform: str = sys.platform) -> int:
    """``ru_maxrss`` in bytes: bytes on macOS, KiB on Linux.

    Raises:
        ValueError: For any other system, whose unit is not known here.
    """
    if platform == "darwin":
        return int(value)
    if platform.startswith("linux"):
        return int(value) * 1024
    raise ValueError(f"the unit of ru_maxrss on {platform} is not known")


def peak_rss_bytes(who: int = resource.RUSAGE_SELF) -> int:
    """Peak RSS of this process (``RUSAGE_SELF``) or of its largest waited-for child."""
    return ru_maxrss_bytes(resource.getrusage(who).ru_maxrss)


def rss_bytes(process: psutil.Process | None = None) -> int:
    """Current RSS of ``process``, by default this process.

    Making the handle costs most of a read, so a caller that reads often keeps its own: a
    worker's three reads per task take 59 us with a new handle each and 20 us with one kept.
    """
    return (process or psutil.Process()).memory_info().rss


def available_bytes() -> int:
    """Memory the system reports as available."""
    return psutil.virtual_memory().available


def total_bytes() -> int:
    """The system's total memory."""
    return psutil.virtual_memory().total


def is_low(available: int, total: int) -> bool:
    """Whether available memory is below LOW_MEMORY_FRACTION of the total."""
    return available < LOW_MEMORY_FRACTION * total


def low_memory_warning(available: int, total: int) -> str:
    """The warning for low memory, giving the share left; the reporter adds its prefix."""
    return (
        f"only {available / MB:.0f} MB available, {100 * available / total:.0f}% "
        f"of the memory (below {100 * LOW_MEMORY_FRACTION:.0f}%)"
    )


def mb_text(n: float | None) -> str:
    """Bytes as whole MB for a log line, rounded once; "unknown" for no reading."""
    return "unknown" if n is None else f"{n / MB:.0f} MB"


def system_memory() -> tuple[int, int]:
    """The available and the total memory, in bytes."""
    memory = psutil.virtual_memory()
    return memory.available, memory.total


class MemoryWatch:
    """Samples this process's RSS, its steps' RSS and the available memory, in a thread.

    Records the largest RSS of this process and, with ``steps``, of its measured steps, and the
    smallest available memory seen, and warns once when the available memory falls below
    LOW_MEMORY_FRACTION of the total. Use it as a context manager.
    """

    def __init__(
        self,
        interval: float = 1.0,
        warn: Callable[[str], None] | None = None,
        system: Callable[[], tuple[int, int]] | None = None,
        steps: bool = False,
    ) -> None:
        """Sample every ``interval`` seconds; ``warn`` prints to stderr by default.

        ``system`` gives the available and total memory; system_memory by default.
        ``steps``: also sample the RSS of this process's children that are pipeline steps, run
        under covsyn.runtime.measure, from outside, as a thread inside a step would make its
        forks unsafe. A step shorter than the interval may be missed.
        """
        self.interval = interval
        self.warn = warn or (
            lambda message: print(f"WARNING: {message}", file=sys.stderr, flush=True)
        )
        self.system = system or system_memory
        self.process = psutil.Process()
        self.parent_peak_bytes: int | None = None  # None until an RSS read succeeds
        self.steps = steps
        # The largest step's sampled RSS; None if no sample caught a step or every read failed
        # (said on stderr).
        self.steps_peak_bytes: int | None = None
        # pid -> handle: children known to be steps, and children that ran something else.
        self._steps: dict[int, psutil.Process] = {}
        self._others: dict[int, psutil.Process] = {}
        try:
            self._own_cmdline = self.process.cmdline()  # a child just forked still shows it
        except psutil.Error:
            self._own_cmdline = None
        self.lowest_available: int | None = None
        self.warning: str | None = None  # the low-memory warning, once made
        self._pending: str | None = None  # a warning to send
        self._attempts = 0
        self.started = False
        self.failed: set[str] = set()  # the reads whose failure was reported
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="memory-watch", daemon=True)

    def sample(self) -> None:
        """Take one sample; the RSS, the available memory and the warning fail separately."""
        self._guarded(self._sample_rss)
        if self.steps:
            self._guarded(self._sample_steps)
        self._guarded(self._sample_available)
        if self._pending is not None:
            self._guarded(self._send_warning)

    def _sample_rss(self) -> None:
        rss = rss_bytes(self.process)
        self.parent_peak_bytes = max(self.parent_peak_bytes or 0, rss)

    def _kept(self, kept: dict[int, psutil.Process], pid: int) -> psutil.Process | None:
        """A kept handle of a running child; one that ended is dropped, as its pid may return."""
        handle = kept.get(pid)
        if handle is not None and not handle.is_running():
            del kept[pid]
            return None
        return handle

    def _sample_steps(self) -> None:
        denied = None
        for child in self.process.children():
            try:
                step = self._kept(self._steps, child.pid)
                if step is None and self._kept(self._others, child.pid) is None:
                    command = child.cmdline()
                    if STEP_MODULE in command:
                        self._steps[child.pid] = step = child
                    elif command != self._own_cmdline:  # exec'd into something else: never
                        self._others[child.pid] = child  # a step, so not read again
                if step is not None:
                    rss = rss_bytes(step)
                    self.steps_peak_bytes = max(self.steps_peak_bytes or 0, rss)
            except psutil.NoSuchProcess:  # ended while it was looked at
                continue
            except psutil.Error as error:  # one child denied stops neither the others nor this
                denied = error
        if denied is not None:
            raise denied  # reported once, by _guarded

    def _sample_available(self) -> None:
        available, total = self.system()
        if self.lowest_available is None or available < self.lowest_available:
            self.lowest_available = available
        if self.warning is None and is_low(available, total):  # once per watch
            self.warning = self._pending = low_memory_warning(available, total)

    def _send_warning(self) -> None:
        # Tried again on the next samples if it fails, but not for ever: a warn that fails
        # after writing part of it would otherwise repeat that part every second.
        self._attempts += 1
        try:
            self.warn(self._pending)
            self._pending = None
        finally:
            if self._attempts >= WARNING_ATTEMPTS:
                self._pending = None

    def _guarded(self, read: Callable[[], None]) -> None:
        """Run one read; its failure (/proc unreadable, a full disk) is reported once, not fatal."""
        try:
            read()
        except Exception as error:
            if read.__name__ not in self.failed:
                self.failed.add(read.__name__)
                with contextlib.suppress(Exception):  # stderr may be the broken part
                    print(f"memory watch: a sample failed ({error!r})", file=sys.stderr)

    def _run(self) -> None:
        while True:
            self.sample()
            if self._stop.wait(self.interval):
                return

    def __enter__(self) -> MemoryWatch:
        """Start sampling."""
        self.thread.start()
        self.started = True
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop sampling, after a last sample."""
        self._stop.set()
        self.thread.join()
        self.sample()  # guarded: must not replace the stage's own result or exception
