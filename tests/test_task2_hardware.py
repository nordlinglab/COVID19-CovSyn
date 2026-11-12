# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 2: detect the performance cores and choose the worker count.

The expected values are the acceptance criteria of the rework plan
(doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 2), checked against recorded
``sysctl`` answers and ``/sys`` trees instead of the machine running the tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from covsyn.runtime import hardware as hw


def sysctl_answers(answers: dict[str, str]):
    """A sysctl stand-in that knows only ``answers`` and fails like a missing key otherwise."""

    def sysctl(key: str) -> str:
        if key not in answers:
            raise OSError(f"sysctl: unknown oid '{key}'")
        return answers[key]

    return sysctl


def denied_sysctl(key: str) -> str:
    """A sysctl that a sandbox denies."""
    raise PermissionError("sysctl denied by the sandbox")


def write_topology(root: Path, cores: dict[int, tuple[int, int]]) -> None:
    """Write /sys/devices/system/cpu/cpu<N>/topology for logical CPU -> (package, core)."""
    for cpu, (package, core) in cores.items():
        topology = root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(f"{package}\n")
        (topology / "core_id").write_text(f"{core}\n")


def smt_topology(physical: int, threads: int = 2) -> dict[int, tuple[int, int]]:
    """Logical CPUs numbered like Linux does: first thread of every core, then the second."""
    return {
        thread * physical + core: (0, core) for thread in range(threads) for core in range(physical)
    }


def mac(answers: dict[str, str], cpus: int) -> hw.Machine:
    """A Mac answering ``answers`` with ``cpus`` logical CPUs."""
    return hw.Machine(
        platform="darwin", sysctl=sysctl_answers(answers), cpu_count=lambda: cpus, affinity=None
    )


def linux(root: Path, cpus: int, affinity: set[int] | None = None) -> hw.Machine:
    """A Linux machine reading ``root`` as /sys and ``root/proc`` as /proc, with ``cpus`` CPUs."""
    return hw.Machine(
        platform="linux",
        sys_root=root,
        proc_root=root / "proc",
        cpu_count=lambda: cpus,
        affinity=(lambda: affinity) if affinity is not None else None,
    )


# --- the acceptance criteria -------------------------------------------------------------


def test_m4_max_counts_its_12_performance_cores():
    """An M4 Max has 12 performance and 4 efficiency cores."""
    answers = {
        "hw.perflevel0.physicalcpu": "12\n",
        "hw.perflevel1.physicalcpu": "4\n",
        "hw.physicalcpu": "16\n",
    }
    assert hw.performance_core_count(mac(answers, 16)).count == 12


def test_intel_mac_counts_its_physical_cores():
    """An Intel Mac has no performance levels, so hw.perflevel0 is an unknown key."""
    found = hw.performance_core_count(mac({"hw.physicalcpu": "8\n"}, 16))
    assert (found.count, found.method) == (8, "sysctl hw.physicalcpu")


def test_hybrid_intel_counts_p_cores_once_each(tmp_path):
    """8 P-cores with two threads each (CPUs 0-15) and 8 E-cores (CPUs 16-23)."""
    cores = smt_topology(8) | {16 + e: (0, 8 + e) for e in range(8)}
    write_topology(tmp_path, cores)
    (tmp_path / "devices" / "cpu_core").mkdir(parents=True)
    (tmp_path / "devices" / "cpu_core" / "cpus").write_text("0-15\n")
    assert hw.performance_core_count(linux(tmp_path, 24)).count == 8


def test_eight_physical_cores_with_smt_count_eight(tmp_path):
    """Two hardware threads per core still count as one core."""
    write_topology(tmp_path, smt_topology(8))
    found = hw.performance_core_count(linux(tmp_path, 16))
    assert (found.count, found.method) == (8, "physical cores in /sys topology")


def test_a_cgroup_quota_of_four_cpus_limits_to_four(tmp_path):
    """A container allowed 4 CPUs gets 4 workers on a 16-core host."""
    write_topology(tmp_path, smt_topology(16))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("400000 100000\n")
    assert hw.performance_core_count(linux(tmp_path, 32)).count == 4


def test_every_read_failing_falls_back_to_os_cpu_count(caplog):
    """A denied sysctl falls back to os.cpu_count() with a warning."""
    machine = hw.Machine(
        platform="darwin", sysctl=denied_sysctl, cpu_count=lambda: 10, affinity=None
    )
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        found = hw.performance_core_count(machine)
    assert (found.count, found.method) == (10, "os.cpu_count() fallback")
    assert "os.cpu_count()" in caplog.text


def test_linux_without_a_topology_falls_back(tmp_path, caplog):
    """No /sys topology to read is a failed read too."""
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        found = hw.performance_core_count(linux(tmp_path, 6))
    assert (found.count, found.method) == (6, "os.cpu_count() fallback")


def test_refuses_when_one_worker_does_not_fit_in_memory():
    """The refusal says how much memory one worker needs."""
    with pytest.raises(hw.InsufficientMemory, match=r"4\.9 GB"):
        hw.default_workers(4.9, cores=16, available_gb=5.0)


# --- the rest of the specification ---------------------------------------------------------


def test_affinity_limits_the_cores(tmp_path):
    """Allowed only CPUs 0, 1 and 8 of 8 two-thread cores: CPUs 0 and 8 share core 0."""
    write_topology(tmp_path, smt_topology(8))
    assert hw.performance_core_count(linux(tmp_path, 16, affinity={0, 1, 8})).count == 2


def test_an_unlimited_cgroup_leaves_the_count(tmp_path):
    """``max`` in cpu.max means no quota."""
    write_topology(tmp_path, smt_topology(8))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("max 100000\n")
    assert hw.performance_core_count(linux(tmp_path, 16)).count == 8


def test_a_fractional_quota_keeps_at_least_one_core(tmp_path):
    """Half a CPU of quota still runs one worker."""
    write_topology(tmp_path, smt_topology(8))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("50000 100000\n")
    assert hw.performance_core_count(linux(tmp_path, 16)).count == 1


@pytest.mark.parametrize(
    ("text", "cpus"), [("0-3,8,10-11\n", {0, 1, 2, 3, 8, 10, 11}), ("5\n", {5}), ("\n", set())]
)
def test_cpu_lists_are_parsed(text, cpus):
    """Ranges and single CPUs, as /sys writes them."""
    assert hw.parse_cpu_list(text) == cpus


@pytest.mark.parametrize("text", ["a-b", "3-1", "1,,2x"])
def test_malformed_cpu_lists_are_rejected(text):
    """Text that is not a CPU list is an error, not a guess."""
    with pytest.raises(ValueError):
        hw.parse_cpu_list(text)


def test_default_workers_is_the_smaller_of_cores_and_memory():
    """Memory binds with 4.9 GB workers, cores with small ones."""
    assert hw.default_workers(4.9, cores=32, available_gb=100.0) == 16  # floor(80 / 4.9)
    assert hw.default_workers(0.5, cores=12, available_gb=100.0) == 12


def test_default_workers_without_a_memory_figure_uses_the_cores():
    """Stages whose peak memory Task 5 has not recorded yet are limited by cores only."""
    assert hw.default_workers(None, cores=12, available_gb=1.0) == 12


@pytest.mark.parametrize("memory", [0.0, -1.0, float("nan")])
def test_default_workers_rejects_an_invalid_memory_figure(memory):
    """Zero, negative or NaN GB per worker is a programming error."""
    with pytest.raises(ValueError):
        hw.default_workers(memory, cores=4, available_gb=10.0)


def test_cli_beats_environment_beats_detection():
    """--workers, then COVSYN_WORKERS, then the detected default."""
    env = {"COVSYN_WORKERS": "6"}
    assert hw.resolve_workers(3, env, None, cores=12) == 3
    assert hw.resolve_workers(None, env, None, cores=12) == 6
    assert hw.resolve_workers(None, {}, None, cores=12) == 12


def test_an_empty_environment_variable_counts_as_unset():
    """launch_phase_d.sh forwards unset variables into tmux as empty strings (B86)."""
    assert hw.resolve_workers(None, {"COVSYN_WORKERS": " "}, None, cores=5) == 5


@pytest.mark.parametrize("value", ["0", "-2", "four", "2.5"])
def test_an_unusable_environment_value_falls_back_with_a_warning(value, caplog):
    """B86: read hours into a chain, a typo must neither stop the run nor mean zero workers."""
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert hw.resolve_workers(None, {"COVSYN_WORKERS": value}, None, cores=4) == 4
    assert "COVSYN_WORKERS" in caplog.text


@pytest.mark.parametrize("value", [0, -2])
def test_an_unusable_workers_option_stops_the_command(value):
    """--workers is read when the command starts, so the user sees the error at once."""
    with pytest.raises(ValueError, match="--workers"):
        hw.resolve_workers(value, {}, None, cores=4)


def test_the_command_prints_counts_method_and_workers(capsys):
    """covsyn-cores reports what it found and how."""
    hw.main(["--memory-per-worker-gb", "4.9"])
    out = capsys.readouterr().out
    for word in ("performance cores", "method", "default workers", "os.cpu_count()"):
        assert word in out


# --- property ---------------------------------------------------------------------------------

anything = st.one_of(st.text(max_size=20), st.integers(-(10**6), 10**6).map(str))


@settings(max_examples=200, deadline=None)
@given(perf=anything, phys=anything, cpus=st.integers(1, 512))
def test_any_sysctl_answer_gives_between_one_and_cpu_count(perf, phys, cpus):
    """Whatever sysctl answers, the count stays within 1..os.cpu_count()."""
    machine = mac({"hw.perflevel0.physicalcpu": perf, "hw.physicalcpu": phys}, cpus)
    assert 1 <= hw.performance_core_count(machine).count <= cpus


@settings(max_examples=100, deadline=None)
@given(
    cpulist=st.text(max_size=20),
    quota=st.text(max_size=20),
    cpus=st.integers(1, 64),
    physical=st.integers(0, 8),
)
def test_any_sys_tree_gives_between_one_and_cpu_count(
    tmp_path_factory, cpulist, quota, cpus, physical
):
    """Whatever /sys contains, the count stays within 1..os.cpu_count()."""
    root = tmp_path_factory.mktemp("sys")
    write_topology(root, smt_topology(physical))
    (root / "devices" / "cpu_core").mkdir(parents=True)
    (root / "devices" / "cpu_core" / "cpus").write_text(cpulist)
    (root / "fs" / "cgroup").mkdir(parents=True)
    (root / "fs" / "cgroup" / "cpu.max").write_text(quota)
    assert 1 <= hw.performance_core_count(linux(root, cpus)).count <= cpus


# --- code review 2026-11-13 -------------------------------------------------------------------


def test_p_cores_outside_the_affinity_count_the_allowed_cores(tmp_path):
    """Pinned to the E-cores of a hybrid Intel, the allowed physical cores are counted."""
    cores = smt_topology(8) | {16 + e: (0, 8 + e) for e in range(8)}
    write_topology(tmp_path, cores)
    (tmp_path / "devices" / "cpu_core").mkdir(parents=True)
    (tmp_path / "devices" / "cpu_core" / "cpus").write_text("0-15\n")
    assert hw.performance_core_count(linux(tmp_path, 24, affinity={16, 17, 18})).count == 3


def test_the_fallback_respects_the_affinity(caplog):
    """A failed read falls back to the CPUs this process may use, not every CPU."""
    machine = hw.Machine(
        platform="darwin", sysctl=denied_sysctl, cpu_count=lambda: 32, affinity=lambda: {0, 1, 2, 3}
    )
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert hw.performance_core_count(machine).count == 4


def test_a_malformed_cgroup_file_keeps_the_core_count(tmp_path, caplog):
    """An unreadable quota is no reason to throw away the cores already counted."""
    write_topology(tmp_path, smt_topology(8))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("garbage\n")
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert hw.performance_core_count(linux(tmp_path, 16)).count == 8
    assert "cpu.max" in caplog.text


def test_the_quota_of_the_process_own_cgroup_is_found(tmp_path):
    """A systemd slice with CPUQuota=400% sets cpu.max in the process's cgroup, not the root."""
    write_topology(tmp_path, smt_topology(16))
    nested = tmp_path / "fs" / "cgroup" / "system.slice" / "covsyn.service"
    nested.mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("max 100000\n")
    (tmp_path / "fs" / "cgroup" / "system.slice" / "cpu.max").write_text("800000 100000\n")
    (nested / "cpu.max").write_text("400000 100000\n")
    proc = tmp_path / "proc" / "self"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/system.slice/covsyn.service\n")
    machine = hw.Machine(
        platform="linux",
        sys_root=tmp_path,
        proc_root=tmp_path / "proc",
        cpu_count=lambda: 32,
        affinity=None,
    )
    assert hw.performance_core_count(machine).count == 4


def test_an_override_above_the_memory_is_capped_with_a_warning(caplog):
    """E98: COVSYN_WORKERS=64 for the optimiser must not start 64 checklist workers."""
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert (
            hw.resolve_workers(None, {"COVSYN_WORKERS": "64"}, 4.9, cores=64, available_gb=50.0)
            == 10  # floor(50 / 4.9): all of the available memory, not 80%
        )
        assert hw.resolve_workers(64, {}, 4.9, cores=64, available_gb=50.0) == 10
    assert "available 50.0 GB holds 10" in caplog.text


def test_e98s_setting_fits_where_the_default_would_be_lower():
    """8 workers of 4.9 GB fit in 45 GB, although the default keeps 20% free and gives 7."""
    assert hw.default_workers(4.9, cores=32, available_gb=45.0) == 7
    assert hw.resolve_workers(None, {"COVSYN_WORKERS": "8"}, 4.9, cores=32, available_gb=45.0) == 8


def test_an_override_below_the_memory_is_taken_as_given():
    """Lowering the count, E98's use, is never capped."""
    assert hw.resolve_workers(None, {"COVSYN_WORKERS": "3"}, 4.9, cores=64, available_gb=500.0) == 3


def test_an_override_starts_one_worker_even_without_the_memory():
    """The refusal tells the user to override; the override then runs one worker."""
    assert hw.resolve_workers(4, {}, 4.9, cores=8, available_gb=1.0) == 1


@pytest.mark.parametrize("value", ["0", "-3", "x"])
def test_the_workers_option_rejects_non_positive_values_at_parsing(value):
    """The parser reports the error, with no traceback after the command started."""
    import argparse

    parser = argparse.ArgumentParser()
    hw.add_workers_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--workers", value])


def test_the_checklist_falls_back_to_the_serial_bootstrap_without_memory(monkeypatch):
    """Short of memory, the checklist computes the intervals in its own process.

    The memory is faked, not the error, so the path from resolve_workers is the real one. The
    module reads sys.argv when first imported, as in test_b92's fixture.
    """
    import importlib
    import sys

    monkeypatch.setattr(sys, "argv", ["verify_phase_d"])
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    verify = importlib.import_module("covsyn.validation.verify_phase_d")
    monkeypatch.setattr(hw, "_available_gb", lambda: 1.0)
    assert verify.bootstrap_workers() == 1


def test_the_fallback_applies_the_cgroup_quota(tmp_path, caplog):
    """No topology to read, but a container quota of 4 CPUs: 4, not os.cpu_count()."""
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("400000 100000\n")
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        found = hw.performance_core_count(linux(tmp_path, 32))
    assert (found.count, found.method) == (4, "os.cpu_count() fallback")


def test_the_root_cgroup_path_is_read_once(tmp_path, caplog):
    """'0::/' (a container's own namespace) names the root, whose bad file warns once."""
    write_topology(tmp_path, smt_topology(8))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("garbage\n")
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/\n")
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert hw.performance_core_count(linux(tmp_path, 16)).count == 8
    assert caplog.text.count("ignoring the CPU quota") == 1


def test_a_cgroup_outside_the_namespace_is_not_followed(tmp_path):
    """'..' in /proc/self/cgroup would lead outside /sys/fs/cgroup; only the root is read."""
    write_topology(tmp_path, smt_topology(8))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    outside = tmp_path / "fs" / "x"
    outside.mkdir(parents=True)
    (outside / "cpu.max").write_text("100000 100000\n")
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/../x\n")
    assert hw.performance_core_count(linux(tmp_path, 16)).count == 8


def test_an_unreadable_cgroup_file_never_stops_the_detection(tmp_path, caplog):
    """A cpu.max the process cannot read (here a directory) sets no limit, on either path."""
    (tmp_path / "fs" / "cgroup" / "cpu.max").mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert hw.performance_core_count(linux(tmp_path, 6)).count == 6  # the fallback
        write_topology(tmp_path, smt_topology(4))
        assert hw.performance_core_count(linux(tmp_path, 8)).count == 4  # the topology
    assert caplog.text.count("ignoring the CPU quota") == 2


def test_cgroup_levels_without_a_quota_are_silent(tmp_path, caplog):
    """Most levels of a cgroup path set no cpu.max; their absence is no warning.

    user.slice is a file here, so its child raises NotADirectoryError, which Path.exists()
    treated as absent too.
    """
    write_topology(tmp_path, smt_topology(4))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "user.slice").write_text("")
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/user.slice/session-1.scope\n")
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert hw.performance_core_count(linux(tmp_path, 8)).count == 4
    assert caplog.text == ""


def test_a_cgroup_name_that_is_not_utf8_is_still_read(tmp_path):
    """A cgroup name is bytes; its quota of 1 CPU binds below the root's 2."""
    import os

    write_topology(tmp_path, smt_topology(4))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("200000 100000\n")
    nested = tmp_path / "fs" / "cgroup" / os.fsdecode(b"job\xff")
    nested.mkdir()
    (nested / "cpu.max").write_text("100000 100000\n")
    (tmp_path / "proc" / "self").mkdir(parents=True)
    (tmp_path / "proc" / "self" / "cgroup").write_bytes(b"0::/job\xff\n")
    assert hw.performance_core_count(linux(tmp_path, 8)).count == 1


def test_an_unreadable_affinity_is_reported(tmp_path, caplog):
    """Counting without the affinity is logged, not silent."""

    def denied() -> set[int]:
        raise PermissionError("sched_getaffinity denied")

    write_topology(tmp_path, smt_topology(4))
    machine = hw.Machine(
        platform="linux",
        sys_root=tmp_path,
        proc_root=tmp_path / "proc",
        cpu_count=lambda: 8,
        affinity=denied,
    )
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        assert hw.performance_core_count(machine).count == 4
    assert "affinity" in caplog.text


def test_the_method_names_the_quota_only_when_it_binds(tmp_path):
    """16 cores, a quota of 12 and 8 CPUs: 8, and the quota is not named as the limit."""
    write_topology(tmp_path, smt_topology(16, threads=1))
    (tmp_path / "fs" / "cgroup").mkdir(parents=True)
    (tmp_path / "fs" / "cgroup" / "cpu.max").write_text("1200000 100000\n")
    found = hw.performance_core_count(linux(tmp_path, 8))
    assert found.count == 8 and "cgroup" not in found.method
