# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 5: memory measurements (doc/plan_Nordling2026_CovSynFireflyNordlingLab.md).

``ru_maxrss`` is in bytes on macOS and in KiB on Linux; the conversion is checked for both
systems whichever one runs the tests, and the 500 MiB task runs in a fresh worker so the
peak it raises is that worker's alone.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import resource
import time

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from covsyn.runtime import memory

MIB = 1024**2


@pytest.mark.parametrize(
    ("platform", "ru_maxrss", "expected"),
    [
        ("darwin", 1024, 1024),  # bytes
        ("darwin", 524_288_000, 524_288_000),
        ("linux", 1024, 1024 * 1024),  # KiB
        ("linux", 512_000, 524_288_000),
    ],
)
def test_ru_maxrss_unit_on_each_system(platform, ru_maxrss, expected):
    """Recorded ru_maxrss values convert to bytes for each system."""
    assert memory.ru_maxrss_bytes(ru_maxrss, platform) == expected


def test_unknown_system_is_refused():
    """Another system's unit is unknown; a wrong guess would be off by a factor of 1024."""
    with pytest.raises(ValueError, match="win32"):
        memory.ru_maxrss_bytes(1, "win32")


@given(st.integers(min_value=0, max_value=2**50))
def test_conversion_is_exact_and_monotone(value):
    """The conversion is exact for any value."""
    assert memory.ru_maxrss_bytes(value, "linux") == value * 1024
    assert memory.ru_maxrss_bytes(value, "darwin") == value


def test_peak_matches_getrusage_on_this_system():
    """The peak is getrusage's, in bytes, on the system running the tests."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = memory.peak_rss_bytes()
    assert memory.ru_maxrss_bytes(raw, memory.sys.platform) <= peak


def test_current_rss_does_not_exceed_the_peak():
    """The current RSS is positive and at most the peak."""
    assert 0 < memory.rss_bytes() <= memory.peak_rss_bytes()


def test_available_and_total_memory():
    """Available memory is positive and at most the total."""
    available, total = memory.available_bytes(), memory.total_bytes()
    assert 0 < available <= total


@pytest.mark.parametrize(
    ("available", "total", "low"),
    [(9, 100, True), (10, 100, False), (50, 100, False), (0, 100, True)],
)
def test_low_memory_is_below_ten_percent(available, total, low):
    """Low means below 10% of the total."""
    assert memory.is_low(available, total) is low


def _fill_500_mib() -> tuple[int, int]:
    before = memory.rss_bytes()
    block = np.ones(500 * MIB // 8)  # np.ones writes every page, so all of it is resident
    after = memory.peak_rss_bytes()
    del block
    return before, after


def test_500_mib_task_raises_the_workers_peak_by_500_mib():
    """The peak rises by the block, measured from the worker's RSS when the task starts.

    From the RSS, not from the peak: a fresh worker's peak includes memory its imports freed,
    so a 500 MiB block raises the peak by less than 500 MiB above it (497 MiB under pytest).
    The 1 MiB allowance is for pages the rest of the process returns meanwhile (4 KiB in a
    trial run); a wrong unit would be off by a factor of 1024.
    """
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        before, after = pool.submit(_fill_500_mib).result()
    assert after - before >= 499 * MIB
    assert after - before < 600 * MIB


def test_watch_records_the_peak_and_warns_once_when_memory_is_low():
    """The watch keeps the smallest available memory and warns once."""
    warnings = []
    samples = iter([(10, 100), (5, 100), (4, 100)])
    last = (4, 100)

    def system():
        return next(samples, last)

    with memory.MemoryWatch(interval=0.01, warn=warnings.append, system=system) as watch:
        time.sleep(0.2)
    assert watch.lowest_available == 4
    assert watch.parent_peak_bytes >= memory.rss_bytes() // 2
    assert len(warnings) == 1
    assert warnings[0].startswith("only") and "5% of the memory" in warnings[0]


def test_watch_stops_its_thread():
    """Leaving the watch stops its thread."""
    with memory.MemoryWatch(interval=0.01) as watch:
        pass
    assert not watch.thread.is_alive()


def _task_reading_100_mib() -> tuple[int, int, int]:
    import os

    from covsyn.calibration import fast_cost
    from covsyn.runtime import timings

    clock = timings.TaskClock(False, fast_cost._WORKER_PROCESS)
    block = np.ones(100 * MIB // 8)
    record = clock.record()
    del block
    return record["pid"], os.getpid(), record["rss_after"] - record["rss_before"]


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_a_worker_measures_itself_not_its_parent(method, demographic_parameters):
    """A task's memory reads are its worker's own, also after a fork: they see its 100 MiB."""
    from covsyn.calibration import fast_cost

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context(method),
        initializer=fast_cost.init_worker, initargs=(demographic_parameters, 1),
    ) as pool:  # fmt: skip
        recorded, actual, grown = pool.submit(_task_reading_100_mib).result()
    assert recorded == actual
    assert grown >= 99 * MIB


def test_rss_of_a_given_process():
    """A kept handle reads the same process."""
    import psutil

    assert 0 < memory.rss_bytes(psutil.Process()) <= memory.peak_rss_bytes()


def test_the_warning_has_no_prefix_of_its_own():
    """Whoever reports it adds "WARNING: ", once."""
    assert not memory.low_memory_warning(5, 100).startswith("WARNING")


def test_a_warning_that_cannot_be_written_does_not_stop_the_sampling():
    """The watch keeps sampling when its warn callback raises, as a full disk would make it."""
    samples = iter([(5, 100), (4, 100), (3, 100)])

    def warn(message):
        raise OSError("No space left on device")

    with memory.MemoryWatch(interval=0.01, warn=warn, system=lambda: next(samples, (3, 100))) as w:
        time.sleep(0.2)
        alive = w.thread.is_alive()
    assert alive and w.lowest_available == 3
