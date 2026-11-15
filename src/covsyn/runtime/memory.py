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

LOW_MEMORY_FRACTION = 0.10  # warn below this share of the total (plan, Task 5)
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


def system_memory() -> tuple[int, int]:
    """The available and the total memory, in bytes."""
    memory = psutil.virtual_memory()
    return memory.available, memory.total


class MemoryWatch:
    """Samples this process's RSS and the available memory once per interval, in a thread.

    Records the largest RSS and the smallest available memory seen, and warns once when the
    available memory falls below LOW_MEMORY_FRACTION of the total. Use it as a context manager.
    """

    def __init__(
        self,
        interval: float = 1.0,
        warn: Callable[[str], None] | None = None,
        system: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        """Sample every ``interval`` seconds; ``warn`` prints to stderr by default.

        ``system`` gives the available and total memory; system_memory by default.
        """
        self.interval = interval
        self.warn = warn or (
            lambda message: print(f"WARNING: {message}", file=sys.stderr, flush=True)
        )
        self.system = system or system_memory
        self.process = psutil.Process()
        self.parent_peak_bytes: int | None = None  # None until an RSS read succeeds
        self.lowest_available: int | None = None
        self.warned = False
        self._pending: str | None = None  # a warning to send
        self.failed: set[str] = set()  # the reads whose failure was reported
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="memory-watch", daemon=True)

    def sample(self) -> None:
        """Take one sample; the RSS, the available memory and the warning fail separately."""
        self._guarded(self._sample_rss)
        self._guarded(self._sample_available)
        if self._pending is not None:
            self._guarded(self._send_warning)

    def _sample_rss(self) -> None:
        rss = rss_bytes(self.process)
        self.parent_peak_bytes = max(self.parent_peak_bytes or 0, rss)

    def _sample_available(self) -> None:
        available, total = self.system()
        if self.lowest_available is None or available < self.lowest_available:
            self.lowest_available = available
        if not self.warned and is_low(available, total):
            self.warned = True
            self._pending = low_memory_warning(available, total)

    def _send_warning(self) -> None:
        self.warn(self._pending)
        self._pending = None  # only once sent: a failed send is tried again next sample

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
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop sampling, after a last sample."""
        self._stop.set()
        self.thread.join()
        self.sample()  # guarded: must not replace the stage's own result or exception
