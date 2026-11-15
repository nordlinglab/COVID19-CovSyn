# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""``--mode profile``: each worker profiles its own tasks; the parent merges them (Task 5).

A worker writes ``results/models/<run-id>/profile/worker_<pid>.prof`` when the pool shuts it
down. The parent adds them with ``pstats.Stats.add`` into ``merged.prof`` and writes the 40
functions with the largest cumulative time to ``results/tables/<run-id>/profile_top.csv``.
"""

from __future__ import annotations

import cProfile
import csv
import os
import threading
from pathlib import Path
from pstats import Stats

MERGED = "merged.prof"
TOP_TABLE = "profile_top.csv"
TOP = 40
TOP_COLUMNS = (
    "rank",
    "function",
    "file",
    "line",
    "calls",
    "primitive_calls",
    "tottime_s",
    "cumtime_s",
    "cumtime_share",
)


class WorkerProfile:
    """A worker's profile, written to ``worker_<pid>.prof`` in profile_dir when it exits.

    A pool worker leaves through threading's shutdown, not through atexit, so the dump is
    registered there (``threading._register_atexit``, which concurrent.futures itself uses).
    The caller enables the profile around each task only, so waiting for work is not counted;
    a worker that ran no task writes nothing, as pstats cannot read an empty profile.
    """

    def __init__(self, profile_dir: Path) -> None:
        """Make the profile and register its dump."""
        self.profile = cProfile.Profile()
        self.path = Path(profile_dir) / f"worker_{os.getpid()}.prof"
        self.used = False
        threading._register_atexit(self.dump)

    def enable(self) -> None:
        """Profile from now on."""
        self.used = True
        self.profile.enable()

    def disable(self) -> None:
        """Stop profiling."""
        self.profile.disable()

    def dump(self) -> None:
        """Write the profile, if a task ran."""
        if self.used:
            self.profile.dump_stats(str(self.path))


def start_worker_profile(profile_dir: Path) -> WorkerProfile:
    """This worker's profile; see WorkerProfile."""
    return WorkerProfile(profile_dir)


def share(path: Path, function: str) -> float:
    """``function``'s cumulative time as a share of the profile's total time."""
    stats = Stats(str(path))
    cumulative = max(
        (stat[3] for func, stat in stats.stats.items() if func[2] == function), default=0.0
    )
    return cumulative / stats.total_tt if stats.total_tt else 0.0


def merge_profiles(profile_dir: Path, tables: Path) -> Path:
    """Merge the workers' profiles into merged.prof, and write profile_top.csv to ``tables``.

    Raises:
        FileNotFoundError: profile_dir holds no worker_<pid>.prof.
    """
    profile_dir = Path(profile_dir)
    workers = sorted(profile_dir.glob("worker_*.prof"))
    if not workers:
        raise FileNotFoundError(f"{profile_dir} holds no worker_<pid>.prof to merge")
    stats = Stats(str(workers[0]))
    for path in workers[1:]:
        stats.add(str(path))
    merged = profile_dir / MERGED
    stats.dump_stats(str(merged))
    ranked = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:TOP]
    tables = Path(tables)
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / TOP_TABLE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TOP_COLUMNS)
        for rank, ((file, line, function), (primitive, calls, tottime, cumtime, _)) in enumerate(
            ranked, start=1
        ):
            writer.writerow(
                [
                    rank,
                    function,
                    file,
                    line,
                    calls,
                    primitive,
                    round(tottime, 6),
                    round(cumtime, 6),
                    round(cumtime / stats.total_tt, 4) if stats.total_tt else 0.0,
                ]
            )
    return merged
