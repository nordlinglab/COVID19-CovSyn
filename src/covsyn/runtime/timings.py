# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The optimiser's timing and memory tables (Task 5).

``results/tables/<run-id>/timings_evaluation.csv`` has one row per evaluation and
``timings_generation.csv`` one per generation, the initial population as generation -1, as
its checkpoint is. Until Task 6 gives each evaluation one worker, an evaluation's simulations
are spread over the pool in tasks of SIMULATIONS_PER_TASK, so its times are sums over its tasks
and its worker memory the largest over its workers.

``LAST`` is the dict fast_cost.cost_function fills for the last evaluation. It lives here, in a
module imported under one name only, for the reason cost_parts.LAST does (finding E71): mutate
it in place, never rebind it.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import psutil

from covsyn.model.data_synthesis_main import STAGES
from covsyn.runtime import memory

LAST: dict = {}

EVALUATION_TABLE = "timings_evaluation.csv"
GENERATION_TABLE = "timings_generation.csv"
EVALUATION_COLUMNS = (
    "generation",
    "candidate",
    "worker_pids",
    "start_unix_s",
    "end_unix_s",
    "simulations",
    "run_covid_s",
    "case_reduction_s",
    "cost_reduction_s",
    "dispatch_s",
    "wait_s",
    "busy_s",
    *(f"{stage}_s" for stage in STAGES),
    "worker_rss_before_mb",
    "worker_rss_after_mb",
    "worker_peak_mb",
)
GENERATION_COLUMNS = (
    "generation",
    "start_unix_s",
    "end_unix_s",
    "wall_s",
    "evaluations",
    "moves_s",
    "dispatch_s",
    "wait_s",
    "checkpoint_s",
    "busy_s",
    "workers",
    "utilisation",
    "parent_rss_mb",
    "worker_rss_sum_mb",
    "worker_rss_max_mb",
    "available_mb",
)
NS = 1e9


def seconds(ns: float) -> float:
    """Nanoseconds as seconds, to the microsecond."""
    return round(ns / NS, 6)


def megabytes(n: float) -> float:
    """Bytes as MB (10^6), to 0.1 MB."""
    return round(n / memory.MB, 1)


class TaskClock:
    """One pool task's timing and memory, measured in the worker."""

    def __init__(self, stage_timing: bool, process: psutil.Process | None = None) -> None:
        """Start the task: its PID, wall-clock start and RSS.

        ``process`` is a handle on this process that the worker keeps; without one, the task
        makes its own for its two reads. Every pool that runs fast_cost's tasks gives one, as
        init_worker, which the tasks need anyway, makes it.
        """
        self.process = process or psutil.Process()
        self.pid = os.getpid()
        self.start = time.time()
        self.started = time.perf_counter_ns()
        self.rss_before = memory.rss_bytes(self.process)
        self.run_covid_ns = 0
        self.reduction_ns = 0
        self.stage_ns: dict[str, int] | None = {} if stage_timing else None
        self.simulations = 0

    def record(self) -> dict:
        """End the task: what the parent needs to fill the evaluation's row."""
        return {
            "pid": self.pid,
            "start": self.start,
            "busy_ns": time.perf_counter_ns() - self.started,
            "simulations": self.simulations,
            "run_covid_ns": self.run_covid_ns,
            "reduction_ns": self.reduction_ns,
            "stage_ns": self.stage_ns,
            "rss_before": self.rss_before,
            "rss_after": memory.rss_bytes(self.process),
            "peak": memory.peak_rss_bytes(),
        }


def summarise_tasks(tasks: Iterable[Mapping]) -> dict:
    """An evaluation's worker fields from its tasks, in submission order.

    Times add over the tasks. Each worker's RSS before is that of its first task, after that of
    its last, and the evaluation's is the largest over its workers; so is its peak.
    ``worker_rss`` maps each PID to its RSS after its last task, for the generation's row.
    """
    tasks = list(tasks)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    peak: dict[int, int] = {}
    stage_ns: dict[str, int] | None = None
    for task in tasks:
        pid = task["pid"]
        first.setdefault(pid, task["rss_before"])
        last[pid] = task["rss_after"]
        peak[pid] = max(peak.get(pid, 0), task["peak"])
        if task["stage_ns"] is not None:
            stage_ns = stage_ns or dict.fromkeys(STAGES, 0)
            for stage, ns in task["stage_ns"].items():
                stage_ns[stage] += ns
    fields = {
        "worker_pids": ";".join(str(pid) for pid in first),
        "simulations": sum(task["simulations"] for task in tasks),
        "run_covid_s": seconds(sum(task["run_covid_ns"] for task in tasks)),
        "case_reduction_s": seconds(sum(task["reduction_ns"] for task in tasks)),
        "busy_s": seconds(sum(task["busy_ns"] for task in tasks)),
        "worker_rss_before_mb": megabytes(max(first.values())) if first else None,
        "worker_rss_after_mb": megabytes(max(last.values())) if last else None,
        "worker_peak_mb": megabytes(max(peak.values())) if peak else None,
        "worker_rss": last,
    }
    for stage in STAGES:
        fields[f"{stage}_s"] = None if stage_ns is None else seconds(stage_ns[stage])
    return fields


def append_row(path: Path, columns: Iterable[str], row: Mapping, new: bool = False) -> None:
    """Append ``row`` to the table at ``path``, under the table's own header.

    ``columns`` is the header of a table that is new, empty (a kill can leave one so), or
    replaced (``new``). A row is written in the header's columns: a value it lacks is left
    blank, as an objective other than fast_cost's, or a failed evaluation's cost parts, fill
    fewer columns, and one the header lacks is left out.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = None
    if not new and path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
    with open(path, "w" if new else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(columns), extrasaction="ignore")
        if not header:
            writer.writeheader()
        writer.writerow({key: "" if value is None else value for key, value in row.items()})


class GenerationClock:
    """A generation's times and memory, collected in the optimiser."""

    def __init__(self, generation: int, workers: int) -> None:
        """Start the generation."""
        self.generation = generation
        self.workers = workers
        self.start = time.time()
        self.started = time.perf_counter_ns()
        self.moves_ns = 0
        self.checkpoint_ns = 0
        self.dispatch_s = 0.0
        self.wait_s = 0.0
        self.busy_s = 0.0
        self.evaluations = 0
        self.worker_rss: dict[int, int] = {}

    def add_evaluation(self, timing: Mapping) -> None:
        """Count one evaluation, with fast_cost's timings when it gave them."""
        self.evaluations += 1
        self.dispatch_s += timing.get("dispatch_s") or 0.0
        self.wait_s += timing.get("wait_s") or 0.0
        self.busy_s += timing.get("busy_s") or 0.0
        self.worker_rss.update(timing.get("worker_rss") or {})

    def row(self, warn=print) -> dict:
        """The generation's row; warns when available memory is below 10% of the total."""
        wall_ns = time.perf_counter_ns() - self.started
        available, total = memory.available_bytes(), memory.total_bytes()
        if memory.is_low(available, total):
            warn(f"WARNING: {memory.low_memory_warning(available, total)}")
        rss = list(self.worker_rss.values())
        wall = wall_ns / NS
        return {
            "generation": self.generation,
            "start_unix_s": round(self.start, 6),
            "end_unix_s": round(time.time(), 6),
            "wall_s": seconds(wall_ns),
            "evaluations": self.evaluations,
            "moves_s": seconds(self.moves_ns),
            "dispatch_s": round(self.dispatch_s, 6),
            "wait_s": round(self.wait_s, 6),
            "checkpoint_s": seconds(self.checkpoint_ns),
            "busy_s": round(self.busy_s, 6),
            "workers": self.workers,
            "utilisation": round(self.busy_s / (self.workers * wall), 4) if wall > 0 else None,
            "parent_rss_mb": megabytes(memory.rss_bytes()),
            "worker_rss_sum_mb": megabytes(sum(rss)) if rss else None,
            "worker_rss_max_mb": megabytes(max(rss)) if rss else None,
            "available_mb": megabytes(available),
        }
