# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Score vectors on seed blocks into a table that a killed pilot resumes from (Task 9).

Both pilots score many (vector, block) pairs at some S. Each finished evaluation is appended to
a CSV and flushed at once, so a pilot killed part way keeps every cost it has and its resume
scores only the rest. The pools hold the objective's inputs once per worker (Task 6), one pool
per S, since a worker's S is set when it starts.
"""

from __future__ import annotations

import concurrent.futures
import csv
import multiprocessing
import os
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from covsyn.runtime import hardware

Evaluate = Callable[[np.ndarray, int], tuple]
MakePool = Callable[[int], concurrent.futures.Executor]


def read_rows(path: Path) -> list[dict[str, str]]:
    """The rows of a CSV written by :func:`score`; none when it does not exist yet."""
    if not Path(path).is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows, all with the first row's keys, to path."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(path: Path, todo: Iterable[Mapping[str, Any]], columns: Sequence[str],
          vector_of: Callable[[Mapping[str, Any]], np.ndarray], evaluate: Evaluate,
          make_pool: MakePool) -> None:
    """Score each task and append it, with its cost, to the CSV at path.

    Args:
        path: The table; created with its header when missing.
        todo: The tasks still to score, each with ``simulations`` and ``seed_offset`` and the
            other columns but ``cost``.
        columns: The table's columns, ``cost`` last.
        vector_of: The parameter vector a task scores.
        evaluate: ``evaluate(P, seed_offset)`` returning the cost first, run in the pool, as
            fast_cost.evaluate_candidate.
        make_pool: The pool for one S, whose workers run S simulations per evaluation.
    """
    todo = list(todo)
    new_file = not Path(path).is_file()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if new_file:
            writer.writeheader()
            f.flush()
        for simulations in dict.fromkeys(t["simulations"] for t in todo):
            group = [t for t in todo if t["simulations"] == simulations]
            with make_pool(simulations) as pool:
                futures = {pool.submit(evaluate, vector_of(t), t["seed_offset"]): t
                           for t in group}  # fmt: skip
                try:
                    for future in concurrent.futures.as_completed(futures):
                        writer.writerow({**futures[future], "cost": float(future.result()[0])})
                        f.flush()  # a killed pilot keeps every finished evaluation
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise


def objective_pools(workers: int | None) -> tuple[Evaluate, MakePool]:
    """The objective, and a spawned pool per S whose workers hold its inputs (Task 6)."""
    from covsyn.calibration import fast_cost

    count = hardware.resolve_workers(workers, os.environ, None)
    with open("./variable/demographic_parameters.pkl", "rb") as f:
        demo = pickle.load(f)
    with open("./variable/processed_contact_tracing_data.pkl", "rb") as f:
        ct = pickle.load(f)
    cheng = (ct["Cheng_contact_array"], ct["Cheng_attack_rate"], ct["norm_weights"])
    taiwan = np.load("./variable/Taiwan_data_matrix.npy")

    def make_pool(simulations: int) -> concurrent.futures.Executor:
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn"),
            initializer=fast_cost.init_worker,
            initargs=(demo, taiwan.shape[1], False, None, cheng, simulations, taiwan))

    return fast_cost.evaluate_candidate, make_pool
