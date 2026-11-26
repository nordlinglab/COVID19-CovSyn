# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The production run's stopping rule: new elite members on fixed monitoring blocks (Task 9).

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 9: every generation, each new
elite member is scored on the same 5 blocks of the monitoring stream, and a member already
scored keeps its score; the run stops when the best monitored mean has not improved by more
than one standard error over 5 generations, or at G_max = 1,000.

Decisions of 2026-11-21: blocks 0-4 of stream 100 at the run's S. After each generation the
best monitored vector so far, the lowest five-block mean (a tie to the one scored first), is
recorded; at generation g >= 5 the run stops when, with d_b the cost on block b of generation
g - 5's best minus that of generation g's, mean(d) <= SD(d) / sqrt(5). When the best is the
same vector, d is 0 and the run stops. Decision of 2026-11-26: the rule applies only from
generation G_plan - 1 on, once the step schedule has run whole (Watch.judge). The fixed blocks
pick the best and judge its progress alike, so this is an operating rule, not a test. The
scores drive the stop only: the elite and the archive keep the optimisation stream's costs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from covsyn.calibration import seed_streams
from covsyn.runtime import runs

STREAM = seed_streams.MONITORING_STREAM
BLOCKS = range(0, 5)
WINDOW = 5  # generations between the snapshots compared
SUMMARY_FILE = "monitoring.json"  # in results/tables/<run-id>/, beside progress.csv
EVENT = "optimiser_stopped"  # the run.log event; no "stage" key, so not a finished stage
# config.json's name for the rule a production run stops by (decision of 2026-11-26): a run
# begun by another, such as the one that stopped run 1 at generation 9, is not resumed.
STOP_RULE = "after_g_plan"


def offsets(simulations: int) -> list[int]:
    """The first seed of each monitoring block at S."""
    return [seed_streams.block_seeds(STREAM, b, simulations).start for b in BLOCKS]


class Monitor:
    """The monitored vectors, their scores, and the best after each generation."""

    def __init__(self, dim: int) -> None:
        """No vector scored yet; vectors have dim parameters."""
        self.dim = dim
        self.vectors: list[np.ndarray] = []
        self.scores: list[np.ndarray] = []
        self.history: list[int] = []  # the best's index after each generation

    def new_vectors(self, vectors: np.ndarray) -> list[np.ndarray]:
        """Those of vectors not yet scored, each once, in order."""
        seen = {v.tobytes() for v in self.vectors}
        found = []
        for vector in np.asarray(vectors, dtype=float):
            if vector.tobytes() not in seen:
                seen.add(vector.tobytes())
                found.append(vector.copy())
        return found

    def add(self, vector: np.ndarray, costs: np.ndarray) -> None:
        """Keep a vector's costs on the monitoring blocks."""
        self.vectors.append(np.asarray(vector, dtype=float).copy())
        self.scores.append(np.asarray(costs, dtype=float).copy())

    def best(self) -> int:
        """The index of the lowest mean so far, a tie to the one scored first."""
        means = [float(np.mean(s)) for s in self.scores]
        return min(range(len(means)), key=lambda i: (means[i], i))

    def record_generation(self) -> None:
        """Note the best after this generation; nothing scored yet notes -1."""
        self.history.append(self.best() if self.scores else -1)

    def stop_reason(self) -> str | None:
        """Why the run stops after the last recorded generation, or None to go on."""
        if len(self.history) <= WINDOW:
            return None
        old, new = self.history[-1 - WINDOW], self.history[-1]
        if old < 0 or new < 0:
            return None
        d = self.scores[old] - self.scores[new]
        se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        improvement = float(np.mean(d))
        if improvement <= se:
            return (f'best monitored mean improved by {improvement:.6g} over {WINDOW} '
                    f'generations, not more than one standard error ({se:.6g})')
        return None

    def state(self) -> dict[str, np.ndarray]:
        """What a checkpoint keeps: each vector with its scores, and the bests."""
        rows = [np.concatenate([v, s]) for v, s in zip(self.vectors, self.scores, strict=True)]
        monitored = np.array(rows, dtype=float).reshape(len(rows), self.dim + len(BLOCKS))
        return {"monitored": monitored, "monitor_best": np.array(self.history, dtype=int)}

    @classmethod
    def from_state(cls, state: dict, dim: int) -> Monitor:
        """The monitor a checkpoint's state describes."""
        monitor = cls(dim)
        for row in np.asarray(state["monitored"], dtype=float):
            monitor.add(row[:dim], row[dim:])
        monitor.history = [int(i) for i in np.asarray(state["monitor_best"])]
        return monitor


class Watch:
    """A production run's monitoring in firefly(): Task 9's stopping rule and its records.

    Unmonitored, it holds nothing and changes nothing, so other runs are as before.
    """

    def __init__(self, on: bool, dim: int, checkpoint: dict | None) -> None:
        """Monitoring when on, from the checkpoint's state when resuming one."""
        self.on = on
        state = checkpoint if on and checkpoint is not None and 'monitored' in checkpoint else None
        self.monitor = Monitor(dim) if state is None else Monitor.from_state(state, dim)
        self.reason = '' if state is None else str(state['stop_reason'])
        self.evaluations = 0 if state is None else int(state['monitor_evaluations'])
        self.seconds = 0.0 if state is None else float(state['monitor_seconds'])

    def stopped(self) -> bool:
        """Whether the rule, or G_max, has stopped the run."""
        return bool(self.reason)

    def score(self, executor, function, members: np.ndarray, simulations: int) -> None:
        """Score the members not yet monitored on the monitoring blocks, one task each."""
        started = time.perf_counter()
        new = self.monitor.new_vectors(members)
        # Every task first, then the results in order, so the pool scores them all at once.
        futures = [[executor.submit(function, vector, offset) for offset in offsets(simulations)]
                   for vector in new]
        for vector, tasks in zip(new, futures, strict=True):
            self.monitor.add(vector, np.array([float(f.result()[0]) for f in tasks]))
            self.evaluations += len(tasks)
        self.seconds += time.perf_counter() - started

    def judge(self, generation: int, max_generations: int, g_plan: int) -> None:
        """Record the generation's best, and the reason to stop, if any.

        The rule applies from generation G_plan - 1 on, so the step schedule runs whole
        (decision of 2026-11-26): pilot 2 chose n and alpha_0 at a budget of G_plan
        generations, and the rule alone stopped production run 1 at generation 9 of 415,
        when its best had stayed one vector for 5 generations.
        """
        self.monitor.record_generation()
        reason = self.monitor.stop_reason() if generation >= g_plan - 1 else None
        if reason is None and generation == max_generations - 1:
            reason = f'reached G_max = {max_generations} generations'
        self.reason = reason or ''

    def state(self) -> dict | None:
        """The checkpoint's monitoring entries; None when not monitoring."""
        if not self.on:
            return None
        return {**self.monitor.state(), 'stop_reason': np.array(self.reason),
                'monitor_evaluations': self.evaluations, 'monitor_seconds': self.seconds}

    def log_stop(self, run_dir: Path, generation: int) -> None:
        """Append the stop to run.log, unless a line already records it (Task 9)."""
        path = Path(run_dir) / 'run.log'
        if any(line.get('event') == EVENT and line.get('generation') == generation
               for line in runs.json_lines(path)):
            return
        runs.append_json_line(path, {'event': EVENT, 'generation': generation,
                                     'reason': self.reason,
                                     'time': datetime.now().isoformat(timespec='seconds')})

    def write_summary(self, tables: Path, simulations: int) -> None:
        """monitoring.json: what the monitoring cost, apart from the optimisation's."""
        summary = {'vectors': len(self.monitor.vectors), 'evaluations': self.evaluations,
                   'simulations': self.evaluations * simulations, 'seconds': self.seconds,
                   'stop_reason': self.reason}
        (Path(tables) / SUMMARY_FILE).write_text(json.dumps(summary, indent=1) + '\n')
