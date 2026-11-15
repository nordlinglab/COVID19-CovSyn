# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 5, items 1, 3 and 4: the evaluation and generation timing tables.

The optimiser runs with the deterministic stand-in objective of the Task 4 tests, which fills
only the optimiser's own columns; fast_cost.cost_function, which fills the workers' columns,
is checked on its own with the real objective.
"""

from __future__ import annotations

import concurrent.futures
import csv

import numpy as np
import pytest

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.data_synthesis_main import STAGES
from covsyn.runtime import runs, timings
from tests.conftest import REPO_ROOT
from tests.test_task4_resume import GENERATIONS, POP, Killed, killed_after, objective, optimise


def table(path, name):
    """A timing table's rows."""
    with open(runs.progress_path(path).parent / name, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def run(tmp_path, monkeypatch):
    """One uninterrupted run with the stand-in objective, and its directory."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "whole"
    path.mkdir()
    optimise(path, objective)
    return path


def evaluations(path):
    """The evaluations the run's last checkpoint counts."""
    return int(fo.load_last_checkpoint(path)["evaluations"])


def test_one_row_per_evaluation(run):
    """Every evaluation, the initial population's as generation -1, in the plan's columns."""
    rows = table(run, timings.EVALUATION_TABLE)
    assert list(rows[0]) == list(timings.EVALUATION_COLUMNS)
    assert len(rows) == evaluations(run)
    initial = [r for r in rows if r["generation"] == "-1"]
    assert [int(r["candidate"]) for r in initial] == list(range(POP))
    assert {int(r["generation"]) for r in rows} == set(range(-1, GENERATIONS))
    assert all(0 <= int(r["candidate"]) < POP for r in rows)
    assert all(float(r["end_unix_s"]) >= float(r["start_unix_s"]) for r in rows)


def test_one_row_per_generation(run):
    """The initial population and every generation, with the plan's times and memory."""
    rows = table(run, timings.GENERATION_TABLE)
    assert list(rows[0]) == list(timings.GENERATION_COLUMNS)
    assert [int(r["generation"]) for r in rows] == list(range(-1, GENERATIONS))
    per_generation = {}
    for r in table(run, timings.EVALUATION_TABLE):
        per_generation[r["generation"]] = per_generation.get(r["generation"], 0) + 1
    for r in rows:
        assert int(r["evaluations"]) == per_generation[r["generation"]]
        assert int(r["workers"]) == 2
        parts = float(r["moves_s"]) + float(r["checkpoint_s"])
        assert 0 <= parts <= float(r["wall_s"])
        assert 0 <= float(r["utilisation"]) <= 1
        assert float(r["parent_rss_mb"]) > 0 and float(r["available_mb"]) > 0
    assert all(float(r["moves_s"]) > 0 for r in rows if r["generation"] != "-1")
    assert all(float(r["checkpoint_s"]) > 0 for r in rows)


def test_a_resumed_run_keeps_one_row_per_evaluation_and_generation(tmp_path, run):
    """Rows written after the last checkpoint are dropped, so none is counted twice."""
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_after(3 * POP + 5))
    optimise(path, objective, resume=True)
    for name, keys in (
        (timings.EVALUATION_TABLE, ("generation", "candidate")),
        (timings.GENERATION_TABLE, ("generation", "evaluations")),
    ):
        got = [tuple(r[k] for k in keys) for r in table(path, name)]
        want = [tuple(r[k] for k in keys) for r in table(run, name)]
        assert got == want


def test_low_available_memory_is_warned_in_the_generation_row(monkeypatch):
    """Below 10% of the total, the row comes with a warning."""
    monkeypatch.setattr(timings.memory, "available_bytes", lambda: 5)
    monkeypatch.setattr(timings.memory, "total_bytes", lambda: 100)
    warnings = []
    timings.GenerationClock(0, 1).row(warn=warnings.append)
    assert len(warnings) == 1 and "below 10%" in warnings[0]


def test_tasks_are_summed_and_memory_is_the_largest_worker():
    """Two workers, three tasks: times add, RSS before is a worker's first, after its last."""

    def task(pid, before, after, peak, ns):
        return {
            "pid": pid, "start": 0.0, "busy_ns": ns, "simulations": 10,
            "run_covid_ns": ns // 2, "reduction_ns": ns // 4, "stage_ns": None,
            "rss_before": before, "rss_after": after, "peak": peak,
        }  # fmt: skip

    fields = timings.summarise_tasks(
        [task(1, 100e6, 110e6, 120e6, 10**9), task(2, 200e6, 150e6, 300e6, 10**9),
         task(1, 110e6, 130e6, 140e6, 10**9)]
    )  # fmt: skip
    assert fields["worker_pids"] == "1;2"
    assert fields["simulations"] == 30
    assert fields["busy_s"] == 3.0 and fields["run_covid_s"] == 1.5
    assert fields["worker_rss_before_mb"] == 200.0
    assert fields["worker_rss_after_mb"] == 150.0
    assert fields["worker_peak_mb"] == 300.0
    assert fields["worker_rss"] == {1: 130e6, 2: 150e6}
    assert all(fields[f"{stage}_s"] is None for stage in STAGES)


def _pool(demographic_parameters, **options):
    columns = np.load(REPO_ROOT / "variable" / "Taiwan_data_matrix.npy").shape[1]
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=2,
        initializer=fast_cost.init_worker,
        initargs=(demographic_parameters, columns, *options.values()),
    )


@pytest.mark.slow
@pytest.mark.parametrize("stage_timing", [False, True])
def test_the_real_objective_fills_the_worker_columns(
    stage_timing, run10_vector, demographic_parameters, cheng_data
):
    """fast_cost.cost_function times its tasks and the workers' memory; the cost is unchanged."""
    with _pool(demographic_parameters) as plain_pool:
        plain = fast_cost.cost_function(
            run10_vector, demographic_parameters, plain_pool, *cheng_data
        )
    with _pool(demographic_parameters, stage_timing=stage_timing) as pool:
        timings.LAST.clear()
        cost = fast_cost.cost_function(run10_vector, demographic_parameters, pool, *cheng_data)
    assert cost == plain
    last = timings.LAST
    assert last["simulations"] == fo.SIMULATIONS_PER_EVALUATION
    assert 1 <= len(last["worker_pids"].split(";")) <= 2
    for key in ("run_covid_s", "case_reduction_s", "cost_reduction_s", "dispatch_s", "wait_s"):
        assert last[key] > 0, key
    assert last["busy_s"] >= last["run_covid_s"] + last["case_reduction_s"]
    assert 0 < last["worker_rss_after_mb"] <= last["worker_peak_mb"]
    assert last["worker_rss_before_mb"] > 0
    stage_values = [last[f"{stage}_s"] for stage in STAGES]
    if stage_timing:
        assert all(v > 0 for v in stage_values)
        assert sum(stage_values) <= last["run_covid_s"]
    else:
        assert all(v is None for v in stage_values)


@pytest.mark.parametrize("cut_off", ["0", "-", "0,3,17"])  # "-" is "-1" cut off
def test_a_row_cut_off_by_a_kill_is_dropped_on_resume(tmp_path, run, cut_off):
    """A kill while writing leaves a row without its newline; resuming drops it, whatever it is."""
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_after(3 * POP + 5))
    evaluation_table = runs.progress_path(path).parent / timings.EVALUATION_TABLE
    with open(evaluation_table, "a", newline="") as f:
        f.write(cut_off)
    optimise(path, objective, resume=True)
    keys = ("generation", "candidate")
    got = [tuple(r[k] for k in keys) for r in table(path, timings.EVALUATION_TABLE)]
    want = [tuple(r[k] for k in keys) for r in table(run, timings.EVALUATION_TABLE)]
    assert got == want


def test_dropping_rows_keeps_csv_line_rules(tmp_path):
    """A cell holding a form feed or a line separator stays one cell, as csv reads it."""
    path = tmp_path / "t.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([["generation", "note"], [0, "a\x0cb"], [1, "c d"], [2, "e"]])
    fo.drop_progress_after(path, 1)
    with open(path, newline="") as f:
        assert list(csv.reader(f)) == [["generation", "note"], ["0", "a\x0cb"], ["1", "c d"]]


def test_dropping_every_row_keeps_the_header(tmp_path):
    """The header stays, so rows appended after it are still read by name."""
    path = tmp_path / "t.csv"
    path.write_text("generation,best_cost\n3,1.0\n")
    fo.drop_progress_after(path, 1)
    assert path.read_bytes() == b"generation,best_cost\r\n"


def test_a_progress_table_left_empty_gets_its_header_again(tmp_path, monkeypatch):
    """A table a kill cut down to nothing is written with its header when generation > 0."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_after(3 * POP + 5))
    runs.progress_path(path).write_text("genera")  # the header, cut off
    optimise(path, objective, resume=True)
    with open(runs.progress_path(path), newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and "best_cost" in rows[0]


def test_dropping_rows_skips_rows_that_are_not_whole(tmp_path):
    """A row short of cells, or without a generation, is dropped rather than kept or fatal."""
    path = tmp_path / "t.csv"
    path.write_text("generation,best_cost\n0,1.0\n0\n,2.0\n1,3.0,9\n1,4.0\n")
    fo.drop_progress_after(path, 1)
    with open(path, newline="") as f:
        assert list(csv.reader(f)) == [["generation", "best_cost"], ["0", "1.0"], ["1", "4.0"]]


def test_a_row_with_other_keys_is_written_under_the_tables_header(tmp_path):
    """A failed evaluation's cost parts differ; its row still lines up with the header."""
    path = tmp_path / "progress.csv"
    timings.append_row(path, ["generation", "a", "b"], {"generation": 0, "a": 1, "b": 2})
    timings.append_row(path, ["generation", "b", "c"], {"generation": 1, "b": 3, "c": 4})
    with open(path, newline="") as f:
        assert list(csv.reader(f)) == [["generation", "a", "b"], ["0", "1", "2"], ["1", "", "3"]]
    fo.drop_progress_after(path, 1)
    with open(path, newline="") as f:
        assert len(list(csv.reader(f))) == 3


def test_a_new_table_replaces_what_was_there(tmp_path):
    """Generation 0 starts the progress table afresh."""
    path = tmp_path / "progress.csv"
    path.write_text("old,header\n1,2\n")
    timings.append_row(path, ["generation"], {"generation": 0}, new=True)
    assert path.read_bytes() == b"generation\r\n0\r\n"
