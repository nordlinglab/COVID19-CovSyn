# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 5, items 1, 3 and 4: the evaluation and generation timing tables.

The optimiser runs with the deterministic stand-in objective of the Task 4 tests, which fills
only the optimiser's own columns, except where a stand-in returns the timing fields fast_cost's
task does; fast_cost.cost_function and evaluate_candidate, which fill the workers' columns, are
checked on their own with the real objective.
"""

from __future__ import annotations

import concurrent.futures
import csv

import numpy as np
import psutil
import pytest

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.data_synthesis_main import STAGES
from covsyn.runtime import runs, timings
from tests import firefly_standins as standins
from tests.conftest import REPO_ROOT
from tests.test_task4_resume import (
    DIM,
    GENERATIONS,
    POP,
    SEED,
    Killed,
    killed_in,
    objective,
    optimise,
)


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


def test_a_run_fills_the_worker_columns_and_drops_no_row_key(tmp_path, monkeypatch):
    """Every row has its worker columns, and no row given to the evaluation table has worker_rss.

    No table drops a key of a row for want of a column in its header.
    """
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "whole"
    path.mkdir()
    rows_given_to_write, dropped_keys, write = [], set(), timings.Table.write

    def recording(self, row):
        """Keep each row given to the evaluation table, write it, and keep any key dropped."""
        if self.path.name == timings.EVALUATION_TABLE:
            rows_given_to_write.append(dict(row))
        write(self, row)
        dropped_keys.update(self.left_out)

    monkeypatch.setattr(timings.Table, "write", recording)
    optimise(path, standins.with_worker_fields)
    assert rows_given_to_write
    assert all("worker_rss" not in row for row in rows_given_to_write)
    assert not dropped_keys
    worker_columns = ("worker_rss_before_mb", "worker_rss_after_mb", "worker_peak_mb")
    evaluation_rows = table(path, timings.EVALUATION_TABLE)
    generation_rows = table(path, timings.GENERATION_TABLE)
    assert evaluation_rows and generation_rows
    for r in evaluation_rows:
        assert r["worker_pids"] and all(float(r[c]) > 0 for c in worker_columns)
    assert all(float(r["worker_rss_max_mb"]) > 0 for r in generation_rows)


def test_a_resumed_run_keeps_one_row_per_evaluation_and_generation(tmp_path, run):
    """Rows written after the last checkpoint are dropped, so none is counted twice."""
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_in(2))
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
        optimise(path, killed_in(2))
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
        optimise(path, killed_in(2))
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


def test_a_row_with_other_keys_is_written_under_the_tables_header(tmp_path, capsys):
    """A failed evaluation's cost parts differ; its row lines up, and a lost column is said."""
    path = tmp_path / "progress.csv"
    first = timings.Table(path, ["generation", "a", "b"])
    first.write({"generation": 0, "a": 1, "b": 2})
    first.close()
    again = timings.Table(path)  # a resumed run's table, over the header already there
    again.write({"generation": 1, "b": 3, "c": 4})
    again.write({"generation": 2, "b": 5, "c": 6})
    again.close()
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["generation", "a", "b"], ["0", "1", "2"], ["1", "", "3"], ["2", "", "5"]]
    assert capsys.readouterr().err.count("'c'") == 1  # said once, not per row
    fo.drop_progress_after(path, 2)
    with open(path, newline="") as f:
        assert len(list(csv.reader(f))) == 4


def test_a_table_stays_open_and_each_row_is_on_disk_at_once(tmp_path):
    """One open per run, not per row (13 ms each on a Windows drive under WSL)."""
    path = tmp_path / "t.csv"
    table = timings.Table(path, ["generation", "x"])
    table.write({"generation": 0, "x": 1})
    opened = table.file
    table.write({"generation": 1, "x": 2})
    assert table.file is opened
    assert path.read_text().splitlines() == ["generation,x", "0,1", "1,2"]
    table.close()
    table.close()  # twice is harmless


def test_a_header_cut_off_by_a_kill_is_written_again_on_resume(tmp_path, run):
    """The resume drops the cut header, and the table starts again under a whole one."""
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_in(2))
    generation_table = runs.progress_path(path).parent / timings.GENERATION_TABLE
    generation_table.write_text("generation,start_un")  # the header, cut off
    optimise(path, objective, resume=True)
    rows = table(path, timings.GENERATION_TABLE)
    assert list(rows[0]) == list(timings.GENERATION_COLUMNS)


def test_an_attempt_without_a_checkpoint_leaves_no_timing_tables(tmp_path):
    """Starting the optimiser afresh discards the killed attempt's tables, cut rows and all."""
    from covsyn.pipeline import stages as ps

    run = tmp_path / "results" / "models" / "20261115-090000_b30c141_phase-d"
    run.mkdir(parents=True)
    tables = runs.progress_path(run).parent
    tables.mkdir(parents=True)
    for name in ("progress.csv", timings.EVALUATION_TABLE, timings.GENERATION_TABLE):
        (tables / name).write_text("generation,start_un")
    ps.discard_optimiser_output(run)
    assert list(tables.iterdir()) == []


def test_a_run_that_fails_closes_its_tables(tmp_path, monkeypatch):
    """However the run ends, its three tables are closed."""
    closed = []
    close = timings.Table.close

    def recording(self):
        closed.append(self.path.name)
        close(self)

    monkeypatch.setattr(timings.Table, "close", recording)
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_in(1))
    assert sorted(closed) == sorted(["progress.csv", *runs.TIMING_TABLES])


def test_a_run_that_fails_shuts_its_pool_down(tmp_path, monkeypatch):
    """No pool of workers is left running behind a failed run."""
    import concurrent.futures

    shut = []
    shutdown = concurrent.futures.ProcessPoolExecutor.shutdown

    def recording(self, *args, **kwargs):
        shut.append((self, kwargs))  # a failed run cancels what has not started
        return shutdown(self, *args, **kwargs)

    monkeypatch.setattr(concurrent.futures.ProcessPoolExecutor, "shutdown", recording)
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_in(1))
    assert shut and shut[-1][1] == {"wait": True, "cancel_futures": True}


def test_a_table_that_fails_to_close_neither_hides_the_error_nor_leaves_others_open(
    tmp_path, monkeypatch
):
    """Every table is closed, and the run's own exception is the one raised."""
    closed = []
    close = timings.Table.close

    def failing(self):
        closed.append(self.path.name)
        close(self)
        if self.path.name == "progress.csv":
            raise OSError("No space left on device")

    monkeypatch.setattr(timings.Table, "close", failing)
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_in(1))
    assert sorted(closed) == sorted(["progress.csv", *runs.TIMING_TABLES])


def test_a_finished_run_raises_a_table_that_failed_to_close(tmp_path, monkeypatch):
    """Without a failure of its own, a run whose table could not be closed says so."""
    close = timings.Table.close
    closed = []

    def failing(self):
        closed.append(self.path.name)
        close(self)
        if self.path.name == "progress.csv":
            raise OSError("No space left on device")

    monkeypatch.setattr(timings.Table, "close", failing)
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "whole"
    path.mkdir()
    with pytest.raises(OSError, match="No space"):
        optimise(path, objective)
    assert sorted(closed) == sorted(["progress.csv", *runs.TIMING_TABLES])


def test_close_tables_closes_all_whatever_one_raises(tmp_path, capsys):
    """Not only an OSError: any error of one close leaves the others to be closed."""

    class Bad:
        def close(self):
            raise ValueError("I/O operation on closed file")

    good = timings.Table(tmp_path / "t.csv", ["generation"])
    good.write({"generation": 0})
    timings.close_tables([Bad(), good], failing=True)
    assert good.file is None
    assert "could not close" in capsys.readouterr().err  # said, as it is not raised
    with pytest.raises(ValueError):
        timings.close_tables([Bad()])


def test_a_second_close_error_is_said(tmp_path, capsys):
    """Only the first is raised; a later one is not lost without a word."""

    class Bad:
        path = "bad.csv"

        def close(self):
            raise OSError("No space left on device")

    with pytest.raises(OSError):
        timings.close_tables([Bad(), Bad()])
    assert capsys.readouterr().err.count("could not close bad.csv") == 1


class SecondInterrupt(BaseException):
    """Stands in for a second Ctrl-C.

    A KeyboardInterrupt that got through would stop the whole pytest session; this
    BaseException, which pytest turns into one test failure, would not.
    """


def test_a_second_interrupt_after_the_shutdown_keeps_the_runs_error(tmp_path, monkeypatch):
    """An interrupt raised by a failed run's shutdown, once it has waited: the run's error."""
    import concurrent.futures
    import inspect

    shutdown = concurrent.futures.ProcessPoolExecutor.shutdown
    signature = inspect.signature(shutdown)
    interrupts = []

    def interrupted(self, *args, **kwargs):
        shutdown(self, *args, **kwargs)  # the workers are joined, then the second interrupt
        call = signature.bind(self, *args, **kwargs)
        call.apply_defaults()  # what the call means, not how it is spelt
        if call.arguments["cancel_futures"]:
            interrupts.append((call.arguments["wait"], call.arguments["cancel_futures"]))
            raise SecondInterrupt

    monkeypatch.setattr(concurrent.futures.ProcessPoolExecutor, "shutdown", interrupted)
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_in(1))
    assert interrupts == [(True, True)]  # it was raised, and kept out


def test_after_a_failure_every_close_error_is_said_and_none_raised(capsys):
    """The run's own error stays the one raised; each table that failed to close is named."""

    class Bad:
        def __init__(self, name):
            self.path = name

        def close(self):
            raise OSError("No space left on device")

    timings.close_tables([Bad("a.csv"), Bad("b.csv")], failing=True)
    err = capsys.readouterr().err
    assert "could not close a.csv" in err and "could not close b.csv" in err


def _gone(pid):
    """Ended and reaped: a zombie, a worker not joined, counts as still there."""
    try:
        return not psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return True


def test_a_failed_run_waits_for_its_running_task_and_leaves_no_worker(tmp_path, monkeypatch):
    """A task running when the run fails is finished, and every worker the pool started ends.

    Candidate 0 fails once another candidate's task runs; that task outlasts the run's cleanup
    by far (2 s), so a shutdown that did not wait would return before it finished, on a slow
    drive too.
    """
    import functools

    flags = tmp_path / "flags"
    flags.mkdir()
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "killed"
    path.mkdir()
    # Candidate 0 of the initial population, which optimise draws on [0, 1] from SEED.
    first = float(fo.Firefly(pop_size=POP, alpha=1, betamin=1, gamma=0.03,
                             seed=SEED).rng.uniform(0, 1, (POP, DIM))[0, 0])  # fmt: skip
    failing = functools.partial(standins.fails_while_another_runs, failing_first_value=first,
                                flags=str(flags))  # fmt: skip
    with pytest.raises(Killed):
        optimise(path, failing)
    ran = {p.name.removeprefix("started_") for p in flags.glob("started_*")}
    assert ran and ran == {p.name.removeprefix("finished_") for p in flags.glob("finished_*")}
    workers = {int(p.read_text()) for pattern in ("started_*", "failed_*")
               for p in flags.glob(pattern)}  # fmt: skip
    assert len(workers) == 2  # the failing task's worker and another
    assert all(_gone(pid) for pid in workers)
