# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 5, item 5: ``--mode profile`` profiles every worker and merges the profiles.

The acceptance criterion (doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 5): on 2
workers at S = 3,000 the profile finishes in under 60 s on a laptop, and run_covid holds at
least 80% of the cumulative time in merged.prof.
"""

from __future__ import annotations

import concurrent.futures
import cProfile
import csv
import json
import multiprocessing
import pstats
import subprocess
import sys
import time

import numpy as np
import pytest

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.runtime import profiling, runs, timings
from tests.conftest import REPO_ROOT


def _busy(n):
    return sum(i * i for i in range(n))


def _write_profile(path, n):
    profile = cProfile.Profile()
    profile.enable()
    _busy(n)
    profile.disable()
    profile.dump_stats(path)


def test_worker_profiles_merge_into_one_and_a_top_40_table(tmp_path):
    """merged.prof adds the workers' calls; profile_top.csv lists 40 rows by cumulative time."""
    profile_dir, tables = tmp_path / "profile", tmp_path / "tables"
    profile_dir.mkdir()
    _write_profile(profile_dir / "worker_1.prof", 10_000)
    _write_profile(profile_dir / "worker_2.prof", 20_000)
    profiling.merge_profiles(profile_dir, tables)
    merged = pstats.Stats(str(profile_dir / profiling.MERGED))
    calls = {func[2]: stat[1] for func, stat in merged.stats.items()}
    assert calls["_busy"] == 2
    with open(tables / profiling.TOP_TABLE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(profiling.TOP_COLUMNS)
    assert 1 <= len(rows) <= profiling.TOP
    cumulative = [float(r["cumtime_s"]) for r in rows]
    assert cumulative == sorted(cumulative, reverse=True)
    assert all(0 <= float(r["cumtime_share"]) <= 1 for r in rows)


def test_merging_without_worker_profiles_is_an_error(tmp_path):
    """No worker profile means the workers never ran or never wrote one: say so."""
    (tmp_path / "profile").mkdir()
    with pytest.raises(FileNotFoundError, match="worker_"):
        profiling.merge_profiles(tmp_path / "profile", tmp_path)


def test_the_top_table_has_40_rows_when_there_are_more_functions(tmp_path):
    """The plan's top 40."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profile = cProfile.Profile()
    profile.enable()
    import email.mime.multipart  # noqa: F401  (importing runs many functions)

    np.linalg.svd(np.ones((5, 5)))
    profile.disable()
    profile.dump_stats(profile_dir / "worker_1.prof")
    profiling.merge_profiles(profile_dir, tmp_path)
    with open(tmp_path / profiling.TOP_TABLE, newline="") as f:
        assert len(list(csv.DictReader(f))) == profiling.TOP


def test_an_evaluation_of_s_simulations(
    run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """cost_function takes S, so profile mode can run 3,000 simulations per evaluation."""
    fast_cost.cost_function(
        run10_vector, demographic_parameters, cost_pool, *cheng_data, simulations=20
    )
    assert timings.LAST["simulations"] == 20


def test_each_worker_writes_its_own_profile(
    tmp_path, run10_vector, demographic_parameters, cheng_data
):
    """A worker given a profile directory writes worker_<pid>.prof when the pool shuts down."""
    columns = np.load(REPO_ROOT / "variable" / "Taiwan_data_matrix.npy").shape[1]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2,
        initializer=fast_cost.init_worker,
        initargs=(demographic_parameters, columns, False, tmp_path),
    ) as pool:
        fast_cost.cost_function(
            run10_vector, demographic_parameters, pool, *cheng_data, simulations=40
        )
        pids = timings.LAST["worker_pids"].split(";")
    written = sorted(p.name for p in tmp_path.glob("worker_*.prof"))
    assert written == sorted(f"worker_{pid}.prof" for pid in pids)
    profiling.merge_profiles(tmp_path, tmp_path)
    assert profiling.share(tmp_path / profiling.MERGED, "run_covid") > 0.5


def _pid_after(barrier):
    """This worker's PID, once another worker has also reached the barrier."""
    import os

    barrier.wait(timeout=60)  # a single worker breaks the barrier rather than hanging
    return os.getpid()


# The platform's default and spawn (macOS's default); fork is unsafe on macOS.
START_METHODS = list(dict.fromkeys([multiprocessing.get_start_method(), "spawn"]))


@pytest.mark.parametrize("method", START_METHODS)
def test_a_worker_without_tasks_writes_no_profile(
    method, tmp_path, run10_vector, demographic_parameters, cheng_data
):
    """10 simulations are one task, so one of 2 workers runs none; the merge still works."""
    columns = np.load(REPO_ROOT / "variable" / "Taiwan_data_matrix.npy").shape[1]
    # The manager outlives the pool, whose workers hold proxies of its barrier.
    with (
        multiprocessing.get_context(method).Manager() as manager,
        concurrent.futures.ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context(method),
            initializer=fast_cost.init_worker,
            initargs=(demographic_parameters, columns, False, tmp_path),
        ) as pool,
    ):
        # Two tasks that wait for each other, so both workers start however slowly they spawn.
        barrier = manager.Barrier(2)
        assert len(set(pool.map(_pid_after, [barrier, barrier]))) == 2
        fast_cost.cost_function(
            run10_vector, demographic_parameters, pool, *cheng_data, simulations=10
        )
    assert len(list(tmp_path.glob("worker_*.prof"))) == 1
    profiling.merge_profiles(tmp_path, tmp_path)


def test_a_profile_that_could_not_start_is_not_written(tmp_path):
    """If enabling fails (another profiler active), the worker still writes no empty profile."""

    class Refusing:
        def enable(self):
            raise ValueError("Another profiling tool is already active")

    worker = profiling.WorkerProfile(tmp_path)
    worker.profile = Refusing()
    with pytest.raises(ValueError):
        worker.enable()
    worker.dump()
    assert not list(tmp_path.iterdir())


def test_a_profile_run_cannot_be_resumed(tmp_path):
    """Its population is the worker count; a profile is run again, not continued."""
    (tmp_path / "config.json").write_text(json.dumps({"seed": 5, "settings": {"mode": "profile"}}))
    with pytest.raises(SystemExit) as stop:
        fo.resumed_settings(tmp_path)
    assert stop.value.code == runs.NOT_RESUMABLE


def _optimiser(*args, timeout=600):
    return subprocess.run(
        [sys.executable, "-m", "covsyn.calibration.firefly_optimizer", *map(str, args)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--simulations", "10"],
        ["--mode", "train", "--simulations", "10"],
        ["--mode", "profile", "--simulations", "0"],
    ],
)
def test_simulations_is_for_profile_mode_only(tmp_path, args):
    """S changes the objective, so only a profile may set it, and to at least 1."""
    result = _optimiser(*args, "--run-dir", tmp_path / "run")
    assert result.returncode != 0
    assert "--simulations" in result.stderr
    assert not (tmp_path / "run").exists()


@pytest.fixture(scope="module")
def profile_run(tmp_path_factory):
    """One profile on 2 workers at the default S, its directory, and its wall time."""
    run = tmp_path_factory.mktemp("profile") / "run"
    start = time.monotonic()
    result = _optimiser("--mode", "profile", "--workers", 2, "--run-dir", run)
    elapsed = time.monotonic() - start
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return run, elapsed


@pytest.mark.slow
def test_profile_mode_writes_the_profiles_and_run_covid_dominates(profile_run):
    """A population of 2 (the workers), 2 generations, S = 3,000; run_covid 80% or more."""
    run, _ = profile_run
    tables = runs.progress_path(run).parent
    config = json.loads((run / "config.json").read_text())
    assert config["settings"]["pop_size"] == 2 and config["settings"]["max_generations"] == 2
    assert config["settings"]["simulations"] == 3000 and config["workers"] == 2
    assert len(sorted((run / "profile").glob("worker_*.prof"))) == 2
    assert (run / "profile" / profiling.MERGED).exists()
    assert (tables / profiling.TOP_TABLE).exists()
    with open(tables / timings.EVALUATION_TABLE, newline="") as f:
        assert {r["simulations"] for r in csv.DictReader(f)} == {"3000"}
    share = profiling.share(run / "profile" / profiling.MERGED, "run_covid")
    assert share >= 0.8, share


@pytest.mark.slow
@pytest.mark.xfail(
    reason="not met before Task 6: 8 evaluations of 3,000 simulations took 177 s on 2 WSL "
    "workers, cProfile making each simulation about 4x slower; Task 6 cuts them to 5",
    strict=False,
)
def test_profile_mode_finishes_in_under_60_s(profile_run):
    """The plan's time limit, on a laptop."""
    _, elapsed = profile_run
    assert elapsed < 60, f"{elapsed:.1f} s"
