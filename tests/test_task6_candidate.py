# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 6, items 2 and 4: one task scores one candidate, as fast_cost.cost_function does.

Acceptance (doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 6): one evaluation equals
fast_cost.cost_function of the baseline for the same vector and seeds, bit for bit. The task
runs all S simulations of the candidate, reduces them and returns the cost, its parts and its
timings; the read-only inputs come once per worker through the pool initializer.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing

import numpy as np
import pytest

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.cost_parts import LAST as LAST_COST_PARTS
from tests.conftest import REPO_ROOT

TAIWAN = np.load(REPO_ROOT / "variable" / "Taiwan_data_matrix.npy")
COLUMNS = TAIWAN.shape[1]


def _same(a: float, b: float) -> bool:
    """Equal to the bit, NaN equal to NaN."""
    return a == b or (a != a and b != b)


def _baseline(vector, demographic_parameters, cheng_data, pool, offset, simulations=None):
    """fast_cost.cost_function's cost and parts, the baseline Task 6 must reproduce."""
    cost = fast_cost.cost_function(vector, demographic_parameters, pool, *cheng_data,
                                   seed_offset=offset, simulations=simulations)  # fmt: skip
    return cost, dict(LAST_COST_PARTS)


def _in_this_process(
    monkeypatch, demographic_parameters, cheng_data, simulations, taiwan_matrix=TAIWAN
):
    """Initialise this process as a worker, undone after the test."""
    for name in ("_WORKER_DEMOGRAPHIC_PARAMETERS", "_WORKER_TAIWAN_MATRIX_COLUMNS",
                 "_WORKER_STAGE_TIMING", "_WORKER_PROFILE", "_WORKER_PROCESS",
                 "_WORKER_CHENG", "_WORKER_SIMULATIONS", "_WORKER_TAIWAN_MATRIX"):  # fmt: skip
        monkeypatch.setattr(fast_cost, name, getattr(fast_cost, name, None), raising=False)
    fast_cost.init_worker(demographic_parameters, COLUMNS, cheng=cheng_data,
                          simulations=simulations, taiwan_matrix=taiwan_matrix)  # fmt: skip


def test_a_candidate_scores_as_cost_function_does(
    monkeypatch, run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """10 simulations on block 3 give the baseline's cost and every part, to the bit."""
    _in_this_process(monkeypatch, demographic_parameters, cheng_data, 10)
    cost, parts, timing = fast_cost.evaluate_candidate(run10_vector, 30)
    want_cost, want_parts = _baseline(run10_vector, demographic_parameters, cheng_data,
                                      cost_pool, 30, simulations=10)  # fmt: skip
    assert cost == want_cost
    assert parts.keys() == want_parts.keys()
    assert all(_same(parts[k], want_parts[k]) for k in want_parts), [
        k for k in want_parts if not _same(parts[k], want_parts[k])]  # fmt: skip
    assert timing["simulations"] == 10
    assert timing["cost_reduction_s"] > 0 and timing["run_covid_s"] > 0
    assert 0 < timing["start_unix_s"] <= timing["end_unix_s"]
    assert timing["busy_s"] >= timing["run_covid_s"] + timing["case_reduction_s"]


def test_a_candidate_leaves_its_vector_unchanged(
    monkeypatch, run10_vector, demographic_parameters, cheng_data
):
    """The penalty reads the vector the candidate was given, whatever the simulations do."""
    _in_this_process(monkeypatch, demographic_parameters, cheng_data, 3)
    vector = run10_vector.copy()
    fast_cost.evaluate_candidate(vector, 0)
    assert vector.tobytes() == run10_vector.tobytes()


def test_a_failed_candidate_is_charged_and_its_parts_are_nan(
    monkeypatch, demographic_parameters, cheng_data, capsys
):
    """As cost_function: a simulation that raises costs FAILED_EVALUATION_COST (E31)."""
    _in_this_process(monkeypatch, demographic_parameters, cheng_data, 3)
    cost, parts, timing = fast_cost.evaluate_candidate(np.zeros(3), 0)  # too short a vector
    assert cost == fo.FAILED_EVALUATION_COST
    assert parts and all(np.isnan(v) for v in parts.values())
    assert {"cost_contact", "cost_outcome"} <= parts.keys()
    assert timing["simulations"] == 0 and timing["worker_pids"]
    assert "charging" in capsys.readouterr().out


def test_without_inputs_a_worker_says_what_is_missing():
    """A pool not initialised with the objective's inputs fails at once, not with a NaN."""
    fresh = fast_cost._WORKER_CHENG
    try:
        fast_cost._WORKER_CHENG = None
        with pytest.raises(RuntimeError, match="init_worker"):
            fast_cost.evaluate_candidate(np.zeros(3), 0)
    finally:
        fast_cost._WORKER_CHENG = fresh


@pytest.mark.slow
@pytest.mark.parametrize("offset", [0, 300])
def test_one_spawned_task_scores_a_whole_evaluation_as_cost_function_does(
    offset, run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """S = 300 on blocks 0 and 1, in a spawned worker, give the baseline's cost and parts."""
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=fast_cost.init_worker,
        initargs=(demographic_parameters, COLUMNS, False, None, cheng_data, None),
    ) as pool:
        cost, parts, timing = pool.submit(fast_cost.evaluate_candidate, run10_vector,
                                          offset).result()  # fmt: skip
    want_cost, want_parts = _baseline(run10_vector, demographic_parameters, cheng_data,
                                      cost_pool, offset)  # fmt: skip
    assert cost == want_cost
    assert all(_same(parts[k], want_parts[k]) for k in want_parts)
    assert timing["simulations"] == fo.SIMULATIONS_PER_EVALUATION
    assert len(timing["worker_pids"].split(";")) == 1


# The seeds below 3,000 on which each run's best vector gives more than one case, the rare
# branch where a secondary case runs its own course; measured over 3,000 seeds (Task 6). They
# move if run_covid's draws change, which the regression tests forbid as well, or if the run's
# firefly_best.txt or the demographic inputs do.
MULTI_CASE_SEEDS = {
    "phaseD_run10": (26, 129, 1226, 1250, 1375),
    "phaseD_run19": (307, 841, 979, 1032, 1179, 1281, 1375, 1830, 2003),
}


@pytest.mark.parametrize("run", sorted(MULTI_CASE_SEEDS))
def test_simulations_leave_the_demographic_parameters_unchanged(run, demographic_parameters):
    """A candidate's S simulations share one copy, so none may change it for the next (Task 6).

    run_covid runs as _simulate calls it, on one shared copy with stage timing, over seeds 0 to
    299 and the seeds with more than one case. Over 3,000 seeds each of runs 10, 18 and 19 it
    changed nothing; this keeps runs 10 and 19 checked in CI.
    """
    import copy
    import pickle

    from covsyn.model.data_synthesis_main import STAGES, run_covid
    from tests.conftest import best_vector

    shared = copy.deepcopy(demographic_parameters)
    before = pickle.dumps(shared)
    vector = best_vector(REPO_ROOT / "results" / "models" / run / "firefly_best.txt")
    given = vector.copy()
    stage_ns = dict.fromkeys(STAGES, 0)
    sizes = {seed: len(run_covid(seed, vector, shared, save_file=False, stage_times=stage_ns)[2])
             for seed in sorted({*range(300), *MULTI_CASE_SEEDS[run]})}  # fmt: skip
    assert pickle.dumps(shared) == before  # what the test guards, so checked first
    assert vector.tobytes() == given.tobytes()
    assert sum(stage_ns.values()) > 0
    single = [seed for seed in MULTI_CASE_SEEDS[run] if sizes[seed] < 2]
    assert not single, (
        f"{run}: seeds {single} no longer give a secondary case, so that branch went unchecked; "
        f"run_covid's draws or the inputs changed, so measure MULTI_CASE_SEEDS again"
    )


def test_a_worker_reads_the_taiwan_matrix_once_not_per_candidate(
    monkeypatch, run10_vector, demographic_parameters, cheng_data
):
    """The matrix comes with the worker's inputs, so a candidate reads no file for it."""
    _in_this_process(monkeypatch, demographic_parameters, cheng_data, 3)
    load = np.load

    def refusing(path, *args, **kwargs):
        if "Taiwan_data_matrix" in str(path):
            raise AssertionError("read again for a candidate")
        return load(path, *args, **kwargs)

    monkeypatch.setattr(fast_cost.np, "load", refusing)
    cost, _, _ = fast_cost.evaluate_candidate(run10_vector, 0)
    assert cost != fo.FAILED_EVALUATION_COST


def test_a_worker_without_the_matrix_reads_it_from_the_file(
    monkeypatch, run10_vector, demographic_parameters, cheng_data
):
    """The fallback gives the cost the matrix passed in gives, to the bit.

    That a worker given the matrix reads no file for a candidate, the test above shows.
    """
    _in_this_process(monkeypatch, demographic_parameters, cheng_data, 3)
    given = fast_cost.evaluate_candidate(run10_vector, 0)[0]
    _in_this_process(monkeypatch, demographic_parameters, cheng_data, 3, taiwan_matrix=None)
    assert fast_cost._WORKER_TAIWAN_MATRIX is None
    assert fast_cost.evaluate_candidate(run10_vector, 0)[0] == given != fo.FAILED_EVALUATION_COST
