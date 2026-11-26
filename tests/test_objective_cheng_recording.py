# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The objective's index case takes a profile of Cheng's period and is recorded as Cheng did (E84).

Every simulation of the objective draws its index case's profile from the simulation seed
(cheng_recording.draw_profile), gives run_covid the profile's age, and bins every case's
contacts with cheng_recording.case_bins, the index case with its arrival. Both the objective
(fast_cost) and its bit-identity reference (firefly_optimizer.cost_function) do so.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from covsyn.calibration import cheng_recording as cr
from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.cost_parts import LAST as LAST_COST_PARTS
from covsyn.runtime import timings


def _same(a: float, b: float) -> bool:
    return a == b or (a != a and b != b)


def test_a_simulation_takes_its_seeds_profile_and_age(run10_vector, demographic_parameters):
    """The index case's age is the profile's, and the profile is the seed's draw."""
    for seed in range(20):
        (demographic, *_), profile = cr.simulate(
            seed, np.array(run10_vector), copy.deepcopy(demographic_parameters))
        assert profile == cr.draw_profile(seed, cr.default_profiles())
        assert demographic[0]["age"] == profile["age"]


def test_the_objectives_bins_are_the_recorded_ones(
    monkeypatch, run10_vector, demographic_parameters
):
    """Each case's bins in the worker's result are case_bins, the index case's with its profile."""
    monkeypatch.setattr(fast_cost, "_WORKER_DEMOGRAPHIC_PARAMETERS", demographic_parameters)
    seeds = list(range(40, 60))
    clock = timings.TaskClock(False, None)
    results = fast_cost._simulate(seeds, np.array(run10_vector), 10, clock)
    checked = 0
    for seed, (_, cases) in zip(seeds, results, strict=True):
        (_, _, courses, contacts), profile = cr.simulate(
            seed, np.array(run10_vector), copy.deepcopy(demographic_parameters))
        for j, (bins, _, _) in enumerate(cases):
            want = cr.case_bins(courses[j], contacts[j], profile if j == 0 else None)
            assert np.array_equal(bins, want)
            checked += 1
    assert checked >= len(seeds)


def test_the_reference_objective_gives_the_same_cost_and_parts(
    monkeypatch, run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """fast_cost and firefly_optimizer agree to the bit on 30 simulations."""
    monkeypatch.setattr(fo, "SIMULATIONS_PER_EVALUATION", 30)
    reference = fo.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    reference_parts = dict(LAST_COST_PARTS)
    fast = fast_cost.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data,
                                   simulations=30)  # fmt: skip
    fast_parts = dict(LAST_COST_PARTS)
    assert fast == reference
    assert fast_parts.keys() == reference_parts.keys()
    assert [k for k in fast_parts if not _same(fast_parts[k], reference_parts[k])] == []


@pytest.mark.parametrize("seed", [0, 1])
def test_the_model_draws_are_those_of_a_run_with_that_age(
    run10_vector, demographic_parameters, seed
):
    """The profile draw consumes none of the model's random numbers."""
    (out, profile) = cr.simulate(seed, np.array(run10_vector),
                                 copy.deepcopy(demographic_parameters))  # fmt: skip
    from covsyn.model.data_synthesis_main import run_covid

    direct = run_covid(seed, np.array(run10_vector), copy.deepcopy(demographic_parameters),
                       save_file=False, source_age=profile["age"])  # fmt: skip
    assert repr(out) == repr(direct)
