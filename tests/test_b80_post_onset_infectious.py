# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B80: the days a symptomatic case stays infectious after onset are charged (5-9 days)."""

import copy

import numpy as np
import pytest

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo

KEY = "infectious_after_onset_median"


def _cases(vector, demographic_parameters, n):
    from covsyn.model.data_synthesis_main import run_covid
    cases = []
    for seed in range(n):
        _, _, courses, contacts = run_covid(seed, vector.copy(),
                                            copy.deepcopy(demographic_parameters),
                                            save_file=False, mode="result")
        cases.append((courses[0], contacts[0], None))
    return cases


def _after_onset(course):
    return course["latent_period"] + course["infectious_period"] - course["incubation_period"]


def test_target_is_the_literature_band_and_charged():
    """Cheng 2020: exposure up to day 5 after onset infects; Cevik 2021: no culture after day 9."""
    assert fo.OUTCOME_TARGETS[KEY] == (5.0, 9.0, 1.0)


def test_measured_median_is_over_symptomatic_cases_only(run10_vector, demographic_parameters):
    cases = _cases(run10_vector, demographic_parameters, 30)
    symptomatic = [c for c, _, _ in cases if np.isfinite(c["incubation_period"])]
    assert 0 < len(symptomatic) < len(cases)
    expected = float(np.median([_after_onset(c) for c in symptomatic]))
    assert fo.measure_outcomes(cases)[KEY] == pytest.approx(expected)


def test_fast_cost_measures_the_same(run10_vector, demographic_parameters):
    cases = _cases(run10_vector, demographic_parameters, 30)
    scalars = [fast_cost._index_case_scalars(c, x, s) for c, x, s in cases]
    assert fast_cost.reduce_outcomes(scalars)[KEY] == fo.measure_outcomes(cases)[KEY]


def test_no_symptomatic_case_leaves_it_unmeasured(run10_vector, demographic_parameters):
    cases = [(c, x, s) for c, x, s in _cases(run10_vector, demographic_parameters, 30)
             if not np.isfinite(c["incubation_period"])]
    assert cases
    assert np.isnan(fo.measure_outcomes(cases)[KEY])
    scalars = [fast_cost._index_case_scalars(c, x, s) for c, x, s in cases]
    assert np.isnan(fast_cost.reduce_outcomes(scalars)[KEY])


def test_a_short_post_onset_period_is_charged():
    inside = {name: np.nan for name in fo.OUTCOME_TARGETS}
    short = dict(inside, **{KEY: 1.0})
    assert fo.outcome_penalty(short) > fo.outcome_penalty(dict(inside, **{KEY: 7.0}))
