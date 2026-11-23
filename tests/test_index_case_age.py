# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""run_covid can be given the index case's age (E84, user decision 2026-11-23).

The calibration's index case takes the age of a case from Cheng et al. (2020)'s period instead
of a random Taiwan resident's. Without the argument run_covid draws it from the population, as
before; the regression digests pin that path.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from covsyn.model.data_synthesis_main import run_covid


def _run(seed, vector, demo, **kwargs):
    return run_covid(seed, np.array(vector), copy.deepcopy(demo), save_file=False, **kwargs)


@pytest.mark.parametrize("age", [0, 33, 100])
def test_the_index_case_gets_the_given_age(run10_vector, demographic_parameters, age):
    """Every seed's first case has exactly that age."""
    for seed in range(5):
        demographic, *_ = _run(seed, run10_vector, demographic_parameters, source_age=age)
        assert demographic[0]["age"] == age


def test_no_age_is_the_old_call(run10_vector, demographic_parameters):
    """source_age=None draws from the population exactly as the call without it."""
    for seed in range(3):
        old = _run(seed, run10_vector, demographic_parameters)
        new = _run(seed, run10_vector, demographic_parameters, source_age=None)
        assert repr(old) == repr(new)


@pytest.mark.parametrize("age", [-1, 101, 33.5, float("nan")])
def test_an_age_outside_the_tables_is_refused(run10_vector, demographic_parameters, age):
    """The age tables run from 0 to 100 whole years."""
    with pytest.raises(ValueError, match="source_age"):
        _run(0, run10_vector, demographic_parameters, source_age=age)
