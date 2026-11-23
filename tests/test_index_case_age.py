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


def _profiles(n):
    return [{"case_id": 1000 + k, "age": 20 + k % 60, "imported": k % 2 == 0,
             "airport": False, "arrival_from_onset": 2 if k % 2 == 0 else None}
            for k in range(n)]


def test_each_source_case_takes_its_own_profile(run10_vector, demographic_parameters):
    """cheng2020 mode's 100 index cases: each gets one profile, its age and its record."""
    profiles = _profiles(100)
    demographic, _, courses, _ = _run(3, run10_vector, demographic_parameters,
                                      mode="cheng2020", source_profiles=profiles)
    tagged = [(c["index_profile"], d["age"]) for c, d in zip(courses, demographic, strict=True)
              if "index_profile" in c]
    assert sorted(p["case_id"] for p, _ in tagged) == [p["case_id"] for p in profiles]
    assert all(age == p["age"] for p, age in tagged)


def test_without_profiles_the_course_records_have_no_profile_key(
    run10_vector, demographic_parameters
):
    """The saved course data keep their old keys, so earlier output is unchanged."""
    _, _, courses, _ = _run(3, run10_vector, demographic_parameters, mode="cheng2020")
    assert all("index_profile" not in c for c in courses)


def test_one_profile_per_source_case_is_required(run10_vector, demographic_parameters):
    """99 profiles for 100 source cases is an error, not a partial assignment."""
    with pytest.raises(ValueError, match="source_profiles"):
        _run(0, run10_vector, demographic_parameters, mode="cheng2020",
             source_profiles=_profiles(99))


def test_an_age_and_profiles_together_are_refused(run10_vector, demographic_parameters):
    """The profiles carry the ages."""
    with pytest.raises(ValueError, match="source_profiles"):
        _run(0, run10_vector, demographic_parameters, source_age=30,
             source_profiles=_profiles(1))


def test_the_cheng2020_synthesis_gives_its_index_cases_chengs_cohort():
    """Fig 6's mode: Cheng's 100 profiles, drawn from the seed; other modes: none."""
    from covsyn.calibration import cheng_recording as cr
    from covsyn.model.data_synthesis_main import source_profiles_for

    cohort = source_profiles_for("cheng2020", 7)
    assert cohort == cr.default_table().cohort(7)
    assert len(cohort) == 100
    assert source_profiles_for("spread_Taiwan_weight", 7) is None
    assert source_profiles_for("result", 7) is None
