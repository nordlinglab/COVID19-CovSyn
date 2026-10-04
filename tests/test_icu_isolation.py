# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Admission to intensive care isolates the case (decision B52, finding E81).

A case admitted to the ICU before its own isolation day is isolated on the admission day:
monitor_isolation_period becomes the ICU day counted from infection, the positive and
negative test dates move with it so the confirming test stays at isolation (C07), and
isolation_route becomes 'critical'. Every other case is left untouched.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from covsyn.model import data_synthesize as ds


def _course(isolation: int, icu: float, infection_day: float = 10.0,
            route: str = 'symptom') -> SimpleNamespace:
    return SimpleNamespace(
        infection_day=infection_day, monitor_isolation_period=isolation,
        date_of_critically_ill=icu, isolation_route=route,
        positive_test_date=infection_day + isolation,
        negative_test_date=np.array([infection_day + isolation - 3.0,
                                     infection_day + isolation - 1.0]))


def _apply(course: SimpleNamespace) -> SimpleNamespace:
    ds.Draw_course_of_disease_data.apply_icu_isolation(course)
    return course


def test_icu_before_isolation_moves_isolation_to_the_icu_day() -> None:
    """ICU on day 6 after infection, isolation planned for day 9: isolated on day 6."""
    course = _apply(_course(isolation=9, icu=16.0))
    assert course.monitor_isolation_period == 6
    assert course.isolation_route == 'critical'


def test_test_dates_move_with_isolation() -> None:
    course = _apply(_course(isolation=9, icu=16.0))
    assert course.positive_test_date == 16.0
    np.testing.assert_array_equal(course.negative_test_date, [13.0, 15.0])


def test_icu_on_the_isolation_day_changes_nothing() -> None:
    before = _course(isolation=6, icu=16.0)
    after = _apply(copy.deepcopy(before))
    assert vars(after).keys() == vars(before).keys()
    assert after.monitor_isolation_period == 6 and after.isolation_route == 'symptom'
    assert after.positive_test_date == before.positive_test_date


def test_icu_after_isolation_changes_nothing() -> None:
    course = _apply(_course(isolation=5, icu=20.0, route='traced'))
    assert course.monitor_isolation_period == 5 and course.isolation_route == 'traced'
    assert course.positive_test_date == 15.0


def test_case_without_icu_changes_nothing() -> None:
    course = _apply(_course(isolation=9, icu=math.nan))
    assert course.monitor_isolation_period == 9 and course.isolation_route == 'symptom'


def test_icu_on_the_infection_day_isolates_on_day_zero() -> None:
    course = _apply(_course(isolation=4, icu=10.0))
    assert course.monitor_isolation_period == 0
    assert course.positive_test_date == 10.0


def test_isolation_never_falls_after_the_icu_day() -> None:
    """C14 on a fractional ICU day: isolation is the whole day on or before it."""
    course = _apply(_course(isolation=9, icu=16.6))
    assert course.infection_day + course.monitor_isolation_period <= 16.6
    assert course.monitor_isolation_period == 6


def test_correction_draws_no_random_number() -> None:
    """The course-of-disease random stream must stay the same as before B52."""
    np.random.seed(7)
    expected = np.random.random()
    np.random.seed(7)
    _apply(_course(isolation=9, icu=16.0))
    assert np.random.random() == expected


@pytest.mark.parametrize("seed", range(40))
def test_every_simulated_icu_case_is_isolated_by_admission(
        seed: int, run10_vector: np.ndarray, demographic_parameters: object) -> None:
    """C14 holds for every case of a whole spread simulation, and C07 still holds."""
    from covsyn.model.data_synthesis_main import run_covid

    _, _, courses, _ = run_covid(seed, run10_vector.copy(), copy.deepcopy(demographic_parameters),
                                 save_file=False, mode='spread_Taiwan_weight')
    for c in courses:
        start = c['infection_day'] + c['monitor_isolation_period']
        if np.isfinite(c['date_of_critically_ill']):
            assert start <= c['date_of_critically_ill']
        assert abs(c['positive_test_date'] - start) <= 1
