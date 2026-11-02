# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B77: school open only on school days. B78: options for the imported first-wave seeds."""

import copy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import covsyn.model.data_synthesize as ds
from covsyn.model import school_calendar as sc

FEB7 = "taiwan_first_outbreak_feb7"


@pytest.mark.parametrize(("day", "expected"), [
    (date(2020, 1, 20), True),    # last day of the autumn term, a Monday
    (date(2020, 1, 21), False),   # winter vacation
    (date(2020, 2, 11), False),   # the original start, postponed
    (date(2020, 2, 15), False),   # make-up Saturday, cancelled
    (date(2020, 2, 24), False),
    (date(2020, 2, 25), True),    # delayed start of the spring term, a Tuesday
    (date(2020, 2, 28), False),   # Peace Memorial Day
    (date(2020, 2, 29), False),   # Saturday
    (date(2020, 3, 2), True),
    (date(2020, 4, 2), False),    # Children's Day observed
    (date(2020, 4, 3), False),    # Tomb Sweeping Day observed
    (date(2020, 5, 1), True),     # Labour Day is a school day
    (date(2020, 6, 25), False),   # Dragon Boat Festival
    (date(2020, 7, 14), True),    # last day of the delayed spring term
    (date(2020, 7, 15), False),   # summer vacation
    (date(2020, 8, 31), True),    # autumn term 109-1
    (date(2020, 10, 9), False),   # National Day holiday
])
def test_school_open_by_date(day, expected):
    assert sc.school_open(day) is expected


def test_undated_calendar_closes_weekends_only():
    assert sc.school_open(date(2020, 1, 21), dated=False)       # a Tuesday in the vacation
    assert not sc.school_open(date(2020, 3, 7), dated=False)    # Saturday
    assert not sc.school_open(date(2020, 3, 8), dated=False)    # Sunday
    assert sc.school_open(date(2021, 1, 11))                    # after DATED_UNTIL: weekday rule


def test_open_days_start_from_the_case_infection_day():
    calendar = sc.SchoolCalendar(date(2020, 2, 20))             # Thursday
    assert calendar.open_days(3, 4).tolist() == [False, False, True, True]   # Feb 23-26
    assert calendar.open_days(0, 0).shape == (0,)
    with pytest.raises(ValueError):
        calendar.open_days(-1, 2)


class _FixedSchool:
    def __init__(self, pattern):
        self.pattern = np.asarray(pattern, dtype=bool)
        self.calls = []

    def open_days(self, first_day, n_days):
        self.calls.append((first_day, n_days))
        return np.resize(self.pattern, n_days)


def test_closed_days_remove_classmate_contacts_only():
    c = ds.Draw_contact_data.__new__(ds.Draw_contact_data)
    c.school_calendar = _FixedSchool([True, False, False, True])
    c.course_of_disease_data_object = type("Course", (), {"infection_day": 6.0})()
    c.school_class_contacts_matrix = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=bool)
    c.household_contacts_matrix = np.ones((2, 4), dtype=bool)
    c.apply_school_calendar()
    assert c.school_calendar.calls == [(6, 4)]
    assert c.school_class_contacts_matrix.astype(int).tolist() == [[1, 0, 0, 0], [0, 0, 0, 1]]
    assert c.household_contacts_matrix.all()
    c.school_class_contacts_matrix = np.empty((0, 4))
    c.apply_school_calendar()
    assert c.school_class_contacts_matrix.shape == (0, 4)


def _b64_vector(run10_vector):
    """Run 10 padded with B54/B62, the B64 work group and the B65 delay (as test_b75)."""
    return np.concatenate([run10_vector, [1.0, 1.67, 21, 1000, 1.0], [1.74], [10.0, 1.0], [1.0]])


def _run(vector, demographic_parameters, seed, mode=FEB7, **options):
    from covsyn.model.data_synthesis_main import run_covid
    return run_covid(seed, vector.copy(), copy.deepcopy(demographic_parameters), save_file=False,
                     mode=mode, **options)


def test_simulated_first_wave_meets_no_classmate_on_a_closed_day(run10_vector,
                                                                 demographic_parameters):
    """The Apr 12 scenario: the Feb 7 one falls almost wholly in the winter vacation."""
    from covsyn.model.data_synthesis_main import FIRST_WAVE_SEED_FILES
    from covsyn.model.work_calendar import first_wave_day_zero

    mode = "taiwan_first_outbreak"
    day_zero = first_wave_day_zero(FIRST_WAVE_SEED_FILES[mode])
    vector = _b64_vector(run10_vector)
    checked = 0
    for seed in range(2):
        _, _, courses, contacts = _run(vector, demographic_parameters, seed, mode=mode)
        for course, contact in zip(courses, contacts):
            matrix = np.asarray(contact["school_class_contacts_matrix"], dtype=bool)
            if matrix.ndim != 2 or matrix.shape[0] == 0:
                continue
            first = int(round(course["infection_day"]))
            closed = ~np.array([sc.school_open(day_zero + timedelta(days=first + j))
                                for j in range(matrix.shape[1])])
            assert not matrix[:, closed].any()
            checked += 1
    assert checked > 0


def test_undated_runs_meet_no_classmate_at_the_weekend(run10_vector, demographic_parameters):
    """Calibration runs: day 0 is the work calendar's random weekday; weekends are closed."""
    from covsyn.model.data_synthesis_main import WORK_CALENDAR_STREAM, run_covid
    from covsyn.model.work_calendar import weekday_calendar

    vector = _b64_vector(run10_vector)
    checked = 0
    for seed in range(15):
        day_zero = weekday_calendar(np.random.default_rng([seed, WORK_CALENDAR_STREAM])).day_zero
        _, _, courses, contacts = run_covid(seed, vector.copy(),
                                            copy.deepcopy(demographic_parameters),
                                            save_file=False, mode="result")
        for course, contact in zip(courses, contacts):
            matrix = np.asarray(contact["school_class_contacts_matrix"], dtype=bool)
            if matrix.ndim != 2 or matrix.shape[0] == 0:
                continue
            first = int(round(course["infection_day"]))
            weekend = np.array([(day_zero + timedelta(days=first + j)).weekday() >= 5
                                for j in range(matrix.shape[1])])
            assert not matrix[:, weekend].any()
            checked += 1
    assert checked > 0


def test_older_vectors_keep_school_open_every_day(run10_vector, demographic_parameters):
    """Run 10 has no B64 work group, hence no calendar: some classmate is met at a weekend."""
    from covsyn.model.data_synthesis_main import run_covid

    met = 0
    for seed in range(15):
        _, _, _, contacts = run_covid(seed, run10_vector.copy(),
                                      copy.deepcopy(demographic_parameters), save_file=False,
                                      mode="result")
        met += sum(np.asarray(c["school_class_contacts_matrix"], dtype=bool).sum()
                   for c in contacts if np.asarray(c["school_class_contacts_matrix"]).ndim == 2)
    assert met > 0


def test_imported_isolation_option_is_validated(run10_vector, demographic_parameters):
    with pytest.raises(ValueError):
        _run(_b64_vector(run10_vector), demographic_parameters, 0, imported_isolation="whenever")


def test_imported_delay_is_drawn_from_the_data(run10_vector, demographic_parameters):
    from covsyn.model.data_synthesis_main import IMPORTED_ONSET_TO_CONFIRMATION

    delays = set()
    for seed in range(6):
        _, _, courses, _ = _run(_b64_vector(run10_vector), demographic_parameters, seed,
                                imported_isolation="imported_delay")
        for c in courses:
            if c.get("seed_type") == "imported" and np.isfinite(c["incubation_period"]):
                delays.add(c["monitor_isolation_period"] - c["incubation_period"])
    assert delays and delays <= set(IMPORTED_ONSET_TO_CONFIRMATION)


def test_imported_onset_to_confirmation_is_the_210_cases():
    from covsyn.model.data_synthesis_main import IMPORTED_ONSET_TO_CONFIRMATION as delays
    assert sum(delays.values()) == 210
    mean = sum(k * n for k, n in delays.items()) / 210
    assert mean == pytest.approx(6.04, abs=0.01)


def test_imported_factor_scales_only_imported_seeds_in_the_community(run10_vector,
                                                                     demographic_parameters):
    layers = ("household", "school", "workplace", "health_care", "municipality")
    infected = {"imported": 0.0, "other seeds": 0.0}
    for seed in range(10):
        _, _, courses, contacts = _run(_b64_vector(run10_vector), demographic_parameters, seed,
                                       imported_factor=0.0)
        for c, x in zip(courses, contacts):
            n = sum(float(np.nansum(np.asarray(x[f"{L}_effective_contacts"] or [], dtype=float)))
                    for L in layers)
            if c.get("seed_type") == "imported":
                infected["imported"] += n
            elif c.get("seed_type") is not None:
                infected["other seeds"] += n
    assert infected["imported"] == 0.0
    assert infected["other seeds"] > 0.0
