# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B75: whether an employed case is at work on a day, by day of the week and national holiday."""

from datetime import date

import numpy as np
import pytest

import covsyn.model.data_synthesize as ds
from covsyn.model import contact_measures as cm
from covsyn.model import work_calendar as wc


@pytest.mark.parametrize(("day", "expected"), [
    (date(2020, 3, 10), wc.WEEKDAY_AT_WORK),   # Tuesday
    (date(2020, 3, 7), wc.SATURDAY_AT_WORK),   # Saturday
    (date(2020, 3, 8), wc.SUNDAY_AT_WORK),     # Sunday
    (date(2020, 1, 1), wc.SUNDAY_AT_WORK),     # New Year, a Wednesday
    (date(2020, 1, 23), wc.SUNDAY_AT_WORK),    # adjusted day off before the Lunar New Year
    (date(2020, 1, 29), wc.SUNDAY_AT_WORK),    # last day of the Lunar New Year holiday
    (date(2020, 1, 30), wc.WEEKDAY_AT_WORK),   # first working day after it
    (date(2020, 2, 15), wc.WEEKDAY_AT_WORK),   # make-up working Saturday
    (date(2020, 2, 28), wc.SUNDAY_AT_WORK),    # Peace Memorial Day, a Friday
    (date(2020, 4, 2), wc.SUNDAY_AT_WORK),     # Children's Day observed, a Thursday
    (date(2020, 4, 3), wc.SUNDAY_AT_WORK),     # Tomb Sweeping Day observed, a Friday
    (date(2020, 5, 1), wc.SUNDAY_AT_WORK),     # Labour Day, a Friday
])
def test_attendance_probability_by_date(day, expected):
    assert wc.attendance_probability(day) == expected


def test_the_values_are_the_sources():
    """Fu 2010 weekdays, MOL 2020 Saturdays, no stipulated Sunday work (user, 2026-10-31)."""
    assert wc.WEEKDAY_AT_WORK == pytest.approx(0.691)
    assert wc.SATURDAY_AT_WORK == pytest.approx(0.0199)
    assert wc.SUNDAY_AT_WORK == 0.0


def test_a_weekday_calendar_ignores_holidays():
    """The calibration calendar has weekdays only: a holiday date counts as its weekday."""
    assert wc.attendance_probability(date(2020, 2, 28), holidays=False) == wc.WEEKDAY_AT_WORK
    assert wc.attendance_probability(date(2020, 2, 15), holidays=False) == wc.SATURDAY_AT_WORK


def test_dated_calendar_follows_the_dates():
    """Day 0 = 2020-01-20 (Monday); days 3-9 are the Lunar New Year holiday (Jan 23-29)."""
    calendar = wc.WorkCalendar(date(2020, 1, 20), np.random.default_rng(0))
    at_work = np.array([calendar.at_work(0, 14) for _ in range(4000)])
    share = at_work.mean(axis=0)
    assert share[3:10].max() == 0.0
    assert share[10:12] == pytest.approx([wc.WEEKDAY_AT_WORK] * 2, abs=0.03)  # Thu, Fri
    assert share[12] < 0.05                                 # Saturday Feb 1
    assert share[13] == 0.0                                 # Sunday Feb 2


def test_at_work_starts_from_the_case_infection_day():
    """Column j of a case infected on simulation day d is the date day_zero + d + j."""
    calendar = wc.WorkCalendar(date(2020, 3, 2), np.random.default_rng(1))  # Monday
    at_work = np.array([calendar.at_work(5, 3) for _ in range(2000)])  # Sat, Sun, Mon
    assert at_work[:, 1].sum() == 0
    assert at_work[:, 0].mean() < 0.05
    assert at_work[:, 2].mean() == pytest.approx(wc.WEEKDAY_AT_WORK, abs=0.04)


def test_weekday_calendar_draws_a_random_weekday_for_day_zero():
    rng = np.random.default_rng(2)
    days = {wc.weekday_calendar(rng).day_zero.weekday() for _ in range(200)}
    assert days == set(range(7))
    assert not wc.weekday_calendar(rng).holidays


def test_at_work_handles_empty_and_rejects_invalid_lengths():
    calendar = wc.WorkCalendar(date(2020, 3, 2), np.random.default_rng(3))
    assert calendar.at_work(0, 0).shape == (0,)
    assert calendar.at_work(0, 400).shape == (400,)
    with pytest.raises(ValueError):
        calendar.at_work(0, -1)
    with pytest.raises(ValueError):
        calendar.at_work(-1, 3)


def test_first_wave_day_zero_is_read_from_the_seed_table():
    assert wc.first_wave_day_zero("variable/taiwan_first_wave_seeds.csv") == date(2020, 1, 9)
    assert wc.first_wave_day_zero("variable/taiwan_first_wave_seeds_mar18_local.csv") == date(2020, 1, 12)


class _FixedCalendar:
    """at_work() returns the given pattern, repeated, from the case's infection day on."""

    def __init__(self, pattern):
        self.pattern = np.asarray(pattern, dtype=bool)
        self.calls = []

    def at_work(self, first_day, n_days):
        self.calls.append((first_day, n_days))
        return np.resize(self.pattern, n_days)


def _contact_data(calendar, infection_day=4.0):
    c = ds.Draw_contact_data.__new__(ds.Draw_contact_data)
    c.work_calendar = calendar
    c.course_of_disease_data_object = type("Course", (), {"infection_day": infection_day})()
    c.workplace_contacts_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    c.household_contacts_matrix = np.ones((2, 4), dtype=bool)
    return c


def test_days_off_remove_workplace_contacts_only():
    calendar = _FixedCalendar([True, False, False, True])
    c = _contact_data(calendar)
    c.apply_work_calendar()
    assert calendar.calls == [(4, 4)]
    assert c.workplace_at_work.tolist() == [True, False, False, True]
    # The coworker met only on day 2 (a day off) is no longer a contact.
    assert c.workplace_contacts_matrix.astype(int).tolist() == [[1, 0, 0, 0], [0, 0, 0, 1]]
    assert c.household_contacts_matrix.all()


def test_a_case_without_coworkers_still_records_its_days_at_work():
    c = _contact_data(_FixedCalendar([False, True]))
    c.workplace_contacts_matrix = np.empty((0, 5))
    c.apply_work_calendar()
    assert c.workplace_at_work.tolist() == [False, True, False, True, False]
    assert c.workplace_contacts_matrix.shape == (0, 5)


def test_contacts_per_day_at_work_counts_only_days_at_work():
    """B75 target: coworker contacts on the days before onset the case was at work."""
    course = {"incubation_period": 4.0}
    contact = {"workplace_contacts_matrix": np.array([[1, 0, 0, 1, 1], [1, 0, 0, 0, 1]]),
               "workplace_at_work": np.array([True, False, False, True, True])}
    assert cm.contacts_per_day_at_work(course, contact) == pytest.approx(1.5)  # days 0 and 3
    assert cm.at_work_days_before_onset(course, contact).tolist() == [2, 1]


def test_contacts_per_day_at_work_without_a_day_at_work_is_nan():
    course = {"incubation_period": 2.0}
    contact = {"workplace_contacts_matrix": np.zeros((0, 4)),
               "workplace_at_work": np.array([False, False, True, True])}
    assert np.isnan(cm.contacts_per_day_at_work(course, contact))
    assert cm.at_work_days_before_onset(course, contact).shape == (0,)


def test_data_without_a_calendar_counts_every_day_as_a_day_at_work():
    """Results saved before B75 have no workplace_at_work: every day was a day at work."""
    course = {"incubation_period": np.nan}
    contact = {"workplace_contacts_matrix": np.array([[1, 0, 1], [0, 0, 1]])}
    assert cm.contacts_per_day_at_work(course, contact) == pytest.approx(1.0)
    assert cm.at_work_days_before_onset(course, contact).tolist() == [1, 0, 2]


def _b64_vector(run10_vector):
    """Run 10 padded with B54/B62 (P[199..204]), the B64 work group and the B65 delay."""
    return np.concatenate([run10_vector, [1.0, 1.67, 21, 1000, 1.0], [1.74], [10.0, 1.0], [1.0]])


def test_simulated_case_meets_no_coworker_on_a_day_off(run10_vector, demographic_parameters):
    import copy

    from covsyn.model.data_synthesis_main import run_covid

    vector = _b64_vector(run10_vector)
    days_off = 0
    for seed in range(20):
        _, _, _, contacts = run_covid(seed, vector.copy(), copy.deepcopy(demographic_parameters),
                                      save_file=False, mode="result")
        for contact in contacts:
            at_work = np.asarray(contact["workplace_at_work"], dtype=bool)
            matrix = np.asarray(contact["workplace_contacts_matrix"], dtype=bool)
            assert matrix.shape[1] == len(at_work)
            assert not matrix[:, ~at_work].any()
            days_off += int((~at_work).sum())
    assert days_off > 0


def test_older_vectors_have_no_work_calendar(run10_vector, demographic_parameters):
    """Run 10 has no B64 work group, so its output keeps its keys and its random stream."""
    import copy

    from covsyn.model.data_synthesis_main import run_covid

    _, _, _, contacts = run_covid(0, run10_vector.copy(), copy.deepcopy(demographic_parameters),
                                  save_file=False, mode="result")
    assert all("workplace_at_work" not in contact for contact in contacts)


def test_workplace_targets_are_fu_2010_days_at_work():
    """B75 replaces B70's average day: 4.96 [4.34, 5.61], zero 27.4% [23.2, 31.8], p90 13
    [10, 16], all charged again now that the model has days off."""
    pytest.importorskip("sklearn")
    from covsyn.calibration.firefly_optimizer import OUTCOME_TARGETS

    assert OUTCOME_TARGETS["daily_workplace_employed"] == (4.34, 5.61, 1.0)
    assert OUTCOME_TARGETS["workplace_day_zero_share"] == (0.232, 0.318, 1.0)
    assert OUTCOME_TARGETS["workplace_day_p90"] == (10.0, 16.0, 1.0)


def test_both_objectives_measure_coworkers_on_days_at_work():
    """fast_cost and measure_outcomes take the same B75 quantity (E65: they must agree)."""
    from covsyn.calibration import fast_cost

    course = {"incubation_period": 4.0, "pre_onset_window": 2.0}
    contact = {"workplace_contacts_matrix": np.array([[1, 0, 0, 1, 1], [1, 0, 0, 0, 1]]),
               "workplace_at_work": np.array([True, False, False, True, True])}
    out = fast_cost.workplace_scalars(course, contact, {"enterprise_size": 5})
    assert out["per_day_workplace_at_work"] == pytest.approx(1.5)
    assert out["workplace_days"].tolist() == [2, 1]


def test_empty_days_at_work_give_no_day():
    """Review: an empty at-work array used to crash in reshape."""
    course = {"incubation_period": 3.0}
    contact = {"workplace_contacts_matrix": [], "workplace_at_work": []}
    assert cm.at_work_days_before_onset(course, contact).shape == (0,)
    assert np.isnan(cm.contacts_per_day_at_work(course, contact))


def test_mismatched_days_at_work_are_rejected():
    """Review: a 2x3 matrix with 2 at-work days used to be reshaped silently to 3x2."""
    course = {"incubation_period": np.nan}
    contact = {"workplace_contacts_matrix": np.ones((2, 3)), "workplace_at_work": [True, True]}
    with pytest.raises(ValueError):
        cm.at_work_days_before_onset(course, contact)


def test_old_data_without_a_window_is_measured_as_before():
    """Review: data saved before B75 keeps B63's one zero day and B55's 0.0 per day."""
    course = {"incubation_period": 3.0}
    contact = {"workplace_contacts_matrix": []}
    assert cm.at_work_days_before_onset(course, contact).tolist() == [0.0]
    assert cm.contacts_per_day_at_work(course, contact) == 0.0


@pytest.mark.parametrize(("day", "expected"), [
    (date(2020, 6, 25), wc.SUNDAY_AT_WORK),    # Dragon Boat Festival, a Thursday
    (date(2020, 6, 26), wc.SUNDAY_AT_WORK),    # its adjusted day off, a Friday
    (date(2020, 10, 1), wc.SUNDAY_AT_WORK),    # Moon Festival, a Thursday
    (date(2020, 10, 2), wc.SUNDAY_AT_WORK),
    (date(2020, 10, 9), wc.SUNDAY_AT_WORK),    # National Day observed, a Friday
    (date(2021, 1, 1), wc.SUNDAY_AT_WORK),     # New Year, a Friday
    (date(2020, 6, 20), wc.SATURDAY_AT_WORK),  # make-up day not confirmed: a Saturday
    (date(2020, 6, 24), wc.WEEKDAY_AT_WORK),
])
def test_holidays_cover_the_whole_simulation_year(day, expected):
    """Review: first-wave runs last 365 days from 2020-01-09, past the old 2020-05-31 table."""
    assert wc.attendance_probability(day) == expected


def test_cached_probabilities_match_the_dates():
    """Review: at_work() reads a per-run probability array instead of one date per day."""
    calendar = wc.WorkCalendar(date(2020, 1, 9), np.random.default_rng(4))
    p = calendar.probabilities(30, 400)
    expected = [wc.attendance_probability(date(2020, 2, 8) + wc.timedelta(days=k))
                for k in range(400)]
    assert p.tolist() == expected


def test_day_zero_is_read_once_per_seed_table():
    """Review: run_covid used to parse the seed table again on every run."""
    wc.first_wave_day_zero.cache_clear()
    for _ in range(3):
        wc.first_wave_day_zero("variable/taiwan_first_wave_seeds.csv")
    assert wc.first_wave_day_zero.cache_info().hits == 2
