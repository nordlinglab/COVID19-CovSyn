# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contacts recorded as Cheng et al. (2020)'s investigation recorded them (E84).

"The period of investigation started at the date at symptom onset (and could be extended to up
to 4 days before symptom onset when epidemiologically indicated) and ended at the date at
COVID-19 confirmation." A contact's bin is the first day of exposure inside that period. An
imported index case met nobody in Taiwan before it arrived, and one found at the airport met
nobody the investigation recorded. A share g of investigations is extended; the objective
mixes the two records' counts in expectation (user decision 2026-11-23, design reviewed in
logic-review_openai_2026-11-23_2130_cost-change-design_reply.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covsyn.calibration import cheng_recording as cr


def _course(onset=5, confirmed=9, infection_day=0):
    return {"incubation_period": onset, "positive_test_date": confirmed + infection_day,
            "infection_day": infection_day}


def _contact(household=(), times=(), **extra):
    """A contact record with only household rows (each a list of contact days, 0-based)."""
    width = 20
    rows = []
    for days in household:
        row = np.zeros(width, dtype=bool)
        row[list(days)] = True
        rows.append(row)
    out = {key: [] for key in cr.MATRIX.values()}
    out.update({f"{layer}_effective_contacts_infection_time": [] for layer in cr.MATRIX})
    out["household_contacts_matrix"] = rows
    out["household_effective_contacts_infection_time"] = list(times)
    out.update(extra)
    return out


def _household(bins):
    return bins[0, 0].tolist(), bins[0, 1].tolist()


# ------------------------------------------------------------------ the investigation period


def test_a_period_from_onset_bins_the_first_day_inside_it():
    """Met on days 2-7, onset day 5: first recorded day 5, bin 0-3."""
    bins = cr.recorded_bins(_course(), _contact([range(2, 8)], [np.nan]), back=0)
    assert _household(bins) == ([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])


def test_an_extended_period_reaches_four_days_before_onset():
    """Same contact with the period from day 1: first recorded day 2, before onset."""
    bins = cr.recorded_bins(_course(), _contact([range(2, 8)], [np.nan]), back=4)
    assert _household(bins)[0] == [1, 0, 0, 0, 0, 0]


def test_both_ends_of_the_period_are_included():
    """Onset 5 (period from 1 when extended), confirmation 9: days 1 and 9 count."""
    first = cr.recorded_bins(_course(), _contact([[1]], [np.nan]), back=4)
    last = cr.recorded_bins(_course(), _contact([[9]], [np.nan]), back=0)
    assert _household(first)[0] == [1, 0, 0, 0, 0, 0]
    assert _household(last)[0] == [0, 0, 1, 0, 0, 0]


def test_contacts_only_outside_the_period_are_not_recorded():
    """Met only on day 0 (before day 1) or day 10 (after confirmation): nothing."""
    bins = cr.recorded_bins(_course(), _contact([[0], [10]], [np.nan, np.nan]), back=4)
    assert bins.sum() == 0


def test_an_infection_counts_with_its_recorded_contact():
    """A recorded contact that was infected is an infection in its bin."""
    bins = cr.recorded_bins(_course(), _contact([range(2, 8)], [3.0]), back=0)
    assert _household(bins) == ([0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])


def test_confirmation_counts_from_the_cases_own_infection():
    """A secondary case infected on day 3: its record runs from its own infection."""
    course = _course(onset=5, confirmed=9, infection_day=3)
    bins = cr.recorded_bins(course, _contact([[9]], [np.nan]), back=0)
    assert _household(bins)[0] == [0, 0, 1, 0, 0, 0]


def test_an_asymptomatic_case_records_nothing():
    """No onset: Cheng's figure leaves the case out."""
    course = _course()
    course["incubation_period"] = np.nan
    assert cr.recorded_bins(course, _contact([range(20)], [np.nan]), back=4).sum() == 0


def test_a_case_without_contacts_records_nothing():
    """The model writes no contacts as an empty list."""
    assert cr.recorded_bins(_course(), _contact(), back=4).sum() == 0


# ------------------------------------------------------------------ arrival


def test_days_before_arrival_are_not_recorded():
    """Arrived on day 7, 2 days after onset: first recorded day 7, bin 0-3."""
    bins = cr.recorded_bins(_course(), _contact([range(2, 9)], [np.nan]), back=4, arrival=7)
    assert _household(bins)[0] == [0, 1, 0, 0, 0, 0]


def test_arrival_moves_a_contact_to_a_later_bin():
    """Arrived on day 10, confirmed on 14: first recorded day 10, five days after onset."""
    course = _course(onset=5, confirmed=14)
    bins = cr.recorded_bins(course, _contact([range(2, 12)], [np.nan]), back=4, arrival=10)
    assert _household(bins)[0] == [0, 0, 1, 0, 0, 0]


def test_arrival_before_the_period_changes_nothing():
    """Arrived on day 0, period from day 1: the same record as without arrival."""
    contact = _contact([range(2, 8)], [np.nan])
    assert np.array_equal(cr.recorded_bins(_course(), contact, back=4, arrival=0),
                          cr.recorded_bins(_course(), contact, back=4))


def test_arrival_after_confirmation_records_nothing():
    """Arrived on day 12, confirmed on 9: the investigation ended before."""
    bins = cr.recorded_bins(_course(), _contact([range(20)], [np.nan]), back=4, arrival=12)
    assert bins.sum() == 0


def test_an_infection_before_arrival_is_not_counted():
    """Infected on day 3, arrival day 6: the contact after arrival counts, its infection not."""
    bins = cr.recorded_bins(_course(), _contact([range(2, 9)], [3.0]), back=4, arrival=6)
    assert _household(bins) == ([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])


def test_an_infection_on_the_day_of_arrival_counts():
    """Infected on day 6, arrival day 6."""
    bins = cr.recorded_bins(_course(), _contact([range(2, 9)], [6.0]), back=4, arrival=6)
    assert _household(bins)[1] == [0, 1, 0, 0, 0, 0]


def test_a_case_found_at_the_airport_records_nothing():
    """Isolated on arrival."""
    bins = cr.recorded_bins(_course(), _contact([range(20)], [3.0]), back=4, airport=True)
    assert bins.sum() == 0


# ------------------------------------------------------------------ groups and events


def test_school_workplace_and_community_add_up_in_the_others_group():
    """One contact each in the three layers of Cheng's 'others'."""
    row = np.zeros(20, dtype=bool)
    row[6] = True
    contact = _contact(school_class_contacts_matrix=[row], workplace_contacts_matrix=[row],
                       municipality_contacts_matrix=[row],
                       school_effective_contacts_infection_time=[np.nan],
                       workplace_effective_contacts_infection_time=[6.0],
                       municipality_effective_contacts_infection_time=[np.nan])
    bins = cr.recorded_bins(_course(), contact, back=0)
    assert bins[2, 0].tolist() == [0, 3, 0, 0, 0, 0]
    assert bins[2, 1].tolist() == [0, 1, 0, 0, 0, 0]


def _event_contact():
    """One ordinary community contact on day 6 and one sampled event contact on day 2."""
    rows = np.zeros((2, 20), dtype=bool)
    rows[0, 6] = True
    rows[1, 2] = True
    expected = np.zeros(20)
    expected[[2, 6]] = [0.5, 0.25]
    infections = np.zeros(20)
    infections[2] = 0.125
    return _contact(municipality_contacts_matrix=list(rows),
                    municipality_effective_contacts_infection_time=[np.nan, 2.0],
                    municipality_event_mask=[False, True],
                    municipality_event_expected_contacts=expected,
                    municipality_event_expected_infections=infections)


def test_event_contacts_are_replaced_by_their_expectation_inside_the_period():
    """B55: the sampled event row goes; the expected 0.5 (day 2) and 0.25 (day 6) come in."""
    bins = cr.recorded_bins(_course(), _event_contact(), back=4)
    assert bins[2, 0].tolist() == [0.5, 1.25, 0, 0, 0, 0]
    assert bins[2, 1].tolist() == [0.125, 0, 0, 0, 0, 0]


def test_expected_event_contacts_outside_the_period_are_not_recorded():
    """From onset (day 5): day 2's expected event contacts fall outside."""
    bins = cr.recorded_bins(_course(), _event_contact(), back=0)
    assert bins[2, 0].tolist() == [0, 1.25, 0, 0, 0, 0]
    assert bins[2, 1].sum() == 0


def test_expected_event_contacts_before_arrival_are_not_recorded():
    """Arrival on day 4: day 2's expected event contacts fall outside."""
    bins = cr.recorded_bins(_course(), _event_contact(), back=4, arrival=4)
    assert bins[2, 0].tolist() == [0, 1.25, 0, 0, 0, 0]


def test_without_a_record_period_the_bins_are_the_objectives_old_ones():
    """No period and no arrival reproduce create_array_cheng2020_fig2 with B55, per group."""
    from covsyn.figures.plot_results import create_array_cheng2020_fig2

    layers = (("Household",), ("Health care",), ("School", "Workplace", "Municipality"))
    for contact in (_contact([range(2, 8), [0], [12]], [3.0, np.nan, np.nan]), _event_contact()):
        old = np.zeros((3, 2, 6))
        for gi, names in enumerate(layers):
            for name in names:
                _, contacts, _, infections = create_array_cheng2020_fig2(
                    [_course()], [contact], layer=name, expected_events=True)
                if contacts.size == 6:
                    old[gi] += (contacts, infections)
        assert np.array_equal(cr.recorded_bins(_course(), contact, back=None), old)


# ------------------------------------------------------------------ the mixture


def test_the_case_bins_mix_the_two_records_by_the_extension_share():
    """The share g of the extended record plus 1 - g of the record from onset."""
    contact = _contact([range(2, 8)], [3.0])
    extended = cr.recorded_bins(_course(), contact, back=4)
    from_onset = cr.recorded_bins(_course(), contact, back=0)
    mixed = cr.case_bins(_course(), contact, share=0.25)
    assert np.allclose(mixed, 0.25 * extended + 0.75 * from_onset)


def test_the_index_case_profile_truncates_at_arrival():
    """Imported, arriving 5 days after onset (day 10): the profile's arrival applies."""
    course = _course(onset=5, confirmed=14)
    contact = _contact([range(2, 12)], [np.nan])
    profile = {"age": 30, "imported": True, "airport": False, "arrival_from_onset": 5}
    bins = cr.case_bins(course, contact, profile, share=1.0)
    assert _household(bins)[0] == [0, 0, 1, 0, 0, 0]


def test_a_local_profile_does_not_truncate():
    """Local: the same as no profile."""
    contact = _contact([range(2, 8)], [3.0])
    profile = {"age": 30, "imported": False, "airport": False, "arrival_from_onset": None}
    assert np.array_equal(cr.case_bins(_course(), contact, profile),
                          cr.case_bins(_course(), contact))


@pytest.mark.parametrize("share", [-0.1, 1.1, float("nan")])
def test_an_extension_share_outside_zero_one_is_refused(share):
    """The extension share g is a share."""
    with pytest.raises(ValueError, match="share"):
        cr.case_bins(_course(), _contact(), share=share)


# ------------------------------------------------------------------ profiles


def _profiles():
    return [{"case_id": 1, "age_low": 20.0, "age_high": 29.0, "imported": True,
             "airport": False, "arrival_from_onset": np.nan},
            {"case_id": 2, "age_low": 50.0, "age_high": 59.0, "imported": True,
             "airport": False, "arrival_from_onset": 3.0},
            {"case_id": 3, "age_low": 30.0, "age_high": 39.0, "imported": False,
             "airport": False, "arrival_from_onset": np.nan}]


def test_the_same_seed_draws_the_same_profile():
    """The draw depends on the simulation seed only."""
    assert cr.draw_profile(7, _profiles()) == cr.draw_profile(7, _profiles())


def test_drawing_a_profile_leaves_the_models_random_state_alone():
    """Its own generator: numpy's and random's global states are untouched."""
    import random

    np.random.seed(1)
    random.seed(1)
    cr.draw_profile(7, _profiles())
    after = (np.random.random(), random.random())
    np.random.seed(1)
    random.seed(1)
    assert after == (np.random.random(), random.random())


def test_the_age_is_a_whole_year_in_the_profiles_band():
    """Over many seeds every age lies in its case's band."""
    bands = {(p["age_low"], p["age_high"]) for p in _profiles()}
    for seed in range(200):
        profile = cr.draw_profile(seed, _profiles())
        assert isinstance(profile["age"], int)
        assert any(low <= profile["age"] <= high for low, high in bands)


def test_an_unknown_arrival_is_drawn_from_the_known_ones():
    """Missing treatment A: case 1's arrival is case 2's, the only known one."""
    for seed in range(200):
        profile = cr.draw_profile(seed, _profiles(), missing="resample")
        if profile["imported"]:
            assert profile["arrival_from_onset"] == 3


def test_an_unknown_arrival_can_be_taken_as_before_onset():
    """Missing treatment B: no truncation."""
    seen = set()
    for seed in range(200):
        profile = cr.draw_profile(seed, _profiles(), missing="before")
        if profile["imported"]:
            seen.add(profile["arrival_from_onset"])
    assert seen == {None, 3}


def test_a_local_profile_has_no_arrival():
    """Local cases are never truncated."""
    for seed in range(200):
        profile = cr.draw_profile(seed, _profiles())
        if not profile["imported"]:
            assert profile["arrival_from_onset"] is None


def test_a_profile_without_an_age_band_leaves_the_age_to_the_model():
    """No band: None, so run_covid draws from the population."""
    profiles = [dict(_profiles()[2], age_low=np.nan, age_high=np.nan)]
    assert cr.draw_profile(0, profiles)["age"] is None


def test_an_unknown_missing_treatment_is_refused():
    """Only 'resample' and 'before'."""
    with pytest.raises(ValueError, match="missing"):
        cr.draw_profile(0, _profiles(), missing="drop")


def test_no_profiles_is_refused():
    """An empty table cannot give an index case."""
    with pytest.raises(ValueError, match="profile"):
        cr.draw_profile(0, [])


def test_the_committed_profiles_load():
    """The real table: 100 cases, as build_cheng_profiles wrote it."""
    profiles = cr.load_profiles()
    assert len(profiles) == 100
    assert sum(p["imported"] for p in profiles) == 71
    assert len(pd.read_csv(cr.PROFILES)) == 100


def test_the_arrival_day_counts_from_infection():
    """Onset day 5 and arrival 3 days after: day 8; local or unknown: None."""
    assert cr.arrival_day({"imported": True, "arrival_from_onset": 3}, 5) == 8
    assert cr.arrival_day({"imported": True, "arrival_from_onset": None}, 5) is None
    assert cr.arrival_day({"imported": False, "arrival_from_onset": None}, 5) is None


def test_a_case_confirmed_before_onset_records_up_to_its_confirmation():
    """Traced and confirmed on day 4, onset on day 6: the period starts no later than day 4.

    From onset the period is [4, 4]; extended it is [0, 4]. Without this the period from onset
    would be empty (code review 2026-11-23); no case of the objective's data is affected.
    """
    course = _course(onset=6, confirmed=4)
    contact = _contact([[4], [1], [5]], [np.nan, np.nan, np.nan])
    assert _household(cr.recorded_bins(course, contact, back=0))[0] == [1, 0, 0, 0, 0, 0]
    assert _household(cr.recorded_bins(course, contact, back=4))[0] == [2, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(("low", "high"), [(100.0, 109.0), (-10.0, -1.0), (90.0, 101.0)])
def test_a_band_outside_the_age_tables_is_refused(low, high):
    """run_covid takes ages 0-100, so such a profile would fail every seed that draws it."""
    profiles = [dict(_profiles()[2], age_low=low, age_high=high)]
    with pytest.raises(ValueError, match="age"):
        cr.draw_profile(0, profiles)


def test_a_cohort_holds_every_profile_once_in_table_order():
    """Fig 6's 100 index cases are Cheng's 100 cases."""
    cohort = cr.ProfileTable(_profiles()).cohort(5)
    assert [p["case_id"] for p in cohort] == [1, 2, 3]


def test_a_cohort_is_the_seeds_and_ages_lie_in_the_bands():
    """Drawn from the seed only; every age in its own case's band."""
    table = cr.ProfileTable(_profiles())
    assert table.cohort(5) == table.cohort(5)
    for seed in range(50):
        for profile, case in zip(table.cohort(seed), _profiles(), strict=True):
            assert case["age_low"] <= profile["age"] <= case["age_high"]


def test_a_cohort_resolves_unknown_arrivals_like_a_draw():
    """Case 1 has no known arrival: resampled from case 2's (3), or none with 'before'."""
    table = cr.ProfileTable(_profiles())
    assert table.cohort(0)[0]["arrival_from_onset"] == 3
    assert table.cohort(0, missing="before")[0]["arrival_from_onset"] is None


def test_the_totals_over_cases_are_the_sum_of_their_recorded_bins():
    """Each case with the profile its course record carries; none gives zeros."""
    profile = {"age": 30, "imported": True, "airport": False, "arrival_from_onset": 5}
    late = dict(_course(onset=5, confirmed=14), index_profile=profile)
    cases = [(_course(), _contact([range(2, 8)], [3.0])), (late, _contact([range(2, 12)], [4.0]))]
    totals = cr.recorded_totals([c for c, _ in cases], [k for _, k in cases])
    want = cr.case_bins(*cases[0]) + cr.case_bins(late, cases[1][1], profile)
    assert np.array_equal(totals, want)
    assert cr.recorded_totals([], []).tolist() == np.zeros((3, 2, 6)).tolist()


def test_contact_rows_without_any_day_record_nothing():
    """A matrix of rows but no day columns, as a window ending before day 0 gives."""
    contact = _contact()
    contact["household_contacts_matrix"] = np.zeros((2, 0), dtype=bool)
    contact["household_effective_contacts_infection_time"] = [np.nan, np.nan]
    assert cr.recorded_bins(_course(), contact, back=4).sum() == 0
