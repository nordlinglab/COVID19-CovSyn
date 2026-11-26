# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cheng et al. (2020)'s recorded first day of exposure, applied to simulated contacts.

Cheng's investigation started at the index case's symptom onset, extended up to 4 days earlier
when indicated, and ended at confirmation; a contact's first day of exposure is the first day
of reported exposure within that period. The simulation knows the true first contact, so to
compare like with like the probe records each contact as the investigation would have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_cheng_window as pcw  # noqa: E402


def _row(days, width=16):
    row = np.zeros(width, dtype=bool)
    row[list(days)] = True
    return row


def test_without_a_window_the_true_first_contact_is_kept():
    """back=None reproduces the simulation's own first contact day."""
    assert pcw.recorded_first_day(_row(range(16)), onset=8, confirmation=15, back=None) == 0


def test_a_window_from_onset_records_onset_for_a_contact_met_since_infection():
    """A household member met every day is recorded from the day the investigation starts."""
    assert pcw.recorded_first_day(_row(range(16)), onset=8, confirmation=15, back=0) == 8


def test_a_window_extended_four_days_records_four_days_before_onset():
    """Extended by 4 days, the same member is recorded from onset - 4."""
    assert pcw.recorded_first_day(_row(range(16)), onset=8, confirmation=15, back=4) == 4


def test_a_contact_met_only_before_the_window_is_not_recorded():
    """The investigation never sees a contact whose exposure ended before it started."""
    assert pcw.recorded_first_day(_row([1, 2]), onset=8, confirmation=15, back=0) is None


def test_a_contact_met_only_after_confirmation_is_not_recorded():
    """The investigation ended at confirmation."""
    assert pcw.recorded_first_day(_row([12, 13]), onset=5, confirmation=10, back=0) is None


def test_a_contact_without_any_day_is_not_recorded():
    """An empty row is no contact."""
    assert pcw.recorded_first_day(_row([]), onset=8, confirmation=15, back=None) is None


def test_the_window_includes_both_ends():
    """Onset - back and confirmation are inside the investigation period."""
    assert pcw.recorded_first_day(_row([4]), onset=8, confirmation=15, back=4) == 4
    assert pcw.recorded_first_day(_row([15]), onset=8, confirmation=15, back=0) == 15


def test_the_bins_are_chengs():
    """Days from onset < 0, 0-3, 4-5, 6-7, 8-9 and > 9."""
    days = [-5, -1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 30]
    assert [pcw.cheng_bin(d) for d in days] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


@given(
    days=st.sets(st.integers(0, 39), max_size=40),
    onset=st.integers(0, 20),
    span=st.integers(0, 19),
    back=st.one_of(st.none(), st.integers(0, 10)),
)
def test_a_recorded_day_is_a_contact_day_inside_the_window_and_no_earlier_than_the_first(
    days, onset, span, back
):
    """Whatever the window, the recorded day is a real contact day within it, never earlier."""
    row = _row(days, width=40)
    confirmation = onset + span
    got = pcw.recorded_first_day(row, onset=onset, confirmation=confirmation, back=back)
    if got is None:
        return
    assert row[got]
    assert got >= min(days)
    if back is not None:
        assert onset - back <= got <= confirmation


def _case(**matrices):
    course = {"incubation_period": 2, "infection_day": 0.0, "positive_test_date": 5.0}
    contact = {key: [] for key in pcw.MATRIX.values()}
    contact.update(matrices)
    return course, contact


def test_a_case_without_contacts_adds_nothing():
    """Empty contact matrices, which the model writes as zero rows, give zero bins."""
    course, contact = _case()
    assert not pcw.case_bins(course, contact, back=0).any()


def test_a_household_member_met_since_infection_moves_from_before_onset_to_onset():
    """The same member falls in < 0 by its true first contact and in 0-3 from onset."""
    course, contact = _case(household_contacts_matrix=[_row(range(6), width=6)])
    contact["household_effective_contacts_infection_time"] = [np.nan]
    assert pcw.case_bins(course, contact, back=None)[0, 0].tolist() == [1, 0, 0, 0, 0, 0]
    assert pcw.case_bins(course, contact, back=0)[0, 0].tolist() == [0, 1, 0, 0, 0, 0]


def test_an_asymptomatic_case_adds_nothing():
    """Cheng's figure leaves out cases without onset."""
    course, contact = _case(household_contacts_matrix=[_row(range(6), width=6)])
    course["incubation_period"] = np.nan
    assert not pcw.case_bins(course, contact, back=None).any()
