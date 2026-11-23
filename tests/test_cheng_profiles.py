# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The index-case profiles of Cheng et al. (2020)'s period, built from the Taiwan case data.

The calibration's index case takes the profile of one case confirmed by 2020-03-18 (the span of
Cheng's study, case 100): its ten-year age group, whether it was imported, whether airport
screening found it, and its arrival counted from onset (E84, user decision 2026-11-23).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from covsyn.data_processing import build_cheng_profiles as bcp


def _cases(**columns):
    base = {"id": [1], "abroad/local": ["Abroad"], "age": ["20 to 29"],
            "way_of_discover": ["hospital notification"], "onset_of_symptom": ["2020-02-01"],
            "date_of_arrival_to_taiwan": ["2020-02-03"], "confirmed_date": ["2020-02-05"]}
    base.update(columns)
    return pd.DataFrame(base)


def test_an_imported_case_arriving_after_onset_has_a_positive_offset():
    """Onset 1 February, arrival 3 February: arrival is 2 days after onset."""
    row = bcp.build_profiles(_cases()).iloc[0]
    assert (row.case_id, row.age_low, row.age_high) == (1, 20, 29)
    assert bool(row.imported) and not bool(row.airport)
    assert row.arrival_from_onset == 2


def test_a_local_case_has_no_arrival():
    """A local case met people in Taiwan all along."""
    row = bcp.build_profiles(_cases(**{"abroad/local": ["Local"]})).iloc[0]
    assert not bool(row.imported)
    assert math.isnan(row.arrival_from_onset)


def test_an_imported_case_without_onset_keeps_an_unknown_arrival():
    """No onset date: the offset is unknown, to be drawn at simulation time."""
    row = bcp.build_profiles(_cases(onset_of_symptom=[None])).iloc[0]
    assert bool(row.imported)
    assert math.isnan(row.arrival_from_onset)


def test_airport_screening_is_flagged():
    """Isolated on arrival: the case met nobody that Cheng's investigation recorded."""
    row = bcp.build_profiles(_cases(way_of_discover=["airport screening"])).iloc[0]
    assert bool(row.airport)


def test_only_cases_confirmed_by_the_end_of_chengs_period_are_kept():
    """Confirmed on 18 March is in; 19 March is out."""
    cases = pd.concat([_cases(id=[1], confirmed_date=["2020-03-18"]),
                       _cases(id=[2], confirmed_date=["2020-03-19"])])
    assert bcp.build_profiles(cases).case_id.tolist() == [1]


def test_no_cases_give_an_empty_table_with_every_column():
    """Empty input is not an error."""
    out = bcp.build_profiles(_cases().iloc[:0])
    assert out.empty
    assert list(out.columns) == list(bcp.COLUMNS)


def test_the_real_data_give_chengs_hundred_cases():
    """Case 100 was confirmed on 18 March 2020; 71 imported, 4 found at the airport, all aged."""
    out = bcp.build_profiles(pd.read_excel(bcp.CASES, sheet_name="Individual Data"))
    assert len(out) == 100
    assert int(out.imported.sum()) == 71
    assert int(out.airport.sum()) == 4
    assert out.age_low.notna().all()
    assert int(out.arrival_from_onset.notna().sum()) == 51


@pytest.mark.parametrize(("label", "band"), [("20 to 29", (20, 29)), ("0 to 9", (0, 9)),
                                             ("80 to 89", (80, 89))])
def test_age_labels_give_their_band(label, band):
    """The data write each group as 'low to high'."""
    assert bcp.age_band(label) == band


@pytest.mark.parametrize("label", [None, float("nan"), "", "x", "20-29", "twenty to 29"])
def test_unreadable_age_labels_give_no_band(label):
    """Anything else is a missing age, never a guess."""
    low, high = bcp.age_band(label)
    assert math.isnan(low) and math.isnan(high)


@given(low=st.integers(0, 120), width=st.integers(0, 20))
def test_any_written_band_is_read_back(low, width):
    """Property: 'a to b' always reads back as (a, b)."""
    assert bcp.age_band(f"{low} to {low + width}") == (low, low + width)


@given(text=st.text(max_size=20))
def test_arbitrary_text_gives_a_band_or_nothing(text):
    """Property: the result is a band with low <= high, or two NaNs."""
    low, high = bcp.age_band(text)
    assert (math.isnan(low) and math.isnan(high)) or low <= high
