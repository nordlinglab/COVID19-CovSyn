# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""An imported index case meets nobody in Taiwan before it arrives (B66's seed rule).

The probe gives each simulated index case the profile of one of Cheng et al.'s 100 cases:
local, imported with its arrival day counted from onset, or isolated at the airport. Contact
days before arrival are removed, a contact left without any day is dropped together with its
infection record, and an airport case meets nobody.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_cheng_arrival as pca  # noqa: E402


def _contact(household_rows, infections):
    contact = {key: [] for key in pca.pcw.MATRIX.values()}
    contact["household_contacts_matrix"] = [np.array(r, dtype=bool) for r in household_rows]
    contact["household_effective_contacts_infection_time"] = infections
    return contact


def test_days_before_arrival_are_removed():
    """A member met on days 0-5 by a case arriving on day 3 is met on days 3-5 only."""
    out = pca.arrived(_contact([[1, 1, 1, 1, 1, 1]], [np.nan]), arrival=3, airport=False)
    assert out["household_contacts_matrix"][0].tolist() == [0, 0, 0, 1, 1, 1]


def test_a_contact_met_only_before_arrival_is_dropped_with_its_infection():
    """The second member, met only on days 0-1, goes; the third keeps its infection record."""
    rows = [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    out = pca.arrived(_contact(rows, [np.nan, 2.0, 5.0]), arrival=2, airport=False)
    assert len(out["household_contacts_matrix"]) == 2
    assert out["household_effective_contacts_infection_time"] == [np.nan, 5.0] or (
        np.isnan(out["household_effective_contacts_infection_time"][0])
        and out["household_effective_contacts_infection_time"][1] == 5.0
    )


def test_an_airport_case_meets_nobody():
    """Isolated on arrival, it has no contact in Taiwan."""
    out = pca.arrived(_contact([[1, 1, 1]], [np.nan]), arrival=1, airport=True)
    assert len(out["household_contacts_matrix"]) == 0


def test_an_arrival_before_infection_removes_nothing():
    """A case already in Taiwan when infected keeps every contact."""
    rows = [[1, 0, 1]]
    out = pca.arrived(_contact(rows, [np.nan]), arrival=-4, airport=False)
    assert out["household_contacts_matrix"][0].tolist() == [1, 0, 1]


def test_the_input_is_not_changed():
    """The simulation's own record stays as it was."""
    contact = _contact([[1, 1, 1]], [np.nan])
    pca.arrived(contact, arrival=2, airport=False)
    assert contact["household_contacts_matrix"][0].tolist() == [1, 1, 1]


def test_a_local_profile_gives_no_arrival():
    """Local cases are not truncated."""
    assert pca.arrival_day({"imported": False, "offset": None, "airport": False}, onset=5) is None


def test_an_imported_profile_counts_its_arrival_from_onset():
    """Arrival 3 days after onset on day 5 of infection is day 8."""
    assert pca.arrival_day({"imported": True, "offset": 3, "airport": False}, onset=5) == 8
