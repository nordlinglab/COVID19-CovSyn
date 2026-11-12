# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B93: hospital transmission is monitored, not given a mechanism of its own.

The professor (2026-11-12): design a hospital-cluster mechanism only if the data has a cluster.
It does -- the recorded links 34 -> 42, 34 -> 45 and 27 -> 32 are all "the same hospital", and
5 of Taiwan's 55 local first-wave cases infected someone in health care, one of them (case 34)
two -- but run 23 already produces it at that rate: 8.2% of index cases infect one or more in
the health-care layer and 1.8% two or more (Taiwan 9.1% and 1.8%). The checklist reports both
shares so a later run that loses it shows.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.model.contact_measures import layer_offspring


def test_counts_each_case_infections_in_the_layer():
    """Infected contacts are 1, others 0; NaN entries are not infections."""
    contacts = [
        {"health_care_effective_contacts": [1, 0, 1]},
        {"health_care_effective_contacts": [0, np.nan]},
        {"health_care_effective_contacts": []},
        {"health_care_effective_contacts": None},
    ]
    assert layer_offspring(contacts, "health_care").tolist() == [2.0, 0.0, 0.0, 0.0]


def test_no_cases():
    """No index cases give an empty array, so a share of them is NaN rather than an error."""
    assert layer_offspring([], "health_care").shape == (0,)


def test_large_input():
    """Many cases: the counts sum to the infections, and each case is counted once."""
    rng = np.random.default_rng(1)
    contacts = [
        {"health_care_effective_contacts": list(rng.integers(0, 2, rng.integers(0, 30)))}
        for _ in range(5000)
    ]
    counts = layer_offspring(contacts, "health_care")
    assert len(counts) == 5000
    assert counts.sum() == sum(sum(c["health_care_effective_contacts"]) for c in contacts)


def test_unknown_layer_is_an_error():
    """A misspelt layer must not read as zero infections."""
    with pytest.raises(KeyError):
        layer_offspring([{"health_care_effective_contacts": [1]}], "hospital")


def test_arrays_as_saved_contact_data_holds_them():
    """Contact data loaded from .npy may hold arrays, whose truth value is ambiguous."""
    contacts = [
        {"health_care_effective_contacts": np.array([1.0, 0.0, 1.0])},
        {"health_care_effective_contacts": np.array([])},
    ]
    assert layer_offspring(contacts, "health_care").tolist() == [2.0, 0.0]
