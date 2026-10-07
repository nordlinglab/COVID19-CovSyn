# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B56 (finding E83): the city check tests for a city effect instead of bounding max / min.

The ratio of the largest to the smallest city mean over 15 small, heavy-tailed city samples is
mostly noise: with the city labels shuffled, which removes any city effect by construction, it
still fails the old bound of 1.6 in 96.6% of shuffles. The check now asks whether the observed
ratio is unusual among shuffles of the labels.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.model import contact_measures as cm
from covsyn.validation import city_effect as ce


def _sample(effect: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cities = np.repeat(np.arange(10), 60)
    scale = 1.0 + effect * cities / 9.0           # city 9 has (1 + effect) times city 0's mean
    values = rng.negative_binomial(0.5, 0.5 / (0.5 + 6.0 * scale))
    return values.astype(float), cities


def test_ratio_uses_cities_with_enough_cases_only() -> None:
    values = np.array([1.0] * 20 + [4.0] * 20 + [100.0] * 3)
    cities = np.array(['a'] * 20 + ['b'] * 20 + ['c'] * 3)
    assert ce.max_min_city_ratio(values, cities, min_cases=20) == pytest.approx(4.0)


def test_ratio_without_enough_cases_is_not_a_number() -> None:
    assert np.isnan(ce.max_min_city_ratio(np.ones(5), np.array(list('abcde')), min_cases=20))


def test_no_city_effect_is_not_detected() -> None:
    values, cities = _sample(effect=0.0)
    assert ce.permutation_p_value(values, cities, permutations=500, seed=1) > 0.05


def test_a_strong_city_effect_is_detected() -> None:
    values, cities = _sample(effect=4.0)
    assert ce.permutation_p_value(values, cities, permutations=500, seed=1) < 0.05


def test_permutation_p_value_is_reproducible_and_never_zero() -> None:
    values, cities = _sample(effect=10.0)
    first = ce.permutation_p_value(values, cities, permutations=200, seed=3)
    assert first == ce.permutation_p_value(values, cities, permutations=200, seed=3)
    assert first >= 1 / 201


# --- B56: contacts per case against Jian et al. 2020 ---------------------------------

def test_contacts_per_case_count_every_layer() -> None:
    contact = {'household_effective_contacts': [0, 1], 'school_effective_contacts': [],
               'workplace_effective_contacts': [0], 'health_care_effective_contacts': None,
               'municipality_effective_contacts': [0, 0, 0]}
    assert cm.contacts_per_case(contact) == 6


# --- E89: contacts per case in Jian et al. 2020's tracing window -----------------------

def _traced_case(onset: float, isolation: int) -> tuple[dict, dict]:
    """Five household contacts, first met on days 0, 2, 3, 6 and 9 (day 0 = infection)."""
    m = np.zeros((5, 12), dtype=bool)
    for row, day in enumerate((0, 2, 3, 6, 9)):
        m[row, day] = True
    contact = {'household_contacts_matrix': m}
    for key in ('school_class', 'workplace', 'health_care', 'municipality'):
        contact[f'{key}_contacts_matrix'] = np.zeros((0, 12), dtype=bool)
    return {'incubation_period': onset, 'monitor_isolation_period': isolation}, contact


def test_traced_window_runs_from_two_days_before_onset_to_isolation() -> None:
    course, contact = _traced_case(onset=5.0, isolation=8)
    # window 3..8: the contacts on days 3 and 6
    assert cm.contacts_in_tracing_window(course, contact) == 2


def test_traced_window_of_an_asymptomatic_case_ends_at_isolation() -> None:
    course, contact = _traced_case(onset=float('nan'), isolation=9)
    # window 7..9: the contact on day 9
    assert cm.contacts_in_tracing_window(course, contact) == 1


def test_traced_window_lead_is_adjustable() -> None:
    course, contact = _traced_case(onset=5.0, isolation=8)
    assert cm.contacts_in_tracing_window(course, contact, lead_days=4) == 3   # days 1..8
    assert cm.contacts_in_tracing_window(course, contact, lead_days=0) == 1   # days 5..8


def test_contact_met_across_the_window_edge_counts_once() -> None:
    course, contact = _traced_case(onset=5.0, isolation=8)
    contact['household_contacts_matrix'][0, :] = True           # met every day
    assert cm.contacts_in_tracing_window(course, contact) == 3
