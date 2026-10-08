# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Expected outcomes in the objective (finding E95).

The objective's infections per index case and per-contact attack rates now use the expected
infections (sums of per-contact infection probabilities, E38) and, in the municipality layer,
the expected event contacts and infections of B55, instead of realised Bernoulli counts. The
share of zero-day pre-onset windows is the Gamma probability of a draw that rounds to zero.
"""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest

from covsyn.model import data_synthesize as ds

GRID = 2.0**20
EVENTS = [0.10, 1.49, 21, 1000, 1.0]
LAYERS = ("household", "school", "workplace", "health_care", "municipality")


def _index_contact(vector: np.ndarray, demographic_parameters: object, seed: int) -> tuple:
    """The index case's course and contact dicts of one simulation."""
    from covsyn.model.data_synthesis_main import run_covid

    _, _, courses, contacts = run_covid(
        seed, vector.copy(), copy.deepcopy(demographic_parameters), save_file=False, mode="result"
    )
    return courses[0], contacts[0]


@pytest.mark.parametrize("seed", range(4))
def test_expected_outcomes_are_saved_on_the_grid(
    seed: int, run10_vector: np.ndarray, demographic_parameters: object
) -> None:
    """Every expected value is saved on the 2**-20 grid, so sums are exact in any order."""
    vector = np.concatenate([run10_vector, EVENTS])
    _, contact = _index_contact(vector, demographic_parameters, seed)
    expected = contact["expected_outcomes"]
    assert set(expected) == {f"expected_infections_{layer}" for layer in LAYERS} | {
        "expected_contacts_municipality"
    }
    for value in expected.values():
        assert value >= 0
        assert value * GRID == np.round(value * GRID)


@pytest.mark.parametrize("seed", range(4))
def test_municipality_expectation_replaces_the_sampled_event_rows(
    seed: int, run10_vector: np.ndarray, demographic_parameters: object
) -> None:
    """Municipality expected contacts are the ordinary rows plus the expected event contacts."""
    vector = np.concatenate([run10_vector, EVENTS])
    _, contact = _index_contact(vector, demographic_parameters, seed)
    rows = len(contact["municipality_effective_contacts"])
    ordinary = rows - int(np.sum(contact["municipality_event_mask"][:rows]))
    events = float(np.sum(contact["municipality_event_expected_contacts"]))
    assert contact["expected_outcomes"]["expected_contacts_municipality"] == pytest.approx(
        ordinary + events, abs=2 / GRID
    )


def test_expected_infections_match_the_realised_ones_on_average(
    run10_vector: np.ndarray, demographic_parameters: object
) -> None:
    """Over many index cases the expected and realised household infections agree."""
    expected, realised = [], []
    for seed in range(150):
        _, contact = _index_contact(run10_vector, demographic_parameters, seed)
        expected.append(contact["expected_outcomes"]["expected_infections_household"])
        realised.append(sum(1 for x in contact["household_effective_contacts"] or [] if x == 1))
    realised_arr = np.asarray(realised, dtype=float)
    standard_error = realised_arr.std(ddof=1) / np.sqrt(len(realised_arr))
    assert abs(np.mean(expected) - realised_arr.mean()) < 4 * standard_error + 1e-9
    # The point of E95: the expectation varies much less than the realised count.
    assert np.std(expected) < realised_arr.std()


def test_measure_outcomes_charges_the_expectation_and_falls_back_without_it(
    run10_vector: np.ndarray, demographic_parameters: object
) -> None:
    """The objective uses expected infections, and realised counts for old output."""
    pytest.importorskip("sklearn")
    from covsyn.calibration import firefly_optimizer as fo

    vector = np.concatenate([run10_vector, EVENTS])
    cases = [(*_index_contact(vector, demographic_parameters, s), None) for s in range(6)]
    measured = fo.measure_outcomes(cases)
    mean = np.mean([c["expected_outcomes"]["expected_infections_household"] for _, c, _ in cases])
    assert measured["infections_per_index_household"] == pytest.approx(mean)

    old = [
        (course, {k: v for k, v in c.items() if k != "expected_outcomes"}, d)
        for course, c, d in cases
    ]
    realised = np.mean(
        [sum(1 for x in c["household_effective_contacts"] or [] if x == 1) for _, c, _ in old]
    )
    assert fo.measure_outcomes(old)["infections_per_index_household"] == pytest.approx(realised)


def test_zero_window_probability_matches_the_draws(run10_vector: np.ndarray) -> None:
    """The Gamma probability of rounding to zero is the share of zero-day windows drawn."""
    pytest.importorskip("sklearn")
    from covsyn.calibration import firefly_optimizer as fo

    P = run10_vector
    owner = SimpleNamespace(incubation_period_shape=P[41], incubation_period_scale=P[42])
    np.random.seed(5)
    random.seed(5)
    n = 40000
    zeros = sum(ds.Draw_course_of_disease_data.draw_pre_onset_window(owner) == 0 for _ in range(n))
    p = fo.pre_onset_zero_probability(P)
    assert zeros / n == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / n))
