# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B62: one scale-free community contact count per case, P(N = n) ~ (n + s)^-alpha.

The professor asked on 2026-10-12 to replace the B27 Gamma-Poisson count plus the B54 yes/no
mass event by a single power-law draw. alpha is P[204], appended so older vectors keep their
model; the shift s is P[28]; with P[204] present no mass event is drawn.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.model import data_synthesize as ds


def test_exponent_is_read_from_p204_only() -> None:
    """Vectors up to P[203] have no exponent; P[204] must be finite and positive."""
    assert ds.community_power_law_exponent(np.ones(204)) is None
    assert ds.community_power_law_exponent(np.r_[np.ones(204), 1.74]) == pytest.approx(1.74)
    with pytest.raises(ValueError):
        ds.community_power_law_exponent(np.r_[np.ones(204), 0.0])
    with pytest.raises(ValueError):
        ds.community_power_law_exponent(np.r_[np.ones(204), np.nan])


@pytest.mark.parametrize("shift", [0.8, 5.0])
def test_counts_follow_the_shifted_power_law(shift: float) -> None:
    """The empirical frequencies of the first counts match (n + s)^-alpha / Z."""
    np.random.seed(0)
    alpha, cap = 1.74, ds.COMMUNITY_MAX_CONTACTS
    draws = np.array([ds.draw_community_contact_count(alpha, shift) for _ in range(100_000)])
    assert draws.min() >= 0 and draws.max() <= cap
    support = np.arange(cap + 1, dtype=float)
    pmf = (support + shift) ** -alpha
    pmf /= pmf.sum()
    for n in range(4):
        observed = np.mean(draws == n)
        assert observed == pytest.approx(pmf[n], abs=4 * np.sqrt(pmf[n] / len(draws)) + 1e-4)


def test_larger_shift_gives_more_contacts() -> None:
    """The shift moves the body: a larger s has a larger median count."""
    np.random.seed(1)
    small = np.median([ds.draw_community_contact_count(1.74, 1.0) for _ in range(20_000)])
    np.random.seed(1)
    large = np.median([ds.draw_community_contact_count(1.74, 10.0) for _ in range(20_000)])
    assert large > small


def test_power_law_mode_draws_no_mass_event(
    run10_vector: np.ndarray, demographic_parameters: object
) -> None:
    """With P[204] the community layer has no event rows, even with an event probability of 1."""
    from covsyn.model.data_synthesis_main import run_covid

    vector = np.concatenate([run10_vector, [1.0, 1.67, 21, 1000, 1.0], [1.74]])
    vector[28] = 5.0
    for seed in range(5):
        _, _, _, contacts = run_covid(seed, vector.copy(), copy.deepcopy(demographic_parameters),
                                      save_file=False, mode="result")
        contact = contacts[0]
        assert contact.get("municipality_event_mask") is None
        rows = np.asarray(contact["municipality_contacts_matrix"]).shape[0]
        assert 0 <= rows <= ds.COMMUNITY_MAX_CONTACTS


def test_warm_start_pads_a_run16_vector() -> None:
    """A 204-value vector of runs 11-16 gets P[204] from the seed vector; others are left alone."""
    seed_vector = np.arange(205, dtype=float)
    padded = fo.pad_pre_b54_vector(np.zeros(204), seed_vector)
    assert padded.shape == (205,) and padded[204] == 204.0
    assert fo.pad_pre_b54_vector(np.zeros(199), seed_vector).shape == (205,)
    assert fo.pad_pre_b54_vector(np.zeros(203), seed_vector).shape == (203,)


def test_committed_bounds_carry_b62() -> None:
    """variable/ holds P[204] = 1.74, the events switched off and P[28] as the shift."""
    import pickle

    lower = np.load("variable/course_parameters_lb.npy")
    upper = np.load("variable/course_parameters_ub.npy")
    index = ds.COMMUNITY_POWER_LAW_INDEX - 37
    assert lower[index] == upper[index] == pytest.approx(1.74)
    event = ds.COMMUNITY_EVENT_FIRST_INDEX - 37
    assert lower[event] == upper[event] == 0.0
    contact = pickle.load(open("variable/contact_parameters.pkl", "rb"))
    assert contact["municipality_lower_bound"][0] == 0.5
    assert contact["municipality_upper_bound"][0] == 20.0


def test_one_day_shape_is_order_free_and_counts_empty_cases() -> None:
    """B63: pooled zero share and p90 do not depend on case order; an empty case is one zero day."""
    from covsyn.model import contact_measures as cm

    days = [np.array([0.0, 2.0, 7.0]), np.array([1.0]), np.array([0.0, 0.0, 9.0, 3.0])]
    assert cm.one_day_shape(days) == cm.one_day_shape(days[::-1])
    zero, p90 = cm.one_day_shape(days)
    assert zero == pytest.approx(3 / 8)
    assert p90 == pytest.approx(np.percentile(np.concatenate(days), 90))
    empty = cm.one_day_counts({"incubation_period": np.nan},
                              {"municipality_contacts_matrix": np.zeros((0, 0))}, "municipality")
    assert empty.tolist() == [0.0]
    assert all(np.isnan(cm.one_day_shape([])))


def test_b63_targets_charge_the_fu2010_shape_and_report_the_tracing_median() -> None:
    """B63: the Fu 2010 one-day shape is charged, the tracing median and tail ratio are reported."""
    targets = fo.OUTCOME_TARGETS
    assert targets["community_day_zero_share"] == (0.598, 0.641, 1.0)
    assert targets["community_day_p90"] == (5.0, 8.0, 1.0)
    assert targets["daily_municipality"][:2] == (1.20, 7.12)
    assert targets["community_median"][2] == 0.0
    assert targets["community_tail_ratio"][2] == 0.0
    import pickle

    contact = pickle.load(open("variable/contact_parameters.pkl", "rb"))
    assert contact["municipality_lower_bound"][1] == 0.0
