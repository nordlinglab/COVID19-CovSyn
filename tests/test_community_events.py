# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mass events in the municipality layer (decision B54).

A case attends at most one mass event before isolation, with probability P[199].
The event size follows a discrete power law P(S = s) proportional to s^-gamma on
[s_min, s_max] (P[200], P[201], P[202]); all event contacts meet the case on one day,
and their per-contact attack rate is the municipality rate times P[203].
Vectors of 199 parameters (run 10 and earlier) have no events at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from covsyn.model import data_synthesize as ds

EVENT = {'probability': 0.1, 'exponent': 1.5, 'min_size': 20, 'max_size': 1000,
         'risk_ratio': 0.05}


def _vector(*event: float) -> np.ndarray:
    return np.concatenate([np.zeros(199), np.asarray(event, dtype=float)])


# --- parameter parsing ----------------------------------------------------------------

def test_run10_vector_has_no_events() -> None:
    """A 199-long vector keeps the pre-B54 model."""
    assert ds.community_event_parameters(np.zeros(199)) is None


def test_event_parameters_are_read_from_p199_to_p203() -> None:
    parsed = ds.community_event_parameters(_vector(0.1, 1.5, 20, 1000, 0.05))
    assert parsed == pytest.approx(EVENT)
    assert isinstance(parsed['min_size'], int) and isinstance(parsed['max_size'], int)


def test_partial_event_block_is_rejected() -> None:
    """Three of the five event parameters is a malformed vector, not 'no events'."""
    with pytest.raises(ValueError):
        ds.community_event_parameters(_vector(0.1, 1.5, 20))


@pytest.mark.parametrize("event", [
    (-0.1, 1.5, 20, 1000, 0.05),   # probability below 0
    (1.1, 1.5, 20, 1000, 0.05),    # probability above 1
    (0.1, 0.0, 20, 1000, 0.05),    # exponent not positive
    (0.1, 1.5, 0, 1000, 0.05),     # minimum size below 1
    (0.1, 1.5, 50, 20, 0.05),      # maximum below minimum
    (0.1, 1.5, 20, 1000, -0.01),   # negative risk ratio
    (0.1, np.nan, 20, 1000, 0.05),  # not a number
])
def test_invalid_event_parameters_are_rejected(event: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        ds.community_event_parameters(_vector(*event))


# --- event size -----------------------------------------------------------------------

def _sizes(n: int, exponent: float = 1.5, min_size: int = 20, max_size: int = 1000,
           seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    return np.array([ds.draw_event_size(exponent, min_size, max_size) for _ in range(n)])


def test_event_size_stays_inside_its_bounds() -> None:
    sizes = _sizes(20000)
    assert sizes.min() >= 20 and sizes.max() <= 1000
    assert sizes.dtype.kind == 'i'


def test_event_size_with_equal_bounds_is_that_size() -> None:
    assert set(_sizes(50, min_size=7, max_size=7)) == {7}


def test_event_size_mean_matches_the_truncated_power_law() -> None:
    support = np.arange(20, 1001)
    weights = support ** -1.5
    expected = np.sum(support * weights) / np.sum(weights)
    sizes = _sizes(200000)
    assert sizes.mean() == pytest.approx(expected, rel=0.03)


def test_event_size_ccdf_is_a_straight_line_on_log_log_axes() -> None:
    """The survival function of a power law with exponent gamma has slope 1 - gamma."""
    sizes = _sizes(200000, max_size=100000)
    grid = np.array([20, 40, 80, 160, 320, 640])
    survival = np.array([np.mean(sizes >= s) for s in grid])
    slope = np.polyfit(np.log(grid), np.log(survival), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.05)


def test_event_size_tail_is_far_heavier_than_the_negative_binomial() -> None:
    """Run 10's NB(6.53, 0.51) gives P(N >= 446) of about 2e-16; events must reach it."""
    assert np.mean(_sizes(100000) >= 446) > 0.01


def test_event_size_is_reproducible_under_a_seed() -> None:
    np.testing.assert_array_equal(_sizes(100, seed=3), _sizes(100, seed=3))


# --- event contacts -------------------------------------------------------------------

def _contact_owner(event: dict | None, end_day: int = 9, onset: float = np.nan) -> SimpleNamespace:
    course = SimpleNamespace(incubation_period=onset, monitor_isolation_period=end_day)
    owner = SimpleNamespace(community_event=event, course_of_disease_data_object=course,
                            generate_logistic_contact_p=ds.Draw_contact_data.generate_logistic_contact_p)
    owner.daily_contact_p = ds.Draw_contact_data.daily_contact_p.__get__(owner)
    return owner


def _event_rows(event: dict | None, room: int = 10**6, seed: int = 0, end_day: int = 9,
                onset: float = np.nan) -> np.ndarray:
    np.random.seed(seed)
    p = [5.0, 0.7, 0.05, 0.05]
    return ds.Draw_contact_data.draw_community_event_contacts(
        _contact_owner(event, end_day, onset), p, 1.0, 0.0, 1.0, end_day, room)


def test_no_event_parameters_means_no_event_contacts() -> None:
    assert _event_rows(None).shape == (0, 10)


def test_probability_zero_never_draws_an_event() -> None:
    for seed in range(50):
        assert _event_rows({**EVENT, 'probability': 0.0}, seed=seed).shape[0] == 0


def test_probability_one_always_draws_one_event_on_one_day() -> None:
    for seed in range(50):
        rows = _event_rows({**EVENT, 'probability': 1.0}, seed=seed)
        assert 20 <= rows.shape[0] <= 1000
        assert rows.shape[1] == 10
        assert np.all(rows.sum(axis=1) == 1), 'each event contact is met exactly once'
        assert len(set(np.argmax(rows, axis=1))) == 1, 'all on the same day'


def test_event_frequency_follows_its_probability() -> None:
    hits = sum(_event_rows(EVENT, seed=s).shape[0] > 0 for s in range(4000))
    assert hits / 4000 == pytest.approx(0.1, abs=0.02)


def test_event_size_is_capped_by_the_remaining_population() -> None:
    rows = _event_rows({**EVENT, 'probability': 1.0, 'min_size': 500}, room=37)
    assert rows.shape[0] == 37


def test_no_room_left_gives_no_event_contacts() -> None:
    assert _event_rows({**EVENT, 'probability': 1.0}, room=0).shape[0] == 0


def test_event_day_lies_inside_a_one_day_window() -> None:
    rows = _event_rows({**EVENT, 'probability': 1.0}, end_day=0)
    assert rows.shape[1] == 1 and np.all(rows[:, 0])


def test_daily_contact_p_is_constant_for_an_asymptomatic_case() -> None:
    owner = _contact_owner(None, onset=np.nan)
    np.testing.assert_array_equal(owner.daily_contact_p([5.0, 0.7, 0.03, 0.01], 1.0, 0.0, 1.0, 4),
                                  np.full(5, 0.03))


# --- search bounds --------------------------------------------------------------------

def test_search_bounds_hold_valid_event_parameters() -> None:
    """Seed, lower and upper vectors of apply_phase_d_parameters all parse (B54)."""
    from covsyn.calibration import apply_phase_d_parameters as apply

    assert apply.COMMUNITY_EVENT_COURSE_INDEX + 37 == ds.COMMUNITY_EVENT_FIRST_INDEX
    assert len(apply.COMMUNITY_EVENT) == len(ds.COMMUNITY_EVENT_FIELDS)
    for column in range(3):
        values = [row[column] for row in apply.COMMUNITY_EVENT]
        assert ds.community_event_parameters(_vector(*values)) is not None
    for seed, lower, upper in apply.COMMUNITY_EVENT:
        assert lower <= seed <= upper
