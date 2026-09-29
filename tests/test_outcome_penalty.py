# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The outcome penalty follows decisions B43 and B50.

B43: a target costs nothing inside its interval; outside it the miss is measured in units of the
interval width, capped at half the interval's centre, and costs miss**2 up to one unit and
2*miss - 1 beyond, times the target weight and OUTCOME_PENALTY_WEIGHT.
B50: the community tail ratio, whose interval is a wide data CI, is measured in units of its
lower bound instead.
"""

from __future__ import annotations

import math

import pytest

from covsyn.calibration import firefly_optimizer as fo


def _alone(name: str, value: float) -> float:
    """Penalty when only one target has a measured value."""
    return fo.outcome_penalty({name: value})


@pytest.mark.parametrize("name", sorted(n for n, (_, _, w) in fo.OUTCOME_TARGETS.items() if w > 0))
def test_inside_the_interval_costs_nothing(name: str) -> None:
    """Every charged target at the middle of its interval adds zero."""
    lo, hi, _ = fo.OUTCOME_TARGETS[name]
    centre = (lo + hi) / 2 if math.isfinite(hi) else lo
    assert _alone(name, centre) == 0.0


def test_quadratic_below_one_unit_with_the_capped_width() -> None:
    """closure_symptomatic [20, 32]: width 12 is below half the centre (13), so 12 is the unit."""
    lo, hi, weight = fo.OUTCOME_TARGETS["closure_symptomatic"]
    assert (lo, hi) == (20.0, 32.0)
    miss = 0.5 / 12.0
    assert _alone("closure_symptomatic", 19.5) == pytest.approx(
        fo.OUTCOME_PENALTY_WEIGHT * weight * miss**2)


def test_linear_beyond_one_unit() -> None:
    """A miss of two units costs 2 * 2 - 1 = 3 units, not 4."""
    lo, hi, weight = fo.OUTCOME_TARGETS["closure_symptomatic"]
    unit = fo.outcome_scale("closure_symptomatic", lo, hi)
    assert _alone("closure_symptomatic", hi + 2 * unit) == pytest.approx(
        fo.OUTCOME_PENALTY_WEIGHT * weight * 3.0)


def test_width_is_capped_at_half_the_centre() -> None:
    """For a band wider than half its centre the unit is half the centre (E66).

    [0, 10] has centre 5, so the cap is 2.5; [20, 32] has centre 26, cap 13, so its width 12 wins.
    """
    assert fo.outcome_scale("any", 0.0, 10.0) == 2.5
    assert fo.outcome_scale("any", 20.0, 32.0) == 12.0


def test_tail_ratio_is_measured_in_units_of_its_lower_bound() -> None:
    """B50: the tail ratio [5.5, 93.1] uses 5.5, so run 10's 4.0 still costs something."""
    lo, hi, weight = fo.OUTCOME_TARGETS["community_tail_ratio"]
    assert (lo, hi) == (5.5, 93.1)
    assert fo.outcome_scale("community_tail_ratio", lo, hi) == lo
    expected = fo.OUTCOME_PENALTY_WEIGHT * weight * ((lo - 4.0) / lo) ** 2
    assert _alone("community_tail_ratio", 4.0) == pytest.approx(expected)
    assert expected > 0.1


def test_missing_and_uncharged_targets_are_ignored() -> None:
    """A NaN measurement and a weight-0 (reported) target never add to the penalty."""
    assert _alone("closure_symptomatic", math.nan) == 0.0
    reported = next(n for n, (_, _, w) in fo.OUTCOME_TARGETS.items() if w == 0)
    assert _alone(reported, 1e9) == 0.0


def test_empty_interval_has_no_unit() -> None:
    """An interval of zero width cannot scale a miss."""
    assert fo.outcome_scale("any", 3.0, 3.0) is None
