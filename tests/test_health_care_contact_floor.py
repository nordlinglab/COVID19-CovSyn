# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Floor on health care contacts per local index case, derived from Cheng et al. 2020.

Fewer contacts than the floor cannot reproduce Cheng's lower bound of medical infections per
local index case (0.0621) even at the upper 95% bound of his per-contact attack rate, 6 / 698
(Table 2: 0.9%, 0.4-1.9%).
"""

from __future__ import annotations

import pytest

from covsyn.calibration import sar_anchors as sa


def test_wilson_interval_reproduces_cheng_printed_interval() -> None:
    """Cheng prints 0.9% (0.4-1.9) for 6 / 698; the Wilson interval rounds to exactly that."""
    lo, hi = sa.wilson_interval(6, 698)
    assert round(100 * lo, 1) == 0.4
    assert round(100 * hi, 1) == 1.9
    assert hi == pytest.approx(0.018625, abs=1e-5)


def test_floor_is_lower_infections_over_upper_attack_rate() -> None:
    """0.06212 infections per local index case / 0.018625 per contact = 3.335 contacts."""
    assert sa.LAYER_INFECTIONS_PER_INDEX["health_care"][0] == pytest.approx(0.062122, abs=1e-5)
    assert sa.health_care_contact_floor() == pytest.approx(3.335, abs=0.002)


def test_floor_is_below_jian_local_mean() -> None:
    """Jian 2020 Table 1: 544 hospital or clinic contacts of 55 local cases, 9.89 per case."""
    assert sa.health_care_contact_floor() < sa.JIAN2020_LOCAL_HEALTH_CARE_PER_CASE
    assert sa.JIAN2020_LOCAL_HEALTH_CARE_PER_CASE == pytest.approx(544 / 55)


def test_wilson_interval_of_zero_events_starts_at_zero() -> None:
    lo, hi = sa.wilson_interval(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.05


def test_wilson_interval_narrows_with_a_large_sample() -> None:
    lo, hi = sa.wilson_interval(8600, 1_000_000)
    assert lo < 0.0086 < hi
    assert hi - lo < 0.0004


@pytest.mark.parametrize("events, trials", [(1, 0), (-1, 10), (11, 10)])
def test_wilson_interval_rejects_impossible_counts(events: int, trials: int) -> None:
    with pytest.raises(ValueError):
        sa.wilson_interval(events, trials)


def test_health_care_lower_bound_is_cheng_per_contact_lower_bound() -> None:
    """Review of B76: the per-contact lower bound no longer presupposes run 18's 2.597 contacts.

    Built from Cheng's lower infections over 2.597 contacts it was 2.39% cumulative, above
    Cheng's own upper per-contact rate of 1.86%, so the floor's contacts could not reach the
    lower half of the infections target. It is now Cheng's Wilson lower bound, 0.395%.
    """
    lower, _, upper = sa.LAYER_CUMULATIVE_SAR["health_care"]
    assert lower == pytest.approx(sa.wilson_interval(6, 698)[0])
    assert lower < upper


def test_committed_bounds_carry_the_health_care_lower_bound() -> None:
    """variable/course_parameters_lb.npy was rebuilt after the change (E53 discipline)."""
    from pathlib import Path

    import numpy as np

    root = Path(__file__).resolve().parents[1]
    lb = np.load(root / "variable" / "course_parameters_lb.npy")
    a, b = sa.ATTACK_RATE_SLICE["health_care"]
    _, expected, _ = sa.attack_rate_block("health_care")
    assert lb[a:b] == pytest.approx(expected)
