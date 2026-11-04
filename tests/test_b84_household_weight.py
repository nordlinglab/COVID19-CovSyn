# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B84: the household contact target is weighted above the layers that already hold.

Run 21 measured 2.115 household contacts a day before onset against the B25 band [2.27, 3.28],
the bootstrap interval reaching only 2.216. The shortfall is a calibration one, not a structural
one: the 1,000 index cases have 2.656 household members on average (14.2% live alone), so the
ceiling is 2.656 contacts a day when every member is met every day and 2.523 at the upper bound
of household_healthy_p. The target carried the same weight as four other layers that were
already inside their bands, so the search had no reason to spend on it.
"""

from __future__ import annotations

from covsyn.calibration.firefly_optimizer import OUTCOME_TARGETS

LAYER_TARGETS = ('daily_household', 'daily_school', 'daily_workplace_employed',
                 'daily_municipality')


def test_the_household_target_is_weighted_above_the_other_layers():
    household = OUTCOME_TARGETS['daily_household'][2]
    assert household > 1.0
    for name in LAYER_TARGETS:
        if name != 'daily_household':
            assert household > OUTCOME_TARGETS[name][2]


def test_the_band_is_unchanged():
    """B84 moves no bound: the band stays the 2020 survey and Fu 2010 figures of B25/B61."""
    assert OUTCOME_TARGETS['daily_household'][:2] == (2.27, 3.28)


def test_the_target_is_still_reachable_inside_the_measured_ceiling():
    """2.27 sits under the 2.523 the household parameters can reach, so weighting is enough."""
    measured_ceiling_at_the_parameter_bound = 2.523
    assert OUTCOME_TARGETS['daily_household'][0] < measured_ceiling_at_the_parameter_bound


def test_the_other_layer_weights_are_untouched():
    assert OUTCOME_TARGETS['daily_school'][2] == 1.0
    assert OUTCOME_TARGETS['daily_workplace_employed'][2] == 1.0
    assert OUTCOME_TARGETS['daily_municipality'][2] == 1.0
    assert OUTCOME_TARGETS['daily_workplace'][2] == 0.0
    assert OUTCOME_TARGETS['daily_health_care'][2] == 0.0
