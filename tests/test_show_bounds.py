# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""show_bounds converts daily attack rates with the measured contact days (review of B75)."""

import numpy as np
import pytest

from covsyn.calibration.sar_anchors import ATTACK_RATE_SLICE, MEAN_CONTACT_DAYS
from covsyn.validation import show_bounds


def test_cumulative_rates_use_mean_contact_days():
    p = np.full(160, 0.01)
    rows = {row[0]: row for row in show_bounds.attack_rate_rows(p, p * 0.5, p * 2)}
    assert set(rows) == set(ATTACK_RATE_SLICE)
    for layer, (_, daily, _, _, cumulative, _, _) in rows.items():
        assert daily == pytest.approx(0.01)
        assert cumulative == pytest.approx(100 * (1 - 0.99 ** MEAN_CONTACT_DAYS[layer]))


def test_each_layer_reads_its_own_block():
    p = np.zeros(160)
    a, b = ATTACK_RATE_SLICE["workplace"]
    p[a:b] = 0.02
    rows = {row[0]: row for row in show_bounds.attack_rate_rows(p, p, p)}
    assert rows["workplace"][1] == pytest.approx(0.02)
    assert rows["household"][1] == 0.0
