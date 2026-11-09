# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""F9: the imported factor is solved on m(q) = A + B q, not by q = 12 / m(1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from calibrate_imported_factor import solve_line  # noqa: E402


def test_the_line_through_m0_and_m1_gives_the_b82_factor():
    """Run 21: A = 7.27, m(1) = 46.93, data 12; the proportional update gave 0.256."""
    q = solve_line(0.0, 7.27, 1.0, 46.93, 12)
    assert q == pytest.approx((12 - 7.27) / (46.93 - 7.27))
    assert 7.27 + (46.93 - 7.27) * q == pytest.approx(12)
    assert q < 12 / 46.93


def test_the_secant_through_two_low_measurements_matches_b82():
    """B82's measurements: q = 0 gives 7.27 and 0.104 gives 11.86; 0.107 was chosen."""
    assert solve_line(0.0, 7.27, 0.104, 11.86, 12) == pytest.approx(0.107, abs=0.002)


def test_the_factor_is_clipped_to_the_unit_interval():
    assert solve_line(0.0, 7.0, 1.0, 9.0, 12) == 1.0
    assert solve_line(0.0, 13.0, 1.0, 40.0, 12) == 0.0


def test_measurements_that_do_not_rise_with_q_are_an_error():
    with pytest.raises(ValueError):
        solve_line(0.0, 10.0, 1.0, 10.0, 12)
