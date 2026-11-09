# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""F9: the imported factor is solved on m(q) = A + B q, not by q = 12 / m(1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def solve_line(monkeypatch_module):
    """The script's solve_line, loaded from its file without putting scripts/ on sys.path."""
    monkeypatch_module.chdir(ROOT)  # the script adds the relative 'src' to sys.path
    spec = importlib.util.spec_from_file_location(
        "calibrate_imported_factor", ROOT / "scripts" / "calibrate_imported_factor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_line


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_the_line_through_m0_and_m1_gives_the_b82_factor(solve_line):
    """Run 21: A = 7.27, m(1) = 46.93, data 12; the proportional update gave 0.256."""
    q = solve_line(0.0, 7.27, 1.0, 46.93, 12)
    assert q == pytest.approx((12 - 7.27) / (46.93 - 7.27))
    assert 7.27 + (46.93 - 7.27) * q == pytest.approx(12)
    assert q < 12 / 46.93


def test_the_line_through_two_low_measurements_matches_b82(solve_line):
    """B82's measurements: q = 0 gives 7.27 and 0.104 gives 11.86; 0.107 was chosen."""
    assert solve_line(0.0, 7.27, 0.104, 11.86, 12) == pytest.approx(0.107, abs=0.002)


def test_a_factor_outside_the_unit_interval_is_returned_unclipped(solve_line):
    """The caller must see that the data cannot be reached, not a clipped 1.0 or 0.0."""
    assert solve_line(0.0, 7.0, 1.0, 9.0, 12) > 1.0
    assert solve_line(0.0, 13.0, 1.0, 40.0, 12) < 0.0


def test_measurements_that_do_not_rise_with_q_are_an_error(solve_line):
    with pytest.raises(ValueError):
        solve_line(0.0, 10.0, 1.0, 10.0, 12)
