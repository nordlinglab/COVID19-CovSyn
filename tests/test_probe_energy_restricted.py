# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The restricted energy term: age and onset to confirmation, complete cases, plain distance.

The 2026-11-16 proposal (E105-E107): compare only what both sides record, on the cases that
record both, with |x - y| instead of log(d + 1e-16), scaled by the Taiwanese side's mean and SD.
The sums are taken over the distinct values with their counts, which must equal the sums over
every pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_energy_restricted as per  # noqa: E402


def _direct(x, y):
    """Half the V- and U-statistics from every pair, the reference the distinct sums must meet."""
    dxy = np.sqrt(((x[:, None] - y[None]) ** 2).sum(-1)).sum()
    dxx = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).sum()
    dyy = np.sqrt(((y[:, None] - y[None]) ** 2).sum(-1)).sum()
    n, m = len(x), len(y)
    half_v = dxy / (n * m) - dxx / (2 * n * n) - dyy / (2 * m * m)
    half_u = dxy / (n * m) - dxx / (2 * n * (n - 1)) - dyy / (2 * m * (m - 1))
    return half_v, half_u


_cells = hnp.arrays(np.float64, st.tuples(st.integers(2, 25), st.just(2)),
                    elements=st.integers(0, 6).map(float))  # fmt: skip


@settings(max_examples=60, deadline=None)
@given(x=_cells, y=_cells)
def test_the_distinct_value_sums_equal_the_sums_over_every_pair(x, y):
    """Ties are what the data is made of, so the shortcut must be exact on them."""
    got = per.energy_halves(x, y)
    assert got == pytest.approx(_direct(x, y), rel=1e-12, abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(x=_cells, y=_cells)
def test_the_v_form_is_never_negative(x, y):
    """Energy distance's V-statistic is a squared distance between distributions."""
    assert per.energy_halves(x, y)[0] >= -1e-12


def test_one_simulated_case_leaves_the_u_form_undefined():
    """With m = 1 there is no pair on the simulated side to average."""
    half_v, half_u = per.energy_halves(np.array([[0.0, 1.0], [1.0, 2.0]]), np.array([[0.0, 1.0]]))
    assert np.isfinite(half_v) and np.isnan(half_u)


def test_no_simulated_case_leaves_both_undefined():
    """Nothing to compare gives NaN, not a number that looks like a fit."""
    assert all(np.isnan(per.energy_halves(np.array([[0.0, 1.0], [1.0, 2.0]]), np.empty((0, 2)))))


def test_only_cases_with_both_columns_are_compared():
    """A case missing age or onset to confirmation is left out, never filled in."""
    rows = np.full((4, 9), np.nan)
    rows[0, [0, 7]] = 30, 5
    rows[1, 0] = 40  # no onset to confirmation
    rows[2, 7] = 3  # no age
    rows[3, [0, 7]] = 50, 2
    assert per.complete_cases(rows).tolist() == [[30.0, 5.0], [50.0, 2.0]]


def test_the_scale_is_taiwans_alone():
    """The simulated side must not move the yardstick it is measured with."""
    taiwan = np.array([[20.0, 2.0], [40.0, 6.0], [60.0, 10.0]])
    a = per.restricted_energy(taiwan, np.array([[30.0, 4.0], [50.0, 8.0]]))
    b = per.restricted_energy(taiwan * 1.0, np.array([[30.0, 4.0], [50.0, 8.0]]))
    scaled = (taiwan - taiwan.mean(0)) / taiwan.std(0)
    sim = (np.array([[30.0, 4.0], [50.0, 8.0]]) - taiwan.mean(0)) / taiwan.std(0)
    assert a == b == pytest.approx(per.energy_halves(scaled, sim))


def test_the_same_sample_on_both_sides_gives_zero_v_form():
    """Identical samples are at distance 0."""
    x = np.array([[20.0, 2.0], [40.0, 6.0], [40.0, 6.0], [60.0, 10.0]])
    assert per.restricted_energy(x, x)[0] == pytest.approx(0.0, abs=1e-12)


def test_with_energy_swaps_the_term_in_every_cost():
    """The new term replaces the old one; the rest of the cost is untouched."""
    rows = [{"cost": "14.0", "energy_old": "-0.3", "energy_u": "0.05", "label": "run19"}]
    out = per.with_energy(rows, "energy_u")
    assert out[0]["cost"] == pytest.approx(14.35) and out[0]["label"] == "run19"


def test_with_energy_keeps_a_failed_evaluations_charge():
    """A failed simulation has no term to swap."""
    rows = [{"cost": "1000000.0", "energy_old": "nan", "energy_u": "nan"}]
    assert per.with_energy(rows, "energy_u")[0]["cost"] == 1e6


def test_with_energy_of_no_rows_is_no_rows():
    """Empty in, empty out."""
    assert per.with_energy([], "energy_u") == []
