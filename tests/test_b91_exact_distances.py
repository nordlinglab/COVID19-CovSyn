# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B91: the energy term's distances are computed coordinate by coordinate.

The old distances came from |x|^2 - 2x.y + |y|^2 through the platform's BLAS. Two identical
cases then came out as residuals of 3-8e-8 instead of 0 on OpenBLAS (375 of 985 such pairs),
and log(d + 1e-16) turned that rounding into -17 instead of -36.8, so Linux and macOS gave
different objectives (E103). The professor (2026-11-12) allowed changing the objective when it
costs no more compute; subtracting coordinate by coordinate gives exact zeros on every
platform, and computing the statistic once instead of twice (the cost discarded estat's one
bootstrap draw) keeps the term cheaper than before.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo


def test_identical_cases_are_exactly_zero_apart():
    """The case E103 found: equal rows at different positions, with large coordinates."""
    rng = np.random.default_rng(3)
    rows = rng.normal(0, 1, (40, 9)) * np.array([1e3, 1, 1, 50, 1, 1, 7, 1, 1])
    rows[rng.random(rows.shape) < 0.6] = np.nan
    x = np.vstack([rows, rows])
    d = fo.rw_nan_euclidean_distances(x, x)
    pairs = d[np.arange(40), np.arange(40) + 40]
    finite = np.isfinite(pairs)
    assert finite.any()
    assert np.all(pairs[finite] == 0.0)


def test_distance_uses_only_coordinates_present_in_both():
    """The docstring's example: [3, na, na, 6] and [1, na, 4, 5] are sqrt(4 + 1) apart."""
    x = np.array([[3, np.nan, np.nan, 6]])
    y = np.array([[1, np.nan, 4, 5]])
    assert fo.rw_nan_euclidean_distances(x, y)[0, 0] == pytest.approx(np.sqrt(5.0))
    assert fo.rw_nan_euclidean_distances(x, y, squared=True)[0, 0] == pytest.approx(5.0)


def test_no_shared_coordinate_gives_nan():
    """A pair with no coordinate observed in both has no distance."""
    x = np.array([[1.0, np.nan]])
    y = np.array([[np.nan, 2.0]])
    assert np.isnan(fo.rw_nan_euclidean_distances(x, y)[0, 0])


def test_distances_match_a_plain_loop():
    """Large input: every entry equals the definition, written as a loop."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 6))
    y = rng.normal(size=(25, 6))
    x[rng.random(x.shape) < 0.4] = np.nan
    y[rng.random(y.shape) < 0.4] = np.nan
    d = fo.rw_nan_euclidean_distances(x, y)
    for i in range(len(x)):
        for j in range(len(y)):
            both = ~np.isnan(x[i]) & ~np.isnan(y[j])
            expected = np.sqrt(np.sum((x[i, both] - y[j, both]) ** 2)) if both.any() else np.nan
            assert d[i, j] == pytest.approx(expected, nan_ok=True)


def test_invalid_input_is_rejected():
    """Rows of different lengths are an error, not a silent broadcast."""
    with pytest.raises(ValueError):
        fo.rw_nan_euclidean_distances(np.ones((2, 3)), np.ones((2, 4)))


def test_the_statistic_equals_estat_without_drawing_random_numbers():
    """The cost needs only the statistic; estat's one bootstrap draw was discarded."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 4))
    y = rng.normal(size=(30, 4))
    x[rng.random(x.shape) < 0.3] = np.nan
    np.random.seed(11)
    _, expected, _ = fo.estat(x, y, nboot=1)
    state = np.random.get_state()[1].copy()
    assert fo.energy_statistic(x, y) == expected
    assert np.array_equal(np.random.get_state()[1], state)
