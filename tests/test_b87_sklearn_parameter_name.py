# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B87: the NaN-tolerant distance must run on the scikit-learn the project declares.

rw_nan_euclidean_distances is CovSyn's copy of sklearn's nan_euclidean_distances, carrying a
row-weighted variant, and it called check_pairwise_arrays with force_all_finite. That name was
deprecated in scikit-learn 1.6 and removed in favour of ensure_all_finite, so on the version
pyproject.toml declares, scikit-learn>=1.9.1, every call raised TypeError. The energy statistic
of the objective goes through it, so the whole cost function was unrunnable on the declared
environment; it kept working only on the compute server's 1.6.1, which still accepted the old
name. The tests below exercise the call rather than the spelling, so either name would pass as
long as the result is right on the installed version.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.calibration.firefly_optimizer import rw_nan_euclidean_distances


def test_a_pair_with_a_missing_coordinate_uses_the_present_ones_unscaled():
    """The row-weighted variant drops sklearn's rescaling, so this is not sklearn's answer.

    The two lines that would apply sklearn's weight, "total coordinates / present
    coordinates", are commented out in the body; B90 measured what restoring them would do and
    left them, correcting the docstring instead. Here only coordinate 0 is present in both
    rows, so the distance is |0 - 1| = 1, where sklearn's rescaled version would give sqrt(2).
    """
    nan = float("nan")
    x = [[0, 1], [1, nan]]
    out = rw_nan_euclidean_distances(x, x)
    assert out == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_a_matrix_without_any_missing_value_is_the_ordinary_distance():
    """With nothing missing the two variants agree, so this pins the plain case."""
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = rw_nan_euclidean_distances(x, x)
    assert out == pytest.approx(np.array([[0.0, 5.0], [5.0, 0.0]]))


def test_distance_to_the_origin():
    """[0, 1] has both coordinates, [1, nan] only the first; neither is rescaled."""
    nan = float("nan")
    x = [[0, 1], [1, nan]]
    out = rw_nan_euclidean_distances(x, [[0, 0]])
    assert out == pytest.approx(np.array([[1.0], [1.0]]))


def test_a_missing_coordinate_is_skipped_and_not_compensated_for():
    """The second coordinate is missing on the left, so its 100 never enters the distance."""
    nan = float("nan")
    out = rw_nan_euclidean_distances([[1.0, nan]], [[4.0, 100.0]])
    assert out == pytest.approx(np.array([[3.0]]))


def test_the_energy_statistic_runs_end_to_end():
    """The objective reaches this function through estat (fast_cost._cost_function)."""
    from covsyn.calibration.firefly_optimizer import estat

    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 3))
    b = rng.normal(loc=0.5, size=(35, 3))
    out = estat(a, b, nboot=1)
    assert np.isfinite(out[1])


def test_it_runs_on_the_declared_scikit_learn():
    """The version pyproject.toml requires is the one that removed the old parameter name."""
    import sklearn
    from packaging.version import Version

    if Version(sklearn.__version__) < Version("1.9.1"):
        pytest.skip(f"installed scikit-learn {sklearn.__version__} predates the declared 1.9.1")
    nan = float("nan")
    assert np.isfinite(rw_nan_euclidean_distances([[0, 1], [1, nan]], [[0, 0]])).all()
