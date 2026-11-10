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

Task 1 of the rework plan carried the pins of requirements.txt into pyproject.toml, so the
declared version is now scikit-learn==1.6.1 and uv.lock installs exactly that one.
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
    """The distance runs on the scikit-learn version pyproject.toml pins, which is installed."""
    import tomllib
    from pathlib import Path

    import sklearn

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pins = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    declared = [pin.split("==")[1] for pin in pins if pin.startswith("scikit-learn==")]
    assert declared, f"pyproject.toml pins no scikit-learn==X.Y.Z: {pins}"
    assert sklearn.__version__ == declared[0]
    nan = float("nan")
    assert np.isfinite(rw_nan_euclidean_distances([[0, 1], [1, nan]], [[0, 0]])).all()


def test_the_call_uses_the_parameter_name_newer_releases_keep():
    """The distance passes ensure_all_finite, the name newer scikit-learn releases keep.

    scikit-learn 1.9 removed force_all_finite, which the pinned 1.6.1 still accepts, so a run on
    the pinned version cannot catch the old name coming back; the source is checked instead.
    """
    import inspect

    from covsyn.calibration import firefly_optimizer

    source = inspect.getsource(firefly_optimizer.rw_nan_euclidean_distances)
    code = "\n".join(line.split("#")[0] for line in source.splitlines())   # not the B87 note
    assert "force_all_finite" not in code
    assert "ensure_all_finite=" in code
