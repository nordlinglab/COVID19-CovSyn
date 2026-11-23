# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which cost terms respond to each parameter, one at a time, at a few base vectors.

The screen behind the decision whether the energy term, or the Cheng terms, can leave the
objective: a parameter that only those terms respond to would be left unconstrained. Each free
parameter moves 0.2 of its normalised range towards the interior; effects are paired
differences over replications, reported with t intervals against a substantive threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_term_sensitivity as pts  # noqa: E402


def test_a_step_goes_up_from_the_lower_half():
    """At 0.3 the parameter moves to 0.5."""
    assert pts.perturbed(np.array([0.3, 0.9]), 0, 0.2).tolist() == pytest.approx([0.5, 0.9])


def test_a_step_goes_down_from_the_upper_half():
    """At 0.9 the parameter moves to 0.7."""
    assert pts.perturbed(np.array([0.3, 0.9]), 1, 0.2).tolist() == pytest.approx([0.3, 0.7])


def test_the_base_vector_is_left_unchanged():
    """The step returns a new vector."""
    x = np.array([0.3, 0.9])
    pts.perturbed(x, 0, 0.2)
    assert x.tolist() == [0.3, 0.9]


@given(x=st.floats(0, 1), step=st.floats(0.01, 0.5))
def test_a_step_stays_inside_the_range_and_has_the_full_length(x, step):
    """Towards the interior, a step of at most 0.5 never leaves [0, 1]."""
    out = pts.perturbed(np.array([x]), 0, step)[0]
    assert 0 <= out <= 1
    assert abs(out - x) == pytest.approx(step)


def test_the_bases_are_the_lowest_the_farthest_low_and_a_middle_row():
    """Lowest cost first, then the low-cost row farthest from it, then the quantile row."""
    x = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.2]])
    costs = np.array([1.0, 2.0, 3.0, 50.0, 100.0])
    low, far, middle = pts.pilot_bases(x, costs, exclude=(), low_share=0.6, middle_quantile=0.75)
    assert (low, far) == (0, 2)
    assert middle == 3


def test_excluded_rows_are_never_chosen():
    """Rows already used as bases (run 19, the literature vector) are skipped."""
    x = np.eye(4)
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    chosen = pts.pilot_bases(x, costs, exclude=(0,), low_share=0.75, middle_quantile=0.5)
    assert 0 not in chosen


def test_an_effect_beyond_the_threshold_is_classed_as_such():
    """The whole interval above delta."""
    assert pts.classify([1.0, 1.1, 0.9, 1.0, 1.05], delta=0.2) == "beyond"


def test_an_effect_inside_the_threshold_is_classed_as_such():
    """The whole interval within +-delta."""
    assert pts.classify([0.01, -0.02, 0.0, 0.015, -0.01], delta=0.2) == "within"


def test_a_noisy_effect_is_imprecise():
    """The interval straddles delta."""
    assert pts.classify([0.5, -0.4, 0.3, -0.2, 0.1], delta=0.2) == "imprecise"


def test_a_negative_effect_beyond_the_threshold_counts():
    """The sign does not matter for beyond."""
    assert pts.classify([-1.0, -1.1, -0.9, -1.0, -1.05], delta=0.2) == "beyond"


def test_identical_differences_are_exact():
    """No spread: the interval is the point."""
    assert pts.classify([0.0] * 5, delta=0.1) == "within"
    assert pts.classify([0.3] * 5, delta=0.1) == "beyond"
