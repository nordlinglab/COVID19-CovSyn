# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 7: the bounds and the literature vector come from one place.

The optimiser and the initial population (Task 7) need the same bounds and the same
literature seed vector: the contact block at the middle of its bounds, which hold no central
estimate, and the course-of-disease block at its documented values.
"""

from __future__ import annotations

import pickle

import numpy as np

from covsyn.calibration import parameter_space
from tests.conftest import REPO_ROOT

VARIABLE = REPO_ROOT / "variable"
CONTACT_KEYS = ("household", "school", "workplace", "health_care", "municipality",
                "overdispersion")  # fmt: skip


def test_the_bounds_are_the_contact_bounds_then_the_course_bounds():
    """In the order run_covid reads P: the six contact blocks, then the course of disease."""
    with open(VARIABLE / "contact_parameters.pkl", "rb") as f:
        contact = pickle.load(f)
    lower = np.hstack([np.array(sum((contact[f"{k}_lower_bound"] for k in CONTACT_KEYS), [])),
                       np.load(VARIABLE / "course_parameters_lb.npy")])  # fmt: skip
    upper = np.hstack([np.array(sum((contact[f"{k}_upper_bound"] for k in CONTACT_KEYS), [])),
                       np.load(VARIABLE / "course_parameters_ub.npy")])  # fmt: skip
    space = parameter_space.load()
    np.testing.assert_array_equal(space.lower, lower)
    np.testing.assert_array_equal(space.upper, upper)
    assert space.lower.shape == (208,) and np.all(space.lower <= space.upper)


def test_the_literature_vector_is_the_middle_then_the_documented_values():
    """The contact block at the middle of its bounds, the course of disease as documented."""
    space = parameter_space.load()
    contact = len(space.lower) - len(np.load(VARIABLE / "course_parameters.npy"))
    np.testing.assert_array_equal(space.literature[:contact],
                                  (space.lower[:contact] + space.upper[:contact]) / 2)  # fmt: skip
    np.testing.assert_array_equal(space.literature[contact:],
                                  np.load(VARIABLE / "course_parameters.npy"))  # fmt: skip
