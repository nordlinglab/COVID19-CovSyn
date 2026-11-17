# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The optimiser's bounds and literature seed vector, read in one place (Task 7).

The optimiser checks a run's bounds against those its initial population was made with, and
the population generator writes them, so both read them here, from ./variable/.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The contact blocks in the order run_covid reads P, before the course of disease.
CONTACT_BLOCKS = ("household", "school", "workplace", "health_care", "municipality",
                  "overdispersion")  # fmt: skip


@dataclass(frozen=True)
class ParameterSpace:
    """Lower and upper bounds, and the literature vector, for every parameter."""

    lower: np.ndarray
    upper: np.ndarray
    literature: np.ndarray


def load(variable: Path = Path("./variable")) -> ParameterSpace:
    """The bounds and the literature vector from the files in ``variable``.

    The bounds are the six contact blocks of contact_parameters.pkl, then the course of
    disease. The literature vector puts the contact block, which has no stored central
    estimate, at the middle of its bounds, and the course of disease at its documented values.
    """
    with open(variable / "contact_parameters.pkl", "rb") as f:
        contact = pickle.load(f)
    contact_lower = np.array(sum((contact[f"{k}_lower_bound"] for k in CONTACT_BLOCKS), []))
    contact_upper = np.array(sum((contact[f"{k}_upper_bound"] for k in CONTACT_BLOCKS), []))
    course = np.load(variable / "course_parameters.npy")
    lower = np.hstack((contact_lower, np.load(variable / "course_parameters_lb.npy")))
    upper = np.hstack((contact_upper, np.load(variable / "course_parameters_ub.npy")))
    n = len(contact_lower)
    literature = np.hstack(((lower[:n] + upper[:n]) / 2, course))
    return ParameterSpace(lower, upper, literature)
