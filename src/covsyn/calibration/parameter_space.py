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

from covsyn.model.data_synthesize import (
    COMMUNITY_EVENT_FIRST_INDEX,
    COMMUNITY_POWER_LAW_INDEX,
    WORK_GROUP_SATURATION_INDEX,
)

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


def clip_warm_start(vector: np.ndarray, lb: np.ndarray, ub: np.ndarray, label: str) -> np.ndarray:
    """Clip a starting vector to the search bounds and say which values moved.

    Rebuilt bounds can exclude an earlier best vector (B75: run 18's attack rates fall outside the
    bounds rebuilt for the work calendar in 17 places), and the firefly then starts somewhere
    other than the vector and cost the run log names.
    """
    clipped = np.clip(vector, lb, ub)
    moved = np.flatnonzero(clipped != vector)
    if len(moved):
        print(
            f"warm start: {label}: {len(moved)} of {len(vector)} values clipped to the bounds "
            f"({', '.join(f'P[{i}]' for i in moved)})",
            flush=True,
        )
    return clipped


def pad_pre_b54_vector(vector: np.ndarray, seed_vector: np.ndarray) -> np.ndarray:
    """Extend a pre-B54 warm-start vector with the seed values of the appended parameters.

    Args:
        vector: A best vector read from an earlier run.
        seed_vector: The literature seed vector of this run.

    Returns:
        The vector with the seed values of the missing tail appended when it has exactly the
        199 values of run 10 and earlier, the 204 values of runs 11-16 (before B62's
        P[204]) or the 205 values of run 17 (before B64's P[205]); otherwise the vector
        unchanged, so a misaligned vector of any other length still fails the caller's shape
        check instead of being padded into a plausible-looking start.
    """
    # B62: a vector of run 11-16 ends before P[204] and is padded the same way; B64: so does
    # a run-17 vector, which ends before P[205].
    if (
        vector.size
        in (COMMUNITY_EVENT_FIRST_INDEX, COMMUNITY_POWER_LAW_INDEX, WORK_GROUP_SATURATION_INDEX)
        and vector.size < seed_vector.size
    ):
        return np.concatenate([vector, seed_vector[vector.size :]])
    return vector
