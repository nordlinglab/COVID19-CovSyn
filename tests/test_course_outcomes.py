# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Outcome dates of a course of disease (decision B58, findings E92 and E94).

ICU admission is no longer capped at the end of infectiousness (E92). Closure can never come
before the positive test, and both ICU outcomes are drawn from distributions truncated at the
earliest possible day rather than clamped onto it, so deaths do not pile up the day after
admission (E94).
"""

from __future__ import annotations

import random

import numpy as np

from covsyn.model import data_synthesize as ds


def _courses(P: np.ndarray, transition: list[float], n: int = 2000) -> list:
    """Draw n courses of disease from vector P with the given transition probabilities."""
    gammas = [
        {"latent_period_shape": P[37], "latent_period_scale": P[38]},
        {"infectious_period_shape": P[39], "infectious_period_scale": P[40]},
        {"incubation_period_shape": P[41], "incubation_period_scale": P[42]},
        {
            "symptom_to_confirmed_shape": P[43],
            "symptom_to_confirmed_scale": P[44],
            "symptom_to_confirmed_loc": P[45],
        },
        {
            "asymptomatic_to_recovered_shape": P[46],
            "asymptomatic_to_recovered_scale": P[47],
            "asymptomatic_to_recovered_loc": P[48],
        },
        {
            "symptomatic_to_critically_ill_shape": P[49],
            "symptomatic_to_critically_ill_scale": P[50],
            "symptomatic_to_critically_ill_loc": P[51],
        },
        {
            "symptomatic_to_recovered_shape": P[52],
            "symptomatic_to_recovered_scale": P[53],
            "symptomatic_to_recovered_loc": P[54],
        },
        {
            "critically_ill_to_recovered_shape": P[55],
            "critically_ill_to_recovered_scale": P[56],
            "critically_ill_to_recovered_loc": P[57],
        },
        {"infection_to_death_shape": P[58], "infection_to_death_scale": P[59]},
        {
            "negative_to_confirmed_shape": P[60],
            "negative_to_confirmed_scale": P[61],
            "negative_to_confirmed_loc": P[62],
        },
    ]
    np.random.seed(3)
    random.seed(3)
    out = []
    for _ in range(n):
        course = ds.Draw_course_of_disease_data(0, *gammas, P[67], transition)
        course.draw_course_of_disease()
        out.append(course)
    return out


def test_no_case_is_closed_before_its_positive_test(run10_vector: np.ndarray) -> None:
    """Every recovered case is released on or after its positive test (E94)."""
    P = run10_vector
    courses = _courses(P, [P[195], P[196], P[197]])
    closed = [c for c in courses if np.isfinite(c.date_of_recovery)]
    assert closed
    assert all(c.date_of_recovery >= float(np.ravel(c.positive_test_date)[0]) for c in closed)


def test_icu_can_follow_the_end_of_infectiousness(run10_vector: np.ndarray) -> None:
    """With every symptomatic case critical, some are admitted after infectiousness (E92)."""
    P = run10_vector
    courses = _courses(P, [P[195], 0.0, 1.0])
    icu = [c for c in courses if np.isfinite(c.date_of_critically_ill)]
    assert icu
    late = [c for c in icu if c.date_of_critically_ill > c.latent_period + c.infectious_period + 1]
    assert late


def test_deaths_do_not_pile_up_the_day_after_icu(run10_vector: np.ndarray) -> None:
    """Deaths follow ICU and are spread out instead of clamped onto ICU + 1 (E94)."""
    P = run10_vector
    courses = _courses(P, [P[195], 0.0, 0.0])
    dead = [c for c in courses if np.isfinite(c.date_of_death)]
    assert len(dead) > 500
    gaps = np.array([c.date_of_death - c.date_of_critically_ill for c in dead])
    assert np.all(gaps >= 1)
    assert np.mean(gaps == 1) < 0.15
