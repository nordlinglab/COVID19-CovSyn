# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The pieces of the CovSyn-paper figures that were wrong in the notebooks (finding E90)."""
from __future__ import annotations

import numpy as np

from covsyn.figures import reproduce_wu2025 as wu


def _case(day: float, incubation: float) -> dict:
    return {'infection_day': day, 'incubation_period': incubation}


def test_serial_interval_is_the_onset_difference_not_the_generation_time() -> None:
    courses = [_case(0, 5), _case(3, 2), _case(4, np.nan)]
    digraph = np.array([['nan', '1', '0', 'nan'], ['1', '2', '3', 'household'],
                        ['2', '3', '4', 'school']], dtype='<U32')
    generation, serial = wu.transmission_intervals(courses, digraph)
    assert generation == [3, 1]               # every edge, the last one included
    assert serial == [0]                      # onset 5 -> 5; the asymptomatic case has none


def test_time_shift_recovers_a_known_delay() -> None:
    simulated = np.cumsum(np.r_[np.zeros(20), np.ones(20), np.zeros(40)])
    observed = simulated[12:]                 # the same curve, 12 days earlier on its own axis
    observed = np.pad(observed, (0, len(simulated) - len(observed)), mode='edge')
    assert wu.best_time_shift(observed, simulated) == 12


def test_still_in_state_counts_cases_whose_duration_reaches_the_day() -> None:
    np.testing.assert_allclose(wu.still_in_state(np.array([0., 2., 2., 5.]), 4),
                               [1.0, 0.75, 0.75, 0.25])
