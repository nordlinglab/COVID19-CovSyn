# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 5, item 2: run_covid's time per stage, behind a flag that leaves the output unchanged."""

from __future__ import annotations

import copy
import pickle
import time

import pytest

from covsyn.model.data_synthesis_main import STAGES, run_covid

SEEDS = range(6)


def test_stages_are_the_ones_the_plan_names():
    """Demographic, social, course of disease, contact draw, and queue handling."""
    assert STAGES == ("demographic", "social", "course_of_disease", "contact_draw", "queue")


def test_without_the_flag_nothing_is_timed(run10_vector, demographic_parameters):
    """The default records nothing and needs no argument."""
    run_covid(0, run10_vector.copy(), copy.deepcopy(demographic_parameters))


@pytest.mark.parametrize("seed", SEEDS)
def test_timing_leaves_the_simulation_unchanged(seed, run10_vector, demographic_parameters):
    """The same seed gives the same cases, bit for bit, with and without the flag."""
    plain = run_covid(seed, run10_vector.copy(), copy.deepcopy(demographic_parameters))
    times: dict[str, int] = {}
    timed = run_covid(
        seed, run10_vector.copy(), copy.deepcopy(demographic_parameters), stage_times=times
    )
    assert pickle.dumps(timed) == pickle.dumps(plain)


def test_every_stage_is_timed_within_the_run(run10_vector, demographic_parameters):
    """Each stage gets a non-negative count of nanoseconds, together within the run's time."""
    times: dict[str, int] = {}
    start = time.perf_counter_ns()
    for seed in SEEDS:
        run_covid(
            seed, run10_vector.copy(), copy.deepcopy(demographic_parameters), stage_times=times
        )
    elapsed = time.perf_counter_ns() - start
    assert set(times) == set(STAGES)
    assert all(isinstance(value, int) and value >= 0 for value in times.values())
    assert all(times[stage] > 0 for stage in STAGES)
    assert sum(times.values()) <= elapsed


def test_times_add_up_across_runs(run10_vector, demographic_parameters):
    """A second run adds to the same dictionary instead of replacing it."""
    times: dict[str, int] = {}
    run_covid(0, run10_vector.copy(), copy.deepcopy(demographic_parameters), stage_times=times)
    first = dict(times)
    run_covid(1, run10_vector.copy(), copy.deepcopy(demographic_parameters), stage_times=times)
    assert all(times[stage] >= first[stage] for stage in STAGES)
    assert sum(times.values()) > sum(first.values())
