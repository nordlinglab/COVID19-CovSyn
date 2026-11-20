# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 8, item 2: an elite of the k lowest-cost fireflies stays unmoved and is re-scored.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 8: keep an elite of the k = 5
lowest-cost fireflies of each generation unmoved, and re-score it on every block. A population
of n <= 5 keeps k = min(5, n - 1), so at least one firefly moves (decision of 2026-11-20).
Generation 0 shares block 0 with the initial population, so its elite keeps its block-0 costs.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from tests import firefly_standins as standins

DIM = 208
POP = 8  # k = 5
GENERATIONS = 3


@pytest.mark.parametrize("n, k", [(1, 0), (2, 1), (3, 2), (5, 4), (6, 5), (100, 5)])
def test_the_elite_is_five_or_all_but_one(n, k):
    """The elite size is k = min(5, n - 1)."""
    assert fo.elite_size(n) == k


def test_the_elite_is_the_k_lowest_costs_ties_by_index():
    """The lowest costs, a tie going to the lower index, so every run picks the same ones."""
    costs = np.array([3.0, 1.0, 2.0, 1.0, 5.0])
    assert fo.elite_members(costs, 3) == [1, 3, 2]
    assert fo.elite_members(costs, 0) == []


def test_a_fixed_firefly_stays_though_a_brighter_one_attracts_it():
    """Firefly 2 (cost 2) would move towards firefly 1 (cost 1); fixed, it stays."""
    start = np.array([[0.1, 0.9], [0.4, 0.4], [0.9, 0.2], [0.6, 0.7]])
    costs = np.array([3.0, 1.0, 2.0, 4.0])
    got = fo.synchronous_moves(start, costs, 0.5, 1.0, 1.0, np.random.default_rng(1),
                               fixed=[1, 2])  # fmt: skip
    np.testing.assert_array_equal(got[[1, 2]], start[[1, 2]])
    assert not np.array_equal(got[0], start[0]) and not np.array_equal(got[3], start[3])


def test_fixing_nothing_moves_as_task_6_did():
    """The fixed set only skips moves; the others draw the same steps in the same order."""
    rng = np.random.default_rng(5)
    start, costs = rng.random((6, 4)), rng.random(6)
    plain = fo.synchronous_moves(start, costs, 0.3, 1.0, 1.0, np.random.default_rng(2))
    fixed = fo.synchronous_moves(start, costs, 0.3, 1.0, 1.0, np.random.default_rng(2),
                                 fixed=[])  # fmt: skip
    np.testing.assert_array_equal(plain, fixed)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A run of POP fireflies on the recording stand-in, and its directory; read only."""
    path = tmp_path_factory.mktemp("elite")
    previous = getattr(fo, "result_path", None)
    fo.result_path = path
    try:
        engine = fo.Firefly(pop_size=POP, alpha_0=0.2, betamin=1, gamma=0.03, seed=4)
        engine.firefly(function=standins.recording, dim=DIM, lb=np.zeros(DIM),
                       ub=np.ones(DIM), demographic_parameters=None,
                       max_generations=GENERATIONS, max_workers=2, Cheng_contact_array=None,
                       Cheng_attack_rate=None, norm_weights=None)  # fmt: skip
    finally:
        fo.result_path = previous
    return path


def _checkpoint(path, generation):
    name = "initial" if generation < 0 else f"generation_{generation:04d}"
    with np.load(path / fo.CHECKPOINTS / f"{name}.npz") as saved:
        return {key: saved[key] for key in saved.files}


def test_the_whole_elite_keeps_its_place_every_generation(run):
    """The k lowest at the start of a generation are where they were, and the rest moved."""
    for generation in range(GENERATIONS):
        before, after = _checkpoint(run, generation - 1), _checkpoint(run, generation)
        elite = fo.elite_members(before["intensities"], fo.elite_size(POP))
        assert len(elite) == 5
        for k in range(POP):
            same = np.array_equal(after["normalized"][k], before["normalized"][k])
            assert same == (k in elite), (generation, k)


def test_generation_0_scores_only_the_fireflies_outside_the_elite(run):
    """The elite keeps its block-0 costs; every firefly is scored from generation 1 on."""
    initial, first = _checkpoint(run, -1), _checkpoint(run, 0)
    assert int(first["evaluations"]) == POP + POP - 5
    elite = fo.elite_members(initial["intensities"], 5)
    np.testing.assert_array_equal(first["intensities"][elite], initial["intensities"][elite])
    for generation in range(1, GENERATIONS):
        evaluations = int(_checkpoint(run, generation)["evaluations"])
        assert evaluations == int(_checkpoint(run, generation - 1)["evaluations"]) + POP
