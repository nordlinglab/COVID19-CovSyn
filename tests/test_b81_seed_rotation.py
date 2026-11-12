# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B81: every generation scores the population on its own 300 seeds.

Run 20 selected a vector whose workplace SAR was 0.024 on the fixed training seeds and 0.070
over ten unseen blocks -- below the lowest of all ten -- while an unoptimised vector sat 9th of
eleven. The search was fitting the fixed sample, so each generation now draws a new block and
the comparisons that steer it use only that generation's scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.revalidation import load_candidates

DIM = 208
POP = 4
GENERATIONS = 6


class _Recorder:
    """Stands in for the objective: a deterministic cost, and every call recorded."""

    def __init__(self):
        # each vector by its bytes, so two different vectors can never be taken for one
        self.calls: list[tuple[bytes, int]] = []

    def __call__(self, P, demo, executor, contacts, attack, weights, seed_offset=0):
        self.calls.append((np.asarray(P, dtype=float).tobytes(), int(seed_offset)))
        # the offset shifts the cost, so a stale score from another generation is visible
        return float(np.sum(P)) % 1.0 + seed_offset / 1e9


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run the optimizer on the stub objective and return (recorder, results)."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    recorder = _Recorder()
    lb, ub = np.zeros(DIM) + 0.1, np.ones(DIM)
    engine = fo.Firefly(pop_size=POP, alpha=1, betamin=1, gamma=0.03, seed=7)
    out = engine.firefly(function=recorder, dim=DIM, lb=lb, ub=ub,
                         demographic_parameters=None, max_generations=GENERATIONS,
                         max_workers=2, Cheng_contact_array=None, Cheng_attack_rate=None,
                         norm_weights=None)
    return recorder, out, tmp_path


def test_offset_of_each_generation_is_its_own_block():
    """Generation g scores on its own 300 seeds; generation 0 shares 0..299 with the initial
    population (F7)."""
    assert fo.generation_seed_offset(0) == 0
    assert fo.generation_seed_offset(5) == 5 * fo.SIMULATIONS_PER_EVALUATION
    offsets = {fo.generation_seed_offset(g) for g in range(120)}
    assert len(offsets) == 120


def test_training_seeds_never_reach_the_validation_blocks():
    """B60 re-scores candidates from 100000; a training block there would leak (E87)."""
    from covsyn.calibration.revalidation import VALIDATION_SEED_START

    last = fo.generation_seed_offset(119) + fo.SIMULATIONS_PER_EVALUATION
    assert last <= VALIDATION_SEED_START


def test_every_evaluation_of_a_generation_uses_that_generation_s_offset(run):
    recorder, _, _ = run
    initial = recorder.calls[:POP]
    assert {offset for _, offset in initial} == {0}
    seen = [offset for _, offset in recorder.calls[POP:]]
    expected = {fo.generation_seed_offset(g) for g in range(GENERATIONS)}
    assert set(seen) == expected
    # the offsets arrive in generation order and never come back
    assert seen == sorted(seen)


def test_generation_0_starts_from_the_initial_scores_without_rescoring(run):
    """The initial population is scored on block 0, generation 0's block, once only (F7).

    Generation 0 evaluates only moved fireflies, on block 0: the first call after the initial
    population is a move on block 0 (it was a re-score on block 300), and no initial vector is
    evaluated on block 0 a second time.
    """
    from collections import Counter

    recorder, _, _ = run
    initial = [P for P, _ in recorder.calls[:POP]]
    assert recorder.calls[POP][1] == 0
    on_block_0 = Counter(P for P, o in recorder.calls if o == 0)
    assert all(on_block_0[P] == initial.count(P) for P in initial)


def test_the_whole_population_is_rescored_when_the_block_changes(run):
    """Including the brightest: a score from another block cannot be compared with this one.

    From generation 1 on; generation 0 shares the initial population's block (F7).
    """
    recorder, _, _ = run
    for g in range(1, GENERATIONS):
        offset = fo.generation_seed_offset(g)
        rescored = [P for P, o in recorder.calls if o == offset]
        assert len(rescored) >= POP


def test_the_returned_best_rows_are_the_final_population(run):
    """firefly_best.txt keeps one row per firefly: reproduce_wu2025 requires equal row counts."""
    _, out, _ = run
    best_fireflies, best_intensities = out[0], out[1]
    fireflies, intensities = out[6], out[7]
    assert len(best_fireflies) == len(best_intensities) == POP
    assert np.array_equal(best_fireflies, fireflies)
    assert np.array_equal(best_intensities, intensities)


def test_the_archive_holds_one_snapshot_per_sampled_generation(run):
    """The candidate pool spreads over the search, not just its last generations."""
    _, _, path = run
    archive = np.atleast_2d(np.loadtxt(path / "firefly_archive.txt"))
    generations = [g for g in range(GENERATIONS) if g % fo.ARCHIVE_EVERY == 0]
    assert len(archive) == len(generations)
    assert archive[:, 0].tolist() == [float(g) for g in generations]
    assert archive.shape[1] == DIM + 2


def test_progress_metrics_record_this_generation_not_a_running_best(run):
    """A cost from another block is not comparable, so no column may accumulate a minimum."""
    import csv

    _, _, path = run
    with open(path / "progress_metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == GENERATIONS
    assert [int(r["generation"]) for r in rows] == list(range(GENERATIONS))
    assert [int(r["seed_offset"]) for r in rows] == [
        fo.generation_seed_offset(g) for g in range(GENERATIONS)
    ]


def test_candidates_include_the_archive(tmp_path):
    """revalidation re-scores the archived snapshots together with the final population."""
    rng = np.random.default_rng(0)
    population = np.column_stack([rng.random((3, DIM)), np.array([1.0, 2.0, 3.0])])
    bests = np.column_stack([np.zeros(3), population[:, :-1], population[:, -1]])
    archive = np.column_stack([np.array([0.0, 5.0]), rng.random((2, DIM)), np.array([4.0, 5.0])])
    np.savetxt(tmp_path / "firefly_result.txt", population)
    np.savetxt(tmp_path / "firefly_best.txt", bests)
    np.savetxt(tmp_path / "firefly_archive.txt", archive)
    vectors, costs = load_candidates(tmp_path)
    assert len(vectors) == 5
    for row in archive:
        assert any(np.allclose(row[1:-1], v) for v in vectors)


def test_candidates_work_without_an_archive(tmp_path):
    """An older run has no archive file; it must still load."""
    rng = np.random.default_rng(1)
    population = np.column_stack([rng.random((2, DIM)), np.array([1.0, 2.0])])
    bests = np.column_stack([np.zeros(2), population[:, :-1], population[:, -1]])
    np.savetxt(tmp_path / "firefly_result.txt", population)
    np.savetxt(tmp_path / "firefly_best.txt", bests)
    vectors, _ = load_candidates(tmp_path)
    assert len(vectors) == 2


def test_the_run_directory_holds_the_configuration(run):
    """Task 3: config.json records the seed, workers, settings and initial population."""
    import hashlib
    import json

    _, _, path = run
    config = json.loads((path / "config.json").read_text())
    assert (config["seed"], config["workers"]) == (7, 2)
    assert config["settings"]["pop_size"] == POP
    assert config["settings"]["max_generations"] == GENERATIONS
    digest = config["initial_population_sha256"]
    assert len(digest) == 64 and int(digest, 16) >= 0
    # The same seed draws the same initial population, so its hash repeats.
    again = fo.Firefly(pop_size=POP, alpha=1, betamin=1, gamma=0.03, seed=7)
    lb, ub = np.zeros(DIM) + 0.1, np.ones(DIM)
    first = lb + again.rng.uniform(0, 1, (POP, DIM)) * (ub - lb)
    assert digest == hashlib.sha256(np.ascontiguousarray(first, dtype=float).tobytes()).hexdigest()


def test_an_unseeded_run_records_a_seed_that_redraws_it():
    """Task 3: without a seed the optimiser draws one, so config.json is never seed null."""
    first = fo.Firefly(pop_size=POP, alpha=1, betamin=1, gamma=0.03)
    again = fo.Firefly(pop_size=POP, alpha=1, betamin=1, gamma=0.03, seed=first.seed)
    assert isinstance(first.seed, int)
    assert np.array_equal(first.rng.uniform(size=5), again.rng.uniform(size=5))
