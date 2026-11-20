# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B81: every generation scores the population on its own 300 seeds.

Run 20 selected a vector whose workplace SAR was 0.024 on the fixed training seeds and 0.070
over ten unseen blocks -- below the lowest of all ten -- while an unoptimised vector sat 9th of
eleven. The search was fitting the fixed sample, so each generation now draws a new block and
the comparisons that steer it use only that generation's scores.
"""

from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.revalidation import load_candidates
from covsyn.runtime import runs, timings
from tests import firefly_standins as standins

DIM = 208
POP = 4
GENERATIONS = 6


def scored(path):
    """Per state (-1 the initial population, then each generation), each firefly's block.

    Task 6: the stand-in runs in the optimiser's workers and returns the block it scored on
    among the cost parts, which every checkpoint keeps by firefly.
    """
    blocks = {}
    for generation in range(-1, GENERATIONS):
        name = "initial" if generation < 0 else f"generation_{generation:04d}"
        with np.load(path / fo.CHECKPOINTS / f"{name}.npz") as saved:
            parts = json.loads(str(saved["cost_parts"]))
        blocks[generation] = [int(p["scored_offset"]) for p in parts]
    return blocks


def evaluations_per_generation(path):
    """The evaluation table's rows counted by generation."""
    with open(runs.progress_path(path).parent / timings.EVALUATION_TABLE, newline="") as f:
        return Counter(int(r["generation"]) for r in csv.DictReader(f))


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run the optimizer on the recording stand-in and return (None, results, directory)."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    lb, ub = np.zeros(DIM) + 0.1, np.ones(DIM)
    engine = fo.Firefly(pop_size=POP, alpha_0=1, betamin=1, gamma=0.03, seed=7)
    out = engine.firefly(function=standins.recording, dim=DIM, lb=lb, ub=ub,
                         demographic_parameters=None, max_generations=GENERATIONS,
                         max_workers=2, Cheng_contact_array=None, Cheng_attack_rate=None,
                         norm_weights=None)
    return None, out, tmp_path


def test_offset_of_each_generation_is_its_own_block():
    """Generation g scores on its own 300 seeds; generation 0 shares block 0 with the initial
    population (F7). Task 7: blocks of the run's stream, from 10^7 on stream 1."""
    first = fo.generation_seed_offset(0)
    assert first == 10**7
    assert fo.generation_seed_offset(5) == first + 5 * fo.SIMULATIONS_PER_EVALUATION
    offsets = {fo.generation_seed_offset(g) for g in range(120)}
    assert len(offsets) == 120


def test_training_seeds_never_reach_the_validation_blocks():
    """B60 re-scores candidates on unseen blocks; a training block there would leak (E87).

    Task 7: the optimisation and validation streams are disjoint by construction.
    """
    from covsyn.calibration.revalidation import validation_seed_blocks

    training = {s for g in range(120) for s in range(fo.generation_seed_offset(g),
                fo.generation_seed_offset(g) + fo.SIMULATIONS_PER_EVALUATION)}  # fmt: skip
    validation = validation_seed_blocks(10, fo.SIMULATIONS_PER_EVALUATION)
    assert not any(seed in training for block in validation for seed in block)


def test_every_evaluation_of_a_generation_uses_that_generation_s_offset(run):
    _, _, path = run
    for generation, blocks in scored(path).items():
        assert set(blocks) == {fo.generation_seed_offset(max(generation, 0))}, generation


def test_generation_0_starts_from_the_initial_scores_without_rescoring(run):
    """The initial population is scored on block 0, generation 0's block, once only (F7).

    Task 6: generation 0 scores every firefly but the elite, which keeps its block-0 costs;
    Task 8: an elite of k.
    """
    _, _, path = run
    k = fo.elite_size(POP)
    assert evaluations_per_generation(path)[0] == POP - k
    with np.load(path / fo.CHECKPOINTS / "initial.npz") as initial, \
            np.load(path / fo.CHECKPOINTS / "generation_0000.npz") as first:  # fmt: skip
        elite = fo.elite_members(initial["intensities"], k)
        np.testing.assert_array_equal(first["intensities"][elite],
                                      initial["intensities"][elite])


def test_the_whole_population_is_rescored_when_the_block_changes(run):
    """Including the brightest: a score from another block cannot be compared with this one.

    From generation 1 on; generation 0 shares the initial population's block (F7).
    """
    _, _, path = run
    counts = evaluations_per_generation(path)
    assert all(counts[g] == POP for g in range(1, GENERATIONS))


def test_the_returned_best_rows_are_the_final_population(run):
    """firefly_best.txt keeps one row per firefly: reproduce_wu2025 requires equal row counts."""
    _, out, _ = run
    best_fireflies, best_intensities = out[0], out[1]
    fireflies, intensities = out[6], out[7]
    assert len(best_fireflies) == len(best_intensities) == POP
    assert np.array_equal(best_fireflies, fireflies)
    assert np.array_equal(best_intensities, intensities)


def test_the_archive_spreads_over_the_search(run):
    """The candidate pool spreads over the search, not just its last generations.

    Task 8 replaced the snapshot of one generation in five by every elite vector, entered on
    the generation listed first (tests/test_task8_archive.py).
    """
    _, _, path = run
    archive = np.loadtxt(path / "firefly_archive.txt", ndmin=2)
    assert archive.shape[1] == DIM + 3
    assert archive[:, 0].min() == 0 and archive[:, 0].max() > 0


def test_progress_metrics_record_this_generation_not_a_running_best(run):
    """A cost from another block is not comparable, so no column may accumulate a minimum."""
    import csv

    _, _, path = run
    with open(path / "progress.csv") as f:  # Task 4: a test directory keeps it in place
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
    assert (config["rng_seed"], config["workers"]) == (7, 2)
    assert config["settings"]["pop_size"] == POP
    assert config["settings"]["max_generations"] == GENERATIONS
    digest = config["initial_population_sha256"]
    assert len(digest) == 64 and int(digest, 16) >= 0
    # The same seed draws the same initial population, so its hash repeats.
    again = fo.Firefly(pop_size=POP, alpha_0=1, betamin=1, gamma=0.03, seed=7)
    lb, ub = np.zeros(DIM) + 0.1, np.ones(DIM)
    first = lb + again.rng.uniform(0, 1, (POP, DIM)) * (ub - lb)
    assert digest == hashlib.sha256(np.ascontiguousarray(first, dtype=float).tobytes()).hexdigest()


def test_an_unseeded_run_records_a_seed_that_redraws_it():
    """Without a seed the run takes the fixed default (Task 7), so rng_seed is never null."""
    first = fo.Firefly(pop_size=POP, alpha_0=1, betamin=1, gamma=0.03)
    again = fo.Firefly(pop_size=POP, alpha_0=1, betamin=1, gamma=0.03, seed=first.seed)
    assert isinstance(first.seed, int)
    assert np.array_equal(first.rng.uniform(size=5), again.rng.uniform(size=5))


def test_the_configuration_is_written_before_the_first_evaluation(monkeypatch, tmp_path):
    """Task 3: a run whose first evaluation fails still leaves its config.json."""
    import json

    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    engine = fo.Firefly(pop_size=POP, alpha_0=1, betamin=1, gamma=0.03, seed=7)
    with pytest.raises(RuntimeError):
        engine.firefly(function=standins.crashes, dim=DIM, lb=np.zeros(DIM) + 0.1, ub=np.ones(DIM),
                       demographic_parameters=None, max_generations=GENERATIONS, max_workers=2,
                       Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None)
    assert json.loads((tmp_path / "config.json").read_text())["rng_seed"] == 7


def test_every_file_the_optimiser_writes_marks_its_directory(run):
    """claim_run_directory's list must cover what a run leaves in its directory."""
    from covsyn.runtime import runs

    _, _, path = run
    written = {p.name for p in path.iterdir()}
    assert written and written <= set(runs.RUN_OUTPUTS), written - set(runs.RUN_OUTPUTS)
