# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Re-scoring candidate vectors on seeds the optimizer never saw (finding E87).

Run 12's best vector sat exactly on the edge of several targets on the objective's 300 fixed
seeds and fell outside them on 1,000 independent simulations. fast_cost.cost_function therefore
takes a seed offset, so the candidates can be scored on fresh seed blocks; the default offset of
0 is the objective exactly as before.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import numpy as np
import pytest

from covsyn.calibration import revalidation as rv


def test_seed_blocks_do_not_overlap_the_objective_or_each_other() -> None:
    """Seed blocks do not overlap the objective or each other."""
    blocks = rv.validation_seed_blocks(4, simulations=300)
    starts = [b.start for b in blocks]
    assert all(b.start >= rv.VALIDATION_SEED_START for b in blocks)
    assert rv.VALIDATION_SEED_START >= 300
    assert all(len(b) == 300 for b in blocks)
    assert len({s for b in blocks for s in b}) == 1200
    assert starts == sorted(starts)


def test_candidates_merge_population_and_personal_bests_without_duplicates(tmp_path) -> None:
    """Candidates merge population and personal bests without duplicates."""
    population = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 0.7]])  # params..., cost
    bests = np.array([[7, 1.0, 2.0, 0.5], [9, 5.0, 6.0, 0.6]])  # iteration, params..., cost
    np.savetxt(tmp_path / "firefly_result.txt", population)
    np.savetxt(tmp_path / "firefly_best.txt", bests)
    vectors, costs = rv.load_candidates(tmp_path)
    assert vectors.shape == (3, 2)
    np.testing.assert_array_equal(costs, [0.5, 0.6, 0.7])  # sorted by training cost
    np.testing.assert_array_equal(vectors[0], [1.0, 2.0])


def test_best_file_is_two_dimensional_and_its_minimum_is_the_chosen_vector(tmp_path) -> None:
    """The best file has at least two rows and its minimum is the chosen vector.

    The report scripts read firefly_best.txt with np.loadtxt(...)[:, -1], which needs at
    least two rows; the file holds every candidate with its validation cost.
    """
    vectors = np.arange(15, dtype=float).reshape(3, 5)
    rv.write_best_file(tmp_path / "firefly_best.txt", vectors, np.array([2.0, 1.25, 3.0]))
    rows = np.loadtxt(tmp_path / "firefly_best.txt")
    assert rows.ndim == 2 and rows.shape == (3, 7)
    best = rows[int(np.argmin(rows[:, -1]))]
    np.testing.assert_allclose(best[1:-1], vectors[1])
    assert best[-1] == pytest.approx(1.25)


def test_best_file_with_a_single_candidate_is_still_two_dimensional(tmp_path) -> None:
    """Best file with a single candidate is still two dimensional."""
    rv.write_best_file(tmp_path / "firefly_best.txt", np.ones((1, 4)), np.array([1.0]))
    assert np.loadtxt(tmp_path / "firefly_best.txt").ndim == 2


def test_output_directory_holds_the_whole_run_for_the_report(tmp_path) -> None:
    """OUT_DIR holds the whole run, with only firefly_best.txt replaced.

    phase_d_chain.sh copies OUT_DIR over the run directory, and show_final reads
    progress_metrics.csv from it.
    """
    run, out = tmp_path / "run", tmp_path / "out"
    run.mkdir()
    out.mkdir()
    for name in ("bound.txt", "progress_metrics.csv", "firefly_result.txt"):
        (run / name).write_text(name)
    (run / "firefly_best.txt").write_text("training")
    (out / "firefly_best.txt").write_text("validation")
    rv.copy_run_files(run, out)
    assert {p.name for p in out.iterdir()} == {
        "bound.txt",
        "progress_metrics.csv",
        "firefly_result.txt",
        "firefly_best.txt",
    }
    assert (out / "firefly_best.txt").read_text() == "validation"


def test_default_seed_offset_is_the_objective_and_others_differ(
    run10_vector: np.ndarray,
    demographic_parameters: Any,
    cheng_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    cost_pool: concurrent.futures.ProcessPoolExecutor,
) -> None:
    """Default seed offset is the objective and others differ."""
    pytest.importorskip("sklearn")
    from covsyn.calibration import fast_cost

    default = fast_cost.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    zero = fast_cost.cost_function(
        run10_vector, demographic_parameters, cost_pool, *cheng_data, seed_offset=0
    )
    moved = fast_cost.cost_function(
        run10_vector,
        demographic_parameters,
        cost_pool,
        *cheng_data,
        seed_offset=rv.VALIDATION_SEED_START,
    )
    assert default == zero
    assert moved != default
