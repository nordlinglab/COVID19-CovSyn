# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 8, item 2: each elite member's running mean over its blocks, stored in the archive.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 8: give each elite member a
running mean over the blocks it has been scored on, counting each block once, and store it in
the archive. Decision of 2026-11-20: the archive holds every parameter vector that was in the
elite during the run, with its mean and the number of blocks counted; a vector that returns
after moving starts a new mean, an identical one continues its record, and generation 0's
reuse of the block-0 costs counts block 0 once. The checkpoint keeps the records, so a resumed
run writes the same archive (tests/test_task4_resume.py).

A generation g's elite members are those it kept unmoved and those its block-g costs put in
the next elite; each has a cost on block g.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.revalidation import load_candidates, read_archive
from tests import firefly_standins as standins

DIM = 208
POP = 8
GENERATIONS = 4


def _record(records, vector):
    """The record of vector: (entered, last block, blocks, total, params...)."""
    rows = [r for r in records if np.array_equal(r[fo.RECORD_PARAMS:], vector)]
    assert len(rows) == 1
    return rows[0]


def test_a_new_member_starts_its_mean_from_its_block():
    """One block counted, the mean its cost, entered on that block's generation."""
    vectors = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    records = fo.record_elite(fo.no_elite_records(2), vectors, np.array([3.0, 1.0, 2.0]),
                              [1, 2], 0)  # fmt: skip
    assert len(records) == 2
    entered, last, blocks, total = _record(records, vectors[2])[: fo.RECORD_PARAMS]
    assert (entered, last, blocks, total) == (0, 0, 1, 2.0)


def test_a_member_kept_in_place_adds_each_block_once():
    """The same vector on blocks 0 and 1 averages both; block 1 again changes nothing."""
    vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
    records = fo.record_elite(fo.no_elite_records(2), vectors, np.array([4.0, 9.0]), [0], 0)
    records = fo.record_elite(records, vectors, np.array([2.0, 9.0]), [0], 1)
    again = fo.record_elite(records, vectors, np.array([7.0, 9.0]), [0], 1)
    np.testing.assert_array_equal(again, records)
    rows = fo.archive_rows(records)
    np.testing.assert_array_equal(rows, [[0, 2, 0.1, 0.2, 3.0]])  # entered, blocks, x, mean


def test_a_member_that_moved_starts_a_new_mean():
    """Firefly 0 at a new position is a new vector, with a record of its own."""
    before, after = np.array([[0.1, 0.2]]), np.array([[0.5, 0.5]])
    records = fo.record_elite(fo.no_elite_records(2), before, np.array([1.0]), [0], 0)
    records = fo.record_elite(records, after, np.array([5.0]), [0], 3)
    rows = fo.archive_rows(records)
    np.testing.assert_array_equal(rows, [[0, 1, 0.1, 0.2, 1.0], [3, 1, 0.5, 0.5, 5.0]])


def test_no_members_leave_no_records():
    """A population of one has no elite, so the archive is empty."""
    records = fo.record_elite(fo.no_elite_records(2), np.array([[0.1, 0.2]]), np.array([1.0]),
                              [], 0)  # fmt: skip
    assert records.shape == (0, fo.RECORD_PARAMS + 2)
    assert fo.archive_rows(records).shape == (0, 5)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A run of POP fireflies, k = 5, and its directory; read only."""
    path = tmp_path_factory.mktemp("archive")
    previous = getattr(fo, "result_path", None)
    fo.result_path = path
    try:
        engine = fo.Firefly(pop_size=POP, alpha_0=0.2, betamin=1, gamma=0.03, seed=6)
        engine.firefly(function=standins.objective, dim=DIM, lb=np.zeros(DIM), ub=np.ones(DIM),
                       demographic_parameters=None, max_generations=GENERATIONS,
                       max_workers=2, Cheng_contact_array=None, Cheng_attack_rate=None,
                       norm_weights=None)  # fmt: skip
    finally:
        fo.result_path = previous
    return path


def _checkpoint(path, generation):
    name = "initial" if generation < 0 else f"generation_{generation:04d}"
    with np.load(path / fo.CHECKPOINTS / f"{name}.npz") as saved:
        return {key: saved[key] for key in saved.files}


def test_each_elite_vector_s_mean_is_over_the_blocks_it_was_in_the_elite(run):
    """Recomputed from the checkpoints: a mean, and a count, per distinct elite vector."""
    k = fo.elite_size(POP)
    scores: dict[bytes, list[float]] = {}
    entered: dict[bytes, int] = {}
    for g in range(GENERATIONS):
        before, after = _checkpoint(run, g - 1), _checkpoint(run, g)
        members = set(fo.elite_members(before["intensities"], k))
        members |= set(fo.elite_members(after["intensities"], k))
        for m in sorted(members):
            key = after["fireflies"][m].tobytes()
            scores.setdefault(key, []).append(float(after["intensities"][m]))
            entered.setdefault(key, g)
    records = _checkpoint(run, GENERATIONS - 1)["elite_records"]
    assert len(records) == len(scores)
    for row in records:
        key = row[fo.RECORD_PARAMS:].tobytes()
        assert row[0] == entered[key]
        assert row[2] == len(scores[key])
        assert row[3] / row[2] == pytest.approx(np.mean(scores[key]))


def test_generation_0_counts_block_0_once(run):
    """The elite kept its block-0 costs, which the initial population already scored."""
    records = _checkpoint(run, 0)["elite_records"]
    assert set(records[:, 2]) == {1.0}  # every record so far has one block, block 0


def test_the_archive_file_holds_the_records_under_its_header(run):
    """firefly_archive.txt: a v2 header, then entered, blocks, params and mean per vector."""
    text = (run / "firefly_archive.txt").read_text()
    assert text.startswith("# " + fo.ARCHIVE_HEADER)
    rows = np.loadtxt(run / "firefly_archive.txt", ndmin=2)
    want = fo.archive_rows(_checkpoint(run, GENERATIONS - 1)["elite_records"])
    np.testing.assert_allclose(rows, want, atol=1e-7)


def test_revalidation_reads_the_archive_s_vectors_and_means(run):
    """Both versions: v2 by its header, the B81 snapshots (generation, params, cost) without."""
    vectors, means = read_archive(run / "firefly_archive.txt")
    rows = np.loadtxt(run / "firefly_archive.txt", ndmin=2)
    np.testing.assert_array_equal(vectors, rows[:, 2:-1])
    np.testing.assert_array_equal(means, rows[:, -1])


def test_revalidation_still_reads_an_archive_of_snapshots(tmp_path):
    """A run from before Task 8 wrote (generation, params..., cost) with no header."""
    snapshots = np.array([[0.0, 0.1, 0.2, 4.0], [5.0, 0.3, 0.4, 5.0]])
    np.savetxt(tmp_path / "firefly_archive.txt", snapshots)
    vectors, costs = read_archive(tmp_path / "firefly_archive.txt")
    np.testing.assert_array_equal(vectors, snapshots[:, 1:-1])
    np.testing.assert_array_equal(costs, snapshots[:, -1])


def test_an_empty_archive_adds_no_candidates(tmp_path):
    """The header alone, as a run without an elite writes it."""
    np.savetxt(tmp_path / "firefly_archive.txt", np.empty((0, 5)), header=fo.ARCHIVE_HEADER)
    vectors, costs = read_archive(tmp_path / "firefly_archive.txt")
    assert len(vectors) == len(costs) == 0


def test_the_candidates_include_every_archived_elite_vector(run, tmp_path):
    """Revalidation's pool: the final population and every vector of the archive."""
    import shutil

    shutil.copy(run / "firefly_archive.txt", tmp_path)
    last = _checkpoint(run, GENERATIONS - 1)
    population = np.column_stack([last["fireflies"], last["intensities"]])
    np.savetxt(tmp_path / "firefly_result.txt", population, fmt="%.7f")
    np.savetxt(tmp_path / "firefly_best.txt",
               np.column_stack([np.zeros(POP), population]), fmt="%.7f")  # fmt: skip
    vectors, _ = load_candidates(tmp_path)
    for row in np.loadtxt(tmp_path / "firefly_archive.txt", ndmin=2):
        assert any(np.array_equal(row[2:-1], v) for v in vectors)
