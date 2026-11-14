# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 4: the optimiser checkpoints every generation and resumes from the last one.

The acceptance criterion of the rework plan (doc/plan_Nordling2026_CovSynFireflyNordlingLab.md,
Task 4): killing the run during the optimiser and resuming it gives the same progress table,
apart from timing columns, as an uninterrupted run. The objective is a deterministic stand-in,
so the comparison is exact.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo

DIM = 208
POP = 4
GENERATIONS = 5
SEED = 11


class Killed(Exception):
    """Stands in for the process being killed."""


def objective(P, demo, executor, contacts, attack, weights, seed_offset=0):
    """Deterministic in the vector and the seed block, like the real objective on fixed seeds."""
    return float(np.sum(np.asarray(P) * np.arange(1, len(P) + 1)) % 1.0) + seed_offset / 1e9


def killed_after(calls):
    """The objective, raising once ``calls`` evaluations have run."""
    count = {"n": 0}

    def function(*args, **kwargs):
        count["n"] += 1
        if count["n"] > calls:
            raise Killed
        return objective(*args, **kwargs)

    return function


def optimise(path, function, resume=False):
    """Run the optimiser into ``path`` and return its result tuple."""
    fo.result_path = path
    engine = fo.Firefly(pop_size=POP, alpha=1, betamin=1, gamma=0.03, seed=SEED)
    return engine.firefly(function=function, dim=DIM, lb=np.zeros(DIM), ub=np.ones(DIM),
                          demographic_parameters=None, max_generations=GENERATIONS,
                          max_workers=2, Cheng_contact_array=None, Cheng_attack_rate=None,
                          norm_weights=None, resume=resume)


def progress(path):
    """The progress table's rows."""
    with open(path / "progress_metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def uninterrupted(tmp_path, monkeypatch):
    """One run of all generations, and its directory."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    path = tmp_path / "whole"
    path.mkdir()
    return optimise(path, objective), path


# The initial population is POP evaluations, then each generation re-scores POP and moves some.
@pytest.mark.parametrize("calls", [POP + 3, 3 * POP + 5, 5 * POP + 1])
def test_a_killed_and_resumed_run_equals_an_uninterrupted_one(tmp_path, uninterrupted, calls):
    """The acceptance criterion: same progress table and same final population."""
    whole, whole_path = uninterrupted
    path = tmp_path / "killed"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_after(calls))
    resumed = optimise(path, objective, resume=True)
    assert progress(path) == progress(whole_path)
    for got, want in zip(resumed, whole, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.loadtxt(path / "firefly_archive.txt"),
                                  np.loadtxt(whole_path / "firefly_archive.txt"))


def test_every_generation_leaves_a_checkpoint(uninterrupted):
    """One checkpoint per generation, as the plan's run directory lists."""
    _, path = uninterrupted
    names = sorted(p.name for p in (path / "checkpoints").iterdir())
    assert names == [f"generation_{g:04d}.npz" for g in range(GENERATIONS)] + ["initial.npz"]


def test_a_finished_run_resumes_to_the_same_result(uninterrupted):
    """Resuming after the last generation evaluates nothing and returns the same population."""
    whole, path = uninterrupted
    again = optimise(path, killed_after(0), resume=True)  # any evaluation would raise
    for got, want in zip(again, whole, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_resuming_without_a_checkpoint_is_an_error(tmp_path):
    """Nothing to resume from is a mistake to report, not a fresh start in an old directory."""
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        optimise(tmp_path, objective, resume=True)


def test_progress_rows_after_the_checkpoint_are_dropped(tmp_path, uninterrupted):
    """Killed between a generation's progress row and its checkpoint, the row is written again."""
    _, whole_path = uninterrupted
    path = tmp_path / "between"
    path.mkdir()
    with pytest.raises(Killed):
        optimise(path, killed_after(3 * POP + 5))
    rows = progress(path)
    last = max(int(p.stem.split("_")[1]) for p in (path / "checkpoints").glob("generation_*"))
    with open(path / "progress_metrics.csv", "a", newline="") as f:  # a row past the checkpoint
        csv.DictWriter(f, fieldnames=list(rows[0])).writerow({**rows[-1], "generation": last + 1})
    optimise(path, objective, resume=True)
    assert progress(path) == progress(whole_path)


def test_the_moves_of_a_generation_come_from_its_own_seed():
    """SeedSequence([seed, g]): generation g's draws need no saved generator state."""
    first = fo.generation_rng(SEED, 3).random(5)
    again = fo.generation_rng(SEED, 3).random(5)
    other = fo.generation_rng(SEED, 4).random(5)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_checkpoints_are_written_whole(uninterrupted):
    """Written to a temporary name and renamed, so a kill never leaves half a checkpoint."""
    _, path = uninterrupted
    assert not list(Path(path / "checkpoints").glob("*.tmp*"))
    with np.load(path / "checkpoints" / f"generation_{GENERATIONS - 1:04d}.npz") as last:
        assert int(last["generation"]) == GENERATIONS - 1
        assert last["normalized"].shape == (POP, DIM)


def test_a_resumed_run_takes_its_seed_from_its_configuration(tmp_path):
    """The optimiser's --resume reads the seed config.json recorded, not a fresh one."""
    import json

    (tmp_path / "config.json").write_text(json.dumps({"seed": 123}))
    assert fo.resumed_seed(tmp_path) == 123


@pytest.mark.parametrize("config", [None, '{"claimed": true}', '{"seed": null}', "not json"])
def test_a_run_without_a_recorded_seed_cannot_be_resumed(tmp_path, config):
    """A claim marker, a missing file or a null seed would resume with the wrong moves."""
    if config is not None:
        (tmp_path / "config.json").write_text(config)
    with pytest.raises(SystemExit, match="cannot be resumed"):
        fo.resumed_seed(tmp_path)
