# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 4: a pipeline run is named by its run id, results/models/<run-id>, and nothing else.

The plan names runs results/models/<run-id> (Task 3) and resumes them with --resume <run-id>
(Task 4). A run named any other way (".", a storage path, a subdirectory, a link) could put its
tables and figures somewhere else, so it is refused before anything is written. Acceptance
cases from 2026-11-15_task4_symlinked_runs_recommendations.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from covsyn.pipeline import config as pc
from covsyn.pipeline import runner as pr
from covsyn.runtime import runs

RUN_ID = "20261115-090000_b30c141_phase-d"


class Recorder:
    """A step runner that runs nothing and records nothing; fails nothing."""

    def __call__(self, args, output, append, env):
        """Leave the files the optimiser would, so later stages have a run to read."""
        args = [str(a) for a in args]
        if args[1] == "covsyn.calibration.firefly_optimizer":
            run = Path(args[args.index("--run-dir") + 1] if "--run-dir" in args
                       else args[args.index("--resume") + 1])
            (run / "firefly_best.txt").write_text("1\n")
        if args[1] == "covsyn.model.data_synthesis_main":
            out = Path(args[args.index("--result_path") + 1])
            (out / "social_data_0.npy").write_text("x")
        if args[1] in ("covsyn.calibration.revalidation", "covsyn.calibration.select_by_checklist"):
            Path(args[3]).mkdir(parents=True, exist_ok=True)
        return 0


@pytest.fixture
def results(tmp_path, monkeypatch):
    """A results root of its own, as RESULTS_ROOT."""
    root = tmp_path / "results"
    (root / "models").mkdir(parents=True)
    monkeypatch.setattr(runs, "RESULTS_ROOT", root)
    return root


def written(root):
    """Every file under root, to show that a refused call wrote none."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_a_run_id_and_its_full_path_name_the_same_run(results):
    """--resume <run-id> and --resume results/models/<run-id> resolve alike."""
    assert pr.resolve_resume(RUN_ID, results) == results / "models" / RUN_ID
    assert pr.resolve_resume(str(results / "models" / RUN_ID), results) == \
        results / "models" / RUN_ID


def test_a_run_id_is_found_from_any_working_directory(results, tmp_path, monkeypatch):
    """The results root is CovSyn's own, not the caller's."""
    monkeypatch.chdir(tmp_path)
    assert pr.resolve_resume(RUN_ID, results) == results / "models" / RUN_ID


@pytest.mark.parametrize("given", ["", ".", "..", "a/b", "../" + RUN_ID, RUN_ID + "/selected",
                                   "./" + RUN_ID])
def test_anything_but_a_run_id_or_its_full_path_is_refused(results, given):
    """An empty or path-like id, a parent step or a subdirectory names no run."""
    before = written(results)
    with pytest.raises(ValueError, match="run id"):
        pr.resolve_resume(given, results)
    assert written(results) == before


def test_a_storage_path_or_dot_inside_the_run_is_refused(results, tmp_path, monkeypatch):
    """Named by where it is stored, the run's tables could move; the id is asked for instead."""
    storage = tmp_path / "storage" / RUN_ID
    storage.mkdir(parents=True)
    with pytest.raises(ValueError, match="results/models"):
        pr.resolve_resume(str(storage), results)
    monkeypatch.chdir(storage)
    with pytest.raises(ValueError, match="run id"):
        pr.resolve_resume(".", results)


def test_a_linked_run_directory_is_not_supported(results, tmp_path):
    """results/models/<id> as a link to other storage is refused, not followed."""
    storage = tmp_path / "storage" / RUN_ID
    storage.mkdir(parents=True)
    (results / "models" / RUN_ID).symlink_to(storage)
    with pytest.raises(ValueError, match="link"):
        pr.run(None, results / "models" / RUN_ID, execute=Recorder(), resume=True,
               results_root=results)
    with pytest.raises(ValueError, match="link"):
        pr.run(pc.load("default"), results / "models" / RUN_ID, execute=Recorder(),
               results_root=results)
    assert written(results) == ["models", f"models/{RUN_ID}"]


@pytest.mark.parametrize("where", ["outside", "subdirectory", "dotdot"])
def test_the_pipeline_entry_checks_the_path_too(results, tmp_path, where):
    """run() refuses a non-standard directory itself, so no caller can go round the CLI."""
    run_dir = {"outside": tmp_path / "elsewhere" / RUN_ID,
               "subdirectory": results / "models" / RUN_ID / "selected",
               "dotdot": results / "models" / RUN_ID / ".." / RUN_ID}[where]
    with pytest.raises(ValueError):
        pr.run(pc.load("default"), run_dir, execute=Recorder(), results_root=results)
    assert written(results) == ["models"]


def test_other_runs_are_never_looked_at(results, tmp_path, monkeypatch):
    """A link in results/models to a dead network share cannot slow this run down."""
    (results / "models" / "old").symlink_to(tmp_path / "gone")  # dangling, like a stale mount
    looked = []
    real = os.stat

    def watching(path, *args, **kwargs):
        if "old" in str(path):
            looked.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", watching)
    assert pr.run(pc.load("default"), results / "models" / RUN_ID, execute=Recorder(),
                  results_root=results) == 0
    assert looked == []


def test_selected_and_revalidation_find_their_runs_table(results):
    """The chain hands steps run/selected and run/revalidation; the table is the run's."""
    run = results / "models" / RUN_ID
    table = results / "tables" / RUN_ID / "progress.csv"
    for sub in ("", "selected", "revalidation"):
        assert runs.progress_path(run / sub if sub else run) == table
