# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 3: name every run and create its directories in one place.

The expected values are the rework plan's (doc/plan_Nordling2026_CovSynFireflyNordlingLab.md,
Task 3): runs are named ``<YYYYMMDD-HHMMSS>_<commit7>[-dirty]_<config-name>`` and live under
``results/models/``, ``results/tables/`` and ``results/figures/``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from covsyn.runtime import runs

WHEN = datetime(2026, 11, 13, 9, 5, 7)


def test_a_run_is_named_by_time_commit_and_configuration():
    """The plan's pattern, with the commit cut to seven characters."""
    run_id = runs.make_run_id("phase-d", when=WHEN, commit="81c93604b908ea", dirty=False)
    assert run_id == "20261113-090507_81c9360_phase-d"


def test_uncommitted_changes_mark_the_run_dirty():
    """A run from a modified checkout cannot be reproduced from its commit alone."""
    run_id = runs.make_run_id("phase-d", when=WHEN, commit="81c9360", dirty=True)
    assert run_id == "20261113-090507_81c9360-dirty_phase-d"


def test_outside_a_git_checkout_the_commit_says_so():
    """No commit to name is recorded as such, not as an empty field."""
    assert runs.make_run_id("x", when=WHEN, commit="", dirty=False) == "20261113-090507_nocommit_x"


def test_the_commit_and_dirty_flag_default_to_this_checkout():
    """Without arguments the id names this repository's HEAD."""
    run_id = runs.make_run_id("phase-d", when=WHEN)
    commit = run_id.split("_")[1].removesuffix("-dirty")
    assert len(commit) == 7 or commit == "nocommit"


@pytest.mark.parametrize("name", ["", "a/b", "../x", "with space", "a_b", ".hidden", "x" * 65])
def test_a_configuration_name_that_cannot_name_a_directory_is_rejected(name):
    """Names go into paths and are split at '_', so only kebab-case words are allowed."""
    with pytest.raises(ValueError):
        runs.make_run_id(name, when=WHEN, commit="81c9360", dirty=False)


@given(st.text(max_size=80))
def test_any_accepted_name_is_safe_in_a_path(name):
    """Whatever the input, an accepted name has no separator, '..', '_' or white space."""
    try:
        run_id = runs.make_run_id(name, when=WHEN, commit="81c9360", dirty=False)
    except ValueError:
        return
    assert run_id.endswith("_" + name)
    assert name and "/" not in name and "\\" not in name and ".." not in name
    assert "_" not in name and not any(c.isspace() for c in name)


@given(st.from_regex(r"[a-z0-9]+(-[a-z0-9]+)*", fullmatch=True).filter(lambda s: len(s) <= 64))
def test_any_kebab_case_name_is_accepted(name):
    """Every lowercase kebab-case name up to 64 characters is a valid configuration name."""
    assert runs.make_run_id(name, when=WHEN, commit="81c9360", dirty=False).endswith(name)


@pytest.mark.parametrize("kind", ["models", "tables", "figures"])
def test_run_directory_creates_each_kind_under_results(tmp_path, kind):
    """A fresh clone has no results/ yet; the function creates the whole path."""
    path = runs.run_directory("20261113-090507_81c9360_phase-d", kind, root=tmp_path / "results")
    assert path == tmp_path / "results" / kind / "20261113-090507_81c9360_phase-d"
    assert path.is_dir()


def test_run_directory_accepts_an_existing_directory(tmp_path):
    """Creating it twice, as a resumed run does, is not an error."""
    first = runs.run_directory("r", root=tmp_path)
    (first / "kept.txt").write_text("1")
    assert runs.run_directory("r", root=tmp_path) == first
    assert (first / "kept.txt").read_text() == "1"


@pytest.mark.parametrize("run_id", ["", "../escape", "a/b", "/abs"])
def test_run_directory_refuses_a_path_as_run_id(tmp_path, run_id):
    """A run id is one directory name; anything else could write outside results/."""
    with pytest.raises(ValueError):
        runs.run_directory(run_id, root=tmp_path)


def test_run_directory_refuses_an_unknown_kind(tmp_path):
    """Only the three directories of the layout exist."""
    with pytest.raises(ValueError):
        runs.run_directory("r", "logs", root=tmp_path)


def test_config_records_what_reproduces_the_run(tmp_path):
    """Settings, seed, workers, machine, versions and the two hashes the plan names."""
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    population = np.arange(6.0).reshape(2, 3)
    path = runs.write_config(
        tmp_path,
        {"pop_size": 2, "gamma": 0.03},
        seed=7,
        workers=4,
        initial_population=population,
        lock=lock,
    )
    config = json.loads(path.read_text())
    assert path == tmp_path / "config.json"
    assert config["settings"] == {"pop_size": 2, "gamma": 0.03}
    assert (config["seed"], config["workers"]) == (7, 4)
    assert config["uv_lock_sha256"] == hashlib.sha256(b"version = 1\n").hexdigest()
    expected = hashlib.sha256(np.ascontiguousarray(population, dtype=float).tobytes()).hexdigest()
    assert config["initial_population_sha256"] == expected
    assert config["machine"]["performance_cores"] >= 1 and config["machine"]["core_method"]
    assert config["machine"]["cpu_model"]
    assert config["versions"]["python"].count(".") == 2
    assert config["versions"]["packages"]["numpy"] == np.__version__


def test_config_without_a_lock_or_population_records_none(tmp_path):
    """An installed copy without uv.lock, or a run that draws its population later."""
    path = runs.write_config(
        tmp_path, {}, seed=None, workers=1, initial_population=None, lock=tmp_path / "missing.lock"
    )
    config = json.loads(path.read_text())
    assert config["uv_lock_sha256"] is None and config["initial_population_sha256"] is None


def test_config_refuses_settings_json_cannot_hold(tmp_path):
    """A setting that would be written as its repr could not be read back."""
    with pytest.raises(TypeError):
        runs.write_config(
            tmp_path,
            {"path": object()},
            seed=0,
            workers=1,
            initial_population=None,
            lock=tmp_path / "uv.lock",
        )


def test_a_script_reads_the_run_directory_from_its_arguments():
    """Task 3: every script that reads a run takes its directory as an argument."""
    from pathlib import Path

    argv = ["show_final", "results/models/r"]
    assert runs.run_dir_argument(argv, 1, "show_final RUN_DIR") == Path("results/models/r")


def test_a_missing_run_directory_stops_with_the_usage(capsys):
    """No legacy default: without the argument the script says what it needs and stops."""
    with pytest.raises(SystemExit) as stop:
        runs.run_dir_argument(["show_final"], 1, "show_final RUN_DIR")
    assert stop.value.code == 2
    assert "usage: show_final RUN_DIR" in capsys.readouterr().err


def test_an_empty_run_directory_argument_is_missing():
    """The shell passes '' for an unset variable; that names no run."""
    with pytest.raises(SystemExit):
        runs.run_dir_argument(["x", ""], 1, "x RUN_DIR")


def test_a_new_run_refuses_an_existing_directory(tmp_path):
    """Two runs in the same second with one configuration must not share a directory."""
    fields = {"when": WHEN, "commit": "81c9360", "dirty": False}
    first = runs.new_run_directory("phase-d", root=tmp_path, **fields)
    assert first == tmp_path / "models" / "20261113-090507_81c9360_phase-d" and first.is_dir()
    with pytest.raises(FileExistsError):
        runs.new_run_directory("phase-d", root=tmp_path, **fields)


@pytest.mark.parametrize("output", ["config.json", "firefly_result.txt", "firefly_best.txt"])
def test_a_given_directory_holding_a_run_is_refused(tmp_path, output):
    """FIREFLY_RUN=results/models/phaseD_run10 must not overwrite the committed run 10."""
    (tmp_path / output).write_text("1")
    with pytest.raises(FileExistsError):
        runs.claim_run_directory(tmp_path)


def test_a_given_new_or_empty_directory_is_claimed(tmp_path):
    """The launcher's fresh directory, or one the user names, is created and used."""
    assert runs.claim_run_directory(tmp_path / "a" / "b") == tmp_path / "a" / "b"
    (tmp_path / "empty").mkdir()
    assert runs.claim_run_directory(tmp_path / "empty").is_dir()


def test_results_live_under_the_repository_root():
    """Wherever a command starts, runs land where .gitignore and the chain expect them."""
    assert runs.RESULTS_ROOT == runs.REPO_ROOT / "results"
    assert (runs.REPO_ROOT / "pyproject.toml").is_file()


def test_versions_are_recorded_without_an_installed_covsyn(monkeypatch, tmp_path):
    """Run from a source tree, covsyn's own metadata is missing; the config is still written."""

    def missing(name):
        raise runs.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runs.metadata, "requires", missing)
    path = runs.write_config(
        tmp_path, {}, seed=1, workers=1, initial_population=None, lock=tmp_path / "uv.lock"
    )
    assert json.loads(path.read_text())["versions"]["packages"]["numpy"] == np.__version__


@pytest.mark.parametrize("output", runs.RUN_OUTPUTS)
def test_every_optimiser_output_marks_a_directory_as_holding_a_run(tmp_path, output):
    """progress_metrics.csv of an earlier run would otherwise get a second run appended."""
    (tmp_path / output).write_text("1")
    with pytest.raises(FileExistsError, match="FIREFLY_RUN"):
        runs.claim_run_directory(tmp_path)


def test_the_command_prints_a_new_run_path_without_creating_it(tmp_path):
    """The launcher sets FIREFLY_RUN to what this prints; the optimiser claims it later."""
    import subprocess
    import sys
    from pathlib import Path

    done = subprocess.run(
        [sys.executable, "-m", "covsyn.runtime.runs", "phase-d", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    printed = Path(done.stdout.strip())
    assert printed.parts[0] == "models" and printed.name.endswith("_phase-d")
    assert not (tmp_path / printed).exists()


def test_the_optimiser_claims_a_given_directory_and_names_a_new_one(tmp_path, monkeypatch):
    """main() goes through runs: a committed run is refused, and no --run-dir is a new run."""
    from covsyn.calibration import firefly_optimizer as fo

    held = tmp_path / "phaseD_run10"
    held.mkdir()
    (held / "firefly_best.txt").write_text("1 2 3\n")
    with pytest.raises(FileExistsError):
        fo.run_directory_for(held, "phase-d")
    assert fo.run_directory_for(tmp_path / "fresh", "phase-d") == tmp_path / "fresh"
    monkeypatch.setattr(runs, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(runs.new_run_directory, "__defaults__", (tmp_path / "results",))
    created = fo.run_directory_for(None, "phase-d")
    assert created.parent == tmp_path / "results" / "models" and created.is_dir()


def test_a_claim_is_atomic(tmp_path, monkeypatch):
    """The first claim creates config.json (valid JSON); a concurrent second one is refused.

    The race is forced by hiding config.json from the first check, as a second process that
    checked before the first one wrote would see it.
    """
    runs.claim_run_directory(tmp_path / "d")
    assert json.loads((tmp_path / "d" / "config.json").read_text()) == {"claimed": True}
    monkeypatch.setattr(runs, "RUN_OUTPUTS", ())
    with pytest.raises(FileExistsError, match="--run-dir"):
        runs.claim_run_directory(tmp_path / "d")


@pytest.mark.parametrize("chain_output", ["revalidation", "selected"])
def test_a_directory_with_chain_output_holds_a_run(tmp_path, chain_output):
    """A run whose optimiser files are gone still has its chain's revalidation or selection."""
    (tmp_path / chain_output).mkdir()
    with pytest.raises(FileExistsError, match="--run-dir"):
        runs.claim_run_directory(tmp_path)


@pytest.mark.parametrize("text", ["-1", "x", "1.5", "+1", "1_0", " 1", "\u0663"])
def test_a_bad_seed_is_refused_when_parsing(text):
    """Refused before the optimiser creates a run directory it would then leave empty."""
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        runs.non_negative_int(text)
    assert runs.non_negative_int("0") == 0


def test_run_outputs_name_every_file_the_optimiser_writes():
    """Every result_path/'name' in the optimiser is on the list, and every listed file is written.

    The run fixture of test_b81 sees only Firefly.firefly's files; main() writes the others.
    """
    import re
    from pathlib import Path

    source = (Path(runs.__file__).parents[1] / "calibration" / "firefly_optimizer.py").read_text(
        encoding="utf-8"
    )
    # Single or double quotes, f-strings included: a name with {} then fails the comparison.
    written = set(re.findall(r"result_path\s*/\s*f?['\"]([^'\"]+)['\"]", source))
    written |= {"config.json"}
    files = {name for name in runs.RUN_OUTPUTS if "." in name}
    assert written == files, (written - files, files - written)


def test_a_dangling_link_counts_as_a_run_output(tmp_path):
    """A config.json link left from a copied run would otherwise be reported as a race."""
    (tmp_path / "config.json").symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError, match="already holds a run"):
        runs.claim_run_directory(tmp_path)


def test_checking_a_directory_creates_nothing(tmp_path):
    """The optimiser's startup check leaves no directory and no claim behind."""
    runs.check_unclaimed(tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("kind", ["file", "dangling"])
def test_a_run_dir_that_is_not_a_directory_fails_the_check(tmp_path, kind):
    """A mistyped --run-dir naming a file fails at startup, not after the inputs load."""
    path = tmp_path / "x"
    if kind == "file":
        path.write_text("1 2 3\n")
    else:
        path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="not a directory"):
        runs.check_unclaimed(path)


def test_the_launchers_check_command_refuses_a_held_run(tmp_path):
    """The check command exits 1 on a directory holding a run and 0 on a fresh one."""
    import subprocess
    import sys

    (tmp_path / "held").mkdir()
    (tmp_path / "held" / "firefly_best.txt").write_text("1\n")
    check = [sys.executable, "-m", "covsyn.runtime.runs", "--check"]
    held = subprocess.run([*check, str(tmp_path / "held")], capture_output=True, text=True)
    fresh = subprocess.run([*check, str(tmp_path / "fresh")], capture_output=True, text=True)
    assert held.returncode == 1 and "already holds a run" in held.stderr
    assert fresh.returncode == 0 and not (tmp_path / "fresh").exists()


def test_a_run_dir_under_a_file_fails_the_check(tmp_path):
    """results/models/run10.txt/new, with run10.txt a file, cannot be created later either."""
    (tmp_path / "run10.txt").write_text("1\n")
    with pytest.raises(FileExistsError, match="run10.txt is not a directory"):
        runs.check_unclaimed(tmp_path / "run10.txt" / "new")


@pytest.mark.parametrize("extra", [[], ["a", "b"]])
def test_the_check_command_needs_exactly_one_directory(extra):
    """A wrong call gets the usage, not a traceback about the configuration name."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "covsyn.runtime.runs", "--check", *extra],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1 and "usage:" in done.stderr and "Traceback" not in done.stderr
