# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 7, items 1 to 3: a fixed, committed initial population.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 7: covsyn-initial-population
--version N writes variable/initial_population_vN.npz, with normalised coordinates in [0, 1]
for n_max = 256 fireflies -- row 0 the literature seed vector, rows 1 to m the warm-start
vectors (run 10 and run 19), the rest a Latin hypercube with a fixed seed -- and the bounds,
the dimension, the generation command and a SHA-256. A run with n fireflies uses its first n
rows, and a run whose bounds or dimension differ from those stored is refused, not clipped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import qmc

from covsyn.calibration import initial_population as ip
from covsyn.calibration import parameter_space
from covsyn.runtime import runs
from tests.conftest import REPO_ROOT, best_vector


@pytest.fixture(scope="module")
def space():
    """The bounds and literature vector the optimiser uses."""
    return parameter_space.load(REPO_ROOT / "variable")


@pytest.fixture(scope="module")
def written(tmp_path_factory, space):
    """A version 1 population written into a fresh directory, and its path."""
    out = tmp_path_factory.mktemp("variable")
    return ip.main(["--version", "1", "--out-dir", str(out)])


def _normalised(vector, space):
    """A vector clipped to the bounds and put in [0, 1], a coincident bound at 0."""
    span = np.where(space.upper > space.lower, space.upper - space.lower, 1.0)
    return np.clip((np.clip(vector, space.lower, space.upper) - space.lower) / span, 0, 1)


def test_the_file_is_named_by_its_version(written):
    """initial_population_vN.npz, N the version."""
    assert written.name == "initial_population_v1.npz"


def test_the_population_holds_256_rows_in_the_unit_box(written, space):
    """n_max = 256 rows of normalised coordinates, with the bounds and the dimension."""
    with np.load(written) as saved:
        rows = saved["normalized"]
        assert rows.shape == (ip.N_MAX, len(space.lower)) == (256, 208)
        assert rows.min() >= 0 and rows.max() <= 1
        np.testing.assert_array_equal(saved["lower"], space.lower)
        np.testing.assert_array_equal(saved["upper"], space.upper)
        assert int(saved["dimension"]) == 208 and int(saved["version"]) == 1
        assert "covsyn-initial-population --version 1" in str(saved["command"])


def test_row_0_is_the_literature_vector_and_then_the_warm_starts(written, space):
    """Run 10's 199 parameters are padded from the literature vector (B54), as before."""
    with np.load(written) as saved:
        rows = saved["normalized"]
        sources = [str(s) for s in saved["warm_starts"]]
    assert sources == list(ip.WARM_STARTS)
    np.testing.assert_array_equal(rows[0], _normalised(space.literature, space))
    for k, source in enumerate(ip.WARM_STARTS, start=1):
        vector = parameter_space.pad_pre_b54_vector(best_vector(REPO_ROOT / source),
                                                    space.literature)  # fmt: skip
        np.testing.assert_array_equal(rows[k], _normalised(vector, space))


def test_the_other_rows_are_a_latin_hypercube_with_a_fixed_seed(written):
    """Each of the 253 remaining rows falls in its own stratum in every dimension."""
    first = 1 + len(ip.WARM_STARTS)
    with np.load(written) as saved:
        rows = saved["normalized"][first:]
        assert int(saved["lhs_seed"]) == ip.LHS_SEED
    want = qmc.LatinHypercube(d=rows.shape[1], rng=ip.LHS_SEED).random(len(rows))
    np.testing.assert_array_equal(rows, want)
    strata = np.floor(rows * len(rows)).astype(int)
    assert all(sorted(column) == list(range(len(rows))) for column in strata.T)


def test_the_sha256_covers_the_rows_and_bounds(written):
    """The stored SHA-256 is that of the rows and the bounds."""
    with np.load(written) as saved:
        assert str(saved["sha256"]) == ip.digest(saved["normalized"], saved["lower"],
                                                  saved["upper"])  # fmt: skip


def test_making_it_again_gives_the_same_population(written, tmp_path):
    """The generator is deterministic, so a version can be checked against it."""
    again = ip.main(["--version", "1", "--out-dir", str(tmp_path)])
    with np.load(written) as a, np.load(again) as b:
        assert str(a["sha256"]) == str(b["sha256"])
        np.testing.assert_array_equal(a["normalized"], b["normalized"])


def test_an_existing_version_is_never_overwritten(written):
    """A new population gets a new version."""
    before = written.read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        ip.main(["--version", "1", "--out-dir", str(written.parent)])
    assert written.read_bytes() == before


@pytest.mark.parametrize("argv", [[], ["--version", "0"], ["--version", "x"]])
def test_a_version_must_be_given_and_positive(argv, tmp_path):
    """Versions count from 1."""
    with pytest.raises(SystemExit):
        ip.main([*argv, "--out-dir", str(tmp_path)])


def test_a_run_takes_the_first_n_rows(written, space):
    """Item 2: a run with n fireflies uses the first n rows."""
    rows = ip.load(written, 7, space.lower, space.upper)
    with np.load(written) as saved:
        np.testing.assert_array_equal(rows, saved["normalized"][:7])


@pytest.mark.parametrize("n", [0, 257])
def test_more_fireflies_than_rows_or_none_are_refused(written, space, n):
    """A population cannot start more fireflies than it holds, nor none.

    As its own exception, which the optimiser turns into runs.NOT_RESUMABLE (decision of
    2026-11-24); another caller, pilot 1 say, exits 1 with the message (code review round 12).
    """
    with pytest.raises(ip.RunSizeOutOfRange, match=f"a run of {n} cannot start") as stopped:
        ip.load(written, n, space.lower, space.upper)
    # A message for code: Python prints it and exits 1.
    assert isinstance(stopped.value, SystemExit) and isinstance(stopped.value.code, str)


def _one_array(path):
    with open(path, "wb") as f:
        np.save(f, np.zeros(3))


def _no_sha256(path):
    np.savez(path, normalized=np.zeros((3, 2)), lower=np.zeros(2), upper=np.ones(2),
             warm_starts=np.array(["a"]))  # fmt: skip


@pytest.mark.parametrize("damage", [_one_array, _no_sha256, lambda path: path.write_bytes(b"")])
def test_a_damaged_population_is_refused_with_a_message(tmp_path, space, written, damage):
    """Not a traceback, and resumable: restoring the file makes a retry succeed (rounds 14, 15)."""
    path = tmp_path / "initial_population_v1.npz"
    damage(path)
    with pytest.raises(SystemExit) as stopped:
        ip.load(path, 3, space.lower, space.upper)
    assert isinstance(stopped.value.code, str) and "cannot be read" in stopped.value.code
    assert "to its last commit" in stopped.value.code
    assert not isinstance(stopped.value, ip.RunSizeOutOfRange)
    path.write_bytes(written.read_bytes())  # restored
    assert len(ip.load(path, 3, space.lower, space.upper)) == 3


def _arrays(path, **changed):
    """A complete population of 3 rows and 2 parameters, with only these arrays changed.

    Complete, so each case reaches the check it was written for (code review round 20).
    """
    arrays = {"normalized": np.zeros((3, 2)), "lower": np.zeros(2), "upper": np.ones(2),
              "sha256": np.array("x"), "warm_starts": np.array(["a"]), "version": np.array(1),
              **changed}  # fmt: skip
    np.savez(path, **arrays)


@pytest.mark.parametrize(("changed", "reason"), [
    ({"normalized": np.array([["a", "b"]])}, "not a non-empty 2-D array of numbers"),
    ({"normalized": np.zeros((0, 2))}, "not a non-empty 2-D array of numbers"),  # no rows
    ({"normalized": np.zeros((3, 0))}, "not a non-empty 2-D array of numbers"),  # no parameter
    ({"normalized": np.zeros(6)}, "not a non-empty 2-D array of numbers"),  # 1-D
    ({"warm_starts": np.array("a")}, "warm starts are not a list"),
    ({"lower": np.array(["a", "b"])}, "not one number a parameter"),  # bounds not numbers
    ({"lower": np.zeros(1), "upper": np.ones(1)}, "not one number a parameter"),  # length
    ({"normalized": np.full((3, 2), np.nan)}, "rows must be finite"),
    ({"normalized": np.full((3, 2), 1.1)}, "rows must fall within"),
    ({"upper": np.full(2, np.inf)}, "bounds must be finite"),
    ({"upper": np.full(2, -1)}, "lower bounds exceed upper bounds"),
    ({"warm_starts": np.array([["a"]])}, "warm starts are not a list"),
    ({"warm_starts": np.array([1])}, "warm starts are not a list"),
    ({"sha256": np.array(["x", "y"])}, "SHA-256 is not a string scalar"),
    ({"version": np.array(0)}, "version must be positive"),
    ({"version": np.array([1, 2])}, "version is not a whole number"),
])  # fmt: skip
def test_a_population_with_damaged_contents_is_unreadable(tmp_path, space, changed, reason):
    """Refused for its own reason, resumably, not a traceback or a run size (rounds 15, 20)."""
    path = tmp_path / "initial_population_v1.npz"
    _arrays(path, **changed)
    with pytest.raises(ip.PopulationUnreadable, match=reason):
        ip.rows_available(path)
    with pytest.raises(SystemExit) as stopped:
        ip.load(path, 1, space.lower, space.upper)
    assert reason in stopped.value.code
    assert not isinstance(stopped.value, ip.RunSizeOutOfRange)


@pytest.mark.parametrize("relative", [True, False])
def test_a_missing_population_is_to_be_restored_from_the_last_commit(tmp_path, space,
                                                                     monkeypatch, relative):
    """A direction, by the kind of failure only: no command whose premises go unchecked.

    Code review rounds 15 to 17 and the contract of 2026-11-24: a relative path also points
    to the working directory, which an absolute one does not depend on.
    """
    monkeypatch.chdir(tmp_path)
    path = Path("variable/initial_population_v1.npz")
    path = path if relative else tmp_path / path
    with pytest.raises(SystemExit) as stopped:
        ip.load(path, 3, space.lower, space.upper)
    message = stopped.value.code
    assert "last commit" in message and "never committed" in message
    assert "git checkout" not in message and "covsyn-initial-population" not in message
    assert ("working directory" in message) == relative


def test_a_directory_in_the_population_s_place_is_named_as_such(tmp_path, space):
    """Neither restoring over it nor a permission helps while it is there (round 16)."""
    path = tmp_path / "initial_population_v1.npz"
    path.mkdir()
    with pytest.raises(SystemExit) as stopped:
        ip.load(path, 3, space.lower, space.upper)
    assert "a directory stands where the file should be" in stopped.value.code


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root reads any file")
def test_a_population_that_may_not_be_read_asks_for_permission(tmp_path, space, written):
    """Restoring would not help (code review round 15)."""
    path = tmp_path / "initial_population_v1.npz"
    path.write_bytes(written.read_bytes())
    path.chmod(0)
    try:
        with pytest.raises(SystemExit) as stopped:
            ip.load(path, 3, space.lower, space.upper)
    finally:
        path.chmod(0o644)
    assert "may read" in stopped.value.code and "restore" not in stopped.value.code


def test_a_bytes_path_loads_as_rows_available_counts(written, space):
    """The two take the same paths (code review round 14)."""
    import os

    path = os.fsencode(written)
    assert len(ip.load(path, ip.rows_available(path), space.lower, space.upper)) == ip.N_MAX


def test_a_missing_population_stays_resumable(tmp_path, space):
    """A file put back makes a retry succeed, so a missing one is an ordinary failure."""
    with pytest.raises(SystemExit) as stopped:
        ip.load(tmp_path / "initial_population_v1.npz", 3, space.lower, space.upper)
    assert stopped.value.code != runs.NOT_RESUMABLE


def test_other_bounds_are_refused_not_clipped(written, space):
    """Item 3: a new population version is then made explicitly."""
    upper = space.upper.copy()
    upper[5] += 1e-9
    with pytest.raises(SystemExit, match="bounds"):
        ip.load(written, 3, space.lower, upper)


def test_another_dimension_is_refused(written, space):
    """Item 3: a run of another dimension needs a new version."""
    with pytest.raises(SystemExit, match="dimension"):
        ip.load(written, 3, space.lower[:-1], space.upper[:-1])


def test_a_changed_population_is_refused(written, space, tmp_path):
    """Rows that no longer match the stored SHA-256 are not used."""
    with np.load(written) as saved:
        data = {key: saved[key] for key in saved.files}
    data["normalized"] = data["normalized"].copy()
    data["normalized"][4, 4] = 0.5
    changed = tmp_path / "initial_population_v1.npz"
    np.savez(changed, **data)
    with pytest.raises(SystemExit, match="SHA-256"):
        ip.load(changed, 3, space.lower, space.upper)


def test_version_1_is_committed_and_matches_the_generator(written):
    """Item 2: the committed file is the one the generator makes."""
    committed = REPO_ROOT / "variable" / "initial_population_v1.npz"
    with np.load(committed) as a, np.load(written) as b:
        assert str(a["sha256"]) == str(b["sha256"])


def test_making_a_population_does_not_load_the_optimiser():
    """The population needs the bounds and two helpers, not the optimiser: no import cycle."""
    code = ("import sys, numpy as np, covsyn.calibration.initial_population as ip; "
            "ip.normalise(np.ones(2), np.zeros(2), np.full(2, 2.0), 'x'); "
            "print('covsyn.calibration.firefly_optimizer' in sys.modules)")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO_ROOT, check=True).stdout.strip()  # fmt: skip
    assert out == "False"


def test_the_command_hands_its_arguments_to_main_and_exits_zero(monkeypatch):
    """The console script runs main() on the command line as given, and exits 0 once done.

    main() reads sys.argv when given none; a returned path would print and give status 1.
    """
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    module, _, function = scripts["covsyn-initial-population"].partition(":")
    calls = []
    monkeypatch.setattr(ip, "main", lambda *args, **kwargs: calls.append((args, kwargs))
                        or Path("/a/run"))  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["covsyn-initial-population", "--given", "x"])
    assert getattr(import_module(module), function)() == 0
    assert calls == [((), {})]  # no arguments of its own, so argparse reads sys.argv
