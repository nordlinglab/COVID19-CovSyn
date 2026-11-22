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

import numpy as np
import pytest
from scipy.stats import qmc

from covsyn.calibration import initial_population as ip
from covsyn.calibration import parameter_space
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
    """A population cannot start more fireflies than it holds, nor none."""
    with pytest.raises(SystemExit, match="firefl"):
        ip.load(written, n, space.lower, space.upper)


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
    import subprocess
    import sys

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
    import sys
    import tomllib
    from importlib import import_module
    from pathlib import Path

    from tests.conftest import REPO_ROOT

    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    module, _, function = scripts["covsyn-initial-population"].partition(":")
    calls = []
    monkeypatch.setattr(ip, "main", lambda *args, **kwargs: calls.append((args, kwargs))
                        or Path("/a/run"))  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["covsyn-initial-population", "--given", "x"])
    assert getattr(import_module(module), function)() == 0
    assert calls == [((), {})]  # no arguments of its own, so argparse reads sys.argv
