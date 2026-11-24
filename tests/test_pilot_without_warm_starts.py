# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A pilot run can start without the warm starts, the fitted vectors of earlier runs.

Pilot 2 of 2026-11-23 (9792f7f) returned run 19, row 2 of the initial population, as the best
vector of 11 of its 12 runs: kept in the elite from generation 0, it was never beaten, so every
setting "agreed" and the rule compared nothing (user decision 2026-11-24: rerun without it).
--without-warm-starts takes the literature vector and then the Latin hypercube rows, leaving out
rows 1 to len(WARM_STARTS); a choice of the pilot mode, while production always leaves them out
(user decision 2026-11-24: its result is to be the calibration's own).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration import initial_population as ip
from covsyn.calibration import parameter_space
from tests.test_task7_optimiser import POPULATION, started  # noqa: F401 (a fixture)

PILOT = ("--mode", "pilot", "--pop-size", "24", "--generations", "415", "--simulations", "3000",
         "--alpha-0", "0.2")  # fmt: skip


def _space():
    space = parameter_space.load()
    return space.lower, space.upper


def test_without_warm_starts_the_rows_are_the_literature_then_the_hypercube():
    """Row 0, then rows 1 + len(WARM_STARTS) onwards."""
    lower, upper = _space()
    rows = ip.load(ip.population_path(1), 24, lower, upper, without_warm_starts=True)
    with np.load(ip.population_path(1)) as saved:
        everything = saved["normalized"]
    skip = 1 + len(ip.WARM_STARTS)
    np.testing.assert_array_equal(rows, np.vstack([everything[:1], everything[skip:skip + 23]]))


def test_with_warm_starts_the_rows_are_unchanged():
    """The default still takes the first n rows."""
    lower, upper = _space()
    with np.load(ip.population_path(1)) as saved:
        first = saved["normalized"][:24]
    np.testing.assert_array_equal(ip.load(ip.population_path(1), 24, lower, upper), first)


def test_without_warm_starts_fewer_fireflies_fit():
    """N_MAX - len(WARM_STARTS) rows are left; one more is refused."""
    lower, upper = _space()
    most = ip.N_MAX - len(ip.WARM_STARTS)
    assert len(ip.load(ip.population_path(1), most, lower, upper, without_warm_starts=True)) == most
    with pytest.raises(SystemExit, match="warm starts"):
        ip.load(ip.population_path(1), most + 1, lower, upper, without_warm_starts=True)


def test_a_pilot_run_can_start_without_the_warm_starts(started):  # noqa: F811
    """The optimiser takes those rows and records that it left the warm starts out."""
    seen = started(*PILOT, "--without-warm-starts")
    lower, upper = _space()
    want = ip.load(POPULATION, 24, lower, upper, without_warm_starts=True)
    np.testing.assert_array_equal(seen["initial_population"], want)
    assert seen["engine"].run_settings["initial_population"]["without_warm_starts"] is True


def test_a_pilot_run_records_that_it_kept_them(started):  # noqa: F811
    """The default is recorded too, so every config.json says which rows the run took."""
    seen = started(*PILOT)
    assert seen["engine"].run_settings["initial_population"]["without_warm_starts"] is False


@pytest.mark.parametrize("mode", ["train", "production"])
def test_leaving_the_warm_starts_out_is_for_the_pilot_only(monkeypatch, tmp_path, mode):
    """Refused in any other mode, before a run directory is claimed.

    Production always leaves them out, the other modes keep them.
    """
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", mode, "--without-warm-starts",
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="--without-warm-starts"):
        fo.main()
    assert not (tmp_path / "r").exists()


def test_a_resume_refuses_the_flag(monkeypatch, tmp_path):
    """A resumed run keeps the rows it started with."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--resume", str(tmp_path),
                                      "--without-warm-starts"])  # fmt: skip
    with pytest.raises(SystemExit, match="--without-warm-starts"):
        fo.main()


def test_the_rows_left_out_are_the_file_s_own_warm_starts(tmp_path, monkeypatch):
    """A file made with one warm start drops one row, whatever WARM_STARTS says now."""
    lower, upper = _space()
    with np.load(ip.population_path(1)) as saved:
        rows = saved["normalized"]
    path = tmp_path / "initial_population_v9.npz"
    np.savez(path, normalized=rows, lower=lower, upper=upper, warm_starts=np.array(["a.txt"]),
             sha256=np.array(ip.digest(rows, lower, upper)))  # fmt: skip
    monkeypatch.setattr(ip, "WARM_STARTS", ("a.txt", "b.txt", "c.txt"))
    got = ip.load(path, 5, lower, upper, without_warm_starts=True)
    np.testing.assert_array_equal(got, np.vstack([rows[:1], rows[2:6]]))


def test_production_always_starts_without_the_warm_starts(started):  # noqa: F811
    """User decision 2026-11-24: production runs as the rerun pilot did, without them."""
    from tests.test_task9_production import PRODUCTION

    seen = started(*PRODUCTION)
    lower, upper = _space()
    want = ip.load(POPULATION, 48, lower, upper, without_warm_starts=True)
    np.testing.assert_array_equal(seen["initial_population"], want)
    assert seen["engine"].run_settings["initial_population"]["without_warm_starts"] is True


def _production_run(path, population):
    """A production run's directory with a checkpoint, its config.json naming its population."""
    import json

    from tests.test_task4_resume import RESUMABLE

    recorded = {"mode": "production", "pop_size": 4, "max_generations": 1000, "simulations": 300,
                "pilot": "20261201-000000_abcdef0_pilot-size", **RESUMABLE,
                "initial_population": population}  # fmt: skip
    (path / "config.json").write_text(json.dumps({"rng_seed": 1, "settings": recorded}))
    (path / "checkpoints").mkdir()
    (path / "checkpoints" / "initial.npz").write_text("x")
    lower, upper = _space()
    np.savetxt(path / "bound.txt", np.vstack((lower, upper)))
    return path


@pytest.mark.parametrize("population", [{"version": 1, "sha256": "x"},
                                        {"version": 1, "sha256": "x",
                                         "without_warm_starts": False}])  # fmt: skip
def test_a_production_run_begun_with_the_warm_starts_does_not_resume(
    monkeypatch, tmp_path, capsys, population
):
    """Code review round 6: its result would not be the calibration's own."""
    from covsyn.runtime import runs

    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--resume",
                                      str(_production_run(tmp_path, population))])  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        fo.main()
    assert stop.value.code == runs.NOT_RESUMABLE
    assert "warm starts" in capsys.readouterr().err


def test_a_production_run_begun_without_them_resumes(monkeypatch, tmp_path):
    """Recorded without the warm starts: the resume goes on."""
    from tests.test_task9_pilot_mode import _resume

    run = _production_run(tmp_path, {"version": 1, "sha256": "x", "without_warm_starts": True})
    seen = _resume(monkeypatch, run)
    assert seen["engine"].pop_size == 4


@pytest.mark.parametrize(("n", "ok"), [(254, True), (255, False), (256, False)])
def test_the_pipeline_refuses_a_production_larger_than_the_rows_left(n, ok):
    """Without the two warm starts 254 rows are left; more is refused when the file is read."""
    from covsyn.pipeline import config as pc
    from tests.test_task9_production_pipeline import PRODUCTION

    text = PRODUCTION.replace("pop_size = 48", f"pop_size = {n}")
    if ok:
        assert pc.parse(text, "t").pop_size == n
    else:
        with pytest.raises(ValueError, match="warm starts"):
            pc.parse(text, "t")


def test_the_population_module_loads_no_scipy():
    """The pipeline's configuration check reads it; only building a population needs scipy."""
    import subprocess

    code = ("import sys; import covsyn.calibration.initial_population; "
            "print('scipy' in sys.modules)")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_pipeline_checks_the_rows_from_any_directory(monkeypatch, tmp_path):
    """The population file is the repository's, wherever the pipeline is started."""
    from covsyn.pipeline import config as pc
    from tests.test_task9_production_pipeline import PRODUCTION

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="warm starts"):
        pc.parse(PRODUCTION.replace("pop_size = 48", "pop_size = 255"), "t")


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04 truncated"])
def test_a_damaged_population_file_leaves_the_refusal_to_the_optimiser(
    monkeypatch, tmp_path, content
):
    """An empty or truncated file does not crash reading the configuration."""
    from covsyn.pipeline import config as pc
    from tests.test_task9_production_pipeline import PRODUCTION

    broken = tmp_path / "initial_population_v1.npz"
    broken.write_bytes(content)
    monkeypatch.setattr(ip, "population_path", lambda version, directory=None: broken)
    assert pc.parse(PRODUCTION, "t").pop_size == 48
