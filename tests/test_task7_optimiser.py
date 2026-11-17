# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 7, items 2 and 3: every run starts from the committed initial population.

A run with n fireflies takes the first n rows of variable/initial_population_v1.npz and
refuses to start when its bounds or dimension differ from those stored. The population holds
the warm starts, so --warm_start and the pipeline's warm_start setting are gone (decision of
2026-11-17): another warm start is another population version.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration import initial_population as ip
from covsyn.calibration import parameter_space
from covsyn.pipeline import config as pc
from covsyn.runtime import runs
from tests import firefly_standins as standins
from tests.conftest import REPO_ROOT

POPULATION = REPO_ROOT / "variable" / "initial_population_v1.npz"


class _Started(Exception):
    """Raised where the optimiser would start evaluating."""


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Run main() up to the optimiser's start; returns what firefly() was given."""
    seen = {}

    def firefly(self, **kwargs):
        seen.update(kwargs, engine=self)
        raise _Started

    monkeypatch.setattr(fo.Firefly, "firefly", firefly)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--run-dir", str(tmp_path / "r"),
                                          *argv])  # fmt: skip
        with pytest.raises(_Started):
            fo.main()
        return seen

    return run


def test_a_run_starts_from_the_first_rows_of_the_population(started):
    """--mode test has 3 fireflies: rows 0 to 2, the literature vector and the warm starts."""
    seen = started("--mode", "test")
    with np.load(POPULATION) as saved:
        np.testing.assert_array_equal(seen["initial_population"], saved["normalized"][:3])
        sha = str(saved["sha256"])
    assert seen["engine"].run_settings["initial_population"] == {"version": 1, "sha256": sha}


def test_the_warm_start_option_is_refused(monkeypatch, tmp_path):
    """The warm starts are rows of the population; another one is a new population version."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--warm_start", "x.txt",
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        fo.main()
    assert "initial population" in str(stop.value.code)  # said on stderr, status 1
    assert not (tmp_path / "r").exists()


def test_other_bounds_stop_the_run_before_it_is_claimed(monkeypatch, tmp_path):
    """Item 3: refused, not clipped, and before the run directory is claimed."""
    space = parameter_space.load()
    upper = space.upper.copy()
    upper[0] += 1.0
    monkeypatch.setattr(fo.parameter_space, "load",
                        lambda: parameter_space.ParameterSpace(space.lower, upper,
                                                               space.literature))  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test",
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="bounds"):
        fo.main()
    assert not (tmp_path / "r").exists()


def test_firefly_starts_from_the_rows_it_is_given(tmp_path, monkeypatch):
    """The initial checkpoint holds the given rows, scaled to the bounds."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    lb, ub = np.full(208, 0.5), np.full(208, 2.5)
    rows = ip.load(POPULATION, 4, *_bounds())
    engine = fo.Firefly(pop_size=4, alpha=1, betamin=1, gamma=0.03, seed=3)
    engine.firefly(function=standins.objective, dim=208, lb=lb, ub=ub,
                   demographic_parameters=None, max_generations=1, max_workers=2,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None,
                   initial_population=rows)  # fmt: skip
    with np.load(runs.last_checkpoint(tmp_path).parent / "initial.npz") as initial:
        np.testing.assert_array_equal(initial["normalized"], rows)
        np.testing.assert_array_equal(initial["fireflies"], rows * (ub - lb) + lb)


def _bounds():
    """The bounds the population was made for."""
    space = parameter_space.load()
    return space.lower, space.upper


def test_the_pipeline_no_longer_takes_a_warm_start():
    """Its default no longer passes one, and a file that sets one is told why it is refused."""
    assert "warm_start" not in {f.name for f in pc.dataclasses.fields(pc.PipelineConfig)}
    with pytest.raises(ValueError, match="initial population"):
        pc.parse('[optimise]\nwarm_start = ["a.txt"]\n', "test.toml")
    lines = (pc.PACKAGED / "default.toml").read_text().splitlines()
    assert not [line for line in lines if line.strip().startswith("warm_start")]


# -- item 4: the move generator's seed -------------------------------------------------------


def test_the_move_seed_is_recorded_as_rng_seed(tmp_path, monkeypatch):
    """config.json holds rng_seed, the seed of the moves (the population is the file's)."""
    import json

    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    engine = fo.Firefly(pop_size=2, alpha=1, betamin=1, gamma=0.03, seed=9)
    engine.firefly(function=standins.objective, dim=208, lb=np.zeros(208), ub=np.ones(208),
                   demographic_parameters=None, max_generations=1, max_workers=1,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None)  # fmt: skip
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["rng_seed"] == 9 and "seed" not in config


def test_without_a_seed_the_moves_use_the_fixed_default():
    """Two runs of one configuration make the same moves without naming a seed (decision)."""
    assert fo.Firefly(pop_size=2, alpha=1, betamin=1, gamma=0.03).seed == fo.DEFAULT_RNG_SEED
    assert fo.DEFAULT_RNG_SEED == 1


def test_the_command_records_the_default_seed(started):
    """No --seed: the run's rng_seed is the default, not a drawn one."""
    assert started("--mode", "test")["engine"].seed == fo.DEFAULT_RNG_SEED


@pytest.mark.parametrize("config", [
    {"seed": 5, "settings": {"mode": "train", "moves": "synchronous"}},
    {"rng_seed": 5, "settings": {"mode": "train", "moves": "synchronous"}},
])  # fmt: skip
def test_a_run_from_before_task_7_cannot_be_resumed(tmp_path, config):
    """Without rng_seed and the population it started from, it began from another population."""
    import json

    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit) as stop:
        fo.resumed_settings(tmp_path)
    assert stop.value.code == runs.NOT_RESUMABLE
