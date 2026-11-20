# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 8, item 1: the step size follows a schedule tied to the planned run length.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 8: alpha_g = alpha_0
(alpha_end / alpha_0)^(g / (G_plan - 1)) for g < G_plan, and alpha_end after, in place of a
factor of 0.97 per generation. Pilot 2 (Task 9) chooses alpha_0, alpha_end and G_plan; until then
alpha_0 = 0.2, alpha_end = 0.001 and G_plan the mode's generations (decision of 2026-11-20).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from tests import firefly_standins as standins
from tests.test_task7_optimiser import started  # noqa: F401 (a fixture)


def test_the_schedule_runs_from_alpha_0_to_alpha_end_over_g_plan():
    """alpha_0 at generation 0, alpha_end at G_plan - 1 and after, geometric between."""
    assert fo.step_size(0, 0.2, 0.001, 5) == 0.2
    assert fo.step_size(4, 0.2, 0.001, 5) == pytest.approx(0.001)
    assert fo.step_size(9, 0.2, 0.001, 5) == 0.001
    assert fo.step_size(2, 0.2, 0.001, 5) == pytest.approx(0.2 * (0.001 / 0.2) ** 0.5)
    steps = [fo.step_size(g, 0.2, 0.001, 5) for g in range(5)]
    ratios = np.array(steps[1:]) / np.array(steps[:-1])
    np.testing.assert_allclose(ratios, ratios[0])


def test_a_one_generation_plan_starts_at_alpha_0_and_then_holds_alpha_end():
    """G_plan = 1 has no generation to shrink over."""
    assert fo.step_size(0, 0.2, 0.001, 1) == 0.2
    assert fo.step_size(1, 0.2, 0.001, 1) == 0.001


@pytest.mark.parametrize(
    "alpha_0, alpha_end, g_plan", [(0, 0.001, 5), (0.2, 0, 5), (0.2, 0.001, 0), (-1, 0.001, 5)]
)
def test_a_schedule_outside_its_range_is_refused(alpha_0, alpha_end, g_plan):
    """Step sizes are positive and the plan at least one generation long."""
    with pytest.raises(ValueError):
        fo.step_size(0, alpha_0, alpha_end, g_plan)


def test_a_run_moves_with_the_scheduled_step(tmp_path, monkeypatch):
    """Each generation's checkpoint holds the step it moved with, and config.json the schedule."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    engine = fo.Firefly(pop_size=3, alpha_0=0.2, betamin=1, gamma=0.03, seed=2,
                        alpha_end=0.001, g_plan=3)  # fmt: skip
    engine.firefly(function=standins.objective, dim=208, lb=np.zeros(208), ub=np.ones(208),
                   demographic_parameters=None, max_generations=4, max_workers=2,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None)  # fmt: skip
    for g in range(4):
        with np.load(tmp_path / fo.CHECKPOINTS / f"generation_{g:04d}.npz") as saved:
            assert float(saved["alpha"]) == fo.step_size(g, 0.2, 0.001, 3), g
    settings = json.loads((tmp_path / "config.json").read_text())["settings"]
    assert (settings["alpha_0"], settings["alpha_end"], settings["g_plan"]) == (0.2, 0.001, 3)


def test_without_a_plan_the_run_plans_its_own_generations(tmp_path, monkeypatch):
    """G_plan defaults to the generations the run is asked for."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    engine = fo.Firefly(pop_size=3, alpha_0=0.2, betamin=1, gamma=0.03, seed=2)
    engine.firefly(function=standins.objective, dim=208, lb=np.zeros(208), ub=np.ones(208),
                   demographic_parameters=None, max_generations=4, max_workers=2,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None)  # fmt: skip
    settings = json.loads((tmp_path / "config.json").read_text())["settings"]
    assert (settings["alpha_end"], settings["g_plan"]) == (fo.DEFAULT_ALPHA_END, 4)


def test_the_modes_start_from_alpha_0_of_0_2(started):  # noqa: F811
    """Until pilot 2 chooses, alpha_0 = 0.2 and alpha_end = 0.001 (decision of 2026-11-20)."""
    engine = started("--mode", "test")["engine"]
    assert (engine.alpha_0, engine.alpha_end) == (0.2, 0.001)
    assert engine.g_plan is None  # the mode's generations


def test_the_command_takes_the_schedule(started):  # noqa: F811
    """--alpha-0, --alpha-end and --g-plan override the defaults."""
    engine = started("--mode", "test", "--alpha-0", "0.05", "--alpha-end", "0.002",
                     "--g-plan", "50")["engine"]  # fmt: skip
    assert (engine.alpha_0, engine.alpha_end, engine.g_plan) == (0.05, 0.002, 50)


@pytest.mark.parametrize("flag, value", [("--alpha-0", "0"), ("--alpha-end", "-1"),
                                         ("--g-plan", "0"), ("--alpha-0", "x")])  # fmt: skip
def test_a_schedule_out_of_range_is_refused_at_startup(monkeypatch, tmp_path, flag, value):
    """Before the run directory is claimed."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test", flag, value,
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit):
        fo.main()
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("flag", ["--alpha-0", "--alpha-end", "--g-plan"])
def test_a_resumed_run_keeps_its_schedule(tmp_path, flag):
    """A resume takes the schedule from config.json; giving one is refused, not ignored."""
    import argparse

    args = argparse.Namespace(resume=tmp_path, run_dir=None, seed=None, mode=None,
                              warm_start=None, **{flag[2:].replace("-", "_"): 1})  # fmt: skip
    with pytest.raises(SystemExit, match="--resume takes"):
        fo.resume_arguments(args)


def test_the_recorded_schedule_is_read_back(tmp_path):
    """config.json's alpha_0, alpha_end and g_plan, for a resumed run to move with."""
    settings = {"alpha_0": 0.05, "alpha_end": 0.002, "g_plan": 50}
    (tmp_path / "config.json").write_text(json.dumps({"settings": settings}))
    assert fo.recorded_schedule(tmp_path) == (0.05, 0.002, 50)


def test_a_run_from_before_task_8_cannot_be_resumed(tmp_path):
    """Without a recorded schedule it moved by the factor of 0.97."""
    (tmp_path / "config.json").write_text(json.dumps({"settings": {"alpha": 1}}))
    with pytest.raises(SystemExit) as stop:
        fo.recorded_schedule(tmp_path)
    assert stop.value.code == fo.runs.NOT_RESUMABLE


def test_a_schedule_that_grows_is_refused():
    """A step that grows, as when alpha_0 and alpha_end are swapped, is a mistake.

    Task 8's steps shrink; equal ends keep one step throughout.
    """
    with pytest.raises(ValueError, match="alpha_end"):
        fo.step_size(0, 0.001, 0.2, 5)
    assert fo.step_size(3, 0.1, 0.1, 5) == 0.1


def test_the_command_refuses_a_schedule_that_grows(monkeypatch, tmp_path):
    """Before the run directory is claimed."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test", "--alpha-0", "0.001",
                                      "--alpha-end", "0.2", "--run-dir", str(tmp_path / "r")])
    with pytest.raises(SystemExit):
        fo.main()
    assert not (tmp_path / "r").exists()

