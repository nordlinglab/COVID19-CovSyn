# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 8: the pipeline runs the optimiser on the step schedule its configuration gives.

Production repeats the schedule pilot 2 selects (Task 8), and covsyn-pipeline runs it (Task 4),
so [optimise] takes alpha_0, alpha_end and g_plan, each optional, checked by the optimiser's own
rules (decision of 2026-11-20). Left out, the optimiser's defaults apply; a resume moves on what
the run recorded in its config.json (tests/test_task8_step.py).
"""

from __future__ import annotations

import dataclasses

import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration import step_schedule
from covsyn.pipeline import config as pc
from tests.test_task4_pipeline import Recorder, run_pipeline


def test_the_optimiser_and_the_pipeline_share_one_schedule():
    """One rule and one set of defaults, in a module the pipeline can read without the model."""
    assert fo.step_size is step_schedule.step_size
    assert (fo.DEFAULT_ALPHA_0, fo.DEFAULT_ALPHA_END) == (0.2, 0.001)
    assert (step_schedule.DEFAULT_ALPHA_0, step_schedule.DEFAULT_ALPHA_END) == (0.2, 0.001)


def test_the_default_configuration_leaves_the_schedule_to_the_optimiser():
    """No schedule set: the optimiser's defaults until pilot 2 chooses."""
    config = pc.load("default")
    assert (config.alpha_0, config.alpha_end, config.g_plan) == (None, None, None)


def test_the_configuration_takes_a_schedule():
    """A whole number is a step size too, as TOML writes 1 for 1.0."""
    config = pc.parse("[optimise]\nalpha_0 = 0.05\nalpha_end = 0.001\ng_plan = 200\n", "t")
    assert (config.alpha_0, config.alpha_end, config.g_plan) == (0.05, 0.001, 200)
    assert pc.parse("[optimise]\nalpha_0 = 1\n", "t").alpha_0 == 1


@pytest.mark.parametrize("text", [
    "[optimise]\nalpha_0 = 0\n",
    "[optimise]\nalpha_0 = -0.1\n",
    "[optimise]\nalpha_end = 0.0\n",
    "[optimise]\ng_plan = 0\n",
    "[optimise]\ng_plan = 2.5\n",
    "[optimise]\nalpha_0 = true\n",
    "[optimise]\nalpha_0 = \"0.2\"\n",
    "[optimise]\nalpha_0 = inf\n",
    "[optimise]\nalpha_0 = nan\n",
    "[optimise]\nalpha_0 = 0.001\nalpha_end = 0.2\n",  # grows
    "[optimise]\nalpha_0 = 0.0005\n",  # below the default alpha_end
])  # fmt: skip
def test_a_schedule_the_optimiser_would_refuse_is_refused_when_read(text):
    """Before any stage starts, as a typo would otherwise run for hours."""
    with pytest.raises(ValueError, match=r"\[optimise\]"):
        pc.parse(text, "t")


def test_the_pipeline_passes_the_schedule_it_is_given(tmp_path):
    """Each setting given becomes the optimiser's flag."""
    config = dataclasses.replace(pc.load("default"), alpha_0=0.05, alpha_end=0.002, g_plan=300)
    recorder = Recorder()
    run_pipeline(tmp_path, config, recorder)
    args = recorder.calls[0]
    assert args[args.index("--alpha-0") + 1] == "0.05"
    assert args[args.index("--alpha-end") + 1] == "0.002"
    assert args[args.index("--g-plan") + 1] == "300"


def test_the_pipeline_passes_no_schedule_when_none_is_given(tmp_path):
    """The optimiser's defaults, and the mode's generations as G_plan."""
    recorder = Recorder()
    run_pipeline(tmp_path, pc.load("default"), recorder)
    assert not {"--alpha-0", "--alpha-end", "--g-plan"} & set(recorder.calls[0])


@pytest.mark.parametrize("schedule", [{}, {"alpha_0": 0.05, "alpha_end": 0.002, "g_plan": 300}])
def test_the_saved_configuration_reads_back_with_or_without_a_schedule(schedule):
    """pipeline.toml, saved with each run, leaves an unset setting out rather than writing None."""
    config = dataclasses.replace(pc.load("default"), **schedule)
    assert pc.parse(pc.to_toml(config), "saved") == config


def test_a_step_size_too_large_for_a_float_is_refused_when_read():
    """A ValueError naming the setting, not a traceback from float()."""
    with pytest.raises(ValueError, match=r"\[optimise\] alpha_0"):
        pc.parse("[optimise]\nalpha_0 = 1" + "0" * 400 + "\n", "t")


@pytest.mark.parametrize("text, unset", [("[optimise]\nalpha_0 = 0.0005\n", "alpha_end"),
                                         ("[optimise]\nalpha_end = 0.5\n", "alpha_0")])  # fmt: skip
def test_a_schedule_refused_for_its_default_end_names_that_end(text, unset):
    """The user set one end; the refusal says the other was left at its default."""
    with pytest.raises(ValueError, match=f"{unset} was left at its default"):
        pc.parse(text, "t")


def test_every_mode_starts_from_the_default_alpha_0():
    """The pipeline checks an unset alpha_0 at the default, so no mode may start elsewhere."""
    assert {mode[1] for mode in fo.OPTIMISER_MODES.values()} == {fo.DEFAULT_ALPHA_0}


def test_the_schedule_a_run_moves_on():
    """Unset ends at the defaults, G_plan at the run's generations; both ends given are kept."""
    assert step_schedule.checked_schedule(None, None, None, 120) == (0.2, 0.001, 120)
    assert step_schedule.checked_schedule(0.05, 0.002, 50, 120) == (0.05, 0.002, 50)
    assert step_schedule.checked_schedule(None, None, None, 7, alpha_0_default=0.1) == (
        0.1, 0.001, 7)  # fmt: skip
    with pytest.raises(ValueError, match="grow"):
        step_schedule.checked_schedule(0.001, 0.2, None, 10)  # both given: no default to blame
