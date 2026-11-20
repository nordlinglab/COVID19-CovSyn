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


def test_each_mode_s_schedule_is_in_one_table():
    """alpha_0 and generations per mode in step_schedule, which main() and the pipeline read.

    The optimiser's modes keep no copy that could differ.
    """
    assert set(step_schedule.MODE_ALPHA_0) == set(step_schedule.MODE_GENERATIONS)
    assert set(step_schedule.MODE_GENERATIONS) == set(fo.OPTIMISER_MODES)
    assert all(len(v) == 3 for v in fo.OPTIMISER_MODES.values())  # pop_size, betamin, gamma


def test_the_pipeline_checks_an_unset_alpha_0_at_the_mode_s(monkeypatch):
    """A mode that starts lower refuses an alpha_end above its own alpha_0."""
    monkeypatch.setitem(step_schedule.MODE_ALPHA_0, "test", 0.05)
    with pytest.raises(ValueError, match="alpha_0 was left at its default"):
        pc.parse('[optimise]\nmode = "test"\nalpha_end = 0.1\n', "t")
    assert pc.parse('[optimise]\nmode = "train"\nalpha_end = 0.1\n', "t").alpha_end == 0.1


@pytest.mark.parametrize("text", ["[optimise]\nalpha_0 = -1\n", "[optimise]\nalpha_0 = 0\n"])
def test_a_step_size_not_positive_is_not_blamed_on_the_other_end(text):
    """Setting alpha_end cannot fix alpha_0 <= 0, so the refusal does not suggest it."""
    with pytest.raises(ValueError) as refused:
        pc.parse(text, "t")
    assert "positive" in str(refused.value) and "left at its default" not in str(refused.value)


@pytest.mark.parametrize("text, said", [
    ("[optimise]\nalpha_0 = 0.0005\n", "give alpha_end too, at most alpha_0"),
    ("[optimise]\nalpha_end = 0.5\n", "give alpha_0 too, at least alpha_end"),
])  # fmt: skip
def test_the_refusal_says_which_values_would_do(text, said):
    """Not only which end to give, but the range it must be in."""
    with pytest.raises(ValueError, match=said):
        pc.parse(text, "t")


def test_an_integer_too_long_to_read_is_refused_naming_the_file():
    """Over Python's 4300-digit limit tomllib itself refuses; the file is still named."""
    with pytest.raises(ValueError, match="^t: "):
        pc.parse("[optimise]\nalpha_0 = 1" + "0" * 5000 + "\n", "t")


@pytest.mark.parametrize("g_plan", [1000, 1001, 10**400])
def test_a_plan_of_any_length_is_taken(g_plan):
    """G_plan uses no seed blocks, so a stream's 1000 do not bound it (decision of 2026-11-20)."""
    assert pc.parse(f"[optimise]\ng_plan = {g_plan}\n", "t").g_plan == g_plan


def test_the_schedule_a_run_moves_on():
    """Unset ends at the defaults, G_plan at the run's generations; both ends given are kept."""
    assert step_schedule.checked_schedule(None, None, None, 120) == (0.2, 0.001, 120)
    assert step_schedule.checked_schedule(0.05, 0.002, 50, 120) == (0.05, 0.002, 50)
    assert step_schedule.checked_schedule(None, None, None, 7, alpha_0_default=0.1) == (
        0.1, 0.001, 7)  # fmt: skip
    with pytest.raises(ValueError, match="grow"):
        step_schedule.checked_schedule(0.001, 0.2, None, 10)  # both given: no default to blame


def test_a_schedule_for_an_unknown_mode_is_refused_when_read():
    """Its alpha_0 is unknown, so the schedule cannot be checked; nor would the run start."""
    with pytest.raises(ValueError, match="mode"):
        pc.parse('[optimise]\nmode = "trian"\nalpha_end = 0.1\n', "t")


def test_every_refusal_names_the_caller_s_settings():
    """Not step_size's own names, whichever check refuses."""
    names = ("--alpha-0", "--alpha-end", "--g-plan")
    for given, said in [((-1.0, None, None), "--alpha-0 and --alpha-end must be positive"),
                        ((None, None, 2.5), "--g-plan must be a whole number")]:  # fmt: skip
        with pytest.raises(ValueError, match=said):
            step_schedule.checked_schedule(*given, 10, names=names)


def test_an_unknown_mode_is_refused_when_read_with_or_without_a_schedule():
    """The optimiser would refuse it, so the pipeline does before any stage starts."""
    with pytest.raises(ValueError, match="mode"):
        pc.parse('[optimise]\nmode = "trian"\n', "t")


def test_a_plan_longer_than_the_mode_s_run_is_warned_about_in_the_log(tmp_path):
    """A typed extra zero (1200 for 120) is in pipeline.log, before the optimiser starts."""
    config = dataclasses.replace(pc.load("default"), g_plan=1200)
    _, run_dir = run_pipeline(tmp_path, config, Recorder())
    log = (run_dir / "pipeline.log").read_text()
    said = "WARNING: [optimise] g_plan 1200 is longer than the run's 120 generations"
    assert said in log and log.index(said) < log.index("stage optimise")


def test_a_plan_as_long_as_the_mode_s_run_is_not_warned_about(tmp_path):
    """The plan's own case: G_plan = G (Task 9, pilot 2)."""
    config = dataclasses.replace(pc.load("default"), g_plan=120)
    _, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert "WARNING: [optimise]" not in (run_dir / "pipeline.log").read_text()


def test_a_resumed_pipeline_does_not_warn_of_its_plan_again(tmp_path):
    """The new run warned once; the log it appends to on resume keeps that one line."""
    config = dataclasses.replace(pc.load("default"), g_plan=1200)
    run_pipeline(tmp_path, config, Recorder(fail={"covsyn.calibration.firefly_optimizer"}))
    _, run_dir = run_pipeline(tmp_path, None, Recorder(), resume=True)
    log = (run_dir / "pipeline.log").read_text()
    assert log.count("run directory") == 2  # the resume logged too
    assert log.count("WARNING: [optimise] g_plan") == 1


def test_a_built_configuration_with_an_unknown_mode_gives_no_schedule_warning():
    """One made without parse(): the optimiser refuses the mode; the warning does not crash."""
    config = dataclasses.replace(pc.load("default"), mode="trian", g_plan=1200)
    assert pc.schedule_warnings(config) == []


def test_reading_a_long_plan_raises_no_warning(recwarn):
    """The log says it, once per run; parse() stays silent, as a resume reads it again."""
    pc.parse("[optimise]\ng_plan = 1200\n", "t")
    assert not [w for w in recwarn if "g_plan" in str(w.message)]
