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
    assert engine.g_plan == fo.step_schedule.MODE_GENERATIONS["test"]  # the mode's generations


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
    with pytest.raises(SystemExit, match="step schedule"):
        fo.main()
    assert not (tmp_path / "r").exists()


def test_a_small_alpha_0_alone_is_told_to_give_alpha_end(monkeypatch, tmp_path):
    """Below the default alpha_end, the refusal names the flag that fixes it."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test", "--alpha-0", "0.0005",
                                      "--run-dir", str(tmp_path / "r")])
    with pytest.raises(SystemExit, match="--alpha-end"):
        fo.main()


def _config_fails_to_read(monkeypatch, run_dir):
    """Make reading run_dir/config.json, and only it, fail as a flaky disk does."""
    read_text = fo.Path.read_text

    def failing(self, *args, **kwargs):
        if self == run_dir / "config.json":
            raise OSError(5, "Input/output error")
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(fo.Path, "read_text", failing)


@pytest.mark.parametrize("reader", ["recorded_schedule", "recorded_stream"])
def test_a_config_that_cannot_be_read_now_may_be_retried(tmp_path, monkeypatch, reader):
    """A read error that may pass is not NOT_RESUMABLE, as in resumed_settings."""
    (tmp_path / "config.json").write_text("{}")
    _config_fails_to_read(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as stop:
        getattr(fo, reader)(tmp_path)
    assert stop.value.code != fo.runs.NOT_RESUMABLE and "could not read" in str(stop.value)


def test_a_recorded_schedule_out_of_range_says_so(tmp_path, capsys):
    """Not that the run predates Task 8, which it does not."""
    settings = {"alpha_0": 0.001, "alpha_end": 0.2, "g_plan": 50}
    (tmp_path / "config.json").write_text(json.dumps({"settings": settings}))
    with pytest.raises(SystemExit) as stop:
        fo.recorded_schedule(tmp_path)
    assert stop.value.code == fo.runs.NOT_RESUMABLE
    said = capsys.readouterr().err
    assert "out of range" in said and "before Task 8" not in said
    assert "alpha_end 0.2 is above alpha_0 0.001" in said  # config.json's keys


def test_a_recorded_plan_out_of_range_is_named_by_its_key(tmp_path, capsys):
    """config.json records g_plan, so the refusal says g_plan, not G_plan."""
    settings = {"alpha_0": 0.2, "alpha_end": 0.001, "g_plan": 0}
    (tmp_path / "config.json").write_text(json.dumps({"settings": settings}))
    with pytest.raises(SystemExit):
        fo.recorded_schedule(tmp_path)
    assert "g_plan must be a whole number" in capsys.readouterr().err


def _engine(**schedule):
    return fo.Firefly(pop_size=3, betamin=1, gamma=0.03, seed=2, **schedule)


def _optimise(engine, generations, resume=False):
    engine.firefly(function=standins.objective, dim=208, lb=np.zeros(208), ub=np.ones(208),
                   demographic_parameters=None, max_generations=generations, max_workers=2,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None,
                   resume=resume)  # fmt: skip


def test_a_resume_with_another_schedule_is_refused(tmp_path, monkeypatch):
    """Through the engine too: config.json's schedule describes the run, so it must hold."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    _optimise(_engine(alpha_0=0.2, g_plan=4), 2)
    with pytest.raises(SystemExit) as stop:
        _optimise(_engine(alpha_0=0.05, g_plan=4), 4, resume=True)
    assert stop.value.code == fo.runs.NOT_RESUMABLE


def test_a_resume_without_a_plan_keeps_the_recorded_one(tmp_path, monkeypatch):
    """A run extended to more generations goes on shrinking over the plan it started with."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    _optimise(_engine(alpha_0=0.2), 2)  # G_plan = 2, recorded
    _optimise(_engine(alpha_0=0.2), 4, resume=True)
    for g in range(4):
        with np.load(tmp_path / fo.CHECKPOINTS / f"generation_{g:04d}.npz") as saved:
            assert float(saved["alpha"]) == fo.step_size(g, 0.2, fo.DEFAULT_ALPHA_END, 2), g


def test_an_alpha_end_alone_above_the_default_alpha_0_is_told_to_give_alpha_0(monkeypatch,
                                                                              tmp_path):
    """The mirror of a small --alpha-0 alone: the refusal names --alpha-0."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test", "--alpha-end", "0.5",
                                      "--run-dir", str(tmp_path / "r")])
    with pytest.raises(SystemExit, match="--alpha-0 was left at its default") as stop:
        fo.main()
    assert "--alpha-end 0.5 is above --alpha-0 0.2" in str(stop.value)  # the flags' names


def test_a_plan_longer_than_the_run_is_run_with_a_warning(tmp_path, monkeypatch):
    """G_plan uses no seed blocks, so it has no cap; a run that ends first is told so.

    Decision of 2026-11-20: a stream's 1000 blocks bound the generations run, not G_plan; a
    typed G_plan of 1200 for 120 is only warned about, as the steps barely shrink.
    """
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    with pytest.warns(UserWarning, match="G_plan 1200 is longer than the run's 2 generations"):
        _optimise(_engine(alpha_0=0.2, g_plan=1200), 2)
    settings = json.loads((tmp_path / "config.json").read_text())["settings"]
    assert settings["g_plan"] == 1200


def test_the_warning_gives_alpha_end_and_the_last_step(tmp_path, monkeypatch):
    """Where the steps end, beside where the plan would have taken them."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    last = fo.step_size(1, 0.2, 0.001, 1200)
    with pytest.warns(UserWarning) as caught:
        _optimise(_engine(alpha_0=0.2, g_plan=1200), 2)
    said = [str(w.message) for w in caught if "G_plan" in str(w.message)]
    assert said == [fo.step_schedule.plan_outlasts_run(1200, 2)
                    + f" (alpha_end 0.001, last step {last:.3g})"]  # fmt: skip


def test_a_resumed_run_does_not_warn_of_its_plan_again(tmp_path, monkeypatch, recwarn):
    """Its start warned that the plan outlasts the run; the resume, by the same rule, does not."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    with pytest.warns(UserWarning, match="G_plan 1200"):
        _optimise(_engine(alpha_0=0.2, g_plan=1200), 2)
    recwarn.clear()
    _optimise(_engine(alpha_0=0.2, g_plan=1200), 3, resume=True)
    said = fo.step_schedule.plan_outlasts_run(1200, 3)
    assert not [w for w in recwarn if said in str(w.message)]


def _record_generations(path, generations):
    """config.json as if the run had started with ``generations``."""
    config = json.loads((path / "config.json").read_text())
    config["settings"]["max_generations"] = generations
    (path / "config.json").write_text(json.dumps(config))


def test_each_resume_shorter_than_its_plan_warns(tmp_path, monkeypatch):
    """Started as long as its plan, resumed shorter: the steps end above alpha_end.

    config.json keeps the generations the run started with, so each such resume warns.
    """
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    _optimise(_engine(alpha_0=0.2, g_plan=4), 1)
    _record_generations(tmp_path, 4)
    for _ in range(2):
        with pytest.warns(UserWarning, match="G_plan 4 is longer than the run's 3 generations"):
            _optimise(_engine(alpha_0=0.2, g_plan=4), 3, resume=True)
        (tmp_path / fo.CHECKPOINTS / "generation_0002.npz").unlink()  # generation 2 to come


def test_a_resume_with_nothing_left_to_run_gives_no_warning(tmp_path, monkeypatch, recwarn):
    """Its steps were all taken, down to alpha_end; none is left to end above it."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    _optimise(_engine(alpha_0=0.2, g_plan=4), 4)
    _optimise(_engine(alpha_0=0.2, g_plan=4), 2, resume=True)
    said = fo.step_schedule.plan_outlasts_run(4, 2)
    assert not [w for w in recwarn if said in str(w.message)]


@pytest.mark.parametrize("recorded", [None, "4", True])
def test_a_recorded_generations_count_not_a_number_still_warns(tmp_path, monkeypatch, recorded):
    """Only a warning hangs on it, so it is read as unknown rather than stopping the resume."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    _optimise(_engine(alpha_0=0.2, g_plan=4), 1)
    _record_generations(tmp_path, recorded)
    with pytest.warns(UserWarning, match="G_plan 4 is longer than the run's 3 generations"):
        _optimise(_engine(alpha_0=0.2, g_plan=4), 3, resume=True)


def test_a_plan_as_long_as_the_run_is_not_warned_about(tmp_path, monkeypatch, recwarn):
    """The plan's own case: G_plan = G (Task 9, pilot 2)."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    _optimise(_engine(alpha_0=0.2, g_plan=2), 2)
    assert not [w for w in recwarn if "G_plan" in str(w.message)]


def test_a_run_longer_than_a_stream_is_refused_before_it_starts(tmp_path, monkeypatch,
                                                                 capsys):
    """Generation g takes block g, so a stream's blocks bound max_generations; nothing is run."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    monkeypatch.setattr(fo.seed_streams, "MAX_BLOCKS", 3)
    _optimise(_engine(alpha_0=0.2), 3)  # blocks 0 to 2: as many as the stream holds
    with pytest.raises(SystemExit) as stop:  # retrying cannot help
        _optimise(_engine(alpha_0=0.2), 4, resume=True)
    assert stop.value.code == fo.runs.NOT_RESUMABLE
    assert "cannot be resumed: max_generations 4" in capsys.readouterr().err  # as given
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.setattr(fo, "result_path", fresh, raising=False)
    with pytest.raises(ValueError, match="max_generations 4 is more than a stream's 3 blocks"):
        _optimise(_engine(alpha_0=0.2), 4)
    assert not (fresh / "config.json").exists()


def test_the_command_starts_from_the_mode_s_alpha_0_the_pipeline_checks(started, monkeypatch):  # noqa: F811
    """main() reads step_schedule.MODE_ALPHA_0, the table the pipeline checks against."""
    monkeypatch.setitem(fo.step_schedule.MODE_ALPHA_0, "test", 0.05)
    assert started("--mode", "test")["engine"].alpha_0 == 0.05


def test_the_command_refuses_a_mode_longer_than_a_stream_before_claiming(monkeypatch, tmp_path):
    """Before the run directory is claimed, so --run-dir stays free for the next try."""
    monkeypatch.setitem(fo.step_schedule.MODE_GENERATIONS, "test", 4)
    monkeypatch.setattr(fo.seed_streams, "MAX_BLOCKS", 3)
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test",
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="max_generations 4 is more than a stream's 3 blocks"):
        fo.main()
    assert not (tmp_path / "r").exists()


def test_the_command_runs_the_mode_s_generations_from_one_table(started, monkeypatch):  # noqa: F811
    """step_schedule.MODE_GENERATIONS, which the pipeline compares g_plan with."""
    monkeypatch.setitem(fo.step_schedule.MODE_GENERATIONS, "test", 4)
    seen = started("--mode", "test")
    assert seen["max_generations"] == 4 and seen["engine"].g_plan == 4


def test_the_help_gives_the_mode_s_alpha_0_as_the_default(monkeypatch, capsys):
    """The default is per mode (MODE_ALPHA_0), so --help does not print one number."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--help"])
    with pytest.raises(SystemExit):
        fo.main()
    assert "default: the mode's" in " ".join(capsys.readouterr().out.split()).split("--alpha-0")[2]


def test_the_rule_for_a_plan_that_outlasts_its_run():
    """One rule, which the optimiser and the pipeline both warn by; no generation, no warning."""
    rule = fo.step_schedule.plan_outlasts_run
    assert rule(120, 120) is None and rule(50, 120) is None and rule(10, 0) is None
    assert rule(1200, 120, "g_plan") == ("g_plan 1200 is longer than the run's 120 "
                                         "generations, so the steps do not reach alpha_end")


def test_a_resume_of_a_mode_longer_than_a_stream_cannot_be_resumed(monkeypatch, tmp_path,
                                                                   capsys):
    """Through the command: NOT_RESUMABLE, so the pipeline does not retry, blaming the mode."""
    from tests.test_task4_resume import RESUMABLE

    settings = {"mode": "test", **RESUMABLE}
    (tmp_path / "config.json").write_text(json.dumps({"rng_seed": 1, "settings": settings}))
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "initial.npz").write_text("x")  # something to continue
    monkeypatch.setattr(fo.seed_streams, "MAX_BLOCKS", 3)
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--resume", str(tmp_path)])
    def snapshot():  # directories too, as None
        return {f: f.read_bytes() if f.is_file() else None for f in tmp_path.rglob("*")}

    before = snapshot()
    with pytest.raises(SystemExit) as stop:
        fo.main()
    assert stop.value.code == fo.runs.NOT_RESUMABLE
    assert "cannot be resumed: the mode's max_generations 10" in capsys.readouterr().err
    assert snapshot() == before  # untouched
