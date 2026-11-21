# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 9, pilot 2: the optimiser runs a population, length and S that the pilot sets.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 9: pilot 2 runs n in {24, 48, 96}
x alpha_0 in {0.05, 0.2}, n x G = 10,000 evaluations, G_plan = G, at the S of pilot 1, on
optimisation streams 1 and 2 (decision of 2026-11-21). The modes so far fix n, the generations
and S, so --mode pilot takes them from the command line, with alpha_0, which no mode's default
stands in for; a resumed pilot run takes them from its config.json, as a resume takes
everything else.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration import parameter_space, step_schedule
from covsyn.pipeline import config as pc
from covsyn.runtime import runs
from tests.test_task4_resume import RESUMABLE
from tests.test_task7_optimiser import POPULATION, started  # noqa: F401 (a fixture)

PILOT = ("--mode", "pilot", "--pop-size", "24", "--generations", "415", "--simulations", "3000",
         "--alpha-0", "0.05")  # fmt: skip


def _without(flag):
    """PILOT without flag and its value."""
    args = list(PILOT)
    i = args.index(flag)
    return args[:i] + args[i + 2:]


def test_a_pilot_run_takes_its_size_length_s_and_alpha_0(started):  # noqa: F811
    """G_plan defaults to the generations and alpha_end to 0.001, as in every mode."""
    seen = started(*PILOT)
    engine = seen["engine"]
    assert (engine.pop_size, seen["max_generations"], engine.simulations) == (24, 415, 3000)
    assert (engine.alpha_0, engine.alpha_end, engine.g_plan) == (0.05, 0.001, 415)
    assert engine.run_settings["mode"] == "pilot"
    with np.load(POPULATION) as saved:
        np.testing.assert_array_equal(seen["initial_population"], saved["normalized"][:24])


def test_a_pilot_run_takes_a_stream(started):  # noqa: F811
    """Each setting runs on streams 1 and 2."""
    assert started(*PILOT, "--stream", "2")["engine"].stream == 2


@pytest.mark.parametrize("flag", ["--pop-size", "--generations", "--simulations", "--alpha-0"])
def test_a_pilot_run_needs_every_setting(monkeypatch, tmp_path, flag):
    """No mode's default stands in for what the pilot sets; refused before the run starts."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", *_without(flag),
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match=flag):
        fo.main()
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("flag, value", [("--pop-size", "1"), ("--pop-size", "257"),
                                         ("--generations", "0"), ("--generations", "1001"),
                                         ("--simulations", "0"), ("--simulations", "10001")])
def test_a_pilot_setting_out_of_range_is_refused(monkeypatch, tmp_path, flag, value):
    """Each within what the run can hold.

    n within the population's 256 rows and above 1, so one firefly moves; a generation per
    block of the stream; S within one block's seeds.
    """
    args = _without(flag) + [flag, value]
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", *args, "--run-dir", str(tmp_path / "r")])
    with pytest.raises(SystemExit, match=flag):
        fo.main()


@pytest.mark.parametrize("flag, value", [("--pop-size", "24"), ("--generations", "10")])
def test_the_size_and_length_are_for_a_pilot_run_only(monkeypatch, tmp_path, flag, value):
    """Every other mode fixes them; given anyway, they are refused rather than ignored."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test", flag, value,
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match=flag):
        fo.main()


def test_s_is_for_a_pilot_or_a_profile_only(monkeypatch, tmp_path):
    """S changes the objective; train and test keep the plan's 300."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test", "--simulations",
                                      "1000", "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="--simulations"):
        fo.main()


def test_the_pilot_mode_has_no_default_alpha_0_or_length():
    """Its alpha_0 and generations are the pilot's; the tables hold None for them."""
    assert step_schedule.MODE_ALPHA_0["pilot"] is None
    assert step_schedule.MODE_GENERATIONS["pilot"] is None


def _pilot_run(path, settings=None):
    """A pilot run's directory with a checkpoint to continue, as config.json records it."""
    recorded = {"mode": "pilot", "pop_size": 4, "max_generations": 3, "simulations": 300,
                **RESUMABLE, "alpha_0": 0.05, "g_plan": 3, **(settings or {})}  # fmt: skip
    (path / "config.json").write_text(json.dumps({"rng_seed": 1, "settings": recorded}))
    (path / "checkpoints").mkdir()
    (path / "checkpoints" / "initial.npz").write_text("x")
    space = parameter_space.load()
    np.savetxt(path / "bound.txt", np.vstack((space.lower, space.upper)))
    return path


class _Started(Exception):
    pass


def _resume(monkeypatch, path, *extra):
    seen = {}

    def firefly(self, **kwargs):
        seen.update(kwargs, engine=self)
        raise _Started

    monkeypatch.setattr(fo.Firefly, "firefly", firefly)
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--resume", str(path), *extra])
    with pytest.raises(_Started):
        fo.main()
    return seen


def test_a_resumed_pilot_run_takes_its_size_length_and_s_from_its_config(monkeypatch, tmp_path):
    """The run continues as it started."""
    seen = _resume(monkeypatch, _pilot_run(tmp_path))
    engine = seen["engine"]
    assert (engine.pop_size, seen["max_generations"], engine.simulations) == (4, 3, 300)
    assert (engine.alpha_0, engine.g_plan) == (0.05, 3)


@pytest.mark.parametrize("flag, value", [("--pop-size", "4"), ("--generations", "3")])
def test_a_resumed_pilot_run_refuses_a_new_size_or_length(monkeypatch, tmp_path, flag, value):
    """Given with --resume, they are refused, as every other setting is."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--resume", str(_pilot_run(tmp_path)),
                                      flag, value])  # fmt: skip
    with pytest.raises(SystemExit, match="--resume takes"):
        fo.main()


@pytest.mark.parametrize("key, value", [("pop_size", None), ("pop_size", 1),
                                        ("max_generations", "3"), ("simulations", 0),
                                        ("pop_size", True)])
def test_a_pilot_run_with_a_recorded_size_out_of_range_cannot_be_resumed(monkeypatch, tmp_path,
                                                                         capsys, key, value):
    """Retrying cannot fix config.json, so NOT_RESUMABLE, naming the setting."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--resume",
                                      str(_pilot_run(tmp_path, {key: value}))])  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        fo.main()
    assert stop.value.code == runs.NOT_RESUMABLE
    assert key in capsys.readouterr().err


def test_the_pipeline_does_not_run_the_pilot_mode():
    """covsyn-pilot-size runs the pilot's runs; the pipeline cannot give their size."""
    with pytest.raises(ValueError, match="pilot"):
        pc.parse('[optimise]\nmode = "pilot"\n', "t")


def test_a_size_out_of_range_names_the_mode():
    """The refusal says which mode it is for."""
    from covsyn.calibration import run_size

    with pytest.raises(ValueError, match="--mode production"):
        run_size.from_command("production", 300, None, 3000)


def test_production_takes_no_generations_to_check():
    """Its G is G_max, not a flag; only n and S are the user's."""
    from covsyn.calibration import run_size

    assert run_size.from_command("production", 48, None, 3000)[::2] == (48, 3000)


def test_the_row_count_is_the_population_s():
    """run_size holds it without importing the population module, so the pipeline loads fast."""
    from covsyn.calibration import initial_population, run_size

    assert run_size.POP_SIZES.stop - 1 == initial_population.N_MAX


def test_reading_the_pipeline_configuration_loads_no_scipy():
    """The configuration is checked before any stage; scipy took 5 s of it (Task 9 review)."""
    import subprocess

    done = subprocess.run([sys.executable, "-c", "import sys, covsyn.pipeline.config; "
                           "print('scipy' in sys.modules)"], capture_output=True, text=True,
                          check=True)  # fmt: skip
    assert done.stdout.strip() == "False"


def test_a_run_whose_config_an_editor_saved_with_a_byte_order_mark_resumes(monkeypatch,
                                                                           tmp_path):
    """The optimiser reads its config.json past a BOM, as the pilots read theirs."""
    run = _pilot_run(tmp_path)
    config = run / "config.json"
    config.write_bytes(b"\xef\xbb\xbf" + config.read_bytes())
    seen = _resume(monkeypatch, run)
    assert seen["engine"].pop_size == 4


def test_every_size_refusal_says_a_whole_number_from_its_range():
    """One wording, from the command line or config.json."""
    from covsyn.calibration import run_size

    with pytest.raises(ValueError, match="--pop-size must be a whole number from 2 to 256"):
        run_size.from_command("pilot", 300, 10, 3000)
    with pytest.raises(ValueError, match="pop_size is 2.5, not a whole number from 2 to 256"):
        run_size.from_settings({"pop_size": 2.5, "max_generations": 3, "simulations": 300})


def test_the_profile_s_refusal_has_the_same_wording(monkeypatch, tmp_path):
    """--mode profile reads S by run_size's rule too."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "profile", "--simulations",
                                      "20000", "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="a whole number from 1 to 10000"):
        fo.main()
