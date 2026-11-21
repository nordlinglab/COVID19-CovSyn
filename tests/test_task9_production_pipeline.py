# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 9: the pipeline runs production on pilot 2's choice.

The production run is a pipeline run (Task 4) whose optimiser takes pilot 2's n, S, step
schedule and run id (decision of 2026-11-21): [optimise] takes pop_size, simulations and pilot,
which mode = "production" requires with alpha_0, alpha_end and g_plan, and every other mode
refuses, as the optimiser does.
"""

from __future__ import annotations

import dataclasses

import pytest

from covsyn.pipeline import config as pc
from tests.test_task4_pipeline import Recorder, run_pipeline

PRODUCTION = """[optimise]
mode = "production"
pop_size = 48
simulations = 3000
alpha_0 = 0.2
alpha_end = 0.001
g_plan = 207
pilot = "20261201-000000_abcdef0_pilot-size"
"""


def test_the_configuration_takes_pilot_2_s_choice():
    """Read with the schedule, the size and the pilot's id."""
    config = pc.parse(PRODUCTION, "t")
    assert (config.mode, config.pop_size, config.simulations) == ("production", 48, 3000)
    assert (config.alpha_0, config.alpha_end, config.g_plan) == (0.2, 0.001, 207)
    assert config.pilot == "20261201-000000_abcdef0_pilot-size"


@pytest.mark.parametrize("key", ["pop_size", "simulations", "pilot", "alpha_0", "alpha_end",
                                 "g_plan"])
def test_production_needs_every_setting_of_pilot_2_s_choice(key):
    """No default stands in for it; refused when read, before any stage."""
    text = "\n".join(line for line in PRODUCTION.splitlines() if not line.startswith(key))
    with pytest.raises(ValueError, match=key):
        pc.parse(text + "\n", "t")


@pytest.mark.parametrize("line", ["pop_size = 48", "simulations = 3000", 'pilot = "x"'])
def test_the_other_modes_refuse_production_s_settings(line):
    """Their optimiser fixes n and S and has no pilot to name."""
    with pytest.raises(ValueError, match=line.split()[0]):
        pc.parse(f'[optimise]\nmode = "train"\n{line}\n', "t")


@pytest.mark.parametrize("line", ["pop_size = 1", "pop_size = 257", "simulations = 0",
                                  "simulations = 10001"])
def test_a_size_out_of_range_is_refused_when_read(line):
    """The optimiser's own ranges: 2 to 256 fireflies, 1 to 10,000 simulations."""
    text = PRODUCTION.replace(f"{line.split()[0]} = ", "#").rstrip() + f"\n{line}\n"
    with pytest.raises(ValueError, match=line.split()[0]):
        pc.parse(text, "t")


def test_the_pipeline_passes_production_s_settings_to_the_optimiser(tmp_path):
    """Each as the optimiser's flag of the same name."""
    recorder = Recorder()
    run_pipeline(tmp_path, pc.parse(PRODUCTION, "t"), recorder)
    args = [str(a) for a in recorder.calls[0]]
    flags = {flag: args[args.index(flag) + 1] for flag in
             ("--mode", "--pop-size", "--simulations", "--pilot", "--g-plan", "--alpha-0",
              "--alpha-end")}  # fmt: skip
    assert flags == {"--mode": "production", "--pop-size": "48", "--simulations": "3000",
                     "--pilot": "20261201-000000_abcdef0_pilot-size", "--g-plan": "207",
                     "--alpha-0": "0.2", "--alpha-end": "0.001"}  # fmt: skip


def test_a_train_run_passes_no_size():
    """As before Task 9."""
    config = pc.load("default")
    assert (config.pop_size, config.simulations, config.pilot) == (None, None, None)


def test_the_saved_production_configuration_reads_back():
    """pipeline.toml, saved with the run, is what a resume reads."""
    config = pc.parse(PRODUCTION, "t")
    assert pc.parse(pc.to_toml(config), "saved") == config
    assert dataclasses.replace(config) == config
