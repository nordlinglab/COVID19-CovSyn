# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 7, item 4: every use takes its seeds from its own stream.

Optimisation scores generation g on block g of its stream (1-99, default 1), the initial
population and generation 0 on block 0; revalidation scores block b of its validation stream
(200-299, default 200); the pipeline's synthesis chooses the candidate on stream 301 and
reports on stream 300 (decision of 2026-11-17: the report keeps E97's independent seeds). The
constants VALIDATION_SEED_START and --seed_start are replaced.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration import revalidation as rv
from covsyn.calibration import seed_streams as ss
from covsyn.model import data_synthesis_main as dsm
from covsyn.pipeline import config as pc
from tests import firefly_standins as standins
from tests.test_task7_optimiser import started  # noqa: F401 (a fixture)

S = fo.SIMULATIONS_PER_EVALUATION


def test_a_generation_is_scored_on_its_block_of_its_stream():
    """Generation g: block g of the run's stream; the default stream is 1."""
    assert fo.generation_seed_offset(0) == ss.block_start(1, 0, S) == 10**7
    assert fo.generation_seed_offset(5, S, 7) == ss.block_start(7, 5, S)
    assert fo.generation_seed_offset(3, 3000) == ss.block_start(1, 3, 3000)


def test_a_run_scores_on_its_stream_and_records_it(tmp_path, monkeypatch):
    """Every candidate, the initial population's too, on the run's stream; config.json says so."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    engine = fo.Firefly(pop_size=3, alpha=1, betamin=1, gamma=0.03, seed=2)
    engine.stream = 4
    engine.firefly(function=standins.recording, dim=208, lb=np.zeros(208), ub=np.ones(208),
                   demographic_parameters=None, max_generations=2, max_workers=2,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None)  # fmt: skip
    for name, block in (("initial", 0), ("generation_0000", 0), ("generation_0001", 1)):
        with np.load(tmp_path / fo.CHECKPOINTS / f"{name}.npz") as saved:
            offsets = {p["scored_offset"] for p in json.loads(str(saved["cost_parts"]))}
        assert offsets == {ss.block_start(4, block, S)}, name
    assert json.loads((tmp_path / "config.json").read_text())["settings"]["stream"] == 4


def test_the_command_takes_the_stream(started):  # noqa: F811
    """--stream sets it; without it, stream 1."""
    assert started("--mode", "test")["engine"].stream == ss.DEFAULT_OPTIMISATION_STREAM


@pytest.mark.parametrize("stream", ["0", "100", "200", "x"])
def test_only_an_optimisation_stream_is_taken(monkeypatch, tmp_path, stream):
    """1 to 99: the others are reserved for monitoring, validation and reporting."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--stream", stream,
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit):
        fo.main()
    assert not (tmp_path / "r").exists()


def test_revalidation_scores_blocks_of_its_validation_stream():
    """Block b of stream 200 by default, of another validation stream when given."""
    assert rv.validation_seed_blocks(3, S) == [ss.block_seeds(200, b, S) for b in range(3)]
    assert rv.validation_seed_blocks(2, S, 205) == [ss.block_seeds(205, b, S) for b in range(2)]
    assert not hasattr(rv, "VALIDATION_SEED_START")
    with pytest.raises(ValueError):
        rv.validation_seed_blocks(2, S, 1)  # an optimisation stream


def test_synthesis_takes_a_stream_not_a_seed_start():
    """Its Monte-Carlo seeds are block 0 of the stream, at the run's count."""
    parser_args = dsm.synthesis_seeds(301, 1000)
    assert parser_args == ss.block_seeds(301, 0, 1000)
    with pytest.raises(ValueError):
        dsm.synthesis_seeds(1, 1000)  # not a reporting or selection stream


def test_the_pipeline_streams():
    """Optimise on stream 1 and revalidate on 200 by default; the seed start is gone."""
    config = pc.load("default")
    assert (config.stream, config.validation_stream) == (1, 200)
    assert "independent_seed_start" not in {f.name for f in pc.dataclasses.fields(config)}
    with pytest.raises(ValueError, match="stream"):
        pc.parse("[synthesise]\nindependent_seed_start = 500000\n", "test.toml")
    for bad in ("[optimise]\nstream = 100\n", "[revalidate]\nstream = 1\n"):
        with pytest.raises(ValueError):
            pc.parse(bad, "test.toml")


@pytest.mark.parametrize("text", ["[synthesise]\nmonte_carlo = 10001\n",
                                  "[revalidate]\nblocks = 1001\n"])  # fmt: skip
def test_the_pipeline_refuses_more_than_a_stream_holds_at_once(text):
    """A block holds at most 10,000 seeds and a stream 1,000 blocks: said at startup.

    Not hours later, when the stage starts.
    """
    with pytest.raises(ValueError, match="at most"):
        pc.parse(text, "test.toml")


def test_a_profile_of_more_simulations_than_a_block_holds_is_refused(monkeypatch, tmp_path):
    """--simulations is S, at most 10,000 per block: refused before the run is claimed."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "profile", "--simulations",
                                      "10001", "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="10000|10,000"):
        fo.main()
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("text", ['[optimise]\nwarm_start = ["a.txt"]\n',
                                  "[synthesise]\nindependent_seed_start = 500000\n"])  # fmt: skip
def test_a_removed_setting_says_a_run_from_before_task_7_cannot_resume(text):
    """A new file is told to drop it; a saved run that sets it cannot be resumed.

    A saved pipeline.toml that still sets one belongs to a run started before Task 7.
    """
    with pytest.raises(ValueError, match="remove it") as error:
        pc.parse(text, "pipeline.toml")
    assert "pipeline.toml that sets it belongs to a run started before Task 7" in str(error.value)


def test_the_synthesis_modes_take_their_seeds_from_the_stream():
    """A pipeline mode takes block 0 of its stream, 301 when none is given, and says which."""
    assert dsm.seed_start_for("spread_Taiwan_weight", None, 1000) == (
        301, ss.block_start(301, 0, 1000))
    assert dsm.seed_start_for("cheng2020", 300, 1000) == (300, ss.block_start(300, 0, 1000))


@pytest.mark.parametrize("mode", ["test", "profile"])
def test_the_developer_modes_keep_their_seeds_and_refuse_a_stream(mode):
    """The test and profile modes run their own seeds from 0, so a stream is refused."""
    assert dsm.seed_start_for(mode, None, 20_000) == (None, None)
    with pytest.raises(ValueError, match="own seeds"):
        dsm.seed_start_for(mode, 300, 1000)


def test_the_selection_synthesis_names_its_stream(tmp_path, monkeypatch):
    """select_by_checklist chooses on stream 301, named, not left to the synthesis's default."""
    from covsyn.calibration import select_by_checklist as sel

    calls = []
    monkeypatch.setattr(sel, "_run", lambda args, log, env=None: calls.append(list(args)))
    with pytest.raises(FileNotFoundError):  # the checks.json it then reads was never written
        sel.evaluate_candidate(np.zeros(3), 1.0, tmp_path, {})
    syntheses = [a for a in calls if "covsyn.model.data_synthesis_main" in a]
    assert syntheses and all(a[a.index("--stream") + 1] == "301" for a in syntheses)
