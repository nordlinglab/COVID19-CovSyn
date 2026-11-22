# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 10, item 1, in the pipeline: production re-scores its archive's best at its S.

Decision of 2026-11-22: in the Task 10 flow, the production pipeline's revalidation re-scores
the archive's best k = 10 distinct vectors on 30 blocks of its validation stream at the run's
S, and the same results serve the selection and cost_uncertainty.csv. Other modes revalidate as
before: every candidate, at the optimiser's 300 simulations, on the configured blocks.
"""

from __future__ import annotations

import dataclasses

import pytest

from covsyn.pipeline import config as pc
from tests.test_task4_pipeline import Recorder, run_pipeline
from tests.test_task9_production_pipeline import PRODUCTION


def _revalidation(recorder):
    args = next(a for a in recorder.calls if "covsyn.calibration.revalidation" in map(str, a))
    return [str(a) for a in args]


def test_production_revalidates_thirty_blocks_unless_told_otherwise():
    """Task 10's 30 blocks, without a [revalidate] blocks of its own."""
    assert pc.parse(PRODUCTION, "t").revalidation_blocks == 30


def test_production_refuses_another_number_of_blocks():
    """Task 10 re-scores on 30 blocks; another count would not be its measurement."""
    with pytest.raises(ValueError, match="30"):
        pc.parse(PRODUCTION + "[revalidate]\nblocks = 10\n", "t")


def test_production_re_scores_the_archive_s_best_ten_at_its_s(tmp_path):
    """--simulations S, --archive-best 10 and 30 blocks reach covsyn-revalidation."""
    recorder = Recorder()
    run_pipeline(tmp_path, pc.parse(PRODUCTION, "t"), recorder)
    args = _revalidation(recorder)
    assert args[args.index("--simulations") + 1] == "3000"
    assert args[args.index("--archive-best") + 1] == "10"
    assert "30" in args  # the blocks


def test_a_train_run_revalidates_as_before(tmp_path):
    """Every candidate, at 300 simulations, on its configured blocks."""
    recorder = Recorder()
    run_pipeline(tmp_path, dataclasses.replace(pc.load("default"), revalidation_blocks=10),
                 recorder)  # fmt: skip
    args = _revalidation(recorder)
    assert "--simulations" not in args and "--archive-best" not in args
    assert "10" in args


def test_the_saved_production_configuration_keeps_its_thirty_blocks():
    """pipeline.toml, which a resume reads, holds the 30."""
    config = pc.parse(PRODUCTION, "t")
    assert pc.parse(pc.to_toml(config), "saved").revalidation_blocks == 30


def test_the_refusal_says_a_run_saved_with_other_blocks_cannot_be_resumed():
    """A pipeline.toml saved before Task 10 with another count: start a new run instead."""
    with pytest.raises(ValueError, match="cannot be resumed"):
        pc.parse(PRODUCTION + "[revalidate]\nblocks = 10\n", "saved")
