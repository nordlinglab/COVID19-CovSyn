# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline regression tests: every post-optimisation step still runs and says the same thing.

The fixture ``tests/fixtures/pipeline_run10.json`` was written by
``tests/make_pipeline_fixtures.py`` on the compute server, with the run 10 vector, at the commit
recorded in its ``generated_at_commit`` (regenerated whenever a decision changes the output; last
for B91-B93).
Data files and report output must match exactly, except the checklist JSON, whose numbers
match to a relative 1e-6 (B91: the platforms' dispersion fits differ at 1e-8); figure steps must
succeed and write the same files (the images themselves are not compared).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pipeline_runner as pr
import pytest
from conftest import FIXTURES
from make_pipeline_fixtures import collect

FIXTURE = json.loads((FIXTURES / "pipeline_run10.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def produced(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the whole pipeline once for this module."""
    workdir: Path = tmp_path_factory.mktemp("pipeline")
    return collect(workdir)


def test_synthetic_data_files_are_identical(produced: dict[str, object]) -> None:
    """Both synthesis modes write exactly the recorded files, byte for byte."""
    assert produced["data"] == FIXTURE["data"]


@pytest.mark.parametrize("step", sorted(FIXTURE["reports"]))
def test_report_step_output_is_unchanged(step: str, produced: dict[str, object]) -> None:
    """Each report step exits the same way and prints the same normalised text."""
    got = produced["reports"][step]
    want = FIXTURE["reports"][step]
    assert got.get("returncode") == want.get("returncode")
    # The fixture decides how a step is compared, and each step must be compared one way.
    if "content" in want or "content" in got:
        assert "content" in want and "content" in got, (
            "the fixture and the pipeline disagree on how this step is recorded; "
            "run make_pipeline_fixtures.py")
        assert pr.json_mismatches(want["content"], got["content"]) == []
    else:
        expected = want.get("stdout_sha256", want.get("sha256"))
        assert expected is not None, "the fixture records no output for this step"
        assert got.get("stdout_sha256", got.get("sha256")) == expected


@pytest.mark.parametrize("step", sorted(FIXTURE["figures"]))
def test_figure_step_still_runs(step: str, produced: dict[str, object]) -> None:
    """Each figure step succeeds and writes the same files as before."""
    assert produced["figures"][step] == FIXTURE["figures"][step]


def test_city_population_scaling_has_a_correlation(produced: dict[str, object]) -> None:
    """The community-contact vs city-population figure is drawn from real populations (E97).

    The city population used to be overwritten by the community layer's missing context size,
    so the figure was empty and its correlation printed as nan.
    """
    figure_stdout = produced["figure_stdout"]
    assert isinstance(figure_stdout, dict)
    stdout = figure_stdout["validate_layers"]
    found = re.search(r"r\(population, mean candidate contacts\) = (\S+)", stdout)
    assert found is not None
    assert found.group(1) != "nan"
    float(found.group(1))
