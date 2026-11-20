# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The checklist's B14 measurement file (AGE_RR_FILE), found by Task 7's review.

The checklist scores B14 on the measurement measure_age_rr writes. It must say which file it
looked for and why it could not use one, rather than crash or score a measurement of another
vector.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest


@pytest.fixture(scope="module")
def vpd():
    """verify_phase_d, imported with its own command line: it reads sys.argv at import."""
    saved = sys.argv
    sys.argv = ["verify_phase_d"]
    try:
        return importlib.import_module("covsyn.validation.verify_phase_d")
    finally:
        sys.argv = saved


@pytest.mark.parametrize("value", [None, ""])
def test_an_unset_or_empty_age_rr_file_means_the_default(vpd, monkeypatch, value):
    """An empty AGE_RR_FILE is taken as unset, not as the working directory."""
    if value is None:
        monkeypatch.delenv("AGE_RR_FILE", raising=False)
    else:
        monkeypatch.setenv("AGE_RR_FILE", value)
    assert vpd.age_rr_file() == vpd.DEFAULT_AGE_RR_FILE


def test_a_missing_measurement_names_the_file_looked_for(vpd, tmp_path):
    """The row says where it looked, so a hand run can tell which file is missing."""
    path = tmp_path / "age_rr.json"
    measurement, problem = vpd.read_age_rr(path, tmp_path)
    assert measurement is None and str(path) in problem and "has not run" in problem


def test_a_directory_is_no_measurement(vpd, tmp_path):
    """A directory given as AGE_RR_FILE is said to be missing, not read and crashed on."""
    measurement, problem = vpd.read_age_rr(tmp_path, tmp_path)
    assert measurement is None and str(tmp_path) in problem


def test_a_whole_measurement_is_read(vpd, tmp_path):
    """The normal case, which the regression pipeline runs, reads the file as before."""
    path = tmp_path / "age_rr.json"
    path.write_text(json.dumps({"n_index_cases": 400}))
    assert vpd.read_age_rr(path, tmp_path) == ({"n_index_cases": 400}, "")


@pytest.mark.parametrize("text", ['{"n_index_cases": 4', "", "[1, 2]"])
def test_a_measurement_cut_short_is_said_to_be_unreadable(vpd, tmp_path, text):
    """A file a kill cut short, or not a measurement, gives a row instead of a crash."""
    path = tmp_path / "age_rr.json"
    path.write_text(text)
    measurement, problem = vpd.read_age_rr(path, tmp_path)
    assert measurement is None and str(path) in problem and "cannot be read" in problem


def _synthesis(tmp_path, parameter_path):
    """A spread synthesis directory whose run_metadata.json names its parameters."""
    spread = tmp_path / "spread"
    spread.mkdir()
    (spread / "run_metadata.json").write_text(json.dumps({"parameter_path": parameter_path}))
    return spread


def test_a_measurement_of_another_vector_is_not_scored(vpd, tmp_path):
    """Left over from another candidate, it would score that candidate's B14 as this one's."""
    path = tmp_path / "age_rr.json"
    path.write_text(json.dumps({"parameters_from": str(tmp_path / "run_b")}))
    spread = _synthesis(tmp_path, str(tmp_path / "run_a"))
    measurement, problem = vpd.read_age_rr(path, spread)
    assert measurement is None and "run_b" in problem and "run_a" in problem


@pytest.mark.parametrize("same", ["run_a", "./run_a/"])
def test_a_measurement_of_the_same_vector_is_scored(vpd, tmp_path, monkeypatch, same):
    """The same parameter directory, however written, is the same vector."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run_a").mkdir()
    path = tmp_path / "age_rr.json"
    path.write_text(json.dumps({"parameters_from": same}))
    measurement, problem = vpd.read_age_rr(path, _synthesis(tmp_path, "run_a"))
    assert measurement is not None and problem == ""


def test_without_a_record_of_either_vector_the_measurement_is_scored(vpd, tmp_path):
    """Nothing to compare: scored as before (an older synthesis or measurement)."""
    path = tmp_path / "age_rr.json"
    path.write_text(json.dumps({"n_index_cases": 4}))
    assert vpd.read_age_rr(path, _synthesis(tmp_path, "run_a"))[0] is not None
    path.write_text(json.dumps({"parameters_from": "run_b"}))
    assert vpd.read_age_rr(path, tmp_path)[0] is not None  # no run_metadata.json here


def test_a_measurement_is_written_whole_or_not_at_all(tmp_path):
    """Written under another name and renamed, so a kill leaves the old file or the new one."""
    from covsyn.validation import measure_age_rr as mar

    out = tmp_path / "age_rr.json"
    mar.write_measurement(out, {"n_index_cases": 4})
    assert json.loads(out.read_text()) == {"n_index_cases": 4}
    with pytest.raises(TypeError):  # fails while writing: the earlier file stays whole
        mar.write_measurement(out, {"n_index_cases": object()})
    assert json.loads(out.read_text()) == {"n_index_cases": 4}
    assert [p.name for p in tmp_path.iterdir()] == ["age_rr.json"]
