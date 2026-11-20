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
def test_a_measurement_of_the_same_vector_is_scored(vpd, tmp_path, same):
    """The same path, written with or without a leading ./ or trailing /, is the same directory.

    The text is compared, not the file system (see read_age_rr's limits).
    """
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


def test_a_measurement_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    """Written under another name and renamed, so a kill leaves the old file or the new one.

    The write is stopped after the data is written and before it is renamed into place, where
    a kill or a full disk would stop it; a writer to the file itself would have cut it short.
    """
    from covsyn.validation import measure_age_rr as mar

    out = tmp_path / "age_rr.json"
    mar.write_measurement(out, {"n_index_cases": 4})
    assert json.loads(out.read_text()) == {"n_index_cases": 4}

    def killed(fd):
        raise OSError("killed before the rename")

    monkeypatch.setattr(mar.os, "fsync", killed)
    with pytest.raises(OSError):
        mar.write_measurement(out, {"n_index_cases": 5, "pad": "x" * 10_000})
    monkeypatch.undo()
    assert json.loads(out.read_text()) == {"n_index_cases": 4}
    assert [p.name for p in tmp_path.iterdir()] == ["age_rr.json"]


def test_two_writers_do_not_share_a_temporary_file(tmp_path, monkeypatch):
    """Each write takes its own temporary name, so two at once cannot swap halves."""
    from covsyn.validation import measure_age_rr as mar

    names = []
    real_replace = mar.os.replace

    def recording(source, target):
        names.append(str(source))
        real_replace(source, target)

    monkeypatch.setattr(mar.os, "replace", recording)
    mar.write_measurement(tmp_path / "age_rr.json", {"n": 1})
    mar.write_measurement(tmp_path / "age_rr.json", {"n": 2})
    assert len(set(names)) == 2 and all(n.startswith(str(tmp_path)) for n in names)


@pytest.mark.parametrize("measured, synthesised", [(["a"], "run_a"), ("run_b", 3), (None, None)])
def test_a_record_that_is_not_a_path_is_taken_as_no_record(vpd, tmp_path, measured, synthesised):
    """Nothing to compare: scored as before, not a crash."""
    path = tmp_path / "age_rr.json"
    path.write_text(json.dumps({"parameters_from": measured}))
    assert vpd.read_age_rr(path, _synthesis(tmp_path, synthesised))[0] is not None


def test_a_relative_and_an_absolute_record_are_not_compared(vpd, tmp_path):
    """A relative path's base is unknown, so it is not judged another directory."""
    path = tmp_path / "age_rr.json"
    path.write_text(json.dumps({"parameters_from": "/elsewhere/run_a"}))
    assert vpd.read_age_rr(path, _synthesis(tmp_path, "results/models/run_a"))[0] is not None


def test_two_relative_records_of_other_directories_are_not_scored(vpd, tmp_path):
    """Two relative paths of other directories, as steps run by hand from the root record."""
    path = tmp_path / "age_rr.json"
    path.write_text(json.dumps({"parameters_from": "results/models/run_b"}))
    measurement, problem = vpd.read_age_rr(path, _synthesis(tmp_path, "results/models/run_a"))
    assert measurement is None and "run_b" in problem


def test_the_missing_row_keeps_a_fixed_target_and_says_why_in_its_note(vpd, monkeypatch):
    """Checklists of two runs compare row by row; the path and reason go in the note."""
    monkeypatch.setattr(vpd, "results", [])
    vpd.missing_age_rr_row("no measurement at /machine/run-1/tables/age_rr.json: ...")
    (row,) = vpd.results
    assert row["decision"] == "B14" and row["target"] == vpd.MISSING_AGE_RR
    assert "/machine/run-1" not in row["target"] and "/machine/run-1" in row["note"]


def test_a_measurement_is_as_readable_as_a_plain_write_makes_it(tmp_path):
    """The file gets the umask's permissions, as write_text gave it, not mkstemp's 0600."""
    import os

    from covsyn.validation import measure_age_rr as mar

    previous = os.umask(0o027)  # one no fixed mode matches by chance
    try:
        out = tmp_path / "age_rr.json"
        mar.write_measurement(out, {"n": 1})
    finally:
        os.umask(previous)
    assert out.stat().st_mode & 0o777 == 0o640


def test_a_failed_permission_change_closes_and_removes_the_temporary_file(tmp_path, monkeypatch):
    """Nothing is left open or on disk when the mode cannot be set."""
    import os

    from covsyn.validation import measure_age_rr as mar

    closed = []
    real_fdopen = os.fdopen

    def tracking(fd, *args, **kwargs):
        f = real_fdopen(fd, *args, **kwargs)
        real_close = f.close

        def close():
            closed.append(fd)
            real_close()

        f.close = close
        return f

    def refusing(*args, **kwargs):
        raise PermissionError("mode changes refused")

    monkeypatch.setattr(mar.os, "fdopen", tracking)
    monkeypatch.setattr(mar.os, "fchmod", refusing)
    with pytest.raises(PermissionError):
        mar.write_measurement(tmp_path / "age_rr.json", {"n": 1})
    assert closed and list(tmp_path.iterdir()) == []
