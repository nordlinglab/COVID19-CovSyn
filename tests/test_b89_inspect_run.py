# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B89: the review probe reads a finished run's own files instead of re-simulating.

After run 22 the per-candidate household contacts and the name of each candidate's failing
check were recomputed from scratch, 5,000 simulations and a repeated checklist, about 35
minutes. Both were already in selected/candidates/*/checks.json, which the chain writes and
keeps, and which reads in about a millisecond. The tests below pin the parts that made it slow
to find: that the per-candidate checklists are read rather than only their counts, that a check
with no measurable value is not reported as a failure, and that an older run without those
files still produces output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import inspect_run as ir  # noqa: E402


def _check(name, value, target, ok, within=False):
    return {"name": name, "value": value, "target": target, "ok": ok, "within_noise": within,
            "decision": "B00", "unit": "", "note": "", "ci": None, "interval": False}


@pytest.fixture
def run(tmp_path):
    """A run directory with the files the chain keeps, in their real places."""
    (tmp_path / "selected" / "candidates" / "candidate_7").mkdir(parents=True)
    (tmp_path / "selected" / "candidates" / "candidate_3").mkdir(parents=True)
    (tmp_path / "validation_reference").mkdir()
    (tmp_path / "phaseD_chain.log").write_text(
        "noise before\n"
        "[chain 11-05 06:34:23] validation best: candidate 7, train 1.0, validation 2.0\n"
        "not a stage line\n"
        "[chain 11-05 08:02:46] ALL DONE\n")
    seven = [_check("contacts per day before onset, household", 2.1152, [2.27, 3.28], False),
             _check("locked age risk ratio input", None, [0.39, 1.0], False),
             _check("offspring dispersion k", 0.2668, [0.2, 0.35], True),
             _check("free text", 5.0, None, True)]
    three = [_check("contacts per day before onset, household", 2.3203, [2.27, 3.28], True),
             _check("samples of 579 cases whose largest is at most 8", 0.5, [2.5, 97.5], False),
             _check("offspring dispersion k", 0.3019, [0.2, 0.35], True, within=True)]
    for name, checks in (("candidate_7", seven), ("candidate_3", three)):
        (tmp_path / "selected" / "candidates" / name / "checks.json").write_text(
            json.dumps({"checks": checks}))
    (tmp_path / "validation_reference" / "phaseD_checks.json").write_text(
        json.dumps({"checks": seven}))
    with open(tmp_path / "selected" / "revalidation.csv", "w", newline="") as f:
        f.write("candidate,training_cost,validation_mean,validation_sd\n"
                "7,14.074,9.9543,4.33\n3,3.1998,10.7304,3.83\n")
    return tmp_path


def test_the_stage_lines_are_read_from_the_chain_log(run):
    """Only the chain's own bracketed lines, so stray output cannot shift the timings."""
    stages = ir.stage_timings(run)
    assert [s[0] for s in stages] == ["11-05 06:34:23", "11-05 08:02:46"]
    assert stages[1][1] == "ALL DONE"


def test_a_missing_chain_log_is_not_an_error(tmp_path):
    """A directory without the log must give an empty list, not raise."""
    assert ir.stage_timings(tmp_path) == []


def test_a_check_with_no_value_is_not_counted_as_a_failure(run):
    """B14's locked age risk ratio has value None; the chain does not count it either."""
    checks = ir.read_checks(run / "validation_reference" / "phaseD_checks.json")
    assert ir.failing(checks) == ["contacts per day before onset, household"]


def test_a_check_whose_value_is_nan_is_not_counted_as_a_failure():
    """verify_phase_d stores an unmeasurable value as NaN with ok None, and json keeps the NaN."""
    stored = json.dumps({"checks": [_check("class size, no cases", float("nan"), [20, 35], None),
                                    _check("household", 2.1, [2.27, 3.28], False)]})
    checks = json.loads(stored)["checks"]
    assert ir.failing(checks) == ["household"]


def test_an_informational_check_is_never_a_failure(run):
    """A check whose target is None is reported only, so it cannot fail."""
    checks = ir.read_checks(run / "validation_reference" / "phaseD_checks.json")
    assert "free text" not in ir.failing(checks)


def test_every_candidate_keeps_its_own_failing_check_names(run):
    """The counts alone were what sent run 22's review back to the simulations."""
    rows = {r["candidate"]: r for r in ir.candidate_table(run, ir.DEFAULT_WATCH)}
    assert set(rows) == {"3", "7"}
    assert rows["7"]["failed"] == ["contacts per day before onset, household"]
    assert rows["3"]["failed"] == ["samples of 579 cases whose largest is at most 8"]


def test_the_watched_values_come_from_each_candidate(run):
    """The household value per candidate, which cost 5,000 simulations to recompute."""
    rows = {r["candidate"]: r for r in ir.candidate_table(run, ir.DEFAULT_WATCH)}
    name = "contacts per day before onset, household"
    assert rows["7"]["watch"][name] == pytest.approx(2.1152)
    assert rows["3"]["watch"][name] == pytest.approx(2.3203)


def test_the_costs_are_joined_from_revalidation_csv(run):
    """The costs and the checklist live in different files and are joined by index."""
    rows = {r["candidate"]: r for r in ir.candidate_table(run, ir.DEFAULT_WATCH)}
    assert float(rows["7"]["validation"]) == pytest.approx(9.9543)
    assert float(rows["3"]["sd"]) == pytest.approx(3.83)


def test_a_candidate_without_checks_json_is_skipped(run):
    """A candidate whose checklist did not finish must not break the table."""
    (run / "selected" / "candidates" / "candidate_9").mkdir()
    rows = ir.candidate_table(run, ir.DEFAULT_WATCH)
    assert {r["candidate"] for r in rows} == {"3", "7"}


def test_a_run_without_candidate_directories_still_reads(tmp_path):
    """An older run has no selected/candidates; the probe must not fail on it."""
    (tmp_path / "selected").mkdir()
    assert ir.candidate_table(tmp_path, ir.DEFAULT_WATCH) == []


def test_the_watch_list_names_the_checks_a_review_returns_to(run):
    """The default watch list covers the checks this review kept asking about."""
    for name in ("contacts per day before onset, household",
                 "samples of 579 cases whose largest is at most 8",
                 "cases infecting 3 or more"):
        assert name in ir.DEFAULT_WATCH


def test_value_of_returns_none_for_an_absent_check(run):
    """A watched name the run does not have must read as absent, not as zero."""
    checks = ir.read_checks(run / "validation_reference" / "phaseD_checks.json")
    assert ir.value_of(checks, "a check that does not exist") is None
