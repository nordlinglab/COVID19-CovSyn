# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B91: the checklist JSON is compared to normal precision, not byte for byte.

The professor (2026-11-12) accepted reproduction "to within normal precision differences between
the packages". After B91 the macOS server and Linux still differ in the checklist JSON by the
dispersion fit (k 0.24445814515766887 against 0.24445813019840637) and by one unit in the last
place of a derived band, so a digest of the file failed on Linux although nothing behaved
differently. Numbers now match to a relative 1e-6, like the regression test's cost; text,
booleans, keys and structure must still match exactly.
"""

from __future__ import annotations

import json
import math

import pipeline_runner as pr
import pytest


def test_numbers_within_normal_precision_match():
    """The two differences the platforms showed are accepted."""
    a = {"checks": [{"value": 0.24445814515766887, "target": [0.0621, 0.36844911035319167]}]}
    b = {"checks": [{"value": 0.24445813019840637, "target": [0.0621, 0.3684491103531918]}]}
    assert pr.json_mismatches(a, b) == []


def test_a_real_change_is_reported_with_its_path():
    """A difference of 1e-5 relative is a change, and the report says where."""
    a = {"checks": [{"name": "k", "value": 0.2444}]}
    b = {"checks": [{"name": "k", "value": 0.2444 * (1 + 1e-5)}]}
    (report,) = pr.json_mismatches(a, b)
    assert report.startswith("checks[0].value: want 0.2444, got ")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ({"ok": True}, {"ok": False}),  # a verdict
        ({"ok": True}, {"ok": 1}),  # a bool is not a number here
        ({"note": "8"}, {"note": "9"}),  # text
        ({"x": 1}, {"y": 1}),  # keys
        ({"x": [1, 2]}, {"x": [1, 2, 3]}),  # length
        ({"x": None}, {"x": 0.0}),  # missing against zero
        ({"x": float("nan")}, {"x": 0.0}),
        ({"n": 1500000}, {"n": 1500001}),  # a count, which carries no rounding
        ({"n": 5}, {"n": 5.0}),  # a count turned into a float
        ({"p": 1e-15}, {"p": 9e-13}),  # tiny values are compared relatively too
    ],
)
def test_anything_but_a_close_number_must_be_equal(a, b):
    """A verdict, note, key, length, count or tiny value that changes is a real change."""
    assert pr.json_mismatches(a, b) != []


def test_nan_matches_nan_and_empty_matches_empty():
    """NaN is how the checklist writes a value with no cases behind it."""
    assert pr.json_mismatches({"x": math.nan, "y": []}, {"x": math.nan, "y": []}) == []


def test_large_input():
    """Many checks: only the one that moved is reported."""
    a = {"checks": [{"value": float(i)} for i in range(5000)]}
    b = {"checks": [{"value": float(i)} for i in range(5000)]}
    b["checks"][4321]["value"] += 1.0
    assert pr.json_mismatches(a, b) == ["checks[4321].value: want 4321.0, got 4322.0"]


def test_shared_keys_are_compared_when_other_keys_differ():
    """One slow pipeline run reports a new key and a moved value together."""
    a = {"checks": {"k": 0.2444}, "old": 1}
    b = {"checks": {"k": 0.3}, "new": 1}
    found = pr.json_mismatches(a, b)
    assert found[0].startswith("<root>: keys differ")
    assert any(m.startswith("checks.k: ") for m in found)


def test_workdir_is_replaced_in_keys_and_escaped_strings(tmp_path):
    """Replaced after parsing, so a path JSON escapes is found, in keys as well as values."""
    workdir = tmp_path / "rün"
    path = tmp_path / "checks.json"
    a, b = str(workdir / "a"), str(workdir / "b")
    path.write_text(json.dumps({a: b}), encoding="utf-8")
    assert "\\u" in path.read_text(encoding="utf-8")
    replace = str(workdir), "<WORKDIR>"
    assert pr.normalised_json(path, workdir) == {a.replace(*replace): b.replace(*replace)}


def test_keys_that_collide_after_replacement_are_an_error(tmp_path):
    """Two keys that differ only by the workdir would otherwise lose one value silently."""
    path = tmp_path / "checks.json"
    path.write_text(json.dumps({str(tmp_path / "a"): 1, "<WORKDIR>/a": 2}), encoding="utf-8")
    with pytest.raises(ValueError):
        pr.normalised_json(path, tmp_path)
