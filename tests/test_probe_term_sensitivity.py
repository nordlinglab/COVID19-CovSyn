# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which cost terms respond to each parameter, one at a time, at a few base vectors.

The screen behind the decision whether the energy term, or the Cheng terms, can leave the
objective: a parameter that only those terms respond to would be left unconstrained. Each free
parameter moves 0.2 of its normalised range towards the interior; effects are paired
differences over replications, reported with t intervals against a substantive threshold.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_term_sensitivity as pts  # noqa: E402


def test_a_step_goes_up_from_the_lower_half():
    """At 0.3 the parameter moves to 0.5."""
    assert pts.perturbed(np.array([0.3, 0.9]), 0, 0.2).tolist() == pytest.approx([0.5, 0.9])


def test_a_step_goes_down_from_the_upper_half():
    """At 0.9 the parameter moves to 0.7."""
    assert pts.perturbed(np.array([0.3, 0.9]), 1, 0.2).tolist() == pytest.approx([0.3, 0.7])


def test_the_base_vector_is_left_unchanged():
    """The step returns a new vector."""
    x = np.array([0.3, 0.9])
    pts.perturbed(x, 0, 0.2)
    assert x.tolist() == [0.3, 0.9]


@given(x=st.floats(0, 1), step=st.floats(0.01, 0.5))
def test_a_step_stays_inside_the_range_and_has_the_full_length(x, step):
    """Towards the interior, a step of at most 0.5 never leaves [0, 1]."""
    out = pts.perturbed(np.array([x]), 0, step)[0]
    assert 0 <= out <= 1
    assert abs(out - x) == pytest.approx(step)


def test_the_bases_are_the_lowest_the_farthest_low_and_a_middle_row():
    """Lowest cost first, then the low-cost row farthest from it, then the quantile row."""
    x = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.2]])
    costs = np.array([1.0, 2.0, 3.0, 50.0, 100.0])
    low, far, middle = pts.pilot_bases(x, costs, exclude=(), low_share=0.6, middle_quantile=0.75)
    assert (low, far) == (0, 2)
    assert middle == 3


def test_excluded_rows_are_never_chosen():
    """Rows already used as bases (run 19, the literature vector) are skipped."""
    x = np.eye(4)
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    chosen = pts.pilot_bases(x, costs, exclude=(0,), low_share=0.75, middle_quantile=0.5)
    assert 0 not in chosen


def test_an_effect_beyond_the_threshold_is_classed_as_such():
    """The whole interval above delta."""
    assert pts.classify([1.0, 1.1, 0.9, 1.0, 1.05], delta=0.2) == "beyond"


def test_an_effect_inside_the_threshold_is_classed_as_such():
    """The whole interval within +-delta."""
    assert pts.classify([0.01, -0.02, 0.0, 0.015, -0.01], delta=0.2) == "within"


def test_a_noisy_effect_is_imprecise():
    """The interval straddles delta."""
    assert pts.classify([0.5, -0.4, 0.3, -0.2, 0.1], delta=0.2) == "imprecise"


def test_a_negative_effect_beyond_the_threshold_counts():
    """The sign does not matter for beyond."""
    assert pts.classify([-1.0, -1.1, -0.9, -1.0, -1.05], delta=0.2) == "beyond"


def test_identical_differences_are_exact():
    """No spread: the interval is the point."""
    assert pts.classify([0.0] * 5, delta=0.1) == "within"
    assert pts.classify([0.3] * 5, delta=0.1) == "beyond"


def test_a_string_case_row_only_invalidates_energy(monkeypatch):
    """An unusable energy row keeps costs and outcomes, and the next evaluation still runs."""
    rows = np.array([[20.0] * 9, [40.0] * 9])
    monkeypatch.setattr(pts, "_TAIWAN", np.array([[10.0, 15.0], [50.0, 45.0]]))

    def simulated(_p, seed, _clock):
        cases = ["bad row", rows[0]] if seed == 1 else rows
        parts = {"cost_contact": 10.0, "cost_outcome": 2.0, "measured_example": 0.3}
        return 12.0, parts, (None, None, cases), None

    monkeypatch.setattr(pts.fast_cost, "_simulate_and_score", simulated)
    failed_energy = pts.score(None, 1)
    assert failed_energy["cost"] == 12.0
    assert failed_energy["cost_contact"] == 10.0
    assert failed_energy["cost_outcome"] == 2.0
    assert failed_energy["measured_example"] == 0.3
    assert np.isnan(failed_energy["energy_u"])
    result = pts.score(None, 2)
    assert result["cost"] == 12.0
    assert np.isfinite(result["energy_u"])


def test_an_unexpected_energy_exception_propagates(monkeypatch):
    """Only the known failed-row diagnostic is converted to NaN."""
    monkeypatch.setattr(pts, "_TAIWAN", np.zeros((2, 2)))
    monkeypatch.setattr(
        pts.fast_cost, "_simulate_and_score",
        lambda *_: (12.0, {}, (None, None, np.zeros((2, 9))), None),
    )

    def broken_energy(*_):
        raise IndexError("unexpected energy error")

    monkeypatch.setattr(pts.per, "restricted_energy", broken_energy)
    with pytest.raises(IndexError, match="unexpected energy error"):
        pts.score(None, 1)


@pytest.mark.parametrize(("energy", "expected_class"), [
    ([np.nan, 0.4, 0.4], "beyond"),
    ([np.nan, np.nan, 0.4], "imprecise"),
    ([np.nan, np.nan, np.nan], "imprecise"),
])
def test_analysis_uses_only_finite_pairs_per_term(tmp_path, energy, expected_class):
    """An invalid energy replication neither removes non-energy pairs nor leaks NaN to JSON."""
    path = tmp_path / "score.csv"
    output = tmp_path / "summary.json"
    rows = []
    for index, value in enumerate(energy):
        for param, energy_u in [(-1, 0.0), (0, value)]:
            rows.append({
                "base": "one", "param": param, "replication": index,
                "cost_contact": 0.2 if param < 0 else 0.6,
                "cost_attack_rate": 0.0, "cost_energy": 0.0, "energy_u": energy_u,
                "cost_outcome": 0.0, "cost_penalty": 0.0,
            })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pts.analyse(SimpleNamespace(out=path, json=output))
    result = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"Nonstandard JSON value: {value}"),
    )["one|0"]
    assert result["contact"]["class_0.1"] == "beyond"
    assert result["contact"]["mean"] == pytest.approx(0.4)
    assert result["energy_new"]["class_0.1"] == expected_class
    valid = sum(np.isfinite(energy))
    assert result["energy_new"]["mean"] == (0.4 if valid else None)
    assert result["energy_new"]["sd"] == (0.0 if valid > 1 else None)


@pytest.mark.parametrize("mismatch", [False, True])
def test_probe_resume_uses_saved_header_or_leaves_file_unchanged(tmp_path, monkeypatch, mismatch):
    """Use the existing column order and reject schema drift before appending a row."""
    path = tmp_path / "probe.csv"
    original = b"base,row,param,replication,measured_old,cost\nrun19,2,-1,0,0.3,1.0\n"
    path.write_bytes(original)
    population = np.arange(5.0).reshape(5, 1)
    monkeypatch.setattr(
        pts.pilot_noise, "_population_and_bounds",
        lambda: (population, np.zeros(1), np.zeros(1)),
    )
    pilot = {("spread", i, 1, 0): i for i in range(5)}
    monkeypatch.setattr(pts.psb, "read_pilot", lambda *_: pilot)
    monkeypatch.setattr(pts, "pilot_bases", lambda *a, **kw: (1, 3, 4))
    monkeypatch.setattr(
        pts.block_scoring, "objective_pools",
        lambda _workers: (None, lambda _s: concurrent.futures.ThreadPoolExecutor(max_workers=1)),
    )
    monkeypatch.setattr(pts.concurrent.futures, "as_completed", lambda submitted: iter(submitted))
    records = iter([
        {"cost": 2.0, "measured_old": 0.4},
        {"cost": 3.0, "measured_new" if mismatch else "measured_old": 0.5},
        {"cost": 4.0, "measured_old": 0.6},
        {"cost": 5.0, "measured_old": 0.7},
    ])
    monkeypatch.setattr(pts, "score", lambda *_: next(records))
    args = SimpleNamespace(out=path, pilot_blocks=tmp_path / "unused", workers=1, replications=1)

    if mismatch:
        with pytest.raises(ValueError, match=r"extra.*measured_new.*missing.*measured_old"):
            pts.run(args)
        assert path.read_bytes() == original
    else:
        pts.run(args)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["cost"] for row in rows] == [str(float(i)) for i in range(1, 6)]
        assert rows[1]["measured_old"] == "0.4"
