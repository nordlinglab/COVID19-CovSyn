# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B92: the largest spreader is charged in the objective, with the checklist's own statistic.

B71 checks where Taiwan's largest tracing count, 8 over 579 cases, falls among the largest of
579 simulated index cases, counting a tie at 8 as half (B85). Run 23 failed it on independent
seeds: 7 of 1000 index cases infected 9 or more and 3 exactly 8, a percentile of 1.0%. The
professor (2026-11-12) chose to charge it and re-optimise. The statistic is computed exactly,
P(max < 8) + P(max = 8) / 2 with each draw independent, so the objective and the checklist judge
the same number without resampling noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.contact_measures import (
    TAIWAN_TRACING_CASES,
    TAIWAN_TRACING_LARGEST,
    largest_spreader_percentile,
)


def _counts(ge9: int, eq8: int, n: int = 1000) -> np.ndarray:
    counts = np.zeros(n)
    counts[:ge9] = 9
    counts[ge9 : ge9 + eq8] = 8
    return counts


def test_taiwan_reference():
    """B71's reference: 579 traced cases, the largest infecting 8."""
    assert (TAIWAN_TRACING_CASES, TAIWAN_TRACING_LARGEST) == (579, 8)


@pytest.mark.parametrize(
    ("ge9", "eq8", "expected"),
    [
        (7, 3, 1.005),  # run 23, independent seeds
        (5, 4, 3.011),  # run 23, calibration seeds
        (5, 3, 3.223),
    ],
)
def test_run_23_values(ge9, eq8, expected):
    """The values computed from run 23's saved contact data (2026-11-11)."""
    assert largest_spreader_percentile(_counts(ge9, eq8)) == pytest.approx(expected, abs=5e-4)


def test_formula():
    """P(max < 8) + P(max = 8) / 2 for 579 independent draws."""
    counts = _counts(7, 3)
    below = np.mean(counts < 8) ** 579
    at_most = np.mean(counts <= 8) ** 579
    assert largest_spreader_percentile(counts) == pytest.approx(100 * (below + at_most) / 2)


def test_extremes():
    """No large spreader, all exactly 8, all above 8, and no cases."""
    assert largest_spreader_percentile(np.zeros(50)) == 100.0
    assert largest_spreader_percentile(np.full(50, 8.0)) == 50.0
    assert largest_spreader_percentile(np.full(50, 20.0)) == 0.0
    assert np.isnan(largest_spreader_percentile(np.array([])))


def test_agrees_with_resampling():
    """The exact value is what 2000 resampled maxima estimate (B71's former statistic)."""
    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(0.2, 0.2 / (0.2 + 0.6), 1000).astype(float)
    largest = np.array([rng.choice(counts, 579).max() for _ in range(20000)])
    resampled = 100 * (np.mean(largest < 8) + np.mean(largest == 8) / 2)
    assert largest_spreader_percentile(counts) == pytest.approx(resampled, abs=0.5)


def _with_offspring(case, infected):
    """A copy of a simulated index case whose community layer infected `infected` people."""
    course, contact, social = case
    contact = {**contact, **{f"{L}_effective_contacts": [] for L in LAYERS}}
    contact["municipality_effective_contacts"] = [1] * infected
    contact["expected_outcomes"] = None
    return course, contact, social


LAYERS = ("household", "school", "workplace", "health_care", "municipality")


def test_both_objectives_measure_it(run10_vector, demographic_parameters):
    """measure_outcomes and fast_cost.reduce_outcomes report the exact percentile."""
    import copy

    from covsyn.calibration import fast_cost
    from covsyn.model.data_synthesis_main import run_covid

    _, social, course, contact = run_covid(
        1, run10_vector.copy(), copy.deepcopy(demographic_parameters), save_file=False
    )
    case = (course[0], contact[0], social[0] if social else None)
    cases = [_with_offspring(case, 9)] * 7 + [_with_offspring(case, 0)] * 993
    expected = 100 * (993 / 1000) ** 579
    assert fo.measure_outcomes(cases)["largest_spreader_percentile"] == pytest.approx(expected)
    scalars = [fast_cost._index_case_scalars(*c) for c in cases]
    assert fast_cost.reduce_outcomes(scalars)["largest_spreader_percentile"] == pytest.approx(
        expected
    )


def test_charged_below_the_band_in_units_of_its_lower_bound():
    """1% costs (2.5 - 1) / 2.5 squared; at or above 2.5 costs nothing."""
    lo, hi, w = fo.OUTCOME_TARGETS["largest_spreader_percentile"]
    assert lo == 2.5 and hi == np.inf and w > 0
    alone = {name: np.nan for name in fo.OUTCOME_TARGETS}
    alone["largest_spreader_percentile"] = 1.0
    miss = (fo.CHARGED_BOUNDS.get("largest_spreader_percentile", (lo, hi))[0] - 1.0) / 2.5
    assert fo.outcome_penalty(alone) == pytest.approx(fo.OUTCOME_PENALTY_WEIGHT * w * miss**2)
    alone["largest_spreader_percentile"] = 50.0
    assert fo.outcome_penalty(alone) == 0.0


def test_no_large_spreader_in_one_evaluation_costs_nothing():
    """100%, no simulated case reaching 8, costs nothing.

    With 300 cases per evaluation, fewer than Taiwan's 579, that happens about 70% of the time
    even when the model matches Taiwan, so charging the upper tail charged noise: run 10 paid
    6.0 for it. Only the lower tail is charged; the checklist, on 1000 or more cases, still
    judges both.
    """
    alone = {name: np.nan for name in fo.OUTCOME_TARGETS}
    alone["largest_spreader_percentile"] = 100.0
    assert fo.outcome_penalty(alone) == 0.0


def test_checklist_uses_the_same_statistic(monkeypatch):
    """verify_phase_d imports the function and the Taiwan reference instead of copying them."""
    import importlib
    import sys

    monkeypatch.setattr(sys, "argv", ["verify_phase_d"])  # it reads sys.argv on import
    v = importlib.import_module("covsyn.validation.verify_phase_d")
    assert v.largest_spreader_percentile is largest_spreader_percentile
    assert v.TAIWAN_TRACING_LARGEST is TAIWAN_TRACING_LARGEST


def test_checklist_note_lists_the_verdicts_inputs():
    """The note gives the counts the exact verdict is computed from, not a range of the largest.

    A range of quantiles cannot share the verdict's mid-rank tie rule, so it could print
    '95% 8-9' beside a failing 1.5% (code review 2026-11-12, round 2).
    """
    source = Path(__file__).resolve().parents[1] / "src/covsyn/validation/verify_phase_d.py"
    text = source.read_text(encoding="utf-8")
    block = text[text.index("percentile of a largest of") : text.index("# B93 (2026-11-12)")]
    assert "infecting 9 or more" in block and "exactly 8" in block
    assert "95%" not in block and "rng" not in block
