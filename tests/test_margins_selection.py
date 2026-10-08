# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Charged targets and the checklist-based choice of the final candidate (decision B60).

The objective charges each outcome against its acceptance interval moved inward by 1.645 times
the checklist's Monte-Carlo standard error (at most a quarter of the width; a lower bound of 0
stays), and the final candidate is the one of the best revalidated ones with the fewest
checklist failures. Independent seeds beyond the 1100 first-wave seed lists wrap around.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from covsyn.calibration import select_by_checklist as sel
from covsyn.calibration import target_margins as tm


def test_bounds_move_inward_by_the_one_sided_margin() -> None:
    """Both bounds move by 1.645 sigma when that is below a quarter of the width."""
    lo, hi = tm.charged_interval(1.0, 3.0, sigma=0.1)
    assert lo == pytest.approx(1.1645) and hi == pytest.approx(2.8355)


def test_shift_is_capped_at_a_quarter_of_the_width() -> None:
    """A noisy target is not collapsed: the shift stops at a quarter of the width."""
    assert tm.charged_interval(0.9, 1.4, sigma=1.0) == pytest.approx((1.025, 1.275))


def test_a_lower_bound_of_zero_is_not_a_constraint() -> None:
    """A one-sided target only moves its upper bound."""
    assert tm.charged_interval(0.0, 0.12, sigma=0.01) == pytest.approx((0.0, 0.10355))


def test_sigma_comes_from_the_bootstrap_interval_in_objective_units() -> None:
    """Sigma is the 95% interval width over 2 * 1.96, converted from percent."""
    checks = [
        {
            "decision": "B23",
            "name": "health care contacts starting before day 4",
            "target": [40.0, 70.0],
            "ci": [35.0, 38.92],
        }
    ]
    margins = tm.compute_margins(checks, {"medical_early_share": (0.4, 0.7, 1.0)})
    m = margins["medical_early_share"]
    assert m["sigma"] == pytest.approx(0.01)
    assert m["charged"] == pytest.approx([0.41645, 0.68355])


def test_a_stricter_objective_bound_is_kept_not_tightened_again() -> None:
    """The checklist interval shrinks, then intersects the objective's own interval."""
    checks = [
        {
            "decision": "B23",
            "name": "health care contacts starting 8+ days after onset",
            "target": [20.0, 50.0],
            "ci": [34.0, 37.92],
        }
    ]
    m = tm.compute_margins(checks, {"medical_late_share": (0.25, 0.50, 1.0)})["medical_late_share"]
    assert m["acceptance"] == pytest.approx([0.20, 0.50])
    assert m["charged"] == pytest.approx([0.25, 0.48355])


def test_integer_medians_get_no_margin() -> None:
    """The bootstrap width of an integer median is the integer grid, not noise."""
    assert "onset_to_confirmation" not in tm.CHECK_FOR_TARGET
    assert "community_median" not in tm.CHECK_FOR_TARGET


def test_a_stale_charged_file_is_refused(tmp_path) -> None:
    """Charged intervals derived from an interval that has since changed raise."""
    path = tmp_path / "charged.json"
    path.write_text(
        json.dumps(
            {"targets": {"icu_to_closure": {"objective": [29.0, 44.0], "charged": [31.5, 41.5]}}}
        )
    )
    assert tm.load_charged_bounds(path, {"icu_to_closure": (29.0, 44.0, 1.0)}) == {
        "icu_to_closure": (31.5, 41.5)
    }
    with pytest.raises(ValueError):
        tm.load_charged_bounds(path, {"icu_to_closure": (25.0, 40.0, 1.0)})


def test_a_missing_checklist_row_is_an_error() -> None:
    """Margins are never silently skipped for a mapped, charged target."""
    with pytest.raises(ValueError):
        tm.compute_margins([], {"medical_early_share": (0.4, 0.7, 1.0)})


def test_committed_charged_targets_agree_with_the_acceptance_intervals() -> None:
    """Every charged interval lies inside the acceptance interval it came from."""
    pytest.importorskip("sklearn")
    from covsyn.calibration import firefly_optimizer as fo

    data = json.loads(tm.CHARGED_TARGETS_FILE.read_text())
    assert set(data["targets"]) == set(fo.CHARGED_BOUNDS)
    for key, m in data["targets"].items():
        lo, hi, weight = fo.OUTCOME_TARGETS[key]
        assert weight > 0
        assert m["objective"] == pytest.approx([lo, hi])
        assert lo <= m["charged"][0] < m["charged"][1] <= hi
        assert m["acceptance"][0] <= m["charged"][0] < m["charged"][1] <= m["acceptance"][1]


def test_checklist_counts_failures_and_noise_passes() -> None:
    """Only judged checks count; ok~ is a pass within the interval."""
    checks = [
        {"ok": True, "within_noise": False},
        {"ok": True, "within_noise": True},
        {"ok": False, "within_noise": False},
        {"ok": None, "within_noise": False},
    ]
    assert sel.checklist_counts(checks) == (1, 1)


def test_the_choice_prefers_fewer_failures_then_fewer_noise_passes_then_cost() -> None:
    """Fewest failures first, then fewest ok~, then the lowest validation cost."""
    rows = [
        {"candidate": 1, "failed": 1, "within_noise": 0, "validation_mean": 2.0},
        {"candidate": 2, "failed": 0, "within_noise": 3, "validation_mean": 9.0},
        {"candidate": 3, "failed": 0, "within_noise": 1, "validation_mean": 5.0},
        {"candidate": 4, "failed": 0, "within_noise": 1, "validation_mean": 4.0},
    ]
    assert min(rows, key=sel.rank_key)["candidate"] == 4


def test_first_wave_seeds_beyond_the_list_wrap_around(
    run10_vector: np.ndarray, demographic_parameters: object
) -> None:
    """An independent seed (B60) reuses the 28 seed days instead of indexing past the list."""
    from covsyn.model.data_synthesis_main import run_covid

    _, _, courses, _ = run_covid(
        500000,
        run10_vector.copy(),
        copy.deepcopy(demographic_parameters),
        save_file=False,
        mode="taiwan_first_outbreak",
    )
    assert len(courses) >= 28
