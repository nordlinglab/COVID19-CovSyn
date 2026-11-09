# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B76: the objective charges health care contacts per index case below Cheng's floor.

The quantity is every candidate health care contact of the index case, the population Cheng
2020's per-contact rate is measured on (review of B76, user 2026-11-01, option A); the floor is
sar_anchors.health_care_contact_floor(), 3.335, with no upper bound. The contacts inside Jian
et al. 2020's tracing window are reported only.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.calibration import sar_anchors as sa
from covsyn.model import contact_measures as cm

pytest.importorskip("sklearn")  # firefly_optimizer imports it at module level
from covsyn.calibration import fast_cost  # noqa: E402
from covsyn.calibration import firefly_optimizer as fo  # noqa: E402

KEY = "health_care_contacts_per_index"


def _case(onset: float, isolation: int, health_days, household_days=()) -> tuple[dict, dict]:
    """One case whose health care and household contacts are each met on one given day."""
    contact = {}
    for layer, days in (("health_care", health_days), ("household", household_days)):
        m = np.zeros((len(days), 12), dtype=bool)
        for row, day in enumerate(days):
            m[row, day] = True
        contact[cm.MATRIX_KEY[layer]] = m
    for layer in ("school", "workplace", "municipality"):
        contact[cm.MATRIX_KEY[layer]] = np.zeros((0, 12), dtype=bool)
    return {"incubation_period": onset, "monitor_isolation_period": isolation}, contact


def test_layers_restrict_the_tracing_window_count() -> None:
    """Window 3..8: health care days 1, 4, 8 and 10 give 2; household day 5 is not counted."""
    course, contact = _case(5.0, 8, (1, 4, 8, 10), (5,))
    assert cm.contacts_in_tracing_window(course, contact, layers=("health_care",)) == 2
    assert cm.contacts_in_tracing_window(course, contact) == 3  # every layer, as before


def test_one_pass_counts_every_layer() -> None:
    """Review finding 10: one walk of the window gives every layer; the total is their sum."""
    course, contact = _case(5.0, 8, (1, 4, 8, 10), (5,))
    counts = cm.tracing_window_counts(course, contact)
    assert counts == {
        "household": 1,
        "school": 0,
        "workplace": 0,
        "health_care": 2,
        "municipality": 0,
    }
    assert sum(counts.values()) == cm.contacts_in_tracing_window(course, contact)


def test_unknown_layer_is_rejected() -> None:
    course, contact = _case(5.0, 8, (4,))
    with pytest.raises(KeyError):
        cm.contacts_in_tracing_window(course, contact, layers=("hospital",))


def test_target_is_the_floor_with_no_upper_bound() -> None:
    lo, hi, weight = fo.OUTCOME_TARGETS[KEY]
    assert lo == pytest.approx(sa.health_care_contact_floor())
    assert hi == np.inf
    assert weight == 1.0


def test_meeting_the_charged_floor_costs_nothing() -> None:
    """B60: the charged floor is the acceptance floor moved up by its Monte-Carlo margin."""
    charged_lo, charged_hi = fo.CHARGED_BOUNDS[KEY]
    assert charged_lo > sa.health_care_contact_floor()
    assert charged_hi == np.inf
    assert fo.outcome_penalty({KEY: charged_lo}) == 0.0
    assert fo.outcome_penalty({KEY: 50.0}) == 0.0


def test_a_miss_is_measured_in_units_of_the_floor() -> None:
    """From the charged floor, in units of the acceptance floor (B50, B60)."""
    floor = sa.health_care_contact_floor()
    charged_lo = fo.CHARGED_BOUNDS[KEY][0]
    miss = (charged_lo - 2.597) / floor
    assert fo.outcome_penalty({KEY: 2.597}) == pytest.approx(fo.OUTCOME_PENALTY_WEIGHT * miss**2)


def test_the_window_count_is_no_longer_an_objective_target() -> None:
    assert "health_care_contacts_traced" not in fo.OUTCOME_TARGETS
    assert not hasattr(fast_cost, "health_care_scalars")


def test_daily_health_care_is_measured_not_charged() -> None:
    """B76 (2026-11-01, user): the unsourced 0.30 cap no longer blocks the floor; band kept."""
    assert fo.OUTCOME_TARGETS["daily_health_care"] == (0.00, 0.30, 0.0)
    assert fo.outcome_penalty({"daily_health_care": 0.47}) == 0.0


# --- review of B76 (2026-11-01) -----------------------------------------------------


def test_open_bound_is_written_as_null() -> None:
    """Strict JSON has no Infinity: the open upper bound of B76 becomes null."""
    import json

    from covsyn.validation.acceptance import json_band

    assert json_band([3.3, np.inf]) == [3.3, None]
    assert json_band([1.0, 2.0]) == [1.0, 2.0]
    assert json_band(None) is None
    assert json_band("text") == "text"
    json.dumps(json_band([3.3, np.inf]), allow_nan=False)  # raises on Infinity


@pytest.mark.parametrize(
    "target, text",
    [
        ([1.0, 2.5], "[1, 2.5]"),
        ([3.3, None], "[3.3, inf]"),
        ([3.3, np.inf], "[3.3, inf]"),
        (None, "(information)"),
    ],
)
def test_band_text_prints_an_open_bound(target, text: str) -> None:
    from covsyn.validation.acceptance import band_text

    assert band_text(target) == text


def test_open_bounds_round_trip_on_either_side() -> None:
    """Second review: one pair of helpers maps open bounds to null and back, both sides."""
    from covsyn.validation.acceptance import band_from_json, json_band

    assert json_band([-np.inf, 2.0]) == [None, 2.0]
    assert band_from_json([None, 2.0]) == (-np.inf, 2.0)
    assert band_from_json([3.0, None]) == (3.0, np.inf)
    assert band_from_json(json_band([3.0, np.inf])) == (3.0, np.inf)


def test_per_contact_lower_bounds_are_a_table() -> None:
    """Second review: the health care override sits in a table next to ANCHOR_FLOOR."""
    assert sa.PER_CONTACT_LOWER == {"health_care": sa.wilson_interval(6, 698)[0]}
