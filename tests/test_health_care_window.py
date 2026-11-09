# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The contact window of each layer follows decisions B23 and B48.

Every layer stops at isolation, except health care, which continues for
HEALTH_CARE_POST_ISOLATION_DAYS (14) days after isolation and is cut short only by death;
it no longer depends on the case-closure date (B48, finding E74).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from covsyn.model import data_synthesize as ds


def _window(layer: str, isolation: int, death: float = math.nan, recovery: float = math.nan,
            infection_day: float = 10.0) -> int:
    course = SimpleNamespace(monitor_isolation_period=isolation, date_of_death=death,
                             date_of_recovery=recovery, infection_day=infection_day)
    return ds.Draw_contact_data.layer_end_day(SimpleNamespace(course_of_disease_data_object=course),
                                              layer)


@pytest.mark.parametrize("layer", ["household", "school", "workplace", "municipality"])
def test_other_layers_stop_at_isolation(layer: str) -> None:
    """Outside health care the window ends on the isolation day."""
    assert _window(layer, isolation=7, death=12.0) == 7


def test_health_care_runs_fourteen_days_past_isolation() -> None:
    """A surviving case keeps its health-care contacts for 14 more days."""
    assert ds.HEALTH_CARE_POST_ISOLATION_DAYS == 14
    assert _window("health_care", isolation=7) == 21


def test_early_closure_does_not_shorten_the_health_care_window() -> None:
    """B48: a case closed 3 days after infection still has the full window."""
    assert _window("health_care", isolation=7, recovery=13.0) == 21


def test_death_cuts_the_health_care_window() -> None:
    """Death on day 15 after infection (absolute day 25) ends the window at 15."""
    assert _window("health_care", isolation=7, death=25.0) == 15


def test_death_before_isolation_leaves_the_isolation_day() -> None:
    """The window is never shorter than the isolation day."""
    assert _window("health_care", isolation=7, death=12.0) == 7
