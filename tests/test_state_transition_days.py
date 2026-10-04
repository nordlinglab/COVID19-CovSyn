# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""State transition days of synthetic cases.

``date_of_critically_ill`` is an absolute simulation day, while ``monitor_isolation_period`` is
counted from the case's own infection day, so the ICU-to-confirmation interval is
``isolation - (ICU day - infection day)``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from covsyn.data_processing.rw_data_processing import extract_state_transition_days_synthetic


def _case(infection_day: float, isolation: int, icu: float, incubation: float = 4.0) -> pd.Series:
    return pd.Series({'infection_day': infection_day, 'incubation_period': incubation,
                      'monitor_isolation_period': isolation, 'date_of_critically_ill': icu,
                      'date_of_recovery': icu + 10.0 if math.isfinite(icu) else 30.0,
                      'date_of_death': math.nan})


@pytest.mark.parametrize("infection_day", [0.0, 10.0, 37.0])
def test_icu_to_confirmation_does_not_depend_on_the_infection_day(infection_day: float) -> None:
    """ICU 5 days and confirmation 8 days after infection: confirmed 3 days after ICU."""
    days = extract_state_transition_days_synthetic(
        [_case(infection_day, isolation=8, icu=infection_day + 5.0)])
    assert days[8][0] == 3.0


def test_icu_to_confirmation_is_zero_when_isolated_at_admission() -> None:
    """B52: a case admitted before its planned isolation is isolated on the admission day."""
    days = extract_state_transition_days_synthetic([_case(20.0, isolation=6, icu=26.0)])
    assert days[8][0] == 0.0


def test_case_without_icu_has_no_icu_to_confirmation() -> None:
    days = extract_state_transition_days_synthetic([_case(20.0, isolation=6, icu=math.nan)])
    assert np.isnan(days[8][0])
