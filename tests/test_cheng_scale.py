# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Cheng contact fit compares contacts per symptomatic index case with Cheng et al. 2020.

Finding E86: the objective summed the contact bins of 300 simulated cases and compared them
with Cheng's totals for 100 cases, pushing CovSyn's contacts per case towards a third of
Cheng's. The bins drop asymptomatic cases (Cheng's six bins count days from symptom onset), so
the CovSyn bins are scaled to Cheng's 91 symptomatic index cases: Cheng's bins hold all 2,761
contacts, of which 91 belong to his 9 asymptomatic cases (Table 2), so the comparison is off
by those 3.3% at most.
"""

from __future__ import annotations

import pytest


def test_contact_scale_maps_the_symptomatic_cases_onto_cheng_cohort() -> None:
    """Contact scale maps the symptomatic cases onto cheng cohort."""
    pytest.importorskip("sklearn")  # firefly_optimizer imports it at module level
    from covsyn.calibration import firefly_optimizer as fo

    assert fo.CHENG_SYMPTOMATIC_INDEX_CASES == 91
    assert fo.cheng_contact_scale(91) == 1.0
    assert fo.cheng_contact_scale(227) == pytest.approx(91 / 227)


def test_contact_scale_without_symptomatic_cases_raises() -> None:
    """No symptomatic case means no Cheng bins, so the scale raises.

    Raising makes cost_function charge the failure cost rather than return a NaN the
    firefly cannot rank.
    """
    pytest.importorskip("sklearn")
    from covsyn.calibration import firefly_optimizer as fo

    with pytest.raises(ValueError):
        fo.cheng_contact_scale(0)


def test_objective_still_simulates_three_hundred_seeds() -> None:
    """Objective still simulates three hundred seeds."""
    pytest.importorskip("sklearn")
    from covsyn.calibration import firefly_optimizer as fo

    assert fo.SIMULATIONS_PER_EVALUATION == 300
