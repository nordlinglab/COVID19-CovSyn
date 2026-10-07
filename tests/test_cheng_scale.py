# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Cheng contact fit compares contacts per 100 cases with Cheng et al. 2020 (finding E86).

Cheng traced the contacts of 100 index cases. The original objective simulated 100 cases three
times and divided the summed contact bins by 3. Run 3 (E38) changed this to 300 cases simulated
once, which left the division at 1, so every run since compared 300 cases of contacts with
Cheng's 100 and pushed CovSyn's contacts per case towards a third of Cheng's.
"""

from __future__ import annotations

import pytest


def test_cheng_repeats_scale_the_simulations_to_cheng_cohorts() -> None:
    pytest.importorskip('sklearn')  # firefly_optimizer imports it at module level
    from covsyn.calibration import firefly_optimizer as fo
    from covsyn.calibration.sar_anchors import CHENG2020_INDEX_CASES

    assert CHENG2020_INDEX_CASES == 100
    assert fo.SIMULATIONS_PER_EVALUATION % CHENG2020_INDEX_CASES == 0
    assert fo.cheng_repeat_number() == fo.SIMULATIONS_PER_EVALUATION // CHENG2020_INDEX_CASES
    assert fo.cheng_repeat_number() == 3


def test_simulated_cases_stay_at_three_hundred() -> None:
    """The fix rescales the comparison only; the objective still simulates 300 seeds."""
    pytest.importorskip('sklearn')
    from covsyn.calibration import firefly_optimizer as fo

    assert fo.SIMULATIONS_PER_EVALUATION == 300
    assert fo.CHENG_INDEX_CASES * fo.cheng_repeat_number() == fo.SIMULATIONS_PER_EVALUATION
