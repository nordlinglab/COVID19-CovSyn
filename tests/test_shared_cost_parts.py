# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finding E71: both copies of firefly_optimizer share one LAST_COST_PARTS.

Running the optimizer as a program loads its module twice, once as ``__main__`` and once by name
when fast_cost imports it.
Before the fix each copy had its own dict, so the running program never saw the cost parts that
fast_cost wrote, and run 7's progress_metrics.csv lost 33 columns without any error.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from covsyn.calibration import cost_parts, firefly_optimizer


def _second_copy() -> ModuleType:
    """Load firefly_optimizer again under another name, as running it as a program does."""
    spec = importlib.util.spec_from_file_location("firefly_optimizer_second_copy",
                                                  firefly_optimizer.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["firefly_optimizer_second_copy"] = module
    spec.loader.exec_module(module)
    return module


def test_two_copies_share_one_cost_parts_dict() -> None:
    """A write through either copy is visible through the other, and so is a clear."""
    second = _second_copy()
    assert second is not firefly_optimizer
    assert firefly_optimizer.LAST_COST_PARTS is cost_parts.LAST
    assert second.LAST_COST_PARTS is cost_parts.LAST

    firefly_optimizer.LAST_COST_PARTS.clear()
    firefly_optimizer.LAST_COST_PARTS.update(cost_contact=1.25, measured_offspring_k=0.3)
    assert dict(second.LAST_COST_PARTS) == {"cost_contact": 1.25, "measured_offspring_k": 0.3}

    second.LAST_COST_PARTS.clear()
    assert not firefly_optimizer.LAST_COST_PARTS


def test_two_copies_hold_identical_constants() -> None:
    """The constants agree in both copies, which is why run 7's fit was unaffected by E71."""
    second = _second_copy()
    for name in ("SIMULATIONS_PER_EVALUATION", "SIMULATIONS_PER_TASK", "ATTACK_RATE_WEIGHT",
                 "OUTCOME_PENALTY_WEIGHT", "PHYSIOLOGY_PENALTY_WEIGHT",
                 "MAX_INTERVAL_WIDTH_OVER_CENTRE"):
        assert getattr(firefly_optimizer, name) == getattr(second, name), name
    assert firefly_optimizer.OUTCOME_TARGETS == second.OUTCOME_TARGETS
