# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests: a restructuring must reproduce Phase D run 10 exactly.

The fixture ``tests/fixtures/regression_run10.json`` was written by
``tests/make_regression_fixtures.py`` at the commit that produced run 10.
Every value is compared for exact equality: the objective is deterministic on its fixed seeds
(finding E30), so any difference is a change in behaviour, not noise.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import math
from typing import Any

import numpy as np
import pytest
import regression_digest as rd
from conftest import FIXTURES

FIXTURE = json.loads((FIXTURES / "regression_run10.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.slow


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def test_cost_function_reproduces_run10(
    run10_vector: np.ndarray,
    demographic_parameters: Any,
    cheng_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    cost_pool: concurrent.futures.ProcessPoolExecutor,
) -> None:
    """The objective returns the recorded total and every recorded cost part, bit for bit."""
    import fast_cost
    from cost_parts import LAST

    total = fast_cost.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    parts = dict(LAST)

    assert _same(float(total), rd.decode_float(FIXTURE["cost_total"]))
    expected = FIXTURE["cost_parts"]
    assert set(parts) == set(expected)
    mismatched = [k for k in expected if not _same(float(parts[k]), rd.decode_float(expected[k]))]
    assert mismatched == []


def test_fast_cost_equals_reference_cost_function(
    run10_vector: np.ndarray,
    demographic_parameters: Any,
    cheng_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    cost_pool: concurrent.futures.ProcessPoolExecutor,
) -> None:
    """Gate 1 of CLAUDE.md: fast_cost and firefly_optimizer.cost_function agree exactly (E65)."""
    import fast_cost
    import firefly_optimizer as fo
    from cost_parts import LAST

    reference_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    try:
        slow = fo.cost_function(run10_vector, demographic_parameters, reference_pool, *cheng_data)
        slow_parts = dict(LAST)
    finally:
        reference_pool.shutdown()
    fast = fast_cost.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    fast_parts = dict(LAST)

    assert _same(float(slow), float(fast))
    assert set(slow_parts) == set(fast_parts)
    assert [k for k in slow_parts if not _same(float(slow_parts[k]), float(fast_parts[k]))] == []


@pytest.mark.parametrize("case", sorted(FIXTURE["synthesis_digests"]))
def test_simulation_output_reproduces_run10(
    case: str, run10_vector: np.ndarray, demographic_parameters: Any
) -> None:
    """One simulation per seed and mode produces exactly the recorded output."""
    from Data_synthesis_main import run_covid

    mode, seed = case.split("/")
    output = run_covid(int(seed), run10_vector.copy(), copy.deepcopy(demographic_parameters),
                       save_file=False, mode=mode)
    assert rd.digest(output) == FIXTURE["synthesis_digests"][case]
