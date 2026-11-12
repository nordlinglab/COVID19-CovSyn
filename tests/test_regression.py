# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests: a restructuring must reproduce Phase D run 10 exactly.

The fixture ``tests/fixtures/regression_run10.json`` was written by
``tests/make_regression_fixtures.py`` with the run 10 vector, at the commit recorded in the
fixture (regenerated whenever a decision changes the output; last for B75).
The objective is deterministic on its fixed seeds (finding E30), and the simulation outputs
are compared by digest, exactly. The cost is compared to a relative 1e-6: the professor
(2026-11-12) accepted reproduction "to within normal precision differences between the
packages" in place of bit-identical results on any machine. Before B91 the platforms differed
by 4e-5 (E103: 72.9278694577941 on the macOS server, 72.9305393810959 under WSL on x86), because
the energy term's BLAS distances turned rounding into log(1e-16) against log(5e-8); B91
computes them exactly. A behavioural change moves the cost far more than 1e-6 (B90's rescaling
moved it by 1%).
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


def _close(a: float, b: float) -> bool:
    """Equal to normal precision across platforms (B91); NaN only matches NaN."""
    return _same(a, b) or math.isclose(a, b, rel_tol=rd.PLATFORM_REL_TOL)


def test_cost_function_reproduces_run10(
    run10_vector: np.ndarray,
    demographic_parameters: Any,
    cheng_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    cost_pool: concurrent.futures.ProcessPoolExecutor,
) -> None:
    """The objective returns the recorded total and every recorded cost part (B91 tolerance)."""
    from covsyn.calibration import fast_cost
    from covsyn.calibration.cost_parts import LAST

    total = fast_cost.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    parts = dict(LAST)

    assert _close(float(total), rd.decode_float(FIXTURE["cost_total"]))
    expected = FIXTURE["cost_parts"]
    assert set(parts) == set(expected)
    mismatched = [k for k in expected if not _close(float(parts[k]), rd.decode_float(expected[k]))]
    assert mismatched == []


@pytest.mark.parametrize("events", [None, (0.10, 1.49, 21, 1000, 1.0)],
                         ids=["run10", "run10-with-B55-events"])
def test_fast_cost_equals_reference_cost_function(
    events: tuple[float, ...] | None,
    run10_vector: np.ndarray,
    demographic_parameters: Any,
    cheng_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    cost_pool: concurrent.futures.ProcessPoolExecutor,
) -> None:
    """Gate 1 of CLAUDE.md: fast_cost and firefly_optimizer.cost_function agree exactly (E65).

    The event case covers the expected-event Cheng bins and the ordinary-contact daily
    measure of B55, which run 10's 199-value vector never reaches.
    """
    from covsyn.calibration import fast_cost
    from covsyn.calibration import firefly_optimizer as fo
    from covsyn.calibration.cost_parts import LAST

    vector = run10_vector if events is None else np.concatenate([run10_vector, events])
    reference_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    try:
        slow = fo.cost_function(vector, demographic_parameters, reference_pool, *cheng_data)
        slow_parts = dict(LAST)
    finally:
        reference_pool.shutdown()
    fast = fast_cost.cost_function(vector, demographic_parameters, cost_pool, *cheng_data)
    fast_parts = dict(LAST)

    assert _same(float(slow), float(fast))
    assert set(slow_parts) == set(fast_parts)
    assert [k for k in slow_parts if not _same(float(slow_parts[k]), float(fast_parts[k]))] == []


@pytest.mark.parametrize("case", sorted(FIXTURE["synthesis_digests"]))
def test_simulation_output_reproduces_run10(
    case: str, run10_vector: np.ndarray, demographic_parameters: Any
) -> None:
    """One simulation per seed and mode produces exactly the recorded output."""
    from covsyn.model.data_synthesis_main import run_covid

    mode, seed = case.split("/")
    output = run_covid(int(seed), run10_vector.copy(), copy.deepcopy(demographic_parameters),
                       save_file=False, mode=mode)
    assert rd.digest(output) == FIXTURE["synthesis_digests"][case]
