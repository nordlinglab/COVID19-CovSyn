# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures for the CovSyn test suite.

The CovSyn modules read their inputs through paths relative to the repository root
(``./variable/...``), so every test runs with the repository root as its working directory,
and ``src/`` is put on ``sys.path`` so the ``covsyn`` package imports without installation.
"""

from __future__ import annotations

import concurrent.futures
import os
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUN10_BEST = REPO_ROOT / "firefly_result" / "phaseD_run10" / "firefly_best.txt"

for entry in (REPO_ROOT / "src", REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


@pytest.fixture(scope="session", autouse=True)
def _run_from_repo_root() -> Iterator[None]:
    """Run every test from the repository root, where the relative input paths resolve."""
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(previous)


def best_vector(path: Path = RUN10_BEST) -> np.ndarray:
    """Return the lowest-cost parameter vector of a firefly_best.txt file."""
    result = np.atleast_2d(np.loadtxt(path))
    return result[int(np.argmin(result[:, -1])), 1:-1].copy()


@pytest.fixture(scope="session")
def run10_vector() -> np.ndarray:
    """The best parameter vector of Phase D run 10 (199 values)."""
    return best_vector()


@pytest.fixture(scope="session")
def demographic_parameters() -> Any:
    """The demographic inputs every simulation samples from."""
    with open(REPO_ROOT / "variable" / "demographic_parameters.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def cheng_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheng et al. (2020) contact counts, attack rates and bin weights used by the objective."""
    with open(REPO_ROOT / "variable" / "processed_contact_tracing_data.pkl", "rb") as f:
        tracing = pickle.load(f)
    return tracing["Cheng_contact_array"], tracing["Cheng_attack_rate"], tracing["norm_weights"]


@pytest.fixture(scope="session")
def cost_pool(demographic_parameters: Any) -> Iterator[concurrent.futures.ProcessPoolExecutor]:
    """A worker pool initialised the way the optimizer initialises it (finding E65)."""
    from covsyn.calibration import fast_cost
    columns = np.load(REPO_ROOT / "variable" / "Taiwan_data_matrix.npy").shape[1]
    workers = min(8, os.cpu_count() or 1)
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=fast_cost.init_worker,
        initargs=(demographic_parameters, columns),
    )
    yield pool
    pool.shutdown()
