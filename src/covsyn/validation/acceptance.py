# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Acceptance verdicts that account for Monte-Carlo noise (decision B57, finding E89).

Each check is measured on 1,000 simulations, and some rest on few events: run 12's candidate 26
failed the latent period at 0.37 standard errors below its own parameter mean, and the
company-size ratio on 19 against 22 infections. A check therefore gets a bootstrap 95% interval
over the simulations, and a value outside the target whose interval still reaches it is
reported as 'within noise' rather than as a failure.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np


def verdict(value: float, target: tuple[float, float],
            interval: tuple[float, float] | None) -> str | None:
    """Judge a value against a target band, allowing for its sampling interval.

    Args:
        value: The measured value.
        target: (lower, upper) acceptance band.
        interval: The value's 95% interval, or None when it has none.

    Returns:
        'ok' inside the band, 'within' outside it but with an interval that overlaps it,
        'fail' otherwise, None for a missing value.
    """
    if value is None or not np.isfinite(value):
        return None
    lower, upper = target
    if lower <= value <= upper:
        return 'ok'
    if interval is not None and interval[0] <= upper and interval[1] >= lower:
        return 'within'
    return 'fail'


def bootstrap_intervals(compute: Callable[[Sequence[Any]], dict[Hashable, float]],
                        units: Sequence[Any], replicates: int = 200, seed: int = 0,
                        level: float = 0.95) -> dict[Hashable, tuple[float, float]]:
    """Percentile bootstrap intervals of every value compute() returns.

    Args:
        compute: Maps a list of units (simulations) to {key: value}.
        units: The units to resample with replacement.
        replicates: Number of bootstrap samples.
        seed: Seed of the resampling, for reproducible intervals.
        level: Coverage of the intervals.

    Returns:
        {key: (lower, upper)} for every key with at least two finite replicates.
    """
    rng = np.random.default_rng(seed)
    collected: dict[Hashable, list[float]] = {}
    for _ in range(replicates):
        sample = [units[i] for i in rng.integers(0, len(units), len(units))]
        for key, value in compute(sample).items():
            if value is not None and np.isfinite(value):
                collected.setdefault(key, []).append(float(value))
    tail = 100 * (1 - level) / 2
    return {key: (float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail)))
            for key, v in collected.items() if len(v) >= 2}
