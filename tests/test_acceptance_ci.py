# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Acceptance verdicts that account for Monte-Carlo noise (finding E89).

A check passes outright when its value lies in the target; it passes as 'within noise' when the
value lies outside but its bootstrap 95% interval overlaps the target; it fails only when the
whole interval lies outside. Run 12's candidate 26 failed the latent period (4.087 against 4.1,
0.37 standard errors below its own parameter mean) and the company-size ratio (4.09 against 3,
bootstrap interval 1.92-8.28) on noise alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.validation import acceptance as ac


@pytest.mark.parametrize("value,ci,expected", [
    (5.0, (4.0, 6.0), 'ok'),            # inside
    (4.087, (3.96, 4.22), 'within'),    # outside, interval overlaps [4.1, 5.5]
    (3.5, (3.3, 3.7), 'fail'),          # interval entirely below
    (6.0, (5.6, 6.4), 'fail'),          # interval entirely above
    (5.6, (5.4, 5.8), 'within'),        # above, interval reaches the upper bound
    (4.0, None, 'fail'),                # no interval: judged on the value alone
    (4.5, None, 'ok'),
])
def test_verdict(value: float, ci: tuple[float, float] | None, expected: str) -> None:
    assert ac.verdict(value, (4.1, 5.5), ci) == expected


def test_verdict_of_a_missing_value_is_none() -> None:
    assert ac.verdict(float('nan'), (4.1, 5.5), (4.0, 4.2)) is None


def test_bootstrap_intervals_cover_the_mean_and_shrink_with_more_data() -> None:
    rng = np.random.default_rng(0)

    def compute(sample):
        values = np.asarray(sample, dtype=float)
        return {('B0', 'mean'): float(values.mean()), ('B0', 'only sometimes'):
                float(values.max()) if len(values) > 3 else float('nan')}

    small = list(rng.normal(10, 2, 50))
    large = list(rng.normal(10, 2, 2000))
    ci_small = ac.bootstrap_intervals(compute, small, replicates=300, seed=1)[('B0', 'mean')]
    ci_large = ac.bootstrap_intervals(compute, large, replicates=300, seed=1)[('B0', 'mean')]
    assert ci_small[0] < 10 < ci_small[1]
    assert (ci_large[1] - ci_large[0]) < (ci_small[1] - ci_small[0])


def test_bootstrap_is_reproducible() -> None:
    data = list(range(100))

    def compute(sample):
        return {('B0', 'mean'): float(np.mean(sample))}

    a = ac.bootstrap_intervals(compute, data, replicates=100, seed=5)
    b = ac.bootstrap_intervals(compute, data, replicates=100, seed=5)
    assert a == b


def test_bootstrap_leaves_out_values_that_are_never_finite() -> None:
    def compute(sample):
        return {('B0', 'never'): float('nan'), ('B0', 'always'): 1.0}

    intervals = ac.bootstrap_intervals(compute, [1, 2, 3], replicates=20, seed=0)
    assert ('B0', 'never') not in intervals
    assert intervals[('B0', 'always')] == (1.0, 1.0)
