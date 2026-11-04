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

# B83 (2026-11-04): acceptance band for the share of cases infecting three or more, measured on
# the population R is measured on. B56 took the band from all 579 first-wave cases, between the
# recorded infector links (0.17%) and the confirmed contacts (4.32%); B67 then moved R onto the
# local index cases (Cheng 2020, 0.62 each, band [0.37, 0.98]) because CovSyn's index case is a
# local case living in the community, but left this share on a population 84% of which was
# imported. The two could not both hold: with R = 0.62 and k = 0.286 a negative binomial puts
# 7.8% at three or more, and no R above 0.40 stays under 4.3% at any accepted k, so the check
# was unsatisfiable whatever the model did. The same two definitions over the 55 local cases of
# validation_reference/taiwan_case_timeline.csv (case_type 'Local') give 1 of 55 = 1.82% by
# links and 7 of 55 = 12.73% by confirmed contacts. The dispersion k keeps the 579-case MLE
# (B61): the local subset gives k = 5.69 by confirmed contacts, which would not constrain the
# dispersion at all.
OFFSPRING_SHARE_GE3 = (1.8, 12.8)


def json_band(target: Any) -> Any:
    """A target band ready for strict JSON: an open (infinite) bound becomes None.

    Args:
        target: A [lower, upper] list, or anything else (None, a string), returned unchanged.

    Returns:
        The band with every non-finite bound replaced by None.
    """
    if not isinstance(target, list):
        return target
    return [x if x is not None and np.isfinite(x) else None for x in target]


def band_from_json(target: Sequence[float | None]) -> tuple[float, float]:
    """(lower, upper) of a band read back from JSON, where an open bound is null.

    Args:
        target: [lower, upper], either None when open.

    Returns:
        The band with an open lower bound as -inf and an open upper bound as +inf.
    """
    lower, upper = target
    return (-np.inf if lower is None else float(lower), np.inf if upper is None else float(upper))


def band_text(target: Any) -> str:
    """'[lower, upper]' for a band, 'inf' for an open bound, '(information)' for none.

    Args:
        target: A [lower, upper] list, either bound None or infinite when open, or None.

    Returns:
        The band as printed in the checklist report.
    """
    if not isinstance(target, list):
        return "(information)"
    return "[%s]" % ", ".join(
        "inf" if x is None or not np.isfinite(x) else "%g" % x for x in target
    )


def verdict(
    value: float, target: tuple[float, float], interval: tuple[float, float] | None
) -> str | None:
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
        return "ok"
    # A single-value target (an anomaly count that must be 0) gets no allowance: a resample
    # that happens to miss the few bad cases would otherwise pass a real defect.
    if interval is not None and lower < upper and interval[0] <= upper and interval[1] >= lower:
        return "within"
    return "fail"


def bootstrap_intervals(
    compute: Callable[..., dict[Hashable, float]],
    *unit_lists: Sequence[Any],
    replicates: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> dict[Hashable, tuple[float, float]]:
    """Percentile bootstrap intervals of every value compute() returns.

    Args:
        compute: Maps one resampled list per unit list to {key: value}.
        *unit_lists: The units to resample with replacement, each list on its own (spread
            and first-outbreak simulations differ in number and must not be paired).
        replicates: Number of bootstrap samples.
        seed: Seed of the resampling, for reproducible intervals.
        level: Coverage of the intervals.

    Returns:
        {key: (lower, upper)} for every key with at least two finite replicates.
    """
    index_sets = bootstrap_index_sets(
        tuple(len(units) for units in unit_lists), replicates=replicates, seed=seed
    )
    values = {}
    for replicate, indices in enumerate(index_sets):
        samples = [
            [units[i] for i in index]
            for units, index in zip(unit_lists, indices, strict=True)
        ]
        values[replicate] = compute(*samples)
    return percentile_intervals(collect_replicates(values), level=level)


def bootstrap_index_sets(
    sizes: Sequence[int], replicates: int = 200, seed: int = 0
) -> list[tuple[np.ndarray, ...]]:
    """The resampling indices of every replicate, in the order the bootstrap draws them.

    One definition of the draw order, used by bootstrap_intervals itself and by a caller that
    evaluates the replicates in a worker pool (S1): drawing every index set in the parent, in
    this order, is what makes the parallel run give the serial run's values. An empty unit list
    consumes no draw, as the serial loop's conditional did.

    Args:
        sizes: Length of each unit list, in the order they are resampled.
        replicates: Number of bootstrap samples.
        seed: Seed of the resampling.

    Returns:
        One tuple of index arrays per replicate, one array per unit list.
    """
    rng = np.random.default_rng(seed)
    empty = np.empty(0, dtype=np.int64)
    return [
        tuple(rng.integers(0, n, n) if n else empty for n in sizes) for _ in range(replicates)
    ]


def collect_replicates(
    values: dict[int, dict[Hashable, float]],
) -> dict[Hashable, list[float]]:
    """The finite value of every key, gathered in replicate order.

    A pool returns the replicates in whatever order they finish, so they are keyed by replicate
    number and sorted here; the serial loop's own rule -- a value is kept only when it is
    neither None nor non-finite -- is applied unchanged.

    Args:
        values: {replicate number: {key: value}}, numbered from 0 without gaps.

    Raises:
        ValueError: A replicate is missing, which would silently shorten the bootstrap.
    """
    expected = set(range(len(values)))
    if set(values) != expected:
        missing = sorted(expected - set(values))
        raise ValueError(f'replicates missing from the bootstrap: {missing}')
    collected: dict[Hashable, list[float]] = {}
    for replicate in sorted(values):
        for key, value in values[replicate].items():
            if value is not None and np.isfinite(value):
                collected.setdefault(key, []).append(float(value))
    return collected


def percentile_intervals(
    collected: dict[Hashable, list[float]], level: float = 0.95
) -> dict[Hashable, tuple[float, float]]:
    """Percentile intervals of the collected bootstrap values, for keys with at least two."""
    tail = 100 * (1 - level) / 2
    return {
        key: (float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail)))
        for key, v in collected.items()
        if len(v) >= 2
    }
