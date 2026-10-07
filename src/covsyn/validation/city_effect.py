# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Is there a city effect in the community contact counts (B56, finding E83)?

Decision B27 removed the city population from the community contact count, and the check of
B27 bounded the ratio of the largest to the smallest city mean at 1.6. Over about 15 cities
with 20-30 index cases each and a heavy-tailed count, that ratio is mostly noise: shuffling the
city labels, which removes any city effect, still gives a median of 2.25 and fails the bound in
96.6% of shuffles (E83). The check therefore compares the observed ratio with its distribution
under shuffled labels.
"""

from __future__ import annotations

import numpy as np


def max_min_city_ratio(values: np.ndarray, cities: np.ndarray, min_cases: int = 20) -> float:
    """Largest over smallest city mean, over the cities with at least min_cases cases.

    Args:
        values: One number per case (community contacts).
        cities: The case's city, same length.
        min_cases: Cities with fewer cases are left out.

    Returns:
        The ratio, or NaN with fewer than two such cities.
    """
    values = np.asarray(values, dtype=float)
    cities = np.asarray(cities)
    names, counts = np.unique(cities, return_counts=True)
    means = [values[cities == name].mean() for name in names[counts >= min_cases]]
    if len(means) < 2:
        return float("nan")
    return float(max(means) / max(min(means), 1e-9))


def permutation_p_value(
    values: np.ndarray,
    cities: np.ndarray,
    permutations: int = 2000,
    seed: int = 0,
    min_cases: int = 20,
) -> float:
    """Share of label shuffles whose ratio is at least the observed one.

    Args:
        values: One number per case (community contacts).
        cities: The case's city, same length.
        permutations: Number of shuffles.
        seed: Seed of the shuffles, for a reproducible p-value.
        min_cases: As for max_min_city_ratio.

    Returns:
        (1 + shuffles with a ratio >= the observed) / (1 + permutations); a small value means
        the cities differ more than shuffled labels do.
    """
    observed = max_min_city_ratio(values, cities, min_cases)
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    cities = np.asarray(cities)
    at_least = sum(
        max_min_city_ratio(values, rng.permutation(cities), min_cases) >= observed
        for _ in range(permutations)
    )
    return (1 + at_least) / (1 + permutations)
