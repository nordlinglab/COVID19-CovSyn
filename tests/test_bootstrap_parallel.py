# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""S1: the bootstrap replicates of the checklist run in parallel without changing any value.

The checklist spent 68 of its 66-69 minutes on 200 serial calls of compute_checks (measured on
run 20, 20.4 s each), on one core of 32. The replicates are independent, so they can run in a
pool; the values must not change. This fixes the pieces that make that possible: the resampling
indices are drawn once, in the parent, in the order bootstrap_intervals draws them, and the
interval arithmetic is shared by the serial and the parallel path.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.validation import acceptance as ac


def test_index_sets_match_the_draws_of_the_serial_bootstrap():
    """The helper reproduces bootstrap_intervals' own resampling, list by list and in order."""
    first, second = [10, 20, 30, 40], [1, 2]
    seen: list[tuple[list[int], list[int]]] = []

    def compute(a, b):
        seen.append((list(a), list(b)))
        return {("B0", "sum"): float(sum(a) + sum(b))}

    ac.bootstrap_intervals(compute, first, second, replicates=7, seed=3)
    sets = ac.bootstrap_index_sets((len(first), len(second)), replicates=7, seed=3)
    assert len(sets) == 7
    rebuilt = [([first[i] for i in a], [second[i] for i in b]) for a, b in sets]
    assert rebuilt == seen


def test_empty_unit_list_consumes_no_draw():
    """bootstrap_intervals skips the draw for an empty list; the helper must skip it too."""
    data = [1, 2, 3]
    drawn: list[list[int]] = []

    def compute(a, b):
        drawn.append(list(a))
        return {("B0", "mean"): float(np.mean(a))}

    ac.bootstrap_intervals(compute, data, [], replicates=5, seed=11)
    sets = ac.bootstrap_index_sets((len(data), 0), replicates=5, seed=11)
    assert [[data[i] for i in a] for a, _ in sets] == drawn
    assert all(len(b) == 0 for _, b in sets)


def test_percentile_intervals_match_the_serial_bootstrap():
    """The shared aggregation gives bootstrap_intervals' own intervals from its own values."""
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    collected: dict[tuple[str, str], list[float]] = {}

    def compute(sample):
        value = float(np.mean(sample))
        collected.setdefault(("B0", "mean"), []).append(value)
        return {("B0", "mean"): value}

    expected = ac.bootstrap_intervals(compute, data, replicates=50, seed=2)
    assert ac.percentile_intervals(collected) == expected


def test_percentile_intervals_drop_keys_with_fewer_than_two_values():
    assert ac.percentile_intervals({("B0", "a"): [1.0]}) == {}
    assert ("B0", "b") in ac.percentile_intervals({("B0", "b"): [1.0, 2.0]})


def test_replicate_values_are_collected_in_replicate_order():
    """A worker pool returns out of order, so the collector must key on the replicate number."""
    out = ac.collect_replicates({2: {("B0", "x"): 3.0}, 0: {("B0", "x"): 1.0},
                                 1: {("B0", "x"): 2.0}})
    assert out == {("B0", "x"): [1.0, 2.0, 3.0]}


def test_collect_replicates_leaves_out_non_finite_values():
    """Same rule as the serial loop: None and non-finite values are not collected."""
    out = ac.collect_replicates({0: {("B0", "x"): 1.0, ("B0", "y"): None},
                                 1: {("B0", "x"): float("nan"), ("B0", "y"): 2.0}})
    assert out == {("B0", "x"): [1.0], ("B0", "y"): [2.0]}


def test_a_missing_replicate_is_an_error():
    """A failed worker must fail the checklist, not silently shorten the bootstrap."""
    with pytest.raises(ValueError):
        ac.collect_replicates({0: {("B0", "x"): 1.0}, 2: {("B0", "x"): 2.0}})
