# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 10, item 2: each parameter's spread across the production runs.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 10: run the production
configuration R = 5 times on five optimisation streams; for each parameter report in
parameter_uncertainty.csv the mean, SD, minimum and maximum of the five selected vectors, and
the range within the indistinguishable set of item 1; flag every parameter whose SD across
runs exceeds 10% of its bound width as not identified by the objective.

Decisions of 2026-11-22: the sample SD (ddof = 1) over the bound width, flagged only when
strictly above 10%; a parameter of zero width is fixed and never flagged; the range comes from
the main run's indistinguishable set, named before the runs start.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from covsyn.calibration import parameter_uncertainty as pu


def _selected():
    """Five runs' vectors over four parameters: steady, spread, at the 10% edge, fixed."""
    return np.array([[1.0, 0.0, 0.0, 5.0],
                     [1.1, 4.0, 0.1, 5.0],
                     [0.9, 8.0, 0.2, 5.0],
                     [1.0, 2.0, 0.0, 5.0],
                     [1.0, 6.0, 0.1, 5.0]])  # fmt: skip


LOWER = np.array([0.0, 0.0, 0.0, 5.0])
UPPER = np.array([2.0, 10.0, 1.0, 5.0])


def test_each_parameter_s_mean_sd_minimum_and_maximum_over_the_runs():
    """Over the five selected vectors, the SD with ddof = 1."""
    table = pu.parameter_table(_selected(), LOWER, UPPER, _selected()[:2])
    for line, values in zip(table, _selected().T, strict=True):
        assert line["mean"] == pytest.approx(values.mean())
        assert line["sd"] == pytest.approx(values.std(ddof=1))
        assert (line["min"], line["max"]) == (values.min(), values.max())


def test_the_sd_is_measured_against_the_bound_width():
    """SD / width per parameter; the spread one, SD 3.16 of width 10, is flagged."""
    table = pu.parameter_table(_selected(), LOWER, UPPER, _selected()[:1])
    assert table[1]["sd_fraction"] == pytest.approx(_selected()[:, 1].std(ddof=1) / 10)
    assert [line["not_identified"] for line in table] == [False, True, False, False]


@pytest.mark.parametrize("fraction, flagged", [(0.10, False), (0.1000001, True), (0.05, False)])
def test_a_parameter_is_flagged_only_strictly_above_ten_percent(fraction, flagged):
    """Exactly 10% of the width is not flagged (decision of 2026-11-22)."""
    assert pu.not_identified(fraction) == flagged


def test_a_parameter_with_no_width_is_fixed_and_never_flagged():
    """Its bounds coincide: no SD fraction, no flag."""
    line = pu.parameter_table(_selected(), LOWER, UPPER, _selected()[:2])[3]
    assert line["fixed"] and not line["not_identified"] and np.isnan(line["sd_fraction"])


def test_the_range_comes_from_the_main_run_s_indistinguishable_set():
    """Its minimum and maximum per parameter, apart from the five runs' own."""
    main_set = np.array([[0.95, 3.0, 0.05, 5.0], [1.05, 5.0, 0.15, 5.0], [1.0, 4.0, 0.1, 5.0]])
    table = pu.parameter_table(_selected(), LOWER, UPPER, main_set)
    for line, values in zip(table, main_set.T, strict=True):
        assert (line["set_min"], line["set_max"]) == (values.min(), values.max())
        assert line["set_size"] == 3


def test_the_table_is_written_one_row_per_parameter(tmp_path):
    """parameter_uncertainty.csv, by parameter index."""
    path = tmp_path / "parameter_uncertainty.csv"
    pu.write_table(path, pu.parameter_table(_selected(), LOWER, UPPER, _selected()[:2]))
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["parameter"]) for r in rows] == [0, 1, 2, 3]


@pytest.mark.parametrize("selected, lower, upper, main_set", [
    (np.zeros((4, 2)), np.zeros(2), np.ones(2), np.zeros((1, 2))),  # not five runs
    (np.zeros((5, 2)), np.zeros(3), np.ones(3), np.zeros((1, 2))),  # bounds of another size
    (np.zeros((5, 2)), np.zeros(2), np.ones(2), np.zeros((0, 2))),  # an empty set
])  # fmt: skip
def test_inputs_that_do_not_fit_are_refused(selected, lower, upper, main_set):
    """R = 5 vectors, bounds and set of their dimension, and a set of at least the best."""
    with pytest.raises(ValueError):
        pu.parameter_table(selected, lower, upper, main_set)
