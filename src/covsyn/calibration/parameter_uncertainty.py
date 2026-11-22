# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Each parameter's spread across the production runs (Task 10, item 2).

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 10: run the production
configuration R = 5 times on five optimisation streams; for each parameter report the mean, SD,
minimum and maximum of the five selected vectors, and the range within the indistinguishable
set of item 1; flag a parameter whose SD across runs exceeds 10% of its bound width as not
identified by the objective (a value drawn uniformly over the bounds has an SD of 29% of it).

Decisions of 2026-11-22: the sample SD (ddof = 1) over the bound width, flagged only when
strictly above 10%; a parameter of zero width is fixed and never flagged; the range comes from
the indistinguishable set of the main run, named before the runs start.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

RUNS = 5
# The 10% threshold, a starting point the plan gives; Task 10's own measurement must back it
# (plan section 1.1), which docs/covsyn_decisions.md records.
NOT_IDENTIFIED_FRACTION = 0.10


def not_identified(sd_fraction: float) -> bool:
    """Whether an SD of this fraction of the bound width marks a parameter as not identified."""
    return bool(sd_fraction > NOT_IDENTIFIED_FRACTION)


def parameter_table(selected: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                    main_set: np.ndarray) -> list[dict[str, Any]]:
    """One row per parameter: the runs' spread, against its bounds, and the main set's range.

    Args:
        selected: The R = 5 runs' selected vectors, one per row.
        lower: The parameters' lower bounds.
        upper: The parameters' upper bounds.
        main_set: The main run's candidates that cannot be told apart from its best, one per
            row, the best among them.

    Raises:
        ValueError: Not R vectors, bounds or a set of another dimension, or an empty set.
    """
    selected = np.asarray(selected, dtype=float)
    main_set = np.asarray(main_set, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if selected.ndim != 2 or len(selected) != RUNS:
        raise ValueError(f"{RUNS} selected vectors are needed, one per run, not {len(selected)}")
    dim = selected.shape[1]
    if lower.shape != (dim,) or upper.shape != (dim,):
        raise ValueError(f"bounds of {lower.shape} and {upper.shape} for {dim} parameters")
    if main_set.ndim != 2 or main_set.shape[1] != dim or not len(main_set):
        raise ValueError(f"the main run's set must hold at least its best, of {dim} parameters")
    width = upper - lower
    table = []
    for i in range(dim):
        values = selected[:, i]
        sd = float(values.std(ddof=1))
        fixed = bool(width[i] == 0)
        fraction = float("nan") if fixed else sd / float(width[i])
        table.append({
            "parameter": i, "lower": float(lower[i]), "upper": float(upper[i]),
            "mean": float(values.mean()), "sd": sd,
            "min": float(values.min()), "max": float(values.max()),
            "sd_fraction": fraction, "fixed": fixed,
            "not_identified": False if fixed else not_identified(fraction),
            "set_min": float(main_set[:, i].min()), "set_max": float(main_set[:, i].max()),
            "set_size": len(main_set),
        })  # fmt: skip
    return table


def write_table(path: Path, table: list[dict[str, Any]]) -> None:
    """Write parameter_uncertainty.csv, one row per parameter."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
