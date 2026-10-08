# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Charged targets: the acceptance intervals moved inward by their Monte-Carlo noise (B60).

The objective's quadratic outcome penalty is weak near an interval's edge, so a calibrated value
tends to settle just at the edge, where the checklist's estimate from 1000 simulations falls
outside about half the time (run 14: four of six edge items). This is the noisy-constraint
problem of chance-constrained programming (Charnes & Cooper 1959): to satisfy a constraint with
probability 1 - alpha under noise of standard error sigma, tighten it by z(1 - alpha) * sigma.
History matching treats the stochastic model's own variance the same way (Vernon, Goldstein &
Bower 2010; Andrianakis et al. 2015).

For each charged outcome that the checklist also judges, sigma is the standard error of the
checklist's estimate, read from its 95% bootstrap interval (B57): sigma = width / (2 * 1.96).
Each finite, active bound moves inward by 1.645 * sigma (one-sided 95%), capped at a quarter of
the interval's width so that no interval collapses. A lower bound of 0 is not a constraint and
is left in place. The checklist itself keeps the original intervals; only the objective aims
inside them.

Usage (repository root, PYTHONPATH=src):
    python -m covsyn.calibration.target_margins CHECKS_JSON [OUT_JSON]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

Z_ONE_SIDED_95 = 1.645
Z_TWO_SIDED_95 = 1.96
MAX_SHIFT_OF_WIDTH = 0.25
CHARGED_TARGETS_FILE = Path(__file__).with_name("charged_targets.json")

# Objective key -> (checklist decision, checklist name, objective units per checklist unit).
CHECK_FOR_TARGET = {
    "daily_household": ("B25", "contacts per day before onset, household", 1.0),
    "daily_school": ("B25", "contacts per day before onset, school", 1.0),
    "daily_workplace": ("B25", "contacts per day before onset, workplace", 1.0),
    "daily_municipality": ("B25", "contacts per day before onset, municipality", 1.0),
    "daily_health_care": ("B25", "contacts per day before onset, health_care", 1.0),
    "asymptomatic_share": ("B33", "asymptomatic share", 0.01),
    "sar_school": ("B9", "cumulative SAR per contact, school", 0.01),
    "sar_workplace": ("B9", "cumulative SAR per contact, workplace", 0.01),
    "medical_late_share": ("B23", "health care contacts starting 8+ days after onset", 0.01),
    "medical_early_share": ("B23", "health care contacts starting before day 4", 0.01),
    "community_median": ("B27", "community contacts per case, median", 1.0),
    "incubation_mean": ("B22", "incubation period, mean", 1.0),
    "pre_onset_window_mean": ("B22", "pre-onset infectious window, mean", 1.0),
    "pre_onset_zero_share": ("B22", "pre-onset window of zero days", 0.01),
    "closure_after_confirmation_symptomatic": (
        "B28",
        "confirmation to case closure, symptomatic",
        1.0,
    ),
    "closure_after_confirmation_asymptomatic": (
        "B28",
        "confirmation to case closure, asymptomatic",
        1.0,
    ),
    "icu_to_closure": ("B28", "ICU to case closure", 1.0),
    "onset_to_icu": ("B28", "onset to ICU", 1.0),
    "onset_to_confirmation": ("B2", "onset to confirmation, median", 1.0),
    "infections_per_index_household": ("B9", "infections per index case, household", 1.0),
    "infections_per_index_health_care": ("B9", "infections per index case, health_care", 1.0),
    "infections_per_index_municipality": ("B9", "infections per index case, municipality", 1.0),
}


def charged_interval(lo: float, hi: float, sigma: float) -> tuple[float, float]:
    """Move the active bounds of [lo, hi] inward by 1.645 sigma, at most a quarter of the width.

    Args:
        lo: Lower bound of the acceptance interval; 0 is treated as no lower constraint.
        hi: Upper bound of the acceptance interval.
        sigma: Standard error of the checklist's estimate, in the interval's units.

    Returns:
        The charged interval.
    """
    shift = min(Z_ONE_SIDED_95 * sigma, MAX_SHIFT_OF_WIDTH * (hi - lo))
    return (lo + shift if lo != 0 else lo, hi - shift)


def compute_margins(checks: list[dict], targets: dict) -> dict[str, dict]:
    """Charged interval and its derivation for every mapped target.

    Args:
        checks: The 'checks' rows of a phaseD_checks.json.
        targets: OUTCOME_TARGETS, {key: (lo, hi, weight)}.

    Returns:
        {key: {'acceptance', 'charged', 'sigma', 'check'}} in objective units.
    """
    by_name = {(c["decision"], c["name"]): c for c in checks}
    out = {}
    for key, (decision, name, unit) in CHECK_FOR_TARGET.items():
        if key not in targets or targets[key][2] <= 0:
            continue
        row = by_name.get((decision, name))
        if row is None or not row.get("ci"):
            raise ValueError(f"checklist row {decision} {name!r} with an interval is missing")
        ci_lo, ci_hi = row["ci"]
        sigma = (ci_hi - ci_lo) / (2 * Z_TWO_SIDED_95) * unit
        lo, hi = float(targets[key][0]), float(targets[key][1])
        out[key] = {
            "acceptance": [lo, hi],
            "charged": list(charged_interval(lo, hi, sigma)),
            "sigma": sigma,
            "check": f"{decision} {name}",
        }
    return out


def load_charged_bounds(path: Path = CHARGED_TARGETS_FILE) -> dict[str, tuple[float, float]]:
    """The charged intervals by objective key, or {} when the file does not exist."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: (float(v["charged"][0]), float(v["charged"][1])) for k, v in data["targets"].items()}


def main() -> None:
    """Compute the charged intervals from a checklist JSON and write them."""
    from covsyn.calibration.firefly_optimizer import OUTCOME_TARGETS

    checks_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else CHARGED_TARGETS_FILE
    checks = json.loads(checks_path.read_text())["checks"]
    margins = compute_margins(checks, OUTCOME_TARGETS)
    out_path.write_text(
        json.dumps(
            {
                "source": str(checks_path),
                "z": Z_ONE_SIDED_95,
                "max_shift_of_width": MAX_SHIFT_OF_WIDTH,
                "targets": margins,
            },
            indent=1,
        )
        + "\n"
    )
    for key, m in margins.items():
        print(
            f"{key:42s} acceptance {m['acceptance']} -> charged "
            f"[{m['charged'][0]:.4g}, {m['charged'][1]:.4g}] (sigma {m['sigma']:.3g})"
        )


if __name__ == "__main__":
    main()
