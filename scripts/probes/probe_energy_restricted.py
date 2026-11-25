# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Would the restricted energy term change pilot 1's choice of S?

The 2026-11-16 proposal for E105-E107 replaces the energy term (log distances over nine
columns with missing values) by an energy distance on age and onset to confirmation only, over
the cases that record both, with plain distances, scaled by the Taiwanese side's mean and SD.
This re-scores every evaluation of pilot 1 on its own seeds, checks each cost against pilot 1's
blocks.csv, keeps the old term and the new one (half the V- and the U-statistic, the halves
the present code's convention uses), and makes pilot 1's choice again with the new term in
place of the old, alone and with the sparse household bin dropped as well.

Usage: uv run python scripts/probes/probe_energy_restricted.py OUT_DIR PILOT1_BLOCKS_CSV
    [--sparse-bins BINS_CSV] [--workers N]
Writes OUT_DIR/energy.csv, one row per evaluation, and OUT_DIR/summary.json. OUT_DIR/energy.csv
must not exist yet: a rerun goes to a new directory. Re-scoring requires the historical
objective that charged energy. The B95 objective reports energy separately.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import probe_sparse_bins as psb
from scipy.spatial.distance import cdist

from covsyn.calibration import block_scoring, fast_cost, pilot_noise
from covsyn.calibration import firefly_optimizer as fo
from covsyn.runtime import timings

VISIBLE = [0, 7]  # age, onset to confirmation: the two columns both sides mostly record
TAIWAN_FULL = "./variable/Taiwan_data_matrix_full.npy"  # all 578 cases, no ">= 3 fields" filter
COLUMNS = (*pilot_noise.COLUMNS, "energy_old", "energy_v", "energy_u", "simulated_cases")
_TAIWAN: np.ndarray | None = None  # each worker's complete Taiwanese cases, read once


def complete_cases(rows: np.ndarray) -> np.ndarray:
    """The age and onset-to-confirmation of the cases that record both."""
    visible = np.asarray(rows, dtype=float)[:, VISIBLE]
    return visible[~np.isnan(visible).any(axis=1)]


def _distinct(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(a, axis=0, return_counts=True)
    return values, counts.astype(float)


def energy_halves(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Half the energy distance's V- and U-statistic between x and y, rows as points.

    The pair sums are taken over the distinct rows weighted by their counts, which is exact,
    since a pair of equal rows adds 0 either way. The U form needs two rows on each side and is
    NaN otherwise; both are NaN when a side is empty.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    (xv, xw), (yv, yw) = _distinct(x), _distinct(y)
    sxy = float(xw @ cdist(xv, yv) @ yw)
    sxx = float(xw @ cdist(xv, xv) @ xw)  # ordered pairs; a row with itself adds 0
    syy = float(yw @ cdist(yv, yv) @ yw)
    half_v = sxy / (n * m) - sxx / (2 * n * n) - syy / (2 * m * m)
    half_u = (sxy / (n * m) - sxx / (2 * n * (n - 1)) - syy / (2 * m * (m - 1))
              if n > 1 and m > 1 else float("nan"))  # fmt: skip
    return half_v, half_u


def restricted_energy(taiwan: np.ndarray, simulated: np.ndarray) -> tuple[float, float]:
    """energy_halves on complete cases, both sides scaled by the Taiwanese mean and SD."""
    mean, sd = taiwan.mean(axis=0), taiwan.std(axis=0)
    return energy_halves((taiwan - mean) / sd, (simulated - mean) / sd)


def with_energy(
    rows: list[dict], column: str, *, cost_includes_energy: bool | None = None
) -> list[dict]:
    """The rows with each cost's old energy term replaced by the one in ``column``.

    The caller must confirm that the archived total cost includes energy. A failed
    evaluation (old term NaN) keeps its charge.
    """
    if cost_includes_energy is not True:
        raise ValueError(
            "energy replacement requires confirmation that the total cost includes energy"
        )
    out = []
    for r in rows:
        old = float(r["energy_old"])
        if np.isnan(old):
            out.append({**r, "cost": float(r["cost"])})
            continue
        out.append({**r, "cost": float(r["cost"]) - old + float(r[column])})
    return out


def _scored_cases(rows) -> np.ndarray:
    """Convert case records after excluding the failed-row diagnostic representation."""
    if any(isinstance(row, str) for row in rows):
        raise ValueError("a simulated case has no test-matrix row")
    return complete_cases(np.array(rows, dtype=float))


def score_energy(P, seed_offset):
    """One pool task: the cost, its old energy term, and the restricted term's two forms."""
    global _TAIWAN
    if _TAIWAN is None:
        _TAIWAN = complete_cases(np.load(TAIWAN_FULL))
    clock = timings.TaskClock(False, fast_cost._WORKER_PROCESS)
    try:
        cost, parts, pooled, _ = fast_cost._simulate_and_score(P, seed_offset, clock)
        simulated = _scored_cases(pooled[2])
    except Exception:
        traceback.print_exc()
        return fo.FAILED_EVALUATION_COST, float("nan"), float("nan"), float("nan"), -1
    half_v, half_u = restricted_energy(_TAIWAN, simulated)
    return cost, parts["cost_energy"], half_v, half_u, len(simulated)


def _key(r: dict) -> tuple:
    return (r["kind"], int(r["row"]), int(r["simulations"]), int(r["block"]))


def _by_vector(rows: list[dict], column: str, reduce) -> dict[str, float]:
    """A column reduced over each vector's blocks, per S, keyed label@S."""
    groups: dict[str, list[float]] = {}
    for r in rows:
        if r["kind"] == "vector":
            groups.setdefault(f"{r['label']}@{r['simulations']}", []).append(float(r[column]))
    return {k: float(reduce(v)) for k, v in sorted(groups.items())}


def _spread_sd(rows: list[dict], column: str) -> float:
    """The SD across the population's rows of each row's mean over the spread blocks."""
    groups: dict[int, list[float]] = {}
    for r in rows:
        if r["kind"] == "spread":
            groups.setdefault(int(r["row"]), []).append(float(r[column]))
    return float(np.std([np.mean(v) for v in groups.values()], ddof=1))


def summarise(
    rows: list[dict], sparse: dict | None, *, cost_includes_energy: bool | None = None
) -> dict[str, Any]:
    """Summarise historical totals that include the energy term being replaced."""
    energy_v = with_energy(rows, "energy_v", cost_includes_energy=cost_includes_energy)
    energy_u = with_energy(rows, "energy_u", cost_includes_energy=cost_includes_energy)
    scored = [r for r in rows if not np.isnan(float(r["energy_old"]))]
    out: dict[str, Any] = {
        "evaluations": len(rows),
        "failed": len(rows) - len(scored),
        "as_pilot_1": pilot_noise.summarise(rows)["choice"],
        "energy_v": pilot_noise.summarise(energy_v)["choice"],
        "energy_u": pilot_noise.summarise(energy_u)["choice"],
        "sd_old": _by_vector(scored, "energy_old", lambda v: np.std(v, ddof=1)),
        "sd_u": _by_vector(scored, "energy_u", lambda v: np.std(v, ddof=1)),
        "mean_old": _by_vector(scored, "energy_old", np.mean),
        "mean_u": _by_vector(scored, "energy_u", np.mean),
        "spread_sd_old": _spread_sd(scored, "energy_old"),
        "spread_sd_u": _spread_sd(scored, "energy_u"),
        "simulated_cases_min": min(int(r["simulated_cases"]) for r in scored),
    }
    if sparse is not None:
        term = f"term_{psb.SPARSE[0]}_{psb.SPARSE[1]}"
        both = [
            {**r, "cost": float(r["cost"]) - float(sparse[_key(r)][term])}
            if not np.isnan(float(r["energy_old"]))
            else r
            for r in energy_u
        ]
        out["energy_u_and_no_sparse_bin"] = pilot_noise.summarise(both)["choice"]
    return out


def main() -> None:
    """Re-score pilot 1 keeping both energy terms, then make its choice with the new one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("pilot_blocks", type=Path)
    parser.add_argument("--sparse-bins", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if not fast_cost.COST_INCLUDES_ENERGY:
        parser.error(
            "This probe is disabled since B95 (aeb0305). "
            "To reproduce its results, check out a commit before aeb0305."
        )
    pilot = psb.read_pilot(args.pilot_blocks)
    sparse = None
    if args.sparse_bins is not None:
        sparse = {_key(r): r for r in psb.read_done(args.sparse_bins).values()}
    population, lower, upper = pilot_noise._population_and_bounds()
    every = pilot_noise.tasks(pilot_noise.pilot_rows(), len(population))
    missing = psb.missing_from_pilot([t.key for t in every], pilot)
    if missing:
        raise SystemExit(f"{args.pilot_blocks} lacks {len(missing)} of pilot 1's evaluations")
    if sparse is not None and any(t.key not in sparse for t in every):
        raise SystemExit(f"{args.sparse_bins} lacks some of pilot 1's evaluations")
    parameters = population * (upper - lower) + lower
    _, make_pool = block_scoring.objective_pools(args.workers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "energy.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; write to a new directory")
    rows = []
    with open(path, "x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for simulations in sorted({t.simulations for t in every}):
            batch = [t for t in every if t.simulations == simulations]
            with make_pool(simulations) as pool:
                futures = {pool.submit(score_energy, parameters[t.row], t.seed_offset): t
                           for t in batch}  # fmt: skip
                for future in concurrent.futures.as_completed(futures):
                    t = futures[future]
                    cost, old, half_v, half_u, cases = future.result()
                    if not np.isclose(cost, pilot[t.key], rtol=1e-9):
                        raise SystemExit(f"{t}: cost {cost}, pilot 1 had {pilot[t.key]}")
                    row = {**t._asdict(), "cost": cost, "energy_old": old, "energy_v": half_v,
                           "energy_u": half_u, "simulated_cases": cases}  # fmt: skip
                    writer.writerow(row)
                    out.flush()
                    rows.append({k: str(v) for k, v in row.items()})
    summary = summarise(rows, sparse, cost_includes_energy=fast_cost.COST_INCLUDES_ENERGY)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                               encoding="utf-8")  # fmt: skip
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
