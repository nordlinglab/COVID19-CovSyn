# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which cost terms respond to each parameter, one at a time, at a few base vectors?

The screen behind the decision whether the energy term, or the Cheng terms (contacts and attack
rates), can leave the objective (professor, 2026-11-23): a parameter that only those terms
respond to would be left unconstrained without them. Design reviewed by a second AI agent
(logic review of 2026-11-23).

Bases: run 19, the literature vector, the lowest-cost row of pilot 1's population spread, the
low-cost row farthest from it, and a middle-cost row. Each free parameter moves STEP of its
normalised range towards the interior. Every point is evaluated on the same R replications
(blocks of validation stream STREAM at S), keeping every cost term, every outcome target's
measured value and the restricted energy term. ``--analyse`` turns the table into paired
effects with t intervals against weighted-cost thresholds.

Usage:
    uv run python scripts/probes/probe_term_sensitivity.py run OUT_CSV PILOT1_BLOCKS_CSV
        [--replications R] [--workers N]
    uv run python scripts/probes/probe_term_sensitivity.py analyse OUT_CSV OUT_JSON
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import probe_energy_restricted as per
import probe_sparse_bins as psb
from scipy import stats

from covsyn.calibration import block_scoring, fast_cost, pilot_noise, seed_streams
from covsyn.calibration import firefly_optimizer as fo
from covsyn.runtime import timings

STEP = 0.2
S = 3000
STREAM = 250  # a validation stream no other use has taken (Task 10 keeps 203-207)
DELTAS = (0.05, 0.1, 0.2)  # weighted-cost thresholds; 0.1 is the primary one
_TAIWAN = None


def perturbed(x: np.ndarray, i: int, step: float) -> np.ndarray:
    """A copy of x with coordinate i moved by step towards the interior of [0, 1]."""
    out = np.array(x, dtype=float)
    out[i] = out[i] + step if out[i] <= 0.5 else out[i] - step
    return out


def pilot_bases(x, costs, exclude, low_share=0.1, middle_quantile=0.5):
    """Three rows: the lowest cost, the low-cost row farthest from it, a middle-cost row.

    Low-cost rows are the lowest ``low_share`` of the rows not excluded; the middle row is the
    remaining row whose cost is closest to that quantile of the rows not excluded.
    """
    rows = [r for r in range(len(costs)) if r not in set(exclude)]
    ranked = sorted(rows, key=lambda r: costs[r])
    low = ranked[: max(2, math.ceil(low_share * len(rows)))]
    first = low[0]
    far = max(low[1:], key=lambda r: float(np.linalg.norm(x[r] - x[first])))
    target = float(np.quantile([costs[r] for r in rows], middle_quantile))
    rest = [r for r in rows if r not in (first, far)]
    middle = min(rest, key=lambda r: abs(costs[r] - target))
    return first, far, middle


def classify(differences, delta: float) -> str:
    """'beyond', 'within' or 'imprecise': the 95% t interval of the mean against +-delta."""
    d = np.asarray(differences, dtype=float)
    half = (
        0.0
        if d.std(ddof=1) == 0
        else stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    )
    low, high = d.mean() - half, d.mean() + half
    if low > delta or high < -delta:
        return "beyond"
    if -delta <= low and high <= delta:
        return "within"
    return "imprecise"


def score(P, seed_offset):
    """One pool task: every term, every outcome target's measured value, the restricted energy."""
    global _TAIWAN
    if _TAIWAN is None:
        _TAIWAN = per.complete_cases(np.load(per.TAIWAN_FULL))
    clock = timings.TaskClock(False, fast_cost._WORKER_PROCESS)
    try:
        cost, parts, pooled, _ = fast_cost._simulate_and_score(P, seed_offset, clock)
    except Exception:
        traceback.print_exc()
        return None
    try:
        simulated = per._scored_cases(pooled[2])
        _, energy_u = per.restricted_energy(_TAIWAN, simulated)
    except Exception:
        traceback.print_exc()
        energy_u = float("nan")
    return {"cost": cost, **{k: v for k, v in parts.items() if isinstance(v, float)},
            "energy_u": energy_u}  # fmt: skip


def run(args) -> None:
    """Evaluate every base and every one-at-a-time step on the same replications."""
    population, lower, upper = pilot_noise._population_and_bounds()
    pilot = psb.read_pilot(args.pilot_blocks)
    spread = {row: np.mean([c for (kind, r, s, b), c in pilot.items()
                            if kind == "spread" and r == row])
              for row in range(len(population))}  # fmt: skip
    costs = np.array([spread[r] for r in range(len(population))])
    bases = {"run19": 2, "literature": 0}
    first, far, middle = pilot_bases(population, costs, exclude=(0, 1, 2))
    bases.update({"low_a": first, "low_b": far, "middle": middle})
    free = [i for i in range(population.shape[1]) if upper[i] > lower[i]]
    points = [(name, row, -1) for name, row in bases.items()]
    points += [(name, row, i) for name, row in bases.items() for i in free]
    path = Path(args.out)
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            done = {(r["base"], int(r["param"]), int(r["replication"])) for r in csv.DictReader(f)}
    _, make_pool = block_scoring.objective_pools(args.workers)
    offsets = [seed_streams.block_seeds(STREAM, b, S).start for b in range(args.replications)]
    columns = None
    with make_pool(S) as pool, open(path, "a", newline="", encoding="utf-8") as out:
        futures = {}
        for name, row, i in points:
            x = population[row] if i < 0 else perturbed(population[row], i, STEP)
            P = x * (upper - lower) + lower
            for rep, off in enumerate(offsets):
                if (name, i, rep) not in done:
                    futures[pool.submit(score, P, off)] = (name, row, i, rep)
        print(f"{len(futures)} evaluations to run; bases {bases}", flush=True)
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            name, row, i, rep = futures[future]
            result = future.result()
            if result is None:
                continue
            record = {"base": name, "row": row, "param": i, "replication": rep, **result}
            if columns is None:
                columns = list(record)
                writer = csv.DictWriter(out, fieldnames=columns)
                if not done:
                    writer.writeheader()
            writer.writerow(record)
            out.flush()
            if n % 200 == 0:
                print(f"{n}/{len(futures)}", flush=True)


def _terms(r: dict) -> dict[str, float]:
    """The weighted terms the decision is about, and each outcome target's weighted penalty."""
    terms = {"contact": r["cost_contact"], "attack": r["cost_attack_rate"],
             "energy_old": r["cost_energy"], "energy_new": r["energy_u"],
             "outcome": r["cost_outcome"], "physiology": r["cost_penalty"]}  # fmt: skip
    for name in fo.OUTCOME_TARGETS:
        key = f"measured_{name}"
        if key in r:
            terms[f"target_{name}"] = fo.outcome_penalty({name: r[key]})
    return terms


def analyse(args) -> None:
    """Paired effects per base, parameter and term, classified against each threshold."""
    with open(args.out, encoding="utf-8") as f:
        rows = [
            {k: (v if k == "base" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)
        ]
    by = {}
    for r in rows:
        by.setdefault((r["base"], int(r["param"])), {})[int(r["replication"])] = _terms(r)
    result: dict[str, Any] = {}
    for (base, i), reps in by.items():
        if i < 0:
            continue
        ref = by[(base, -1)]
        common = sorted(set(reps) & set(ref))
        if len(common) < 2:
            continue
        entry = {}
        for term in reps[common[0]]:
            d = [reps[k][term] - ref[k][term] for k in common]
            classes = {f"class_{delta}": classify(d, delta) for delta in DELTAS}
            entry[term] = {"mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)), **classes}
        result[f"{base}|{i}"] = entry
    Path(args.json).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"{len(result)} base-parameter pairs analysed", flush=True)


def main() -> None:
    """Run the screen or analyse its table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("pilot_blocks", type=Path)
    r.add_argument("--replications", type=int, default=5)
    r.add_argument("--workers", type=int, default=None)
    a = sub.add_parser("analyse")
    a.add_argument("out")
    a.add_argument("json")
    args = parser.parse_args()
    run(args) if args.command == "run" else analyse(args)


if __name__ == "__main__":
    main()
