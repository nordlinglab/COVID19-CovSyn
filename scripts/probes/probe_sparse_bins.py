# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Does a Cheng bin with almost no contacts decide pilot 1's choice of S?

Pilot 1's outlier (run 19, S = 10,000, block 23, cost 34.19 against 12.7-14.6) came from one bin
of the attack-rate term, household contacts 0 to 3 days after onset, which holds 0 to 3 contacts
per 10,000 simulations: 1 infected of 2 added 20.49. This re-scores every evaluation of pilot 1,
on the same seeds, keeping each bin's term, checks each cost against pilot 1's own blocks.csv,
and makes pilot 1's choice again with and without that bin.

Usage: uv run python scripts/probes/probe_sparse_bins.py OUT_DIR PILOT1_BLOCKS_CSV [--workers N]
Writes OUT_DIR/bins.csv, one row per evaluation, appended to as found so a rerun resumes, and
OUT_DIR/summary.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from covsyn.calibration import block_scoring, fast_cost, pilot_noise
from covsyn.calibration import firefly_optimizer as fo
from covsyn.runtime import timings

GROUPS, BINS = 3, 6
TERMS = [f"term_{g}_{b}" for g in range(GROUPS) for b in range(BINS)]
COLUMNS = (*pilot_noise.COLUMNS, "attack", "household_0_3_contacts",
           "household_0_3_infections", *TERMS)  # fmt: skip
SPARSE = (0, 1)  # household, 0 to 3 days after onset


def attack_terms(bins: np.ndarray, cheng: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each (group, bin)'s share of cost_attack_rate, as fast_cost._cost computes it.

    A bin without contacts has rate 0, as in the objective.
    """
    contacts, infections = bins[:, 0].astype(float), bins[:, 1].astype(float)
    rate = np.divide(infections, contacts, out=np.zeros_like(infections), where=contacts != 0)
    peak = np.max(cheng)
    return fo.ATTACK_RATE_WEIGHT * ((rate / peak - cheng / peak) * weights) ** 2


def without_bin(rows: list[dict], group: int, bin_: int) -> list[dict]:
    """The rows with that bin's term taken out of each cost."""
    column = f"term_{group}_{bin_}"
    return [{**r, "cost": float(r["cost"]) - float(r[column])} for r in rows]


def score_bins(P, seed_offset):
    """One pool task: the cost, its attack-rate part and the pooled Cheng bins."""
    simulations = fast_cost._WORKER_SIMULATIONS
    seeds = list(range(seed_offset, seed_offset + simulations))
    clock = timings.TaskClock(False, fast_cost._WORKER_PROCESS)
    results = fast_cost._simulate(seeds, copy.deepcopy(P),
                                  fast_cost._WORKER_TAIWAN_MATRIX_COLUMNS, clock)  # fmt: skip
    pooled = fast_cost._pool([results], simulations)
    cost, parts = fast_cost._cost(P, pooled, simulations, fast_cost._WORKER_TAIWAN_MATRIX,
                                  *fast_cost._WORKER_CHENG)  # fmt: skip
    return cost, parts["cost_attack_rate"], pooled[1]


def main() -> None:
    """Re-score pilot 1 keeping each bin's term, then make its choice with and without one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("pilot_blocks", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "bins.csv"
    with open(args.pilot_blocks, newline="") as f:
        pilot = {(r["kind"], int(r["row"]), int(r["simulations"]), int(r["block"])):
                 float(r["cost"]) for r in csv.DictReader(f)}  # fmt: skip
    done = {}
    if path.exists():
        with open(path, newline="") as f:
            done = {(r["kind"], int(r["row"]), int(r["simulations"]), int(r["block"])): r
                    for r in csv.DictReader(f)}  # fmt: skip
    population, lower, upper = pilot_noise._population_and_bounds()
    parameters = population * (upper - lower) + lower
    _, make_pool = block_scoring.objective_pools(args.workers)
    with open("./variable/processed_contact_tracing_data.pkl", "rb") as f:
        ct = pickle.load(f)
    cheng, weights = np.asarray(ct["Cheng_attack_rate"]), np.asarray(ct["norm_weights"])
    todo = [t for t in pilot_noise.tasks(pilot_noise.pilot_rows(), len(population))
            if t.key not in done]  # fmt: skip
    new = not path.exists()
    with open(path, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        for simulations in sorted({t.simulations for t in todo}):
            batch = [t for t in todo if t.simulations == simulations]
            with make_pool(simulations) as pool:
                futures = {pool.submit(score_bins, parameters[t.row], t.seed_offset): t
                           for t in batch}  # fmt: skip
                for future in concurrent.futures.as_completed(futures):
                    t = futures[future]
                    cost, attack, bins = future.result()
                    terms = attack_terms(bins, cheng, weights)
                    if not np.isclose(terms.sum(), attack, rtol=1e-9, atol=1e-12):
                        raise SystemExit(f"{t}: bins give {terms.sum()}, the objective {attack}")
                    if t.key in pilot and not np.isclose(cost, pilot[t.key], rtol=1e-9):
                        raise SystemExit(f"{t}: cost {cost}, pilot 1 had {pilot[t.key]}")
                    row = {**t._asdict(), "cost": cost, "attack": attack,
                           "household_0_3_contacts": int(bins[0, 0, 1]),
                           "household_0_3_infections": int(bins[0, 1, 1]),
                           **dict(zip(TERMS, terms.ravel(), strict=True))}  # fmt: skip
                    writer.writerow(row)
                    out.flush()
                    done[t.key] = row
    rows = list(done.values())
    summary = {"evaluations": len(rows), "checked_against_pilot_1": sum(k in pilot for k in done),
               "as_pilot_1": pilot_noise.summarise(rows)["choice"],
               "without_household_0_3": pilot_noise.summarise(without_bin(rows, *SPARSE))["choice"],
               "household_0_3": {
                   str(s): {"evaluations": sum(int(r["simulations"]) == s for r in rows),
                            "with_an_infection": sum(int(r["simulations"]) == s and
                                                     int(r["household_0_3_infections"]) > 0
                                                     for r in rows),
                            "largest_term": max(float(r[f"term_{SPARSE[0]}_{SPARSE[1]}"])
                                                for r in rows if int(r["simulations"]) == s)}
                   for s in sorted({int(r["simulations"]) for r in rows})}}  # fmt: skip
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
