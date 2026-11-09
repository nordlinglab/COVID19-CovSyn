# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where should the mass-event probability P[199] be floored (run 13, after B58)?

Run 13 settled at P[199] = 0.0035, a fifth of run 12's 0.0171. E85 showed that what pushes
the event probability down is the objective's fixed-seed noise, not the fit itself, so the
user decided to floor P[199]. This probe chooses the floor on the CURRENT objective (E86
scaling, B55 expected events, B58 course timing): it holds every other parameter of the run 13
choice and evaluates fast_cost.cost_function on many unseen 300-seed blocks per value, so
each point is the expected cost and its parts rather than one draw.

With --compensate the community attack rate P[170:195] is scaled at each value so that the
infections per index case stay at the base vector's level, as the optimizer can do when the
event risk ratio is locked at 1 (B54); without it the attack rate is held fixed. B59 (floor
0.05) was decided on the compensated probe.

Usage (repository root, PYTHONPATH=src):
    python scripts/probes/probe_event_floor.py BEST_TXT OUT_CSV [BLOCKS] [--compensate]
"""

import concurrent.futures
import csv
import pickle
import subprocess
import sys

import numpy as np

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.cost_parts import LAST

PROBABILITIES = [0.0, 0.0035, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
# Far from the objective's seeds 0..299, the chain's 0..999 and revalidation's 100000+.
SEED_START = 300000
PARTS = [
    "cost_contact_others",
    "cost_contact_household",
    "cost_contact_healthcare",
    "cost_attack_rate",
    "cost_energy",
    "cost_outcome",
    "cost_penalty",
]
MEASURED = [
    "community_median",
    "community_p90",
    "community_tail_ratio",
    "daily_municipality",
    "infections_per_index_municipality",
    "sar_municipality",
    "offspring_k",
]


def evaluate(vector, demo, pool, cheng, blocks):
    """Mean and SD of the cost, its parts and the measured outcomes over the seed blocks."""
    values = {k: [] for k in ["total", *PARTS, *MEASURED]}
    for b in range(blocks):
        offset = SEED_START + b * fo.SIMULATIONS_PER_EVALUATION
        values["total"].append(
            float(fast_cost.cost_function(vector, demo, pool, *cheng, seed_offset=offset))
        )
        for k in PARTS:
            values[k].append(float(LAST.get(k, np.nan)))
        for k in MEASURED:
            values[k].append(float(LAST.get("measured_" + k, np.nan)))
    row = {}
    for k, v in values.items():
        row[k] = float(np.nanmean(v))
        row[k + "_sd"] = float(np.nanstd(v, ddof=1))
    return row


def main():
    """Evaluate every probability on BLOCKS seed blocks and write mean and SD per quantity."""
    args = [a for a in sys.argv[1:] if a != "--compensate"]
    compensate = "--compensate" in sys.argv
    best_path, out_path = args[0], args[1]
    blocks = int(args[2]) if len(args) > 2 else 20
    best = np.atleast_2d(np.loadtxt(best_path))
    base = best[int(np.argmin(best[:, -1])), 1:-1].copy()
    with open("./variable/demographic_parameters.pkl", "rb") as f:
        demo = pickle.load(f)
    with open("./variable/processed_contact_tracing_data.pkl", "rb") as f:
        ct = pickle.load(f)
    cheng = (ct["Cheng_contact_array"], ct["Cheng_attack_rate"], ct["norm_weights"])
    columns = np.load("./variable/Taiwan_data_matrix.npy").shape[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    print(
        f"commit {commit}; base {best_path}; P[199] there {base[199]:.4f}; {blocks} blocks of "
        f"{fo.SIMULATIONS_PER_EVALUATION} seeds from {SEED_START}; compensate {compensate}",
        flush=True,
    )
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns)
    )
    key = "infections_per_index_municipality"
    target = evaluate(base, demo, pool, cheng, blocks)[key] if compensate else np.nan
    rows = []
    for p in PROBABILITIES:
        vector = base.copy()
        vector[199] = p
        factor = 1.0
        if compensate:
            factor = target / evaluate(vector, demo, pool, cheng, blocks)[key]
            vector[170:195] = base[170:195] * factor
        row = {
            "event_probability": p,
            "attack_rate_factor": factor,
            **evaluate(vector, demo, pool, cheng, blocks),
        }
        rows.append(row)
        print(
            f"P[199]={p:.4f} SAR x{factor:.3f} total {row['total']:.3f} "
            f"(sd {row['total_sd']:.3f}) others {row['cost_contact_others']:.3f} outcome "
            f"{row['cost_outcome']:.3f} tail {row['community_tail_ratio']:.2f} inf/idx "
            f"{row[key]:.4f}",
            flush=True,
        )
    pool.shutdown()
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
