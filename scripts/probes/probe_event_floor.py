# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where should the mass-event probability P[199] be floored (run 13, after B58)?

Run 13 settled at P[199] = 0.0035, a fifth of run 12's 0.0171. E85 showed that what pushes
the event probability down is the objective's fixed-seed noise, not the fit itself, so the
user decided to floor P[199]. This probe chooses the floor on the CURRENT objective (E86
scaling, B55 expected events, B58 course timing): it holds every other parameter of the run 13
choice and evaluates fast_cost.cost_function on many unseen 300-seed blocks per value, so
each point is the expected cost and its parts rather than one draw.

Usage (repository root, PYTHONPATH=src):
    python scripts/probes/probe_event_floor.py BEST_TXT OUT_CSV [BLOCKS]
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


def main():
    """Evaluate every probability on BLOCKS seed blocks and write mean and SD per quantity."""
    best_path, out_path = sys.argv[1], sys.argv[2]
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 20
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
        f"{fo.SIMULATIONS_PER_EVALUATION} seeds from {SEED_START}",
        flush=True,
    )
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns)
    )
    rows = []
    for p in PROBABILITIES:
        vector = base.copy()
        vector[199] = p
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
        row = {"event_probability": p}
        for k, v in values.items():
            row[k] = float(np.nanmean(v))
            row[k + "_sd"] = float(np.nanstd(v, ddof=1))
        rows.append(row)
        print(
            f"P[199]={p:.4f} total {row['total']:.3f} (sd {row['total_sd']:.3f}) others "
            f"{row['cost_contact_others']:.3f} outcome {row['cost_outcome']:.3f} tail "
            f"{row['community_tail_ratio']:.2f} daily {row['daily_municipality']:.2f} "
            f"inf/idx {row['infections_per_index_municipality']:.4f}",
            flush=True,
        )
    pool.shutdown()
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
