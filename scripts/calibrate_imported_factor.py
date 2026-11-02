"""Calibrate the transmission factor of the imported first-wave seeds in the community (B78 C).

In the 2020-04-12 scenario the 118 imported seeds living in the community (69 found there and
49 recalled into home quarantine on 2020-03-18, B73) infected 12 local cases in the data
(validation_reference/taiwan_local_case_sources.csv), 0.102 each; the run 19 model gives about
0.32 each. The factor q multiplies their infection probabilities until they enter home
quarantine (the breach probability applies from then on, B74). The script runs the scenario with
q = 1, takes the mean number these seeds infect, m(1), and sets q = 12 / m(1); a second step
multiplies q by 12 / m(q) and the last set of runs checks it, as calibrate_quarantine_breach.py
does for the breach probability.

Usage (repository root):
    python scripts/calibrate_imported_factor.py <firefly_best.txt> [runs] [workers] [isolation]

isolation is the B78 option A setting (model, observed or imported_delay); calibrate with the
one that will be used, since earlier isolation leaves less for the factor to remove.
"""

import concurrent.futures
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from covsyn.model.data_synthesis_main import run_covid  # noqa: E402

LAYERS = ("household", "school", "workplace", "health_care", "municipality")
SEED_START = 710000
SEEDS = "variable/taiwan_first_wave_seeds.csv"
SOURCES = "validation_reference/taiwan_local_case_sources.csv"


def observed_offspring():
    """Local cases infected by the imported seeds living in the community, and their number."""
    seeds = pd.read_csv(SEEDS)
    community = set(seeds.case_id[(seeds.seed_type == "imported") & (seeds.status == "community")])
    infector = pd.to_numeric(pd.read_csv(SOURCES).infector_id, errors="coerce")
    return int(infector.isin(community).sum()), len(community)


def community_offspring(seed, P, dp, factor, isolation):
    """People infected by the imported community seeds of one run."""
    _, _, course, contact = run_covid(seed, P, dp, mode="taiwan_first_outbreak",
                                      imported_factor=factor, imported_isolation=isolation)
    return sum(sum(float(np.nansum(np.asarray(x[f"{layer}_effective_contacts"] or [], dtype=float)))
                   for layer in LAYERS)
               for c, x in zip(course, contact)
               if c.get("seed_type") == "imported" and c.get("seed_status") == "community")


def mean_offspring(P, dp, factor, isolation, runs, workers):
    """Mean over runs of the people the community seeds infect, with its 5-95% range."""
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        out = np.array(list(ex.map(community_offspring, range(SEED_START, SEED_START + runs),
                                   [P] * runs, [dp] * runs, [factor] * runs, [isolation] * runs)))
    return out.mean(), np.percentile(out, [5, 95])


def main():
    rows = np.atleast_2d(np.loadtxt(sys.argv[1]))
    P = rows[np.argmin(rows[:, -1]), 1:-1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    isolation = sys.argv[4] if len(sys.argv) > 4 else "model"
    with open("variable/demographic_parameters.pkl", "rb") as f:
        dp = pickle.load(f)
    k, n = observed_offspring()
    m1, band = mean_offspring(P, dp, 1.0, isolation, runs, workers)
    print(f"isolation {isolation}; q = 1: {n} community seeds infect {m1:.2f} "
          f"(5-95% of runs {band[0]:.0f}-{band[1]:.0f}), data {k}; {runs} runs")
    q = k / m1
    for step in range(2):   # the second step corrects the departure from q * m(1)
        mq, band = mean_offspring(P, dp, q, isolation, runs, workers)
        print(f"q = {q:.4f} gives {mq:.2f} people (5-95% of runs {band[0]:.0f}-{band[1]:.0f}), "
              f"data {k}")
        if step == 0:
            q *= k / mq


if __name__ == "__main__":
    main()
