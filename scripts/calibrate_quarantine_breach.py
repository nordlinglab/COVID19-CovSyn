"""Calibrate the quarantine breach probability of the first-wave seeds (B74).

A seed in home quarantine meets the layers of a seed outside quarantine (health care excepted)
with every infection probability multiplied by the breach probability p. The script runs the
2020-04-12 scenario (taiwan_first_outbreak, 122 quarantined seeds) with p = 1, takes the mean
number of people a quarantined seed infects, m(1), and sets p = (observed / 122) / m(1) for the
two observed counts: 3 confirmed household infections during quarantine and 5 with the two
unclear ones. The infection probabilities are small, so m(p) is close to p * m(1); a second
step multiplies p by observed / m(p) and the last set of runs checks the result.

Usage (repository root):
    python scripts/calibrate_quarantine_breach.py <firefly_best.txt> [runs] [workers]
"""

import concurrent.futures
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "src")
from covsyn.model.data_synthesis_main import run_covid  # noqa: E402

LAYERS = ("household", "school", "workplace", "health_care", "municipality")
OBSERVED = {"confirmed": 3, "with_unclear": 5}
SEED_START = 700000


def quarantined_offspring(seed, P, dp, breach):
    """People infected by each quarantined seed of one run, and the number of such seeds."""
    _, _, course, contact = run_covid(seed, P, dp, mode="taiwan_first_outbreak",
                                      quarantine_breach=breach)
    infected = [sum(float(np.nansum(np.asarray(x[f"{layer}_effective_contacts"] or [], dtype=float)))
                    for layer in LAYERS)
                for c, x in zip(course, contact) if c.get("seed_status") == "quarantine"]
    return sum(infected), len(infected)


def mean_offspring(P, dp, breach, runs, workers):
    """Mean over runs of the people a quarantined seed infects, with its 5-95% range."""
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(quarantined_offspring, range(SEED_START, SEED_START + runs),
                          [P] * runs, [dp] * runs, [breach] * runs))
    per_run = np.array([s / n for s, n in out])
    return per_run.mean(), np.percentile(per_run, [5, 95]), out[0][1]


def main():
    rows = np.atleast_2d(np.loadtxt(sys.argv[1]))
    P = rows[np.argmin(rows[:, -1]), 1:-1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    with open("variable/demographic_parameters.pkl", "rb") as f:
        dp = pickle.load(f)
    m1, band, n = mean_offspring(P, dp, 1.0, runs, workers)
    print(f"p = 1: {n} quarantined seeds, {m1:.4f} infected per seed "
          f"(5-95% of runs {band[0]:.4f}-{band[1]:.4f}), {runs} runs")
    for name, k in OBSERVED.items():
        p = k / n / m1
        for step in range(2):   # the second step corrects the small departure from p * m(1)
            mp, band, _ = mean_offspring(P, dp, p, runs, workers)
            print(f"{name}: observed {k}/{n} = {k / n:.4f} per seed; p = {p:.4f} gives {mp:.4f} per seed "
                  f"({mp * n:.2f} people, 5-95% of runs {band[0] * n:.1f}-{band[1] * n:.1f})")
            if step == 0:
                p *= k / n / mp


if __name__ == "__main__":
    main()
