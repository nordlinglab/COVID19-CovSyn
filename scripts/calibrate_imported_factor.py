"""Calibrate the transmission factor of the imported first-wave seeds in the community (B78 C).

In the 2020-04-12 scenario the 118 imported seeds living in the community (69 found there and
49 recalled into home quarantine on 2020-03-18, B73) infected 12 local cases in the data
(validation_reference/taiwan_local_case_sources.csv), 0.102 each; the run 19 model gives about
0.32 each. The factor q multiplies their infection probabilities until they enter home
quarantine (the breach probability applies from then on, B74).

Since B82 the seeds that arrived before the border control are held at 1.0, so the people the
community seeds infect are m(q) = A + B q with A = m(0) > 0, and the proportional update
q = 12 / m(1) this script used to make overshot (0.256 against about 0.11 on run 21's vector;
code review 2026-11-09, F9). The script now measures m(0) and m(1), solves A + B q = 12, takes
one secant step through m(0) and m(q) -- two low-q measurements, as B82 found necessary -- and
checks the result with a last set of runs.

Usage (repository root):
    python scripts/calibrate_imported_factor.py <firefly_best.txt> [runs] [workers] [isolation]

isolation is the B78 option A setting (model or imported_delay, the default since the
meeting of 2026-11-02); calibrate with the one that will be used, since earlier isolation
leaves less for the factor to remove.
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


def solve_line(q_low, m_low, q_high, m_high, target):
    """The q at which the line through (q_low, m_low) and (q_high, m_high) reaches target.

    Args:
        q_low, m_low: A lower factor and the mean people infected at it.
        q_high, m_high: A higher factor and the mean people infected at it.
        target: The people infected in the data.

    Returns:
        The factor, clipped to [0, 1].

    Raises:
        ValueError: The two measurements do not rise with q, so no line can be solved.
    """
    if m_high <= m_low:
        raise ValueError(f"m({q_high:.4f}) = {m_high:.2f} is not above m({q_low:.4f}) = {m_low:.2f}")
    q = q_low + (target - m_low) * (q_high - q_low) / (m_high - m_low)
    return float(min(max(q, 0.0), 1.0))


def main():
    rows = np.atleast_2d(np.loadtxt(sys.argv[1]))
    P = rows[np.argmin(rows[:, -1]), 1:-1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    isolation = sys.argv[4] if len(sys.argv) > 4 else "imported_delay"
    with open("variable/demographic_parameters.pkl", "rb") as f:
        dp = pickle.load(f)
    k, n = observed_offspring()

    def measure(q):
        m, band = mean_offspring(P, dp, q, isolation, runs, workers)
        print(f"q = {q:.4f}: {n} community seeds infect {m:.2f} "
              f"(5-95% of runs {band[0]:.0f}-{band[1]:.0f}), data {k}; {runs} runs")
        return m

    print(f"isolation {isolation}")
    m0, m1 = measure(0.0), measure(1.0)
    if m0 >= k:
        print(f"the seeds held at 1.0 alone infect {m0:.2f} >= {k}; no q in [0, 1] reaches the data")
        return
    q = solve_line(0.0, m0, 1.0, m1, k)
    q = solve_line(0.0, m0, q, measure(q), k)   # secant through two low-q measurements
    print(f"calibrated q = {q:.4f}")
    measure(q)


if __name__ == "__main__":
    main()
