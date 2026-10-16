"""B62 reachability: can the community targets be met together with the scale-free count?

Starts from a finished run's best vector, switches it to B62 (P[199] = 0, P[204] = alpha) and
scans the shift s = P[28], the repeat-contact probability P[29] (0.29 is the share Fu 2010 meets
daily) and the healthy community daily contact probability P[30]; each point is 300 index cases,
measured with the objective's own measure_outcomes (E100, B63).

Usage (CovSyn root): python scripts/probes/probe_community_b62.py FIREFLY_BEST_TXT [ALPHA]
"""
import concurrent.futures
import copy
import pickle
import sys

import numpy as np

from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.data_synthesis_main import run_covid

SEEDS = range(300)
KEYS = ("community_day_zero_share", "community_day_p90", "community_median",
        "community_zero_share", "daily_municipality", "infections_per_index_municipality")


def index_case(args):
    seed, vector, demo = args
    _, _, courses, contacts = run_covid(seed, vector.copy(), copy.deepcopy(demo), save_file=False,
                                        mode="result")
    return courses[0], contacts[0], None


def main():
    demo = pickle.load(open("variable/demographic_parameters.pkl", "rb"))
    best = np.atleast_2d(np.loadtxt(sys.argv[1]))
    base = best[int(np.argmin(best[:, -1])), 1:-1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.74
    base = np.r_[base[:204], alpha]
    base[199] = 0.0
    base[30] = 0.01
    print("targets:", {k: fo.OUTCOME_TARGETS[k][:2] for k in KEYS})
    with concurrent.futures.ProcessPoolExecutor(32) as pool:
        for s in (0.5, 1.0, 2.0, 3.0, 5.0):
            for p29, p30 in ((0.29, 0.01), (0.0, 0.01), (0.29, 0.05), (0.0, 0.05)):
                v = base.copy()
                v[28], v[29], v[30] = s, p29, p30
                cases = list(pool.map(index_case, [(k, v, demo) for k in SEEDS], chunksize=10))
                m = fo.measure_outcomes(cases)
                print(f"s {s:4.1f} P[29] {p29:.2f} P[30] {p30:.2f}: "
                      + "  ".join(f"{k} {m[k]:.3g}" for k in KEYS), flush=True)


if __name__ == "__main__":
    main()
