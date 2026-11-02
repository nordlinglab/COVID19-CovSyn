# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""How many copied seeds the February scenarios need to reach the data (B79).

Professor meeting 2026-11-02: the February scenarios infect too few local cases; seeds missed by
the surveillance may be added, copied from the community seeds (build_first_wave_seeds.
copy_community_seeds), as long as they are about 20-30% more seeds. Needing about twice as many
would mean the infection probability is too low, and the model parameters are revisited instead.

For each scenario and each n = 0, 1, ..., the script writes the seed table with n copies (one
fixed generator, so the sets are nested), runs it, and prints the mean local cases per run
against the reference (compare_first_wave), with n as a share of the community seeds. The
first n whose mean reaches the reference is the answer.

Usage (repository root):
    python scripts/calibrate_copied_seeds.py <firefly_best.txt> [runs] [workers] [max_copies]
"""

import concurrent.futures
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from covsyn.data_processing import build_first_wave_seeds as bfs  # noqa: E402
from covsyn.model.data_synthesis_main import run_covid  # noqa: E402
from covsyn.validation.compare_first_wave import SOURCES, reference_local_cases  # noqa: E402

SEED_START = 720000
LOCAL = (None, "local_unknown", "local_from_imported")


def local_cases(seed, P, dp, mode, seed_file):
    """Local cases of one run: the non-seed cases and the local seeds, copies included."""
    _, _, course, _ = run_covid(seed, P, dp, mode=mode, seed_file=seed_file)
    return sum(1 for c in course if c.get("seed_type") in LOCAL)


def mean_local(P, dp, mode, seed_file, runs, workers):
    """Mean local cases over runs, with the 5-95% range of the runs."""
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        out = np.array(list(ex.map(local_cases, range(SEED_START, SEED_START + runs),
                                   [P] * runs, [dp] * runs, [mode] * runs,
                                   [seed_file] * runs)), dtype=float)
    return out.mean(), np.percentile(out, [5, 95])


def main():
    rows = np.atleast_2d(np.loadtxt(sys.argv[1]))
    P = rows[np.argmin(rows[:, -1]), 1:-1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    max_copies = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    with open("variable/demographic_parameters.pkl", "rb") as f:
        dp = pickle.load(f)
    cases = pd.read_excel(bfs.CASES, sheet_name="Individual Data")
    sources = pd.read_csv(SOURCES, dtype=str, keep_default_na=False)
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in bfs.COPY_SCENARIOS:
            mode = f"taiwan_first_outbreak_{scenario}"
            base = pd.read_csv(bfs.write_copied_seeds(scenario, 0, cases, Path(tmp)))
            target = len(reference_local_cases(base, sources))
            community = int((base.status == "community").sum())
            print(f"[{scenario}] reference {target} local cases; {len(base)} seeds, "
                  f"{community} in the community; {runs} runs")
            for n in range(max_copies + 1):
                path = bfs.write_copied_seeds(scenario, n, cases, Path(tmp))
                mean, band = mean_local(P, dp, mode, str(path), runs, workers)
                print(f"  n = {n:2d} ({n / community:5.1%} of the community seeds): "
                      f"{mean:5.1f} local cases (5-95% of runs {band[0]:.0f}-{band[1]:.0f})",
                      flush=True)
                if mean >= target:
                    break


if __name__ == "__main__":
    main()
