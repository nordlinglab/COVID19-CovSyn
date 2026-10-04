# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the regression fixtures that pin CovSyn's output before any restructuring.

Run from the repository root, on the compute server's pinned environment, whenever a decision
intentionally changes simulated output; first at the commit that produced Phase D run 10, last
for B52 (isolation at ICU admission):

    python tests/make_regression_fixtures.py

It records, for the run 10 best parameter vector,
(1) the total cost and every cost part of ``fast_cost.cost_function``, and
(2) digests of ``run_covid`` output for fixed seeds in both simulation modes the pipeline uses.
The regression tests then require every later commit to reproduce these exactly.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

import regression_digest as rd  # noqa: E402

SEEDS = list(range(8))
# "result" is what the objective simulates; the other two are the modes data_synthesis.sh runs.
MODES = ["result", "spread_Taiwan_weight", "taiwan_first_outbreak"]


def main() -> None:
    """Compute and write ``tests/fixtures/regression_run10.json``."""
    os.chdir(REPO_ROOT)
    from covsyn.calibration import fast_cost
    from conftest import best_vector
    from covsyn.model.data_synthesis_main import run_covid
    from covsyn.calibration.cost_parts import LAST

    with open("variable/demographic_parameters.pkl", "rb") as f:
        demo = pickle.load(f)
    with open("variable/processed_contact_tracing_data.pkl", "rb") as f:
        tracing = pickle.load(f)
    cheng = (tracing["Cheng_contact_array"], tracing["Cheng_attack_rate"], tracing["norm_weights"])
    columns = np.load("variable/Taiwan_data_matrix.npy").shape[1]
    vector = best_vector()

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=min(8, os.cpu_count() or 1),
        initializer=fast_cost.init_worker,
        initargs=(demo, columns),
    )
    total = fast_cost.cost_function(vector, demo, pool, *cheng)
    parts = {k: rd.encode_float(float(v)) for k, v in LAST.items()}
    pool.shutdown()

    synthesis = {}
    for mode in MODES:
        for seed in SEEDS:
            output = run_covid(seed, vector.copy(), copy.deepcopy(demo), save_file=False, mode=mode)
            synthesis[f"{mode}/{seed}"] = rd.digest(output)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=REPO_ROOT, check=False).stdout.strip()
    fixture = {
        "generated_at_commit": commit,
        "parameter_vector": "firefly_result/phaseD_run10/firefly_best.txt (lowest-cost row)",
        "cost_total": rd.encode_float(float(total)),
        "cost_parts": parts,
        "synthesis_digests": synthesis,
    }
    out = HERE / "fixtures" / "regression_run10.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"cost {total!r}, {len(parts)} parts, {len(synthesis)} synthesis digests -> {out}")


if __name__ == "__main__":
    main()
