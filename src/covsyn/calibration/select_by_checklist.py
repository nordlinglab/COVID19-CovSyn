# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose among the best revalidated candidates by the acceptance checklist (B60).

Revalidation (E87, E95) ranks the candidates by their objective on unseen seeds, but the
objective does not contain every checklist item (the enterprise-size ratio, for one) and
charges near-edge misses lightly, so the lowest-cost candidate can still fail a check. This
runs the full checklist for the TOP candidates by validation cost, each on its own synthetic
data and age risk ratio measurement, and keeps the one with the fewest failures, then the
fewest results that pass only within their 95% interval, then the lowest validation cost.

The data are generated on the chain's usual seeds 0..999. Because the choice is made on them,
the chain reports the chosen candidate's checklist again on an independent seed set as well.

Usage (repository root, PYTHONPATH=src):
    python -m covsyn.calibration.select_by_checklist REVALIDATION_DIR OUT_DIR [TOP] [CPU_CORES]

Writes OUT_DIR (a copy of REVALIDATION_DIR whose firefly_best.txt holds only the chosen vector),
OUT_DIR/selection.csv and the candidates' checklists under OUT_DIR/candidates/.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from covsyn.calibration.revalidation import write_best_file

MONTE_CARLO = 1000
AGE_RR_CASES = 20000
MODES = (("spread", "spread_Taiwan_weight"), ("first", "taiwan_first_outbreak"))


def checklist_counts(checks: list[dict]) -> tuple[int, int]:
    """Number of failed checks and of checks passing only within their 95% interval."""
    judged = [c for c in checks if c.get("ok") is not None]
    failed = sum(1 for c in judged if c["ok"] is False)
    within_noise = sum(1 for c in judged if c["ok"] and c.get("within_noise"))
    return failed, within_noise


def rank_key(row: dict) -> tuple[int, int, float]:
    """Sort key: fewest failures, then fewest ok~, then lowest validation cost."""
    return (int(row["failed"]), int(row["within_noise"]), float(row["validation_mean"]))


def _run(args: list[str], log, env: dict | None = None) -> None:
    subprocess.run(
        [sys.executable, *args], check=True, stdout=log, stderr=subprocess.STDOUT, env=env
    )


def evaluate_candidate(vector: np.ndarray, cost: float, workdir: Path, cpu_cores: int) -> dict:
    """Generate the candidate's data, measure its age risk ratio and run the checklist."""
    workdir.mkdir(parents=True, exist_ok=True)
    write_best_file(workdir / "firefly_best.txt", vector[None, :], np.array([cost]))
    with open(workdir / "log.txt", "w") as log:
        for name, mode in MODES:
            # data_synthesis_main does not create its result path, and a missing one makes
            # every save fail inside the worker pool without failing the process.
            (workdir / name).mkdir(parents=True, exist_ok=True)
            _run(
                [
                    "-m",
                    "covsyn.model.data_synthesis_main",
                    "--mode",
                    mode,
                    "--monte_carlo_number",
                    str(MONTE_CARLO),
                    "--result_path",
                    str(workdir / name),
                    "--cpu_cores",
                    str(cpu_cores),
                    "--parameter_path",
                    str(workdir),
                ],
                log,
            )
        age_rr = workdir / "age_rr.json"
        _run(
            [
                "-m",
                "covsyn.validation.measure_age_rr",
                str(workdir),
                str(AGE_RR_CASES),
                "--out",
                str(age_rr),
            ],
            log,
        )
        env = {**os.environ, "AGE_RR_FILE": str(age_rr)}
        _run(
            [
                "-m",
                "covsyn.validation.verify_phase_d",
                str(workdir / "spread"),
                str(workdir / "first"),
                str(workdir / "checks.json"),
            ],
            log,
            env=env,
        )
    failed, within_noise = checklist_counts(
        json.loads((workdir / "checks.json").read_text())["checks"]
    )
    # The candidate's data are large and only its counts are needed from here on.
    for name, _ in MODES:
        shutil.rmtree(workdir / name, ignore_errors=True)
    return {"failed": failed, "within_noise": within_noise}


def main() -> None:
    """Evaluate the TOP revalidated candidates and write the chosen one to OUT_DIR."""
    source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    cpu_cores = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    with open(source / "revalidation.csv") as f:
        table = list(csv.DictReader(f))
    best = np.atleast_2d(np.loadtxt(source / "firefly_best.txt"))
    vectors = best[:, 1:-1]
    ranked = sorted(table, key=lambda r: float(r["validation_mean"]))[:top]
    if out_dir.resolve() == source.resolve():
        raise SystemExit("OUT_DIR must differ from REVALIDATION_DIR")
    if len(best) != len(table):
        raise SystemExit(
            f"{source}: {len(table)} rows in revalidation.csv but "
            f"{len(best)} in firefly_best.txt; not a revalidation output"
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(source, out_dir)
    rows = []
    for r in ranked:
        index = int(r["candidate"])
        cost = float(r["validation_mean"])
        counts = evaluate_candidate(
            vectors[index], cost, out_dir / "candidates" / f"candidate_{index}", cpu_cores
        )
        rows.append({"candidate": index, "validation_mean": cost, **counts})
        print(
            f"candidate {index}: validation {cost:.4f}, {counts['failed']} failed, "
            f"{counts['within_noise']} ok~",
            flush=True,
        )
    chosen = min(rows, key=rank_key)
    with open(out_dir / "selection.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], "chosen"])
        writer.writeheader()
        writer.writerows([{**r, "chosen": r is chosen} for r in rows])
    write_best_file(
        out_dir / "firefly_best.txt",
        vectors[chosen["candidate"]][None, :],
        np.array([chosen["validation_mean"]]),
    )
    print(
        f"checklist choice: candidate {chosen['candidate']} ({chosen['failed']} failed, "
        f"{chosen['within_noise']} ok~, validation {chosen['validation_mean']:.4f}) -> "
        f"{out_dir / 'firefly_best.txt'}"
    )


if __name__ == "__main__":
    main()
