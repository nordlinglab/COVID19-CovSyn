# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare a first-wave scenario (B66) with the local cases it should reproduce.

The reference is the set of reported local cases whose chain of infection starts at one of the
scenario's seeds: following each local case's infector in
validation_reference/taiwan_local_case_sources.csv until it reaches an imported case or a local
case of unknown source (the root). For the 'all' scenario every root is a seed, so the reference
is all 55 local cases; for 'feb7' only the chains rooted before 2020-02-07 count.
The simulated local cases of a run are its non-seed cases plus its local seeds.

Usage (repository root, PYTHONPATH=src):
    python -m covsyn.validation.compare_first_wave RESULT_DIR SEED_CSV
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = Path("validation_reference/taiwan_local_case_sources.csv")
LAYERS = ("household", "school", "workplace", "health_care", "municipality")


def chain_root(case_id: int, infector: dict[int, int | None]) -> int:
    """Follow the recorded infectors of a local case back to its root."""
    seen = {case_id}
    while infector.get(case_id) is not None:
        case_id = infector[case_id]
        if case_id in seen:
            break
        seen.add(case_id)
    return case_id


def reference_local_cases(seeds: pd.DataFrame, sources: pd.DataFrame) -> list[int]:
    """The reported local cases whose chain of infection starts at one of these seeds."""
    infector = {int(r.id): (int(r.infector_id) if str(r.infector_id).strip() not in ("", "nan")
                            else None)
                for r in sources.itertuples()}
    roots = set(seeds.case_id.astype(int))
    return sorted(c for c in infector if chain_root(c, infector) in roots)


def simulated(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Local cases per run, and infections caused by every imported seed."""
    local, imported = [], []
    for f in sorted(glob.glob(str(directory / "course_of_disease_data_*.npy"))):
        k = Path(f).stem.split("_")[-1]
        course = list(np.load(f, allow_pickle=True))
        contact = list(np.load(directory / f"contact_data_{k}.npy", allow_pickle=True))
        local.append(sum(1 for c in course if c.get("seed_type") in (None, "local_unknown")))
        for c, k_ in zip(course, contact, strict=True):
            if c.get("seed_type") == "imported":
                imported.append(sum(float(np.nansum(np.asarray(k_[f"{L}_effective_contacts"] or [],
                                                                dtype=float))) for L in LAYERS))
    return np.array(local, dtype=float), np.array(imported, dtype=float)


def main() -> None:
    """Print the simulated local cases and the imported seeds' offspring against the reference."""
    directory, seed_csv = Path(sys.argv[1]), Path(sys.argv[2])
    seeds = pd.read_csv(seed_csv)
    reference = reference_local_cases(seeds, pd.read_csv(SOURCES, dtype=str, keep_default_na=False))
    local, imported = simulated(directory)
    rng = np.random.default_rng(0)
    boot = [rng.choice(local, len(local)).mean() for _ in range(2000)] if len(local) else [np.nan]
    print(f"{seed_csv.name}: {len(seeds)} seeds; reference local cases {len(reference)}")
    print(f"simulated local cases per run: mean {local.mean():.1f} (95% of the mean "
          f"{np.percentile(boot, 2.5):.1f}-{np.percentile(boot, 97.5):.1f}), median "
          f"{np.median(local):.0f}, 5-95% of runs {np.percentile(local, 5):.0f}-"
          f"{np.percentile(local, 95):.0f}; n = {len(local)}")
    if len(imported):
        print(f"infections per imported seed: {imported.mean():.3f} "
              f"(Cheng 2020: 0.056, exact Poisson 95% 0.015-0.14)")


if __name__ == "__main__":
    main()
