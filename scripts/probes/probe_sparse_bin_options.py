# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which way of handling the nearly empty household 0-3 day bin changes pilot 1 how?

Reads the bins.csv probe_sparse_bins.py wrote (pilot 1's 992 evaluations, each with its bin
terms and its household 0-3 day contacts) and, without simulating, re-makes pilot 1's choice
of S under each option: the bin as it is, dropped, dropped below a number of contacts, or its
rate shrunk toward Cheng's by a prior of k contacts. Merging the bin with its neighbour needs
the neighbour's counts, which bins.csv lacks, so it is not scored here.

Usage: uv run python scripts/probes/probe_sparse_bin_options.py BINS_CSV OUT_JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import probe_sparse_bins as psb

from covsyn.calibration import pilot_noise

TERM = f"term_{psb.SPARSE[0]}_{psb.SPARSE[1]}"
OPTIONS = [("as_pilot_1", None), ("drop", None), ("threshold", 10), ("threshold", 30),
           ("threshold", 100), ("shrink", 10), ("shrink", 30), ("shrink", 100)]  # fmt: skip


def _kept_share(option: str, value: float | None, contacts: int) -> float:
    """The share of the bin's term an option keeps.

    Shrinking the rate toward Cheng's, (i + k c) / (n + k), puts it n / (n + k) of the way from
    c, and the term is that distance squared.
    """
    if option == "as_pilot_1":
        return 1.0
    if option == "drop":
        return 0.0
    if option == "threshold":
        return 0.0 if contacts < value else 1.0
    if option == "shrink":
        return (contacts / (contacts + value)) ** 2
    raise ValueError(f"unknown option {option!r}")


def rescored(rows: list[dict], option: str, value: float | None) -> list[dict]:
    """The rows with each cost rewritten as the option scores the household 0-3 day bin.

    A failed evaluation (contacts -1) keeps its charge: it has no bin to rescore.

    Raises:
        ValueError: an unknown option.
    """
    _kept_share(option, value, 0)  # refuses an unknown option even when there are no rows
    out = []
    for r in rows:
        contacts = int(r["household_0_3_contacts"])
        if contacts < 0:
            out.append(dict(r))
            continue
        term = float(r[TERM])
        cost = float(r["cost"]) - term + term * _kept_share(option, value, contacts)
        out.append({**r, "cost": cost})
    return out


def describe(rows: list[dict], option: str, value: float | None) -> dict[str, Any]:
    """Pilot 1's choice under the option, and each vector's SD and mean at S = 3,000 and 10,000."""
    summary = pilot_noise.summarise(rescored(rows, option, value))
    vectors = {f"{v['label']}@{v['simulations']}": {"mean": v["mean"], "sd": v["sd"]}
               for v in summary["vectors"] if v["simulations"] >= 3_000}  # fmt: skip
    return {"option": option, "value": value, "choice": summary["choice"], "vectors": vectors}


def contacts_by_vector(rows: list[dict]) -> dict[str, Any]:
    """Per vector and S: the bin's contacts (min, median, max) and its mean term."""
    found: dict[str, Any] = {}
    for key in sorted({(r["label"], int(r["simulations"])) for r in rows if r["kind"] == "vector"},
                      key=lambda k: (k[1], k[0])):  # fmt: skip
        mine = [r for r in rows if (r["label"], int(r["simulations"])) == key]
        n = [int(r["household_0_3_contacts"]) for r in mine]
        found[f"{key[0]}@{key[1]}"] = {
            "contacts_min": min(n), "contacts_median": float(np.median(n)),
            "contacts_max": max(n),
            "term_mean": float(np.mean([float(r[TERM]) for r in mine])),
            "term_max": max(float(r[TERM]) for r in mine),
        }  # fmt: skip
    return found


def main() -> None:
    """Score every option on bins.csv and write the result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bins", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = list(psb.read_done(args.bins).values())
    result = {
        "evaluations": len(rows),
        "contacts": contacts_by_vector(rows),
        "options": [describe(rows, option, value) for option, value in OPTIONS],
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
