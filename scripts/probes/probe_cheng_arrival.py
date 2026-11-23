# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cheng's bins with both mechanisms: the investigation's start and the case's arrival.

Two mechanisms can put a household member's recorded first exposure after onset: the
investigation started at onset unless extended up to 4 days earlier (Cheng et al. 2020,
Methods), and an imported case met nobody in Taiwan before it arrived. Of Cheng's period
(confirmed by 18 March 2020: 100 cases in the Taiwan case data, 71 imported), 20 of the 51
imported cases with both dates arrived after onset. Each simulated index case takes the profile
of one of the 100 cases: local, imported with its arrival counted from onset, or isolated at
the airport (meets nobody). Contacts are then recorded from onset or from 4 days before, mixed
in shares g = 0, 0.25, 0.5, 0.75 and 1, and compared with Cheng's bins. The 20 imported cases
without both dates are given, in turn, an arrival drawn from the 51 and an arrival before
onset. Only the index case is truncated; its secondary cases (about 2 in 1,000) are kept.

Usage: uv run python scripts/probes/probe_cheng_arrival.py OUT_JSON [--simulations S]
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import probe_cheng_window as pcw

from covsyn.calibration import pilot_noise
from covsyn.model.data_synthesis_main import run_covid

CASES = "data/structured_course_of_disease_data/figshare_taiwan_covid.xlsx"
CHENG_END = "2020-03-18"
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
PROFILE_SEED = 20261123  # its own generator, so the model's random draws are untouched


def arrival_day(profile: dict, onset: int):
    """The day of arrival counted from infection, or None for a case that was not imported."""
    if not profile["imported"]:
        return None
    return int(onset + profile["offset"])


def arrived(contact: dict, arrival, airport: bool) -> dict:
    """A copy of the contacts with every day before arrival removed (all, at the airport).

    A contact left without any day is dropped, and its infection record with it.
    """
    out = dict(contact)
    for layer, key in pcw.MATRIX.items():
        matrix = np.array(contact[key], dtype=bool)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            continue
        if airport:
            matrix[:, :] = False
        elif arrival is not None and arrival > 0:
            matrix[:, :arrival] = False
        keep = np.flatnonzero(matrix.any(axis=1))
        out[key] = [matrix[i] for i in keep]
        infections = contact.get(f"{layer}_effective_contacts_infection_time")
        if infections is not None:
            out[f"{layer}_effective_contacts_infection_time"] = [
                infections[i] for i in keep if i < len(infections)
            ]
    return out


def profiles(missing: str) -> list[dict]:
    """Cheng's period's cases as index-case profiles; missing imported dates as ``missing`` says.

    missing="resample": drawn from the imported cases with both dates; "before": arrived before
    onset, so nothing after onset is removed.
    """
    d = pd.read_excel(CASES, sheet_name="Individual Data")
    confirmed = pd.to_datetime(d.confirmed_date, errors="coerce")
    d = d[confirmed <= CHENG_END]
    onset = pd.to_datetime(d.onset_of_symptom, errors="coerce")
    arrival = pd.to_datetime(d.date_of_arrival_to_taiwan, errors="coerce")
    imported = d["abroad/local"].astype(str).str.lower().str.startswith("abroad")
    airport = d["way_of_discover"].astype(str).str.strip() == "airport screening"
    offset = (arrival - onset).dt.days
    known = offset[imported & offset.notna()].astype(int).tolist()
    rng = np.random.default_rng(PROFILE_SEED)
    out = []
    for imp, air, off in zip(imported, airport, offset, strict=True):
        if imp and pd.isna(off):
            off = int(rng.choice(known)) if missing == "resample" else -30
        out.append({"imported": bool(imp), "airport": bool(air),
                    "offset": None if not imp else int(off)})  # fmt: skip
    return out


def simulate(P, simulations: int, cases: list[dict]) -> dict[int, np.ndarray]:
    """Pooled bins from onset (0) and from 4 days before (4), each index case a drawn profile."""
    with open("./variable/demographic_parameters.pkl", "rb") as f:
        demo = pickle.load(f)
    rng = np.random.default_rng(PROFILE_SEED + 1)
    pooled = {back: np.zeros((len(pcw.GROUPS), 2, 6)) for back in (0, 4)}
    for seed in range(pcw.STREAM_OFFSET, pcw.STREAM_OFFSET + simulations):
        profile = cases[int(rng.integers(len(cases)))]
        _, _, courses, contacts = run_covid(seed, P, copy.deepcopy(demo), save_file=False)
        for j, (course, contact) in enumerate(zip(courses, contacts, strict=True)):
            onset = course["incubation_period"]
            if j == 0 and onset is not None and not np.isnan(onset):
                contact = arrived(contact, arrival_day(profile, int(onset)), profile["airport"])
            for back in (0, 4):
                pooled[back] += pcw.case_bins(course, contact, back)
    return pooled


def main() -> None:
    """Score each extension share under both treatments of the missing arrival dates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--simulations", type=int, default=3000)
    args = parser.parse_args()
    with open("./variable/processed_contact_tracing_data.pkl", "rb") as f:
        cheng = np.asarray(pickle.load(f)["Cheng_contact_array"], dtype=float)
    population, lower, upper = pilot_noise._population_and_bounds()
    P = population[2] * (upper - lower) + lower  # run 19
    result: dict[str, Any] = {"simulations": args.simulations, "cheng": cheng.astype(int).tolist()}
    for missing in ("resample", "before"):
        cases = profiles(missing)
        pooled = simulate(P, args.simulations, cases)
        for g in SHARES:
            mixed = g * pooled[4] + (1 - g) * pooled[0]
            result[f"{missing}|g={g}"] = pcw.summary(mixed, cheng)
        print(missing, "done", flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
