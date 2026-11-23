# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Do the Cheng bins move towards Cheng's when contacts are recorded as Cheng recorded them?

Cheng et al. (2020, Methods): "The period of investigation started at the date at symptom onset
(and could be extended to up to 4 days before symptom onset when epidemiologically indicated)
and ended at the date at COVID-19 confirmation." A contact's first day of exposure is the first
day of reported exposure within that period. The objective bins the simulation's TRUE first
contact (plot_results.create_array_cheng2020_fig2), so a household member met since infection
always falls before onset. This bins every simulated case's contacts three ways: the true first
contact, a period from onset, and a period from 4 days before onset; with the index case's age
drawn as now (the population) or from the Taiwan first-wave cases up to 18 March. The first way
is checked against create_array_cheng2020_fig2 case by case.

Usage: uv run python scripts/probes/probe_cheng_window.py OUT_JSON [--simulations S]
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from covsyn.calibration import pilot_noise
from covsyn.figures.plot_results import create_array_cheng2020_fig2
from covsyn.model import data_synthesize
from covsyn.model.data_synthesis_main import run_covid

GROUPS = (("household", ("household",)), ("health_care", ("health_care",)),
          ("others", ("school", "workplace", "municipality")))  # fmt: skip
MATRIX = {"household": "household_contacts_matrix", "school": "school_class_contacts_matrix",
          "workplace": "workplace_contacts_matrix", "health_care": "health_care_contacts_matrix",
          "municipality": "municipality_contacts_matrix"}  # fmt: skip
CHENG_LAYER = {"household": "Household", "school": "School", "workplace": "Workplace",
               "health_care": "Health care", "municipality": "Municipality"}  # fmt: skip
WINDOWS = {"true_first_contact": None, "from_onset": 0, "from_onset_minus_4": 4}
SEEDS_MAR18 = "./variable/taiwan_first_wave_seeds_mar18.csv"
STREAM_OFFSET = 2010000000 + 13 * 3000  # pilot 1's validation stream 201, block 13 at S = 3,000


def recorded_first_day(row: np.ndarray, onset: int, confirmation: int, back: int | None):
    """The first contact day the investigation records, or None when it records none.

    back=None keeps the true first contact; otherwise the period runs from onset - back to
    confirmation, both included.
    """
    days = np.flatnonzero(row)
    if back is not None:
        days = days[(days >= onset - back) & (days <= confirmation)]
    return int(days[0]) if days.size else None


def cheng_bin(days_from_onset: int) -> int:
    """Cheng's bin of a first day of exposure: < 0, 0-3, 4-5, 6-7, 8-9, > 9 days from onset."""
    for edge, index in ((0, 0), (4, 1), (6, 2), (8, 3), (10, 4)):
        if days_from_onset < edge:
            return index
    return 5


def case_bins(course: dict, contact: dict, back: int | None) -> np.ndarray:
    """One case's contacts and infections per group and bin, shape (3, 2, 6).

    An asymptomatic case (no onset) adds nothing, as in Cheng's figure. The confirmation day is
    counted from the case's own infection, as its contact matrices are.
    """
    out = np.zeros((len(GROUPS), 2, 6))
    onset = course["incubation_period"]
    if onset is None or np.isnan(onset):
        return out
    confirmation = int(course["positive_test_date"] - course["infection_day"])
    for gi, (_, layers) in enumerate(GROUPS):
        for layer in layers:
            matrix = np.asarray(contact[MATRIX[layer]], dtype=bool)
            if matrix.ndim != 2 or matrix.shape[0] == 0:  # the model writes no contacts as []
                continue
            infected = contact.get(f"{layer}_effective_contacts_infection_time") or []
            for m, row in enumerate(matrix):
                day = recorded_first_day(row, int(onset), confirmation, back)
                if day is None:
                    continue
                b = cheng_bin(day - int(onset))
                out[gi, 0, b] += 1
                if m < len(infected) and not np.isnan(infected[m]):
                    out[gi, 1, b] += 1
    return out


def original_bins(course: dict, contact: dict) -> np.ndarray:
    """The objective's own binning, without B55's expected mass events, for the check."""
    out = np.zeros((len(GROUPS), 2, 6))
    for gi, (_, layers) in enumerate(GROUPS):
        for layer in layers:
            _, contacts, _, infections = create_array_cheng2020_fig2(
                [course], [contact], layer=CHENG_LAYER[layer], expected_events=False
            )
            if np.size(contacts) == 6:
                out[gi, 0] += contacts
                out[gi, 1] += infections
    return out


def _taiwan_age_draw(bands):
    def draw_age(self):
        if np.isnan(self.age):  # only the index case enters without an age
            low, high = random.choice(bands)
            self.age = random.randint(low, high)
        return self.age

    return draw_age


def simulate(P, simulations: int, ages: str, bands) -> dict[str, np.ndarray]:
    """Pooled bins per window over the simulations, after checking the true-contact binning."""
    with open("./variable/demographic_parameters.pkl", "rb") as f:
        demo = pickle.load(f)
    original = data_synthesize.Draw_demographic_data.draw_age
    if ages == "taiwan":
        data_synthesize.Draw_demographic_data.draw_age = _taiwan_age_draw(bands)
    pooled = {name: np.zeros((len(GROUPS), 2, 6)) for name in WINDOWS}
    try:
        for seed in range(STREAM_OFFSET, STREAM_OFFSET + simulations):
            _, _, courses, contacts = run_covid(seed, P, copy.deepcopy(demo), save_file=False)
            for course, contact in zip(courses, contacts, strict=True):
                for name, back in WINDOWS.items():
                    bins = case_bins(course, contact, back)
                    if back is None and not np.array_equal(bins, original_bins(course, contact)):
                        raise SystemExit(
                            f"seed {seed}: the true-contact bins differ from the objective's"
                        )
                    pooled[name] += bins
    finally:
        data_synthesize.Draw_demographic_data.draw_age = original
    return pooled


def summary(bins: np.ndarray, cheng: np.ndarray) -> dict[str, Any]:
    """Per group: the share of contacts per bin, Cheng's, and the total variation distance."""
    out = {}
    for gi, (name, _) in enumerate(GROUPS):
        share = bins[gi, 0] / bins[gi, 0].sum() if bins[gi, 0].sum() else np.zeros(6)
        theirs = cheng[gi] / cheng[gi].sum()
        distance = round(float(0.5 * np.abs(share - theirs).sum()), 3)
        out[name] = {"contacts": bins[gi, 0].astype(int).tolist(),
                     "infections": bins[gi, 1].astype(int).tolist(),
                     "share": np.round(share, 3).tolist(),
                     "cheng_share": np.round(theirs, 3).tolist(),
                     "total_variation": distance}  # fmt: skip
    return out


def main() -> None:
    """Bin two vectors' simulations every way and write the comparison with Cheng."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--simulations", type=int, default=3000)
    args = parser.parse_args()
    with open("./variable/processed_contact_tracing_data.pkl", "rb") as f:
        cheng = np.asarray(pickle.load(f)["Cheng_contact_array"], dtype=float)
    seeds = pd.read_csv(SEEDS_MAR18)
    pairs = zip(seeds.age_low, seeds.age_high, strict=True)
    bands = [(int(lo), int(hi)) for lo, hi in pairs if np.isfinite(lo) and np.isfinite(hi)]
    population, lower, upper = pilot_noise._population_and_bounds()
    result: dict[str, Any] = {"simulations": args.simulations, "cheng": cheng.astype(int).tolist()}
    for label, row in (("run19", 2), ("literature", 0)):
        P = population[row] * (upper - lower) + lower
        for ages in ("population", "taiwan"):
            pooled = simulate(P, args.simulations, ages, bands)
            for name, bins in pooled.items():
                result[f"{label}|{ages}|{name}"] = summary(bins, cheng)
            print(label, ages, "done", flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
