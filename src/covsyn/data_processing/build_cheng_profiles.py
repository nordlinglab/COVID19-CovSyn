# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the index-case profiles of Cheng et al. (2020)'s period from the Taiwan case data (E84).

Cheng's contact data come from the investigations of the first 100 confirmed cases (confirmed by
2020-03-18), 71 of them imported. The calibration compares its index cases with these, so since
2026-11-23 (user decision) each simulated index case takes the profile of one of them: its
ten-year age group (the model draws the age within it, as B66 does for the first-wave seeds),
whether it was imported, whether airport screening found it (isolated on arrival, so the
investigation recorded no contacts in Taiwan), and the day of arrival counted from onset. An
imported case without both dates keeps an unknown arrival, which the calibration draws from
the known ones (covsyn.calibration.cheng_recording).

Usage: uv run python -m covsyn.data_processing.build_cheng_profiles [CASES_XLSX]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASES = Path("data/structured_course_of_disease_data/figshare_taiwan_covid.xlsx")
OUT = Path("variable/cheng_period_profiles.csv")
CHENG_END = pd.Timestamp("2020-03-18")  # confirmation of case 100
COLUMNS = ("case_id", "age_low", "age_high", "imported", "airport", "arrival_from_onset")
_BAND = re.compile(r"([0-9]+) to ([0-9]+)")


def age_band(label) -> tuple[float, float]:
    """The (low, high) of an age label written 'low to high'; two NaNs for anything else."""
    match = _BAND.fullmatch(label.strip()) if isinstance(label, str) else None
    if match is None or int(match[1]) > int(match[2]):
        return np.nan, np.nan
    return int(match[1]), int(match[2])


def build_profiles(cases: pd.DataFrame, end: pd.Timestamp = CHENG_END) -> pd.DataFrame:
    """One row per case confirmed by ``end``, with the columns of COLUMNS.

    Args:
        cases: The 'Individual Data' sheet of the Taiwan case data.
        end: Last confirmation date kept.
    """
    confirmed = pd.to_datetime(cases.confirmed_date, errors="coerce")
    d = cases[confirmed <= end]
    imported = d["abroad/local"].astype(str).str.strip() == "Abroad"
    onset = pd.to_datetime(d.onset_of_symptom, errors="coerce")
    arrival = pd.to_datetime(d.date_of_arrival_to_taiwan, errors="coerce")
    bands = [age_band(label) for label in d.age]
    out = pd.DataFrame({
        "case_id": d.id.astype(int).to_numpy(),
        "age_low": [low for low, _ in bands],
        "age_high": [high for _, high in bands],
        "imported": imported.to_numpy(),
        "airport": (d.way_of_discover.astype(str).str.strip() == "airport screening").to_numpy(),
        "arrival_from_onset": (arrival - onset).dt.days.where(imported).to_numpy(dtype=float),
    }, columns=list(COLUMNS))
    return out.sort_values("case_id", ignore_index=True)


def main() -> None:
    """Write the profiles and print a summary."""
    cases_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CASES
    out = build_profiles(pd.read_excel(cases_path, sheet_name="Individual Data"))
    out.to_csv(OUT, index=False)
    print(f"{len(out)} profiles -> {OUT}: {int(out.imported.sum())} imported "
          f"({int(out.airport.sum())} at the airport, "
          f"{int(out.arrival_from_onset.notna().sum())} with both dates)")


if __name__ == "__main__":
    main()
