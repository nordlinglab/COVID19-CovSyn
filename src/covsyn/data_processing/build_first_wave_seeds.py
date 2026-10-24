# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the seed cases of the two Taiwan first-wave scenarios (B66) from the Taiwan case data.

The professor asked on 2026-10-12 and 2026-10-19 that the first-wave simulation start from the
cases that came from abroad or whose source is unknown, put in the infection queue with the
properties the case data give them, and that cases kept isolated on entry not be counted as
spreading in society. Taiwan's entry quarantine in the first wave was home quarantine (Feb 6-7
for China, Hong Kong and Macau, Mar 19 for every arrival; supervised facility quarantine for all
arrivals only from 2021-01-15), so two scenarios are built (user decision 2026-10-24):

``all`` (FIRST_WAVE_END = 2020-04-12, the confirmation of the last local case of the first wave)
    Every imported case that arrived by then and the 13 local cases of unknown source. The
    restrictions grow over time: each imported seed's status comes from how it was detected, and
    from 2020-03-19 every arrival is at least in home quarantine (policy).
    Compared with the 55 local cases of the first wave.

``feb7`` (before 2020-02-07, the "first three weeks before the border control" of 2026-10-12)
    The imported cases that arrived before 2020-02-07 and the local cases of unknown source
    infected before it. No policy restriction applies yet; a case caught by airport screening
    is still isolated on arrival, as it was.
    Compared with the local cases whose chain of infection starts at one of these seeds
    (covsyn.validation.compare_first_wave).

Local cases with a known source are left for the model to produce
(validation_reference/taiwan_local_case_sources.csv).

Infection date, fixed once (2026-10-19 decision, sensitivity is a cut-off shift instead)
    * the recorded earliest infection date when there is one;
    * otherwise onset minus the median incubation of the cases with both dates;
    * no onset but symptoms listed: confirmation minus that median incubation and the median
      onset-to-confirmation delay;
    * no onset and no symptoms: confirmation minus the median infection-to-confirmation delay.
    The estimate is then clipped into what the record allows: not before the first departure
    abroad, not after the arrival in Taiwan, the onset or the confirmation.

Status after arrival
    From how the case was detected: airport screening -> 'airport'; home quarantine ->
    'quarantine'; anything else -> 'community'. In the ``all`` scenario an arrival on or after
    2020-03-19 that would be 'community' becomes 'quarantine' (status_source 'policy 2020-03-19').
    No imported case of the data arrived from China, Hong Kong or Macau between Feb 7 and Mar 18,
    so the Feb 7 rule changes nobody. Local seeds are always 'community'.

Age is kept as the ten-year group; the simulation draws the year within it (professor,
2026-10-19). Day 0 of a scenario is its earliest seed infection.

Usage (repository root):
    python -m covsyn.data_processing.build_first_wave_seeds [case_xlsx] [sources_csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASES = Path("data/structured_course_of_disease_data/figshare_taiwan_covid.xlsx")
SOURCES = Path("validation_reference/taiwan_local_case_sources.csv")
FIRST_WAVE_END = pd.Timestamp("2020-04-12")
BORDER_CONTROL = pd.Timestamp("2020-02-07")
ALL_ARRIVALS_QUARANTINED = pd.Timestamp("2020-03-19")
# scenario -> (output file, last arrival, local seeds infected before, home quarantine policy)
SCENARIOS = {
    "all": (Path("variable/taiwan_first_wave_seeds.csv"), FIRST_WAVE_END, None, True),
    "feb7": (Path("variable/taiwan_first_wave_seeds_feb7.csv"),
             BORDER_CONTROL - pd.Timedelta(days=1), BORDER_CONTROL, False),
}
STATUS = {"airport screening": "airport", "home quarantine": "quarantine"}


def median_days(later: pd.Series, earlier: pd.Series) -> float:
    """Median of (later - earlier) in days over the rows that have both."""
    days = (later - earlier).dt.days.dropna()
    return float(np.median(days))


def build(cases: pd.DataFrame, sources: pd.DataFrame,
          end: pd.Timestamp = FIRST_WAVE_END, local_before: pd.Timestamp | None = None,
          policy: bool = True) -> pd.DataFrame:
    """The seed table: one row per seed case, ordered by infection day.

    Args:
        cases: The 'Individual Data' sheet of the Taiwan case data.
        sources: The local-case source classification.
        end: Last arrival date of an imported seed.
        local_before: Keep only local seeds infected before this date; None keeps all 13.
        policy: Apply the home quarantine of every arrival from 2020-03-19.
    """
    d = cases.copy()
    for column in ("onset_of_symptom", "confirmed_date", "earliest_infection_date",
                   "date_of_arrival_to_taiwan"):
        d[column] = pd.to_datetime(d[column], errors="coerce")
    # Departure dates are sometimes only a month ("2020-02"); those give no bound.
    d["departure"] = pd.to_datetime(d["date_of_departure_1"].astype(str), format="%Y-%m-%d",
                                    errors="coerce")
    incubation = median_days(d.onset_of_symptom, d.earliest_infection_date)
    onset_to_confirmation = median_days(d.confirmed_date, d.onset_of_symptom)
    infection_to_confirmation = median_days(d.confirmed_date, d.earliest_infection_date)

    imported = d["abroad/local"] == "Abroad"
    d["arrival"] = d.date_of_arrival_to_taiwan.where(
        ~imported | d.date_of_arrival_to_taiwan.notna(), d.confirmed_date)
    unknown = set(sources.loc[sources.source_class == "unknown", "id"].astype(int))
    seed = (imported & (d.arrival <= end)) | d.id.isin(unknown)
    s = d[seed].copy()

    symptom = s.symptom.astype(str).str.strip().str.upper()
    listed = ~symptom.isin(["", "X", "NAN"])
    estimate = s.onset_of_symptom - pd.Timedelta(days=incubation)
    estimate = estimate.fillna(
        (s.confirmed_date - pd.Timedelta(days=incubation + onset_to_confirmation)).where(listed))
    estimate = estimate.fillna(s.confirmed_date - pd.Timedelta(days=infection_to_confirmation))
    infection = s.earliest_infection_date.fillna(estimate)
    rule = np.where(s.earliest_infection_date.notna(), "recorded",
                    np.where(s.onset_of_symptom.notna(), "onset - incubation",
                             np.where(listed, "confirmation - (incubation + onset to confirmation)",
                                      "confirmation - infection to confirmation")))

    s_imported = s["abroad/local"] == "Abroad"
    upper = pd.concat([s.onset_of_symptom, s.confirmed_date,
                       s.arrival.where(s_imported)], axis=1).min(axis=1)
    lower = s.departure.where(s_imported)
    clipped = infection.clip(upper=upper)
    clipped = clipped.where(lower.isna() | (clipped >= lower), lower.clip(upper=upper))

    status, source = [], []
    for way, imp, arrival in zip(s.way_of_discover, s_imported, s.arrival, strict=True):
        state = STATUS.get(str(way).strip(), "community") if imp else "community"
        origin = "data"
        if policy and imp and state == "community" and arrival >= ALL_ARRIVALS_QUARANTINED:
            state, origin = "quarantine", "policy 2020-03-19"
        status.append(state)
        source.append(origin if imp else "local seed")

    age = s.age.astype(str).str.extract(r"(\d+)\s*to\s*(\d+)").astype(float)
    out = pd.DataFrame({
        "case_id": s.id.astype(int),
        "seed_type": np.where(s_imported, "imported", "local_unknown"),
        "age_low": age[0], "age_high": age[1],
        "gender": s.gender.where(s.gender.isin(["Male", "Female"]), ""),
        "infection_date": clipped.dt.date,
        "infection_rule": rule,
        "clipped": (clipped != infection),
        "arrival_date": s.arrival.where(s_imported).dt.date,
        "status": status,
        "status_source": source,
        "confirmed_date": s.confirmed_date.dt.date,
    })
    if local_before is not None:
        keep = (out.seed_type == "imported") | (pd.to_datetime(out.infection_date) < local_before)
        out = out[keep]
    day0 = pd.to_datetime(out.infection_date).min()
    out["infection_day"] = (pd.to_datetime(out.infection_date) - day0).dt.days
    out["arrival_day_from_infection"] = (pd.to_datetime(out.arrival_date)
                                         - pd.to_datetime(out.infection_date)).dt.days
    out = out.sort_values(["infection_day", "case_id"]).reset_index(drop=True)
    out.attrs.update(day0=str(day0.date()), incubation_median=incubation,
                     onset_to_confirmation_median=onset_to_confirmation,
                     infection_to_confirmation_median=infection_to_confirmation)
    return out


def main() -> None:
    """Write both scenarios' seed tables and print a summary."""
    cases_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CASES
    sources_path = Path(sys.argv[2]) if len(sys.argv) > 2 else SOURCES
    cases = pd.read_excel(cases_path, sheet_name="Individual Data")
    sources = pd.read_csv(sources_path)
    for name, (out_path, end, local_before, policy) in SCENARIOS.items():
        seeds = build(cases, sources, end, local_before, policy)
        seeds.to_csv(out_path, index=False)
        a = seeds.attrs
        print(f"[{name}] {len(seeds)} seeds -> {out_path}; day 0 = {a['day0']}")
        print(seeds.groupby(["seed_type", "status", "status_source"]).size().to_string())
    print(f"medians (days): incubation {a['incubation_median']}, onset to confirmation "
          f"{a['onset_to_confirmation_median']}, infection to confirmation "
          f"{a['infection_to_confirmation_median']}")


if __name__ == "__main__":
    main()
