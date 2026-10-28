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

``feb29`` and ``mar18`` (added 2026-10-25 after run 18)
    The same construction to 2020-02-29 (before the March wave of returning residents) and to
    2020-03-18, the confirmation of case 100 and so the span of Cheng et al. (2020); no policy
    restriction, since every arrival was home quarantined only from 2020-03-19. ``mar18`` is the
    main comparison for the imported seeds; ``all`` is kept for information, as only 7 local
    cases are added after 2020-03-18.

``mar18_local`` (B68, local start)
    No imported seed: the local cases infected by an imported case that arrived by 2020-03-18
    (14 in the data) and the 11 local cases of unknown source infected by then. The model then
    only has to carry local transmission, which is what the calibration fits (B67), while the
    imported cases' own offspring are reported separately. Each linked seed is infected no
    earlier than its infector's arrival, and is traced from its infector's confirmation
    (``source_confirmed_day``), as it was in the data.

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

Recall into home quarantine on 2020-03-18 (B73, user decision 2026-10-28)
    On 2020-03-18 the CECC applied home quarantine retroactively to the people who had arrived
    from Europe, Egypt, Turkey or Dubai (transit included) between 2020-03-05 and 2020-03-14
    (MOHW press release 2020-03-18, https://mohw.gov.tw/cp-199-52426-2.html). Such a seed whose
    status is 'community' stays in the community until 2020-03-18 and is in home quarantine from
    then on (``quarantine_from_day_from_infection``); the policy did not exist before, so nothing
    changes earlier. The place is matched on the case's visited countries.

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
# confirmation of case 100, the last case of Cheng et al. (2020)
CHENG2020_LAST_CONFIRMATION = pd.Timestamp("2020-03-18")
# scenario -> (output file, last arrival, local seeds infected before, home quarantine policy)
SCENARIOS = {
    "all": (Path("variable/taiwan_first_wave_seeds.csv"), FIRST_WAVE_END, None, True),
    "feb7": (Path("variable/taiwan_first_wave_seeds_feb7.csv"),
             BORDER_CONTROL - pd.Timedelta(days=1), BORDER_CONTROL, False),
    "feb29": (Path("variable/taiwan_first_wave_seeds_feb29.csv"),
              pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-01"), False),
    "mar18": (Path("variable/taiwan_first_wave_seeds_mar18.csv"),
              CHENG2020_LAST_CONFIRMATION, CHENG2020_LAST_CONFIRMATION + pd.Timedelta(days=1),
              False),
}
# scenario -> (output file, last arrival of the infectors, local unknown seeds infected before)
LOCAL_START_SCENARIOS = {
    "mar18_local": (Path("variable/taiwan_first_wave_seeds_mar18_local.csv"),
                    CHENG2020_LAST_CONFIRMATION,
                    CHENG2020_LAST_CONFIRMATION + pd.Timedelta(days=1)),
}
STATUS = {"airport screening": "airport", "home quarantine": "quarantine"}
# B73: retroactive home quarantine announced 2020-03-18 (MOHW, https://mohw.gov.tw/cp-199-52426-2.html)
RECALL_DATE = pd.Timestamp("2020-03-18")
RECALL_ARRIVALS = (pd.Timestamp("2020-03-05"), pd.Timestamp("2020-03-14"))
RECALL_PLACES = ("Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia", "Bulgaria",
                 "Croatia", "Cyprus", "Czech", "Denmark", "England", "Estonia", "Finland", "France",
                 "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia",
                 "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco",
                 "Montenegro", "Netherlands", "Macedonia", "Norway", "Poland", "Portugal", "Romania",
                 "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
                 "Switzerland", "Ukraine", "United Kingdom", "UK", "Vatican", "Europe",
                 "Egypt", "Turkey", "Dubai", "UAE", "United Arab Emirates")


def median_days(later: pd.Series, earlier: pd.Series) -> float:
    """Median of (later - earlier) in days over the rows that have both."""
    days = (later - earlier).dt.days.dropna()
    return float(np.median(days))


def build(cases: pd.DataFrame, sources: pd.DataFrame,
          end: pd.Timestamp = FIRST_WAVE_END, local_before: pd.Timestamp | None = None,
          policy: bool = True, local_start: bool = False) -> pd.DataFrame:
    """The seed table: one row per seed case, ordered by infection day.

    Args:
        cases: The 'Individual Data' sheet of the Taiwan case data.
        sources: The local-case source classification.
        end: Last arrival date of an imported seed.
        local_before: Keep only local seeds infected before this date; None keeps all 13.
        policy: Apply the home quarantine of every arrival from 2020-03-19.
        local_start: Replace the imported seeds by the local cases they infected (B68).
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
    infector_arrival = pd.Series(dtype="datetime64[ns]")
    infector_confirmed = pd.Series(dtype="datetime64[ns]")
    if local_start:
        infectors = d[seed & imported].set_index("id")
        infector_id = pd.to_numeric(sources.infector_id, errors="coerce")
        named = infector_id.notna() & infector_id.isin(infectors.index)
        infector_of = dict(zip(sources.id[named].astype(int), infector_id[named].astype(int)))
        seed = d.id.isin(set(infector_of) | unknown)
        s = d[seed].copy()
        source = s.id.map(infector_of)
        infector_arrival = source.map(infectors.arrival)
        infector_confirmed = source.map(infectors.confirmed_date)
    else:
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
    if local_start:
        lower = lower.fillna(infector_arrival)
    clipped = infection.clip(upper=upper)
    clipped = clipped.where(lower.isna() | (clipped >= lower), lower.clip(upper=upper))

    places = s["visited_countries/cities"].astype(str).str.lower()
    recalled = (s_imported & s.arrival.between(*RECALL_ARRIVALS)
                & places.apply(lambda v: any(p.lower() in v for p in RECALL_PLACES)))
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
        "seed_type": np.where(s_imported, "imported",
                              np.where(s.id.isin(unknown), "local_unknown",
                                       "local_from_imported")),
        "age_low": age[0], "age_high": age[1],
        "gender": s.gender.where(s.gender.isin(["Male", "Female"]), ""),
        "infection_date": clipped.dt.date,
        "infection_rule": rule,
        "clipped": (clipped != infection),
        "arrival_date": s.arrival.where(s_imported).dt.date,
        "status": status,
        "status_source": source,
        "recalled": recalled & (pd.Series(status, index=s.index) == "community"),
        "confirmed_date": s.confirmed_date.dt.date,
        "source_confirmed_date": pd.to_datetime(infector_confirmed.reindex(s.index)).dt.date,
    })
    if local_before is not None:
        # only the seeds of unknown source are cut by their infection date; a local seed
        # infected by an imported case belongs to its infector's scenario whenever it was infected
        keep = ((out.seed_type != "local_unknown")
                | (pd.to_datetime(out.infection_date) < local_before))
        out = out[keep]
    day0 = pd.to_datetime(out.infection_date).min()
    out["infection_day"] = (pd.to_datetime(out.infection_date) - day0).dt.days
    out["arrival_day_from_infection"] = (pd.to_datetime(out.arrival_date)
                                         - pd.to_datetime(out.infection_date)).dt.days
    out["source_confirmed_day"] = (pd.to_datetime(out.source_confirmed_date) - day0).dt.days
    out["quarantine_from_day_from_infection"] = (
        (RECALL_DATE - pd.to_datetime(out.infection_date)).dt.days.where(out.recalled))
    out = out.drop(columns="recalled")
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
    for name, (out_path, end, local_before) in LOCAL_START_SCENARIOS.items():
        seeds = build(cases, sources, end, local_before, policy=False, local_start=True)
        seeds.to_csv(out_path, index=False)
        print(f"[{name}] {len(seeds)} seeds -> {out_path}; day 0 = {seeds.attrs['day0']}")
        print(seeds.groupby("seed_type").size().to_string())
    print(f"medians (days): incubation {a['incubation_median']}, onset to confirmation "
          f"{a['onset_to_confirmation_median']}, infection to confirmation "
          f"{a['infection_to_confirmation_median']}")


if __name__ == "__main__":
    main()
