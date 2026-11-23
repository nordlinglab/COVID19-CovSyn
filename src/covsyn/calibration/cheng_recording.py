# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contacts recorded as Cheng et al. (2020)'s investigation recorded them (E84).

Cheng et al. (2020, Methods): "The period of investigation started at the date at symptom onset
(and could be extended to up to 4 days before symptom onset when epidemiologically indicated)
and ended at the date at COVID-19 confirmation." The objective used to bin the simulation's TRUE
first contact, so a household member met since infection always fell before onset and Cheng's
household bins after onset could not be reproduced (finding E84; probes probe_cheng_window and
probe_cheng_arrival, 2026-11-23). Since then (user decision 2026-11-23):

* a contact's bin is its first day of exposure inside the investigation period, both ends
  included; the period ends at the case's confirmation, counted from its own infection;
* a share EXTENSION_SHARE of investigations starts EXTENSION_DAYS before onset, the rest at
  onset; the two records' counts are mixed in expectation, so no random number is drawn and
  the counts still add across cases and workers;
* each simulated index case takes the profile of one of Cheng's 100 cases (draw_profile): its
  age, and for an imported case its arrival, before which it met nobody in Taiwan, so nothing
  before arrival is recorded and an infection before arrival is not counted; a case found at
  the airport has no recorded contact. Only the record changes, not the simulation, so the
  other cost terms see the same transmission (design review
  logic-review_openai_2026-11-23_2130_cost-change-design_reply.md, point 2).

EXTENSION_SHARE = 0.9: the share for which the household contacts before onset match Cheng's
100 of 151, between 0.85 and 0.92 under the two treatments of the 20 imported cases without
both dates (probe_cheng_arrival, run 19); not a free parameter. Sensitivity: 0.5, 0.75, 1.0.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

import numpy as np

from covsyn.figures.plot_results import _cheng_day_bin
from covsyn.model.data_synthesis_main import run_covid

PROFILES = Path("variable/cheng_period_profiles.csv")  # build_cheng_profiles
PROFILE_STREAM = 84  # the profile draw's own generator, [seed, PROFILE_STREAM] (E84)
EXTENSION_SHARE = 0.9
EXTENSION_DAYS = 4
MISSING_ARRIVAL = "resample"  # treatment A; "before" (B) is the sensitivity
MISSING_TREATMENTS = ("resample", "before")

# Cheng's three groups, in the objective's order, and the layers each is built from.
GROUPS = (("household", ("household",)), ("health_care", ("health_care",)),
          ("others", ("school", "workplace", "municipality")))
MATRIX = {"household": "household_contacts_matrix", "school": "school_class_contacts_matrix",
          "workplace": "workplace_contacts_matrix", "health_care": "health_care_contacts_matrix",
          "municipality": "municipality_contacts_matrix"}


def load_profiles(path: Path = PROFILES) -> list[dict]:
    """The profiles table as dicts with float ages and arrivals (NaN when unknown)."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{"case_id": int(row["case_id"]),
                 "age_low": float(row["age_low"]) if row["age_low"] else np.nan,
                 "age_high": float(row["age_high"]) if row["age_high"] else np.nan,
                 "imported": row["imported"] == "True",
                 "airport": row["airport"] == "True",
                 "arrival_from_onset": (float(row["arrival_from_onset"])
                                        if row["arrival_from_onset"] else np.nan)}
                for row in csv.DictReader(f)]


@functools.cache
def default_profiles() -> list[dict]:
    """The committed profiles, read once per process."""
    return load_profiles()


def simulate(seed, P, demographic_parameters, profiles=None, **run_covid_kwargs):
    """run_covid for the objective: the index case takes the seed's profile and its age.

    Returns:
        (run_covid's output, the profile).
    """
    profile = draw_profile(seed, default_profiles() if profiles is None else profiles)
    out = run_covid(seed, P, demographic_parameters, save_file=False, source_age=profile["age"],
                    **run_covid_kwargs)
    return out, profile


def draw_profile(seed: int, profiles: list[dict], missing: str = MISSING_ARRIVAL) -> dict:
    """The index-case profile of the simulation with this seed.

    Drawn with its own generator, so the model's random draws are untouched. The age is a whole
    year within the case's band (None without one, leaving it to the model); an imported case
    without a known arrival takes one drawn from the known arrivals ("resample") or none, i.e.
    arrived before onset ("before").

    Returns:
        {'case_id', 'age', 'imported', 'airport', 'arrival_from_onset'}, the arrival an int
        or None.
    """
    if missing not in MISSING_TREATMENTS:
        raise ValueError(f"missing must be one of {MISSING_TREATMENTS}, got {missing!r}")
    if not profiles:
        raise ValueError("no index-case profile to draw from")
    for p in profiles:  # run_covid's age tables run from 0 to 100 (code review 2026-11-23)
        if np.isfinite([p["age_low"], p["age_high"]]).all() and not (
                0 <= p["age_low"] <= p["age_high"] <= 100):
            raise ValueError(f"case {p['case_id']}: age band {p['age_low']}-{p['age_high']} "
                             "lies outside 0-100")
    rng = np.random.default_rng([int(seed), PROFILE_STREAM])
    case = profiles[int(rng.integers(len(profiles)))]
    low, high = case["age_low"], case["age_high"]
    age = int(rng.integers(int(low), int(high) + 1)) if np.isfinite([low, high]).all() else None
    arrival = None
    if case["imported"]:
        if np.isfinite(case["arrival_from_onset"]):
            arrival = int(case["arrival_from_onset"])
        elif missing == "resample":
            known = [p["arrival_from_onset"] for p in profiles
                     if p["imported"] and np.isfinite(p["arrival_from_onset"])]
            arrival = int(known[int(rng.integers(len(known)))]) if known else None
    return {"case_id": case["case_id"], "age": age, "imported": case["imported"],
            "airport": case["airport"], "arrival_from_onset": arrival}


def arrival_day(profile: dict, onset: int):
    """The day of arrival counted from infection, or None (local, or arrival unknown)."""
    if not profile["imported"] or profile["arrival_from_onset"] is None:
        return None
    return int(onset + profile["arrival_from_onset"])


def recorded_bins(course: dict, contact: dict, back, arrival=None, airport=False) -> np.ndarray:
    """One case's recorded contacts and infections per group and Cheng bin, shape (3, 2, 6).

    Args:
        course: The case's course-of-disease record.
        contact: The case's contact record.
        back: Days before onset the investigation starts; None records the true first contact
            (the objective before E84).
        arrival: The day, counted from infection, before which nothing is recorded; None for
            no arrival.
        airport: Found at the airport: nothing is recorded.
    """
    out = np.zeros((len(GROUPS), 2, 6))
    onset = course["incubation_period"]
    if onset is None or np.isnan(onset) or airport:
        return out
    onset = int(onset)
    if back is None:
        start, end = -np.inf, np.inf
    else:
        end = course["positive_test_date"] - course["infection_day"]
        # A traced case can be confirmed before onset; its investigation then starts no later
        # than its confirmation, so the period is not empty (code review 2026-11-23).
        start = min(onset, end) - back
    if arrival is not None:
        start = max(start, arrival)
    for gi, (_, layers) in enumerate(GROUPS):
        for layer in layers:
            _record_layer(out[gi], contact, layer, onset, start, end, arrival)
    return out


def _record_layer(out, contact, layer, onset, start, end, arrival):
    """Add one layer's recorded contacts and infections to out, shape (2, 6)."""
    matrix = np.asarray(contact[MATRIX[layer]], dtype=bool)
    events = None
    if layer == "municipality":
        mask = contact.get("municipality_event_mask")
        expected = contact.get("municipality_event_expected_contacts")
        if mask is not None and expected is not None:
            events = np.asarray(mask, dtype=bool)
    if matrix.ndim == 2 and matrix.shape[0]:
        times = np.full(matrix.shape[0], np.nan)
        recorded = np.asarray(contact.get(f"{layer}_effective_contacts_infection_time") or [],
                              dtype=float)
        times[:len(recorded)] = recorded  # the loop stops when the population runs out
        if events is not None and len(events) != matrix.shape[0]:
            raise ValueError(f"municipality_event_mask has {len(events)} entries for "
                             f"{matrix.shape[0]} contact rows")
        for m, row in enumerate(matrix):
            if events is not None and events[m]:
                continue  # B55: replaced by the expectation below
            days = np.flatnonzero(row)
            days = days[(days >= start) & (days <= end)]
            if days.size == 0:
                continue
            b = _cheng_day_bin(int(days[0]) - onset)
            out[0, b] += 1
            if not np.isnan(times[m]) and (arrival is None or times[m] >= arrival):
                out[1, b] += 1
    if events is not None:
        expected_infections = contact.get("municipality_event_expected_infections")
        for day, count in enumerate(contact["municipality_event_expected_contacts"]):
            if start <= day <= end:
                b = _cheng_day_bin(day - onset)
                out[0, b] += count
                if expected_infections is not None:
                    out[1, b] += expected_infections[day]


def case_bins(course: dict, contact: dict, profile: dict | None = None,
              share: float = EXTENSION_SHARE) -> np.ndarray:
    """The objective's bins for one case: share x extended record + (1 - share) x from onset.

    profile, the index case's (draw_profile), adds its arrival or airport isolation; None for a
    secondary case.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"the extension share must lie in [0, 1], got {share!r}")
    arrival, airport = None, False
    onset = course["incubation_period"]
    if profile is not None and onset is not None and not np.isnan(onset):
        arrival = arrival_day(profile, int(onset))
        airport = profile["airport"]
    extended = recorded_bins(course, contact, EXTENSION_DAYS, arrival, airport)
    from_onset = recorded_bins(course, contact, 0, arrival, airport)
    return share * extended + (1.0 - share) * from_onset
