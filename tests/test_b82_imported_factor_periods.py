# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B82: the imported community factor follows the border-control regime, not one global value.

B78 C multiplies the infection probabilities of every imported seed living in the community by
one factor, whose stated purpose is the home quarantine. Taiwan had no border screening and no
home quarantine before 2020-02-07, and the factor was calibrated over the 118 community seeds of
the 2020-04-12 scenario, 102 of which (86%) arrived later; applying it to the 16 that arrived
earlier suppresses exactly the seeds the February scenarios run on. The data itself puts the
local cases per seed 7.1x higher before 2020-02-07 (17 from 21 seeds) than after (38 from 333),
so a single factor cannot hold both periods.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from covsyn.model.data_synthesis_main import (BORDER_CONTROL_DATE, FIRST_WAVE_SEED_FILES,
                                              imported_factor_for, load_first_wave_seeds)

FEB7 = "taiwan_first_outbreak_feb7"
ALL = "taiwan_first_outbreak"
LAYERS = ("household", "school", "workplace", "health_care", "municipality")


def _run(vector, demographic_parameters, seed, mode=FEB7, **options):
    from covsyn.model.data_synthesis_main import run_covid
    return run_covid(seed, vector.copy(), copy.deepcopy(demographic_parameters), save_file=False,
                     mode=mode, **options)


def test_the_border_control_date_is_the_2020_02_07_entry_ban():
    assert BORDER_CONTROL_DATE == "2020-02-07"


@pytest.mark.parametrize(("arrival", "expected"), [
    ("2020-01-12", 1.0),     # the earliest seed, no screening and no quarantine existed
    ("2020-02-03", 1.0),     # the latest feb7 arrival
    ("2020-02-06", 1.0),     # the day before the ban
    ("2020-02-07", 0.25),    # the ban takes effect, so the factor applies from this day
    ("2020-03-19", 0.25),
    ("2020-04-11", 0.25),    # the latest seed of the 2020-04-12 scenario
])
def test_the_factor_applies_only_from_the_border_control_date(arrival, expected):
    assert imported_factor_for(arrival, 0.25) == expected


def test_a_seed_without_an_arrival_date_keeps_the_calibrated_factor():
    """A local seed of unknown source has no arrival date; it is not an imported seed, so the
    value is never read, and the function must not fail or silently return the pre-ban 1.0."""
    assert imported_factor_for(None, 0.25) == 0.25
    assert imported_factor_for("", 0.25) == 0.25


def test_a_factor_of_one_is_unchanged_in_both_periods():
    assert imported_factor_for("2020-01-20", 1.0) == 1.0
    assert imported_factor_for("2020-03-20", 1.0) == 1.0


def test_the_seed_table_keeps_the_arrival_date():
    seeds = load_first_wave_seeds(FIRST_WAVE_SEED_FILES[ALL])
    dated = [s for s in seeds if s["arrival_date"]]
    assert len(dated) == 341
    assert min(s["arrival_date"] for s in dated) == "2020-01-12"
    assert max(s["arrival_date"] for s in dated) == "2020-04-11"
    assert all(s["arrival_date"] is None for s in seeds if s["seed_type"] == "local_unknown")


def test_every_february_community_seed_arrived_before_the_ban():
    """The feb7 scenario's engine: 16 imported community seeds, every one of them pre-ban, so
    the regime factor leaves all of them untouched."""
    seeds = load_first_wave_seeds(FIRST_WAVE_SEED_FILES[FEB7])
    community = [s for s in seeds if s["seed_type"] == "imported" and s["status"] == "community"]
    assert len(community) == 16
    assert all(imported_factor_for(s["arrival_date"], 0.25) == 1.0 for s in community)


def test_the_april_scenario_splits_its_community_seeds_across_the_two_periods():
    seeds = load_first_wave_seeds(FIRST_WAVE_SEED_FILES[ALL])
    community = [s for s in seeds if s["seed_type"] == "imported" and s["status"] == "community"]
    before = [s for s in community if imported_factor_for(s["arrival_date"], 0.25) == 1.0]
    assert len(community) == 118
    assert len(before) == 16


def test_february_seeds_still_infect_when_the_factor_is_zero(run10_vector,
                                                             demographic_parameters):
    """The regression B82 fixes: with imported_factor = 0 the old code silenced every February
    community seed. They arrived before the ban, so they must still infect."""
    infected = 0.0
    for seed in range(6):
        _, _, courses, contacts = _run(run10_vector, demographic_parameters, seed,
                                       imported_factor=0.0)
        for c, x in zip(courses, contacts, strict=True):
            if c.get("seed_type") != "imported" or c.get("seed_status") != "community":
                continue
            infected += sum(
                float(np.nansum(np.asarray(x[f"{L}_effective_contacts"] or [], dtype=float)))
                for L in LAYERS)
    assert infected > 0.0


def test_a_post_ban_seed_is_still_silenced_by_a_zero_factor(run10_vector,
                                                            demographic_parameters):
    """The 2020-04-12 scenario keeps B78's behaviour for the seeds the factor was fitted on.

    A seed recalled into home quarantine on 2020-03-18 is excluded: from the recall day B73
    replaces the imported factor with the breach probability, so 49 of the 118 community seeds
    go on transmitting however small the factor is.
    """
    seeds = {s["case_id"]: s for s in load_first_wave_seeds(FIRST_WAVE_SEED_FILES[ALL])}
    silenced = {i for i, s in seeds.items()
                if s["seed_type"] == "imported" and s["status"] == "community"
                and s["quarantine_from_day"] is None
                and imported_factor_for(s["arrival_date"], 0.0) == 0.0}
    assert len(silenced) == 53     # 118 community seeds, less 16 pre-ban and 49 recalled

    infected = 0.0
    for seed in range(3):
        _, _, courses, contacts = _run(run10_vector, demographic_parameters, seed, mode=ALL,
                                       imported_factor=0.0)
        for c, x in zip(courses, contacts, strict=True):
            if c.get("seed_case_id") not in silenced:
                continue
            infected += sum(
                float(np.nansum(np.asarray(x[f"{L}_effective_contacts"] or [], dtype=float)))
                for L in LAYERS)
    assert infected == 0.0
