# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B64 work group, B65 tracing delay, B66 first-wave seeds, B67 Cheng local index cases."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import covsyn.model.data_synthesize as ds
from covsyn.calibration import sar_anchors
from covsyn.data_processing.build_first_wave_seeds import build


def test_work_group_parameters_need_the_appended_values():
    """Vectors up to P[205] keep B31's group; a and k must be positive."""
    assert ds.work_group_parameters(np.ones(205)) is None
    assert ds.work_group_parameters(np.r_[np.ones(205), 12.0, 0.5]) == (12.0, 0.5)
    with pytest.raises(ValueError):
        ds.work_group_parameters(np.r_[np.ones(205), 0.0, 0.5])


def test_tracing_delay_is_the_constant_for_older_vectors():
    """P[207] replaces TRACING_DELAY_MEAN only when present."""
    assert ds.tracing_delay_mean(np.ones(207)) == ds.TRACING_DELAY_MEAN
    assert ds.tracing_delay_mean(np.r_[np.ones(207), 2.5]) == 2.5


def test_work_group_never_exceeds_the_coworkers_and_small_companies_meet_everyone():
    """The group is capped by the coworkers and saturates near a in large companies."""
    np.random.seed(0)
    assert ds.draw_work_group(0, 10.0, 1.0) == 0
    groups = [ds.draw_work_group(4, 10.0, 1.0) for _ in range(2000)]
    assert max(groups) <= 4
    # mean (n - 1) a / (a + n - 1) = 4 * 10 / 14 = 2.86 before the cap at 4
    large = [ds.draw_work_group(999, 10.0, 5.0) for _ in range(2000)]
    assert 8.0 < np.mean(large) < 12.0


def test_fixed_gender_is_kept():
    """A seed's gender from the case data is not redrawn."""
    import pickle

    d = pickle.load(open("variable/demographic_parameters.pkl", "rb"))
    demo = ds.Draw_demographic_data(d[0], d[1], d[2], d[3], d[4], 30, gender="Female")
    demo.draw_demographic_data()
    assert demo.gender == "Female" and demo.age == 30


def test_cheng_local_scale_is_local_over_all_infections_per_index_case():
    """The Cheng per-index targets are scaled to local index cases (B67)."""
    assert sar_anchors.CHENG2020_LOCAL_SCALE == pytest.approx((18 / 29) / (22 / 100))
    lo, centre, hi = sar_anchors.LAYER_INFECTIONS_PER_INDEX["household"]
    assert centre == pytest.approx(0.10 * sar_anchors.CHENG2020_LOCAL_SCALE)
    assert lo < centre < hi


def test_first_wave_seeds_from_the_case_data():
    """The committed seed table matches a rebuild from the case data (B66)."""
    cases = pd.read_excel(Path("data/structured_course_of_disease_data/figshare_taiwan_covid.xlsx"),
                          sheet_name="Individual Data")
    seeds = build(cases, pd.read_csv("validation_reference/taiwan_local_case_sources.csv"))
    assert (seeds.seed_type == "local_unknown").sum() == 13
    assert set(seeds.status) <= {"airport", "quarantine", "community"}
    assert (seeds.loc[seeds.seed_type == "local_unknown", "status"] == "community").all()
    imported = seeds[seeds.seed_type == "imported"]
    assert (imported.arrival_day_from_infection >= 0).all()
    assert seeds.infection_day.min() == 0
    committed = pd.read_csv("variable/taiwan_first_wave_seeds.csv")
    assert len(committed) == len(seeds)
    # restrictions grow over time: from 2020-03-19 no imported seed lives in the community
    late = pd.to_datetime(imported.arrival_date) >= pd.Timestamp("2020-03-19")
    assert (imported.loc[late, "status"] != "community").all()


def test_feb7_scenario_has_no_policy_and_its_reference_is_seventeen_local_cases():
    """The 2/7 seeds carry no quarantine; 17 reported local cases descend from them."""
    from covsyn.validation.compare_first_wave import reference_local_cases

    sources = pd.read_csv("validation_reference/taiwan_local_case_sources.csv",
                          dtype=str, keep_default_na=False)
    feb7 = pd.read_csv("variable/taiwan_first_wave_seeds_feb7.csv")
    assert (feb7.status != "quarantine").all()
    assert (pd.to_datetime(feb7.arrival_date.dropna()) < pd.Timestamp("2020-02-07")).all()
    assert set(feb7.loc[feb7.seed_type == "local_unknown", "case_id"]) == {19, 24, 27}
    assert len(reference_local_cases(feb7, sources)) == 17
    every = pd.read_csv("variable/taiwan_first_wave_seeds.csv")
    assert len(reference_local_cases(every, sources)) == 55


def test_feb29_and_mar18_scenarios_match_their_cut_offs():
    """Cut-offs 2020-02-29 and 2020-03-18 (case 100, Cheng 2020): 28 and 48 local cases."""
    from covsyn.validation.compare_first_wave import reference_local_cases

    sources = pd.read_csv("validation_reference/taiwan_local_case_sources.csv",
                          dtype=str, keep_default_na=False)
    for name, end, n_seeds, n_local in (("feb29", "2020-02-29", 26, 28),
                                        ("mar18", "2020-03-18", 186, 48)):
        seeds = pd.read_csv(f"variable/taiwan_first_wave_seeds_{name}.csv")
        assert len(seeds) == n_seeds
        assert (pd.to_datetime(seeds.arrival_date.dropna()) <= pd.Timestamp(end)).all()
        assert (seeds.status_source != "policy 2020-03-19").all()
        assert len(reference_local_cases(seeds, sources)) == n_local


def test_local_start_scenario_replaces_imported_seeds_by_their_local_offspring():
    """B68: 14 local cases infected by an imported case plus 11 unknown; same 48 local cases."""
    from covsyn.validation.compare_first_wave import reference_local_cases

    sources = pd.read_csv("validation_reference/taiwan_local_case_sources.csv",
                          dtype=str, keep_default_na=False)
    seeds = pd.read_csv("variable/taiwan_first_wave_seeds_mar18_local.csv")
    assert seeds.seed_type.value_counts().to_dict() == {"local_from_imported": 14,
                                                        "local_unknown": 11}
    assert (seeds.status == "community").all() and seeds.arrival_date.isna().all()
    linked = seeds[seeds.seed_type == "local_from_imported"]
    assert linked.source_confirmed_day.notna().all()
    assert seeds.loc[seeds.seed_type == "local_unknown", "source_confirmed_day"].isna().all()
    assert len(reference_local_cases(seeds, sources)) == 48


def test_quarantine_breach_has_the_two_calibrated_options():
    """B74: the breach probability for 3 and for 5 infections by quarantined seeds."""
    assert set(ds.QUARANTINE_BREACH) == {"confirmed", "with_unclear"}
    assert 0 < ds.QUARANTINE_BREACH["confirmed"] < ds.QUARANTINE_BREACH["with_unclear"] < 1


class _Course:
    monitor_isolation_period = 9
    incubation_period = 5.0


def _contact_data(rule):
    c = ds.Draw_contact_data.__new__(ds.Draw_contact_data)
    c.seed_rule = rule
    for name in ("household_contacts_matrix", "school_class_contacts_matrix",
                 "workplace_contacts_matrix", "health_care_contacts_matrix",
                 "municipality_contacts_matrix"):
        setattr(c, name, np.ones((3, 10), dtype=bool))
    return c


def test_seed_rule_removes_days_before_arrival_and_layers_the_status_forbids():
    """No contact before arrival; quarantine closes health care only (B68, B74), airport all."""
    c = _contact_data({"arrival_day": 4, "status": "quarantine"})
    c.apply_seed_rule()
    assert c.household_contacts_matrix[:, :4].sum() == 0
    assert c.household_contacts_matrix[:, 4:].all()
    assert c.workplace_contacts_matrix[:, 4:].all()
    assert c.municipality_contacts_matrix[:, 4:].all()
    assert c.health_care_contacts_matrix.shape[0] == 0
    c = _contact_data({"arrival_day": 0, "status": "airport"})
    c.apply_seed_rule()
    assert c.household_contacts_matrix.shape[0] == 0
    assert c.health_care_contacts_matrix.shape[0] == 0
    c = _contact_data({"arrival_day": 2, "status": "community"})
    c.apply_seed_rule()
    assert c.municipality_contacts_matrix[:, :2].sum() == 0
    assert c.municipality_contacts_matrix[:, 2:].all()


def test_warm_start_pads_a_run_17_vector():
    """A 205-value run-17 vector gets P[205..207] from the seed vector."""
    pytest.importorskip("sklearn")
    from covsyn.calibration import firefly_optimizer as fo

    seed_vector = np.arange(208, dtype=float)
    padded = fo.pad_pre_b54_vector(np.zeros(205), seed_vector)
    assert padded.shape == (208,) and padded[205] == 205.0


def test_seeds_recalled_into_quarantine_on_march_18():
    """B73: 'community' arrivals from Europe etc. of Mar 5-14 are quarantined from Mar 18."""
    seeds = pd.read_csv("variable/taiwan_first_wave_seeds.csv", parse_dates=["infection_date"])
    recalled = seeds.dropna(subset=["quarantine_from_day_from_infection"])
    assert len(recalled) == 49 and (recalled.status == "community").all()
    days = (pd.Timestamp("2020-03-18") - recalled.infection_date).dt.days
    assert (recalled.quarantine_from_day_from_infection == days).all()
    feb29 = pd.read_csv("variable/taiwan_first_wave_seeds_feb29.csv")
    assert feb29.quarantine_from_day_from_infection.isna().all()


def test_recalled_seed_loses_health_care_from_the_recall_day():
    """B73: health care is closed from the recall day on, the other layers stay open."""
    c = _contact_data({"arrival_day": 2, "status": "community", "quarantine_from_day": 6})
    c.apply_seed_rule()
    assert c.health_care_contacts_matrix[:, 2:6].all()
    assert c.health_care_contacts_matrix[:, 6:].sum() == 0
    assert c.workplace_contacts_matrix[:, 2:].all()
