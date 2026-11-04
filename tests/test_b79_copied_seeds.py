# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Professor meeting 2026-11-02: B74 and B78 defaults, B79 copied seeds for February."""

import copy
import inspect

import numpy as np
import pandas as pd
import pytest

from covsyn.data_processing import build_first_wave_seeds as bfs

FEB7 = "variable/taiwan_first_wave_seeds_feb7.csv"


@pytest.fixture(scope="module")
def cases():
    return pd.read_excel(bfs.CASES, sheet_name="Individual Data")


@pytest.fixture(scope="module")
def feb7():
    return pd.read_csv(FEB7)


def _copies(table):
    return table[table.copied_from.notna()]


def test_quarantine_breach_default_counts_the_two_cases_met_on_the_way_home():
    """B74: 5 of the 122 quarantined seeds infected someone, the trip home included."""
    from covsyn.model.data_synthesis_main import run_covid
    assert inspect.signature(run_covid).parameters["quarantine_breach"].default == "with_unclear"


def test_imported_isolation_keeps_model_and_imported_delay_and_defaults_to_the_data():
    """B78: A2 + C chosen; A1 ('observed') is dropped."""
    from covsyn.model.data_synthesis_main import IMPORTED_ISOLATION_OPTIONS, run_covid
    assert IMPORTED_ISOLATION_OPTIONS == ("model", "imported_delay")
    default = inspect.signature(run_covid).parameters["imported_isolation"].default
    assert default == "imported_delay"


def test_zero_copies_return_the_table_unchanged(feb7, cases):
    out = bfs.copy_community_seeds(feb7, cases, 0, np.random.default_rng(0))
    assert len(out) == len(feb7)
    assert out.copied_from.isna().all()


def test_negative_count_is_rejected(feb7, cases):
    with pytest.raises(ValueError):
        bfs.copy_community_seeds(feb7, cases, -1, np.random.default_rng(0))


def test_no_community_seed_to_copy_is_rejected(feb7, cases):
    airport_only = feb7[feb7.status == "airport"]
    with pytest.raises(ValueError):
        bfs.copy_community_seeds(airport_only, cases, 1, np.random.default_rng(0))


def test_copies_come_from_community_seeds_and_keep_their_properties(feb7, cases):
    out = bfs.copy_community_seeds(feb7, cases, 40, np.random.default_rng(1))
    copies = _copies(out)
    assert len(out) == len(feb7) + 40 and len(copies) == 40
    source = feb7.set_index("case_id")
    assert set(copies.copied_from.astype(int)) <= set(feb7.case_id[feb7.status == "community"])
    for column in ("seed_type", "status", "age_low", "age_high", "gender", "arrival_date",
                   "confirmed_date"):
        kept = source.loc[copies.copied_from.astype(int), column].to_numpy()
        assert pd.Series(copies[column].to_numpy()).equals(pd.Series(kept)), column


def test_copies_have_new_ids_outside_the_case_data(feb7, cases):
    """The ids must not match a reported case, or the reference cases would change."""
    copies = _copies(bfs.copy_community_seeds(feb7, cases, 30, np.random.default_rng(2)))
    assert copies.case_id.is_unique
    assert not set(copies.case_id) & set(cases.id.astype(int))


def test_copy_dates_are_drawn_not_fixed_and_stay_within_the_record(feb7, cases):
    out = bfs.copy_community_seeds(feb7, cases, 200, np.random.default_rng(3))
    copies = _copies(out)
    infection = pd.to_datetime(copies.infection_date)
    assert copies.groupby("copied_from").infection_date.nunique().max() > 1
    record = cases.set_index("id")
    onset = pd.to_datetime(record.loc[copies.copied_from.astype(int), "onset_of_symptom"],
                           errors="coerce").to_numpy()
    confirmed = pd.to_datetime(copies.confirmed_date).to_numpy()
    arrival = pd.to_datetime(copies.arrival_date).to_numpy()
    assert (infection.to_numpy() <= confirmed).all()
    assert not (infection.to_numpy() > onset).any()
    assert not (infection.to_numpy() > arrival).any()
    assert copies.infection_rule.str.startswith("copy:").all()


def test_infection_days_are_recounted_from_the_earliest_seed(feb7, cases):
    out = bfs.copy_community_seeds(feb7, cases, 50, np.random.default_rng(4))
    dates = pd.to_datetime(out.infection_date)
    assert out.infection_day.min() == 0
    assert (out.infection_day == (dates - dates.min()).dt.days).all()
    arrival = (pd.to_datetime(out.arrival_date) - dates).dt.days
    assert out.arrival_day_from_infection.equals(arrival)


def test_a_larger_set_extends_a_smaller_one(feb7, cases):
    """The same generator gives nested sets, so adding seeds never redraws the earlier ones."""
    small = _copies(bfs.copy_community_seeds(feb7, cases, 3, np.random.default_rng(5)))
    large = _copies(bfs.copy_community_seeds(feb7, cases, 8, np.random.default_rng(5)))
    columns = ["case_id", "copied_from", "infection_date"]
    pd.testing.assert_frame_equal(small[columns].sort_values("case_id").reset_index(drop=True),
                                  large[large.case_id.isin(small.case_id)][columns]
                                  .sort_values("case_id").reset_index(drop=True))


def test_copied_seeds_are_written_only_for_february(tmp_path, cases):
    path = bfs.write_copied_seeds("feb7", 2, cases, directory=tmp_path)
    assert path.name == "taiwan_first_wave_seeds_feb7_copies2.csv"
    assert len(_copies(pd.read_csv(path))) == 2
    with pytest.raises(ValueError):
        bfs.write_copied_seeds("mar18", 2, cases, directory=tmp_path)


def test_reference_cases_do_not_change_with_copies(feb7, cases):
    from covsyn.validation.compare_first_wave import SOURCES, reference_local_cases
    sources = pd.read_csv(SOURCES, dtype=str, keep_default_na=False)
    out = bfs.copy_community_seeds(feb7, cases, 10, np.random.default_rng(6))
    assert reference_local_cases(out, sources) == reference_local_cases(feb7, sources)


def test_run_covid_reads_the_copied_seed_file(tmp_path, cases, run10_vector,
                                              demographic_parameters):
    """A copied local seed is a local case of the run, so the comparison counts it."""
    from covsyn.model.data_synthesis_main import run_covid
    path = bfs.write_copied_seeds("feb7", 5, cases, directory=tmp_path)
    vector = np.concatenate([run10_vector, [1.0, 1.67, 21, 1000, 1.0], [1.74], [10.0, 1.0], [1.0]])
    _, _, courses, _ = run_covid(0, vector, copy.deepcopy(demographic_parameters),
                                 save_file=False, mode="taiwan_first_outbreak_feb7",
                                 seed_file=str(path))
    seeded = [c for c in courses if c.get("seed_type") is not None]
    assert len(seeded) == len(pd.read_csv(path))
    assert any(c["seed_case_id"] < 0 for c in seeded)


def test_calibrated_q_and_p_are_the_defaults_under_imported_delay():
    """The calibrated constants, with the measurement each one came from.

    p: run 19 candidate 60 under A2, 400 runs (Mac, 2026-11-02), 0.061 / 0.099.
    q: B82 (2026-11-04) recalibrated it on the run 21 selected vector and made it the POST-BAN
    value, the 16 seeds that arrived before the 2020-02-07 entry ban being held at 1.0. The
    relation is then m(q) = A + B*q with A measured at 7.27 of the data's 12 infections, so
    the old proportional calibration no longer applies and 0.294 became 0.107.
    """
    import covsyn.model.data_synthesize as ds
    from covsyn.model.data_synthesis_main import IMPORTED_FACTOR, run_covid
    assert IMPORTED_FACTOR == pytest.approx(0.107)
    assert inspect.signature(run_covid).parameters["imported_factor"].default == IMPORTED_FACTOR
    assert ds.QUARANTINE_BREACH == {"confirmed": 0.061, "with_unclear": 0.099}
