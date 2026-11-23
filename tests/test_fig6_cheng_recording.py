# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fig 6 bins its simulated cohort as the objective does: as Cheng's investigation recorded it.

Code review 2026-11-23: the figure still binned the true first contact, so after E84 the
calibrated model would have looked unlike Cheng where the objective matched the recorded bins.
Each index case is recorded with the profile its course record carries (run_covid's
source_profiles, the cheng2020 synthesis).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from covsyn.calibration import cheng_recording as cr
from covsyn.calibration.firefly_optimizer import CHENG_SYMPTOMATIC_INDEX_CASES
from covsyn.figures import reproduce_wu2025 as rw
from covsyn.model.data_synthesis_main import (
    run_covid,
    source_profiles_for,
    source_profiles_metadata,
)


def test_fig6_bins_are_the_recorded_bins_of_each_case(
    tmp_path, run10_vector, demographic_parameters
):
    """Per run: the sum of case_bins, each index case with its profile, on Cheng's scale."""
    want = []
    for run, seed in enumerate((11, 12)):
        _, _, courses, contacts = run_covid(
            seed, np.array(run10_vector), copy.deepcopy(demographic_parameters),
            save_file=False, mode="cheng2020",
            source_profiles=source_profiles_for("cheng2020", seed))
        np.save(tmp_path / f"course_of_disease_data_{run}.npy", np.array(courses, dtype=object))
        np.save(tmp_path / f"contact_data_{run}.npy", np.array(contacts, dtype=object))
        bins = sum(cr.case_bins(c, k, c.get("index_profile"))
                   for c, k in zip(courses, contacts, strict=True))
        symptomatic = sum(not np.isnan(c["incubation_period"]) for c in courses)
        want.append(bins * CHENG_SYMPTOMATIC_INDEX_CASES / symptomatic)
    assert all("index_profile" in c for c in courses)
    data, runs = rw.cheng_bins(tmp_path, ["Household", "Health care", "All"])
    assert runs == 2
    for run, bins in enumerate(want):
        assert np.allclose(data["Household|contacts"][run], bins[0, 0])
        assert np.allclose(data["Health care|infections"][run], bins[1, 1])
        assert np.allclose(data["All|contacts"][run], bins[:, 0].sum(axis=0))


def test_fig6_refuses_data_whose_index_cases_have_no_profile(
    tmp_path, run10_vector, demographic_parameters
):
    """Data synthesised before E84 would be binned without arrival or airport isolation."""
    _, _, courses, contacts = run_covid(
        11, np.array(run10_vector), copy.deepcopy(demographic_parameters), save_file=False,
        mode="cheng2020")
    np.save(tmp_path / "course_of_disease_data_0.npy", np.array(courses, dtype=object))
    np.save(tmp_path / "contact_data_0.npy", np.array(contacts, dtype=object))
    with pytest.raises(ValueError, match="index_profile"):
        rw.cheng_bins(tmp_path, ["Household"])


def test_fig6_refuses_an_unknown_layer(tmp_path):
    """Only Cheng's groups and their sum can be drawn."""
    with pytest.raises(ValueError, match="School"):
        rw.cheng_bins(tmp_path, ["School"])


def test_the_cheng2020_metadata_names_the_profiles(tmp_path):
    """A reader can tell E84 data from older data."""
    import json

    from covsyn.model.data_synthesis_main import write_run_metadata

    write_run_metadata(tmp_path, mode="cheng2020", **source_profiles_metadata("cheng2020"))
    record = json.loads((tmp_path / "run_metadata.json").read_text())
    assert record["source_profiles"] == "variable/cheng_period_profiles.csv"
    assert record["profile_stream"] == cr.PROFILE_STREAM
    assert source_profiles_metadata("spread_Taiwan_weight") == {}
