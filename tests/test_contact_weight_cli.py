# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The spread_Taiwan_weight contact multiplier is a run argument (referee item B2).

The benchmark spans several transmission regimes, so the multiplier that was written into the
mode is passed in instead; the default of 1 is the calibrated Taiwan model. Every generated
dataset records how it was made.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from covsyn.model import data_synthesis_main as dsm


class _Stop(Exception):
    """Raised by the patched weight function to end the simulation early."""


@pytest.mark.parametrize("weight", [1.0, 7.0])
def test_the_weight_reaches_the_contact_multiplier(
    weight: float, run10_vector: np.ndarray, demographic_parameters: object, monkeypatch
) -> None:
    """run_covid hands its contact_weight to apply_contact_weight in spread_Taiwan_weight."""
    seen = []

    def record(_P, w):
        seen.append(w)
        raise _Stop

    monkeypatch.setattr(dsm, "apply_contact_weight", record)
    with pytest.raises(_Stop):
        dsm.run_covid(
            0,
            run10_vector.copy(),
            copy.deepcopy(demographic_parameters),
            mode="spread_Taiwan_weight",
            contact_weight=weight,
        )
    assert seen == [weight]


def test_the_default_weight_is_the_calibrated_model(
    run10_vector: np.ndarray, demographic_parameters: object, monkeypatch
) -> None:
    """Without an argument the multiplier is 1."""
    seen = []

    def record(_P, w):
        seen.append(w)
        raise _Stop

    monkeypatch.setattr(dsm, "apply_contact_weight", record)
    with pytest.raises(_Stop):
        dsm.run_covid(
            0,
            run10_vector.copy(),
            copy.deepcopy(demographic_parameters),
            mode="spread_Taiwan_weight",
        )
    assert seen == [1.0]


def test_run_metadata_records_the_settings(tmp_path) -> None:
    """The dataset directory holds the mode, weight, seeds and code version that made it."""
    dsm.write_run_metadata(
        tmp_path,
        mode="spread_Taiwan_weight",
        contact_weight=5.0,
        seed_start=0,
        monte_carlo_number=300,
        parameter_path="p",
    )
    record = json.loads((tmp_path / "run_metadata.json").read_text())
    assert record["contact_weight"] == 5.0 and record["monte_carlo_number"] == 300
    assert "covsyn_commit" in record and "written" in record
