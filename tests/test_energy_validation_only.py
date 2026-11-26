# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The energy term is reported but not charged (user decision 2026-11-23).

The term screen (scripts/probes/probe_term_sensitivity.py, CovSyn e737d4c) found no parameter
that only the energy term responds to, so it left the objective and stays as a validation
value in cost_energy. The total skips it rather than weighting it by 0, so a NaN or infinite
energy cannot reach the total (design review 2026-11-23, point 5).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.cost_parts import LAST as LAST_COST_PARTS

TAIWAN = np.load("variable/Taiwan_data_matrix.npy")
CHARGED = ("cost_contact", "cost_attack_rate", "cost_penalty", "cost_outcome")


def _candidate(monkeypatch, demographic_parameters, cheng_data, vector):
    for name in ("_WORKER_DEMOGRAPHIC_PARAMETERS", "_WORKER_TAIWAN_MATRIX_COLUMNS",
                 "_WORKER_STAGE_TIMING", "_WORKER_PROFILE", "_WORKER_PROCESS",
                 "_WORKER_CHENG", "_WORKER_SIMULATIONS", "_WORKER_TAIWAN_MATRIX"):  # fmt: skip
        monkeypatch.setattr(fast_cost, name, getattr(fast_cost, name, None), raising=False)
    fast_cost.init_worker(demographic_parameters, TAIWAN.shape[1], cheng=cheng_data,
                          simulations=10, taiwan_matrix=TAIWAN)  # fmt: skip
    cost, parts, _ = fast_cost.evaluate_candidate(vector, 0)
    return cost, parts


def test_the_total_is_the_charged_terms_without_energy(
    monkeypatch, run10_vector, demographic_parameters, cheng_data
):
    """Contacts, attack rates, physiology and outcomes; energy is still reported."""
    cost, parts = _candidate(monkeypatch, demographic_parameters, cheng_data, run10_vector)
    assert cost == pytest.approx(sum(parts[k] for k in CHARGED), rel=1e-12)
    assert math.isfinite(parts["cost_energy"]) and parts["cost_energy"] != 0


@pytest.mark.parametrize("energy", [float("nan"), float("inf")])
def test_an_unusable_energy_does_not_reach_the_total(
    monkeypatch, run10_vector, demographic_parameters, cheng_data, energy
):
    """A NaN or infinite energy is reported as it is and the total stays finite."""
    monkeypatch.setattr(fo, "energy_statistic", lambda *args, **kwargs: energy)
    cost, parts = _candidate(monkeypatch, demographic_parameters, cheng_data, run10_vector)
    assert math.isfinite(cost)
    reported = parts["cost_energy"]
    assert reported == energy or (math.isnan(energy) and math.isnan(reported))


def test_the_reference_objective_charges_the_same_terms(
    monkeypatch, run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """firefly_optimizer.cost_function, the bit-identity reference, also leaves energy out."""
    monkeypatch.setattr(fo, "SIMULATIONS_PER_EVALUATION", 10)
    cost = fo.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    parts = dict(LAST_COST_PARTS)
    assert cost == pytest.approx(sum(parts[k] for k in CHARGED), rel=1e-12)
    assert math.isfinite(parts["cost_energy"])


def test_energy_provenance_matches_both_objective_totals(
    monkeypatch, run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """Both real objective paths must agree with the energy provenance marker."""
    fast_total, fast_parts = _candidate(
        monkeypatch, demographic_parameters, cheng_data, run10_vector
    )
    monkeypatch.setattr(fo, "SIMULATIONS_PER_EVALUATION", 10)
    reference_total = fo.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    reference_parts = dict(LAST_COST_PARTS)

    assert type(fast_cost.COST_INCLUDES_ENERGY) is bool
    charged = (*CHARGED, "cost_energy") if fast_cost.COST_INCLUDES_ENERGY else CHARGED
    for total, parts in ((fast_total, fast_parts), (reference_total, reference_parts)):
        assert math.isfinite(parts["cost_energy"]) and parts["cost_energy"] != 0
        assert total == pytest.approx(sum(parts[key] for key in charged), rel=1e-12)


def _raise(*args, **kwargs):
    raise ValueError("energy failed")


def test_an_energy_error_is_reported_as_nan_not_a_failed_evaluation(
    monkeypatch, run10_vector, demographic_parameters, cheng_data
):
    """Energy is not charged, so its failure must not charge FAILED_EVALUATION_COST."""
    monkeypatch.setattr(fo, "energy_statistic", _raise)
    cost, parts = _candidate(monkeypatch, demographic_parameters, cheng_data, run10_vector)
    assert cost == pytest.approx(sum(parts[k] for k in CHARGED), rel=1e-12)
    assert math.isnan(parts["cost_energy"])


def test_the_reference_objective_survives_an_energy_error(
    monkeypatch, run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """The same in firefly_optimizer.cost_function."""
    monkeypatch.setattr(fo, "SIMULATIONS_PER_EVALUATION", 10)
    monkeypatch.setattr(fo, "energy_statistic", _raise)
    cost = fo.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    parts = dict(LAST_COST_PARTS)
    assert cost == pytest.approx(sum(parts[k] for k in CHARGED), rel=1e-12)
    assert math.isnan(parts["cost_energy"])


def test_a_failure_building_the_energy_matrix_is_reported_as_nan(
    monkeypatch, run10_vector, demographic_parameters, cheng_data
):
    """The test-matrix rows are only for energy, so their failure must not fail the evaluation."""
    monkeypatch.setattr(fast_cost, "convert_synthetic_data_to_test_matrix", _raise)
    cost, parts = _candidate(monkeypatch, demographic_parameters, cheng_data, run10_vector)
    assert cost == pytest.approx(sum(parts[k] for k in CHARGED), rel=1e-12)
    assert math.isnan(parts["cost_energy"])


def test_the_reference_survives_a_failure_building_the_energy_matrix(
    monkeypatch, run10_vector, demographic_parameters, cheng_data, cost_pool
):
    """The same in firefly_optimizer.cost_function."""
    monkeypatch.setattr(fo, "SIMULATIONS_PER_EVALUATION", 10)
    monkeypatch.setattr(fo, "convert_synthetic_data_to_test_matrix", _raise)
    cost = fo.cost_function(run10_vector, demographic_parameters, cost_pool, *cheng_data)
    parts = dict(LAST_COST_PARTS)
    assert cost == pytest.approx(sum(parts[k] for k in CHARGED), rel=1e-12)
    assert math.isnan(parts["cost_energy"])


def test_a_failing_test_matrix_reports_once_not_per_case(
    monkeypatch, capsys, run10_vector, demographic_parameters, cheng_data
):
    """A converter failing for every case prints one traceback, not one per case."""
    monkeypatch.setattr(fast_cost, "convert_synthetic_data_to_test_matrix", _raise)
    _candidate(monkeypatch, demographic_parameters, cheng_data, run10_vector)
    assert capsys.readouterr().err.count("Traceback") == 1
