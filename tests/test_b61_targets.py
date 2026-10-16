# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B61: the offspring dispersion estimator, the one-day contact counts, and the recomputed targets.

The objective, its fast twin and the acceptance check now estimate the offspring dispersion k
by maximum likelihood, the estimator of Taiwan's k = 0.29 target, through one shared function.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, stats

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.target_margins import load_charged_bounds
from covsyn.model import contact_measures as cm


def _two_parameter_mle(x: np.ndarray) -> float:
    """Reference: both NegBin parameters free (plot_todolist923.nb_fit)."""

    def nll(p):
        r, m = np.exp(p)
        return -stats.nbinom.logpmf(x, r, r / (r + m)).sum()

    fit = optimize.minimize(nll, [0.0, np.log(x.mean())], method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 10000})
    return float(np.exp(fit.x[0]))


@pytest.mark.parametrize("k", [0.15, 0.3, 1.0])
def test_dispersion_matches_the_two_parameter_mle(k: float) -> None:
    """The profile MLE equals the full two-parameter MLE."""
    rng = np.random.default_rng(1)
    x = rng.negative_binomial(k, k / (k + 0.4), size=3000).astype(float)
    assert cm.offspring_dispersion(x) == pytest.approx(_two_parameter_mle(x), rel=1e-3)


def test_dispersion_recovers_a_known_k() -> None:
    """A large sample recovers the k it was drawn with."""
    rng = np.random.default_rng(2)
    x = rng.negative_binomial(0.29, 0.29 / (0.29 + 0.43), size=200_000)
    assert cm.offspring_dispersion(x) == pytest.approx(0.29, rel=0.05)


def test_dispersion_edge_cases_keep_the_old_conventions() -> None:
    """No infections gives NaN; no overdispersion gives the ceiling, as before."""
    assert np.isnan(cm.offspring_dispersion(np.zeros(50)))
    assert np.isnan(cm.offspring_dispersion([]))
    assert cm.offspring_dispersion([1.0] * 50) == cm.NO_OVERDISPERSION_K
    assert cm.offspring_dispersion([0, 1, 2] * 30) == cm.NO_OVERDISPERSION_K


def test_dispersion_does_not_depend_on_case_order() -> None:
    """The two objective implementations add cases in a different order (E65)."""
    rng = np.random.default_rng(3)
    x = rng.negative_binomial(0.3, 0.3 / 0.73, size=300).astype(float)
    assert cm.offspring_dispersion(x) == cm.offspring_dispersion(x[::-1])
    assert cm.offspring_dispersion(x) == cm.offspring_dispersion(rng.permutation(x))


def test_both_objective_implementations_agree_bit_for_bit(
    run10_vector: np.ndarray, demographic_parameters: object
) -> None:
    """measure_outcomes and fast_cost.reduce_outcomes give the same outcomes on simulated cases."""
    import copy

    from covsyn.model.data_synthesis_main import run_covid

    cases = []
    for seed in range(40):
        _, _, courses, contacts = run_covid(seed, run10_vector.copy(),
                                            copy.deepcopy(demographic_parameters),
                                            save_file=False, mode="result")
        cases.append((courses[0], contacts[0], None))
    slow = fo.measure_outcomes(cases)
    # Reversed: the fast implementation collects the cases in a different order (E65).
    scalars = [fast_cost._index_case_scalars(course, contact) for course, contact, _ in cases[::-1]]
    fast = fast_cost.reduce_outcomes(scalars)
    assert np.isfinite(slow["offspring_k"])
    for key in ("offspring_k", "daily_household", "daily_school", "daily_municipality",
                "community_day_zero_share", "community_day_p90"):  # B63 too
        assert fast[key] == slow[key], key


def test_daily_counts_average_to_the_per_day_mean() -> None:
    """daily_contacts_before_onset keeps the window of contacts_per_day_before_onset."""
    rng = np.random.default_rng(4)
    matrix = (rng.random((12, 9)) < 0.3).astype(float)
    contact = {"municipality_contacts_matrix": matrix}
    for onset in (np.nan, 0, 1, 4, 20):
        course = {"incubation_period": onset}
        days = cm.daily_contacts_before_onset(course, contact, "municipality")
        assert days.mean() == pytest.approx(
            cm.contacts_per_day_before_onset(course, contact, "municipality"))
        assert len(days) == (9 if np.isnan(onset) else min(9, max(onset, 1)))


def test_charged_targets_hold_the_b61_bands() -> None:
    """The charged intervals were rebuilt for the new household and school targets."""
    charged = load_charged_bounds(targets=fo.OUTCOME_TARGETS)
    for key in ("daily_household", "daily_school", "offspring_k"):
        lo, hi = fo.OUTCOME_TARGETS[key][:2]
        assert lo <= charged[key][0] < charged[key][1] <= hi
    assert fo.OUTCOME_TARGETS["daily_household"][:2] == (2.27, 3.28)
    assert fo.OUTCOME_TARGETS["daily_school"][:2] == (1.15, 1.72)
