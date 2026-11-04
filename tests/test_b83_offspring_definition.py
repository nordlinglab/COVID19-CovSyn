# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B83: the share infecting three or more is measured on the population R is measured on.

B56 took the band for the share of cases infecting three or more from the whole first-wave
tracing set, between the recorded infector links (0.17%) and the confirmed contacts (4.32%).
B67 then moved R's band onto Cheng 2020's local index cases (0.62 each, band [0.37, 0.98])
because CovSyn's index case is a local case living in the community, but left the share on the
old population, 84% of whose 579 cases were imported. The two cannot both hold: a negative
binomial with R = 0.62 and k = 0.286 puts 7.8% of cases at three or more, and the share stays
above 4.3% for every R above 0.40. Restricted to the 55 local cases, the same two definitions
give 1.82% and 12.73%, which is the band that matches R's population.
"""

from __future__ import annotations

from math import exp, lgamma

import numpy as np
import pytest

from covsyn.calibration.firefly_optimizer import OUTCOME_TARGETS
from covsyn.validation.acceptance import OFFSPRING_SHARE_GE3


def nb_share_ge3(mean: float, k: float) -> float:
    """P(X >= 3) of a negative binomial with this mean and dispersion, as a percentage."""
    def pmf(x):
        return (exp(lgamma(x + k) - lgamma(k) - lgamma(x + 1))
                * (k / (k + mean)) ** k * (mean / (k + mean)) ** x)
    return 100 * (1.0 - sum(pmf(x) for x in (0, 1, 2)))


def test_the_band_is_the_local_cases_two_definitions():
    """1.82% = 1 of the 55 local cases by recorded links, 12.73% = 7 of 55 by confirmed
    contacts (recompute_taiwan_targets, taiwan_case_timeline case_type == 'Local')."""
    assert OFFSPRING_SHARE_GE3 == (1.8, 12.8)


def test_the_old_band_was_unreachable_at_the_accepted_r():
    """The defect B83 fixes: at B67's R the old 4.3% ceiling fails for every accepted k, so no
    parameter vector could pass the check."""
    for k in (OUTCOME_TARGETS['offspring_k'][0], 0.286, OUTCOME_TARGETS['offspring_k'][1]):
        assert nb_share_ge3(0.62, k) > 4.3        # the point estimate of B67's band
        assert nb_share_ge3(0.98, k) > 4.3        # and its upper end
        # the old ceiling survived only at the very bottom of R's band: the largest R that
        # holds it is 0.372 at k = 0.20, 0.401 at k = 0.286 and 0.420 at k = 0.35
        assert nb_share_ge3(0.43, k) > 4.3


@pytest.mark.parametrize("k", [0.20, 0.267, 0.286, 0.35])
def test_the_new_band_holds_at_the_r_of_its_own_source(k):
    """Cheng 2020's local index cases infected 0.62 each, the estimate B67's R band is built on;
    the share implied there must sit inside the band, which is what the old one failed."""
    assert OFFSPRING_SHARE_GE3[0] <= nb_share_ge3(0.62, k) <= OFFSPRING_SHARE_GE3[1]


@pytest.mark.parametrize("k", [0.20, 0.267, 0.286, 0.35])
def test_the_new_band_holds_over_most_of_the_r_interval(k):
    """It holds from the lower end of B67's interval up to R = 0.95."""
    for r in np.linspace(0.37, 0.95, 25):
        assert OFFSPRING_SHARE_GE3[0] <= nb_share_ge3(float(r), k) <= OFFSPRING_SHARE_GE3[1]


@pytest.mark.parametrize("k", [0.286, 0.35])
def test_the_residual_mismatch_at_the_very_top_of_the_r_interval(k):
    """Documented limit, not a defect to paper over: R's band is an exact Poisson interval on
    18 of 29 cases, while this band is the range between two case definitions, so the two are
    not the same kind of quantity and cannot agree everywhere. At R = 0.98, the very top of
    that interval, the implied share passes 12.8%. The gap is small, no parameter vector has
    come near that corner (run 21 measured R = 0.602), and closing it would mean replacing an
    empirical endpoint, 7 of the 55 local cases, with an extrapolation."""
    assert 12.8 < nb_share_ge3(0.98, k) < 13.5


def test_the_dispersion_target_stays_on_the_579_case_estimate():
    """The 55 local cases give k = 5.69 by confirmed contacts, so a local band would not
    constrain the dispersion at all; k keeps the first-wave MLE (B61)."""
    assert OUTCOME_TARGETS['offspring_k'][:2] == (0.20, 0.35)
