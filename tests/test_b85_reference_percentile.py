# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B85: a first-wave scenario is judged by where the observation falls, not by the mean.

Taiwan's first wave happened once, and a scenario's simulated counts are right-skewed, so
dividing the one observed count by the simulated mean understates the agreement. B71 already
settled this for the offspring maximum -- "the largest count grows with the number of cases and
rests on one draw... The check passes when 8 is not in either 2.5% tail of the simulated
largest" -- and the same rule belongs here. Judged by the mean, run 21's feb29 reached 12.7 of
28 and looked like a 45% shortfall needing 50% more seeds; judged by the distribution it sat at
the 96.7th percentile, a real failure, which B82 moved to 89.5% and inside the band.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.validation.compare_first_wave import reference_percentile, reference_verdict


def test_the_percentile_is_the_share_of_runs_below_the_observation():
    """The plain reading: half the runs below means the 50th percentile."""
    runs = np.arange(100, dtype=float)      # 0..99
    assert reference_percentile(runs, 50) == pytest.approx(50.5)
    assert reference_percentile(runs, 0) == pytest.approx(0.5)
    assert reference_percentile(runs, 99) == pytest.approx(99.5)


def test_a_tie_counts_as_half_a_run():
    """Discrete counts tie often, so a tie must not push the observation into a tail."""
    runs = np.array([5.0] * 100)
    assert reference_percentile(runs, 5) == pytest.approx(50.0)
    assert reference_percentile(runs, 4) == pytest.approx(0.0)
    assert reference_percentile(runs, 6) == pytest.approx(100.0)


def test_an_observation_inside_the_band_passes():
    """Anywhere between the two tails is agreement, not a near miss."""
    runs = np.arange(1000, dtype=float)
    for observed in (30, 500, 960):
        assert reference_verdict(runs, observed) == "ok"


@pytest.mark.parametrize("observed", [5, 10, 980, 995])
def test_an_observation_in_either_tail_fails(observed):
    """Both tails count: the model may not over-produce either."""
    runs = np.arange(1000, dtype=float)
    assert reference_verdict(runs, observed) == "fail"


def test_the_band_edges_are_the_2_5_percent_tails_of_b71():
    """The threshold is B71's, so the two checks of one draw agree."""
    runs = np.arange(1000, dtype=float)
    # 2.5% and 97.5% of 0..999 are 24.5 and 974.5; just inside them passes, outside fails
    assert reference_verdict(runs, 25) == "ok"
    assert reference_verdict(runs, 974) == "ok"
    assert reference_verdict(runs, 24) == "fail"
    assert reference_verdict(runs, 975) == "fail"


def test_a_mean_ratio_and_the_percentile_can_disagree():
    """The defect B85 fixes, on a right-skewed sample like a scenario's own counts.

    The observation sits well above the mean, so the mean ratio reads as a large shortfall,
    while the distribution holds the observation comfortably: both statements are arithmetically
    right and only the second answers whether the model reproduces what happened once.
    """
    rng = np.random.default_rng(0)
    runs = rng.negative_binomial(2, 2 / (2 + 18.0), size=20000).astype(float)
    assert runs.mean() > np.median(runs)                 # right-skewed, as the scenarios are
    observed = 28
    assert runs.mean() / observed < 0.70                 # the mean ratio looks like a failure
    assert reference_verdict(runs, observed) == "ok"     # the distribution says otherwise


def test_an_observation_the_distribution_does_reject():
    """The test still bites: far enough out, the verdict fails whatever the mean ratio says."""
    rng = np.random.default_rng(1)
    runs = rng.negative_binomial(2, 2 / (2 + 18.0), size=20000).astype(float)
    assert reference_verdict(runs, int(np.percentile(runs, 99.5))) == "fail"


def test_no_runs_gives_no_verdict():
    """An empty directory must give no verdict rather than a spurious one."""
    assert reference_percentile(np.array([]), 10) is None
    assert reference_verdict(np.array([]), 10) is None


def test_a_largest_tied_at_the_observation_is_not_pushed_into_the_upper_tail():
    """B71's largest of 579: P(<= 7) = 0.90, P(= 8) = 0.08, P(>= 9) = 0.02 (code review F4).

    8 is in neither 2.5% tail (P(max >= 8) = 10%), but counting every tie as below gave 98,
    which failed (2.5, 97.5). With a tie as half a sample the check passes.
    """
    largest = np.array([7] * 900 + [8] * 80 + [9] * 20)
    assert 100 * np.mean(largest <= 8) > 97.5
    assert reference_percentile(largest, 8) == pytest.approx(94.0)
    assert reference_verdict(largest, 8) == "ok"
