import argparse
import concurrent.futures
import contextlib
import copy
import csv
import io
import json
import multiprocessing
import numpy as np
import pickle
import sys
import time
import traceback
import warnings
import os
from pathlib import Path
from numpy.random import default_rng
from scipy.stats import genextreme
from tqdm import tqdm

from covsyn.model.data_synthesis_main import run_covid
from covsyn.calibration import (initial_population, parameter_space, revalidation, seed_streams,
                                step_schedule)
from covsyn.runtime import hardware, profiling, runs, timings
from covsyn.calibration.sar_anchors import (CHENG2020_INDEX_CASES, LAYER_CUMULATIVE_SAR,
                                            LAYER_INFECTIONS_PER_INDEX,
                                            health_care_contact_floor)
from covsyn.model.contact_measures import (at_work_days_before_onset, contacts_per_day_at_work,
                                           contacts_per_day_before_onset, is_employed,
                                           largest_spreader_percentile, offspring_dispersion,
                                           one_day_counts, one_day_shape)
from covsyn.calibration.target_margins import load_charged_bounds
from covsyn.model.data_synthesize import *
from covsyn.model.data_synthesize import COMMUNITY_EVENT_FIELDS, COMMUNITY_EVENT_FIRST_INDEX
from covsyn.figures.plot_results import *
from covsyn.data_processing.rw_data_processing import convert_synthetic_data_to_test_matrix
# from sklearn.metrics.pairwise import nan_euclidean_distances
# Note: RW_nan_euclidean_distances is a copy code from sklearn but only comment out the weighting parts.
# "distances /= present_count" and "distances *= X.shape[1]"
from sklearn.metrics.pairwise import *
from sklearn.utils.validation import _deprecate_positional_args
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# Multivariate test

# This code is modified based on sklearn.metrics.pairwise nan_euclidean_distances function


@_deprecate_positional_args
def rw_nan_euclidean_distances(X, Y=None, *, squared=False,
                               missing_values=np.nan, copy=True):
    """Euclidean distance over the coordinates present in both samples, NOT rescaled.

    Compute the distance between each pair of samples in X and Y, where Y=X is assumed if
    Y=None, ignoring any coordinate missing in either sample:

        dist(x, y) = sqrt(sq. distance over the coordinates present in both)

    For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]`` is
    ``sqrt((3-1)^2 + (6-5)^2)``.

    This is where it departs from sklearn's ``nan_euclidean_distances``, which this function
    was copied from: sklearn multiplies the squared distance by
    ``total coordinates / present coordinates``, so that pairs observed on different numbers of
    coordinates are comparable. This function never did (the two lines were commented out
    from the first commit until B91 rewrote the computation), so the behaviour is inherited
    rather than decided here.

    B90 (2026-11-06) measured what restoring them would do, because it is not obvious: the
    objective's data matrix is only 36% present, its rows carry 3 to 6 of 9 coordinates, and
    pairs share 1 to 5, while the synthetic side is about 94% present, so the two sides are not
    alike and the asymmetry does not cancel. Restoring the rescaling moved all five of run 23's
    shortlisted candidates by the same +0.105 and changed no ranking, because the energy term is
    -0.249 against a total of 9.3 with an sd of 0.046 against 4.2. It was therefore left alone,
    and this docstring was corrected to describe what the code does instead.

    If all the coordinates are missing, or there are no common present coordinates, NaN is
    returned for that pair.

    Read more in the :ref:`User Guide <metrics>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    X : array-like of shape=(n_samples_X, n_features)

    Y : array-like of shape=(n_samples_Y, n_features), default=None

    squared : bool, default=False
        Return squared Euclidean distances.

    missing_values : np.nan or int, default=np.nan
        Representation of missing value.

    copy : bool, default=True
        Ignored since B91: the inputs are never written, so they need no protective copy
        (check_pairwise_arrays may still convert them, e.g. lists or integers); the parameter
        is kept so existing calls stay valid.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)

    See Also
    --------
    paired_distances : Distances between pairs of elements of X and Y.

    Examples
    --------
    Without sklearn's rescaling by the share of present coordinates, a pair that shares one
    coordinate is 1.0 apart where sklearn's nan_euclidean_distances gives 1.41421356
    (code review 2026-11-09, F10):

    >>> nan = float("NaN")
    >>> X = [[0, 1], [1, nan]]
    >>> rw_nan_euclidean_distances(X, X) # distance between rows of X
    array([[0., 1.],
           [1., 0.]])

    >>> # get distance to origin
    >>> rw_nan_euclidean_distances(X, [[0, 0]])
    array([[1.],
           [1.]])

    References
    ----------
    * John K. Dixon, "Pattern Recognition with Partly Missing Data",
      IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
      10, pp. 617 - 621, Oct. 1979.
      http://ieeexplore.ieee.org/abstract/document/4310090/
    """

    # B87: 'ensure_all_finite' replaced 'force_all_finite', which scikit-learn 1.6 deprecated
    # and 1.9 removed. The project declares scikit-learn>=1.9.1, so the old name raised
    # TypeError on the declared environment and every call through estat failed; 1.6 accepts
    # both, so the compute server kept working and hid it.
    ensure_all_finite = 'allow-nan' if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 ensure_all_finite=ensure_all_finite, copy=False)
    # copy is ignored since B91: nothing below writes to X or Y, so copying them on each of
    # energy()'s three calls only allocated (code review 2026-11-12).
    if not is_scalar_nan(missing_values):
        X = np.where(X == missing_values, np.nan, X)
        Y = np.where(Y == missing_values, np.nan, Y)
    # B91 (2026-11-12): subtract coordinate by coordinate instead of expanding
    # |x|^2 - 2x.y + |y|^2 through the platform's BLAS. The expansion left identical cases
    # 3-8e-8 apart on OpenBLAS but 0 on Accelerate, and energy() takes log(d + 1e-16), so the
    # objective differed between Linux and macOS (E103). Exact differences make equal cases
    # exactly 0 on every platform.
    # The work arrays are allocated once and reused across coordinates (code review
    # 2026-11-12). Each difference is squared in the inputs' own precision and summed in
    # float64, and a missing difference is set to 0 before squaring, which adds exactly what
    # np.where(present, d*d, 0) added: bit for bit the first B91 version, float32 or float64.
    shape = (X.shape[0], Y.shape[0])
    squared_sum = np.zeros(shape)
    none_shared = np.ones(shape, dtype=bool)
    difference = np.empty(shape, dtype=X.dtype)  # check_pairwise_arrays gave X and Y one dtype
    missing = np.empty(shape, dtype=bool)
    for k in range(X.shape[1]):
        np.subtract(X[:, k, None], Y[None, :, k], out=difference)
        np.isnan(difference, out=missing)
        np.copyto(difference, 0.0, where=missing)
        np.multiply(difference, difference, out=difference)
        squared_sum += difference
        np.logical_and(none_shared, missing, out=none_shared)
    distances = squared_sum if squared else np.sqrt(squared_sum)
    distances[none_shared] = np.nan
    return distances


def _standardised_stack(x, y):
    stack = np.vstack([x, y])
    return (stack - np.nanmean(stack, axis=0))/np.nanstd(stack, axis=0)


def energy_statistic(x, y, method='log'):
    """The statistic of estat, without its bootstrap (B91).

    The objective discarded the one bootstrap draw it asked estat for, so it computed the
    energy twice and drew from the global random state; this computes it once.
    """
    return energy(*np.split(_standardised_stack(x, y), [len(x)]), method)


def estat(x, y, nboot=1000, replace=False, method='log', fitting=False):
    '''
    Energy distance statistics test.
    Reference
    ---------
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Brian Lau, multdist, https://github.com/brian-lau/multdist

    '''
    n, N = len(x), len(x) + len(y)
    stack = _standardised_stack(x, y)
    if replace:
        def rand(x): return np.random.randint(x, size=x)
    else:
        rand = np.random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting == True:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x, y, method='log'):
    dx = rw_nan_euclidean_distances(x, x)
    dx = dx[np.triu_indices(dx.shape[0], k=1)]

    dy = rw_nan_euclidean_distances(y, y)
    dy = dy[np.triu_indices(dy.shape[0], k=1)]

    dxy = rw_nan_euclidean_distances(x, y)

    e = 10**-16
    # dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx+e), np.log(dy+e), np.log(dxy+e)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = np.nansum(dxy)/(n * m) - np.nansum(dx)/n**2 - np.nansum(dy)/m**2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z

# def RW_nan_euclidean_distances(x, y):

#     size = np.max(np.shape(x))
#     distances = np.zeros([size, size])
#     for i, x_row in enumerate(x):
#         for j, y_row in enumerate(y):
#             distances[i, j] = np.sqrt(np.nansum((x_row - y_row)**2))

#     return distances


# How many simulations one pool task carries. One simulation is only ~0.8 ms of compute,
# so submitting 300 of them as 300 separate tasks spends almost all of its time in
# multiprocessing overhead: measured on the remote M3 Ultra with 32 workers, the pool round
# trip was 0.100 s of a 0.121 s evaluation, i.e. a parallel efficiency of 6%. Batching the
# same 300 simulations into tasks of this size cut that to 0.020 s (4.9x); 5 per task gave
# 0.024 s and 20 per task 0.031 s, so the optimum is flat around 10. Changing it cannot
# change any result: every simulation still runs run_covid with its own seed and the results
# are flattened back in seed order.
SIMULATIONS_PER_TASK = 10


class _CompletedSimulation:
    """Gives an already-computed simulation result the .result() interface.

    generate_contact_result(), convert_synthetic_data_to_test_matrix() and the index-case
    loop in _cost_function() all expect the Future objects the pool used to hand back one
    per simulation. Batching means the pool now returns a LIST of results per task, so each
    one is wrapped here and every consumer keeps working unchanged.
    """

    __slots__ = ('_value',)

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def _run_simulation_batch(seeds, P, demographic_parameters):
    """Run several simulations inside one pool task. See SIMULATIONS_PER_TASK."""
    return [run_covid(seed, P, demographic_parameters, save_file=False) for seed in seeds]


def extract_course_and_contact(results, case_number):
    """Pull the course-of-disease and contact records out of the simulation results once.

    This used to sit inside generate_contact_result(), which _cost_function() calls once per
    Cheng layer, so the same 300 results were walked three times, every record was fetched
    twice per walk (result.result()[2] and result.result()[3]), and np.append on a growing
    object array made each walk quadratic. Doing it once per evaluation instead of three
    times is the second half of the speed-up recorded at SIMULATIONS_PER_TASK; the arrays
    produced are identical to what the np.append loop produced.
    """
    course_of_disease_data_list = []
    contact_data_list = []
    for result in results:
        value = result.result()
        course_of_disease_data_list.extend(value[2])
        contact_data_list.extend(value[3])
    return (np.array(course_of_disease_data_list[0:case_number], dtype=object),
            np.array(contact_data_list[0:case_number], dtype=object))


def generate_contact_result(course_of_disease_data_list, contact_data_list, layer):
    # Transform the data to contact bars as Cheng2020
    if (layer == 'Household') | (layer == 'Health care'):
        _, contact_array, _, infection_array = create_array_cheng2020_fig2(
            course_of_disease_data_list, contact_data_list, layer=layer)
    elif layer == 'Cheng others':
        _, school_contact_array, _, school_infection_array = create_array_cheng2020_fig2(
            course_of_disease_data_list, contact_data_list, layer='School')
        _, workplace_contact_array, _, workplace_infection_array = create_array_cheng2020_fig2(
            course_of_disease_data_list, contact_data_list, layer='Workplace')
        _, municipality_contact_array, _, municipality_infection_array = create_array_cheng2020_fig2(
            course_of_disease_data_list, contact_data_list, layer='Municipality',
            expected_events=True)   # B55
        contact_array = np.sum(np.vstack(
            (school_contact_array, workplace_contact_array, municipality_contact_array)), axis=0)
        infection_array = np.sum(np.vstack(
            (school_infection_array, workplace_infection_array, municipality_infection_array)), axis=0)
        # contact_array = school_contact_array + \
        #     workplace_contact_array + municipality_contact_array
        # infection_array = school_infection_array + \
        #     workplace_infection_array + municipality_infection_array

    return (contact_array, infection_array)


# def cost_function(P, demographic_parameters):
#     source_case_number = 100
#     repeat_number = 3
#     seeds = range(source_case_number*repeat_number)
#     # print('Multiprocess start')
#     with concurrent.futures.ProcessPoolExecutor() as executor:  # Multiprocessing
#         results = [executor.submit(run_covid, seeds[i], P, demographic_parameters, save_file=False)
#                    for i in seeds]
#     # print('Multiprocess end')
#     # print('Time: ', time.asctime(time.localtime(time.time())))

#     # Household
#     Cheng_contact_array = np.array([100, 39, 6, 4, 2, 0])
#     Cheng_household_attack_rate = np.array([4, 5.1, 16.7, 0, 0, 0])
#     household_attack_rate_weight = np.array(
#         [12.19512195, 6.49350649, 1.87265918, 2.04081633, 1.52207002, 1])
#     contact_array, infection_array = generate_contact_result(
#         results, layer='Household', case_number=source_case_number*repeat_number)
#     household_attack_rate = infection_array/contact_array
#     household_cost = np.sum(
#         (contact_array/repeat_number-Cheng_contact_array)**2)
#     household_attack_rate_cost = np.nansum(
#         ((household_attack_rate-Cheng_household_attack_rate)*household_attack_rate_weight)**2)

#     # Health care
#     Cheng_contact_array = np.array([236, 150, 38, 17, 110, 146])
#     Cheng_health_care_attack_rate = np.array([0.8, 2, 2.6, 0, 0, 0])
#     health_care_attack_rate_weight = np.array(
#         [35.71428571, 20, 7.69230769, 5.43478261, 30.3030303, 38.46153846])
#     contact_array, infection_array = generate_contact_result(
#         results, layer='Health care', case_number=source_case_number*repeat_number)
#     health_care_attack_rate = infection_array/contact_array
#     health_care_cost = np.sum(
#         (contact_array/repeat_number-Cheng_contact_array)**2)
#     health_care_attack_rate_cost = np.nansum(
#         ((health_care_attack_rate-Cheng_health_care_attack_rate)*health_care_attack_rate_weight)**2)

#     # Cheng others
#     # non_household_contact+other_contact
#     Cheng_contact_array = np.array([399, 678, 172,  98, 337, 138])
#     Cheng_others_attack_rate = np.array([0, 0, 0.6, 0, 0, 0])
#     others_attack_rate_weight = np.array(
#         [100, 166.66666667, 31.25, 23.80952381, 90.90909091, 30.3030303])
#     contact_array, infection_array = generate_contact_result(
#         results, layer='Cheng others', case_number=source_case_number*repeat_number)
#     others_attack_rate = infection_array/contact_array
#     others_cost = np.sum((contact_array/repeat_number-Cheng_contact_array)**2)
#     others_attack_rate_cost = np.nansum(
#         ((others_attack_rate-Cheng_others_attack_rate)*others_attack_rate_weight)**2)

#     # Energy distance
#     # Load Taiwan data matrix
#     taiwan_data_matrix = np.load('./variable/Taiwan_data_matrix.npy')

#     synthetic_data_matrix = convert_synthetic_data_to_test_matrix(
#         results, taiwan_data_matrix, source_case_number*repeat_number)
#     # Energy statistics test
#     _, energy_cost, _ = estat(
#         taiwan_data_matrix, synthetic_data_matrix, nboot=1)

#     # Objective function
#     energy_weight = 1000
#     print('household_cost: ', household_cost)
#     print('household_attack_rate_cost: ', household_attack_rate_cost)
#     print('health_care_cost: ', health_care_cost)
#     print('health_care_attack_rate_cost: ', health_care_attack_rate_cost)
#     print('others_cost: ', others_cost)
#     print('others_attack_rate_cost: ', others_attack_rate_cost)
#     print('weighted energy cost: ', energy_weight*energy_cost)
#     print()
#     cost = household_cost + household_attack_rate_cost + health_care_cost + \
#         health_care_attack_rate_cost + others_cost + \
#         others_attack_rate_cost + energy_weight*energy_cost
#     # print('cost: ', cost)
#     return (cost)

PHYSIOLOGY_PENALTY_WEIGHT = 2.0

# Weight applied to the secondary-attack-rate terms of the objective. Decomposing the
# previous best solution showed the contact-count terms carried 96.6% of the layer cost and
# the attack-rate terms only 3.4%, because every attack rate is normalised by the global
# maximum (16.7%, the household 4-5 day bin) which crushes the small-rate layers. Making
# the two families contribute equally at that point needs ~28; 10 is used as a deliberately
# conservative first step, since the balance moves as the fit changes.
ATTACK_RATE_WEIGHT = 10.0

# Filled in by cost_function() on every evaluation so the firefly loop can log which part
# of the objective is driving the fit (see progress.csv).
#
# E71: this dict lives in its own module, not here. This file is run as a script, so while it
# executes it is `__main__`; when fast_cost imports `firefly_optimizer` Python loads a SECOND
# copy under the real name, with its own globals. Every constant is identical in both copies, so
# the fit is unaffected -- but a dict is not, and run 7 filled the imported copy's dict while the
# running __main__ read its own empty one, losing every cost_* and measured_* column from
# progress_metrics.csv (34 columns instead of 73). cost_parts is only ever imported under its own
# name, so both copies bind the same object. Mutate it in place; never rebind it.
from covsyn.calibration.cost_parts import LAST as LAST_COST_PARTS


def physiology_penalty(P, weight=PHYSIOLOGY_PENALTY_WEIGHT):
    """Soft constraint on the disease-course PARAMETERS. 0 when every quantity is inside its
    literature range; grows (relative, squared) the further outside.

    Phase D (B22, B26, B33) moved most of this penalty to measured quantities, because a
    Gamma mean computed from shape x scale is not what the simulation produces once the
    draws are truncated (finding E23). What is left here are the three quantities that are
    drawn without truncation, plus the two new ones introduced in Phase D:
      * P[41] / P[42] no longer describe the incubation period but the PRE-ONSET WINDOW
        (incubation = latent + window, B22 (3)); about 2 days of pre-symptomatic
        infectiousness (Cheng 2020 found transmission concentrated in this window).
      * P[43:46] is onset -> confirmation. It is NOT pinned to Taiwan's post-March median of
        1 day (B26, finding E27): the contact data this objective is fitted to is Cheng
        2020's cohort of 15 Jan - 18 Mar 2020, when the delay was about 5 days, and 41% of
        Cheng's non-household contacts start more than 3 days after onset -- contacts that
        cannot exist at all if every case is isolated one day after onset. The range is
        therefore left wide (1-6 days) and the contact bins decide; the fitted value is
        reported against both of Taiwan's observed medians (finding E33).
    The recovery times, the asymptomatic share and the severity cascade are now measured
    from the simulation in outcome_penalty().
    """
    def ou(x, lo, hi):
        if x <= 0:
            return 1.0
        return max(0.0, x / hi - 1.0) + max(0.0, lo / x - 1.0)
    pen = 0.0
    latent_mean      = P[37] * P[38]
    infectious_mean  = P[39] * P[40]
    pre_onset_window = P[41] * P[42]
    onset_to_confirm = P[43] * P[44] + P[45]
    # B3 (2026-09-25): the latent target is widened from [4.1, 4.5] to the whole literature
    # reported-mean range. The narrow version was B1's way of forcing the latent period into
    # the range at the expense of the generation time, but E45 showed the range itself is
    # built from mixed statistics -- 4.1 is Cheng 2020's MEDIAN and 5.5 is Xin 2022's MEAN --
    # so aiming at its lower edge is aiming at a number that is not a mean at all. Inside the
    # range the term costs nothing, which is also what the professor asked for (todolist 1.11).
    pen += ou(latent_mean,      4.1, 5.5)  ** 2
    # B5 put every term on the literature reported-mean range; the infectious period kept
    # the older 5-10 days, so run 4 and run 11 paid for a mean of 4.2 days that lies inside
    # the reported range of 3.45-20 (E25). B55 applies B5 here as well.
    pen += ou(infectious_mean,  3.45, 20.0) ** 2
    pen += ou(pre_onset_window, 1.0, 3.0)  ** 2
    # Widened from [1, 6] once B2 gave this quantity a measured target of 5-7 days: a
    # Gamma with median 7 has a mean near 7.8, so the old ceiling made the new target
    # unreachable in the same way E34 made the latent target unreachable. The ceiling is
    # now the published model's own upper bound of 12.02 days.
    pen += ou(onset_to_confirm, 1.0, 12.0) ** 2
    # The age risk ratios P[63:67] are not penalised here: they are locked by lb == ub in
    # parameters_for_initialization.py and set by hand, so a penalty term would only add a
    # constant that differs between trial values and would make trial runs incomparable.
    return weight * pen


# ---------------------------------------------------------------------------------------
# Phase D measured-outcome targets (covsyn_decisions.md B17, B19, B25, B26, B28, B33).
# Each entry is an acceptance INTERVAL, not a point: inside it the term costs nothing. They
# are measured on the index case of every simulation, which is the only case whose whole
# course and whole contact record is inside the evaluation window.
# ---------------------------------------------------------------------------------------
# E66, 2026-09-27: raised 2.0 -> 6.0, calibrated on run 6's own measured values rather than
# chosen. Run 6's objective decomposed as cost_contact 1.6956, cost_attack_rate 0.3092,
# cost_outcome 0.0419: every decision B17-B42 encoded as a measured target was competing for
# 2.0% of the objective, which is why six substantive fixes moved the checklist so little.
# Capping the miss scale (MAX_INTERVAL_WIDTH_OVER_CENTRE) multiplies the raw penalty by 3.9 on
# those same values, and this weight takes the block to about 20% of contact + attack + outcome
# -- enough to move the search, not enough to abandon the Cheng contact fit that is the
# published model's objective.
#
# WATCH THIS ON RUN 7: the risk of a heavier outcome block is that the optimizer pays for it out
# of the Cheng fit. cost_contact was 1.6956 on run 6; if it climbs a long way while the outcome
# targets improve, this weight is too high, not the targets wrong. The other failure mode is
# E23/E34 -- a target that cannot be reached inside the bounds gets traded away and drags the
# decision backwards -- so every per-index target was checked reachable before raising this:
# medical needs 0.06 per index over 2.244 candidate contacts = 2.7% per contact against a
# ceiling of 3.9% cumulative, household 0.10 over 2.601 = 3.8% against 12.0%, community 0.01
# over 5.773 = 0.17% against a range of 0.08-1.93%.
OUTCOME_PENALTY_WEIGHT = 6.0

# Each entry is (low, high, weight). Inside the interval the term costs nothing. The
# intervals are widened to about two standard errors of ONE evaluation (100 index cases),
# because a target narrower than the sampling noise would charge the optimizer for
# randomness: with 100 cases the asymptomatic share alone has a standard error of 4 points.
# Weight 0 means the quantity is measured and written to progress.csv but not
# charged -- used where 100 cases cannot say anything (about 10 ICU cases and 1 death per
# evaluation), in which case the calibration is carried by the parameter itself.
OUTCOME_TARGETS = {
    # B17: negative-binomial dispersion of the offspring count. Taiwan's own tracing gives
    # k = 0.29; cluster-driven case finding biases the observed k upward, so 0.1-0.3.
    # Tightened from [0.1, 0.3] onto Taiwan's own estimate of 0.29. The loose version let the
    # infectiousness multiplier sit at its most dispersed setting, which both mis-states the
    # dispersion and makes every layer attack rate impossible to measure from a few hundred
    # index cases, because the infections then cluster into a handful of them (E38).
    'offspring_k':            (0.20, 0.35, 1.0),
    # B92 (2026-11-12, professor): B71's check, charged. Taiwan's largest tracing count (8 over
    # 579 cases) must not fall in either 2.5% tail of the largest of 579 simulated index cases.
    # Run 23 failed it on independent seeds (1.0%: 7 of 1000 index cases infected 9 or more),
    # and the largest spreader was not in the objective, so re-optimising alone could not fix
    # it. The miss is in units of the lower bound (OUTCOME_SCALE_BY_LOWER_BOUND): in units of
    # the band's width, 95 points, falling from 2.5% to 0% would cost almost nothing.
    # Only the lower tail is charged (user, 2026-11-12): 100% means no case reached 8, which
    # with 300 cases per evaluation, fewer than Taiwan's 579, happens about 70% of the time
    # even when the model matches Taiwan; charging it cost run 10 a flat 6.0. The checklist,
    # on 1000 or more cases, still judges both tails.
    'largest_spreader_percentile': (2.5, np.inf, 1.0),
    # B25: close contacts per day BEFORE symptom onset, from the 2020 national survey
    # (6.03 contacts/day) split by the setting composition of Fu 2012.
    # B61 (2026-10-15): checked against Fu 2010's national contact diaries (Zenodo
    # 10.5281/zenodo.6385759), counting only contacts of 15 minutes or more -- the close-contact
    # duration of the 2020 survey and of contact tracing -- one setting per contact (home >
    # school > work > community), per person per day over all 1,954 diaries: household 3.28,
    # school 1.72, workplace 1.65, community 1.76 (covid19-research
    # scripts/summarize_taiwan_contact_data.py). Rule: the lower bound is the 2020 survey value,
    # the upper bound Fu 2010; a band that already holds Fu 2010 is kept.
    # * household [2.27, 3.28]: 2.27 is the 2020 survey's household figure (Table 4). Fu 2010's
    #   "home" is a place, so it includes visitors; by relationship Fu 2012 gives 3.22 a day (all
    #   durations). With about 2.8 other members and a daily probability capped at 0.95, the
    #   model reaches about 2.66, inside the band.
    # * school [1.15, 1.72]: 1.15 is the 2020 survey split by Fu 2012's setting shares.
    # * workplace and community keep [1.20, 1.80]; Fu 2010 (1.65, 1.76) falls inside.
    # B84 (2026-11-04, user): weight 1.0 -> 3.0. Run 21 measured 2.115 against this band, its
    # bootstrap interval reaching only 2.216, while the other four layer targets were already
    # inside theirs, so the search had no reason to spend on this one. The shortfall is a
    # calibration one: the 1,000 index cases have 2.656 household members on average (14.2%
    # live alone), so the ceiling is 2.656 contacts a day when every member is met every day
    # and 2.523 at the upper bound of household_healthy_p, both above the 2.27 lower bound.
    # No bound moves; only the weight.
    'daily_household':        (2.27, 3.28, 3.0),
    'daily_school':           (1.15, 1.72, 1.0),
    # B64 (2026-10-24): reported only. Fu 2010's 1.65 is over every diary, including the people
    # with no job and the employed on a day off; when B64 set it aside the model had no days off.
    # B75 adds them, so the two are comparable again, but the charged workplace targets remain
    # daily_workplace_employed and the one-day shape below, from the same diaries.
    'daily_workplace':        (1.20, 1.80, 0.0),
    # B64: coworker contacts of the employed (enterprise size >= 1) index cases per day before
    # onset: Fu 2010 work contacts of 15 minutes or more met daily or almost daily, weighted to
    # the 2010 population by sex and age as Fu et al. (2012) weight their sample.
    # B75 (2026-10-31, user) replaces B70's average day of the week (2.95 [2.56, 3.36]): the
    # model now has days off (work_calendar), so the target is a day at work again, measured on
    # the days before onset the case was at work: 4.96 [4.34, 5.61] over the 416 weekday diaries
    # with a work contact (covid19-research scripts/summarize_taiwan_contact_data.py at_work()).
    'daily_workplace_employed': (4.34, 5.61, 1.0),
    # B63 (2026-10-16, E100): community upper bound 1.80 -> 7.12. Under B62's scale-free count
    # the mean is set by the tail, and Fu 2010 records at most 40 contacts per person a day
    # (3.9% of diaries at the cap), so its 1.76 is a lower bound: the same law fitted to it
    # (alpha 1.737, s 0.790) has mean 2.08 at the diary cap and 7.12 at CovSyn's cap of 1000,
    # while its zero share (58-60%) and p90 (6-8) hardly move. The shape is charged by
    # community_day_zero_share and community_day_p90 below.
    'daily_municipality':     (1.20, 7.12, 1.0),
    # Health care is not one of the survey's everyday settings; a healthy person does not
    # meet medical staff daily, so this one is bounded from above only.
    # B76 (2026-11-01, user): measured only. The 0.30 had no source (set during Phase D) and,
    # with the timing shape held, blocks B76's health care contact floor: 3x run 18's P[21]
    # meets the floor at 0.47 a day. Data put the pre-onset rate anywhere in 0.08-2.1 a day:
    # 15.40 outpatient visits per person in 2019 (NHI, MOHW 110 negotiation indicators) times
    # 2 close-contact staff or 50 traced contacts per visit (Chung et al. 2020, JKMS 35:e431).
    'daily_health_care':      (0.00, 0.30, 0.0),
    # B33: the severity cascade recomputed from the Taiwan contact-tracing file itself
    # (579 cases): asymptomatic 23.7%, symptomatic->ICU 12.7%, ICU->death 12.5%, CFR 1.2%.
    # B61 (2026-10-15), recomputed on the public 578-row file (recompute_taiwan_targets.py):
    # an ICU case without an onset date counts as symptomatic with a missing date, so
    # asymptomatic 20.4% (118/578), symptomatic->ICU 12.6% (58/460), ICU->death 12.1% (7/58),
    # CFR 1.21% (7/578); the cascade then gives the observed CFR exactly. The old 12.7% put all
    # ICU cases over the cases with an onset date. Every band below already holds these values.
    # All four are measured only (weight 0). Each is exactly what one parameter already
    # says -- P[195] IS the asymptomatic share, P[196] the ICU share, P[197] the death share
    # given ICU, with the age normalisation built so that they keep that meaning -- and the
    # objective is evaluated on 100 FIXED seeds, which contain about 10 ICU cases and 1
    # death. Fitting a 12.7% rate to such a sample would not calibrate the model, it would
    # only overfit those seeds (that particular block happens to hold 1 ICU case). They are
    # constrained by narrow bounds in apply_phaseD_parameters.py instead, and checked for
    # real on the 1000-run Monte Carlo afterwards.
    # E58, 2026-09-27: the asymptomatic share is now CHARGED, at the same [20, 28] the
    # checklist uses. Leaving it at weight 0 rested on two claims that run 5 disproved.
    # (1) "P[195] IS the asymptomatic share, so its narrow bounds are enough" -- they are not:
    # the age normalisation sits on top, so with P[195] pegged at its ceiling of 0.28 the
    # realised share came out 29.2%, outside the band the bounds were meant to guarantee.
    # (2) "charging it would only overfit the 100 fixed seeds" -- the objective has run on 300
    # simulations with the expected-infection estimator since E38, so that risk is much
    # smaller now. The optimizer had a clear reason to push it up: an asymptomatic case is
    # never isolated by its own symptoms (B26), so raising this share is the cheapest way to
    # buy the transmission run 5 was short of. The other three stay measured-only, where the
    # "one parameter already says it" argument does hold.
    'asymptomatic_share':     (0.20, 0.28, 1.0),
    'icu_share_of_symptom':   (0.06, 0.20, 0.0),
    'death_share_of_icu':     (0.08, 0.18, 0.0),
    'case_fatality':          (0.008, 0.018, 0.0),
    # The per-layer CUMULATIVE secondary attack rate, i.e. infections per candidate contact
    # over the whole contact window -- the quantity the calibration anchors are stated in
    # (B8, B9, B13). Until now the anchors were enforced only through the conversion of a
    # cumulative rate into a per-day attack rate, and that conversion assumes how many days
    # a contact is met. B25 changed exactly that, so the conversion stopped holding and the
    # first Phase D run ended with the community layer 27x above its anchor and the
    # household layer below its lower bound (finding E36). The anchor and its bounds are
    # now measured on the simulation and charged directly, which is the same lesson as E23
    # and E34: constrain the quantity the decision is about, not a parameter that used to
    # imply it. Intervals are the anchor bounds recorded in parameters_for_initialization.py.
    'sar_household':          (0.046, 0.101, 1.0),
    'sar_school':             (0.010, 0.040, 1.0),
    'sar_workplace':          (0.015, 0.050, 1.0),
    'sar_health_care':        (0.001, 0.016, 1.0),
    'sar_municipality':       (0.001, 0.010, 1.0),
    # B23: share of medical contacts whose first contact falls 8 or more days after onset.
    # Cheng 2020 has 37% of the medical contacts there, because a hospitalised patient keeps
    # meeting staff. Lengthening the contact window alone does not deliver it -- the
    # optimizer simply moves the symptomatic contact peak earlier -- so it is charged here.
    'medical_late_share':     (0.25, 0.50, 1.0),
    # E59, 2026-09-27: charging ONLY the late tail was a one-sided target and the optimizer
    # answered it one-sidedly. Cheng's medical contacts sit mostly EARLY -- his six bins
    # (<0, 0-3, 4-5, 6-7, 8-9, >9 days from onset) are 33.9 / 21.5 / 5.5 / 2.4 / 15.8 / 20.9%,
    # so 55.4% of them start before day 4 -- while run 5 put 4.4 / 4.6 / 10.4 / 31.0 / 45.5 /
    # 4.2% there, i.e. 9.0% early. The shape came out reversed, the late share overshot its
    # own ceiling (50.7% against [25, 50]), and no acceptance item could see it because only
    # the tail was described. Charging both ends pins the distribution instead of one side of
    # it. The interval is wide because CovSyn's denominator is its own candidate contacts
    # while Cheng's is the contacts Taiwan's tracers chose to record.
    'medical_early_share':    (0.40, 0.70, 1.0),
    # B76 (2026-11-01, user, E61 option a'): every candidate health care contact of the index
    # case, floor only. Run 18 gave 2.597, so Cheng's medical infections could only be reached
    # with a daily attack rate 2.5x the household's. The floor, 3.335, is Cheng's lower bound of
    # medical infections per local index case over the upper Wilson bound of his per-contact
    # rate 6 / 698 (sar_anchors.health_care_contact_floor, calibration_sources.md section 10).
    # Review (user, option A): the floor is counted on every contact, as Cheng's rate is (37% of
    # his medical contacts start 8+ days after onset), not inside Jian's tracing window, which
    # verify_phase_d reports against Jian's 9.89 for local cases. No upper bound: no data gives one.
    'health_care_contacts_per_index': (health_care_contact_floor(), np.inf, 1.0),
    # B27: the shape of the community contact distribution. The Taiwan tracing records have a
    # median of 7 contacts per index case but a p90 of 172 and a maximum of 850; the second
    # Phase D run reproduced the median (9) and nothing else (p90 18, max 33), because
    # nothing in the objective rewarded the tail.
    #
    # The third run charged the RATIO p90/median, target 3-25, to avoid fighting the
    # contacts-per-day target. The optimizer reached 3.00 -- by dropping the median from 9 to
    # 2 while the p90 fell to 12 (finding E50). A ratio with no anchor on its level is a
    # textbook Goodhart target: there are two ways to raise it and the cheap one is to shrink
    # the denominator. Both order statistics are now charged on their own ABSOLUTE level and
    # the ratio is only reported. The intervals are wide because these are single order
    # statistics of a heavy-tailed count measured on a few hundred index cases, and because
    # the tracing p90 of 172 counts every named contact whereas CovSyn counts candidates.
    # E73: over EVERY index case, zeros included -- the population verify_phaseD.py checks and
    # the one Taiwan's median of 7.5 is taken over. Until now the objective used the non-zero
    # cases only, so on run 6 it saw 6 while the checklist saw 0 and called it a failure. This
    # also prices the zero-inflation of E67 directly: every case with no community contact now
    # pulls the charged median down.
    # B63 (2026-10-16, E100): reported, no longer charged. Under B62 a median of 3 or more per
    # case cannot coexist with the one-day Fu 2010 shape, and the 38 tracing records are cases
    # published with their contacts, a selected sample (E88). The one-day Fu 2010 shape below,
    # from the same data as B62's exponent, is charged instead.
    'community_median':       (3.0, 15.0, 0.0),
    # E72: reported, not charged. The interval's centre is 110 while CovSyn's candidate-contact
    # scale reaches 16-44, so even B43's capped scale is 55 and missing the lower bound by 4 cost
    # 0.03 of a 1.82 objective -- the optimizer ignored it, correctly. An interval whose centre
    # the model cannot reach is not an acceptance test. The SHAPE is charged instead, below.
    'community_p90':          (20.0, 200.0, 0.0),
    # E72: charged again, which is only safe now. E50 removed it because the optimizer reached a
    # ratio of 3.0 by pressing the median from 9 down to 2; with the all-case median charged at
    # [3, 15] that move now costs about 2.7 under B43's scaling, far more than growing the tail,
    # so the hole is shut. The interval is NOT the old [3, 25]: run 7's ratio of 4.0 sat inside
    # that, so charging it unchanged would have changed nothing. Taiwan's tracing gives
    # 172 / 7 = 24.6; a p90/median ratio is a shape statistic and travels between the tracing's
    # named-contact denominator and CovSyn's candidate-contact denominator better than either
    # order statistic does on its own, but it is still measured on 81 traced cases, so the
    # interval is set to roughly a factor of three either side of that: [8, 40]. Run 6 reached
    # 7.35 while holding a non-zero median of 6, so 8 is within reach.
    #
    # B50 (E76 / E77), 2026-09-28: [8, 40] was a judgement, and it was NOT reachable: around the
    # Cheng-fitted region no point held the median at 3 with a ratio of 8 (best 6.32), because
    # raising the community contact mean to hold the median costs the Cheng contact fit
    # (probe_tail_price_detail.py). The band is now the data's own uncertainty: the bootstrap
    # 95% CI of p90 / median over the 38 non-zero 2020 first-wave cases with a known uninfected
    # count, [5.5, 93.1] around 26.0 (the old "n = 81" counted those 38 twice, E77). Its miss is
    # scaled by the lower bound (OUTCOME_SCALE_BY_LOWER_BOUND), otherwise B43's half-centre cap
    # (24.7) would make a miss from 4 to 5.5 cost 0.02 and the target would not bite at all.
    # B63: reported, no longer charged, for the same reason as community_median (E100).
    'community_tail_ratio':   (5.5, 93.1, 0.0),
    # E67, 2026-09-27: run 6 finally grew the community tail (p90 13 -> 44.1, max 48 ->
    # 196) and paid for it with the centre -- the median fell 3 -> 0, i.e. more than half
    # the index cases stopped having any community contact at all. With the attack rate
    # capped by B38 the only remaining source of a heavy tail is the contact COUNT
    # dispersion P[198], and the cheapest way to raise that is to zero most cases out.
    # community_median is charged and should now bite (E66 rescaling), so this is
    # reported rather than charged: it is the number that says whether the median moved
    # because the distribution improved or because the zeros were merely reshuffled.
    'community_zero_share':   (0.0, 0.30, 0.0),
    # B63 (2026-10-16): the one-day community shape of Fu 2010 (Zenodo 10.5281/zenodo.6385759),
    # community-only contacts of 15 minutes or more per person per day, n = 1,954: zero share
    # 62.0% (bootstrap 95% 59.8-64.1%) and p90 6 (5-6; 8 for the fitted law without the
    # diary cap, so the upper bound is 8). Measured on the index cases' days before onset,
    # pooled (contact_measures.one_day_shape). Same data as B62's exponent.
    'community_day_zero_share': (0.598, 0.641, 1.0),
    'community_day_p90':        (5.0, 8.0, 1.0),
    # B64 / B75: the one-day shape of the same diaries on a day at work, which pins the work
    # group's saturation P[205] and spread P[206]: zero share 27.4% (bootstrap 95% 23.2-31.8%),
    # p90 13 (10-16). B70 reported them only, because the model had no days off and the zero days
    # of an average day are mostly days off; B75 models those and charges the shape again.
    'workplace_day_zero_share': (0.232, 0.318, 1.0),
    'workplace_day_p90':        (10.0, 16.0, 1.0),
    # B22: incubation back to the literature reported-mean range now that it is built as
    # latent + window rather than drawn and truncated.
    'incubation_mean':        (3.9, 8.0, 1.0),
    # E60, 2026-09-27: the pre-onset infectious window is charged on what the SIMULATION
    # produced, not on the parameter product. physiology_penalty() charges P[41] * P[42], the
    # MEAN of the Gamma, and run 5 satisfied it completely (1.041 in [1, 3], penalty 0.0000)
    # while the realised window averaged 0.990 and 23.9% of cases got a window of zero days --
    # a Gamma with a mean near one day puts a quarter of its mass below a day. Constraining a
    # parameter that used to imply the quantity instead of the quantity itself is the same
    # mistake as E23 and E34; both bands here are the ones verify_phaseD.py already checks.
    'pre_onset_window_mean':  (1.0, 3.0, 1.0),
    # B80 (2026-11-02, user): days a symptomatic case stays infectious after onset, median.
    # Run 19 ended infectiousness a median 1 day after onset, with the mean infectious period at
    # its lower bound of 4 days: Cheng 2020's day 4-5 bins hold only 3 infections, so the
    # objective barely saw it, and the February seeds that infected people in hospital days
    # after onset could not be reproduced. Literature: Cheng et al. (2020, JAMA Intern Med,
    # doi:10.1001/jamainternmed.2020.2020) find infections among contacts first exposed up to
    # day 5 after onset and none later; Cevik et al. (2021, Lancet Microbe,
    # doi:10.1016/S2666-5247(20)30172-5) find no culturable virus after day 9 in non-severe
    # cases. The presymptomatic share was checked first and is already in the literature range
    # (run 19 about 50%; He et al. 2020 and Ashcroft et al. 2020 about 44%), so it is not charged.
    'infectious_after_onset_median': (5.0, 9.0, 1.0),
    'pre_onset_zero_share':   (0.0, 0.12, 1.0),
    # B28: date_of_recovery is the day the case is CLOSED (released from isolation), not a
    # clinical recovery at 14-20 days.
    # E91 (2026-10-08): case closure is measured from CONFIRMATION, the quantity Taiwan's
    # records hold. B28's "onset -> release about 25 days" had been applied to infection ->
    # closure (20-32), which left out the ~5-day incubation period; run 12 sat at 20.7 days,
    # i.e. about 9 days after confirmation against Taiwan's 24. Bands: bootstrap 95% interval
    # of the Taiwan mean confirmed -> recovery (figshare file, symptomatic 27.0, n=56;
    # asymptomatic 23.3, n=23).
    'closure_after_confirmation_symptomatic':  (22.0, 32.0, 1.0),
    'closure_after_confirmation_asymptomatic': (17.0, 31.0, 1.0),
    # E93: nothing constrained ICU -> closure and run 12 settled at 6.4 days; Taiwan 36.2
    # (n=28, bootstrap 95% interval of the mean 28.9-44.0).
    'icu_to_closure':         (29.0, 44.0, 1.0),
    # E92: once ICU admission is no longer capped at the end of infectiousness, its timing
    # needs a target of its own. Taiwan onset -> ICU 7.2 (n=40, 95% interval 5.4-9.2).
    'onset_to_icu':           (5.4, 9.2, 1.0),
    # B2 (2026-09-25): median days from symptom onset to confirmation, for symptomatic index
    # cases. Until now this was left free inside a 1-6 day physiological range with NO target
    # (B26, E27, E33), and all three Phase D runs settled at 1.0 day -- the shortest value
    # available. That single number is the knot behind nine of run 3's seventeen failures:
    # with confirmation one day after onset the case is isolated immediately, so there are
    # almost no post-onset contacts, which forces the pre-onset window down (0.99 d), the
    # share of contacts starting before onset up (71% against Cheng's 27.5%), the daily
    # contact total down (4.71 against 5-7) and case closure down (16.1 d against 20-32).
    #
    # Three independent sources put it at 5-7 days, none of them at 1:
    #   * Taiwan's own tracing file, the same 579 cases the rest of B33 comes from: median 6
    #     days over the n=442 symptomatic cases (finding E33).
    #   * Ge 2021, Zhejiang: median onset to isolation 5 days (IQR 2-8).
    #   * The published CovSyn model itself fits onset to confirmation at a mean of 8.17 days
    #     (bounds 4.42-12.02) -- the 1 day is ours, not the paper's.
    # The "1 day" of B26/E27 is Taiwan's POST-March-2020 figure, after tracing had scaled up;
    # the contact data this objective is fitted to is Cheng's 15 Jan - 18 Mar cohort.
    'onset_to_confirmation':  (5.0, 7.0, 1.0),
}

# The five sar_* intervals ARE the calibration anchors, so they are taken from the module
# that also builds the daily attack-rate bounds rather than written out a second time. Until
# 2026-09-25 the anchors existed in three places with three different sets of numbers, and
# the one this dict quoted ("recorded in parameters_for_initialization.py") was the one that
# never reached variable/course_parameters_*.npy.
for _layer, (_anchor_lo, _anchor_centre, _anchor_hi) in LAYER_CUMULATIVE_SAR.items():
    OUTCOME_TARGETS['sar_' + _layer] = (_anchor_lo, _anchor_hi, 1.0)

# E56, 2026-09-27: for the three layers Cheng reports, what is CHARGED is the number of people
# one index case infects in that layer, and the per-contact rate is only reported.
#
# The per-contact version divided Cheng's infection count by CovSyn's own candidate-contact
# count, which made the target a ratio whose denominator the search controls. Run 5 is the
# demonstration: E55 raised the health care anchor 2.5x to get medical transmission up, the
# optimizer cut health care candidate contacts by 37% (daily medical contacts 0.25 -> 0.09),
# and medical infections per index case FELL from 0.0160 to 0.0140 against Cheng's 0.06. The
# per-contact rate was satisfied; the thing it stood for got worse. Charging the product
# removes the lever, and it also takes the anchors off the treadmill of being re-derived from
# the previous run's contact counts (the circularity admitted in sar_anchors.py) -- Cheng's
# 10, 6 and 1 infections per 100 index cases do not move when CovSyn's contact counts do.
#
# school and workplace keep their per-contact targets: Cheng has no category for either, so
# there is no infections-per-index-case figure to charge. State that asymmetry in the thesis.
for _layer, (_lo, _centre, _hi) in LAYER_INFECTIONS_PER_INDEX.items():
    OUTCOME_TARGETS['infections_per_index_' + _layer] = (_lo, _hi, 1.0)
    OUTCOME_TARGETS['sar_' + _layer] = (OUTCOME_TARGETS['sar_' + _layer][0],
                                        OUTCOME_TARGETS['sar_' + _layer][1], 0.0)
# school and workplace are reported at weight 0 rather than left out, because these five
# numbers are what made E56 legible: run 5's school infections per index case halved (0.088 ->
# 0.042) as a side effect of the contact count collapsing, and nothing in the progress CSV
# showed it. Anything that can move this much unnoticed belongs in the log.
for _layer in ('school', 'workplace'):
    OUTCOME_TARGETS['infections_per_index_' + _layer] = (0.0, np.inf, 0.0)


def measure_outcomes(index_cases):
    """Summarise the index cases of one cost-function evaluation.

    index_cases is a list of (course_of_disease_dict, contact_dict, social_dict); the social
    dict (None when absent) only decides which index cases are employed (B64).
    Returns the dict of measured quantities named in OUTCOME_TARGETS (missing entries are
    simply not penalised, which is what happens when a quantity has no cases behind it).
    """
    layers = ['household', 'school', 'workplace', 'health_care', 'municipality']
    # Every key is always present (NaN when a quantity has no cases behind it) so that the
    # progress CSV keeps a stable set of columns across generations.
    measured = {name: np.nan for name in OUTCOME_TARGETS}
    if not index_cases:
        return measured

    offspring = np.array([sum(np.nansum(np.asarray(contact[f'{L}_effective_contacts'], dtype=float))
                              for L in layers)
                          for _, contact, _ in index_cases], dtype=float)
    # B61: maximum likelihood, the estimator Taiwan's k = 0.29 comes from (was the method of
    # moments). NaN when no case infects anyone; 100 when there is no overdispersion.
    k = offspring_dispersion(offspring)
    if np.isfinite(k):
        measured['offspring_k'] = k
    measured['largest_spreader_percentile'] = largest_spreader_percentile(offspring)  # B92

    for layer in layers:
        # B55: ordinary contacts only, see contact_measures.contacts_per_day_before_onset.
        per_day = [contacts_per_day_before_onset(course, contact, layer)
                   for course, contact, _ in index_cases]
        measured[f'daily_{layer}'] = float(np.mean(per_day))

    # E95: expected infections (and, in the municipality layer, expected event contacts) in
    # place of the realised counts, when every index case carries them.
    expected = [contact.get('expected_outcomes') for _, contact, _ in index_cases]
    use_expected = all(e is not None for e in expected)
    for layer in layers:
        candidate = effective = 0
        for (_, contact, _), exp in zip(index_cases, expected, strict=True):
            eff = contact[f'{layer}_effective_contacts']
            eff = [] if eff is None else list(eff)
            if use_expected:
                candidate += (exp['expected_contacts_municipality'] if layer == 'municipality'
                              else len(eff))
                effective += exp[f'expected_infections_{layer}']
            else:
                candidate += len(eff)
                effective += sum(1 for x in eff if x == 1)
        if candidate:
            measured[f'sar_{layer}'] = effective / candidate
        # E56: the charged quantity for the three layers Cheng reports. Not divided by the
        # candidate contacts, so the optimizer cannot satisfy it by removing contacts -- which
        # is exactly what it did with the per-contact version on run 5. Also recorded for
        # school and workplace, where it is reported but not charged (Cheng has no category).
        key = f'infections_per_index_{layer}'
        if key in measured:
            measured[key] = effective / len(index_cases)
        if layer == 'health_care':
            # B76: every candidate health care contact per index case, zeros included.
            measured['health_care_contacts_per_index'] = candidate / len(index_cases)

    # B23 / B27: the two distribution-shape targets. Both ends of the medical contact timing
    # are charged since E59 -- "early" is Cheng's first two bins (<0 and 0-3 days from onset),
    # "late" his last two (8-9 and >9), using the same bin edges as
    # compare_healthcare_municipality.py so the figure and the objective cannot disagree.
    early = late = total = 0
    for course, contact, _ in index_cases:
        onset = course['incubation_period']
        if onset is None or np.isnan(onset):
            continue
        matrix = np.asarray(contact['health_care_contacts_matrix'], dtype=float)
        if matrix.size == 0:
            continue
        first = np.argmax(matrix > 0, axis=1) - onset
        total += len(first)
        early += int((first < 4).sum())
        late += int((first >= 8).sum())
    if total:
        measured['medical_late_share'] = late / total
        measured['medical_early_share'] = early / total

    community = np.array([len(contact['municipality_effective_contacts'] or [])
                          for _, contact, _ in index_cases], dtype=float)
    measured['community_zero_share'] = float(np.mean(community == 0))
    # E73: the charged median is over every index case, matching verify_phaseD.py and the
    # tracing reference. The tail statistics stay on the non-zero cases, which is also what
    # verify_phaseD.py reports, because a p90 over a sample that is a quarter zeros mostly
    # measures the zeros.
    measured['community_median'] = float(np.median(community))
    nonzero = community[community > 0]
    if len(nonzero) > 20:
        median, p90 = float(np.median(nonzero)), float(np.percentile(nonzero, 90))
        measured['community_p90'] = p90
        if median > 0:
            measured['community_tail_ratio'] = p90 / median
    # B63: the one-day community shape, every index case's days before onset pooled as diary
    # days and compared with Fu 2010 (contact_measures.one_day_shape, shared with fast_cost).
    measured['community_day_zero_share'], measured['community_day_p90'] = one_day_shape(
        [one_day_counts(course, contact, 'municipality') for course, contact, _ in index_cases])
    # B64: the workplace targets are over the employed index cases only (enterprise size >= 1,
    # read from the social data the index case carries).
    employed = [(course, contact) for course, contact, social in index_cases
                if is_employed(social)]
    if employed:
        # B75: the days before onset the case was at work, as Fu 2010's diaries of a day at work.
        measured['daily_workplace_employed'] = float(np.nanmean(
            [contacts_per_day_at_work(course, contact) for course, contact in employed]))
        measured['workplace_day_zero_share'], measured['workplace_day_p90'] = one_day_shape(
            [at_work_days_before_onset(course, contact) for course, contact in employed])

    incubation = np.array([c['incubation_period'] for c, _, _ in index_cases], dtype=float)
    symptomatic = ~np.isnan(incubation)
    measured['asymptomatic_share'] = float(np.mean(~symptomatic))
    if symptomatic.any():
        measured['incubation_mean'] = float(np.nanmean(incubation))

    # E60: the realised pre-onset infectious window, not the Gamma mean physiology_penalty()
    # charges. Same field and same two statistics verify_phaseD.py checks under B22.
    pre_onset = np.array([c['pre_onset_window'] for c, _, _ in index_cases], dtype=float)
    if np.isfinite(pre_onset).any():
        measured['pre_onset_window_mean'] = float(np.nanmean(pre_onset))
        # E96: over symptomatic cases only. nanmean(pre_onset == 0) counted the asymptomatic
        # cases (NaN == 0 is False, not NaN) and understated the share by about a quarter.
        measured['pre_onset_zero_share'] = float(np.mean(pre_onset[np.isfinite(pre_onset)] == 0))
    # B80: infectious days after onset, symptomatic cases only.
    after_onset = np.array([c['latent_period'] + c['infectious_period'] - c['incubation_period']
                            for c, _, _ in index_cases], dtype=float)
    if np.isfinite(after_onset).any():
        measured['infectious_after_onset_median'] = float(np.nanmedian(after_onset))

    # Onset -> confirmation, realised rather than read off the parameters. It is negative for
    # a case isolated before its own onset; an index case has no tracing source so that is
    # rare here, but the draws are kept rather than clipped so the median is the same
    # quantity Taiwan's records report.
    onset_to_confirmation = np.array(
        [c['positive_test_date'] - c['infection_day'] - c['incubation_period']
         for c, _, _ in index_cases], dtype=float)[symptomatic]
    if np.isfinite(onset_to_confirmation).any():
        measured['onset_to_confirmation'] = float(np.nanmedian(onset_to_confirmation))

    icu = np.array([not np.isnan(c['date_of_critically_ill']) for c, _, _ in index_cases])
    dead = np.array([not np.isnan(c['date_of_death']) for c, _, _ in index_cases])
    if symptomatic.sum():
        measured['icu_share_of_symptom'] = float(icu.sum() / symptomatic.sum())
    if icu.sum():
        measured['death_share_of_icu'] = float(dead.sum() / icu.sum())
    measured['case_fatality'] = float(dead.mean())

    # E91 / E92 / E93: closure from confirmation, ICU -> closure, onset -> ICU, computed by
    # the same function fast_cost.reduce_outcomes calls, so the two stay bit-identical.
    def column(key):
        return np.array([c[key] for c, _, _ in index_cases], dtype=float)
    measured.update(course_timing(column('date_of_recovery'), column('positive_test_date'),
                                  column('date_of_critically_ill'), column('infection_day'),
                                  column('incubation_period'), symptomatic))
    return measured


def course_timing(recovery: np.ndarray, positive: np.ndarray, critical: np.ndarray,
                  infection: np.ndarray, incubation: np.ndarray,
                  symptomatic: np.ndarray) -> dict[str, float]:
    """Closure after confirmation, ICU -> closure and onset -> ICU, as means (E91-E93).

    Args:
        recovery: Closure day of each case (absolute day, NaN if the case died).
        positive: Positive test day (absolute).
        critical: ICU admission day (absolute, NaN if never critical).
        infection: Infection day.
        incubation: Incubation period (NaN for asymptomatic cases).
        symptomatic: Boolean mask of symptomatic cases.

    Returns:
        The measured values that exist in this sample.
    """
    out = {}
    closure = recovery - positive
    for key, mask in (('closure_after_confirmation_symptomatic', symptomatic),
                      ('closure_after_confirmation_asymptomatic', ~symptomatic)):
        if np.isfinite(closure[mask]).any():
            out[key] = float(np.nanmean(closure[mask]))
    icu_stay = recovery - critical
    if np.isfinite(icu_stay).any():
        out['icu_to_closure'] = float(np.nanmean(icu_stay))
    onset_to_icu = critical - infection - incubation
    if np.isfinite(onset_to_icu).any():
        out['onset_to_icu'] = float(np.nanmean(onset_to_icu))
    return out


# E66, 2026-09-27: how wide an interval is allowed to be before it stops setting the scale of
# the miss, as a fraction of its own centre. Measuring the miss in units of the interval width
# means the widest intervals produce the weakest gradients -- and the widest intervals are the
# ones derived from small Poisson counts, i.e. exactly the quantities the calibration is about.
# On run 6 the medical layer sat at 0.015 infections per index case against Cheng's 0.06, a
# factor of four out, and cost 0.0041 of an objective of 1.87, because its interval
# [0.022, 0.131] is 1.8 times as wide as its centre.
MAX_INTERVAL_WIDTH_OVER_CENTRE = 0.5


# B50: targets whose band is a wide data CI (lower bound far below the centre) have their miss
# measured in units of the lower bound instead of B43's capped width, so the target still bites
# near its lower end. Kept to the one target that needs it.
# B76: a floor with no upper bound has no width either, so its miss is in units of the floor.
# B92: the largest-spreader percentile is a floor, 2.5 points above 0.
OUTCOME_SCALE_BY_LOWER_BOUND = {'community_tail_ratio', 'health_care_contacts_per_index',
                                'largest_spreader_percentile'}


# B60: the intervals the objective charges, from target_margins.py; OUTCOME_TARGETS keeps the
# acceptance intervals that the checklist judges.
CHARGED_BOUNDS = load_charged_bounds(targets=OUTCOME_TARGETS)


def outcome_scale(name, lo, hi):
    """Unit a target's miss is measured in; None when the band is empty. Shared with
    calibrate_outcome_weight.py so the two cannot drift apart (lesson 4)."""
    width = hi - lo
    if width <= 0:
        return None
    if name in OUTCOME_SCALE_BY_LOWER_BOUND and lo > 0:
        return lo
    centre = abs(lo + hi) / 2.0
    scale = min(width, MAX_INTERVAL_WIDTH_OVER_CENTRE * centre) if centre > 0 else width
    return scale if scale > 0 else width


def pre_onset_zero_probability(P: np.ndarray) -> float:
    """Probability that a symptomatic case gets a zero-day pre-onset window (E95).

    draw_pre_onset_window() rounds a Gamma(P[41], P[42]) draw, so the window is zero exactly
    when the draw is at most 0.5 (Python rounds 0.5 to 0); the cap at the infectious period
    cannot make it zero because that period is at least one day (C02). This is the expectation
    of the share E60 measured on the simulated index cases, without the noise of 300 of them.
    """
    from scipy import stats

    return float(stats.gamma.cdf(0.5, a=P[41], scale=P[42]))


def outcome_penalty(measured, weight=OUTCOME_PENALTY_WEIGHT):
    """Relative distance outside the acceptance interval, summed over targets.

    The miss is measured in units of the acceptance interval's width, CAPPED at
    MAX_INTERVAL_WIDTH_OVER_CENTRE of the interval's centre, so that widening an interval can
    no longer make a large relative error cheap (E66). It stays quadratic while the miss is
    below one unit and linear beyond, so a badly-placed quantity points the search in the right
    direction without outweighing the whole Cheng contact fit.

    What is NOT restored here is the original ratio form, max(0, lo/x - 1): that is the one
    that blew up as a quantity approached zero, charging 249 units for a late-medical-contact
    share of 0.001 against a lower bound of 0.25, which alone was 80% of the objective. The
    cap below is bounded -- the worst a vanishing quantity can score is lo divided by the
    capped scale -- so it does not bring that behaviour back.
    """
    pen = 0.0
    for name, (lo, hi, w) in OUTCOME_TARGETS.items():
        x = measured.get(name, np.nan)
        if w <= 0 or not np.isfinite(x):
            continue
        scale = outcome_scale(name, lo, hi)
        if scale is None:
            continue
        # B60: the miss is measured from the charged interval (the acceptance interval moved
        # inward by its Monte-Carlo noise) in units of the ACCEPTANCE interval, so only where
        # the penalty starts moves, not how steeply it grows.
        charged_lo, charged_hi = CHARGED_BOUNDS.get(name, (lo, hi))
        miss = max(0.0, charged_lo - x, x - charged_hi) / scale
        pen += w * (miss ** 2 if miss <= 1.0 else 2.0 * miss - 1.0)
    return weight * pen


# A parameter vector that makes a simulation crash (a Gamma truncated beyond its numerical
# support, say) is a bad vector, not a reason to lose the whole run. It is charged this cost
# and the traceback is printed so the failure is still visible in the log.
FAILED_EVALUATION_COST = 1e6


# Simulations per objective evaluation. 100 was too few: the layer attack rate then has a
# standard error of 2.8-6.2 percentage points against anchor intervals half that wide, so the
# optimizer fitted the particular 100 seeds rather than the model (E38). 300 halves that. The
# generation count comes down from 200 to 120 to pay for it -- the second run's cost curve was
# already nearly flat after 150 (1.36 -> 1.27).
SIMULATIONS_PER_EVALUATION = 300
# B81 (2026-11-03): every generation scores the population on its own block of seeds. Run 20
# selected a vector whose workplace SAR was 0.024 on the fixed block 0..299 and 0.070 over ten
# unseen blocks -- below the lowest of the ten -- while an unoptimised vector sat 9th of eleven,
# so the search was fitting the fixed sample (winner's curse over 664,435 evaluations).
# Generation g scores on block g. The initial population is scored on block 0, so generation 0
# moves on those scores. Before the code review of 2026-11-09 (F7) generation 0 scored the same
# population again on block 300 and discarded the block-0 scores: one population of evaluations
# for a different noise draw, not a better search. Runs from then on therefore take other paths
# than runs 20-22. 120 generations ended at 35,999, clear of revalidation's blocks from 100,000
# (E87); since Task 7 each run scores on its own optimisation stream, which no validation
# stream meets (seed_streams).
# Costs from different blocks are NOT comparable, so nothing keeps a running minimum across
# generations: the attraction, the elite and the progress row all use this generation's scores,
# and the candidates are re-scored on one common set of blocks afterwards. Until Task 8 the
# archive held a snapshot of the best firefly of one generation in five; it now holds every
# elite vector with its mean over the blocks it was scored on there (record_elite).


# Task 7: the move generator's seed when none is given.
DEFAULT_RNG_SEED = 1

# Task 8: the step size schedule, in step_schedule so the pipeline checks its settings by the
# same rule without loading the model.
DEFAULT_ALPHA_0 = step_schedule.DEFAULT_ALPHA_0
DEFAULT_ALPHA_END = step_schedule.DEFAULT_ALPHA_END
step_size = step_schedule.step_size

# Task 6: the synchronous firefly algorithm (Yang, 2010). config.json records it, and a run
# that does not, started before Task 6 with its sequential moves, is not resumed with these.
MOVES = 'synchronous'


# Task 8: the elite's size; a population of n <= 5 keeps n - 1, so one firefly still moves
# (decision of 2026-11-20).
ELITE_SIZE = 5


def elite_size(pop_size: int) -> int:
    """The number of fireflies a generation keeps unmoved, k = min(5, n - 1) (Task 8)."""
    return max(0, min(ELITE_SIZE, pop_size - 1))


def elite_members(costs, k: int) -> list[int]:
    """The indices of the k lowest costs, lowest first, a tie going to the lower index."""
    return [int(i) for i in np.argsort(np.asarray(costs), kind='stable')[:k]]


# Task 8: an elite record is (entered, last block, blocks, total cost, params...): the
# generation whose block first scored the vector in the elite, the last such block, how many
# blocks, and the sum of their costs. The checkpoint keeps the records, so a resumed run goes on
# counting where it stopped.
RECORD_PARAMS = 4
ARCHIVE_HEADER = revalidation.ARCHIVE_HEADER


def no_elite_records(dim: int) -> np.ndarray:
    """The records of a run before any generation: none, with room for dim parameters."""
    return np.empty((0, RECORD_PARAMS + dim))


def record_elite(records, fireflies, intensities, members, block: int) -> np.ndarray:
    """The records after adding each member's cost on block (Task 8).

    A vector already recorded adds the cost unless that block is already counted, so a block
    counts once; any other vector, such as a member that moved, starts a record of its own.
    Members are taken in ascending order, so new records come in a fixed order.

    Returns:
        A new array; ``records`` is left unchanged.
    """
    records = np.array(records, dtype=float)
    for m in sorted(members):
        vector = np.asarray(fireflies[m], dtype=float)
        same = np.flatnonzero(np.all(records[:, RECORD_PARAMS:] == vector, axis=1))
        if len(same) == 0:
            new = np.concatenate(([block, block, 1, intensities[m]], vector))
            records = np.vstack([records, new])
        elif records[same[0], 1] != block:
            row = records[same[0]]  # a view, so the update lands in records
            row[1] = block
            row[2] += 1
            row[3] += intensities[m]
    return records


def archive_rows(records) -> np.ndarray:
    """firefly_archive.txt's rows, (entered, blocks, params..., mean cost), from the records."""
    records = np.asarray(records, dtype=float)
    return np.column_stack([records[:, 0], records[:, 2], records[:, RECORD_PARAMS:],
                            records[:, 3] / np.maximum(records[:, 2], 1)])


def synchronous_moves(normalized, costs, alpha, betamin, gamma, rng, fixed=()):
    """Every firefly's new normalised position, from the positions and costs given (Task 6).

    Firefly i moves towards each strictly brighter firefly j (costs[j] < costs[i]), for i and
    then j in ascending order, by beta (x_j - x_i) plus a uniform step of up to alpha / 2
    per coordinate, beta = betamin exp(-gamma r) with r the squared distance (B51) from i's
    position so far to j's start position, and is clipped to [0, 1] after each move. Nothing
    is evaluated between moves, so the brightest firefly, and one as bright, stays put; the
    steps are drawn from rng in that fixed order, one vector per move. A firefly in ``fixed``,
    the elite (Task 8), makes no move and draws no step.

    Returns:
        The new positions; ``normalized`` is left unchanged.
    """
    start = np.asarray(normalized, dtype=float)
    moved = start.copy()
    fixed = set(fixed)
    for i in range(len(start)):
        if i in fixed:
            continue
        for j in range(len(start)):
            if costs[j] < costs[i]:
                r = np.sum(np.square(moved[i] - start[j]), axis=-1)
                beta = betamin * np.exp(-gamma * r)
                steps = alpha * (rng.random(start.shape[1]) - 0.5)
                moved[i] = np.clip(moved[i] + beta * (start[j] - moved[i]) + steps, 0, 1)
    return moved


def generation_rng(seed: int, generation: int) -> np.random.Generator:
    """The move generator of a generation, from SeedSequence([seed, generation]) (Task 4).

    Each generation draws its moves from its own generator, so a resumed run needs no saved
    generator state to make the same moves as an uninterrupted one.
    """
    return default_rng(np.random.SeedSequence([seed, generation]))


CHECKPOINTS = runs.CHECKPOINTS


def save_checkpoint(directory: Path, generation: int, state: dict) -> None:
    """Write generation's state to directory/checkpoints/generation_<g>.npz, all or nothing.

    It is written under a temporary name and renamed, so a run killed while writing leaves the
    previous checkpoint as the last one.
    """
    folder = Path(directory) / CHECKPOINTS
    folder.mkdir(exist_ok=True)
    # Generation -1 is the evaluated initial population, so a run killed in generation 0 resumes.
    name = 'initial' if generation < 0 else f'generation_{generation:04d}'
    final = folder / f'{name}.npz'
    temporary = folder / f'{name}.tmp.npz'
    np.savez(temporary, generation=generation, **state)
    os.replace(temporary, final)


def load_last_checkpoint(directory: Path) -> dict:
    """The state of the last finished generation in directory/checkpoints, as a dict.

    Raises:
        FileNotFoundError: The directory holds no checkpoint to resume from.
    """
    last = runs.last_checkpoint(directory)
    if last is None:
        raise FileNotFoundError(f'{directory} holds no checkpoint to resume from')
    with np.load(last) as saved:
        return {name: saved[name] for name in saved.files}


def check_progress_table(run_dir: Path) -> None:
    """Refuse to resume a run without a checkpoint, or whose finished generations lost their
    progress table (Task 4).

    Raises:
        SystemExit: runs.NOT_RESUMABLE, as at startup; see runs.resume_problem.
    """
    problem = runs.resume_problem(run_dir)
    if problem is not None:
        not_resumable(problem)


def drop_progress_after(path: Path, generation: int) -> None:
    """Keep the table's rows up to generation; later ones were written after its checkpoint.

    Also drops a last row a kill cut off, which lacks its newline (Task 5): its generation
    could read as an earlier one, or as no number at all.
    """
    if not path.exists():
        return
    text = path.read_text(newline='')
    if text and not text.endswith('\n'):
        text = text[:text.rfind('\n') + 1]
    reader = csv.DictReader(io.StringIO(text, newline=''))  # csv's own line rules
    rows = [r for r in reader if _whole_row(r) and int(r['generation']) <= generation]
    with open(path, 'w', newline='') as f:
        if reader.fieldnames:  # the header stays when every row goes
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _whole_row(row: dict) -> bool:
    """Whether a table row has every cell, no extra one, and a generation that is a number."""
    if None in row or None in row.values():
        return False
    try:
        int(row['generation'])
    except (KeyError, ValueError):
        return False
    return True


def generation_seed_offset(generation, simulations=SIMULATIONS_PER_EVALUATION,
                           stream=seed_streams.DEFAULT_OPTIMISATION_STREAM):
    """First simulation seed of a generation's own block (B81); generation 0 shares block 0
    with the initial population, whose scores it starts from (F7). A block is one evaluation's
    simulations, which only profile mode changes (Task 5). Task 7: block g of the run's
    optimisation stream, s x 10^7 + g x S, so no run meets another use's seeds."""
    return seed_streams.block_start(stream, generation, simulations)

# E86: the objective summed the contact bins of all 300 simulated cases and compared them
# with Cheng et al. 2020's totals for 100 cases (the original 100 cases x 3 repeats lost its
# division by 3 when E38 raised the simulations), pushing contacts per case towards a third of
# Cheng's from run 3 to run 11. The bins hold symptomatic cases only (days from onset), so
# they are scaled to Cheng's 91 symptomatic index cases; Cheng's bins also hold the 91
# contacts of his 9 asymptomatic cases (Table 2), 3.3% of his 2,761, which cannot be
# separated by setting and are left in.
CHENG_SYMPTOMATIC_INDEX_CASES = CHENG2020_INDEX_CASES - 9


def cheng_contact_scale(symptomatic_cases: int) -> float:
    """Factor that puts contact bins summed over these symptomatic cases on Cheng's scale."""
    # No symptomatic case leaves no Cheng bins to scale; raising lets cost_function charge
    # FAILED_EVALUATION_COST instead of returning a NaN the firefly cannot rank.
    if symptomatic_cases <= 0:
        raise ValueError('no symptomatic case among the pooled cases')
    return CHENG_SYMPTOMATIC_INDEX_CASES / symptomatic_cases


def cost_function(P, demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate,
                  norm_weights, seed_offset=0):
    try:
        return _cost_function(P, demographic_parameters, executor, Cheng_contact_array,
                              Cheng_attack_rate, norm_weights, seed_offset)
    except Exception:
        traceback.print_exc()
        print('cost_function: simulation failed for this parameter vector, charging '
              f'{FAILED_EVALUATION_COST:g}', flush=True)
        LAST_COST_PARTS.clear()
        LAST_COST_PARTS.update(cost_contact=np.nan, cost_attack_rate=np.nan,
                               cost_energy=np.nan, cost_penalty=np.nan, cost_outcome=np.nan,
                               cost_contact_household=np.nan, cost_contact_healthcare=np.nan,
                               cost_contact_others=np.nan,
                               **{f'measured_{k}': np.nan for k in OUTCOME_TARGETS})
        return FAILED_EVALUATION_COST


def _cost_function(P, demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate,
                   norm_weights, seed_offset=0):
    source_case_number = SIMULATIONS_PER_EVALUATION
    repeat_number = 1
    seeds = range(seed_offset, seed_offset + source_case_number * repeat_number)

    # Submit simulations to the SHARED (persistent) process pool passed in by firefly().
    # Downstream convert_synthetic_data_to_test_matrix()/generate_contact_result() call
    # future.result(), which blocks until each simulation finishes, so no explicit wait
    # is needed here. Reusing one pool avoids re-spawning/re-importing all workers on
    # every evaluation (the dominant cost on macOS 'spawn').
    # Batched into tasks of SIMULATIONS_PER_TASK simulations each: one simulation is only
    # ~0.8 ms, so one task per simulation left the pool 94% idle on dispatch overhead.
    seeds_copy = list(seeds)
    P_copy = copy.deepcopy(P)
    demographic_parameters_copy = copy.deepcopy(demographic_parameters)
    batches = [seeds_copy[i:i + SIMULATIONS_PER_TASK]
               for i in range(0, len(seeds_copy), SIMULATIONS_PER_TASK)]
    futures = [executor.submit(_run_simulation_batch, batch, P_copy, demographic_parameters_copy)
               for batch in batches]
    # Flattened back in seed order, so every consumer sees the same sequence as before.
    results = [_CompletedSimulation(value) for future in futures for value in future.result()]

    # print('len results', len(results))
    # results = []
    # for i in seeds:
    #     result = run_covid(seeds[i], P, demographic_parameters, save_file=False)
    #     print(result)
    #     results.append(result)

    # Constants
    max_Cheng_contact = np.max(Cheng_contact_array)
    norm_Cheng_contact_array = Cheng_contact_array/max_Cheng_contact

    max_Cheng_attack_rate = np.max(Cheng_attack_rate)
    norm_Cheng_attack_rate = Cheng_attack_rate/max_Cheng_attack_rate

    # weights = np.array([[12.19512195, 6.49350649, 1.87265918, 2.04081633, 1.52207002, 1],
    #                     [35.71428571, 20, 7.69230769,
    #                         5.43478261, 30.3030303, 38.46153846],
    #                     [100, 166.66666667, 31.25, 23.80952381, 90.90909091, 30.3030303]])
    # max_weights = np.max(weights)
    # norm_weights = weights/max_weights
    # weights can be found in plot_compare_previous_studies.ipynb

    layers = ['Household', 'Health care', 'Cheng others']

    contact_costs = []
    attack_rate_costs = []
    # Walk the simulation results ONCE, not once per layer (see extract_course_and_contact).
    course_of_disease_data_list, contact_data_list = extract_course_and_contact(
        results, source_case_number * repeat_number)
    # print('Time spend before loop: ', time.time() - start_t)
    # start_t = time.time()
    # E86: contacts per Cheng cohort of symptomatic index cases.
    contact_scale = cheng_contact_scale(sum(
        1 for course in course_of_disease_data_list if not np.isnan(course['incubation_period'])))
    for i, layer in enumerate(layers):
        contact_array, infection_array = generate_contact_result(
            course_of_disease_data_list, contact_data_list, layer=layer)
        contact_array = contact_array.astype(float)
        norm_contact_array = contact_array/max_Cheng_contact
        infection_array = infection_array.astype(float)
        attack_rate = np.divide(infection_array, contact_array, out=np.zeros_like(
            infection_array), where=contact_array != 0)
        norm_attack_rate = attack_rate/max_Cheng_attack_rate
        norm_Cheng_data = norm_Cheng_contact_array[i]
        norm_Cheng_attack = norm_Cheng_attack_rate[i]

        # Calculate costs
        if layer == 'Health care':
            health_care_weights = np.array([1, 1, 1, 1, 2, 2])
            cost = np.sum(
                ((norm_contact_array * contact_scale -
                  norm_Cheng_data)*health_care_weights) ** 2)
        else:
            cost = np.sum(
                (norm_contact_array * contact_scale - norm_Cheng_data) ** 2)
        attack_rate_cost = np.nansum(
            ((norm_attack_rate - norm_Cheng_attack) * norm_weights[i]) ** 2)
        # if layer == 'Household':
        #     print('Layer: ', layer)
        #     print('norm_Cheng_attack: ', norm_Cheng_attack)
        #     print('norm_attack_rate: ', norm_attack_rate)
        #     print('attack_rate_cost: ', attack_rate_cost)
        #     print()
        # print('Attack rate cost: ', attack_rate_cost)
        contact_costs.append(cost)
        attack_rate_costs.append(attack_rate_cost)
    # print('Time spend in loop: ', time.time() - start_t)
    # start_t = time.time()
    # Energy distance
    taiwan_data_matrix = np.load('./variable/Taiwan_data_matrix.npy')
    synthetic_data_matrix = convert_synthetic_data_to_test_matrix(
        results, taiwan_data_matrix, source_case_number * repeat_number)
    energy_cost = energy_statistic(taiwan_data_matrix, synthetic_data_matrix)  # B91
    # energy_cost = max(energy_cost, 0)  # Prevent negative energy cost
    # Objective function
    # print('Energy cost: ', energy_cost)
    energy_weight = 1
    contact_cost = sum(contact_costs)
    attack_cost = ATTACK_RATE_WEIGHT * sum(attack_rate_costs)
    penalty = physiology_penalty(P)
    # Phase D: quantities that can only be judged on what the simulation actually produced
    # -- offspring dispersion, daily contacts per setting, the severity cascade and the
    # time to case closure. Measured on the index case of every simulation.
    index_cases = []
    for result in results:
        _, social_list, course_list, contact_list = result.result()
        if course_list and contact_list:
            index_cases.append((course_list[0], contact_list[0],
                                social_list[0] if social_list else None))
    measured = measure_outcomes(index_cases)
    # The sampled share stays in the log next to the charged probability (E95, E96).
    measured['pre_onset_zero_share_sampled'] = measured['pre_onset_zero_share']
    measured['pre_onset_zero_share'] = pre_onset_zero_probability(P)
    outcome = outcome_penalty(measured)
    total_cost = contact_cost + attack_cost + energy_weight * energy_cost + penalty + outcome
    LAST_COST_PARTS.clear()
    LAST_COST_PARTS.update(
        cost_contact=float(contact_cost), cost_attack_rate=float(attack_cost),
        cost_energy=float(energy_weight * energy_cost), cost_penalty=float(penalty),
        cost_outcome=float(outcome),
        cost_contact_household=float(contact_costs[0]),
        cost_contact_healthcare=float(contact_costs[1]),
        cost_contact_others=float(contact_costs[2]),
        **{f'measured_{k}': float(v) for k, v in measured.items()})
    # print('Total cost: ', total_cost)
    # print()

    # print(f'Household Cost: {costs[0]:.4f}')
    # print(f'Health Care Cost: {costs[1]:.4f}')
    # print(f'Cheng Others Cost: {costs[2]:.4f}')
    # print(f'Energy Cost: {energy_cost:.4f}')
    # print('Total cost: ', total_cost)
    # print()
    # print('Time spend in cost: ', time.time() - start_t)
    return total_cost


def decode_metrics(P):
    """Decode a 198-dim firefly parameter vector into interpretable, human-checkable
    quantities (gamma means, key probabilities, per-layer attack rates) for live
    monitoring. Index mapping matches run_covid() in Data_synthesis_main.py."""
    m = {}
    m['latent_mean'] = P[37] * P[38]
    m['infectious_mean'] = P[39] * P[40]
    # P[41] / P[42] describe the PRE-ONSET WINDOW since Phase D (B22 (3)); the incubation
    # period is latent + window, so it is derived here rather than read off directly.
    m['presymptomatic_window'] = P[41] * P[42]
    m['incubation_mean'] = m['latent_mean'] + m['presymptomatic_window']
    m['symptom_to_confirmed_mean'] = P[43] * P[44] + P[45]
    m['asymptomatic_to_recovered_mean'] = P[46] * P[47] + P[48]
    m['sympt_to_crit_mean'] = P[49] * P[50] + P[51]
    m['sympt_to_recovered_mean'] = P[52] * P[53] + P[54]
    m['crit_to_recovered_mean'] = P[55] * P[56] + P[57]
    m['infection_to_death_mean'] = P[58] * P[59]
    # B17: P[35] / P[36] are now the Gamma shape and the cap of the case-level
    # infectiousness multiplier, not a per-contact lottery rate and weight.
    m['infectiousness_shape_k'] = P[35]
    m['infectiousness_cap'] = P[36]
    if len(P) > 198:
        # E35: dispersion of the community contact count, independent of infectiousness.
        m['community_dispersion_k'] = P[198]
    if len(P) >= COMMUNITY_EVENT_FIRST_INDEX + len(COMMUNITY_EVENT_FIELDS):
        # B54: the two searched mass-event parameters.
        m['community_event_probability'] = P[COMMUNITY_EVENT_FIRST_INDEX]
        m['community_event_risk_ratio'] = P[COMMUNITY_EVENT_FIRST_INDEX
                                            + COMMUNITY_EVENT_FIELDS.index('risk_ratio')]
    m['age_risk_0_19'] = P[63]
    m['age_risk_20_39'] = P[64]
    m['age_risk_40_59'] = P[65]
    m['age_risk_60p'] = P[66]
    m['infect_to_recov_p'] = P[195]
    m['symptom_to_recov_p'] = P[196]
    m['crit_to_recov_p'] = P[197]
    for name, i in [('household', 70), ('school', 95), ('workplace', 120),
                    ('healthcare', 145), ('municipality', 170)]:
        m['attack_' + name] = float(np.mean(P[i:i + 25]))
    m['school_healthy_p'] = P[9]
    m['school_symptom_p'] = P[10]
    m['household_healthy_p'] = P[2]
    m['workplace_healthy_p'] = P[16]
    m['health_care_gate'] = P[21]
    # B27: P[28] is the mean number of distinct community contacts per case, not a
    # probability against the city population.
    m['municipality_contacts_lambda'] = P[28]
    m['municipality_healthy_p'] = P[30]
    return {k: float(v) for k, v in m.items()}


class Firefly:
    def __init__(self, pop_size, alpha_0, betamin, gamma, seed=None,
                 alpha_end=DEFAULT_ALPHA_END, g_plan=None):
        self.pop_size = pop_size
        # Task 8: the step size schedule, see step_size; g_plan None is the run's generations.
        self.alpha_0 = alpha_0
        self.alpha_end = alpha_end
        self.g_plan = g_plan
        self.betamin = betamin
        self.gamma = gamma
        # Task 7: the seed of the moves, config.json's rng_seed; without one, the fixed default,
        # so two runs of one configuration make the same moves (decision of 2026-11-17).
        self.seed = DEFAULT_RNG_SEED if seed is None else seed
        self.rng = default_rng(self.seed)
        # Task 3: recorded in config.json with the settings above; main() adds its own.
        self.run_settings = {}
        # Task 5: S, run_covid's stage timing in the workers, and, in profile mode, where the
        # workers write their profiles.
        self.simulations = SIMULATIONS_PER_EVALUATION
        self.stage_timing = False
        self.profile_dir = None
        # Task 7: the run's optimisation stream, whose block g generation g is scored on.
        self.stream = seed_streams.DEFAULT_OPTIMISATION_STREAM

    @staticmethod
    def _state(normalized, fireflies, intensities, cost_parts, elite_records, worst_fireflies,
               worst_intensities, worst_iteration, evaluations, alpha, dim):
        """What a checkpoint holds: everything the next generation reads (Task 4)."""
        return {'normalized': np.asarray(normalized, dtype=float),
                'fireflies': np.asarray(fireflies, dtype=float),
                'intensities': np.asarray(intensities, dtype=float),
                'cost_parts': np.array(json.dumps(cost_parts)),
                'elite_records': np.asarray(elite_records, dtype=float).reshape(
                    len(elite_records), RECORD_PARAMS + dim),
                'worst_fireflies': np.asarray(worst_fireflies, dtype=float),
                'worst_intensities': np.asarray(worst_intensities, dtype=float),
                'worst_iteration': np.asarray(worst_iteration, dtype=float),
                'evaluations': evaluations, 'alpha': alpha}

    def firefly(self, function, dim, lb, ub, demographic_parameters, max_generations, max_workers, Cheng_contact_array, Cheng_attack_rate, norm_weights, initial_population=None, resume=False):
        # Task 7: a run starts from the first pop_size rows of the committed initial population,
        # in normalised coordinates, which main() reads and checks against the bounds: the
        # literature vector (firefly 0), the warm starts (B49, E75: run 8 finished at 3.80
        # while run 7's best vector scores 2.75 on run 8's own objective), then a Latin
        # hypercube. Without one, as in the tests, the population is drawn from the seed.
        if initial_population is None:
            normalized_fireflies = self.rng.uniform(0, 1, (self.pop_size, dim))
        else:
            normalized_fireflies = np.array(initial_population, dtype=float)
            if normalized_fireflies.shape != (self.pop_size, dim):
                raise ValueError(f'initial population of shape {normalized_fireflies.shape}, '
                                 f'expected {(self.pop_size, dim)}')
        # inverse of min-max normalization
        fireflies = normalized_fireflies*(ub-lb) + lb
        # Task 4: a resumed run continues from its last checkpoint; its config.json stands.
        if resume:
            check_progress_table(result_path)  # a checkpoint, and its table: else NOT_RESUMABLE
        checkpoint = load_last_checkpoint(result_path) if resume else None
        # Task 3: recorded before any evaluation, so a run that fails early still has it.
        g_plan = max_generations if self.g_plan is None else self.g_plan
        if checkpoint is not None:
            # Task 8: a resume moves on the schedule config.json records, which describes the
            # run; one extended to more generations keeps shrinking over the plan it started
            # with.
            recorded = recorded_schedule(result_path)
            g_plan = recorded[2] if self.g_plan is None else g_plan
            if (self.alpha_0, self.alpha_end, g_plan) != recorded:
                not_resumable(f'{result_path} cannot be resumed with the step schedule '
                              f'{(self.alpha_0, self.alpha_end, g_plan)}: it started with '
                              f'{recorded}')
        step_size(0, self.alpha_0, self.alpha_end, g_plan)  # refuses a schedule out of range
        settings = {'pop_size': self.pop_size, 'alpha_0': self.alpha_0,
                    'alpha_end': self.alpha_end, 'g_plan': g_plan, 'betamin': self.betamin,
                    'gamma': self.gamma, 'max_generations': max_generations, 'dimensions': dim,
                    'simulations': self.simulations, 'stage_timing': self.stage_timing,
                    **self.run_settings, 'moves': MOVES,  # Task 6: last, so nothing overrides it
                    'stream': self.stream}
        if checkpoint is None:
            runs.write_config(result_path, settings, rng_seed=self.seed, workers=max_workers,
                              initial_population=fireflies)
        # Create ONE persistent process pool reused for every evaluation in this run.
        # (Previously each evaluation created and tore down its own pool, re-spawning
        # 24 workers thousands of times.)
        #
        # Task 6: one task per candidate runs function(vector, seed_offset) in a worker, which
        # init_worker has given the read-only inputs once. The workers start by spawn on every
        # system, so macOS and Linux behave the same.
        from covsyn.calibration.fast_cost import init_worker as _init_reduction_worker
        _taiwan = np.load('./variable/Taiwan_data_matrix.npy')
        if self.profile_dir is not None:
            Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        intensities = np.empty(self.pop_size)
        cost_parts = [{} for _ in range(self.pop_size)]
        # Task 5: one row per evaluation and per generation, beside progress.csv.
        tables = runs.progress_path(result_path).parent
        evaluation_table = timings.Table(tables / timings.EVALUATION_TABLE,
                                         timings.EVALUATION_COLUMNS)
        generation_table = timings.Table(tables / timings.GENERATION_TABLE,
                                         timings.GENERATION_COLUMNS)
        progress_table = timings.Table(runs.progress_path(result_path))  # Tables open lazily
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_reduction_worker,
            initargs=(demographic_parameters, _taiwan.shape[1], self.stage_timing,
                      self.profile_dir, (Cheng_contact_array, Cheng_attack_rate, norm_weights),
                      self.simulations, _taiwan))
        failed = False
        try:  # the pool and the tables are closed however the run ends
            clock = timings.GenerationClock(-1, max_workers)

            def evaluate(candidates, vectors, offset):
                """Score the candidates at once, one task each, on the block from offset.

                Their costs and parts go to intensities and cost_parts by candidate index, and
                their rows to the evaluation table in that order, whatever order the tasks end
                in (Task 6).
                """
                submitted = {}
                for k in candidates:
                    dispatching = time.perf_counter_ns()
                    future = executor.submit(function, vectors[k], offset)
                    submitted[k] = (future, time.time(),
                                    timings.seconds(time.perf_counter_ns() - dispatching))
                for k, (future, start, dispatch_s) in submitted.items():
                    waiting = time.perf_counter_ns()
                    intensities[k], parts, timing = future.result()
                    cost_parts[k] = dict(parts)
                    timing = {**timing, 'dispatch_s': dispatch_s,
                              'wait_s': timings.seconds(time.perf_counter_ns() - waiting)}
                    # The task's own start and end, as the worker measured them; an objective
                    # that gives none gets the submission and the collection of its result.
                    timing.setdefault('start_unix_s', round(start, 6))
                    timing.setdefault('end_unix_s', round(time.time(), 6))
                    row = {'generation': clock.generation, 'candidate': k, **timing}
                    row.pop('worker_rss', None)  # per-PID RSS for the generation row, not a column
                    evaluation_table.write(row)
                    clock.add_evaluation(timing)

            def finish_generation(state):
                """Checkpoint the generation, then write its timing row."""
                started = time.perf_counter_ns()
                save_checkpoint(result_path, clock.generation, state)
                clock.checkpoint_ns += time.perf_counter_ns() - started
                generation_table.write(clock.row())

            if checkpoint is not None:
                print(f'resuming after generation {int(checkpoint["generation"])}', flush=True)
            else:
                print(f'evaluating initial population: {self.pop_size} fireflies '
                      f'({self.simulations} simulations each)...', flush=True)
                evaluate(range(self.pop_size), fireflies,
                         generation_seed_offset(0, self.simulations, self.stream))
                for k in range(self.pop_size):
                    print(f'  initial firefly {k+1}/{self.pop_size}: cost={intensities[k]:.4f}',
                          flush=True)
            print(f'initial population evaluated: best_cost={np.min(intensities):.4f}', flush=True)
            # B49 / E78: the initial population counts. It used to start from 10e10 and was only
            # updated after a firefly MOVED, so an initial point -- a warm start of 2.49 in the test
            # of run 9 -- was never recorded as the best.
            # B81: no running minimum over generations -- see generation_seed_offset. The archive
            # carries positions from across the search into the candidate pool instead; Task 8:
            # every elite vector, with its mean over the blocks it was scored on there.
            elite_records = no_elite_records(dim)
            worst_fireflies = np.zeros(np.shape(fireflies))
            worst_intensities = np.zeros(np.shape(intensities))
            worst_iteration = np.zeros(np.shape(intensities))
            evaluations = self.pop_size
            new_alpha = step_size(0, self.alpha_0, self.alpha_end, g_plan)
            start = 0
            if checkpoint is None:
                # Save the first random guess
                result = np.hstack((fireflies, np.matrix(intensities).T))
                np.savetxt(result_path/'firefly_result_first_initial_guess.txt',
                           result, fmt='%.7f')
                finish_generation(self._state(
                    normalized_fireflies, fireflies, intensities, cost_parts, elite_records,
                    worst_fireflies, worst_intensities, worst_iteration, evaluations, new_alpha,
                    dim))
            else:
                normalized_fireflies = checkpoint['normalized'].copy()
                fireflies = checkpoint['fireflies'].copy()
                intensities = checkpoint['intensities'].copy()
                cost_parts = json.loads(str(checkpoint['cost_parts']))
                elite_records = checkpoint['elite_records'].copy()
                worst_fireflies = checkpoint['worst_fireflies'].copy()
                worst_intensities = checkpoint['worst_intensities'].copy()
                worst_iteration = checkpoint['worst_iteration'].copy()
                evaluations = int(checkpoint['evaluations'])
                start = int(checkpoint['generation']) + 1
                for table in (progress_table, evaluation_table, generation_table):
                    drop_progress_after(table.path, start - 1)

            # Task 8: the elite the next generation keeps, from the costs it starts with; each
            # generation's costs choose the one after, so it is chosen once.
            elite = elite_members(intensities, elite_size(self.pop_size))
            for generation in tqdm(range(start, max_generations)):
                # B81: this generation's block, and every firefly scored on it -- the brightest
                # included, since its score came from another block and cannot be compared here.
                offset = generation_seed_offset(generation, self.simulations, self.stream)
                clock = timings.GenerationClock(generation, max_workers)
                new_alpha = step_size(generation, self.alpha_0, self.alpha_end, g_plan)  # Task 8
                # Task 6: every firefly moves on the positions and costs the generation starts
                # from, then all are scored at once. B49 / E78: the brightest stays where it is;
                # Task 8: so do the rest of the k lowest, all re-scored on this block.
                moving = time.perf_counter_ns()
                normalized_fireflies = synchronous_moves(
                    normalized_fireflies, intensities, new_alpha, self.betamin, self.gamma,
                    generation_rng(self.seed, generation), fixed=elite)  # Task 4
                fireflies = normalized_fireflies*(ub-lb) + lb
                clock.moves_ns += time.perf_counter_ns() - moving
                # Generation 0's block is the initial one, on which the elite is already scored
                # (F7); every other candidate is scored on this generation's block.
                candidates = [k for k in range(self.pop_size)
                              if generation > 0 or k not in elite]
                print(f'  gen {generation} | scoring {len(candidates)} fireflies | '
                      f'evals {evaluations}', flush=True)
                evaluate(candidates, fireflies, offset)
                for k in candidates:  # in candidate order, so the counts are the same however run
                    evaluations += 1
                    if intensities[k] > worst_intensities[k]:
                        worst_fireflies[k] = fireflies[k]
                        worst_intensities[k] = intensities[k]
                        worst_iteration[k] = evaluations
                # Per-generation progress for live monitoring: lower best_cost = better fit.
                # B81: this generation's own block, so the column is not a convergence curve --
                # a rise can be the block, not the search.
                print(f'generation {generation}: best_cost={np.min(intensities):.4f} '
                      f'offset={offset} evals={evaluations}', flush=True)

                # Append this generation's best solution's decoded metrics to a CSV so the
                # run can be plotted/inspected live (and aborted early if values look wrong).
                best_idx = int(np.argmin(intensities))
                # Task 8: the members kept in place, and those this block puts in the next
                # elite, each with its cost on this block; generation 0's kept members their
                # block-0 costs, counted here only.
                next_elite = elite_members(intensities, elite_size(self.pop_size))
                elite_records = record_elite(elite_records, fireflies, intensities,
                                             set(elite) | set(next_elite), generation)
                elite = next_elite
                row = {'generation': generation, 'seed_offset': offset,
                       'best_cost': float(np.min(intensities))}
                row.update(cost_parts[best_idx])
                row.update(decode_metrics(fireflies[best_idx]))
                # Task 4: results/tables/<run-id>/progress.csv. A new run starts it, and a resume
                # has trimmed it to its checkpoint; rows go under its header (Task 5).
                progress_table.write(row)
                # Task 4: after the progress row, so a resumed run drops a row written past it.
                # Task 5: the timing row follows the checkpoint, whose time it records; a kill
                # between the two leaves that generation without one.
                finish_generation(self._state(
                    normalized_fireflies, fireflies, intensities, cost_parts, elite_records,
                    worst_fireflies, worst_intensities, worst_iteration, evaluations, new_alpha,
                    dim))

            executor.shutdown()  # the workers write their profiles as they exit
        except BaseException:
            # A failed or stopped run cancels the tasks not yet started and waits for the rest:
            # those running and those already queued, at most 2 x workers + 1 tasks of one
            # candidate each (S simulations, about 0.7 s at S = 300 on WSL, if each simulation
            # ends: the wait has no timeout), so, unless Ctrl-C is pressed again, no worker
            # outlives the run. A Ctrl-C in a terminal reaches the workers too, and may end
            # their tasks, or break the pool, before those tasks would have finished; the wait
            # then ends sooner.
            # A second Ctrl-C cuts the wait short, as asked; neither it nor any error of the
            # shutdown replaces the run's own error.
            failed = True
            with contextlib.suppress(BaseException):
                executor.shutdown(wait=True, cancel_futures=True)
            raise
        finally:  # whatever ended the run, shutdown included; its own error stays the one raised
            timings.close_tables((progress_table, evaluation_table, generation_table),
                                 failing=failed)
        if self.profile_dir is not None:  # Task 5
            merged = profiling.merge_profiles(self.profile_dir, tables)
            print(f'profile: {merged}, top {profiling.TOP} in {tables / profiling.TOP_TABLE}; '
                  f'run_covid {100 * profiling.share(merged, "run_covid"):.1f}% of the time',
                  flush=True)
        # B81: firefly_best.txt keeps one row per firefly -- reproduce_wu2025 and
        # firefly_txt_to_csv require the four result files to have equal row counts -- so the
        # final population fills it, and the elite vectors go to firefly_archive.txt, which
        # revalidation.load_candidates adds to the candidate pool when it exists (Task 8).
        np.savetxt(result_path/'firefly_archive.txt', archive_rows(elite_records), fmt='%.7f',
                   header=ARCHIVE_HEADER)
        best_iteration = np.full(self.pop_size, float(evaluations))
        return (np.array(fireflies, dtype=float), np.array(intensities, dtype=float),
                best_iteration, worst_fireflies, worst_intensities, worst_iteration,
                fireflies, intensities)


# Each mode's (pop_size, alpha_0, betamin, gamma, max_generations); main() runs these, and a run
# recorded with another mode cannot be continued.
OPTIMISER_MODES = {
    # B51 (E79): gamma was 0.131, but r is the SQUARED distance over all 199 normalised
    # dimensions, whose expectation between two uniform points is d/6 = 33; at the median r of
    # 29.7 that gave beta = 0.02, i.e. no attraction -- run 9 never moved off its warm start in
    # 120 generations. gamma = 6/d makes beta = 1/e = 0.37 at the typical distance and lets it
    # approach 1 as fireflies close in.
    # Task 8: alpha_0 starts the step schedule (step_size); it was the step that shrank by 0.97,
    # from 1.
    'train': (100, DEFAULT_ALPHA_0, 1, 0.03, 120),
    'test': (3, DEFAULT_ALPHA_0, 1, 0.01, 10),
    # Task 5: the population is the worker count, set in main(), and S comes from the command
    # line; a profile is run again, never resumed.
    'profile': (None, DEFAULT_ALPHA_0, 1, 0.01, 2),
}
PROFILE_SIMULATIONS = 3000  # the plan's default S for a profile


def not_resumable(message: str) -> None:
    """Stop with runs.NOT_RESUMABLE: the run cannot be resumed, so retrying cannot help."""
    print(message, file=sys.stderr)
    raise SystemExit(runs.NOT_RESUMABLE)


def resumed_settings(run_dir: Path) -> tuple[int, str, bool]:
    """The seed, mode and stage timing a run recorded in its config.json, to resume it with.

    The mode sets the population size and the generations, which must match the checkpoint
    (Task 4). The stage timing, False when not recorded, keeps the timing table's stage columns
    as they started (Task 5).

    Raises:
        SystemExit: With code runs.NOT_RESUMABLE when retrying cannot help: config.json is
            missing, not a file, unreadable as JSON, or lacks an rng_seed of at least 0 or a mode
            of OPTIMISER_MODES. With a message (status 1) for any other read error, which may
            pass.
    """
    try:
        config = json.loads((Path(run_dir) / 'config.json').read_text(encoding='utf-8'))
        seed, mode = config['rng_seed'], config['settings']['mode']
        stage_timing = config['settings'].get('stage_timing') is True
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError, KeyError,
            TypeError) as error:  # what retrying cannot change
        not_resumable(f'{run_dir} cannot be resumed: no rng_seed and mode in its config.json '
                      f'({error!r})')
    except OSError as error:  # a disk that may come back: retrying can help, so not that code
        raise SystemExit(f'{run_dir}: could not read its config.json ({error!r})')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        not_resumable(f'{run_dir} cannot be resumed: its config.json rng_seed is {seed!r}')
    if not isinstance(mode, str) or mode not in OPTIMISER_MODES:  # a list would not hash
        not_resumable(f'{run_dir} cannot be resumed: its config.json mode is {mode!r}')
    if mode == 'profile':  # Task 5: its population is the worker count of that run
        not_resumable(f'{run_dir} is a profile, which is run again rather than resumed')
    if 'initial_population' not in config['settings']:  # Task 7
        not_resumable(f'{run_dir} was started before the committed initial population; its '
                      f'generations cannot be continued from it')
    if config['settings'].get('moves') != MOVES:  # Task 6
        not_resumable(f'{run_dir} was started with the moves of before Task 6; its generations '
                      f'cannot be continued with the synchronous ones')
    return seed, mode, stage_timing


def recorded_settings(run_dir: Path) -> dict:
    """The settings a run recorded in its config.json, for a resume to read (Task 8).

    Raises:
        SystemExit: With code runs.NOT_RESUMABLE when retrying cannot help: config.json is
            missing, not a file, unreadable as JSON, or has no settings table. With a message
            (status 1) for any other read error, which may pass, as in resumed_settings.
    """
    try:
        settings = json.loads((Path(run_dir) / 'config.json').read_text(encoding='utf-8'))[
            'settings']
        if not isinstance(settings, dict):
            raise TypeError(f'settings is {settings!r}')
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError, KeyError,
            TypeError) as error:  # what retrying cannot change
        not_resumable(f'{run_dir} cannot be resumed: no settings in its config.json ({error!r})')
    except OSError as error:  # a disk that may come back: retrying can help, so not that code
        raise SystemExit(f'{run_dir}: could not read its config.json ({error!r})') from error
    return settings


def recorded_stream(run_dir: Path) -> int:
    """The optimisation stream a run recorded in its config.json, to resume it on (Task 7).

    Raises:
        SystemExit: runs.NOT_RESUMABLE when it is missing or not an optimisation stream; see
            recorded_settings for a config.json that cannot be read.
    """
    try:
        stream = recorded_settings(run_dir)['stream']
    except KeyError as error:
        not_resumable(f'{run_dir} cannot be resumed: no stream in its config.json ({error!r})')
    try:
        return seed_streams.require(stream, seed_streams.OPTIMISATION_STREAMS, 'optimisation')
    except ValueError as error:
        not_resumable(f'{run_dir} cannot be resumed: its config.json stream: {error}')


def check_resumed_bounds(run_dir: Path, lower: np.ndarray, upper: np.ndarray) -> None:
    """Refuse to resume a run under bounds other than those it started with (Task 7).

    Its checkpoints hold positions within the bounds in its bound.txt, which savetxt writes
    exactly; resuming under others would move them in another box.

    Raises:
        SystemExit: runs.NOT_RESUMABLE when bound.txt is missing, unreadable or different.
    """
    path = Path(run_dir) / 'bound.txt'
    if not path.is_file():
        not_resumable(f'{run_dir} cannot be resumed: no bound.txt records the bounds it '
                      f'started with')
    try:
        with warnings.catch_warnings():  # an empty file only warns, and reads as no values
            warnings.simplefilter('error', UserWarning)
            recorded = np.loadtxt(path, ndmin=2)  # rows stay rows, of one value too
    except (OSError, ValueError, UserWarning) as error:
        not_resumable(f'{run_dir} cannot be resumed: its bound.txt cannot be read ({error!r})')
    if len(recorded) != 2:
        not_resumable(f'{run_dir} cannot be resumed: its bound.txt does not hold a lower and an '
                      f'upper row (it holds {len(recorded)})')
    if recorded.shape != (2, len(lower)) or not (np.array_equal(recorded[0], lower)
                                                  and np.array_equal(recorded[1], upper)):
        not_resumable(f'{run_dir} started with other bounds or another dimension than '
                      f'./variable now gives; it cannot be resumed under them')


def positive_float(text: str) -> float:
    """A command-line number greater than 0."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{text!r} is not a number') from None
    if not (value > 0 and np.isfinite(value)):
        raise argparse.ArgumentTypeError(f'{text} is not greater than 0')
    return value


def positive_int(text: str) -> int:
    """A command-line whole number of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{text!r} is not a whole number') from None
    if value < 1:
        raise argparse.ArgumentTypeError(f'{value} is not at least 1')
    return value


def recorded_schedule(run_dir: Path) -> tuple[float, float, int]:
    """The step schedule a run recorded in its config.json, to resume it with (Task 8).

    Raises:
        SystemExit: runs.NOT_RESUMABLE when it is missing, as in a run started before Task 8,
            which moved by a factor of 0.97, or out of range; see recorded_settings for a
            config.json that cannot be read.
    """
    settings = recorded_settings(run_dir)
    try:
        schedule = settings['alpha_0'], settings['alpha_end'], settings['g_plan']
    except KeyError as error:
        not_resumable(f'{run_dir} cannot be resumed: no step schedule in its config.json, as '
                      f'in a run started before Task 8 ({error!r})')
    try:
        step_size(0, *schedule)
    except (ValueError, TypeError) as error:
        not_resumable(f'{run_dir} cannot be resumed: its recorded step schedule {schedule} is '
                      f'out of range ({error})')
    return schedule


def optimisation_stream(text: str) -> int:
    """--stream: one of the optimisation streams, 1 to 99 (Task 7)."""
    try:
        return seed_streams.require(int(text), seed_streams.OPTIMISATION_STREAMS, 'optimisation')
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def resume_arguments(args: argparse.Namespace) -> str:
    """The mode to run in; with --resume, the run's recorded seed and mode (Task 4).

    A resumed run continues as it started, so an option that would change it is refused rather
    than ignored.

    Raises:
        SystemExit: --resume with --run-dir, --seed, --mode, --warm_start, --simulations,
            --stream, --alpha-0, --alpha-end, --g-plan or --stage-timings.
    """
    if args.resume is None:
        return 'train' if args.mode is None else args.mode  # '' is refused as unknown below
    given = [flag for flag, value in (('--run-dir', args.run_dir), ('--seed', args.seed),
                                      ('--mode', args.mode), ('--warm_start', args.warm_start),
                                      ('--simulations', getattr(args, 'simulations', None)),
                                      ('--stream', getattr(args, 'stream', None)),
                                      ('--alpha-0', getattr(args, 'alpha_0', None)),
                                      ('--alpha-end', getattr(args, 'alpha_end', None)),
                                      ('--g-plan', getattr(args, 'g_plan', None)),
                                      ('--stage-timings',
                                       getattr(args, 'stage_timings', False) or None))
             if value is not None]  # --seed 0 and a bare --warm_start are given too
    if given:  # a mistake in the command, reported before anything is read
        sys.exit(f'--resume takes the directory, seed, mode, stream and step schedule from the '
                 f'run; remove {", ".join(given)}')
    # Then the checkpoints: a run killed before any has nothing to continue, whatever its
    # config.json holds, and that is the reason to report.
    problem = runs.resume_problem(args.resume)
    if problem is not None:
        not_resumable(problem)
    args.seed, mode, args.stage_timings = resumed_settings(args.resume)
    args.stream = recorded_stream(args.resume)
    args.alpha_0, args.alpha_end, args.g_plan = recorded_schedule(args.resume)
    return mode


def run_directory_for(run_dir: Path | None, config_name: str) -> Path:
    """The run's directory: ``run_dir`` if given and holding no run, else a new one (Task 3)."""
    if run_dir is not None:
        return runs.claim_run_directory(run_dir)
    return runs.new_run_directory(config_name)


def main() -> None:
    """Run the optimiser from the command line (``covsyn-optimise``)."""
    # Firefly.firefly writes its output to the module-level result_path, which test_b81
    # sets directly; Task 3 of the rework plan replaces it with an explicit run directory.
    global result_path
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Firefly optimization')
    # parser.add_argument('--result_path', type=str, default='./Firefly_result/')
    parser.add_argument('--mode', type=str, default=None,
                        help='train (default), test or profile; a resumed run keeps its own')
    # parser.add_argument('--shift_percentage', type=float, default=0.3)
    hardware.add_workers_argument(parser)  # Task 2: was --max_workers, os.cpu_count() (D3)
    # B49: firefly_best.txt files of earlier runs seeded fireflies 1..; since Task 7 the warm
    # starts are rows of the initial population, and the option is refused with that reason.
    parser.add_argument('--warm_start', type=str, nargs='*', default=None,
                        help=argparse.SUPPRESS)
    # Task 3: the run writes into results/models/<run-id>/; covsyn-pipeline names it with
    # covsyn.runtime.runs and passes it here, so the chain knows where to read.
    parser.add_argument('--run-dir', type=Path, default=None,
                        help='run directory (default: results/models/<new run id>)')
    parser.add_argument('--config-name', default='phase-d',
                        help='configuration name in a new run id (lowercase kebab-case)')
    parser.add_argument('--resume', type=Path, default=None,
                        help='run directory to continue from its last checkpoint (Task 4)')
    parser.add_argument('--seed', type=runs.non_negative_int, default=None,
                        help="seed of the moves, config.json's rng_seed "
                             f'(default {DEFAULT_RNG_SEED})')
    # Task 5
    parser.add_argument('--simulations', type=int, default=None,
                        help=f'profile mode: simulations per evaluation, S '
                             f'(default {PROFILE_SIMULATIONS})')
    parser.add_argument('--stream', type=optimisation_stream, default=None,
                        help='optimisation stream, 1 to 99, whose block g scores generation g '
                             f'(default {seed_streams.DEFAULT_OPTIMISATION_STREAM})')
    # Task 8: the step schedule, which pilot 2 (Task 9) chooses
    parser.add_argument('--alpha-0', type=positive_float, default=None,
                        help=f'step size of generation 0 (default {DEFAULT_ALPHA_0})')
    parser.add_argument('--alpha-end', type=positive_float, default=None,
                        help=f'step size from generation G_plan - 1 on, at most --alpha-0 '
                             f'(default {DEFAULT_ALPHA_END})')
    parser.add_argument('--g-plan', type=positive_int, default=None,
                        help="planned run length G_plan (default: the mode's generations)")
    parser.add_argument('--stage-timings', action='store_true',
                        help="time run_covid's stages into timings_evaluation.csv")
    args = parser.parse_args()
    if args.warm_start is not None and args.resume is None:
        sys.exit('--warm_start is gone (Task 7): the warm starts are rows of the initial '
                 'population; make a new version with covsyn-initial-population to change them')

    cores = hardware.performance_core_count()
    max_workers = hardware.resolve_workers(args.workers, os.environ, None, cores=cores.count)
    print(f'performance cores: {cores.count} ({cores.method}); workers: {max_workers}')
    # result_path = Path(args.result_path)
    mode = resume_arguments(args)  # every resume check, before any input loads
    if args.resume is None and args.run_dir is not None:
        runs.check_unclaimed(args.run_dir)  # fails at startup; the claim comes after the inputs
    print(f'mode: {mode}')
    # shift_percentage = args.shift_percentage
    if mode not in OPTIMISER_MODES:
        # A non-zero status, so the launcher and the chain see the failure.
        sys.exit(f'mode error: unknown --mode {mode!r}')
    pop_size, alpha_0, betamin, gamma, max_generations = OPTIMISER_MODES[mode]
    # Task 5: a profile has n = workers, 2 generations, and S from the command line.
    if mode == 'profile':
        pop_size = max_workers
        simulations = PROFILE_SIMULATIONS if args.simulations is None else args.simulations
        if not 1 <= simulations <= seed_streams.MAX_SIMULATIONS:  # Task 7: one block's seeds
            sys.exit(f'--simulations must be 1 to {seed_streams.MAX_SIMULATIONS}, '
                     f'not {simulations}')
    elif args.simulations is not None:
        sys.exit('--simulations changes the objective, so it is for --mode profile only')
    else:
        simulations = SIMULATIONS_PER_EVALUATION

    # Firefly
    # Task 8: the step schedule, the mode's alpha_0 unless given; a resume's comes from its
    # config.json (resume_arguments).
    fa = Firefly(pop_size, alpha_0 if args.alpha_0 is None else args.alpha_0, betamin, gamma,
                 seed=args.seed,
                 alpha_end=DEFAULT_ALPHA_END if args.alpha_end is None else args.alpha_end,
                 g_plan=max_generations if args.g_plan is None else args.g_plan)
    try:  # Task 8: refused here, before the run directory is claimed
        step_size(0, fa.alpha_0, fa.alpha_end, fa.g_plan)
    except ValueError as error:
        hint = ('; give --alpha-end too, at most --alpha-0'
                if args.alpha_end is None and fa.alpha_end > fa.alpha_0 else '')
        sys.exit(f'step schedule: {error}{hint}')
    fa.run_settings = {'mode': mode}
    fa.simulations = simulations
    fa.stage_timing = args.stage_timings
    fa.stream = (seed_streams.DEFAULT_OPTIMISATION_STREAM if args.stream is None
                 else args.stream)

    # Task 7: the bounds come from one place, which the initial population is made from too.
    space = parameter_space.load()
    lower_bound, upper_bound = space.lower, space.upper
    # Task 7: the first pop_size rows of the committed population, refused, not clipped, when
    # the bounds or the dimension differ from those it was made with. A resumed run continues
    # from its checkpoint and keeps the config.json that names its population.
    population = None
    if args.resume is not None:
        check_resumed_bounds(args.resume, lower_bound, upper_bound)
    else:
        population_file = initial_population.population_path(initial_population.CURRENT_VERSION)
        population = initial_population.load(population_file, pop_size, lower_bound, upper_bound)
        fa.run_settings['initial_population'] = initial_population.describe(population_file)

    # Load demographic data
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demographic_parameters = pickle.load(f)

    # Load Cheng2020 data
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        processed_contact_tracing_data = pickle.load(f)

    Cheng_contact_array = processed_contact_tracing_data['Cheng_contact_array']
    Cheng_attack_rate = processed_contact_tracing_data['Cheng_attack_rate']
    norm_weights = processed_contact_tracing_data['norm_weights']

    # Task 3: claimed only now, after every input has loaded, so a mistyped warm start or a
    # missing input file leaves no claimed directory behind.
    result_path = (args.resume if args.resume is not None
                   else run_directory_for(args.run_dir, args.config_name))
    print(f'run directory: {result_path}')

    # Optimization
    if mode == 'profile':  # Task 5: the workers profile their tasks into the run directory
        fa.profile_dir = result_path / 'profile'
    if args.resume is None:  # a resumed run keeps the bounds it started with (Task 7)
        np.savetxt(result_path/'bound.txt',
                   np.vstack((lower_bound, upper_bound)))
    # E65: fast_cost.cost_function reduces inside the workers -- 1.63x on the steady-state
    # benchmark, on top of the 2.46x the batching gave (run 5 0.1284 s -> run 6 0.0523 s
    # per evaluation). It is verified bit-identical to cost_function() below on nine
    # parameter vectors from two runs; cost_function() stays as the reference the
    # verification compares against, so keep both and re-run verify_fast_cost.py after any
    # change to either.
    # Task 6: each candidate is one task, fast_cost.evaluate_candidate, which runs its S
    # simulations and reduces them in the worker; it is verified bit-identical to
    # fast_cost.cost_function by the same script.
    from covsyn.calibration.fast_cost import evaluate_candidate
    best_fireflies, best_intensities, best_iteration, worst_firefly, worst_intensities, worst_iteration, \
        fireflies, intensities = fa.firefly(
            function=evaluate_candidate, dim=len(lower_bound), lb=lower_bound, ub=upper_bound,
            demographic_parameters=demographic_parameters, max_generations=max_generations, max_workers=max_workers, Cheng_contact_array=Cheng_contact_array,
            Cheng_attack_rate=Cheng_attack_rate, norm_weights=norm_weights,
            initial_population=population,
            resume=args.resume is not None)
    fireflies_result = np.hstack((fireflies, np.matrix(intensities).T))
    best_result = np.hstack(
        (np.matrix(best_iteration).T, best_fireflies, np.matrix(best_intensities).T))
    worst_result = np.hstack(
        (np.matrix(worst_iteration).T, worst_firefly, np.matrix(worst_intensities).T))

    np.savetxt(result_path/'firefly_result.txt',
               fireflies_result, fmt='%.7f')
    np.savetxt(result_path/'firefly_best.txt', best_result, fmt='%.7f')
    np.savetxt(result_path/'firefly_worst.txt', worst_result, fmt='%.7f')
    print("--- Done %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
