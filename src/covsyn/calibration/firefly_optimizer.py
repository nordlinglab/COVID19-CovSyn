import argparse
import concurrent.futures
import copy
import cProfile
import csv
import numpy as np
import pickle
import pstats
import sys
import time
import traceback
import multiprocessing
from pathlib import Path
from numpy.random import default_rng
from scipy.stats import genextreme
from tqdm import tqdm

from covsyn.model.data_synthesis_main import run_covid
from covsyn.calibration.sar_anchors import LAYER_CUMULATIVE_SAR, LAYER_INFECTIONS_PER_INDEX
from covsyn.model.contact_measures import contacts_per_day_before_onset
from covsyn.model.data_synthesize import *
from covsyn.figures.plot_results import *
from covsyn.data_processing.rw_data_processing import convert_synthetic_data_to_test_matrix
# from sklearn.metrics.pairwise import nan_euclidean_distances
# Note: RW_nan_euclidean_distances is a copy code from sklearn but only comment out the weighting parts.
# "distances /= present_count" and "distances *= X.shape[1]"
from sklearn.metrics.pairwise import *
from sklearn.utils._mask import _get_mask
from sklearn.utils.validation import _deprecate_positional_args
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# Multivariate test

# This code is modified based on sklearn.metrics.pairwise nan_euclidean_distances function


@_deprecate_positional_args
def rw_nan_euclidean_distances(X, Y=None, *, squared=False,
                               missing_values=np.nan, copy=True):
    """Calculate the euclidean distances in the presence of missing values.

    Compute the euclidean distance between each pair of samples in X and Y,
    where Y=X is assumed if Y=None. When calculating the distance between a
    pair of samples, this formulation ignores feature coordinates with a
    missing value in either sample and scales up the weight of the remaining
    coordinates:

        dist(x,y) = sqrt(weight * sq. distance from present coordinates)
        where,
        weight = Total # of coordinates / # of present coordinates

    For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]``
    is:

        .. math::
            \\sqrt{\\frac{4}{2}((3-1)^2 + (6-5)^2)}

    If all the coordinates are missing or if there are no common present
    coordinates then NaN is returned for that pair.

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
        Make and use a deep copy of X and Y (if Y exists).

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)

    See Also
    --------
    paired_distances : Distances between pairs of elements of X and Y.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import nan_euclidean_distances
    >>> nan = float("NaN")
    >>> X = [[0, 1], [1, nan]]
    >>> nan_euclidean_distances(X, X) # distance between rows of X
    array([[0.        , 1.41421356],
           [1.41421356, 0.        ]])

    >>> # get distance to origin
    >>> nan_euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])

    References
    ----------
    * John K. Dixon, "Pattern Recognition with Partly Missing Data",
      IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
      10, pp. 617 - 621, Oct. 1979.
      http://ieeexplore.ieee.org/abstract/document/4310090/
    """

    force_all_finite = 'allow-nan' if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 force_all_finite=force_all_finite, copy=copy)
    # Get missing mask for X
    missing_X = _get_mask(X, missing_values)

    # Get missing mask for Y
    missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)

    # set missing values to zero
    X[missing_X] = 0
    Y[missing_Y] = 0

    distances = euclidean_distances(X, Y, squared=True)

    # Adjust distances for missing values
    XX = X * X
    YY = Y * Y
    distances -= np.dot(XX, missing_Y.T)
    distances -= np.dot(missing_X, YY.T)

    np.clip(distances, 0, None, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        np.fill_diagonal(distances, 0.0)

    present_X = 1 - missing_X
    present_Y = present_X if Y is X else ~missing_Y
    present_count = np.dot(present_X, present_Y.T)
    distances[present_count == 0] = np.nan
    # avoid divide by zero
    np.maximum(1, present_count, out=present_count)
    # distances /= present_count
    # distances *= X.shape[1]

    if not squared:
        np.sqrt(distances, out=distances)

    return distances


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
    stack = np.vstack([x, y])
    stack = (stack - np.nanmean(stack, axis=0))/np.nanstd(stack, axis=0)
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
# of the objective is driving the fit (see progress_metrics.csv).
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
# Weight 0 means the quantity is measured and written to progress_metrics.csv but not
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
    # B25: close contacts per day BEFORE symptom onset, from the 2020 national survey
    # (6.03 contacts/day) split by the setting composition of Fu 2012.
    'daily_household':        (1.50, 2.20, 1.0),
    'daily_school':           (0.90, 1.40, 1.0),
    'daily_workplace':        (1.20, 1.80, 1.0),
    'daily_municipality':     (1.20, 1.80, 1.0),
    # Health care is not one of the survey's everyday settings; a healthy person does not
    # meet medical staff daily, so this one is bounded from above only.
    'daily_health_care':      (0.00, 0.30, 1.0),
    # B33: the severity cascade recomputed from the Taiwan contact-tracing file itself
    # (579 cases): asymptomatic 23.7%, symptomatic->ICU 12.7%, ICU->death 12.5%, CFR 1.2%.
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
    'community_median':       (3.0, 15.0, 1.0),
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
    'community_tail_ratio':   (5.5, 93.1, 1.0),
    # E67, 2026-09-27: run 6 finally grew the community tail (p90 13 -> 44.1, max 48 ->
    # 196) and paid for it with the centre -- the median fell 3 -> 0, i.e. more than half
    # the index cases stopped having any community contact at all. With the attack rate
    # capped by B38 the only remaining source of a heavy tail is the contact COUNT
    # dispersion P[198], and the cheapest way to raise that is to zero most cases out.
    # community_median is charged and should now bite (E66 rescaling), so this is
    # reported rather than charged: it is the number that says whether the median moved
    # because the distribution improved or because the zeros were merely reshuffled.
    'community_zero_share':   (0.0, 0.30, 0.0),
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
    'pre_onset_zero_share':   (0.0, 0.12, 1.0),
    # B28: date_of_recovery is the day the case is CLOSED (released from isolation), which
    # Taiwan's records put at about 25 days from onset, not a clinical recovery at 14-20.
    'closure_symptomatic':    (20.0, 32.0, 1.0),
    'closure_asymptomatic':   (20.0, 32.0, 1.0),
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

    index_cases is a list of (course_of_disease_dict, contact_dict, demographic_dict).
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
    mean, var = offspring.mean(), offspring.var(ddof=1) if len(offspring) > 1 else 0.0
    if var > mean > 0:
        measured['offspring_k'] = mean ** 2 / (var - mean)
    elif mean > 0:
        measured['offspring_k'] = 100.0        # Poisson or tighter: no overdispersion at all

    for layer in layers:
        # B55: ordinary contacts only, see contact_measures.contacts_per_day_before_onset.
        per_day = [contacts_per_day_before_onset(course, contact, layer)
                   for course, contact, _ in index_cases]
        measured[f'daily_{layer}'] = float(np.mean(per_day))

    for layer in layers:
        candidate = effective = 0
        for _, contact, _ in index_cases:
            eff = contact[f'{layer}_effective_contacts']
            eff = [] if eff is None else list(eff)
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
        measured['pre_onset_zero_share'] = float(np.nanmean(pre_onset == 0))

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

    closure = np.array([c['date_of_recovery'] - c['infection_day'] for c, _, _ in index_cases],
                       dtype=float)
    if np.isfinite(closure[symptomatic]).any():
        measured['closure_symptomatic'] = float(np.nanmean(closure[symptomatic]))
    if np.isfinite(closure[~symptomatic]).any():
        measured['closure_asymptomatic'] = float(np.nanmean(closure[~symptomatic]))
    return measured


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
OUTCOME_SCALE_BY_LOWER_BOUND = {'community_tail_ratio'}


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
        miss = max(0.0, lo - x, x - hi) / scale
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


def cost_function(P, demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate, norm_weights):
    try:
        return _cost_function(P, demographic_parameters, executor, Cheng_contact_array,
                              Cheng_attack_rate, norm_weights)
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


def _cost_function(P, demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate, norm_weights):
    source_case_number = SIMULATIONS_PER_EVALUATION
    repeat_number = 1
    seeds = range(source_case_number * repeat_number)

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
                (((norm_contact_array / repeat_number -
                 norm_Cheng_data)*health_care_weights) ** 2))
        else:
            cost = np.sum(
                ((norm_contact_array / repeat_number - norm_Cheng_data) ** 2))
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
    _, energy_cost, _ = estat(
        taiwan_data_matrix, synthetic_data_matrix, nboot=1)
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
        _, _, course_list, contact_list = result.result()
        if course_list and contact_list:
            index_cases.append((course_list[0], contact_list[0], None))
    measured = measure_outcomes(index_cases)
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


def pad_pre_b54_vector(vector: np.ndarray, seed_vector: np.ndarray) -> np.ndarray:
    """Extend a pre-B54 warm-start vector with the seed values of the appended parameters.

    Args:
        vector: A best vector read from an earlier run.
        seed_vector: The literature seed vector of this run.

    Returns:
        The vector with seed_vector[199:] appended when it has exactly the 199 values of
        run 10 and earlier; otherwise the vector unchanged, so a misaligned vector of any
        other length still fails the caller's shape check instead of being padded into a
        plausible-looking start.
    """
    if vector.size == COMMUNITY_EVENT_FIRST_INDEX < seed_vector.size:
        return np.concatenate([vector, seed_vector[vector.size:]])
    return vector


class Firefly:
    def __init__(self, pop_size, alpha, betamin, gamma, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)

    def firefly(self, function, dim, lb, ub, demographic_parameters, max_generations, max_workers, Cheng_contact_array, Cheng_attack_rate, norm_weights, seed_vector=None, warm_start_vectors=()):
        normalized_fireflies = self.rng.uniform(0, 1, (self.pop_size, dim))
        # fireflies = self.rng.uniform(lb, ub, (self.pop_size, dim))
        # inverse of min-max normalization
        fireflies = normalized_fireflies*(ub-lb) + lb
        # Seed firefly 0 at the literature central estimate instead of leaving the whole
        # population random, so the run always contains the documented starting point.
        if seed_vector is not None:
            span = np.where((ub - lb) > 0, ub - lb, 1.0)
            normalized_fireflies[0] = np.clip((np.clip(seed_vector, lb, ub) - lb) / span, 0, 1)
            fireflies[0] = normalized_fireflies[0]*(ub-lb) + lb
        # B49 (E75): fireflies 1.. start at the best vectors of earlier runs. Every run used to
        # restart from the literature seed, and run 8 finished at 3.80 while run 7's best vector
        # scores 2.75 on run 8's own objective -- a known better point the search never found.
        span = np.where((ub - lb) > 0, ub - lb, 1.0)
        for k, vector in enumerate(warm_start_vectors, start=1):
            if k >= self.pop_size:
                break
            normalized_fireflies[k] = np.clip((np.clip(vector, lb, ub) - lb) / span, 0, 1)
            fireflies[k] = normalized_fireflies[k]*(ub-lb) + lb
            print(f'warm start: firefly {k} set from an earlier best vector', flush=True)
        # Create ONE persistent process pool reused for every evaluation in this run.
        # (Previously each evaluation created and tore down its own pool, re-spawning
        # 24 workers thousands of times.)
        #
        # The initializer hands each worker the read-only data once instead of once per task,
        # which is what fast_cost's worker-side reduction needs (E65). It sets globals the
        # plain cost_function() below never reads, so a run using that instead is unaffected.
        from covsyn.calibration.fast_cost import init_worker as _init_reduction_worker
        _columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_reduction_worker,
            initargs=(demographic_parameters, _columns))
        print(f'evaluating initial population: {self.pop_size} fireflies '
              f'(100 simulations each)...', flush=True)
        intensities = np.empty(self.pop_size)
        cost_parts = [{} for _ in range(self.pop_size)]
        best_parts = [{} for _ in range(self.pop_size)]
        for k in range(self.pop_size):
            intensities[k] = function(
                fireflies[k], demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate, norm_weights)
            cost_parts[k] = dict(LAST_COST_PARTS)
            print(f'  initial firefly {k+1}/{self.pop_size}: cost={intensities[k]:.4f}', flush=True)
        print(f'initial population evaluated: best_cost={np.min(intensities):.4f}', flush=True)
        # B49 / E78: the initial population counts. It used to start from 10e10 and was only
        # updated after a firefly MOVED, so an initial point -- a warm start of 2.49 in the test
        # of run 9 -- was never recorded as the best.
        best_fireflies = np.array(fireflies, dtype=float)
        best_intensities = np.array(intensities, dtype=float)
        best_iteration = np.arange(1, self.pop_size + 1, dtype=float)
        best_parts = [dict(parts) for parts in cost_parts]
        worst_fireflies = np.zeros(np.shape(fireflies))
        worst_intensities = np.zeros(np.shape(intensities))
        worst_iteration = np.zeros(np.shape(intensities))
        # Save the first random guess
        result = np.hstack((fireflies, np.matrix(intensities).T))
        np.savetxt(result_path/'firefly_result_first_initial_guess.txt',
                   result, fmt='%.7f')

        evaluations = self.pop_size
        new_alpha = self.alpha
        # search_range = ub - lb

        for generation in tqdm(range(max_generations)):
            new_alpha *= 0.97
            for i in range(self.pop_size):
                if i % 10 == 0:
                    print(f'  gen {generation} | firefly {i}/{self.pop_size} | '
                          f'evals {evaluations} | best {np.min(intensities):.4f}', flush=True)
                # B49 / E78: the brightest firefly stays where it is. The move condition below
                # is >=, which includes j == i, so every firefly -- the best one too -- took a
                # random step of up to new_alpha / 2 of the whole range every generation and the
                # population never kept its best point. Only the current best is exempt; every
                # other firefly moves exactly as before.
                if i == int(np.argmin(intensities)):
                    continue
                for j in range(self.pop_size):
                    # print('i, j: ', i, j)
                    if intensities[i] >= intensities[j]:
                        r = np.sum(
                            np.square(normalized_fireflies[i] - normalized_fireflies[j]), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * r)
                        # steps = new_alpha * \
                        #     (self.rng.random(dim) - 0.5) * search_range
                        steps = new_alpha*(self.rng.random(dim) - 0.5)
                        normalized_fireflies[i] += beta * \
                            (normalized_fireflies[j] -
                             normalized_fireflies[i]) + steps
                        # fireflies[i] = np.clip(fireflies[i], lb, ub)
                        normalized_fireflies[i] = np.clip(
                            normalized_fireflies[i], 0, 1)
                        fireflies[i] = normalized_fireflies[i]*(ub-lb) + lb
                        # print(fireflies[i])
                        intensities[i] = function(
                            fireflies[i], demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate, norm_weights)
                        evaluations += 1
                        # (per-evaluation print removed; see per-generation summary below)
                        cost_parts[i] = dict(LAST_COST_PARTS)
                        if intensities[i] < best_intensities[i]:
                            best_fireflies[i] = fireflies[i]
                            best_intensities[i] = intensities[i]
                            best_iteration[i] = evaluations
                            best_parts[i] = dict(cost_parts[i])
                        if intensities[i] > worst_intensities[i]:
                            worst_fireflies[i] = fireflies[i]
                            worst_intensities[i] = intensities[i]
                            worst_iteration[i] = evaluations
            # Per-generation progress for live monitoring: lower best_cost = better fit
            print(f'generation {generation}: best_cost={np.min(best_intensities):.4f} '
                  f'pop_best={np.min(intensities):.4f} evals={evaluations}', flush=True)

            # Append the current global-best solution's decoded metrics to a CSV so the
            # run can be plotted/inspected live (and aborted early if values look wrong).
            best_idx = int(np.argmin(best_intensities))
            row = {'generation': generation,
                   'best_cost': float(np.min(best_intensities))}
            row.update(best_parts[best_idx])
            row.update(decode_metrics(best_fireflies[best_idx]))
            csv_path = result_path / 'progress_metrics.csv'
            with open(csv_path, 'w' if generation == 0 else 'a', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=list(row.keys()))
                if generation == 0:
                    writer.writeheader()
                writer.writerow(row)

        executor.shutdown()
        return (best_fireflies, best_intensities, best_iteration, worst_fireflies, worst_intensities, worst_iteration, fireflies, intensities)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Firefly optimization')
    # parser.add_argument('--result_path', type=str, default='./Firefly_result/')
    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--shift_percentage', type=float, default=0.3)
    parser.add_argument('--max_workers', type=int,
                        default=multiprocessing.cpu_count())
    # B49: firefly_best.txt files of earlier runs; each one's lowest-cost row seeds a firefly.
    parser.add_argument('--warm_start', type=str, nargs='*', default=[])
    args = parser.parse_args()

    max_workers = args.max_workers
    print('Available CPU cores (max_workers): ', multiprocessing.cpu_count())
    print('set max_workers: ', max_workers)
    # result_path = Path(args.result_path)
    mode = args.mode
    print(f'mode: {mode}')
    # shift_percentage = args.shift_percentage
    if mode == 'train':
        # pop_size = 50
        pop_size = 100
        alpha = 1
        betamin = 1
        # B51 (E79): gamma was 0.131, but r below is the SQUARED distance over all 199
        # normalised dimensions, whose expectation between two uniform points is d/6 = 33; at
        # the median r of 29.7 that gave beta = 0.02, i.e. no attraction -- run 9 never moved
        # off its warm start in 120 generations. gamma = 6/d makes beta = 1/e = 0.37 at the
        # typical distance and lets it approach 1 as fireflies close in.
        gamma = 0.03
        max_generations = 120
        # max_generations = 100
    elif mode == 'test':
        pop_size = 3
        alpha = 1
        betamin = 1
        gamma = 0.01
        max_generations = 10
    elif mode == 'profile':
        pop_size = 10
        alpha = 1
        betamin = 1
        gamma = 0.01
        max_generations = 3
    else:
        print('mode error')
        sys.exit()
    result_path = Path(
        f'./Firefly_result_pop_size_{pop_size}_alpha_{alpha}_betamin_{betamin}_gamma_{gamma}_max_generations_{max_generations}')
    if not result_path.is_dir():
        result_path.mkdir(parents=True)

    # Firefly
    fa = Firefly(pop_size, alpha, betamin, gamma)

    with open('./variable/contact_parameters.pkl', 'rb') as f:
        contact_parameters = pickle.load(f)
    household_lower_bound = contact_parameters['household_lower_bound']
    household_upper_bound = contact_parameters['household_upper_bound']
    school_lower_bound = contact_parameters['school_lower_bound']
    school_upper_bound = contact_parameters['school_upper_bound']
    workplace_lower_bound = contact_parameters['workplace_lower_bound']
    workplace_upper_bound = contact_parameters['workplace_upper_bound']
    health_care_lower_bound = contact_parameters['health_care_lower_bound']
    health_care_upper_bound = contact_parameters['health_care_upper_bound']
    municipality_lower_bound = contact_parameters['municipality_lower_bound']
    municipality_upper_bound = contact_parameters['municipality_upper_bound']
    overdispersion_lower_bound = contact_parameters['overdispersion_lower_bound']
    overdispersion_upper_bound = contact_parameters['overdispersion_upper_bound']

    lower_bound = np.array(household_lower_bound + school_lower_bound +
                           workplace_lower_bound + health_care_lower_bound + municipality_lower_bound +
                           overdispersion_lower_bound)
    upper_bound = np.array(household_upper_bound + school_upper_bound +
                           workplace_upper_bound + health_care_upper_bound + municipality_upper_bound +
                           overdispersion_upper_bound)

    # Load course of disease data
    course_parameters = np.load(
        './variable/course_parameters.npy')
    course_parameters_lb = np.load('./variable/course_parameters_lb.npy')
    course_parameters_ub = np.load('./variable/course_parameters_ub.npy')
    # Extend lb and hb to include the lower and upper bound of the course of disease
    lower_bound = np.hstack(
        (lower_bound, course_parameters_lb))
    # lower_bound[lower_bound < 0] = 0
    upper_bound = np.hstack(
        (upper_bound, course_parameters_ub))
    # upper_bound_tmp = upper_bound[65::]  # upper limit for probability
    # upper_bound_tmp[upper_bound_tmp > 1] = 1
    # upper_bound[65::] = upper_bound_tmp

    # Literature central estimate for firefly 0: documented values for the course of
    # disease block, midpoint of the bounds for the contact block (which has no central
    # estimate stored).
    seed_vector = np.hstack(((lower_bound[:37] + upper_bound[:37]) / 2, course_parameters))

    warm_start_vectors = []
    for path in args.warm_start:
        earlier = np.atleast_2d(np.loadtxt(path))
        vector = earlier[int(np.argmin(earlier[:, -1])), 1:-1]
        padded = pad_pre_b54_vector(vector, seed_vector)
        if padded.shape != lower_bound.shape:
            sys.exit(f'warm start {path}: {vector.size} parameters, expected {lower_bound.size}')
        warm_start_vectors.append(padded)
        print(f'warm start from {path}: recorded cost {earlier[:, -1].min():.4f}'
              + (f' for its {vector.size} parameters; P[{vector.size}:] from the seed vector'
                 if padded.size != vector.size else ''), flush=True)

    # Load demographic data
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demographic_parameters = pickle.load(f)

    # Load Cheng2020 data
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        processed_contact_tracing_data = pickle.load(f)

    Cheng_contact_array = processed_contact_tracing_data['Cheng_contact_array']
    Cheng_attack_rate = processed_contact_tracing_data['Cheng_attack_rate']
    norm_weights = processed_contact_tracing_data['norm_weights']

    # Optimization
    if mode == 'profile':
        # Cprofile
        with cProfile.Profile() as pr:
            fa.firefly(function=cost_function, dim=len(lower_bound), lb=lower_bound, ub=upper_bound,
                       demographic_parameters=demographic_parameters,
                       max_generations=max_generations, 
                       Cheng_contact_array=Cheng_contact_array,
                       Cheng_attack_rate=Cheng_attack_rate,
                       norm_weights=norm_weights)
        print("--- Done %s seconds ---" % (time.time() - start_time))
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats()
        stats.dump_stats(filename='./profile_results/firefly_profiling.prof')
    else:
        np.savetxt(result_path/'bound.txt',
                   np.vstack((lower_bound, upper_bound)))
        # E65: fast_cost.cost_function reduces inside the workers -- 1.63x on the steady-state
        # benchmark, on top of the 2.46x the batching gave (run 5 0.1284 s -> run 6 0.0523 s
        # per evaluation). It is verified bit-identical to cost_function() below on nine
        # parameter vectors from two runs; cost_function() stays as the reference the
        # verification compares against, so keep both and re-run verify_fast_cost.py after any
        # change to either.
        from covsyn.calibration.fast_cost import cost_function as reducing_cost_function
        best_fireflies, best_intensities, best_iteration, worst_firefly, worst_intensities, worst_iteration, \
            fireflies, intensities = fa.firefly(
                function=reducing_cost_function, dim=len(lower_bound), lb=lower_bound, ub=upper_bound,
                demographic_parameters=demographic_parameters, max_generations=max_generations, max_workers=max_workers, Cheng_contact_array=Cheng_contact_array,
                Cheng_attack_rate=Cheng_attack_rate, norm_weights=norm_weights,
                seed_vector=seed_vector, warm_start_vectors=warm_start_vectors)
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
