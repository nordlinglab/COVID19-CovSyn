import argparse
import concurrent.futures
import copy
import cProfile
import numpy as np
import pickle
import pstats
import sys
import time
import multiprocessing
from pathlib import Path
from numpy.random import default_rng
from scipy.stats import genextreme
from tqdm import tqdm

from Data_synthesis_main import run_covid
from Data_synthesize import *
from plot_results import *
from rw_data_processing import convert_synthetic_data_to_test_matrix
# from sklearn.metrics.pairwise import nan_euclidean_distances
# Note: RW_nan_euclidean_distances is a copy code from sklearn but only comment out the weighting parts.
# "distances /= present_count" and "distances *= X.shape[1]"
from sklearn.metrics.pairwise import *
from sklearn.utils._mask import _get_mask
from sklearn.utils.validation import _deprecate_positional_args

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


def generate_contact_result(results, layer, case_number):
    # Input multiprocess object
    # Load data
    course_of_disease_data_list = np.array([])
    contact_data_list = np.array([])
    for result in results:
        course_of_disease_data_list = np.append(
            course_of_disease_data_list, result.result()[2])
        contact_data_list = np.append(contact_data_list, result.result()[3])

    # Set number of source cases
    number_source_cases = case_number
    course_of_disease_data_list = course_of_disease_data_list[0:number_source_cases]
    contact_data_list = contact_data_list[0:number_source_cases]

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
            course_of_disease_data_list, contact_data_list, layer='Municipality')
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

def cost_function(P, demographic_parameters, max_workers):
    source_case_number = 100
    repeat_number = 1
    seeds = range(source_case_number * repeat_number)

    # start_t = time.time()
    # Parallel processing
    # Multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        seeds_copy = copy.deepcopy(seeds)
        P_copy = copy.deepcopy(P)
        demographic_parameters_copy = copy.deepcopy(demographic_parameters)
        results = [executor.submit(run_covid, seeds_copy[i], P_copy, demographic_parameters_copy, save_file=False)
                   for i in seeds_copy]

    # print('len results', len(results))
    # results = []
    # for i in seeds:
    #     result = run_covid(seeds[i], P, demographic_parameters, save_file=False)
    #     print(result)
    #     results.append(result)

    # Constants
    Cheng_contact_array = np.array([[100, 39, 6, 4, 2, 0],
                                   [236, 150, 38, 17, 110, 146],
                                   [399, 678, 172, 98, 337, 138]])
    max_Cheng_contact = np.max(Cheng_contact_array)
    norm_Cheng_contact_array = Cheng_contact_array/max_Cheng_contact

    Cheng_attack_rate = np.array([[4, 5.1, 16.7, 0, 0, 0],
                                 [0.8, 2, 2.6, 0, 0, 0],
                                 [0, 0, 0.6, 0, 0, 0]])/100
    max_Cheng_attack_rate = np.max(Cheng_attack_rate)
    norm_Cheng_attack_rate = Cheng_attack_rate/max_Cheng_attack_rate

    # weights = np.array([[12.19512195, 6.49350649, 1.87265918, 2.04081633, 1.52207002, 1],
    #                     [35.71428571, 20, 7.69230769,
    #                         5.43478261, 30.3030303, 38.46153846],
    #                     [100, 166.66666667, 31.25, 23.80952381, 90.90909091, 30.3030303]])
    # max_weights = np.max(weights)
    # norm_weights = weights/max_weights
    # weights can be found in plot_compare_previous_studies.ipynb
    norm_weights = np.array([[1, 0.53246753, 0.15355805, 0.16734694, 0.12480974, 0.082],
                             [0.92857143, 0.52, 0.2, 0.14130435, 0.78787879, 1],
                             [0.6, 1, 0.1875, 0.14285714, 0.54545455, 0.18181818]])

    layers = ['Household', 'Health care', 'Cheng others']

    costs = []
    # print('Time spend before loop: ', time.time() - start_t)
    # start_t = time.time()
    for i, layer in enumerate(layers):
        contact_array, infection_array = generate_contact_result(
            results, layer=layer, case_number=source_case_number * repeat_number)
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
        total_cost = cost + attack_rate_cost
        costs.append(total_cost)
    # print('Time spend in loop: ', time.time() - start_t)
    # start_t = time.time()
    # Energy distance
    taiwan_data_matrix = np.load('./variable/Taiwan_data_matrix.npy')
    synthetic_data_matrix = convert_synthetic_data_to_test_matrix(
        results, taiwan_data_matrix, source_case_number * repeat_number)
    _, energy_cost, _ = estat(
        taiwan_data_matrix, synthetic_data_matrix, nboot=100)
    # energy_cost = max(energy_cost, 0)  # Prevent negative energy cost
    # Objective function
    # print('Energy cost: ', energy_cost)
    energy_weight = 1
    total_cost = sum(costs) + energy_weight * energy_cost
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


class Firefly:
    def __init__(self, pop_size, alpha, betamin, gamma, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)

    def firefly(self, function, dim, lb, ub, demographic_parameters, max_generations, max_workers):
        normalized_fireflies = self.rng.uniform(0, 1, (self.pop_size, dim))
        # fireflies = self.rng.uniform(lb, ub, (self.pop_size, dim))
        # inverse of min-max normalization
        fireflies = normalized_fireflies*(ub-lb) + lb
        intensities = np.apply_along_axis(
            function, 1, fireflies, demographic_parameters, max_workers)
        best_fireflies = np.zeros(np.shape(fireflies))
        best_intensities = np.ones(np.shape(intensities))*10e10
        best_iteration = np.zeros(np.shape(intensities))
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
                            fireflies[i], demographic_parameters, max_workers)
                        evaluations += 1
                        print(
                            f'generation {generation}, evaluation {evaluations}')
                        if intensities[i] < best_intensities[i]:
                            best_fireflies[i] = fireflies[i]
                            best_intensities[i] = intensities[i]
                            best_iteration[i] = evaluations
                        if intensities[i] > worst_intensities[i]:
                            worst_fireflies[i] = fireflies[i]
                            worst_intensities[i] = intensities[i]
                            worst_iteration[i] = evaluations
            # print('')

        return (best_fireflies, best_intensities, best_iteration, worst_fireflies, worst_intensities, worst_iteration, fireflies, intensities)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Firefly optimization')
    # parser.add_argument('--result_path', type=str, default='./Firefly_result/')
    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--shift_percentage', type=float, default=0.3)
    parser.add_argument('--max_workers', type=int,
                        default=multiprocessing.cpu_count())
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
        gamma = 0.131
        max_generations = 200
        # max_generations = 150
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

    # Load demographic data
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demographic_parameters = pickle.load(f)

    # Optimization
    if mode == 'profile':
        # Cprofile
        with cProfile.Profile() as pr:
            fa.firefly(function=cost_function, dim=198, lb=lower_bound, ub=upper_bound,
                       demographic_parameters=demographic_parameters,
                       max_generations=max_generations)
        print("--- Done %s seconds ---" % (time.time() - start_time))
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats()
        stats.dump_stats(filename='./profile_results/firefly_profiling.prof')
    else:
        np.savetxt(result_path/'bound.txt',
                   np.vstack((lower_bound, upper_bound)))
        best_fireflies, best_intensities, best_iteration, worst_firefly, worst_intensities, worst_iteration, \
            fireflies, intensities = fa.firefly(
                function=cost_function, dim=198, lb=lower_bound, ub=upper_bound,
                demographic_parameters=demographic_parameters, max_generations=max_generations, max_workers=4)
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
