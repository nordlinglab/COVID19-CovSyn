"""Can the community layer reach a tail ratio of 8-40 while keeping an all-case median of 3-15?

HANDOVER lesson 2, and findings E23 / E34: charging a target that the bounds cannot reach makes
the optimizer trade it away and can drive the decision backwards. E72 widens the charged tail
ratio to [8, 40] and E73 moves the charged median onto every index case, zeros included, so both
have to be satisfiable at once. Run 6 reached a ratio of 7.35 with a non-zero median of 6, which
is close but not inside, so this samples the parameters that shape the community layer and reports
what the bounds can actually produce.

Varies P[28] (mean community contacts per case), P[30] (municipality healthy-day probability),
P[31]-P[34] (symptomatic probability and the logistic shape) and P[198] (the contact-COUNT
dispersion), leaving every other parameter at the run 7 best fit.
"""
import concurrent.futures
import copy
import glob
import os
import pickle

import numpy as np

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.data_synthesis_main import run_covid

N_DRAWS = 100
N_SIMS = 400
MEDIAN_TARGET = fo.OUTCOME_TARGETS['community_median'][:2]
RATIO_TARGET = fo.OUTCOME_TARGETS['community_tail_ratio'][:2]
VARY = [28, 29, 30, 31, 32, 33, 34, 198]


def shape_of(seeds, P):
    demographic_parameters = fast_cost._WORKER_DEMOGRAPHIC_PARAMETERS
    counts = []
    for seed in seeds:
        _demo, _social, _course, contact_list = run_covid(
            seed, P, copy.deepcopy(demographic_parameters), save_file=False)
        if contact_list:
            counts.append(len(contact_list[0]['municipality_effective_contacts'] or []))
    return counts


def main():
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]
    d = [x for x in sorted(glob.glob('Firefly_result_pop_size_100_*'))
         if os.path.exists(x + '/firefly_best.txt')][0]
    res = np.loadtxt(d + '/firefly_best.txt')
    base = res[int(np.argmin(res[:, -1])), 1:-1]

    bounds = np.loadtxt(d + '/bound.txt')
    lb, ub = bounds[0], bounds[1]
    print('parameters from %s' % d)
    print('charged targets: all-case median %s, tail ratio %s\n' % (MEDIAN_TARGET, RATIO_TARGET))
    for i in VARY:
        print('  P[%3d] in [%.5g, %.5g], run 7 best %.5g' % (i, lb[i], ub[i], base[i]))

    rng = np.random.default_rng(1)
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns))
    rows = []
    for _ in range(N_DRAWS):
        P = base.copy()
        for i in VARY:
            P[i] = lb[i] + rng.random() * (ub[i] - lb[i])
        chunk = max(N_SIMS // 32, 1)
        batches = [list(range(i, min(i + chunk, N_SIMS))) for i in range(0, N_SIMS, chunk)]
        counts = []
        for future in [executor.submit(shape_of, b, P) for b in batches]:
            counts += future.result()
        counts = np.array(counts, dtype=float)
        if not len(counts):
            continue
        median_all = float(np.median(counts))
        nonzero = counts[counts > 0]
        if len(nonzero) < 20 or np.median(nonzero) <= 0:
            continue
        ratio = float(np.percentile(nonzero, 90) / np.median(nonzero))
        rows.append((median_all, ratio, float(np.mean(counts == 0)), P[VARY].copy()))
    executor.shutdown()

    median_all = np.array([r[0] for r in rows])
    ratio = np.array([r[1] for r in rows])
    print('\n%d of %d draws produced a usable community distribution' % (len(rows), N_DRAWS))
    print('all-case median : min %.1f  median %.1f  max %.1f'
          % (median_all.min(), np.median(median_all), median_all.max()))
    print('tail ratio      : min %.2f  median %.2f  max %.2f'
          % (ratio.min(), np.median(ratio), ratio.max()))

    in_median = (median_all >= MEDIAN_TARGET[0]) & (median_all <= MEDIAN_TARGET[1])
    in_ratio = (ratio >= RATIO_TARGET[0]) & (ratio <= RATIO_TARGET[1])
    both = in_median & in_ratio
    print('\ndraws inside the median band : %d' % int(in_median.sum()))
    print('draws inside the ratio band  : %d' % int(in_ratio.sum()))
    print('draws inside BOTH            : %d' % int(both.sum()))

    if both.sum():
        k = int(np.argmax(both))
        print('\nan example reaching both:')
        print('  all-case median %.1f   tail ratio %.2f   zero share %.1f%%'
              % (rows[k][0], rows[k][1], 100 * rows[k][2]))
        for i, value in zip(VARY, rows[k][3]):
            print('  P[%3d] = %.5g' % (i, value))
    else:
        k = int(np.argmax(np.where(in_median, ratio, -np.inf)))
        print('\nNOT REACHED by this sampling. Best draw that keeps the median in band:')
        print('  all-case median %.1f   tail ratio %.2f   zero share %.1f%%'
              % (rows[k][0], rows[k][1], 100 * rows[k][2]))
        print('  -> if this stays below %.1f, the ratio band is too high and would be traded'
              ' away (E23 / E34)' % RATIO_TARGET[0])


if __name__ == '__main__':
    main()
