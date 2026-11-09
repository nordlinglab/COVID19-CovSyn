"""Is the new medical_early_share target reachable inside the bounds?

HANDOVER lesson 2: check that a penalty target can be reached before charging it, or the
optimizer trades it away and the decision ends up inverted (E23, E34 -- the latent target of
4.1-4.5 was unreachable, so the fit went to 3.53, BELOW the literature floor).

Samples the health care contact block P[21:28] uniformly inside its bounds, leaves every other
parameter at the seed, and reports the range of medical_early_share and medical_late_share that
the bounds can actually produce. Also reports whether both targets can be met at once.
"""
import concurrent.futures
import pickle

import numpy as np

from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.data_synthesis_main import run_covid

N_DRAWS = 120
N_SIMS = 120
EARLY_TARGET = fo.OUTCOME_TARGETS['medical_early_share'][:2]
LATE_TARGET = fo.OUTCOME_TARGETS['medical_late_share'][:2]


def shape_of(results):
    early = late = total = 0
    for value in results:
        course_list, contact_list = value[2], value[3]
        if not course_list or not contact_list:
            continue
        course, contact = course_list[0], contact_list[0]
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
    if not total:
        return np.nan, np.nan, 0
    return early / total, late / total, total


def main():
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    contact_parameters = pickle.load(open('./variable/contact_parameters.pkl', 'rb'))
    order = ['household', 'school', 'workplace', 'health_care', 'municipality']
    lb, ub = [], []
    for L in order:
        lb += list(contact_parameters[L + '_lower_bound'])
        ub += list(contact_parameters[L + '_upper_bound'])
    lb += list(contact_parameters['overdispersion_lower_bound'])
    ub += list(contact_parameters['overdispersion_upper_bound'])
    lb = np.array(lb + list(np.load('./variable/course_parameters_lb.npy')))
    ub = np.array(ub + list(np.load('./variable/course_parameters_ub.npy')))
    course = np.load('./variable/course_parameters.npy')
    seed = np.hstack(((lb[:37] + ub[:37]) / 2, course))

    rng = np.random.default_rng(0)
    print('health care contact block bounds (P[21..27]):')
    names = ['contact_p', 'contact_previous_day_p', 'healthy_p', 'symptom_p',
             'steepness', 'symptom_phase', 'recover_phase']
    for j, name in enumerate(names):
        print('  P[%2d] %-24s [%.4g, %.4g]' % (21 + j, name, lb[21 + j], ub[21 + j]))
    print('\ntarget: early in [%.2f, %.2f], late in [%.2f, %.2f]'
          % (EARLY_TARGET[0], EARLY_TARGET[1], LATE_TARGET[0], LATE_TARGET[1]))
    print('Cheng 2020: early 0.554, late 0.367\n')

    ex = concurrent.futures.ProcessPoolExecutor(max_workers=32)
    rows = []
    for draw in range(N_DRAWS):
        P = seed.copy()
        P[21:28] = lb[21:28] + rng.random(7) * (ub[21:28] - lb[21:28])
        futures = [ex.submit(run_covid, s, P, demo, False) for s in range(N_SIMS)]
        early, late, total = shape_of([f.result() for f in futures])
        rows.append((early, late, total, P[21:28].copy()))
    ex.shutdown()

    early = np.array([r[0] for r in rows], dtype=float)
    late = np.array([r[1] for r in rows], dtype=float)
    ok = np.isfinite(early)
    print('%d of %d draws produced medical contacts' % (ok.sum(), N_DRAWS))
    print('medical_early_share  min %.3f  median %.3f  max %.3f'
          % (np.nanmin(early), np.nanmedian(early), np.nanmax(early)))
    print('medical_late_share   min %.3f  median %.3f  max %.3f'
          % (np.nanmin(late), np.nanmedian(late), np.nanmax(late)))

    in_early = (early >= EARLY_TARGET[0]) & (early <= EARLY_TARGET[1])
    in_late = (late >= LATE_TARGET[0]) & (late <= LATE_TARGET[1])
    print('\ndraws inside the early band : %d' % int(np.nansum(in_early)))
    print('draws inside the late band  : %d' % int(np.nansum(in_late)))
    print('draws inside BOTH           : %d' % int(np.nansum(in_early & in_late)))

    if np.nansum(in_early & in_late):
        k = int(np.nanargmax((in_early & in_late) * np.ones_like(early)))
        print('\nan example vector meeting both:')
        for j, name in enumerate(names):
            print('  P[%2d] %-24s %.5g' % (21 + j, name, rows[k][3][j]))
        print('  early %.3f  late %.3f' % (rows[k][0], rows[k][1]))
    else:
        best = int(np.nanargmax(early))
        print('\nNOT REACHABLE by this sampling. Closest on the early share:')
        for j, name in enumerate(names):
            print('  P[%2d] %-24s %.5g' % (21 + j, name, rows[best][3][j]))
        print('  early %.3f  late %.3f' % (rows[best][0], rows[best][1]))


if __name__ == '__main__':
    main()
