# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Is B76's floor on health care contacts reachable inside the bounds?

HANDOVER lesson 2: check that a penalty target can be reached before charging it. Starts from
a parameter vector (run 18 candidate 80 by default), draws the health care contact block
P[21:28] uniformly inside its bounds, and measures every index case's outcomes with the
objective's own measure_outcomes() ('result' mode, as the calibration runs). Reports how many
draws meet the floor together with the other charged health care contact targets.

Uniform draws over seven dimensions rarely land on the timing shape (B40 found 1 in 120), so
the grid mode instead scales the contact probability P[21] and the healthy-day probability
P[23] of the base vector up to their ceilings and keeps the shape parameters P[25:28] fixed.

Usage: python scripts/probes/probe_health_care_floor.py [parameter_dir] [draws] [simulations]
       python scripts/probes/probe_health_care_floor.py [parameter_dir] grid [simulations]
"""
import concurrent.futures
import pickle
import sys
from pathlib import Path

import numpy as np

from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.data_synthesis_main import run_covid

BLOCK = slice(21, 28)
NAMES = ['contact_p', 'contact_previous_day_p', 'healthy_p', 'symptom_p',
         'steepness', 'symptom_phase', 'recover_phase']
KEYS = ['health_care_contacts_traced', 'daily_health_care', 'medical_early_share',
        'medical_late_share', 'infections_per_index_health_care']


def load_vector(directory):
    """The lowest-cost row of firefly_best.txt, as data_synthesis_main reads it."""
    rows = np.array([[float(x) for x in line.split(' ')]
                     for line in (Path(directory) / 'firefly_best.txt').read_text().splitlines()])
    return rows[np.argmin(rows[:, -1]), 1:-1]


def block_bounds():
    """Bounds of P[21:28], the health care block of variable/contact_parameters.pkl."""
    with open('./variable/contact_parameters.pkl', 'rb') as f:
        contact_parameters = pickle.load(f)
    return (np.asarray(contact_parameters['health_care_lower_bound'], dtype=float),
            np.asarray(contact_parameters['health_care_upper_bound'], dtype=float))


def outcomes(executor, P, demo, simulations):
    """measure_outcomes() over the index cases of `simulations` runs."""
    futures = [executor.submit(run_covid, s, P, demo, False) for s in range(simulations)]
    cases = []
    for f in futures:
        _, social, course, contact = f.result()
        if course and contact:
            cases.append((course[0], contact[0], social[0] if social else None))
    return fo.measure_outcomes(cases)


def meets(measured, key):
    lo, hi, _ = fo.OUTCOME_TARGETS[key]
    return lo <= measured[key] <= hi


GRID_FACTORS = (1.0, 1.5, 2.0, 3.0, 4.0)


def grid(base, lb, ub, demo, simulations):
    """P[21] and P[23] times GRID_FACTORS, clipped to their bounds; the rest at the base."""
    print('\n%7s %7s  ' % ('x P21', 'x P23') + '  '.join('%10s' % k[:10] for k in KEYS)
          + '  all four')
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
        for a in GRID_FACTORS:
            for b in GRID_FACTORS:
                P = base.copy()
                P[21] = min(base[21] * a, ub[0])
                P[23] = min(base[23] * b, ub[2])
                m = outcomes(ex, P, demo, simulations)
                ok = all(meets(m, k) for k in KEYS[:4])
                print('%7.1f %7.1f  ' % (a, b) + '  '.join('%10.3f' % m[k] for k in KEYS)
                      + '  %s' % ('yes' if ok else ''))


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else 'firefly_result/phaseD_run18'
    if len(sys.argv) > 2 and sys.argv[2] == 'grid':
        with open('./variable/demographic_parameters.pkl', 'rb') as f:
            demo = pickle.load(f)
        lb, ub = block_bounds()
        grid(load_vector(directory), lb, ub, demo,
             int(sys.argv[3]) if len(sys.argv) > 3 else 200)
        return
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    simulations = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    base = load_vector(directory)
    lb, ub = block_bounds()
    for j, name in enumerate(NAMES):
        print('  P[%2d] %-24s [%.4g, %.4g]  base %.4g'
              % (21 + j, name, lb[j], ub[j], base[21 + j]))
    for key in KEYS:
        print('  target %-34s %s' % (key, fo.OUTCOME_TARGETS[key][:2]))

    rng = np.random.default_rng(0)
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
        measured = outcomes(ex, base, demo, simulations)
        print('\nbase vector: ' + '  '.join('%s %.3f' % (k, measured[k]) for k in KEYS))
        for _ in range(draws):
            P = base.copy()
            P[BLOCK] = lb + rng.random(len(lb)) * (ub - lb)
            rows.append((P[BLOCK].copy(), outcomes(ex, P, demo, simulations)))

    values = {k: np.array([m[k] for _, m in rows], dtype=float) for k in KEYS}
    for k in KEYS:
        print('%-34s min %.3f  median %.3f  max %.3f'
              % (k, np.nanmin(values[k]), np.nanmedian(values[k]), np.nanmax(values[k])))
    floor = np.array([meets(m, KEYS[0]) for _, m in rows])
    shape = np.array([all(meets(m, k) for k in KEYS[1:4]) for _, m in rows])
    print('\ndraws meeting the floor              : %d of %d' % (floor.sum(), draws))
    print('draws meeting the three contact targets: %d' % shape.sum())
    print('draws meeting the floor AND those three: %d' % (floor & shape).sum())
    for label, mask in (('floor and shape', floor & shape), ('floor', floor)):
        if mask.any():
            k = int(np.argmax(np.where(mask, values[KEYS[0]], -np.inf)))
            print('\nexample (%s):' % label)
            for j, name in enumerate(NAMES):
                print('  P[%2d] %-24s %.5g' % (21 + j, name, rows[k][0][j]))
            print('  ' + '  '.join('%s %.3f' % (key, rows[k][1][key]) for key in KEYS))
            break


if __name__ == '__main__':
    main()
