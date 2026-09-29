"""Audit CovSyn parameter bounds for values that are too large / too small.

Two checks:
  1. Static bound audit: for every gamma-distributed duration, compute the mean range
     achievable WITHIN the bounds (shape*scale+loc) and compare to a literature range.
     A bound is flagged when its whole range, or part of it, falls outside literature.
  2. Best-fit audit (optional): given a Firefly result dir, report any of the 198
     parameters that sit within `tol` of a bound (pegged = optimizer wanted to go
     further, i.e. the bound is actively constraining the fit).

Usage:
    python audit_bounds.py
    python audit_bounds.py ./Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200
"""
import sys
import pickle
import numpy as np
from pathlib import Path

# course_parameters index -> (shape, scale, loc-or-None, literature_mean_range, label)
DURATIONS = [
    (0, 1, None, (2, 5), 'latent'),
    (2, 3, None, (4, 14), 'infectious'),
    (4, 5, None, (4, 8), 'incubation'),
    (6, 7, 8, (2, 8), 'symptom->confirmed'),
    # NOTE on "recovery": in figshare_taiwan_covid.xlsx recovery == official de-isolation
    # (repeated negative PCR), so these are long by design. Reference ranges below reflect
    # the actual Taiwan cohort, not generic viral-clearance literature.
    (9, 10, 11, (10, 75), 'asymptomatic->recovered'),    # Taiwan data n=2 only (31,73 d): distribution statistically unfounded
    (12, 13, 14, (3, 14), 'symptomatic->critically_ill'),
    (15, 16, 17, (10, 40), 'symptomatic->recovered'),
    (18, 19, 20, (10, 60), 'critically_ill->recovered'),  # Taiwan data n=28, median ~30 d (strict discharge criteria)
    (21, 22, None, (14, 24), 'infection->death'),
    (23, 24, 25, (0, 10), 'negative->confirmed'),
]


def static_audit():
    lb = np.load('variable/course_parameters_lb.npy')
    ub = np.load('variable/course_parameters_ub.npy')
    print('=== 1. Duration bounds vs literature (mean = shape*scale+loc) ===')
    print(f'{"parameter":30s} {"min..max in bounds":>20s} {"literature":>12s}  flag')
    for s, sc, lo, (litlo, lithi), name in DURATIONS:
        loc_lb = lb[lo] if lo is not None else 0.0
        loc_ub = ub[lo] if lo is not None else 0.0
        mmin = lb[s] * lb[sc] + loc_lb
        mmax = ub[s] * ub[sc] + loc_ub
        flag = ''
        if mmin > lithi:
            flag = 'CRITICAL: even minimum > literature max'
        elif mmax > lithi * 1.3:
            flag = 'wide: max far above literature'
        if mmin < 0:
            flag += ' / can go negative'
        print(f'{name:30s} {mmin:8.2f} .. {mmax:6.2f} {litlo:5d}-{lithi:<5d}  {flag}')

    # Age-risk and transition probabilities
    print('\n=== 2. Age-risk ratios & transition probabilities ===')
    print('age_risk lb', np.round(lb[26:30], 3), 'ub', np.round(ub[26:30], 3),
          '(40-59 & 60+ lb < 1 means older-age risk can be suppressed below literature)')
    print('transition_p lb', np.round(lb[158:161], 3), 'ub', np.round(ub[158:161], 3),
          '(symptom->recovered lb low => high ICU fraction allowed)')

    # Attack-rate cross-layer dominance
    print('\n=== 3. Attack-rate cross-layer comparison (upper bounds) ===')
    aru = ub[33:158].reshape(5, 25)
    layers = ['household', 'school', 'workplace', 'healthcare', 'municipality']
    hh = aru[0].max()
    for i, L in enumerate(layers):
        flag = ' <-- exceeds household' if (i != 0 and aru[i].max() > hh) else ''
        print(f'  {L:13s} max attack-rate ub {aru[i].max():.4f}{flag}')


def bestfit_audit(result_dir, tol=0.01):
    cp = pickle.load(open('variable/contact_parameters.pkl', 'rb'))
    order = ['household', 'school', 'workplace', 'health_care', 'municipality']
    lo, hi = [], []
    for L in order:
        lo += cp[L + '_lower_bound']
        hi += cp[L + '_upper_bound']
    lo += cp['overdispersion_lower_bound']
    hi += cp['overdispersion_upper_bound']
    full_lb = np.hstack([np.array(lo), np.load('variable/course_parameters_lb.npy')])
    full_ub = np.hstack([np.array(hi), np.load('variable/course_parameters_ub.npy')])

    res = np.array([[float(n) for n in line.split()]
                    for line in open(Path(result_dir) / 'firefly_best.txt')])
    P = res[np.argmin(res[:, -1]), 1:-1]
    span = np.where(full_ub > full_lb, full_ub - full_lb, 1.0)
    at_lo = (P - full_lb) / span < tol
    at_hi = (full_ub - P) / span < tol
    print(f'\n=== 4. Best-fit params pegged at a bound (within {tol:.0%}) in {result_dir} ===')
    pegged = np.where(at_lo | at_hi)[0]
    if len(pegged) == 0:
        print('  none pegged.')
    for i in pegged:
        side = 'LOWER' if at_lo[i] else 'UPPER'
        print(f'  P[{i:3d}] = {P[i]:.5g}  pegged at {side} bound [{full_lb[i]:.5g}, {full_ub[i]:.5g}]')


if __name__ == '__main__':
    static_audit()
    if len(sys.argv) > 1:
        bestfit_audit(sys.argv[1])
