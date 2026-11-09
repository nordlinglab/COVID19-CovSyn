"""Prove the batching refactor did not change a single number.

Imports the pre-refactor optimizer as firefly_optimizer_baseline and the refactored one as
firefly_optimizer, evaluates the same parameter vector through both, and compares the total
cost and every entry of LAST_COST_PARTS exactly. Also times both.
"""
import concurrent.futures
import pickle
import time
from pathlib import Path

import numpy as np

from covsyn.calibration import firefly_optimizer as new
import firefly_optimizer_baseline as old

D = Path('Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')


def main():
    res = np.loadtxt(D / 'firefly_best.txt')
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    args = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])

    # Three different parameter vectors, not just the best one, so the comparison also
    # covers vectors the optimizer would have rejected.
    rows = [int(np.argmin(res[:, -1])), 0, len(res) // 2]
    ex = concurrent.futures.ProcessPoolExecutor(max_workers=32)
    old.cost_function(res[rows[0], 1:-1], demo, ex, *args)          # warm the pool
    all_ok = True
    for row in rows:
        P = res[row, 1:-1]

        t0 = time.perf_counter()
        c_old = old.cost_function(P, demo, ex, *args)
        t_old = time.perf_counter() - t0
        parts_old = dict(old.LAST_COST_PARTS)

        t0 = time.perf_counter()
        c_new = new.cost_function(P, demo, ex, *args)
        t_new = time.perf_counter() - t0
        parts_new = dict(new.LAST_COST_PARTS)

        same_total = (c_old == c_new)
        diffs = []
        for k in sorted(set(parts_old) | set(parts_new)):
            a, b = parts_old.get(k), parts_new.get(k)
            if not (a == b or (a != a and b != b)):             # NaN == NaN counts as equal
                diffs.append('%s %r != %r' % (k, a, b))
        ok = same_total and not diffs
        all_ok = all_ok and ok
        print('row %-4d cost old %.12f  new %.12f  identical=%s  parts compared %d  %s'
              % (row, c_old, c_new, same_total, len(parts_old), 'OK' if ok else 'MISMATCH'))
        print('         time old %.4f s   new %.4f s   speedup %.2fx' % (t_old, t_new, t_old / t_new))
        for d in diffs:
            print('         DIFF ' + d)

    # steady-state timing over more evaluations
    P = res[rows[0], 1:-1]
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        old.cost_function(P, demo, ex, *args)
    t_old = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        new.cost_function(P, demo, ex, *args)
    t_new = (time.perf_counter() - t0) / reps
    ex.shutdown()
    print('\nsteady state over %d evaluations: old %.4f s  new %.4f s  speedup %.2fx'
          % (reps, t_old, t_new, t_old / t_new))
    evals = 638502
    print('run 5 took 22.78 h for %d evaluations (%.4f s each)' % (evals, 82000 / evals))
    print('the same search at the new speed: %.1f h' % (evals * t_new / 3600))
    print('\nRESULT: %s' % ('identical output, safe to use' if all_ok else 'OUTPUT CHANGED -- do not use'))


if __name__ == '__main__':
    main()
