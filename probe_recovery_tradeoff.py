"""What did run 8 buy by shortening symptomatic recovery from 23 to 8 days?

Run 8's best vector has P[52..54] (the onset-to-recovery Gamma) at a mean of 8.18 days against
23.08 in run 7, and symptomatic case closure fell 28.4 -> 18.6 d, out of its [20, 32] band.
Closure is charged, but a half-day miss under a 12-day scale costs about 0.01, so it is almost
free to trade. This evaluates run 8's best vector twice -- as found, and with only P[52..54]
put back to run 7's values -- and prints every cost part and measured value that moves.

The objective is deterministic on its fixed seeds (E30), so the difference is the attribution.

Usage: python probe_recovery_tradeoff.py
"""
import concurrent.futures
import pickle

import numpy as np

import fast_cost
import firefly_optimizer as fo
from cost_parts import LAST

RUN7 = 'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_120_run7/firefly_best.txt'
RUN8 = 'ARCHIVE_20260928_run_phaseD8/firefly_120/firefly_best.txt'
RECOVERY = [52, 53, 54]


def best_vector(path):
    result = np.loadtxt(path)
    row = int(np.argmin(result[:, -1]))
    return result[row, 1:-1].copy(), result[row, -1]


def main():
    p7, _ = best_vector(RUN7)
    p8, recorded = best_vector(RUN8)
    swapped = p8.copy()
    swapped[RECOVERY] = p7[RECOVERY]

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns))
    results = {}
    for name, P in (('run8', p8), ('run8 + run7 recovery', swapped)):
        cost = fast_cost.cost_function(P, demo, pool, *cheng)
        results[name] = (cost, dict(LAST))
    pool.shutdown()

    base_cost, base = results['run8']
    swap_cost, swap = results['run8 + run7 recovery']
    print('recorded run 8 cost %.6f, recomputed %.6f (match: %s)'
          % (recorded, base_cost, abs(base_cost - recorded) < 1e-9))
    print('P[52..54]: run 8 %s (mean %.2f)  run 7 %s (mean %.2f)\n'
          % (np.round(p8[RECOVERY], 4), p8[52] * p8[53] + p8[54],
             np.round(p7[RECOVERY], 4), p7[52] * p7[53] + p7[54]))
    print('%-40s %12s %12s %12s' % ('part', 'run8', 'swapped', 'swap - run8'))
    print('%-40s %12.4f %12.4f %+12.4f' % ('TOTAL', base_cost, swap_cost, swap_cost - base_cost))
    for key in sorted(set(base) | set(swap)):
        a, b = base.get(key, np.nan), swap.get(key, np.nan)
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if np.isfinite(a) and np.isfinite(b) and abs(a - b) < 1e-9:
            continue
        band = ''
        target = key[len('measured_'):] if key.startswith('measured_') else None
        if target in fo.OUTCOME_TARGETS:
            lo, hi, w = fo.OUTCOME_TARGETS[target]
            band = '[%.4g, %.4g]%s' % (lo, hi, '' if w > 0 else ' info')
        print('%-40s %12.4f %12.4f %+12.4f  %s' % (key, a, b, b - a, band))


if __name__ == '__main__':
    main()
