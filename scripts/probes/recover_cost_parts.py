"""Recover a finished run's cost decomposition when progress_metrics.csv lost the columns.

Run 7 was launched as `python -m covsyn.calibration.firefly_optimizer`, which makes that file __main__. fast_cost
then did `import firefly_optimizer as fo`, which loaded a SECOND copy of the module under its
real name, so it filled that copy's LAST_COST_PARTS while the running __main__ read its own,
permanently empty dict. The fit itself is unaffected -- both copies hold identical constants -- but the
progress CSV lost every cost_* and measured_* column (34 columns instead of 73).

The objective is deterministic on its fixed seeds (E30), so evaluating the run's best vector
once reproduces exactly the decomposition the optimizer saw. Prints it, and the measured values,
against the acceptance intervals.

Usage: python recover_cost_parts.py [firefly_result_dir]
"""
import concurrent.futures
import glob
import os
import pickle
import sys

import numpy as np

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else [
        d for d in sorted(glob.glob('Firefly_result_pop_size_100_*'))
        if os.path.exists(d + '/firefly_best.txt')][0]
    result = np.loadtxt(os.path.join(directory, 'firefly_best.txt'))
    row = int(np.argmin(result[:, -1]))
    P = result[row, 1:-1]
    print('best vector: row %d of %s, recorded cost %.6f\n' % (row, directory, result[row, -1]))

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns))
    cost = fast_cost.cost_function(P, demo, pool, *cheng)
    pool.shutdown()
    parts = dict(fo.LAST_COST_PARTS)
    print('recomputed cost: %.6f  (matches the recorded value: %s)\n'
          % (cost, abs(cost - result[row, -1]) < 1e-9))

    print('=== objective decomposition ===')
    order = ['cost_contact', 'cost_attack_rate', 'cost_energy', 'cost_penalty', 'cost_outcome']
    total = sum(parts[k] for k in order)
    for key in order:
        print('  %-24s %9.4f   %6.1f%%' % (key, parts[key], 100 * parts[key] / total))
    print('  %-24s %9.4f' % ('total', total))
    for key in ('cost_contact_household', 'cost_contact_healthcare', 'cost_contact_others'):
        print('  %-24s %9.4f' % (key, parts[key]))

    print('\n=== measured values against the intervals ===')
    print('  %-34s %10s %-20s %-9s %s' % ('target', 'value', 'interval', 'state', 'verdict'))
    for name, (lo, hi, w) in sorted(fo.OUTCOME_TARGETS.items()):
        value = parts.get('measured_' + name, np.nan)
        state = 'charged' if w > 0 else 'reported'
        if not np.isfinite(value):
            verdict = 'no value'
        elif w <= 0:
            verdict = ''
        else:
            verdict = 'ok' if lo <= value <= hi else 'OUT'
        print('  %-34s %10.5g %-20s %-9s %s'
              % (name, value, '[%.4g, %.4g]' % (lo, hi), state, verdict))


if __name__ == '__main__':
    main()
