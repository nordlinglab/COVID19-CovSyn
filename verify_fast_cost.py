"""Does fast_cost.cost_function return exactly what firefly_optimizer.cost_function returns?

fast_cost moves the Cheng binning, the test matrix and the per-case half of measure_outcomes
into the worker processes. Nothing about that is allowed to change a number, so this evaluates
the same parameter vectors through both and compares the total cost and all 31+ cost parts.

Run with a small --workers while another job owns the machine: correctness does not need speed.
Timing is only meaningful on an idle machine, so it is reported but flagged.

Usage: python verify_fast_cost.py [--workers N] [--vectors N]
"""
import argparse
import concurrent.futures
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import fast_cost
import firefly_optimizer as fo


def load_vectors(count):
    """Parameter vectors to test: the run 5 best, the worst, and evenly spaced others."""
    for directory in sorted(Path('.').glob('Firefly_result_pop_size_100_*')):
        best = directory / 'firefly_best.txt'
        if best.exists():
            res = np.loadtxt(best)
            order = [int(np.argmin(res[:, -1])), int(np.argmax(res[:, -1]))]
            step = max(len(res) // max(count - 2, 1), 1)
            order += list(range(0, len(res), step))
            seen, rows = set(), []
            for i in order:
                if i not in seen:
                    seen.add(i)
                    rows.append(i)
                if len(rows) >= count:
                    break
            print(f'parameter vectors from {directory}')
            return [(i, res[i, 1:-1]) for i in rows]
    raise SystemExit('no firefly result directory found to take parameter vectors from')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--vectors', type=int, default=4)
    args = parser.parse_args()

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]

    vectors = load_vectors(args.vectors)
    print(f'{fo.SIMULATIONS_PER_EVALUATION} simulations per evaluation, '
          f'{fo.SIMULATIONS_PER_TASK} per task, {args.workers} workers\n')

    slow_pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    fast_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=fast_cost.init_worker,
        initargs=(demo, columns))
    # Warm both pools so the first vector is not paying for process startup.
    fo.cost_function(vectors[0][1], demo, slow_pool, *cheng)
    fast_cost.cost_function(vectors[0][1], demo, fast_pool, *cheng)

    all_ok = True
    for row, P in vectors:
        np.random.seed(12345)
        t0 = time.perf_counter()
        slow = fo.cost_function(P, demo, slow_pool, *cheng)
        t_slow = time.perf_counter() - t0
        slow_parts = dict(fo.LAST_COST_PARTS)

        np.random.seed(12345)
        t0 = time.perf_counter()
        fast = fast_cost.cost_function(P, demo, fast_pool, *cheng)
        t_fast = time.perf_counter() - t0
        fast_parts = dict(fo.LAST_COST_PARTS)

        diffs = []
        if slow != fast:
            diffs.append('TOTAL %.17g != %.17g  (delta %.3g)' % (slow, fast, fast - slow))
        for key in sorted(set(slow_parts) | set(fast_parts)):
            a, b = slow_parts.get(key), fast_parts.get(key)
            if a is None or b is None:
                diffs.append('%s missing on one side (%r / %r)' % (key, a, b))
            elif not (a == b or (a != a and b != b)):
                diffs.append('%s %.17g != %.17g  (delta %.3g)' % (key, a, b, b - a))
        all_ok = all_ok and not diffs
        print('row %-4d cost %.12f  parts %d  %s'
              % (row, slow, len(slow_parts), 'IDENTICAL' if not diffs else 'MISMATCH'))
        print('         time  slow %.4f s   fast %.4f s   ratio %.2fx  (timing only valid on an idle machine)'
              % (t_slow, t_fast, t_slow / t_fast if t_fast else float('nan')))
        for d in diffs[:12]:
            print('         ' + d)

    slow_pool.shutdown()
    fast_pool.shutdown()
    print('\nRESULT: %s' % ('identical on every vector -- safe to adopt for run 7'
                            if all_ok else 'OUTPUT DIFFERS -- do not adopt'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
