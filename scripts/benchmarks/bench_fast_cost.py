"""Steady-state timing of fast_cost against firefly_optimizer, warmup discarded.

verify_fast_cost.py proves the two agree; this measures what the difference is worth. Both
pools are warmed properly first: spawning 32 macOS workers and running the fast pool's
initializer costs seconds, and that showed up as an apparent 0.01x on the first evaluations.

Run this on an idle machine only.
"""
import argparse
import concurrent.futures
import glob
import os
import pickle
import time

import numpy as np

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=15)
    args = parser.parse_args()

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]

    d = [x for x in sorted(glob.glob('Firefly_result_pop_size_100_*'))
         if os.path.exists(x + '/firefly_best.txt')][0]
    res = np.loadtxt(d + '/firefly_best.txt')
    # A spread of vectors, so the timing is not just the best fit's data volume: the search
    # spends most of its evaluations on vectors that produce MORE cases and contacts, which is
    # why the batching benchmark said 2.83x and the real run 6 gave 1.94x.
    rows = [int(np.argmin(res[:, -1])), 0, len(res) // 3, 2 * len(res) // 3,
            int(np.argmax(res[:, -1]))]
    vectors = [res[i, 1:-1] for i in rows]
    print(f'parameters from {d}, {len(vectors)} vectors, {args.workers} workers')
    print(f'{fo.SIMULATIONS_PER_EVALUATION} simulations per evaluation, '
          f'{fo.SIMULATIONS_PER_TASK} per task\n')

    slow_pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    fast_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=fast_cost.init_worker, initargs=(demo, columns))

    for i in range(args.warmup):
        fo.cost_function(vectors[i % len(vectors)], demo, slow_pool, *cheng)
        fast_cost.cost_function(vectors[i % len(vectors)], demo, fast_pool, *cheng)
    print(f'warmup done ({args.warmup} evaluations each)\n')

    slow_times, fast_times = [], []
    for i in range(args.reps):
        P = vectors[i % len(vectors)]
        t0 = time.perf_counter()
        fo.cost_function(P, demo, slow_pool, *cheng)
        slow_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fast_cost.cost_function(P, demo, fast_pool, *cheng)
        fast_times.append(time.perf_counter() - t0)
    slow_pool.shutdown()
    fast_pool.shutdown()

    slow = np.array(slow_times)
    fast = np.array(fast_times)
    print('%-28s %8s %8s %8s' % ('', 'median', 'mean', 'min'))
    print('%-28s %8.4f %8.4f %8.4f' % ('firefly_optimizer (run 6)', np.median(slow),
                                       slow.mean(), slow.min()))
    print('%-28s %8.4f %8.4f %8.4f' % ('fast_cost', np.median(fast), fast.mean(), fast.min()))
    ratio = np.median(slow) / np.median(fast)
    print('\nspeedup on the median: %.2fx' % ratio)

    # Scale to the real search. Run 6 measured 0.0647 s per evaluation over its own log, which
    # is higher than any benchmark because the search visits vectors with more cases; apply the
    # ratio to that rather than to the benchmark number.
    real = 0.0647
    evaluations = 638502
    print('run 6 real rate      : %.4f s -> %.1f h for %d evaluations'
          % (real, real * evaluations / 3600, evaluations))
    print('the same at this ratio: %.4f s -> %.1f h'
          % (real / ratio, real / ratio * evaluations / 3600))


if __name__ == '__main__':
    main()
