"""Does fast_cost.cost_function return exactly what firefly_optimizer.cost_function returns?

fast_cost moves the Cheng binning, the test matrix and the per-case half of measure_outcomes
into the worker processes. Nothing about that is allowed to change a number, so this evaluates
the same parameter vectors through both and compares the total cost and all 31+ cost parts.

Task 6: the optimiser scores each candidate in one task, fast_cost.evaluate_candidate, in a
spawned worker. It is compared with fast_cost.cost_function the same way, on block 0 and on
generation 1's block.

Run with a small --workers while another job owns the machine: correctness does not need speed.
Timing is only meaningful on an idle machine, so it is reported but flagged.

Usage: uv run python scripts/gates/verify_fast_cost.py RUN_DIR [--workers N] [--vectors N]
"""
import os
import argparse
import concurrent.futures
import multiprocessing
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.runtime import hardware


def load_vectors(count, directory):
    """Parameter vectors to test from a run: its best, its worst, and evenly spaced others."""
    best = Path(directory) / 'firefly_best.txt'
    if not best.exists():
        raise SystemExit(f'{best} not found')
    res = np.atleast_2d(np.loadtxt(best))  # one row would load as 1-D
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


def compare(label, cost_a, parts_a, cost_b, parts_b):
    """Every difference between two costs and their parts, NaN equal to NaN."""
    diffs = []
    if cost_a != cost_b:
        diffs.append(f'{label}TOTAL {cost_a:.17g} != {cost_b:.17g}  '
                     f'(delta {cost_b - cost_a:.3g})')
    for key in sorted(set(parts_a) | set(parts_b)):
        a, b = parts_a.get(key), parts_b.get(key)
        if a is None or b is None:
            diffs.append(f'{label}{key} missing on one side ({a!r} / {b!r})')
        elif not (a == b or (a != a and b != b)):
            diffs.append(f'{label}{key} {a:.17g} != {b:.17g}  (delta {b - a:.3g})')
    return diffs


def main():
    parser = argparse.ArgumentParser()
    hardware.add_workers_argument(parser)  # Task 2 (D3)
    parser.add_argument('--vectors', type=int, default=4)
    parser.add_argument('run_dir', help='run directory to take vectors from (Task 3)')
    args = parser.parse_args()
    args.workers = hardware.resolve_workers(args.workers, os.environ, None)

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]

    vectors = load_vectors(args.vectors, args.run_dir)
    print(f'{fo.SIMULATIONS_PER_EVALUATION} simulations per evaluation, '
          f'{fo.SIMULATIONS_PER_TASK} per task, {args.workers} workers\n')

    slow_pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    fast_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=fast_cost.init_worker,
        initargs=(demo, columns))
    # Task 6: one candidate per task, in a spawned worker as the optimiser makes them.
    candidate_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context('spawn'),
        initializer=fast_cost.init_worker, initargs=(demo, columns, False, None, cheng))
    # Warm the pools so the first vector is not paying for process startup.
    fo.cost_function(vectors[0][1], demo, slow_pool, *cheng)
    fast_cost.cost_function(vectors[0][1], demo, fast_pool, *cheng)
    candidate_pool.submit(fast_cost.evaluate_candidate, vectors[0][1], 0).result()

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

        t0 = time.perf_counter()
        candidate, candidate_parts, _ = candidate_pool.submit(
            fast_cost.evaluate_candidate, P, 0).result()
        t_candidate = time.perf_counter() - t0
        # Generation 1's block too, as the optimiser scores every generation after 0 there.
        offset = fo.generation_seed_offset(1)
        fast_1 = fast_cost.cost_function(P, demo, fast_pool, *cheng, seed_offset=offset)
        fast_1_parts = dict(fo.LAST_COST_PARTS)
        candidate_1, candidate_1_parts, _ = candidate_pool.submit(
            fast_cost.evaluate_candidate, P, offset).result()

        diffs = compare('', slow, slow_parts, fast, fast_parts)
        diffs += compare('one task: ', fast, fast_parts, candidate, candidate_parts)
        diffs += compare(f'one task, seeds from {offset}: ', fast_1, fast_1_parts, candidate_1,
                         candidate_1_parts)
        all_ok = all_ok and not diffs
        print('row %-4d cost %.12f  parts %d  %s'
              % (row, slow, len(slow_parts), 'IDENTICAL' if not diffs else 'MISMATCH'))
        print('         time  slow %.4f s   fast %.4f s   ratio %.2fx  (timing only valid on an idle machine)'
              % (t_slow, t_fast, t_slow / t_fast if t_fast else float('nan')))
        print('         one task  %.4f s on 1 worker' % t_candidate)
        for d in diffs[:12]:
            print('         ' + d)

    slow_pool.shutdown()
    fast_pool.shutdown()
    candidate_pool.shutdown()
    print('\nRESULT: %s' % ('identical on every vector -- safe to adopt for run 7'
                            if all_ok else 'OUTPUT DIFFERS -- do not adopt'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
