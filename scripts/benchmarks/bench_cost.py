"""Split one cost_function evaluation into (a) getting the 300 simulations back from the
pool and (b) everything the parent process then does with them.

Reads only. Prints a cProfile of cost_function measured in the parent process, so pool
dispatch shows up as time inside future.result() while the aggregation shows up as
generate_contact_result / np.append.
"""
import cProfile
import concurrent.futures
import io
import pickle
import pstats
import time
from pathlib import Path

import numpy as np

from covsyn.calibration import firefly_optimizer as fo
from covsyn.model.data_synthesis_main import run_covid

D = Path('Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')


def main():
    res = np.loadtxt(D / 'firefly_best.txt')
    P = res[int(np.argmin(res[:, -1])), 1:-1]
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    args = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])

    n = fo.SIMULATIONS_PER_EVALUATION
    ex = concurrent.futures.ProcessPoolExecutor(max_workers=32)
    fo.cost_function(P, demo, ex, *args)

    # (a) pool round trip only: submit n tasks and collect them, nothing else
    t0 = time.perf_counter()
    for _ in range(3):
        futs = [ex.submit(run_covid, i, P, demo, False) for i in range(n)]
        _ = [f.result() for f in futs]
    t_pool = (time.perf_counter() - t0) / 3

    # (b) the same, but batched into one task per worker
    def batched(k):
        chunks = [range(i, min(i + k, n)) for i in range(0, n, k)]
        t0 = time.perf_counter()
        for _ in range(3):
            futs = [ex.submit(_batch, list(c), P, demo) for c in chunks]
            _ = [r for f in futs for r in f.result()]
        return (time.perf_counter() - t0) / 3, len(chunks)

    print('n simulations per evaluation: %d' % n)
    print('\n(a) pool round trip, 1 sim per task : %.4f s  (%d tasks)' % (t_pool, n))
    for k in (5, 10, 20, 38):
        t, c = batched(k)
        print('(b) pool round trip, %2d sims/task  : %.4f s  (%d tasks)  speedup %.1fx'
              % (k, t, c, t_pool / t))

    # (c) full cost_function, profiled
    t0 = time.perf_counter()
    for _ in range(3):
        fo.cost_function(P, demo, ex, *args)
    t_full = (time.perf_counter() - t0) / 3
    print('\n(c) full cost_function             : %.4f s' % t_full)
    print('    of which pool round trip        : %.4f s (%.0f%%)'
          % (t_pool, 100 * t_pool / t_full))
    print('    parent-side aggregation         : %.4f s (%.0f%%)'
          % (t_full - t_pool, 100 * (t_full - t_pool) / t_full))

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        fo.cost_function(P, demo, ex, *args)
    pr.disable()
    ex.shutdown()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats('tottime').print_stats(18)
    print('\n(d) cost_function hot spots (tottime, 3 evaluations)')
    for line in buf.getvalue().splitlines():
        if line.strip():
            print('   ' + line[:140])


def _batch(seeds, P, demo):
    return [run_covid(s, P, demo, save_file=False) for s in seeds]


if __name__ == '__main__':
    main()
