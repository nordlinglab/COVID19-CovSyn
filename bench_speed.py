"""Where does a firefly generation's time actually go?

Measures, on the real parameter vector of run 5:
  1. how big demographic_parameters is when pickled -- it is passed as an argument to EVERY
     one of the 300 tasks per evaluation, so its serialised size is paid 300x per evaluation
  2. the pure in-process compute cost of one run_covid
  3. the wall time of one cost_function evaluation through the persistent pool
  4. a cProfile of run_covid, to see whether the simulation itself has a hot spot

Nothing is written; this only reads variable/ and the run 5 firefly result.
"""
import cProfile
import concurrent.futures
import io
import pickle
import pstats
import time
from pathlib import Path

import numpy as np

import firefly_optimizer as fo
from Data_synthesis_main import run_covid

D = Path('Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')


def main():
    res = np.loadtxt(D / 'firefly_best.txt')
    P = res[int(np.argmin(res[:, -1])), 1:-1]

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng_contact = ct['Cheng_contact_array']
    cheng_ar = ct['Cheng_attack_rate']
    norm_w = ct['norm_weights']

    print('SIMULATIONS_PER_EVALUATION = %d' % fo.SIMULATIONS_PER_EVALUATION)

    # ---------------------------------------------------------------- 1. payload size
    t0 = time.perf_counter()
    blob = pickle.dumps(demo)
    t_pickle = time.perf_counter() - t0
    print('\n1. demographic_parameters: %.2f MB, pickle %.1f ms'
          % (len(blob) / 1e6, 1000 * t_pickle))
    print('   type %s' % type(demo).__name__)
    if isinstance(demo, dict):
        for k, v in list(demo.items())[:12]:
            n = len(pickle.dumps(v))
            print('     %-34s %-12s %8.3f MB' % (k, type(v).__name__, n / 1e6))
    n = fo.SIMULATIONS_PER_EVALUATION
    print('   paid per evaluation if passed per task: %.1f MB (%d x)' % (n * len(blob) / 1e6, n))
    print('   pickle cost per evaluation:             %.0f ms' % (1000 * n * t_pickle))

    # ---------------------------------------------------------------- 2. one simulation
    reps = 20
    t0 = time.perf_counter()
    for i in range(reps):
        run_covid(i, P, demo, save_file=False)
    t_sim = (time.perf_counter() - t0) / reps
    print('\n2. run_covid in-process: %.1f ms per simulation (%d reps)' % (1000 * t_sim, reps))
    print('   serial compute per evaluation: %.2f s' % (n * t_sim))

    # ---------------------------------------------------------------- 3. one evaluation
    for workers in (32,):
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        fo.cost_function(P, demo, ex, cheng_contact, cheng_ar, norm_w)     # warm the pool
        t0 = time.perf_counter()
        for _ in range(3):
            fo.cost_function(P, demo, ex, cheng_contact, cheng_ar, norm_w)
        t_eval = (time.perf_counter() - t0) / 3
        ex.shutdown()
        eff = n * t_sim / (workers * t_eval)
        print('\n3. cost_function with %d workers: %.3f s per evaluation' % (workers, t_eval))
        print('   parallel efficiency: %.0f%%  (ideal %.3f s)'
              % (100 * eff, n * t_sim / workers))
        print('   -> 120 generations x ~5300 evaluations = %.1f h'
              % (120 * 5321 * t_eval / 3600))

    # ---------------------------------------------------------------- 4. profile the sim
    print('\n4. run_covid hot spots (10 simulations)')
    pr = cProfile.Profile()
    pr.enable()
    for i in range(10):
        run_covid(i, P, demo, save_file=False)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats('cumulative').print_stats(22)
    for line in buf.getvalue().splitlines():
        if line.strip():
            print('   ' + line[:150])


if __name__ == '__main__':
    main()
