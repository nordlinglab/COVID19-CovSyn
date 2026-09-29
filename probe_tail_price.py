"""What does it cost the rest of the objective to grow the community tail to a ratio of 8?

Finding E76: run 8 held the community tail ratio at 4.0 against a charged band of [8, 40], and the
tail term was about 0.67 of its 0.77 outcome cost. The optimizer declined to pay whatever growing
the tail costs elsewhere, so before re-weighting the target (B50) this measures that price.

Starting from the best vectors of run 7 and run 8, it lowers the community contact-count
dispersion k = P[198] (lower k, heavier tail) and raises the mean P[28] to hold the all-case
median, and evaluates the full objective at every point. For each point it reports the tail
ratio, the all-case median, and the cost WITHOUT the tail term -- the difference from the
unmodified vector is what the optimizer saves by not growing the tail. The same grid is the
reachability gate (lesson 2) for the re-weighted target.

Usage: python probe_tail_price.py
"""
import concurrent.futures
import pickle

import numpy as np

import fast_cost
import firefly_optimizer as fo
from cost_parts import LAST

STARTS = {
    'run7': 'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_120_run7/firefly_best.txt',
    'run8': 'ARCHIVE_20260928_run_phaseD8/firefly_120/firefly_best.txt',
}
K_VALUES = [None, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12]
LAMBDA_FACTORS = [1.0, 1.5, 2.2, 3.0]


def best_vector(path):
    result = np.atleast_2d(np.loadtxt(path))
    return result[int(np.argmin(result[:, -1])), 1:-1].copy()


def tail_term(measured):
    lo, hi, w = fo.OUTCOME_TARGETS['community_tail_ratio']
    x = measured.get('measured_community_tail_ratio', np.nan)
    if not np.isfinite(x) or w <= 0:
        return 0.0
    return fo.outcome_penalty({'community_tail_ratio': x})


def main():
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]
    lb, ub = np.loadtxt('ARCHIVE_20260928_run_phaseD8/firefly_120/bound.txt')
    print('P[198] bounds [%g, %g], P[28] bounds [%g, %g]' % (lb[198], ub[198], lb[28], ub[28]))
    print('tail band %s, median band %s\n' % (fo.OUTCOME_TARGETS['community_tail_ratio'][:2],
                                              fo.OUTCOME_TARGETS['community_median'][:2]))

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns))
    header = '%-5s %6s %6s | %6s %6s %6s %5s | %8s %8s %8s %8s %9s' % (
        'start', 'k', 'lam', 'ratio', 'med', 'p90', 'zero', 'total', 'no_tail', 'contact',
        'outcome', 'd_no_tail')
    for name, path in STARTS.items():
        base = best_vector(path)
        print(header)
        reference = None
        for k in K_VALUES:
            for factor in LAMBDA_FACTORS:
                if k is None and factor != 1.0:
                    continue
                P = base.copy()
                if k is not None:
                    P[198] = k
                P[28] = min(base[28] * factor, ub[28])
                total = fast_cost.cost_function(P, demo, pool, *cheng)
                parts = dict(LAST)
                no_tail = total - tail_term(parts)
                if reference is None:
                    reference = no_tail
                print('%-5s %6.3f %6.2f | %6.2f %6.1f %6.1f %5.2f | %8.4f %8.4f %8.4f %8.4f %+9.4f' % (
                    name, P[198], P[28], parts.get('measured_community_tail_ratio', np.nan),
                    parts.get('measured_community_median', np.nan),
                    parts.get('measured_community_p90', np.nan),
                    parts.get('measured_community_zero_share', np.nan), total, no_tail,
                    parts.get('cost_contact', np.nan), parts.get('cost_outcome', np.nan),
                    no_tail - reference), flush=True)
        print()
    pool.shutdown()


if __name__ == '__main__':
    main()
