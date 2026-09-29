"""Which outcome targets blow up when the community contact mean is raised to hold the median?

Follow-up to probe_tail_price.py: raising P[28] to keep the all-case median at 3 while lowering
k = P[198] cost +20 to +90 of outcome penalty. This prints every charged target that is out of
its band at three points around run 7's best vector, with its share of the outcome cost, so the
price can be read as either structural or something the optimizer could re-balance.

Usage: python probe_tail_price_detail.py
"""
import concurrent.futures
import pickle

import numpy as np

import fast_cost
import firefly_optimizer as fo
from cost_parts import LAST

RUN7 = 'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_120_run7/firefly_best.txt'
# (label, k = P[198], factor on P[28], P[29]). P[29] is the probability of meeting yesterday's
# contact again: lowering it spreads the same daily contact count over MORE distinct people,
# which is the only lever that raises the distinct count without raising daily_municipality.
POINTS = [('run 7 as found', None, 1.0, None)] + [
    ('k %.2f, lambda x%.1f, P29 %.2f' % (k, f, p29), k, f, p29)
    for p29 in (0.6, 0.5) for k in (0.45, 0.3, 0.2) for f in (1.0, 1.5, 2.2, 3.0)]


def main():
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]
    result = np.atleast_2d(np.loadtxt(RUN7))
    base = result[int(np.argmin(result[:, -1])), 1:-1]

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns))
    for label, k, factor, p29 in POINTS:
        P = base.copy()
        if k is not None:
            P[198] = k
        if p29 is not None:
            P[29] = p29
        P[28] = base[28] * factor
        total = fast_cost.cost_function(P, demo, pool, *cheng)
        parts = dict(LAST)
        print('=== %s: total %.4f  contact %.4f  attack %.4f  penalty %.4f  outcome %.4f'
              '  | ratio %.2f median %.1f daily %.2f inf %.3f' % (
            label, total, parts['cost_contact'], parts['cost_attack_rate'],
            parts['cost_penalty'], parts['cost_outcome'],
            parts.get('measured_community_tail_ratio', np.nan),
            parts.get('measured_community_median', np.nan),
            parts.get('measured_daily_municipality', np.nan),
            parts.get('measured_infections_per_index_municipality', np.nan)), flush=True)
        for name, (lo, hi, w) in sorted(fo.OUTCOME_TARGETS.items()):
            x = parts.get('measured_' + name, np.nan)
            if w <= 0 or not np.isfinite(x) or lo <= x <= hi:
                continue
            term = fo.outcome_penalty({name: x})
            print('  OUT %-34s %10.4g  [%.4g, %.4g]  term %8.4f' % (name, x, lo, hi, term))
    pool.shutdown()


if __name__ == '__main__':
    main()
