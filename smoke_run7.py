"""Smoke test before launching Phase D run 6.

cost_function() swallows exceptions and charges 1e6 (finding E31), and 1e6 looks like a
plausible cost, so a whole run can finish before anyone notices it was garbage. This checks,
on ONE evaluation of the seed vector:

  (a) the cost is far below FAILED_EVALUATION_COST
  (b) the seed vector lies inside the rebuilt bounds
  (c) every measured_* field the objective charges actually has a value
  (d) the new run-6 targets are present, charged, and reachable from the seed
  (e) the cross-layer daily attack-rate ceiling holds
"""
import concurrent.futures
import pickle
import time

import numpy as np

import firefly_optimizer as fo
import sar_anchors


def main():
    print('=== (e) cross-layer ceiling (E61) ===')
    breaches = sar_anchors.cross_layer_ceiling_report()
    for layer, ceiling, household in breaches:
        print('  anchor derivation still wants %s at %.5f > household %.5f (capped away)'
              % (layer, ceiling, household))
    for layer in sar_anchors.LAYERS:
        upper = sar_anchors.attack_rate_block(layer)[2]
        print('  %-13s applied daily ceiling %.5f' % (layer, upper.max()))
    household_cap = sar_anchors.attack_rate_block('household')[2].max()
    worst = max((sar_anchors.attack_rate_block(L)[2].max()
                 for L in sar_anchors.LAYERS if L != 'household'))
    assert worst <= household_cap + 1e-12, 'a layer is still above the household ceiling'
    print('  OK: no layer exceeds the household ceiling\n')

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    args = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])

    contact_parameters = pickle.load(open('./variable/contact_parameters.pkl', 'rb'))
    order = ['household', 'school', 'workplace', 'health_care', 'municipality']
    lb, ub = [], []
    for L in order:
        lb += list(contact_parameters[L + '_lower_bound'])
        ub += list(contact_parameters[L + '_upper_bound'])
    lb += list(contact_parameters['overdispersion_lower_bound'])
    ub += list(contact_parameters['overdispersion_upper_bound'])
    lb = np.array(lb + list(np.load('./variable/course_parameters_lb.npy')))
    ub = np.array(ub + list(np.load('./variable/course_parameters_ub.npy')))
    course = np.load('./variable/course_parameters.npy')
    seed = np.hstack(((lb[:37] + ub[:37]) / 2, course))

    print('=== (b) seed vector inside the rebuilt bounds ===')
    print('  length %d  (bounds %d)' % (len(seed), len(lb)))
    assert len(seed) == len(lb) == len(ub), 'length mismatch between seed and bounds'
    outside = np.where((seed < lb - 1e-12) | (seed > ub + 1e-12))[0]
    if len(outside):
        for i in outside[:20]:
            print('  P[%d] = %.6g outside [%.6g, %.6g]' % (i, seed[i], lb[i], ub[i]))
    assert not len(outside), '%d seed values outside the bounds' % len(outside)
    print('  OK: every seed value is inside its bounds\n')

    print('=== (a) one cost evaluation ===')
    ex = concurrent.futures.ProcessPoolExecutor(max_workers=32)
    t0 = time.perf_counter()
    cost = fo.cost_function(seed, demo, ex, *args)
    elapsed = time.perf_counter() - t0
    ex.shutdown()
    parts = dict(fo.LAST_COST_PARTS)
    print('  cost %.4f in %.2f s  (failure sentinel is %g)'
          % (cost, elapsed, fo.FAILED_EVALUATION_COST))
    assert np.isfinite(cost), 'cost is not finite'
    assert cost < fo.FAILED_EVALUATION_COST / 100, 'cost looks like a swallowed exception'
    for k in sorted(parts):
        if k.startswith('cost_'):
            print('  %-32s %.4f' % (k, parts[k]))
    print('  OK\n')

    print('=== (c)/(d) every charged target has a measured value ===')
    missing = []
    for name, (lo, hi, w) in sorted(fo.OUTCOME_TARGETS.items()):
        value = parts.get('measured_' + name, np.nan)
        state = 'charged' if w > 0 else 'reported'
        flag = ''
        if not np.isfinite(value):
            flag = ' <-- NO VALUE'
            if w > 0:
                missing.append(name)
        elif w > 0 and not (lo <= value <= hi):
            flag = ' <-- outside [%.4g, %.4g] at the seed' % (lo, hi)
        print('  %-30s %-9s %12.5g%s' % (name, state, value, flag))
    assert not missing, 'charged targets with no measured value: %s' % missing
    print('\n  OK: all %d charged targets measured' %
          sum(1 for _, _, w in fo.OUTCOME_TARGETS.values() if w > 0))
    print('\nSMOKE TEST PASSED -- safe to start the run')


if __name__ == '__main__':
    main()
