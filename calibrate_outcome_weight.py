"""What is the outcome block worth, before and after the E66 rescaling?

Reads the measured_* values the last generation of a finished run actually produced, and prints
each target's contribution under the old width-scaled miss and the new capped-width miss, so the
weight can be set against a real objective rather than guessed. Run 6's decomposition was
cost_contact 1.6956, cost_attack_rate 0.3092, cost_energy -0.1772, cost_penalty 0.0039,
cost_outcome 0.0419 -- i.e. every Phase D decision together was 2.2% of the objective.

Usage: python calibrate_outcome_weight.py [firefly_result_dir]
"""
import csv
import glob
import os
import sys

import numpy as np

import firefly_optimizer as fo

COST_CONTACT_RUN6 = 1.6956
COST_ATTACK_RUN6 = 0.3092


def old_miss(x, lo, hi):
    width = hi - lo
    return max(0.0, lo - x, x - hi) / width if width > 0 else 0.0


def new_miss(x, lo, hi, name=None):
    scale = fo.outcome_scale(name, lo, hi)
    return max(0.0, lo - x, x - hi) / scale if scale else 0.0


def shaped(miss):
    return miss ** 2 if miss <= 1.0 else 2.0 * miss - 1.0


def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = [d for d in sorted(glob.glob('Firefly_result_pop_size_100_*'))
                     if os.path.exists(d + '/progress_metrics.csv')][0]
    rows = list(csv.DictReader(open(directory + '/progress_metrics.csv')))
    last = rows[-1]
    print('measured values from %s, generation %s\n' % (directory, last['generation']))

    print('%-34s %10s %10s %9s %9s %9s' %
          ('target', 'value', 'interval', 'old miss', 'new miss', 'new pen'))
    old_total = new_total = 0.0
    lines = []
    for name, (lo, hi, w) in sorted(fo.OUTCOME_TARGETS.items()):
        if w <= 0:
            continue
        raw = last.get('measured_' + name)
        if raw is None:
            continue
        x = float(raw)
        if not np.isfinite(x):
            continue
        om, nm = old_miss(x, lo, hi), new_miss(x, lo, hi, name)
        op, np_ = w * shaped(om), w * shaped(nm)
        old_total += op
        new_total += np_
        if nm > 0:
            lines.append((np_, '%-34s %10.5g %10s %9.3f %9.3f %9.4f'
                          % (name, x, '[%.3g, %.3g]' % (lo, hi), om, nm, np_)))
    for _, line in sorted(lines, reverse=True):
        print(line)

    print('\nraw penalty sum: old %.4f   new %.4f   (ratio %.2fx)'
          % (old_total, new_total, new_total / old_total if old_total else float('nan')))
    for weight in (fo.OUTCOME_PENALTY_WEIGHT, 4.0, 6.0, 8.0, 10.0):
        cost = weight * new_total
        total = COST_CONTACT_RUN6 + COST_ATTACK_RUN6 + cost
        print('  weight %5.1f -> cost_outcome %7.4f   = %5.1f%% of contact+attack+outcome'
              % (weight, cost, 100 * cost / total))
    print('\nfor reference, at run 6 the old scheme gave cost_outcome %.4f (%.1f%%)'
          % (fo.OUTCOME_PENALTY_WEIGHT * old_total,
             100 * fo.OUTCOME_PENALTY_WEIGHT * old_total
             / (COST_CONTACT_RUN6 + COST_ATTACK_RUN6 + fo.OUTCOME_PENALTY_WEIGHT * old_total)))


if __name__ == '__main__':
    main()
