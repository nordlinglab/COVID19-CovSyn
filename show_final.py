"""Show the final generation's decoded metrics and which penalty terms are still active."""
import csv
import numpy as np
from pathlib import Path
from firefly_optimizer import PHYSIOLOGY_PENALTY_WEIGHT

D = Path('Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')
rows = list(csv.DictReader(open(D / 'progress_metrics.csv')))
last = rows[-1]
print('=== final generation (%s of %d) ===' % (last['generation'], len(rows)))
for k, v in last.items():
    if k != 'generation':
        print('  %-32s %s' % (k, round(float(v), 4)))

res = np.loadtxt(D / 'firefly_best.txt')
i = int(np.argmin(res[:, -1]))
P = res[i, 1:-1]
print('\n=== best firefly (cost %.4f) ===' % res[i, -1])


def ou(x, lo, hi):
    return 0.0 if x <= 0 else max(0.0, x / hi - 1.0) + max(0.0, lo / x - 1.0)


# These are the four terms physiology_penalty() actually charges. The table used to list six
# plus a symptom->ICU term, carried over from before Phase D moved the recovery times, the
# asymptomatic share and the severity cascade into outcome_penalty(); worse, it labelled
# P[41]*P[42] "incubation_mean" against a target of 5.2-8.0 when B22 (3) made that pair the
# PRE-ONSET WINDOW, target 1-3 days. The result was a table reporting "TOTAL penalty 38.95"
# and five OUT rows for a run whose real cost_penalty was 0.0004 (finding E54). Keep this
# list in step with firefly_optimizer.physiology_penalty().
terms = [('latent_mean', P[37] * P[38], 4.1, 5.5),
         ('infectious_mean', P[39] * P[40], 5.0, 10.0),
         ('pre_onset_window', P[41] * P[42], 1.0, 3.0),
         ('onset_to_confirmation', P[43] * P[44] + P[45], 1.0, 12.0)]
print('%-26s %9s  %-14s %s' % ('penalty term', 'value', 'target', 'contribution'))
tot = 0.0
for n, v, lo, hi in terms:
    c = PHYSIOLOGY_PENALTY_WEIGHT * ou(v, lo, hi) ** 2
    tot += c
    print('%-26s %9.3f  [%.1f, %.1f]%s%.4f %s' % (n, v, lo, hi, ' ' * 5, c, '<-- OUT' if c > 1e-6 else ''))
print('%-26s %9s  %-14s %.4f' % ('TOTAL penalty', '', '', tot))
print('  (the recovery times, the asymptomatic share and the severity cascade are NOT here:')
print('   they are measured on the simulation by outcome_penalty(), see the measured_* rows)')
print('\nage risk ratios (locked): %s' % np.round(P[63:67], 3).tolist())
print('overdispersion: rate %.4f weight %.3f  -> expected multiplier %.3f'
      % (P[35], P[36], (1 - P[35]) + P[35] * P[36]))
for name, j in [('household', 70), ('school', 95), ('workplace', 120), ('healthcare', 145), ('municipality', 170)]:
    print('  mean daily attack rate %-13s %.5f' % (name, float(np.mean(P[j:j + 25]))))

lb = np.loadtxt(D / 'bound.txt')[0]
ub = np.loadtxt(D / 'bound.txt')[1]
span = np.where(ub - lb > 0, ub - lb, 1.0)
frac = (P - lb) / span
pegged = [k for k in range(len(P)) if (ub[k] > lb[k]) and (frac[k] < 0.02 or frac[k] > 0.98)]
print('\nparameters pegged at a bound: %d of %d' % (len(pegged), int((ub > lb).sum())))
print('  indices:', pegged[:40])
