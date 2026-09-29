"""Disease-course distributions of the optimized parameters against their acceptance ranges.

Draws many courses of disease straight from the fitted parameters (no contact process, no
transmission) and shows each quantity against the range the calibration actually used.

Decision B24: the bands are now the SAME ranges the objective is built on -- the literature
reported-mean ranges kept in physiology_penalty() and the Taiwan targets kept in
OUTCOME_TARGETS (firefly_optimizer.py) -- instead of a separate set that contradicted them,
and the pass/fail marks are gone: a mean sitting outside a band is a finding to read with
the rest of the validation, not a test this figure can decide on its own.

Usage: python -m covsyn.figures.plot_validation [param_dir] [out_dir] [n_courses]
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from covsyn.model.data_synthesize import Draw_course_of_disease_data

PARAM = Path(sys.argv[1] if len(sys.argv) > 1 else
             'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else '.')
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
OUT.mkdir(parents=True, exist_ok=True)

res = np.loadtxt(PARAM / 'firefly_best.txt')
P = res[np.argmin(res[:, -1]), 1:-1]
latent = {'latent_period_shape': P[37], 'latent_period_scale': P[38]}
infect = {'infectious_period_shape': P[39], 'infectious_period_scale': P[40]}
incub = {'incubation_period_shape': P[41], 'incubation_period_scale': P[42]}
s2i = {'symptom_to_confirmed_shape': P[43], 'symptom_to_confirmed_scale': P[44], 'symptom_to_confirmed_loc': P[45]}
a2r = {'asymptomatic_to_recovered_shape': P[46], 'asymptomatic_to_recovered_scale': P[47], 'asymptomatic_to_recovered_loc': P[48]}
s2c = {'symptomatic_to_critically_ill_shape': P[49], 'symptomatic_to_critically_ill_scale': P[50], 'symptomatic_to_critically_ill_loc': P[51]}
s2r = {'symptomatic_to_recovered_shape': P[52], 'symptomatic_to_recovered_scale': P[53], 'symptomatic_to_recovered_loc': P[54]}
c2r = {'critically_ill_to_recovered_shape': P[55], 'critically_ill_to_recovered_scale': P[56], 'critically_ill_to_recovered_loc': P[57]}
i2d = {'infection_to_death_shape': P[58], 'infection_to_death_scale': P[59]}
n2c = {'negative_to_confirmed_shape': P[60], 'negative_to_confirmed_scale': P[61], 'negative_to_confirmed_loc': P[62]}

np.random.seed(0)
random.seed(0)
courses = []
for _ in range(N):
    o = Draw_course_of_disease_data(0, latent, infect, incub, s2i, a2r, s2c, s2r, c2r, i2d, n2c,
                                    P[67], [P[195], P[196], P[197]])
    o.draw_course_of_disease()
    courses.append(o)


def missing(x):
    return x is None or (isinstance(x, float) and (np.isnan(x) or x <= -1 or x >= 1e9))


def g(f):
    return np.array([f(c) for c in courses if f(c) is not None], float)


# (name, data, low, high, where the range comes from)
metrics = [
    ('Latent period', g(lambda c: c.latent_period), 4.1, 4.5,
     'held at the lower edge of 4.1-5.5 so the generation time stays as low as it can (B1)'),
    ('Incubation period', g(lambda c: None if missing(c.incubation_period) else c.incubation_period), 3.9, 8.0,
     'literature reported means; now built as latent + window (B22)'),
    ('Infectious period', g(lambda c: c.infectious_period), 5.0, 10.0,
     'inside the reported 3.45-20 range'),
    ('Pre-symptomatic window', g(lambda c: None if missing(c.incubation_period) else c.incubation_period - c.latent_period), 1.0, 3.0,
     'about 2 days of pre-symptomatic infectiousness (B22)'),
    ('Onset -> case closure', g(lambda c: None if (missing(c.incubation_period) or missing(c.date_of_recovery))
                                else c.date_of_recovery - c.incubation_period), 20.0, 32.0,
     'date_of_recovery is release from isolation, about 25 days in Taiwan (B28)'),
    ('Infection -> closure (asymptomatic)', g(lambda c: None if (not missing(c.incubation_period) or missing(c.date_of_recovery))
                                              else c.date_of_recovery), 20.0, 32.0,
     'same definition, asymptomatic cases (B28)'),
    ('Onset -> death', g(lambda c: None if (missing(c.incubation_period) or missing(c.date_of_death))
                         else c.date_of_death - c.incubation_period), 14.0, 21.0,
     'literature'),
    ('Onset -> confirmation', g(lambda c: None if missing(c.incubation_period)
                                else float(np.ravel(c.positive_test_date)[0]) - c.incubation_period), 1.0, 6.0,
     'Taiwan: 5 days before March 2020 (Cheng cohort), 1 day after; left free between them (E33)'),
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()
for ax, (name, data, lo, hi, source) in zip(axes, metrics):
    if len(data) == 0:
        ax.set_visible(False)
        continue
    ax.hist(data, bins=range(0, int(max(data.max(), hi)) + 2), density=True,
            color='#4C78A8', alpha=0.75, edgecolor='white')
    ax.axvspan(lo, hi, color='#59A14F', alpha=0.18, label=f'calibration range {lo:g}-{hi:g} d')
    mean = data.mean()
    ax.axvline(mean, color='#E15759', lw=2, label=f'CovSyn mean {mean:.1f} d')
    if name == 'Onset -> confirmation':
        for day, text in ((1, 'Taiwan after Mar 2020'), (5, 'Taiwan before Mar 2020')):
            ax.axvline(day, color='black', ls=':', lw=1.2)
            ax.text(day, ax.get_ylim()[1] * 0.92, f' {text}', fontsize=7, rotation=90, va='top')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel(f'days\n{source}', fontsize=7)
    ax.legend(fontsize=8)

# age risk ratios: the locked input vs the target it is tuned against
ax = axes[8]
ours = [P[63], P[64], P[65], P[66]]
cheng = [0.52, 1.0, 1.83, 1.32]
x = np.arange(4)
w = 0.38
ax.bar(x - w / 2, ours, w, label='CovSyn input (locked, B14)', color='#4C78A8')
ax.bar(x + w / 2, cheng, w, label='target: Cheng 2020, all infections', color='#59A14F')
ax.set_xticks(x)
ax.set_xticklabels(['0-19', '20-39', '40-59', '60+'])
ax.set_title('Age risk ratios: input vs target', fontsize=11, fontweight='bold')
ax.set_ylabel('relative risk')
ax.set_xlabel('the input is NOT the output: what the simulation produces is measured by\n'
              'rr_exact.py and checked by verify_phaseD.py (B2, B14)', fontsize=7)
ax.legend(fontsize=8)
for i, value in enumerate(ours):
    ax.text(i - w / 2, value + 0.03, f'{value:.2f}', ha='center', fontsize=8)

fig.suptitle('CovSyn disease course vs the ranges used to calibrate it', fontsize=14, fontweight='bold')
fig.text(0.01, 0.005, 'Courses are drawn directly from the fitted parameters with no infector, so every case is '
         'isolated by its own symptoms or not at all; in a full simulation most cases are isolated earlier by '
         'contact tracing (B26), which shortens the contact window but not these durations. Ranges are the ones the '
         'objective actually uses (physiology_penalty and OUTCOME_TARGETS in firefly_optimizer.py); a mean outside '
         'its band is a finding, not a verdict (B24).', fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
target = OUT / 'covsyn_vs_literature.png'
fig.savefig(target, dpi=130)
print('saved', target, '| courses drawn:', N)
