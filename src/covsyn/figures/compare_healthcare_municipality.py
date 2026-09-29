"""Health care and community layers against the Taiwan data that exists for them.

These are the two layers with the weakest references, so this figure puts every observation
we actually have next to the simulation, and says how large each sample is.

  (a) health care, WHEN the contacts happen relative to symptom onset, against Cheng 2020's
      medical contacts (697 contacts of 100 index cases). This is the comparison that showed
      the model produces no contact at all 8 or more days after onset while Cheng has 37%
      of the medical contacts there -- an isolated patient keeps meeting staff, the model
      stops meeting them.
  (b) health care, contacts per case and attack rate, against Cheng 2020, Huang 2021 and the
      Taiwan contact-tracing records.
  (c) community, the TAIL of the contacts-per-case distribution, drawn as P(X >= x) on log
      axes, against the tracing records. The medians agree; the tail does not.
  (d) community, attack rate against its calibration anchor and Cheng's pooled "others".

Usage: python -m covsyn.figures.compare_healthcare_municipality [spread_dir] [out_dir]
"""
import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPREAD = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'validation_figures_phaseD')
OUT.mkdir(parents=True, exist_ok=True)
REF, SIM = '#59A14F', '#3a7ca5'
LIT = '#B07AA1'

# Cheng et al. 2020 (JAMA Intern Med), medical contacts by days from onset of the index case:
# < 0, 0-3, 4-5, 6-7, 8-9, > 9 days. 6 infections among 697 contacts of 100 index cases.
with open('variable/processed_contact_tracing_data.pkl', 'rb') as f:
    CHENG = np.asarray(pickle.load(f)['Cheng_contact_array'], dtype=float)
CHENG_MEDICAL = CHENG[1]
BINS = ['before\nonset', '0-3', '4-5', '6-7', '8-9', '>9']

# ------------------------------------------------------------------ simulation
first_day, hc_per_case, mu_per_case = [], [], []
hc_cand = hc_eff = mu_cand = mu_eff = 0
for f in sorted(glob.glob(str(SPREAD / 'contact_data_*.npy')),
                key=lambda p: int(Path(p).stem.split('_')[-1])):
    k = int(Path(f).stem.split('_')[-1])
    contact = np.load(f, allow_pickle=True)
    course = np.load(SPREAD / f'course_of_disease_data_{k}.npy', allow_pickle=True)
    for c, cd in zip(contact, course):
        hc = list(c['health_care_effective_contacts'] or [])
        mu = list(c['municipality_effective_contacts'] or [])
        hc_per_case.append(len(hc)); mu_per_case.append(len(mu))
        hc_cand += len(hc); hc_eff += sum(1 for x in hc if x == 1)
        mu_cand += len(mu); mu_eff += sum(1 for x in mu if x == 1)
        onset = cd['incubation_period']
        if onset is None or np.isnan(onset):      # Cheng's figure has no asymptomatic cases
            continue
        matrix = np.asarray(c['health_care_contacts_matrix'], dtype=float)
        if matrix.size:
            first_day.extend((np.argmax(matrix > 0, axis=1) - onset).tolist())
first_day = np.array(first_day, dtype=float)
hc_per_case = np.array(hc_per_case, dtype=float)
mu_per_case = np.array(mu_per_case, dtype=float)
covsyn_medical = np.array([(first_day < 0).sum(), ((first_day >= 0) & (first_day <= 3)).sum(),
                           ((first_day >= 4) & (first_day <= 5)).sum(),
                           ((first_day >= 6) & (first_day <= 7)).sum(),
                           ((first_day >= 8) & (first_day <= 9)).sum(), (first_day > 9).sum()], dtype=float)

# ------------------------------------------------------------------ observations
tracing = pd.read_csv('validation_reference/taiwan_tracing_contacts_per_case.csv')


def observed(layer):
    x = tracing[tracing['layer'] == layer]
    return x[x['uninfected'].notna()]['total'].to_numpy(dtype=float)


hc_obs, mu_obs = observed('health_care'), observed('municipality')


def wilson(k, n, z=1.96):
    if not n:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(c - h, 0), 100 * (c + h)


fig, ax = plt.subplots(2, 2, figsize=(17, 11))

# ---------------------------------------------------------------- (a) timing, health care
x = np.arange(6)
w = 0.38
ax[0, 0].bar(x - w / 2, 100 * CHENG_MEDICAL / CHENG_MEDICAL.sum(), w, color=REF,
             edgecolor='black', label=f'Cheng 2020 medical contacts (n={int(CHENG_MEDICAL.sum()):,})')
ax[0, 0].bar(x + w / 2, 100 * covsyn_medical / max(covsyn_medical.sum(), 1), w, color=SIM,
             edgecolor='black', label=f'CovSyn health care (n={int(covsyn_medical.sum()):,})')
for i in range(6):
    ax[0, 0].text(i + w / 2, 100 * covsyn_medical[i] / max(covsyn_medical.sum(), 1) + 1,
                  f'{100 * covsyn_medical[i] / max(covsyn_medical.sum(), 1):.0f}%', ha='center', fontsize=8)
    ax[0, 0].text(i - w / 2, 100 * CHENG_MEDICAL[i] / CHENG_MEDICAL.sum() + 1,
                  f'{100 * CHENG_MEDICAL[i] / CHENG_MEDICAL.sum():.0f}%', ha='center', fontsize=8)
ax[0, 0].set_xticks(x)
ax[0, 0].set_xticklabels(BINS)
ax[0, 0].set_ylabel('% of that layer\'s contacts')
ax[0, 0].set_xlabel('days from symptom onset of the index case to the first contact')
late_cheng = 100 * CHENG_MEDICAL[4:].sum() / CHENG_MEDICAL.sum()
late_covsyn = 100 * covsyn_medical[4:].sum() / max(covsyn_medical.sum(), 1)
ax[0, 0].set_title('(a) Health care: WHEN are the contacts made?\n'
                   f'8 or more days after onset: Cheng {late_cheng:.0f}%, CovSyn {late_covsyn:.0f}%',
                   fontsize=11, fontweight='bold')
ax[0, 0].legend(fontsize=9)
ax[0, 0].grid(axis='y', alpha=0.25)

# ---------------------------------------------------------------- (b) health care level
labels, values, errors, colours = [], [], [], []
p, lo, hi = wilson(6, 698)
labels.append('Cheng 2020\n6/698 contacts\nof 100 index cases'); values.append(p); errors.append((p - lo, hi - p)); colours.append(REF)
p, lo, hi = wilson(8, 455)
labels.append('Huang 2021\nhospital cluster\n8/455'); values.append(p); errors.append((p - lo, hi - p)); colours.append(LIT)
p, lo, hi = wilson(hc_eff, hc_cand)
labels.append(f'CovSyn\n{hc_eff}/{hc_cand:,}'); values.append(p); errors.append((p - lo, hi - p)); colours.append(SIM)
ax[0, 1].bar(range(len(values)), values, color=colours, edgecolor='black',
             yerr=np.array(errors).T, capsize=5)
ax[0, 1].axhspan(0.1, 1.6, color='gray', alpha=0.18, label='calibration anchor 0.86% (bounds 0.1-1.6%)')
ax[0, 1].set_xticks(range(len(labels)))
ax[0, 1].set_xticklabels(labels, fontsize=8)
ax[0, 1].set_ylabel('secondary attack rate per contact (%)')
ax[0, 1].set_title('(b) Health care: how often does a medical contact get infected?\n'
                   f'CovSyn contacts per case {hc_per_case.mean():.2f} vs tracing median '
                   f'{np.median(hc_obs) if len(hc_obs) else float("nan"):.0f} (n={len(hc_obs)} records)',
                   fontsize=11, fontweight='bold')
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(axis='y', alpha=0.25)

# ---------------------------------------------------------------- (c) community tail
def survival(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, 1.0 - np.arange(len(values)) / len(values)


for data, colour, label in ((mu_obs, REF, f'Taiwan contact tracing (n={len(mu_obs)} index cases)'),
                            (mu_per_case[mu_per_case > 0], SIM, f'CovSyn (n={int((mu_per_case > 0).sum()):,} cases)')):
    if len(data) == 0:
        continue
    v, s = survival(data)
    ax[1, 0].step(np.maximum(v, 0.5), s, where='post', color=colour, lw=2, label=label)
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')
ax[1, 0].set_xlabel('community contacts of one case, x')
ax[1, 0].set_ylabel('P(contacts >= x)')
if len(mu_obs):
    ax[1, 0].set_title('(c) Community: the tail, not the middle, is what differs\n'
                       f'median {np.median(mu_obs):.0f} vs {np.median(mu_per_case[mu_per_case > 0]):.0f}, '
                       f'mean {mu_obs.mean():.0f} vs {mu_per_case[mu_per_case > 0].mean():.0f}, '
                       f'max {mu_obs.max():.0f} vs {mu_per_case.max():.0f}',
                       fontsize=11, fontweight='bold')
ax[1, 0].legend(fontsize=9)
ax[1, 0].grid(alpha=0.25, which='both')

# ---------------------------------------------------------------- (d) community level
labels, values, errors, colours = [], [], [], []
p, lo, hi = wilson(1, 1836)
labels.append('Cheng 2020 "others"\n1/1836\n(school+work+community)'); values.append(p); errors.append((p - lo, hi - p)); colours.append(REF)
p, lo, hi = wilson(mu_eff, mu_cand)
labels.append(f'CovSyn community\n{mu_eff}/{mu_cand:,}'); values.append(p); errors.append((p - lo, hi - p)); colours.append(SIM)
ax[1, 1].bar(range(len(values)), values, color=colours, edgecolor='black',
             yerr=np.array(errors).T, capsize=5)
ax[1, 1].axhspan(0.1, 1.0, color='gray', alpha=0.18, label='calibration anchor 0.2% (bounds 0.1-1.0%)')
ax[1, 1].set_xticks(range(len(labels)))
ax[1, 1].set_xticklabels(labels, fontsize=8)
ax[1, 1].set_ylabel('secondary attack rate per contact (%)')
ax[1, 1].set_title('(d) Community: how often does a community contact get infected?',
                   fontsize=11, fontweight='bold')
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(axis='y', alpha=0.25)

fig.suptitle(f'Health care and community layers vs every Taiwan observation we have ({SPREAD.name})',
             fontsize=14, fontweight='bold')
fig.text(0.01, 0.005,
         'Panel (a): Cheng et al. 2020 traced the contacts of 100 index cases from 4 days before symptom onset to isolation and '
         'recorded when each contact happened; CovSyn is scored the same way (first contact day minus onset, symptomatic cases only). '
         'Panel (b): Cheng 2020 Table 2 medical contacts; Huang, Tu & Lai 2021 hospital cluster; the anchor is the cumulative attack '
         'rate handed to the optimizer. Panel (c): Taiwan CDC contact tracing, "friend" plus "other (unknown) contact" per index case, '
         'counting only the records where the uninfected contacts were also reported, against every CovSyn case with at least one '
         'community contact; a straight line on these axes is a power-law tail. Panel (d): Cheng\'s "others" pools school, workplace '
         'and community, so it is a lower bound for the community layer alone. Every observed sample is small and not random: these '
         'are exploratory comparisons, not formal validation.', fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
target = OUT / 'fig_healthcare_community_vs_taiwan.png'
fig.savefig(target, dpi=150)
print('saved', target)
print('health care  bins  Cheng %s' % (100 * CHENG_MEDICAL / CHENG_MEDICAL.sum()).round(1))
print('health care  bins CovSyn %s' % (100 * covsyn_medical / max(covsyn_medical.sum(), 1)).round(1))
print('community    observed median %.0f mean %.1f p90 %.0f max %.0f (n=%d)'
      % (np.median(mu_obs), mu_obs.mean(), np.percentile(mu_obs, 90), mu_obs.max(), len(mu_obs)))
nz = mu_per_case[mu_per_case > 0]
print('community    CovSyn   median %.0f mean %.1f p90 %.0f max %.0f (n=%d)'
      % (np.median(nz), nz.mean(), np.percentile(nz, 90), nz.max(), len(nz)))
