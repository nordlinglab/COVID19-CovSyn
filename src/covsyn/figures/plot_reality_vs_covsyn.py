"""Every quantity CovSyn is validated against, with the observation on one side and the
simulation on the other, on a single page.

Each row is one comparison. The dot on the left of each pair is what was observed in Taiwan
(or in the literature, where Taiwan has nothing); the dot on the right is what the simulation
produced. The axis is the ratio between them on a log scale, so 1 means the model reproduces
the observation, 2 means twice as large and 0.5 means half. The actual pair of numbers is
printed on every row, because a ratio alone hides whether a disagreement matters: 0.5 on a
household attack rate and 0.5 on a case fatality are not the same kind of error.

Only quantities with a real observation behind them are included. Where the only reference is
a literature RANGE rather than a point (the disease-course durations), the midpoint of the
range is used as the observation and the range is drawn as a band.

Usage: python -m covsyn.figures.plot_reality_vs_covsyn [spread_dir] [first_outbreak_dir] [out_dir]
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

# E63: the calibration anchors are READ from the module that defines them, never copied.
# Four of the values below used to be written out here by hand and had drifted three
# decisions behind (B1 lowered household to 3.73%, E55 raised health care 2.5x and moved
# the community anchor, B2 replaced the 1-day onset-to-confirmation with 5-7 days), so this
# figure was scoring run 5 against targets that no longer existed.
from covsyn.calibration.sar_anchors import LAYER_CUMULATIVE_SAR, LAYER_INFECTIONS_PER_INDEX

SPREAD = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight')
FIRST = Path(sys.argv[2] if len(sys.argv) > 2 else 'synthetic_data_results_taiwan_first_outbreak')
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else 'validation_figures_phaseD')
OUT.mkdir(parents=True, exist_ok=True)
LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
MATRIX = {L: ('school_class_contacts_matrix' if L == 'school' else f'{L}_contacts_matrix') for L in LAYERS}
REF, SIM = '#59A14F', '#3a7ca5'

# ------------------------------------------------------------------ load the simulation
runs = []
for f in sorted(glob.glob(str(SPREAD / 'course_of_disease_data_*.npy')),
                key=lambda p: int(Path(p).stem.split('_')[-1])):
    k = int(Path(f).stem.split('_')[-1])
    runs.append({'course': list(np.load(f, allow_pickle=True)),
                 'contact': list(np.load(SPREAD / f'contact_data_{k}.npy', allow_pickle=True)),
                 'social': list(np.load(SPREAD / f'social_data_{k}.npy', allow_pickle=True)),
                 'demo': list(np.load(SPREAD / f'demographic_data_{k}.npy', allow_pickle=True))})
course = [c for r in runs for c in r['course']]
contact = [c for r in runs for c in r['contact']]
social = [s for r in runs for s in r['social']]
demo = [d for r in runs for d in r['demo']]
index_course = [r['course'][0] for r in runs]
index_contact = [r['contact'][0] for r in runs]
index_social = [r['social'][0] for r in runs]
index_demo = [r['demo'][0] for r in runs]
print(f'{SPREAD}: {len(runs)} simulations, {len(course)} cases')


def field(key, source):
    return np.array([x.get(key, np.nan) for x in source], dtype=float)


def layer_sar(layer, cases):
    candidate = effective = 0
    for c in cases:
        eff = list(c[f'{layer}_effective_contacts'] or [])
        candidate += len(eff)
        effective += sum(1 for x in eff if x == 1)
    return 100 * effective / candidate if candidate else np.nan


def daily_contacts(layer):
    per_day = []
    for c, k in zip(index_contact, index_course):
        m = np.asarray(c[MATRIX[layer]], dtype=float)
        if m.size == 0 or m.shape[1] == 0:
            per_day.append(0.0)
            continue
        onset = k['incubation_period']
        days = m.shape[1] if (onset is None or np.isnan(onset)) else int(min(m.shape[1], max(onset, 1)))
        per_day.append(m[:, :days].sum() / max(days, 1))
    return float(np.mean(per_day))


def contact_bins(layers, only_symptomatic=True):
    """Share of contacts whose first contact falls in each of Cheng's six day bins."""
    counts = np.zeros(6)
    for c, k in zip(index_contact, index_course):
        onset = k['incubation_period']
        if only_symptomatic and (onset is None or np.isnan(onset)):
            continue
        for layer in layers:
            m = np.asarray(c[MATRIX[layer]], dtype=float)
            if m.size == 0:
                continue
            first = np.argmax(m > 0, axis=1) - onset
            counts += [(first < 0).sum(), ((first >= 0) & (first <= 3)).sum(),
                       ((first >= 4) & (first <= 5)).sum(), ((first >= 6) & (first <= 7)).sum(),
                       ((first >= 8) & (first <= 9)).sum(), (first > 9).sum()]
    return 100 * counts / max(counts.sum(), 1)


ages = field('age', index_demo)
classes = field('school_class_size', index_social)
enterprise = field('enterprise_size', index_social)
others = field('household_size', index_social)
incubation = field('incubation_period', index_course)
latent = field('latent_period', index_course)
infectious = field('infectious_period', index_course)
positive = field('positive_test_date', index_course)
infection_day = field('infection_day', index_course)
recovery = field('date_of_recovery', index_course) - infection_day
symptomatic = ~np.isnan(incubation)
icu = ~np.isnan(field('date_of_critically_ill', index_course))
dead = ~np.isnan(field('date_of_death', index_course))
community = np.array([len(c['municipality_effective_contacts'] or []) for c in index_contact], dtype=float)
coworker = np.array([len(c['workplace_effective_contacts'] or []) for c in index_contact], dtype=float)
medical = np.array([len(c['health_care_effective_contacts'] or []) for c in index_contact], dtype=float)
offspring = np.array([sum(np.nansum(np.asarray(c[f'{L}_effective_contacts'], dtype=float)) for L in LAYERS)
                      for c in index_contact], dtype=float)
k_hat = offspring.mean() ** 2 / (offspring.var(ddof=1) - offspring.mean()) \
    if offspring.var(ddof=1) > offspring.mean() else np.inf

# age risk ratio, measured over every candidate contact
contact_ages, infected_ages = [], []
for c in index_contact:
    for L in LAYERS:
        contact_ages.extend(np.asarray(c[f'{L}_contact_ages'], dtype=float).tolist())
        infected_ages.extend([a for a in np.asarray(c[f'{L}_secondary_contact_ages'], dtype=float)
                              if np.isfinite(a)])
contact_ages, infected_ages = np.array(contact_ages), np.array(infected_ages)
rr = []
for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 200)]:
    denominator = ((contact_ages >= lo) & (contact_ages < hi)).sum()
    rr.append(((infected_ages >= lo) & (infected_ages < hi)).sum() / denominator if denominator else np.nan)
rr = np.array(rr) / rr[1]

hh_bins = contact_bins(['household'])
hc_bins = contact_bins(['health_care'])

# Generation time: days from the infector being infected to the infectee being infected. The
# contact record stores exactly that offset for every effective contact.
generation = np.array([float(t) for c in index_contact for L in LAYERS
                       for e, t in zip(list(c[f'{L}_effective_contacts'] or []),
                                       list(c[f'{L}_effective_contacts_infection_time'] or []))
                       if e == 1 and np.isfinite(t)], dtype=float)

first_cases = first_deaths = np.nan
if FIRST.exists():
    sizes, deaths = [], []
    for f in sorted(glob.glob(str(FIRST / 'course_of_disease_data_*.npy'))):
        items = list(np.load(f, allow_pickle=True))
        sizes.append(len(items))
        deaths.append(sum(1 for c in items if not np.isnan(c['date_of_death'])))
    if sizes:
        first_cases, first_deaths = float(np.mean(sizes)), float(np.mean(deaths))

tracing = pd.read_csv('validation_reference/taiwan_tracing_contacts_per_case.csv')


def observed_layer(layer):
    x = tracing[tracing['layer'] == layer]
    return x[x['uninfected'].notna()]['total'].to_numpy(dtype=float)


def infections_per_index(layer):
    """How many people the average index case infects in this layer (E56)."""
    infected = 0
    for c in index_contact:
        eff = c.get(layer + '_effective_contacts')
        if eff is None:
            continue
        infected += sum(1 for x in eff if x == 1)
    return infected / len(index_contact) if index_contact else np.nan


# ------------------------------------------------------------------ the comparison table
# (section, label, observed, CovSyn, unit, source)
rows = [
    ('Social structure', 'household members other than the case', 2.78, others.mean(), '', 'MOI 2021, person-weighted'),
    ('Social structure', 'living alone', 13.5, 100 * np.mean(others == 0), '%', 'MOI 2021'),
    ('Social structure', 'class size, elementary', 24.09,
     classes[(ages >= 7) & (ages <= 12) & (classes > 0)].mean(), '', 'MOE, student-weighted'),
    ('Social structure', 'class size, junior high', 27.29,
     classes[(ages >= 13) & (ages <= 15) & (classes > 0)].mean(), '', 'MOE'),
    ('Social structure', 'class size, senior high', 32.59,
     classes[(ages >= 16) & (ages <= 18) & (classes > 0)].mean(), '', 'MOE'),
    ('Social structure', 'class size, university', 88.33,
     classes[(ages >= 19) & (ages <= 22) & (classes > 0)].mean(), '', 'MOE'),
    ('Social structure', 'enterprise size of an employed case', 141.9,
     enterprise[enterprise > 0].mean(), '', 'census 2016, employee-weighted'),

    ('Contacts', 'close contacts per day, all settings', 6.03,
     sum(daily_contacts(L) for L in LAYERS), '', 'national survey 2020'),
    ('Contacts', 'close contacts per day, household', 1.86, daily_contacts('household'), '', 'survey x Fu 2012'),
    ('Contacts', 'close contacts per day, school', 1.16, daily_contacts('school'), '', 'survey x Fu 2012'),
    ('Contacts', 'close contacts per day, workplace', 1.49, daily_contacts('workplace'), '', 'survey x Fu 2012'),
    ('Contacts', 'close contacts per day, community', 1.49, daily_contacts('municipality'), '', 'survey x Fu 2012'),
    ('Contacts', 'community contacts per case, median', 7.0,
     float(np.median(community[community > 0])), '', 'CDC tracing, n=81'),
    ('Contacts', 'community contacts per case, p90', 172.0,
     float(np.percentile(community[community > 0], 90)), '', 'CDC tracing, n=81'),
    ('Contacts', 'coworker contacts per case, median', 2.5,
     float(np.median(coworker[coworker > 0])) if (coworker > 0).any() else np.nan, '', 'CDC tracing, n=6'),
    ('Contacts', 'medical contacts per case, median', 10.0,
     float(np.median(medical[medical > 0])) if (medical > 0).any() else np.nan, '', 'CDC tracing, n=8'),

    ('Transmission', 'attack rate, household', 100 * LAYER_CUMULATIVE_SAR['household'][1],
     layer_sar('household', index_contact), '%', 'calibration anchor (B1, Cheng 2020)'),
    ('Transmission', 'attack rate, school', 2.38, layer_sar('school', index_contact), '%', 'Huang 2021 classroom'),
    ('Transmission', 'attack rate, workplace', 100 * LAYER_CUMULATIVE_SAR['workplace'][1],
     layer_sar('workplace', index_contact), '%', 'calibration anchor'),
    ('Transmission', 'attack rate, health care', 100 * LAYER_CUMULATIVE_SAR['health_care'][1],
     layer_sar('health_care', index_contact), '%', 'calibration anchor (E55, Cheng 2020)'),
    ('Transmission', 'attack rate, community', 100 * LAYER_CUMULATIVE_SAR['municipality'][1],
     layer_sar('municipality', index_contact), '%', 'calibration anchor (E55, Cheng 2020 "others")'),
    # E56: the per-contact rate above is a ratio whose denominator the model chooses, so these
    # three rows are the ones to read. They are what Cheng actually counted -- 10, 6 and 1
    # infections over 100 index cases -- and they are what the objective charges since run 6.
    ('Transmission', 'infections per index case, household', LAYER_INFECTIONS_PER_INDEX['household'][1],
     infections_per_index('household'), '', 'Cheng 2020: 10 per 100 index cases'),
    ('Transmission', 'infections per index case, health care', LAYER_INFECTIONS_PER_INDEX['health_care'][1],
     infections_per_index('health_care'), '', 'Cheng 2020: 6 per 100 index cases'),
    ('Transmission', 'infections per index case, community', LAYER_INFECTIONS_PER_INDEX['municipality'][1],
     infections_per_index('municipality'), '', 'Cheng 2020: 1 per 100 index cases'),
    ('Transmission', 'household contacts starting before onset', 66.2, hh_bins[0], '%', 'Cheng 2020'),
    ('Transmission', 'medical contacts starting 8+ days after onset', 36.7, hc_bins[4] + hc_bins[5], '%', 'Cheng 2020'),
    # E70: Cheng's 0-19 ratio rests on one infection among 281 contacts, so the comparison
    # point is the multi-source range instead (Zhang 2020 0.34, Viner 2021 0.56, Uthman 2024
    # wild-type 0.58, Madewell 2020 0.59; midpoint 0.50). See todolist 1.13 and decision N5.
    ('Transmission', 'age risk ratio 0-19 (20-39 = 1)', 0.50, rr[0], '',
     'Zhang/Viner/Uthman/Madewell, range 0.34-0.77 (NOT Cheng: his 0-19 is 1/281)'),
    ('Transmission', 'age risk ratio 40-59', 1.83, rr[2], '', 'Cheng 2020'),
    ('Transmission', 'age risk ratio 60+', 1.32, rr[3], '', 'Cheng 2020'),
    ('Transmission', 'offspring dispersion k', 0.29, k_hat, '', 'CDC tracing, negative binomial MLE'),
    ('Transmission', 'cases infecting 3 or more', 4.3, 100 * np.mean(offspring >= 3), '%', 'CDC tracing'),

    ('Disease course', 'asymptomatic share', 23.7, 100 * np.mean(~symptomatic), '%', 'CDC tracing, 579 cases'),
    ('Disease course', 'symptomatic to ICU', 12.7, 100 * icu.sum() / max(symptomatic.sum(), 1), '%', 'CDC tracing'),
    ('Disease course', 'ICU to death', 12.5, 100 * dead.sum() / max(icu.sum(), 1), '%', 'CDC tracing'),
    ('Disease course', 'case fatality', 1.2, 100 * dead.mean(), '%', 'CDC tracing'),
    ('Disease course', 'latent period', 4.8, latent.mean(), 'd', 'literature reported means 4.1-5.5'),
    ('Disease course', 'incubation period', 5.95, np.nanmean(incubation), 'd', 'literature 3.9-8.0'),
    ('Disease course', 'infectious period', 11.7, infectious.mean(), 'd', 'literature 3.45-20'),
    # B2 (2026-09-25): 5-7 days, median 6, from Taiwan's own tracing file (n=442), Ge 2021
    # (5 d) and the published CovSyn fit (8.17 d). The 1.0 here was the pre-B2 figure (E33),
    # Taiwan's POST-March-2020 number, and it made the model's correct 5 days look 5x wrong.
    ('Disease course', 'onset to confirmation, median', 6.0,
     float(np.nanmedian((positive - infection_day - incubation)[symptomatic])), 'd', 'Taiwan tracing median, n=442 (B2)'),
    ('Disease course', 'infection to case closure', 25.0, float(np.nanmean(recovery[symptomatic])), 'd',
     'Taiwan, onset to release ~25 d'),
    ('Disease course', 'generation time', 4.05, generation.mean() if len(generation) else np.nan, 'd',
     'literature 2.9-5.2'),

    ('First wave', 'cumulative cases from 28 seeds', 55.0, first_cases, '', 'observed Taiwan first wave'),
    ('First wave', 'deaths', 3.0, first_deaths, '', 'observed Taiwan first wave'),
]
rows = [r for r in rows if np.isfinite(r[3])]

# ------------------------------------------------------------------ draw
fig, ax = plt.subplots(figsize=(15, 0.42 * len(rows) + 3))
y = 0
ticks, labels, section_lines = [], [], []
current = None
for section, label, obs, sim, unit, source in rows:
    if section != current:
        if current is not None:
            section_lines.append(y + 0.5)
            y += 1
        current = section
        ax.text(0.013, y, section, fontsize=12, fontweight='bold', va='center')
        y += 1
    ratio = sim / obs if obs else np.nan
    ax.plot([1, ratio], [y, y], color='0.75', lw=1.5, zorder=1)
    ax.scatter([1], [y], color=REF, s=70, zorder=3, edgecolor='black', linewidth=0.5)
    ax.scatter([ratio], [y], color=SIM, s=70, zorder=3, edgecolor='black', linewidth=0.5)
    ticks.append(y)
    labels.append(label)
    ax.text(9.3, y, f'{obs:g}{unit}  vs  {sim:.2f}{unit}', fontsize=8, va='center', ha='right')
    ax.text(9.6, y, source, fontsize=7, va='center', ha='left', color='0.35')
    y += 1

ax.axvspan(0.8, 1.25, color=REF, alpha=0.10, zorder=0)
ax.axvline(1, color=REF, lw=1.5, zorder=2)
for line in section_lines:
    ax.axhline(line, color='0.9', lw=0.8)
ax.set_xscale('log')
ax.set_xlim(0.02, 30)
ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 25])
ax.set_xticklabels(['1/20', '1/10', '1/4', '1/2', 'match', '2x', '4x', '10x', '25x'])
ax.set_ylim(y, -1.2)
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('CovSyn divided by the observation (log scale) — green line = the observation, '
              'shaded band = within 25%', fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_title(f'Every validated quantity: Taiwan observation vs CovSyn ({SPREAD.name}, '
             f'{len(runs):,} simulations, {len(course):,} cases)', fontsize=14, fontweight='bold', pad=15)
ax.scatter([], [], color=REF, s=70, edgecolor='black', linewidth=0.5, label='observed / reference')
ax.scatter([], [], color=SIM, s=70, edgecolor='black', linewidth=0.5, label='CovSyn')
ax.legend(loc='lower right', fontsize=10)
fig.text(0.01, 0.005,
         'Every row is one comparison and both halves are the SAME quantity measured the same way. Sources: Ministry of the Interior '
         '2021 household structure; Ministry of Education enrolment files; 2016 industry and service census; the 2020 national contact '
         'survey split by the setting composition of Fu 2012; Taiwan CDC contact tracing (taiwan_covid_figshare.xlsx, the records where '
         'uninfected contacts were also reported); Cheng et al. 2020 (JAMA Intern Med); Huang, Tu & Lai 2021; and the literature '
         'reported-mean ranges for the disease-course durations, whose midpoint is used as the observation. Tracing samples are small '
         'and not random (n = 6 to 81), so the contact rows are exploratory. The attack-rate rows use index cases only, matching the '
         'design of the tracing studies.', fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.035, 1, 1])
target = OUT / 'fig_reality_vs_covsyn_scorecard.png'
fig.savefig(target, dpi=150)
print('saved', target)
worst = sorted(rows, key=lambda r: -abs(np.log(max(r[3] / r[2], 1e-9))) if r[2] else 0)[:8]
print('\nfurthest from the observation:')
for section, label, obs, sim, unit, _ in worst:
    print('  %-52s observed %8.2f   CovSyn %8.2f   ratio %6.2f' % (label, obs, sim, sim / obs))
