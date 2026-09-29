"""Every quantity for which BOTH sides have a distribution, drawn as split violins.

Left half of each violin is the Taiwan observation, right half is CovSyn, and both halves are
the same quantity measured the same way. Quantities where the only reference is a single
number or a literature range are not here -- they are in the scorecard figure instead, because
half a violin against a point estimate invites reading a spread that was never observed.

  (a) social contexts: the household, class and enterprise a case belongs to, against the
      census distributions those are drawn from
  (b) contacts per case in each layer, against the contact-tracing records
  (c) course-of-disease durations that the tracing file records dates for
  (d) secondary infections per case and the age of the cases

Usage: python -m covsyn.figures.plot_violin_reality_vs_covsyn [spread_dir] [out_dir] [case_timeline_csv]
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from covsyn.data_processing.taiwan_reference import (LEVELS, MARKER_NOTE, REF_COLOR, SIM_COLOR, household_pmfs,
                              load_demographics, pmf_samples, school_classmates_all,
                              school_reference, split_violins, tracing_contacts,
                              workplace_establishment_distributions)

SPREAD = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'validation_figures_phaseD')
# The per-case dates and age bands, extracted from the workbook by extract_tracing_reference.py
# so this script needs nothing but the CSV.
TIMELINE = Path(sys.argv[3] if len(sys.argv) > 3 else 'validation_reference/taiwan_case_timeline.csv')
OUT.mkdir(parents=True, exist_ok=True)
LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
LABEL = {'household': 'Household', 'school': 'School', 'workplace': 'Workplace',
         'health_care': 'Health care', 'municipality': 'Community'}

# ------------------------------------------------------------------ CovSyn
runs = []
for f in sorted(glob.glob(str(SPREAD / 'course_of_disease_data_*.npy')),
                key=lambda p: int(Path(p).stem.split('_')[-1])):
    k = int(Path(f).stem.split('_')[-1])
    runs.append((list(np.load(f, allow_pickle=True)),
                 list(np.load(SPREAD / f'contact_data_{k}.npy', allow_pickle=True)),
                 list(np.load(SPREAD / f'social_data_{k}.npy', allow_pickle=True)),
                 list(np.load(SPREAD / f'demographic_data_{k}.npy', allow_pickle=True))))
index_course = [r[0][0] for r in runs]
index_contact = [r[1][0] for r in runs]
index_social = [r[2][0] for r in runs]
index_demo = [r[3][0] for r in runs]
print(f'{SPREAD}: {len(runs)} simulations')


def field(key, source):
    return np.array([x.get(key, np.nan) for x in source], dtype=float)


contacts = {L: np.array([len(c[f'{L}_effective_contacts'] or []) for c in index_contact], dtype=float)
            for L in LAYERS}
offspring = np.array([sum(np.nansum(np.asarray(c[f'{L}_effective_contacts'], dtype=float)) for L in LAYERS)
                      for c in index_contact], dtype=float)
ages = field('age', index_demo)
classes = field('school_class_size', index_social)
enterprise = field('enterprise_size', index_social)
others = field('household_size', index_social)
incubation = field('incubation_period', index_course)
infection_day = field('infection_day', index_course)
onset_to_confirm = field('positive_test_date', index_course) - infection_day - incubation
closure = field('date_of_recovery', index_course) - infection_day - incubation

# ------------------------------------------------------------------ Taiwan observations
DEMO = load_demographics()
sizes, _, _, pmf_person = household_pmfs(DEMO)
tw_household = pmf_samples(sizes - 1, pmf_person)
try:
    tw_class = school_classmates_all(school_reference()) + 1      # class size including the student
except Exception as exc:
    print('WARNING: school reference unavailable:', repr(exc))
    tw_class = None
size_values, _, pmf_worker = workplace_establishment_distributions(DEMO)
tw_enterprise = pmf_samples(size_values, pmf_worker)

timeline = pd.read_csv(TIMELINE)
timeline = timeline[timeline['dataset'] == 'first_wave_2020']


def duration(column, lo=-30, hi=200):
    x = timeline[column].dropna()
    return x[(x >= lo) & (x <= hi)].to_numpy(dtype=float)


def share_negative(x):
    """Cases confirmed before their symptoms started -- found by screening or by tracing."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return 100 * np.mean(x < 0) if len(x) else np.nan


tw_incubation = duration('infection_to_onset')
tw_onset_to_confirm = duration('onset_to_confirmation')
events = pd.read_csv('validation_reference/taiwan_infection_events.csv')
events = events[events['dataset'] == 'first_wave_2020']
per_source = events.groupby('source_id').size()
tw_offspring = np.zeros(len(timeline))
tw_offspring[:len(per_source)] = per_source.to_numpy()      # everyone else infected nobody on record
tw_age = timeline['age_band_midpoint'].dropna().to_numpy(dtype=float)

tracing = {}
for layer in LAYERS:
    values, _, has_uninfected = tracing_contacts(layer)
    tracing[layer] = values[has_uninfected] if values is not None else np.array([])

# ------------------------------------------------------------------ draw
fig, ax = plt.subplots(4, 1, figsize=(16, 24))

split_violins(ax[0], [
    ('Household\nmembers other than the case', tw_household, others, 'MOI 2021, size-weighted'),
    ('School\nclass size incl. the student', tw_class, classes[classes > 0], 'MOE, student-weighted'),
    ('Workplace\nstaff of the enterprise', tw_enterprise, enterprise[enterprise > 0], 'census 2016, employee-weighted'),
], 'Taiwan census', 'CovSyn', REF_COLOR, '#9ecae1', 'people in the context')
ax[0].set_title('(a) The social contexts a case belongs to', fontsize=12, fontweight='bold')

split_violins(ax[1], [(LABEL[L],
                       tracing[L] if len(tracing[L]) else None,
                       contacts[L][contacts[L] > 0] if (contacts[L] > 0).any() else contacts[L],
                       f'tracing n={len(tracing[L])}') for L in LAYERS],
              'Taiwan contact tracing', 'CovSyn', REF_COLOR, SIM_COLOR, 'contacts of one case')
ax[1].set_title('(b) Contacts per case, by layer — tracing samples are small and not random',
                fontsize=12, fontweight='bold')

split_violins(ax[2], [
    ('Incubation\ninfection to onset', tw_incubation, incubation[~np.isnan(incubation)],
     f'tracing n={len(tw_incubation)}'),
    ('Onset to confirmation', tw_onset_to_confirm, onset_to_confirm[~np.isnan(onset_to_confirm)],
     f'tracing n={len(tw_onset_to_confirm)}; confirmed BEFORE onset: '
     f'observed {share_negative(tw_onset_to_confirm):.0f}%, CovSyn {share_negative(onset_to_confirm):.0f}% '
     '(drawn at 0 on a log axis)'),
], 'Taiwan contact tracing', 'CovSyn', REF_COLOR, '#E8A33D', 'days')
ax[2].set_title('(c) Course of disease, for the two durations the tracing file dates both ends of',
                fontsize=12, fontweight='bold')

split_violins(ax[3], [
    ('Secondary infections\nper case', tw_offspring, offspring, f'tracing n={len(tw_offspring)}, first wave'),
    ('Age of the case', tw_age, ages, f'tracing n={len(tw_age)}'),
], 'Taiwan contact tracing', 'CovSyn', REF_COLOR, SIM_COLOR, 'count / years')
ax[3].set_title('(d) Who gets infected and how many they infect', fontsize=12, fontweight='bold')

fig.suptitle(f'Taiwan observation (left half) vs CovSyn (right half) — {SPREAD.name}, {len(runs):,} simulations',
             fontsize=15, fontweight='bold')
fig.text(0.01, 0.004, MARKER_NOTE + ' Both halves of every violin are the same quantity. Panel (a): the distributions CovSyn '
         'samples its contexts from, so agreement here is a check that the sampling is person-weighted, not independent evidence. '
         'Panel (b): close contacts per index case recorded in Taiwan CDC contact tracing, counting only the cases where the '
         'uninfected contacts were also reported, against CovSyn cases with at least one contact in that layer. Panel (c): dates from '
         'the same tracing file (earliest infection date to onset; onset to confirmation) -- note that the tracing median for onset to '
         'confirmation is 6 days over the whole file, which is the period Cheng 2020 covers, not the 1 day recorded for after March '
         '2020. Panel (d): secondary infections counted from the recorded infector of each first-wave case, so it counts only onward '
         'infections that were actually detected, and case age is the mid-point of the reported age band.', fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.975])
target = OUT / 'fig_violin_reality_vs_covsyn.png'
fig.savefig(target, dpi=140)
print('saved', target)
for name, obs, sim in [('household members', tw_household, others),
                       ('contacts, community', tracing['municipality'], contacts['municipality']),
                       ('incubation', tw_incubation, incubation[~np.isnan(incubation)]),
                       ('onset to confirmation', tw_onset_to_confirm, onset_to_confirm[~np.isnan(onset_to_confirm)]),
                       ('secondary infections', tw_offspring, offspring),
                       ('case age', tw_age, ages)]:
    if obs is None or not len(obs):
        continue
    print('%-24s observed median %6.1f mean %7.2f | CovSyn median %6.1f mean %7.2f'
          % (name, np.median(obs), obs.mean(), np.median(sim), sim.mean()))
