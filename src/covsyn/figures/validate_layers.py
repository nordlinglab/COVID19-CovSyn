"""Phase A validation (todo.md sections 2.1, 2.2 and 3) on the latest CovSyn dataset
(run5 firefly parameters): contacts per social layer, compared with Taiwan reference data.

Every figure uses split violins with one violin per social layer: the LEFT half is the
observed / reference distribution and the RIGHT half is the CovSyn simulation.

Definitions follow plot_diagnostics.py:
  (1) context size       social_data size field (household_size, school_class_size,
                         work_group_size, clinic_size); the community layer has none (B27)
  (2) candidate contacts len({layer}_effective_contacts): people contacted on at least one
                         day between infection and monitored isolation
  (3) effective contacts entries of {layer}_effective_contacts equal to 1 (infections)

Usage: python -m covsyn.figures.validate_layers [spread_dir] [out_dir] [index_case_dir]
"""
import csv
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from covsyn.data_processing.taiwan_reference import (EFF_COLOR, IMPORTED_COLOR, LEVELS, MARKER_NOTE, REF_COLOR, RNG, SIM_COLOR,
                              half_points, half_violin, household_pmfs, load_demographics,
                              municipality_population, pmf_samples, school_classmates_all, school_reference,
                              split_violins, to_axis, weighted_samples, workplace_establishment_distributions,
                              write_household_pmf_csv)

SPREAD = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight_MC1000'
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'validation_figures_phaseD')
# Optional third argument: a separate dataset to take the index cases from. Without it the
# index cases are the first case of every spread simulation, which is what they are.
INDEX = sys.argv[3] if len(sys.argv) > 3 else None
PARAM = Path('Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')
OUT.mkdir(exist_ok=True)
for superseded in ('fig_household_candidate_contacts.png', 'fig_school_candidate_contacts.png'):
    (OUT / superseded).unlink(missing_ok=True)
LAYERS = [('household', 'household_size', 'household_contacts_matrix'),
          ('school', 'school_class_size', 'school_class_contacts_matrix'),
          ('workplace', 'work_group_size', 'workplace_contacts_matrix'),
          ('health_care', 'clinic_size', 'health_care_contacts_matrix'),
          ('municipality', None, 'municipality_contacts_matrix')]
LAYER_LABEL = {'household': 'Household', 'school': 'School', 'workplace': 'Workplace',
               'health_care': 'Health care', 'municipality': 'Municipality'}
DEMO = load_demographics()
MUNICIPALITY_POP = municipality_population(DEMO)
LIT_COLOR = '#B07AA1'
# Huang, Tu & Lai 2021 (J Microbiol Immunol Infect) Table 1: close contacts of the two workplace case series
HUANG_WORKPLACE = [41.0, 24.0]      # case 160 (3 secondary cases), case 277 (2, workplace + household)


# --------------------------------------------------------------------------- loading
def sim_index(path):
    return int(Path(path).stem.split('_')[-1])


def load_cases(result_dir):
    """One row per infected case: age, municipality and per-layer (1)/(2)/(3)."""
    rows = []
    for f in sorted(glob.glob(f'{result_dir}/social_data_*.npy'), key=sim_index):
        k = sim_index(f)
        social = np.load(f, allow_pickle=True)
        contact = np.load(f'{result_dir}/contact_data_{k}.npy', allow_pickle=True)
        demographic = np.load(f'{result_dir}/demographic_data_{k}.npy', allow_pickle=True)
        course = np.load(f'{result_dir}/course_of_disease_data_{k}.npy', allow_pickle=True)
        for position, (s, c, g, cd) in enumerate(zip(social, contact, demographic, course)):
            recovery = cd.get('date_of_recovery')
            recovers = recovery is not None and np.isfinite(float(recovery)) and float(recovery) >= 0
            row = {'age': g.get('age'), 'municipality': s.get('municipality'),
                   'municipality_size': MUNICIPALITY_POP.get(s.get('municipality'), np.nan),
                   'enterprise_size': float(s.get('enterprise_size') or 0),
                   'recovers': bool(recovers), 'position_in_sim': position}
            for layer, size_key, matrix_key in LAYERS:
                eff = c.get(layer + '_effective_contacts')
                eff = [] if eff is None else list(eff)
                matrix = c.get(matrix_key)
                row[layer + '_rows'] = matrix.shape[0] if isinstance(matrix, np.ndarray) and matrix.ndim == 2 else 0
                # B27: the community layer no longer draws its contacts from the city
                # population, so it has no context size at all. The city population is still
                # recorded, but only to show that the contact count is now independent of it
                # (finding E4), never as an opportunity set.
                size = np.nan if size_key is None else s.get(size_key)
                row[layer + '_size'] = np.nan if size is None else float(size)
                row[layer + '_cand'] = len(eff)
                row[layer + '_eff'] = int(sum(1 for x in eff if x == 1))
            rows.append(row)
    return pd.DataFrame(rows)


def summary_row(name, x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [name, 0] + [''] * 6
    return [name, len(x), round(x.mean(), 3), round(float(np.median(x)), 3),
            round(float(np.percentile(x, 25)), 3), round(float(np.percentile(x, 75)), 3),
            round(float(np.percentile(x, 95)), 3), round(float((x == 0).mean()), 3)]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


# --------------------------------------------------------------------------- data
spread = load_cases(SPREAD)
index = load_cases(INDEX) if INDEX else spread[spread['position_in_sim'] == 0].reset_index(drop=True)
index_source = f'from {INDEX}' if INDEX else 'first case of each simulation'
print(f'spread dataset {SPREAD}: {len(spread)} cases | index cases ({index_source}): {len(index)} cases')
for label, df in (('spread', spread), ('index', index)):
    for layer, _, _ in LAYERS:
        bad = df[df[layer + '_rows'] != df[layer + '_cand']]
        if len(bad):
            print(f'  {label:6s} {layer:12s} rows!=len(effective) {len(bad):5d} | effective empty {int((bad[layer + "_cand"] == 0).sum()):5d}'
                  f' | never recovers {int((~bad["recovers"]).sum()):5d} | first case of simulation {int((bad["position_in_sim"] == 0).sum()):5d}')

students = spread[spread['school_size'] > 0]
workers = spread[spread['workplace_size'] > 0]
table = [['quantity', 'n', 'mean', 'median', 'p25', 'p75', 'p95', 'share_zero']]

# --------------------------------------------------------------------------- references
sizes, pmf_input, pmf_household, pmf_person = household_pmfs(DEMO)
tw_household_person = pmf_samples(sizes - 1, pmf_person)
tw_household_household = pmf_samples(sizes - 1, pmf_household)
write_household_pmf_csv(DEMO, OUT / 'reference_household_pmf.csv')


try:
    school_ref = school_reference()
    tw_school_all = school_classmates_all(school_ref)
except Exception as exc:          # keep the figures buildable if a raw file changes format
    print('WARNING: raw school reference failed:', repr(exc))
    school_ref, tw_school_all = {}, None

# The workplace reference a random WORKER experiences: the 2016 industry and service census
# weighted by employees, which is exactly what B31 makes the model draw its enterprise size
# from, so panel (a) can compare them directly instead of pointing at another figure.
size_values, pmf_establishment, pmf_worker = workplace_establishment_distributions(DEMO)
tw_per_worker = pmf_samples(size_values, pmf_worker)

# --------------------------------------------------------------------------- figure 1
fig, axes = plt.subplots(2, 1, figsize=(17, 14))
split_violins(axes[0], [
    ('Household', tw_household_person, spread['household_size'], 'household members other than the case'),
    ('School', tw_school_all, students['school_size'] - 1, 'classmates, students only'),
    ('Workplace enterprise', tw_per_worker, workers['enterprise_size'], 'staff of the enterprise;\nTaiwan: census weighted by employees'),
    ('Workplace group', 'no Taiwan data\n(size fixed from\nChen 2022)', workers['workplace_size'], 'work group actually met'),
    ('Health care', None, spread['health_care_size'], 'clinic daily patient volume'),
], 'Taiwan', 'CovSyn (1) context size', REF_COLOR, '#9ecae1', 'people who could be contacted')
axes[0].set_title('(a) Opportunity set: Taiwan reference vs CovSyn context size. The workplace is now two-stage '
                  '(enterprise, then the group met inside it, B31); the community layer has no context size at all '
                  'since it stopped drawing from the city population (B27)', fontsize=11)
split_violins(axes[1], [
    ('Household', tw_household_person, spread['household_cand'], ''),
    ('School', tw_school_all, students['school_cand'], 'students only'),
    ('Workplace', 'see workplace\ncandidate-contact\nexternal comparison', workers['workplace_cand'], 'employed only'),
    ('Health care', 'see health care\ncandidate-contact\nexternal comparison', spread['health_care_cand'], ''),
    ('Municipality', 'see municipality\ncandidate-contact\nexternal comparison', spread['municipality_cand'], ''),
], 'Taiwan', 'CovSyn (2) candidate contacts', REF_COLOR, SIM_COLOR, 'people')
axes[1].set_title('(b) Taiwan reference (possible contacts) vs CovSyn candidate contacts', fontsize=11)
fig.suptitle(f'Contacts by social layer — Taiwan reference vs CovSyn ({SPREAD}, {len(spread):,} cases)',
             fontsize=13, fontweight='bold')
fig.text(0.01, 0.004, MARKER_NOTE + ' Household reference: household structure by city (Ministry of the Interior, 2021), '
         'household size experienced by a random person (size-weighted), cities weighted by population. School reference: '
         'Ministry of Education enrolment files, students/classes per school-grade weighted by students; university = department-year '
         'cohort. Workplace: see the establishment-size sampling diagnostic and the candidate-contact external comparison figures. '
         'Health care and municipality candidate contacts: see their external comparison figures (Taiwan contact tracing, exploratory); '
         'no reference exists yet for the health care context size (clinic daily patient volume).', fontsize=7, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(OUT / 'fig_layers_reference_vs_covsyn.png', dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------- figure 2
if school_ref:
    fig, axes = plt.subplots(2, 1, figsize=(15, 14))
    groups_size, groups_cand = [], []
    for level, lo, hi in LEVELS:
        v, w = school_ref[level]
        ref = weighted_samples(v - 1, w)
        in_level = students[(students['age'] >= lo) & (students['age'] <= hi)]
        groups_size.append((level, ref, in_level['school_size'] - 1, ''))
        groups_cand.append((level, ref, in_level['school_cand'], ''))
        table.append(summary_row(f'school {level} Taiwan classmates per student', ref))
        table.append(summary_row(f'school {level} CovSyn (1) classmates', in_level['school_size'] - 1))
        table.append(summary_row(f'school {level} CovSyn (2) candidate', in_level['school_cand']))
    split_violins(axes[0], groups_size, 'Taiwan', 'CovSyn (1) classmates', REF_COLOR, '#9ecae1', 'classmates')
    axes[0].set_title('(a) Classmates per student: Taiwan vs CovSyn context size', fontsize=11)
    split_violins(axes[1], groups_cand, 'Taiwan', 'CovSyn (2) candidate contacts', REF_COLOR, SIM_COLOR, 'people')
    axes[1].set_title('(b) Classmates per student (Taiwan) vs CovSyn candidate school contacts', fontsize=11)
    fig.suptitle('School layer by education level', fontsize=13, fontweight='bold')
    fig.text(0.01, 0.004, MARKER_NOTE + ' CovSyn samples class size from school_p[age], a histogram over schools rather than '
             'students, and school_class_size includes the case (1 is subtracted here). Period: during the 2020 first wave the spring '
             'semester started on 25 February after a two-week delay and schools stayed open; the Ministry of Education closed a class '
             'for 14 days after one confirmed case and a school after two, so ordinary class sizes are the reference. The 84-day spread '
             'simulation has no calendar date and no closure mechanism.', fontsize=7, wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT / 'fig_school_levels_reference_vs_covsyn.png', dpi=150)
    plt.close(fig)

# --------------------------------------------------------------------------- figure 3
layer_names = [layer for layer, _, _ in LAYERS]
subset = {'household': spread, 'school': students, 'workplace': workers, 'health_care': spread, 'municipality': spread}
total_infections = sum(int(spread[L + '_eff'].sum()) for L in layer_names)
groups = []
with open(OUT / 'transmission_per_candidate_contact.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['layer', 'candidate_total', 'effective_total', 'probability', 'ci_low', 'ci_high', 'share_of_infections'])
    for L in layer_names:
        k, n = int(spread[L + '_eff'].sum()), int(spread[L + '_cand'].sum())
        p, a, b = wilson(k, n)
        share = k / max(total_infections, 1)
        w.writerow([L, n, k, round(p, 6), round(a, 6), round(b, 6), round(share, 4)])
        groups.append((LAYER_LABEL[L], subset[L][L + '_cand'], subset[L][L + '_eff'],
                       f'infected per candidate {100 * p:.2f}% ({100 * a:.2f}-{100 * b:.2f})\nshare of infections {100 * share:.1f}%'))
fig, ax = plt.subplots(figsize=(17, 8))
split_violins(ax, groups, 'candidate (2)', 'effective (3)', SIM_COLOR, EFF_COLOR, 'people per case')
ax.set_title(f'CovSyn candidate vs effective contacts by social layer ({SPREAD}, {len(spread):,} cases)',
             fontsize=12, fontweight='bold')
fig.text(0.01, 0.004, 'Left half = CovSyn candidate contacts, right half = CovSyn effective contacts (infections). '
         'Red diamond = mean, white dot = median, black bar = interquartile range. '
         'Infection probability = total effective / total candidate contacts with 95% Wilson interval.', fontsize=7)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT / 'fig_layers_candidate_vs_effective.png', dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------- workplace (two figures, two purposes)
# (A) Workplace sampling diagnostic: does CovSyn draw the enterprise the way a WORKER meets
# it, and how much smaller is the group actually met inside it (B31)?
tw_per_establishment = pmf_samples(size_values, pmf_establishment)
fig, ax = plt.subplots(figsize=(15, 8.5))
split_violins(ax, [
    ('Enterprise size counted\nper ESTABLISHMENT\n(the rule CovSyn used before B31)',
     tw_per_establishment, workers['enterprise_size'], ''),
    ('Enterprise size counted\nper WORKER\n(the rule CovSyn uses now)',
     tw_per_worker, workers['enterprise_size'], ''),
    ('Work group actually met\ninside the enterprise\n(no Taiwan data, fixed from Chen 2022)',
     None, workers['workplace_size'], ''),
], 'Taiwan 2016 census', 'CovSyn', REF_COLOR, '#9ecae1', 'persons')
ax.set_title('Workplace sampling diagnostic: enterprise size, then the group met inside it (B31)',
             fontsize=13, fontweight='bold')
fig.text(0.01, 0.004, MARKER_NOTE + ' Purpose: check which census weighting the enterprise size follows and how far the work group '
         'falls below it. An infected person is a worker, not a company, so the per-worker curve is the one to match (finding E3). The '
         'work group is capped by a log-normal (median 10) that does not grow with the company, because Chen 2022 (BMJ Open '
         '12:e055643) found the number of cases per workplace cluster essentially constant (median 4) while the median staff count of '
         'the enterprises ran from 16 to 650. Taiwan: Industry and Service Census, enterprise units by number of persons employed, end '
         'of 2016 (Tables 18 and 18-2), persons spread uniformly within each size class as in parameters_for_initialization.py, '
         f'industries mixed by full-time employment share. CovSyn: employed cases in {SPREAD}.', fontsize=7, wrap=True)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT / 'fig_workplace_establishment_size_sampling_diagnostic.png', dpi=150)
plt.close(fig)
table.append(summary_row('workplace Taiwan 2016 establishment size, per establishment', tw_per_establishment))
table.append(summary_row('workplace Taiwan 2016 establishment size, per worker', tw_per_worker))
table.append(summary_row('workplace CovSyn enterprise size', workers['enterprise_size']))

# (B) Workplace candidate-contact external comparison: observed coworker close contacts vs CovSyn
coworker_csv = Path('validation_reference/taiwan_tracing_contacts_per_case.csv')
if coworker_csv.exists():
    observed = pd.read_csv(coworker_csv)
    observed = observed[observed['layer'] == 'workplace'].rename(columns={'total': 'total_coworker_close_contacts'})
    contacted = workers[workers['workplace_cand'] > 0]
    groups = []
    for dataset, label in [('first_wave_2020', 'Taiwan contact tracing\nJan-Nov 2020'),
                           ('extended_to_2021', 'Taiwan contact tracing\nJan 2020-Jan 2021')]:
        o = observed[observed['dataset'] == dataset]
        if o.empty:
            continue
        local = o['case_type'].str.lower() == 'local'
        colors = np.where(local, REF_COLOR, IMPORTED_COLOR)
        left = [('points', o['total_coworker_close_contacts'].to_numpy(float), colors, 'o'),
                ('points', HUANG_WORKPLACE, [LIT_COLOR] * len(HUANG_WORKPLACE), '^')]      # F7
        groups.append((label, left, contacted['workplace_cand'],
                       f'{int(local.sum())} local / {int((~local).sum())} imported index cases'))
        table.append(summary_row(f'workplace observed coworker close contacts {dataset}', o['total_coworker_close_contacts']))
    table.append(summary_row('workplace CovSyn (2) candidate, employed with >=1 workplace contact', contacted['workplace_cand']))
    fig, ax = plt.subplots(figsize=(14, 8.5))
    split_violins(ax, groups, 'observed', 'CovSyn (2) candidate', REF_COLOR, SIM_COLOR, 'coworkers per case')
    ax.legend(handles=[Patch(facecolor=REF_COLOR, edgecolor='black', label='left: contact tracing, local index case'),
                       Patch(facecolor=IMPORTED_COLOR, edgecolor='black', label='left: contact tracing, imported index case'),
                       Line2D([], [], marker='^', ls='none', markerfacecolor=LIT_COLOR, markeredgecolor='black', ms=8,
                              label='left: published cluster investigation (Huang 2021, cases 160 and 277)'),
                       Patch(facecolor=SIM_COLOR, edgecolor='black', label='right: CovSyn candidate contacts')],
              loc='upper left', fontsize=8)
    ax.set_title('Workplace candidate-contact external comparison — EXPLORATORY, LIMITED EVIDENCE',
                 fontsize=13, fontweight='bold')
    fig.text(0.01, 0.004, 'Left = coworker close contacts per index case recorded in Taiwan CDC contact tracing (infected + uninfected '
             'coworkers; index cases with at least one coworker record only; each point is one index case). Index cases 755-758 belong to '
             'one cluster and share one contact list, as do 745-748, so points are not independent. Right = CovSyn workplace candidate '
             f'contacts for employed cases with at least one workplace candidate contact (mean over all employed cases: '
             f'{workers["workplace_cand"].mean():.2f}). Red diamond = mean, white dot = median, black bar = interquartile range; the '
             'statistics use the contact-tracing points only. Triangles = the two workplace case series in Huang, Tu & Lai 2021 '
             '(Table 1: case 160 with 41 close contacts and 3 secondary cases, case 277 with 24 contacts and 2 secondary cases, '
             'workplace + household); these were selected for publication because transmission occurred, so they are biased upwards. '
             'Case 601 (3 coworkers) is recorded only in the press-release text and is included from 2026-09-21. '
             'Small, non-random samples: exploratory comparison, not a formal validation.', fontsize=7, wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT / 'fig_workplace_candidate_contact_external_comparison.png', dpi=150)
    plt.close(fig)
else:
    print('WARNING: missing', coworker_csv, '- run extract_tracing_reference.py first')

# --------------------------------------------------------------------------- health care and municipality external comparisons
tracing_csv = Path('validation_reference/taiwan_tracing_contacts_per_case.csv')
EXTERNAL = [('health_care', 'Health care', "'the same hospital'",
             'CovSyn health care context size is the daily patient volume of a clinic.'),
            ('municipality', 'Municipality', "'friend' and 'other (unknown) contact'",
             'The tracing category "other" pools contacts outside the listed relationships, the closest analogue of the municipality '
             'layer. CovSyn municipality candidate contacts scale with city population (see the municipality scaling figure).')]
if tracing_csv.exists():
    tracing = pd.read_csv(tracing_csv)
    for layer, name, columns, layer_note in EXTERNAL:
        contacted = spread[spread[layer + '_cand'] > 0]
        groups = []
        for dataset, label in [('first_wave_2020', 'Taiwan contact tracing\nJan-Nov 2020'),
                               ('extended_to_2021', 'Taiwan contact tracing\nJan 2020-Jan 2021')]:
            o = tracing[(tracing['dataset'] == dataset) & (tracing['layer'] == layer)]
            if o.empty:
                continue
            local = o['case_type'].str.lower() == 'local'
            values = o['total'].to_numpy(float)
            left = ('points', values, np.where(local, REF_COLOR, '#B6D7A8')) if len(o) < 30 else values   # D2
            groups.append((label, left, contacted[layer + '_cand'],
                           f'{int(local.sum())} local / {int((~local).sum())} imported index cases'))
            table.append(summary_row(f'{layer} observed tracing contacts {dataset}', values))
        table.append(summary_row(f'{layer} CovSyn (2) candidate, cases with >=1 {layer} contact', contacted[layer + '_cand']))
        fig, ax = plt.subplots(figsize=(14, 8.5))
        split_violins(ax, groups, 'observed', 'CovSyn (2) candidate', REF_COLOR, SIM_COLOR, 'contacts per case')
        ax.legend(handles=[Patch(facecolor=REF_COLOR, edgecolor='black', label='left: observed (density, or point = local index case)'),
                           Patch(facecolor='#B6D7A8', edgecolor='black', label='left: observed point, imported index case'),
                           Patch(facecolor=SIM_COLOR, edgecolor='black', label='right: CovSyn candidate contacts')],
                  loc='upper left', fontsize=8)
        ax.set_title(f'{name} candidate-contact external comparison — EXPLORATORY, LIMITED EVIDENCE',
                     fontsize=13, fontweight='bold')
        fig.text(0.01, 0.004, f'Left = {name.lower()} close contacts per index case recorded in Taiwan CDC contact tracing under '
                 f'{columns} (distinct infected case IDs + uninfected contacts; index cases with at least one record only; individual '
                 'points when fewer than 30 index cases, otherwise a density). The press releases do not report the tracing window or a '
                 'duration threshold. Index cases in one cluster can share one contact list, so observations are not independent. '
                 f'Right = CovSyn {name.lower()} candidate contacts for cases with at least one such contact (mean over all cases: '
                 f'{spread[layer + "_cand"].mean():.2f}). {layer_note} Red diamond = mean, white dot = median, black bar = interquartile '
                 'range. Small, non-random samples: exploratory comparison, not a formal validation.', fontsize=7, wrap=True)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(OUT / f'fig_{layer}_candidate_contact_external_comparison.png', dpi=150)
        plt.close(fig)
else:
    print('WARNING: missing', tracing_csv, '- run extract_tracing_reference.py first')

# --------------------------------------------------------------------------- figure 4
res = np.loadtxt(PARAM / 'firefly_best.txt')
P = res[int(np.argmin(res[:, -1])), 1:-1]
by_city = spread.groupby('municipality').agg(pop=('municipality_size', 'first'), cand=('municipality_cand', 'mean'))
by_city_index = index.groupby('municipality').agg(pop=('municipality_size', 'first'), cand=('municipality_cand', 'mean'))
r = np.corrcoef(by_city_index['pop'], by_city_index['cand'])[0, 1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(by_city_index['pop'] / 1e6, by_city_index['cand'], color='#1f4e79', label='index cases')
ax.scatter(by_city['pop'] / 1e6, by_city['cand'], color=SIM_COLOR, marker='x', label=f'spread cases ({len(spread):,})')
xs = np.linspace(0, max(MUNICIPALITY_POP.values()), 50)
# Before B27 the community layer drew Binomial(city population, p), so the expected number of
# contacts was a straight line through the origin -- a case in Taipei met twelve times as many
# people as one in Taitung purely because of where it lived (finding E4). The community layer
# now draws Poisson(nu * lambda), independent of the city, so the expectation is a flat line.
ax.axhline(P[28], color='gray', ls='--', label=f'model: lambda = {P[28]:.2f} contacts per case, independent of the city')
ax.set_xlabel('city population (millions)')
ax.set_ylabel('mean community candidate contacts per case')
ax.set_ylim(bottom=0)
ax.set_title('Community contacts no longer scale with city population\n'
             f'(correlation with population r = {r:.2f}; it was proportional before B27, finding E4)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'fig_municipality_population_scaling.png', dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------- tables
table.append(summary_row('household Taiwan random person (size-1)', tw_household_person))
table.append(summary_row('household Taiwan random household (size-1)', tw_household_household))
if tw_school_all is not None:
    table.append(summary_row('school Taiwan classmates per student, all levels', tw_school_all))
index_subset = {'household': index, 'school': index[index['school_size'] > 0],
                'workplace': index[index['workplace_size'] > 0], 'health_care': index, 'municipality': index}
for L in layer_names:
    s = subset[L]
    table.append(summary_row(f'{L} CovSyn (1) context size', s[L + '_size'] - (1 if L == 'school' else 0)))
    table.append(summary_row(f'{L} CovSyn (2) candidate', s[L + '_cand']))
    table.append(summary_row(f'{L} CovSyn (2) candidate, index cases', index_subset[L][L + '_cand']))
    table.append(summary_row(f'{L} CovSyn (3) effective', s[L + '_eff']))
with open(OUT / 'candidate_contact_stats.csv', 'w', newline='') as f:
    csv.writer(f).writerows(table)
for line in table:
    print(' | '.join(str(x) for x in line))
print('municipality r(population, mean candidate contacts) =', round(r, 3),
      '| lambda (mean community contacts per case) =', P[28])
print('saved to', OUT)
