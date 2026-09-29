"""Phase B validation (todo.md sections 4 and 5) on the latest CovSyn dataset (run5 parameters).

  fig_layer_attack_rate_comparison.png
      forest plot of the secondary attack rate per layer: CovSyn vs Cheng 2020, Huang 2021 and the
      calibration anchors the optimizer was given (a proportion, so not a split violin; F10)
  fig_secondary_infection_relationship_distribution.png
      share of secondary infections by layer: CovSyn vs Taiwan contact tracing (infector-linked
      events, unknown infector excluded, todo 5) and vs Cheng 2020's settings

CovSyn attack rate = effective / candidate contacts (C2 in covsyn_decisions.md): candidate contacts
are everyone contacted on at least one day between infection and isolation, with no duration
threshold. Relationship-to-layer mapping: F9 in covsyn_decisions.md (extract_tracing_reference.py).

Usage: python validate_infection.py [spread_dir] [out_dir] [index_case_dir]
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
from matplotlib.lines import Line2D

SPREAD = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight_MC1000'
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'validation_figures_phaseD')
# Optional third argument: a separate dataset to take the index cases from. Without it the
# index cases are the first case of every spread simulation, which is what they are.
INDEX = sys.argv[3] if len(sys.argv) > 3 else None
REFERENCE = Path('validation_reference')
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
LABEL = {'household': 'Household', 'school': 'School', 'workplace': 'Workplace',
         'health_care': 'Health care', 'municipality': 'Municipality'}
OTHERS = ['school', 'workplace', 'municipality']        # Cheng 2020 "others" (as in cheng_gate.py)
REF_COLOR, SIM_COLOR, SPREAD_COLOR, LIT_COLOR = '#59A14F', '#3a7ca5', '#9ecae1', '#B07AA1'
LOCAL_CASES_FIRST_WAVE = 55                              # 'Local' rows in taiwan_covid_figshare.xlsx

# Cumulative SAR (lb, centre, ub) handed to the optimizer, parameters_for_initialization.py (run5)
ANCHORS = {'household': (4.6, 6.62, 10.1), 'school': (1.0, 2.3, 4.0), 'workplace': (1.5, 3.4, 5.0),
           'health_care': (0.1, 0.86, 1.6), 'municipality': (0.1, 0.2, 1.0)}


def sim_index(path):
    return int(Path(path).stem.split('_')[-1])


def load_layer_counts(result_dir):
    """One row per infected case: simulation index and per-layer candidate / effective contacts."""
    rows = []
    for f in sorted(glob.glob(f'{result_dir}/contact_data_*.npy'), key=sim_index):
        k = sim_index(f)
        for position, c in enumerate(np.load(f, allow_pickle=True)):
            row = {'sim': k, 'position_in_sim': position}
            for L in LAYERS:
                eff = c.get(L + '_effective_contacts')
                eff = [] if eff is None else list(eff)
                row[L + '_cand'] = len(eff)
                row[L + '_eff'] = int(sum(1 for x in eff if x == 1))
            rows.append(row)
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), c + h


def pct(k, n):
    return tuple(100 * v for v in wilson(k, n))


spread = load_layer_counts(SPREAD)
index = (load_layer_counts(INDEX) if INDEX
         else spread[spread['position_in_sim'] == 0].reset_index(drop=True))
index_source = INDEX if INDEX else 'first case of each simulation'
print(f'spread {SPREAD}: {len(spread)} cases in {spread["sim"].nunique()} simulations | '
      f'index ({index_source}): {len(index)} cases in {index["sim"].nunique()} simulations')


def covsyn_sar(df, layers):
    k = int(sum(df[L + '_eff'].sum() for L in layers))
    n = int(sum(df[L + '_cand'].sum() for L in layers))
    return k, n


def per_run_range(df, layers):
    """2.5-97.5 percentile of the SAR in one simulation (100 index cases, the size of Cheng 2020)."""
    g = df.groupby('sim')[[L + s for L in layers for s in ('_cand', '_eff')]].sum()
    cand = g[[L + '_cand' for L in layers]].sum(axis=1)
    eff = g[[L + '_eff' for L in layers]].sum(axis=1)
    sar = 100 * eff[cand > 0] / cand[cand > 0]
    return float(np.percentile(sar, 2.5)), float(np.percentile(sar, 97.5))


# --------------------------------------------------------------------------- figure 1: attack rate
def covsyn_entries(layers):
    k, n = covsyn_sar(index, layers)
    p, lo, hi = pct(k, n)
    r = per_run_range(index, layers)
    ks, ns = covsyn_sar(spread, layers)
    ps, los, his = pct(ks, ns)
    return [dict(series='index', p=p, lo=lo, hi=hi, range=r, k=k, n=n,
                 text=f'CovSyn index cases {p:.2f}% ({k:,}/{n:,}); 95% of 100-case runs {r[0]:.2f}-{r[1]:.2f}%'),
            dict(series='spread', p=ps, lo=los, hi=his, k=ks, n=ns,
                 text=f'CovSyn spread {ps:.2f}% ({lo_hi(los, his)}; {ks:,}/{ns:,})')]


def lo_hi(lo, hi):
    return f'{lo:.2f}-{hi:.2f}'


def anchor_entry(layer):
    lb, centre, ub = ANCHORS[layer]
    return dict(series='anchor', p=centre, lo=lb, hi=ub, text=f'calibration anchor {centre:g}% (bounds {lb:g}-{ub:g}%)')


def reported(series, p, lo, hi, text):
    return dict(series=series, p=p, lo=lo, hi=hi, text=text)


def counted(series, k, n, text):
    p, lo, hi = pct(k, n)
    return dict(series=series, p=p, lo=lo, hi=hi, k=k, n=n, text=f'{text} {p:.2f}% ({lo_hi(lo, hi)}; {k}/{n:,})')


ROWS = [
    ('All settings', covsyn_entries(LAYERS) + [
        reported('cheng_clinical', 0.7, 0.4, 1.0, 'Cheng 2020 clinical 0.7% (0.4-1.0; 2,761 contacts)'),
        counted('cheng_all', 22, 2761, 'Cheng 2020 all infections'),
        reported('huang', 0.88, 0.42, 1.69, 'Huang 2021 pooled 0.88% (CrI 0.42-1.69; 15 case series)')]),
    ('Household', covsyn_entries(['household']) + [
        anchor_entry('household'),
        reported('cheng_clinical', 4.6, 2.3, 9.3, 'Cheng 2020 clinical 4.6% (2.3-9.3; 7/151)'),
        counted('cheng_all', 10, 151, 'Cheng 2020 all infections')]),
    ('Non-household family\n(no CovSyn layer)', [
        reported('cheng_clinical', 5.3, 2.1, 12.8, 'Cheng 2020 clinical 5.3% (2.1-12.8; 4/76)'),
        counted('cheng_all', 5, 76, 'Cheng 2020 all infections')]),
    ('School', covsyn_entries(['school']) + [
        anchor_entry('school'),
        counted('huang', 3, 126, 'Huang 2021 classroom case series (cases 39 + 59)')]),
    ('Workplace', covsyn_entries(['workplace']) + [
        anchor_entry('workplace'),
        counted('huang', 3, 41, 'Huang 2021 workplace cluster (case 160)'),
        counted('huang', 2, 24, 'Huang 2021 workplace + household (case 277)')]),
    ('Health care', covsyn_entries(['health_care']) + [
        anchor_entry('health_care'),
        reported('cheng_clinical', 0.9, 0.4, 1.9, 'Cheng 2020 clinical = all infections 0.9% (0.4-1.9; 6/698)'),
        counted('huang', 8, 455, 'Huang 2021 hospital cluster')]),
    ('Municipality', covsyn_entries(['municipality']) + [anchor_entry('municipality')]),
    ('Others\n(school + workplace\n+ municipality)', covsyn_entries(OTHERS) + [
        reported('cheng_clinical', 0.1, 0.0, 0.3, 'Cheng 2020 "others" clinical = all 0.1% (0.0-0.3; 1/1,836)')]),
]
SERIES = {'index': ("CovSyn under Cheng 2020's design: 100 index cases, one generation (same model, "
                    'other scenario); band = 95% of those runs', 'o', SIM_COLOR, SIM_COLOR),
          'spread': ('CovSyn, all cases of the 84-day spread simulations', 's', SPREAD_COLOR, '#1f4e79'),
          'anchor': ('Calibration anchor given to the optimizer (bar = bounds)', '|', 'black', 'black'),
          'cheng_clinical': ('Cheng 2020, clinical attack rate as reported (symptomatic only)', 'D', REF_COLOR, REF_COLOR),
          'cheng_all': ('Cheng 2020, all infections incl. 4 asymptomatic (Wilson CI)', 'D', 'white', REF_COLOR),
          'huang': ('Huang, Tu & Lai 2021, Taiwan CDC case series', '^', LIT_COLOR, LIT_COLOR)}
XMIN, XMAX = 0.01, 40

fig, ax = plt.subplots(figsize=(17, 13))
y, step, gap = 0.0, 0.34, 0.55
yticks, ylabels, records = [], [], []
for row_label, entries in ROWS:
    top = y
    for e in entries:
        label, marker, face, edge = SERIES[e['series']]
        if e['series'] == 'anchor':
            ax.hlines(y, e['lo'], e['hi'], color='#c8c8c8', lw=7, zorder=1)
            ax.plot(e['p'], y, marker='|', color='black', ms=16, mew=2, zorder=2)
        else:
            if 'range' in e:
                ax.hlines(y, max(e['range'][0], XMIN), e['range'][1], color=edge, lw=9, alpha=0.22, zorder=1)
            ax.hlines(y, max(e['lo'], XMIN), e['hi'], color=edge, lw=1.6, zorder=2)
            ax.scatter(max(e['p'], XMIN), y, marker=marker, s=70, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3)
        ax.text(1.01, y, e['text'], transform=ax.get_yaxis_transform(), fontsize=8, va='center')
        records.append([row_label.replace('\n', ' '), SERIES[e['series']][0], round(e['p'], 4), round(e['lo'], 4),
                        round(e['hi'], 4), e.get('k', ''), e.get('n', ''),
                        '' if 'range' not in e else f'{e["range"][0]:.4f}-{e["range"][1]:.4f}'])
        y -= step
    yticks.append((top + y + step) / 2)
    ylabels.append(row_label)
    ax.axhline(y + step - gap / 2 - 0.02, color='#dddddd', lw=0.8)
    y -= gap
ax.set_xscale('log')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(y + gap / 2, 0.4)
ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
ax.set_xticklabels(['0.01', '0.03', '0.1', '0.3', '1', '3', '10', '30'])
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=10)
ax.set_xlabel('secondary attack rate (%) — log scale; lower limits of 0 are drawn at 0.01%')
ax.grid(axis='x', alpha=0.3)
fig.legend(handles=[Line2D([], [], marker=m, ls='none', markerfacecolor=f, markeredgecolor=e, color=e, ms=8, mew=1.6, label=l)
                    for l, m, f, e in SERIES.values()], loc='lower center', bbox_to_anchor=(0.39, 0.075), ncol=2, fontsize=8.5)
ax.set_title('Secondary attack rate by social layer — CovSyn vs Taiwan literature', fontsize=13, fontweight='bold')
fig.text(0.01, 0.006,
         'CovSyn: attack rate = effective / candidate contacts; candidate = contacted on at least one day from infection to isolation, no '
         f'duration threshold. Index cases: {INDEX} ({len(index):,} index cases, one generation, like Cheng 2020); spread: {SPREAD} '
         f'({len(spread):,} cases). CovSyn intervals are 95% Wilson intervals. Cheng et al. 2020 (JAMA Intern Med): 100 index cases, '
         'Jan 15-Mar 18 2020; close contact = face-to-face >15 min without PPE, traced from symptom onset (up to 4 days before when '
         'indicated) to confirmation; clinical rate counts symptomatic secondary cases; all-infection counts add the 3 household and 1 '
         'non-household-family asymptomatic cases. Cheng "others" pools every setting except family and health care, compared here with '
         'CovSyn school + workplace + municipality. Huang, Tu & Lai 2021 (J Microbiol Immunol Infect): Taiwan CDC case series, Jan 21-'
         'Apr 8 2020, pooled by proportional meta-analysis (credible interval). Its single case series are shown here as raw counts with '
         'Wilson intervals, and their denominator is EVERY traced contact of that case series, not only the contacts in the setting '
         'where transmission happened, so they are not setting-specific attack rates; the paper\'s own shrunk Bayesian medians are '
         'lower (case 160: 3.60%, case 277: 3.02%, hospital: 1.56%). The CovSyn school anchor of 2.3% is exactly the classroom series '
         'shown here (3/126). The two CovSyn series are the same model under two scenarios, not independent evidence. Calibration '
         'anchors: cumulative SAR converted to daily attack rates in parameters_for_initialization.py.', fontsize=7.5, wrap=True)
fig.subplots_adjust(left=0.12, right=0.66, top=0.95, bottom=0.2)
fig.savefig(OUT / 'fig_layer_attack_rate_comparison.png', dpi=150)
plt.close(fig)
with open(OUT / 'layer_attack_rate_comparison.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['row', 'series', 'sar_percent', 'ci_low', 'ci_high', 'infected', 'contacts', 'per_run_95_range'])
    w.writerows(records)


# --------------------------------------------------------------------------- figure 2: relationship distribution
def bootstrap_shares(df, groups, reps=2000):
    """Share of effective contacts per group, with 95% bootstrap intervals over simulations."""
    per_sim = df.groupby('sim')[[L + '_eff' for L in LAYERS]].sum()
    counts = np.column_stack([per_sim[[L + '_eff' for L in g]].sum(axis=1).to_numpy() for g in groups])
    total = counts.sum()
    share = counts.sum(axis=0) / total
    draws = RNG.integers(0, len(counts), size=(reps, len(counts)))
    boot = np.array([counts[d].sum(axis=0) / max(counts[d].sum(), 1) for d in draws])
    return counts.sum(axis=0), int(total), share, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def multinomial_shares(counts):
    n = int(sum(counts))
    ci = np.array([wilson(k, n) for k in counts])
    return np.asarray(counts), n, ci[:, 0], ci[:, 1], ci[:, 2]


def grouped_bars(ax, categories, series, ylabel):
    width = 0.8 / len(series)
    for j, (name, color, (counts, n, share, lo, hi)) in enumerate(series):
        x = np.arange(len(categories)) + (j - (len(series) - 1) / 2) * width
        ax.bar(x, 100 * share, width * 0.92, color=color, edgecolor='black', linewidth=0.6, label=f'{name} (n={n:,})')
        ax.errorbar(x, 100 * share, yerr=[100 * (share - lo), 100 * (hi - share)], fmt='none', ecolor='black', capsize=3, lw=1)
        for xi, k, s, h in zip(x, counts, share, hi):
            ax.text(xi, 100 * h + 1.2, f'{int(k)}' if n < 1000 else f'{100 * s:.0f}%', ha='center', fontsize=7.5)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')


events = pd.read_csv(REFERENCE / 'taiwan_infection_events.csv')
first = events[events['dataset'] == 'first_wave_2020']
local = first[first['infectee_type'] == 'Local']
known = local[local['layer'] != 'no_covsyn_layer']
tw_counts = known['layer'].value_counts().reindex(LAYERS, fill_value=0).to_numpy()
tw = multinomial_shares(tw_counts)
five = [[L] for L in LAYERS]
cov_spread = bootstrap_shares(spread, five)

cheng_groups = [('Household\n(+ non-household family)', ['household']), ('Health care', ['health_care']),
                ('Others\n(school, workplace, municipality)', OTHERS)]
cheng = multinomial_shares([15, 6, 1])      # Cheng 2020 Table 2, all infections: household 10 + non-household family 5
tw_cheng = multinomial_shares([tw_counts[LAYERS.index('household')], tw_counts[LAYERS.index('health_care')],
                               sum(tw_counts[LAYERS.index(L)] for L in OTHERS)])
cov_index_cheng = bootstrap_shares(index, [g for _, g in cheng_groups])
cov_spread_cheng = bootstrap_shares(spread, [g for _, g in cheng_groups])

fig, axes = plt.subplots(2, 1, figsize=(13, 13))
grouped_bars(axes[0], [LABEL[L] for L in LAYERS],
             [('Taiwan contact tracing, local infectees with a recorded infector', REF_COLOR, tw),
              ('CovSyn spread simulations', SIM_COLOR, cov_spread)], '% of secondary infections with a known layer')
axes[0].set_title('(a) Five CovSyn layers: Taiwan contact tracing (first wave 2020) vs CovSyn', fontsize=11)
grouped_bars(axes[1], [c for c, _ in cheng_groups],
             [('Cheng 2020, all secondary infections', REF_COLOR, cheng),
              ('Taiwan contact tracing, local infectees', '#B6D7A8', tw_cheng),
              ("CovSyn under Cheng's design (100 index cases, one generation)", SIM_COLOR, cov_index_cheng),
              ('CovSyn spread simulations', SPREAD_COLOR, cov_spread_cheng)], '% of secondary infections')
axes[1].set_title("(b) Cheng 2020's exposure settings", fontsize=11)
n_no_layer = int((local['layer'] == 'no_covsyn_layer').sum())
n_imported = int((first['infectee_type'] != 'Local').sum())
fig.suptitle('Secondary infections by relationship — Taiwan vs CovSyn', fontsize=13, fontweight='bold')
fig.text(0.01, 0.006,
         f'Taiwan contact tracing: taiwan_covid_figshare.xlsx (Taiwan CDC press releases, Jan-Nov 2020; tracing window not reported). '
         f'{len(first)} cases record an infector; {n_imported} infectees were imported and are excluded, leaving {len(local)} local infectees, '
         f'{n_no_layer} of them via a relationship with no CovSyn layer; the other {LOCAL_CASES_FIRST_WAVE - len(local)} of the '
         f'{LOCAL_CASES_FIRST_WAVE} local cases have no recorded infector and are excluded as unknown (todo 5). The extended workbook '
         'adds no infector links. Relationships mapped to layers as in covsyn_decisions.md F9 (couple, parent/child, grandparent, sibling, '
         'family, live together = household; school; coworker = workplace; same hospital = health care; friend, other = municipality); '
         'when several apply, household > school > workplace > health care > municipality. Bars = share, whiskers = 95% Wilson interval '
         '(Taiwan, Cheng) or bootstrap over simulations (CovSyn); numbers above bars are counts (percentages when n >= 1,000). Cheng et al. 2020: 22 '
         'secondary cases among contacts of 100 index cases (Table 2), non-household family merged into household, "others" = every '
         f'other setting. CovSyn: effective contacts per layer in {SPREAD} and {INDEX}.', fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
fig.savefig(OUT / 'fig_secondary_infection_relationship_distribution.png', dpi=150)
plt.close(fig)

with open(OUT / 'secondary_infection_relationship_distribution.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['panel', 'source', 'category', 'count', 'total', 'share', 'ci_low', 'ci_high'])
    for panel, categories, series in [
            ('a', LAYERS, [('Taiwan tracing local', tw), ('CovSyn spread', cov_spread)]),
            ('b', [c.replace('\n', ' ') for c, _ in cheng_groups],
             [('Cheng 2020', cheng), ('Taiwan tracing local', tw_cheng), ('CovSyn index', cov_index_cheng),
              ('CovSyn spread', cov_spread_cheng)])]:
        for name, (counts, n, share, lo, hi) in series:
            for c, k, s, a, b in zip(categories, counts, share, lo, hi):
                w.writerow([panel, name, c, int(k), n, round(float(s), 4), round(float(a), 4), round(float(b), 4)])

for r in records:
    print(' | '.join(str(x) for x in r))
print('Taiwan local infectees by layer:', dict(zip(LAYERS, tw_counts.tolist())), '| no CovSyn layer:', n_no_layer)
print('CovSyn spread share:', dict(zip(LAYERS, np.round(cov_spread[2], 3).tolist())))
print('saved to', OUT)
