"""Validation figures rebuilt to the conclusions of the 2026-09-23 meeting (todolist923.md).

  T1  fig_T1_epi_periods.png          todolist 2.1 / 1.4: epidemiological periods against the
                                      literature. Literature ranges drawn underneath, simulation on
                                      top, the black mean line last; symptomatic and asymptomatic
                                      infectious periods on the same scale; axes clipped at 20 days
                                      (R0 at 8) with the full 95% CI written in the panel; the
                                      pre-/post-symptomatic periods labelled with their source
                                      (Byrne et al. 2020, finding E46).
  T2  fig_T2_candidate_contacts.png   todolist 2.2: candidate contacts per case on the full range.
                                      The old stage-3/4 panels clipped the degree at 40
                                      (plot_diagnostics.py np.clip), which made the peak at 40.
  T3  fig_T3_effective_contacts.png   todolist 2.4 / 1.9: secondary infections per case, Taiwan
                                      first. Taiwan gives two definitions and both are shown (E80).
  T4  fig_T4{A,B,C}_*.png             todolist 2.3 / 1.10 / 1.11: every checked quantity as its
                                      ACTUAL value against its interval, inside / outside marked,
                                      split into (A) input reproduction, (B) calibration targets
                                      the optimizer is charged on -- not independent -- and
                                      (C) independent validation. Values and intervals are read
                                      from phaseD_checks.json, i.e. the same numbers as the
                                      acceptance checklist (lesson 4).

Usage: python plot_todolist923.py [spread_dir] [firefly_dir] [checks_json] [out_dir]
"""
import glob
import json
import random
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SPREAD = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight'
FIREFLY = sys.argv[2] if len(sys.argv) > 2 else \
    'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200'
CHECKS = sys.argv[3] if len(sys.argv) > 3 else 'validation_reference/phaseD_checks.json'
OUT = Path(sys.argv[4] if len(sys.argv) > 4 else 'validation_figures_todolist923')
OUT.mkdir(parents=True, exist_ok=True)
NOTEBOOK = 'plot_result/plot_synthetic_data.ipynb'
LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
LAYER_LABEL = {'household': 'Household', 'school': 'School (students)',
               'workplace': 'Workplace (employed)', 'health_care': 'Health care',
               'municipality': 'Community'}

SIM = '#2c6e9b'       # simulation
CI = '#cfe3cf'        # literature 95% CI range
MEAN = '#f0c27b'      # literature reported-mean range
IN, OUTC = '#2e8b57', '#c0392b'
DAY_CLIP, R0_CLIP = 20.0, 4.0


# ------------------------------------------------------------------ literature (notebook)
def notebook_dicts():
    """The report_* literature dictionaries of plot_synthetic_data.ipynb (finding E45)."""
    nb = json.load(open(NOTEBOOK, encoding='utf-8'))
    src = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c.get('cell_type') == 'code')

    def extract(name):
        i = src.find(name + ' = ')
        if i < 0:
            return None
        j = i + len(name + ' = ')
        while src[j] in ' \n\t':
            j += 1
        if src[j] not in '{[':
            return None
        opn, cls, depth = src[j], {'{': '}', '[': ']'}[src[j]], 0
        for k in range(j, len(src)):
            depth += (src[k] == opn) - (src[k] == cls)
            if depth == 0:
                return eval(src[j:k + 1], {'np': np})
        return None

    def span(d):
        means = [v[0] for v in d.values() if not np.isnan(v[0])]
        los = [v[1] for v in d.values() if len(v) > 1 and not np.isnan(v[1])]
        his = [v[2] for v in d.values() if len(v) > 2 and not np.isnan(v[2])]
        return min(means), max(means), min(los), max(his), len(d)

    names = {'latent': 'report_latent_period', 'incubation': 'report_incubation_period',
             'infectious': 'report_infectious_period',
             'asymptomatic': 'report_asymptomatic_infectious_period',
             'presymptomatic': 'report_presymptomatic_infectious_period',
             'postsymptomatic': 'report_postsymptomatic_infectious_period',
             'generation': 'report_generation_time', 'serial': 'report_serial_interval'}
    lit = {k: span(extract(v)) for k, v in names.items() if extract(v)}
    rows = (extract('R0') or []) + (extract('R0_RW') or [])
    nums = [[x for x in r[1:] if isinstance(x, (int, float)) and not np.isnan(x)] for r in rows]
    nums = [n for n in nums if n]
    if nums:
        lit['R0'] = (min(n[0] for n in nums), max(n[0] for n in nums),
                     min(min(n) for n in nums), max(max(n) for n in nums), len(nums))
    return lit


# ------------------------------------------------------------------ simulation
def best_vector():
    result = np.atleast_2d(np.loadtxt(Path(FIREFLY) / 'firefly_best.txt'))
    return result[int(np.argmin(result[:, -1])), 1:-1]


def drawn_courses(P, n=20000):
    """Course-of-disease durations drawn from the fitted parameters, as plot_10panel.py does."""
    from Data_synthesize import Draw_course_of_disease_data
    mk = lambda *t: {k: P[i] for k, i in t}
    args = (mk(('latent_period_shape', 37), ('latent_period_scale', 38)),
            mk(('infectious_period_shape', 39), ('infectious_period_scale', 40)),
            mk(('incubation_period_shape', 41), ('incubation_period_scale', 42)),
            mk(('symptom_to_confirmed_shape', 43), ('symptom_to_confirmed_scale', 44), ('symptom_to_confirmed_loc', 45)),
            mk(('asymptomatic_to_recovered_shape', 46), ('asymptomatic_to_recovered_scale', 47), ('asymptomatic_to_recovered_loc', 48)),
            mk(('symptomatic_to_critically_ill_shape', 49), ('symptomatic_to_critically_ill_scale', 50), ('symptomatic_to_critically_ill_loc', 51)),
            mk(('symptomatic_to_recovered_shape', 52), ('symptomatic_to_recovered_scale', 53), ('symptomatic_to_recovered_loc', 54)),
            mk(('critically_ill_to_recovered_shape', 55), ('critically_ill_to_recovered_scale', 56), ('critically_ill_to_recovered_loc', 57)),
            mk(('infection_to_death_shape', 58), ('infection_to_death_scale', 59)),
            mk(('negative_to_confirmed_shape', 60), ('negative_to_confirmed_scale', 61), ('negative_to_confirmed_loc', 62)))
    np.random.seed(0)
    random.seed(0)
    out = {k: [] for k in ('latent', 'incubation', 'infectious', 'symptomatic', 'asymptomatic',
                           'presymptomatic', 'postsymptomatic')}
    for _ in range(n):
        o = Draw_course_of_disease_data(0, *args, P[67], [P[195], P[196], P[197]])
        o.draw_course_of_disease()
        out['latent'].append(o.latent_period)
        out['infectious'].append(o.infectious_period)
        if isinstance(o.incubation_period, float) and np.isnan(o.incubation_period):
            out['asymptomatic'].append(o.infectious_period)
        else:
            pre = o.incubation_period - o.latent_period
            out['incubation'].append(o.incubation_period)
            out['symptomatic'].append(o.infectious_period)
            out['presymptomatic'].append(pre)
            out['postsymptomatic'].append(o.infectious_period - pre)
    return {k: np.asarray(v, float) for k, v in out.items()}


def spread_outputs():
    """Per-case contact counts, offspring of index cases, generation / serial intervals, R0."""
    from R0_network import R0_average_effective_contact

    def case_id(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(f) else int(f)

    def missing(x):
        return x is None or (isinstance(x, float) and (np.isnan(x) or x <= -1 or x >= 1e9))

    cand = {L: [] for L in LAYERS}
    degree, index_offspring, gen, serial, r0 = [], [], [], [], []
    files = sorted(glob.glob(f'{SPREAD}/contact_data_*.npy'), key=lambda p: int(p.split('_')[-1].split('.')[0]))
    for f in files:
        sim = int(f.split('_')[-1].split('.')[0])
        contact = list(np.load(f, allow_pickle=True))
        course = list(np.load(f'{SPREAD}/course_of_disease_data_{sim}.npy', allow_pickle=True))
        social = list(np.load(f'{SPREAD}/social_data_{sim}.npy', allow_pickle=True))
        digraph = np.load(f'{SPREAD}/transmission_digraph_{sim}.npy', allow_pickle=True)
        if contact:
            r0.append(R0_average_effective_contact(contact))
        for i, c in enumerate(contact):
            s = social[i] if i < len(social) else {}
            active = {'school': float(s.get('school_class_size') or 0) > 0,
                      'workplace': float(s.get('work_group_size') or 0) > 0}
            total_c, total_e = 0, 0
            for L in LAYERS:
                e = c[f'{L}_effective_contacts'] or []
                total_c += len(e)
                total_e += int(np.nansum([1 if x else 0 for x in e]))
                if active.get(L, True):
                    cand[L].append(len(e))
            degree.append(total_c)
            if i == 0:
                index_offspring.append(total_e)
        by = {i + 1: course[i] for i in range(len(course))}
        for edge in digraph:
            p, q = case_id(edge[0]), case_id(edge[1])
            if p in by and q in by:
                gen.append(by[q]['infection_day'] - by[p]['infection_day'])
                if not missing(by[p]['incubation_period']) and not missing(by[q]['incubation_period']):
                    serial.append(by[q]['infection_day'] + by[q]['incubation_period']
                                  - by[p]['infection_day'] - by[p]['incubation_period'])
    arr = lambda x: np.asarray(x, float)
    return ({L: arr(v) for L, v in cand.items()}, arr(degree), arr(index_offspring),
            arr(gen), arr(serial), arr(r0), len(files))


# ------------------------------------------------------------------ Taiwan reference
def taiwan_offspring():
    """Two Taiwan definitions of secondary infections per case, 2020 first wave (579 cases).

    (a) the tracing file's count of INFECTED CONTACTS per case, summed over settings -- the
        source of the B17 targets (R 0.43, k 0.29, >=3 4.3%, max 8). It counts contacts who were
        confirmed, which includes co-exposed people (same tour group, same flight), so it is an
        upper bound on onward transmission.
    (b) the recorded infector -> infectee links, a lower bound (links are only recorded when the
        infector was identified)."""
    contacts = pd.read_csv('validation_reference/taiwan_tracing_contacts_per_case.csv')
    timeline = pd.read_csv('validation_reference/taiwan_case_timeline.csv')
    events = pd.read_csv('validation_reference/taiwan_infection_events.csv')
    ds = 'first_wave_2020'
    n = int((timeline.dataset == ds).sum())
    a = np.zeros(n)
    per = contacts[contacts.dataset == ds].groupby('case_id').infected.sum().values
    a[:len(per)] = per
    b = np.zeros(n)
    links = events[events.dataset == ds].groupby('source_id').size().values
    b[:len(links)] = links
    return a, b


def nb_fit(x):
    x = np.asarray(x, float)
    if x.mean() <= 0:
        return np.nan

    def nll(p):
        r, m = np.exp(p)
        return -stats.nbinom.logpmf(x, r, r / (r + m)).sum()
    return float(np.exp(optimize.minimize(nll, [0.0, np.log(x.mean())], method='Nelder-Mead').x[0]))


# ------------------------------------------------------------------ T1
def figure_t1(courses, gen, serial, r0, lit):
    panels = [
        ('Latent period', courses['latent'], 'latent', DAY_CLIP, None),
        ('Incubation period', courses['incubation'], 'incubation', DAY_CLIP, None),
        ('Infectious period, all cases', courses['infectious'], 'infectious', DAY_CLIP, None),
        # The review has no symptomatic-only estimate. The band shown is the ALL-case infectious
        # period (the same 9 studies as panel 3), labelled as such; the post-symptomatic values the
        # old 10-panel figure used here measure only the part after onset.
        ('Infectious period, symptomatic cases', courses['symptomatic'], 'infectious', DAY_CLIP,
         'literature band = infectious period of ALL cases (same studies as panel 3); '
         'no symptomatic-only estimate is reported'),
        ('Infectious period, asymptomatic cases', courses['asymptomatic'], 'asymptomatic', DAY_CLIP, None),
        ('Pre-symptomatic infectious period\n(infectious onset to symptom onset)', courses['presymptomatic'],
         'presymptomatic', DAY_CLIP, 'Byrne et al. 2020, BMJ Open (E46)'),
        ('Post-symptomatic infectious period\n(symptom onset to end of infectiousness)', courses['postsymptomatic'],
         'postsymptomatic', DAY_CLIP, 'Byrne et al. 2020, BMJ Open (E46)'),
        ('Generation time', gen, 'generation', DAY_CLIP, None),
        ('Serial interval', serial, 'serial', DAY_CLIP, None),
        ('R0 (mean effective contacts per case, per simulation)', r0, 'R0', R0_CLIP, None)]
    fig, axes = plt.subplots(4, 3, figsize=(16, 20))
    axes = axes.ravel()
    for ax, (title, data, key, clip, note) in zip(axes, panels):
        data = data[np.isfinite(data)]
        is_r0 = key == 'R0'
        lo_edge = -5 if key == 'serial' else 0
        hi_edge = clip
        info = []
        if key in lit:
            ml, mh, cl, ch, n_studies = lit[key]
            # literature underneath (zorder 0-1)
            ax.axvspan(max(cl, lo_edge), min(ch, hi_edge), color=CI, zorder=0,
                       label='literature 95% CI range')
            ax.axvspan(ml, mh, color=MEAN, zorder=1, label='literature reported-mean range')
            if ch > hi_edge:
                ax.annotate(f'95% CI continues to {ch:g}', xy=(hi_edge, 0.97), xycoords=('data', 'axes fraction'),
                            ha='right', va='top', fontsize=8, color='#4a7a4a')
            mean = float(np.mean(data))
            inside_mean = ml <= mean <= mh
            inside_ci = cl <= mean <= ch
            info.append(f'{n_studies} studies; reported means {ml:g}-{mh:g}, CI {cl:g}-{ch:g}')
            info.append(f'CovSyn mean {mean:.2f}: '
                        + ('inside the reported-mean range' if inside_mean else
                           'inside the 95% CI range only' if inside_ci else 'OUTSIDE the 95% CI range'))
        else:
            info.append(f'CovSyn mean {np.mean(data):.2f}')
        # simulation on top (zorder 2)
        if is_r0:
            bins = np.linspace(0, hi_edge, 33)
        else:
            bins = np.arange(lo_edge - 0.5, hi_edge + 1.0, 1.0)
        shown = np.clip(data, bins[0], bins[-1])
        weights = np.full(len(shown), 1.0 / len(shown))
        ax.hist(shown, bins=bins, weights=weights, color=SIM, alpha=0.9, zorder=2,
                edgecolor='white', linewidth=0.4, label='CovSyn')
        beyond = float(np.mean(data > hi_edge))
        if beyond >= 0.0005:
            info.append(f'{100 * beyond:.1f}% of CovSyn values > {hi_edge:g} (in the last bin)')
        if is_r0:
            info.append(f'{100 * np.mean(data == 0):.0f}% of simulations have R0 = 0 (the index case infected '
                        'nobody); the old log-axis panel dropped them')
        # reference line last (zorder 5)
        ax.axvline(np.mean(data), color='black', ls='--', lw=2, zorder=5, label='CovSyn mean')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('R0' if is_r0 else 'days')
        ax.set_ylabel('share of cases' if not is_r0 else 'share of simulations')
        if note:
            info.append(note)
        ax.text(0.5, -0.17, '\n'.join(textwrap.fill(line, 80) for line in info), transform=ax.transAxes,
                ha='center', va='top', fontsize=7.8)
    # symptomatic and asymptomatic on the same y-scale as well as the same x-scale
    ymax = max(axes[3].get_ylim()[1], axes[4].get_ylim()[1])
    axes[3].set_ylim(0, ymax)
    axes[4].set_ylim(0, ymax)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[len(panels):]:
        ax.axis('off')
    axes[-1].legend(handles, labels, loc='center', fontsize=10, frameon=False)
    axes[-2].text(0.0, 0.5, textwrap.fill(
        'Drawing order: literature 95% CI range (light green) and reported-mean range (orange) '
        'underneath, CovSyn on top, CovSyn mean (black dashed) last. Day axes are clipped at '
        f'{DAY_CLIP:g} days and R0 at {R0_CLIP:g}; where a literature CI extends further its full end '
        'is written in the panel. Symptomatic and asymptomatic infectious periods share both axes. '
        'Durations are drawn from the fitted parameters (20,000 courses); generation time, serial '
        'interval and R0 come from the 1,000 spread simulations. The target is to lie inside the '
        'literature range, not on a literature mean (todolist 1.11).', 60),
        transform=axes[-2].transAxes, va='center', fontsize=9)
    fig.suptitle('Epidemiological periods: CovSyn against the literature ranges', fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.07, hspace=0.62, wspace=0.22)
    fig.savefig(OUT / 'fig_T1_epi_periods.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ T2
def figure_t2(cand, degree):
    tracing = pd.read_csv('validation_reference/taiwan_tracing_contacts_per_case.csv')
    tracing = tracing[(tracing.dataset == 'first_wave_2020') & tracing.uninfected.notna()]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()
    ax = axes[0]
    bins = np.arange(-0.5, degree.max() + 1.5, 1)
    ax.hist(degree, bins=bins, color=SIM, zorder=2)
    ax.axvspan(40.5, bins[-1], color='#f6d5d5', zorder=0, label='> 40: all clipped into ONE bar before')
    ax.set_yscale('log')
    ax.set_title('All settings: candidate contacts per case, full range', fontweight='bold')
    ax.set_xlabel('candidate contacts per case')
    ax.set_ylabel('cases (log)')
    over = int((degree > 40).sum())
    ax.text(0.97, 0.95, f'n = {len(degree):,} cases\nmax {int(degree.max())}\n'
            f'{over} cases ({100 * over / len(degree):.1f}%) above 40\n'
            f'the old stage-3/4 panels drew these as a peak at 40', transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5, bbox=dict(facecolor='white', edgecolor='#bbbbbb'))
    ax.legend(loc='lower left', fontsize=8)
    for ax, L in zip(axes[1:], LAYERS):
        # cumulative distribution: no binning, so nothing is clipped or merged
        x = np.sort(cand[L])
        obs = np.sort(tracing[tracing.layer == L].total.to_numpy(float))
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=SIM, lw=2.2, zorder=2,
                label=f'CovSyn candidate contacts (n={len(x):,}, max {int(x.max())})')
        if len(obs):
            ax.plot(obs, np.arange(1, len(obs) + 1) / len(obs), 'o', color='#2e8b57', ms=6, zorder=3,
                    markeredgecolor='black', label=f'Taiwan tracing 2020 (n={len(obs)}, max {int(obs.max())})')
        top = max(x.max(), obs.max() if len(obs) else 0, 1)
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xlim(0, top * 1.5)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.set_title(LAYER_LABEL[L], fontweight='bold')
        ax.set_xlabel('contacts per case (symlog axis, no clipping)')
        ax.set_ylabel('cumulative share of cases')
        ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('Candidate contacts per case on the full range (todolist 2.2)', fontsize=15, fontweight='bold')
    fig.text(0.01, 0.005, textwrap.fill(
        'Candidate contact = a person met on at least one day between infection and isolation (health care: '
        'to isolation + 14 days, B48). School counts students only, workplace employed cases only. Taiwan: CDC '
        'contact-tracing records of the 2020 first wave where the uninfected contacts were also reported; '
        'the samples are small and non-random (n in each legend), so the overlay is exploratory. '
        'The tracing file does not break every case\'s total down by setting for every case (todolist 1.8, F9).', 230),
        fontsize=8.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT / 'fig_T2_candidate_contacts.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ T3
def figure_t3(index_offspring):
    tw_a, tw_b = taiwan_offspring()
    series = [('Taiwan (a): infected contacts per case\n(tracing file; includes co-exposed contacts)', tw_a, '#2e8b57'),
              ('Taiwan (b): recorded infector -> infectee links\n(only links with an identified infector)', tw_b, '#8fbf8f'),
              ('CovSyn: secondary infections of the index case\n(1,000 spread simulations)', index_offspring, SIM)]
    top = int(max(s[1].max() for s in series))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={'width_ratios': [2.2, 1]})
    width = 0.27
    ks = np.arange(top + 1)
    for j, (label, x, color) in enumerate(series):
        share = np.bincount(x.astype(int), minlength=top + 1) / len(x)
        ax.bar(ks + (j - 1) * width, share, width, color=color, label=label, zorder=2)
    ax.set_yscale('log')
    ax.set_xticks(ks)
    ax.set_xlabel('secondary infections per case')
    ax.set_ylabel('share of cases (log)')
    ax.set_title('Secondary infections per case: Taiwan first, then CovSyn (todolist 2.4 / 1.9)', fontweight='bold')
    ax.legend(fontsize=8.5)
    rows = []
    for label, x, _ in series:
        rows.append([label.split(':')[0], f'{len(x):,}', f'{x.mean():.3f}', f'{nb_fit(x):.2f}',
                     f'{100 * np.mean(x >= 3):.1f}%', f'{int(x.max())}'])
    bx.axis('off')
    table = bx.table(cellText=rows, colLabels=['series', 'n', 'mean (R)', 'NB k', '>= 3', 'max'],
                     loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    bx.text(0.0, 0.45, textwrap.fill(
        'Taiwan (a) reproduces the B17 targets exactly (R 0.428, k 0.286, 4.3% >= 3, max 8), but it counts '
        'every confirmed contact of a case, including people exposed together with it (tour groups, flights): '
        '248 "infected contacts" for 579 cases, while Taiwan had about 55 local cases in the first wave. '
        'It is therefore an upper bound on onward transmission. Taiwan (b), the recorded links, is a lower '
        'bound. CovSyn counts only infections caused by the case, which is the definition of (b); its mean lies '
        'between the two (finding E80).', 55), transform=bx.transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_T3_effective_contacts.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ T4
# Which group every checklist line belongs to. B = the optimizer is charged on it (OUTCOME_TARGETS,
# physiology_penalty, the Cheng contact-timing fit or the attack-rate anchors), so a pass shows the
# fit worked, not that the model is right. A = the reference is itself a model input. C = neither.
GROUP = {
    'A': {
        'household: other members per case': 'MOI 2021 household structure (input)',
        'household: living alone': 'MOI 2021 (input)',
        'class size, elementary (7-12)': 'MOE enrolment (input)',
        'class size, junior high (13-15)': 'MOE enrolment (input)',
        'class size, senior high (16-18)': 'MOE enrolment (input)',
        'class size, university (19-22)': 'MOE enrolment (input)',
        'work group size (median)': 'Chen 2022 work-group model (input, B31)',
        'symptomatic to ICU': 'Taiwan tracing 56/442 (P[196] set from it, B33)',
        'ICU to death': 'Taiwan tracing (P[197] set from it, B33)',
        'case fatality': 'Taiwan tracing (product of the two above)',
        'ICU rate 60+ / 20-39': 'age gradient input SYMPTOM_TO_ICU_AGE_RR (B20)',
        'Taiwan scenario mean case age': 'ages of the 28 seeded Taiwan cases (input, B34)',
        'cases with candidate contacts but no effective-contact record': 'implementation check (B32)',
        'cases with no contact in any layer': 'implementation check (B32)',
    },
    'B': {
        'contacts per day before onset, household': 'charged: daily_household',
        'contacts per day before onset, school': 'charged: daily_school',
        'contacts per day before onset, workplace': 'charged: daily_workplace',
        'contacts per day before onset, health_care': 'charged: daily_health_care',
        'contacts per day before onset, municipality': 'charged: daily_municipality',
        'contacts per day, all layers': 'sum of the five charged daily targets',
        'cumulative SAR per contact, school': 'charged: sar_school',
        'cumulative SAR per contact, workplace': 'charged: sar_workplace',
        'infections per index case, household': 'charged: Cheng 2020 0.10 (B37)',
        'infections per index case, health_care': 'charged: Cheng 2020 0.06 (B37)',
        'infections per index case, municipality': 'charged: Cheng 2020 0.01 (B37)',
        'offspring dispersion k': 'charged: offspring_k',
        'community contacts per case, median': 'charged: community_median',
        'community contacts, p90 / median': 'charged: community_tail_ratio (B50)',
        'incubation period, mean': 'charged: incubation_mean',
        'pre-onset infectious window, mean': 'charged: pre_onset_window_mean',
        'pre-onset window of zero days': 'charged: pre_onset_zero_share',
        'latent period, mean': 'charged: physiology_penalty',
        'onset to confirmation, median': 'charged: onset_to_confirmation (B2)',
        'contacts starting before symptom onset': 'Cheng 2020 contact-timing bins (cost_contact)',
        'health care contacts starting 8+ days after onset': 'charged: medical_late_share',
        'health care contacts starting before day 4': 'charged: medical_early_share',
        'infection to case closure, symptomatic': 'charged: closure_symptomatic',
        'infection to case closure, asymptomatic': 'charged: closure_asymptomatic',
        'asymptomatic share': 'charged: asymptomatic_share (B39)',
    },
    'C': {
        'R (mean offspring of an index case)': 'Taiwan tracing definition (a), see T3 / E80',
        'cases infecting 3 or more': 'Taiwan tracing definition (a), see T3 / E80',
        'largest number infected by one case': 'Taiwan tracing definition (a)',
        'infections per case, 250+ staff / 1-49 staff': 'Chen 2022 cluster size by firm size',
        'infections per staff, 1-49 staff / 250+ staff': 'Chen 2022',
        'household: share of days each member is met': 'Fu 2012 contact diary',
        'largest / smallest city mean community contacts': 'design requirement (E4)',
        'infection to isolation, mean': 'Taiwan tracing (B26)',
        'measured age risk ratio, 0-19': 'Zhang/Viner/Uthman/Madewell range (B46)',
        'measured age risk ratio, 40-59': 'Cheng 2020; the locked input comes from the same study',
        'measured age risk ratio, 60+': 'Cheng 2020; the locked input comes from the same study',
    },
}
GROUP_TITLE = {
    'A': ('fig_T4A_input_reproduction.png',
          'A. Input reproduction / implementation checks: the reference is itself a CovSyn input'),
    'B': ('fig_T4B_calibration_targets.png',
          'B. Calibration targets: the optimizer is charged on these, so a pass shows the fit, not independent validity'),
    'C': ('fig_T4C_independent_validation.png',
          'C. Independent validation: neither a model input nor a charged target'),
}


def figure_t4(checks):
    by_name = {c['name']: c for c in checks}
    placed = set()
    for group, (fname, title) in GROUP_TITLE.items():
        rows = [(n, src) for n, src in GROUP[group].items()
                if n in by_name and isinstance(by_name[n]['target'], list)
                and len(by_name[n]['target']) == 2 and by_name[n]['value'] is not None]
        placed |= {n for n, _ in rows}
        fig, axes = plt.subplots(len(rows), 1, figsize=(15, 0.62 * len(rows) + 1.6), squeeze=False)
        n_in = 0
        for ax, (name, src) in zip(axes[:, 0], rows):
            c = by_name[name]
            lo, hi = float(c['target'][0]), float(c['target'][1])
            v = float(c['value'])
            unit = c['unit'] or ''
            inside = lo <= v <= hi
            n_in += inside
            left, right = min(lo, v), max(hi, v)
            pad = 0.15 * (right - left) if right > left else max(abs(v), 1) * 0.2
            ax.set_xlim(left - pad, right + pad)
            ax.axvspan(lo, hi, ymin=0.25, ymax=0.75, color='#d9d9d9', zorder=1)
            ax.plot([v], [0.5], 'o', ms=10, color=IN if inside else OUTC, zorder=3,
                    markeredgecolor='black')
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=7.5, pad=1)
            for s in ('top', 'right', 'left'):
                ax.spines[s].set_visible(False)
            ax.text(-0.01, 0.5, f'{name}', transform=ax.transAxes, ha='right', va='center', fontsize=9)
            ax.text(1.01, 0.5, f'{v:.3g}{unit} in [{lo:.3g}, {hi:.3g}]  ' + ('inside' if inside else 'OUTSIDE')
                    + f'\n{src}', transform=ax.transAxes, ha='left', va='center', fontsize=7.8,
                    color='black' if inside else OUTC)
        fig.suptitle(f'{title}\n{n_in} of {len(rows)} inside their interval  '
                     '(grey band = interval, dot = CovSyn value in real units)', fontsize=11.5, fontweight='bold')
        fig.subplots_adjust(left=0.30, right=0.72, top=1 - 1.1 / (0.62 * len(rows) + 1.6), bottom=0.03, hspace=1.1)
        fig.savefig(OUT / fname, dpi=130)
        plt.close(fig)
    checked = {c['name'] for c in checks if isinstance(c['target'], list) and len(c['target']) == 2
               and c['value'] is not None}
    unplaced = sorted(checked - placed)
    if unplaced:
        print('checks with an interval that are in no T4 group:', unplaced)


def main():
    checks = json.load(open(CHECKS, encoding='utf-8'))['checks']
    lit = notebook_dicts()
    courses = drawn_courses(best_vector())
    cand, degree, index_offspring, gen, serial, r0, n_sims = spread_outputs()
    print(f'{n_sims} simulations, {len(degree):,} cases, {len(index_offspring)} index cases')
    figure_t1(courses, gen, serial, r0, lit)
    figure_t2(cand, degree)
    figure_t3(index_offspring)
    figure_t4(checks)
    tw_a, tw_b = taiwan_offspring()
    print('Taiwan (a) R %.3f k %.3f; (b) R %.3f; CovSyn index R %.3f k %.3f'
          % (tw_a.mean(), nb_fit(tw_a), tw_b.mean(), index_offspring.mean(), nb_fit(index_offspring)))
    print('degree > 40: %d of %d' % ((degree > 40).sum(), len(degree)))
    print('saved to', OUT)


if __name__ == '__main__':
    main()
