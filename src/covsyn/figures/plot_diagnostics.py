"""CovSyn validation diagnostic figures (Stage 3-5 of covsyn_validation_workflow.md).
Stage 3: infection probability chain by social layer, with Taiwan references where they exist.
Stage 4: attack rate & infectiousness (offspring / degree / overdispersion / age risk ratio).
Stage 5: epidemic growth curves.

Stages 3 and 4 follow the meeting feedback (todo.md 6): show distributions rather than means only,
and compare with an external reference wherever one exists, naming the source and denominator.
Stage 5 has no matching observed scenario (the Taiwan outbreak comparison uses the
taiwan_first_outbreak mode, see tw_check.py), so it stays CovSyn-only.

Usage: python -m covsyn.figures.plot_diagnostics [spread_dir] [out_dir]
Only the spread dataset is plotted: the cheng2020 mode is another CovSyn configuration, so showing
it next to CovSyn results would add no independent evidence.
"""
import glob, sys, pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from covsyn.data_processing.taiwan_reference import (IMPORTED_COLOR, MARKER_NOTE, household_pmfs, load_demographics, pmf_samples,
                              school_classmates_all, school_reference, split_violins, to_axis, tracing_contacts,
                              workplace_establishment_distributions)

SYN = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight'
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else '.')
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = Path('validation_reference/taiwan_infection_events.csv')
LAYERS = [('household', 'household_effective_contacts', 'household_secondary_contact_ages', 'household_contact_ages'),
          ('school', 'school_effective_contacts', 'school_secondary_contact_ages', 'school_contact_ages'),
          ('workplace', 'workplace_effective_contacts', 'workplace_secondary_contact_ages', 'workplace_contact_ages'),
          ('healthcare', 'health_care_effective_contacts', 'health_care_secondary_contact_ages', 'health_care_contact_ages'),
          ('municipality', 'municipality_effective_contacts', 'municipality_secondary_contact_ages', 'municipality_contact_ages')]
NA = lambda x: x is None or (isinstance(x, float) and (np.isnan(x) or x <= -1 or x >= 1e9))
BLUE, ORANGE, GREEN, LIGHT_BLUE = '#3a7ca5', '#E8A33D', '#59A14F', '#9ecae1'
# Cheng et al. 2020: secondary cases per index case by setting (22 cases from 100 index cases).
# Household includes the 5 non-household-family cases; "others" (1 case) pools school+workplace+municipality.
CHENG_PER_INDEX = {'household': 0.15, 'healthcare': 0.06}
CHENG_PER_INDEX_OTHERS = 0.01

# Cheng et al. 2020 (JAMA Intern Med), Table 2, all infections: cases / contacts per setting.
CHENG_SAR = {'household': (10, 151), 'healthcare': (6, 698)}          # 'others' 1/1836 is school+workplace+municipality pooled
CHENG_OTHERS = (1, 1836)
CHENG_SHARE = {'household': 15, 'healthcare': 6, 'others': 1}          # household includes 5 non-household family
# Cumulative SAR anchors handed to the optimizer (parameters_for_initialization.py, run5)
ANCHOR = {'household': 6.62, 'school': 2.3, 'workplace': 3.4, 'healthcare': 0.86, 'municipality': 0.2}
# Cheng 2020 age risk ratio, all infections (covsyn_decisions.md B2)
CHENG_RR = [0.52, 1.0, 1.83, 1.32]

DEMO = load_demographics()
MUNICIPALITY_POP = {k: float(v) for k, v in DEMO[6].items()}
SIZE_KEY = {'household': 'household_size', 'school': 'school_class_size', 'workplace': 'work_group_size',
            'healthcare': 'clinic_size', 'municipality': None}

cand = {L[0]: 0 for L in LAYERS}; eff = {L[0]: 0 for L in LAYERS}
candpc = {L[0]: [] for L in LAYERS}; effpc = {L[0]: [] for L in LAYERS}
ctxpc = {L[0]: [] for L in LAYERS}          # (1) context size, to keep the three quantities apart
age_cand, age_eff = [], []
offspring, degree = [], []
case_infday, case_off = [], []
inf_day, conf_day, rec_day, death_day = [], [], [], []
growth = []

for f in sorted(glob.glob(SYN + '/contact_data_*.npy'), key=lambda p: int(p.split('_')[-1].split('.')[0])):
    sim = int(f.split('_')[-1].split('.')[0])
    C = list(np.load(f, allow_pickle=True))
    course = list(np.load(SYN + f'/course_of_disease_data_{sim}.npy', allow_pickle=True))
    social = list(np.load(SYN + f'/social_data_{sim}.npy', allow_pickle=True))
    sim_inf = []
    for i in range(min(len(C), len(course))):
        cd, cs = C[i], course[i]
        s = social[i] if i < len(social) else {}
        in_context = {'school': float(s.get('school_class_size') or 0) > 0,
                      'workplace': float(s.get('work_group_size') or 0) > 0}
        te, tc = 0, 0
        for nm, ek, ak, allk in LAYERS:
            e = cd[ek]
            # ages of every candidate contact (D4); fall back to the infected-only ages of older datasets
            a = cd.get(allk) if isinstance(cd, dict) else None
            if a is None or len(a) != len(e):
                a = cd[ak]
            n = len(e); se = int(np.nansum([1 if x else 0 for x in e]))
            cand[nm] += n; eff[nm] += se
            if in_context.get(nm, True):    # students only / employed only for the per-case distributions
                candpc[nm].append(n); effpc[nm].append(se)
                # B27: the community layer no longer has an opportunity set -- it draws its
                # contacts directly, so the city population is not a context size any more
                # (it was, and that was finding E4).
                key = SIZE_KEY[nm]
                size = np.nan if key is None else s.get(key)
                size = np.nan if size is None else float(size)
                ctxpc[nm].append(size - 1 if nm == 'school' else size)   # school_class_size includes the case
            tc += n; te += se
            for k in range(len(e)):
                if not NA(a[k]):
                    age_cand.append(float(a[k])); age_eff.append(1 if e[k] else 0)
        offspring.append(te); degree.append(tc)
        if not NA(cs['infection_day']):
            case_infday.append(cs['infection_day']); case_off.append(te)
    for cs in course:
        if not NA(cs['infection_day']):
            inf_day.append(cs['infection_day']); sim_inf.append(int(cs['infection_day']))
        pt = np.ravel(cs['positive_test_date'])[0] if cs['positive_test_date'] is not None else None
        if not NA(pt): conf_day.append(pt)
        if not NA(cs['date_of_recovery']): rec_day.append(cs['date_of_recovery'])
        if not NA(cs['date_of_death']): death_day.append(cs['date_of_death'])
    growth.append(sorted(sim_inf))

names = [L[0] for L in LAYERS]
ar_layer = [eff[n] / cand[n] if cand[n] else 0 for n in names]
total_eff = sum(eff[n] for n in names)


def wilson(k, n, z=1.96):
    if not n:
        return np.nan, np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), c + h


# --------------------------------------------------------------------------- Taiwan references (D1: left half)
sizes, _, _, pmf_person = household_pmfs(DEMO)
ref_household = pmf_samples(sizes - 1, pmf_person)         # household of a random person, MOI 2021
try:
    ref_school = school_classmates_all(school_reference())  # classmates per student, MOE files
except Exception as exc:
    print('WARNING: raw school reference failed:', repr(exc)); ref_school = None
# No Taiwan workgroup data exists (covsyn_decisions.md G), so the reference is the whole establishment
# a random worker is employed in (2016 industry and service census, size-weighted).
size_values, _, pmf_worker = workplace_establishment_distributions(DEMO)
ref_workplace = pmf_samples(size_values, pmf_worker)


def tracing_reference(layer):
    """Close contacts per index case from Taiwan CDC contact tracing.

    The colour says how complete the record is, because that changes what the number means:
    dark green = the uninfected contacts were reported too, so the total is a real contact count;
    hollow = only the infected contacts are known, so the total is a lower bound.
    Densities are used from 30 observations, individual points below that (D2).
    """
    values, is_local, has_uninfected = tracing_contacts(layer)
    if values is None:
        return None, ''
    full, partial = values[has_uninfected], values[~has_uninfected]
    specs = []
    if len(full):
        specs.append(('violin', full) if len(full) >= 30 else ('points', full, [GREEN] * len(full), 'o'))
    if len(partial):
        specs.append(('points', partial, ['white'] * len(partial), 'v', GREEN))
    note = (f'{len(full)} full records, {len(partial)} infected-only\n'
            f'{int(is_local.sum())} local / {int((~is_local).sum())} imported index cases')
    return specs, note


# ===================== STAGE 3 =====================
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.15, 1])
ax = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2]),
      fig.add_subplot(gs[2, 0]), None, fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]

# (1) opportunity set vs opportunity set: how many people are in the context at all
split_violins(ax[0], [
    ('Household', ref_household, ctxpc['household'], 'members other than the case;\nTaiwan: household of a random person'),
    ('School', ref_school, ctxpc['school'], 'classmates, students only;\nTaiwan: classmates per student'),
    ('Workplace', ref_workplace, ctxpc['workplace'], 'work group met, employed only;\nTaiwan: whole establishment of a random worker\n(the model caps the group, B31)'),
    ('Health care', None, ctxpc['healthcare'], 'clinic daily patient volume'),
    ('Municipality', 'no opportunity set:\ncontacts drawn\ndirectly (B27)', [], 'not applicable'),
], 'Taiwan', 'CovSyn (1) context size', GREEN, LIGHT_BLUE, 'people', label_fontsize=6.5)
ax[0].set_title('(1) Opportunity set: Taiwan (left half) vs CovSyn context size (right half) — same quantity. '
                'The community layer has none: it draws Poisson(nu x lambda) contacts directly, so it no longer '
                'scales with city population (B27, finding E4)', fontsize=10)

# (2) candidate contacts vs candidate contacts: people actually contacted, per index case
groups = []
for layer, key, extra in [('Household', 'household', ''), ('School', 'school', 'students only'),
                          ('Workplace', 'workplace', 'employed only'), ('Health care', 'health_care', ''),
                          ('Municipality', 'municipality', '')]:
    specs, note = tracing_reference(key)
    groups.append((layer, specs, candpc[key if key != 'health_care' else 'healthcare'],
                   (extra + '\n' if extra else '') + note))
split_violins(ax[1], groups, 'Taiwan tracing', 'CovSyn (2) candidate', GREEN, BLUE, 'people per case', label_fontsize=6.5)
ax[1].legend(handles=[
    Patch(facecolor=GREEN, edgecolor='black', label='left: Taiwan, uninfected contacts reported (real contact count)'),
    Line2D([], [], marker='v', ls='none', markerfacecolor='white', markeredgecolor=GREEN, ms=8,
           label='left: Taiwan, only infected contacts known (lower bound)'),
    Patch(facecolor=BLUE, edgecolor='black', label='right: CovSyn (2) candidate contacts')],
    loc='upper left', fontsize=7.5)
ax[1].set_title('(2) Candidate contacts: Taiwan contact tracing (left half) vs CovSyn (right half) — same quantity, '
                'EXPLORATORY (small non-random samples)', fontsize=10)

x = np.arange(len(names))
p_ci = [wilson(eff[n], cand[n]) for n in names]
ax[2].bar(x, [100 * p for p, _, _ in p_ci], color=BLUE, label='CovSyn')
ax[2].errorbar(x, [100 * p for p, _, _ in p_ci],
               yerr=[[100 * (p - lo) for p, lo, _ in p_ci], [100 * (hi - p) for p, _, hi in p_ci]],
               fmt='none', ecolor='black', capsize=3, lw=1)
for i, n in enumerate(names):
    if n in CHENG_SAR:
        k, tot = CHENG_SAR[n]
        p, lo, hi = wilson(k, tot)
        ax[2].errorbar(i, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt='D', color=GREEN, capsize=3, ms=7,
                       label='Cheng 2020 (all infections)' if i == 0 else None)
    ax[2].scatter(i, ANCHOR[n], marker='_', s=260, color='black', linewidths=2,
                  label='calibration anchor' if i == 0 else None)
ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=30)
ax[2].set_ylabel('secondary attack rate (%)')
ax[2].set_title('Attack rate by layer (effective / candidate)\nCheng "others" = school+workplace+municipality: '
                f'{100 * CHENG_OTHERS[0] / CHENG_OTHERS[1]:.2f}%', fontsize=10)
ax[2].legend(fontsize=8)

share = np.array([eff[n] / total_eff if total_eff else 0 for n in names])
tw_counts = np.zeros(len(names))
tw_n = 0
if REFERENCE.exists():
    import csv
    with open(REFERENCE, newline='') as fh:
        rows = [r for r in csv.DictReader(fh)
                if r['dataset'] == 'first_wave_2020' and r['infectee_type'] == 'Local' and r['layer'] != 'no_covsyn_layer']
    key = {'household': 'household', 'school': 'school', 'workplace': 'workplace',
           'healthcare': 'health_care', 'municipality': 'municipality'}
    tw_counts = np.array([sum(1 for r in rows if r['layer'] == key[n]) for n in names], float)
    tw_n = int(tw_counts.sum())
w = 0.38
ax[3].bar(x - w / 2, 100 * share, w, color=BLUE, label=f'CovSyn ({total_eff:,} infections)')
if tw_n:
    tw_share = tw_counts / tw_n
    ci = np.array([wilson(k, tw_n) for k in tw_counts])
    ax[3].bar(x + w / 2, 100 * tw_share, w, color=GREEN, label=f'Taiwan contact tracing (n={tw_n})')
    ax[3].errorbar(x + w / 2, 100 * tw_share, yerr=[100 * (tw_share - ci[:, 1]), 100 * (ci[:, 2] - tw_share)],
                   fmt='none', ecolor='black', capsize=3, lw=1)
ax[3].set_xticks(x); ax[3].set_xticklabels(names, rotation=30)
ax[3].set_ylabel('% of secondary infections')
ax[3].set_title('Share of secondary infections by layer\nCheng 2020: household 68%, health care 27%, others 5% (22 cases)',
                fontsize=10)
ax[3].legend(fontsize=8)

mean_eff = [float(np.mean(effpc[n])) for n in names]
bars = ax[4].bar(x, mean_eff, 0.6, color=ORANGE, label=f'CovSyn ({len(offspring):,} cases)')
ax[4].bar_label(bars, fmt='%.3f', fontsize=7)
for i, n in enumerate(names):
    if n in CHENG_PER_INDEX:
        ax[4].scatter(i, CHENG_PER_INDEX[n], marker='D', color=GREEN, s=60, zorder=4,
                      label='Cheng 2020, per index case' if i == 0 else None)
ax[4].set_xticks(x); ax[4].set_xticklabels(names, rotation=30)
ax[4].set_ylabel('infections per case')
ax[4].set_title('Mean effective contacts (infections) by layer\n'
                f'Cheng 2020 "others" = school+workplace+municipality: {CHENG_PER_INDEX_OTHERS:.2f} per index case, '
                f'CovSyn {sum(np.mean(effpc[n]) for n in ("school", "workplace", "municipality")):.3f}', fontsize=9)
ax[4].legend(fontsize=7)

ax[6].hist(np.clip(degree, 0, 40), bins=range(0, 42), color=BLUE); ax[6].set_title('Candidate contacts per case (degree)')
ax[6].set_xlabel('candidate contacts'); ax[6].set_yscale('log')
ax[7].hist(np.clip(offspring, 0, 20), bins=range(0, 22), color=BLUE); ax[7].set_title('Effective contacts per case (offspring)')
ax[7].set_xlabel('secondary infections'); ax[7].set_yscale('log')
fig.suptitle(f'Stage 3 — Infection probability chain by social layer ({SYN}, {len(offspring):,} cases)',
             fontsize=14, fontweight='bold')
fig.text(0.01, 0.005, MARKER_NOTE + ' Each comparison uses the SAME quantity on both halves. Panel (1) opportunity set: household = '
         'household of a random person (Ministry of the Interior 2021, size-weighted), school = classmates per student (Ministry of '
         'Education enrolment files), workplace = the whole establishment a random worker is employed in (industry and service census '
         '2016, size-weighted; no Taiwan workgroup data exists, so this is the entire company, while CovSyn caps the group actually '
         'met inside it following Chen 2022, B31); health care has no reference yet, and the community layer no longer has an '
         'opportunity set at all - it draws Poisson(nu x lambda) contacts directly, independent of the city (B27, which fixes E4). '
         'Panel (2) candidate contacts = contacted on at least one day from infection to isolation, no duration threshold (C2), '
         'against close contacts per index case recorded in Taiwan CDC contact tracing (taiwan_covid_figshare.xlsx, relationships '
         'mapped as in F9). The press releases report the uninfected contacts for only some index cases: dark green (density or filled '
         'points) are the cases where they were reported, so the total is a real contact count; hollow triangles are cases where only '
         'the infected contacts are known, so that total is a lower bound and sits near 1. The statistics (mean, median, box) use the '
         'dark green group only, and each label gives both counts. EXPLORATORY: small non-random samples, mostly imported index cases, '
         'tracing window not reported, index cases of one cluster can share a contact list. '
         'Mean effective contacts panel: CovSyn = infections per case in these spread simulations (all generations); Cheng et al. 2020 '
         '= their 22 secondary cases divided by their 100 index cases (household includes the 5 non-household-family cases), so the '
         'two denominators are not identical. Every CovSyn number in this figure comes from the spread dataset only. '
         'Attack-rate and share panels: Cheng et al. 2020 Table 2 (100 index cases, traced from symptom onset, face-to-face >15 min '
         'without PPE); share panel uses local infectees with a recorded infector. Whiskers are 95% Wilson intervals.',
         fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.96]); fig.savefig(OUT / 'covsyn_stage3.png', dpi=120); plt.close(fig)

# ===================== STAGE 4 =====================
rr_source = f'{total_eff:,} infections'
age_cand = np.array(age_cand); age_eff = np.array(age_eff)
bins = [(0, 19), (20, 39), (40, 59), (60, 120)]; blab = ['0-19', '20-39', '40-59', '60+']
grp_contacts = np.array([float(((age_cand >= lo) & (age_cand <= hi)).sum()) for lo, hi in bins])
grp_infected = np.array([float(age_eff[(age_cand >= lo) & (age_cand <= hi)].sum()) for lo, hi in bins])
grp_sar = np.divide(grp_infected, grp_contacts, out=np.zeros(4), where=grp_contacts > 0)
rr = grp_sar / grp_sar[1] if grp_sar[1] else np.zeros(4)
with open('./variable/demographic_parameters.pkl', 'rb') as fp:
    age_p = np.array(pickle.load(fp)[0], float)
pop_age = np.array([age_p[lo:min(hi + 1, len(age_p))].sum() for lo, hi in bins]); pop_age /= pop_age.sum()

fig, ax = plt.subplots(2, 3, figsize=(16, 9)); ax = ax.ravel()
off = np.array(offspring)
ax[0].hist(np.clip(off, 0, 20), bins=range(0, 22), color=BLUE); ax[0].set_yscale('log')
ax[0].set_title('Offspring distribution (secondary infections / infector)'); ax[0].set_xlabel('# infected')
frac = [np.mean(off == k) for k in range(0, 8)]
ax[1].bar(range(0, 8), frac, color=BLUE); ax[1].set_title('Fraction of infectors infecting k people (overdispersion)')
ax[1].set_xlabel('k'); ax[1].set_ylabel('fraction')
ax[2].hist(np.clip(degree, 0, 40), bins=range(0, 42), color=BLUE); ax[2].set_yscale('log')
ax[2].set_title('Degree distribution (contacts / case)'); ax[2].set_xlabel('contacts')

xa = np.arange(4); wa = 0.38
ax[3].bar(xa - wa / 2, rr, wa, color=BLUE, label='CovSyn (measured)')
ax[3].bar(xa + wa / 2, CHENG_RR, wa, color=GREEN, label='Cheng 2020 (all infections)')
ax[3].axhline(1, color='gray', ls=':')
ax[3].set_xticks(xa); ax[3].set_xticklabels(blab); ax[3].legend(fontsize=8)
ax[3].set_title('Age risk ratio of the secondary attack rate\n' + f'(reference 20-39; {rr_source.split(",")[-1].strip()})', fontsize=10)
ax[3].set_ylabel('risk ratio')
for i, v in enumerate(rr):
    ax[3].text(i - wa / 2, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)

ax[4].bar(np.arange(len(names)), [100 * a for a in ar_layer], color=ORANGE)
ax[4].set_xticks(np.arange(len(names))); ax[4].set_xticklabels(names, rotation=30)
ax[4].set_ylabel('attack rate (%)'); ax[4].set_title('Attack rate by social layer')
top = np.sort(off)[::-1]; cum = np.cumsum(top) / max(top.sum(), 1)
ax[5].plot(np.arange(1, len(top) + 1) / len(top) * 100, cum * 100, color=BLUE)
ax[5].set_title('Cumulative share of infections by top infectors')
ax[5].set_xlabel('% of infectors (ranked)'); ax[5].set_ylabel('% of all infections')
fig.suptitle('Stage 4 — Attack rate & infectiousness (offspring / overdispersion / age)', fontsize=14, fontweight='bold')
fig.text(0.01, 0.005, 'Age risk ratio uses every candidate contact as the denominator ({layer}_contact_ages, covsyn_decisions.md D4); '
         'the age risk ratios are locked model inputs (B2), so this panel is a consistency check, not an independent validation. '
         f'Population age distribution for reference: {", ".join(f"{b} {100*p:.0f}%" for b, p in zip(blab, pop_age))}.',
         fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.96]); fig.savefig(OUT / 'covsyn_stage4.png', dpi=120); plt.close(fig)

# ===================== STAGE 5 =====================
T = 90
def daily(arr):
    d = np.zeros(T)
    for v in arr:
        iv = int(round(v))
        if 0 <= iv < T: d[iv] += 1
    return d
di, dc, dr, dd = daily(inf_day), daily(conf_day), daily(rec_day), daily(death_day)
ci, cc, cr, cd_ = np.cumsum(di), np.cumsum(dc), np.cumsum(dr), np.cumsum(dd)
active = ci - cr - cd_
case_infday = np.array(case_infday); case_off = np.array(case_off)
Rt = np.array([case_off[(case_infday >= t) & (case_infday < t + 1)].mean() if ((case_infday >= t) & (case_infday < t + 1)).sum() else np.nan for t in range(T)])
fig, ax = plt.subplots(3, 3, figsize=(16, 13)); ax = ax.ravel()
days = np.arange(T)
ax[0].bar(days, di, color=BLUE); ax[0].set_title('Daily infections')
ax[1].plot(days, ci, color=BLUE); ax[1].set_title('Cumulative infections')
ax[2].bar(days, dc, color=BLUE); ax[2].set_title('Daily confirmed cases')
ax[3].plot(days, cc, color=BLUE); ax[3].set_title('Cumulative confirmed cases')
ax[4].plot(days, active, color=BLUE); ax[4].set_title('Active infected cases')
ax[5].plot(days, cr, color=GREEN, label='recovered'); ax[5].plot(days, cd_, color='#d62728', label='death')
ax[5].set_title('Cumulative recovered / death'); ax[5].legend()
ax[6].axhline(1, color='gray', ls=':'); ax[6].plot(days, Rt, color=ORANGE)
ax[6].set_title('Rt (mean offspring by infection day)'); ax[6].set_ylabel('Rt')
for g in growth[:40]:
    if g:
        u, c = np.unique(g, return_counts=True); ax[7].plot(u, np.cumsum(c), color=BLUE, alpha=0.15)
ax[7].set_title('Growth curves across seeds (40 sims, cumulative)')
ax[8].hist([len(g) for g in growth], bins=range(1, max(2, max(len(g) for g in growth) + 1)), color=BLUE)
ax[8].set_yscale('log'); ax[8].set_title('Outbreak size distribution (cases / sim)'); ax[8].set_xlabel('cases')
for a in ax[:8]: a.set_xlabel('day')
fig.suptitle(f'Stage 5 — Epidemic growth curves ({SYN}, {len(growth):,} simulations, contact_weight=1)',
             fontsize=14, fontweight='bold')
fig.text(0.01, 0.005, 'Each simulation starts from one seed case and runs 84 days. No external reference: the observed Taiwan first '
         'wave is a different scenario (28 local index cases seeded on their reported confirmation dates, E29) and is compared '
         'separately with the taiwan_first_outbreak mode.',
         fontsize=7.5, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97]); fig.savefig(OUT / 'covsyn_stage5.png', dpi=120); plt.close(fig)

print('saved', OUT / 'covsyn_stage3.png', OUT / 'covsyn_stage4.png', OUT / 'covsyn_stage5.png')
print('cases total:', len(offspring), '| attack rate by layer:', {n: round(r, 4) for n, r in zip(names, ar_layer)})
print('share of infections:', {n: round(s, 3) for n, s in zip(names, share)}, '| Taiwan tracing counts:',
      {n: int(c) for n, c in zip(names, tw_counts)})
print('age groups contacts:', grp_contacts.tolist(), 'infected:', grp_infected.tolist(), 'RR:', np.round(rr, 3).tolist())
print('mean offspring (R proxy):', round(np.mean(offspring), 3), '| max outbreak:', max(len(g) for g in growth))
