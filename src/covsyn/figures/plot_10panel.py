"""Reproduce the 10-panel CovSyn-vs-literature figure (latent, incubation, infectious,
asymptomatic/symptomatic/pre-symptomatic/post-symptomatic infectious period, generation time,
serial interval, R0) using the optimized parameters + the notebook's report_* literature dicts.
Style matches plot_synthetic_data.ipynb: blue=Simulation, dashed=Sim mean,
orange=Reported mean range, green=Reported 95% CI range."""
import json, glob, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from covsyn.model.data_synthesize import Draw_course_of_disease_data
from covsyn.model.r0_network import R0_average_effective_contact

# Usage: python -m covsyn.figures.plot_10panel [spread_dir] [out_dir] [parameter_dir]
NB = 'plot_result/plot_synthetic_data.ipynb'
SYN = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
PARAM = sys.argv[3] if len(sys.argv) > 3 else 'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200'
OUT.mkdir(parents=True, exist_ok=True)
N = 20000

# ---------- literature extraction (brace/bracket matching) ----------
nb = json.load(open(NB, encoding='utf-8'))
src = '\n'.join(''.join(c.get('source', [])) for c in nb.get('cells', []) if c.get('cell_type') == 'code')
def extract(name):
    i = src.find(name + ' = ')
    if i < 0:
        return None
    j = i + len(name + ' = ')
    while src[j] in ' \n\t':
        j += 1
    if src[j] not in '{[':
        return None
    opn, cls = src[j], {'{': '}', '[': ']'}[src[j]]
    depth = 0
    for k in range(j, len(src)):
        if src[k] == opn: depth += 1
        elif src[k] == cls:
            depth -= 1
            if depth == 0:
                try:
                    return eval(src[j:k + 1], {'np': np})
                except Exception as e:
                    print(name, 'eval fail', e); return None
    return None
def rng(d):
    means = [v[0] for v in d.values() if not np.isnan(v[0])]
    los = [v[1] for v in d.values() if len(v) > 1 and not np.isnan(v[1])]
    his = [v[2] for v in d.values() if len(v) > 2 and not np.isnan(v[2])]
    return (min(means), max(means), min(los), max(his))
LIT = {k: rng(extract(v)) for k, v in {
    'latent': 'report_latent_period', 'incubation': 'report_incubation_period',
    'infectious': 'report_infectious_period', 'asymptomatic': 'report_asymptomatic_infectious_period',
    'presymptomatic': 'report_presymptomatic_infectious_period', 'postsymptomatic': 'report_postsymptomatic_infectious_period',
    'generation': 'report_generation_time', 'serial': 'report_serial_interval'}.items() if extract(v)}
# R0 reported (lists of [label, mean, lo, hi])
R0rows = (extract('R0') or []) + (extract('R0_RW') or [])
if R0rows:
    nums = lambda r: [x for x in r[1:] if isinstance(x, (int, float)) and not np.isnan(x)]
    LIT['R0'] = (min(nums(r)[0] for r in R0rows if nums(r)), max(nums(r)[0] for r in R0rows if nums(r)),
                 min(min(nums(r)) for r in R0rows if nums(r)), max(max(nums(r)) for r in R0rows if nums(r)))

# ---------- simulation: disease course (drawn) ----------
res = np.loadtxt(Path(PARAM) / 'firefly_best.txt'); P = res[np.argmin(res[:, -1]), 1:-1]
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
import random; random.seed(0)
latent, incub, infectious, asx_inf, sym_inf, pre_inf, post_inf = ([] for _ in range(7))
isnan = lambda x: isinstance(x, float) and np.isnan(x)
for _ in range(N):
    o = Draw_course_of_disease_data(0, *args, P[67], [P[195], P[196], P[197]])
    o.draw_course_of_disease()
    latent.append(o.latent_period); infectious.append(o.infectious_period)
    if isnan(o.incubation_period):                     # asymptomatic
        asx_inf.append(o.infectious_period)
    else:                                              # symptomatic
        incub.append(o.incubation_period)
        sym_inf.append(o.infectious_period)
        pre_inf.append(o.incubation_period - o.latent_period)
        post_inf.append(o.infectious_period - (o.incubation_period - o.latent_period))

# ---------- simulation: generation/serial/R0 (spread data) ----------
NA = lambda x: x is None or (isinstance(x, float) and (np.isnan(x) or x <= -1 or x >= 1e9))
def toid(v):
    try: f = float(v)
    except Exception: return None
    return None if np.isnan(f) else int(f)
gen, serial, R0 = [], [], []
for f in sorted(glob.glob(f'{SYN}/course_of_disease_data_*.npy'), key=lambda p: int(p.split('_')[-1].split('.')[0])):
    sim = int(f.split('_')[-1].split('.')[0])
    items = list(np.load(f, allow_pickle=True))
    cdat = list(np.load(f'{SYN}/contact_data_{sim}.npy', allow_pickle=True))
    dg = np.load(f'{SYN}/transmission_digraph_{sim}.npy', allow_pickle=True)
    if len(cdat):
        R0.append(R0_average_effective_contact(cdat))
    by = {i + 1: items[i] for i in range(len(items))}
    for e in dg:
        p, c = toid(e[0]), toid(e[1])
        if p in by and c in by:
            gen.append(by[c]['infection_day'] - by[p]['infection_day'])
            if not NA(by[p]['incubation_period']) and not NA(by[c]['incubation_period']):
                serial.append((by[c]['infection_day'] + by[c]['incubation_period']) - (by[p]['infection_day'] + by[p]['incubation_period']))

# ---------- plot ----------
panels = [
    ('Latent period', latent, 'latent', False), ('Incubation period', incub, 'incubation', False),
    ('Infectious period', infectious, 'infectious', False),
    ('Asymptomatic cases infectious period', asx_inf, 'asymptomatic', False),
    ('Symptomatic cases infectious period', sym_inf, 'postsymptomatic', False),
    ('Pre-symptomatic cases infectious period', pre_inf, 'presymptomatic', False),
    ('Post-symptomatic cases infectious period', post_inf, 'postsymptomatic', False),
    ('Generation time', gen, 'generation', False), ('Serial interval', serial, 'serial', False),
    ('R0', R0, 'R0', True)]
fig, axes = plt.subplots(4, 3, figsize=(16, 18)); axes = axes.ravel()
for idx, (title, data, key, islog) in enumerate(panels):
    ax = axes[idx]; data = np.array([d for d in data if d is not None and not isnan(float(d))], float)
    if len(data) == 0:
        ax.set_title(title + ' (no data)'); continue
    if islog:
        data = data[data > 0]
        ax.hist(data, bins=np.logspace(-1, 2, 30), color='#3a7ca5', label='Simulation')
        ax.set_xscale('log')
    else:
        ax.hist(data, bins=np.arange(-0.5, int(np.nanmax(data)) + 1.5, 1), rwidth=0.8, color='#3a7ca5', label='Simulation')
    ax.axvline(np.nanmean(data), color='k', ls='--', lw=2, label='Simulation mean')
    if key in LIT:
        ml, mh, cl, ch = LIT[key]
        ax.axvspan(cl, ch, alpha=0.3, color='#7DBA84', label='Reported 95% CI')
        ax.axvspan(ml, mh, alpha=0.45, color='#E8A33D', label='Reported mean')
    ax.set_xlabel(title + ' (days)' if not islog else 'R0'); ax.set_ylabel('Frequency')
    if idx == 0:
        ax.legend(fontsize=8)
for j in range(len(panels), len(axes)):
    axes[j].set_visible(False)
fig.suptitle(f'CovSyn (optimized, penalty-fixed) vs literature — 10 panels ({SYN})', fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT / 'covsyn_10panel.png', dpi=120)
print('saved', OUT / 'covsyn_10panel.png')
print('LIT keys:', list(LIT.keys()))
print('sim n: latent', len(latent), 'incub', len(incub), 'gen', len(gen), 'serial', len(serial), 'R0', len(R0))
