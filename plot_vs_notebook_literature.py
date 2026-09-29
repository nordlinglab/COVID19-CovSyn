"""Plot CovSyn (optimized) disease-course distributions against the literature review
defined in plot_result/plot_synthetic_data.ipynb (the user's own report_* dictionaries).
Two literature bands per panel, mirroring the notebook: wide = CI range (min low .. max high),
narrow = mean range (min study-mean .. max study-mean)."""
import sys, json, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from Data_synthesize import Draw_course_of_disease_data

# Usage: python plot_vs_notebook_literature.py [spread_dir] [out_dir] [parameter_dir]
NB = 'plot_result/plot_synthetic_data.ipynb'
SYN = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
PARAM = sys.argv[3] if len(sys.argv) > 3 else 'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200'
OUT.mkdir(parents=True, exist_ok=True)
N = 20000

# ---- 1. extract literature report_* dicts from the notebook ----
# Robust: pull each `report_X = {...}` literal by brace-matching and eval it in isolation,
# so surrounding (possibly broken) notebook code can't stop the extraction.
nb = json.load(open(NB, encoding='utf-8'))
fullsrc = '\n'.join(''.join(c.get('source', [])) for c in nb.get('cells', []) if c.get('cell_type') == 'code')

def extract_dict(name):
    i = -1
    for key in (name + ' = {', name + ' ={', name + '={'):
        i = fullsrc.find(key)
        if i >= 0:
            break
    if i < 0:
        return None
    j = fullsrc.index('{', i)
    depth = 0
    for k in range(j, len(fullsrc)):
        if fullsrc[k] == '{':
            depth += 1
        elif fullsrc[k] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return eval(fullsrc[j:k + 1], {'np': np})
                except Exception as e:
                    print(name, 'eval failed:', e)
                    return None
    return None

def lit_ranges(d):
    means = [v[0] for v in d.values() if v[0] is not None and not np.isnan(v[0])]
    los = [v[1] for v in d.values() if len(v) > 1 and not np.isnan(v[1])]
    his = [v[2] for v in d.values() if len(v) > 2 and not np.isnan(v[2])]
    return (min(means), max(means), min(los), max(his), len(d))

LIT = {}
mapping = {
    'Latent period': 'report_latent_period',
    'Incubation period': 'report_incubation_period',
    'Infectious period': 'report_infectious_period',
    'Generation time': 'report_generation_time',
    'Serial interval': 'report_serial_interval',
}
for label, var in mapping.items():
    d = extract_dict(var)
    if d:
        LIT[label] = lit_ranges(d)
        m = LIT[label]
        print(f'{label:18s} mean-range [{m[0]:.2f},{m[1]:.2f}]  CI-range [{m[2]:.2f},{m[3]:.2f}]  ({m[4]} studies)')
    else:
        print(f'{label}: NOT FOUND in notebook')

# ---- 2. synthetic latent / incubation / infectious from drawn courses ----
res = np.loadtxt(Path(PARAM) / 'firefly_best.txt'); P = res[np.argmin(res[:, -1]), 1:-1]
mk = lambda *ix_keys: {k: P[i] for k, i in ix_keys}
lat = mk(('latent_period_shape', 37), ('latent_period_scale', 38))
inf = mk(('infectious_period_shape', 39), ('infectious_period_scale', 40))
inc = mk(('incubation_period_shape', 41), ('incubation_period_scale', 42))
s2i = mk(('symptom_to_confirmed_shape', 43), ('symptom_to_confirmed_scale', 44), ('symptom_to_confirmed_loc', 45))
a2r = mk(('asymptomatic_to_recovered_shape', 46), ('asymptomatic_to_recovered_scale', 47), ('asymptomatic_to_recovered_loc', 48))
s2c = mk(('symptomatic_to_critically_ill_shape', 49), ('symptomatic_to_critically_ill_scale', 50), ('symptomatic_to_critically_ill_loc', 51))
s2r = mk(('symptomatic_to_recovered_shape', 52), ('symptomatic_to_recovered_scale', 53), ('symptomatic_to_recovered_loc', 54))
c2r = mk(('critically_ill_to_recovered_shape', 55), ('critically_ill_to_recovered_scale', 56), ('critically_ill_to_recovered_loc', 57))
i2d = mk(('infection_to_death_shape', 58), ('infection_to_death_scale', 59))
n2c = mk(('negative_to_confirmed_shape', 60), ('negative_to_confirmed_scale', 61), ('negative_to_confirmed_loc', 62))
np.random.seed(0)
import random; random.seed(0)
latent_a, incub_a, infect_a = [], [], []
for _ in range(N):
    o = Draw_course_of_disease_data(0, lat, inf, inc, s2i, a2r, s2c, s2r, c2r, i2d, n2c, P[67], [P[195], P[196], P[197]])
    o.draw_course_of_disease()
    latent_a.append(o.latent_period); infect_a.append(o.infectious_period)
    if not (isinstance(o.incubation_period, float) and np.isnan(o.incubation_period)):
        incub_a.append(o.incubation_period)

# ---- 3. generation time & serial interval from synthetic transmission pairs ----
NA = lambda x: x is None or (isinstance(x, float) and (np.isnan(x) or x <= -1 or x >= 1e9))
def to_id(v):
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    return None if np.isnan(f) else int(f)
gen, serial = [], []
for f in sorted(glob.glob(f'{SYN}/course_of_disease_data_*.npy'), key=lambda p: int(p.split('_')[-1].split('.')[0])):
    sim = int(f.split('_')[-1].split('.')[0])
    items = list(np.load(f, allow_pickle=True))
    dg = np.load(f'{SYN}/transmission_digraph_{sim}.npy', allow_pickle=True)
    by = {i + 1: items[i] for i in range(len(items))}
    for e in dg:
        pid, cid = to_id(e[0]), to_id(e[1])
        if pid is None or cid is None or pid not in by or cid not in by:
            continue
        p, c = by[pid], by[cid]
        gen.append(c['infection_day'] - p['infection_day'])
        if not NA(p['incubation_period']) and not NA(c['incubation_period']):
            serial.append((c['infection_day'] + c['incubation_period']) - (p['infection_day'] + p['incubation_period']))

# ---- 4. plot ----
panels = [
    ('Latent period', np.array(latent_a, float)),
    ('Incubation period', np.array(incub_a, float)),
    ('Infectious period', np.array(infect_a, float)),
    ('Generation time', np.array(gen, float)),
    ('Serial interval', np.array(serial, float)),
]
fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.ravel()
for ax, (name, data) in zip(axes, panels):
    if len(data) == 0:
        ax.set_visible(False); continue
    hi_x = int(np.nanpercentile(data, 99)) + 2
    ax.hist(data, bins=range(0, hi_x + 1), density=True, color='#4C78A8', alpha=0.75, edgecolor='white')
    m = np.nanmean(data)
    ax.axvline(m, color='k', ls='--', lw=2, label=f'CovSyn mean={m:.1f}d')
    if name in LIT:
        ml, mh, cl, ch, ns_ = LIT[name]
        ax.axvspan(cl, ch, color='#FF9DA6', alpha=0.30, label=f'lit. CI range [{cl:.1f},{ch:.1f}]')
        ax.axvspan(ml, mh, color='#59A14F', alpha=0.40, label=f'lit. mean range [{ml:.1f},{mh:.1f}] ({ns_} studies)')
    ax.set_title(name, fontsize=12, fontweight='bold'); ax.set_xlabel('days'); ax.legend(fontsize=8)
axes[5].set_visible(False)
fig.suptitle('CovSyn (optimized) vs literature review (from plot_synthetic_data.ipynb)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'covsyn_vs_notebook_literature.png', dpi=130)
print('\nsaved', OUT / 'covsyn_vs_notebook_literature.png')
print(f'synthetic n: latent {len(latent_a)}, incub {len(incub_a)}, infectious {len(infect_a)}, gen {len(gen)}, serial {len(serial)}')
