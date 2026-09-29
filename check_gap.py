"""Compare the latest firefly run's synthetic data against the literature reported-mean
ranges taken from plot_result/plot_synthetic_data.ipynb."""
import glob, json, random
import numpy as np
from pathlib import Path
from Data_synthesize import Draw_course_of_disease_data

PARAM = 'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200'
SYN = 'synthetic_data_results_spread_Taiwan_weight'
N = 20000

# ---------- literature reported-mean ranges ----------
nb = json.load(open('plot_result/plot_synthetic_data.ipynb', encoding='utf-8'))
src = '\n'.join(''.join(c.get('source', [])) for c in nb.get('cells', []) if c.get('cell_type') == 'code')
def ext(name):
    i = -1
    for key in (name + ' = {', name + ' ={', name + '={'):
        i = src.find(key)
        if i >= 0:
            break
    if i < 0:
        return None
    j = src.index('{', i); d = 0
    for k in range(j, len(src)):
        if src[k] == '{': d += 1
        elif src[k] == '}':
            d -= 1
            if d == 0:
                try: return eval(src[j:k+1], {'np': np})
                except Exception: return None
LIT = {}
for label, var in [('Latent', 'report_latent_period'), ('Incubation', 'report_incubation_period'),
                   ('Infectious', 'report_infectious_period'), ('Generation time', 'report_generation_time'),
                   ('Serial interval', 'report_serial_interval')]:
    d = ext(var)
    if d:
        means = [v[0] for v in d.values() if v[0] is not None and not np.isnan(v[0])]
        LIT[label] = (min(means), max(means))

# ---------- best firefly ----------
res = np.loadtxt(Path(PARAM) / 'firefly_best.txt')
best_row = int(np.argmin(res[:, -1]))
P = res[best_row, 1:-1]
print('=== best firefly (cost %.4f, row %d of %d) ===' % (res[best_row, -1], best_row, len(res)))
print('param gamma means: latent %.2f  infectious %.2f  incubation %.2f'
      % (P[37]*P[38], P[39]*P[40], P[41]*P[42]))
print('age risk ratios  : [%.3f, %.3f, %.3f, %.3f]' % tuple(P[63:67]))
print('overdispersion   : rate %.3f weight %.2f' % (P[35], P[36]))
print('transition p     : asx->R %.3f  sym->R %.3f  crit->R %.3f' % (P[195], P[196], P[197]))
for name, i in [('household', 70), ('school', 95), ('workplace', 120), ('healthcare', 145), ('municipality', 170)]:
    print('   mean daily attack rate %-13s %.4f' % (name, float(np.mean(P[i:i+25]))))

# ---------- realised course-of-disease ----------
mk = lambda *kv: {k: P[i] for k, i in kv}
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
np.random.seed(0); random.seed(0)
L, F, I = [], [], []
for _ in range(N):
    o = Draw_course_of_disease_data(0, lat, inf, inc, s2i, a2r, s2c, s2r, c2r, i2d, n2c, P[67], [P[195], P[196], P[197]])
    o.draw_course_of_disease()
    L.append(o.latent_period); F.append(o.infectious_period)
    if not (isinstance(o.incubation_period, float) and np.isnan(o.incubation_period)):
        I.append(o.incubation_period)

# ---------- generation time / serial interval / R0 from synthetic output ----------
NA = lambda x: x is None or (isinstance(x, float) and (np.isnan(x) or x <= -1 or x >= 1e9))
def to_id(v):
    try: f = float(v)
    except (ValueError, TypeError): return None
    return None if np.isnan(f) else int(f)
gen, serial, offspring_all = [], [], []
files = sorted(glob.glob(f'{SYN}/course_of_disease_data_*.npy'),
               key=lambda p: int(p.split('_')[-1].split('.')[0]))
print('\nsynthetic sims found:', len(files))
for f in files:
    sim = int(f.split('_')[-1].split('.')[0])
    items = list(np.load(f, allow_pickle=True))
    try: dg = np.load(f'{SYN}/transmission_digraph_{sim}.npy', allow_pickle=True)
    except Exception: continue
    by = {i + 1: items[i] for i in range(len(items))}
    nchild = {}
    for e in dg:
        pid, cid = to_id(e[0]), to_id(e[1])
        if pid is None or cid is None or pid not in by or cid not in by:
            continue
        nchild[pid] = nchild.get(pid, 0) + 1
        p, c = by[pid], by[cid]
        gen.append(c['infection_day'] - p['infection_day'])
        if not NA(p['incubation_period']) and not NA(c['incubation_period']):
            serial.append((c['infection_day'] + c['incubation_period']) - (p['infection_day'] + p['incubation_period']))
    for cid in by:
        offspring_all.append(nchild.get(cid, 0))

rows = [('Latent', L), ('Incubation', I), ('Infectious', F),
        ('Generation time', gen), ('Serial interval', serial)]
print('\n%-17s %9s   %-16s %s' % ('quantity', 'CovSyn', 'reported mean', 'gap'))
print('-' * 62)
for name, data in rows:
    if not len(data):
        print('%-17s   (no data)' % name); continue
    m = float(np.nanmean(np.array(data, float)))
    if name in LIT:
        lo, hi = LIT[name]
        if m < lo: gap = 'LOW  by %.2f d' % (lo - m)
        elif m > hi: gap = 'HIGH by %.2f d' % (m - hi)
        else: gap = 'OK'
        print('%-17s %8.2f d   [%.2f, %.2f]%s%s' % (name, m, lo, hi, ' ' * 4, gap))
    else:
        print('%-17s %8.2f d   (no lit range)' % (name, m))
if offspring_all:
    print('%-17s %8.2f     [0.30, 6.70]     %s' % ('R0 (mean offspring)', float(np.mean(offspring_all)),
          'OK' if 0.3 <= np.mean(offspring_all) <= 6.7 else 'OUT'))
print('\nn: latent %d, incub %d, infectious %d, gen %d, serial %d, cases %d'
      % (len(L), len(I), len(F), len(gen), len(serial), len(offspring_all)))
