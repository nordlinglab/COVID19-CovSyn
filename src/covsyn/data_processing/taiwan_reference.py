"""Taiwan reference distributions and split-violin helpers.

Shared by validate_layers.py (Phase A external comparisons) and plot_diagnostics.py (stage 3),
so both figures use exactly the same references and the same drawing conventions:
split violins with the LEFT half = observed / reference and the RIGHT half = CovSyn
(covsyn_decisions.md D1-D3; individual points instead of a density when n < 30, D2).
"""
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

RNG = np.random.default_rng(0)
TICKS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
LEVELS = [('Elementary\n(age 7-12)', 7, 12), ('Junior high\n(age 13-15)', 13, 15),
          ('Senior high\n(age 16-18)', 16, 18), ('University\n(age 19-22)', 19, 22)]
REF_COLOR, SIM_COLOR, EFF_COLOR, LIGHT_SIM, IMPORTED_COLOR = '#59A14F', '#3a7ca5', '#E8A33D', '#9ecae1', '#B6D7A8'
MARKER_NOTE = ('Left half = observed/reference, right half = CovSyn. Red diamond = mean, white dot = median, '
               'black bar = interquartile range. Each half is a kernel density on log(1+x), scaled to the same width.')
TRACING_CSV = Path('validation_reference/taiwan_tracing_contacts_per_case.csv')


def load_demographics(path='./variable/demographic_parameters.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def municipality_population(demo):
    return {k: float(v) for k, v in demo[6].items()}


def pmf_samples(values, pmf, n=100000):
    pmf = np.asarray(pmf, float)
    return RNG.choice(np.asarray(values, float), size=n, p=pmf / pmf.sum())


def weighted_samples(values, weights, n=100000):
    w = np.asarray(weights, float)
    return RNG.choice(np.asarray(values, float), size=n, p=w / w.sum())


# --------------------------------------------------------------------------- references
def household_pmfs(demo):
    """Household size pmf as CovSyn samples it, per household, and as experienced by a random person."""
    fam = demo[5]
    pop = municipality_population(demo)
    cities = [c for c in pop if c in fam]
    sizes = np.arange(1, len(fam[cities[0]]) + 1)
    pmf_input, pmf_household, pmf_person = (np.zeros(len(sizes)) for _ in range(3))
    for c in cities:
        p = np.asarray(fam[c], float)
        mean_size = float(np.sum(sizes * p))
        pmf_input += pop[c] * p                          # what CovSyn samples
        pmf_household += (pop[c] / mean_size) * p        # random household
        pmf_person += pop[c] * sizes * p / mean_size     # household experienced by a random person
    pmf_input, pmf_household, pmf_person = (x / x.sum() for x in (pmf_input, pmf_household, pmf_person))
    return sizes, pmf_input, pmf_household, pmf_person


def write_household_pmf_csv(demo, path):
    sizes, pmf_input, pmf_household, pmf_person = household_pmfs(demo)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['household_size', 'other_members', 'taiwan_random_household', 'taiwan_random_person', 'covsyn_sampling_input'])
        for row in zip(sizes, sizes - 1, pmf_household, pmf_person, pmf_input):
            w.writerow([row[0], row[1]] + [round(v, 5) for v in row[2:]])


def school_reference(base=Path('data/demographic_data')):
    """Classmates per student from the raw Ministry of Education files, using the same columns as
    get_school_data() but weighting every school-grade by its students. Returns {level: (values, weights)}."""
    drop = ['金門縣', '連江縣', '澎湖縣']
    ref = {}

    def per_grade(df, class_pos, student_pos, n_grades):
        vals, weights = [], []
        for i in range(n_grades):
            classes = pd.to_numeric(df.iloc[:, class_pos + i], errors='coerce')
            pupils = (pd.to_numeric(df.iloc[:, student_pos + 2 * i], errors='coerce')
                      + pd.to_numeric(df.iloc[:, student_pos + 2 * i + 1], errors='coerce'))
            m = (classes > 0) & (pupils > 0)
            vals += list((pupils[m] / classes[m]).values)
            weights += list(pupils[m].values)
        return np.array(vals, float), np.array(weights, float)

    def read(name, extra=None):
        df = pd.read_excel(base / name)
        df.columns = df.iloc[1]
        df = df[2:]
        df = df[~df['縣市名稱'].isin(drop)]
        return df if extra is None else extra(df)

    ref[LEVELS[0][0]] = per_grade(read('國民小學校別資料.xls'), 6, 15, 6)
    ref[LEVELS[1][0]] = per_grade(read('國民中學校別資料.xlsx'), 6, 12, 3)
    ref[LEVELS[2][0]] = per_grade(read('高級中等學校校別資料檔.xls', lambda d: d[d['學程(等級)別'] != 'J']), 9, 16, 3)
    u = read('大專校院各校科系別學生數.xlsx', lambda d: d[(d['等級別'] == 'B 學士') & (d['縣市名稱'] != '71 金門縣')])
    vals, weights = [], []
    for i in range(4):   # no class counts: the department-year cohort is the contact group
        cohort = (pd.to_numeric(u.iloc[:, 9 + 2 * i], errors='coerce')
                  + pd.to_numeric(u.iloc[:, 9 + 2 * i + 1], errors='coerce'))
        m = cohort > 0
        vals += list(cohort[m].values)
        weights += list(cohort[m].values)
    ref[LEVELS[3][0]] = (np.array(vals, float), np.array(weights, float))
    return ref


def school_classmates_all(school_ref, n=100000):
    """Classmates per student (class size minus the student) pooled over all education levels."""
    values = np.concatenate([v for v, _ in school_ref.values()])
    weights = np.concatenate([w for _, w in school_ref.values()])
    return weighted_samples(values - 1, weights, n)


def workplace_establishment_distributions(demo):
    """Establishment size (persons employed) from the industry and service census as used by CovSyn:
    industry drawn by full-time employment share, then establishment size either by establishment
    count (the rule CovSyn uses) or size-biased (the establishment of a random worker)."""
    workplace_p, job = demo[8], demo[4]
    full_time = np.asarray(job['full_time_job_p'], float)
    per_establishment, per_worker = np.zeros(1), np.zeros(1)
    for j, name in enumerate(job['job_list']):
        if name not in workplace_p:
            continue
        p = np.asarray(workplace_p[name], float)
        share = full_time[0][j] + full_time[1][j] if full_time.ndim == 2 else full_time[j]
        size = np.arange(len(p))
        n = max(len(per_establishment), len(p))
        per_establishment = np.pad(per_establishment, (0, n - len(per_establishment)))
        per_worker = np.pad(per_worker, (0, n - len(per_worker)))
        per_establishment[:len(p)] += share * p
        per_worker[:len(p)] += share * size * p / max(float(np.sum(size * p)), 1e-12)
    size = np.arange(len(per_establishment))
    return size, per_establishment / per_establishment.sum(), per_worker / per_worker.sum()


def tracing_contacts(layer, dataset='first_wave_2020', path=TRACING_CSV):
    """Close contacts per index case recorded in Taiwan CDC contact tracing for one CovSyn layer.

    Returns (values, is_local, has_uninfected) or (None, None, None) when the extract is missing.
    `has_uninfected` marks the index cases whose uninfected contacts were reported: only for those is
    the total a real contact count. For the rest it is the number of infected contacts, a lower bound.
    """
    path = Path(path)
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    df = df[(df['layer'] == layer) & (df['dataset'] == dataset)]
    if df.empty:
        return None, None, None
    return (df['total'].to_numpy(float), (df['case_type'].str.lower() == 'local').to_numpy(),
            df['uninfected'].notna().to_numpy())


# --------------------------------------------------------------------------- split violins
def to_axis(x):
    # log(1+x) is defined for x > -1 only, and a duration measured between two reported dates
    # can be negative (a case confirmed by screening before its symptoms started). Clipping at
    # 0 keeps such a value on the axis instead of sending it to -inf and killing the density;
    # callers that care about the negative part must count it separately.
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    return np.log1p(np.asarray(x, float))


def half_violin(ax, center, data, side, color, width=0.42):
    """Draw one half of a violin (KDE on log(1+x)), with IQR bar, median and mean."""
    x = np.asarray(data, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return x
    y = to_axis(x)
    sign = -1 if side == 'left' else 1
    if np.ptp(y) < 1e-9:
        ax.plot([center, center + sign * width], [y[0], y[0]], color=color, lw=4, solid_capstyle='butt')
    else:
        grid = np.linspace(y.min(), y.max(), 256)
        density = gaussian_kde(y)(grid)
        density = density / density.max() * width
        ax.fill_betweenx(grid, center, center + sign * density, facecolor=color,
                         edgecolor='black', linewidth=0.6, alpha=0.9)
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    offset = sign * 0.07
    ax.vlines(center + offset, to_axis(q1), to_axis(q3), color='black', lw=3)
    ax.scatter(center + offset, to_axis(median), color='white', edgecolor='black', zorder=4, s=24)
    ax.scatter(center + offset, to_axis(x.mean()), color='red', marker='D', zorder=5, s=22)
    return x


def half_points(ax, center, values, colors, width=0.42, edgecolor='black', marker='o', stats=True):
    """Left half for very small samples: individual observations instead of a density (D2)."""
    x = np.asarray(values, float)
    jitter = RNG.uniform(0.12, width, size=len(x))
    ax.scatter(center - jitter, to_axis(x), c=list(colors), edgecolor=edgecolor, s=42, zorder=4, marker=marker)
    if stats:
        q1, median, q3 = np.percentile(x, [25, 50, 75])
        ax.vlines(center - 0.06, to_axis(q1), to_axis(q3), color='black', lw=3)
        ax.scatter(center - 0.06, to_axis(median), color='white', edgecolor='black', zorder=5, s=24)
        ax.scatter(center - 0.06, to_axis(x.mean()), color='red', marker='D', zorder=6, s=22)
    return x


def _draw_left(ax, center, spec, left_color, stats):
    """One drawing instruction for a left half: ('violin', data) or ('points', values, colors, marker, edge)."""
    if spec[0] == 'violin':
        return half_violin(ax, center, spec[1], 'left', left_color)
    values, colors = spec[1], spec[2]
    marker = spec[3] if len(spec) > 3 else 'o'
    edgecolor = spec[4] if len(spec) > 4 else 'black'
    return half_points(ax, center, values, colors, edgecolor=edgecolor, marker=marker, stats=stats)


def split_violins(ax, groups, left_name, right_name, left_color, right_color, ylabel, label_fontsize=8):
    """groups: list of (category, left data or None, right data, note).

    `left` may be an array (density), None ('reference pending'), a str (shown as a note), or
    ('points', values, colors) for a small sample."""
    labels, top = [], 0.0
    for i, (category, left, right, note) in enumerate(groups):
        if isinstance(left, list):          # several drawing instructions; the first one carries the statistics
            for k, spec in enumerate(left):
                shown = _draw_left(ax, i, spec, left_color, stats=(k == 0))
                if len(shown):
                    top = max(top, float(np.max(shown)))
                if k == 0:
                    left_text = f'{left_name}: mean {shown.mean():.2f}, median {np.median(shown):.1f} (n={len(shown)})'
        elif isinstance(left, tuple) and len(left) == 3 and left[0] == 'points':
            shown = half_points(ax, i, left[1], left[2])
            left_text = f'{left_name}: mean {shown.mean():.2f}, median {np.median(shown):.1f} (n={len(shown)})'
            top = max(top, float(np.max(shown)))
        elif left is None or isinstance(left, str):
            ax.text(i - 0.21, to_axis(2), 'reference\npending' if left is None else left, ha='center', va='center',
                    fontsize=8, color='gray', style='italic')
            left_text = f'{left_name}: pending' if left is None else f'{left_name}: separate figure'
        else:
            shown = half_violin(ax, i, left, 'left', left_color)
            left_text = f'{left_name}: mean {shown.mean():.2f} (n={len(shown):,})'
            top = max(top, float(np.max(shown)))
        shown = half_violin(ax, i, right, 'right', right_color)
        right_text = f'{right_name}: mean {shown.mean():.2f} (n={len(shown):,})' if len(shown) else f'{right_name}: no data'
        if len(shown):
            top = max(top, float(np.max(shown)))
        ax.vlines(i, 0, to_axis(max(top, 1)), color='black', lw=0.4, alpha=0.4)
        labels.append(f'{category}\n{left_text}\n{right_text}' + (f'\n{note}' if note else ''))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=label_fontsize)
    ticks = [t for t in TICKS if t <= max(top, 1) * 1.05] + [next((t for t in TICKS if t > max(top, 1) * 1.05), TICKS[-1])]
    ax.set_yticks(to_axis(ticks))
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylim(-0.1, to_axis(ticks[-1]))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel + ' (log(1+x) scale)')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(handles=[Patch(facecolor=left_color, edgecolor='black', label=f'left half: {left_name}'),
                       Patch(facecolor=right_color, edgecolor='black', label=f'right half: {right_name}')],
              loc='upper left', fontsize=8)
