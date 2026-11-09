"""The workplace sampling options that finding E3 was decided on, kept as the record of that choice.

CovSyn used to draw work_group_size from the 2016 industry and service census with every
ENTERPRISE UNIT equally likely. An infected person is a worker, not a company, so the
size-weighted (per worker) distribution is the one a random employee experiences. This figure
is what that comparison looked like, and it is why decision B31 moved the model to per-worker
sampling -- with a second stage on top, because no single contact probability could reproduce
both the mean and the median of the tracing records (panel b): the enterprise is drawn per
worker and the group actually met inside it is capped by a log-normal that does not grow with
the company (Chen 2022). The CovSyn series below is therefore the CURRENT model, whose work
group is that capped group; fig_workplace_establishment_size_sampling_diagnostic.png in
validate_layers.py shows the enterprise size against the same census curves.

Usage: python -m covsyn.figures.workplace_sampling_options [spread_dir] [out_dir] [parameter_dir]
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from covsyn.data_processing.taiwan_reference import (REF_COLOR, SIM_COLOR, half_violin, load_demographics, to_axis, tracing_contacts,
                              workplace_establishment_distributions)

SPREAD = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight_MC1000'
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'validation_figures_run5')
PARAM = Path(sys.argv[3] if len(sys.argv) > 3 else 'Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
N = 200000
TRACING_MEDIAN = 2.5          # coworkers per index case, Taiwan CDC tracing, full records only
ALT_COLOR, CUR_COLOR = '#B07AA1', '#E8A33D'

DEMO = load_demographics()
size, pmf_establishment, pmf_worker = workplace_establishment_distributions(DEMO)
res = np.loadtxt(PARAM / 'firefly_best.txt')
P = res[int(np.argmin(res[:, -1])), 1:-1]
p0 = float(P[14])             # workplace contact probability, Binomial(work_group_size, p0)

# ---------------------------------------------------------------- CovSyn as it runs now
context, candidate, effective = [], [], []
for f in sorted(glob.glob(f'{SPREAD}/social_data_*.npy'), key=lambda p: int(Path(p).stem.split('_')[-1])):
    k = int(Path(f).stem.split('_')[-1])
    social = np.load(f, allow_pickle=True)
    contact = np.load(f'{SPREAD}/contact_data_{k}.npy', allow_pickle=True)
    for s, c in zip(social, contact):
        n = float(s.get('work_group_size') or 0)
        if n <= 0:
            continue
        eff = list(c.get('workplace_effective_contacts') or [])
        context.append(n); candidate.append(len(eff)); effective.append(sum(1 for x in eff if x == 1))
context, candidate, effective = np.array(context), np.array(candidate), np.array(effective)
sar = effective.sum() / max(candidate.sum(), 1)     # infections per workplace candidate contact


def draw(pmf, p):
    n = RNG.choice(size, size=N, p=pmf / pmf.sum())
    return n, RNG.binomial(np.maximum(n.astype(int), 0), p)


def fit_p(pmf, target, statistic):
    lo, hi = 1e-5, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        _, c = draw(pmf, mid)
        if statistic(c) < target:
            lo = mid
        else:
            hi = mid
    return mid


n_est, c_est = draw(pmf_establishment, p0)
n_wrk, c_wrk = draw(pmf_worker, p0)
p_mean = fit_p(pmf_worker, candidate.mean(), np.mean)
p_median = fit_p(pmf_worker, TRACING_MEDIAN, np.median)
_, c_fit_mean = draw(pmf_worker, p_mean)
_, c_fit_median = draw(pmf_worker, p_median)

tracing, is_local, has_uninfected = tracing_contacts('workplace')
full = tracing[has_uninfected] if tracing is not None else np.array([])

TICKS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def violin(ax, i, data, color):
    half_violin(ax, i, data, 'left', color)
    return half_violin(ax, i, data, 'right', color)


def finish(ax, labels, ylabel, top):
    ticks = [t for t in TICKS if t <= top * 1.05] + [next((t for t in TICKS if t > top * 1.05), TICKS[-1])]
    ax.set_yticks(to_axis(ticks)); ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylim(-0.1, to_axis(ticks[-1]))
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel + ' (log(1+x) scale)')
    ax.grid(axis='y', alpha=0.25)


fig, axes = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.5, 1]})

# (a) which establishment-size distribution
for i, (data, color) in enumerate([(n_est, REF_COLOR), (n_wrk, ALT_COLOR), (context, SIM_COLOR)]):
    violin(axes[0], i, data, color)
finish(axes[0], [f'Census\nper establishment\nmean {n_est.mean():.1f}, median {np.median(n_est):.0f}',
                 f'Census\nper worker\nmean {n_wrk.mean():.1f}, median {np.median(n_wrk):.0f}',
                 f'CovSyn now\nwork group size\nmean {context.mean():.1f}, median {np.median(context):.0f}'],
       'persons employed in the establishment', float(np.max(n_wrk)))
axes[0].set_title('(a) Which census weighting does CovSyn follow?\nit follows "per establishment" (E3)', fontsize=11)

# (b) resulting candidate contacts under each rule
series = [(candidate, SIM_COLOR, f'CovSyn now\n(per establishment, p0={p0:.2f})'),
          (c_wrk, ALT_COLOR, f'Per worker\nsame p0={p0:.2f}'),
          (c_fit_mean, ALT_COLOR, f'Per worker\np0={p_mean:.3f} (mean kept)'),
          (c_fit_median, ALT_COLOR, f'Per worker\np0={p_median:.3f} (median kept)')]
labels = []
for i, (data, color, label) in enumerate(series):
    violin(axes[1], i, data, color)
    labels.append(f'{label}\nmean {np.mean(data):.1f}, median {np.median(data):.0f}\n'
                  f'p95 {np.percentile(data, 95):.0f}, zero {np.mean(np.asarray(data) == 0) * 100:.0f}%')
if len(full):
    jitter = RNG.uniform(-0.3, 0.3, size=len(full))
    axes[1].scatter(len(series) + jitter, to_axis(full), c=REF_COLOR, edgecolor='black', s=45, zorder=4)
    axes[1].scatter(len(series), to_axis(np.median(full)), color='white', edgecolor='black', s=30, zorder=5)
    labels.append(f'Taiwan contact tracing\ncoworkers per index case\nmean {full.mean():.1f}, median {np.median(full):.1f}\n'
                  f'(n={len(full)}, full records only)')
finish(axes[1], labels, 'coworkers per employed case', float(np.percentile(c_wrk, 99.5)))
axes[1].set_title('(b) Coworkers contacted under each sampling rule\nno single contact probability reproduces both the mean '
                  'and the median of the tracing records', fontsize=11)

# (c) what that does to transmission
per_case = [np.mean(candidate) * sar, np.mean(c_wrk) * sar, np.mean(c_fit_mean) * sar, np.mean(c_fit_median) * sar]
bars = axes[2].bar(range(4), per_case, color=[SIM_COLOR, ALT_COLOR, ALT_COLOR, ALT_COLOR])
axes[2].bar_label(bars, fmt='%.3f', fontsize=9)
axes[2].set_xticks(range(4))
axes[2].set_xticklabels(['CovSyn now', 'per worker\nsame p0', 'per worker\nmean kept', 'per worker\nmedian kept'], fontsize=8)
axes[2].set_ylabel('workplace infections per employed case')
axes[2].set_title(f'(c) Effect on transmission\n(attack rate per candidate contact held at {100 * sar:.2f}%)', fontsize=11)

fig.suptitle('Workplace layer: the per-establishment vs per-worker choice behind decision B31',
             fontsize=13, fontweight='bold')
fig.text(0.01, 0.005, 'Census = industry and service census 2016 (enterprise units by persons employed, Tables 18/18-2), industries '
         f'mixed by full-time employment share, persons spread uniformly inside each size class. CovSyn now = {SPREAD}, employed '
         f'cases only (n={len(context):,}); its work group is the capped group of decision B31, so it sits below both census curves by construction. Simulated '
         f'rules draw {N:,} work group sizes from the census and then Binomial(size, p0) candidate contacts. Taiwan contact tracing = '
         'coworkers per index case where the uninfected contacts were also reported (taiwan_covid_figshare.xlsx, first wave 2020); '
         'small, non-random, all imported index cases. Panel (c) keeps the measured attack rate per candidate contact fixed, so it '
         'shows the mechanical effect of the contact count alone; the whole model would have to be refitted.', fontsize=7.5, wrap=True)
fig.legend(handles=[Patch(facecolor=SIM_COLOR, edgecolor='black', label='CovSyn as it runs now'),
                    Patch(facecolor=ALT_COLOR, edgecolor='black', label='per-worker sampling (E3 fix)'),
                    Patch(facecolor=REF_COLOR, edgecolor='black', label='Taiwan census / contact tracing'),
                    Line2D([], [], marker='D', ls='none', color='red', ms=6, label='mean (white dot = median)')],
           loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.075))
fig.tight_layout(rect=[0, 0.12, 1, 0.95])
fig.savefig(OUT / 'fig_workplace_sampling_options.png', dpi=150)
print('saved', OUT / 'fig_workplace_sampling_options.png')
print(f'p0={p0:.4f}  attack rate per candidate={sar:.4f}')
for (data, _, label) in series:
    print(f'{label.splitlines()[0]:22s} mean {np.mean(data):7.2f} median {np.median(data):5.0f} '
          f'p95 {np.percentile(data, 95):6.0f} zero {np.mean(np.asarray(data) == 0):.2f} '
          f'infections/case {np.mean(data) * sar:.3f}')
if len(full):
    print(f'Taiwan tracing (n={len(full)}) mean {full.mean():.2f} median {np.median(full):.1f}')
