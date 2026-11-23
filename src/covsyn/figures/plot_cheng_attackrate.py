"""Cheng-2020-comparable secondary attack rate diagnostic, WITH literature overlay.
Records the contacts as the firefly cost target does (E84, cheng_recording: the first day of
exposure inside Cheng's investigation period, each contact counted once, layers grouped as
Cheng: Household / Health care / Others), so the synthetic attack rate is defined like Cheng
2020 Fig 2 and like the objective."""
import glob, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from covsyn.calibration.cheng_recording import recorded_totals

# Usage:
#     uv run python -m covsyn.figures.plot_cheng_attackrate [data_dir] [out_dir]
# cheng2020-mode data matches Cheng's design
SYN = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
OUT.mkdir(parents=True, exist_ok=True)
BINS = ['<0', '0-3', '4-5', '6-7', '8-9', '>9']
# Cheng 2020 literature (from parameters_for_training.get_processed_contact_tracing_data)
CHENG_AR = {'Household': [4, 5.1, 16.7, 0, 0, 0], 'Health care': [0.8, 2, 2.6, 0, 0, 0], 'Others': [0, 0, 0.6, 0, 0, 0]}

# load all infectors (flat) across sims
course, contact = [], []
for f in sorted(glob.glob(SYN + '/course_of_disease_data_*.npy'), key=lambda p: int(p.split('_')[-1].split('.')[0])):
    sim = int(f.split('_')[-1].split('.')[0])
    course += list(np.load(f, allow_pickle=True))
    contact += list(np.load(SYN + f'/contact_data_{sim}.npy', allow_pickle=True))
print('total infectors:', len(course))

totals = recorded_totals(course, contact)
groups = {name: (totals[g, 0], totals[g, 1])
          for g, name in enumerate(('Household', 'Health care', 'Others'))}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (name, (c, i)) in zip(axes, groups.items()):
    ar = np.divide(i, c, out=np.zeros(6), where=c > 0) * 100
    overall = 100 * i.sum() / c.sum() if c.sum() else 0
    cheng = CHENG_AR[name]
    cheng_overall = 100 * np.sum(np.array(cheng) / 100 * np.array([1] * 6))  # informational only
    x = np.arange(6)
    ax.bar(x - 0.2, c / c.sum() * 100 if c.sum() else c, 0.4, color='#9ecae1', label='Synthetic contacts (%)')
    ax2 = ax.twinx()
    ax2.plot(x, ar, 'o-', color='#3a7ca5', lw=2, label='Synthetic attack rate')
    ax2.plot(x, cheng, 's--', color='#E8A33D', lw=2, label='Cheng 2020 attack rate')
    ax.set_xticks(x); ax.set_xticklabels(BINS); ax.set_xlabel('Days from onset to first exposure')
    ax.set_ylabel('Contacts (% of layer)'); ax2.set_ylabel('Clinical attack rate, %')
    ax.set_title(f'{name}\nsynthetic overall AR={overall:.2f}%', fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
fig.suptitle(f'Cheng-2020-comparable secondary attack rate (synthetic vs literature) — {SYN}', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / 'covsyn_cheng_attackrate.png', dpi=130)
print('saved', OUT / 'covsyn_cheng_attackrate.png')
for name, (c, i) in groups.items():
    ar = np.divide(i, c, out=np.zeros(6), where=c > 0) * 100
    print(f'{name:12s} overall AR={100*i.sum()/c.sum():.2f}%  per-bin={np.round(ar,1).tolist()}  Cheng={CHENG_AR[name]}')
