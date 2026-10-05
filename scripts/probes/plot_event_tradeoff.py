# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot the output of probe_event_tradeoff.py: four quantities against the event probability.

Usage: python scripts/probes/plot_event_tradeoff.py PROBE_CSV OUT_PNG [RUN11_P199]
"""
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURFACE, BAND = '#1f1f1e', '#6b6a63', '#e6e5df', '#fcfcfb', '#eceae3'
CHENG_OTHERS_PER_100 = 1822          # Cheng et al. 2020 Table 1 'Others', binned (n = 100 cases)
TAIL_BAND = (5.5, 93.1)              # tracing data bootstrap 95% CI (E77, B50)
INFECTIONS_UPPER = 0.03              # infections per index case, municipality (E57)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_xlabel('mass-event probability per case, P[199]', fontsize=9, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)


def mark_run11(ax, p199, y=0.0):
    ax.axvline(p199, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
    ax.annotate(f'run 11 chose {p199:.3f}', (p199, y), xycoords=('data', 'axes fraction'),
                xytext=(4, 4), textcoords='offset points', fontsize=8, color=MUTED, va='bottom')


def line(ax, d, column, color, marker='o', label=None):
    ax.plot(d['event_probability'], d[column], color=color, linewidth=2, marker=marker,
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)


def main():
    d = pd.read_csv(sys.argv[1]).sort_values('event_probability')
    p199 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0122
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURFACE)
    (a, b), (c, e) = axes  # e: panel D

    style(a, 'A  Community tail ratio, p90 / median (3,000 cases)', 'tail ratio')
    a.axhspan(TAIL_BAND[0], d['tail_ratio'].max() * 1.15, color=BAND, linewidth=0)
    a.text(0.2, d['tail_ratio'].max() * 1.13, 'target band: tracing data 95% CI, 5.5 to 93.1  ',
           fontsize=8, color=MUTED, va='top', ha='right')
    line(a, d, 'tail_ratio', BLUE)
    a.set_ylim(0, d['tail_ratio'].max() * 1.15)
    mark_run11(a, p199)

    style(b, "B  Cheng 'others' contacts per 100 cases (3,000 cases)", 'contacts per 100 cases')
    b.axhline(CHENG_OTHERS_PER_100, color=INK, linewidth=1)
    b.text(0.2, CHENG_OTHERS_PER_100, f'Cheng et al. 2020: {CHENG_OTHERS_PER_100}  ',
           fontsize=8, color=INK, va='top', ha='right')
    line(b, d, 'others_contacts_per_100_cases', BLUE)
    b.set_ylim(0, max(d['others_contacts_per_100_cases'].max(), CHENG_OTHERS_PER_100) * 1.12)
    mark_run11(b, p199)

    style(c, "C  Cheng 'others' contact cost", 'cost')
    line(c, d, 'expected_cost_contact_others', BLUE, 'o', 'expected (mean of 3,000 cases)')
    line(c, d, 'objective_cost_contact_others', ORANGE, 's', 'what the optimizer saw (100 fixed seeds)')
    c.legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left')
    c.set_ylim(0, d['objective_cost_contact_others'].max() * 1.15)
    mark_run11(c, p199, y=0.45)

    style(e, 'D  Community infections per index case (3,000 cases)', 'infections per index case')
    e.axhline(INFECTIONS_UPPER, color=INK, linewidth=1)
    e.text(0.2, INFECTIONS_UPPER, 'target upper bound 0.03  ', fontsize=8, color=INK, va='top',
           ha='right')
    line(e, d, 'community_infections_per_index', BLUE)
    e.set_ylim(0, max(d['community_infections_per_index'].max(), INFECTIONS_UPPER) * 1.2)
    mark_run11(e, p199)

    fig.suptitle('Run 11 best vector with only the mass-event probability varied (CovSyn B54)',
                 x=0.06, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(sys.argv[2], dpi=200, facecolor=SURFACE)
    print('wrote', sys.argv[2])


if __name__ == '__main__':
    main()
