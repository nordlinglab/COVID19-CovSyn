# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every data comparison of the CovSyn paper (Wu & Nordling 2025, medRxiv), redrawn with the
corrected model.

The paper compares CovSyn with data or literature in Fig 3 and Table 2 (epidemiological
periods and R0), Fig 4 (optimizer convergence), Fig 5 (state transitions against the Taiwan
dataset), Fig 6 (Cheng et al. 2020 contacts and attack rates) and Fig 7 and Table 3 (the
Taiwan first outbreak). Each is rebuilt here from the notebook that drew it
(plot_result/plot_synthetic_data.ipynb, plot_result/plot_firefly.ipynb,
notebooks/Course_synthesis.ipynb, plot_result/plot_compare_previous_studies.ipynb and
notebooks/taiwan_first_outbreak.ipynb), with the same layout and data sources, and with the
notebooks' errors corrected (finding E90):

* Fig 3: the serial interval was computed with generate_generation_time, so the paper's serial
  interval (8.2 d) is the generation time again. It is the difference in onset days here.
  generate_generation_time also skips the last transmission edge; every edge is used here.
* Fig 5: the simulated 'confirmed' date was the isolation date. It is the positive test date
  here, which is what the Taiwan confirmed_date records. The simulated courses come from the
  first-outbreak runs, so severity depends on the cases' ages as it does in the model.
* Fig 6: the contact intervals were 5-95 % percentiles labelled as 95 % intervals; they are
  2.5-97.5 % here. Contacts are scaled to Cheng's 91 symptomatic index cases, as the objective
  does (E86).
* Fig 7: the simulated deaths were shifted by the fitted delay while the simulated cases were
  not, and the delay was chosen by a bisection that returned one shift and dated the axis with
  another. One shift, found by exhaustive search, is applied to everything here.

Usage (repository root, PYTHONPATH=src):
    python -m covsyn.figures.reproduce_wu2025 fig3 FIRST_OUTBREAK_DIR OUT_DIR
    python -m covsyn.figures.reproduce_wu2025 fig4 FIREFLY_RUN_DIR OUT_DIR [--no-evaluate]
    python -m covsyn.figures.reproduce_wu2025 fig5 FIRST_OUTBREAK_DIR OUT_DIR
    python -m covsyn.figures.reproduce_wu2025 fig6 CHENG2020_DIR OUT_DIR
    python -m covsyn.figures.reproduce_wu2025 fig7 FIRST_OUTBREAK_DIR OUT_DIR
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn  # noqa: E402

PALETTE = seaborn.color_palette()
STYLE = Path('rw_visualization.mplstyle')
SYNTHETIC_DATA_NOTEBOOK = Path('plot_result/plot_synthetic_data.ipynb')
TAIWAN_XLSX = Path('data/structured_course_of_disease_data/figshare_taiwan_covid.xlsx')
TAIWAN_POPULATION = 23008366
# notebooks/taiwan_first_outbreak.ipynb: the 28 local cases used as seed infections.
SOURCE_CONFIRMED_DATES = [
    '2020/1/28', '2020/1/30', '2020/2/3', '2020/2/19', '2020/2/23', '2020/2/28', '2020/3/5',
    '2020/3/13', '2020/3/18', '2020/3/18', '2020/3/19', '2020/3/20', '2020/3/22', '2020/3/20',
    '2020/3/24', '2020/3/26', '2020/3/26', '2020/3/28', '2020/3/28', '2020/3/29', '2020/3/31',
    '2020/3/31', '2020/4/2', '2020/4/2', '2020/4/3', '2020/4/4', '2020/4/8', '2020/4/12']
# The values the paper reports (Results and Table 2), printed next to the new ones.
PUBLISHED = {'latent': 3.8, 'incubation': 5.1, 'infectious': 21.8, 'generation': 8.2,
             'serial': 8.2, 'R0': 0.4}
BINS = ['<0', '0-3', '4-5', '6-7', '8-9', '>9']


def _style() -> None:
    if STYLE.exists():
        plt.style.use(str(STYLE))


def iter_runs(directory: Path, with_digraph: bool = False) -> Iterator[tuple]:
    """Yield (courses, contacts[, digraph]) for every Monte-Carlo run, in run order."""
    files = glob.glob(str(Path(directory) / 'course_of_disease_data_*.npy'))
    for f in sorted(files, key=lambda p: int(re.findall(r'(\d+)\.npy$', p)[0])):
        index = re.findall(r'(\d+)\.npy$', f)[0]
        courses = list(np.load(f, allow_pickle=True))
        contacts = list(np.load(Path(directory) / f'contact_data_{index}.npy', allow_pickle=True))
        if with_digraph:
            digraph = np.load(Path(directory) / f'transmission_digraph_{index}.npy',
                              allow_pickle=True)
            yield courses, contacts, digraph
        else:
            yield courses, contacts


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------------------ Fig 3
def literature_ranges(notebook: Path = SYNTHETIC_DATA_NOTEBOOK) -> dict[str, tuple]:
    """(mean low, mean high, CI low, CI high) per quantity, computed by the notebook's own cells.

    The cells that define the report_* dictionaries and the R0 lists are executed unchanged,
    so the ranges are exactly those of the published figure.
    """
    import scipy.stats as stats

    cells = json.load(open(notebook, encoding='utf-8'))['cells']
    namespace: dict[str, Any] = {'np': np, 'stats': stats, 'print': lambda *a, **k: None}
    for cell in cells:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and ('report_' in source or 'R0_RW' in source) \
                and 'add_subplot' not in source and 'contact_data_list' not in source:
            exec(source, namespace)
    ns = namespace

    def spread(report: dict) -> tuple:
        values = np.array([v for v in report.values()], dtype=float)
        return (np.nanmin(values[:, 0]), np.nanmax(values[:, 0]),
                np.nanmin(values), np.nanmax(values))

    return {
        'latent': (ns['min_mean_latent_period'], ns['max_mean_latent_period'],
                   ns['min_latent_period'], ns['max_latent_period']),
        'incubation': (ns['min_mean_incubation_period'], ns['max_mean_incubation_period'],
                       ns['min_incubation_period'], ns['max_incubation_period']),
        'infectious': spread(ns['report_infectious_period']),
        'asymptomatic': spread(ns['report_asymptomatic_infectious_period']),
        'symptomatic': spread(ns['report_infectious_period']),
        'presymptomatic': spread(ns['report_presymptomatic_infectious_period']),
        'postsymptomatic': spread(ns['report_postsymptomatic_infectious_period']),
        'generation': (ns['min_mean_generation_time'], ns['max_mean_generation_time'],
                       ns['min_generation_time'], ns['max_generation_time']),
        'serial': (ns['min_mean_serial_interval'], ns['max_mean_serial_interval'],
                   ns['min_serial_interval'], ns['max_serial_interval']),
        'R0': (ns['min_mean_R0'], ns['max_mean_R0'], ns['min_R0'], ns['max_R0']),
    }


def transmission_intervals(courses: list[dict], digraph: np.ndarray) -> tuple[list, list]:
    """Generation times and serial intervals of every transmission edge of one run.

    Each digraph row is [source id, case id, infection day, layer] with 1-based ids and
    'nan' as the source of a seed case. The serial interval needs both cases symptomatic.
    """
    generation, serial = [], []
    for source, target, *_ in digraph:
        if str(source) == 'nan':
            continue
        s, t = courses[int(float(source)) - 1], courses[int(float(target)) - 1]
        generation.append(t['infection_day'] - s['infection_day'])
        onset_s = s['infection_day'] + s['incubation_period']
        onset_t = t['infection_day'] + t['incubation_period']
        if not (np.isnan(onset_s) or np.isnan(onset_t)):
            serial.append(onset_t - onset_s)
    return generation, serial


def epidemiological_statistics(directory: Path) -> tuple[dict[str, np.ndarray], int]:
    """Per-case periods, per-edge intervals and per-run R0 of the first-outbreak runs."""
    from covsyn.model.r0_network import R0_average_effective_contact

    out: dict[str, list] = {k: [] for k in ('latent', 'incubation', 'infectious', 'asymptomatic',
                                            'symptomatic', 'presymptomatic', 'postsymptomatic',
                                            'generation', 'serial', 'R0')}
    runs = 0
    for courses, contacts, digraph in iter_runs(directory, with_digraph=True):
        runs += 1
        for c in courses:
            out['latent'].append(c['latent_period'])
            out['infectious'].append(c['infectious_period'])
            if np.isnan(c['incubation_period']):
                out['asymptomatic'].append(c['infectious_period'])
            else:
                window = c['incubation_period'] - c['latent_period']
                out['incubation'].append(c['incubation_period'])
                out['symptomatic'].append(c['infectious_period'])
                out['presymptomatic'].append(window)
                out['postsymptomatic'].append(c['infectious_period'] - window)
        generation, serial = transmission_intervals(courses, digraph)
        out['generation'] += generation
        out['serial'] += serial
        out['R0'].append(R0_average_effective_contact(contacts))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}, runs


def fig3(directory: Path, out_dir: Path) -> None:
    """Fig 3 (10 panels, as the notebook's last version) and Table 2."""
    _style()
    data, runs = epidemiological_statistics(directory)
    lit = literature_ranges()
    panels = [('latent', 'Latent period (days)'), ('incubation', 'Incubation period (days)'),
              ('infectious', 'Infectious period (days)'),
              ('asymptomatic', 'Asymptomatic cases infectious period (days)'),
              ('symptomatic', 'Symptomatic cases infectious period (days)'),
              ('presymptomatic', 'Pre-symptomatic cases infectious period (days)'),
              ('postsymptomatic', 'Post-symptomatic cases infectious period (days)'),
              ('generation', 'Generation time (days)'), ('serial', 'Serial interval (days)'),
              ('R0', '$R_0$')]
    fig = plt.figure(figsize=(14, 16))
    for i, (key, label) in enumerate(panels):
        ax = fig.add_subplot(4, 3, i + 1)
        x = data[key][~np.isnan(data[key])]
        if key == 'R0':
            ax.hist(x, bins=max(len(np.unique(x)), 1), color=PALETTE[0])
            ax.set_xscale('log')
        else:
            lo = int(np.floor(x.min())) if key == 'serial' else 0
            ax.hist(x, bins=np.arange(lo - 0.5, int(np.nanmax(x)) + 1.5, 1), rwidth=0.8,
                    color=PALETTE[0], label='Simulation')
            ax.set_xlim(lo if key == 'serial' else (1 if key == 'latent' else 0), 50)
        ax.axvline(np.mean(x), color='k', linestyle='dashed', linewidth=2, label='Simulation mean')
        mean_lo, mean_hi, ci_lo, ci_hi = lit[key]
        ax.axvspan(mean_lo, mean_hi, alpha=0.3, color=PALETTE[3], label='Reported mean')
        ax.axvspan(ci_lo, ci_hi, alpha=0.3, color=PALETTE[1], label='Reported 95% CI')
        ax.set_xlabel(label)
        if i % 3 == 0:
            ax.set_ylabel('Frequency')
        if i == 0:
            ax.legend()
    fig.suptitle(f'CovSyn after the corrections: {runs} first-outbreak runs, '
                 f'{len(data["latent"]):,} cases', y=1.0)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(out_dir / 'wu2025_fig3_epidemiological_statistics.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)

    rows = []
    for key, _ in panels:
        x = data[key][~np.isnan(data[key])]
        mean_lo, mean_hi, ci_lo, ci_hi = lit[key]
        rows.append({'quantity': key, 'n': len(x), 'covsyn_mean': round(float(np.mean(x)), 2),
                     'covsyn_median': round(float(np.median(x)), 2),
                     'covsyn_p2.5': round(float(np.percentile(x, 2.5)), 2),
                     'covsyn_p97.5': round(float(np.percentile(x, 97.5)), 2),
                     'published_covsyn_mean': PUBLISHED.get(key, ''),
                     'literature_mean_low': round(float(mean_lo), 2),
                     'literature_mean_high': round(float(mean_hi), 2),
                     'literature_ci_low': round(float(ci_lo), 2),
                     'literature_ci_high': round(float(ci_hi), 2),
                     'mean_in_reported_mean_range': bool(mean_lo <= np.mean(x) <= mean_hi),
                     'mean_in_reported_ci_range': bool(ci_lo <= np.mean(x) <= ci_hi)})
    _write_csv(out_dir / 'wu2025_table2_epidemiological_parameters.csv', rows)
    for r in rows:
        print(f"{r['quantity']:16s} CovSyn {r['covsyn_mean']:6.2f} (published "
              f"{r['published_covsyn_mean']!s:>4s}) reported means "
              f"[{r['literature_mean_low']}, {r['literature_mean_high']}] "
              f"CI [{r['literature_ci_low']}, {r['literature_ci_high']}]")


# ------------------------------------------------------------------------------ Fig 4
def _load_rows(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path))


def _seriation(linkage: np.ndarray, n: int, index: int) -> list[int]:
    if index < n:
        return [index]
    left, right = int(linkage[index - n, 0]), int(linkage[index - n, 1])
    return _seriation(linkage, n, left) + _seriation(linkage, n, right)


def serial_distance_matrix(dist: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Distance matrix reordered by Ward clustering, as plot_firefly.ipynb draws it."""
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    n = len(dist)
    linkage = hierarchy.linkage(squareform(dist, checks=False), method='ward')
    order = _seriation(linkage, n, 2 * n - 2)
    return dist[np.ix_(order, order)], order


COST_PARTS = [('cost_contact_household', 'Household contact cost'),
              ('cost_contact_healthcare', 'Health care contact cost'),
              ('cost_contact_others', 'Others contact cost'),
              ('cost_attack_rate', 'Attack rate cost'), ('cost_energy', 'Energy cost'),
              ('cost_outcome', 'Outcome penalty'), ('cost_penalty', 'Physiology penalty'),
              ('total', 'Total cost')]


def evaluate_cost_parts(vectors: dict[str, np.ndarray], cache: Path,
                        workers: int = 32) -> list[dict]:
    """Objective and its parts for every firefly at every stage, cached in a CSV."""
    if cache.exists():
        with open(cache) as f:
            return [{k: (v if k == 'stage' else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(f)]
    import concurrent.futures
    import pickle

    from covsyn.calibration import fast_cost
    from covsyn.calibration.cost_parts import LAST

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=fast_cost.init_worker, initargs=(demo, columns))
    rows = []
    for stage, matrix in vectors.items():
        for firefly, vector in enumerate(matrix):
            total = float(fast_cost.cost_function(vector, demo, pool, *cheng))
            rows.append({'stage': stage, 'firefly': firefly, 'total': total,
                         **{k: float(LAST.get(k, np.nan)) for k, _ in COST_PARTS[:-1]}})
            print(f'{stage} {firefly:3d} {total:.4f}', flush=True)
    pool.shutdown()
    _write_csv(cache, rows)
    return rows


def fig4(run_dir: Path, out_dir: Path, evaluate: bool = True) -> None:
    """Fig 4: cost against distance from the best firefly, the clustered distance matrix of
    the final fireflies, and each cost part at the initial, worst, best and final positions."""
    from scipy.spatial import distance_matrix

    _style()
    bounds = _load_rows(run_dir / 'bound.txt')
    lower, upper = bounds[0], bounds[1]
    span = np.where(upper > lower, upper - lower, np.nan)

    def normalise(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - lower) / span)

    initial = _load_rows(run_dir / 'firefly_result_first_initial_guess.txt')
    best = _load_rows(run_dir / 'firefly_best.txt')
    worst = _load_rows(run_dir / 'firefly_worst.txt')
    final = _load_rows(run_dir / 'firefly_result.txt')
    stages = {'Initial': (initial[:, :-1], initial[:, -1]), 'Worst': (worst[:, 1:-1], worst[:, -1]),
              'Best': (best[:, 1:-1], best[:, -1]), 'Final': (final[:, :-1], final[:, -1])}
    best_index = int(np.argmin(best[:, -1]))
    reference = normalise(best[best_index, 1:-1])
    distance = {k: np.linalg.norm(normalise(v) - reference, axis=1) for k, (v, _) in stages.items()}
    cost = {k: c for k, (_, c) in stages.items()}
    worst_first = worst[:, 0] < best[:, 0]

    def trajectories(ax: Any) -> None:
        ax.plot(distance['Initial'], cost['Initial'], '.', color=PALETTE[2], markersize=10,
                label='Initial guess')
        ax.plot(distance['Worst'], cost['Worst'], 'kX', markersize=8, label='Worst fireflies')
        ax.plot(distance['Best'], cost['Best'], 'k^', markersize=8, label='Best fireflies')
        ax.plot(distance['Final'], cost['Final'], '.', color=PALETTE[0], markersize=10,
                label='Final fireflies')
        for i in range(len(best)):
            order = ['Initial', 'Worst', 'Best', 'Final'] if worst_first[i] else \
                ['Initial', 'Best', 'Worst', 'Final']
            width, alpha = (1.0, 1.0) if i == best_index else (0.3, 0.2)
            for a, b in zip(order[:-1], order[1:]):
                ax.plot([distance[a][i], distance[b][i]], [cost[a][i], cost[b][i]], 'k',
                        alpha=alpha, linewidth=width)

    rows = evaluate_cost_parts({k: v for k, (v, _) in stages.items()},
                               out_dir / 'wu2025_fig4_cost_parts.csv') if evaluate else []
    n_part_rows = int(np.ceil(len(COST_PARTS) / 2)) if rows else 0
    fig = plt.figure(figsize=(14, 6 + 4 * n_part_rows))
    grid = fig.add_gridspec(1 + n_part_rows, 2, height_ratios=[1.5] + [1] * n_part_rows)

    ax = fig.add_subplot(grid[0, 0])
    trajectories(ax)
    ax.set_xlabel('Distance from the best firefly')
    ax.set_ylabel('Cost')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=8, numpoints=1)
    zoom_x = (distance['Final'].min(), distance['Final'].max())
    zoom_y = (min(cost['Final'].min(), cost['Best'].min()), cost['Final'].max())
    pad_x, pad_y = 0.05 * (zoom_x[1] - zoom_x[0]) + 1e-3, 0.05 * (zoom_y[1] - zoom_y[0]) + 1e-3
    inset = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
    trajectories(inset)
    inset.set_xlim(zoom_x[0] - pad_x, zoom_x[1] + pad_x)
    inset.set_ylim(zoom_y[0] - pad_y, zoom_y[1] + pad_y)
    for spine in inset.spines.values():
        spine.set_color('red')
    inset.tick_params(colors='red', labelsize=7)
    ax.indicate_inset_zoom(inset, edgecolor='red')

    ax = fig.add_subplot(grid[0, 1])
    sorted_dist, order = serial_distance_matrix(
        distance_matrix(normalise(stages['Final'][0]), normalise(stages['Final'][0])))
    seaborn.heatmap(sorted_dist, ax=ax, xticklabels=False, yticklabels=False,
                    cbar_kws={'label': 'Euclidean distance'})
    ax.set_xlabel('Firefly (clustered order)')
    ax.set_ylabel('Firefly (clustered order)')

    summary = []
    if rows:
        names = ['Initial', 'Worst', 'Best', 'Final']
        for p, (key, label) in enumerate(COST_PARTS):
            ax = fig.add_subplot(grid[1 + p // 2, p % 2])
            values = np.array([[r[key] for r in rows if r['stage'] == s] for s in names])
            for i in range(values.shape[1]):
                ax.plot(range(4), values[:, i], '*--', linewidth=0.5, markersize=3, alpha=0.6)
            ax.plot(range(4), values.mean(axis=1), 'k^-', markersize=8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(names)
            ax.set_ylabel(label)
            if np.all(values > 0):
                ax.set_yscale('log')
        for s in names:
            stored = cost[s]
            again = np.array([r['total'] for r in rows if r['stage'] == s])
            summary.append({'stage': s, 'fireflies': len(stored),
                            'max_abs_difference_from_stored_cost':
                                float(np.max(np.abs(again - stored)))})
        _write_csv(out_dir / 'wu2025_fig4_reproducibility.csv', summary)
        for r in summary:
            print(f"{r['stage']:8s} re-evaluated cost vs stored: max |difference| "
                  f"{r['max_abs_difference_from_stored_cost']:.3g}")
    plt.tight_layout()
    fig.savefig(out_dir / 'wu2025_fig4_firefly_convergence.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'best firefly {best_index}, cost {best[best_index, -1]:.4f}; final cost range '
          f'{cost["Final"].min():.4f}-{cost["Final"].max():.4f}')


# ------------------------------------------------------------------------------ Fig 5
TRANSITIONS = [  # (key, source, target, x limit) in the notebook's panel order
    ('IA_to_IS', '$\\mathrm{I}^\\mathrm{A}$', '$\\mathrm{I}^\\mathrm{S}$', 51),
    ('IA_to_C', '$\\mathrm{I}^\\mathrm{A}$', '$\\mathrm{C}$', 51),
    ('IA_to_R', '$\\mathrm{I}^\\mathrm{A}$', '$\\mathrm{R}$', 121),
    ('IS_to_IC', '$\\mathrm{I}^\\mathrm{S}$', '$\\mathrm{I}^\\mathrm{C}$', 51),
    ('IS_to_C', '$\\mathrm{I}^\\mathrm{S}$', '$\\mathrm{C}$', 51),
    ('IS_to_R', '$\\mathrm{I}^\\mathrm{S}$', '$\\mathrm{R}$', 121),
    ('IC_to_R', '$\\mathrm{I}^\\mathrm{C}$', '$\\mathrm{R}$', 201),
    ('IC_to_D', '$\\mathrm{I}^\\mathrm{C}$', '$\\mathrm{D}$', 71)]


def simulated_transition_days(directory: Path) -> dict[str, np.ndarray]:
    """Days spent in each state before each transition, over every simulated case.

    Dates of ICU, recovery, death and the positive test are absolute days; the latent and
    incubation periods are relative to the case's infection day.
    """
    days: dict[str, list] = {k: [] for k, *_ in TRANSITIONS}
    for courses, _ in iter_runs(directory):
        for c in courses:
            infection = c['infection_day']
            positive = float(np.ravel(c['positive_test_date'])[0])
            icu, recovery, death = c['date_of_critically_ill'], c['date_of_recovery'], \
                c['date_of_death']
            if np.isnan(c['incubation_period']):
                days['IA_to_C'].append(positive - infection)
                if not np.isnan(recovery):
                    days['IA_to_R'].append(recovery - infection)
                continue
            onset = infection + c['incubation_period']
            days['IA_to_IS'].append(c['incubation_period'])
            days['IS_to_C'].append(positive - onset)
            if np.isnan(icu):
                if not np.isnan(recovery):
                    days['IS_to_R'].append(recovery - onset)
            else:
                days['IS_to_IC'].append(icu - onset)
                if not np.isnan(recovery):
                    days['IC_to_R'].append(recovery - icu)
                if not np.isnan(death):
                    days['IC_to_D'].append(death - icu)
    return {k: np.floor(np.asarray(v, dtype=float)) for k, v in days.items()}


def taiwan_transition_days() -> dict[str, Any]:
    """The Taiwan dataset's transition days, selected exactly as Course_synthesis.ipynb does."""
    import pandas as pd

    from covsyn.data_processing.rw_data_processing import clean_taiwan_data, extract_state_data

    sheet = pd.read_excel(TAIWAN_XLSX, sheet_name=0)
    sheet.columns = sheet.columns.str.strip().str.lower().str.replace(' ', '_') \
        .str.replace('(', '').str.replace(')', '')
    data = clean_taiwan_data(sheet, 1, 579)
    out = {
        'IA_to_IS': extract_state_data(data, 'earliest_infection_date', 'onset_of_symptom'),
        'IA_to_R': extract_state_data(data, 'earliest_infection_date', 'recovery',
                                      'onset_of_symptom'),
        'IS_to_IC': extract_state_data(data, 'onset_of_symptom', 'icu'),
        'IS_to_R': extract_state_data(data, 'onset_of_symptom', 'recovery', 'icu'),
        'IC_to_R': extract_state_data(data, 'icu', 'recovery'),
        'IC_to_D': extract_state_data(data, 'icu', 'death_date'),
    }
    out['IA_to_C'] = extract_state_data(data, 'earliest_infection_date', 'confirmed_date',
                                        'onset_of_symptom').drop(out['IA_to_R'].index)
    out['IS_to_C'] = extract_state_data(data, 'onset_of_symptom', 'confirmed_date') \
        .drop(out['IS_to_IC'].index).drop(out['IS_to_R'].index)
    return out


def still_in_state(days: np.ndarray, length: int) -> np.ndarray:
    """Share of cases still in the state on day k = 0..length-1 (duration >= k)."""
    return (days[None, :] >= np.arange(length)[:, None]).mean(axis=1)


def fig5(directory: Path, out_dir: Path, bootstrap: int = 1000, seed: int = 0) -> None:
    """Fig 5: Kaplan-Meier curves of the Taiwan data against bootstrapped simulated curves,
    each bootstrap sample as large as the Taiwan sample."""
    from covsyn.data_processing.rw_data_processing import state_transition_plot

    _style()
    rng = np.random.default_rng(seed)
    simulated = simulated_transition_days(directory)
    taiwan = taiwan_transition_days()
    fig = plt.figure(figsize=(18, 18))
    rows = []
    for p, (key, source, target, xlim) in enumerate(TRANSITIONS):
        ax = fig.add_subplot(3, 3, p + 1)
        plt.sca(ax)
        tw = taiwan[key]
        tw_days = tw.dt.days.to_numpy().astype(float)
        sim = simulated[key]
        state_transition_plot(tw, source, target, xlim, save_fig=False)
        length = int(max(xlim, np.nanmax(sim) + 2 if len(sim) else xlim))
        if len(sim):
            samples = rng.choice(sim, size=(bootstrap, len(tw_days)), replace=True)
            curves = np.array([still_in_state(s, length) for s in samples])
            mean = curves.mean(axis=0)
            lb, ub = np.percentile(curves, [2.5, 97.5], axis=0)
            edges = np.arange(length + 1) - 1
            ax.stairs(mean, edges=edges, linewidth=1.5, color='g',
                      label=f'Simulation ({len(sim):,} cases)')
            ax.fill_between(edges[:-1], lb, ub, step='post', color='g', alpha=0.2)
            tw_curve = still_in_state(tw_days, length)
            rows.append({'transition': key, 'taiwan_n': len(tw_days), 'simulated_n': len(sim),
                         'taiwan_mean_days': round(float(tw_days.mean()), 2),
                         'simulated_mean_days': round(float(sim.mean()), 2),
                         'area_between_curves_signed': round(float(np.sum(mean - tw_curve)), 2),
                         'area_between_curves_absolute':
                             round(float(np.sum(np.abs(mean - tw_curve))), 2),
                         'share_of_taiwan_curve_inside_95ci': round(float(np.mean(
                             (tw_curve[:int(np.max(tw_days)) + 1] >= lb[:int(np.max(tw_days)) + 1])
                             & (tw_curve[:int(np.max(tw_days)) + 1]
                                <= ub[:int(np.max(tw_days)) + 1]))), 2)})
        ax.set_xlim(-1, xlim)
        ax.set_xlabel('Day' if p >= 6 else '')
        ax.set_ylabel('Proportion of cases' if p % 3 == 0 else '')
        ax.legend(fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / 'wu2025_fig5_state_transitions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    _write_csv(out_dir / 'wu2025_fig5_state_transitions.csv', rows)
    for r in rows:
        print(f"{r['transition']:9s} Taiwan n={r['taiwan_n']:3d} mean {r['taiwan_mean_days']:6.2f} | "
              f"CovSyn mean {r['simulated_mean_days']:6.2f} | area {r['area_between_curves_signed']:+7.2f}"
              f" | Taiwan curve inside 95% band {r['share_of_taiwan_curve_inside_95ci']:.0%}")


# ------------------------------------------------------------------------------ Fig 6
def cheng_bins(directory: Path, layers: list[str]) -> tuple[dict[str, np.ndarray], int]:
    """Contacts, infections and attack rate per Cheng bin, one row per Monte-Carlo run.

    Contacts and infections are scaled to Cheng's 91 symptomatic index cases (E86), since
    only symptomatic cases have an onset to bin contacts against.
    """
    from covsyn.calibration.firefly_optimizer import CHENG_SYMPTOMATIC_INDEX_CASES
    from covsyn.figures.plot_results import create_array_cheng2020_fig2

    out: dict[str, list] = {f'{layer}|{q}': [] for layer in layers
                            for q in ('contacts', 'infections', 'attack')}
    runs = 0
    for courses, contacts in iter_runs(directory):
        symptomatic = sum(not np.isnan(c['incubation_period']) for c in courses)
        if symptomatic == 0:
            continue
        runs += 1
        scale = CHENG_SYMPTOMATIC_INDEX_CASES / symptomatic
        for layer in layers:
            _, contact, _, infection = create_array_cheng2020_fig2(courses, contacts, layer=layer)
            contact = np.zeros(6) if np.size(contact) == 0 else np.asarray(contact, float)
            infection = np.zeros(6) if np.size(infection) == 0 else np.asarray(infection, float)
            attack = np.divide(infection, contact, out=np.full(6, np.nan), where=contact > 0) * 100
            out[f'{layer}|contacts'].append(contact * scale)
            out[f'{layer}|infections'].append(infection * scale)
            out[f'{layer}|attack'].append(attack)
    return {k: np.array(v) for k, v in out.items()}, runs


def fig6(directory: Path, out_dir: Path) -> None:
    """Fig 6: CovSyn against Cheng et al. 2020 for household, health care and all contacts."""
    from covsyn.figures.plot_results import plot_cheng2020_bar_chart

    _style()
    layers = ['Household', 'Health care', 'All']
    data, runs = cheng_bins(directory, layers)
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11))
    x = np.arange(6)
    width = 0.15
    rows = []
    for row, layer in enumerate(layers):
        _, _, (obs_contacts, obs_infected, obs_attack, obs_lb, obs_ub) = \
            plot_cheng2020_bar_chart(layer=layer, save_fig=False)
        plt.close()
        contacts, infections, attack = (data[f'{layer}|{q}'] for q in
                                        ('contacts', 'infections', 'attack'))
        mean_c, mean_i, mean_a = contacts.mean(0), infections.mean(0), np.nanmean(attack, 0)
        lb_c, ub_c = np.percentile(contacts, [2.5, 97.5], axis=0)
        lb_i, ub_i = np.percentile(infections, [2.5, 97.5], axis=0)
        lb_a, ub_a = np.nanpercentile(attack, [2.5, 97.5], axis=0)

        ax1 = axes[row, 0]
        ax2 = ax1.twinx()
        ax1.bar(x - 2 * width, obs_infected, width=width, color='#b2df8a',
                label='Observed secondary infection')
        ax1.bar(x - width, mean_i, width=width, color='#33a02c', label='CovSyn secondary infection')
        ax1.errorbar(x - width, mean_i, yerr=[mean_i - lb_i, np.maximum(ub_i - mean_i, 0)],
                     fmt='.k', capsize=1, linewidth=1, markersize=3)
        ax2.bar(x + width, obs_contacts, width=width, color='#a6cee3', label='Observed close contacts')
        ax2.bar(x + 2 * width, mean_c, width=width, color='#1f78b4', label='CovSyn close contacts')
        ax2.errorbar(x + 2 * width, mean_c, yerr=[mean_c - lb_c, ub_c - mean_c], fmt='.k',
                     capsize=1, linewidth=1, markersize=3)
        ax1.set_xlabel('Days from onset to first exposure')
        ax1.set_ylabel(f'{layer} secondary infected cases')
        ax2.set_ylabel(f'{layer} close contacts')
        ax1.set_xticks(x)
        ax1.set_xticklabels(BINS)
        for axis, colour, side in ((ax1, '#33a02c', 'left'), (ax2, '#1f78b4', 'right')):
            axis.spines[side].set_color(colour)
            axis.tick_params(axis='y', colors=colour)
            axis.yaxis.label.set_color(colour)
        ax1.set_ylim(0, 1.15 * max(np.max(obs_infected), np.max(ub_i), 1))
        ax2.set_ylim(0, 1.15 * max(np.max(obs_contacts), np.max(ub_c)))
        if row == 0:
            lines = ax1.get_legend_handles_labels()
            more = ax2.get_legend_handles_labels()
            ax1.legend(lines[0] + more[0], lines[1] + more[1], loc='upper right', fontsize=8)

        ax3 = axes[row, 1]
        ax3.plot(x - 0.15, obs_attack, 'o--', color='#ff7f00', label='Observed data')
        ax3.plot(x + 0.15, mean_a, 's--', color='#e41a1c', label='CovSyn')
        ax3.errorbar(x - 0.15, obs_attack, yerr=[obs_attack - obs_lb, obs_ub - obs_attack],
                     fmt='none', ecolor='#ff7f00', capsize=1, alpha=0.8)
        ax3.errorbar(x + 0.15, mean_a, yerr=[np.maximum(mean_a - lb_a, 0),
                                             np.maximum(ub_a - mean_a, 0)],
                     fmt='none', ecolor='#e41a1c', capsize=1, alpha=0.8)
        ax3.set_xlabel('Days from onset to first exposure')
        ax3.set_ylabel(f'{layer} secondary attack rate (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(BINS)
        ax3.set_ylim(0, 1.15 * max(np.nanmax(obs_attack), np.nanmax(mean_a)))
        if row == 0:
            ax3.legend(loc='upper right')
        for b in range(6):
            rows.append({'layer': layer, 'bin': BINS[b],
                         'cheng_contacts': int(obs_contacts[b]),
                         'covsyn_contacts': round(float(mean_c[b]), 1),
                         'covsyn_contacts_95': f'[{lb_c[b]:.1f}, {ub_c[b]:.1f}]',
                         'cheng_infected': int(obs_infected[b]),
                         'covsyn_infected': round(float(mean_i[b]), 2),
                         'covsyn_infected_95': f'[{lb_i[b]:.2f}, {ub_i[b]:.2f}]',
                         'cheng_attack_pct': float(obs_attack[b]),
                         'cheng_attack_95': f'[{obs_lb[b]}, {obs_ub[b]}]',
                         'covsyn_attack_pct': round(float(mean_a[b]), 2),
                         'covsyn_attack_95': f'[{lb_a[b]:.2f}, {ub_a[b]:.2f}]',
                         'covsyn_mean_in_cheng_95': bool(obs_lb[b] <= mean_a[b] <= obs_ub[b])})
    fig.suptitle(f'CovSyn ({runs} Monte-Carlo runs of 100 index cases, scaled to 91 '
                 'symptomatic) against Cheng et al. 2020', y=1.0)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(out_dir / 'wu2025_fig6_cheng2020.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    _write_csv(out_dir / 'wu2025_fig6_cheng2020.csv', rows)
    for layer in layers:
        r = [x for x in rows if x['layer'] == layer]
        print(f"{layer:11s} contacts Cheng {sum(x['cheng_contacts'] for x in r):5d} CovSyn "
              f"{sum(x['covsyn_contacts'] for x in r):7.1f} | infected Cheng "
              f"{sum(x['cheng_infected'] for x in r):3d} CovSyn {sum(x['covsyn_infected'] for x in r):6.2f}"
              f" | attack-rate bins inside Cheng's 95% CI: "
              f"{sum(x['covsyn_mean_in_cheng_95'] for x in r)}/6")


# ------------------------------------------------------------------------------ Fig 7
def taiwan_daily_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily local confirmed cases and deaths from 2020-01-27, 365 days, as the notebook builds
    them from the Summary sheet (only the three local deaths are kept)."""
    import pandas as pd

    summary = pd.read_excel(TAIWAN_XLSX, sheet_name='Summary')
    summary = summary.loc[(summary['announce_date'] >= pd.to_datetime('2020-01-27'))
                          & (pd.to_datetime('2020-08-01') >= summary['announce_date'])]
    local = summary['number_of_local_positive_cases'].to_numpy()
    keep = ~np.isnan(local)
    local = np.flip(local[keep].astype(int))
    unknown = np.flip(summary['number_of_unknown_positive_cases'].to_numpy()[keep].astype(int))
    dates = np.flip(summary['announce_date'].to_numpy()[keep])
    deaths = np.flip(summary['dead'].to_numpy()[keep].astype(int))
    deaths[np.where(deaths == 5)[0][0]:] = 3

    full_dates = np.arange(dates[0], dates[-1] + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    day_dates = dates.astype('datetime64[D]')

    def fill(values: np.ndarray) -> np.ndarray:
        out, last = [], None
        for d in full_dates:
            hit = np.where(day_dates == d)[0]
            last = values[hit[-1]] if len(hit) else last
            out.append(last)
        return np.array(out, dtype=float)

    daily_cases = np.diff(fill(local)) + np.diff(fill(unknown))
    daily_deaths = np.diff(fill(deaths))
    daily_cases = np.pad(daily_cases, (0, 365 - len(daily_cases)))
    daily_deaths = np.pad(daily_deaths, (0, 365 - len(daily_deaths)))
    return np.arange(full_dates[0], full_dates[0] + np.timedelta64(365, 'D')), daily_cases, \
        daily_deaths


def best_time_shift(observed_cumulative: np.ndarray, simulated_cumulative: np.ndarray,
                    max_shift: int = 57) -> int:
    """The delay, in days, that best aligns the observed and simulated cumulative cases.

    Same loss as the notebook (observed padded with leading zeros, simulation padded with its
    last value), searched over every shift instead of by bisection.
    """
    losses = []
    for s in range(max_shift + 1):
        obs = np.pad(observed_cumulative, (s, 0))
        sim = np.pad(simulated_cumulative, (0, s), constant_values=simulated_cumulative[-1])
        losses.append(np.mean((obs - sim) ** 2))
    return int(np.argmin(losses))


def goodness_of_fit(actual: np.ndarray, predicted: np.ndarray, window: int = 1) -> dict[str, float]:
    """MAE, MSE, RMSE and R2 of the daily and cumulative series after a centred moving mean."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if window > 1:
        kernel = np.ones(window) / window
        actual = np.convolve(actual, kernel, mode='same')
        predicted = np.convolve(predicted, kernel, mode='same')
    out = {}
    for suffix, a, p in (('', actual, predicted),
                         (' (Cumulative)', np.cumsum(actual), np.cumsum(predicted))):
        mse = mean_squared_error(a, p)
        out.update({f'MAE{suffix}': mean_absolute_error(a, p), f'MSE{suffix}': mse,
                    f'RMSE{suffix}': float(np.sqrt(mse)), f'R2{suffix}': r2_score(a, p)})
    return out


def fig7(directory: Path, out_dir: Path) -> None:
    """Fig 7 and Table 3: the Taiwan first outbreak against the seeded CovSyn runs."""
    import matplotlib.dates as mdates
    from datetime import timedelta
    from matplotlib.dates import DateFormatter

    from covsyn.data_processing.rw_data_processing import \
        transform_course_object_to_population_data
    from covsyn.upstream import scoring

    _style()
    time_limit = 365
    dates, tw_cases, tw_deaths = taiwan_daily_series()
    cases, deaths = [], []
    for courses, contacts in iter_runs(directory):
        if len(courses) < len(SOURCE_CONFIRMED_DATES) + 1:
            continue
        series = transform_course_object_to_population_data(
            courses, contacts, time_limit=time_limit - 1, population_size=TAIWAN_POPULATION)
        cases.append(series[4])
        deaths.append(series[10])
    cases, deaths = np.array(cases, dtype=float), np.array(deaths, dtype=float)

    seeds = np.zeros(time_limit)
    first = min(datetime.strptime(d, '%Y/%m/%d') for d in SOURCE_CONFIRMED_DATES)
    for d in SOURCE_CONFIRMED_DATES:
        seeds[(datetime.strptime(d, '%Y/%m/%d') - first).days] += 1

    shift = best_time_shift(np.cumsum(tw_cases), np.cumsum(cases, axis=1).mean(axis=0))
    # One shift for everything: the observed series start `shift` days later, the simulated
    # ones (cases, deaths and the seeds alike) are extended at the end.
    sim_cases = np.pad(cases, ((0, 0), (0, shift)))
    sim_deaths = np.pad(deaths, ((0, 0), (0, shift)))
    obs_cases = np.pad(tw_cases, (shift, 0))
    obs_deaths = np.pad(tw_deaths, (shift, 0))
    seeds = np.pad(seeds, (0, shift))
    axis_dates = np.arange(dates[0] - np.timedelta64(shift, 'D'), dates[-1] + np.timedelta64(1, 'D'))
    window = slice(0, 28 * 6)
    axis_dates = axis_dates[window]

    rows = []
    for label, size in (('daily', 1), ('weekly', 7), ('monthly', 31)):
        for name, actual, matrix in (('Confirmed', obs_cases, sim_cases),
                                     ('Deaths', obs_deaths, sim_deaths)):
            fits = [goodness_of_fit(actual[window], m[window], size) for m in matrix]
            row: dict[str, Any] = {'': f'{label} {name}'}
            for metric in fits[0]:
                values = [f[metric] for f in fits]
                row[metric] = round(float(np.mean(values)), 2)
                row[f'{metric} lb'] = round(float(np.percentile(values, 2.5)), 2)
                row[f'{metric} ub'] = round(float(np.percentile(values, 97.5)), 2)
            rows.append(row)
    _write_csv(out_dir / 'wu2025_table3_goodness_of_fit.csv', rows)

    cum_cases, cum_deaths = np.cumsum(sim_cases, 1)[:, window], np.cumsum(sim_deaths, 1)[:, window]
    obs_cum_cases, obs_cum_deaths = np.cumsum(obs_cases)[window], np.cumsum(obs_deaths)[window]
    cum_seeds = np.cumsum(seeds)[window]
    wis = {}
    for key, obs, sim in (('cases', obs_cum_cases, cum_cases), ('deaths', obs_cum_deaths, cum_deaths)):
        q = {0.025: np.quantile(sim, 0.025, axis=0), 0.975: np.quantile(sim, 0.975, axis=0)}
        wis[key] = scoring.weighted_interval_score(obs, alphas=[0.05], q_dict=q)[0]

    plot_dates = axis_dates.astype('datetime64[D]').astype(datetime)
    start = plot_dates[0]
    majors = [start + timedelta(days=28 * k) for k in range(7)]
    minors = [m + timedelta(days=7 * j) for m in majors[:-1] for j in range(1, 4)]

    def draw(ax: Any) -> None:
        ax.plot(plot_dates, obs_cum_cases, ':', color=PALETTE[0])
        ax.plot(plot_dates, obs_cum_deaths, ':', color=PALETTE[2])
        up_c = np.concatenate([[True], np.diff(obs_cum_cases) > 0])
        up_d = np.concatenate([[True], np.diff(obs_cum_deaths) > 0])
        ax.plot(plot_dates[up_c], obs_cum_cases[up_c], '*', color=PALETTE[0], markersize=8,
                label='Observed local confirmed cases')
        ax.plot(plot_dates[up_d], obs_cum_deaths[up_d], 'X', color=PALETTE[2], markersize=8,
                label='Observed local deaths')
        ax.plot(plot_dates, cum_cases.mean(0), color=PALETTE[0], label='CovSyn local confirmed cases')
        ax.fill_between(plot_dates, *np.percentile(cum_cases, [2.5, 97.5], axis=0),
                        color=PALETTE[0], alpha=0.3, label='95% CI for CovSyn cases')
        ax.plot(plot_dates, cum_deaths.mean(0), color=PALETTE[2], label='CovSyn local deaths')
        ax.fill_between(plot_dates, *np.percentile(cum_deaths, [2.5, 97.5], axis=0),
                        color=PALETTE[2], alpha=0.3, label='95% CI for CovSyn deaths')
        ax.plot(plot_dates, cum_seeds, '--', color=PALETTE[3], label='Source infected cases')
        ax.set_xticks([mdates.date2num(d) for d in majors])
        ax.set_xticks([mdates.date2num(d) for d in minors], minor=True)
        ax.grid(True, which='major', linestyle='dotted', color='gray', alpha=0.7)
        ax.set_xlim(start - timedelta(days=2), majors[-1] + timedelta(days=2))

    for name, with_wis in (('wu2025_fig7_first_outbreak.png', False),
                           ('wu2025_fig7_first_outbreak_wis.png', True)):
        if with_wis:
            fig, (ax, ax_w) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})
        else:
            fig, ax = plt.subplots(figsize=(9.5, 6))
        draw(ax)
        ax.legend(numpoints=1, loc='best', fontsize=8)
        ax.set_ylabel('Cumulative number of cases')
        bottom = ax
        if with_wis:
            ax.set_xticklabels([])
            ax_w.plot(plot_dates, wis['cases'], color=PALETTE[0], label='Confirmed cases')
            ax_w.plot(plot_dates, wis['deaths'], color=PALETTE[2], label='Deaths')
            ax_w.set_xticks([mdates.date2num(d) for d in majors])
            ax_w.set_xticks([mdates.date2num(d) for d in minors], minor=True)
            ax_w.grid(True, which='major', linestyle='dotted', color='gray', alpha=0.7)
            ax_w.set_xlim(start - timedelta(days=2), majors[-1] + timedelta(days=2))
            ax_w.set_ylabel('Weighted interval score')
            ax_w.legend()
            bottom = ax_w
        bottom.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        bottom.tick_params(axis='x', rotation=30)
        plt.setp(bottom.get_xticklabels(), ha='right')
        plt.tight_layout()
        fig.savefig(out_dir / name, dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f'{len(cases)} runs with at least one transmission; time shift {shift} days '
          f'(axis starts {axis_dates[0]})')
    print(f'cumulative local cases over the window: observed {obs_cum_cases[-1]:.0f}, CovSyn mean '
          f'{cum_cases[:, -1].mean():.1f} (95% {np.percentile(cum_cases[:, -1], 2.5):.0f}-'
          f'{np.percentile(cum_cases[:, -1], 97.5):.0f})')
    print(f'cumulative local deaths: observed {obs_cum_deaths[-1]:.0f}, CovSyn mean '
          f'{cum_deaths[:, -1].mean():.2f}')
    for r in rows:
        print(f"{r['']:18s} RMSE {r['RMSE']:7.2f}  R2 {r['R2']:6.2f}  RMSE(cum) "
              f"{r['RMSE (Cumulative)']:7.2f}  R2(cum) {r['R2 (Cumulative)']:6.2f}")


def main() -> None:
    figure, source, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if figure == 'fig3':
        fig3(source, out)
    elif figure == 'fig4':
        fig4(source, out, evaluate='--no-evaluate' not in sys.argv)
    elif figure == 'fig5':
        fig5(source, out)
    elif figure == 'fig6':
        fig6(source, out)
    elif figure == 'fig7':
        fig7(source, out)
    else:
        raise SystemExit(f'unknown figure {figure}; use fig3, fig4, fig5, fig6 or fig7')


if __name__ == '__main__':
    main()
