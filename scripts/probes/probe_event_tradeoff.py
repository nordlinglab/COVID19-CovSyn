# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""What does the mass-event probability P[199] trade against? (B54, run 11)

Run 11 searched P[199] in [0, 0.20] and settled at 0.012, which left the community tail ratio
(p90 / median of non-zero community contacts) at 3.6 against the tracing data's [5.5, 93.1].
This probe holds every other parameter of the run 11 best vector and varies P[199] only.

The objective scores 100 fixed seeds, and one extra random draw per case moves it by several
units (run 10: 2.03 -> 7.35), so a single evaluation per point would mostly show seed noise.
Each point is therefore measured on many index cases, and the Cheng 'others' contact cost is
computed from the MEAN bin counts per 100 cases, i.e. the cost the objective would charge on
average. One ordinary objective evaluation per point is printed alongside for comparison.

Usage (repository root, PYTHONPATH=src):
    python scripts/probes/probe_event_tradeoff.py BEST_TXT OUT_CSV [N_INDEX_CASES]
"""
import concurrent.futures
import copy
import csv
import pickle
import subprocess
import sys

import numpy as np

from covsyn.calibration import fast_cost
from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.cost_parts import LAST
from covsyn.figures.plot_results import create_array_cheng2020_fig2
from covsyn.model.data_synthesis_main import run_covid

PROBABILITIES = [0.0, 0.012, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
OTHERS = [i for i, (name, _) in enumerate(fast_cost.CHENG_GROUPS) if name == 'Cheng others'][0]
SEEDS_PER_TASK = 50


def measure_seeds(P, seeds):
    """Per seed: Cheng bins summed over all cases, index community contacts and infections."""
    demo = copy.deepcopy(fast_cost._WORKER_DEMOGRAPHIC_PARAMETERS)
    out = []
    for seed in seeds:
        _, _, courses, contacts = run_covid(seed, P.copy(), demo, save_file=False, mode='result')
        # The SAMPLED bins (expected_events=False), as E85 measured them; the objective now
        # bins the expected event contacts instead (B55).
        bins = np.zeros((len(fast_cost.CHENG_GROUPS), 2, 6))
        for c, k in zip(courses, contacts):
            for gi, (_group, layers) in enumerate(fast_cost.CHENG_GROUPS):
                for layer in layers:
                    _, cnt, _, inf = create_array_cheng2020_fig2([c], [k], layer=layer)
                    if cnt.size == 6:
                        bins[gi, 0] += cnt
                        bins[gi, 1] += inf
        effective = np.asarray(contacts[0]['municipality_effective_contacts'] or [], dtype=float)
        out.append((bins, len(courses), len(effective), float(np.nansum(effective))))
    return out


def summarise(rows, cheng_contact):
    bins = sum(r[0] for r in rows)
    cases = sum(r[1] for r in rows)
    community = np.array([r[2] for r in rows], dtype=float)
    nonzero = community[community > 0]
    per_100 = bins[OTHERS, 0] * 100.0 / cases
    scale = np.max(cheng_contact)
    expected_others_cost = float(np.sum((per_100 / scale - cheng_contact[OTHERS] / scale) ** 2))
    return {
        'median_all': float(np.median(community)),
        'p90_nonzero': float(np.percentile(nonzero, 90)),
        'tail_ratio': float(np.percentile(nonzero, 90) / np.median(nonzero)),
        'mean_contacts': float(community.mean()),
        'max_contacts': float(community.max()),
        'community_infections_per_index': float(np.mean([r[3] for r in rows])),
        'others_contacts_per_100_cases': float(per_100.sum()),
        'expected_cost_contact_others': expected_others_cost,
    }


def top_outcome_terms(parts, count=4):
    """The outcome targets that contribute most to cost_outcome, as 'name=penalty(measured)'."""
    terms = []
    for name, (_lo, _hi, w) in fo.OUTCOME_TARGETS.items():
        x = parts.get('measured_' + name, np.nan)
        if w > 0 and np.isfinite(x):
            terms.append((fo.outcome_penalty({name: x}), name, x))
    terms.sort(reverse=True)
    return '; '.join(f'{n}={pen:.3g}({x:.3g})' for pen, n, x in terms[:count] if pen > 0)


def main():
    best_path, out_path = sys.argv[1], sys.argv[2]
    n_cases = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    best = np.atleast_2d(np.loadtxt(best_path))
    base = best[int(np.argmin(best[:, -1])), 1:-1].copy()
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    cheng_contact = np.asarray(ct['Cheng_contact_array'], dtype=float)
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            check=False).stdout.strip()
    print(f'commit {commit}; base {best_path}; P[199] there {base[199]:.4f}; '
          f'{n_cases} index cases per point; Cheng others contacts per 100 cases '
          f'{cheng_contact[OTHERS].sum():.0f}', flush=True)

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns))
    rows_out = []
    for p in PROBABILITIES:
        P = base.copy()
        P[199] = p
        tasks = [pool.submit(measure_seeds, P, list(range(s, min(s + SEEDS_PER_TASK, n_cases))))
                 for s in range(0, n_cases, SEEDS_PER_TASK)]
        stats = summarise([r for t in tasks for r in t.result()], cheng_contact)
        total = fast_cost.cost_function(P, demo, pool, *cheng)
        stats.update(event_probability=p, objective_total=float(total),
                     objective_cost_contact_others=float(LAST.get('cost_contact_others', np.nan)),
                     objective_cost_outcome=float(LAST.get('cost_outcome', np.nan)),
                     objective_tail_ratio=float(LAST.get('measured_community_tail_ratio', np.nan)))
        stats['objective_top_outcome_terms'] = top_outcome_terms(LAST)
        rows_out.append(stats)
        print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}=[{v}]'
                       for k, v in stats.items()), flush=True)
    pool.shutdown()

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f'wrote {out_path} (commit {commit})')


if __name__ == '__main__':
    main()
