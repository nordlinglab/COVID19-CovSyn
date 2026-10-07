# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Re-score an optimizer run's candidates on seeds it never saw (finding E87).

The objective evaluates every vector on the same 300 seeds, so the lowest training cost is
partly a property of those seeds: run 12's best vector sat on the edge of several targets on
them and fell outside on 1,000 independent simulations. This scores the final population and
every firefly's personal best on fresh, disjoint seed blocks and picks the lowest mean.

Usage (repository root, PYTHONPATH=src):
    python -m covsyn.calibration.revalidation FIREFLY_DIR OUT_DIR [BLOCKS]

Writes OUT_DIR/revalidation.csv (every candidate) and OUT_DIR/firefly_best.txt (every candidate
with its validation cost, so its minimum is the chosen vector), and copies the run's other files
(bound.txt, progress_metrics.csv, ...), so the post-run chain can be pointed at OUT_DIR.
"""
from __future__ import annotations

import concurrent.futures
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Far above the objective's seeds 0..299 and the chain's Monte-Carlo seeds 0..999.
VALIDATION_SEED_START = 100000
REPORTED_OUTCOMES = ('community_tail_ratio', 'community_median', 'daily_municipality',
                     'medical_early_share', 'medical_late_share', 'closure_asymptomatic',
                     'infections_per_index_health_care', 'infections_per_index_municipality',
                     'offspring_k')


def validation_seed_blocks(blocks: int, simulations: int) -> list[range]:
    """Disjoint seed blocks of one objective evaluation each, starting at VALIDATION_SEED_START.

    Args:
        blocks: Number of blocks.
        simulations: Simulations per objective evaluation.

    Returns:
        One range of seeds per block.
    """
    return [range(VALIDATION_SEED_START + b * simulations,
                  VALIDATION_SEED_START + (b + 1) * simulations) for b in range(blocks)]


def load_candidates(firefly_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """The final population and every personal best of a run, without duplicates.

    Args:
        firefly_dir: Directory holding firefly_result.txt (params..., cost) and
            firefly_best.txt (iteration, params..., cost).

    Returns:
        (vectors, training costs), sorted by training cost.
    """
    population = np.atleast_2d(np.loadtxt(Path(firefly_dir) / 'firefly_result.txt'))
    bests = np.atleast_2d(np.loadtxt(Path(firefly_dir) / 'firefly_best.txt'))
    vectors = np.vstack([population[:, :-1], bests[:, 1:-1]])
    costs = np.concatenate([population[:, -1], bests[:, -1]])
    _, first = np.unique(vectors, axis=0, return_index=True)
    order = sorted(first, key=lambda i: costs[i])
    return vectors[order], costs[order]


def write_best_file(path: Path, vectors: np.ndarray, costs: np.ndarray) -> None:
    """Write candidates in firefly_best.txt's layout (row number, params..., cost).

    The report scripts index the loaded file as a matrix, so a single candidate is written
    twice rather than as a one-dimensional row; the lowest cost is the chosen vector.
    """
    vectors = np.atleast_2d(vectors)
    costs = np.atleast_1d(costs)
    rows = np.column_stack([np.arange(len(vectors), dtype=float), vectors, costs])
    if len(rows) == 1:
        rows = np.vstack([rows, rows])
    np.savetxt(path, rows, fmt='%.7f')


def copy_run_files(firefly_dir: Path, out_dir: Path) -> None:
    """Copy every file of the optimizer run except the two this module writes itself.

    phase_d_chain.sh replaces the run directory with OUT_DIR before the run report, and
    show_final reads progress_metrics.csv from it, so OUT_DIR must hold the whole run.
    """
    for source in Path(firefly_dir).iterdir():
        if source.is_file() and source.name not in ('firefly_best.txt', 'revalidation.csv'):
            shutil.copy(source, Path(out_dir) / source.name)


def score(vector: np.ndarray, demo: Any, pool: concurrent.futures.Executor,
          cheng: tuple[np.ndarray, np.ndarray, np.ndarray],
          seed_blocks: list[range]) -> tuple[list[float], dict[str, float]]:
    """Objective cost and reported outcomes of one vector on each seed block."""
    from covsyn.calibration import fast_cost
    from covsyn.calibration.cost_parts import LAST

    totals, outcomes = [], {name: [] for name in REPORTED_OUTCOMES}
    for block in seed_blocks:
        totals.append(float(fast_cost.cost_function(vector, demo, pool, *cheng,
                                                    seed_offset=block.start)))
        for name in REPORTED_OUTCOMES:
            outcomes[name].append(float(LAST.get('measured_' + name, np.nan)))
    return totals, {name: float(np.nanmean(v)) for name, v in outcomes.items()}


def main() -> None:
    import pickle

    from covsyn.calibration import fast_cost
    from covsyn.calibration import firefly_optimizer as fo

    firefly_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    out_dir.mkdir(parents=True, exist_ok=True)
    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    with open('./variable/processed_contact_tracing_data.pkl', 'rb') as f:
        ct = pickle.load(f)
    cheng = (ct['Cheng_contact_array'], ct['Cheng_attack_rate'], ct['norm_weights'])
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            check=False).stdout.strip()
    vectors, train = load_candidates(firefly_dir)
    seed_blocks = validation_seed_blocks(blocks, fo.SIMULATIONS_PER_EVALUATION)
    print(f'commit {commit or "(not a git checkout)"}; {len(vectors)} candidates from '
          f'{firefly_dir}; {blocks} blocks of {fo.SIMULATIONS_PER_EVALUATION} seeds from '
          f'{VALIDATION_SEED_START}', flush=True)

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, initializer=fast_cost.init_worker, initargs=(demo, columns))
    rows = []
    for index, (vector, training_cost) in enumerate(zip(vectors, train)):
        totals, outcomes = score(vector, demo, pool, cheng, seed_blocks)
        rows.append({'candidate': index, 'training_cost': float(training_cost),
                     'validation_mean': float(np.mean(totals)),
                     'validation_sd': float(np.std(totals, ddof=1)) if blocks > 1 else 0.0,
                     **{f'block_{b}': t for b, t in enumerate(totals)}, **outcomes})
        print(f'{index:3d} train {training_cost:.4f} validation {np.mean(totals):.4f} '
              f'(sd {rows[-1]["validation_sd"]:.4f})', flush=True)
    pool.shutdown()

    with open(out_dir / 'revalidation.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    chosen = min(rows, key=lambda r: r['validation_mean'])
    write_best_file(out_dir / 'firefly_best.txt', vectors,
                    np.array([r['validation_mean'] for r in rows]))
    copy_run_files(firefly_dir, out_dir)
    training_best = rows[0]
    print(f'\ntraining best: candidate 0, train {training_best["training_cost"]:.4f}, '
          f'validation {training_best["validation_mean"]:.4f}')
    print(f'validation best: candidate {chosen["candidate"]}, train {chosen["training_cost"]:.4f}, '
          f'validation {chosen["validation_mean"]:.4f} -> {out_dir / "firefly_best.txt"}')


if __name__ == '__main__':
    main()
