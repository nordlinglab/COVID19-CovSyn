# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Re-score an optimizer run's candidates on seeds it never saw (finding E87).

The objective evaluates every vector on the same 300 seeds, so the lowest training cost is
partly a property of those seeds: run 12's best vector sat on the edge of several targets on
them and fell outside on 1,000 independent simulations. This scores the final population and
every firefly's personal best on fresh, disjoint seed blocks and picks the lowest mean.

Usage (repository root, after uv sync):
    uv run python -m covsyn.calibration.revalidation FIREFLY_DIR OUT_DIR [BLOCKS] [--workers N]

Writes OUT_DIR/revalidation.csv (every candidate) and OUT_DIR/firefly_best.txt (every candidate
with its validation cost, so its minimum is the chosen vector), and copies the run's other files
(bound.txt, ...), so the post-run chain can be pointed at OUT_DIR; the progress table stays in
results/tables/<run-id>/, where runs.progress_path finds it from OUT_DIR too.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from covsyn.calibration import run_size, seed_streams
from covsyn.runtime import hardware

REPORTED_OUTCOMES = (
    "community_tail_ratio",
    "community_median",
    "daily_municipality",
    "medical_early_share",
    "medical_late_share",
    "closure_after_confirmation_asymptomatic",
    "icu_to_closure",
    "onset_to_icu",
    "infections_per_index_health_care",
    "infections_per_index_municipality",
    "offspring_k",
)


def validation_seed_blocks(
    blocks: int, simulations: int, stream: int = seed_streams.DEFAULT_VALIDATION_STREAM
) -> list[range]:
    """Blocks 0 to blocks - 1 of a validation stream, one objective evaluation each (Task 7).

    Until Task 7 they started at seed 100,000; each use now takes its own stream of 200-299,
    so every use gets fresh blocks and none meets the optimisation's.

    Args:
        blocks: Number of blocks.
        simulations: Simulations per objective evaluation.
        stream: The validation stream, 200 to 299.

    Returns:
        One range of seeds per block.

    Raises:
        ValueError: The stream is not a validation stream, or a block falls outside it.
    """
    seed_streams.require(stream, seed_streams.VALIDATION_STREAMS, "validation")
    return [seed_streams.block_seeds(stream, b, simulations) for b in range(blocks)]


# Task 8: the first line of firefly_archive.txt from Task 8 on, after np.savetxt's '# '. Its rows
# are (entered, blocks, params..., mean cost), one per vector that was in the elite; before, the
# file had no header and held B81's snapshots (generation, params..., cost).
ARCHIVE_HEADER = "firefly_archive v2: entered blocks params... mean_cost (Task 8)"


def _hide_no_data() -> None:
    """Hide numpy's warning that a file holds no rows, and no other (in catch_warnings)."""
    warnings.filterwarnings("ignore", message=".*(no data|[Ee]mpty input file).*",
                            category=UserWarning)  # fmt: skip


def _rows(path: Path, **kwargs: Any) -> np.ndarray:
    """np.loadtxt of a run's file, whose errors name the file, as loadtxt's do not."""
    try:
        return np.loadtxt(path, **kwargs)
    except ValueError as error:  # UnicodeDecodeError too
        raise ValueError(f"{path}: {error}") from error


def _first_line(path: Path) -> str:
    """A file's first line, as UTF-8; an error names the file."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.readline()
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: not UTF-8 text ({error.reason})") from error


def read_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The vectors of a firefly_archive.txt and their costs, of either version.

    Returns:
        (vectors, costs): from Task 8 on, each elite vector's mean over the blocks it was
        scored on in the elite; before, each snapshot's cost on its generation's block. Both
        are empty when the file holds no rows.
    """
    version_2 = _first_line(path).startswith("# " + ARCHIVE_HEADER)
    with warnings.catch_warnings():  # a file of no rows is an empty archive, not a mistake
        _hide_no_data()
        rows = _rows(path, ndmin=2, encoding="utf-8")  # the header is a comment
    if rows.size == 0:
        return np.empty((0, 0)), np.empty(0)
    first = 2 if version_2 else 1
    return rows[:, first:-1], rows[:, -1]


# Task 9's pilot 2 and Task 10 re-score a run's best k = 10 archive candidates.
ARCHIVE_CANDIDATES = 10


def ranked_archive(path: Path, k: int = ARCHIVE_CANDIDATES) -> tuple[np.ndarray, np.ndarray]:
    """A run's best k distinct archive vectors and their means, lowest first.

    By the archive's mean cost, a tie to the vector that entered the elite first, then to the
    earlier row (Task 9's pilot 2 and Task 10, decisions of 2026-11-21 and 2026-11-22).

    Raises:
        ValueError: The archive is not Task 8's (v2): its rows are not elite vectors with
            their means.
    """
    if _first_line(path).strip() != f"# {ARCHIVE_HEADER}":
        raise ValueError(f"{path} is not a v2 archive (Task 8), so it has no means to rank")
    with warnings.catch_warnings():  # a file of no rows is an empty archive, not a mistake
        _hide_no_data()
        rows = _rows(path, ndmin=2, encoding="utf-8")
    if rows.size == 0:
        return np.empty((0, 0)), np.empty(0)
    order = np.lexsort((np.arange(len(rows)), rows[:, 0], rows[:, -1]))
    vectors: list[np.ndarray] = []
    means: list[float] = []
    for i in order:
        vector = rows[i, 2:-1]
        if not any(np.array_equal(vector, v) for v in vectors):
            vectors.append(vector)
            means.append(float(rows[i, -1]))
        if len(vectors) == k:
            break
    width = rows.shape[1] - 3
    return np.array(vectors).reshape(len(vectors), width), np.array(means)


def archive_candidates(path: Path, k: int = ARCHIVE_CANDIDATES) -> np.ndarray:
    """A run's best k distinct archive vectors, as ranked_archive orders them."""
    return ranked_archive(path, k)[0]


def load_candidates(firefly_dir: Path, archive_best: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The final population and every personal best of a run, without duplicates.

    firefly_archive.txt is added when present, so the pool spans the search and not only its
    end: from Task 8 on every vector that was in the elite, with its mean over the blocks it
    was scored on there; before, B81's snapshot of one generation in five. Its costs come from
    different seed blocks and are not comparable with each other; they order the rows and
    nothing else, because every candidate is re-scored below (read_archive).

    Args:
        firefly_dir: Directory holding firefly_result.txt (params..., cost),
            firefly_best.txt (iteration, params..., cost) and optionally firefly_archive.txt.

        archive_best: Take only the archive's best this many distinct vectors (Task 10),
            not the final population nor the personal bests.

    Returns:
        (vectors, training costs), sorted by training cost.

    Raises:
        ValueError: A file it reads is malformed, or, with archive_best, the archive is not
            Task 8's (v2); the message names the file.
        OSError: A file it reads is missing or cannot be read.
    """
    if archive_best is not None:
        return ranked_archive(Path(firefly_dir) / "firefly_archive.txt", archive_best)
    result_path = Path(firefly_dir) / "firefly_result.txt"
    best_path = Path(firefly_dir) / "firefly_best.txt"
    with warnings.catch_warnings():  # a file of no rows is refused below, by name
        _hide_no_data()
        population = _rows(result_path, ndmin=2)
        bests = _rows(best_path, ndmin=2)
    for path, table in ((result_path, population), (best_path, bests)):
        if table.size == 0:
            raise ValueError(f"{path}: no rows")
    rows = [population[:, :-1], bests[:, 1:-1]]
    values = [population[:, -1], bests[:, -1]]
    paths = [result_path, best_path]
    archive_path = Path(firefly_dir) / "firefly_archive.txt"
    if archive_path.exists() and archive_path.stat().st_size:
        vectors, costs = read_archive(archive_path)
        if len(vectors):
            rows.append(vectors)
            values.append(costs)
            paths.append(archive_path)
    if rows[0].shape[1] == 0:
        raise ValueError(f"{result_path}: no parameters, only costs")
    if len({r.shape[1] for r in rows}) > 1:
        widths = ", ".join(f"{p} {r.shape[1]}" for p, r in zip(paths, rows, strict=True))
        raise ValueError(f"the vectors differ in length: {widths}")
    vectors = np.vstack(rows)
    costs = np.concatenate(values)
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
    np.savetxt(path, rows, fmt="%.7f")


# Written here, or the pipeline's records of the whole run rather than of the optimiser.
NOT_COPIED = ("firefly_best.txt", "revalidation.csv", "optimiser.log", "run.log",
              "pipeline.toml", "pipeline.log", "pipeline.lock")


def copy_run_files(firefly_dir: Path, out_dir: Path) -> None:
    """Copy every file of the optimizer run except the two this module writes itself.

    The steps after it read the run from OUT_DIR, so OUT_DIR must hold the whole run; the
    pipeline's own records of the run (Task 4) stay with the run itself.
    """
    for source in Path(firefly_dir).iterdir():
        if source.is_file() and source.name not in NOT_COPIED:
            shutil.copy(source, Path(out_dir) / source.name)


def score(
    vector: np.ndarray,
    demo: Any,
    pool: concurrent.futures.Executor,
    cheng: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed_blocks: list[range],
    simulations: int | None = None,
) -> tuple[list[float], dict[str, float]]:
    """Objective cost and reported outcomes of one vector on each seed block, of S simulations.

    simulations: S, the optimiser's 300 when None (Task 10: the production run's S).
    """
    from covsyn.calibration import fast_cost
    from covsyn.calibration.cost_parts import LAST

    totals: list[float] = []
    outcomes: dict[str, list[float]] = {name: [] for name in REPORTED_OUTCOMES}
    for block in seed_blocks:
        totals.append(
            float(fast_cost.cost_function(vector, demo, pool, *cheng, seed_offset=block.start,
                                    simulations=simulations))
        )
        for name in REPORTED_OUTCOMES:
            outcomes[name].append(float(LAST.get("measured_" + name, np.nan)))
    return totals, {name: float(np.nanmean(v)) for name, v in outcomes.items()}


def _finite_block_cost(value: float, candidate: int, block: str) -> float:
    """Read a block cost and reject nonfinite input before using the candidate."""
    cost = float(value)
    if not np.isfinite(cost):
        raise ValueError(f"candidate {candidate} has nonfinite cost for {block}")
    return cost


def cost_uncertainty(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each candidate's mean cost +- SE, and its paired difference to the best (Task 10).

    Over the blocks of revalidation.csv's rows (block_0, block_1, ...): the best is the lowest
    mean, a tie to the earlier row, the archive's order; d = candidate - best, block by block,
    with a paired t interval of n - 1 degrees of freedom. The best is chosen on these blocks,
    so the interval is exploratory (decision of 2026-11-22). A candidate whose interval holds 0
    cannot be told apart from the best. Fewer than two blocks also count as
    indistinguishable: the data are insufficient to estimate an SE or interval.
    The validation requires finite block costs. Failed evaluations use the finite
    FAILED_EVALUATION_COST sentinel. Nonfinite values signal an upstream fault and are
    rejected. Finite arithmetic overflow outside the observed cost scale is out of scope.

    Raises:
        ValueError: No candidates, mismatched block keys, or a nonfinite block cost.
    """
    from scipy import stats

    if not rows:
        raise ValueError("cost uncertainty requires at least one candidate")
    block_keys = {key for key in rows[0] if key.startswith("block_")}
    columns = sorted(block_keys, key=lambda key: (int(key[6:]), key))
    costs = []
    for row in rows:
        if {key for key in row if key.startswith("block_")} != block_keys:
            reference = rows[0]["candidate"]
            raise ValueError(
                f"candidate {row['candidate']} has different block keys from {reference}"
            )
        costs.append(np.array([
            _finite_block_cost(row[key], row["candidate"], key) for key in columns
        ]))
    means = [float(c.mean()) if len(c) else np.nan for c in costs]
    best = min(range(len(rows)), key=lambda i: (means[i], i))
    table = []
    for i, c in enumerate(costs):
        n = len(c)
        d = c - costs[best]
        difference = float(d.mean()) if n else np.nan
        if n < 2:
            se = low = high = np.nan
            indistinguishable = True
        else:
            se = float(c.std(ddof=1) / np.sqrt(n))
            half = (float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
                    if i != best else 0.0)  # fmt: skip
            low, high = difference - half, difference + half
            indistinguishable = bool(low <= 0 <= high)
        table.append({"candidate": rows[i]["candidate"], "mean": means[i],
                      "se": se, "difference_to_best": difference, "ci_low": low, "ci_high": high,
                      "indistinguishable": indistinguishable, "best": i == best})
    return table


def main(argv: list[str] | None = None) -> None:
    """Re-score every candidate of FIREFLY_DIR and write OUT_DIR."""
    import pickle

    from covsyn.calibration import fast_cost
    from covsyn.calibration import firefly_optimizer as fo

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("firefly_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("blocks", type=int, nargs="?", default=10)  # E95: 4 left it noisy
    parser.add_argument(
        "--stream",
        type=int,
        default=seed_streams.DEFAULT_VALIDATION_STREAM,
        help="validation stream, 200 to 299, whose blocks 0.. score the candidates (Task 7)",
    )
    parser.add_argument("--simulations", type=int, default=None,
                        help="simulations per block, S (default: the optimiser's 300); Task 10 "
                             "re-scores at the production run's S")  # fmt: skip
    parser.add_argument("--archive-best", type=int, default=None,
                        help="score only the archive's best this many distinct vectors (Task 10)")
    hardware.add_workers_argument(parser)  # Task 2: was 32 workers (D3)
    args = parser.parse_args(argv)
    simulations = fo.SIMULATIONS_PER_EVALUATION if args.simulations is None else args.simulations
    try:  # before anything is read: S within one block of a stream
        run_size.check_simulations(simulations, "--simulations")
    except ValueError as error:
        parser.error(str(error))
    if args.blocks < 1:
        parser.error(f"the blocks must be 1 or more, not {args.blocks}")
    if args.archive_best is not None and args.archive_best < 1:
        parser.error(f"--archive-best must be 1 or more, not {args.archive_best}")
    firefly_dir, out_dir, blocks = args.firefly_dir, args.out_dir, args.blocks
    workers = hardware.resolve_workers(args.workers, os.environ, None)
    try:
        vectors, train = load_candidates(firefly_dir, args.archive_best)
    except (ValueError, OSError) as error:  # such as an archive from before Task 8
        raise SystemExit(str(error)) from None  # each names its file
    if not len(vectors):  # before anything is written or read
        reason = " (its archive is empty)" if args.archive_best is not None else ""
        raise SystemExit(f"{firefly_dir}: no candidates to revalidate{reason}")
    try:
        seed_blocks = validation_seed_blocks(blocks, simulations, args.stream)
    except ValueError as error:
        parser.error(str(error))
    with open("./variable/demographic_parameters.pkl", "rb") as f:
        demo = pickle.load(f)
    with open("./variable/processed_contact_tracing_data.pkl", "rb") as f:
        ct = pickle.load(f)
    cheng = (ct["Cheng_contact_array"], ct["Cheng_attack_rate"], ct["norm_weights"])
    columns = np.load("./variable/Taiwan_data_matrix.npy").shape[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    print(
        f"commit {commit or '(not a git checkout)'}; {len(vectors)} candidates from "
        f"{firefly_dir}; {blocks} blocks of {simulations} seeds of stream "
        f"{args.stream}, from {seed_blocks[0].start if seed_blocks else '-'}",
        flush=True,
    )

    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=fast_cost.init_worker, initargs=(demo, columns)
    ) as pool:
        for index, (vector, training_cost) in enumerate(zip(vectors, train, strict=True)):
            totals, outcomes = score(vector, demo, pool, cheng, seed_blocks, simulations)
            totals = [
                _finite_block_cost(value, index, f"block_{block}")
                for block, value in enumerate(totals)
            ]
            rows.append(
                {
                    "candidate": index,
                    "training_cost": float(training_cost),
                    "validation_mean": float(np.mean(totals)),
                    "validation_sd": float(np.std(totals, ddof=1)) if blocks > 1 else 0.0,
                    **{f"block_{b}": t for b, t in enumerate(totals)},
                    **outcomes,
                }
            )
            print(
                f"{index:3d} train {training_cost:.4f} validation {np.mean(totals):.4f} "
                f"(sd {rows[-1]['validation_sd']:.4f})",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "revalidation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Task 10: each candidate's mean +- SE and paired difference to the best, beside it.
    uncertainty = cost_uncertainty(rows)
    with open(out_dir / "cost_uncertainty.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(uncertainty[0]))
        writer.writeheader()
        writer.writerows(uncertainty)
    chosen = min(rows, key=lambda r: r["validation_mean"])
    write_best_file(
        out_dir / "firefly_best.txt", vectors, np.array([r["validation_mean"] for r in rows])
    )
    copy_run_files(firefly_dir, out_dir)
    training_best = rows[0]
    print(
        f"\ntraining best: candidate 0, train {training_best['training_cost']:.4f}, "
        f"validation {training_best['validation_mean']:.4f}"
    )
    print(
        f"validation best: candidate {chosen['candidate']}, train {chosen['training_cost']:.4f}, "
        f"validation {chosen['validation_mean']:.4f} -> {out_dir / 'firefly_best.txt'}"
    )


if __name__ == "__main__":
    main()
