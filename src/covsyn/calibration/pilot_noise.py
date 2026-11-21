# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pilot 1 of Task 9: the objective's noise against the simulations per evaluation, S.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 9. It scores the literature seed
(row 0 of the initial population), the run-19 vector (row 2) and two random Latin-hypercube
rows at S in {300, 1,000, 3,000, 10,000}, 30 blocks each, and every row of the population on
two blocks at S = 10,000 for the cost spread. It then chooses the smallest S at which the
largest SD of a paired difference between the vectors is at most 10% of that spread.

Decisions of 2026-11-21: validation stream 201; each S, and the spread, on blocks whose seeds
no other part of the pilot uses, so means are compared across S unpaired; the random rows
drawn without replacement from rows 3-255 by numpy's default_rng(ROW_SEED); the spread the
sample SD across the 256 rows of each row's mean on its two blocks; all six pairs must pass; no
S passing chooses none, and the measurements go to the professor.

Usage (repository root, after uv sync):
    uv run covsyn-pilot-noise [--workers N]
    uv run covsyn-pilot-noise --resume results/tables/<run-id>

Writes results/tables/<run-id>/: config.json (the command, commit, rows, seed, stream and
blocks), blocks.csv (one row per evaluation, appended as each finishes, so a killed pilot
resumes where it stopped), and the summaries vectors.csv, pairs.csv and choice.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from covsyn.calibration import block_scoring, initial_population, parameter_space, seed_streams
from covsyn.runtime import hardware, runs

STREAM = 201
# Each S on its own blocks of the stream: block b at S starts at seed b x S, so these ranges
# (seeds 0-8,999, 9,000-38,999, 39,000-128,999 and 130,000-429,999) do not overlap.
BLOCKS = {300: range(0, 30), 1_000: range(9, 39), 3_000: range(13, 43), 10_000: range(13, 43)}
SPREAD_SIMULATIONS = 10_000
SPREAD_BLOCKS = (43, 44)  # seeds 430,000-449,999, after the last S's
ROW_SEED = 20261121
RANDOM_ROWS = range(3, initial_population.N_MAX)  # the Latin hypercube, after the warm starts
THRESHOLD = 0.10  # of the spread
FAILED_EVALUATION_COST = 1e6  # firefly_optimizer.FAILED_EVALUATION_COST, without the model

CONFIG_FILE = "config.json"
BLOCKS_FILE = "blocks.csv"
VECTORS_FILE = "vectors.csv"
PAIRS_FILE = "pairs.csv"
CHOICE_FILE = "choice.json"
COLUMNS = ("kind", "label", "row", "simulations", "block", "seed_offset", "cost")


class Task(NamedTuple):
    """One evaluation: a vector or a population row on one block of the stream at S."""

    kind: str  # "vector" or "spread"
    label: str  # the vector's name; "" for a spread row
    row: int
    simulations: int
    block: int
    seed_offset: int

    @property
    def key(self) -> tuple[str, int, int, int]:
        """What identifies it in blocks.csv."""
        return self.kind, self.row, self.simulations, self.block


def pilot_rows(seed: int = ROW_SEED) -> dict[str, int]:
    """The four vectors' rows of the initial population: rows 0 and 2, and two random ones."""
    drawn = np.random.default_rng(seed).choice(np.array(RANDOM_ROWS), size=2, replace=False)
    return {"literature": 0, "run19": 2, "random_a": int(drawn[0]), "random_b": int(drawn[1])}


def tasks(vectors: dict[str, int], rows: int) -> list[Task]:
    """Every evaluation of the pilot: the vectors at each S, then the population's rows."""
    found = []
    for simulations, blocks in BLOCKS.items():
        for block in blocks:
            offset = seed_streams.block_seeds(STREAM, block, simulations).start
            found += [Task("vector", label, row, simulations, block, offset)
                      for label, row in vectors.items()]  # fmt: skip
    for block in SPREAD_BLOCKS:
        offset = seed_streams.block_seeds(STREAM, block, SPREAD_SIMULATIONS).start
        found += [Task("spread", "", row, SPREAD_SIMULATIONS, block, offset)
                  for row in range(rows)]  # fmt: skip
    return found


def _sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The pilot's summary from blocks.csv's rows.

    Returns:
        vectors: per S and vector, its blocks, mean and SD; pairs: per S and pair of vectors,
        the mean and SD of their difference block by block; spread: the SD across rows of each
        row's mean over the spread blocks, and each block's own; choice: per S the largest
        paired SD, its ratio to the spread and whether it is within THRESHOLD of it, and the
        smallest S that is (None when none, or when any evaluation failed); failed_evaluations:
        costs of a failed simulation.
    """
    cost = {(r["kind"], r["label"], int(r["row"]), int(r["simulations"]), int(r["block"])):
            float(r["cost"]) for r in rows}  # fmt: skip
    present = {label for kind, label, *_ in cost if kind == "vector"}
    labels = [label for label in pilot_rows() if label in present]  # not the finishing order
    labels += sorted(present - set(labels))
    vectors: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for simulations in BLOCKS:
        by = {(label, block): c for (kind, label, _, s, block), c in cost.items()
              if kind == "vector" and s == simulations}  # fmt: skip
        blocks = sorted({block for _, block in by})
        for label in labels:
            values = [by[(label, b)] for b in blocks if (label, b) in by]
            vectors.append({"simulations": simulations, "label": label, "blocks": len(values),
                            "mean": float(np.mean(values)), "sd": _sd(values)})  # fmt: skip
        for a, b in itertools.combinations(labels, 2):
            d = [by[(a, k)] - by[(b, k)] for k in blocks if (a, k) in by and (b, k) in by]
            pairs.append({"simulations": simulations, "pair": f"{a}-{b}", "blocks": len(d),
                          "mean_difference": float(np.mean(d)), "sd_difference": _sd(d)})
    spread_cost = {(row, block): c for (kind, _, row, _, block), c in cost.items()
                   if kind == "spread"}  # fmt: skip
    row_ids = sorted({row for row, _ in spread_cost})
    means = [float(np.mean([spread_cost[(row, b)] for b in SPREAD_BLOCKS])) for row in row_ids]
    spread: dict[str, Any] = {"rows": len(row_ids), "simulations": SPREAD_SIMULATIONS,
                              "sd": _sd(means),
              **{f"sd_block_{b}": _sd([spread_cost[(row, b)] for row in row_ids])
                 for b in SPREAD_BLOCKS}}  # fmt: skip
    threshold = THRESHOLD * spread["sd"]
    # A failed evaluation's 1e6 is no cost: it would swamp the spread, so no S passes.
    failed = sum(c >= FAILED_EVALUATION_COST for c in cost.values())
    per_simulations = []
    for simulations in BLOCKS:
        largest = max(p["sd_difference"] for p in pairs if p["simulations"] == simulations)
        per_simulations.append({"simulations": simulations, "max_pair_sd": largest,
                                "ratio": largest / spread["sd"],
                                "meets": bool(largest <= threshold) and not failed})
    chosen = next((line["simulations"] for line in per_simulations if line["meets"]), None)
    choice = {"threshold_fraction": THRESHOLD, "spread_sd": spread["sd"], "threshold": threshold,
              "per_simulations": per_simulations, "chosen_simulations": chosen,
              "failed_evaluations": failed}
    return {"vectors": vectors, "pairs": pairs, "spread": spread, "choice": choice,
            "failed_evaluations": failed}


def write_summary(out_dir: Path, warn: bool = True) -> dict[str, Any]:
    """Summarise out_dir's blocks.csv into vectors.csv, pairs.csv and choice.json.

    warn: as read_table's; run() warns here only about a mark added since its first read.

    Raises:
        block_scoring.TableError: blocks.csv is gone, or not a regular file.
    """
    rows = block_scoring.read_table(Path(out_dir) / BLOCKS_FILE, warn=warn, required=True).rows
    summary = summarise(rows)
    block_scoring.write_csv(Path(out_dir) / VECTORS_FILE, summary["vectors"])
    block_scoring.write_csv(Path(out_dir) / PAIRS_FILE, summary["pairs"])
    choice = {**summary["choice"], "spread": summary["spread"]}
    (Path(out_dir) / CHOICE_FILE).write_text(json.dumps(choice, indent=1) + "\n", encoding="utf-8")
    return summary


def run(out_dir: Path, population: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        evaluate: block_scoring.Evaluate, make_pool: block_scoring.MakePool) -> dict[str, Any]:
    """Score every evaluation not yet in out_dir/blocks.csv, then write the summaries.

    Args:
        out_dir: The pilot's directory.
        population: The initial population's normalised rows, all of them.
        lower: The parameters' lower bounds.
        upper: The parameters' upper bounds.
        evaluate: ``evaluate(P, seed_offset)`` returning the cost first, run in the pool, as
            fast_cost.evaluate_candidate.
        make_pool: The pool for one S, whose workers run S simulations per evaluation.
    """
    path = Path(out_dir) / BLOCKS_FILE
    first = block_scoring.read_table(path)  # warns about a mark found now
    done = {(r["kind"], int(r["row"]), int(r["simulations"]), int(r["block"]))
            for r in first.rows}  # fmt: skip
    todo = [t._asdict() for t in tasks(pilot_rows(), len(population)) if t.key not in done]
    parameters = np.asarray(population, dtype=float) * (upper - lower) + lower
    block_scoring.score(path, todo, COLUMNS, lambda t: parameters[t["row"]], evaluate, make_pool,
                        exists=first.exists)  # appended to as found, or created afresh
    return write_summary(out_dir, warn=not first.marked)  # a mark added since: warned now


def _population_and_bounds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The committed initial population, all its rows, checked against today's bounds."""
    space = parameter_space.load()
    path = initial_population.population_path(initial_population.CURRENT_VERSION)
    population = initial_population.load(path, initial_population.N_MAX, space.lower,
                                         space.upper)  # fmt: skip
    return population, space.lower, space.upper


def _evaluator(workers: int | None) -> tuple[block_scoring.Evaluate, block_scoring.MakePool]:
    """The objective and its pools; replaced in the tests."""
    return block_scoring.objective_pools(workers)


def main(argv: list[str] | None = None) -> Path:
    """Run pilot 1, or resume it, and return its directory."""
    parser = argparse.ArgumentParser(prog="covsyn-pilot-noise", description=main.__doc__)
    parser.add_argument("--resume", type=Path, default=None,
                        help="continue the pilot in this results/tables/<run-id> directory")
    parser.add_argument("--results-root", type=Path, default=runs.RESULTS_ROOT,
                        help=argparse.SUPPRESS)  # tests
    hardware.add_workers_argument(parser)
    args = parser.parse_args(argv)
    workers = None
    if args.resume is not None:
        out = args.resume
        try:
            config = json.loads((out / CONFIG_FILE).read_text(encoding="utf-8-sig"))
            recorded_workers = config.get("workers")
        # RecursionError: JSON nested too deep for json.loads
        except (OSError, ValueError, TypeError, AttributeError, RecursionError) as error:
            parser.error(f"{out} holds no pilot to resume ({error!r})")
        try:  # the pilot is there; its recorded count may be wrong
            workers = block_scoring.pilot_workers(args.workers, os.environ, recorded_workers)
        except ValueError as error:
            parser.error(f"pilot 1 in {out}: {error}")
        try:  # the pilot is there; only its record of the workers is at stake
            block_scoring.record_workers(out, workers)
        except PermissionError as error:  # refused: say what was refused
            parser.error(str(error))
        except (OSError, ValueError, RecursionError) as error:  # see record_workers' Raises
            # Named once: by the error, when it names its file, else here.
            where = out / CONFIG_FILE if getattr(error, "filename", None) is None else CONFIG_FILE
            parser.error(f"could not record the workers in {where}: {error}")
    else:
        out = runs.run_directory(runs.make_run_id("pilot-noise"), "tables", args.results_root,
                                 new=True)  # fmt: skip
    population, lower, upper = _population_and_bounds()
    evaluate, make_pool = _evaluator(args.workers if workers is None else workers)
    if args.resume is None:
        settings = {
            "command": shlex.join(["covsyn-pilot-noise", *(sys.argv[1:] if argv is None
                                                           else argv)]),
            "commit": (runs._git("rev-parse", "HEAD") or "").strip() or None,
            "rows": pilot_rows(), "row_seed": ROW_SEED,
            "row_sampler": "numpy.random.default_rng(row_seed).choice(rows 3-255, 2, "
                           "replace=False)",
            "stream": STREAM,
            "blocks": {str(s): [b.start, b.stop - 1] for s, b in BLOCKS.items()},
            "spread": {"simulations": SPREAD_SIMULATIONS, "blocks": list(SPREAD_BLOCKS),
                       "rows": len(population)},
            "threshold_fraction": THRESHOLD,
        }
        workers = hardware.resolve_workers(args.workers, os.environ, None)
        runs.write_config(out, settings, rng_seed=None, workers=workers,
                          initial_population=population)
    try:
        summary = run(out, population, lower, upper, evaluate, make_pool)
    except block_scoring.TableError as error:  # a table gone or out of place: say which
        parser.error(str(error))
    print(json.dumps(summary["choice"], indent=1))
    return out


if __name__ == "__main__":
    main()
