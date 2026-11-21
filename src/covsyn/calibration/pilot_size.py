# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pilot 2 of Task 9: population size and step size, at the S of pilot 1.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 9. It runs the optimiser's pilot
mode for n in {24, 48, 96} x alpha_0 in {0.05, 0.2}, each on optimisation streams 1 and 2,
with a budget of 10,000 evaluations and G_plan = G; it then re-scores each run's best archive
candidates on validation stream 202 and chooses the setting with the lowest mean validated cost
whose two runs agree.

Decisions of 2026-11-21: G = floor((10,000 - n) / n), the initial evaluation counted; at most 10
distinct archive vectors per run, by ascending archive mean, ties to the one that entered the
elite first; blocks 0-29 pick each run's representative (screening), blocks 30-59 score the
representatives (confirmation); a setting's two runs agree when the mean of their differences d,
block by block over the confirmation blocks, is at most SD(d) / sqrt(30); the chosen setting is
the lowest mean of its two representatives' confirmation costs among those that agree, and none
agreeing chooses none, for the professor to decide.

Usage (repository root, after uv sync and pilot 1):
    uv run covsyn-pilot-size --noise results/tables/<pilot-1 run-id> [--workers N]
    uv run covsyn-pilot-size --resume results/tables/<run-id>

Each run is an optimiser run in results/models/<run-id>_pilot-size-n<n>-a<alpha_0>-s<stream>;
a resume keeps the finished ones and continues the others from their checkpoints. The pilot
writes results/tables/<run-id>/: config.json, runs.csv, screening.csv and confirmation.csv
(each scored block appended as it finishes), settings.csv and choice.json.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from covsyn.calibration import block_scoring, seed_streams, step_schedule
from covsyn.calibration.revalidation import ARCHIVE_HEADER
from covsyn.runtime import hardware, runs

POP_SIZES = (24, 48, 96)
ALPHA_0S = (0.05, 0.2)
STREAMS = (1, 2)
BUDGET = 10_000
CANDIDATES = 10
VALIDATION_STREAM = 202
SCREEN_BLOCKS = range(0, 30)
CONFIRM_BLOCKS = range(30, 60)
OPTIMISER = "covsyn.calibration.firefly_optimizer"
FINISHED = "firefly_worst.txt"  # the optimiser's main() writes it last
# A pilot 2's directory name, as runs.make_run_id makes it; its runs share the part before it.
RUN_ID = re.compile(r"[0-9]{8}-[0-9]{6}_(?:[0-9a-f]{7}|nocommit)(?:-dirty)?_pilot-size")

CONFIG_FILE = "config.json"
RUNS_FILE = "runs.csv"
SCREEN_FILE = "screening.csv"
CONFIRM_FILE = "confirmation.csv"
SETTINGS_FILE = "settings.csv"
CHOICE_FILE = "choice.json"
SCREEN_COLUMNS = ("run", "candidate", "simulations", "block", "seed_offset", "cost")
CONFIRM_COLUMNS = ("run", "pop_size", "alpha_0", "stream", "candidate", "simulations", "block",
                   "seed_offset", "cost")  # fmt: skip

Execute = Callable[[Sequence, Path | None, bool, Mapping[str, str]], int]


def generations(pop_size: int) -> int:
    """G for n: the initial n evaluations and n per generation within the budget."""
    return (BUDGET - pop_size) // pop_size


class Setting(NamedTuple):
    """One run of the pilot."""

    pop_size: int
    alpha_0: float
    stream: int

    @property
    def generations(self) -> int:
        """Its G, which is also its G_plan."""
        return generations(self.pop_size)


def settings() -> list[Setting]:
    """The pilot's twelve runs, in the order they run."""
    return [Setting(n, a, s) for n in POP_SIZES for a in ALPHA_0S for s in STREAMS]


def run_name(prefix: str, setting: Setting) -> str:
    """The run's directory name under results/models: the pilot's prefix and the setting."""
    alpha = f"{setting.alpha_0:g}".replace(".", "p")
    return f"{prefix}_pilot-size-n{setting.pop_size}-a{alpha}-s{setting.stream}"


def optimiser_args(setting: Setting, simulations: int, run_dir: Path) -> list[str]:
    """The optimiser's command for a fresh run; G_plan = G is the pilot mode's default."""
    return ["-m", OPTIMISER, "--mode", "pilot", "--pop-size", str(setting.pop_size),
            "--generations", str(setting.generations), "--simulations", str(simulations),
            "--alpha-0", f"{setting.alpha_0:g}", "--stream", str(setting.stream),
            "--run-dir", str(run_dir)]  # fmt: skip


def archive_candidates(path: Path, k: int = CANDIDATES) -> np.ndarray:
    """A run's best k distinct archive vectors, lowest mean first, ties to the earlier entry.

    Raises:
        ValueError: The archive is not Task 8's (v2): its rows are not elite vectors with
            their means.
    """
    with open(path) as f:
        if f.readline().strip() != f"# {ARCHIVE_HEADER}":
            raise ValueError(f"{path} is not a v2 archive (Task 8), so it has no means to rank")
    rows = np.loadtxt(path, ndmin=2)
    order = np.lexsort((np.arange(len(rows)), rows[:, 0], rows[:, -1]))
    chosen: list[np.ndarray] = []
    for i in order:
        vector = rows[i, 2:-1]
        if not any(np.array_equal(vector, v) for v in chosen):
            chosen.append(vector)
        if len(chosen) == k:
            break
    return np.array(chosen).reshape(len(chosen), rows.shape[1] - 3)


def compare_settings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each setting's paired comparison of its two runs over the confirmation blocks."""
    cost = {(int(r["pop_size"]), float(r["alpha_0"]), int(r["stream"]), int(r["block"])):
            float(r["cost"]) for r in rows}  # fmt: skip
    table = []
    for n in POP_SIZES:
        for a in ALPHA_0S:
            blocks = sorted({b for (n_, a_, _, b) in cost if (n_, a_) == (n, a)})
            first = np.array([cost[(n, a, STREAMS[0], b)] for b in blocks])
            second = np.array([cost[(n, a, STREAMS[1], b)] for b in blocks])
            d = first - second
            se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
            table.append({"pop_size": n, "alpha_0": a, "generations": generations(n),
                          "cost_stream_1": float(first.mean()),
                          "cost_stream_2": float(second.mean()),
                          "mean_cost": float(np.mean(np.concatenate([first, second]))),
                          "mean_difference": float(d.mean()), "paired_se": se,
                          "agree": bool(abs(d.mean()) <= se)})  # fmt: skip
    return table


def choose(table: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The agreeing setting of lowest mean cost, with its G as G_plan; None when none agree."""
    agreeing = [line for line in table if line["agree"]]
    if not agreeing:
        return None
    best = min(agreeing, key=lambda line: line["mean_cost"])
    return {"pop_size": best["pop_size"], "alpha_0": best["alpha_0"],
            "generations": best["generations"], "g_plan": best["generations"],
            "alpha_end": step_schedule.DEFAULT_ALPHA_END,  # the pilot mode's
            "mean_cost": best["mean_cost"]}  # fmt: skip


def _evaluations(run_dir: Path) -> int | None:
    """The evaluations of the run's last checkpoint, to check the budget.

    None without one, or with one that cannot be read: the count is only reported.
    """
    last = runs.last_checkpoint(run_dir)
    if last is None:
        return None
    try:
        with np.load(last) as saved:
            return int(saved["evaluations"]) if "evaluations" in saved.files else None
    except (OSError, ValueError):
        return None


def _optimise(models_root: Path, prefix: str, simulations: int, execute: Execute,
              workers: int | None) -> None:
    """Run each setting not yet finished: afresh, or from its checkpoint (as the pipeline).

    Each run takes workers through COVSYN_WORKERS, a resumed one included; None passes none.
    """
    env = {} if workers is None else {hardware.ENV_VARIABLE: str(workers)}
    from covsyn.pipeline import stages

    for setting in settings():
        run_dir = Path(models_root) / run_name(prefix, setting)
        if (run_dir / FINISHED).is_file():  # the optimiser's last file: the run finished
            continue
        resume = stages.has_checkpoint(run_dir)
        if resume:
            args = ["-m", OPTIMISER, "--resume", str(run_dir)]
        else:
            if run_dir.exists():  # killed before its first checkpoint: its claim would refuse
                stages.discard_optimiser_output(run_dir)
            args = optimiser_args(setting, simulations, run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        status = execute(args, run_dir / "optimiser.log", resume, env)
        if status != 0:
            raise RuntimeError(f"run {run_dir.name} failed with status {status}; see "
                               f"{run_dir / 'optimiser.log'}, then resume the pilot")


def _screen_tasks(models_root: Path, prefix: str, simulations: int,
                  done: set) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Each run's candidates on the screening blocks, those not yet scored."""
    candidates, todo = {}, []
    for setting in settings():
        name = run_name(prefix, setting)
        candidates[name] = archive_candidates(Path(models_root) / name / "firefly_archive.txt")
        for c in range(len(candidates[name])):
            for b in SCREEN_BLOCKS:
                if (name, c, b) not in done:
                    offset = seed_streams.block_seeds(VALIDATION_STREAM, b, simulations).start
                    todo.append({"run": name, "candidate": c, "simulations": simulations,
                                 "block": b, "seed_offset": offset})  # fmt: skip
    return todo, candidates


def representatives(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Each run's candidate of lowest screening mean, a tie to the lower index."""
    costs: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        costs.setdefault(r["run"], {}).setdefault(int(r["candidate"]), []).append(float(r["cost"]))
    return {run: min(by, key=lambda c: (float(np.mean(by[c])), c)) for run, by in costs.items()}


def run(out_dir: Path, models_root: Path, prefix: str, simulations: int, execute: Execute,
        evaluate: block_scoring.Evaluate, make_pool: block_scoring.MakePool,
        workers: int | None = None) -> dict[str, Any]:
    """Run, screen and confirm whatever is not yet done, then write the summaries.

    workers: every optimiser run's workers, given it as COVSYN_WORKERS; main() passes the
    pilot's, recorded in its config.json, so a resume runs with the count it started with.
    None, as in a test, passes none.
    """
    out_dir = Path(out_dir)
    _optimise(models_root, prefix, simulations, execute, workers)

    screen_path = out_dir / SCREEN_FILE
    done = {(r["run"], int(r["candidate"]), int(r["block"]))
            for r in block_scoring.read_rows(screen_path)}  # fmt: skip
    todo, candidates = _screen_tasks(models_root, prefix, simulations, done)
    block_scoring.score(screen_path, todo, SCREEN_COLUMNS,
                        lambda t: candidates[t["run"]][t["candidate"]], evaluate, make_pool)
    chosen = representatives(block_scoring.read_rows(screen_path))

    confirm_path = out_dir / CONFIRM_FILE
    confirmed = {(r["run"], int(r["block"])) for r in block_scoring.read_rows(confirm_path)}
    todo = []
    for setting in settings():
        name = run_name(prefix, setting)
        for b in CONFIRM_BLOCKS:
            if (name, b) not in confirmed:
                todo.append({"run": name, "pop_size": setting.pop_size,
                             "alpha_0": setting.alpha_0, "stream": setting.stream,
                             "candidate": chosen[name], "simulations": simulations, "block": b,
                             "seed_offset": seed_streams.block_seeds(
                                 VALIDATION_STREAM, b, simulations).start})  # fmt: skip
    block_scoring.score(confirm_path, todo, CONFIRM_COLUMNS,
                        lambda t: candidates[t["run"]][t["candidate"]], evaluate, make_pool)

    table = compare_settings(block_scoring.read_rows(confirm_path))
    choice = choose(table)
    runs_table = []
    for setting in settings():
        name = run_name(prefix, setting)
        runs_table.append({"run": name, **setting._asdict(), "generations": setting.generations,
                           "evaluations": _evaluations(Path(models_root) / name),
                           "candidates": len(candidates[name]),
                           "representative": chosen[name]})  # fmt: skip
    block_scoring.write_csv(out_dir / RUNS_FILE, runs_table)
    block_scoring.write_csv(out_dir / SETTINGS_FILE, table)
    summary = {"chosen": choice, "simulations": simulations, "budget": BUDGET,
               "validation_stream": VALIDATION_STREAM,
               "screen_blocks": [SCREEN_BLOCKS.start, SCREEN_BLOCKS.stop - 1],
               "confirm_blocks": [CONFIRM_BLOCKS.start, CONFIRM_BLOCKS.stop - 1],
               "candidates_per_run": CANDIDATES}  # fmt: skip
    (out_dir / CHOICE_FILE).write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _evaluator(workers: int | None) -> tuple[block_scoring.Evaluate, block_scoring.MakePool]:
    """The objective and its pools; replaced in the tests."""
    return block_scoring.objective_pools(workers)


def main(argv: list[str] | None = None) -> Path:
    """Run pilot 2, or resume it, and return its directory."""
    import os

    from covsyn.pipeline.runner import run_module

    parser = argparse.ArgumentParser(prog="covsyn-pilot-size", description=main.__doc__)
    parser.add_argument("--noise", type=Path, default=None,
                        help="pilot 1's results/tables/<run-id>, whose choice.json gives S")
    parser.add_argument("--resume", type=Path, default=None,
                        help="continue the pilot in this results/tables/<run-id> directory")
    parser.add_argument("--results-root", type=Path, default=runs.RESULTS_ROOT,
                        help=argparse.SUPPRESS)  # tests
    hardware.add_workers_argument(parser)
    args = parser.parse_args(argv)
    if (args.noise is None) == (args.resume is None):
        parser.error("give --noise for a new pilot or --resume for one begun")
    if args.resume is not None:
        # Absolute, as the optimiser runs from the repository root, without "..", and not
        # resolved, so a linked tables directory keeps its results/ siblings.
        out = Path(os.path.abspath(args.resume))
        if (out.is_symlink() or out.parent.name != "tables"
                or not RUN_ID.fullmatch(out.name)):
            parser.error(f"{args.resume} is not results/tables/<run-id> of a pilot 2, whose "
                         f"name and place give its runs' names and directory")
        try:
            config = json.loads((out / CONFIG_FILE).read_text())
            simulations = config["settings"]["simulations"]
            workers = block_scoring.pilot_workers(args.workers, os.environ,
                                                  config.get("workers"))  # fmt: skip
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            parser.error(f"{out} holds no pilot to resume ({error!r})")
        try:  # the pilot is there; only its record of the workers is at stake
            block_scoring.record_workers(out, workers)
        except PermissionError as error:  # refused: say what was refused
            parser.error(str(error))
        except (OSError, ValueError) as error:  # ValueError: a NaN read, not writable as JSON
            # The error names the file; the message adds what was being done.
            parser.error(f"could not record the workers in {CONFIG_FILE}: {error}")
        root = out.parent.parent
    else:
        try:
            noise = json.loads((args.noise / "choice.json").read_text())
            simulations = noise["chosen_simulations"]
            failed = noise.get("failed_evaluations", 0)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            parser.error(f"{args.noise} holds no pilot 1 choice ({error!r})")
        if failed:
            parser.error(f"pilot 1 in {args.noise} had {failed} failed evaluations, whose "
                         f"1e6 is no cost; its S is not pilot 2's to run")
        if simulations is None:
            parser.error(f"pilot 1 in {args.noise} chose no S: no S met its criterion, which "
                         f"the professor decides first")
        workers = hardware.resolve_workers(args.workers, os.environ, None)
        root = Path(os.path.abspath(args.results_root))
        out = runs.run_directory(runs.make_run_id("pilot-size"), "tables", root, new=True)
        runs.write_config(out, {
            "command": shlex.join(["covsyn-pilot-size", *(sys.argv[1:] if argv is None
                                                          else argv)]),
            "commit": (runs._git("rev-parse", "HEAD") or "").strip() or None,
            "noise": str(args.noise), "simulations": simulations,
            "runs": [s._asdict() | {"generations": s.generations} for s in settings()],
            "budget": BUDGET, "candidates_per_run": CANDIDATES,
            "validation_stream": VALIDATION_STREAM,
            "screen_blocks": [SCREEN_BLOCKS.start, SCREEN_BLOCKS.stop - 1],
            "confirm_blocks": [CONFIRM_BLOCKS.start, CONFIRM_BLOCKS.stop - 1],
        }, rng_seed=None, workers=workers, initial_population=None)
    prefix = out.name.rsplit("_", 1)[0]  # <stamp>_<commit>, shared by its runs
    evaluate, make_pool = _evaluator(workers)
    execute = functools.partial(run_module, keep_fds=())
    summary = run(out, root / "models", prefix, simulations, execute, evaluate, make_pool,
                  workers=workers)
    print(json.dumps(summary, indent=1))
    return out


if __name__ == "__main__":
    main()
