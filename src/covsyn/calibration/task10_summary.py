# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 10's summary of the five production runs (items 1 and 2), after they finish.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 10: the main run's
cost_uncertainty.csv (item 1), and parameter_uncertainty.csv over the five selected vectors with
the range of the main run's indistinguishable set (item 2, parameter_uncertainty.py).

Each run is a covsyn-pipeline production run. From its directory the summary reads:
revalidation/cost_uncertainty.csv, its archive candidates' costs on its 30 validation blocks;
firefly_archive.txt, those candidates at full precision, ranked as the revalidation ranked them,
which must round to revalidation/firefly_best.txt's 7-decimal rows; selected/selection.csv, the
checklist's choice among them, which must round to selected/firefly_best.txt; bound.txt, its
bounds; pipeline.toml, its streams, S and configuration; config.json, the stopping rule it ran
by; and run.log, the optimiser's stop and each stage's time.

It refuses runs that cannot be summarised together: not five, a run twice or in both lists,
another S or other bounds, a stream two runs share, a list of two configurations, runs that
differ in a setting both lists share (SHARED), a runner-up of production's configuration or off
its pair's streams, production runs whose n, alpha_0, alpha_end or G_plan is not pilot 2's
results/tables/<pilot>/choice.json (beside results/models/; required without --competitor too,
so a copy of the runs elsewhere needs it as well), or a run whose config.json does not record
monitoring.STOP_RULE.

Decision of 2026-11-26: pilot 2's runner-up (--competitor) runs on the same five pairs of
optimisation and validation streams, and each run's best validation mean is compared with its
pair's: comparison.csv, and a paired t interval over the five differences in summary.json.

Usage (repository root):
    covsyn-task10-summary OUT_DIR --main I --runs RUN_DIR x5 [--competitor RUN_DIR x5]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from covsyn.calibration import monitoring, revalidation
from covsyn.calibration import parameter_uncertainty as pu
from covsyn.calibration.monitoring import EVENT as STOP_EVENT
from covsyn.pipeline import config as pc
from covsyn.runtime import runs as rt

COST_TABLE = Path("revalidation") / "cost_uncertainty.csv"
CANDIDATES = Path("revalidation") / "firefly_best.txt"
SELECTED = Path("selected") / "firefly_best.txt"
SELECTION = Path("selected") / "selection.csv"
ARCHIVE = Path("firefly_archive.txt")
# write_best_file's 7 decimals round by at most 5e-8; a little more allows for the parse.
ROUNDING = 6e-8


def _vectors(path: Path) -> np.ndarray:
    """A firefly_best.txt's rows (number, params..., cost) as their params, one per row."""
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing")
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] < 3:
        raise ValueError(f"{path}: not a firefly_best.txt (number, params..., cost)")
    return rows


def candidates(run: Path) -> np.ndarray:
    """The run's scored candidates at full precision, by candidate number.

    revalidation/firefly_best.txt keeps 7 decimals, which on a parameter of bound width 8e-6
    is a step of about 1% of it; the candidates are the archive's best, as the revalidation
    ranked them (revalidation.ranked_archive), whose file keeps every digit.

    Raises:
        ValueError: The archive's ranking does not round to firefly_best.txt's rows.
    """
    rounded = _vectors(Path(run) / CANDIDATES)
    # write_best_file writes a lone candidate twice, so the report scripts read a matrix.
    if len(rounded) == 2 and np.array_equal(rounded[0], rounded[1]):
        rounded = rounded[:1]
    path = Path(run) / ARCHIVE
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing")
    full = revalidation.ranked_archive(path, len(rounded))[0]
    if (
        full.shape != rounded[:, 1:-1].shape
        or not np.array_equal(rounded[:, 0], np.arange(len(rounded)))
        or not np.allclose(full, rounded[:, 1:-1], rtol=0, atol=ROUNDING)
    ):
        raise ValueError(f"{run}: the archive's best do not round to {CANDIDATES}'s rows")
    return full


def selected_vector(run: Path) -> np.ndarray:
    """The checklist's choice at full precision: selection.csv's chosen candidate.

    Raises:
        ValueError: Not one chosen candidate, or one that does not round to
            selected/firefly_best.txt's vector.
    """
    path = Path(run) / SELECTION
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing")
    with open(path, newline="", encoding="utf-8") as f:
        chosen = [int(r["candidate"]) for r in csv.DictReader(f) if _true(r["chosen"])]
    if len(chosen) != 1:
        raise ValueError(f"{path}: {len(chosen)} candidates chosen, not one")
    full = candidates(run)
    if not 0 <= chosen[0] < len(full):
        raise ValueError(f"{path}: candidate {chosen[0]} is not among the {len(full)} scored")
    vector = full[chosen[0]]
    if not np.allclose(vector, _vectors(Path(run) / SELECTED)[0, 1:-1], rtol=0, atol=ROUNDING):
        raise ValueError(f"{run}: candidate {chosen[0]} does not round to {SELECTED}")
    return vector


def bounds(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """bound.txt's lower and upper rows."""
    path = Path(run) / "bound.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing")
    rows = np.loadtxt(path, ndmin=2)
    if len(rows) != 2:
        raise ValueError(f"{path}: {len(rows)} rows, not a lower and an upper")
    return rows[0], rows[1]


def _true(text: str) -> bool:
    if text not in ("True", "False"):
        raise ValueError(f"not True or False: {text!r}")
    return text == "True"


def cost_rows(run: Path) -> list[dict[str, Any]]:
    """cost_uncertainty.csv's rows, typed; exactly one is the best.

    Raises:
        ValueError: No candidates, or not exactly one best.
    """
    path = Path(run) / COST_TABLE
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{"candidate": int(r["candidate"]), "mean": float(r["mean"]), "se": float(r["se"]),
                 "indistinguishable": _true(r["indistinguishable"]), "best": _true(r["best"])}
                for r in csv.DictReader(f)]  # fmt: skip
    if not rows:
        raise ValueError(f"{path}: no candidates")
    if sum(r["best"] for r in rows) != 1:
        raise ValueError(f"{path}: {sum(r['best'] for r in rows)} rows marked best, not one")
    return rows


def best_row(run: Path) -> dict[str, Any]:
    """The cost table's best candidate."""
    return next(r for r in cost_rows(run) if r["best"])


def indistinguishable_set(run: Path) -> np.ndarray:
    """The vectors of the candidates that cannot be told apart from the best, the best too.

    By candidate number, at full precision (see candidates).

    Raises:
        ValueError: A candidate number the revalidation did not score.
    """
    full = candidates(run)
    chosen = []
    for row in cost_rows(run):
        if row["indistinguishable"] or row["best"]:
            if not 0 <= row["candidate"] < len(full):
                raise ValueError(f"{run}: candidate {row['candidate']} is not in {CANDIDATES}")
            chosen.append(full[row["candidate"]])
    return np.array(chosen)


def run_record(run: Path) -> dict[str, Any]:
    """What summary.json keeps of a run: its best cost, set size, stop and stage times.

    run.log holds a stage's line once it finishes; a stage resumed from a checkpoint times only
    its last attempt and is marked partial, so partial_stages lists the times that understate
    their stage.

    Raises:
        ValueError: run.log records no stop of the optimiser, which every production run has.
    """
    best = best_row(run)
    lines = rt.json_lines(Path(run) / "run.log")
    stops = [line.get("reason") for line in lines if line.get("event") == STOP_EVENT]
    if not stops:
        raise ValueError(f"{Path(run) / 'run.log'}: no optimiser stop recorded")
    return {
        "run_id": Path(run).name,
        "best_candidate": best["candidate"],
        "best_mean": best["mean"],
        "best_se": best["se"],
        "set_size": sum(r["indistinguishable"] or r["best"] for r in cost_rows(run)),
        "stop_reason": stops[-1],
        "stage_seconds": {line["stage"]: line.get("seconds") for line in lines if "stage" in line},
        "partial_stages": [
            line["stage"] for line in lines if "stage" in line and line.get("partial")
        ],
    }


def settings(run: Path) -> tuple[int, int, int]:
    """The run's optimisation stream, validation stream and S, from its pipeline.toml."""
    path = Path(run) / "pipeline.toml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing")
    config = pc.load(str(path))
    if config.mode != "production" or config.simulations is None:
        raise ValueError(f"{path}: not a production run")
    # The optimiser's config.json records the rule it stopped by (2026-11-26): a run of the
    # earlier one, such as run 1 that stopped at generation 9, is not production's.
    recorded = Path(run) / "config.json"
    if not recorded.is_file():
        raise FileNotFoundError(f"{recorded}: missing")
    try:
        data = json.loads(recorded.read_text(encoding="utf-8-sig"))
    except ValueError as error:  # JSONDecodeError, or text that is not UTF-8
        raise ValueError(f"{recorded}: not JSON ({error})") from None
    recorded_settings = data.get("settings") if isinstance(data, dict) else None
    if not isinstance(recorded_settings, dict):
        raise ValueError(f"{recorded}: no settings object")
    rule = recorded_settings.get("stop_rule")
    if rule != monitoring.STOP_RULE:
        raise ValueError(
            f"{run} ran with another stopping rule ({rule!r}, not {monitoring.STOP_RULE!r})"
        )
    return config.stream, config.validation_stream, config.simulations


def check_streams(runs: list[Path], competitor: list[Path] | None) -> None:
    """One S throughout, a stream pair per run, and the runner-up's run i on run i's pair.

    The comparison pairs run i with the runner-up's run i on their shared validation blocks;
    out of order, the pairs' differences would hold the blocks' noise too.

    Raises:
        ValueError: Another S, a stream two runs share, or a pair on other streams.
    """
    pairs = [settings(r) for r in runs]
    others = [] if competitor is None else [settings(r) for r in competitor]
    sizes = {s for _, _, s in pairs + others}
    if len(sizes) != 1:
        raise ValueError(f"the runs are at different S: {sorted(sizes)}")
    for index, kind in ((0, "optimisation"), (1, "validation")):
        used = [p[index] for p in pairs]
        for stream in used:
            if used.count(stream) > 1:
                raise ValueError(f"two runs share {kind} stream {stream}")
    for run, other, mine, theirs in zip(runs, competitor or [], pairs, others, strict=False):
        if mine[:2] != theirs[:2]:
            raise ValueError(
                f"{other} runs on streams {theirs[:2]}, not its pair {run}'s "
                f"{mine[:2]}; give the runner-up's runs in the same order"
            )


# The settings pilot 2 chose, which every run of one configuration shares.
CONFIGURATION = ("pop_size", "alpha_0", "alpha_end", "g_plan")
# The settings both configurations share, so their validation costs compare.
SHARED = ("pilot", "revalidation_blocks", "select_top")


def check_configurations(runs: list[Path], competitor: list[Path] | None) -> None:
    """One configuration per list, and the runner-up's another.

    The stream check cannot see a production run swapped with its runner-up pair, which shares
    its streams; their pipeline.toml tells them apart.

    Raises:
        ValueError: A list of two configurations, the runner-up on production's, or runs that
            differ in a setting both share (SHARED).
    """
    keys = []
    for what, group in (("production", runs), ("competitor", competitor or [])):
        if not group:
            continue
        configs = [pc.load(str(Path(r) / "pipeline.toml")) for r in group]
        for name in CONFIGURATION:
            values = {getattr(c, name) for c in configs}
            if len(values) != 1:
                raise ValueError(f"{what}: the runs differ in {name}: {sorted(map(str, values))}")
        keys.append(tuple(getattr(configs[0], name) for name in CONFIGURATION))
    if len(keys) == 2 and keys[0] == keys[1]:
        raise ValueError("the runner-up runs the same configuration as production")
    every = [pc.load(str(Path(r) / "pipeline.toml")) for r in [*runs, *(competitor or [])]]
    for name in SHARED:
        values = {getattr(c, name) for c in every}
        if len(values) != 1:
            raise ValueError(f"the runs differ in {name}: {sorted(map(str, values))}")
    check_pilot_choice(runs)


def check_pilot_choice(runs: list[Path]) -> None:
    """The production runs run pilot 2's choice, which tells --runs from --competitor.

    choice.json is read from results/tables/<pilot>/, beside results/models/<run-id>/.

    Raises:
        ValueError: The production runs' n, alpha_0, alpha_end or G_plan is not pilot 2's, or
            choice.json is not JSON or has no chosen setting.
        FileNotFoundError: choice.json is missing.
    """
    config = pc.load(str(Path(runs[0]) / "pipeline.toml"))
    # abspath, as the pipeline uses (runner.py, stages.py): an absolute run path keeps a linked
    # run directory's own place, and '..' collapses by name, so a path through a link and then
    # '..' is not supported, here or there; a relative path joins the physical working
    # directory, so run from the repository root, as the pipeline is.
    results = Path(os.path.abspath(runs[0])).parent.parent
    path = results / "tables" / str(config.pilot) / "choice.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing, pilot 2's choice that production runs")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as error:  # JSONDecodeError, or text that is not UTF-8
        raise ValueError(f"{path}: not JSON ({error})") from None
    chosen = data.get("chosen") if isinstance(data, dict) else None
    if not isinstance(chosen, dict):
        raise ValueError(f"{path}: no chosen setting")
    for name in ("pop_size", "alpha_0", "alpha_end", "g_plan"):
        if chosen.get(name) != getattr(config, name):
            raise ValueError(
                f"the production runs' {name} is {getattr(config, name)!r}, not "
                f"pilot 2's {chosen.get(name)!r} ({path}); are --runs and "
                f"--competitor swapped?"
            )


def _check_runs(runs: list[Path], what: str) -> None:
    if len(runs) != pu.RUNS:
        raise ValueError(f"{what}: {pu.RUNS} runs are needed, not {len(runs)}")
    if len({Path(r).resolve() for r in runs}) != len(runs):
        raise ValueError(f"{what}: a run is given twice")


def compare(production: list[Path], competitor: list[Path]) -> tuple[list[dict], dict]:
    """Each pair's best validation means, and a paired t interval over the differences.

    difference = competitor - production, so a positive one favours production.
    """
    from scipy import stats

    rows = []
    for i, (a, b) in enumerate(zip(production, competitor, strict=True), start=1):
        cost_a, cost_b = best_row(a)["mean"], best_row(b)["mean"]
        rows.append({"pair": i, "production_run": Path(a).name, "competitor_run": Path(b).name,
                     "production_best_mean": cost_a, "competitor_best_mean": cost_b,
                     "difference": cost_b - cost_a})  # fmt: skip
    d = np.array([r["difference"] for r in rows])
    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    mean = float(d.mean())
    return rows, {
        "production_mean": float(np.mean([r["production_best_mean"] for r in rows])),
        "competitor_mean": float(np.mean([r["competitor_best_mean"] for r in rows])),
        "mean_difference": mean, "ci_low": mean - half, "ci_high": mean + half,
        "production_lower_in": int((d > 0).sum()),
    }  # fmt: skip


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Inside a new output, where a killed summary leaves it; the output is then incomplete.
STAGING = ".partial"


def _remove(out: Path, staging: Path, moved: list[Path]) -> bool:
    """Remove a failed summary's out, the staging directory, the incomplete mark, last.

    A moved file that will not go leaves the mark beside it, so out still reads as incomplete.

    Returns:
        Whether anything was left in out.
    """
    left = False
    for path in moved:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            left = True
    if not left:
        shutil.rmtree(staging, ignore_errors=True)
        with contextlib.suppress(OSError):
            out.rmdir()
    return left


def summarise(
    runs: list[Path], main: int, out: Path, competitor: list[Path] | None = None
) -> dict[str, Any]:
    """Write parameter_uncertainty.csv, the cost tables, summary.json and comparison.csv.

    Args:
        runs: The five production run directories, in stream order.
        main: The main run, 1 to 5, named before the runs started.
        out: The directory to write, which must not exist yet.
        competitor: The runner-up's five runs, pair by pair with ``runs``.

    Raises:
        ValueError: Runs that cannot be summarised together (see the module docstring),
            main outside 1-5, or a malformed file (see candidates, selected_vector, cost_rows,
            indistinguishable_set and run_record).
        FileNotFoundError: A run's file or pilot 2's choice.json is missing, or ``out`` or a
            directory above it links to nothing; the message names it.
        FileExistsError: ``out`` exists or a directory above it is a file.
    """
    runs = [Path(r) for r in runs]
    _check_runs(runs, "production")
    if competitor is not None:
        competitor = [Path(r) for r in competitor]
        _check_runs(competitor, "competitor")
        if {r.resolve() for r in runs} & {r.resolve() for r in competitor}:
            raise ValueError("a run is given as both a production run and the runner-up's")
    if not 1 <= main <= pu.RUNS:
        raise ValueError(f"the main run must be 1 to {pu.RUNS}, not {main}")
    check_streams(runs, competitor)
    check_configurations(runs, competitor)
    lower, upper = bounds(runs[0])
    for run in [*runs[1:], *(competitor or [])]:  # one search space for every run
        other = bounds(run)
        if not (np.array_equal(other[0], lower) and np.array_equal(other[1], upper)):
            raise ValueError(f"{run} has other bounds than {runs[0]}")
    selected = np.array([selected_vector(r) for r in runs])
    table = pu.parameter_table(selected, lower, upper, indistinguishable_set(runs[main - 1]))
    records = [run_record(r) for r in runs]
    comparison = None
    if competitor is not None:
        comparison = (*compare(runs, competitor), [run_record(r) for r in competitor])
    out = Path(out)
    for level in (out, *out.parents):  # a link to an unmounted disk, here or above
        if level.is_symlink() and not level.exists():
            raise FileNotFoundError(f"{level} links to {os.readlink(level)}, which does not exist")
    if out.exists():
        raise FileExistsError(f"{out} exists; give a new directory")
    summary: dict[str, Any] = {
        "main": main, "runs": records,
        "not_identified": sum(bool(row["not_identified"]) for row in table),
        "fixed": sum(bool(row["fixed"]) for row in table), "parameters": len(table),
    }  # fmt: skip
    # out must be new (decision of 2026-11-26): the mkdir that makes it is the claim, so no
    # other summary writes there, and a failure removes it whole.
    for level in out.parents:
        if level.exists() and not level.is_dir():
            raise FileExistsError(f"{level}, above {out}, is not a directory")
    made = [level for level in out.parents if not level.exists()]  # deepest first
    staging = out / STAGING  # files move up only once all are written
    moved: list[Path] = []
    claimed = False
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.mkdir()
        except FileExistsError:  # after the check above: likely writing it now, not killed
            raise FileExistsError(
                f"{out} exists: another summary made it as this one started; give a new directory"
            ) from None
        claimed = True
        staging.mkdir()
        pu.write_table(staging / "parameter_uncertainty.csv", table)
        shutil.copyfile(runs[main - 1] / COST_TABLE, staging / "cost_uncertainty.csv")
        for i, run in enumerate(runs, start=1):
            shutil.copyfile(run / COST_TABLE, staging / f"cost_uncertainty_run{i}.csv")
        if comparison is not None:
            rows, summary["comparison"], summary["competitor_runs"] = comparison
            _write(staging / "comparison.csv", rows)
        text = json.dumps(summary, indent=1) + "\n"
        (staging / "summary.json").write_text(text, encoding="utf-8")
        for path in sorted(staging.iterdir()):
            moved.append(out / path.name)  # before the move, so an interrupt after it is cleaned
            path.rename(out / path.name)
        staging.rmdir()
    except BaseException as error:
        # .partial goes only once every file is in place: then the output stands.
        if claimed and not (moved and not staging.exists()) and _remove(out, staging, moved):
            error.add_note(f"{out} is left incomplete")
        # The parents made for out, if empty; another summary into a sibling that made one its
        # parent at the same moment may then fail to make its own out, a race left as it is.
        for level in made:
            with contextlib.suppress(OSError):
                level.rmdir()
        raise
    return summary


def main(argv: list[str] | None = None) -> None:
    """``covsyn-task10-summary``: summarise the five production runs."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("out", type=Path, help="a new directory, such as results/tables/<id>")
    parser.add_argument("--main", type=int, required=True, help="the main run, 1 to 5")
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="the five production run directories, in stream order",
    )
    parser.add_argument(
        "--competitor",
        type=Path,
        nargs="+",
        help="the runner-up's five runs, on the same streams in the same order",
    )
    args = parser.parse_args(argv)
    try:
        summary = summarise(args.runs, args.main, args.out, args.competitor)
    except (ValueError, OSError) as error:
        message = "\n".join([f"covsyn-task10-summary: {error}", *getattr(error, "__notes__", [])])
        raise SystemExit(message) from None
    print(
        f"{summary['parameters']} parameters, {summary['not_identified']} not identified; "
        f"wrote {args.out}"
    )
    if "comparison" in summary:
        c = summary["comparison"]
        print(
            f"competitor - production: {c['mean_difference']:.3f} "
            f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}], production lower in "
            f"{c['production_lower_in']} of {pu.RUNS}"
        )


if __name__ == "__main__":
    main()
