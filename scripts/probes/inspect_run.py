# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read a finished Phase D run's own files and print what a review of it asks for.

Every number here is already on disk when the chain finishes. Reading them costs milliseconds,
while re-deriving them costs simulations: after run 22 the per-candidate household contacts and
the name of each candidate's failing check were recomputed with 5,000 simulations and a repeated
checklist, about 35 minutes, when selected/candidates/*/checks.json held both and reads in 1 ms.
This probe exists so that does not happen again.

What the chain keeps, and what this reads:
  phaseD_chain.log                       stage timings and the selection line
  Firefly_result_*/progress_metrics.csv  every measured_* quantity, per generation
  selected/revalidation.csv              per candidate: training cost, validation mean and sd
  selected/candidates/*/checks.json      per candidate: the FULL checklist, not just the counts
  validation_reference/phaseD_checks.json            the selected vector's checklist
  validation_reference/phaseD_checks_independent.json  the same on independent seeds
  phaseD_report.txt                      the standard report

Usage (from a run directory, or with one as the argument):
    uv run python scripts/probes/inspect_run.py [RUN_DIR] [--watch "check name" ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# The checks a review keeps coming back to; --watch adds more.
DEFAULT_WATCH = (
    "contacts per day before onset, household",
    "cases infecting 3 or more",
    "samples of 579 cases, percentile of a largest of 8",
    "offspring dispersion k",
    "R (mean offspring of an index case)",
)


# A check renamed when its statistic changed, mapped to its former name and what that measured,
# so an older run is read under the current name and marked, rather than as a second row that
# is blank on every run (code review 2026-11-09, F4).
FORMER_NAMES = {
    "samples of 579 cases, percentile of a largest of 8":
        ("samples of 579 cases whose largest is at most 8", "older run: P(largest <= 8)"),
}


def watched_value(checks: list[dict], name: str) -> tuple[float | None, str]:
    """The value of a watched check, read under its former name in an older run.

    Returns:
        The value, or None when absent, and a note saying what an older run's value measured,
        or an empty string.
    """
    value = value_of(checks, name)
    if value is None and name in FORMER_NAMES:
        former, note = FORMER_NAMES[name]
        value = value_of(checks, former)
        if value is not None:
            return value, note
    return value, ""


def stage_timings(run: Path) -> list[tuple[str, str]]:
    """The chain's own [chain MM-DD HH:MM:SS] lines, as (timestamp, text)."""
    log = run / "phaseD_chain.log"
    if not log.exists():
        return []
    out = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"\[chain (\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def failing(checks: list[dict]) -> list[str]:
    """Names of the checks that failed, leaving out those with no measurable value.

    A value that could not be measured is stored as None or, by verify_phase_d.check, as NaN
    with ok None; json reads the NaN back as a float, so only ok False counts as a failure.
    """
    return [c["name"] for c in checks
            if c["target"] is not None and c["value"] is not None and c["ok"] is False]


def value_of(checks: list[dict], name: str) -> float | None:
    """The measured value of one check, or None when it is absent."""
    return next((c["value"] for c in checks if c["name"] == name), None)


def read_checks(path: Path) -> list[dict] | None:
    """The checks of one checks.json, or None when the file is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text())["checks"]


def candidate_table(run: Path, watch: tuple[str, ...]) -> list[dict]:
    """One row per candidate the checklist was run on, from its kept checks.json."""
    rows = []
    costs = {}
    reval = run / "selected" / "revalidation.csv"
    if reval.exists():
        with open(reval) as f:
            for r in csv.DictReader(f):
                costs[r["candidate"]] = r
    for d in sorted((run / "selected" / "candidates").glob("candidate_*")):
        checks = read_checks(d / "checks.json")
        if checks is None:
            continue
        index = d.name.removeprefix("candidate_")
        watched = {name: watched_value(checks, name) for name in watch}
        c = costs.get(index, {})
        rows.append({
            "candidate": index,
            "train": c.get("training_cost"),
            "validation": c.get("validation_mean"),
            "sd": c.get("validation_sd"),
            "failed": failing(checks),
            "watch": {name: watched[name][0] for name in watch},
            "notes": {name: watched[name][1] for name in watch},
        })
    return rows


def _fmt(x, width=9, places=4):
    return " " * width if x is None else f"{float(x):>{width}.{places}f}"


def main() -> None:
    """Print the timings, the selection, the per-candidate table and the checklists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", default=".", type=Path)
    parser.add_argument("--watch", action="append", default=[])
    args = parser.parse_args()
    run = args.run
    watch = DEFAULT_WATCH + tuple(args.watch)

    stages = stage_timings(run)
    if stages:
        print(f"=== stages ({stages[0][0]} to {stages[-1][0]}) ===")
        for when, text in stages:
            print(f"  {when}  {text[:96]}")

    rows = candidate_table(run, watch)
    if rows:
        print("\n=== the candidates the checklist was run on ===")
        print(f'  {"candidate":<10} {"train":>9} {"validation":>10} {"sd":>7}  failed')
        for r in rows:
            failed = "; ".join(r["failed"]) or "none"
            print(f'  {r["candidate"]:<10} {_fmt(r["train"])} '
                  f'{_fmt(r["validation"], 10)} {_fmt(r["sd"], 7, 2)}  {failed}')
        print("\n  watched values")
        for name in watch:
            line = "  ".join(f'{r["candidate"]} {_fmt(r["watch"].get(name), 8)}' for r in rows)
            notes = sorted({r["notes"][name] for r in rows if r["notes"].get(name)})
            print(f"    {name[:46]:<46} {line}" + (f'  ({"; ".join(notes)})' if notes else ""))
    else:
        print("\n(no selected/candidates/*/checks.json: an older run, or the chain is unfinished)")

    for tag, label in (("", "selected, training seeds"), ("_independent", "independent seeds")):
        checks = read_checks(run / "validation_reference" / f"phaseD_checks{tag}.json")
        if checks is None:
            continue
        print(f"\n=== checklist: {label} ===")
        print(f'  failed: {"; ".join(failing(checks)) or "none"}')
        within = [c["name"] for c in checks if c.get("within_noise")]
        print(f'  only within the 95% interval: {"; ".join(within) or "none"}')
        for name in watch:
            v, note = watched_value(checks, name)
            if v is not None:
                print(f"    {name[:46]:<46} {_fmt(v)}" + (f"  ({note})" if note else ""))


if __name__ == "__main__":
    main()
