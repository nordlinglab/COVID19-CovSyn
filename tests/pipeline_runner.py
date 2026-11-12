# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the post-optimisation pipeline on a small fixed dataset, for the pipeline regression tests.

``phaseD_chain.sh`` turns a finished firefly run into synthetic data, the acceptance checklist,
the run report, the constraint table and the validation figures.
Moving those scripts can break them silently (finding E71), so this runs every step on a small
dataset built from the run 10 best vector with fixed seeds, and records what each step produced:
a digest of every data file, the text each report script prints, the checklist JSON itself
(compared to a relative PLATFORM_REL_TOL, B91), and the figures each plot script writes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import regression_digest as rd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = "firefly_result/phaseD_run10"
MONTE_CARLO = 40
MODES = ("spread_Taiwan_weight", "taiwan_first_outbreak")


def _run(args: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    return subprocess.run([sys.executable, *args], cwd=REPO_ROOT, env=env, text=True,
                          capture_output=True, check=False, timeout=1800)


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(workdir: Path) -> dict[str, str]:
    """Synthesise both pipeline modes into ``workdir``; return a digest per data file."""
    digests = {}
    for mode in MODES:
        out = workdir / mode
        out.mkdir(parents=True, exist_ok=True)
        done = _run(["-m", "covsyn.model.data_synthesis_main", "--mode", mode, "--monte_carlo_number",
                     str(MONTE_CARLO), "--result_path", str(out), "--workers", "4",
                     "--parameter_path", RUN_DIR], workdir)
        if done.returncode:
            raise RuntimeError(f"data synthesis failed for {mode}:\n{done.stderr[-2000:]}")
        for f in sorted(out.glob("*.npy")):
            digests[f"{mode}/{f.name}"] = file_digest(f)
    return digests


def report_steps(workdir: Path) -> dict[str, list[str]]:
    """Command line of every text-producing step, keyed by a stable name."""
    spread, first = str(workdir / MODES[0]), str(workdir / MODES[1])
    checks = str(workdir / "phaseD_checks.json")
    # Order matters and follows phaseD_chain.sh: measure_age_rr.py writes
    # validation_reference/age_rr.json, which verify_phaseD.py reads.
    return {
        "measure_age_rr": ["-m", "covsyn.validation.measure_age_rr", RUN_DIR, "400"],
        "verify_phaseD": ["-m", "covsyn.validation.verify_phase_d", spread, first, checks],
        "check_constraints": ["-m", "covsyn.validation.check_constraints", spread, first, "--out",
                              str(workdir / "constraint_check")],
        "rr_exact": ["-m", "covsyn.validation.rr_exact", spread],
        "tw_check": ["-m", "covsyn.validation.tw_check", first],
        "measure_days": ["-m", "covsyn.validation.measure_days", spread],
        "show_final": ["-m", "covsyn.validation.show_final", RUN_DIR],
        "check_gap": ["-m", "covsyn.validation.check_gap", RUN_DIR],
        "show_bounds": ["-m", "covsyn.validation.show_bounds"],
    }


def figure_steps(workdir: Path) -> dict[str, list[str]]:
    """Command line of every figure step of phaseD_chain.sh, writing into ``workdir/figures``."""
    spread, first = str(workdir / MODES[0]), str(workdir / MODES[1])
    figs = str(workdir / "figures")
    checks = str(workdir / "phaseD_checks.json")
    return {
        "validate_layers": ["-m", "covsyn.figures.validate_layers", spread, figs, RUN_DIR],
        "validate_infection": ["-m", "covsyn.figures.validate_infection", spread, figs],
        "plot_diagnostics": ["-m", "covsyn.figures.plot_diagnostics", spread, figs],
        "plot_10panel": ["-m", "covsyn.figures.plot_10panel", spread, figs, RUN_DIR],
        "plot_cheng_attackrate": ["-m", "covsyn.figures.plot_cheng_attackrate", spread, figs],
        "plot_vs_notebook_literature": ["-m", "covsyn.figures.plot_vs_notebook_literature", spread, figs,
                                        RUN_DIR],
        "plot_validation": ["-m", "covsyn.figures.plot_validation", RUN_DIR, figs],
        "workplace_sampling_options": ["-m", "covsyn.figures.workplace_sampling_options", spread, figs, RUN_DIR],
        "plot_reality_vs_covsyn": ["-m", "covsyn.figures.plot_reality_vs_covsyn", spread, first, figs],
        "plot_violin_reality_vs_covsyn": ["-m", "covsyn.figures.plot_violin_reality_vs_covsyn", spread, figs],
        "compare_healthcare_municipality": ["-m", "covsyn.figures.compare_healthcare_municipality", spread, figs],
        "plot_todolist923": ["-m", "covsyn.figures.plot_todolist923", spread, RUN_DIR, checks, figs + "/todolist923"],
    }


def run_step(args: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
    """Run one pipeline step from the repository root."""
    return _run(args, workdir)


_ELAPSED = re.compile(r"^\s*\d+(\.\d+)?\s*s\s*$")


def normalise_text(text: str, workdir: Path) -> str:
    """Remove what legitimately differs between runs.

    That is the temporary directory, progress bars, and elapsed-time lines such as the
    ``"  27.7 s"`` that measure_age_rr.py prints; simulated numbers are never touched.
    """
    text = text.replace(str(workdir), "<WORKDIR>")
    return "\n".join(line for line in text.splitlines()
                     if "it/s]" not in line and "s/it]" not in line and not _ELAPSED.match(line))


def normalised_file_digest(path: Path, workdir: Path) -> str:
    """SHA-256 of a text file after replacing the temporary directory, which it may record."""
    return hashlib.sha256(path.read_text(encoding="utf-8").replace(str(workdir), "<WORKDIR>")
                          .encode()).hexdigest()


def _replace_workdir(value: object, workdir: str) -> object:
    if isinstance(value, str):
        return value.replace(workdir, "<WORKDIR>")
    if isinstance(value, dict):
        replaced = {k.replace(workdir, "<WORKDIR>"): _replace_workdir(v, workdir)
                    for k, v in value.items()}
        if len(replaced) != len(value):
            raise ValueError("two keys differ only by the temporary directory")
        return replaced
    if isinstance(value, list):
        return [_replace_workdir(v, workdir) for v in value]
    return value


def normalised_json(path: Path, workdir: Path) -> object:
    """A JSON file's content with the temporary directory replaced in its keys and strings.

    Replaced after parsing, so a path that JSON escapes (non-ASCII, backslashes) is still found.
    """
    return _replace_workdir(json.loads(path.read_text(encoding="utf-8")), str(workdir))


def json_mismatches(want: object, got: object, where: str = "") -> list[str]:
    """Where two JSON values differ beyond normal precision (B91), with both values.

    Floats match to a relative regression_digest.PLATFORM_REL_TOL, with no absolute floor so
    tiny probabilities are still compared relatively, and NaN matches NaN. Everything else must
    be equal and of the same type: integers (counts carry no platform rounding), booleans, text,
    None, keys and lengths, since a verdict or a note that changes is a real change.

    Returns:
        One 'path: want ..., got ...' line per difference.
    """
    path = where or "<root>"
    if isinstance(want, float) and isinstance(got, float):
        same = (math.isnan(want) and math.isnan(got)) or math.isclose(want, got,
                                                                      rel_tol=rd.PLATFORM_REL_TOL)
        return [] if same else [f"{path}: want {want!r}, got {got!r}"]
    if isinstance(want, dict) and isinstance(got, dict):
        # Keys that differ are reported, and the shared keys are still compared, so one slow
        # pipeline run shows every difference.
        differ = sorted(set(want) ^ set(got))
        found = [f"{path}: keys differ, {differ}"] if differ else []
        return found + [m for k in sorted(set(want) & set(got))
                        for m in json_mismatches(want[k], got[k], f"{where}.{k}" if where else k)]
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return [f"{path}: length {len(want)}, got {len(got)}"]
        return [m for i, (a, b) in enumerate(zip(want, got, strict=True))
                for m in json_mismatches(a, b, f"{where}[{i}]")]
    if type(want) is type(got) and want == got:
        return []
    return [f"{path}: want {want!r}, got {got!r}"]
