# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the post-optimisation pipeline on a small fixed dataset, for the pipeline regression tests.

``phaseD_chain.sh`` turns a finished firefly run into synthetic data, the acceptance checklist,
the run report, the constraint table and the validation figures.
Moving those scripts can break them silently (finding E71), so this runs every step on a small
dataset built from the run 10 best vector with fixed seeds, and records what each step produced:
a digest of every data file, the text each report script prints, and the figures each plot
script writes.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = "firefly_result/phaseD_run10"
# Several report scripts read the optimizer output from this fixed name, which
# phaseD_chain.sh creates by copying the run directory onto it.
COMPAT_DIR = "Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200"
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
    shutil.rmtree(REPO_ROOT / COMPAT_DIR, ignore_errors=True)
    shutil.copytree(REPO_ROOT / RUN_DIR, REPO_ROOT / COMPAT_DIR)
    digests = {}
    for mode in MODES:
        out = workdir / mode
        out.mkdir(parents=True, exist_ok=True)
        done = _run(["Data_synthesis_main.py", "--mode", mode, "--monte_carlo_number",
                     str(MONTE_CARLO), "--result_path", str(out), "--cpu_cores", "4",
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
        "measure_age_rr": ["measure_age_rr.py", COMPAT_DIR, "400"],
        "verify_phaseD": ["verify_phaseD.py", spread, first, checks],
        "check_constraints": ["check_constraints.py", spread, first, "--out",
                              str(workdir / "constraint_check")],
        "rr_exact": ["rr_exact.py", spread],
        "tw_check": ["tw_check.py", first],
        "measure_days": ["measure_days.py", spread],
        "show_final": ["show_final.py"],
        "check_gap": ["check_gap.py"],
        "show_bounds": ["show_bounds.py"],
    }


def figure_steps(workdir: Path) -> dict[str, list[str]]:
    """Command line of every figure step of phaseD_chain.sh, writing into ``workdir/figures``."""
    spread, first = str(workdir / MODES[0]), str(workdir / MODES[1])
    figs = str(workdir / "figures")
    checks = str(workdir / "phaseD_checks.json")
    return {
        "validate_layers": ["validate_layers.py", spread, figs],
        "validate_infection": ["validate_infection.py", spread, figs],
        "plot_diagnostics": ["plot_diagnostics.py", spread, figs],
        "plot_10panel": ["plot_10panel.py", spread, figs],
        "plot_cheng_attackrate": ["plot_cheng_attackrate.py", spread, figs],
        "plot_vs_notebook_literature": ["plot_vs_notebook_literature.py", spread, figs],
        "plot_validation": ["plot_validation.py", COMPAT_DIR, figs],
        "workplace_sampling_options": ["workplace_sampling_options.py", spread, figs, COMPAT_DIR],
        "plot_reality_vs_covsyn": ["plot_reality_vs_covsyn.py", spread, first, figs],
        "plot_violin_reality_vs_covsyn": ["plot_violin_reality_vs_covsyn.py", spread, figs],
        "compare_healthcare_municipality": ["compare_healthcare_municipality.py", spread, figs],
        "plot_todolist923": ["plot_todolist923.py", spread, RUN_DIR, checks, figs + "/todolist923"],
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
