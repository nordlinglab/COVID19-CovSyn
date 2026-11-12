# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Name every run and create its directories in one place (Task 3).

A run is named ``<YYYYMMDD-HHMMSS>_<commit7>[-dirty]_<config-name>`` and writes under the
parent repository's layout: ``results/models/<run-id>/`` (parameters, populations, logs and
``config.json``), ``results/tables/<run-id>/`` and ``results/figures/<run-id>/``.
:func:`run_directory` and :func:`claim_run_directory` are the only places these
directories are created, so a fresh clone needs none of them in advance. ``results/`` is under
the repository root wherever a command starts.

Usage: ``uv run python -m covsyn.runtime.runs CONFIG_NAME`` prints the path of a new run
directory, which launch_phase_d.sh passes to the optimiser (``--run-dir``), which claims it,
and to the chain.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from covsyn.runtime import hardware

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
# What an optimiser run writes; a directory holding any of them already holds a run.
RUN_OUTPUTS = (
    "config.json",
    "bound.txt",
    "firefly_result_first_initial_guess.txt",
    "progress_metrics.csv",
    "firefly_archive.txt",
    "firefly_result.txt",
    "firefly_best.txt",
    "firefly_worst.txt",
    "revalidation",  # written into the run directory by the chain
    "selected",
)
KINDS = ("models", "tables", "figures")
# Lowercase kebab-case: safe in any path, and free of the "_" that separates the id's fields.
CONFIG_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
CONFIG_NAME_MAX = 64


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def make_run_id(
    config_name: str,
    when: datetime | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """The id of a new run: ``<YYYYMMDD-HHMMSS>_<commit7>[-dirty]_<config-name>``.

    Args:
        config_name: Lowercase kebab-case name of the configuration, at most 64 characters.
        when: Start time; now when None.
        commit: Commit the code is at; this checkout's HEAD when None, and ``nocommit`` when
            there is none to name.
        dirty: Whether the checkout has uncommitted changes; read from git when None.

    Raises:
        ValueError: ``config_name`` is not lowercase kebab-case of at most 64 characters.
    """
    if len(config_name) > CONFIG_NAME_MAX or not CONFIG_NAME.fullmatch(config_name):
        raise ValueError(
            f"configuration name must be lowercase kebab-case of at most "
            f"{CONFIG_NAME_MAX} characters, got {config_name!r}"
        )
    when = when or datetime.now()
    if commit is None:
        commit = (_git("rev-parse", "HEAD") or "").strip()
    if dirty is None:
        dirty = bool((_git("status", "--porcelain") or "").strip()) if commit else False
    stamp = commit[:7] if commit else "nocommit"
    return f"{when:%Y%m%d-%H%M%S}_{stamp}{'-dirty' if dirty else ''}_{config_name}"


def run_directory(
    run_id: str, kind: str = "models", root: Path = RESULTS_ROOT, new: bool = False
) -> Path:
    """Create, if needed, and return ``root/kind/run_id``.

    Args:
        run_id: One directory name.
        kind: models, tables or figures.
        root: The results directory.
        new: Require that the directory does not exist yet, as for a fresh run: two runs
            started in the same second with the same configuration get the same id.

    Raises:
        ValueError: ``kind`` is not one of models, tables and figures, or ``run_id`` is not a
            single directory name.
        FileExistsError: ``new`` and the directory exists.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not run_id or run_id in (".", "..") or Path(run_id).name != run_id or "\\" in run_id:
        raise ValueError(f"a run id is one directory name, got {run_id!r}")
    path = Path(root) / kind / run_id
    path.mkdir(parents=True, exist_ok=not new)
    return path


def new_run_path(config_name: str, root: Path = RESULTS_ROOT, **id_fields: Any) -> Path:
    """The models directory a new run would claim; ``id_fields`` go to :func:`make_run_id`.

    Nothing is created: :func:`claim_run_directory` creates and claims it in one step.
    """
    return Path(root) / "models" / make_run_id(config_name, **id_fields)


def new_run_directory(config_name: str, root: Path = RESULTS_ROOT, **id_fields: Any) -> Path:
    """Create and claim the models directory of a new run, as :func:`claim_run_directory` does.

    Raises:
        FileExistsError: Another run with the same id, started in the same second, claimed it.
    """
    return claim_run_directory(new_run_path(config_name, root, **id_fields))


def check_unclaimed(path: Path) -> None:
    """Refuse a directory that holds a run, without creating or claiming anything.

    The optimiser calls it as it starts, so a FIREFLY_RUN holding a run fails within the
    launcher's startup check, and claims the directory only once its inputs have loaded.

    Raises:
        FileExistsError: The directory holds any of RUN_OUTPUTS, a dangling link included.
    """
    path = Path(path)
    found = [n for n in RUN_OUTPUTS if (path / n).exists() or (path / n).is_symlink()]
    if found:
        raise FileExistsError(
            f"{path} already holds a run ({', '.join(found)}); give a new directory "
            "(FIREFLY_RUN or --run-dir), or leave it out for a fresh one"
        )


def claim_run_directory(path: Path) -> Path:
    """Create, if needed, a run directory given by path, refusing one that holds a run already.

    The launcher's FIREFLY_RUN or the optimiser's --run-dir may name any directory, a committed
    run such as results/models/phaseD_run10 included, which a new run must not overwrite.

    Raises:
        FileExistsError: The directory holds any file an optimiser run writes (RUN_OUTPUTS), a
            failed run's config.json included, or another run claimed it at the same time.
    """
    path = Path(path)
    check_unclaimed(path)
    path.mkdir(parents=True, exist_ok=True)
    # Claimed by creating config.json exclusively, which is atomic: of two runs started on one
    # directory at once, the second fails here. It holds valid JSON until write_config fills it.
    try:
        with open(path / "config.json", "x", encoding="utf-8") as claim:
            claim.write('{"claimed": true}\n')
    except FileExistsError:
        raise FileExistsError(
            f"{path} was claimed by another run just now; give a new directory "
            "(FIREFLY_RUN or --run-dir), or leave it out for a fresh one"
        ) from None
    return path


def non_negative_int(text: str) -> int:
    """An argparse type for a seed, checked before any run directory is created."""
    import argparse

    # Plain ASCII digits only: int() would also take '+1', '1_0' and other scripts' digits,
    # which a seed copied from config.json never contains.
    value = int(text) if re.fullmatch(r"[0-9]+", text) else -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a whole number of at least 0, got {text!r}")
    return value


def run_dir_argument(argv: list[str], position: int, usage: str) -> Path:
    """The run directory a script was given at ``argv[position]``.

    Scripts that read a run take its directory as an argument instead of a fixed legacy name,
    so without it this prints the usage and exits with status 2, as argparse does.
    """
    if len(argv) > position and argv[position]:
        return Path(argv[position])
    print(
        f"usage: {usage}\nthe run directory, such as results/models/<run-id>, is required",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return hardware.Machine().sysctl("machdep.cpu.brand_string").strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_versions() -> dict[str, str]:
    """The installed version of numpy and of every package covsyn depends on."""
    names = {"numpy"}
    try:
        requirements = metadata.requires("covsyn") or []
    except metadata.PackageNotFoundError:  # run from a source tree that is not installed
        requirements = []
    for requirement in requirements:
        if "extra ==" not in requirement:
            names.add(re.split(r"[ <>=!~;\[]", requirement, maxsplit=1)[0])
    versions = {}
    for name in sorted(names):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_config(
    run_dir: Path,
    settings: Mapping[str, Any],
    seed: int | None,
    workers: int,
    initial_population: np.ndarray | None,
    lock: Path = REPO_ROOT / "uv.lock",
) -> Path:
    """Write ``run_dir/config.json`` with what it takes to reproduce the run.

    That is every setting, the random seed and the worker count; the CPU model and the detected
    cores; the Python and package versions; and the SHA-256 of ``uv.lock`` and of the initial
    population (float64, C order), None where either does not exist.

    Raises:
        TypeError: A setting JSON cannot hold.
    """
    cores = hardware.performance_core_count()
    population = (
        None
        if initial_population is None
        else _sha256(np.ascontiguousarray(initial_population, dtype=float).tobytes())
    )
    config = {
        "settings": dict(settings),
        "seed": seed,
        "workers": workers,
        "machine": {
            "cpu_model": _cpu_model(),
            "performance_cores": cores.count,
            "core_method": cores.method,
            "platform": platform.platform(),
        },
        "versions": {"python": platform.python_version(), "packages": _package_versions()},
        "uv_lock_sha256": _sha256(lock.read_bytes()) if lock.is_file() else None,
        "initial_population_sha256": population,
    }
    text = json.dumps(config, indent=1, allow_nan=False)  # raises TypeError on a non-JSON value
    path = Path(run_dir) / "config.json"
    path.write_text(text + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    # An optional second argument replaces the results root, for testing.
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_ROOT
    named = new_run_path(sys.argv[1] if len(sys.argv) > 1 else "phase-d", root)
    try:
        print(named.relative_to(Path.cwd()))
    except ValueError:  # started outside the repository
        print(named)
