# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""``covsyn-pipeline``: run the Phase D stages in order, and resume an interrupted run (Task 4).

One process runs the stages of scripts/phase_d_chain.sh in order, so neither tmux nor a signal
that the optimiser has finished is needed. Detach it with what ships with both systems::

    nohup uv run covsyn-pipeline --config default > pipeline.out 2>&1 &

Each finished stage appends one JSON line to ``results/models/<run-id>/run.log``: the stage,
its end time, duration, workers, and the peak memory of this process and of the largest step.
``--resume <run-id>`` skips the stages in run.log and continues the optimiser from its last
checkpoint. The configuration is saved with the run as ``pipeline.toml``, so a resumed run
keeps its settings.

Usage:
    uv run covsyn-pipeline --config default
    uv run covsyn-pipeline --resume 20261115-090000_b30c141_phase-d
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from covsyn.pipeline import config as pc
from covsyn.pipeline import stages as ps
from covsyn.runtime import hardware, runs

SAVED_CONFIG = "pipeline.toml"
RUN_LOG = "run.log"
PIPELINE_LOG = "pipeline.log"  # every progress line, warning and error, kept with the run
LOCK = "pipeline.lock"
Execute = Callable[[Sequence, Path | None, bool, Mapping[str, str]], int]


def run_module(
    args: Sequence,
    output: Path | None,
    append: bool,
    env: Mapping[str, str],
    keep_fds: Sequence[int] = (),
) -> int:
    """Run ``python args`` from the repository root, its output into ``output``; its status.

    ``keep_fds`` stay open in the step: the run's lock, so a step left running when the
    pipeline was killed still holds it and a resume cannot start a second one on the run.
    """
    environment = {**os.environ, **env}
    command = [sys.executable, *map(str, args)]
    options = {
        "cwd": runs.REPO_ROOT,
        "env": environment,
        "check": False,
        "pass_fds": tuple(keep_fds),
    }
    if output is None:
        return subprocess.run(command, **options).returncode
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a" if append else "w", encoding="utf-8") as f:
        return subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, **options).returncode


def say(message: str, log: Path | None = None) -> None:
    """A timestamped progress line, printed and, given a run's pipeline.log, kept there."""
    line = f"[pipeline {datetime.now():%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    if log is not None:
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _peak_mb(who: int) -> float:
    """Peak resident memory in MB; ru_maxrss is in bytes on macOS and in KiB on Linux."""
    peak = resource.getrusage(who).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6


def finished_stages(run_dir: Path) -> list[str]:
    """The stages run.log records as finished; a line cut off by a kill is not one."""
    path = Path(run_dir) / RUN_LOG
    if not path.exists():
        return []
    stages = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            stages.append(json.loads(line)["stage"])
        except (ValueError, KeyError, TypeError):
            continue
    return stages


def record_stage(
    run_dir: Path,
    stage: str,
    seconds: float,
    workers: int,
    warnings: Sequence[str] = (),
) -> None:
    """Append one line to run.log for a finished stage.

    The worker peak is the largest of every step run so far (RUSAGE_CHILDREN cannot be reset);
    Task 5 replaces it with each stage's own peak.
    """
    line = {
        "stage": stage,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(seconds, 3),
        "workers": workers,
        "parent_peak_mb": round(_peak_mb(resource.RUSAGE_SELF), 1),
        "worker_peak_mb": round(_peak_mb(resource.RUSAGE_CHILDREN), 1),
        "warnings": list(warnings),
    }
    path = Path(run_dir) / RUN_LOG
    # A kill while writing leaves a line without its newline; start after it, not on it.
    cut_off = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            cut_off = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(("\n" if cut_off else "") + json.dumps(line) + "\n")


def run(
    config: pc.PipelineConfig | None,
    run_dir: Path,
    execute: Execute | None = None,
    resume: bool = False,
) -> int:
    """Run the stages not yet finished; 0 when all finished, 1 when a stage stopped the run.

    Args:
        config: The settings of a new run; None when resuming, which reads the run's own.
        run_dir: The run's models directory.
        execute: Runs one step; :func:`run_module` by default, a recorder in tests.
        resume: Continue the run in ``run_dir`` instead of starting one there.

    Raises:
        FileNotFoundError: Resuming a directory that holds no pipeline run.
        FileExistsError: Starting a run in a directory that holds one.
        RuntimeError: Another pipeline, or a step one left running, holds the run.
    """
    # Absolute, as steps run from the repository root; not resolved, so a symlinked models
    # directory keeps its results/ siblings.
    run_dir = runs.canonical_run_dir(run_dir)  # absolute, and through results/models if linked
    if resume:
        saved = run_dir / SAVED_CONFIG
        if not saved.is_file():
            raise FileNotFoundError(f"{run_dir} holds no pipeline run to resume ({SAVED_CONFIG})")
        config = pc.parse(saved.read_text(encoding="utf-8"), str(saved))
    else:
        runs.check_unclaimed(run_dir)  # the optimiser claims it; this fails before anything runs
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / SAVED_CONFIG).write_text(pc.to_toml(config), encoding="utf-8")
    lock = acquire_lock(run_dir)
    try:
        if execute is None:
            execute = functools.partial(run_module, keep_fds=(lock,))
        report = functools.partial(say, log=run_dir / PIPELINE_LOG)
        return _run_stages(config, run_dir, execute, resume, report)
    finally:
        os.close(lock)  # the lock goes with the last descriptor, a step's included


def acquire_lock(run_dir: Path) -> int:
    """Lock the run and return the lock's descriptor, or refuse.

    Raises:
        RuntimeError: Another pipeline, or a step one left running, holds the lock.
    """
    fd = os.open(Path(run_dir) / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError(
            f"{run_dir} is in use by another pipeline, or by a step one left "
            "running; wait for it, or stop it, before resuming"
        ) from None
    return fd


def _run_stages(
    config: pc.PipelineConfig,
    run_dir: Path,
    execute: Execute,
    resume: bool,
    report: Callable[[str], None],
) -> int:
    env = {} if config.workers is None else {hardware.ENV_VARIABLE: str(config.workers)}
    # The steps' default: the setting, else COVSYN_WORKERS, else the cores (Task 2); each
    # step's own count, after its memory cap, is Task 5's to record.
    workers = hardware.resolve_workers(None, {**os.environ, **env}, None)

    def step(args, output=None, append=False, env=None):
        return execute(args, output, append, {**environment, **(env or {})})

    environment = env
    layout = ps.Layout(run_dir)
    for directory in (layout.tables, layout.figures, layout.logs):
        directory.mkdir(parents=True, exist_ok=True)
    if os.environ.get("VERIFY_WORKERS") or os.environ.get("CPU_CORES"):
        report(
            "WARNING: VERIFY_WORKERS and CPU_CORES are no longer read; set workers in the "
            "configuration or COVSYN_WORKERS"
        )
    done = set(finished_stages(run_dir))
    report(f"run directory {run_dir}" + (f"; finished: {', '.join(sorted(done))}" if done else ""))
    for name, function in ps.STAGES:
        if name in done:
            continue
        ctx = ps.Context(config, layout, step)
        report(f"stage {name}")
        start = time.monotonic()
        try:
            function(ctx, resume) if name == "optimise" else function(ctx)
        except ps.StageFailed as error:
            report(
                f"ERROR: {error}; stopping. Resume with: covsyn-pipeline --resume "
                f"{resume_name(run_dir)}"
            )
            return 1
        finally:
            for warning in ctx.warnings:
                report(f"WARNING: {warning}")
        record_stage(run_dir, name, time.monotonic() - start, workers, ctx.warnings)
    report("ALL DONE")
    return 0


def resume_name(run_dir: Path) -> str:
    """What to give --resume: the run id under results/models, else the whole path."""
    models = runs.RESULTS_ROOT / "models"
    try:
        inside = models.is_dir() and os.path.samefile(run_dir.parent, models)
    except OSError:
        inside = False
    return run_dir.name if inside else str(run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """``covsyn-pipeline``: start a run from a configuration, or resume one."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        default="default",
        help="a packaged configuration, such as default, or a TOML file",
    )
    group.add_argument(
        "--resume", metavar="RUN_ID", help="continue results/models/<RUN_ID>, or a run directory"
    )
    args = parser.parse_args(argv)
    if args.resume:
        given = Path(args.resume)
        run_dir = (
            runs.canonical_run_dir(given)
            if given.is_dir()
            else runs.RESULTS_ROOT / "models" / args.resume
        )
        return run(None, run_dir, resume=True)
    config = pc.load(args.config)
    return run(config, runs.new_run_path(config.config_name, runs.RESULTS_ROOT))


if __name__ == "__main__":
    sys.exit(main())
