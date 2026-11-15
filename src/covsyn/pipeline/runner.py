# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""``covsyn-pipeline``: run the Phase D stages in order, and resume an interrupted run (Task 4).

One process runs the stages of scripts/phase_d_chain.sh in order, so neither tmux nor a signal
that the optimiser has finished is needed. Detach it with what ships with both systems::

    nohup uv run covsyn-pipeline --config default > pipeline.out 2>&1 &

Each finished stage appends one JSON line to ``results/models/<run-id>/run.log``: the stage,
its end time, duration, workers, the peak memory of its steps' own processes and of their
largest worker (Task 5), and that of this process. A stage that an earlier run on the same
configuration shows would not fit in the available memory is refused.
``--resume <run-id>`` skips the stages in run.log and continues the optimiser from its last
checkpoint. The configuration is saved with the run as ``pipeline.toml``, so a resumed run
keeps its settings.

Usage:
    uv run covsyn-pipeline --config default
    uv run covsyn-pipeline --resume 20261115-090000_b30c141_phase-d
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from covsyn.pipeline import config as pc
from covsyn.pipeline import stages as ps
from covsyn.runtime import hardware, measure, memory, runs

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
    if measure.ENV in environment:  # Task 5: the step records its own and its workers' peaks
        command[1:1] = ["-m", "covsyn.runtime.measure"]
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


class Reporter:
    """Timestamped progress lines, printed and, given a run's pipeline.log, kept there.

    The run's log is written first, so a terminal that went away loses nothing from it; neither
    a full disk nor a broken or missing stdout may replace a stage's status or error. The memory
    watch reports from its own thread (Task 5): the lock keeps pipeline.log's lines whole, and
    is not held while printing, so a stdout nobody reads cannot stop the watch.
    """

    def __init__(self, log: Path | None = None) -> None:
        """Report into ``log`` as well as to stdout."""
        self.log = log
        self.lock = threading.Lock()
        self.log_failed = False

    def __call__(self, message: str) -> None:
        """Report one line."""
        line = f"[pipeline {datetime.now():%m-%d %H:%M:%S}] {message}\n"
        if self.log is not None:
            with self.lock:  # the once-flag too, as two threads may fail together
                try:
                    with open(self.log, "a", encoding="utf-8") as f:
                        f.write(line)
                except OSError as error:
                    if not self.log_failed:  # said once, not for every line after
                        self.log_failed = True
                        with contextlib.suppress(Exception):
                            sys.stderr.write(f"could not write {self.log}: {error!r}\n")
        if sys.stdout is not None:  # None when started without one
            with contextlib.suppress(OSError, ValueError):  # a broken pipe, a closed stdout
                sys.stdout.write(line)  # in one write call, not print's two
                sys.stdout.flush()


def json_lines(path: Path) -> list[dict]:
    """The JSON objects of a JSON-lines file; a line cut off by a kill, or no object, is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    lines = []
    for text in path.read_text(errors="replace").splitlines():
        try:
            line = json.loads(text)
        except ValueError:
            continue
        if isinstance(line, dict):
            lines.append(line)
    return lines


def finished_stages(run_dir: Path) -> list[str]:
    """The stages run.log records as finished; a line cut off by a kill is not one."""
    return [line["stage"] for line in json_lines(Path(run_dir) / RUN_LOG) if "stage" in line]


def _number(value: object) -> bool:
    """A finite number: run.log may hold NaN or Infinity, which json reads as floats."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stage_memory(path: Path) -> dict:
    """A stage's peaks from its steps' measurements: the largest over the steps, or None.

    ``steps_measured`` names the steps that wrote their measurement.
    """
    lines = json_lines(path)

    def largest(key):
        values = [line[key] for line in lines if _number(line.get(key))]
        return max(values) if values else None

    return {
        "parent_peak_mb": largest("parent_peak_mb"),
        "worker_peak_mb": largest("worker_peak_mb"),
        "steps_measured": [str(line.get("step", "")) for line in lines],
    }


def stage_peaks(measurements: Path, watch: memory.MemoryWatch) -> dict:
    """A stage's memory, as run.log records it."""
    return {
        **stage_memory(measurements),
        "pipeline_peak_mb": memory.megabytes(watch.parent_peak_bytes),  # None: not read
        # The plan's once-a-second RSS of the step, the process that owns the pool: for the
        # reader; the memory check uses the step's own measured peak.
        "step_sampled_peak_mb": memory.megabytes(watch.steps_peak_bytes),
        "lowest_available_mb": memory.megabytes(watch.lowest_available),
    }


class Estimate(NamedTuple):
    """An earlier run's peaks in MB for a stage, and which run they come from.

    ``pooled``: its steps spawned a pool, measured as a worker peak above 0.
    """

    step: float
    worker: float
    run: str
    pooled: bool

    def need(self, workers: int) -> float:
        """The memory in MB this stage would need with ``workers``."""
        return self.step + workers * self.worker


def _lines(run: Path, *parts: str) -> list[dict]:
    """A run's JSON-lines file; nothing if it is not a readable file.

    Nothing too if the run, the file or a folder on the way is a link (Task 4: a dead share
    must not stall or stop the run).
    """
    path = run.joinpath(*parts)
    try:
        if run.is_symlink() or any(
            run.joinpath(*parts[: k + 1]).is_symlink() for k in range(len(parts))
        ):
            return []
        if not path.is_file():
            return []
        return json_lines(path)
    except OSError:
        return []


def other_runs(run_dir: Path) -> list[Path]:
    """The other run directories beside run_dir on its configuration.

    Links are skipped without being followed, a run's linked pipeline.toml too, as is anything
    but a directory: a link in results/models to a dead share must not stall the run (Task 4).
    A run whose configuration cannot be read is skipped alone, so the others still count.

    Args:
        run_dir: This run's directory, results/models/<run-id>.

    Returns:
        The other runs' directories whose pipeline.toml equals this run's.

    Raises:
        OSError: results/models or this run's own pipeline.toml cannot be read.
    """
    saved = (run_dir / SAVED_CONFIG).read_text(encoding="utf-8")
    found = []
    with os.scandir(run_dir.parent) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):  # False for a link too
                continue
            other = Path(entry.path)
            config = other / SAVED_CONFIG
            try:
                if other.name == run_dir.name or config.is_symlink() or not config.is_file():
                    continue
                if config.read_text(encoding="utf-8", errors="replace") == saved:
                    found.append(other)
            except OSError:  # unreadable, or gone meanwhile: this run alone is skipped
                continue
    return found


def finished_at(line: Mapping) -> datetime | None:
    """When a run.log line's stage finished, as an aware time; None if missing or invalid.

    A time without an offset is taken as this machine's local time: the order of such times
    from machines in other time zones cannot be recovered.
    """
    value = line.get("finished")
    if not isinstance(value, str):
        return None
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return None
    return when.astimezone() if when.tzinfo is None else when


def complete(line: Mapping) -> bool:
    """Whether a run.log line measured its whole stage, so the memory check may use it.

    Each step the stage ran wrote its measurement, matched by name, not only by count (a step
    without a pool has no worker peak), and the stage ran whole: an optimiser resumed from a
    checkpoint measures only the rest.
    """
    steps, measured = line.get("steps"), line.get("steps_measured")
    if not (isinstance(steps, list) and isinstance(measured, list) and steps):
        return False
    return (
        sorted(map(str, steps)) == sorted(map(str, measured))
        and _number(line.get("parent_peak_mb"))
        and line.get("partial") is False
    )


def histories(others: Sequence[Path]) -> dict[Path, list[dict]]:
    """Each other run's run.log lines, read once for the memory check (see :func:`_lines`)."""
    return {other: _lines(other, RUN_LOG) for other in others}


def earlier_estimate(
    run_dir: Path,
    stage: str,
    others: Sequence[Path] | None = None,
    warn: Callable[[str], None] | None = None,
    history: Mapping[Path, list[dict]] | None = None,
) -> list[Estimate]:
    """The stage's latest complete earlier measurements on this configuration, by kind.

    Among another run's run.log lines that measured the whole stage (:func:`complete`), the
    latest finished, by finished time then run id, of each kind (all of them where one run
    finished the stage more than once at that time): with a pool (a worker peak above
    0), and without (its steps ran serially, under a memory cap say). This run may run either
    way, and a serial run does all its work in the step, so both count and the larger need
    decides. A later incomplete line is skipped and an earlier complete one used; stopped
    attempts are not used. The latest line of a kind keeps counting however old it is, until
    a later one of that kind replaces it: the conservative outcome. The plan names "an earlier
    run on the same configuration"; the latest complete one of each kind keeps the rule simple
    to check.

    Args:
        run_dir: This run's directory, itself never used.
        stage: The stage to be checked.
        others: The other runs, as :func:`other_runs` lists them; listed here when None.
        warn: Told of each complete line skipped for want of a valid finished time.
        history: Each other run's run.log lines (:func:`histories`); read here when None.

    Returns:
        The latest complete measurements of each kind (more than one only where one run
        finished the stage twice at one time); empty when no earlier run measured the stage
        whole.
    """
    others = other_runs(run_dir) if others is None else others
    history = histories(others) if history is None else history
    latest: dict[bool, tuple] = {}  # kind -> ((time, run), the estimates at that time)
    for other in others:
        for line in history.get(other, []):
            if line.get("stage") != stage or not complete(line):
                continue
            when = finished_at(line)
            if when is None:
                if warn is not None:
                    warn(f"WARNING: {other.name}'s {stage} has no valid finished time; not used")
                continue
            worker = line.get("worker_peak_mb")
            pooled = _number(worker) and worker > 0
            estimate = Estimate(
                line["parent_peak_mb"], worker if pooled else 0.0, other.name, pooled
            )
            # The latest, by time then run id; lines of one run at one time all count, as which
            # needs more depends on this run's workers.
            key = (when, other.name)
            if pooled not in latest or key > latest[pooled][0]:
                latest[pooled] = (key, [estimate])
            elif key == latest[pooled][0]:
                latest[pooled][1].append(estimate)
    return [estimate for _, estimates in latest.values() for estimate in estimates]


def memory_problem(
    run_dir: Path,
    stage: str,
    workers: int,
    others: Sequence[Path] | None = None,
    report: Callable[[str], None] | None = None,
    history: Mapping[Path, list[dict]] | None = None,
) -> str | None:
    """Why the stage would not fit in the available memory, or None (Task 5).

    A conservative check at the stage's start, not a guarantee. The need is estimated from the
    latest complete earlier measurements on this configuration, with a pool and without
    (:func:`earlier_estimate`; several of a kind where one run finished the stage twice at one
    time), each as the step's own peak plus this run's workers x the peak per worker; the
    largest decides. The step's peak is a
    conservative addition to the plan's formula. The sum is not the peaks' simultaneous total,
    and data, code, start method, worker count or other processes may change the next run's
    need; stopped attempts are not used, so a stage once stopped for want of memory may run out
    again. On Linux, a forked worker's RSS also counts pages it shares with its parent, which
    the estimate then counts once per worker, so a stage that would fit may be refused.

    Args:
        run_dir: This run's directory.
        stage: The stage about to start.
        workers: This run's workers.
        others: The other runs, as :func:`other_runs` lists them.
        report: Told when no earlier measurement can be used, and of lines skipped for want
            of a valid finished time.
        history: Each other run's run.log lines, read once (:func:`histories`).

    Returns:
        The reason to refuse the stage, or None when it fits or cannot be checked.
    """
    others = other_runs(run_dir) if others is None else others
    history = histories(others) if history is None else history
    estimates = earlier_estimate(run_dir, stage, others, report, history)
    if not estimates:
        if report is not None:
            report(
                f"memory check for stage {stage}: no complete earlier measurement on this "
                "configuration, so its need cannot be estimated"
            )
        return None
    # The larger need decides; at an equal need, the run id, the kind, then the peaks, so the
    # message does not depend on the order the runs or their lines are listed in.
    estimate = max(estimates, key=lambda e: (e.need(workers), e.run, e.pooled, e.step, e.worker))
    needed = estimate.need(workers)
    available = memory.available_bytes() / memory.MB
    if needed <= available:
        return None
    if estimate.pooled:
        source = (
            f"{estimate.step:.0f} MB for its steps and {workers} workers x "
            f"{estimate.worker:.0f} MB, the peaks of run {estimate.run} on this configuration"
        )
    else:
        source = f"its steps' peak with no pool, in run {estimate.run} on this configuration"
    # Fewer workers can help only with workers to spare, and only if every estimate would fit
    # with one: a serial need, and a step's own peak, do not shrink with the worker count.
    if workers > 1 and all(e.need(1) <= available for e in estimates):
        advice = "free memory, or try fewer workers, which need not save memory in proportion"
    else:
        advice = "free memory: fewer workers will not make it fit"
    return (
        f"stage {stage} needs about {needed:.0f} MB ({source}), but only {available:.0f} MB "
        f"is available; {advice}"
    )


def record_stage(
    run_dir: Path,
    stage: str,
    seconds: float,
    workers: int,
    warnings: Sequence[str] = (),
    peaks: Mapping | None = None,
) -> None:
    """Append one line to run.log for a finished stage.

    ``peaks``: the largest peak of the stage's steps' own processes and of their workers, None
    where no step measured one (:func:`stage_memory`); the steps' RSS the pipeline sampled once
    a second (step_sampled_peak_mb), and its own (pipeline_peak_mb), None where no sample
    caught one; and the least available memory during the stage.
    """
    peaks = peaks or {}
    line = {
        "stage": stage,
        # With its offset, so runs from machines in other time zones keep their order.
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(seconds, 3),
        "workers": workers,
        "parent_peak_mb": peaks.get("parent_peak_mb"),
        "worker_peak_mb": peaks.get("worker_peak_mb"),
        "lowest_available_mb": peaks.get("lowest_available_mb"),
        "pipeline_peak_mb": peaks.get("pipeline_peak_mb"),
        "step_sampled_peak_mb": peaks.get("step_sampled_peak_mb"),
        # Whether the memory check may use this line (complete): the names of the steps the
        # stage ran and of those that wrote their measurement, and whether it ran only the
        # rest of its work.
        "steps": peaks.get("steps"),
        "steps_measured": peaks.get("steps_measured"),
        "partial": peaks.get("partial", False),
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


class NoPipelineRun(FileNotFoundError):
    """The directory holds no pipeline run to resume."""


class RunInUse(RuntimeError):
    """Another pipeline, or a step one left running, holds the run's lock."""


def standard_run_dir(run_dir: Path | str, results_root: Path) -> Path:
    """The run directory, if it is results_root/models/<run-id> written out; else refused.

    The plan names runs results/models/<run-id> and resumes them by run id (Tasks 3 and 4).
    A run named otherwise, or a link there to other storage, could put its tables and figures
    in another place, so nothing but this one name is accepted, and no other run is looked at.

    Raises:
        ValueError: Not results_root/models/<run-id>, a step through "." or "..", or a link.
    """
    text = str(run_dir)
    if any(part in (".", "..") for part in text.replace("\\", "/").split("/")):
        raise ValueError(f"{text!r} names no run: give its run id, or results/models/<run id>")
    path = Path(os.path.abspath(text))
    models = Path(os.path.abspath(results_root)) / "models"
    # A link above results/ (macOS's /tmp, a linked home) may name it, as RESULTS_ROOT is
    # resolved; but the path must still read .../results/models/<id>, so the storage path of a
    # linked results/models is refused like any other.
    named = (path.parent.name, path.parent.parent.name) == ("models", models.parent.name)
    if not named or os.path.realpath(path.parent) != os.path.realpath(models):
        raise ValueError(f"{text} is not results/models/<run id> ({models}); give the run id")
    if path.is_symlink():
        raise ValueError(
            f"{path} is a link; a run behind a link is not supported, as its "
            "tables and figures could go elsewhere"
        )
    return models / path.name  # one spelling, so its tables and figures have one place


def resolve_resume(given: str, results_root: Path) -> Path:
    """The run --resume names: a run id, or results/models/<run-id> written out.

    Raises:
        ValueError: Anything else, ".", a storage path or a subdirectory included.
    """
    if not given or given in (".", "..") or any(c.isspace() for c in given):
        raise ValueError(f"{given!r} names no run: give its run id")
    if "/" not in given and "\\" not in given:
        return standard_run_dir(Path(results_root) / "models" / given, results_root)
    if not os.path.isabs(given):
        raise ValueError(
            f"{given!r} names no run: give its run id, or results/models/<run id> as a whole path"
        )
    return standard_run_dir(given, results_root)


def run(
    config: pc.PipelineConfig | None,
    run_dir: Path,
    execute: Execute | None = None,
    resume: bool = False,
    results_root: Path | None = None,
) -> int:
    """Run the stages not yet finished; 0 when all finished, 1 when a stage stopped the run.

    Args:
        config: The settings of a new run; None when resuming, which reads the run's own.
        run_dir: The run's models directory.
        execute: Runs one step; :func:`run_module` by default, a recorder in tests.
        resume: Continue the run in ``run_dir`` instead of starting one there.
        results_root: The results directory ``run_dir`` must be in; RESULTS_ROOT by default,
            a scratch one in tests.

    Raises:
        FileNotFoundError: Resuming a directory that holds no pipeline run.
        FileExistsError: Starting a run in a directory that holds one.
        RuntimeError: Another pipeline, or a step one left running, holds the run.
        ValueError: ``run_dir`` is not results_root/models/<run-id> (see standard_run_dir).
    """
    # Absolute, as steps run from the repository root; not resolved, so a symlinked models
    # directory keeps its results/ siblings.
    run_dir = standard_run_dir(run_dir, runs.RESULTS_ROOT if results_root is None else results_root)
    if resume:
        saved = run_dir / SAVED_CONFIG
        if not saved.is_file():
            raise NoPipelineRun(f"{run_dir} holds no pipeline run to resume ({SAVED_CONFIG})")
        config = pc.parse(saved.read_text(encoding="utf-8"), str(saved))
    else:
        runs.check_unclaimed(run_dir)  # the optimiser claims it; this fails before anything runs
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / SAVED_CONFIG).write_text(pc.to_toml(config), encoding="utf-8")
    lock = acquire_lock(run_dir)
    try:
        if execute is None:
            execute = functools.partial(run_module, keep_fds=(lock,))
        report = Reporter(run_dir / PIPELINE_LOG)
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
        raise RunInUse(
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
    others = None  # listed and read once, when a stage first needs the memory check
    history: dict[Path, list[dict]] = {}
    report(f"run directory {run_dir}" + (f"; finished: {', '.join(sorted(done))}" if done else ""))
    for name, function in ps.STAGES:
        if name in done:
            continue
        # Task 5: the stage's steps append their peaks here, one file per attempt, so a resumed
        # stage counts only its own and a failed attempt's stay where pipeline.log says.
        measurements = layout.logs / f"memory_{name}_{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
        environment = {**env, measure.ENV: str(measurements)}
        ctx = ps.Context(config, layout, step)
        report(f"stage {name}")
        start = time.monotonic()
        # The pipeline's own RSS and the available memory, once a second (Task 5). A warning is
        # reported at once: a stage killed for want of memory may never reach its end.
        watch = memory.MemoryWatch(warn=lambda message: report(f"WARNING: {message}"), steps=True)
        finished = False
        try:
            if others is None:
                try:
                    others = other_runs(run_dir)
                except OSError as error:  # the check goes without them, as a new run would
                    report(f"WARNING: could not list the other runs ({error!r})")
                    others = []
                history = histories(others)
            # A stage the configuration turns off runs no step: nothing to check.
            off = ps.turned_off(name, config)
            problem = (
                None if off else memory_problem(run_dir, name, workers, others, report, history)
            )
            if problem is not None:
                raise ps.StageFailed(problem)
            with watch:
                function(ctx, resume) if name == "optimise" else function(ctx)
            finished = True
        except ps.StageFailed as error:
            hint = (
                f" Resume with: covsyn-pipeline --resume {resume_name(run_dir)}"
                if error.resumable
                else ""
            )
            report(f"ERROR: {error}; stopping.{hint}")
            return 1
        finally:
            for warning in ctx.warnings:
                report(f"WARNING: {warning}")
            # Whatever stopped the stage, unless the memory check refused it before it started.
            if not finished and watch.started:
                steps = (
                    f"the steps' peaks are in {measurements}"
                    if measurements.exists()
                    else "no step reported its peaks"
                )

                report(
                    f"memory in stage {name}: lowest available "
                    f"{memory.mb_text(watch.lowest_available)}, this process at most "
                    f"{memory.mb_text(watch.parent_peak_bytes)}, a step at most "
                    f"{memory.mb_text(watch.steps_peak_bytes)} (sampled once a second); {steps}"
                )
        peaks = {**stage_peaks(measurements, watch), "steps": ctx.steps, "partial": ctx.partial}
        # The watch's warning for run.log, once, however often its report was tried, and even
        # if no report got through: the memory was low whatever the output did.
        warnings = [*ctx.warnings, *([] if watch.warning is None else [watch.warning])]
        record_stage(run_dir, name, time.monotonic() - start, workers, warnings, peaks)
        measurements.unlink(missing_ok=True)  # in run.log now; a stopped attempt's stays
    report("ALL DONE")
    return 0


def resume_name(run_dir: Path) -> str:
    """What to give --resume: the run id, which names results/models/<run-id>."""
    return run_dir.name


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
        "--resume",
        metavar="RUN_ID",
        help="continue results/models/<RUN_ID>; give the run id, or that path in full",
    )
    args = parser.parse_args(argv)
    if args.resume is not None:  # '' too: an unset $RUN_ID must not start a new run
        try:
            run_dir = resolve_resume(args.resume, runs.RESULTS_ROOT)
            if not run_dir.is_dir():
                raise ValueError(f"no run {args.resume!r} in {run_dir.parent}")
        except ValueError as error:
            parser.error(str(error))
        try:
            return run(None, run_dir, resume=True)
        except (NoPipelineRun, RunInUse) as error:  # what the command can explain; not a stage
            print(f"covsyn-pipeline: {error}", file=sys.stderr)
            return 1
    config = pc.load(args.config)
    return run(config, runs.new_run_path(config.config_name, runs.RESULTS_ROOT))


if __name__ == "__main__":
    sys.exit(main())
