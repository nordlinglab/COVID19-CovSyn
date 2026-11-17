# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stages of the pipeline, each the steps of scripts/phase_d_chain.sh it replaces (Task 4).

Every step is an existing module run as ``python -m``, as the chain ran it, so its output is
the chain's; what differs is where it goes. The chain wrote into the working directory
(synthetic_data_results_*, independent_*, phaseD_report.txt, validation_reference/...), so each
run overwrote the last; here a run writes only into its own directories (Task 3):

- ``results/models/<run-id>/``: the optimiser's files, revalidation/, selected/, and the
  simulations under synthetic/ (stream 301) and independent/ (stream 300, Task 7);
- ``results/tables/<run-id>/``: the age risk ratio, both checklists and the run report;
- ``results/figures/<run-id>/``: the validation, todolist923 and Wu 2025 figures.

What stops the run is what stopped the chain: a failed optimiser, revalidation, selection or
synthesis, and a missing reported figure. The other steps only warned there, and warn here.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from covsyn.calibration import seed_streams
from covsyn.pipeline.config import PipelineConfig
from covsyn.runtime import measure, runs

SPREAD_MODE = "spread_Taiwan_weight"
FIRST_MODE = "taiwan_first_outbreak"
CHENG_MODE = "cheng2020"


class StageFailed(RuntimeError):
    """A step the chain stopped on failed, so the run stops after this stage."""

    def __init__(self, message: str, resumable: bool = True) -> None:
        """``resumable`` False: resuming would fail the same way, so no resume is suggested."""
        super().__init__(message)
        self.resumable = resumable


class Layout:
    """Where a run's files go: its models directory and the tables and figures beside it."""

    def __init__(self, run_dir: Path) -> None:
        """Lay out the run in ``run_dir``; outside results/models the run holds all three."""
        # Absolute, as every step runs from the repository root.
        self.run = Path(os.path.abspath(run_dir))
        if self.run.parent.name == "models":
            results = self.run.parent.parent
            self.tables = results / "tables" / self.run.name
            self.figures = results / "figures" / self.run.name
        else:
            self.tables = self.run / "tables"
            self.figures = self.run / "figures"
        self.logs = self.run / "logs"

    def synthetic(self, mode: str) -> Path:
        """The simulations of stream 301, on which the candidate is chosen (Task 7)."""
        return self.run / "synthetic" / mode

    def independent(self, mode: str) -> Path:
        """The simulations of the independent seeds, from which everything is reported (E97)."""
        return self.run / "independent" / mode


class Context:
    """What a stage needs: the configuration, the layout, and a way to run a step."""

    def __init__(self, config: PipelineConfig, layout: Layout, step: Callable[..., int]) -> None:
        """Hold ``config`` and ``layout``; ``step`` runs one module and returns its status."""
        self.config = config
        self.layout = layout
        self._run_step = step
        self.warnings: list[str] = []
        # Task 5: each step this stage ran, as its measurement names it ("-m <module>"), so the
        # memory check can tell a stage whose every step measured; and whether the stage ran
        # only the rest of its work (an optimiser continued from a checkpoint).
        self.steps: list[str] = []
        self.partial = False

    def step(self, args: list, partial: bool = False, **options) -> int:
        """Run one step and return its status.

        ``partial``: the step does only the rest of the stage's work, so the stage's memory
        measurement is not of the whole stage (Task 5); must() and may() pass it on.
        """
        self.partial = self.partial or partial
        args = [str(a) for a in args]
        self.steps.append(measure.step_name(args))
        return self._run_step(args, **options)

    @property
    def candidate(self) -> Path:
        """The directory whose best vector every stage after the selection reads."""
        if not turned_off("select", self.config):
            return self.layout.run / "selected"
        if not turned_off("revalidate", self.config):
            return self.layout.run / "revalidation"
        return self.layout.run

    def must(self, args: list, what: str, resuming: bool = False, **options) -> None:
        """Run a step the chain stopped on.

        ``resuming``: the optimiser continuing a run, whose status runs.NOT_RESUMABLE means
        retrying cannot help; from any other step, that status is an ordinary failure.
        """
        status = self.step(args, **options)
        if resuming and status == runs.NOT_RESUMABLE:
            log = options.get("output")
            raise StageFailed(
                f"{what} failed: the run cannot be resumed (see {log})", resumable=False
            )
        if status != 0:
            raise StageFailed(f"{what} failed")

    def may(self, args: list, what: str, **options) -> None:
        """Run a step the chain only warned about."""
        if self.step(args, **options) != 0:
            self.warnings.append(f"{what} failed")


def optimise(ctx: Context, resume: bool) -> None:
    """The optimiser, into the run directory; continued from its checkpoint when resuming."""
    run = ctx.layout.run
    log = run / "optimiser.log"
    if resume and has_checkpoint(run):
        problem = runs.resume_problem(run)
        if problem is not None:  # the optimiser would refuse too; resuming cannot help
            raise StageFailed(problem, resumable=False)
        args = ["-m", "covsyn.calibration.firefly_optimizer", "--resume", run]
        # It runs only the generations left, so its memory is not the whole stage's.
        ctx.must(args, "the optimiser", resuming=True, partial=True, output=log, append=True)
        return
    if resume:
        # Killed before its first checkpoint (while scoring the initial population): nothing
        # to continue, and the claim it left would refuse a fresh start, so the optimiser's
        # partial output goes and the optimiser starts again.
        discard_optimiser_output(run)
    args = [
        "-m",
        "covsyn.calibration.firefly_optimizer",
        "--mode",
        ctx.config.mode,
        "--run-dir",
        run,
        "--stream",
        str(ctx.config.stream),
    ]
    ctx.must(args, "the optimiser", output=log)


def has_checkpoint(run: Path) -> bool:
    """Whether the optimiser left a finished checkpoint to resume from (not a .tmp one)."""
    return runs.last_checkpoint(run) is not None


# The optimiser's own files, from runs.RUN_OUTPUTS: not the chain's directories after it.
OPTIMISER_OUTPUTS = tuple(
    name for name in runs.RUN_OUTPUTS if name not in ("revalidation", "selected")
)


def discard_optimiser_output(run: Path) -> None:
    """Remove what an optimiser that never reached a checkpoint left in the run directory."""
    for name in OPTIMISER_OUTPUTS:
        path = Path(run) / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    # In results/tables/<run-id>/: progress.csv and the timing tables (Task 5).
    progress = runs.progress_path(run)
    for table in (progress, *(progress.parent / name for name in runs.TIMING_TABLES)):
        if table.exists():
            table.unlink()


def turned_off(stage: str, config: PipelineConfig) -> bool:
    """Whether the configuration turns the stage off, so it runs no step."""
    if stage == "revalidate":
        return not config.revalidate
    if stage == "select":
        return not (config.revalidate and config.select)
    return False


def revalidate(ctx: Context) -> None:
    """Re-score the candidates on unseen seed blocks (B56, E87), starting afresh."""
    if turned_off("revalidate", ctx.config):
        return
    out = ctx.layout.run / "revalidation"
    shutil.rmtree(out, ignore_errors=True)
    ctx.must(
        [
            "-m",
            "covsyn.calibration.revalidation",
            ctx.layout.run,
            out,
            str(ctx.config.revalidation_blocks),
            "--stream",
            str(ctx.config.validation_stream),
        ],
        "revalidation",
        output=ctx.layout.logs / "revalidate.log",
    )


def select(ctx: Context) -> None:
    """Run the checklist for the best revalidated candidates and keep the best (B60)."""
    if turned_off("select", ctx.config):
        return
    ctx.must(
        [
            "-m",
            "covsyn.calibration.select_by_checklist",
            ctx.layout.run / "revalidation",
            ctx.layout.run / "selected",
            str(ctx.config.select_top),
        ],
        "checklist selection",
        output=ctx.layout.logs / "select.log",
    )


def _synthesis(ctx: Context, mode: str, out: Path, stream: int) -> None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)  # data_synthesis_main does not create its result path
    args = [
        "-m",
        "covsyn.model.data_synthesis_main",
        "--mode",
        mode,
        "--monte_carlo_number",
        str(ctx.config.monte_carlo),
        "--result_path",
        out,
        "--parameter_path",
        ctx.candidate,
    ]
    args += ["--stream", str(stream)]  # Task 7
    ctx.must(
        args,
        f"data synthesis of {out.relative_to(ctx.layout.run)}",
        output=ctx.layout.logs / "synthesise.log",
        append=True,
    )


def synthesise(ctx: Context) -> None:
    """Both modes on streams 301 and 300, and Cheng 2020 for Fig 6 on 300 (Task 7).

    The candidate is chosen on stream 301 and reported from stream 300 (E97).
    """
    for mode in (SPREAD_MODE, FIRST_MODE):
        _synthesis(ctx, mode, ctx.layout.synthetic(mode), seed_streams.SELECTION_STREAM)
    for mode in (SPREAD_MODE, FIRST_MODE, CHENG_MODE):
        _synthesis(ctx, mode, ctx.layout.independent(mode), seed_streams.REPORTING_STREAM)
    # A synthesis can report success and write no simulation (chain step 5).
    for mode in (SPREAD_MODE, FIRST_MODE):
        if not list(ctx.layout.independent(mode).glob("social_data_*.npy")):
            raise StageFailed(f"{ctx.layout.independent(mode)} holds no simulation")


def repository_path(setting: str) -> Path | None:
    """A path setting, relative to the repository root as every step runs there; None if empty."""
    if not setting:
        return None
    path = Path(setting)
    return path if path.is_absolute() else runs.REPO_ROOT / path


def check(ctx: Context) -> None:
    """The age risk ratio, then the checklist on both seed sets, and the previous run's."""
    tables = ctx.layout.tables
    age_rr = tables / "age_rr.json"
    # B14 is judged on 20000 index cases, not the spread output (E68); before the checklist.
    ctx.may(
        [
            "-m",
            "covsyn.validation.measure_age_rr",
            ctx.candidate,
            str(ctx.config.age_rr_cases),
            "--out",
            age_rr,
        ],
        "the age risk ratio measurement",
        output=tables / "age_rr.txt",
    )
    env = {"AGE_RR_FILE": str(age_rr)}
    ctx.may(
        [
            "-m",
            "covsyn.validation.verify_phase_d",
            ctx.layout.synthetic(SPREAD_MODE),
            ctx.layout.synthetic(FIRST_MODE),
            tables / "checks.json",
        ],
        "the checklist",
        output=tables / "checks.txt",
        env=env,
    )
    ctx.may(
        [
            "-m",
            "covsyn.validation.verify_phase_d",
            ctx.layout.independent(SPREAD_MODE),
            ctx.layout.independent(FIRST_MODE),
            tables / "checks_independent.json",
        ],
        "the independent checklist",
        output=tables / "checks_independent.txt",
        env=env,
    )
    previous = repository_path(ctx.config.previous_checks)
    if previous is not None and previous.is_file():
        ctx.may(
            ["-m", "covsyn.validation.compare_phase_d_runs", previous, tables / "checks.json"],
            "the run comparison",
            output=tables / "previous_vs_current.txt",
        )


REPORT = ("show_final", "check_gap", "rr_exact", "tw_check", "show_bounds")


def report(ctx: Context) -> None:
    """The standard run report, the contact days, and the per-case constraints."""
    tables, candidate = ctx.layout.tables, ctx.candidate
    out = tables / "report.txt"
    arguments = {
        "show_final": [candidate],
        "check_gap": [candidate, ctx.layout.synthetic(SPREAD_MODE)],
        "rr_exact": [ctx.layout.synthetic(SPREAD_MODE)],
        "tw_check": [ctx.layout.synthetic(FIRST_MODE)],
        "show_bounds": [],
    }
    out.write_text("")
    for name in REPORT:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"=== {name} ===\n")
        ctx.may(
            ["-m", f"covsyn.validation.{name}", *arguments[name]],
            f"the report's {name}",
            output=out,
            append=True,
        )
    # MEAN_CONTACT_DAYS in sar_anchors.py is a model output, re-measured after every run (E53).
    with open(out, "a", encoding="utf-8") as f:
        f.write("=== measure_days (update MEAN_CONTACT_DAYS in sar_anchors.py with this) ===\n")
    ctx.may(
        ["-m", "covsyn.validation.measure_days", ctx.layout.synthetic(SPREAD_MODE)],
        "measure_days",
        output=out,
        append=True,
    )
    ctx.may(
        [
            "-m",
            "covsyn.validation.check_constraints",
            ctx.layout.independent(SPREAD_MODE),
            ctx.layout.independent(FIRST_MODE),
            "--out",
            tables / "constraint_check",
        ],
        "check_constraints",
        output=ctx.layout.logs / "report.log",
    )


# The validation figures of the chain's step 5, with their arguments after the spread data.
VALIDATION_FIGURES = (
    "validate_layers",
    "validate_infection",
    "plot_diagnostics",
    "plot_10panel",
    "plot_cheng_attackrate",
    "plot_vs_notebook_literature",
    "plot_validation",
    "workplace_sampling_options",
    "plot_reality_vs_covsyn",
    "plot_violin_reality_vs_covsyn",
    "compare_healthcare_municipality",
)


def validation_figure_args(
    name: str, spread: Path, first: Path, out: Path, candidate: Path
) -> list:
    """The arguments of one validation figure, as phase_d_chain.sh gave them."""
    return {
        "validate_layers": [spread, out, candidate],
        "plot_10panel": [spread, out, candidate],
        "plot_vs_notebook_literature": [spread, out, candidate],
        "plot_validation": [candidate, out],
        "workplace_sampling_options": [spread, out, candidate],
        "plot_reality_vs_covsyn": [spread, first, out],
    }.get(name, [spread, out])


def figures(ctx: Context) -> None:
    """The validation figures, todolist923, and the Wu & Nordling 2025 comparison (E90, E97)."""
    figures_dir, log = ctx.layout.figures, ctx.layout.logs / "figures.log"
    spread, first = ctx.layout.independent(SPREAD_MODE), ctx.layout.independent(FIRST_MODE)
    validation = figures_dir / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    for name in VALIDATION_FIGURES:
        ctx.may(
            [
                "-m",
                f"covsyn.figures.{name}",
                *validation_figure_args(name, spread, first, validation, ctx.candidate),
            ],
            f"the figure {name}",
            output=log,
            append=True,
        )
    ctx.may(
        [
            "-m",
            "covsyn.figures.plot_todolist923",
            spread,
            ctx.candidate,
            ctx.layout.tables / "checks_independent.json",
            figures_dir / "todolist923",
        ],
        "plot_todolist923",
        output=log,
        append=True,
    )
    # Removed first, so a figure that fails is missing rather than stale.
    wu = figures_dir / "wu2025"
    shutil.rmtree(wu, ignore_errors=True)
    wu.mkdir(parents=True)
    failed = []
    for figure, source in (
        ("fig3", first),
        ("fig5", first),
        ("fig7", first),
        # Fig 4 plots the optimiser's own history, not the selected copy.
        ("fig4", ctx.layout.run),
        ("fig6", ctx.layout.independent(CHENG_MODE)),
    ):
        if (
            ctx.step(
                ["-m", "covsyn.figures.reproduce_wu2025", figure, source, wu],
                output=log,
                append=True,
            )
            != 0
        ):
            failed.append(figure)
    if failed:  # a missing reported figure is an error, as a missing checklist is (F6)
        raise StageFailed(f"Wu 2025 figures failed: {', '.join(failed)}")


STAGES = (
    ("optimise", optimise),
    ("revalidate", revalidate),
    ("select", select),
    ("synthesise", synthesise),
    ("check", check),
    ("report", report),
    ("figures", figures),
)
