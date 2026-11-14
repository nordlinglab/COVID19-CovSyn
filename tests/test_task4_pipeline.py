# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 4: covsyn-pipeline runs the stages of phase_d_chain.sh in order and resumes.

The stages, their commands and what stops the run follow scripts/phase_d_chain.sh as it was at
b30c141; every output goes into the run's own directories (Task 3). The step runner is replaced
by a recorder, so no simulation runs: each test checks what the pipeline asks for, in order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from covsyn.pipeline import config as pc
from covsyn.pipeline import runner as pr
from covsyn.pipeline import stages as ps

STAGES = ["optimise", "revalidate", "select", "synthesise", "check", "report", "figures"]


class Recorder:
    """Records each step and creates the files a real step would leave, exiting with ``fail``."""

    def __init__(self, fail=()):
        """Fail the steps of the modules in ``fail``."""
        self.calls: list[list[str]] = []
        self.envs: list[dict] = []
        self.fail = set(fail)

    def __call__(self, args, output, append, env):
        """Record one step, as runner.run_module would run it."""
        args = [str(a) for a in args]
        self.calls.append(args)
        self.envs.append(dict(env))
        module = args[1]
        if module == "covsyn.calibration.firefly_optimizer":
            run = Path(
                args[args.index("--resume") + 1]
                if "--resume" in args
                else args[args.index("--run-dir") + 1]
            )
            for name in ("firefly_best.txt", "firefly_result.txt", "config.json"):
                (run / name).write_text("1 2 3\n")
        if module == "covsyn.model.data_synthesis_main":
            out = Path(args[args.index("--result_path") + 1])
            (out / "social_data_0.npy").write_text("x")
        if module in ("covsyn.calibration.revalidation", "covsyn.calibration.select_by_checklist"):
            out = Path(args[3])
            out.mkdir(parents=True, exist_ok=True)
            (out / "firefly_best.txt").write_text("1 2 3\n")
        if output is not None:
            with open(output, "a" if append else "w") as f:
                f.write(f"output of {module}\n")
        return 1 if module in self.fail or " ".join(args) in self.fail else 0


def modules(recorder):
    """The module each recorded step ran, in order."""
    return [c[1] for c in recorder.calls]


@pytest.fixture
def config():
    """The default configuration with a short warm start."""
    return pc.load("default")


def run_pipeline(tmp_path, config, recorder, resume=False):
    """Run the pipeline into a fresh results tree under tmp_path."""
    root = tmp_path / "results"
    run_dir = root / "models" / "20261115-090000_b30c141_phase-d"
    status = pr.run(config, run_dir, execute=recorder, resume=resume)
    return status, run_dir


# --- configuration ----------------------------------------------------------------------------


def test_the_default_configuration_matches_the_chains_defaults(config):
    """The chain's defaults: revalidation on 10 blocks (E95), the best 5 checked (B60), ..."""
    assert (config.revalidate, config.revalidation_blocks) == (True, 10)
    assert (config.select, config.select_top) == (True, 5)
    assert (config.monte_carlo, config.independent_seed_start) == (1000, 500000)
    assert config.age_rr_cases == 20000
    assert config.warm_start == ("results/models/phaseD_run10/firefly_best.txt",)
    assert config.mode == "train" and config.workers is None


def test_a_configuration_file_can_be_named_by_path(tmp_path):
    """--config takes a packaged name or a TOML file."""
    path = tmp_path / "mine.toml"
    path.write_text("[select]\ntop = 3\n")
    config = pc.load(str(path))
    assert config.select_top == 3 and config.revalidation_blocks == 10


@pytest.mark.parametrize(
    "text",
    [
        '[select]\ntop = "five"\n',
        "[select]\ntop = 0\n",
        "[nonsense]\nx = 1\n",
        "[select]\ntopp = 3\n",
        "[run]\nworkers = -1\n",
    ],
)
def test_a_wrong_setting_is_refused_with_its_name(tmp_path, text):
    """A typo in a setting would otherwise run hours with the default."""
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ValueError):
        pc.load(str(path))


def test_an_unknown_configuration_name_is_refused():
    """Neither a packaged configuration nor a file."""
    with pytest.raises(FileNotFoundError):
        pc.load("no-such-config")


# --- the stages, in order -----------------------------------------------------------------------


def test_the_stages_run_in_the_chains_order(tmp_path, config):
    """optimise, revalidate, select, synthesise, check, report, figures."""
    recorder = Recorder()
    status, run_dir = run_pipeline(tmp_path, config, recorder)
    assert status == 0
    assert [
        json.loads(line)["stage"] for line in (run_dir / "run.log").read_text().splitlines()
    ] == STAGES
    order = modules(recorder)
    assert order[0] == "covsyn.calibration.firefly_optimizer"
    assert order[1:3] == [
        "covsyn.calibration.revalidation",
        "covsyn.calibration.select_by_checklist",
    ]
    assert order.index("covsyn.validation.measure_age_rr") < order.index(
        "covsyn.validation.verify_phase_d"
    )  # the checklist reads its JSON (B14, E68)


def test_the_optimiser_writes_into_the_run_directory_with_the_warm_start(tmp_path, config):
    """Like launch_phase_d.sh: --mode train, the run directory, and the warm-start files."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    args = recorder.calls[0]
    assert args[args.index("--run-dir") + 1] == str(run_dir)
    assert args[args.index("--mode") + 1] == "train"
    assert args[args.index("--warm_start") + 1 :] == list(config.warm_start)


def test_revalidation_and_selection_stay_in_the_run_directory(tmp_path, config):
    """Task 3: run/revalidation and run/selected, with the chain's block and candidate counts."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    revalidation = recorder.calls[1]
    assert revalidation[2:] == [str(run_dir), str(run_dir / "revalidation"), "10"]
    select = recorder.calls[2]
    assert select[2:] == [str(run_dir / "revalidation"), str(run_dir / "selected"), "5"]


def test_every_later_stage_reads_the_selected_vector(tmp_path, config):
    """The synthesis, the age risk ratio and the run report use run/selected."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    selected = str(run_dir / "selected")
    for args in recorder.calls:
        if args[1] == "covsyn.model.data_synthesis_main":
            assert args[args.index("--parameter_path") + 1] == selected
        if args[1] in (
            "covsyn.validation.measure_age_rr",
            "covsyn.validation.show_final",
            "covsyn.validation.check_gap",
        ):
            assert args[2] == selected


def test_without_revalidation_the_optimiser_output_is_used(tmp_path, config):
    """REVALIDATE=0 in the chain: neither revalidation nor selection, the run itself is read."""
    recorder = Recorder()
    config = pc.replace(config, revalidate=False)
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    assert "covsyn.calibration.revalidation" not in modules(recorder)
    assert "covsyn.calibration.select_by_checklist" not in modules(recorder)
    synthesis = next(a for a in recorder.calls if a[1] == "covsyn.model.data_synthesis_main")
    assert synthesis[synthesis.index("--parameter_path") + 1] == str(run_dir)


def test_without_selection_the_revalidated_vector_is_used(tmp_path, config):
    """SELECT=0: the revalidation's best is read."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, pc.replace(config, select=False), recorder)
    synthesis = next(a for a in recorder.calls if a[1] == "covsyn.model.data_synthesis_main")
    assert synthesis[synthesis.index("--parameter_path") + 1] == str(run_dir / "revalidation")


def test_the_synthesis_runs_both_modes_and_the_independent_seeds(tmp_path, config):
    """Seeds 0..999 for choosing, 500000.. for reporting (B60, E97), and Cheng 2020 for Fig 6."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    syntheses = [a for a in recorder.calls if a[1] == "covsyn.model.data_synthesis_main"]
    plan = [
        (
            a[a.index("--mode") + 1],
            Path(a[a.index("--result_path") + 1]).relative_to(run_dir),
            a[a.index("--seed_start") + 1] if "--seed_start" in a else None,
        )
        for a in syntheses
    ]
    assert plan == [
        ("spread_Taiwan_weight", Path("synthetic/spread_Taiwan_weight"), None),
        ("taiwan_first_outbreak", Path("synthetic/taiwan_first_outbreak"), None),
        ("spread_Taiwan_weight", Path("independent/spread_Taiwan_weight"), "500000"),
        ("taiwan_first_outbreak", Path("independent/taiwan_first_outbreak"), "500000"),
        ("cheng2020", Path("independent/cheng2020"), "500000"),
    ]
    assert all(a[a.index("--monte_carlo_number") + 1] == "1000" for a in syntheses)


def test_the_checklists_score_this_runs_age_risk_ratio(tmp_path, config):
    """AGE_RR_FILE names this run's age_rr.json, never validation_reference's."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    tables = ps.Layout(run_dir).tables
    age = next(a for a in recorder.calls if a[1] == "covsyn.validation.measure_age_rr")
    assert age[2:4] == [str(run_dir / "selected"), "20000"]
    assert age[age.index("--out") + 1] == str(tables / "age_rr.json")
    checks = [
        (a, e)
        for a, e in zip(recorder.calls, recorder.envs, strict=True)
        if a[1] == "covsyn.validation.verify_phase_d"
    ]
    assert [a[4] for a, _ in checks] == [
        str(tables / "checks.json"),
        str(tables / "checks_independent.json"),
    ]
    assert all(e["AGE_RR_FILE"] == str(tables / "age_rr.json") for _, e in checks)


def test_outputs_go_into_the_runs_tables_and_figures(tmp_path, config):
    """Task 3's layout: results/tables/<run-id>/ and results/figures/<run-id>/."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    layout = ps.Layout(run_dir)
    assert layout.tables == tmp_path / "results" / "tables" / run_dir.name
    assert layout.figures == tmp_path / "results" / "figures" / run_dir.name
    assert (layout.tables / "report.txt").read_text().count("=== ") == 6
    todo = next(a for a in recorder.calls if a[1] == "covsyn.figures.plot_todolist923")
    assert todo[2:] == [
        str(run_dir / "independent" / "spread_Taiwan_weight"),
        str(run_dir / "selected"),
        str(layout.tables / "checks_independent.json"),
        str(layout.figures / "todolist923"),
    ]
    fig4 = next(a for a in recorder.calls if a[1:3] == ["covsyn.figures.reproduce_wu2025", "fig4"])
    assert fig4[3] == str(run_dir)  # the optimiser's own history, not the selected copy


def test_nothing_is_written_outside_the_run(tmp_path, config, monkeypatch):
    """The chain wrote phaseD_report.txt, validation_reference/... in the working directory."""
    monkeypatch.chdir(tmp_path)
    run_pipeline(tmp_path, config, Recorder())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["results"]


def test_the_previous_run_is_compared_when_given(tmp_path, config):
    """PREVIOUS_CHECKS in the chain; an empty setting skips the comparison."""
    previous = tmp_path / "old_checks.json"
    previous.write_text("{}")
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, pc.replace(config, previous_checks=str(previous)), recorder)
    compare = next(a for a in recorder.calls if a[1] == "covsyn.validation.compare_phase_d_runs")
    assert compare[2:] == [str(previous), str(ps.Layout(run_dir).tables / "checks.json")]
    recorder = Recorder()
    run_pipeline(tmp_path / "again", config, recorder)
    assert "covsyn.validation.compare_phase_d_runs" not in modules(recorder)


def test_the_workers_setting_reaches_every_step(tmp_path, config):
    """COVSYN_WORKERS, which every module reads (Task 2)."""
    recorder = Recorder()
    run_pipeline(tmp_path, pc.replace(config, workers=6), recorder)
    assert {e["COVSYN_WORKERS"] for e in recorder.envs} == {"6"}


# --- failures ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "covsyn.calibration.firefly_optimizer",
        "covsyn.calibration.revalidation",
        "covsyn.calibration.select_by_checklist",
        "covsyn.model.data_synthesis_main",
    ],
)
def test_a_failed_step_the_chain_stopped_on_stops_the_pipeline(tmp_path, config, module):
    """The optimiser, revalidation, selection and synthesis stop the chain; nothing runs after."""
    recorder = Recorder(fail={module})
    status, run_dir = run_pipeline(tmp_path, config, recorder)
    assert status != 0
    assert modules(recorder)[-1] == module
    done = (
        [json.loads(line)["stage"] for line in (run_dir / "run.log").read_text().splitlines()]
        if (run_dir / "run.log").exists()
        else []
    )
    assert len(done) < len(STAGES)


@pytest.mark.parametrize(
    "module",
    [
        "covsyn.validation.measure_age_rr",
        "covsyn.validation.verify_phase_d",
        "covsyn.validation.show_final",
        "covsyn.figures.validate_layers",
    ],
)
def test_a_failed_step_the_chain_warned_about_does_not_stop_it(tmp_path, config, module, capsys):
    """The chain only warned about these; the pipeline goes on and says so."""
    recorder = Recorder(fail={module})
    status, _ = run_pipeline(tmp_path, config, recorder)
    assert status == 0
    assert "covsyn.figures.reproduce_wu2025" in modules(recorder)
    assert "WARNING" in capsys.readouterr().out


def test_a_failed_wu_figure_fails_the_pipeline_after_the_others(tmp_path, config):
    """A missing reported figure is an error (F6), reported once every figure has been tried."""

    class Fig3Fails(Recorder):
        def __call__(self, args, output, append, env):
            status = super().__call__(args, output, append, env)
            return (
                1
                if [str(a) for a in args[1:3]] == ["covsyn.figures.reproduce_wu2025", "fig3"]
                else status
            )

    recorder = Fig3Fails()
    status, run_dir = run_pipeline(tmp_path, config, recorder)
    wu = [a[2] for a in recorder.calls if a[1] == "covsyn.figures.reproduce_wu2025"]
    assert wu == ["fig3", "fig5", "fig7", "fig4", "fig6"]
    assert status != 0
    stages = [json.loads(line)["stage"] for line in (run_dir / "run.log").read_text().splitlines()]
    assert "figures" not in stages  # so a resumed run draws them again


def test_a_synthesis_that_wrote_no_simulation_stops_before_the_figures(tmp_path, config):
    """It reported success but left no social_data_*.npy, as the chain's step 5 checked."""

    class Empty(Recorder):
        def __call__(self, args, output, append, env):
            status = super().__call__(args, output, append, env)
            if args[1] == "covsyn.model.data_synthesis_main":
                for f in Path(args[args.index("--result_path") + 1]).glob("social_data_*"):
                    f.unlink()
            return status

    recorder = Empty()
    status, _ = run_pipeline(tmp_path, config, recorder)
    assert status != 0 and "covsyn.validation.verify_phase_d" not in modules(recorder)


# --- run.log and resume -----------------------------------------------------------------------


def test_each_finished_stage_is_one_line_of_run_log(tmp_path, config):
    """Stage, end time, duration, workers, and the peak memory of parent and workers."""
    _, run_dir = run_pipeline(tmp_path, config, Recorder())
    lines = [json.loads(line) for line in (run_dir / "run.log").read_text().splitlines()]
    assert len(lines) == len(STAGES)
    for line in lines:
        assert set(line) >= {
            "stage",
            "finished",
            "seconds",
            "workers",
            "parent_peak_mb",
            "worker_peak_mb",
        }
        assert line["seconds"] >= 0 and line["parent_peak_mb"] > 0


def test_resume_skips_the_finished_stages(tmp_path, config):
    """Killed during the check stage, the resumed run starts the check stage again."""
    recorder = Recorder(fail={"covsyn.model.data_synthesis_main"})
    status, run_dir = run_pipeline(tmp_path, config, recorder)
    assert status != 0
    recorder = Recorder()
    status = pr.run(None, run_dir, execute=recorder, resume=True)
    assert status == 0
    assert modules(recorder)[0] == "covsyn.model.data_synthesis_main"
    assert [
        json.loads(line)["stage"] for line in (run_dir / "run.log").read_text().splitlines()
    ] == STAGES


def test_resume_continues_the_optimiser_from_its_checkpoint(tmp_path, config):
    """An unfinished optimise stage restarts as --resume, not as a new run."""
    recorder = Recorder(fail={"covsyn.calibration.firefly_optimizer"})
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "generation_0003.npz").write_text("x")
    recorder = Recorder()
    pr.run(None, run_dir, execute=recorder, resume=True)
    assert recorder.calls[0][2:] == ["--resume", str(run_dir)]


def test_resume_uses_the_runs_own_configuration(tmp_path, config):
    """The configuration is saved with the run, so a resumed run keeps its settings."""
    recorder = Recorder(fail={"covsyn.model.data_synthesis_main"})
    _, run_dir = run_pipeline(tmp_path, pc.replace(config, monte_carlo=40), recorder)
    recorder = Recorder()
    pr.run(None, run_dir, execute=recorder, resume=True)
    synthesis = recorder.calls[0]
    assert synthesis[synthesis.index("--monte_carlo_number") + 1] == "40"


def test_resuming_a_finished_run_runs_nothing(tmp_path, config):
    """Every stage is in run.log."""
    _, run_dir = run_pipeline(tmp_path, config, Recorder())
    recorder = Recorder()
    assert pr.run(None, run_dir, execute=recorder, resume=True) == 0
    assert recorder.calls == []


def test_resuming_a_directory_that_is_no_pipeline_run_is_an_error(tmp_path):
    """Without the saved configuration there is nothing to resume."""
    (tmp_path / "x").mkdir()
    with pytest.raises(FileNotFoundError, match="pipeline"):
        pr.run(None, tmp_path / "x", execute=Recorder(), resume=True)


def test_a_new_run_refuses_a_directory_holding_a_run(tmp_path, config):
    """As the optimiser's claim does (Task 3)."""
    run_dir = tmp_path / "results" / "models" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "firefly_best.txt").write_text("1\n")
    with pytest.raises(FileExistsError):
        pr.run(config, run_dir, execute=Recorder())


# --- the command line -------------------------------------------------------------------------


def test_the_command_resumes_a_run_by_id(tmp_path, monkeypatch):
    """--resume <run-id> names results/models/<run-id>."""
    seen = {}

    def fake_run(config, run_dir, execute=None, resume=False):
        seen.update(config=config, run_dir=run_dir, resume=resume)
        return 0

    monkeypatch.setattr(pr, "run", fake_run)
    monkeypatch.setattr(pr.runs, "RESULTS_ROOT", tmp_path / "results")
    assert pr.main(["--resume", "20261115-090000_b30c141_phase-d"]) == 0
    assert seen["run_dir"] == tmp_path / "results" / "models" / "20261115-090000_b30c141_phase-d"
    assert seen["resume"] and seen["config"] is None


def test_the_command_starts_a_new_run_with_a_configuration(tmp_path, monkeypatch):
    """--config default names a new run under results/models."""
    seen = {}
    monkeypatch.setattr(
        pr,
        "run",
        lambda config, run_dir, execute=None, resume=False: (
            seen.update(config=config, run_dir=run_dir) or 0
        ),
    )
    monkeypatch.setattr(pr.runs, "RESULTS_ROOT", tmp_path / "results")
    assert pr.main(["--config", "default"]) == 0
    assert seen["run_dir"].parent == tmp_path / "results" / "models"
    assert seen["run_dir"].name.endswith("_phase-d") and not seen["run_dir"].exists()


def test_the_real_step_runner_runs_a_module_from_the_repository_root(tmp_path):
    """The default runner starts the module with this interpreter and captures its output."""
    out = tmp_path / "out.txt"
    status = pr.run_module(["-c", "import os; print(os.getcwd())"], out, False, {})
    assert status == 0 and Path(out.read_text().strip()) == pr.runs.REPO_ROOT
    assert pr.run_module(["-c", "import sys; sys.exit(3)"], None, False, {}) == 3
    assert sys.executable  # the interpreter the runner uses


def test_an_earlier_wu_figure_is_removed_before_drawing(tmp_path, config):
    """A figure that fails is missing, not stale (code review 2026-11-09), as in the chain."""
    root = tmp_path / "results"
    run_dir = root / "models" / "r"
    stale = root / "figures" / "r" / "wu2025" / "fig3.png"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    pr.run(config, run_dir, execute=Recorder())
    assert not stale.exists()


def test_a_resumed_revalidation_starts_afresh(tmp_path, config):
    """Old candidates in run/revalidation must not mix with the new ones."""
    recorder = Recorder(fail={"covsyn.calibration.revalidation"})
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    (run_dir / "revalidation" / "stale.txt").write_text("old")
    pr.run(None, run_dir, execute=Recorder(), resume=True)
    assert not (run_dir / "revalidation" / "stale.txt").exists()


# --- code review 2026-11-15 (high) ---------------------------------------------------------


def test_an_optimiser_killed_before_its_first_checkpoint_starts_again(tmp_path, config):
    """Killed while scoring the initial population: its claim is cleared and it runs anew."""
    recorder = Recorder(fail={"covsyn.calibration.firefly_optimizer"})
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    (run_dir / "config.json").write_text('{"claimed": true}\n')  # what the claim left
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "initial.tmp.npz").write_text("cut off")
    recorder = Recorder()
    pr.run(None, run_dir, execute=recorder, resume=True)
    first = recorder.calls[0]
    assert "--run-dir" in first and "--resume" not in first
    assert not (run_dir / "checkpoints").exists() or not list((run_dir / "checkpoints").iterdir())


def test_a_relative_previous_checks_is_read_from_the_repository_root(tmp_path, config):
    """Every step runs from the repository root, so a setting's path is read from there."""
    recorder = Recorder()
    name = "results/tables/phaseD_run10/progress.csv"  # a committed file, as a stand-in
    _, run_dir = run_pipeline(tmp_path, pc.replace(config, previous_checks=name), recorder)
    compare = next(a for a in recorder.calls if a[1] == "covsyn.validation.compare_phase_d_runs")
    assert compare[2] == str(pr.runs.REPO_ROOT / name)


def test_warnings_are_kept_with_the_run(tmp_path, config):
    """In pipeline.log and the stage's run.log line, not only in pipeline.out.

    The next resume overwrites pipeline.out.
    """
    _, run_dir = run_pipeline(tmp_path, config, Recorder(fail={"covsyn.validation.show_final"}))
    assert "WARNING: the report's show_final failed" in (run_dir / "pipeline.log").read_text()
    lines = {json.loads(x)["stage"]: json.loads(x) for x in (run_dir / "run.log").read_text()
             .splitlines()}
    assert lines["report"]["warnings"] == ["the report's show_final failed"]
    assert lines["check"]["warnings"] == []


def test_a_run_in_use_cannot_be_resumed(tmp_path, config):
    """A second pipeline, or a step one left running, holds the lock; a resume is refused."""
    import fcntl
    import os

    recorder = Recorder(fail={"covsyn.model.data_synthesis_main"})
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    fd = os.open(run_dir / "pipeline.lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # as the orphaned optimiser would hold it
    try:
        with pytest.raises(RuntimeError, match="in use"):
            pr.run(None, run_dir, execute=Recorder(), resume=True)
    finally:
        os.close(fd)
    assert pr.run(None, run_dir, execute=Recorder(), resume=True) == 0


def test_every_step_inherits_the_runs_lock(tmp_path, config):
    """A real step holds the lock too, so it outlives a killed pipeline."""
    seen = []

    def step(args, output, append, env):
        seen.append(pr._lock_fds)
        return Recorder()(args, output, append, env)

    run_pipeline(tmp_path, config, step)
    assert seen and all(len(fds) == 1 for fds in seen)
    assert pr._lock_fds == ()  # released when the run ends


def test_a_cut_off_run_log_line_does_not_stop_a_resume(tmp_path, config):
    """A kill while writing run.log leaves half a line; it is not a finished stage."""
    recorder = Recorder(fail={"covsyn.model.data_synthesis_main"})
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    with open(run_dir / "run.log", "a") as f:
        f.write('{"stage": "synth')
    recorder = Recorder()
    assert pr.run(None, run_dir, execute=recorder, resume=True) == 0
    assert modules(recorder)[0] == "covsyn.model.data_synthesis_main"


def test_run_log_records_the_workers_the_steps_use(tmp_path, config, monkeypatch):
    """COVSYN_WORKERS from the shell, when the configuration sets none."""
    monkeypatch.setenv("COVSYN_WORKERS", "7")
    _, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert {json.loads(x)["workers"] for x in (run_dir / "run.log").read_text().splitlines()} \
        == {7}


def test_check_gap_reads_this_runs_spread_simulations(tmp_path, config):
    """It read a fixed synthetic_data_results_spread_Taiwan_weight in the working directory."""
    recorder = Recorder()
    _, run_dir = run_pipeline(tmp_path, config, recorder)
    gap = next(a for a in recorder.calls if a[1] == "covsyn.validation.check_gap")
    assert gap[2:] == [str(run_dir / "selected"),
                       str(run_dir / "synthetic" / "spread_Taiwan_weight")]


def test_a_relative_run_directory_is_made_absolute(tmp_path, config, monkeypatch):
    """Steps run from the repository root, so a path relative to elsewhere must not reach them."""
    monkeypatch.chdir(tmp_path)
    recorder = Recorder()
    pr.run(config, Path("results/models/rel"), execute=recorder)
    optimiser = recorder.calls[0]
    assert optimiser[optimiser.index("--run-dir") + 1] == str(tmp_path / "results/models/rel")


def test_the_old_worker_variables_are_reported(tmp_path, config, monkeypatch, capsys):
    """The chain warned about VERIFY_WORKERS and CPU_CORES; so does the pipeline."""
    monkeypatch.setenv("VERIFY_WORKERS", "8")
    run_pipeline(tmp_path, config, Recorder())
    assert "VERIFY_WORKERS and CPU_CORES are no longer read" in capsys.readouterr().out


def test_tw_check_lists_earlier_runs_only_when_given_no_run(tmp_path):
    """Given this run, as the pipeline gives it, the report is about this run alone."""
    import subprocess

    old = pr.runs.REPO_ROOT / "synthetic_data_results_taiwan_first_outbreak_zz_test_only"
    old.mkdir()
    try:
        given = subprocess.run([sys.executable, "-m", "covsyn.validation.tw_check", str(tmp_path)],
                               cwd=pr.runs.REPO_ROOT, capture_output=True, text=True)
    finally:
        old.rmdir()
    assert "previous:" not in given.stdout, given.stdout + given.stderr
