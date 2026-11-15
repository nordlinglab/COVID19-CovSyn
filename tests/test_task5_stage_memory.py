# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 5, item 4: each pipeline stage's peak memory in run.log, and the memory check.

Every step runs in a fresh process with a fresh pool, so its own ``ru_maxrss`` is the parent's
peak in that step and its ``RUSAGE_CHILDREN`` the largest worker's. The step reports both
through ``covsyn.runtime.measure``; the pipeline keeps the largest of a stage's steps.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from covsyn.pipeline import config as pc
from covsyn.pipeline import runner as pr
from covsyn.runtime import measure, memory
from tests.test_task4_pipeline import STAGES, Recorder, prun, run_pipeline

MIB = 1024**2

STEP = textwrap.dedent(
    """
    import concurrent.futures
    import multiprocessing
    import sys

    import numpy as np


    def fill(mib):
        block = np.ones(mib * 1024**2 // 8)
        return int(block.size)


    if __name__ == "__main__":
        parent = np.ones(200 * 1024**2 // 8)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(2, mp_context=context) as pool:
            sizes = list(pool.map(fill, [300, 100]))
        sys.exit(int(sys.argv[1]))
    """
)


@pytest.fixture
def config():
    """The default configuration."""
    return pc.load("default")


@pytest.fixture
def step_module(tmp_path, monkeypatch):
    """A step that fills 200 MiB itself and 300 MiB in its largest worker."""
    (tmp_path / "memory_step.py").write_text(STEP)
    return {"PYTHONPATH": str(tmp_path)}


def test_a_measured_step_reports_its_own_and_its_largest_workers_peak(tmp_path, step_module):
    """The measured peaks are those of the step and its largest worker, in MB."""
    log = tmp_path / "memory.jsonl"
    env = {**step_module, measure.ENV: str(log)}
    assert pr.run_module(["-m", "memory_step", "0"], None, False, env) == 0
    (line,) = [json.loads(text) for text in log.read_text().splitlines()]
    assert line["step"] == "-m memory_step"
    assert line["parent_peak_mb"] >= 200 * MIB / memory.MB
    assert line["worker_peak_mb"] >= 300 * MIB / memory.MB
    assert line["worker_peak_mb"] < line["parent_peak_mb"] + 300 * MIB / memory.MB


def test_a_measured_step_runs_no_thread_of_its_own(tmp_path):
    """A step that forks its workers must be single-threaded when it forks (Python 3.12+)."""
    out = tmp_path / "out.txt"
    env = {measure.ENV: str(tmp_path / "memory.jsonl")}
    code = "import threading; print(threading.active_count())"
    assert pr.run_module(["-c", code], out, False, env) == 0
    assert out.read_text().strip() == "1"


def test_a_measured_step_keeps_its_exit_status(tmp_path, step_module):
    """A failing step still fails, and is still measured."""
    log = tmp_path / "memory.jsonl"
    env = {**step_module, measure.ENV: str(log)}
    assert pr.run_module(["-m", "memory_step", "3"], None, False, env) == 3
    assert pr.run_module(["-c", "import sys; sys.exit(4)"], None, False, env) == 4
    lines = [json.loads(text) for text in log.read_text().splitlines()]
    assert [line["step"] for line in lines] == ["-m memory_step", "-c"]
    assert lines[1]["worker_peak_mb"] is None  # no worker


def test_a_step_without_the_variable_runs_unmeasured(tmp_path):
    """Outside the pipeline nothing is wrapped."""
    out = tmp_path / "out.txt"
    code = "import sys; print('covsyn.runtime.measure' in sys.modules)"
    assert pr.run_module(["-c", code], out, False, {}) == 0
    assert out.read_text().strip() == "False"


class Measuring(Recorder):
    """A recorder whose steps report the peaks in ``peaks``, by module."""

    def __init__(self, peaks, fail=()):
        """Report ``peaks[module]`` (parent MB, worker MB) for each step."""
        super().__init__(fail)
        self.peaks = peaks

    def __call__(self, args, output, append, env):
        """Record the step, and write its measurement where the pipeline asks."""
        status = super().__call__(args, output, append, env)
        module = str(args[1])
        if module in self.peaks:
            parent, worker = self.peaks[module]
            line = {"step": f"-m {module}", "parent_peak_mb": parent, "worker_peak_mb": worker}
            with open(env[measure.ENV], "a") as f:
                f.write(json.dumps(line) + "\n")
        return status


def run_log(run_dir):
    """run.log's lines by stage."""
    lines = [json.loads(text) for text in (run_dir / "run.log").read_text().splitlines()]
    return {line["stage"]: line for line in lines}


def test_run_log_has_each_stages_own_peaks(tmp_path, config):
    """The largest over a stage's steps, not over the whole run so far."""
    peaks = {
        "covsyn.calibration.firefly_optimizer": (300.0, 900.0),
        "covsyn.model.data_synthesis_main": (500.0, 2000.0),  # synthesise
        "covsyn.validation.verify_phase_d": (700.0, None),  # check
        "covsyn.validation.measure_age_rr": (100.0, 4900.0),  # check
    }
    _, run_dir = run_pipeline(tmp_path, config, Measuring(peaks))
    lines = run_log(run_dir)
    assert list(lines) == STAGES
    assert (lines["optimise"]["parent_peak_mb"], lines["optimise"]["worker_peak_mb"]) == (300, 900)
    assert lines["synthesise"]["worker_peak_mb"] == 2000.0
    assert (lines["check"]["parent_peak_mb"], lines["check"]["worker_peak_mb"]) == (700, 4900)
    for line in lines.values():
        assert line["pipeline_peak_mb"] > 0


def test_a_stage_whose_steps_report_nothing_has_no_peaks(tmp_path, config):
    """Unmeasured is blank, not a number from another stage."""
    _, run_dir = run_pipeline(tmp_path, config, Recorder())
    for line in run_log(run_dir).values():
        assert line["parent_peak_mb"] is None and line["worker_peak_mb"] is None


def test_low_memory_during_a_stage_is_a_warning_of_it(tmp_path, config, monkeypatch):
    """The pipeline samples the available memory; below 10% it warns in run.log and pipeline.log."""
    monkeypatch.setattr(memory, "system_memory", lambda: (5, 100))
    _, run_dir = run_pipeline(tmp_path, config, Recorder())
    for line in run_log(run_dir).values():
        assert any("below 10%" in w for w in line["warnings"])
        assert line["lowest_available_mb"] is not None
    assert "below 10%" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_low_memory_is_reported_while_the_stage_runs(tmp_path, config, monkeypatch):
    """Live, once per stage, with one prefix: a stage killed for want of memory may not end."""
    import time
    from pathlib import Path

    monkeypatch.setattr(memory, "system_memory", lambda: (5, 100))
    seen = []

    class Watching(Recorder):
        def __call__(self, args, output, append, env):
            log = Path(env[measure.ENV]).parent.parent / pr.PIPELINE_LOG  # run/logs/...
            deadline = time.monotonic() + 5
            while "below 10%" not in log.read_text() and time.monotonic() < deadline:
                time.sleep(0.01)
            seen.append("below 10%" in log.read_text())
            return super().__call__(args, output, append, env)

    _, run_dir = run_pipeline(tmp_path, config, Watching())
    assert seen and all(seen)
    text = (run_dir / pr.PIPELINE_LOG).read_text()
    assert text.count("WARNING: only") == len(STAGES) and "WARNING: WARNING" not in text


def test_a_failed_stage_leaves_its_memory_readings(tmp_path, config):
    """The least available memory, and a file of the steps' peaks that a resume keeps."""
    import re
    from pathlib import Path

    peaks = {"covsyn.model.data_synthesis_main": (500.0, 9000.0)}
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.model.data_synthesis_main"})
    )
    assert status == 1
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "memory in stage synthesise: lowest available" in log
    (named,) = re.findall(r"peaks are in (\S+memory_synthesise\S*\.jsonl)", log)
    assert prun(None, run_dir, execute=Recorder(), resume=True) == 0
    assert Path(named).exists()


def test_a_stage_stopped_by_an_interrupt_leaves_its_memory_readings(tmp_path, config):
    """Any exit before the stage's end, not only a failed step."""

    class Interrupted(Recorder):
        def __call__(self, args, output, append, env):
            if "covsyn.model.data_synthesis_main" in map(str, args):
                raise KeyboardInterrupt
            return super().__call__(args, output, append, env)

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(tmp_path, config, Interrupted())
    run_dir = tmp_path / "results" / "models" / "20261115-090000_b30c141_phase-d"
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "memory in stage synthesise" in log
    assert "no step reported its peaks" in log and "memory_synthesise" not in log


def earlier_run(
    tmp_path, config, name, stage, worker_peak_mb, workers, parent_peak_mb=100.0, task5=True
):
    """A finished run beside the next one, with the same configuration unless told otherwise.

    ``task5``: its run.log line is of Task 5's format, with the stage's own peaks; before it,
    worker_peak_mb was the largest over the run so far.
    """
    run_dir = tmp_path / "results" / "models" / name
    run_dir.mkdir(parents=True)
    (run_dir / pr.SAVED_CONFIG).write_text(config)
    line = {"stage": stage, "workers": workers, "parent_peak_mb": parent_peak_mb,
            "worker_peak_mb": worker_peak_mb}  # fmt: skip
    if task5:
        line["pipeline_peak_mb"] = 50.0
    (run_dir / pr.RUN_LOG).write_text(json.dumps(line) + "\n")
    return run_dir


def test_a_stage_that_would_not_fit_is_refused(tmp_path, config, monkeypatch):
    """Workers x the peak per worker of an earlier run on this configuration > available."""
    earlier_run(tmp_path, pc.to_toml(config), "20261101-000000_aaaaaaa_phase-d",
                "optimise", 5000.0, 4)  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    recorder = Recorder()
    status, run_dir = run_pipeline(tmp_path, config, recorder)
    assert status == 1
    assert recorder.calls == []
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "optimise" in log and "10100 MB" in log and "8000 MB" in log
    assert "Resume with" in log


def test_the_check_uses_only_runs_on_the_same_configuration(tmp_path, config, monkeypatch):
    """Another configuration's peaks say nothing about this one's."""
    earlier_run(tmp_path, "# another\n", "20261101-000000_aaaaaaa_phase-d", "optimise",
                5000.0, 4)  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0


def test_the_steps_own_peak_counts_too(tmp_path, config, monkeypatch):
    """6000 MB in the step itself and 2 x 500 MB in its workers do not fit in 6500 MB."""
    earlier_run(tmp_path, pc.to_toml(config), "20261101-000000_aaaaaaa_phase-d",
                "optimise", 500.0, 4, parent_peak_mb=6000.0)  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(6.5e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 1


def test_a_run_log_from_before_task_5_is_not_used(tmp_path, config, monkeypatch):
    """Its worker peak is the run's so far, not the stage's, so it says nothing of the stage."""
    earlier_run(tmp_path, pc.to_toml(config), "20261101-000000_aaaaaaa_phase-d",
                "optimise", 5000.0, 4, task5=False)  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0


def test_a_line_that_is_not_an_object_is_skipped(tmp_path, config, monkeypatch):
    """A run.log or measurement line that parses but is not a JSON object stops nothing."""
    run_dir = earlier_run(tmp_path, pc.to_toml(config), "20261101-000000_aaaaaaa_phase-d",
                          "optimise", 3000.0, 4)  # fmt: skip
    with open(run_dir / pr.RUN_LOG, "a") as f:
        f.write("[]\n7\n")
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    assert pr.finished_stages(run_dir) == ["optimise"]
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0
    (tmp_path / "m.jsonl").write_text('[]\n{"parent_peak_mb": 5}\n')
    assert pr.stage_memory(tmp_path / "m.jsonl")["parent_peak_mb"] == 5


def test_a_stage_that_fits_runs(tmp_path, config, monkeypatch):
    """2 workers x 3000 MB fit in 8000 MB."""
    earlier_run(tmp_path, pc.to_toml(config), "20261101-000000_aaaaaaa_phase-d",
                "optimise", 3000.0, 4)  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0


def test_a_resumed_stage_starts_its_measurements_afresh(tmp_path, config):
    """The steps of a stage killed earlier do not count for the stage run again."""
    peaks = {"covsyn.model.data_synthesis_main": (500.0, 9000.0)}
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.model.data_synthesis_main"})
    )
    assert status == 1
    peaks = {"covsyn.model.data_synthesis_main": (400.0, 1000.0)}
    assert prun(None, run_dir, execute=Measuring(peaks), resume=True) == 0
    assert run_log(run_dir)["synthesise"]["worker_peak_mb"] == 1000.0


def test_only_stopped_attempts_leave_their_measurements(tmp_path, config):
    """A finished stage's peaks are in run.log; its file goes. A stopped attempt's stays."""
    peaks = {"covsyn.model.data_synthesis_main": (500.0, 9000.0),
             "covsyn.calibration.firefly_optimizer": (300.0, 900.0)}  # fmt: skip
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.model.data_synthesis_main"})
    )
    assert status == 1
    left = sorted(p.name.split("_")[1] for p in (run_dir / "logs").glob("memory_*.jsonl"))
    assert left == ["synthesise"]
    assert prun(None, run_dir, execute=Measuring(peaks), resume=True) == 0
    left = sorted(p.name.split("_")[1] for p in (run_dir / "logs").glob("memory_*.jsonl"))
    assert left == ["synthesise"]  # the stopped attempt's, not the finished one's


def test_a_stage_whose_memory_could_not_be_read_still_finishes(tmp_path, config, monkeypatch):
    """No available-memory reading is a blank in run.log, not a crash after the stage."""

    def failing():
        raise OSError("/proc unreadable")

    monkeypatch.setattr(memory, "system_memory", failing)
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 0
    assert all(line["lowest_available_mb"] is None for line in run_log(run_dir).values())


def test_a_pipeline_log_that_cannot_be_written_keeps_the_status(tmp_path, config):
    """A full disk under pipeline.log stops no report: a failed stage still gives status 1."""
    run_dir = tmp_path / "results" / "models" / "20261115-090000_b30c141_phase-d"
    (run_dir / pr.PIPELINE_LOG).mkdir(parents=True)  # every write to it fails
    status, _ = run_pipeline(tmp_path, config, Recorder(fail={"covsyn.model.data_synthesis_main"}))
    assert status == 1


def test_a_broken_stdout_still_leaves_pipeline_log(tmp_path, config, monkeypatch):
    """The terminal going away loses nothing from the run's own log."""
    import sys

    class Broken:
        def write(self, text):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    monkeypatch.setattr(sys, "stdout", Broken())
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 0
    assert "ALL DONE" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_each_line_goes_to_stdout_in_one_write(monkeypatch):
    """So the watch thread's lines cannot split another's."""
    import sys

    writes = []

    class Recording:
        def write(self, text):
            writes.append(text)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", Recording())
    pr.say("stage optimise")
    assert len(writes) == 1 and writes[0].endswith("stage optimise\n")


def test_a_stopped_stage_says_what_it_knows_when_a_reading_failed(tmp_path, config, monkeypatch):
    """No available-memory reading still leaves the line, the pipeline's peak and the file."""

    def failing():
        raise OSError("/proc/meminfo unreadable")

    monkeypatch.setattr(memory, "system_memory", failing)
    peaks = {"covsyn.model.data_synthesis_main": (500.0, 9000.0)}
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.model.data_synthesis_main"})
    )
    assert status == 1
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "memory in stage synthesise: lowest available unknown, this process at most" in log
    assert "memory_synthesise_" in log


def test_a_pipeline_without_stdout_still_runs(tmp_path, config, monkeypatch):
    """Started detached with no stdout (sys.stdout is None), it reports into pipeline.log."""
    import sys

    monkeypatch.setattr(sys, "stdout", None)
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 0 and "ALL DONE" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_an_unwritable_pipeline_log_is_said_once(tmp_path, config, capsys):
    """Not once per line for the rest of the run."""
    run_dir = tmp_path / "results" / "models" / "20261115-090000_b30c141_phase-d"
    (run_dir / pr.PIPELINE_LOG).mkdir(parents=True)
    run_pipeline(tmp_path, config, Recorder())
    assert capsys.readouterr().err.count("could not write") == 1
