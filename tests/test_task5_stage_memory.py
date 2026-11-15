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
    assert line["lowest_available_mb"] > 0 and line["low_memory"] is False


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
        """Report ``peaks[module]`` (parent MB, worker MB, low memory) for each step."""
        super().__init__(fail)
        self.peaks = peaks

    def __call__(self, args, output, append, env):
        """Record the step, and write its measurement where the pipeline asks."""
        status = super().__call__(args, output, append, env)
        module = str(args[1])
        if module in self.peaks:
            parent, worker, low = self.peaks[module]
            line = {"step": f"-m {module}", "parent_peak_mb": parent, "worker_peak_mb": worker,
                    "lowest_available_mb": 1000.0, "low_memory": low}  # fmt: skip
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
        "covsyn.calibration.firefly_optimizer": (300.0, 900.0, False),
        "covsyn.model.data_synthesis_main": (500.0, 2000.0, False),  # synthesise
        "covsyn.validation.verify_phase_d": (700.0, None, False),  # check
        "covsyn.validation.measure_age_rr": (100.0, 4900.0, False),  # check
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


def test_low_memory_in_a_step_is_a_warning_of_its_stage(tmp_path, config):
    """The step warns in its log; the pipeline repeats it in run.log and pipeline.log."""
    peaks = {"covsyn.model.data_synthesis_main": (500.0, 2000.0, True)}
    _, run_dir = run_pipeline(tmp_path, config, Measuring(peaks))
    warnings = run_log(run_dir)["synthesise"]["warnings"]
    assert any("below 10%" in w for w in warnings)
    assert "below 10%" in (run_dir / pr.PIPELINE_LOG).read_text()


def earlier_run(tmp_path, config, name, stage, worker_peak_mb, workers):
    """A finished run beside the next one, with the same configuration unless told otherwise."""
    run_dir = tmp_path / "results" / "models" / name
    run_dir.mkdir(parents=True)
    (run_dir / pr.SAVED_CONFIG).write_text(config)
    line = {"stage": stage, "workers": workers, "parent_peak_mb": 100.0,
            "worker_peak_mb": worker_peak_mb}  # fmt: skip
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
    assert "optimise" in log and "10000 MB" in log and "8000 MB" in log
    assert "Resume with" in log


def test_the_check_uses_only_runs_on_the_same_configuration(tmp_path, config, monkeypatch):
    """Another configuration's peaks say nothing about this one's."""
    earlier_run(tmp_path, "# another\n", "20261101-000000_aaaaaaa_phase-d", "optimise",
                5000.0, 4)  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0


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
    peaks = {"covsyn.model.data_synthesis_main": (500.0, 9000.0, False)}
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.model.data_synthesis_main"})
    )
    assert status == 1
    peaks = {"covsyn.model.data_synthesis_main": (400.0, 1000.0, False)}
    assert prun(None, run_dir, execute=Measuring(peaks), resume=True) == 0
    assert run_log(run_dir)["synthesise"]["worker_peak_mb"] == 1000.0
