# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 5, item 4: each pipeline stage's peak memory in run.log, and the memory check.

Every step runs in a fresh process with a fresh pool, so its own ``ru_maxrss`` is the parent's
peak in that step and its ``RUSAGE_CHILDREN`` the largest worker's. The step reports both
through ``covsyn.runtime.measure``; the pipeline keeps the largest of a stage's steps.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path

import pytest

from covsyn.pipeline import config as pc
from covsyn.pipeline import runner as pr
from covsyn.pipeline import stages as ps
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


def test_a_failed_stage_leaves_its_memory_readings(tmp_path, config, monkeypatch):
    """The least available memory, and a file of the steps' peaks that a resume keeps."""
    # Its resume reads the stopped attempt's peaks (made up, and large): let them fit.
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(1e15))
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
    if task5:  # measured whole: the memory check may use it
        line.update(pipeline_peak_mb=50.0, finished="2026-11-01T10:00:00+08:00",
                    steps=[OPTIMISER_STEP], steps_measured=[OPTIMISER_STEP],
                    partial=False)  # fmt: skip
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


def test_a_resumed_stage_starts_its_measurements_afresh(tmp_path, config, monkeypatch):
    """The steps of a stage killed earlier do not count for the stage run again."""
    # Its resume reads the stopped attempt's peaks (made up, and large): let them fit.
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(1e15))
    peaks = {"covsyn.model.data_synthesis_main": (500.0, 9000.0)}
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.model.data_synthesis_main"})
    )
    assert status == 1
    peaks = {"covsyn.model.data_synthesis_main": (400.0, 1000.0)}
    assert prun(None, run_dir, execute=Measuring(peaks), resume=True) == 0
    assert run_log(run_dir)["synthesise"]["worker_peak_mb"] == 1000.0


def test_only_stopped_attempts_leave_their_measurements(tmp_path, config, monkeypatch):
    """A finished stage's peaks are in run.log; its file goes. A stopped attempt's stays."""
    # Its resume reads the stopped attempt's peaks (made up, and large): let them fit.
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(1e15))
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
    pr.Reporter()("stage optimise")
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


def test_a_refused_stage_writes_no_memory_line(tmp_path, config, monkeypatch):
    """The memory check refused it before it started: nothing was measured."""
    earlier_run(tmp_path, pc.to_toml(config), "20261101-000000_aaaaaaa_phase-d",
                "optimise", 5000.0, 4)  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1 and "memory in stage" not in (run_dir / pr.PIPELINE_LOG).read_text()


def test_a_warning_whose_report_failed_is_still_in_run_log(tmp_path, config, monkeypatch):
    """The stage's record has it once, though its report failed after writing pipeline.log."""
    monkeypatch.setattr(memory, "system_memory", lambda: (5, 100))
    call = pr.Reporter.__call__

    def failing(self, message):
        call(self, message)
        if message.startswith("WARNING: only"):
            raise RuntimeError("a custom stream")

    monkeypatch.setattr(pr.Reporter, "__call__", failing)
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 0 and list(run_log(run_dir)) == STAGES
    for line in run_log(run_dir).values():
        assert sum("below 10%" in w for w in line["warnings"]) == 1


def test_a_warning_whose_report_wrote_nothing_is_in_run_log_once(tmp_path, config, monkeypatch):
    """The memory was low whatever the output did: run.log says so, once."""
    monkeypatch.setattr(memory, "system_memory", lambda: (5, 100))
    call = pr.Reporter.__call__

    def failing(self, message):
        if message.startswith("WARNING: only"):
            raise RuntimeError("a closed stream")  # before writing anything
        call(self, message)

    monkeypatch.setattr(pr.Reporter, "__call__", failing)
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 0 and list(run_log(run_dir)) == STAGES
    assert "below 10%" not in (run_dir / pr.PIPELINE_LOG).read_text()  # no report got through
    for line in run_log(run_dir).values():
        assert sum("below 10%" in w for w in line["warnings"]) == 1


def test_run_log_has_the_steps_rss_sampled_once_a_second(tmp_path, config):
    """The plan's sampled parent RSS: the pipeline samples each step, the pool's owner.

    The step holds its memory for 3.5 s, so the once-a-second samples catch it at least twice,
    however long a loaded machine takes to import numpy.
    """
    code = "import numpy as np, time; b = np.ones(200 * 1024**2 // 8); time.sleep(3.5)"

    class Running(Recorder):
        def __call__(self, args, output, append, env):
            if str(args[1]) == "covsyn.calibration.firefly_optimizer":
                pr.run_module(["-c", code], None, False, env)  # a measured step
            return super().__call__(args, output, append, env)

    _, run_dir = run_pipeline(tmp_path, config, Running())
    lines = run_log(run_dir)
    assert lines["optimise"]["step_sampled_peak_mb"] >= 199 * MIB / memory.MB
    assert lines["figures"]["step_sampled_peak_mb"] is None  # no step process ran


def test_a_stopped_stage_reports_its_sampled_step_peak(tmp_path, config):
    """A step killed before its measurement is written leaves only the sampled RSS."""
    code = (
        "import numpy as np, time, os; b = np.ones(200 * 1024**2 // 8); time.sleep(3.5); "
        "os._exit(9)"  # killed: no atexit, so no measurement line
    )

    class Killed(Recorder):
        def __call__(self, args, output, append, env):
            if str(args[1]) == "covsyn.calibration.firefly_optimizer":
                pr.run_module(["-c", code], None, False, env)
                return 1
            return super().__call__(args, output, append, env)

    status, run_dir = run_pipeline(tmp_path, config, Killed())
    assert status == 1
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "no step reported its peaks" in log
    (sampled,) = re.findall(r"a step at most (\d+) MB", log)
    assert int(sampled) >= 199 * MIB / memory.MB


def test_a_step_that_wrote_no_peak_makes_its_stage_unusable(tmp_path, config, monkeypatch):
    """A step killed before measuring leaves its stage incomplete: no estimate next time."""
    code = (
        "import numpy as np, time, os; b = np.ones(200 * 1024**2 // 8); time.sleep(3.5); "
        "os._exit(0)"  # no atexit, so no measurement line, but a finished stage
    )

    class Unmeasured(Recorder):
        def __call__(self, args, output, append, env):
            if str(args[1]) == "covsyn.calibration.firefly_optimizer":
                pr.run_module(["-c", code], None, False, env)
            return super().__call__(args, output, append, env)

    _, first = run_pipeline(tmp_path, config, Unmeasured())
    line = run_log(first)["optimise"]
    assert (line["steps"], line["steps_measured"]) == ([OPTIMISER_STEP], [])  # killed unmeasured
    second = first.parent / "20261116-090000_b30c141_phase-d"
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(100e6))
    assert prun(config, second, execute=Recorder()) == 0
    assert (
        "stage optimise: no complete earlier measurement" in (second / pr.PIPELINE_LOG).read_text()
    )


def test_a_stopped_stage_is_not_used_next_time(tmp_path, config, monkeypatch):
    """Only finished stages are estimates; a stopped one is reported, not used."""
    status, first = run_pipeline(
        tmp_path, config, Measuring({"covsyn.calibration.firefly_optimizer": (9000.0, 9000.0)},
                                    fail={"covsyn.calibration.firefly_optimizer"})
    )  # fmt: skip
    assert status == 1
    second = first.parent / "20261116-090000_b30c141_phase-d"
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(100e6))
    assert prun(config, second, execute=Recorder()) == 0


def test_run_log_says_whether_a_stage_was_measured_whole(tmp_path, config):
    """Steps run, steps measured, whether it ran only the rest, and a time with its offset."""
    peaks = {"covsyn.calibration.firefly_optimizer": (300.0, 900.0)}
    _, run_dir = run_pipeline(tmp_path, config, Measuring(peaks))
    line = run_log(run_dir)["optimise"]
    assert (line["steps"], line["steps_measured"], line["partial"]) == (
        [OPTIMISER_STEP],
        [OPTIMISER_STEP],
        False,
    )
    assert pr.complete(line)
    assert pr.finished_at(line).utcoffset() is not None


def test_a_resumed_optimiser_measured_only_the_rest(tmp_path, config, monkeypatch):
    """Continued from a checkpoint, its line is partial: not taken for the whole stage."""
    peaks = {"covsyn.calibration.firefly_optimizer": (300.0, 900.0)}
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.calibration.firefly_optimizer"})
    )
    assert status == 1
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "generation_0000.npz").write_text("x")  # one to continue from
    table = tmp_path / "results" / "tables" / run_dir.name / "progress.csv"
    table.write_text("generation,best_cost\n0,1.0\n")  # its generation's row (Task 4)
    assert prun(None, run_dir, execute=Measuring(peaks), resume=True) == 0
    line = run_log(run_dir)["optimise"]
    assert line["partial"] is True and not pr.complete(line)


@pytest.mark.parametrize("finished", [None, "unknown", "2026-13-40T99:00:00", 5])
def test_a_line_without_a_valid_time_is_skipped_and_said(tmp_path, config, monkeypatch,
                                                         finished):  # fmt: skip
    """Not sorted as text, where "unknown" would come after every date."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 7000.0, 1000.0, finished)],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 100.0, 100.0, "2026-11-02T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 0  # the valid 100 + 2 x 100 decides
    assert (
        "20261101-000000_aaaaaaa_phase-d's optimise has no valid finished time"
        in (run_dir / pr.PIPELINE_LOG).read_text()
    )


def test_equal_times_are_ordered_by_run_id(tmp_path, config, monkeypatch):
    """The same second in two runs: the larger run id, the later run, decides."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 100.0, 100.0, "2026-11-02T10:00:00+08:00")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 7000.0, 1000.0, "2026-11-02T10:00:00+08:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1
    assert "run 20261102-000000_aaaaaaa_phase-d" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_times_in_other_zones_keep_their_order(tmp_path, config, monkeypatch):
    """10:00 in Taipei came before 09:00 in London that day."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 7000.0, 1000.0, "2026-11-02T09:00:00+00:00")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 100.0, 100.0, "2026-11-02T10:00:00+08:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 1  # London's, the later, decides


OPTIMISER_STEP = "-m covsyn.calibration.firefly_optimizer"


def _line(stage, parent, worker, finished, steps=(OPTIMISER_STEP,),
          measured=(OPTIMISER_STEP,), partial=False, workers=8):  # fmt: skip
    """A run.log line of Task 5's format; whole, so usable, unless told otherwise."""
    return {"stage": stage, "parent_peak_mb": parent, "worker_peak_mb": worker,
            "pipeline_peak_mb": 50.0, "finished": finished, "steps": list(steps),
            "steps_measured": list(measured), "partial": partial, "workers": workers}  # fmt: skip


def _earlier(tmp_path, config, name, finished=()):
    """A run beside the next one on the same configuration, with these run.log lines."""
    run = tmp_path / "results" / "models" / name
    (run / "logs").mkdir(parents=True)
    (run / pr.SAVED_CONFIG).write_text(pc.to_toml(config))
    (run / pr.RUN_LOG).write_text("".join(json.dumps(x) + "\n" for x in finished))
    return run


def test_the_latest_finished_line_is_found_by_its_time_not_its_files(tmp_path, config, monkeypatch):
    """A later finished stage wins over an earlier one, however the files were touched."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 100.0, 100.0, "2026-11-03T10:00:00")],
    )
    old = _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 7000.0, 1000.0, "2026-11-01T10:00:00")],
    )
    (old / pr.RUN_LOG).touch()  # newest file, oldest line
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0  # 100 + 2 x 100 fits; the 2026-11-03 line is the latest


def test_a_linked_run_or_config_is_not_looked_into(tmp_path, config, monkeypatch):
    """A link in results/models, to a dead share say, is skipped without following it.

    So is a run whose pipeline.toml is a link.
    """
    models = tmp_path / "results" / "models"
    (models / "real").mkdir(parents=True)
    (models / "old").symlink_to(tmp_path / "gone")
    (models / "real" / pr.SAVED_CONFIG).symlink_to(tmp_path / "gone.toml")
    # A run with a real configuration whose logs are links.
    logged = models / "logged"
    logged.mkdir()
    (logged / pr.SAVED_CONFIG).write_text(pc.to_toml(config))
    (logged / pr.RUN_LOG).symlink_to(tmp_path / "gone.log")
    (logged / "logs").symlink_to(tmp_path / "gone")
    followed = []

    def watching(function):
        def watched(path=".", *args, **kwargs):
            # lstat and stat(follow_symlinks=False) look at a link without following it.
            follows = function.__name__ != "lstat" and kwargs.get("follow_symlinks", True)
            parts = set(Path(str(path)).parts)
            linked = {"old", "gone", "gone.toml", "gone.log"} & parts or (
                ("real" in parts and str(path).endswith(pr.SAVED_CONFIG))
                or ("logged" in parts and (str(path).endswith(pr.RUN_LOG) or "logs" in parts))
            )
            if follows and linked:
                followed.append((function.__name__, str(path)))
            return function(path, *args, **kwargs)

        return watched

    run_dir = tmp_path / "results" / "models" / "20261115-090000_b30c141_phase-d"
    run_dir.mkdir()
    (run_dir / pr.SAVED_CONFIG).write_text(pc.to_toml(config))
    assert "logged" in [p.name for p in pr.other_runs(run_dir)]  # its logs are what is tested
    (run_dir / pr.SAVED_CONFIG).unlink()
    run_dir.rmdir()
    for name in ("scandir", "stat", "lstat"):
        monkeypatch.setattr(os, name, watching(getattr(os, name)))
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0 and followed == []


def test_a_later_finished_stage_without_its_measurement_does_not_drop_an_earlier_one(
    tmp_path, config, monkeypatch
):
    """A later stage not measured whole is skipped; the earlier whole one is used."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 3000.0, "2026-11-01T10:00:00")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        # two steps ran, one measured: not whole, so not used
        finished=[
            _line(
                "optimise", 600.0, None, "2026-11-02T10:00:00", steps=(OPTIMISER_STEP, "-m other")
            )
        ],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "4")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1  # 500 + 4 x 3000, from the run that measured
    assert (
        "the peaks of run 20261101-000000_aaaaaaa_phase-d on this"
        in (run_dir / pr.PIPELINE_LOG).read_text()
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_another_runs_unreadable_log_stops_nothing(tmp_path, config, monkeypatch):
    """A sibling's run.log that is a directory, or unreadable, is skipped, not fatal.

    The unreadable one would refuse the stage if it were read.
    """
    unreadable = _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 9000.0, 0.0, "2026-11-01T10:00:00")],
    )
    (unreadable / pr.RUN_LOG).chmod(0)
    folder = _earlier(tmp_path, config, "20261102-000000_aaaaaaa_phase-d")
    (folder / pr.RUN_LOG).unlink()
    (folder / pr.RUN_LOG).mkdir()
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_one_unreadable_configuration_skips_only_its_run(tmp_path, config, monkeypatch):
    """The other runs still count, so the check is not left without any."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 7000.0, 1000.0, "2026-11-01T10:00:00")],
    )
    hidden = _earlier(tmp_path, config, "20261102-000000_aaaaaaa_phase-d")
    (hidden / pr.SAVED_CONFIG).chmod(0)
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 1  # 7000 + 2 x 1000, from the readable run


def test_a_finished_resume_lists_no_other_run(tmp_path, config, monkeypatch):
    """With every stage done, nothing beside the run is read."""
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 0
    calls = []
    monkeypatch.setattr(pr, "other_runs", lambda path: calls.append(path) or [])
    assert prun(None, run_dir, execute=Recorder(), resume=True) == 0
    assert calls == []


def test_a_resume_without_a_checkpoint_measures_the_whole_optimiser(tmp_path, config):
    """Killed before its first checkpoint, the optimiser starts again: not partial."""
    peaks = {"covsyn.calibration.firefly_optimizer": (300.0, 900.0)}
    status, run_dir = run_pipeline(
        tmp_path, config, Measuring(peaks, fail={"covsyn.calibration.firefly_optimizer"})
    )
    assert status == 1
    assert prun(None, run_dir, execute=Measuring(peaks), resume=True) == 0
    line = run_log(run_dir)["optimise"]
    assert line["partial"] is False and pr.complete(line)


def test_an_older_pooled_line_counts_beside_a_later_serial_one(tmp_path, config, monkeypatch):
    """A run that spawned no pool says nothing of one with a pool: the pooled one counts too.

    Its larger need decides.
    """
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 900.0, "2026-11-01T10:00:00", workers=8)],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, None, "2026-11-02T10:00:00", workers=1)],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(5e9))
    monkeypatch.setenv("COVSYN_WORKERS", "8")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1  # 500 + 8 x 900, from the run with a pool
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "run 20261101-000000_aaaaaaa_phase-d" in log


def test_one_step_measured_twice_does_not_stand_in_for_another(tmp_path, config, monkeypatch):
    """The steps are matched by name, not counted."""
    twice = (OPTIMISER_STEP, OPTIMISER_STEP)
    line = _line("optimise", 500.0, 900.0, "2026-11-01T10:00:00",
                 steps=(OPTIMISER_STEP, "-m other"), measured=twice)  # fmt: skip
    _earlier(tmp_path, config, "20261101-000000_aaaaaaa_phase-d", finished=[line])
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(1e9))
    monkeypatch.setenv("COVSYN_WORKERS", "8")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0  # not complete, so not used


def test_a_stage_turned_off_is_not_said_to_be_unchecked(tmp_path, config):
    """Revalidation off: it runs no step, so it is not reported; a stage that runs still is."""
    import dataclasses

    off = dataclasses.replace(config, revalidate=False)
    status, run_dir = run_pipeline(tmp_path, off, Recorder())
    assert status == 0
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "stage revalidate: no complete" not in log and "stage select: no complete" not in log
    assert "stage optimise: no complete" in log


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_peak_that_is_no_finite_number_is_not_used(tmp_path, config, monkeypatch, bad):
    """NaN or Infinity in a run.log line refuses nothing and is not used."""
    run = _earlier(tmp_path, config, "20261101-000000_aaaaaaa_phase-d")
    (run / pr.RUN_LOG).write_text(json.dumps(_line("optimise", bad, 900.0, "2026-11-01T10:00:00")))
    monkeypatch.setenv("COVSYN_WORKERS", "2")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 0


def test_a_stage_that_never_had_a_pool_is_estimated_from_its_step(tmp_path, config, monkeypatch):
    """No complete line has a worker peak: the latest one's step peak is the need."""
    _earlier(tmp_path, config, "20261101-000000_aaaaaaa_phase-d",
             finished=[_line("report", 9000.0, None, "2026-11-01T10:00:00", steps=("-m r",),
                             measured=("-m r",))])  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1
    assert "stage report needs about 9000 MB" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_an_unusable_pooled_line_does_not_drop_a_usable_pool_less_one(
    tmp_path, config, monkeypatch
):
    """A pooled line without a valid time cannot be used, so the pool-less one is."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 900.0, "garbage")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 9000.0, None, "2026-11-02T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1
    assert "run 20261102-000000_aaaaaaa_phase-d" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_a_turned_off_stage_is_not_checked_at_all(tmp_path, config, monkeypatch):
    """Its old lines cannot refuse it, as it runs nothing."""
    import dataclasses

    _earlier(tmp_path, dataclasses.replace(config, revalidate=False),
             "20261101-000000_aaaaaaa_phase-d",
             finished=[_line("revalidate", 9e9, 9e9, "2026-11-01T10:00:00")])  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    status, _ = run_pipeline(tmp_path, dataclasses.replace(config, revalidate=False), Recorder())
    assert status == 0


def test_a_later_serial_run_that_needed_more_decides(tmp_path, config, monkeypatch):
    """Serial, all the work in the step: its 9000 MB beat the older pool's 500 + 8 x 900."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 900.0, "2026-11-01T10:00:00")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 9000.0, None, "2026-11-02T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "8")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1
    assert "run 20261102-000000_aaaaaaa_phase-d" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_select_turned_off_is_not_checked(tmp_path, config, monkeypatch):
    """Revalidation on, selection off: select runs nothing, and later stages read revalidation."""
    import dataclasses

    off = dataclasses.replace(config, select=False)
    _earlier(
        tmp_path,
        off,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("select", 9e9, 9e9, "2026-11-01T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    recorder = Recorder()
    status, run_dir = run_pipeline(tmp_path, off, recorder)
    assert status == 0
    assert "stage select: no complete" not in (run_dir / pr.PIPELINE_LOG).read_text()
    synthesis = [c for c in recorder.calls if c[1] == "covsyn.model.data_synthesis_main"]
    revalidation, selected = str(run_dir / "revalidation"), str(run_dir / "selected")
    assert synthesis and all(revalidation in c for c in synthesis)
    assert not any(arg.startswith(selected) for c in recorder.calls for arg in c)


def test_an_older_serial_line_counts_beside_a_later_pooled_one(tmp_path, config, monkeypatch):
    """This run may run serially too: the older serial 9000 MB beats 500 + 1 x 900."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 9000.0, None, "2026-11-01T10:00:00")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 900.0, "2026-11-02T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "1")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "run 20261101-000000_aaaaaaa_phase-d" in log and "with no pool" in log


def test_two_lines_of_one_run_at_one_time_take_the_larger(tmp_path, config, monkeypatch):
    """Same run, same second: the larger need with this run's workers, not the larger step."""
    _earlier(tmp_path, config, "20261101-000000_aaaaaaa_phase-d",
             finished=[_line("optimise", 7000.0, 100.0, "2026-11-01T10:00:00"),
                       _line("optimise", 100.0, 5000.0, "2026-11-01T10:00:00")])  # fmt: skip
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(10e9))
    monkeypatch.setenv("COVSYN_WORKERS", "8")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1  # 100 + 8 x 5000, not 7000 + 8 x 100
    assert "needs about 40100 MB" in (run_dir / pr.PIPELINE_LOG).read_text()


def test_a_zero_worker_peak_is_no_pool(tmp_path, config, monkeypatch):
    """A worker peak of 0 MB measured no worker: the older pooled line counts too."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 900.0, "2026-11-01T10:00:00")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 0.0, "2026-11-02T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(5e9))
    monkeypatch.setenv("COVSYN_WORKERS", "8")
    status, _ = run_pipeline(tmp_path, config, Recorder())
    assert status == 1


def test_a_step_run_by_may_can_be_partial(tmp_path, config):
    """Partial reaches the stage through may() as well as must()."""
    context = ps.Context(config, ps.Layout(tmp_path / "run"), lambda *a, **k: 0)
    context.may(["-m", "x"], "x", partial=True)
    assert context.partial is True and context.steps == ["-m x"]


def test_fewer_workers_is_advised_only_where_it_could_help(tmp_path, config, monkeypatch):
    """Not with one worker already, nor while a serial need alone does not fit."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 7000.0, None, "2026-11-01T10:00:00")],
    )
    _earlier(
        tmp_path,
        config,
        "20261102-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 500.0, 900.0, "2026-11-02T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(6e9))
    monkeypatch.setenv("COVSYN_WORKERS", "8")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1  # 500 + 8 x 900 decides, but the serial 7000 does not fit either
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "fewer workers will not make it fit" in log and "try fewer workers" not in log


def test_one_worker_is_not_told_to_use_fewer(tmp_path, config, monkeypatch):
    """With one worker there are no fewer to try."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 7000.0, 2000.0, "2026-11-01T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "1")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "stage optimise needs about 9000 MB" in log  # refused by the memory check
    assert "fewer workers will not make it fit" in log and "try fewer workers" not in log


def test_fewer_workers_is_not_advised_when_one_worker_would_not_fit(tmp_path, config, monkeypatch):
    """A pooled step of 9000 MB does not fit in 8000 with any worker count."""
    _earlier(
        tmp_path,
        config,
        "20261101-000000_aaaaaaa_phase-d",
        finished=[_line("optimise", 9000.0, 100.0, "2026-11-01T10:00:00")],
    )
    monkeypatch.setattr(pr.memory, "available_bytes", lambda: int(8e9))
    monkeypatch.setenv("COVSYN_WORKERS", "8")
    status, run_dir = run_pipeline(tmp_path, config, Recorder())
    assert status == 1
    log = (run_dir / pr.PIPELINE_LOG).read_text()
    assert "fewer workers will not make it fit" in log and "try fewer workers" not in log
