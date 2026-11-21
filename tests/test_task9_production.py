# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 9, production run: monitor new elite members and stop when they no longer improve.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 9: use G_plan from pilot 2; every
generation, score each new elite member on the same 5 blocks of the monitoring stream, a member
already scored keeping its score; stop when the best monitored mean has not improved by more
than one standard error over 5 generations, or at G_max = 1,000, and record the reason in
run.log.

Decisions of 2026-11-21: blocks 0-4 of stream 100 at the run's S; at generation g, with
snapshots at g and g - 5, d_b = (cost of g - 5's best on block b) - (cost of g's best on block
b), and the run stops when mean(d) <= SD(d) / sqrt(5), which holds when the best is the same
vector; the optimiser writes the stop to run.log, keeps it in its checkpoint, and a resume
writes it if missing; production requires n, S, G_plan, alpha_0, alpha_end and pilot 2's id;
the monitoring scores only drive the stop, and their cost is recorded apart.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration import monitoring, seed_streams, step_schedule
from covsyn.runtime import runs
from tests import firefly_standins as standins
from tests.test_task7_optimiser import started  # noqa: F401 (a fixture)

DIM = 208


def test_the_monitoring_blocks_are_blocks_0_to_4_of_stream_100():
    """At the run's S, whose block b starts at 100 x 10^7 + b x S."""
    assert (monitoring.STREAM, monitoring.BLOCKS, monitoring.WINDOW) == (100, range(0, 5), 5)
    assert monitoring.offsets(300) == [seed_streams.block_seeds(100, b, 300).start
                                       for b in range(5)]  # fmt: skip


def test_only_vectors_not_yet_scored_are_new_each_once():
    """A member already scored keeps its score; two fireflies at one position are one vector."""
    monitor = monitoring.Monitor(dim=2)
    a, b = np.array([0.1, 0.2]), np.array([0.3, 0.4])
    assert [v.tolist() for v in monitor.new_vectors(np.array([a, b, a]))] == [a.tolist(),
                                                                              b.tolist()]
    monitor.add(a, np.ones(5))
    assert [v.tolist() for v in monitor.new_vectors(np.array([a, b]))] == [b.tolist()]


def test_the_best_is_the_lowest_monitored_mean_ties_to_the_first_scored():
    """Over its five monitoring blocks."""
    monitor = monitoring.Monitor(dim=1)
    monitor.add(np.array([1.0]), np.full(5, 3.0))
    monitor.add(np.array([2.0]), np.full(5, 2.0))
    monitor.add(np.array([3.0]), np.full(5, 2.0))
    assert monitor.best() == 1


def _monitor_with_bests(costs_per_generation):
    """A monitor whose best at each generation has the given 5 costs."""
    monitor = monitoring.Monitor(dim=1)
    for g, costs in enumerate(costs_per_generation):
        monitor.add(np.array([float(g)]), np.asarray(costs, dtype=float))
        monitor.record_generation()
    return monitor


def test_no_stop_before_five_generations_of_snapshots():
    """The rule compares generation g with g - 5, so g >= 5."""
    monitor = _monitor_with_bests([np.full(5, 10.0)] * 5)  # generations 0-4: no g - 5 yet
    assert monitor.stop_reason() is None


def test_the_same_best_for_five_generations_stops():
    """No improvement and no SE: mean(d) = 0 <= 0."""
    monitor = monitoring.Monitor(dim=1)
    monitor.add(np.array([0.0]), np.full(5, 10.0))
    for _ in range(6):
        monitor.record_generation()
    assert "5 generations" in monitor.stop_reason()


def test_an_improvement_beyond_one_paired_se_continues():
    """mean(d) > SD(d) / sqrt(5)."""
    old = np.array([10.0, 11.0, 9.0, 10.5, 9.5])
    new = old - np.array([2.0, 2.1, 1.9, 2.0, 2.0])  # mean 2, SE about 0.03
    monitor = _monitor_with_bests([old] + [old + 1] * 4 + [new])
    assert monitor.stop_reason() is None


def test_an_improvement_within_one_paired_se_stops():
    """mean(d) <= SD(d) / sqrt(5)."""
    old = np.array([10.0, 11.0, 9.0, 10.5, 9.5])
    new = old - np.array([3.0, -2.0, 2.5, -2.5, 0.5])  # mean 0.3, SE about 1.2
    monitor = _monitor_with_bests([old] + [old + 1] * 4 + [new])
    reason = monitor.stop_reason()
    d = old - new
    assert d.mean() <= d.std(ddof=1) / np.sqrt(5) and reason is not None


def test_the_state_goes_into_a_checkpoint_and_back():
    """So a resumed run stops where an uninterrupted one would."""
    monitor = _monitor_with_bests([np.arange(5.0) + g for g in range(7)])
    again = monitoring.Monitor.from_state(monitor.state(), dim=1)
    np.testing.assert_array_equal(again.state()["monitored"], monitor.state()["monitored"])
    np.testing.assert_array_equal(again.state()["monitor_best"], monitor.state()["monitor_best"])
    assert again.stop_reason() == monitor.stop_reason()


# --- in the optimiser, on the stand-in objective ----------------------------------------------


def _engine(seed=3):
    engine = fo.Firefly(pop_size=6, alpha_0=0.2, betamin=1, gamma=0.03, seed=seed, g_plan=20)
    engine.monitor = True
    return engine


def _optimise(engine, generations=40, resume=False, function=standins.objective):
    return engine.firefly(function=function, dim=DIM, lb=np.zeros(DIM), ub=np.ones(DIM),
                          demographic_parameters=None, max_generations=generations,
                          max_workers=2, Cheng_contact_array=None, Cheng_attack_rate=None,
                          norm_weights=None, resume=resume)  # fmt: skip


def _last(path):
    last = runs.last_checkpoint(path)
    with np.load(last) as saved:
        return {key: saved[key] for key in saved.files}


@pytest.fixture(scope="module")
def stopped(tmp_path_factory):
    """A monitored run that stopped before its 40 generations; read only."""
    path = tmp_path_factory.mktemp("production")
    previous = getattr(fo, "result_path", None)
    fo.result_path = path
    try:
        _optimise(_engine())
    finally:
        fo.result_path = previous
    return path


def test_a_monitored_run_stops_and_says_why_in_run_log(stopped):
    """One event line, its generation and reason as the checkpoint holds them."""
    last = _last(stopped)
    assert int(last["generation"]) < 39 and str(last["stop_reason"])
    events = [line for line in runs.json_lines(stopped / "run.log") if "event" in line]
    assert len(events) == 1
    assert events[0]["event"] == "optimiser_stopped"
    assert events[0]["generation"] == int(last["generation"])
    assert events[0]["reason"] == str(last["stop_reason"])
    assert "stage" not in events[0]  # the pipeline's finished stages are its own lines


def test_the_monitored_scores_are_the_objective_on_the_monitoring_blocks(stopped):
    """Each monitored vector's five costs, on blocks 0-4 of stream 100."""
    last = _last(stopped)
    monitored = last["monitored"]
    for row in monitored:
        vector, costs = row[:DIM], row[DIM:]
        want = [standins.cost(vector, o) for o in monitoring.offsets(fo.SIMULATIONS_PER_EVALUATION)]
        np.testing.assert_allclose(costs, want)


def test_every_new_elite_member_was_monitored(stopped):
    """Each vector the archive holds, every elite member there was, has its five scores."""
    last = _last(stopped)
    monitored = {row[:DIM].tobytes() for row in last["monitored"]}
    records = last["elite_records"]
    next_elite = {r[fo.RECORD_PARAMS:].tobytes() for r in records}
    assert monitored <= next_elite and monitored  # the elite after each generation


def test_the_monitoring_cost_is_recorded_apart(stopped):
    """monitoring.json in the run's tables: vectors, evaluations, simulations and seconds."""
    summary = json.loads((runs.progress_path(stopped).parent / monitoring.SUMMARY_FILE).read_text())
    last = _last(stopped)
    assert summary["vectors"] == len(last["monitored"])
    assert summary["evaluations"] == 5 * summary["vectors"]
    assert summary["simulations"] == summary["evaluations"] * fo.SIMULATIONS_PER_EVALUATION
    assert summary["seconds"] >= 0 and summary["stop_reason"] == str(last["stop_reason"])


def test_a_stopped_run_resumed_runs_no_more_and_logs_the_stop_once(stopped, tmp_path,
                                                                  monkeypatch):
    """Its stop is in its checkpoint, so the resume ends there, and run.log has it already."""
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(stopped, copy)
    monkeypatch.setattr(fo, "result_path", copy, raising=False)
    before = _last(copy)
    _optimise(_engine(), resume=True)
    assert int(_last(copy)["generation"]) == int(before["generation"])
    events = [line for line in runs.json_lines(copy / "run.log") if "event" in line]
    assert len(events) == 1


def test_a_resume_writes_a_stop_that_run_log_lost(stopped, tmp_path, monkeypatch):
    """Killed between the checkpoint and the log line: the resume writes it."""
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(stopped, copy)
    (copy / "run.log").unlink()
    monkeypatch.setattr(fo, "result_path", copy, raising=False)
    _optimise(_engine(), resume=True)
    events = [line for line in runs.json_lines(copy / "run.log") if "event" in line]
    assert len(events) == 1 and events[0]["reason"] == str(_last(copy)["stop_reason"])


def test_a_run_that_reaches_g_max_says_so(tmp_path, monkeypatch):
    """Its last generation, max_generations - 1, and G_max as the reason."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    monkeypatch.setattr(monitoring, "WINDOW", 50)  # the rule cannot stop it in 4 generations
    _optimise(_engine(), generations=4)
    events = [line for line in runs.json_lines(tmp_path / "run.log") if "event" in line]
    assert len(events) == 1 and "G_max" in events[0]["reason"]
    assert events[0]["generation"] == 3


def test_a_killed_monitored_run_resumes_to_the_same_result(tmp_path, monkeypatch):
    """The monitored scores and snapshots are in the checkpoint, so the stop is the same."""
    whole, killed = tmp_path / "whole", tmp_path / "killed"
    whole.mkdir(), killed.mkdir()
    monkeypatch.setattr(fo, "result_path", whole, raising=False)
    _optimise(_engine())
    stop_generation = int(_last(whole)["generation"])
    assert stop_generation >= 3
    monkeypatch.setattr(fo, "result_path", killed, raising=False)
    draw = fo.generation_rng

    def stopping(seed, g):
        if g == 3:
            raise standins.Killed
        return draw(seed, g)

    monkeypatch.setattr(fo, "generation_rng", stopping)
    with pytest.raises(standins.Killed):
        _optimise(_engine())
    monkeypatch.setattr(fo, "generation_rng", draw)
    _optimise(_engine(), resume=True)
    a, b = _last(whole), _last(killed)
    for key in ("fireflies", "intensities", "monitored", "monitor_best", "elite_records"):
        np.testing.assert_array_equal(a[key], b[key])
    assert str(a["stop_reason"]) == str(b["stop_reason"])


def test_an_unmonitored_run_is_as_before(tmp_path, monkeypatch):
    """Train, test, profile and pilot runs neither monitor nor stop early."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    engine = fo.Firefly(pop_size=4, alpha_0=0.2, betamin=1, gamma=0.03, seed=3)
    engine.firefly(function=standins.objective, dim=DIM, lb=np.zeros(DIM), ub=np.ones(DIM),
                   demographic_parameters=None, max_generations=3, max_workers=2,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None)  # fmt: skip
    last = _last(tmp_path)
    assert int(last["generation"]) == 2 and "monitored" not in last
    assert not (tmp_path / "run.log").exists()


# --- the command ------------------------------------------------------------------------------

PRODUCTION = ("--mode", "production", "--pop-size", "48", "--simulations", "3000",
              "--g-plan", "207", "--alpha-0", "0.2", "--alpha-end", "0.001",
              "--pilot", "20261201-000000_abcdef0_pilot-size")  # fmt: skip


def _without(flag):
    args = list(PRODUCTION)
    i = args.index(flag)
    return args[:i] + args[i + 2:]


def test_a_production_run_takes_pilot_2_s_settings_and_runs_to_g_max(started):  # noqa: F811
    """n, S, G_plan, alpha_0, alpha_end and pilot 2's id; at most 1,000 generations."""
    seen = started(*PRODUCTION)
    engine = seen["engine"]
    assert (engine.pop_size, engine.simulations, seen["max_generations"]) == (48, 3000, 1000)
    assert (engine.alpha_0, engine.alpha_end, engine.g_plan) == (0.2, 0.001, 207)
    assert engine.monitor is True
    assert engine.run_settings["pilot"] == "20261201-000000_abcdef0_pilot-size"


@pytest.mark.parametrize("flag", ["--pop-size", "--simulations", "--g-plan", "--alpha-0",
                                  "--alpha-end", "--pilot"])
def test_a_production_run_needs_every_setting_pilot_2_chose(monkeypatch, tmp_path, flag):
    """No default stands in for pilot 2's choice (decision of 2026-11-21)."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", *_without(flag),
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match=flag):
        fo.main()
    assert not (tmp_path / "r").exists()


def test_a_production_run_has_no_length_but_g_max(monkeypatch, tmp_path):
    """It stops by its rule or at G_max = 1,000; --generations is the pilot's."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", *PRODUCTION, "--generations", "10",
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="--generations"):
        fo.main()


def test_the_production_mode_s_tables():
    """No default alpha_0; G_max = 1,000 generations."""
    assert step_schedule.MODE_ALPHA_0["production"] is None
    assert step_schedule.MODE_GENERATIONS["production"] == 1000


def test_pilot_is_for_a_production_run_only(monkeypatch, tmp_path):
    """Every other mode refuses it."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--mode", "test", "--pilot", "x",
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="--pilot"):
        fo.main()


def test_a_missing_size_is_named_for_the_production_mode(monkeypatch, tmp_path):
    """Not as the pilot mode's: the user asked for production."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", *_without("--pop-size"),
                                      "--run-dir", str(tmp_path / "r")])  # fmt: skip
    with pytest.raises(SystemExit, match="--mode production needs --pop-size"):
        fo.main()


def test_the_help_names_every_mode_and_where_alpha_0_is_required(monkeypatch, capsys):
    """--help lists pilot and production, and that they require --alpha-0."""
    monkeypatch.setattr(sys, "argv", ["covsyn-optimise", "--help"])
    with pytest.raises(SystemExit):
        fo.main()
    text = " ".join(capsys.readouterr().out.split())
    assert "pilot" in text.split("--mode")[2] and "production" in text.split("--mode")[2]
    assert "required" in text.split("--alpha-0")[2].split("--alpha-end")[0]


class _Future:
    def __init__(self, events, value):
        self.events, self.value = events, value

    def result(self):
        self.events.append("result")
        return self.value


class _Executor:
    """Records the order of submissions and waits."""

    def __init__(self):
        self.events = []

    def submit(self, function, vector, offset):
        self.events.append("submit")
        return _Future(self.events, function(vector, offset))


def test_the_new_members_are_all_submitted_before_any_is_awaited():
    """So the pool scores every new member's blocks at once, not one member at a time."""
    watch = monitoring.Watch(True, 3, None)
    executor = _Executor()
    members = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    watch.score(executor, standins.objective, members, 300)
    assert executor.events == ["submit"] * 15 + ["result"] * 15
    for vector, costs in zip(watch.monitor.vectors, watch.monitor.scores, strict=True):
        np.testing.assert_array_equal(costs, [standins.cost(vector, o)
                                              for o in monitoring.offsets(300)])  # fmt: skip
