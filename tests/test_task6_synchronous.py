# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 6: the synchronous firefly algorithm, one candidate per worker.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 6 (Yang, 2010):

1. From the positions and costs at the start of generation g, every firefly moves towards
   each brighter firefly in a fixed order, without evaluating between moves; the elite, so far
   the brightest firefly, does not move.
2. All n candidates are submitted at once, one task each, and scored on block g; generation 0
   shares block 0 with the initial population, so its elite keeps its initial cost.
3. Results are stored by candidate index, never by completion order.
5. The pool starts its workers by spawn on every system.

Acceptance: runs with 1, 3 and 8 workers give byte-identical progress.csv files, apart from
the timing columns.
"""

from __future__ import annotations

import concurrent.futures
import csv
import functools
import itertools
import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from covsyn.calibration import firefly_optimizer as fo
from covsyn.runtime import runs, timings
from tests import firefly_standins as standins
from tests.conftest import REPO_ROOT

DIM = 208  # decode_metrics reads the progress row from a full vector
POP = 5
GENERATIONS = 4
SEED = 3


def optimise(path, function, workers=2, resume=False):
    """Run the optimiser on a stand-in objective into ``path``."""
    fo.result_path = path
    engine = fo.Firefly(pop_size=POP, alpha_0=1, betamin=1, gamma=0.5, seed=SEED)
    return engine.firefly(function=function, dim=DIM, lb=np.zeros(DIM), ub=np.ones(DIM),
                          demographic_parameters=None, max_generations=GENERATIONS,
                          max_workers=workers, Cheng_contact_array=None,
                          Cheng_attack_rate=None, norm_weights=None, resume=resume)  # fmt: skip


def checkpoint(path, generation):
    """The state a generation's checkpoint holds; -1 is the initial population's."""
    name = "initial" if generation < 0 else f"generation_{generation:04d}"
    with np.load(path / fo.CHECKPOINTS / f"{name}.npz") as saved:
        state = {key: saved[key] for key in saved.files}
    state["cost_parts"] = json.loads(str(state["cost_parts"]))
    return state


def table(path, name):
    """A table's rows from beside progress.csv."""
    with open(runs.progress_path(path).parent / name, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def scrambled_run(tmp_path_factory):
    """A run on 3 workers whose tasks finish out of order, and its directory; read only."""
    path = tmp_path_factory.mktemp("scrambled")
    previous = getattr(fo, "result_path", None)
    try:
        optimise(path, standins.scrambled, workers=3)
    finally:
        fo.result_path = previous
    return path


# -- item 1: moves ---------------------------------------------------------------------------


def _move(x, towards, alpha, betamin, gamma, rng):
    """One move of Yang (2010) in normalised coordinates, r the squared distance (B51)."""
    beta = betamin * np.exp(-gamma * np.sum((x - towards) ** 2))
    return np.clip(x + beta * (towards - x) + alpha * (rng.random(len(x)) - 0.5), 0, 1)


def test_each_firefly_moves_towards_each_brighter_one_from_the_start_positions():
    """Firefly 2 moves towards firefly 0's start position, not its new one, then to 1's."""
    start = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.1]])
    costs = np.array([2.0, 1.0, 3.0])  # 1 brightest, then 0, then 2
    alpha, betamin, gamma = 0.3, 1.0, 2.0
    draws = np.random.default_rng(9)
    want = start.copy()
    want[0] = _move(start[0], start[1], alpha, betamin, gamma, draws)
    want[2] = _move(start[2], start[0], alpha, betamin, gamma, draws)
    want[2] = _move(want[2], start[1], alpha, betamin, gamma, draws)
    got = fo.synchronous_moves(start, costs, alpha, betamin, gamma, np.random.default_rng(9))
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(start, [[0.2, 0.2], [0.8, 0.8], [0.5, 0.1]])  # untouched


def test_the_elite_and_a_firefly_as_bright_as_it_stay():
    """Only a strictly brighter firefly attracts, so a tie, and the brightest, do not move."""
    start = np.array([[0.1, 0.9], [0.4, 0.4], [0.9, 0.2]])
    costs = np.array([1.0, 1.0, 2.0])
    got = fo.synchronous_moves(start, costs, 0.5, 1.0, 1.0, np.random.default_rng(1))
    np.testing.assert_array_equal(got[:2], start[:2])
    assert not np.array_equal(got[2], start[2])


def test_moves_stay_in_the_normalised_box():
    """Every coordinate stays in [0, 1], however large the step."""
    rng = np.random.default_rng(4)
    start = rng.random((8, 5))
    got = fo.synchronous_moves(start, rng.random(8), 50.0, 1.0, 0.1, rng)
    assert got.min() >= 0 and got.max() <= 1


def test_one_firefly_does_not_move():
    """A population of one has nothing brighter to move towards."""
    start = np.array([[0.3, 0.6]])
    got = fo.synchronous_moves(start, np.array([1.0]), 1.0, 1.0, 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(got, start)


def test_the_elite_keeps_its_position_every_generation(scrambled_run):
    """The brightest firefly at the start of a generation is where it was at its end."""
    for generation in range(GENERATIONS):
        before = checkpoint(scrambled_run, generation - 1)
        after = checkpoint(scrambled_run, generation)
        elite = int(np.argmin(before["intensities"]))
        np.testing.assert_array_equal(after["normalized"][elite], before["normalized"][elite])


# -- item 2: evaluation ----------------------------------------------------------------------


def test_each_generation_evaluates_every_candidate_once(scrambled_run):
    """Each generation has n evaluations, generation 0 n - 1: its elite keeps its cost."""
    counts = Counter(int(r["generation"]) for r in table(scrambled_run,
                                                         timings.EVALUATION_TABLE))  # fmt: skip
    assert counts == {-1: POP, 0: POP - 1, **{g: POP for g in range(1, GENERATIONS)}}
    assert int(checkpoint(scrambled_run, GENERATIONS - 1)["evaluations"]) == sum(counts.values())
    initial, first = checkpoint(scrambled_run, -1), checkpoint(scrambled_run, 0)
    elite = int(np.argmin(initial["intensities"]))
    assert first["intensities"][elite] == initial["intensities"][elite]


def test_every_candidate_of_a_generation_is_scored_on_its_block(scrambled_run):
    """Common random numbers: block g for all of generation g, block 0 for generation 0."""
    for generation in range(-1, GENERATIONS):
        block = fo.generation_seed_offset(max(generation, 0))
        offsets = {p["scored_offset"] for p in checkpoint(scrambled_run, generation)["cost_parts"]}
        assert offsets == {block}, generation


# -- item 3: collection ----------------------------------------------------------------------


def test_costs_are_stored_by_candidate_not_by_completion(scrambled_run):
    """Each firefly's cost is its own vector's, though the tasks finished out of order."""
    for generation in range(-1, GENERATIONS):
        state = checkpoint(scrambled_run, generation)
        block = fo.generation_seed_offset(max(generation, 0))
        want = [standins.cost(vector, block) for vector in state["fireflies"]]
        np.testing.assert_array_equal(state["intensities"], want)
    rows = table(scrambled_run, timings.EVALUATION_TABLE)
    per_generation = {}
    for r in rows:
        per_generation.setdefault(r["generation"], []).append(int(r["candidate"]))
    assert all(c == sorted(c) for c in per_generation.values())


# -- item 5: spawn ---------------------------------------------------------------------------


def test_the_pool_starts_its_workers_by_spawn(tmp_path, monkeypatch):
    """The same start method on macOS and Linux."""
    contexts = []
    real = concurrent.futures.ProcessPoolExecutor

    def recording(*args, **kwargs):
        contexts.append(kwargs.get("mp_context"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fo.concurrent.futures, "ProcessPoolExecutor", recording)
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    optimise(tmp_path, standins.objective)
    assert [c.get_start_method() for c in contexts] == ["spawn"]


_POOL = concurrent.futures.ProcessPoolExecutor  # the real one, which the tests below replace


class _Spy:
    """A ProcessPoolExecutor that records its worker count, submissions and collections."""

    events: list = []
    workers: list = []

    def __init__(self, *args, **kwargs):
        self.pool = _POOL(*args, **kwargs)
        _Spy.workers.append(kwargs.get("max_workers", args[0] if args else None))

    def submit(self, *args, **kwargs):
        _Spy.events.append("submit")
        return _Collected(self.pool.submit(*args, **kwargs))

    def shutdown(self, *args, **kwargs):
        return self.pool.shutdown(*args, **kwargs)


class _Collected:
    """A future whose collection is recorded."""

    def __init__(self, future):
        self.future = future

    def result(self, *args, **kwargs):
        _Spy.events.append("result")
        return self.future.result(*args, **kwargs)


def test_a_generation_s_candidates_are_all_in_flight_before_any_is_collected(tmp_path, monkeypatch):
    """A generation's tasks all go out before any result is collected.

    They go to a pool built for the workers asked for; which candidates they score,
    test_costs_are_stored_by_candidate_not_by_completion checks.
    """
    monkeypatch.setattr(_Spy, "events", [])
    monkeypatch.setattr(_Spy, "workers", [])
    monkeypatch.setattr(fo.concurrent.futures, "ProcessPoolExecutor", _Spy)
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    optimise(tmp_path, standins.objective, workers=3)
    assert _Spy.workers == [3]
    runs_of = [len(list(group)) for _, group in itertools.groupby(_Spy.events)]
    # initial population, generation 0 without its elite, then every generation
    assert runs_of == [POP, POP, POP - 1, POP - 1] + [POP, POP] * (GENERATIONS - 1)


# -- a resumed run ---------------------------------------------------------------------------


def test_a_run_from_before_task_6_cannot_be_resumed(tmp_path):
    """Its moves were the sequential ones; continuing would mix the two algorithms."""
    config = {"seed": 5, "settings": {"mode": "train"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit) as stop:
        fo.resumed_settings(tmp_path)
    assert stop.value.code == runs.NOT_RESUMABLE


def test_a_run_records_its_moves(tmp_path, monkeypatch):
    """config.json says the run moves synchronously, which a resume requires."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    optimise(tmp_path, standins.objective)
    settings = json.loads((tmp_path / "config.json").read_text())["settings"]
    assert settings["moves"] == fo.MOVES


def test_no_other_setting_can_change_the_recorded_moves(tmp_path, monkeypatch):
    """The moves config.json records are the algorithm's, whatever the run's own settings say."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    monkeypatch.setattr(fo.Firefly, "__init__", _with_settings(fo.Firefly.__init__))
    optimise(tmp_path, standins.objective)
    assert json.loads((tmp_path / "config.json").read_text())["settings"]["moves"] == fo.MOVES


def _with_settings(init):
    """Firefly.__init__, then run settings that name moves."""

    def wrapped(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.run_settings = {"moves": "sequential"}

    return wrapped


def test_a_kill_during_an_evaluation_ends_the_run_with_its_error(tmp_path, monkeypatch):
    """A task that raises stops the run, and the generation is left without a checkpoint."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    block = fo.generation_seed_offset(2)
    with pytest.raises(standins.Killed):
        optimise(tmp_path, functools.partial(standins.killed_on, block_offset=block))
    assert runs.last_checkpoint(tmp_path).name == "generation_0001.npz"


# -- acceptance ------------------------------------------------------------------------------


def _test_mode_run(tmp_path, workers):
    """covsyn-optimise --mode test on ``workers`` workers; its progress.csv and evaluations."""
    run = tmp_path / f"workers_{workers}"
    subprocess.run(
        [sys.executable, "-m", "covsyn.calibration.firefly_optimizer", "--mode", "test",
         "--seed", "5", "--workers", str(workers), "--run-dir", str(run)],
        cwd=REPO_ROOT, check=True, timeout=1800,
    )  # fmt: skip
    return runs.progress_path(run).read_bytes(), table(run, timings.EVALUATION_TABLE)


@pytest.mark.slow
def test_the_command_gives_the_same_progress_table_on_1_and_3_workers(tmp_path):
    """covsyn-optimise --mode test, on the real objective at S = 300, as a user runs it."""
    one, evaluations = _test_mode_run(tmp_path, 1)
    assert one.count(b"\n") > 1
    assert all(len(r["worker_pids"].split(";")) == 1 for r in evaluations)
    assert all(int(r["simulations"]) == fo.SIMULATIONS_PER_EVALUATION for r in evaluations)
    assert _test_mode_run(tmp_path, 3)[0] == one


WIDE = 9  # more fireflies than the 8 workers, so all 8 can hold a candidate at once


def _real_run(path, workers, run10_vector, demographic_parameters, cheng_data):
    """WIDE fireflies within 5% of run 10's vector on the real objective at S = 20."""
    from covsyn.calibration.fast_cost import evaluate_candidate

    fo.result_path = path
    engine = fo.Firefly(pop_size=WIDE, alpha_0=0.2, betamin=1, gamma=0.03, seed=SEED)
    engine.simulations = 20
    engine.firefly(function=evaluate_candidate, dim=len(run10_vector),
                   lb=np.minimum(run10_vector * 0.95, run10_vector * 1.05),
                   ub=np.maximum(run10_vector * 0.95, run10_vector * 1.05),
                   demographic_parameters=demographic_parameters, max_generations=3,
                   max_workers=workers, Cheng_contact_array=cheng_data[0],
                   Cheng_attack_rate=cheng_data[1], norm_weights=cheng_data[2])  # fmt: skip
    return runs.progress_path(path).read_bytes(), table(path, timings.EVALUATION_TABLE)


@pytest.mark.slow
def test_1_3_and_8_workers_give_the_same_progress_table(
    tmp_path, monkeypatch, run10_vector, demographic_parameters, cheng_data
):
    """The acceptance criterion, on the real objective; progress.csv has no timing column."""
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    tables = {}
    for workers in (1, 3, 8):
        path = tmp_path / f"workers_{workers}"
        path.mkdir()
        tables[workers], evaluations = _real_run(path, workers, run10_vector,
                                                 demographic_parameters, cheng_data)  # fmt: skip
        used = len({r["worker_pids"] for r in evaluations})
        # With more fireflies than workers, all 8 can hold a candidate at once. How many get one
        # depends on how soon each is up -- a spawned worker takes about 10 s to import on WSL
        # and a task 50 ms; 6 did there -- so the count is printed, not checked.
        print(f"{workers} workers: {used} ran a candidate")
        assert used == 1 if workers == 1 else used <= workers
    assert tables[1].count(b"\n") == 4  # the header and 3 generations
    assert tables[3] == tables[1] and tables[8] == tables[1]
