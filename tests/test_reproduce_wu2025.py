# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The pieces of the CovSyn-paper figures that were wrong in the notebooks (finding E90)."""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.figures import reproduce_wu2025 as wu


def _case(day: float, incubation: float) -> dict:
    return {"infection_day": day, "incubation_period": incubation}


def test_serial_interval_is_the_onset_difference_not_the_generation_time() -> None:
    """Serial interval is the onset difference not the generation time."""
    courses = [_case(0, 5), _case(3, 2), _case(4, np.nan)]
    digraph = np.array(
        [["nan", "1", "0", "nan"], ["1", "2", "3", "household"], ["2", "3", "4", "school"]],
        dtype="<U32",
    )
    generation, serial = wu.transmission_intervals(courses, digraph)
    assert generation == [3, 1]  # every edge, the last one included
    assert serial == [0]  # onset 5 -> 5; the asymptomatic case has none


def test_time_shift_recovers_a_known_delay() -> None:
    """Time shift recovers a known delay."""
    simulated = np.cumsum(np.r_[np.zeros(20), np.ones(20), np.zeros(40)])
    observed = simulated[12:]  # the same curve, 12 days earlier on its own axis
    observed = np.pad(observed, (0, len(simulated) - len(observed)), mode="edge")
    assert wu.best_time_shift(observed, simulated) == 12


def test_still_in_state_counts_cases_whose_duration_reaches_the_day() -> None:
    """Still in state counts cases whose duration reaches the day."""
    np.testing.assert_allclose(
        wu.still_in_state(np.array([0.0, 2.0, 2.0, 5.0]), 4), [1.0, 0.75, 0.75, 0.25]
    )


def test_a_run_with_compacted_case_ids_is_skipped_not_mispaired() -> None:
    """Case ids that no longer match list positions give no intervals."""
    courses = [_case(0, 5), _case(3, 2)]
    digraph = np.array([["nan", "1", "0", "nan"], ["1", "3", "3", "household"]], dtype="<U32")
    assert wu.transmission_intervals(courses, digraph) == ([], [])


def _optimizer_run(directory, rotated: bool) -> None:
    """The four result files of a 5-firefly, 3-parameter run; under B81 best is the final."""
    rng = np.random.default_rng(0)
    pop, dim = 5, 3
    np.savetxt(directory / "bound.txt", np.array([np.zeros(dim), np.ones(dim)]))
    np.savetxt(directory / "firefly_result_first_initial_guess.txt",
               np.column_stack([rng.random((pop, dim)), rng.random(pop) + 2]))
    final = np.column_stack([rng.random((pop, dim)), rng.random(pop) + 1])
    np.savetxt(directory / "firefly_result.txt", final)
    iteration = np.full(pop, 500.0) if rotated else rng.integers(100, 500, pop).astype(float)
    best = final if rotated else np.column_stack([rng.random((pop, dim)), rng.random(pop) + 1])
    np.savetxt(directory / "firefly_best.txt", np.column_stack([iteration, best]))
    np.savetxt(directory / "firefly_worst.txt",
               np.column_stack([np.full(pop, 50.0), rng.random((pop, dim)), rng.random(pop) + 3]))


def test_a_b81_run_draws_no_separate_best_stage(tmp_path, monkeypatch) -> None:
    """Under B81 firefly_best.txt is the final population, so Best would repeat Final (F2)."""
    plotted = []
    original = wu.plt.Axes.plot

    def spy(self, *args, **kwargs):
        plotted.append(kwargs.get("label"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(wu.plt.Axes, "plot", spy)
    for rotated in (True, False):
        run = tmp_path / f"run_{rotated}"
        run.mkdir()
        _optimizer_run(run, rotated)
        plotted.clear()
        best = np.loadtxt(run / "firefly_best.txt")
        assert wu.best_repeats_final(best, np.loadtxt(run / "firefly_result.txt")) is rotated
        wu.fig4(run, run, evaluate=False)
        assert (run / "wu2025_fig4_firefly_convergence.png").exists()
        assert ("Best fireflies" in plotted) is not rotated


def test_a_b81_run_with_a_failed_evaluation_still_repeats_its_final():
    """A NaN cost from a failed evaluation must not bring the duplicate Best stage back."""
    final = np.array([[0.1, 0.2, 1.5], [0.3, 0.4, np.nan]])
    best = np.column_stack([np.full(2, 500.0), final])
    assert wu.best_repeats_final(best, final)


def test_a_converged_pre_b81_run_keeps_its_best_stage():
    """Bests equal to the final positions, but reached at different iterations: not B81."""
    final = np.array([[0.1, 0.2, 1.5], [0.3, 0.4, 2.5]])
    best = np.column_stack([np.array([120.0, 340.0]), final])
    assert not wu.best_repeats_final(best, final)


def _stub_objective(monkeypatch):
    """fast_cost stand-in: the total and every part are functions of the vector."""
    from covsyn.calibration import fast_cost
    from covsyn.calibration.cost_parts import LAST

    def cost_function(vector, *args):
        LAST.clear()
        LAST.update({key: float(np.sum(vector)) + i for i, (key, _) in enumerate(wu.COST_PARTS[:-1])})
        return float(np.sum(vector))

    monkeypatch.setattr(fast_cost, "init_worker", lambda *a: None)
    monkeypatch.setattr(fast_cost, "cost_function", cost_function)


def test_cost_parts_are_recomputed_even_when_a_previous_table_exists(tmp_path, monkeypatch):
    """No cache: a table left by an earlier run is replaced, its total and every part."""
    _stub_objective(monkeypatch)
    table = tmp_path / "wu2025_fig4_cost_parts.csv"
    stale = {"stage": "Final", "firefly": 0, "total": 999.0}
    stale.update({key: 999.0 for key, _ in wu.COST_PARTS[:-1]})
    wu._write_csv(table, [stale])
    rows = wu.evaluate_cost_parts({"Final": np.array([[1.0, 2.0]])}, table, workers=1)
    assert rows[0]["total"] == 3.0
    assert [rows[0][key] for key, _ in wu.COST_PARTS[:-1]] == [
        3.0 + i for i in range(len(wu.COST_PARTS) - 1)]
    assert "999" not in table.read_text()


def test_fig4_writes_its_tables_beside_the_figure(tmp_path, monkeypatch):
    _stub_objective(monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    _optimizer_run(run, rotated=True)
    out = tmp_path / "figures"
    out.mkdir()
    wu.fig4(run, out, evaluate=True)
    assert (out / "wu2025_fig4_cost_parts.csv").exists()
    assert (out / "wu2025_fig4_reproducibility.csv").exists()
    assert not any(p.name.startswith("wu2025_fig4_cost_parts") for p in run.iterdir())


def test_fig4_without_evaluation_removes_an_earlier_run_s_tables(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _optimizer_run(run, rotated=False)
    out = tmp_path / "figures"
    out.mkdir()
    for name in ("wu2025_fig4_cost_parts.csv", "wu2025_fig4_reproducibility.csv"):
        (out / name).write_text("another run")
    wu.fig4(run, out, evaluate=False)
    assert (out / "wu2025_fig4_firefly_convergence.png").exists()
    assert not (out / "wu2025_fig4_cost_parts.csv").exists()
    assert not (out / "wu2025_fig4_reproducibility.csv").exists()
