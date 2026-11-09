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


def test_a_cached_cost_parts_file_with_other_stages_is_not_reused(tmp_path) -> None:
    """A cache from a B81 run has no Best rows; a pre-B81 fig4 into the same place recomputes."""
    cache = tmp_path / "wu2025_fig4_cost_parts.csv"
    assert wu.read_cost_parts_cache(cache, {"Initial", "Final"}) is None
    wu._write_csv(cache, [{"stage": s, "firefly": 0, "total": 1.0} for s in ("Initial", "Final")])
    rows = wu.read_cost_parts_cache(cache, {"Initial", "Final"})
    assert [r["stage"] for r in rows] == ["Initial", "Final"]
    assert wu.read_cost_parts_cache(cache, {"Initial", "Best", "Final"}) is None


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


def test_the_cost_parts_cache_lives_in_its_run_and_is_named_by_its_vectors(tmp_path):
    """Another optimizer run, or the same run's other stages, never reuses this cache."""
    one = {"Initial": np.zeros((2, 3)), "Final": np.ones((2, 3))}
    other = {"Initial": np.zeros((2, 3)), "Final": np.full((2, 3), 2.0)}
    path = wu.cost_parts_cache_path(tmp_path, one, "v1")
    assert path.parent == tmp_path and path.name.startswith("wu2025_fig4_cost_parts_")
    assert path == wu.cost_parts_cache_path(tmp_path, dict(one), "v1")
    assert path != wu.cost_parts_cache_path(tmp_path, other, "v1")
    assert path != wu.cost_parts_cache_path(tmp_path, one, "v2")


def _package(root, source: bytes):
    package = root / "covsyn"
    (package / "calibration").mkdir(parents=True)
    (package / "calibration" / "fast_cost.py").write_bytes(source)
    data = root / "variable"
    data.mkdir()
    (data / "demographic_parameters.pkl").write_bytes(b"data")
    return package, data


def test_the_cost_version_changes_with_any_module_or_data_file(tmp_path):
    """Any module the objective imports, or any model data file, changes the version."""
    package, data = _package(tmp_path, b"WEIGHT = 1\n")
    before = wu.cost_version(package, data)
    (package / "calibration" / "fast_cost.py").write_bytes(b"WEIGHT = 2\n")
    after_code = wu.cost_version(package, data)
    (data / "demographic_parameters.pkl").write_bytes(b"other data")
    assert len({before, after_code, wu.cost_version(package, data)}) == 3


def test_the_cost_version_ignores_line_endings(tmp_path):
    """A Windows checkout of a commit gives the same version as a Mac checkout."""
    lf_root, crlf_root = tmp_path / "lf", tmp_path / "crlf"
    lf = wu.cost_version(*_package(lf_root, b"A = 1\nB = 2\n"))
    crlf = wu.cost_version(*_package(crlf_root, b"A = 1\r\nB = 2\r\n"))
    assert lf == crlf


def test_the_cost_version_needs_the_model_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        wu.cost_version(tmp_path, tmp_path / "variable")


def test_the_cost_version_orders_files_by_their_path_string(tmp_path):
    """Windows orders Path objects ignoring case; the digest must not depend on the platform."""
    import hashlib

    package, data = _package(tmp_path, b"x = 1\n")
    (package / "B.py").write_bytes(b"b\n")
    (package / "a.py").write_bytes(b"a\n")
    expected = hashlib.sha256()
    for name, body in (("B.py", b"b\n"), ("a.py", b"a\n"), ("calibration/fast_cost.py", b"x = 1\n")):
        expected.update(name.encode())
        expected.update(body)
    expected.update(b"demographic_parameters.pkl")
    expected.update(b"data")
    assert wu.cost_version(package, data) == expected.hexdigest()
