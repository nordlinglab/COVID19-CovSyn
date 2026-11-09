# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""F9: the imported factor is solved on m(q) = A + B q, not by q = 12 / m(1)."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def _loaded_script():
    """The script as a module, loaded from its file; cwd and sys.path are restored on exit."""
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)  # the script adds the relative 'src' to sys.path
        mp.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location(
            "calibrate_imported_factor", ROOT / "scripts" / "calibrate_imported_factor.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


@pytest.fixture(scope="module")
def script():
    with _loaded_script() as module:
        yield module


def test_loading_the_script_leaves_sys_path_as_it_was():
    before = list(sys.path)
    with _loaded_script():
        pass
    assert sys.path == before


def test_the_line_through_m0_and_m1_gives_the_b82_factor(script):
    """Run 21: A = 7.27, m(1) = 46.93, data 12; the proportional update gave 0.256."""
    q = script.solve_line(0.0, 7.27, 1.0, 46.93, 12)
    assert q == pytest.approx((12 - 7.27) / (46.93 - 7.27))
    assert q < 12 / 46.93


def test_the_line_through_two_low_measurements_matches_b82(script):
    """B82's measurements: q = 0 gives 7.27 and 0.104 gives 11.86; 0.107 was chosen."""
    assert script.solve_line(0.0, 7.27, 0.104, 11.86, 12) == pytest.approx(0.107, abs=0.002)


def test_measurements_that_do_not_rise_with_q_are_an_error(script):
    with pytest.raises(ValueError):
        script.solve_line(0.0, 10.0, 1.0, 10.0, 12)


def _curve(q):
    """A concave m(q) like B82's: A = 7.27, steeper near 0 than a line to m(1) = 46.93."""
    return 7.27 + 39.66 * q ** 0.8


def test_calibration_converges_on_a_curve_from_a_distant_start(script):
    """Starting at 0.9 when the root is near 0.1 still ends within the tolerance."""
    calls = []
    q = script.calibrate(lambda x: calls.append(x) or _curve(x), 12, 0.9)
    assert abs(_curve(q) - 12) <= 0.5
    assert calls[:2] == [0.0, 0.9]


def test_a_start_near_the_root_needs_three_batches(script):
    calls = []
    script.calibrate(lambda x: calls.append(x) or _curve(x), 12, 0.0605)
    assert len(calls) == 3


@pytest.mark.parametrize("q0", [0.0, -0.1, 1.5, float("nan")])
def test_a_starting_factor_outside_the_unit_interval_is_refused_before_any_run(script, q0):
    calls = []
    with pytest.raises(ValueError):
        script.calibrate(lambda x: calls.append(x) or _curve(x), 12, q0)
    assert calls == []


def test_seeds_held_at_one_that_already_reach_the_data_are_an_error(script):
    with pytest.raises(ValueError, match="alone infect"):
        script.calibrate(lambda x: 13.0 + x, 12, 0.1)


def test_a_target_beyond_q_1_is_an_error_not_a_clipped_factor(script):
    with pytest.raises(ValueError, match="outside"):
        script.calibrate(lambda x: 7.0 + 2.0 * x, 12, 0.5)


def test_no_convergence_within_the_steps_is_an_error(script):
    noisy = iter([7.27, 9.0, 20.0, 5.0, 20.0, 5.0, 20.0])
    with pytest.raises(ValueError):
        script.calibrate(lambda x: next(noisy), 12, 0.1, max_steps=3)


def test_main_exits_non_zero_when_no_factor_reaches_the_data(script, monkeypatch, tmp_path):
    best = tmp_path / "firefly_best.txt"
    best.write_text("0 0.5 0.5 1.0\n")
    monkeypatch.setattr(script, "observed_offspring", lambda: (12, 118))
    monkeypatch.setattr(script, "mean_offspring", lambda *a: (13.0, (10, 16)))
    monkeypatch.setattr(script.pickle, "load", lambda f: None)
    monkeypatch.setattr(sys, "argv", ["calibrate_imported_factor.py", str(best), "4", "1"])
    with pytest.raises(SystemExit) as stop:
        script.main()
    assert stop.value.code != 0 and "alone infect" in str(stop.value.code)
