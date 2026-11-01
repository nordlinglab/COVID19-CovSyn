# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""show_bounds converts daily attack rates with the measured contact days (review of B75)."""

import numpy as np
import pytest

from covsyn.calibration.sar_anchors import ATTACK_RATE_SLICE, MEAN_CONTACT_DAYS
from covsyn.validation import show_bounds


def test_cumulative_rates_use_mean_contact_days():
    p = np.full(160, 0.01)
    rows = {row[0]: row for row in show_bounds.attack_rate_rows(p, p * 0.5, p * 2)}
    assert set(rows) == set(ATTACK_RATE_SLICE)
    for layer, (_, daily, _, _, cumulative, _, _) in rows.items():
        assert daily == pytest.approx(0.01)
        assert cumulative == pytest.approx(100 * (1 - 0.99 ** MEAN_CONTACT_DAYS[layer]))


def test_each_layer_reads_its_own_block():
    p = np.zeros(160)
    a, b = ATTACK_RATE_SLICE["workplace"]
    p[a:b] = 0.02
    rows = {row[0]: row for row in show_bounds.attack_rate_rows(p, p, p)}
    assert rows["workplace"][1] == pytest.approx(0.02)
    assert rows["household"][1] == 0.0


# --- review of B76: the days the bounds were built with travel with the bounds ----------


def test_explicit_days_override_the_current_ones():
    p = np.full(160, 0.01)
    days = {layer: 2.0 for layer in ATTACK_RATE_SLICE}
    rows = {row[0]: row for row in show_bounds.attack_rate_rows(p, p, p, days)}
    assert rows["workplace"][4] == pytest.approx(100 * (1 - 0.99**2))


def test_days_are_read_from_the_bounds_directory(tmp_path):
    days = {layer: 1.5 for layer in ATTACK_RATE_SLICE}
    (tmp_path / "attack_rate_contact_days.json").write_text(
        '{"days": %s, "measured_on": "test"}' % str(days).replace("'", '"')
    )
    assert show_bounds.bounds_contact_days(tmp_path) == (days, "test")


def test_missing_days_fall_back_to_the_current_ones(tmp_path):
    days, source = show_bounds.bounds_contact_days(tmp_path)
    assert days == MEAN_CONTACT_DAYS
    assert "not recorded" in source


def test_committed_bounds_record_their_days():
    from pathlib import Path

    from covsyn.calibration.sar_anchors import CONTACT_DAYS_MEASURED_ON

    variable = Path(__file__).resolve().parents[1] / "variable"
    assert show_bounds.bounds_contact_days(variable) == (
        MEAN_CONTACT_DAYS,
        CONTACT_DAYS_MEASURED_ON,
    )


def test_main_reads_the_given_directory(tmp_path, capsys):
    """Second review: show_bounds can print an archived run's bounds with that run's days."""
    p = np.full(208, 0.01)
    for name, v in (
        ("course_parameters", p),
        ("course_parameters_lb", p / 2),
        ("course_parameters_ub", p * 2),
    ):
        np.save(tmp_path / f"{name}.npy", v)
    days = {layer: 1.0 for layer in ATTACK_RATE_SLICE}
    (tmp_path / "attack_rate_contact_days.json").write_text(
        '{"days": %s, "measured_on": "archived run"}' % str(days).replace("'", '"')
    )
    show_bounds.main(tmp_path)
    out = capsys.readouterr().out
    assert "archived run" in out
    assert "1.00" in out  # 0.01 daily over one day is 1.00% cumulative
