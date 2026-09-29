# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-case constraint checker (Table A, N2) passes valid cases and catches each violation.

Times follow the model: latent, incubation, infectious and isolation periods count from the case's
infection; infection_day and the date_of_* fields are absolute; the infectious window is the closed
interval [latent, latent + infectious].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from covsyn.validation import check_constraints as cc

NAN = math.nan


def _case(infection_day: float = 0.0, **overrides: Any) -> dict[str, Any]:
    """A symptomatic case that satisfies every constraint.

    Relative to its infection: onset on day 5, infectious on days 3-9, isolated and tested on
    day 8, closed on day 25.
    """
    case = {"infection_day": infection_day, "latent_period": 3, "incubation_period": 5,
            "infectious_period": 6, "monitor_isolation_period": 8, "isolation_route": "symptom",
            "positive_test_date": infection_day + 8, "date_of_critically_ill": NAN,
            "date_of_death": NAN, "date_of_recovery": infection_day + 25}
    case.update(overrides)
    return case


def _run(tmp_path: Path, cases: list[dict[str, Any]], edges: list[list[Any]] | None = None
         ) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    np.save(tmp_path / "course_of_disease_data_0.npy", np.array(cases, dtype=object),
            allow_pickle=True)
    np.save(tmp_path / "transmission_digraph_0.npy",
            np.array(edges or [["nan", "1", "0.0", "nan"]], dtype=object), allow_pickle=True)
    counts, _, bad = cc.check([str(tmp_path)])
    return counts, bad


def _failed(counts: dict[str, list[int]]) -> set[str]:
    return {cid for cid, (_, fail, _) in counts.items() if fail}


def test_a_valid_case_passes_every_constraint(tmp_path: Path) -> None:
    """No constraint fails for a case built to satisfy them all."""
    counts, bad = _run(tmp_path, [_case()])
    assert _failed(counts) == set()
    assert bad == []


def test_icu_before_isolation_violates_c14(tmp_path: Path) -> None:
    """E81: a case in intensive care on day 6 but isolated on day 8."""
    counts, bad = _run(tmp_path, [_case(date_of_critically_ill=6.0)])
    assert _failed(counts) == {"C14"}
    assert [b["constraint"] for b in bad] == ["C14"]


def test_icu_on_the_isolation_day_is_allowed(tmp_path: Path) -> None:
    """Isolation on the day of ICU admission satisfies C14."""
    counts, _ = _run(tmp_path, [_case(date_of_critically_ill=8.0)])
    assert "C14" not in _failed(counts)


def test_death_on_the_day_of_icu_admission_is_allowed(tmp_path: Path) -> None:
    """C11 is <=: the model lets a case die the day it enters intensive care."""
    counts, _ = _run(tmp_path, [_case(date_of_critically_ill=9.0, date_of_death=9.0,
                                      date_of_recovery=NAN)])
    assert "C11" not in _failed(counts)


@pytest.mark.parametrize(("generation_interval", "layer", "violated"), [
    (2.0, "household", "T01"),       # before the infector's latent period of 3 ends
    (10.0, "health_care", "T02"),    # after its infectious window [3, 9] ends
    (9.0, "household", "T03"),       # outside health care, after isolation on day 8
    (9.0, "health_care", None),      # health care may continue after isolation
    (8.0, "household", None),        # the isolation day itself is still inside the window
])
def test_transmission_timing(tmp_path: Path, generation_interval: float, layer: str,
                             violated: str | None) -> None:
    """Each transmission-level rule is checked against the infector's own course."""
    infectee = _case(infection_day=generation_interval)
    edges = [["nan", "1", "0.0", "nan"], ["1", "2", str(generation_interval), layer]]
    counts, _ = _run(tmp_path, [_case(), infectee], edges)
    failed = {c for c in _failed(counts) if c.startswith("T")}
    assert failed == ({violated} if violated else set())


def test_a_case_infected_twice_violates_t05(tmp_path: Path) -> None:
    """An infectee has one infector."""
    edges = [["nan", "1", "0.0", "nan"], ["1", "2", "4.0", "household"],
             ["1", "2", "5.0", "household"]]
    counts, _ = _run(tmp_path, [_case(), _case(infection_day=4.0)], edges)
    assert "T05" in _failed(counts)


def test_asymptomatic_cases_skip_the_symptom_constraints(tmp_path: Path) -> None:
    """C03, C04 and C06 do not apply without a symptom onset."""
    counts, _ = _run(tmp_path, [_case(incubation_period=NAN, isolation_route="untraced",
                                      monitor_isolation_period=9)])
    for cid in ("C03", "C04", "C06"):
        assert counts[cid][2] == 1, cid
    assert _failed(counts) == set()
