# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 9: the pilots' tables survive a kill at any point of a write.

block_scoring appends each finished evaluation to a CSV, so a pilot resumes from it. A kill can
leave the file empty, before its header was written, or its last row cut off; the next run must
then still write a header, and start its rows on a line of their own.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np

from covsyn.calibration import block_scoring

COLUMNS = ("name", "simulations", "seed_offset", "cost")


def _evaluate(P, seed_offset):
    return float(np.sum(P) + seed_offset), {}, {}


def _threads(simulations):
    return concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _score(path, names):
    todo = [{"name": n, "simulations": 10, "seed_offset": i} for i, n in enumerate(names)]
    block_scoring.score(path, todo, COLUMNS, lambda t: np.ones(2), _evaluate, _threads)


def test_an_empty_table_gets_its_header(tmp_path):
    """Killed after creating the file, before the header: the resume writes it."""
    path = tmp_path / "t.csv"
    path.write_text("")
    _score(path, ["a", "b"])
    rows = block_scoring.read_rows(path)
    assert sorted(r["name"] for r in rows) == ["a", "b"]


def test_a_cut_off_row_does_not_swallow_the_next(tmp_path):
    """Killed mid-row: the resume starts on a new line, and the cut-off row is not a row."""
    path = tmp_path / "t.csv"
    _score(path, ["a"])
    with open(path, "a") as f:
        f.write("b,10,1")  # no cost and no newline
    _score(path, ["c"])
    rows = block_scoring.read_rows(path)
    assert [r["name"] for r in rows if r["cost"] not in (None, "")] == ["a", "c"]


def test_read_rows_skips_a_cut_off_row(tmp_path):
    """A row with a missing cost is not a finished evaluation."""
    path = tmp_path / "t.csv"
    path.write_text("name,simulations,seed_offset,cost\na,10,0,1.5\nb,10,1\n")
    assert [r["name"] for r in block_scoring.read_rows(path)] == ["a"]
