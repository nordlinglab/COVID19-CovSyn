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
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
    assert [r["name"] for r in rows] == ["a", "c"]
    assert not [line for line in path.read_text().splitlines() if line.startswith("b")]


def test_read_rows_skips_a_cut_off_row(tmp_path):
    """A row with a missing cost is not a finished evaluation."""
    path = tmp_path / "t.csv"
    path.write_text("name,simulations,seed_offset,cost\na,10,0,1.5\nb,10,1\n")
    assert [r["name"] for r in block_scoring.read_rows(path)] == ["a"]


ROWS = [{"name": n, "simulations": 10, "seed_offset": i, "cost": 1.5 + i}
        for i, n in enumerate(["a", "b", "c", "d"])]


def _table_bytes():
    """The bytes write_csv gives ROWS, the same on every call."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        block_scoring.write_csv(Path(d) / "t.csv", ROWS)
        return (Path(d) / "t.csv").read_bytes()


def _written(tmp_path):
    path = tmp_path / "t.csv"
    block_scoring.write_csv(path, ROWS)
    return path, path.read_bytes()


DATA_LENGTH = len(_table_bytes())


@pytest.mark.parametrize("cut", range(DATA_LENGTH + 1))
def test_a_table_cut_at_every_byte_reads_as_its_whole_rows(tmp_path, cut):
    """Each offset in turn, so none is left to chance."""
    _check_cut(tmp_path, cut)


@settings(max_examples=100, deadline=None)
@given(cut=st.integers(min_value=0, max_value=DATA_LENGTH))
def test_a_table_cut_at_any_byte_reads_as_its_whole_rows(tmp_path_factory, cut):
    """Every complete row and no partial one, wherever a kill stopped the write.

    The next score then appends a valid table.
    """
    _check_cut(tmp_path_factory.mktemp("cut"), cut)


def _check_cut(tmp_path, cut):
    path, data = _written(tmp_path)
    path.write_bytes(data[:cut])
    whole = data[:cut].count(b"\n") - 1  # the header's line ends first
    rows = block_scoring.read_rows(path)
    assert [r["name"] for r in rows] == [r["name"] for r in ROWS[:max(whole, 0)]]
    _score(path, ["z"])
    again = block_scoring.read_rows(path)
    assert [r["name"] for r in again] == [r["name"] for r in ROWS[:max(whole, 0)]] + ["z"]


def test_an_unusable_covsyn_workers_falls_back_to_the_recorded_count():
    """Not to the cores, which record_workers would then write over the record."""
    for typo in ("abc", "0", "-1"):
        assert block_scoring.pilot_workers(None, {"COVSYN_WORKERS": typo}, 3) == 3


def test_a_usable_covsyn_workers_comes_before_the_record():
    """--workers, else COVSYN_WORKERS, else the record."""
    assert block_scoring.pilot_workers(None, {"COVSYN_WORKERS": "5"}, 3) == 5
    assert block_scoring.pilot_workers(2, {"COVSYN_WORKERS": "5"}, 3) == 2


def test_the_workers_are_written_whole_or_not_at_all(tmp_path, monkeypatch):
    """A kill or a full disk while writing leaves the old config.json, not a cut-off one."""
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {"simulations": 300}, "workers": 8}))

    def full_disk(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    with pytest.raises(OSError):
        block_scoring.record_workers(tmp_path, 3)
    assert json.loads(path.read_text())["workers"] == 8
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json"]  # no temporary left


def test_a_read_only_config_is_not_replaced(tmp_path):
    """Its protection holds: the write is refused, and the file and its mode stay."""
    import json
    import os
    import stat

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    path.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            block_scoring.record_workers(tmp_path, 3)
        assert json.loads(path.read_text())["workers"] == 2
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o444
    finally:
        path.chmod(0o644)


def test_a_linked_config_is_not_replaced_by_a_file(tmp_path):
    """The link stays a link; the write is refused."""
    import json

    real = tmp_path / "real.json"
    real.write_text(json.dumps({"settings": {}, "workers": 2}))
    out = tmp_path / "run"
    out.mkdir()
    (out / "config.json").symlink_to(real)
    with pytest.raises(PermissionError):
        block_scoring.record_workers(out, 3)
    assert (out / "config.json").is_symlink()


def test_the_new_config_keeps_the_old_mode(tmp_path):
    """The replacement takes the file's permissions."""
    import json
    import os
    import stat

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    path.chmod(0o640)
    block_scoring.record_workers(tmp_path, 3)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640


def test_a_config_that_is_not_an_object_is_a_value_error(tmp_path):
    """Valid JSON but no object: no workers to record."""
    (tmp_path / "config.json").write_text("[1, 2]")
    with pytest.raises(ValueError):
        block_scoring.record_workers(tmp_path, 3)


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=8), recorded=st.integers(min_value=1, max_value=64))
def test_covsyn_workers_is_used_when_usable_and_the_record_otherwise(text, recorded):
    """Usable means hardware's own rule: a whole number of at least 1."""
    from covsyn.runtime import hardware

    try:
        usable = hardware._override(text, hardware.ENV_VARIABLE)
    except ValueError:
        usable = None
    got = block_scoring.pilot_workers(None, {"COVSYN_WORKERS": text}, recorded)
    assert got == (recorded if usable is None or not text.strip() else usable)
