# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 9: the pilots' tables survive a kill at any point of a write.

block_scoring appends each finished evaluation to a CSV, so a pilot resumes from it. A kill can
leave the file empty, before its header was written, or its last row cut off; the next run must
then still write a header, and start its rows on a line of their own.
"""

from __future__ import annotations

import concurrent.futures
import errno

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


ROWS = [
    {"name": n, "simulations": 10, "seed_offset": i, "cost": 1.5 + i}
    for i, n in enumerate(["a", "b", "c", "d"])
]


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
    assert [r["name"] for r in rows] == [r["name"] for r in ROWS[: max(whole, 0)]]
    _score(path, ["z"])
    again = block_scoring.read_rows(path)
    assert [r["name"] for r in again] == [r["name"] for r in ROWS[: max(whole, 0)]] + ["z"]


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
        usable = hardware.parse_workers(text, hardware.ENV_VARIABLE)
    except ValueError:
        usable = None
    got = block_scoring.pilot_workers(None, {"COVSYN_WORKERS": text}, recorded)
    assert got == (recorded if usable is None or not text.strip() else usable)


def test_a_read_only_config_is_refused_by_its_mode_bits(tmp_path, monkeypatch):
    """As root os.access says yes to everything; the mode still marks it read-only."""
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    path.chmod(0o444)
    monkeypatch.setattr(os, "access", lambda *a, **kw: True)  # as root
    try:
        with pytest.raises(PermissionError):
            block_scoring.record_workers(tmp_path, 3)
    finally:
        path.chmod(0o644)


def test_a_hard_linked_config_is_not_split(tmp_path):
    """Replacing it would leave the other name with the old workers."""
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    os.link(path, tmp_path / "backup.json")
    with pytest.raises(PermissionError):
        block_scoring.record_workers(tmp_path, 3)
    assert json.loads((tmp_path / "backup.json").read_text())["workers"] == 2


def test_a_directory_that_cannot_be_synced_does_not_undo_a_written_config(tmp_path, monkeypatch):
    """The rename has happened; a file system that refuses a directory fsync is no failure."""
    import json
    import os
    import stat

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    real_fsync = os.fsync

    def no_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(22, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", no_directory_fsync)
    block_scoring.record_workers(tmp_path, 3)
    assert json.loads(path.read_text())["workers"] == 3


def test_the_temporary_is_never_more_readable_than_the_config(tmp_path, monkeypatch):
    """Created with the config's mode, so a private record has no world-readable copy."""
    import json
    import os
    import stat

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    path.chmod(0o600)
    modes = []
    real_fsync = os.fsync

    def watching(fd):
        for tmp in tmp_path.glob("config.json.*.tmp"):  # its own name, made afresh
            modes.append(stat.S_IMODE(os.stat(tmp).st_mode))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", watching)
    block_scoring.record_workers(tmp_path, 3)
    assert modes and all(m == 0o600 for m in modes)


def test_the_workers_parser_is_hardware_s_public_one():
    """Not a private helper of another module."""
    from covsyn.runtime import hardware

    assert hardware.parse_workers("4", "x") == 4
    with pytest.raises(ValueError):
        hardware.parse_workers("0", "x")


def test_a_link_left_at_a_temporary_path_is_not_followed(tmp_path):
    """No fixed temporary name: a file it might point to stays untouched."""
    import json

    notes = tmp_path / "notes.txt"
    notes.write_text("mine\n")
    (tmp_path / "config.json").write_text(json.dumps({"settings": {}, "workers": 2}))
    (tmp_path / "config.json.tmp").symlink_to(notes)
    block_scoring.record_workers(tmp_path, 3)
    assert notes.read_text() == "mine\n"
    assert not (tmp_path / "config.json").is_symlink()
    assert json.loads((tmp_path / "config.json").read_text())["workers"] == 3


@pytest.mark.parametrize(
    "setup, cause",
    [
        (0o444, "read-only"),
        ("hard", "hard-linked"),
        ("symbolic", "a symbolic link"),
        ("directory", "read-only directory"),
        ("owner", "owned by another user"),
    ],
)
def test_the_refusal_names_its_cause(tmp_path, monkeypatch, setup, cause):
    """Each refusal says why, a writable but hard-linked config.json included."""
    import json
    import os

    run = tmp_path / "run"
    run.mkdir()
    path = run / "config.json"
    if setup == "symbolic":
        real = tmp_path / "real.json"
        real.write_text(json.dumps({"settings": {}, "workers": 2}))
        path.symlink_to(real)
    else:
        path.write_text(json.dumps({"settings": {}, "workers": 2}))
    if setup == "hard":
        os.link(path, tmp_path / "backup.json")
    elif setup == "directory":
        if os.geteuid() == 0:
            pytest.skip("root writes into a read-only directory, so there is nothing to refuse")
        run.chmod(0o555)
    elif setup == "owner":
        monkeypatch.setattr(os, "geteuid", lambda: os.stat(path).st_uid + 1)
    elif isinstance(setup, int):
        path.chmod(setup)
    try:
        with pytest.raises(PermissionError, match=cause):
            block_scoring.record_workers(run, 3)
    finally:
        run.chmod(0o755)
        if not path.is_symlink():
            path.chmod(0o644)


def test_a_failed_rename_closes_the_descriptor_once(tmp_path, monkeypatch):
    """The temporary's descriptor is closed exactly once, however the write ends."""
    import json
    import os

    (tmp_path / "config.json").write_text(json.dumps({"settings": {}, "workers": 2}))
    closed = []
    real_close = os.close
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd) or real_close(fd))

    def failing(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError):
        block_scoring.record_workers(tmp_path, 3)
    assert len(closed) == len(set(closed)) and closed  # each once
    assert not list(tmp_path.glob("config.json.*.tmp"))


def test_a_failed_write_closes_the_descriptor_and_leaves_no_temporary(tmp_path, monkeypatch):
    """A full disk mid-write: the old config.json stays, and nothing is left open or behind."""
    import json
    import os

    (tmp_path / "config.json").write_text(json.dumps({"settings": {}, "workers": 2}))
    closed = []
    real_close = os.close
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd) or real_close(fd))

    def full(fd, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", full)
    with pytest.raises(OSError):
        block_scoring.record_workers(tmp_path, 3)
    assert len(closed) == 1
    assert not list(tmp_path.glob("config.json.*.tmp"))
    assert json.loads((tmp_path / "config.json").read_text())["workers"] == 2


def test_a_group_that_cannot_be_kept_is_refused(tmp_path, monkeypatch):
    """The replacement would take the user's group, and lab members their access with it."""
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))

    def refused(fd, uid, gid):
        raise PermissionError(1, "Operation not permitted")

    _other_group(monkeypatch, path, refused)
    with pytest.raises(PermissionError, match="group.*workers stay 2.*resume with 3"):
        block_scoring.record_workers(tmp_path, 3)
    monkeypatch.undo()
    assert json.loads(path.read_text())["workers"] == 2
    assert not list(tmp_path.glob("config.json.*.tmp"))


def _other_group(monkeypatch, path, fchown):
    """config.json reported in another group than a new file gets; fchown as given."""
    import os

    real_stat = os.stat
    group = real_stat(path).st_gid + 1

    class Stat:
        def __init__(self, st):
            self.st = st

        def __getattr__(self, name):
            return group if name == "st_gid" else getattr(self.st, name)

    monkeypatch.setattr(os, "stat", lambda p, *a, **kw: Stat(real_stat(p, *a, **kw))
                        if str(p) == str(path) else real_stat(p, *a, **kw))  # fmt: skip
    monkeypatch.setattr(os, "fchown", fchown)
    return group


def test_the_group_is_given_to_the_replacement(tmp_path, monkeypatch):
    """fchown(temporary, -1, config.json's group): the owner unchanged, the group kept."""
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    calls, handles = [], []
    group = _other_group(monkeypatch, path, lambda fd, uid, gid: calls.append((fd, uid, gid)))
    real_mkstemp = block_scoring.tempfile.mkstemp

    def recording(*args, **kwargs):
        handle, name = real_mkstemp(*args, **kwargs)
        handles.append(handle)
        return handle, name

    monkeypatch.setattr(block_scoring.tempfile, "mkstemp", recording)
    block_scoring.record_workers(tmp_path, 3)
    assert calls == [(handles[0], -1, group)]  # the temporary's descriptor
    monkeypatch.undo()
    assert json.loads(path.read_text())["workers"] == 3
    assert os.path.exists(path)


@pytest.mark.parametrize("errno_", sorted({errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}))
def test_a_file_system_that_cannot_set_the_group_is_a_refusal(tmp_path, monkeypatch, errno_):
    """EINVAL, EOPNOTSUPP or ENOTSUP (macOS's) from fchown is a refusal naming the group.

    The pilots report a PermissionError as it says (tests/test_task9_pilot_size.py).
    """
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))

    def refused(fd, uid, gid):
        raise OSError(errno_, os.strerror(errno_))

    _other_group(monkeypatch, path, refused)
    with pytest.raises(PermissionError, match="group.*workers stay 2.*resume with 3"):
        block_scoring.record_workers(tmp_path, 3)
    monkeypatch.undo()
    assert json.loads(path.read_text())["workers"] == 2
    assert not list(tmp_path.glob("config.json.*.tmp"))


def test_the_group_is_set_before_the_mode(tmp_path, monkeypatch):
    """Changing the group clears setuid and setgid bits, so the mode comes after it."""
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    order = []
    _other_group(monkeypatch, path, lambda fd, uid, gid: order.append("chown"))
    real_fchmod = os.fchmod
    monkeypatch.setattr(os, "fchmod", lambda fd, mode: order.append("chmod")
                        or real_fchmod(fd, mode))  # fmt: skip
    block_scoring.record_workers(tmp_path, 3)
    assert order == ["chown", "chmod"]


def test_short_writes_still_write_the_whole_record(tmp_path, monkeypatch):
    """A file system that takes a few bytes per write gets every byte, in order."""
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, bytes(data[:3])))
    block_scoring.record_workers(tmp_path, 3)
    monkeypatch.undo()
    assert json.loads(path.read_text()) == {"settings": {"simulations": 300}, "workers": 3}


def test_a_write_that_takes_nothing_fails_rather_than_spins(tmp_path, monkeypatch):
    """A file system whose write returns 0 for a non-empty buffer: an error, not a hang."""
    import json
    import os

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {}, "workers": 2}))
    calls = []

    def nothing(fd, data):
        calls.append(1)
        if len(calls) > 50:  # without the guard: fail, not hang the suite
            raise AssertionError("record_workers kept writing after 0 bytes were taken")
        return 0

    monkeypatch.setattr(os, "write", nothing)
    with pytest.raises(OSError, match="short write") as info:
        block_scoring.record_workers(tmp_path, 3)
    assert info.value.errno == errno.EIO and info.value.filename == str(path)
    monkeypatch.undo()
    assert json.loads(path.read_text())["workers"] == 2
    assert not list(tmp_path.glob("config.json.*.tmp"))


def test_the_tables_are_utf_8_whatever_the_locale(tmp_path, monkeypatch):
    """A non-ASCII value round-trips through write_csv, score and read_rows as UTF-8."""
    import builtins
    import io as io_module

    encodings = []
    real_open = builtins.open

    def recording(file, mode="r", *args, **kwargs):
        if str(file).endswith(".csv") and "b" not in mode:  # text opens; bytes have none
            # Path.open passes it by position, after buffering
            encodings.append(kwargs.get("encoding", args[1] if len(args) > 1 else None))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording)
    monkeypatch.setattr(io_module, "open", recording)
    path = tmp_path / "t.csv"
    block_scoring.write_csv(path, [{"name": "\u0141\u00f3d\u017a", "simulations": 10,
                                    "seed_offset": 0, "cost": 1.5}])  # fmt: skip
    _score(path, ["\u00e9t\u00e9"])
    rows = block_scoring.read_rows(path)
    monkeypatch.undo()
    assert [r["name"] for r in rows] == ["\u0141\u00f3d\u017a", "\u00e9t\u00e9"]
    assert encodings and set(encodings) == {"utf-8"}
    assert path.read_bytes().decode("utf-8")  # the bytes are UTF-8
