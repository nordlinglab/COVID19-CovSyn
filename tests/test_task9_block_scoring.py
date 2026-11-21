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
import re

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
    """write_csv and score open their tables as UTF-8, and a non-ASCII value round-trips.

    read_rows decodes bytes, which no open records; its UTF-8 decoding is what the round trip
    and test_a_row_cut_inside_a_character_still_reads_the_finished_rows show.
    """
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
    # Without an encoding of its own, an open gets io.text_encoding's: "utf-8" in UTF-8 mode,
    # which would hide a missing one, so it gives "locale" here.
    monkeypatch.setattr(io_module, "text_encoding", lambda encoding, *a: encoding or "locale")
    path = tmp_path / "t.csv"
    block_scoring.write_csv(path, [{"name": "\u0141\u00f3d\u017a", "simulations": 10,
                                    "seed_offset": 0, "cost": 1.5}])  # fmt: skip
    _score(path, ["\u00e9t\u00e9"])
    rows = block_scoring.read_rows(path)
    monkeypatch.undo()
    assert [r["name"] for r in rows] == ["\u0141\u00f3d\u017a", "\u00e9t\u00e9"]
    assert encodings and set(encodings) == {"utf-8"}


def test_a_row_cut_inside_a_character_still_reads_the_finished_rows(tmp_path):
    """The kill split a multi-byte character; the finished rows before it are read."""
    path = tmp_path / "t.csv"
    path.write_bytes(
        b"name,simulations,seed_offset,cost\na,10,0,1.5\n\xc3\xa9t\xc3\xa9,10,1,2.5\n\xc3"
    )
    assert [r["name"] for r in block_scoring.read_rows(path)] == ["a", "\u00e9t\u00e9"]


def test_a_table_saved_with_a_byte_order_mark_reads_its_header(tmp_path, caplog):
    """A spreadsheet's "CSV UTF-8" adds EF BB BF; the first column keeps its name.

    It is read, with a warning naming the file: a spreadsheet may have shortened the costs.
    """
    import logging

    path = tmp_path / "t.csv"
    path.write_bytes(b"\xef\xbb\xbfname,simulations,seed_offset,cost\na,10,0,1.5\n")
    with caplog.at_level(logging.WARNING, logger=block_scoring.log.name):
        assert [r["name"] for r in block_scoring.read_rows(path)] == ["a"]
    assert str(path) in caplog.text and "rewrote" in caplog.text


def test_a_table_without_a_byte_order_mark_gives_no_warning(tmp_path, caplog):
    """The pilots' own tables, as they write them, read quietly."""
    import logging

    path = tmp_path / "t.csv"
    path.write_bytes(b"name,simulations,seed_offset,cost\na,10,0,1.5\n")
    with caplog.at_level(logging.WARNING, logger=block_scoring.log.name):
        block_scoring.read_rows(path)
    assert not caplog.text


def test_a_read_that_asks_for_no_warning_gives_none(tmp_path, caplog):
    """warn=False: a pilot's later reads of a table it has warned about in this run."""
    import logging

    path = tmp_path / "t.csv"
    path.write_bytes(b"\xef\xbb\xbfname,simulations,seed_offset,cost\na,10,0,1.5\n")
    with caplog.at_level(logging.WARNING, logger=block_scoring.log.name):
        assert [r["name"] for r in block_scoring.read_rows(path, warn=False)] == ["a"]
    assert not caplog.text


@pytest.fixture
def socket_path():
    """A path holding a bound AF_UNIX socket, in a short directory: $TMPDIR, else /tmp.

    pytest's own tmp_path, or a long $TMPDIR, can pass macOS's 104-byte cap on a socket's path.
    """
    import os
    import shutil
    import socket
    import sys
    import tempfile
    from pathlib import Path

    cap = 104 if sys.platform == "darwin" else 108  # bytes of an AF_UNIX path, with its NUL
    directory = Path(tempfile.mkdtemp(prefix="cs"))
    if len(os.fsencode(directory / "t.csv")) >= cap:  # a long $TMPDIR
        directory.rmdir()
        try:
            directory = Path(tempfile.mkdtemp(prefix="cs", dir="/tmp"))
        except OSError:
            pytest.skip("no directory short enough for a socket's path")
    listener = socket.socket(socket.AF_UNIX)
    try:
        path = directory / "t.csv"
        listener.bind(str(path))
        yield path
    finally:
        listener.close()
        shutil.rmtree(directory)


def test_a_missing_table_at_the_first_read_reads_as_absent(tmp_path):
    """Missing when a pilot first looks: no rows, no mark, and exists False, so it starts anew."""
    path = tmp_path / "t.csv"
    table = block_scoring.read_table(path)
    assert (table.rows, table.marked, table.exists) == ([], False, False)
    path.write_text("name,simulations,seed_offset,cost\na,10,0,1.5\n")
    assert block_scoring.read_table(path).exists


def test_a_table_known_to_exist_that_went_missing_is_an_error(tmp_path):
    """A later read, after the pilot saw it: stop, naming it, rather than start anew."""
    path = tmp_path / "t.csv"
    with pytest.raises(block_scoring.TableError, match=re.escape(str(path))):
        block_scoring.read_table(path, required=True)


def test_a_special_file_is_refused_before_a_read_could_block(tmp_path):
    """A FIFO at the table's path: refused at once, not read forever."""
    import os

    path = tmp_path / "t.csv"
    os.mkfifo(path)
    with pytest.raises(block_scoring.TableError, match="not a regular file"):
        block_scoring.read_table(path)


@pytest.mark.parametrize("what", ["directory", "parent", "socket"])
def test_a_path_that_cannot_be_a_table_is_named(tmp_path, request, what):
    """A directory or a socket where the table belongs, or a file where its directory belongs."""
    if what == "directory":
        path = tmp_path / "t.csv"
        path.mkdir()
    elif what == "socket":
        path = request.getfixturevalue("socket_path")
    else:
        (tmp_path / "out").write_text("a file, not a directory")
        path = tmp_path / "out" / "t.csv"
    with pytest.raises(block_scoring.TableError, match=re.escape(str(path))):
        block_scoring.read_table(path)


def test_a_table_known_to_exist_is_not_recreated_when_it_went_missing(tmp_path):
    """Appending to a table the pilot read: if it is gone, stop; do not create an empty one."""
    path = tmp_path / "t.csv"
    with pytest.raises(block_scoring.TableError, match=re.escape(str(path))):
        block_scoring.score(path, [{"name": "a", "simulations": 10, "seed_offset": 0}], COLUMNS,
                            lambda t: np.ones(2), _evaluate, _threads, exists=True)  # fmt: skip
    assert not path.exists()


def test_a_new_table_that_appeared_meanwhile_is_not_appended_to(tmp_path):
    """Absent at the first read, there now: stop, so a restart reads what it holds."""
    path = tmp_path / "t.csv"
    path.write_text("name,simulations,seed_offset,cost\nb,10,1,2.5\n")
    before = path.read_bytes()
    with pytest.raises(block_scoring.TableError, match="start it again"):
        block_scoring.score(path, [{"name": "a", "simulations": 10, "seed_offset": 0}], COLUMNS,
                            lambda t: np.ones(2), _evaluate, _threads, exists=False)  # fmt: skip
    assert path.read_bytes() == before


def _task():
    return [{"name": "a", "simulations": 10, "seed_offset": 0}]


@pytest.mark.parametrize("entry", ["dangling link", "directory", "fifo"])
def test_a_new_table_s_path_taken_by_another_entry_is_named(tmp_path, entry):
    """Not "a table appeared, start again", which a restart would only repeat."""
    import os

    path = tmp_path / "t.csv"
    if entry == "dangling link":
        path.symlink_to(tmp_path / "nowhere" / "x.csv")
    elif entry == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises(block_scoring.TableError) as error:
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip
    assert "start it again" not in str(error.value) and str(path) in str(error.value)
    if entry == "dangling link":
        assert "symbolic link" in str(error.value) and "nowhere" in str(error.value)


@pytest.mark.parametrize("entry", ["directory", "fifo", "parent", "dangling link", "socket"])
def test_appending_to_a_table_that_is_no_regular_file_is_refused(tmp_path, request, entry):
    """The exists=True path: trimmed and appended only as a regular file."""
    import os

    path = tmp_path / "t.csv"
    if entry == "directory":
        path.mkdir()
    elif entry == "fifo":
        os.mkfifo(path)
    elif entry == "dangling link":  # the table it pointed at is gone
        path.symlink_to(tmp_path / "nowhere.csv")
    elif entry == "socket":
        path = request.getfixturevalue("socket_path")
    else:
        (tmp_path / "out").write_text("a file")
        path = tmp_path / "out" / "t.csv"
    with pytest.raises(block_scoring.TableError, match=re.escape(str(path))):
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=True)  # fmt: skip


def test_a_new_table_under_a_parent_that_is_a_file_is_named(tmp_path):
    """The create path names a parent that is no directory, as the read path does."""
    (tmp_path / "out").write_text("a file")
    path = tmp_path / "out" / "t.csv"
    with pytest.raises(block_scoring.TableError, match="parent"):
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip


def test_a_table_cut_off_before_any_newline_gets_its_header(tmp_path):
    """A kill inside the header itself: trimmed to nothing, then written whole."""
    path = tmp_path / "t.csv"
    path.write_text("name,simul")
    _score(path, ["a"])
    assert [r["name"] for r in block_scoring.read_rows(path)] == ["a"]


def test_a_new_table_takes_its_mode_from_the_umask_alone(tmp_path):
    """0o666 less the umask, as open(path, "a") gave it: a shared group keeps write access."""
    import os
    import stat

    old = os.umask(0o002)
    try:
        path = tmp_path / "t.csv"
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o664


def test_a_live_link_to_a_table_that_appeared_says_start_again(tmp_path):
    """The link points at a table now: restarting reads it, so that is the advice."""
    target = tmp_path / "shared.csv"
    target.write_text("name,simulations,seed_offset,cost\nb,10,1,2.5\n")
    path = tmp_path / "t.csv"
    path.symlink_to(target)
    with pytest.raises(block_scoring.TableError, match="start it again"):
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip


def test_a_link_removed_while_its_target_is_named_still_gives_a_refusal(tmp_path, monkeypatch):
    """A link seen, then gone before its target is named: still a TableError, not an OSError."""
    import os

    path = tmp_path / "t.csv"
    path.symlink_to(tmp_path / "nowhere.csv")

    def gone(p):
        raise FileNotFoundError(2, "No such file or directory", str(p))

    monkeypatch.setattr(os, "readlink", gone)
    with pytest.raises(block_scoring.TableError, match=re.escape(str(path))):
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip


def test_a_table_the_system_refuses_is_a_table_error(tmp_path, monkeypatch):
    """PermissionError opening a table becomes a TableError naming it; others stay as they are."""
    import os

    path = tmp_path / "t.csv"
    path.write_text("name,simulations,seed_offset,cost\n")
    real_open = os.open

    def refused(p, flags, *args, **kwargs):
        if str(p) == str(path):
            raise PermissionError(13, "Permission denied", str(p))
        return real_open(p, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refused)
    with pytest.raises(block_scoring.TableError, match=re.escape(str(path))):
        block_scoring.read_table(path)


def test_a_failure_wrapping_the_descriptor_closes_it_once(tmp_path, monkeypatch):
    """A text wrapper that fails: the raw file is closed, and os.close is not called again."""
    import io
    import os

    closed, raws = [], []
    real_close, real_fileio = os.close, io.FileIO
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd) or real_close(fd))
    monkeypatch.setattr(io, "FileIO", lambda *a, **kw: raws.append(real_fileio(*a, **kw))
                        or raws[-1])  # fmt: skip

    def failing(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(io, "TextIOWrapper", failing)
    with pytest.raises(MemoryError):
        block_scoring.score(tmp_path / "t.csv", _task(), COLUMNS, lambda t: np.ones(2),
                            _evaluate, _threads, exists=False)  # fmt: skip
    assert len(raws) == 1 and raws[0].closed and closed == []  # closed once, by the raw file


@pytest.mark.parametrize("code", [errno.EMFILE, errno.ENOSPC, errno.EIO])
def test_an_error_that_is_not_the_table_s_stays_as_it_is(tmp_path, monkeypatch, code):
    """Too many open files, a full disk, an I/O error: raised as such, not as a TableError."""
    import os

    path = tmp_path / "t.csv"
    path.write_text("name,simulations,seed_offset,cost\n")

    def failing(p, flags, *args, **kwargs):
        raise OSError(code, os.strerror(code), str(p))

    monkeypatch.setattr(os, "open", failing)
    with pytest.raises(OSError) as error:
        block_scoring.read_table(path)
    assert not isinstance(error.value, block_scoring.TableError) and error.value.errno == code
    for path_, exists in ((tmp_path / "new.csv", False), (path, True)):  # create, append
        with pytest.raises(OSError) as error:
            block_scoring.score(path_, _task(), COLUMNS, lambda t: np.ones(2), _evaluate,
                                _threads, exists=exists)  # fmt: skip
        assert not isinstance(error.value, block_scoring.TableError)
        assert error.value.errno == code


def test_a_new_table_in_a_read_only_directory_is_a_table_error(tmp_path):
    """The create path names the refusal, as the read path does."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root writes into a read-only directory")
    run = tmp_path / "run"
    run.mkdir()
    run.chmod(0o555)
    try:
        with pytest.raises(block_scoring.TableError, match=re.escape(str(run / "t.csv"))):
            block_scoring.score(run / "t.csv", _task(), COLUMNS, lambda t: np.ones(2),
                                _evaluate, _threads, exists=False)  # fmt: skip
    finally:
        run.chmod(0o755)


@pytest.mark.parametrize("target", ["directory", "loop"])
def test_a_link_to_what_cannot_be_a_table_says_what_it_leads_to(tmp_path, target):
    """A live link to a directory is no table; a link loop leads to no file at all."""
    path = tmp_path / "t.csv"
    if target == "directory":
        (tmp_path / "d").mkdir()
        path.symlink_to(tmp_path / "d")
        said = "not a regular file"
    else:
        path.symlink_to(path)
        said = "leads to no file"
    with pytest.raises(block_scoring.TableError, match=said):
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip


def test_a_path_whose_existence_cannot_be_checked_is_a_table_error(tmp_path, monkeypatch):
    """exists=None checks the path; a refused check is the table's refusal too."""
    from pathlib import Path

    def refused(self, *a, **kw):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "exists", refused)
    with pytest.raises(block_scoring.TableError):
        block_scoring.score(tmp_path / "t.csv", _task(), COLUMNS, lambda t: np.ones(2),
                            _evaluate, _threads)  # fmt: skip


def test_a_link_to_a_table_out_of_reach_is_named_as_refused(tmp_path, monkeypatch):
    """The target exists but cannot be reached: refused, not "leads to no file"."""
    import os

    path = tmp_path / "t.csv"
    path.symlink_to(tmp_path / "locked" / "blocks.csv")
    real_stat = os.stat

    def refused(p, *a, **kw):
        if str(p) == str(path) and kw.get("follow_symlinks", True):  # following the link
            raise PermissionError(13, "Permission denied", str(p))
        return real_stat(p, *a, **kw)

    monkeypatch.setattr(os, "stat", refused)
    with pytest.raises(block_scoring.TableError) as error:
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip
    assert "Permission denied" in str(error.value) and "leads to no file" not in str(error.value)
    assert "symbolic link" in str(error.value) and "locked" in str(error.value)  # which, where


@pytest.mark.parametrize("code", sorted({errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENXIO,
                                         errno.ENODEV}))
def test_a_refusal_at_open_names_a_socket_but_not_a_regular_file(tmp_path, monkeypatch,
                                                                 socket_path, code):
    """A socket refused at open (EOPNOTSUPP on macOS, ENXIO on Linux): what is there decides.

    A socket is "not a regular file"; a regular file refused the same way keeps the error.
    """
    import os

    regular = tmp_path / "t.csv"
    regular.write_text("")
    link = tmp_path / "link.csv"  # a link to a regular table is a table too
    link.symlink_to(regular)
    socket_link = tmp_path / "socket-link.csv"  # a link to a socket is not
    socket_link.symlink_to(socket_path)

    def refused(p, flags, *args, **kwargs):
        raise OSError(code, os.strerror(code), str(p))

    monkeypatch.setattr(os, "open", refused)
    try:
        for path in (socket_path, socket_link):
            with pytest.raises(block_scoring.TableError, match="not a regular file"):
                block_scoring.read_table(path)
        for path in (regular, link):
            with pytest.raises(OSError) as error:
                block_scoring.read_table(path)
            assert not isinstance(error.value, block_scoring.TableError), path
            assert error.value.errno == code
    finally:
        monkeypatch.undo()  # before the socket's clean-up, which opens files too


@pytest.mark.parametrize("code", sorted({errno.EOPNOTSUPP, errno.ENOTSUP, errno.EAGAIN}))
def test_a_new_table_s_unsupported_or_busy_create_keeps_its_error(tmp_path, monkeypatch, code):
    """Nothing is there: neither "not a regular file" nor "held by another program"."""
    import os

    real_open = os.open

    def refused(p, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            raise OSError(code, os.strerror(code), str(p))
        return real_open(p, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refused)
    with pytest.raises(OSError) as error:
        block_scoring.score(tmp_path / "t.csv", _task(), COLUMNS, lambda t: np.ones(2),
                            _evaluate, _threads, exists=False)  # fmt: skip
    assert not isinstance(error.value, block_scoring.TableError) and error.value.errno == code


def test_an_entry_gone_again_before_it_is_named_is_not_said_to_have_appeared(tmp_path,
                                                                             monkeypatch):
    """EEXIST, then nothing there: the refusal says it changed, not "start it again"."""
    import os

    path = tmp_path / "t.csv"
    real_open, real_lstat = os.open, os.lstat

    def exists_then_gone(p, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(p))
        return real_open(p, flags, *args, **kwargs)

    def gone(p, *a, **kw):
        if str(p) == str(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(p))
        return real_lstat(p, *a, **kw)

    monkeypatch.setattr(os, "open", exists_then_gone)
    monkeypatch.setattr(os, "lstat", gone)
    with pytest.raises(block_scoring.TableError) as error:
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip
    assert "start it again" not in str(error.value) and "changed" in str(error.value)


def test_a_table_another_program_holds_says_to_resume_when_it_lets_go(tmp_path, monkeypatch):
    """A lease (EAGAIN) only means wait: the refusal says so."""
    import os

    path = tmp_path / "t.csv"
    path.write_text("")

    def leased(p, flags, *args, **kwargs):
        raise OSError(errno.EAGAIN, os.strerror(errno.EAGAIN), str(p))

    monkeypatch.setattr(os, "open", leased)
    with pytest.raises(block_scoring.TableError, match="held by another program.*resume"):
        block_scoring.read_table(path)


@pytest.mark.parametrize("bare", [OSError(errno.ENOENT), OSError(errno.ENOENT, None)],
                         ids=["code-only", "errno-no-text"])
def test_a_change_without_a_reason_text_still_says_what_happened(tmp_path, monkeypatch, bare):
    """An OSError without strerror, its code in args or in errno: the errno's own text."""
    import os

    path = tmp_path / "t.csv"
    real_open, real_lstat = os.open, os.lstat

    def exists(p, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(p))
        return real_open(p, flags, *args, **kwargs)

    def refused(p, *a, **kw):
        if str(p) == str(path):
            raise bare
        return real_lstat(p, *a, **kw)

    monkeypatch.setattr(os, "open", exists)
    monkeypatch.setattr(os, "lstat", refused)
    with pytest.raises(block_scoring.TableError) as error:
        block_scoring.score(path, _task(), COLUMNS, lambda t: np.ones(2), _evaluate, _threads,
                            exists=False)  # fmt: skip
    assert "None" not in str(error.value) and os.strerror(errno.ENOENT) in str(error.value)


@pytest.mark.parametrize("bare", [OSError(), OSError("not a number")])
def test_an_error_with_no_text_at_all_is_named_by_its_kind(bare):
    """No strerror, no errno, no text: the error's kind, not empty parentheses."""
    assert block_scoring._reason(bare)
