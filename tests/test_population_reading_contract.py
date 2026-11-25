# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The reading contract of initial_population.load() and rows_available() (2026-11-24).

Each file state of the contract, through both entry points: what each returns or raises,
the exit status, the path named once (in a directory whose name ends in a parenthesis), the
advice's direction, and that the file loads once put right. The contract itself is the
comment above initial_population.PopulationUnreadable.
"""

from __future__ import annotations

import io
import os
import random
import re
import struct
import zipfile

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from covsyn.calibration import initial_population as ip
from covsyn.calibration import parameter_space
from tests.conftest import REPO_ROOT

COMMITTED = REPO_ROOT / "variable" / "initial_population_v1.npz"


@pytest.fixture(scope="module")
def space():
    """The bounds the committed population was made with."""
    return parameter_space.load(REPO_ROOT / "variable")


@pytest.fixture
def place(tmp_path):
    """Where the file goes: a directory with a space and a closing parenthesis in its name."""
    directory = tmp_path / "a dir (x)"
    directory.mkdir()
    return directory / "initial_population_v1.npz"


def _put_right(path):
    path.write_bytes(COMMITTED.read_bytes())


def _missing(path):
    return path


def _a_directory(path):
    path.mkdir()
    return path


def _a_file_for_a_directory(path):
    path.parent.joinpath("blocker").write_text("x")
    return path.parent / "blocker" / path.name


def _empty(path):
    path.write_bytes(b"")
    return path


def _no_rows(path):
    """The committed file with its rows emptied, else complete (code review round 20)."""
    with np.load(COMMITTED) as saved:
        arrays = {key: saved[key] for key in saved.files}
    np.savez(path, **{**arrays, "normalized": arrays["normalized"][:0]})
    return path


def _a_non_array_member(path):
    """A complete archive with a member that is not an NPY array."""
    with zipfile.ZipFile(COMMITTED) as source, zipfile.ZipFile(path, "w") as target:
        for name in source.namelist():
            data = b"plain text" if name == "normalized.npy" else source.read(name)
            target.writestr(name, data)
    return path


def _a_forged_header(path):
    """The committed file but normalized.npy, which claims 10**11 rows and holds 48 bytes.

    Complete otherwise, so no missing array refuses it first (code review rounds 18, 21).
    """
    with np.load(COMMITTED) as saved:
        np.savez(path, **{key: saved[key] for key in saved.files})
    with zipfile.ZipFile(path) as plain:
        members = {name: plain.read(name) for name in plain.namelist()}
    forged = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        forged, {"descr": "<f8", "fortran_order": False, "shape": (10**11, 2)})
    forged.write(b"\0" * 48)
    members["normalized.npy"] = forged.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _unsupported_lzma(path):
    """Change only the compression of the required normalized member."""
    with zipfile.ZipFile(COMMITTED) as source, zipfile.ZipFile(path, "w") as target:
        for name in source.namelist():
            method = zipfile.ZIP_LZMA if name == "normalized.npy" else zipfile.ZIP_STORED
            target.writestr(name, source.read(name), compress_type=method)
    return path


# state, its reason (each its own, so no other fault refuses the file first; round 21), the
# advice it gets, whether putting the file right at the path fixes it
REFUSED = [
    (_missing, "FileNotFoundError", "last commit", True),
    (_a_directory, "IsADirectoryError", "a directory stands where the file should be", False),
    (_a_file_for_a_directory, "NotADirectoryError", "should be a directory is not one", False),
    (_empty, "empty population file", "last commit", True),
    (_a_non_array_member, "not an NPY array", "last commit", True),
    (_no_rows, "not a non-empty 2-D array", "last commit", True),
    (_a_forged_header, "header size does not match the member", "last commit", True),
    (_unsupported_lzma, "unsupported ZIP compression for normalized", "last commit", True),
]


@pytest.mark.parametrize(("state", "reason", "advice", "fixable"), REFUSED)
def test_load_refuses_a_bad_file_in_words_and_resumably(place, space, state, reason, advice,
                                                        fixable):
    """Guarantees 2, 4, 5: a SystemExit with a message (exit 1), the path once, a direction."""
    path = state(place)
    with pytest.raises(SystemExit) as stopped:
        ip.load(path, 3, space.lower, space.upper)
    message = stopped.value.code
    assert isinstance(message, str) and not isinstance(stopped.value, ip.RunSizeOutOfRange)
    assert message.count(str(path)) == 1 and advice in message
    assert re.search(reason, message) and "KeyError" not in message
    assert "git " not in message and "--out-dir" not in message
    if fixable:
        _put_right(path)
        assert len(ip.load(path, 3, space.lower, space.upper)) == 3


@pytest.mark.parametrize(("state", "reason", "advice", "fixable"), REFUSED)
def test_rows_available_refuses_the_same_files(place, state, reason, advice, fixable):
    """Guarantee 2: PopulationUnreadable naming the path once, its cause kept for the advice."""
    path = state(place)
    with pytest.raises(ip.PopulationUnreadable) as unreadable:
        ip.rows_available(path, True)
    assert str(unreadable.value).count(str(path)) == 1
    assert re.search(reason, str(unreadable.value))
    assert advice in ip.remedy(unreadable.value.__cause__, path)
    if fixable:
        _put_right(path)
        assert ip.rows_available(path, True) == ip.N_MAX - len(ip.WARM_STARTS)


def test_compressed_population_preserves_rows_and_enforces_expanded_limit(
    place,
    space,
    monkeypatch,
):
    """Ordinary NumPy compression is supported within the expanded byte limit."""
    with np.load(COMMITTED) as saved:
        np.savez_compressed(place, **{key: saved[key] for key in saved.files})

    expected = ip.load(COMMITTED, 3, space.lower, space.upper)
    np.testing.assert_array_equal(ip.load(place, 3, space.lower, space.upper), expected)
    monkeypatch.setattr(ip, "_POPULATION_BYTE_LIMIT", place.stat().st_size)
    with pytest.raises(ip.PopulationUnreadable, match="expanded population archive exceeds"):
        ip.rows_available(place)


def test_inconsistent_header_is_refused_before_numpy_loads(place, monkeypatch):
    """Refuse mismatched member lengths before decoding their arrays."""
    path = _a_forged_header(place)

    def unexpected_load(*args, **kwargs):
        pytest.fail("NumPy was called before the archive member size was checked")

    monkeypatch.setattr(ip.np, "load", unexpected_load)
    with pytest.raises(ip.PopulationUnreadable, match="header size does not match the member"):
        ip.rows_available(path)


def test_unused_zip_members_obey_compression_policy(place):
    """The ZIP allowlist covers metadata members even when the reader skips their contents."""
    with zipfile.ZipFile(COMMITTED) as source, zipfile.ZipFile(place, "w") as target:
        for name in source.namelist():
            method = zipfile.ZIP_BZIP2 if name == "command.npy" else zipfile.ZIP_STORED
            target.writestr(name, source.read(name), compress_type=method)
    with pytest.raises(ip.PopulationUnreadable, match="unsupported ZIP compression for command"):
        ip.rows_available(place)


def test_member_offsets_are_checked_before_open(place, monkeypatch):
    """A directory entry outside the file is refused before any member opens."""
    _put_right(place)
    infolist = zipfile.ZipFile.infolist

    def out_of_bounds(archive):
        entries = infolist(archive)
        entries[0].header_offset = place.stat().st_size
        return entries

    monkeypatch.setattr(zipfile.ZipFile, "infolist", out_of_bounds)
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *_: pytest.fail("member was opened"))
    with pytest.raises(ip.PopulationUnreadable, match="ZIP member outside the archive"):
        ip.rows_available(place)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [("overlap", "ZIP member outside the archive"), ("sizes", "inconsistent ZIP stored sizes")],
)
def test_zip_member_metadata_is_consistent_before_open(place, monkeypatch, damage, reason):
    """Unused members must fit their own region and have consistent stored sizes."""
    _put_right(place)
    infolist = zipfile.ZipFile.infolist

    def invalid_entries(archive):
        entries = sorted(infolist(archive), key=lambda entry: entry.header_offset)
        index = next(i for i, entry in enumerate(entries) if entry.filename == "command.npy")
        entry = entries[index]
        if damage == "overlap":
            entry.compress_size = entries[index + 1].header_offset - entry.header_offset - 30
            entry.file_size = entry.compress_size
        else:
            entry.file_size += 1
        return entries

    monkeypatch.setattr(zipfile.ZipFile, "infolist", invalid_entries)
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *_: pytest.fail("member was opened"))
    with pytest.raises(ip.PopulationUnreadable, match=reason):
        ip.rows_available(place)


def test_streaming_zip_data_descriptors_are_supported(place, space):
    """Writing to a non-seekable stream still produces a readable population."""

    class Stream(io.BytesIO):
        def seek(self, *args):
            raise io.UnsupportedOperation("stream cannot seek")

    stream = Stream()
    with zipfile.ZipFile(COMMITTED) as source, zipfile.ZipFile(
        stream, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
    place.write_bytes(stream.getvalue())
    expected = ip.load(COMMITTED, 3, space.lower, space.upper)
    np.testing.assert_array_equal(ip.load(place, 3, space.lower, space.upper), expected)


@pytest.mark.parametrize(
    "header",
    [
        b"{'descr': '<f8', 'fortran_order': False, 'shape': (), 1: 0, }",
        b"{'descr': (), 'fortran_order': False, 'shape': (), }",
        b"{'descr': '>f8', 'fortran_order': False, 'shape': (), }",
        b"{'descr': '<f8', 'fortran_order': True, 'shape': (), }",
        b"{'descr': '<f8', 'fortran_order': False, 'shape': (1, 1, 1), }",
    ],
)
def test_noncanonical_npy_header_is_refused_before_parser(header, place, monkeypatch):
    """Only supported header syntax may reach the NumPy header parser."""
    header += b"\n"
    payload = np.lib.format.magic(1, 0) + struct.pack("<H", len(header)) + header + bytes(8)
    with zipfile.ZipFile(COMMITTED) as source, zipfile.ZipFile(place, "w") as target:
        for name in source.namelist():
            target.writestr(name, payload if name == "normalized.npy" else source.read(name))
    monkeypatch.setattr(
        np.lib.format, "read_array_header_1_0", lambda *_: pytest.fail("header reached NumPy")
    )
    with pytest.raises(ip.PopulationUnreadable, match="unsupported NPY header"):
        ip.rows_available(place)


@given(
    shape=st.lists(st.integers(0, 8), max_size=2).map(tuple),
    descriptor=st.sampled_from(("<f8", "<i8", "<U1", "<U64")),
    version=st.sampled_from(((1, 0), (2, 0))),
)
def test_writer_headers_preflight_and_parse(shape, descriptor, version):
    """Canonical scalar, vector and matrix headers agree with NumPy's parser."""
    array = np.zeros(shape, dtype=descriptor)
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=version, allow_pickle=False)
    np.testing.assert_array_equal(ip._decode_member(buffer.getvalue(), "member", "test"), array)


def test_unexpected_zip_member_open_error_propagates(place, monkeypatch):
    """A library state failure is distinct from corrupt member metadata."""
    _put_right(place)
    failure = ValueError("unexpected ZIP member open failure")

    def fail_open(*args):
        raise failure

    monkeypatch.setattr(zipfile.ZipFile, "open", fail_open)
    with pytest.raises(ValueError) as caught:
        ip.rows_available(place)
    assert caught.value is failure


def test_unexpected_population_validation_error_propagates(place, monkeypatch):
    """An error in our size calculation must leave with its original type."""
    _put_right(place)
    failure = ValueError("unexpected size calculation failure")

    def fail_size(_shape):
        raise failure

    monkeypatch.setattr(ip.math, "prod", fail_size)
    with pytest.raises(ValueError) as caught:
        ip.rows_available(place)
    assert caught.value is failure


def test_seeded_population_format_failures_stay_at_the_file_boundary(place, space):
    """Exercise both readers across repeatable archive and header read failures."""
    rng = random.Random(1731)
    original = COMMITTED.read_bytes()
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    compressed = io.BytesIO()
    with zipfile.ZipFile(compressed, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    del members["normalized.npy"]

    def check_readers():
        try:
            ip.rows_available(place)
        except ip.PopulationUnreadable as error:
            assert str(error).count(str(place)) == 1
        try:
            ip.load(place, 1, space.lower, space.upper)
        except SystemExit as error:
            assert not isinstance(error, ip.RunSizeOutOfRange)
            assert str(error).count(str(place)) == 1

    for archive_bytes in (original, compressed.getvalue()):
        for _ in range(64):
            place.write_bytes(archive_bytes[:rng.randrange(len(archive_bytes))])
            check_readers()
        for _ in range(64):
            damaged = bytearray(archive_bytes)
            damaged[rng.randrange(len(damaged))] ^= 1 << rng.randrange(8)
            place.write_bytes(damaged)
            check_readers()

    for _ in range(128):
        header = b"{" + rng.randbytes(rng.randrange(1, 40)) + b"\n"
        payload = np.lib.format.magic(1, 0) + struct.pack("<H", len(header)) + header
        with zipfile.ZipFile(place, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("normalized.npy", payload)
            for name, data in members.items():
                archive.writestr(name, data)
        check_readers()


def test_the_relative_path_the_optimiser_reads_is_named_once(tmp_path, space, monkeypatch):
    """Guarantee 2 on the path as the optimiser and pilot 1 give it (code review round 18)."""
    monkeypatch.chdir(tmp_path)
    path = ip.population_path(ip.CURRENT_VERSION)
    with pytest.raises(SystemExit) as stopped:
        ip.load(path, 3, space.lower, space.upper)
    assert stopped.value.code.count(path.name) == 1, stopped.value.code


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root reads any file")
def test_a_file_this_user_may_not_read_gets_the_permission_direction(place, space):
    """Guarantee 5: restoring would not help, so it is not advised."""
    _put_right(place)
    place.chmod(0)
    try:
        with pytest.raises(SystemExit) as stopped:
            ip.load(place, 3, space.lower, space.upper)
        with pytest.raises(ip.PopulationUnreadable):
            ip.rows_available(place)
    finally:
        place.chmod(0o644)
    assert "may read" in stopped.value.code and "last commit" not in stopped.value.code


@pytest.mark.parametrize("path", [None, 3])
def test_a_path_that_is_not_a_path_is_a_programming_error(space, path):
    """Guarantee 1, at both entry points."""
    with pytest.raises(TypeError):
        ip.load(path, 3, space.lower, space.upper)
    with pytest.raises(TypeError):
        ip.rows_available(path)


@pytest.mark.parametrize(("path", "why"), [("variable/nul\0x.npz", "NUL"),
                                           (b"variable/nul\0x.npz", "NUL"),
                                           ("", "empty"), (b"", "empty"),
                                           ("\ud800/x.npz", "encode")])  # fmt: skip
def test_a_path_no_file_can_have_is_a_programming_error(space, path, why):
    """Guarantee 1: what the string alone shows names no file (rounds 20 to 22)."""
    for read in (lambda: ip.load(path, 3, space.lower, space.upper),
                 lambda: ip.rows_available(path)):  # fmt: skip
        with pytest.raises(ValueError, match=why) as refused:
            read()
        assert not isinstance(refused.value, ip.PopulationUnreadable)


@given(name=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", max_size=40))
def test_generated_population_paths_keep_argument_and_file_errors_distinct(
    tmp_path_factory,
    name,
):
    """An invalid argument and an absent file take different error paths."""
    root = tmp_path_factory.getbasetemp()
    for path in ("", f"{name}\0.npz", f"{name}\ud800.npz"):
        with pytest.raises(ValueError) as invalid:
            ip.rows_available(path)
        assert not isinstance(invalid.value, ip.PopulationUnreadable)

    with pytest.raises(ip.PopulationUnreadable) as missing:
        ip.rows_available(root / "missing-populations" / f"{name}.npz")
    assert isinstance(missing.value.__cause__, FileNotFoundError)


def test_both_entry_points_read_the_path_as_given(place, space):
    """A trailing slash is not dropped by one and kept by the other (code review round 21)."""
    _put_right(place)
    given = f"{place}/"
    with pytest.raises(SystemExit) as stopped:
        ip.load(given, 3, space.lower, space.upper)
    with pytest.raises(ip.PopulationUnreadable) as unreadable:
        ip.rows_available(given)
    assert given in stopped.value.code and given in str(unreadable.value)


def test_a_file_edited_without_its_sha256_is_refused_by_load_only(place, space):
    """Guarantee 3: load() checks the SHA-256; rows_available() counts the rows."""
    with np.load(COMMITTED) as saved:
        data = {key: saved[key] for key in saved.files}
    data["normalized"] = data["normalized"].copy()
    data["normalized"][4, 4] = 0.5
    np.savez(place, **data)
    with pytest.raises(SystemExit, match="SHA-256") as stopped:
        ip.load(place, 3, space.lower, space.upper)
    assert stopped.value.code.count(str(place)) == 1
    assert ip.rows_available(place) == ip.N_MAX


@pytest.mark.parametrize("version", [None, np.array([1, 2])])
def test_invalid_version_metadata_is_refused_by_load(place, space, version):
    """The version metadata is validated before a run records it."""
    with np.load(COMMITTED) as saved:
        data = {key: saved[key] for key in saved.files if key != "version"}
    if version is not None:
        data["version"] = version
    np.savez(place, **data)
    with pytest.raises(SystemExit, match="cannot be read") as stopped:
        ip.load(place, 3, space.lower, space.upper)
    assert not isinstance(stopped.value, ip.RunSizeOutOfRange)


@pytest.mark.parametrize("dropped", ["version", "sha256"])
def test_the_design_check_refuses_what_load_would(place, dropped):
    """rows_available() reads every array load() does (code review round 19)."""
    with np.load(COMMITTED) as saved:
        np.savez(place, **{key: saved[key] for key in saved.files if key != dropped})
    with pytest.raises(ip.PopulationUnreadable):
        ip.rows_available(place)


def test_an_n_beyond_a_good_file_is_a_run_size_not_a_file_fault(place, space):
    """Guarantees 3, 4: RunSizeOutOfRange, exit 1 here; only the optimiser maps it to 3."""
    _put_right(place)
    with pytest.raises(ip.RunSizeOutOfRange) as stopped:
        ip.load(place, ip.N_MAX + 1, space.lower, space.upper)
    assert isinstance(stopped.value.code, str)
    assert stopped.value.code.count(str(place)) == 1
    assert ip.rows_available(place) == ip.N_MAX
