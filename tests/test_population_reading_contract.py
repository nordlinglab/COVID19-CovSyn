# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The reading contract of initial_population.load() and rows_available() (2026-11-24).

Each file state of the contract, through both entry points: what each returns or raises,
the exit status, the path named once (in a directory whose name ends in a parenthesis), the
advice's direction, and that the file loads once put right. The contract itself is the
comment above initial_population.PopulationUnreadable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

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


def _a_forged_header(path):
    """normalized.npy claims 10**11 rows and holds 48 bytes (code review round 18)."""
    import io
    import zipfile

    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        member, {"descr": "<f8", "fortran_order": False, "shape": (10**11, 2)})
    member.write(b"\0" * 48)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("normalized.npy", member.getvalue())
        for name, array in (("lower", np.zeros(2)), ("upper", np.ones(2)),
                            ("warm_starts", np.array(["a"]))):  # fmt: skip
            stored = io.BytesIO()
            np.save(stored, array)
            archive.writestr(f"{name}.npy", stored.getvalue())
    return path


def _corrupt_lzma(path):
    """An archive whose members are LZMA-compressed, the data corrupted (round 19)."""
    import zipfile

    with np.load(COMMITTED) as saved:
        arrays = {key: saved[key] for key in saved.files}
    np.savez(path, **arrays)
    with zipfile.ZipFile(path) as plain:
        members = {name: plain.read(name) for name in plain.namelist()}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_LZMA) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    raw = bytearray(path.read_bytes())
    for i in range(200, 230):
        raw[i] ^= 0xFF
    path.write_bytes(bytes(raw))
    return path


# state, the advice it gets, whether putting the file right at the path fixes it
REFUSED = [
    (_missing, "last commit", True),
    (_a_directory, "a directory stands where the file should be", False),
    (_a_file_for_a_directory, "should be a directory is not one", False),
    (_empty, "last commit", True),
    (_no_rows, "last commit", True),
    (_a_forged_header, "last commit", True),
    (_corrupt_lzma, "last commit", True),
]


@pytest.mark.parametrize(("state", "advice", "fixable"), REFUSED)
def test_load_refuses_a_bad_file_in_words_and_resumably(place, space, state, advice, fixable):
    """Guarantees 2, 4, 5: a SystemExit with a message (exit 1), the path once, a direction."""
    path = state(place)
    with pytest.raises(SystemExit) as stopped:
        ip.load(path, 3, space.lower, space.upper)
    message = stopped.value.code
    assert isinstance(message, str) and not isinstance(stopped.value, ip.RunSizeOutOfRange)
    assert message.count(str(path)) == 1 and advice in message
    assert "git " not in message and "--out-dir" not in message
    if fixable:
        _put_right(path)
        assert len(ip.load(path, 3, space.lower, space.upper)) == 3


@pytest.mark.parametrize(("state", "advice", "fixable"), REFUSED)
def test_rows_available_refuses_the_same_files(place, state, advice, fixable):
    """Guarantee 2: PopulationUnreadable naming the path once, its cause kept for the advice."""
    path = state(place)
    with pytest.raises(ip.PopulationUnreadable) as unreadable:
        ip.rows_available(path, True)
    assert str(unreadable.value).count(str(path)) == 1
    assert advice in ip.remedy(unreadable.value.__cause__, path)
    if fixable:
        _put_right(path)
        assert ip.rows_available(path, True) == ip.N_MAX - len(ip.WARM_STARTS)


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
                                           ("", "empty"), (b"", "empty")])  # fmt: skip
def test_a_path_no_file_can_have_is_a_programming_error(space, path, why):
    """Guarantee 1: a NUL byte or nothing names no file (code review rounds 20, 21)."""
    for read in (lambda: ip.load(path, 3, space.lower, space.upper),
                 lambda: ip.rows_available(path)):  # fmt: skip
        with pytest.raises(ValueError, match=why) as refused:
            read()
        assert not isinstance(refused.value, ip.PopulationUnreadable)


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
def test_a_version_describe_cannot_read_is_refused_by_load(place, space, version):
    """What describe() reads after load(), load() checks (guarantee 2, round 18)."""
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
