# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The fixed initial population every run starts from (Task 7, items 1 to 3).

``covsyn-initial-population --version N`` writes variable/initial_population_vN.npz: the
normalised coordinates, in [0, 1], of N_MAX fireflies -- row 0 the literature seed vector,
rows 1 to m the warm-start vectors of WARM_STARTS, the rest a Latin hypercube with the fixed
seed LHS_SEED -- with the bounds, the dimension, the warm starts' sources, the generation
command and a SHA-256 of the rows and bounds. A run with n fireflies takes the first n rows.
A run whose bounds or dimension differ from the stored ones is refused rather than clipped,
and a new version is then made; an existing version is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import sys
import tokenize
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from covsyn.calibration.parameter_space import ParameterSpace

N_MAX = 256
# The rework plan's date; any fixed value would do, but it must never change for a version.
LHS_SEED = 20261109
# The best vectors of the warm starts, in row order: run 10, the pipeline's warm start until
# Task 7, and run 19 (B49, E75).
WARM_STARTS = ("results/models/phaseD_run10/firefly_best.txt",
               "results/models/phaseD_run19/firefly_best.txt")  # fmt: skip
VARIABLE = Path("./variable")
CURRENT_VERSION = 1  # the population every run starts from


def _parameter_space():
    """The parameter space module, imported when a population is built or clipped.

    It loads the model and with it scipy, which reading a population does without; only
    building or clipping one needs it.
    """
    from covsyn.calibration import parameter_space

    return parameter_space


def population_path(version: int, directory: Path = VARIABLE) -> Path:
    """Where version ``version`` of the initial population is kept."""
    return Path(directory) / f"initial_population_v{version}.npz"


def digest(normalized: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> str:
    """The SHA-256 of the rows and the bounds, as float64 bytes in that order."""
    sha = hashlib.sha256()
    for array in (normalized, lower, upper):
        sha.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return sha.hexdigest()


def normalise(vector: np.ndarray, lower: np.ndarray, upper: np.ndarray, label: str) -> np.ndarray:
    """A vector's coordinates in [0, 1], clipped to the bounds first, saying what was clipped.

    A parameter whose bounds coincide has span 1, so its coordinate is 0.
    """
    span = np.where(upper > lower, upper - lower, 1.0)
    return np.clip(
        (_parameter_space().clip_warm_start(vector, lower, upper, label) - lower) / span, 0, 1
    )


def best_of(path: Path) -> np.ndarray:
    """The lowest-cost vector of a firefly_best.txt file (iteration, params..., cost)."""
    rows = np.atleast_2d(np.loadtxt(path))
    return rows[int(np.argmin(rows[:, -1])), 1:-1].copy()


def build(space: ParameterSpace, root: Path = Path(".")) -> np.ndarray:
    """The N_MAX rows of a population, from the bounds, the literature vector and WARM_STARTS.

    Raises:
        SystemExit: A warm start does not have the bounds' dimension once padded (B54).
    """
    # Imported here: reading a population loads no scipy; only building one does.
    from scipy.stats import qmc

    lower, upper = space.lower, space.upper
    rows = [normalise(space.literature, lower, upper, "literature vector")]
    for source in WARM_STARTS:
        vector = _parameter_space().pad_pre_b54_vector(best_of(root / source), space.literature)
        if vector.shape != lower.shape:
            raise SystemExit(f"warm start {source}: {vector.size} parameters, "
                             f"expected {lower.size}")  # fmt: skip
        rows.append(normalise(vector, lower, upper, source))
    hypercube = qmc.LatinHypercube(d=len(lower), rng=LHS_SEED).random(N_MAX - len(rows))
    return np.vstack([np.array(rows), hypercube])


def write(path: Path, normalized: np.ndarray, space: ParameterSpace,
          version: int, command: str) -> None:  # fmt: skip
    """Write the population and what it was made from, under a temporary name first."""
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, normalized=normalized, lower=space.lower, upper=space.upper,
             dimension=len(space.lower), version=version, lhs_seed=LHS_SEED,
             warm_starts=np.array(WARM_STARTS, dtype=str), command=np.array(command),
             sha256=np.array(digest(normalized, space.lower, space.upper)))  # fmt: skip
    temporary.replace(path)


class RunSizeOutOfRange(SystemExit):
    """The run's n is below 1 or beyond the rows the population holds.

    A SystemExit with its message, as load()'s other refusals; its own class so the optimiser
    can tell it apart: the n is the run's setting, which a resume keeps, so it exits with
    runs.NOT_RESUMABLE (decision of 2026-11-24, code review round 12). A profile's n is its
    worker count, and a profile is run again, never resumed, so the status holds there too:
    it is run again with fewer workers (code review round 22).
    """


def load(path: str | bytes | os.PathLike, n: int, lower: np.ndarray, upper: np.ndarray,
         without_warm_starts: bool = False) -> np.ndarray:
    """Return the first ``n`` validated rows, optionally excluding warm starts."""
    rows, _ = load_with_metadata(path, n, lower, upper, without_warm_starts)
    return rows


def load_with_metadata(
    path: str | bytes | os.PathLike,
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
    without_warm_starts: bool = False,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """The first ``n`` rows and verified metadata from a single read of ``path``.

    without_warm_starts leaves out the warm starts the file records (rows 1 to their number),
    the fitted vectors of earlier runs, so the run starts from the literature vector and the
    Latin hypercube (pilot 2 rerun, user decision 2026-11-24).

    Keeps the reading contract above PopulationUnreadable.

    Raises:
        TypeError: ``path`` is not a path.
        ValueError: ``path`` is empty, holds a NUL byte or cannot be encoded.
        RunSizeOutOfRange: n is not 1 to the rows available.
        SystemExit: A known read error, invalid format, a SHA-256 mismatch, or a mismatch
            in dimension or bounds.

    Unexpected library exceptions propagate unchanged.
    """
    # Missing or damaged: refused with a message, resumable once remedied (rounds 14, 15).
    try:
        saved = _read(path)  # as given, as rows_available() reads it (code review round 21)
    except PopulationUnreadable as error:
        raise SystemExit(f"{error}; {remedy(error.__cause__, Path(os.fsdecode(path)))}") from error
    path = os.fsdecode(path)  # named as given in the refusals below
    normalized = saved["normalized"]
    stored_lower, stored_upper = saved["lower"], saved["upper"]
    sha = str(saved["sha256"])
    # The file's own warm starts, which the code's WARM_STARTS may no longer match
    # (code review round 4).
    warm_starts = len(saved["warm_starts"])
    if digest(normalized, stored_lower, stored_upper) != sha:
        raise SystemExit(f"{path}: the rows or bounds no longer match its SHA-256")
    if without_warm_starts:
        normalized = np.vstack([normalized[:1], normalized[1 + warm_starts:]])
    if not 1 <= n <= len(normalized):
        without = " without the warm starts" if without_warm_starts else ""
        raise RunSizeOutOfRange(f"{path} holds {len(normalized)} fireflies{without}; a run "
                                f"of {n} cannot start")
    if len(lower) != normalized.shape[1]:
        raise SystemExit(f"{path} was made for dimension {normalized.shape[1]}, not {len(lower)}; "
                         f"make a new version")  # fmt: skip
    if not (np.array_equal(lower, stored_lower) and np.array_equal(upper, stored_upper)):
        moved = np.flatnonzero((lower != stored_lower) | (upper != stored_upper))
        raise SystemExit(f"{path} was made for other bounds (P[{', '.join(map(str, moved))}] "
                         f"differ); make a new version rather than clip")  # fmt: skip
    return normalized[:n].copy(), {"version": int(saved["version"]), "sha256": sha}


# The reading contract of load() and rows_available() (user decision of 2026-11-24, after
# code review rounds 11 to 17 and a logic review):
# 1. A path the string alone shows to name no file is a programming error, raised before
#    any file is touched: TypeError for another type, ValueError for an empty path, a NUL
#    byte, or a str the filesystem cannot encode (UnicodeEncodeError). Whatever the system
#    says on opening it, a name too long included, is a failure of the file (guarantee 2).
# 2. Explicit format checks and the known exceptions listed at each file read or decode
#    operation give a refusal naming the path once with a short reason. Unexpected library
#    exceptions and errors in our own checks propagate unchanged.
# 3. load() checks in this order: the file and its contents, its SHA-256, n against the rows
#    (RunSizeOutOfRange), the dimension, the bounds. No rows at all is a fault of the file.
# 4. Of load()'s refusals, only RunSizeOutOfRange is mapped by the optimiser to
#    runs.NOT_RESUMABLE; the others exit 1, so none of them by itself makes a run not
#    resumable.
# 5. The advice (remedy) follows the exception's type, and for a missing file whether its
#    path is relative (a test of the string alone); it makes no filesystem or git call, and
#    gives a direction, never a command whose premises go unchecked.
# Scope (code review rounds 18, 19): the guarantees are load()'s and rows_available()'s; the
# optimiser, pilot 1 (which exits 1 for every refusal of load(), RunSizeOutOfRange
# included) and pilot 2's design check pass their refusals on in those words. A command-line
# entry point keeps its own status for its argument and design refusals (argparse exits 2;
# pilot 2 so reports a population its design check cannot read, round 22),
# pilot 2 reports a failed run as a RuntimeError naming the run's log, the pipeline as "the
# optimiser failed (see <its log>)"; neither suggests a resume for status runs.NOT_RESUMABLE,
# a fresh start included (rounds 18, 20).
# Imprecision outside these guarantees, such as advice that suits only some repository
# states, is not a defect.
#
# The 2026-11-25 decision replaces round 19's broad catch. The ZIP allowlist restricts
# decompression to STORED and DEFLATE. Each decoder has its own tested exception list,
# and application checks run outside those boundaries.

class PopulationUnreadable(ValueError):
    """The population file is missing, unreadable, or fails a format check."""


def rows_available(path: str | bytes | os.PathLike, without_warm_starts: bool = False) -> int:
    """How many fireflies a run can take from the population at ``path``, as load() counts.

    Keeps the reading contract above PopulationUnreadable. The file's SHA-256 is not
    checked, so a file edited without being hashed again passes here and load() refuses it
    (code review round 16).

    Raises:
        TypeError: ``path`` is not a path, a programming error (code review round 11).
        ValueError: ``path`` is empty, holds a NUL byte or cannot be encoded, so names no
            file (code review rounds 20 to 22).
        PopulationUnreadable: A known read error or failed format check.

    Unexpected library exceptions propagate unchanged.
    """
    # The arrays load() reads, so what passes here loads, unless its SHA-256 no longer
    # matches, which only load() checks (rounds 15, 16, 19).
    saved = _read(path)
    normalized, warm_starts = saved["normalized"], saved["warm_starts"]
    if not without_warm_starts:
        return len(normalized)
    # As load() takes them: row 0 stays, rows 1 to w go (code review round 12).
    return len(normalized[:1]) + len(normalized[1 + len(warm_starts):])


# Both load() and rows_available() decode the same members (code review round 19).
ARRAYS = ("normalized", "lower", "upper", "sha256", "warm_starts", "version")


# The committed v1 archive and its expanded members each take less than 0.5 MiB.
# Leave room for wider future populations while bounding the ZIP and NPY reads.
_POPULATION_BYTE_LIMIT = 8 * 1024 * 1024


# ZIP flags that require features outside the population format.
_UNSUPPORTED_ZIP_FLAGS = (1 << 0) | (1 << 5) | (1 << 6)


@contextmanager
def _reading(shown: str, *expected: type[Exception]) -> Iterator[None]:
    """Translate the documented file errors of one operation."""
    try:
        yield
    except expected as error:
        reason = error.strerror if isinstance(error, OSError) and error.strerror else error
        raise PopulationUnreadable(
            f"{shown} cannot be read ({type(error).__name__}: {reason})"
        ) from error


def _read_archive(path: str | bytes | os.PathLike, shown: str) -> dict[str, np.ndarray]:
    """Read the required arrays from a bounded snapshot, checking headers before loading."""
    with _reading(shown, OSError):
        source = open(path, "rb")
    with source:
        with _reading(shown, OSError):
            data = source.read(_POPULATION_BYTE_LIMIT + 1)
    if not data:
        raise PopulationUnreadable(f"{shown} cannot be read (empty population file)")
    if len(data) > _POPULATION_BYTE_LIMIT:
        raise PopulationUnreadable(
            f"{shown} cannot be read (population archive exceeds its size limit)"
        )
    if data.startswith(np.lib.format.MAGIC_PREFIX):
        raise PopulationUnreadable(f"{shown} cannot be read (one array, not an .npz archive)")

    arrays = {}
    with _reading(shown, zipfile.BadZipFile, UnicodeDecodeError, NotImplementedError):
        archive = zipfile.ZipFile(io.BytesIO(data))
    with archive:
        members = archive.infolist()
        by_name = {member.filename: member for member in members}
        if len(by_name) != len(members):
            raise PopulationUnreadable(f"{shown} cannot be read (duplicate ZIP member names)")
        if sum(member.file_size for member in members) > _POPULATION_BYTE_LIMIT:
            raise PopulationUnreadable(
                f"{shown} cannot be read (expanded population archive exceeds its size limit)"
            )
        _check_zip_members(members, len(data), shown)
        for key in ARRAYS:
            member = by_name.get(f"{key}.npy")
            if member is None:
                raise PopulationUnreadable(f"{shown} cannot be read (missing ZIP member {key}.npy)")
            with _reading(shown, zipfile.BadZipFile, ValueError):
                source = archive.open(member)
            with source:
                with _reading(shown, zipfile.BadZipFile, EOFError, zlib.error):
                    payload = source.read(member.file_size + 1)
            arrays[key] = _decode_member(payload, key, shown)
    return arrays


def _check_zip_members(members: list[zipfile.ZipInfo], size: int, shown: str) -> None:
    """Apply container rules even to entries whose contents we do not use."""
    for member in members:
        name = member.filename.removesuffix(".npy")
        if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise PopulationUnreadable(
                f"{shown} cannot be read (unsupported ZIP compression for {name})"
            )
        if member.flag_bits & _UNSUPPORTED_ZIP_FLAGS:
            raise PopulationUnreadable(f"{shown} cannot be read (unsupported ZIP flags for {name})")
        # The fixed local header occupies 30 bytes before any name or extra data.
        end = member.header_offset + 30 + member.compress_size
        if member.header_offset < 0 or not 0 <= member.compress_size <= size or end > size:
            raise PopulationUnreadable(f"{shown} cannot be read (ZIP member outside the archive)")


def _decode_member(payload: bytes, key: str, shown: str) -> np.ndarray:
    """Check a member's claimed size before NumPy allocates its array."""
    if not payload.startswith(np.lib.format.MAGIC_PREFIX):
        raise PopulationUnreadable(f"{shown} cannot be read ({key} is not an NPY array)")
    buffer = io.BytesIO(payload)
    with _reading(shown, ValueError):
        version = np.lib.format.read_magic(buffer)
    if version == (1, 0):
        read_header = np.lib.format.read_array_header_1_0
    elif version == (2, 0):
        read_header = np.lib.format.read_array_header_2_0
    else:
        raise PopulationUnreadable(
            f"{shown} cannot be read ({key} uses unsupported NPY format {version})"
        )
    with _reading(shown, ValueError, OverflowError, tokenize.TokenError, SyntaxError):
        shape, _, dtype = read_header(buffer)
    remaining = len(payload) - buffer.tell()
    if dtype.hasobject or dtype.itemsize == 0 or any(size < 0 for size in shape):
        raise PopulationUnreadable(
            f"{shown} cannot be read ({key} has an unsupported NPY dtype or shape)"
        )
    if math.prod(shape) * dtype.itemsize != remaining:
        raise PopulationUnreadable(
            f"{shown} cannot be read ({key} header size does not match the member)"
        )
    buffer.seek(0)
    with _reading(shown, ValueError, OverflowError):
        return np.load(buffer, allow_pickle=False)


def _read(path: str | bytes | os.PathLike) -> dict[str, np.ndarray]:
    """The ARRAYS of the population file, for load() and rows_available() alike.

    Raises:
        TypeError: ``path`` is not a path, a programming error (code review round 11).
        ValueError: ``path`` is empty, holds a NUL byte or cannot be encoded, so names no
            file (code review rounds 20 to 22).
        PopulationUnreadable: A known file reading error or a failed format check.
    """
    if not isinstance(path, str | bytes | os.PathLike):  # what np.load takes
        raise TypeError(f"path must be a path, got {type(path).__name__}")
    shown = os.fsdecode(path)  # a bytes path named as text (code review round 13)
    if "\0" in shown:  # names no file, so no remedy for a file applies (round 20)
        raise ValueError(f"path holds a NUL byte: {shown!r}")
    if not shown:  # names no file either (round 21)
        raise ValueError("path is empty")
    os.fsencode(path)  # a str the filesystem cannot encode: UnicodeEncodeError (round 22)
    arrays = _read_archive(path, shown)
    # What load() computes with: rows it can stack and hash, at least one of them, of at least
    # one parameter (rounds 14, 15); a warm-start list it can count; bounds of numbers.
    rows = arrays["normalized"]
    if (rows.ndim != 2 or 0 in rows.shape or rows.dtype.kind not in "iuf"
            or rows.dtype.itemsize > 8):
        raise PopulationUnreadable(f"{shown} cannot be read (its rows are not a non-empty 2-D "
                                   f"array of numbers)")
    if not np.isfinite(rows).all():
        raise PopulationUnreadable(f"{shown} cannot be read (its rows must be finite)")
    if np.any((rows < 0) | (rows > 1)):
        raise PopulationUnreadable(f"{shown} cannot be read (its rows must fall within [0, 1])")
    warm_starts = arrays["warm_starts"]
    if warm_starts.ndim != 1 or warm_starts.dtype.kind not in "US":
        raise PopulationUnreadable(f"{shown} cannot be read (its warm starts are not a list "
                                   f"of strings)")
    slots = len(rows) - 1  # row 0 is reserved for the literature seed
    if len(warm_starts) > slots:
        raise PopulationUnreadable(
            f"{shown} cannot be read (lists {len(warm_starts)} warm starts but holds only {slots})"
        )
    lower, upper = arrays["lower"], arrays["upper"]
    if any(bound.shape != rows.shape[1:] or bound.dtype.kind not in "iuf"
           or bound.dtype.itemsize > 8 for bound in (lower, upper)):
        raise PopulationUnreadable(f"{shown} cannot be read (its bounds are not one number a "
                                   f"parameter)")
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise PopulationUnreadable(f"{shown} cannot be read (its bounds must be finite)")
    if np.any(lower > upper):
        raise PopulationUnreadable(f"{shown} cannot be read (its lower bounds exceed upper bounds)")
    version = arrays["version"]
    if version.ndim != 0 or version.dtype.kind not in "iu":
        raise PopulationUnreadable(f"{shown} cannot be read (its version is not a whole number)")
    if int(version) < 1:
        raise PopulationUnreadable(f"{shown} cannot be read (its version must be positive)")
    sha256 = arrays["sha256"]
    if sha256.ndim != 0 or sha256.dtype.kind != "U":
        raise PopulationUnreadable(f"{shown} cannot be read (its SHA-256 is not a string scalar)")
    return arrays


def remedy(cause: BaseException | None, path: Path) -> str:
    """What to do about a population file that could not be read: a direction, not a command.

    Chosen by the kind of failure only, with no filesystem or git call that could fail in
    turn, and no command whose premises (the repository, its index, the working directory)
    go unchecked (contract of 2026-11-24, after code review rounds 11 to 17). The file is
    restored, not remade, as a remade one may differ: the warm starts must be on disk, and
    the Latin hypercube follows scipy's version.
    """
    # No path: the refusal names it already (guarantee 2, code review round 18).
    committed = "restore the file in the CovSyn repository to its last commit"
    if isinstance(cause, PermissionError):
        return "check that this user may read the file and every directory on its path"
    if isinstance(cause, IsADirectoryError):
        return f"a directory stands where the file should be; {committed}"
    if isinstance(cause, NotADirectoryError):
        return f"a part of its path that should be a directory is not one; {committed}"
    if isinstance(cause, FileNotFoundError):
        relative = ("" if path.is_absolute()
                    else "it is read relative to the working directory, so check that too; ")
        # No command: covsyn-initial-population writes under the working directory, which may
        # not be where the file is read (code review round 18).
        return f"{relative}{committed}, or make it as a new version if it was never committed"
    return committed


def main(argv: list[str] | None = None) -> Path:
    """Write a new version of the initial population (``covsyn-initial-population``)."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", type=int, required=True, help="the version to write, 1 up")
    parser.add_argument("--out-dir", type=Path, default=VARIABLE,
                        help="where to write it (default ./variable)")  # fmt: skip
    args = parser.parse_args(argv)
    if args.version < 1:
        parser.error(f"--version must be 1 or more, not {args.version}")
    path = population_path(args.version, args.out_dir)
    if path.exists():
        raise SystemExit(f"{path} exists; a new population is a new version")
    space = _parameter_space().load()
    command = " ".join(["covsyn-initial-population", *argv])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(path, build(space), space, args.version, command)
    print(f"wrote {path}: {N_MAX} fireflies of dimension {len(space.lower)}")
    return path


def cli() -> int:
    """The console script: run main() on sys.argv, and exit 0 once it returns.

    main() returns the path it wrote, for callers; a console script exits with what its
    function returns, so a path would print and give status 1.
    """
    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
