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
import os
import sys
import zipfile
import zlib
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

    It loads the model and with it scipy, which the pipeline's configuration check, reading
    this module for the rows a run may take, does without.
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
    # Imported here: the pipeline's configuration check reads this module and loads no scipy.
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
             warm_starts=np.array(WARM_STARTS), command=np.array(command),
             sha256=np.array(digest(normalized, space.lower, space.upper)))  # fmt: skip
    temporary.replace(path)


class RunSizeOutOfRange(SystemExit):
    """The run's n is below 1 or beyond the rows the population holds.

    A SystemExit with its message, as load()'s other refusals; its own class so the optimiser
    can tell it apart: the n is the run's setting, which a resume keeps, so it exits with
    runs.NOT_RESUMABLE (decision of 2026-11-24, code review round 12).
    """


def load(path: str | bytes | os.PathLike, n: int, lower: np.ndarray, upper: np.ndarray,
         without_warm_starts: bool = False) -> np.ndarray:
    """The first ``n`` rows of the population at ``path``, for a run with these bounds.

    without_warm_starts leaves out the warm starts the file records (rows 1 to their number),
    the fitted vectors of earlier runs, so the run starts from the literature vector and the
    Latin hypercube (pilot 2 rerun, user decision 2026-11-24).

    Keeps the reading contract above READ_ERRORS.

    Raises:
        TypeError: ``path`` is not a path.
        RunSizeOutOfRange: n is not 1 to the rows available.
        SystemExit: The file is missing, cannot be read, or its rows do not match its
            SHA-256, or the dimension or the bounds differ from those it was made with.
    """
    path = Path(os.fsdecode(path))  # the paths rows_available takes (code review round 14)
    # Missing or damaged: refused with a message, resumable once remedied (rounds 14, 15).
    try:
        # version too, which the optimiser's describe() reads next (code review round 18)
        saved = _read(path, ("normalized", "lower", "upper", "sha256", "warm_starts", "version"))
    except PopulationUnreadable as error:
        raise SystemExit(f"{error}; {remedy(error.__cause__, path)}") from error
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
    return normalized[:n].copy()


# The reading contract of load() and rows_available() (user decision of 2026-11-24, after
# code review rounds 11 to 17 and a logic review):
# 1. A path that is not a path is a programming error, raised as TypeError before any file
#    is touched.
# 2. A failure of the file -- opening, decoding, or contents load() cannot compute with --
#    is a worded refusal naming the path once with a short reason, never a traceback; any
#    other exception is a programming error and raises as itself.
# 3. load() checks in this order: the file and its contents, its SHA-256, n against the rows
#    (RunSizeOutOfRange), the dimension, the bounds. No rows at all is a fault of the file.
# 4. Only the optimiser maps RunSizeOutOfRange to runs.NOT_RESUMABLE; every other refusal
#    exits 1, so none of them by itself makes a run not resumable.
# 5. The advice (remedy) follows the exception's type alone, makes no filesystem or git call,
#    and gives a direction, never a command whose premises go unchecked.
# Imprecision outside these guarantees, such as advice that suits only some repository
# states, is not a defect.
#
# What reading and decoding an .npz can raise: OSError (missing, unreadable), EOFError
# (empty), ValueError (not an .npz, pickled data), KeyError (an array missing), BadZipFile
# (truncated), zlib.error (a corrupt stream), NotImplementedError (an unknown compression),
# RuntimeError (zipfile's other refusals), MemoryError and OverflowError (a header claiming
# a shape too large to allocate or index, code review round 18). The file is 256 rows, so a
# MemoryError while reading it is taken for a damaged header, not a machine out of memory.
READ_ERRORS = (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile, zlib.error,
               NotImplementedError, RuntimeError, MemoryError, OverflowError)


class PopulationUnreadable(ValueError):
    """The population file is missing or cannot be read, whatever the reason."""


def rows_available(path: str | bytes | os.PathLike, without_warm_starts: bool = False) -> int:
    """How many fireflies a run can take from the population at ``path``, as load() counts.

    Keeps the reading contract above READ_ERRORS. The file's SHA-256 is not checked, so a
    file edited without being hashed again passes here and load() refuses it (code review
    round 16).

    Raises:
        TypeError: ``path`` is not a path, a programming error (code review round 11).
        PopulationUnreadable: The file cannot be read or decoded -- missing, empty,
            truncated, a corrupt or unknown compression, rows that are not a non-empty 2-D
            array of numbers, bounds that are not numbers -- so a caller catches this alone;
            any other failure, a programming error, raises as itself (code review rounds 7
            to 15).
    """
    # Every array load() computes with, so what passes here loads, unless its SHA-256 no
    # longer matches, which only load() checks (rounds 15, 16).
    saved = _read(path, ("normalized", "lower", "upper", "warm_starts"))
    normalized, warm_starts = saved["normalized"], saved["warm_starts"]
    if not without_warm_starts:
        return len(normalized)
    # As load() takes them: row 0 stays, rows 1 to w go (code review round 12).
    return len(normalized[:1]) + len(normalized[1 + len(warm_starts):])


def _read(path: str | bytes | os.PathLike, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The arrays ``keys`` of the population file, for load() and rows_available() alike.

    Raises:
        TypeError: ``path`` is not a path, a programming error (code review round 11).
        PopulationUnreadable: The file cannot be read or decoded (see READ_ERRORS), holds one
            array instead of an archive, or its rows, warm starts or bounds are not what
            load() computes with.
    """
    if not isinstance(path, str | bytes | os.PathLike):  # what np.load takes
        raise TypeError(f"path must be a path, got {type(path).__name__}")
    shown = os.fsdecode(path)  # a bytes path named as text (code review round 13)
    try:
        saved = np.load(path)
        # A single .npy array loads as an array, not an archive (code review round 12).
        archive = isinstance(saved, np.lib.npyio.NpzFile)
        if archive:
            with saved:
                arrays = {key: saved[key] for key in keys}
    except READ_ERRORS as error:
        # An OSError's text repeats the path; its strerror does not (code review round 11).
        reason = error.strerror if isinstance(error, OSError) and error.strerror else error
        raise PopulationUnreadable(f"{shown} cannot be read ({type(error).__name__}: "
                                   f"{reason})") from error
    if not archive:
        raise PopulationUnreadable(f"{shown} cannot be read (one array, not an .npz archive)")
    # What load() computes with: rows it can stack and hash, at least one of them, of at least
    # one parameter (rounds 14, 15); a warm-start list it can count; bounds of numbers.
    rows = arrays["normalized"]
    if rows.ndim != 2 or 0 in rows.shape or rows.dtype.kind not in "iuf":
        raise PopulationUnreadable(f"{shown} cannot be read (its rows are not a non-empty 2-D "
                                   f"array of numbers)")
    if arrays["warm_starts"].ndim == 0:
        raise PopulationUnreadable(f"{shown} cannot be read (its warm starts are not a list)")
    if any(arrays[key].shape != rows.shape[1:] or arrays[key].dtype.kind not in "iuf"
           for key in ("lower", "upper") if key in arrays):  # one bound a parameter (round 16)
        raise PopulationUnreadable(f"{shown} cannot be read (its bounds are not one number a "
                                   f"parameter)")
    if "version" in arrays and (arrays["version"].ndim != 0
                                or arrays["version"].dtype.kind not in "iu"):
        raise PopulationUnreadable(f"{shown} cannot be read (its version is not a whole number)")
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


def describe(path: Path) -> dict:
    """The version and SHA-256 config.json records for the population at ``path``."""
    with np.load(path) as saved:
        return {"version": int(saved["version"]), "sha256": str(saved["sha256"])}


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
