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
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from covsyn.calibration import parameter_space

N_MAX = 256
# The rework plan's date; any fixed value would do, but it must never change for a version.
LHS_SEED = 20261109
# The best vectors of the warm starts, in row order: run 10, the pipeline's warm start until
# Task 7, and run 19 (B49, E75).
WARM_STARTS = ("results/models/phaseD_run10/firefly_best.txt",
               "results/models/phaseD_run19/firefly_best.txt")  # fmt: skip
VARIABLE = Path("./variable")
CURRENT_VERSION = 1  # the population every run starts from


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
        (parameter_space.clip_warm_start(vector, lower, upper, label) - lower) / span, 0, 1
    )


def best_of(path: Path) -> np.ndarray:
    """The lowest-cost vector of a firefly_best.txt file (iteration, params..., cost)."""
    rows = np.atleast_2d(np.loadtxt(path))
    return rows[int(np.argmin(rows[:, -1])), 1:-1].copy()


def build(space: parameter_space.ParameterSpace, root: Path = Path(".")) -> np.ndarray:
    """The N_MAX rows of a population, from the bounds, the literature vector and WARM_STARTS.

    Raises:
        SystemExit: A warm start does not have the bounds' dimension once padded (B54).
    """
    lower, upper = space.lower, space.upper
    rows = [normalise(space.literature, lower, upper, "literature vector")]
    for source in WARM_STARTS:
        vector = parameter_space.pad_pre_b54_vector(best_of(root / source), space.literature)
        if vector.shape != lower.shape:
            raise SystemExit(f"warm start {source}: {vector.size} parameters, "
                             f"expected {lower.size}")  # fmt: skip
        rows.append(normalise(vector, lower, upper, source))
    hypercube = qmc.LatinHypercube(d=len(lower), rng=LHS_SEED).random(N_MAX - len(rows))
    return np.vstack([np.array(rows), hypercube])


def write(path: Path, normalized: np.ndarray, space: parameter_space.ParameterSpace,
          version: int, command: str) -> None:  # fmt: skip
    """Write the population and what it was made from, under a temporary name first."""
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, normalized=normalized, lower=space.lower, upper=space.upper,
             dimension=len(space.lower), version=version, lhs_seed=LHS_SEED,
             warm_starts=np.array(WARM_STARTS), command=np.array(command),
             sha256=np.array(digest(normalized, space.lower, space.upper)))  # fmt: skip
    temporary.replace(path)


def load(path: Path, n: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """The first ``n`` rows of the population at ``path``, for a run with these bounds.

    Raises:
        SystemExit: The file is missing or its rows do not match its SHA-256, n is not 1 to
            N_MAX, or the dimension or the bounds differ from those it was made with.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{path} not found: make it with covsyn-initial-population --version N")
    with np.load(path) as saved:
        normalized = saved["normalized"]
        stored_lower, stored_upper = saved["lower"], saved["upper"]
        sha = str(saved["sha256"])
    if digest(normalized, stored_lower, stored_upper) != sha:
        raise SystemExit(f"{path}: the rows or bounds no longer match its SHA-256")
    if not 1 <= n <= len(normalized):
        raise SystemExit(f"{path} holds {len(normalized)} fireflies; a run of {n} cannot start")
    if len(lower) != normalized.shape[1]:
        raise SystemExit(f"{path} was made for dimension {normalized.shape[1]}, not {len(lower)}; "
                         f"make a new version")  # fmt: skip
    if not (np.array_equal(lower, stored_lower) and np.array_equal(upper, stored_upper)):
        moved = np.flatnonzero((lower != stored_lower) | (upper != stored_upper))
        raise SystemExit(f"{path} was made for other bounds (P[{', '.join(map(str, moved))}] "
                         f"differ); make a new version rather than clip")  # fmt: skip
    return normalized[:n].copy()


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
    space = parameter_space.load()
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
