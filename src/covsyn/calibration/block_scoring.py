# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Score vectors on seed blocks into a table that a killed pilot resumes from (Task 9).

Both pilots score many (vector, block) pairs at some S. Each finished evaluation is appended to
a CSV and flushed at once, so a pilot killed part way keeps every cost it has and its resume
scores only the rest. The pools hold the objective's inputs once per worker (Task 6), one pool
per S, since a worker's S is set when it starts.
"""

from __future__ import annotations

import concurrent.futures
import csv
import io
import json
import logging
import multiprocessing
import os
import pickle
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from covsyn.runtime import hardware

log = logging.getLogger(__name__)

Evaluate = Callable[[np.ndarray, int], tuple]
MakePool = Callable[[int], concurrent.futures.Executor]


def read_rows(path: Path) -> list[dict[str, str]]:
    """The finished rows of a CSV written by :func:`score`; none when it does not exist yet.

    A last line without its newline, which a kill cut off, and a row short of a value are not
    finished evaluations and are left out.
    """
    if not Path(path).is_file():
        return []
    text = Path(path).read_text(newline="")
    if text and not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    return [row for row in csv.DictReader(io.StringIO(text, newline=""))
            if None not in row and None not in row.values()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows, all with the first row's keys, to path."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(path: Path, todo: Iterable[Mapping[str, Any]], columns: Sequence[str],
          vector_of: Callable[[Mapping[str, Any]], np.ndarray], evaluate: Evaluate,
          make_pool: MakePool) -> None:
    """Score each task and append it, with its cost, to the CSV at path.

    Args:
        path: The table; created with its header when missing.
        todo: The tasks still to score, each with ``simulations`` and ``seed_offset`` and the
            other columns but ``cost``.
        columns: The table's columns, ``cost`` last.
        vector_of: The parameter vector a task scores.
        evaluate: ``evaluate(P, seed_offset)`` returning the cost first, run in the pool, as
            fast_cost.evaluate_candidate.
        make_pool: The pool for one S, whose workers run S simulations per evaluation.
    """
    todo = list(todo)
    path = Path(path)
    _drop_cut_off_line(path)
    # A kill can leave the file empty, before its header: then the header is written now.
    empty = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if empty:
            writer.writeheader()
            f.flush()
        for simulations in dict.fromkeys(t["simulations"] for t in todo):
            group = [t for t in todo if t["simulations"] == simulations]
            with make_pool(simulations) as pool:
                futures = {pool.submit(evaluate, vector_of(t), t["seed_offset"]): t
                           for t in group}  # fmt: skip
                try:
                    for future in concurrent.futures.as_completed(futures):
                        writer.writerow({**futures[future], "cost": float(future.result()[0])})
                        f.flush()  # a killed pilot keeps every finished evaluation
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise


def _drop_cut_off_line(path: Path) -> None:
    """Remove a last line a kill cut off, so the next row starts on a line of its own.

    It holds no finished evaluation (read_rows leaves it out), only part of one.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def pilot_workers(given: int | None, env: Mapping[str, str], recorded: Any) -> int:
    """A pilot's workers: --workers, else COVSYN_WORKERS, else its recorded count, else the default.

    So a resume runs with the count it started with unless told otherwise.

    Raises:
        ValueError: --workers, or the recorded count, is not a positive whole number.
    """
    if given is not None or recorded is None:
        return hardware.resolve_workers(given, env, None)
    if type(recorded) is not int or recorded < 1:
        raise ValueError(f"config.json's workers is {recorded!r}, not a positive whole number")
    requested = env.get(hardware.ENV_VARIABLE, "")
    if not requested.strip():  # empty counts as unset, as resolve_workers takes it (B86)
        return recorded
    try:
        return hardware.parse_workers(requested, hardware.ENV_VARIABLE)  # hardware's rule
    except ValueError as error:
        # Unusable: the record, not the cores, since the run started with it (resolve_workers
        # would fall to the cores, which record_workers would then write over the record).
        log.warning("%s; using the %d workers the pilot recorded", error, recorded)
        return recorded


def record_workers(out_dir: Path, workers: int) -> None:
    """Write the workers a resume runs with into config.json, which records the run.

    Written whole or not at all: a temporary file, flushed to disk, replaces it, so a kill, a
    full disk or a crash leaves the old one, which every later resume needs. The replacement
    keeps the file's mode; a config.json that is read-only, a link, hard-linked, owned by
    another user or in a read-only directory is not replaced. A process killed outright, which
    runs no clean-up, leaves its config.json.<random>.tmp, a stale copy that can be deleted;
    it is not removed here, as it may be another resume's at work.

    Raises:
        PermissionError: config.json is read-only, a link, hard-linked, another user's, or
            in a read-only directory.
        OSError: config.json cannot be read, or the new one cannot be written.
        ValueError: config.json is not a JSON object.
    """
    path = Path(out_dir) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} is not a JSON object")
    if config.get("workers") == workers:
        return
    mode = stat.S_IMODE(path.lstat().st_mode)
    # Replacing a link, a hard-linked file or a read-only one would undo what it was made for;
    # the mode bits count too, since os.access says yes to everything for root.
    if path.is_symlink():
        cause = "a symbolic link"
    elif path.stat().st_nlink > 1:
        cause = f"hard-linked, with {path.stat().st_nlink} names"
    elif not os.access(path, os.W_OK) or not mode & 0o222:
        cause = "read-only"
    elif path.stat().st_uid != os.geteuid():  # a replacement would be this user's, not theirs
        cause = "owned by another user"
    elif not os.access(path.parent, os.W_OK):
        cause = "in a read-only directory"
    else:
        cause = ""
    if cause:
        raise PermissionError(f"{path} is {cause}, so its workers stay "
                              f"{config.get('workers')!r}; give a writable copy of its own to "
                              f"resume with {workers}")
    config["workers"] = workers
    # A name of its own, created afresh (O_EXCL, no link followed), so neither a leftover link
    # nor another resume's temporary is written through; private until given config.json's mode.
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        try:
            f = os.fdopen(handle, "w", encoding="utf-8")
        except BaseException:
            os.close(handle)  # fdopen never took it; once it has, only the file closes it
            raise
        with f:
            os.fchmod(f.fileno(), mode)
            f.write(json.dumps(config, indent=1, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    try:  # the rename itself, on file systems that need it; some refuse a directory fsync
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        pass


def objective_pools(workers: int | None) -> tuple[Evaluate, MakePool]:
    """The objective, and a spawned pool per S whose workers hold its inputs (Task 6)."""
    from covsyn.calibration import fast_cost

    count = hardware.resolve_workers(workers, os.environ, None)
    with open("./variable/demographic_parameters.pkl", "rb") as f:
        demo = pickle.load(f)
    with open("./variable/processed_contact_tracing_data.pkl", "rb") as f:
        ct = pickle.load(f)
    cheng = (ct["Cheng_contact_array"], ct["Cheng_attack_rate"], ct["norm_weights"])
    taiwan = np.load("./variable/Taiwan_data_matrix.npy")

    def make_pool(simulations: int) -> concurrent.futures.Executor:
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn"),
            initializer=fast_cost.init_worker,
            initargs=(demo, taiwan.shape[1], False, None, cheng, simulations, taiwan))

    return fast_cost.evaluate_candidate, make_pool
