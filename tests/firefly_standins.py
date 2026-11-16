# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stand-in objectives for the optimiser tests, run in its spawned workers (Task 6).

Firefly.firefly submits ``function(P, seed_offset)`` to a pool started by spawn, one task per
candidate, and expects ``(cost, cost parts, timing fields)`` back. A spawned worker imports
the function by its module and name, so these live at module level here; a closure or a
lambda would not reach the worker.
"""

from __future__ import annotations

import time

import numpy as np


class Killed(Exception):
    """Stands in for the process being killed."""


def cost(P, seed_offset):
    """Deterministic in the vector and the seed block, like the real objective on fixed seeds."""
    return float(np.sum(np.asarray(P) * np.arange(1, len(P) + 1)) % 1.0) + seed_offset / 1e9


def objective(P, seed_offset):
    """The stand-in cost, with no parts and no timings."""
    return cost(P, seed_offset), {}, {}


def recording(P, seed_offset):
    """The stand-in cost, its parts recording the block it was scored on."""
    return cost(P, seed_offset), {"scored_offset": float(seed_offset)}, {}


def scrambled(P, seed_offset):
    """The stand-in cost after a wait set by the vector, so tasks finish out of order."""
    time.sleep(0.05 * (int(np.sum(np.asarray(P)) * 1000) % 7))
    return recording(P, seed_offset)


def killed_on(P, seed_offset, block_offset=None):
    """Raise Killed on the block starting at block_offset, on every block when it is None.

    Bind block_offset with functools.partial, which a spawned worker can unpickle.
    """
    if block_offset is None or seed_offset == block_offset:
        raise Killed
    return objective(P, seed_offset)


def crashes(P, seed_offset):
    """A task whose worker fails, as a broken simulation setup would."""
    raise RuntimeError("worker crashed")


def fails_while_another_runs(P, seed_offset, failing_first_value, flags):
    """Fail one candidate while another task is still running.

    The vector whose first value is failing_first_value records its PID and raises Killed once
    another task has started; every other task records its PID, sleeps 2 s, and records that it
    finished. The
    run then fails while a task is still running, which its shutdown must wait for.
    """
    import os
    from pathlib import Path

    flags = Path(flags)
    if float(P[0]) == failing_first_value:
        deadline = time.monotonic() + 60
        while not any(flags.glob("started_*")):
            if time.monotonic() > deadline:
                raise TimeoutError("no other task started")
            time.sleep(0.01)
        (flags / f"failed_{os.getpid()}").write_text(str(os.getpid()))
        raise Killed
    pid = os.getpid()
    (flags / f"started_{pid}_{seed_offset}_{float(P[0])!r}").write_text(str(pid))
    time.sleep(2.0)
    (flags / f"finished_{pid}_{seed_offset}_{float(P[0])!r}").write_text(str(pid))
    return objective(P, seed_offset)


def with_worker_fields(P, seed_offset):
    """The stand-in cost, with the timing fields fast_cost's task gives, measured here."""
    from covsyn.runtime import timings

    return cost(P, seed_offset), {}, timings.summarise_tasks([timings.TaskClock(False).record()])
