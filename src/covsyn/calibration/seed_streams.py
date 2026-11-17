# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulation seeds from (stream, block) (Task 7, item 4).

Block b of stream s covers s x 10^7 + b x S to s x 10^7 + (b + 1) x S - 1, so streams are
disjoint for S up to 10,000 and up to 1,000 blocks each, and a use that takes its own stream
never meets another's seeds. The reserved streams follow the rework plan
(doc/plan_Nordling2026_CovSynFireflyNordlingLab.md in covid19-research, Task 7):

====== =========================================================================
0      none: it holds the seeds of every earlier run (the objective's 0..299 and
       later blocks, the chain's 0..999, revalidation from 100,000, reporting
       from 500,000)
1-99   optimisation, one stream per run
100    the stopping rule's monitoring blocks (Task 9)
200-299 validation, one stream per use
300    the reporting checklist and report, on seeds the candidate was not chosen
       on (E97)
301    the synthesis the candidate is chosen on
====== =========================================================================

The plan lists the run report under 301; the report stays on 300, as E97 decided, and the
difference is put to the plan's author.
"""

from __future__ import annotations

STREAM_SPAN = 10**7
MAX_SIMULATIONS = 10_000
MAX_BLOCKS = 1_000

EARLIER_STREAM = 0
OPTIMISATION_STREAMS = range(1, 100)
MONITORING_STREAM = 100
VALIDATION_STREAMS = range(200, 300)
REPORTING_STREAM = 300
SELECTION_STREAM = 301

DEFAULT_OPTIMISATION_STREAM = 1
DEFAULT_VALIDATION_STREAM = 200


def _whole(value, name: str) -> int:
    """The value as an int, refusing a bool, a float or anything else."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number, not {value!r}")
    return value


def _described(allowed) -> str:
    """A range as "a to b", any other collection as its members."""
    if isinstance(allowed, range):
        return f"{allowed.start} to {allowed.stop - 1}"
    return ", ".join(str(stream) for stream in allowed)


def require(stream, allowed, use: str) -> int:
    """The stream, if it is one of ``allowed``, the streams of ``use``.

    Raises:
        ValueError: It is not a whole number, or not one of them; the message names the use.
    """
    stream = _whole(stream, f"the {use} stream")
    if stream not in allowed:
        raise ValueError(f"{stream} is not one of the {use} streams ({_described(allowed)})")
    return stream


def block_seeds(stream: int, block: int, simulations: int) -> range:
    """The seeds of block ``block`` of ``stream`` at ``simulations`` per block.

    Raises:
        ValueError: The stream is not one of 0 to 301, the block not one of 0 to 999, or
            the simulations not 1 to 10,000: beyond those a block could reach the next
            stream.
    """
    stream = _whole(stream, "stream")
    block = _whole(block, "block")
    simulations = _whole(simulations, "simulations")
    if not EARLIER_STREAM <= stream <= SELECTION_STREAM:
        raise ValueError(f"stream {stream} is not one of {EARLIER_STREAM} to {SELECTION_STREAM}")
    if not 0 <= block < MAX_BLOCKS:
        raise ValueError(f"block {block} is not one of 0 to {MAX_BLOCKS - 1}")
    if not 1 <= simulations <= MAX_SIMULATIONS:
        raise ValueError(f"{simulations} simulations per block is not 1 to {MAX_SIMULATIONS}")
    start = stream * STREAM_SPAN + block * simulations
    return range(start, start + simulations)


def block_start(stream: int, block: int, simulations: int) -> int:
    """The first seed of block ``block`` of ``stream``; see block_seeds."""
    return block_seeds(stream, block, simulations).start
