# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A pilot or production run's n, G and S, which no mode fixes (Task 9).

Pilot 2's runs take n, G and S from the command line, and the production run n and S, with G
its G_max; a resume takes them from config.json. Each must be within what a run can hold: n
within the initial population's rows and above 1, so one firefly moves; a generation per block
of the stream; S within one block's seeds.
"""

from __future__ import annotations

from typing import Any

from covsyn.calibration import seed_streams

# initial_population.N_MAX, the population's rows, held here so that the pipeline, which reads
# it, need not import the population module and scipy (a test keeps the two equal).
POPULATION_ROWS = 256
POP_SIZES = range(2, POPULATION_ROWS + 1)
GENERATIONS = range(1, seed_streams.MAX_BLOCKS + 1)
SIMULATIONS = range(1, seed_streams.MAX_SIMULATIONS + 1)


def _within(allowed: range) -> str:
    return f"{allowed.start} to {allowed.stop - 1}"


def from_command(mode: str, pop_size: int | None, generations: int | None,
                 simulations: int | None) -> tuple[int, int | None, int]:
    """n, G and S as given on the command line, each required and in range.

    G is the pilot mode's only: production runs to G_max, so its generations are given as None
    and returned unchecked.

    Raises:
        ValueError: One is missing or out of range, naming its flag and the mode.
    """
    checks = [("--pop-size", pop_size, POP_SIZES)]
    if mode == "pilot":
        checks.append(("--generations", generations, GENERATIONS))
    checks.append(("--simulations", simulations, SIMULATIONS))
    for flag, value, allowed in checks:
        if value is None:
            raise ValueError(f"--mode {mode} needs {flag}, which no default stands in for")
        if value not in allowed:
            raise ValueError(f"--mode {mode}: {flag} must be {_within(allowed)}, not {value}")
    return pop_size, generations, simulations


def from_settings(settings: dict[str, Any]) -> tuple[int, int, int]:
    """n, G and S as config.json's settings record them (pop_size, max_generations, simulations).

    Raises:
        ValueError: One is missing, not a whole number, or out of range, naming its key.
    """
    found = []
    for key, allowed in (("pop_size", POP_SIZES), ("max_generations", GENERATIONS),
                         ("simulations", SIMULATIONS)):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
            raise ValueError(f"{key} is {value!r}, not {_within(allowed)}")
        found.append(value)
    return found[0], found[1], found[2]
