# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The pipeline's settings, read from a TOML file (Task 4).

``--config default`` names ``default.toml`` beside this module; any other value is a path. Every
setting has the default of that file, a setting the file does not know is an error, and so is a
value of the wrong type, as a typo would otherwise run for hours on the default.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from covsyn.calibration import run_size, seed_streams, step_schedule

PACKAGED = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineConfig:
    """Every setting of a pipeline run."""

    mode: str = "train"
    config_name: str = "phase-d"
    stream: int = seed_streams.DEFAULT_OPTIMISATION_STREAM  # Task 7
    # Task 8: the step schedule; None leaves each to the optimiser's default
    alpha_0: float | None = None
    alpha_end: float | None = None
    g_plan: int | None = None
    # Task 9: production's n, S and pilot 2's run id; None in every other mode
    pop_size: int | None = None
    simulations: int | None = None
    pilot: str | None = None
    revalidate: bool = True
    revalidation_blocks: int = 10
    validation_stream: int = seed_streams.DEFAULT_VALIDATION_STREAM  # Task 7
    select: bool = True
    select_top: int = 5
    monte_carlo: int = 1000
    age_rr_cases: int = 20000
    previous_checks: str = ""
    workers: int | None = None  # None: each step's own default


# Settings that existed and are gone, with the reason a file that still sets one is given.
REMOVED = {
    ("optimise", "warm_start"): "the warm starts are rows of the initial population since "
    "Task 7; make a new version with covsyn-initial-population to change them",
    ("synthesise", "independent_seed_start"): "since Task 7 the reported synthesis takes its "
    "seeds from stream 300 and the chosen one from stream 301",
}

# Task 7: counts a stream bounds -- a block holds at most 10,000 seeds, a stream 1,000 blocks,
# and the age risk ratio's seeds, the first of a stream, at most 10^7 -- checked when the file
# is read rather than when the stage starts.
MAXIMA = {
    "monte_carlo": seed_streams.MAX_SIMULATIONS,
    "simulations": seed_streams.MAX_SIMULATIONS,  # Task 9: one block's seeds
    "pop_size": run_size.POP_SIZES.stop - 1,  # Task 9: the initial population's rows
    "revalidation_blocks": seed_streams.MAX_BLOCKS,
    "age_rr_cases": seed_streams.STREAM_SPAN,
}

# Task 7: settings that name a stream, and the streams each may name.
STREAMS = {
    "stream": (seed_streams.OPTIMISATION_STREAMS, "optimisation"),
    "validation_stream": (seed_streams.VALIDATION_STREAMS, "validation"),
}

# TOML section and key -> field, with the check each value must pass.
SCHEMA: dict[str, dict[str, tuple[str, type, Any]]] = {
    "optimise": {
        "mode": ("mode", str, None),
        "config_name": ("config_name", str, None),
        "stream": ("stream", int, None),
        "alpha_0": ("alpha_0", float, None),
        "alpha_end": ("alpha_end", float, None),
        "g_plan": ("g_plan", int, 1),
        "pop_size": ("pop_size", int, 2),  # Task 9: at least one firefly moves
        "simulations": ("simulations", int, 1),
        "pilot": ("pilot", str, None),
    },
    "revalidate": {
        "enabled": ("revalidate", bool, None),
        "blocks": ("revalidation_blocks", int, 1),
        "stream": ("validation_stream", int, None),
    },
    "select": {"enabled": ("select", bool, None), "top": ("select_top", int, 1)},
    "synthesise": {
        "monte_carlo": ("monte_carlo", int, 1),
    },
    "check": {
        "age_rr_cases": ("age_rr_cases", int, 1),
        "previous_checks": ("previous_checks", str, None),
    },
    "run": {"workers": ("workers", int, 0)},
}


def _value(section: str, key: str, raw: Any) -> tuple[str, Any]:
    field, kind, minimum = SCHEMA[section][key]
    # TOML writes a whole step size such as 1 as an integer.
    if kind is float and isinstance(raw, int) and not isinstance(raw, bool):
        try:
            raw = float(raw)
        except OverflowError:
            raise ValueError(f"[{section}] {key} is too large, a {len(str(abs(raw)))}-digit "
                             f"integer") from None
    # bool is an int in Python; a count given as true must not pass.
    if not isinstance(raw, kind) or (kind is int and isinstance(raw, bool)):
        raise ValueError(f"[{section}] {key} must be {kind.__name__}, got {raw!r}")
    if minimum is not None and raw < minimum:
        raise ValueError(f"[{section}] {key} must be at least {minimum}, got {raw!r}")
    if field in MAXIMA and raw > MAXIMA[field]:
        raise ValueError(f"[{section}] {key} must be at most {MAXIMA[field]}, got {raw!r}")
    if field in STREAMS:
        try:
            seed_streams.require(raw, *STREAMS[field])
        except ValueError as error:
            raise ValueError(f"[{section}] {key}: {error}") from None
    if field == "workers":
        return field, raw or None
    return field, raw


def parse(text: str, source: str) -> PipelineConfig:
    """The configuration a TOML text gives; ``source`` names it in errors.

    Raises:
        ValueError: An unknown section or key, a value of the wrong type, or one out of range.
    """
    try:
        data = tomllib.loads(text)
    except ValueError as error:  # TOMLDecodeError, or an integer over Python's digit limit
        raise ValueError(f"{source}: {error}") from None
    values: dict[str, Any] = {}
    for section, keys in data.items():
        if section not in SCHEMA or not isinstance(keys, dict):
            raise ValueError(f"{source}: unknown section [{section}]")
        for key, raw in keys.items():
            if (section, key) in REMOVED:
                raise ValueError(f"{source}: [{section}] {key} is gone: {REMOVED[section, key]}. "
                                 f"In a new configuration, remove it; a saved pipeline.toml that "
                                 f"sets it belongs to a run started before Task 7, which cannot "
                                 f"be resumed")  # fmt: skip
            if key not in SCHEMA[section]:
                raise ValueError(f"{source}: unknown setting {key!r} in [{section}]")
            field, value = _value(section, key, raw)
            values[field] = value
    _check_schedule(values, source)
    _check_production(values, source)
    return PipelineConfig(**values)


# Task 10: the blocks production's revalidation re-scores the archive's best on.
TASK10_BLOCKS = 30

# Task 9: what production takes from pilot 2, which no default stands in for.
PRODUCTION_SETTINGS = ("pop_size", "simulations", "pilot", "alpha_0", "alpha_end", "g_plan")


def _check_production(values: dict[str, Any], source: str) -> None:
    """Production needs pilot 2's choice whole; every other mode fixes n and S (Task 9)."""
    mode = values.get("mode", PipelineConfig.mode)
    if mode == "production":
        # Task 10: its revalidation is the cost's measurement, on 30 blocks (decision of
        # 2026-11-22); unset, the 30 apply, and another count is refused.
        blocks = values.setdefault("revalidation_blocks", TASK10_BLOCKS)
        if blocks != TASK10_BLOCKS:
            raise ValueError(f"{source}: [revalidate] blocks is {blocks}, but production re-scores "
                             f"on {TASK10_BLOCKS} blocks (Task 10). Remove the setting; a saved "
                             f"run whose pipeline.toml sets another count cannot be resumed, so "
                             f"start a new run")  # fmt: skip
        missing = [key for key in PRODUCTION_SETTINGS if values.get(key) is None]
        if missing:
            raise ValueError(f"{source}: [optimise] mode \"production\" needs "
                             f"{', '.join(missing)}, pilot 2's choice")
        _check_production_rows(values["pop_size"], source)
        return
    given = [key for key in ("pop_size", "simulations", "pilot") if values.get(key) is not None]
    if given:
        raise ValueError(f"{source}: [optimise] {', '.join(given)} is for mode \"production\" "
                         f"only; mode {mode!r} fixes it")


def _check_production_rows(pop_size: int, source: str) -> None:
    """Refuse an n beyond the rows production has without the warm starts (2026-11-24).

    Read from the population file, as the optimiser counts them; a file that cannot be read
    leaves the refusal to the optimiser, which loads it before claiming its run directory.
    """
    from covsyn.calibration import initial_population
    from covsyn.runtime import runs

    try:  # the repository's file, wherever the pipeline is started (code review round 7)
        rows = initial_population.rows_available(initial_population.population_path(
            initial_population.CURRENT_VERSION, runs.REPO_ROOT / "variable"), True)
    except Exception:  # any unreadable file: the optimiser loads and refuses it (round 7)
        return
    if pop_size > rows:
        raise ValueError(f"{source}: [optimise] pop_size is {pop_size}, but production starts "
                         f"without the warm starts, which leaves {rows} fireflies")


def _check_schedule(values: dict[str, Any], source: str) -> None:
    """Refuse a mode or schedule the optimiser would refuse, unset ends at its defaults (Task 8).

    A g_plan longer than the mode's generations is not refused: see schedule_warnings.
    """
    mode = values.get("mode", PipelineConfig.mode)
    if mode not in step_schedule.MODE_GENERATIONS:  # the optimiser's modes
        raise ValueError(f"{source}: [optimise] mode {mode!r} is not one of "
                         f"{sorted(step_schedule.MODE_GENERATIONS)}")
    if step_schedule.MODE_GENERATIONS[mode] is None:  # Task 9: its size is the pilot's to give
        raise ValueError(f"{source}: [optimise] mode {mode!r} is run by covsyn-pilot-size, "
                         f"not the pipeline")
    if all(values.get(field) is None for field in ("alpha_0", "alpha_end", "g_plan")):
        return
    generations = step_schedule.MODE_GENERATIONS[mode]
    try:
        step_schedule.checked_schedule(
            values.get("alpha_0"), values.get("alpha_end"), values.get("g_plan"), generations,
            alpha_0_default=step_schedule.MODE_ALPHA_0[mode])
    except ValueError as error:
        raise ValueError(f"{source}: [optimise] step schedule: {error}") from None


def schedule_warnings(config: PipelineConfig) -> list[str]:
    """What the pipeline's log warns of in the step schedule before a run starts (Task 8).

    A g_plan longer than the mode's generations runs, as the optimiser does, but may be a typo
    (decision of 2026-11-20); the log keeps it with the run, which a warning from parse() would
    not reach.
    """
    generations = step_schedule.MODE_GENERATIONS.get(config.mode)
    if config.g_plan is None or generations is None:  # no mode: the optimiser refuses it
        return []
    outlasts = step_schedule.plan_outlasts_run(config.g_plan, generations, "g_plan")
    return [] if outlasts is None else [f"[optimise] {outlasts}"]


def load(name: str) -> PipelineConfig:
    """The configuration named ``name``: a packaged one, such as ``default``, or a TOML path.

    Raises:
        FileNotFoundError: Neither a packaged configuration nor a file.
        ValueError: See :func:`parse`.
    """
    packaged = PACKAGED / f"{name}.toml"
    path = packaged if "/" not in name and packaged.is_file() else Path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no configuration {name!r}: neither {packaged} nor a file")
    return parse(path.read_text(encoding="utf-8"), str(path))


def to_toml(config: PipelineConfig) -> str:
    """The configuration as TOML that :func:`parse` reads back, saved with each run."""
    lines = ["# The configuration this run started with (covsyn-pipeline, Task 4)."]
    for section, keys in SCHEMA.items():
        lines.append(f"\n[{section}]")
        for key, (field, _, _) in keys.items():
            value = getattr(config, field)
            if field == "workers":
                value = value or 0
            if isinstance(value, tuple):
                value = list(value)
            if value is None:  # Task 8: unset, so the optimiser's default; TOML has no None
                continue
            lines.append(f"{key} = {_toml(value)}")
    return "\n".join(lines) + "\n"


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return str(value)


def replace(config: PipelineConfig, **changes: Any) -> PipelineConfig:
    """A copy of ``config`` with ``changes``."""
    return dataclasses.replace(config, **changes)
