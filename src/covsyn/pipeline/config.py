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

from covsyn.calibration import seed_streams

PACKAGED = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineConfig:
    """Every setting of a pipeline run."""

    mode: str = "train"
    config_name: str = "phase-d"
    stream: int = seed_streams.DEFAULT_OPTIMISATION_STREAM  # Task 7
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
    except tomllib.TOMLDecodeError as error:
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
    return PipelineConfig(**values)


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
