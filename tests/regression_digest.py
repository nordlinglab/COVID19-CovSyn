# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical fingerprints of simulation output, for the regression tests.

A restructuring must not change a single simulated number.
Comparing the full output of a simulation would need large fixtures, so each output is reduced to
a SHA-256 digest of a canonical byte encoding: arrays by dtype, shape and raw bytes, mappings by
sorted key, and scalars by ``repr``, which round-trips floats exactly.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def _encode(obj: Any, out: hashlib._Hash) -> None:
    """Feed a canonical encoding of ``obj`` into the hash ``out``."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            out.update(b"objarray")
            out.update(repr(obj.shape).encode())
            for item in obj.ravel():
                _encode(item, out)
        else:
            out.update(f"array|{obj.dtype.str}|{obj.shape}|".encode())
            out.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, Mapping):
        out.update(b"map{")
        for key in sorted(obj, key=str):
            out.update(repr(key).encode())
            _encode(obj[key], out)
        out.update(b"}")
    elif isinstance(obj, list | tuple):
        out.update(f"seq{len(obj)}[".encode())
        for item in obj:
            _encode(item, out)
        out.update(b"]")
    elif isinstance(obj, float | np.floating):
        value = float(obj)
        out.update(b"nan" if math.isnan(value) else repr(value).encode())
    elif isinstance(obj, np.integer | np.bool_):
        out.update(repr(obj.item()).encode())
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        _encode(vars(obj), out)
    else:
        out.update(repr(obj).encode())
    out.update(b";")


def digest(obj: Any) -> str:
    """Return the SHA-256 hex digest of the canonical encoding of ``obj``."""
    out = hashlib.sha256()
    _encode(obj, out)
    return out.hexdigest()


def encode_float(value: float) -> float | str:
    """Store a float in JSON exactly; NaN, which JSON cannot hold, becomes the string ``"nan"``."""
    return "nan" if isinstance(value, float) and math.isnan(value) else value


def decode_float(value: float | str) -> float:
    """Inverse of :func:`encode_float`."""
    return math.nan if value == "nan" else float(value)
