# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run one pipeline step and record its peak memory (Task 5).

``python -m covsyn.runtime.measure -m <module> <args>`` runs the module as ``python -m`` would,
then appends one JSON line to the file named by ``COVSYN_MEASURE``: the step's own peak RSS and
its largest worker's. ``-c <code>`` and a script path work too.

A step is a fresh process with a fresh pool, so its ``ru_maxrss`` covers that step alone, and
``RUSAGE_CHILDREN`` its largest worker. The line is written at exit, after the pools' workers
have been joined, so the largest worker is known; the step's exit status is kept. The step runs
no thread of its own, as one would make every fork of its pool workers unsafe; the pipeline
samples the available memory instead.
"""

from __future__ import annotations

import atexit
import json
import os
import resource
import runpy
import sys

from covsyn.runtime import memory

ENV = "COVSYN_MEASURE"


def _mb(n: int) -> float | None:
    return round(n / memory.MB, 1) if n else None  # no child: RUSAGE_CHILDREN is 0


def _record(path: str, step: str) -> None:
    line = {
        "step": step,
        "parent_peak_mb": _mb(memory.peak_rss_bytes(resource.RUSAGE_SELF)),
        "worker_peak_mb": _mb(memory.peak_rss_bytes(resource.RUSAGE_CHILDREN)),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def run(argv: list[str]) -> None:
    """Run ``argv`` as the arguments of ``python`` would be."""
    if argv[0] == "-m":
        sys.argv = [argv[1], *argv[2:]]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    elif argv[0] == "-c":
        sys.argv = ["-c", *argv[2:]]
        exec(compile(argv[1], "<string>", "exec"), {"__name__": "__main__"})
    else:
        sys.argv = list(argv)
        runpy.run_path(argv[0], run_name="__main__")


def main() -> None:
    """Measure the step given on the command line."""
    argv = sys.argv[1:]
    if not argv:
        sys.exit("usage: uv run python -m covsyn.runtime.measure -m <module> [args]")
    path = os.environ.get(ENV)
    if path is not None:
        step = {"-m": f"-m {argv[1]}", "-c": "-c"}.get(argv[0], argv[0])
        atexit.register(_record, path, step)
    run(argv)


if __name__ == "__main__":
    main()
