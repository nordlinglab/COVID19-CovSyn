# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/data_synthesis.sh, and the worker count the checklist reads (B86, Task 2).

These tests come from test_b86_launch_env_forwarding.py, which tested launch_phase_d.sh and
phase_d_chain.sh until covsyn-pipeline replaced both (Task 4). data_synthesis.sh stays as a way
to generate the two datasets by hand, and B86's rule for an unusable worker count stands.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pipeline_runner
import pytest

SYNTHESIS = Path(__file__).resolve().parent.parent / "scripts" / "data_synthesis.sh"
SCRIPT = SYNTHESIS  # read by the checklist test below for the source tree
SYNTHESIS_FILES = {"_files": {"run/firefly_best.txt": "1 2 3\n"}}
MODES = list(pipeline_runner.MODES)


def _record_dir(tmp_path: Path) -> Path:
    """Where the stand-ins of a scratch run record their calls."""
    return tmp_path / "record"


def _read(path: Path) -> str:
    """A file the stand-ins or the chain wrote, as UTF-8 with no newline translation."""
    with path.open(encoding="utf-8", newline="") as file:
        return file.read()


def _records(path: Path, context: str = "") -> list[list[str]]:
    """The calls a stand-in recorded, in order: fields ended by 0x1F, calls by 0x1E.

    ``context``, such as a run's output and log, is added to a failure's message.
    """
    assert path.exists(), f"no call was recorded in {path.name}\n{context}"
    text = _read(path)
    assert text.endswith("\x1e"), f"the last record in {path.name} is cut off\n{context}"
    return [call.split("\x1f")[:-1] for call in text.split("\x1e")[:-1]]


def _uv_runs(record: Path, context: str = "") -> list[list[str]]:
    """The arguments of every uv run recorded, in order, leaving out the synthesis stand-in's."""
    return [call for call in _records(record / "runs", context) if call[:1] != ["SYNTHESIS"]]


def _synthesis_runs(record: Path, context: str = "") -> list[tuple[str, str]]:
    """The (--mode, --parameter_path) of every uv run the stand-in recorded, in order."""
    runs = []
    for words in _uv_runs(record, context):
        assert "--mode" in words and "--parameter_path" in words, words
        runs.append((words[words.index("--mode") + 1], words[words.index("--parameter_path") + 1]))
    return runs


def _stand_ins(tmp_path: Path) -> tuple[Path, Path]:
    """A bin directory with stand-ins for uv and sleep, and the directory they record in."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = _record_dir(tmp_path)
    record.mkdir()
    (bin_dir / "uv").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "${{UV_PROJECT_ENVIRONMENT:-}}" > "{record}/uv_$1_environment"\n'
        f'echo "uv $*" >> "{record}/calls"\n'
        '[ "$1" = sync ] && exit "${UV_SYNC_STATUS:-0}"\n'
        '[ "$1" = run ] || exit 0\n'
        # One record per run: arguments ended by \x1f, the run by \x1e, so any argument,
        # newlines included, comes back as it was.
        f'printf "%s\\x1f" "$@" >> "{record}/runs"; printf "\\x1e" >> "{record}/runs"\n'
        'exit "${UV_STATUS:-0}"\n'
    )
    (bin_dir / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    for tool in ("uv", "sleep"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir, record


def _run(tmp_path: Path, script: Path, env: dict[str, str]) -> tuple[Path, Path, int, str]:
    """Run a copy of a script from a scratch repository root with the stand-ins on PATH."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(script, root / "scripts" / script.name)
    args = env.pop("_args", [])
    for name, text in env.pop("_files", {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    bin_dir, record = _stand_ins(tmp_path)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **env}
    log = root / "synthesis.log"  # never written; shown if present

    def show(how: str, output: str) -> None:
        # Printed for pytest, which shows a failing test's captured output: whichever check
        # fails, and however (an assertion, an error or a timeout), the output and the chain
        # log are there. Undecodable bytes are replaced, so showing them never fails a test.
        print(f"--- {script.name} {how}; its output:\n{output}")
        if log.exists():
            print(f"--- phaseD_chain.log:\n{log.read_bytes().decode('utf-8', 'replace')}")

    try:
        result = subprocess.run(
            ["bash", str(root / "scripts" / script.name), *args],
            env=env,
            cwd=root,
            capture_output=True,
            encoding="utf-8",  # as the timeout path decodes, and never failing on a byte
            errors="replace",
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as timeout:
        # A timeout's captured output is bytes even though the run decodes its output, so it
        # is decoded here the same way; without this, joining it would fail.
        parts = [timeout.stdout or b"", timeout.stderr or b""]
        show("timed out", "".join(p.decode("utf-8", "replace") for p in parts))
        raise
    output = result.stdout + result.stderr
    show(f"exited {result.returncode}", output)
    return root, record, result.returncode, output


def test_the_synthesis_alone_needs_the_run_directory(tmp_path):
    """Task 3: run on its own without PARAMETER_PATH, data_synthesis.sh stops and says why.

    There is no legacy copy to fall back on, and a time-stamped run name cannot be guessed.
    """
    root, record, status, output = _run(tmp_path, SYNTHESIS, dict(SYNTHESIS_FILES))
    assert status != 0 and "PARAMETER_PATH is not set" in output, output
    assert not (record / "runs").exists(), output  # nothing ran


def test_the_synthesis_uses_the_parameter_path_it_is_given(tmp_path):
    """With PARAMETER_PATH set, both modes run on that directory."""
    root, record, status, output = _run(
        tmp_path, SYNTHESIS, {"PARAMETER_PATH": "run", **SYNTHESIS_FILES}
    )
    assert status == 0, output
    assert _synthesis_runs(record, output) == [(mode, "run") for mode in MODES], output


def test_the_synthesis_alone_stops_on_an_environment_that_differs(tmp_path):
    """Run on its own, data_synthesis.sh stops before any run when the check fails.

    Its message names the environment in use.
    """
    root, record, status, output = _run(
        tmp_path,
        SYNTHESIS,
        {
            "PARAMETER_PATH": "run",
            "UV_SYNC_STATUS": "2",
            "UV_PROJECT_ENVIRONMENT": "/opt/env",
            **SYNTHESIS_FILES,
        },
    )
    assert status == 1
    assert "/opt/env differs from uv.lock" in output
    assert (record / "uv_sync_environment").read_text().strip() == "/opt/env"
    assert (record / "calls").read_text().splitlines() == ["uv sync --locked --check --inexact"]
    assert not (record / "runs").exists()


@pytest.mark.parametrize(
    "args", [["--environment_checked"], ["run10"], ["--environment-checked", "extra"]]
)
def test_the_synthesis_rejects_an_argument_it_does_not_know(tmp_path, args):
    """A misspelled option or a stray argument stops the script before 2000 simulations."""
    root, record, status, output = _run(
        tmp_path, SYNTHESIS, {"PARAMETER_PATH": "run", "_args": args, **SYNTHESIS_FILES}
    )
    assert status == 2
    assert "Usage" in output
    assert not (record / "calls").exists()


def test_the_synthesis_called_by_the_chain_skips_the_check(tmp_path):
    """With --environment-checked, a lock that changed mid-run does not stop the synthesis.

    The chain checked the environment at its start, and the synthesis runs in it.
    """
    root, record, status, output = _run(
        tmp_path,
        SYNTHESIS,
        {
            "PARAMETER_PATH": "run",
            "UV_SYNC_STATUS": "2",
            "UV_PROJECT_ENVIRONMENT": "/opt/env",
            "_args": ["--environment-checked"],
            **SYNTHESIS_FILES,
        },
    )
    assert status == 0, output
    assert not (record / "uv_sync_environment").exists()
    assert (record / "uv_run_environment").read_text().strip() == "/opt/env"
    assert _synthesis_runs(record, output) == [(mode, "run") for mode in MODES], output


def test_the_checklist_uses_the_shared_helper_for_its_worker_count():
    """Both the forwarding and the parsing have to agree on one definition (Task 2)."""
    checklist = (
        SCRIPT.parent.parent / "src" / "covsyn" / "validation" / "verify_phase_d.py"
    ).read_text(encoding="utf-8")
    assert "hardware.resolve_workers(" in checklist
    assert "CHECKLIST_GB_PER_WORKER" in checklist


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("8", 8),  # E98's setting for this shared machine, about 39 GB
        ("1", 1),
        ("12", 12),  # above the cores, which only set a default
        (" 8 ", 8),  # a forwarded value can carry whitespace
    ],
)
def test_an_explicit_worker_count_is_taken_as_given(value, expected):
    """An explicit setting overrides the default, beyond the cores included."""
    from covsyn.runtime.hardware import resolve_workers

    env = {"COVSYN_WORKERS": value}
    assert resolve_workers(None, env, 4.9, cores=4, available_gb=100.0) == expected


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_unset_or_empty_value_falls_back_to_the_default(value):
    """The shell forwards '' when the variable is unset, which must not raise or mean zero."""
    from covsyn.runtime.hardware import resolve_workers

    env = {} if value is None else {"COVSYN_WORKERS": value}
    assert resolve_workers(None, env, 4.9, cores=48, available_gb=1000.0) == 48  # the cores
    assert resolve_workers(None, env, 4.9, cores=48, available_gb=50.0) == 8  # the memory


@pytest.mark.parametrize("value", ["0", "-1", "nonsense", "3.5"])
def test_a_value_that_is_not_a_positive_count_falls_back(value):
    """A typo must not silently run the checklist on one worker or crash it."""
    from covsyn.runtime.hardware import resolve_workers

    env = {"COVSYN_WORKERS": value}
    assert resolve_workers(None, env, 4.9, cores=32, available_gb=1000.0) == 32
