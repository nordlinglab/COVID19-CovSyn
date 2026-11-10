# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 1 of the optimisation rework plan: ``git clone`` and ``uv sync`` install CovSyn.

The package installs in editable mode with every dependency pinned in ``pyproject.toml`` and
``uv.lock``, so no script, test, or user sets ``PYTHONPATH``.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The pins of requirements.txt at f8b7ccd, carried over unchanged, plus psutil (Task 5).
RUNTIME_PINS = (
    "bidict==0.23.1",
    "lifelines==0.30.0",
    "matplotlib==3.10.3",
    "networkx==3.4.2",
    "numpy==2.2.6",
    "pandas==2.2.3",
    "scikit-learn==1.6.1",
    "scipy==1.15.3",
    "seaborn==0.13.2",
    "tqdm==4.67.1",
    "openpyxl==3.1.5",
    "xlrd>=2.0.1",
)

# The entry points for the optimiser and for every chain module that has a main function.
# The chain modules that run at import time keep running as ``uv run python -m``.
ENTRY_POINTS = {
    "covsyn-optimise": "covsyn.calibration.firefly_optimizer:main",
    "covsyn-revalidation": "covsyn.calibration.revalidation:main",
    "covsyn-select-by-checklist": "covsyn.calibration.select_by_checklist:main",
    "covsyn-data-synthesis": "covsyn.model.data_synthesis_main:main",
    "covsyn-measure-age-rr": "covsyn.validation.measure_age_rr:main",
    "covsyn-verify-phase-d": "covsyn.validation.verify_phase_d:main",
    "covsyn-show-bounds": "covsyn.validation.show_bounds:cli",
    "covsyn-measure-days": "covsyn.validation.measure_days:main",
    "covsyn-compare-phase-d-runs": "covsyn.validation.compare_phase_d_runs:main",
    "covsyn-plot-todolist923": "covsyn.figures.plot_todolist923:main",
    "covsyn-check-constraints": "covsyn.validation.check_constraints:main",
    "covsyn-reproduce-wu2025": "covsyn.figures.reproduce_wu2025:main",
}

INSTALL_COMMANDS = (
    "git clone git@github.com:nordlinglab/COVID19-CovSyn.git",
    "cd COVID19-CovSyn",
    "uv sync",
    "uv run pytest",
)


def _normalise(requirement: str) -> str:
    return requirement.replace(" ", "").replace("_", "-").lower()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    """The parsed ``pyproject.toml``."""
    return tomllib.loads(PYPROJECT.read_text())


def test_project_table_names_the_package_and_python(pyproject):
    """The ``[project]`` table names ``covsyn`` and Python 3.13 only."""
    project = pyproject["project"]
    assert project["name"] == "covsyn"
    assert project["requires-python"] == ">=3.13,<3.14"


def test_runtime_dependencies_keep_the_pins(pyproject):
    """Every pin of ``requirements.txt`` carries over unchanged, and psutil is added."""
    declared = {_normalise(d) for d in pyproject["project"]["dependencies"]}
    for pin in RUNTIME_PINS:
        assert _normalise(pin) in declared, pin
    assert any(d.startswith("psutil") for d in declared)


def test_build_system_and_dev_group(pyproject):
    """A build backend installs ``src/covsyn``; the dev group holds the test and lint tools."""
    assert pyproject["build-system"]["build-backend"]
    dev = {_normalise(d).split("=")[0].split(">")[0] for d in pyproject["dependency-groups"]["dev"]}
    assert {"pytest", "ruff", "hypothesis"} <= dev


def test_uv_files_replace_requirements_txt():
    """``uv.lock`` and ``.python-version`` are committed and ``requirements.txt`` is gone."""
    assert not (REPO_ROOT / "requirements.txt").exists()
    assert (REPO_ROOT / "uv.lock").is_file()
    assert (REPO_ROOT / ".python-version").read_text().strip() == "3.13"


def test_running_python_is_3_13():
    """The environment ``uv sync`` built runs Python 3.13."""
    assert sys.version_info[:2] == (3, 13)


def test_package_imports_without_pythonpath(tmp_path):
    """``covsyn`` imports from outside the repository with ``PYTHONPATH`` unset, from ``src/``."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import covsyn; print(covsyn.__file__)"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve().is_relative_to(REPO_ROOT / "src" / "covsyn")


def test_entry_points_are_declared(pyproject):
    """``[project.scripts]`` declares every entry point with its target."""
    scripts = pyproject["project"]["scripts"]
    for name, target in ENTRY_POINTS.items():
        assert scripts.get(name) == target, name


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_entry_point_is_installed_and_callable(name, monkeypatch):
    """Each installed entry point loads a function that takes no required argument."""
    installed = {ep.name: ep for ep in entry_points(group="console_scripts")}
    assert name in installed
    # verify_phase_d reads its arguments at import, as the console script imports it with the
    # user's arguments; pytest's own arguments must not reach it.
    monkeypatch.setattr(sys, "argv", [name])
    function = installed[name].load()
    required = [
        p
        for p in inspect.signature(function).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    assert not required, required


def _script_files() -> list[Path]:
    files = [p for p in (REPO_ROOT / "scripts").rglob("*") if p.suffix in {".sh", ".py"}]
    return [*files, REPO_ROOT / "tests" / "pipeline_runner.py"]


@pytest.mark.parametrize("path", _script_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_script_sets_pythonpath_or_picks_an_interpreter(path):
    """Scripts run through ``uv run`` instead of ``PYTHONPATH`` and a ``PYTHON`` variable."""
    text = path.read_text()
    assert "PYTHONPATH" not in text
    assert "PYTHON=" not in text
    assert '"$PYTHON"' not in text


def test_readme_gives_the_four_install_commands():
    """The README's installation section is the four commands of the plan."""
    readme = (REPO_ROOT / "README.md").read_text()
    for command in INSTALL_COMMANDS:
        assert command in readme, command
    assert "PYTHONPATH" not in readme


def test_ci_workflow_runs_the_fast_tests_on_both_systems():
    """CI installs from the lock file and runs the fast tests on Linux and macOS."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text()
    for needle in (
        "ubuntu-latest",
        "macos-latest",
        "uv sync --locked",
        'uv run pytest -m "not slow"',
    ):
        assert needle in workflow, needle


PYTHON_SCRIPT = re.compile(r"\bpython3? [\w./-]+\.py\b")
# uv run options that take a value; any other option stands alone.
UV_VALUE_OPTIONS = r"--with|--python|-p|--extra|--group|--directory|--project|--env-file"
THROUGH_UV = re.compile(
    rf"\buv run(?:\s+(?:(?:{UV_VALUE_OPTIONS})(?:\s+|=)[^\s|;&]+|--?[\w-]+(?:=[^\s|;&]+)?))*\s+$"
)


@pytest.mark.parametrize("path", _script_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_usage_lines_run_scripts_through_uv(path):
    """Usage lines run a script from the repository root through ``uv run``.

    A bare ``python script.py`` finds no ``covsyn`` without ``PYTHONPATH``.
    """
    bare = [
        line.strip()
        for line in path.read_text().splitlines()
        for match in PYTHON_SCRIPT.finditer(line)
        if not THROUGH_UV.search(line[: match.start()])
    ]
    assert not bare, bare


@pytest.mark.parametrize(
    ("line", "through_uv"),
    [
        ("uv run python scripts/x.py", True),
        ("uv run --no-sync python scripts/x.py", True),
        ("uv run --with pandas python scripts/x.py", True),
        ("uv run --python 3.13 -q python scripts/x.py", True),
        ("python scripts/x.py", False),
        ("Requires uv run setup first, then: python scripts/x.py", False),
        ("see uv run --frozen then python scripts/x.py", False),
        ("uv run --env-file=.env python scripts/x.py", True),
    ],
)
def test_usage_line_check_accepts_only_uv_options(line, through_uv):
    """Only option-shaped words may stand between ``uv run`` and ``python``."""
    match = PYTHON_SCRIPT.search(line)
    assert bool(THROUGH_UV.search(line[: match.start()])) == through_uv


UV_RUN = r"""(?:\buv|['"]?\$\{?UV\}?['"]?) run\b.*"""


def test_chain_scripts_do_not_resync_the_environment():
    """The chain's ``uv run`` calls use the environment as installed.

    The optimiser runs for days in the same ``.venv``; a sync in the middle of a run would
    reinstall packages under it and could change the lock file.
    """
    for name in ("phase_d_chain.sh", "data_synthesis.sh", "launch_phase_d.sh"):
        text = (REPO_ROOT / "scripts" / name).read_text()
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        commands = [c for line in code for c in re.findall(UV_RUN, line)]
        assert commands, name
        for command in commands:
            assert "--no-sync" in command, (name, command)


def test_launch_checks_uv_and_the_environment_before_the_sessions():
    """The launcher checks uv and the environment once, before the sessions start.

    It stops with a message when uv is missing, and checks the environment against uv.lock,
    as the sessions run with --no-sync.

    tmux gives a new session the PATH of the shell that runs new-session, so the uv found
    here is the uv both sessions find.
    """
    text = (REPO_ROOT / "scripts" / "launch_phase_d.sh").read_text()
    first_session = text.index("tmux new-session")
    assert "command -v uv >/dev/null || {" in text[:first_session]
    assert "uv sync --locked" in text[:first_session]


def test_readme_explains_the_slow_tests_on_linux():
    """The README warns that the slow regression tests reproduce macOS fixtures (E103)."""
    readme = (REPO_ROOT / "README.md").read_text()
    assert "E103" in readme
    assert "[Tests](#tests)" in readme
