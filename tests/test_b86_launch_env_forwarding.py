# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B86: launch_phase_d.sh must forward every variable the chain reads into its tmux session.

tmux does not give a new session the launching shell's environment, as the script's own comment
says, so each variable is passed to the session with tmux -e. VERIFY_WORKERS was not, which
means setting it before the launch was silently dropped and the checklist kept its default of
min(cpu_count, 32). That default is what E98 measured: 32 bootstrap workers need about 157 GB,
and on this shared machine another user held 294 GB of the 512, so run 21's checklist swapped
and each candidate took 19-28 minutes against the 4.3 measured in isolation.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "launch_phase_d.sh"
CHAIN = SCRIPT.parent / "phase_d_chain.sh"
SOURCE = SCRIPT.parent.parent / "src" / "covsyn"
# Set by the chain or its modules themselves, not by the user: OPTIMIZER_RUN after
# revalidation, AGE_RR_FILE by select_by_checklist for each candidate's checklist.
CHAIN_INTERNAL = {"OPTIMIZER_RUN", "AGE_RR_FILE"}
SHELL_SPECIAL = {"BASH_SOURCE"}
NAME = r"[A-Z][A-Z0-9_]*"
ENVIRON = {"environ"}
MAPPING = {"env"}


def _name(node: ast.AST) -> str:
    return getattr(node, "attr", getattr(node, "id", ""))


def _environment_reads(path: Path) -> set[str]:
    """Names a module reads from the environment, found in its code, not its comments.

    Covers ``os.environ`` or ``environ`` subscripts, ``.get`` and ``.setdefault``, ``getenv``
    and ``in environ``, and the same on a mapping passed in as ``env``, as
    ``worker_count(os.environ)`` reads VERIFY_WORKERS.
    """
    sources = ENVIRON | MAPPING
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        key = None
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"get", "setdefault"}:
                if _name(func.value) in sources:
                    key = node.args[0]
            elif _name(func) == "getenv":
                key = node.args[0]
        elif isinstance(node, ast.Subscript) and _name(node.value) in sources:
            key = node.slice
        elif isinstance(node, ast.Compare) and any(isinstance(o, ast.In) for o in node.ops):
            if any(_name(c) in sources for c in node.comparators):
                key = node.left
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and \
                re.fullmatch(NAME, key.value):
            names.add(key.value)
    return names


def _import_closure(module: str) -> set[Path]:
    """The covsyn source files a module imports, directly or through other covsyn modules.

    Relative imports resolve against the importing module, and each package's ``__init__.py``
    counts, as it runs on every import of the package.
    """
    seen, todo, files = set(), [module], set()
    while todo:
        name = todo.pop()
        if name in seen or not name.startswith("covsyn"):
            continue
        seen.add(name)
        parts = name.split(".")
        for depth in range(1, len(parts)):
            init = SOURCE.parent.joinpath(*parts[:depth], "__init__.py")
            if init.is_file():
                files.add(init)
        path = SOURCE.parent / (name.replace(".", "/") + ".py")
        package = SOURCE.parent.joinpath(*parts, "__init__.py")
        path = path if path.is_file() else package
        if not path.is_file():
            continue
        files.add(path)
        base = parts if path.name == "__init__.py" else parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                prefix = base[: len(base) - node.level + 1] if node.level else []
                target = ".".join([*prefix, *(node.module or "").split(".")]).strip(".")
                todo.append(target)
                todo.extend(f"{target}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                todo.extend(a.name for a in node.names)
    return files


ENVIRONMENT_NAMES = {"environ", "environb", "getenv", "getenvb"}


def _mentions_environment(path: Path) -> bool:
    """Whether a module touches the environment at all.

    A copy, an alias or a mapping handed to a reader such as ``worker_count`` reads the
    environment as surely as a lookup does, so the firefly session's modules may not touch it:
    no ``os.environ``, ``environb``, ``getenv`` or ``getenvb``, under any import name, and no
    ``getattr`` on ``os``.
    """
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.Name, ast.Attribute)) and _name(node) in ENVIRONMENT_NAMES:
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "os" and \
                any(a.name in ENVIRONMENT_NAMES or a.name == "*" for a in node.names):
            return True
        if isinstance(node, ast.Call) and _name(node.func) == "getattr" and node.args and \
                _name(node.args[0]) == "os":
            return True
    return False


# Every covsyn module the firefly session runs; the chain runs the rest, and these too.
FIREFLY_FILES = _import_closure("covsyn.calibration.firefly_optimizer")


def _shell_code(path: Path) -> str:
    """A shell script without its comment lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _assigned(code: str) -> set[str]:
    """Names a shell script assigns with ``NAME=`` at the start of a line.

    The chain gives each variable it reads from the environment a default, ``${NAME:-...}``,
    and assigns its own variables with ``NAME=``; test_the_chain_follows_the_convention fails
    loudly on any other form, rather than this scan guessing at it.
    """
    return set(re.findall(rf"^\s*({NAME})=", code, re.MULTILINE))


def _chain_variables() -> list[str]:
    """Every variable the chain or a module it runs reads from the environment.

    The chain reads one when it gives it a default or uses it without assigning it; PYTHON is
    gone, as the chain runs every module through uv run (Task 1).
    """
    chain = _shell_code(CHAIN)
    referenced = set(re.findall(rf"\$\{{?({NAME})", chain))
    assigned = _assigned(chain)
    defaulted = set(re.findall(rf"\$\{{({NAME}):-", chain))
    modules = {name for path in SOURCE.rglob("*.py") for name in _environment_reads(path)}
    return sorted(((referenced - assigned) | defaulted | modules) - CHAIN_INTERNAL - SHELL_SPECIAL)


CHAIN_VARIABLES = _chain_variables()


@pytest.fixture
def chain_env() -> list[str]:
    """The names launch_phase_d.sh passes into the chain session."""
    text = _shell_code(SCRIPT)
    match = re.search(r"^CHAIN_ENV=\((.*?)\)", text, re.DOTALL | re.MULTILINE)
    assert match, "launch_phase_d.sh no longer lists the chain's variables in CHAIN_ENV"
    return match.group(1).split()


@pytest.mark.parametrize("variable", CHAIN_VARIABLES)
def test_the_chain_session_is_given_every_variable_it_reads(chain_env, variable):
    """A variable missing here is dropped without any error at launch."""
    assert variable in chain_env


def test_the_firefly_modules_do_not_touch_the_environment():
    """No module the firefly session imports reads the environment.

    The session gets WARM_START alone, which the shell passes to the optimiser as an argument.
    """
    assert len(FIREFLY_FILES) > 2
    for path in FIREFLY_FILES:
        assert not _mentions_environment(path), path.name


@pytest.mark.parametrize(("source", "touches"), [
    ("import os\nworker_count(dict(os.environ))", True),
    ("import os\nenv = os.environ\nworker_count(env)", True),
    ("from os import getenv\ngetenv('A')", True),
    ("from os import environ as e\nworker_count(e)", True),
    ("from os import getenv as g\ng('A')", True),
    ("import os\nos.environb", True),
    ("import os\ngetattr(os, 'environ')", True),
    ("def worker_count(env):\n    return env.get('A')", False),
    ("# os.environ\nprint('os.environ')", False),
])
def test_touching_the_environment_is_found(tmp_path, source, touches):
    """Copies and aliases of the environment count, a mapping parameter or text does not."""
    path = tmp_path / "module.py"
    path.write_text(source)
    assert _mentions_environment(path) == touches


def test_the_import_closure_follows_relative_imports_and_packages(tmp_path, monkeypatch):
    """A relative import and a package ``__init__.py`` are part of what a module runs."""
    package = tmp_path / "covsyn" / "calibration"
    package.mkdir(parents=True)
    (tmp_path / "covsyn" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("import os\nos.environ['B']")
    (package / "main.py").write_text("from .helper import f\n")
    (package / "helper.py").write_text("def f():\n    pass\n")
    monkeypatch.setattr(sys.modules[__name__], "SOURCE", tmp_path / "covsyn")
    closure = _import_closure("covsyn.calibration.main")
    files = {p.relative_to(tmp_path).as_posix() for p in closure}
    assert files == {"covsyn/__init__.py", "covsyn/calibration/__init__.py",
                     "covsyn/calibration/main.py", "covsyn/calibration/helper.py"}
    assert any(_mentions_environment(p) for p in closure)


def test_the_chain_follows_the_convention():
    """Every variable the chain uses is assigned with NAME= or read with a ${NAME:-} default.

    A variable used any other way, say by for, read or local, would be taken for user input
    and its forwarding asked for; this fails instead, so its form can be checked by hand.
    """
    chain = _shell_code(CHAIN)
    referenced = set(re.findall(rf"\$\{{?({NAME})", chain)) - SHELL_SPECIAL
    unaccounted = referenced - _assigned(chain) - set(re.findall(rf"\$\{{({NAME}):-", chain))
    assert not unaccounted


@pytest.mark.parametrize(("code", "names"), [
    ("A=1", {"A"}), ("    A=1", {"A"}), ("echo A=1", set()), ("for A in 1; do :; done", set()),
])
def test_the_chain_scan_sees_line_start_assignments(code, names):
    """Only NAME= at the start of a line counts as an assignment."""
    assert _assigned(code) == names


@pytest.mark.parametrize(("source", "expected"), [
    ("import os\nos.environ['A']", {"A"}),
    ("import os\nos.environ.get('A')", {"A"}),
    ("import os\nos.environ.setdefault('A', '1')", {"A"}),
    ("import os\nos.getenv('A')", {"A"}),
    ("from os import getenv\ngetenv('A')", {"A"}),
    ("from os import environ\n'A' in environ", {"A"}),
    ("from os import environ\nenviron['A']", {"A"}),
    ("def f(env):\n    return env.get('A')", {"A"}),
    ("# os.environ['A']\nprint('os.getenv(B)')", set()),
    ("d = {}\nd.get('A')", set()),
])
def test_environment_reads_are_found_in_code_only(tmp_path, source, expected):
    """The scan finds every spelling of an environment read and nothing in comments or text."""
    path = tmp_path / "module.py"
    path.write_text(source)
    assert _environment_reads(path) == expected


def _stand_ins(tmp_path: Path) -> tuple[Path, Path]:
    """A bin directory with stand-ins for tmux, uv and sleep, and the directory they record in.

    new-session applies -c and -e, runs the firefly command and keeps the chain session's
    environment; has-session reports the firefly session live, once started, only when
    FIREFLY_LIVE is set.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "record"
    record.mkdir()
    (bin_dir / "uv").write_text(
        "#!/usr/bin/env bash\n"
        f'[ "$1" = run ] || exit 0\n'
        f'printf "%s\\n" "$@" > "{record}/uv_run_args"\n'
        'exit "${UV_STATUS:-0}"\n')
    (bin_dir / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "tmux").write_text(f"""#!/usr/bin/env bash
case "$1" in
  has-session) [ -n "${{FIREFLY_LIVE:-}}" ] && [ -e "{record}/firefly_started" ]; exit ;;
  new-session) shift ;;
  *) exit 0 ;;
esac
envs=(); dir=.; name=
while [ $# -gt 0 ]; do
  case "$1" in
    -d) shift ;;
    -s) name=$2; shift 2 ;;
    -c) dir=$2; shift 2 ;;
    -e) envs+=("$2"); shift 2 ;;
    *) break ;;
  esac
done
if [ "$name" = firefly ]; then
  touch "{record}/firefly_started"
  [ -n "${{KILL_FIREFLY:-}}" ] || (cd "$dir" && env "${{envs[@]}}" "$@")
else
  env -i "${{envs[@]}}" > "{record}/chain_env"
fi
""")
    for tool in ("uv", "sleep", "tmux"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir, record


def _run(tmp_path: Path, script: Path, env: dict[str, str]) -> tuple[Path, Path, int, str]:
    """Run a copy of a script from a scratch repository root with the stand-ins on PATH."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(script, root / "scripts" / script.name)
    for name, text in env.pop("_files", {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    bin_dir, record = _stand_ins(tmp_path)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **env}
    result = subprocess.run(["bash", str(root / "scripts" / script.name)], env=env, cwd=root,
                            capture_output=True, text=True, timeout=60, check=False)
    return root, record, result.returncode, result.stdout + result.stderr


def test_launch_forwards_the_values_unchanged(tmp_path):
    """Run the launcher with stand-ins for tmux and uv and record what each session gets.

    A value with a quote arrives intact, an unset one arrives empty, the space-separated
    WARM_START reaches the optimiser as separate files whatever the user's login shell, and
    the firefly session leaves the optimiser's exit status where the chain looks for it.
    """
    root, record, status, output = _run(tmp_path, SCRIPT, {
        "WARM_START": "a/firefly_best.txt b/firefly_best.txt",
        "FIREFLY_RUN": "run's dir",
        "CPU_CORES": "8",
        "_files": {"firefly_phaseD.exit": "1\n"},
    })
    assert status == 0, output
    args = (record / "uv_run_args").read_text().split("\n")
    start = args.index("--warm_start")
    assert args[start + 1:start + 3] == ["a/firefly_best.txt", "b/firefly_best.txt"]
    chain = dict(line.split("=", 1) for line in (record / "chain_env").read_text().splitlines())
    assert (root / chain["FIREFLY_STATUS"]).read_text().strip() == "0"
    assert chain["FIREFLY_RUN"] == "run's dir"
    assert chain["CPU_CORES"] == "8"
    assert chain["SELECT_TOP"] == ""
    assert set(chain) == set(CHAIN_VARIABLES)


def test_launch_uses_a_status_file_the_user_names(tmp_path):
    """A FIREFLY_STATUS set before the launch is the file the session writes and the chain reads.

    The path is relative to the repository root.
    """
    root, record, status, output = _run(tmp_path, SCRIPT, {
        "FIREFLY_STATUS": "runs/x.exit", "_files": {"runs/.keep": ""},
    })
    assert status == 0, output
    chain = dict(line.split("=", 1) for line in (record / "chain_env").read_text().splitlines())
    assert chain["FIREFLY_STATUS"] == "runs/x.exit"
    assert (root / "runs" / "x.exit").read_text().strip() == "0"


def test_session_checks_match_the_name_exactly():
    """Every session check matches the session name exactly.

    tmux matches -t firefly as a prefix when no session has that name, so firefly_old would
    count; "=" makes the match exact.
    """
    for script in (SCRIPT, CHAIN):
        targets = re.findall(r"has-session -t (\S+)", _shell_code(script))
        assert targets, script.name
        assert all(t.startswith("=") for t in targets), (script.name, targets)


def test_launch_stops_when_the_optimiser_fails_at_startup(tmp_path):
    """An optimiser that exits non-zero at once fails the launch, and no chain starts."""
    root, record, status, output = _run(tmp_path, SCRIPT, {"UV_STATUS": "3"})
    assert status == 1
    assert "ended at startup with status 3" in output
    assert not (record / "chain_env").exists()


def test_a_firefly_session_killed_at_startup_fails_the_launch(tmp_path):
    """A session killed before the optimiser exits fails the launch.

    It leaves "running" and no session, which the launcher reports instead of starting the
    chain.
    """
    root, record, status, output = _run(tmp_path, SCRIPT, {"KILL_FIREFLY": "1"})
    assert status == 1
    assert "status killed" in output
    assert (root / "firefly_phaseD.exit").read_text().strip() == "running"
    assert not (record / "chain_env").exists()


def test_a_running_firefly_session_starts_the_chain(tmp_path):
    """While the optimiser still runs at the check, the launcher starts the chain."""
    root, record, status, output = _run(tmp_path, SCRIPT, {"KILL_FIREFLY": "1",
                                                          "FIREFLY_LIVE": "1"})
    assert status == 0, output
    assert (root / "firefly_phaseD.exit").read_text().strip() == "running"
    assert (record / "chain_env").exists()


@pytest.mark.parametrize(("content", "launched", "stops"), [
    ("1", True, True),
    ("137", True, True),
    ("running", True, True),    # killed before the optimiser exited
    (None, True, True),         # status file gone
    ("0", True, False),
    ("1", False, False),        # a hand-run chain ignores a stale status
    (None, False, False),
])
def test_the_chain_stops_when_the_firefly_failed(tmp_path, content, launched, stops):
    """A failed or killed optimiser stops a launched chain before it reads FIREFLY_RUN.

    A successful one, or a chain started by hand, goes on to its first module, the
    revalidation (here a stand-in uv that fails, so the chain stops there).
    """
    files = {"run/firefly_result.txt": "1 2 3\n"}
    if content is not None:
        files["firefly_phaseD.exit"] = content + "\n"
    env = {"FIREFLY_RUN": "run", "UV_STATUS": "9", "_files": files}
    if launched:
        env["FIREFLY_STATUS"] = "firefly_phaseD.exit"
    root, record, status, output = _run(tmp_path, CHAIN, env)
    log = (root / "phaseD_chain.log").read_text()
    assert status != 0, output
    assert ("the firefly session ended with status" in log) == stops, log
    assert (record / "uv_run_args").exists() != stops


def test_the_chain_reads_the_variables_review_found_dropped():
    """The variables the 2026-11-10 code reviews found dropped are all found by the scan.

    Seven are read by the chain itself; VERIFY_WORKERS (B86) by the checklist module.
    """
    dropped = {"CPU_CORES", "FIREFLY_RUN", "REVALIDATE", "REVALIDATION_BLOCKS", "SELECT",
               "SELECT_TOP", "VERIFY_WORKERS", "PREVIOUS_CHECKS"}
    assert dropped <= set(CHAIN_VARIABLES)


@pytest.mark.parametrize("variable", CHAIN_VARIABLES)
def test_the_variable_is_documented_in_the_usage_header(variable):
    """The header is where a run is configured from, so an undocumented variable is unusable."""
    header = SCRIPT.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert variable in header


def test_the_checklist_uses_the_shared_helper_for_its_worker_count():
    """Both the forwarding and the parsing have to agree on one definition."""
    checklist = (SCRIPT.parent.parent / "src" / "covsyn" / "validation"
                 / "verify_phase_d.py").read_text(encoding="utf-8")
    assert "worker_count(" in checklist


@pytest.mark.parametrize(("value", "expected"), [
    ("8", 8),        # E98's setting for this shared machine, about 39 GB
    ("1", 1),
    ("64", 64),      # above the cap, because the cap is only a default
    (" 8 ", 8),      # a forwarded value can carry whitespace
])
def test_an_explicit_worker_count_is_taken_as_given(value, expected):
    """An explicit setting overrides the default, above the cap included."""
    from covsyn.validation.acceptance import worker_count
    assert worker_count({"VERIFY_WORKERS": value}, cpus=32) == expected


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_unset_or_empty_value_falls_back_to_the_default(value):
    """The shell forwards '' when the variable is unset, which must not raise or mean zero."""
    from covsyn.validation.acceptance import worker_count
    env = {} if value is None else {"VERIFY_WORKERS": value}
    assert worker_count(env, cpus=48) == 32      # the cap
    assert worker_count(env, cpus=4) == 4        # fewer cores than the cap


@pytest.mark.parametrize("value", ["0", "-1", "nonsense", "3.5"])
def test_a_value_that_is_not_a_positive_count_falls_back(value):
    """A typo must not silently run the checklist on one worker or crash it."""
    from covsyn.validation.acceptance import worker_count
    assert worker_count({"VERIFY_WORKERS": value}, cpus=32) == 32
