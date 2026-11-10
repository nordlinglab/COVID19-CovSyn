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
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "launch_phase_d.sh"
CHAIN = SCRIPT.parent / "phase_d_chain.sh"
SYNTHESIS = SCRIPT.parent / "data_synthesis.sh"
SOURCE = SCRIPT.parent.parent / "src" / "covsyn"
# The scripts the chain runs, each with the variables the chain sets on its command line.
CHAIN_SCRIPTS = {SYNTHESIS: {"PARAMETER_PATH"}}
# Set by the chain or its modules themselves, not by the user: OPTIMIZER_RUN after
# revalidation, AGE_RR_FILE by select_by_checklist for each candidate's checklist.
CHAIN_INTERNAL = {"OPTIMIZER_RUN", "AGE_RR_FILE"}
SHELL_SPECIAL = {"BASH_SOURCE"}
NAME = r"[A-Z][A-Z0-9_]*"
# One definition of an environment access for both scans: the mappings and the getters of os,
# and a mapping passed in as ``env``, as ``worker_count(os.environ)`` reads VERIFY_WORKERS.
ENVIRONMENT_MAPPINGS = {"environ", "environb"}
ENVIRONMENT_GETTERS = {"getenv", "getenvb"}
ENVIRONMENT_NAMES = ENVIRONMENT_MAPPINGS | ENVIRONMENT_GETTERS
MAPPING_PARAMETER = "env"


def _name(node: ast.AST) -> str:
    return getattr(node, "attr", getattr(node, "id", ""))


def _environment_names(tree: ast.AST) -> dict[str, str]:
    """The names a module uses for the environment accesses of os.

    Imported aliases (``from os import environ as e``) and local ones (``e = os.environ``)
    included.
    """
    names = {name: name for name in ENVIRONMENT_NAMES}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in ENVIRONMENT_NAMES:
                    names[alias.asname or alias.name] = alias.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Attribute)):
            value = _name(node.value)
            value = names.get(value, value) if isinstance(node.value, ast.Name) else value
            if value in ENVIRONMENT_NAMES:
                names.update({t.id: value for t in node.targets if isinstance(t, ast.Name)})
    return names


def _environment_reads(path: Path) -> set[str]:
    """Names a module reads from the environment, found in its code, not its comments.

    Covers subscripts, ``.get``, ``.setdefault``, ``.pop`` and ``in`` on ``os.environ`` or on
    a mapping passed in as ``env``, and ``getenv`` with the key given by position or as
    ``key=``, under any name the module imports or assigns them to. A mapping parameter by
    another name than ``env``, or a key built at run time, is beyond a static scan.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _environment_names(tree)

    def kind(node: ast.AST) -> str:
        name = _name(node)
        if isinstance(node, ast.Name):
            name = aliases.get(name, name)
        if name in ENVIRONMENT_MAPPINGS or name == MAPPING_PARAMETER:
            return "mapping"
        return "getter" if name in ENVIRONMENT_GETTERS else ""

    names = set()
    for node in ast.walk(tree):
        key = None
        if isinstance(node, ast.Call):
            func = node.func
            first = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "key"), None)
            if isinstance(func, ast.Attribute) and func.attr in {"get", "setdefault", "pop"}:
                if kind(func.value) == "mapping":
                    key = first
            elif kind(func) == "getter":
                key = first
        elif isinstance(node, ast.Subscript) and kind(node.value) == "mapping":
            key = node.slice
        elif isinstance(node, ast.Compare) and any(isinstance(o, ast.In) for o in node.ops):
            if any(kind(c) == "mapping" for c in node.comparators):
                key = node.left
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and \
                re.fullmatch(NAME, key.value):
            names.add(key.value)
    return names


def _mentions_environment(path: Path) -> bool:
    """Whether a module touches the environment at all.

    A copy, an alias or a mapping handed to a reader such as ``worker_count`` reads the
    environment as surely as a lookup does, so the firefly session's modules may not touch it:
    no environment access of os under any name, and no such name as a string outside a
    docstring, which covers ``getattr(os, "environ")`` and ``vars(os)["environ"]``. A name
    built at run time (``"env" + "iron"``) is beyond a static scan.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _environment_names(tree)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
        if isinstance(node, ast.Attribute) and node.attr in ENVIRONMENT_NAMES:
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "os" and \
                any(a.name in ENVIRONMENT_NAMES or a.name == "*" for a in node.names):
            return True
        if isinstance(node, ast.Constant) and node.value in ENVIRONMENT_NAMES and \
                id(node) not in docstrings:
            return True
    return False


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


# Every covsyn module the firefly session runs; the chain runs the rest, and these too.
FIREFLY_FILES = _import_closure("covsyn.calibration.firefly_optimizer")


def _shell_lines(path: Path) -> list[str]:
    """A shell script's lines, comment lines dropped first and continuation lines then joined.

    bash does not continue a comment onto the next line, so the order matters.
    """
    lines, pending = [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not pending and line.lstrip().startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        lines.append(pending + line)
        pending = ""
    return [*lines, pending] if pending else lines


def _first_command(line: str) -> list[str]:
    """The words of a line's first command, command substitutions folded into one word."""
    while (folded := re.sub(r"\$\([^()]*\)", "X", line)) != line:
        line = folded
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        words = list(lexer)
    except ValueError:
        return []
    command = []
    for word in words:
        if word and set(word) <= set(";&|"):
            break
        command.append(word)
    return command


def _top_level_assignment(line: str) -> str | None:
    """The name a line assigns, when it assigns it unconditionally.

    Only NAME=value at the very start of a line, as the whole first command, counts: an
    indented one may sit in a branch that does not run, and NAME=value followed by a command
    sets NAME for that command only.
    """
    match = re.match(rf"({NAME})=", line)
    if match and len(_first_command(line)) == 1:
        return match.group(1)
    return None


def _shell_variables(lines: list[str]) -> tuple[set[str], set[str], set[str]]:
    """The names a script references, assigns and reads with a ``${NAME:-}`` default.

    A name counts as assigned only when the script assigns it unconditionally, at the top
    level, before it first uses it; a script that uses a name first, or assigns it only in a
    branch, reads it from the environment.
    """
    referenced, defaulted, first_use, first_assignment = set(), set(), {}, {}
    for number, line in enumerate(lines):
        for name in re.findall(rf"\$\{{?({NAME})", line):
            referenced.add(name)
            first_use.setdefault(name, number)
        defaulted |= set(re.findall(rf"\$\{{({NAME}):-", line))
        name = _top_level_assignment(line)
        if name:
            first_assignment.setdefault(name, number)
    assigned = {name for name, number in first_assignment.items()
                if number < first_use.get(name, len(lines))}
    return referenced - SHELL_SPECIAL, assigned, defaulted


def _environment_variables(path: Path) -> set[str]:
    """The names a script reads from the environment: defaulted or used before assigned."""
    referenced, assigned, defaulted = _shell_variables(_shell_lines(path))
    return (referenced - assigned) | defaulted


def _chain_variables() -> list[str]:
    """Every variable the chain, a script it runs or a module it runs reads from the environment.

    PYTHON is gone, as the chain runs every module through uv run (Task 1).
    """
    found = _environment_variables(CHAIN)
    for script, set_by_chain in CHAIN_SCRIPTS.items():
        found |= _environment_variables(script) - set_by_chain
    found |= {name for path in SOURCE.rglob("*.py") for name in _environment_reads(path)}
    return sorted(found - CHAIN_INTERNAL)


CHAIN_VARIABLES = _chain_variables()


@pytest.fixture
def chain_env() -> list[str]:
    """The names launch_phase_d.sh passes into the chain session."""
    text = "\n".join(_shell_lines(SCRIPT))
    match = re.search(r"^CHAIN_ENV=\((.*?)\)", text, re.DOTALL | re.MULTILINE)
    assert match, "launch_phase_d.sh no longer lists the chain's variables in CHAIN_ENV"
    return match.group(1).split()


@pytest.mark.parametrize("variable", CHAIN_VARIABLES)
def test_the_chain_session_is_given_every_variable_it_reads(chain_env, variable):
    """A variable missing here is dropped without any error at launch."""
    assert variable in chain_env


@pytest.mark.parametrize(("script", "names"), [(s.name, n) for s, n in CHAIN_SCRIPTS.items()])
def test_the_chain_sets_what_it_does_not_forward(script, names):
    """The chain sets on a script's command line every variable it does not forward.

    Such a variable is left out of the forwarding, so without it the tmux server's value would
    reach the script.
    """
    calls = [line for line in _shell_lines(CHAIN) if f"scripts/{script}" in line]
    assert calls, script
    for line in calls:
        for name in names:
            assert re.match(rf"^\s*{name}=\S+\s+bash scripts/{script}", line), line


def test_every_script_the_chain_runs_is_scanned():
    """A script the chain runs and the scan skips would have its variables dropped unseen."""
    called = set(re.findall(r"\b(?:bash|sh|source|\.)\s+scripts/(\S+\.sh)",
                            "\n".join(_shell_lines(CHAIN))))
    assert called == {script.name for script in CHAIN_SCRIPTS}


def test_the_synthesis_runs_on_the_copied_run():
    """The chain points the synthesis at the run it copied onto $FIREFLY, nothing else."""
    calls = [line for line in _shell_lines(CHAIN) if "scripts/data_synthesis.sh" in line]
    assert calls and all(line.lstrip().startswith('PARAMETER_PATH="./$FIREFLY" ') for line in calls)


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
    ("import os as o\ngetattr(o, 'environ')", True),
    ("import os\nvars(os)['environ']", True),
    ("def worker_count(env):\n    return env.get('A')", False),
    ("# os.environ\nprint('os.environ')", False),
    ("def f():\n    '''environ'''", False),
    ("x = 1\n'environ'", True),     # a bare string statement that is no docstring
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


@pytest.mark.parametrize("script", [CHAIN, *CHAIN_SCRIPTS], ids=lambda p: p.name)
def test_the_scripts_follow_the_convention(script):
    """Every variable a script uses is assigned with NAME= or read with a ${NAME:-} default.

    A variable used any other way, say by for, read or local, would be taken for user input
    and its forwarding asked for; this fails instead, so its form can be checked by hand.
    """
    referenced, assigned, defaulted = _shell_variables(_shell_lines(script))
    assert not referenced - assigned - defaulted


@pytest.mark.parametrize(("lines", "assigned"), [
    (["A=1", "echo $A"], {"A"}),
    (["A=\"$(cd \"$(dirname x)\" && pwd)\"", "echo $A"], {"A"}),
    (["A=1; echo hi", "echo $A"], {"A"}),
    (["A=$B cmd", "echo $A"], set()),
    (["if x; then", "  A=1", "fi", "echo $A"], set()),   # may not run
    (["echo A=1"], set()),
    (["A=1 cmd", "echo $A"], set()),                # sets A for cmd only
    (["A=\"x y\" cmd"], set()),
    (["echo $A", "A=1"], set()),                    # read before it is assigned
    (["for A in 1; do :; done"], set()),
])
def test_the_scan_takes_only_assignments_made_before_use(lines, assigned):
    """Only NAME= on a line of its own, before the first use, makes a name the script's own."""
    assert _shell_variables(lines)[1] == assigned


def test_a_comment_does_not_continue(tmp_path):
    """A comment ending in a backslash does not swallow the next line, as in bash."""
    script = tmp_path / "s.sh"
    script.write_text("# note \\\necho ${SECRET:-x}\n")
    assert _environment_variables(script) == {"SECRET"}


def test_continuation_lines_are_joined(tmp_path):
    """A NAME= argument on a continuation line belongs to the command it continues."""
    script = tmp_path / "s.sh"
    script.write_text("run x \\\n  A=1\necho $A\n")
    assert _shell_variables(_shell_lines(script))[1] == set()


@pytest.mark.parametrize(("source", "expected"), [
    ("import os\nos.environ['A']", {"A"}),
    ("import os\nos.environ.get('A')", {"A"}),
    ("import os\nos.environ.setdefault('A', '1')", {"A"}),
    ("import os\nos.getenv('A')", {"A"}),
    ("from os import getenv\ngetenv('A')", {"A"}),
    ("from os import environ\n'A' in environ", {"A"}),
    ("from os import environ\nenviron['A']", {"A"}),
    ("from os import getenv as g\ng('A')", {"A"}),
    ("from os import environ as e\ne.get('A')", {"A"}),
    ("import os\ne = os.environ\ne.get('A')", {"A"}),
    ("import os\nos.environ.pop('A')", {"A"}),
    ("import os\nos.getenv(key='A')", {"A"}),
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
  has-session)
    # STATUS_ON_CHECK: the optimiser exits, with that status, while the launcher checks.
    if [ -n "${{STATUS_ON_CHECK:-}}" ] && [ -e "{record}/firefly_started" ]; then
      echo "$STATUS_ON_CHECK" > firefly_phaseD.exit; exit 1
    fi
    [ -n "${{FIREFLY_LIVE:-}}" ] && [ -e "{record}/firefly_started" ]; exit ;;
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
        targets = re.findall(r"has-session -t (\S+)", "\n".join(_shell_lines(script)))
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


@pytest.mark.parametrize(("exit_status", "launched"), [("0", True), ("3", False)])
def test_an_optimiser_exiting_during_the_check_keeps_its_status(tmp_path, exit_status, launched):
    """An optimiser that exits during the launcher's check keeps its own status.

    The status is read again once the session is gone, so the exit is not reported as killed.
    """
    root, record, status, output = _run(tmp_path, SCRIPT, {"KILL_FIREFLY": "1",
                                                          "STATUS_ON_CHECK": exit_status})
    assert (status == 0) == launched, output
    assert "killed" not in output
    assert (record / "chain_env").exists() == launched


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
