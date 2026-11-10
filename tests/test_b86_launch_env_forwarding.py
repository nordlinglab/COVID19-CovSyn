# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""B86: launch_phase_d.sh must forward every variable the chain reads into its tmux session.

tmux does not give a new session the launching shell's environment, as the script's own comment
says, so each variable is named on the session's command line. VERIFY_WORKERS was not, which
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
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "launch_phase_d.sh"
CHAIN = SCRIPT.parent / "phase_d_chain.sh"
SOURCE = SCRIPT.parent.parent / "src" / "covsyn"
# Set by the chain or its modules themselves, not by the user: OPTIMIZER_RUN after
# revalidation, AGE_RR_FILE by select_by_checklist for each candidate's checklist.
CHAIN_INTERNAL = {"OPTIMIZER_RUN", "AGE_RR_FILE"}
SHELL_SPECIAL = {"BASH_SOURCE"}
# The modules the firefly session runs; every other module runs in the chain.
FIREFLY_MODULES = ("calibration/firefly_optimizer.py", "calibration/fast_cost.py")
NAME = r"[A-Z][A-Z0-9_]*"


def _environment_reads(path: Path) -> set[str]:
    """Names a module reads from the environment, found in its code, not its comments.

    Covers ``os.environ[...]``, ``os.environ.get``/``setdefault``, ``os.getenv``, ``... in
    os.environ`` and ``env.get``/``env[...]`` on a mapping passed from ``os.environ``, as
    ``worker_count`` reads VERIFY_WORKERS.
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        key = None
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"get", "setdefault"}:
                if isinstance(func.value, (ast.Name, ast.Attribute)) and \
                        getattr(func.value, "attr", getattr(func.value, "id", "")) in {"environ",
                                                                                     "env"}:
                    key = node.args[0]
            elif isinstance(func, ast.Attribute) and func.attr == "getenv":
                key = node.args[0]
        elif isinstance(node, ast.Subscript):
            target = node.value
            if getattr(target, "attr", getattr(target, "id", "")) in {"environ", "env"}:
                key = node.slice
        elif isinstance(node, ast.Compare) and any(isinstance(o, ast.In) for o in node.ops):
            if any(getattr(c, "attr", "") == "environ" for c in node.comparators):
                key = node.left
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and \
                re.fullmatch(NAME, key.value):
            names.add(key.value)
    return names


def _shell_code(path: Path) -> str:
    """A shell script without its comment lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _chain_variables() -> list[str]:
    """Every variable the chain or a module it runs reads from the environment.

    The chain reads one when it gives it a default or uses it without assigning it; PYTHON is
    gone, as the chain runs every module through uv run (Task 1).
    """
    chain = _shell_code(CHAIN)
    referenced = set(re.findall(rf"\$\{{?({NAME})", chain))
    assigned = set(re.findall(
        rf"(?:^|[;&|]\s*|\b(?:export|local|declare|readonly|for|read)\s+(?:-\w+\s+)*)({NAME})"
        rf"(?:=|\s+in\b|\s*$)", chain, re.MULTILINE))
    defaulted = set(re.findall(rf"\$\{{({NAME}):-", chain))
    modules = {
        name
        for path in SOURCE.rglob("*.py")
        if str(path.relative_to(SOURCE)) not in FIREFLY_MODULES
        for name in _environment_reads(path)
    }
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


def test_the_firefly_modules_read_only_what_the_firefly_session_is_given():
    """The firefly session gets WARM_START alone, so its modules may read nothing else."""
    for module in FIREFLY_MODULES:
        assert _environment_reads(SOURCE / module) <= {"WARM_START"}, module


def test_launch_forwards_the_values_unchanged(tmp_path):
    """Run the launcher with stand-ins for tmux and uv and record what each session gets.

    A value with a quote arrives intact, an unset one arrives empty, and the space-separated
    WARM_START reaches the optimiser as separate files whatever the user's login shell.
    """
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / SCRIPT.name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "record"
    record.mkdir()
    (bin_dir / "uv").write_text(
        "#!/usr/bin/env bash\n"
        f'[ "$1" = run ] && printf "%s\\n" "$@" > "{record}/uv_run_args"\n'
        "exit 0\n")
    # new-session: apply -c and -e, then run the command when it is the firefly one, and keep
    # the chain session's environment; has-session reports the firefly session once it exists.
    (bin_dir / "tmux").write_text(f"""#!/usr/bin/env bash
case "$1" in
  has-session) [ "$3" = firefly ] && [ -e "{record}/firefly" ]; exit ;;
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
touch "{record}/$name"
if [ "$name" = firefly ]; then (cd "$dir" && env "${{envs[@]}}" "$@")
else env -i "${{envs[@]}}" > "{record}/chain_env"; fi
""")
    for tool in ("uv", "tmux"):
        (bin_dir / tool).chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "WARM_START": "a/firefly_best.txt b/firefly_best.txt",
        "FIREFLY_RUN": "run's dir",
        "CPU_CORES": "8",
    }
    result = subprocess.run(["bash", str(root / "scripts" / SCRIPT.name)], env=env, cwd=root,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    args = (record / "uv_run_args").read_text().split("\n")
    start = args.index("--warm_start")
    assert args[start + 1:start + 3] == ["a/firefly_best.txt", "b/firefly_best.txt"]
    chain = dict(line.split("=", 1) for line in (record / "chain_env").read_text().splitlines())
    assert chain["FIREFLY_RUN"] == "run's dir"
    assert chain["CPU_CORES"] == "8"
    assert chain["SELECT_TOP"] == ""
    assert set(chain) == set(CHAIN_VARIABLES)


def test_the_variables_are_given_as_tmux_options_not_in_the_command():
    """Each value goes through tmux -e, never pasted into a command line.

    A quote in a value then cannot break the command, and no count is hard-coded: the right
    worker count depends on free memory.
    """
    text = _shell_code(SCRIPT)
    assert 'chain_env_args+=(-e "$name=${!name:-}")' in text
    assert '"${chain_env_args[@]}" bash scripts/phase_d_chain.sh' in text


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
