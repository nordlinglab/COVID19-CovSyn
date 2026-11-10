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

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "launch_phase_d.sh"
CHAIN = SCRIPT.parent / "phase_d_chain.sh"
SYNTHESIS = SCRIPT.parent / "data_synthesis.sh"
SOURCE = SCRIPT.parent.parent / "src" / "covsyn"
# Every variable the chain, data_synthesis.sh or a module they run reads from the environment;
# the launcher passes each into the chain session. VERIFY_WORKERS is read by the checklist
# module (worker_count), the others by the two scripts.
CHAIN_VARIABLES = ("PREVIOUS_CHECKS", "VERIFY_WORKERS", "FIREFLY_RUN", "REVALIDATE",
                   "REVALIDATION_BLOCKS", "SELECT", "SELECT_TOP", "CPU_CORES", "FIREFLY_STATUS",
                   "UV_PROJECT_ENVIRONMENT")
# Set by the chain itself: OPTIMIZER_RUN after revalidation, PARAMETER_PATH on its call of
# data_synthesis.sh, so a value left in the tmux server's environment cannot reach either.
SET_BY_CHAIN = {"OPTIMIZER_RUN", "PARAMETER_PATH"}
# Set by the launcher on the chain process alone, not on its tmux session, so that a window
# opened there later does not inherit it.
SET_FOR_THE_CHAIN_PROCESS = {"ENVIRONMENT_CHECKED"}
SYNTHESIS_CALL = re.compile(r'^\s*PARAMETER_PATH="\./\$FIREFLY" bash scripts/data_synthesis\.sh\b')


def _code(path: Path) -> list[str]:
    """A script's lines without its comment lines."""
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")]


def _defaults(path: Path) -> set[str]:
    """The variables a script reads from the environment with a default, ${NAME:-...}."""
    return set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):-", "\n".join(_code(path))))


@pytest.fixture
def chain_env() -> list[str]:
    """The names launch_phase_d.sh passes into the chain session."""
    match = re.search(r"^CHAIN_ENV=\((.*?)\)", "\n".join(_code(SCRIPT)), re.DOTALL | re.MULTILINE)
    assert match, "launch_phase_d.sh no longer lists the chain's variables in CHAIN_ENV"
    return match.group(1).split()


def test_the_chain_session_is_given_every_variable_on_the_list(chain_env):
    """The launcher forwards exactly the listed variables; a missing one is dropped silently."""
    assert sorted(chain_env) == sorted(CHAIN_VARIABLES)


def test_the_list_holds_every_default_the_scripts_read():
    """A ${NAME:-} added to the chain or to data_synthesis.sh must join the list.

    The scripts read the environment only this way; a variable read otherwise is not caught
    here, a limit accepted for scripts that Task 4 of the rework plan replaces.
    """
    read = (_defaults(CHAIN) | _defaults(SYNTHESIS)) - SET_BY_CHAIN - SET_FOR_THE_CHAIN_PROCESS
    assert read <= set(CHAIN_VARIABLES)


def test_the_chain_runs_only_the_synthesis_script_and_sets_its_parameter_path():
    """The chain calls data_synthesis.sh in one fixed form, setting PARAMETER_PATH itself."""
    calls = [line for line in _code(CHAIN) if re.search(r"\.sh\b", line)]
    assert calls
    for line in calls:
        assert SYNTHESIS_CALL.match(line), line


# Read by a module but set by the chain's own code where it needs a value: select_by_checklist
# sets AGE_RR_FILE per candidate, so the chain clears any value the tmux server holds.
CLEARED_BY_CHAIN = {"AGE_RR_FILE"}


# The ways a module reads the environment in this code base: os.environ.get, os.getenv,
# environ[...] under any module name (not an assignment, which sets it for a child) and a
# mapping of it passed as env (worker_count). A del is left out by _environment_reads.
ENVIRONMENT_READ = re.compile(
    r"""\b(?:environ\.get|getenv|env\.get)\(\s*['"]([A-Z][A-Z0-9_]*)"""
    r"""|\benviron\[\s*['"]([A-Z][A-Z0-9_]*)['"]\s*\](?!\s*=[^=])""")


def _environment_reads(text: str) -> set[str]:
    """The variables a module's text reads from the environment, as ENVIRONMENT_READ finds them.

    A del of os.environ[...] is left out, as it sets nothing for this process to read.
    """
    names = set()
    for match in ENVIRONMENT_READ.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if re.match(r"\s*del\s", text[line_start:]):
            continue                            # every item of a del statement
        names.add(match.group(1) or match.group(2))
    return names


@pytest.mark.parametrize(("text", "names"), [
    ("os.environ.get('A')", {"A"}),
    ("os.getenv('A')", {"A"}),
    ("os.environ['A']", {"A"}),
    ("environ['A']", {"A"}),
    ("os.environ['A'] == 'x'", {"A"}),
    ("env.get('A')", {"A"}),
    ("os.environ['A'] = '1'", set()),              # sets it for a child process
    ("del os.environ['A']", set()),
    ("del  os.environ['A']", set()),
    ("del environ['A']", set()),
    ("myenviron['A']", set()),
    ("myenviron.get('A')", set()),
    ("_os.environ['A']", {"A"}),                   # os imported under another name
    ("del os.environ['A'], os.environ['B']", set()),
])
def test_environment_reads_are_told_from_writes(text, names):
    """A read counts; an assignment or a del of os.environ[...] does not."""
    assert _environment_reads(text) == names


@given(st.from_regex(r"[A-Z][A-Z0-9_]{0,10}", fullmatch=True), st.integers(1, 3),
       st.sampled_from(["os.environ", "environ", "_os.environ"]))
def test_reads_and_dels_of_any_name_are_told_apart(name, spaces, mapping):
    """For any variable name, a subscript read counts and a del of it does not."""
    assert _environment_reads(f"x = {mapping}['{name}']") == {name}
    assert _environment_reads(f"del{' ' * spaces}{mapping}['{name}']") == set()
    assert _environment_reads(f"{mapping}['{name}'] = '1'") == set()


def test_every_module_environment_read_is_forwarded_or_cleared():
    """A module's environment read either is on the list or the chain clears it first.

    The chain clears it at the top level, before it first runs verify_phase_d, which reads it.
    """
    read = set()
    for path in SOURCE.rglob("*.py"):
        read |= _environment_reads(path.read_text(encoding="utf-8"))
    assert "VERIFY_WORKERS" in read                 # read through worker_count's env mapping
    assert read <= set(CHAIN_VARIABLES) | CLEARED_BY_CHAIN
    code = _code(CHAIN)
    first_verify = next(i for i, line in enumerate(code) if "verify_phase_d" in line)
    for name in CLEARED_BY_CHAIN:
        unset = next(i for i, line in enumerate(code) if line == f"unset {name}")
        assert unset < first_verify, name


def test_the_synthesis_checks_the_environment_unless_the_launcher_did():
    """data_synthesis.sh checks the environment against uv.lock before it runs without syncing.

    Run on its own, as the README shows, nothing else checks it. Under the launcher, which
    checked it once and sets ENVIRONMENT_CHECKED, a lock pulled during the run must not stop the
    chain halfway, as every step runs in the environment the launcher checked. uv's own message
    is kept, as it names what differs.
    """
    code = "\n".join(_code(SYNTHESIS))
    check = re.search(r'if \[ -z "\$\{ENVIRONMENT_CHECKED:-\}" \]; then\n(.*?)\nfi', code, re.S)
    assert check, "the check no longer depends on ENVIRONMENT_CHECKED"
    sync = next(line for line in check.group(1).splitlines() if "uv sync" in line)
    assert "--locked --check --inexact" in sync
    assert "/dev/null" not in sync                  # uv's message names what differs
    assert code.index("uv sync --locked --check") < code.index("uv run --no-sync")


def test_the_firefly_modules_do_not_read_the_environment():
    """The firefly session gets WARM_START alone, which the shell passes as an argument."""
    for name in ("firefly_optimizer.py", "fast_cost.py"):
        text = (SOURCE / "calibration" / name).read_text(encoding="utf-8")
        assert not re.search(r"\b(?:environ|getenv)\b", text), name


def _stand_ins(tmp_path: Path) -> tuple[Path, Path]:
    """A bin directory with stand-ins for tmux, uv and sleep, and the directory they record in.

    new-session applies -c and -e and runs the firefly command with only those variables, PATH
    and the stand-in uv's UV_STATUS, as a real session gets the tmux server's environment and
    the client's PATH; it keeps the chain session's environment. has-session reports a session
    named in LIVE_SESSIONS, and the firefly session live, once started, only when FIREFLY_LIVE
    is set.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "record"
    record.mkdir()
    (bin_dir / "uv").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "${{UV_PROJECT_ENVIRONMENT:-}}" > "{record}/uv_$1_environment"\n'
        '[ "$1" = sync ] && exit "${UV_SYNC_STATUS:-0}"\n'
        '[ "$1" = run ] || exit 0\n'
        f'printf "%s\\n" "$@" > "{record}/uv_run_args"\n'
        'exit "${UV_STATUS:-0}"\n')
    (bin_dir / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "tmux").write_text(f"""#!/usr/bin/env bash
case "$1" in
  has-session)
    case " ${{LIVE_SESSIONS:-}} " in *" ${{3#=}} "*) exit 0 ;; esac
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
  [ -n "${{KILL_FIREFLY:-}}" ] || (cd "$dir" &&
    env -i PATH="$PATH" UV_STATUS="${{UV_STATUS:-0}}" "${{envs[@]}}" "$@")
else
  env -i "${{envs[@]}}" > "{record}/chain_env"
  printf "%s\\n" "$@" > "{record}/chain_command"
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
    # uv sync in the launcher and uv run in both sessions use the same environment.
    assert chain["UV_PROJECT_ENVIRONMENT"] == ".venv"
    assert (record / "uv_sync_environment").read_text().strip() == ".venv"
    assert (record / "uv_run_environment").read_text().strip() == ".venv"
    # ENVIRONMENT_CHECKED goes to the chain process only, not to its session.
    command = (record / "chain_command").read_text().split("\n")
    script = command.index("scripts/phase_d_chain.sh")
    assert command[0] == "env" and command[script - 1] == "bash"
    assert set(command[1:script - 1]) == {f"{name}=1" for name in SET_FOR_THE_CHAIN_PROCESS}
    assert not SET_FOR_THE_CHAIN_PROCESS & set(chain)
    assert set(chain) == set(CHAIN_VARIABLES)


def test_launch_runs_both_sessions_in_the_environment_the_user_names(tmp_path):
    """The launcher's uv sync and both sessions' uv run see one UV_PROJECT_ENVIRONMENT."""
    root, record, status, output = _run(tmp_path, SCRIPT, {"UV_PROJECT_ENVIRONMENT": "/opt/env"})
    assert status == 0, output
    assert (record / "uv_sync_environment").read_text().strip() == "/opt/env"
    assert (record / "uv_run_environment").read_text().strip() == "/opt/env"
    chain = dict(line.split("=", 1) for line in (record / "chain_env").read_text().splitlines())
    assert chain["UV_PROJECT_ENVIRONMENT"] == "/opt/env"


SYNTHESIS_FILES = {"_files": {"run/firefly_best.txt": "1 2 3\n"}}


def test_the_synthesis_alone_stops_on_an_environment_that_differs(tmp_path):
    """Run on its own, data_synthesis.sh stops before any run when the check fails.

    Its message names the environment in use.
    """
    root, record, status, output = _run(tmp_path, SYNTHESIS, {
        "PARAMETER_PATH": "run", "UV_SYNC_STATUS": "2", "UV_PROJECT_ENVIRONMENT": "/opt/env",
        **SYNTHESIS_FILES})
    assert status == 1
    assert "/opt/env differs from uv.lock" in output
    assert (record / "uv_sync_environment").read_text().strip() == "/opt/env"
    assert not (record / "uv_run_args").exists()


def test_the_synthesis_under_the_launcher_skips_the_check(tmp_path):
    """With ENVIRONMENT_CHECKED, a lock that changed mid-run does not stop the synthesis.

    It runs in the environment the launcher checked.
    """
    root, record, status, output = _run(tmp_path, SYNTHESIS, {
        "PARAMETER_PATH": "run", "UV_SYNC_STATUS": "2", "ENVIRONMENT_CHECKED": "1",
        "UV_PROJECT_ENVIRONMENT": "/opt/env", **SYNTHESIS_FILES})
    assert status == 0, output
    assert not (record / "uv_sync_environment").exists()
    assert (record / "uv_run_environment").read_text().strip() == "/opt/env"


def test_the_chain_runs_in_the_environment_it_is_given(tmp_path):
    """The chain's own uv run uses the UV_PROJECT_ENVIRONMENT the launcher passes."""
    root, record, status, output = _run(tmp_path, CHAIN, {
        "FIREFLY_RUN": "run", "UV_STATUS": "9", "UV_PROJECT_ENVIRONMENT": "/opt/env",
        "_files": {"run/firefly_result.txt": "1 2 3\n"}})
    # The chain's own run() reached the revalidation, which the stand-in uv fails.
    assert status != 0
    assert "revalidation failed" in (root / "phaseD_chain.log").read_text()
    assert "covsyn.calibration.revalidation" in (record / "uv_run_args").read_text()
    assert (record / "uv_run_environment").read_text().strip() == "/opt/env"


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
        code = "\n".join(_code(script))
        targets = re.findall(r"has-session -t (\S+)", code)
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


@pytest.mark.parametrize("live", ["firefly", "chain"])
def test_launch_refuses_while_a_session_runs(tmp_path, live):
    """A second launch beside a live firefly or chain session stops before touching anything."""
    root, record, status, output = _run(tmp_path, SCRIPT, {
        "LIVE_SESSIONS": live, "_files": {"firefly_phaseD.exit": "0\n"},
    })
    assert status == 1
    assert "already running" in output
    assert not (record / "uv_run_args").exists()
    assert (root / "firefly_phaseD.exit").read_text() == "0\n"


def _header_sections() -> tuple[str, str]:
    """The launcher header's inputs, and what it sets whatever the caller sets."""
    header = SCRIPT.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    inputs, marker, set_by_it = header.partition("# Set by this script")
    assert marker, "the launcher header no longer has a '# Set by this script' section"
    return inputs, set_by_it


@pytest.mark.parametrize("variable", sorted(SET_FOR_THE_CHAIN_PROCESS))
def test_what_the_launcher_sets_is_documented_as_set_by_it(variable):
    """A variable the launcher sets whatever the caller sets is documented apart from inputs.

    The launcher also clears a value the caller exports before any tmux server starts.
    """
    inputs, set_by_it = _header_sections()
    assert variable in set_by_it
    assert variable not in inputs
    code = _code(SCRIPT)
    unset = code.index(f"unset {variable}")
    assert unset < next(i for i, line in enumerate(code) if "tmux new-session" in line)


@pytest.mark.parametrize("variable", CHAIN_VARIABLES)
def test_the_variable_is_documented_in_the_usage_header(variable):
    """The header's inputs are where a run is configured from; an undocumented one is unusable."""
    assert variable in _header_sections()[0]


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
