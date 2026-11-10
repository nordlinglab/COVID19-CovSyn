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
from typing import NamedTuple

import pipeline_runner
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
CHAIN_VARIABLES = (
    "PREVIOUS_CHECKS",
    "VERIFY_WORKERS",
    "FIREFLY_RUN",
    "REVALIDATE",
    "REVALIDATION_BLOCKS",
    "SELECT",
    "SELECT_TOP",
    "CPU_CORES",
    "FIREFLY_STATUS",
    "UV_PROJECT_ENVIRONMENT",
)
# Set by the chain itself: OPTIMIZER_RUN after revalidation, PARAMETER_PATH on its call of
# data_synthesis.sh, so a value left in the tmux server's environment cannot reach either.
SET_BY_CHAIN = {"OPTIMIZER_RUN", "PARAMETER_PATH"}


def _code(path: Path) -> list[str]:
    """A script's lines without its comment lines.

    Split at newlines only, and a comment is a line whose first character after spaces and
    tabs is #, as bash reads a line; other white space or separators hide nothing.
    """
    with path.open(encoding="utf-8", newline="") as file:  # keep a lone CR, as bash does
        lines = file.read().split("\n")
    return [line for line in lines if not line.lstrip(" \t").startswith("#")]


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
    read = (_defaults(CHAIN) | _defaults(SYNTHESIS)) - SET_BY_CHAIN
    assert read <= set(CHAIN_VARIABLES)


# The only chain lines that name a script, word for word: one message and the fixed call. A
# golden list, not a parser of bash: any change to a line naming a script fails here, to be
# checked by hand.
SCRIPT_LINES = [
    '    say "then start scripts/phase_d_chain.sh itself '
    '(not launch_phase_d.sh, which starts a new"',
    'PARAMETER_PATH="./$FIREFLY" bash scripts/data_synthesis.sh --environment-checked '
    '>> "$LOG" 2>&1 \\',
]


def test_the_chain_names_only_the_golden_script_lines():
    """The code lines naming a script are word for word the golden ones.

    This checks the spelling only; test_the_chain_runs_the_synthesis_at_step_2_on_the_runs_copy
    and test_a_failed_synthesis_stops_the_chain check what bash runs.
    """
    naming = [line for line in _code(CHAIN) if re.search(r"\.sh\b", line)]
    assert naming == SCRIPT_LINES


SYNTHESIS_FILES = {"_files": {"run/firefly_best.txt": "1 2 3\n"}}


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


MODES = list(pipeline_runner.MODES)


def test_the_synthesis_alone_reads_the_run_the_chain_copies_onto(tmp_path, chain_at_step_2):
    """Run on its own without PARAMETER_PATH, data_synthesis.sh reads the chain's copy.

    The directory comes from a chain run, so no directory name is written in the test.
    """
    directory = chain_at_step_2.directory
    root, record, status, output = _run(
        tmp_path, SYNTHESIS, {"_files": {f"{directory}/firefly_best.txt": "1 2 3\n"}}
    )
    assert status == 0, output
    assert _synthesis_runs(record, output) == [(mode, f"./{directory}") for mode in MODES], output


def test_the_synthesis_uses_the_parameter_path_it_is_given(tmp_path):
    """With PARAMETER_PATH set, both modes run on that directory."""
    root, record, status, output = _run(
        tmp_path, SYNTHESIS, {"PARAMETER_PATH": "run", **SYNTHESIS_FILES}
    )
    assert status == 0, output
    assert _synthesis_runs(record, output) == [(mode, "run") for mode in MODES], output


def _synthesis_stand_in(record: Path) -> str:
    """A data_synthesis.sh that records its call in the uv stand-in's record.

    The call goes in order with the uv runs (SYNTHESIS, PARAMETER_PATH as its own field, then
    the arguments), and the stand-in exits with SYNTHESIS_STATUS.
    """
    return (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f'printf "%s\\x1f" SYNTHESIS "$PARAMETER_PATH" "$@" >> "{record}/runs"\n'
        f'printf "\\x1e" >> "{record}/runs"\n'
        'exit "${SYNTHESIS_STATUS:-0}"\n'
    )


class ChainRun(NamedTuple):
    """A chain run to its synthesis step."""

    root: Path  # the scratch repository
    record: Path  # where the stand-ins record their calls
    status: int
    context: str  # output and log, for assertion messages
    log: str  # phaseD_chain.log, which the user reads
    directory: str  # what the synthesis was told to read: PARAMETER_PATH without ./


def _chain_to_step_2(tmp_path: Path, synthesis_status: str) -> ChainRun:
    """Run the chain to its synthesis step with a stand-in data_synthesis.sh.

    With REVALIDATE=0 no uv run happens before the synthesis, so every recorded uv run comes
    after it.
    """
    root, record, status, output = _run(
        tmp_path,
        CHAIN,
        {
            "FIREFLY_RUN": "run",
            "REVALIDATE": "0",
            "UV_STATUS": "9",  # every later module fails; step 3b stops the run
            "SYNTHESIS_STATUS": synthesis_status,
            "_files": {
                "run/firefly_best.txt": "1 2 3\n",
                "scripts/data_synthesis.sh": _synthesis_stand_in(_record_dir(tmp_path)),
            },
        },
    )
    log_path = root / "phaseD_chain.log"
    log = _read(log_path) if log_path.exists() else ""
    context = output + log
    syntheses = [call for call in _records(record / "runs", context) if call[:1] == ["SYNTHESIS"]]
    assert len(syntheses) == 1 and syntheses[0][2:] == ["--environment-checked"], context
    path = syntheses[0][1]
    assert path.startswith("./"), context
    return ChainRun(root, record, status, context, log, path[2:])


@pytest.fixture(scope="module")
def chain_at_step_2(tmp_path_factory) -> ChainRun:
    """One chain run to step 2 with a synthesis that succeeds, shared by the tests below."""
    return _chain_to_step_2(tmp_path_factory.mktemp("chain"), "0")


def test_the_chain_runs_the_synthesis_at_step_2_on_the_runs_copy(chain_at_step_2):
    """Run to step 2 with REVALIDATE=0: the synthesis runs once, on the run's copy.

    That copy is the directory the regression harness checks, and the log reports success.
    Only this path is run; other settings reach the same call by reading of the script.
    """
    run = chain_at_step_2
    assert (run.root / run.directory / "firefly_best.txt").exists(), run.context
    assert run.directory == pipeline_runner.COMPAT_DIR, run.context
    assert "synthetic data done" in run.log, run.context
    # The steps after it run uv: recorded after the synthesis, in order.
    calls = _records(run.record / "runs", run.context)
    assert calls[0][0] == "SYNTHESIS" and len(calls) > 1, run.context


def test_a_failed_synthesis_stops_the_chain(tmp_path):
    """A failed synthesis stops the chain, and the log, which the user reads, says so."""
    run = _chain_to_step_2(tmp_path, "1")
    assert run.status != 0, run.context
    # The synthesis is the only call recorded: none comes before it (REVALIDATE=0), and a chain
    # that went on would run uv in its next step.
    calls = _records(run.record / "runs", run.context)
    assert [call[0] for call in calls] == ["SYNTHESIS"], run.context
    assert "] ERROR: data synthesis failed" in run.log, run.context
    assert "synthetic data done" not in run.log, run.context


# Read by a module but set by the chain's own code where it needs a value: select_by_checklist
# sets AGE_RR_FILE per candidate, so the chain clears any value the tmux server holds.
CLEARED_BY_CHAIN = {"AGE_RR_FILE"}


# The ways a module reads the environment in this code base: os.environ.get, os.getenv,
# environ[...] under any module name (not an assignment, which sets it for a child) and a
# mapping of it passed as env (worker_count). A del is left out by _environment_reads.
ENVIRONMENT_READ = re.compile(
    r"""\b(?:environ\.get|getenv|env\.get)\(\s*['"]([A-Z][A-Z0-9_]*)"""
    r"""|\benviron\[\s*['"]([A-Z][A-Z0-9_]*)['"]\s*\](?!\s*=[^=])"""
)


def _environment_reads(text: str) -> set[str]:
    """The variables a module's text reads from the environment, as ENVIRONMENT_READ finds them.

    A del of os.environ[...] is left out, as it sets nothing for this process to read.
    """
    names = set()
    for match in ENVIRONMENT_READ.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if re.match(r"\s*del\s", text[line_start:]):
            continue  # every item of a del statement
        names.add(match.group(1) or match.group(2))
    return names


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("os.environ.get('A')", {"A"}),
        ("os.getenv('A')", {"A"}),
        ("os.environ['A']", {"A"}),
        ("environ['A']", {"A"}),
        ("os.environ['A'] == 'x'", {"A"}),
        ("env.get('A')", {"A"}),
        ("os.environ['A'] = '1'", set()),  # sets it for a child process
        ("del os.environ['A']", set()),
        ("del  os.environ['A']", set()),
        ("del environ['A']", set()),
        ("myenviron['A']", set()),
        ("myenviron.get('A')", set()),
        ("_os.environ['A']", {"A"}),  # os imported under another name
        ("del os.environ['A'], os.environ['B']", set()),
    ],
)
def test_environment_reads_are_told_from_writes(text, names):
    """A read counts; an assignment or a del of os.environ[...] does not."""
    assert _environment_reads(text) == names


@given(
    st.from_regex(r"[A-Z][A-Z0-9_]{0,10}", fullmatch=True),
    st.integers(1, 3),
    st.sampled_from(["os.environ", "environ", "_os.environ"]),
)
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
    assert "VERIFY_WORKERS" in read  # read through worker_count's env mapping
    assert read <= set(CHAIN_VARIABLES) | CLEARED_BY_CHAIN
    code = _code(CHAIN)
    first_verify = next(i for i, line in enumerate(code) if "verify_phase_d" in line)
    for name in CLEARED_BY_CHAIN:
        unset = next(i for i, line in enumerate(code) if line == f"unset {name}")
        assert unset < first_verify, name


def test_the_chain_checks_the_environment_once_before_it_waits(tmp_path):
    """The chain checks the environment against uv.lock at its start, before anything runs.

    The check changes nothing, so it is safe while the optimiser runs; done at the start, it
    stops a chain on a stale environment before hours of work, and a lock pulled later does not
    stop the chain halfway. uv's own message is kept in the log.
    """
    root, record, status, output = _run(
        tmp_path,
        CHAIN,
        {
            "FIREFLY_RUN": "run",
            "UV_SYNC_STATUS": "2",
            "UV_PROJECT_ENVIRONMENT": "/opt/env",
            "_files": {"run/firefly_result.txt": "1 2 3\n"},
        },
    )
    log = (root / "phaseD_chain.log").read_text()
    assert status != 0
    assert "/opt/env differs from uv.lock" in log
    assert (record / "uv_sync_environment").read_text().strip() == "/opt/env"
    # The failing check is the only call: nothing ran on the stale environment.
    assert (record / "calls").read_text().splitlines() == ["uv sync --locked --check --inexact"]


@pytest.mark.parametrize("launched", [True, False])
def test_a_failed_check_says_how_to_go_on(tmp_path, launched):
    """The chain stops with nothing run and shows how to go on, with the values it was given.

    It names the environment to sync and the repository root, says to start the chain itself,
    not the launcher, and lists every setting as a tmux -e option, empty ones included, so a
    launched chain's FIREFLY_STATUS is not lost and a left-out setting cannot take a stale
    value from the tmux server on restart.
    """
    env = {"UV_SYNC_STATUS": "2", "UV_PROJECT_ENVIRONMENT": "/opt/env"}
    if launched:
        env["FIREFLY_STATUS"] = "firefly.exit"
    root, record, status, output = _run(tmp_path, CHAIN, env)
    log = (root / "phaseD_chain.log").read_text()
    assert status != 0
    assert (record / "calls").read_text().splitlines() == ["uv sync --locked --check --inexact"]
    assert "Nothing has run" in log and f"run in {root}" in log
    assert "UV_PROJECT_ENVIRONMENT=/opt/env uv sync --locked --inexact" in log
    assert "not launch_phase_d.sh" in log
    assert "empty ones included" in log
    assert ("  -e FIREFLY_STATUS=firefly.exit" in log) == launched
    shown = [line.split("  -e ", 1)[1] for line in log.splitlines() if "  -e " in line]
    assert sorted(item.split("=", 1)[0] for item in shown) == sorted(CHAIN_VARIABLES)
    assert "SELECT=" in shown  # an empty one is shown too
    assert "  -e UV_PROJECT_ENVIRONMENT=/opt/env" in log


def test_the_failed_check_lists_every_forwarded_variable():
    """The settings the chain shows on a failed check are the launcher's list."""
    code = "\n".join(_code(CHAIN)).replace("\\\n", " ")
    match = re.search(r"for name in ([A-Z0-9_ ]+); do", code)
    assert match, "the chain no longer lists its settings on a failed check"
    assert sorted(match.group(1).split()) == sorted(CHAIN_VARIABLES)


def test_the_chain_checks_before_it_waits_for_the_optimiser(tmp_path):
    """A passing check comes before the first look for the firefly session."""
    root, record, status, output = _run(
        tmp_path,
        CHAIN,
        {"FIREFLY_RUN": "run", "UV_STATUS": "9", "_files": {"run/firefly_result.txt": "1 2 3\n"}},
    )
    calls = (record / "calls").read_text().splitlines()
    assert calls[0] == "uv sync --locked --check --inexact"
    assert calls[1] == "tmux has-session -t =firefly"
    assert sum(call.startswith("uv sync") for call in calls) == 1  # checked once
    assert status != 0 and "revalidation failed" in (root / "phaseD_chain.log").read_text()


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
    (bin_dir / "tmux").write_text(f"""#!/usr/bin/env bash
case "$1" in
  has-session)
    echo "tmux $*" >> "{record}/calls"
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
    args = env.pop("_args", [])
    for name, text in env.pop("_files", {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    bin_dir, record = _stand_ins(tmp_path)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **env}
    result = subprocess.run(
        ["bash", str(root / "scripts" / script.name), *args],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = result.stdout + result.stderr
    # Printed for pytest, which shows a failing test's captured output: whichever check fails,
    # and however (an assertion or an error), the script's output and the chain log are there.
    print(f"--- {script.name} exited {result.returncode}; its output:\n{output}")
    log = root / "phaseD_chain.log"
    if log.exists():
        print(f"--- phaseD_chain.log:\n{_read(log)}")
    return root, record, result.returncode, output


def test_launch_forwards_the_values_unchanged(tmp_path):
    """Run the launcher with stand-ins for tmux and uv and record what each session gets.

    A value with a quote arrives intact, an unset one arrives empty, the space-separated
    WARM_START reaches the optimiser as separate files whatever the user's login shell, and
    the firefly session leaves the optimiser's exit status where the chain looks for it.
    """
    root, record, status, output = _run(
        tmp_path,
        SCRIPT,
        {
            "WARM_START": "a/firefly_best.txt b/firefly_best.txt",
            "FIREFLY_RUN": "run's dir",
            "CPU_CORES": "8",
            "_files": {"firefly_phaseD.exit": "1\n"},
        },
    )
    assert status == 0, output
    args = _uv_runs(record, output)[-1]
    start = args.index("--warm_start")
    assert args[start + 1 : start + 3] == ["a/firefly_best.txt", "b/firefly_best.txt"]
    chain = dict(line.split("=", 1) for line in (record / "chain_env").read_text().splitlines())
    assert (root / chain["FIREFLY_STATUS"]).read_text().strip() == "0"
    assert chain["FIREFLY_RUN"] == "run's dir"
    assert chain["CPU_CORES"] == "8"
    assert chain["SELECT_TOP"] == ""
    # uv sync in the launcher and uv run in both sessions use the same environment.
    assert chain["UV_PROJECT_ENVIRONMENT"] == ".venv"
    assert (record / "uv_sync_environment").read_text().strip() == ".venv"
    assert (record / "uv_run_environment").read_text().strip() == ".venv"
    command = (record / "chain_command").read_text().split("\n")
    assert command[:2] == ["bash", "scripts/phase_d_chain.sh"]
    assert set(chain) == set(CHAIN_VARIABLES)


def test_launch_runs_both_sessions_in_the_environment_the_user_names(tmp_path):
    """The launcher's uv sync and both sessions' uv run see one UV_PROJECT_ENVIRONMENT."""
    root, record, status, output = _run(tmp_path, SCRIPT, {"UV_PROJECT_ENVIRONMENT": "/opt/env"})
    assert status == 0, output
    assert (record / "uv_sync_environment").read_text().strip() == "/opt/env"
    assert (record / "uv_run_environment").read_text().strip() == "/opt/env"
    chain = dict(line.split("=", 1) for line in (record / "chain_env").read_text().splitlines())
    assert chain["UV_PROJECT_ENVIRONMENT"] == "/opt/env"


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


def test_the_chain_runs_in_the_environment_it_is_given(tmp_path):
    """The chain's own uv run uses the UV_PROJECT_ENVIRONMENT the launcher passes."""
    root, record, status, output = _run(
        tmp_path,
        CHAIN,
        {
            "FIREFLY_RUN": "run",
            "UV_STATUS": "9",
            "UV_PROJECT_ENVIRONMENT": "/opt/env",
            "_files": {"run/firefly_result.txt": "1 2 3\n"},
        },
    )
    # The chain's own run() reached the revalidation, which the stand-in uv fails.
    assert status != 0
    assert "revalidation failed" in (root / "phaseD_chain.log").read_text()
    assert "covsyn.calibration.revalidation" in _uv_runs(record, output)[-1]
    assert (record / "uv_run_environment").read_text().strip() == "/opt/env"


def test_launch_uses_a_status_file_the_user_names(tmp_path):
    """A FIREFLY_STATUS set before the launch is the file the session writes and the chain reads.

    The path is relative to the repository root.
    """
    root, record, status, output = _run(
        tmp_path,
        SCRIPT,
        {
            "FIREFLY_STATUS": "runs/x.exit",
            "_files": {"runs/.keep": ""},
        },
    )
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
    root, record, status, output = _run(
        tmp_path, SCRIPT, {"KILL_FIREFLY": "1", "STATUS_ON_CHECK": exit_status}
    )
    assert (status == 0) == launched, output
    assert "killed" not in output
    assert (record / "chain_env").exists() == launched


def test_a_running_firefly_session_starts_the_chain(tmp_path):
    """While the optimiser still runs at the check, the launcher starts the chain."""
    root, record, status, output = _run(
        tmp_path, SCRIPT, {"KILL_FIREFLY": "1", "FIREFLY_LIVE": "1"}
    )
    assert status == 0, output
    assert (root / "firefly_phaseD.exit").read_text().strip() == "running"
    assert (record / "chain_env").exists()


@pytest.mark.parametrize(
    ("content", "launched", "stops"),
    [
        ("1", True, True),
        ("137", True, True),
        ("running", True, True),  # killed before the optimiser exited
        (None, True, True),  # status file gone
        ("0", True, False),
        ("1", False, False),  # a hand-run chain ignores a stale status
        (None, False, False),
    ],
)
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
    assert (record / "runs").exists() != stops


@pytest.mark.parametrize("live", ["firefly", "chain"])
def test_launch_refuses_while_a_session_runs(tmp_path, live):
    """A second launch beside a live firefly or chain session stops before touching anything."""
    root, record, status, output = _run(
        tmp_path,
        SCRIPT,
        {
            "LIVE_SESSIONS": live,
            "_files": {"firefly_phaseD.exit": "0\n"},
        },
    )
    assert status == 1
    assert "already running" in output
    assert not (record / "runs").exists()
    assert (root / "firefly_phaseD.exit").read_text() == "0\n"


@pytest.mark.parametrize("variable", CHAIN_VARIABLES)
def test_the_variable_is_documented_in_the_usage_header(variable):
    """The header is where a run is configured from, so each variable needs its own entry."""
    text = SCRIPT.read_text(encoding="utf-8")
    header, marker, _ = text.partition("set -euo pipefail")
    assert marker, "launch_phase_d.sh no longer has its set -euo pipefail line"
    assert re.search(rf"^#   {variable} +\S", header, re.MULTILINE), variable


def test_the_checklist_uses_the_shared_helper_for_its_worker_count():
    """Both the forwarding and the parsing have to agree on one definition."""
    checklist = (
        SCRIPT.parent.parent / "src" / "covsyn" / "validation" / "verify_phase_d.py"
    ).read_text(encoding="utf-8")
    assert "worker_count(" in checklist


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("8", 8),  # E98's setting for this shared machine, about 39 GB
        ("1", 1),
        ("64", 64),  # above the cap, because the cap is only a default
        (" 8 ", 8),  # a forwarded value can carry whitespace
    ],
)
def test_an_explicit_worker_count_is_taken_as_given(value, expected):
    """An explicit setting overrides the default, above the cap included."""
    from covsyn.validation.acceptance import worker_count

    assert worker_count({"VERIFY_WORKERS": value}, cpus=32) == expected


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_unset_or_empty_value_falls_back_to_the_default(value):
    """The shell forwards '' when the variable is unset, which must not raise or mean zero."""
    from covsyn.validation.acceptance import worker_count

    env = {} if value is None else {"VERIFY_WORKERS": value}
    assert worker_count(env, cpus=48) == 32  # the cap
    assert worker_count(env, cpus=4) == 4  # fewer cores than the cap


@pytest.mark.parametrize("value", ["0", "-1", "nonsense", "3.5"])
def test_a_value_that_is_not_a_positive_count_falls_back(value):
    """A typo must not silently run the checklist on one worker or crash it."""
    from covsyn.validation.acceptance import worker_count

    assert worker_count({"VERIFY_WORKERS": value}, cpus=32) == 32
