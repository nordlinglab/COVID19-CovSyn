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

import env_scan as es
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "launch_phase_d.sh"
CHAIN = SCRIPT.parent / "phase_d_chain.sh"
SYNTHESIS = SCRIPT.parent / "data_synthesis.sh"
SOURCE = SCRIPT.parent.parent / "src" / "covsyn"
# The scripts the chain runs, each with the variables the chain sets on its command line and
# the value it sets: the synthesis runs on the run the chain copied onto $FIREFLY.
CHAIN_SCRIPTS = {SYNTHESIS: {"PARAMETER_PATH": '"./$FIREFLY"'}}
# Set by the chain or its modules themselves, not by the user: OPTIMIZER_RUN after
# revalidation, AGE_RR_FILE by select_by_checklist for each candidate's checklist.
CHAIN_INTERNAL = {"OPTIMIZER_RUN", "AGE_RR_FILE"}
# Every covsyn module the firefly session runs; the chain runs the rest, and these too.
FIREFLY_FILES = es.import_closure("covsyn.calibration.firefly_optimizer", SOURCE)


def _lines(path: Path) -> list[es.ShellLine]:
    return es.shell_lines(path.read_text(encoding="utf-8"))


def _chain_variables() -> list[str]:
    """Every variable the chain, a script it runs or a module it runs reads from the environment.

    PYTHON is gone, as the chain runs every module through uv run (Task 1).
    """
    found = es.environment_variables(CHAIN)
    for script, set_by_chain in CHAIN_SCRIPTS.items():
        found |= es.environment_variables(script) - set(set_by_chain)
    found |= {name for path in SOURCE.rglob("*.py") for name in es.environment_reads(path)}
    return sorted(found - CHAIN_INTERNAL)


CHAIN_VARIABLES = _chain_variables()


@pytest.fixture
def chain_env() -> list[str]:
    """The names launch_phase_d.sh passes into the chain session."""
    text = "\n".join(line.text for line in _lines(SCRIPT))
    match = re.search(r"^CHAIN_ENV=\((.*?)\)", text, re.DOTALL | re.MULTILINE)
    assert match, "launch_phase_d.sh no longer lists the chain's variables in CHAIN_ENV"
    return match.group(1).split()


@pytest.mark.parametrize("variable", CHAIN_VARIABLES)
def test_the_chain_session_is_given_every_variable_it_reads(chain_env, variable):
    """A variable missing here is dropped without any error at launch."""
    assert variable in chain_env


@pytest.mark.parametrize(("script", "values"), [(s.name, v) for s, v in CHAIN_SCRIPTS.items()])
def test_the_chain_sets_what_it_does_not_forward(script, values):
    """The chain sets on a script's command line every variable it does not forward.

    Such a variable is left out of the forwarding, so without it the tmux server's value would
    reach the script.
    """
    calls = [line.text for line in _lines(CHAIN) if script in es.called_scripts([line])]
    assert calls, script
    for line in calls:
        for name, value in values.items():
            assert re.match(rf"^\s*{name}={re.escape(value)}\s", line), line


def test_every_script_the_chain_runs_is_scanned():
    """A script the chain runs and the scan skips would have its variables dropped unseen."""
    assert es.called_scripts(_lines(CHAIN)) == {script.name for script in CHAIN_SCRIPTS}


def test_the_firefly_modules_do_not_touch_the_environment():
    """No module the firefly session imports reads the environment.

    The session gets WARM_START alone, which the shell passes to the optimiser as an argument.
    """
    assert len(FIREFLY_FILES) > 2
    for path in FIREFLY_FILES:
        assert not es.mentions_environment(path), path.name


@pytest.mark.parametrize("script", [CHAIN, *CHAIN_SCRIPTS], ids=lambda p: p.name)
def test_the_scripts_follow_the_convention(script):
    """Every variable a script uses is assigned before use or read with a ${NAME:-} default.

    A variable used any other way would be taken for user input and its forwarding asked for;
    this fails instead, so its form can be checked by hand.
    """
    referenced, assigned, defaulted = es.shell_variables(_lines(script))
    assert not referenced - assigned - defaulted


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
        code = "\n".join(line.text for line in _lines(script))
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
