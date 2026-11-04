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

import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "launch_phase_d.sh"
# Every variable the chain, or anything it starts, reads from the environment.
CHAIN_VARIABLES = ("PYTHON", "PREVIOUS_CHECKS", "VERIFY_WORKERS")


@pytest.fixture
def chain_session() -> str:
    """The tmux command that starts the chain session, however many lines it spans."""
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"tmux new-session -d -s chain\b(.*?phase_d_chain\.sh\s*\")", text,
                      re.DOTALL)
    assert match, "the chain session is no longer started by a tmux new-session command"
    return match.group(1)


@pytest.mark.parametrize("variable", CHAIN_VARIABLES)
def test_the_chain_session_is_given_every_variable_it_reads(chain_session, variable):
    """A variable missing here is dropped without any error at launch."""
    assert f"{variable}=" in chain_session


def test_verify_workers_is_passed_through_rather_than_fixed(chain_session):
    """The script must not hard-code a worker count: the right one depends on free memory."""
    assert "VERIFY_WORKERS='${VERIFY_WORKERS:-}'" in chain_session


def test_the_variable_is_documented_in_the_usage_header():
    """The header is where a run is configured from, so an undocumented variable is unusable."""
    header = SCRIPT.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "VERIFY_WORKERS" in header


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
