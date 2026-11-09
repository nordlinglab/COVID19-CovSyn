# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""F6: step 9 of phase_d_chain.sh never reports a stale or missing Wu 2025 figure as done.

Step 9 is cut out of the chain and run with stand-ins for its run and say functions, so the
shell logic is tested without the model: which figures fail, what is left in wu2025_figures,
the last message, and the exit status.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

CHAIN = Path(__file__).resolve().parent.parent / "scripts" / "phase_d_chain.sh"
STEP_9 = "# ---------------------------------------------------------------- 9. Wu"

# The bash on PATH, by full path. On Windows the one in System32 is the WSL launcher, which
# does not pass the environment through, so only another bash, such as Git's, will do.
BASH = shutil.which("bash")
if BASH is not None and "system32" in BASH.lower():
    BASH = None
pytestmark = pytest.mark.skipif(BASH is None, reason="needs a bash that is not WSL's launcher")


def _step_9() -> str:
    text = CHAIN.read_text(encoding="utf-8").replace("\r\n", "\n")
    return text[text.index(STEP_9):]


def _environment(failing: str) -> dict[str, str]:
    """Only what bash needs, so a developer's BASH_ENV or chain variables cannot leak in."""
    keep = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TMP", "TEMP") if k in os.environ}
    return {**keep, "FAILING": failing}


def _run_step_9(tmp_path: Path, failing: str) -> tuple[int, str, list[str]]:
    """Run step 9 where `failing` is a space-separated list of figures, or 'cheng2020'."""
    figures = tmp_path / "wu2025_figures"
    figures.mkdir()
    (figures / "wu2025_fig5_state_transitions.png").write_text("previous run")
    (figures / "wu2025_fig6_cheng2020.png").write_text("previous run")
    (figures / "wu2025_fig4_cost_parts_0123456789ab.csv").write_text("cache")
    (figures / "archive").mkdir()
    (tmp_path / "step9.sh").write_text(_step_9(), encoding="utf-8", newline="\n")
    stub = (
        "set -euo pipefail\n"
        "LOG=log; REPORT_FIRST=f; REPORT_CHENG=c; FIREFLY=x; FIREFLY_RUN=y\n"
        'say() { echo "SAY $*"; }\n'
        "run() {\n"
        '    for f in $FAILING; do\n'
        '        case " $* " in *" $f "*|*"--mode $f "*) return 1 ;; esac\n'
        "    done\n"
        "}\n"
        "source ./step9.sh\n"
    )
    done = subprocess.run([BASH, "-c", stub], cwd=tmp_path, capture_output=True,
                          encoding="utf-8", env=_environment(failing))
    said = [line for line in done.stdout.splitlines() if line.startswith("SAY ")]
    return done.returncode, said[-1] if said else "", sorted(p.name for p in figures.iterdir())


def test_a_clean_run_ends_all_done_and_leaves_no_previous_file(tmp_path):
    """Even an old cost-parts cache goes: the cache now lives in the optimizer run."""
    status, last, left = _run_step_9(tmp_path, "")
    assert status == 0 and last == "SAY ALL DONE"
    assert left == ["archive"]


@pytest.mark.parametrize("figure", ["fig3", "fig4", "fig5", "fig6", "fig7"])
def test_a_failed_figure_ends_with_an_error_and_leaves_no_stale_file(tmp_path, figure):
    status, last, left = _run_step_9(tmp_path, figure)
    assert status == 1
    assert last == f"SAY DONE WITH AN ERROR: Wu 2025 figures that failed: {figure}"
    assert "wu2025_fig5_state_transitions.png" not in left
    assert "wu2025_fig6_cheng2020.png" not in left


def test_a_failed_cheng2020_synthesis_stops_and_names_earlier_failures(tmp_path):
    status, last, left = _run_step_9(tmp_path, "fig4 cheng2020")
    assert status == 1
    assert "cheng2020 synthesis failed" in last and "fig4" in last and last.endswith("stopping")
    assert "wu2025_fig6_cheng2020.png" not in left
