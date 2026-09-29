#!/usr/bin/env bash
# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Launch a Phase D run: the firefly optimizer in its own tmux session, then the unattended chain.
#
# Usage: scripts/launch_phase_d.sh
# Environment:
#   PYTHON           interpreter with the CovSyn dependencies (default python3)
#   WARM_START       space-separated firefly_best.txt files whose best vectors seed the initial
#                    population (B49); default: the committed run 10 result
#   PREVIOUS_CHECKS  phaseD_checks.json of the previous run, for the chain's comparison step
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
PYPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
WARM_START="${WARM_START:-firefly_result/phaseD_run10/firefly_best.txt}"

if tmux has-session -t firefly 2>/dev/null || tmux has-session -t chain 2>/dev/null; then
  echo "ERROR: a firefly or chain session is already running, refusing to start"
  tmux ls
  exit 1
fi
# tmux does not pass this shell's environment to a new session, so it is given explicitly.
tmux new-session -d -s firefly \
  "cd '$ROOT' && PYTHONPATH='$PYPATH' '$PYTHON' -m covsyn.calibration.firefly_optimizer \
   --mode train --warm_start $WARM_START > firefly_phaseD.log 2>&1"
sleep 2
tmux new-session -d -s chain \
  "cd '$ROOT' && PYTHON='$PYTHON' PREVIOUS_CHECKS='${PREVIOUS_CHECKS:-}' bash scripts/phase_d_chain.sh"
sleep 1
tmux ls
echo "--- firefly log head ---"
head -6 firefly_phaseD.log 2>/dev/null || true
