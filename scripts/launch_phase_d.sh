#!/usr/bin/env bash
# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Launch a Phase D run: the firefly optimizer in its own tmux session, then the unattended chain.
#
# Usage: scripts/launch_phase_d.sh
# Environment:
#   WARM_START       space-separated firefly_best.txt files whose best vectors seed the initial
#                    population (B49); default: the committed run 10 result
#   PREVIOUS_CHECKS  phaseD_checks.json of the previous run, for the chain's comparison step
#   VERIFY_WORKERS   bootstrap workers for the checklist (B86, E98); each holds about 4.9 GB,
#                    so the default of min(cores, 32) needs about 157 GB and swaps when the
#                    machine is shared. Unset means that default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WARM_START="${WARM_START:-firefly_result/phaseD_run10/firefly_best.txt}"

if tmux has-session -t firefly 2>/dev/null || tmux has-session -t chain 2>/dev/null; then
  echo "ERROR: a firefly or chain session is already running, refusing to start"
  tmux ls
  exit 1
fi
# tmux does not pass this shell's environment to a new session, so it is given explicitly;
# the session inherits the tmux server's PATH, which may lack uv, so uv goes by its full path.
UV="$(command -v uv)"
tmux new-session -d -s firefly \
  "cd '$ROOT' && '$UV' run --no-sync python -m covsyn.calibration.firefly_optimizer \
   --mode train --warm_start $WARM_START > firefly_phaseD.log 2>&1"
sleep 2
tmux new-session -d -s chain \
  "cd '$ROOT' && PREVIOUS_CHECKS='${PREVIOUS_CHECKS:-}' \
   VERIFY_WORKERS='${VERIFY_WORKERS:-}' bash scripts/phase_d_chain.sh"
sleep 1
tmux ls
echo "--- firefly log head ---"
head -6 firefly_phaseD.log 2>/dev/null || true
