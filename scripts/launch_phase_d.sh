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
#   FIREFLY_RUN      the optimizer output directory the chain reads; default: the directory the
#                    optimizer's default settings name
#   REVALIDATE       1 (default) re-scores the candidates on unseen seeds (B56); 0 skips it
#   REVALIDATION_BLOCKS  unseen seed blocks for that re-scoring (default 10, E95)
#   SELECT           1 (default) runs the checklist on the best revalidated candidates (B60)
#   SELECT_TOP       how many candidates that selection checks (default 5)
#   CPU_CORES        workers for the selection's simulations (default 24)
# An unset variable reaches the chain empty, which the chain reads as its default.
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
   VERIFY_WORKERS='${VERIFY_WORKERS:-}' FIREFLY_RUN='${FIREFLY_RUN:-}' \
   REVALIDATE='${REVALIDATE:-}' REVALIDATION_BLOCKS='${REVALIDATION_BLOCKS:-}' \
   SELECT='${SELECT:-}' SELECT_TOP='${SELECT_TOP:-}' CPU_CORES='${CPU_CORES:-}' \
   bash scripts/phase_d_chain.sh"
sleep 1
tmux ls
echo "--- firefly log head ---"
head -6 firefly_phaseD.log 2>/dev/null || true
