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
#   CPU_CORES        workers for the selection's and the data synthesis's simulations
#                    (default 24)
#   FIREFLY_STATUS   the file where the firefly session leaves its exit status, which the
#                    chain checks before it uses FIREFLY_RUN; default firefly_phaseD.exit,
#                    relative to the repository root. The launcher fails if the firefly session
#                    ends within its first 2 s with a status other than 0; a later failure
#                    stops the chain.
#   UV_PROJECT_ENVIRONMENT  the environment uv checks here and both sessions run in; default
#                    .venv
# An unset variable reaches the chain empty, which the chain reads as its default; the launcher
# always gives FIREFLY_STATUS and UV_PROJECT_ENVIRONMENT a value.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WARM_START="${WARM_START:-firefly_result/phaseD_run10/firefly_best.txt}"

if tmux has-session -t =firefly 2>/dev/null || tmux has-session -t =chain 2>/dev/null; then
  echo "ERROR: a firefly or chain session is already running, refusing to start"
  tmux ls
  exit 1
fi
command -v uv >/dev/null || { echo "ERROR: uv not found on PATH; see the README"; exit 1; }
# The sessions run uv run --no-sync, so the environment is checked against uv.lock here, once,
# before anything runs on it; --inexact leaves alone packages the lock does not list. The
# sessions get the tmux server's environment, so the environment checked here is passed to
# them, lest they run in another one.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv}"
uv sync --locked --inexact
# A tmux session gets the tmux server's environment, not this shell's, so every variable a
# session reads is passed with -e. PATH is the exception: tmux gives a new session the PATH of
# the shell that runs new-session (tested on tmux 3.6), so the uv found above is found there.
# Each session command is given as separate arguments, which tmux runs directly rather than
# through the user's default shell: bash then expands the variables, so WARM_START splits into
# its files under any login shell, and a quote in a value cannot break the command line.
CHAIN_ENV=(PREVIOUS_CHECKS VERIFY_WORKERS FIREFLY_RUN REVALIDATE REVALIDATION_BLOCKS SELECT
           SELECT_TOP CPU_CORES FIREFLY_STATUS UV_PROJECT_ENVIRONMENT)
# The firefly session replaces "running" with the optimiser's exit status, so a session that is
# killed before the optimiser exits leaves "running", which stops the chain too.
FIREFLY_STATUS="${FIREFLY_STATUS:-firefly_phaseD.exit}"
echo running > "$FIREFLY_STATUS"
chain_env_args=()
for name in "${CHAIN_ENV[@]}"; do chain_env_args+=(-e "$name=${!name:-}"); done
# shellcheck disable=SC2016
tmux new-session -d -s firefly -c "$ROOT" -e "WARM_START=$WARM_START" \
  -e "UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT" \
  -e "FIREFLY_STATUS=$FIREFLY_STATUS" bash -c \
  'uv run --no-sync python -m covsyn.calibration.firefly_optimizer \
   --mode train --warm_start $WARM_START > firefly_phaseD.log 2>&1; \
   echo $? > "$FIREFLY_STATUS.tmp" && mv "$FIREFLY_STATUS.tmp" "$FIREFLY_STATUS"'
sleep 2
# The status is renamed into place, so it is never read half-written. "running" with no
# session left means the session was killed before the optimiser exited; the file is read again
# after the session check, as the optimiser may have exited in between. The "=" makes tmux match
# the session name exactly rather than as a prefix.
status=$(cat "$FIREFLY_STATUS")
if [ "$status" = running ] && ! tmux has-session -t =firefly 2>/dev/null; then
  status=$(cat "$FIREFLY_STATUS")
  if [ "$status" = running ]; then status='killed'; fi
fi
if [ "$status" != running ] && [ "$status" != 0 ]; then
  echo "ERROR: the firefly session ended at startup with status $status; not starting the chain"
  tail -n 20 firefly_phaseD.log 2>/dev/null || true
  exit 1
fi
tmux new-session -d -s chain -c "$ROOT" "${chain_env_args[@]}" bash scripts/phase_d_chain.sh
sleep 1
tmux ls
echo "--- firefly log head ---"
head -6 firefly_phaseD.log 2>/dev/null || true
