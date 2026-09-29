#!/bin/bash
# Launch a Phase D run: firefly in its own tmux session, then the unattended chain.
set -u
cd ~/COVID19-CovSyn
PYTHON=/Users/rex/covsyn/bin/python
# B49 (E75): the best vectors of earlier runs seed fireflies 1 and 2 of the initial population.
WARM="--warm_start Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_120_run9/firefly_best.txt Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_120_run8/firefly_best.txt"
if tmux has-session -t firefly 2>/dev/null || tmux has-session -t chain 2>/dev/null; then
  echo "ERROR: a firefly or chain session is already running, refusing to start"
  tmux ls
  exit 1
fi
tmux new-session -d -s firefly "$PYTHON firefly_optimizer.py --mode train $WARM > firefly_phaseD.log 2>&1"
sleep 2
tmux new-session -d -s chain "bash phaseD_chain.sh"
sleep 1
tmux ls
echo "--- firefly log head ---"
head -6 firefly_phaseD.log 2>/dev/null
