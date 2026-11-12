#!/usr/bin/env bash
# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Generate the synthetic datasets from the fitted parameters.
#
# B36: the Taiwan first-wave scenario is kept as an end-to-end demonstration (it is the only
# output compared with a real epidemic curve), so it is generated alongside the single-seed
# spread data that every validation figure uses.
#
# Usage: scripts/data_synthesis.sh [--environment-checked]
#   --environment-checked  the caller (phase_d_chain.sh) checked the environment against
#                    uv.lock, so this script does not check it again
# Environment:
#   COVSYN_WORKERS  worker processes (default: the performance cores, Task 2)
#   PARAMETER_PATH  directory holding firefly_best.txt (default: the name phase_d_chain.sh copies
#                   the run onto)
#   UV_PROJECT_ENVIRONMENT  the environment the runs use (default .venv)
set -euo pipefail

usage="Usage: scripts/data_synthesis.sh [--environment-checked]"
if [ $# -gt 1 ] || { [ $# -eq 1 ] && [ "$1" != --environment-checked ]; }; then
    echo "$usage" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
modes=('spread_Taiwan_weight' 'taiwan_first_outbreak')
parameter_path="${PARAMETER_PATH:-./Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200}"
# A5: at least 1000 Monte Carlo runs; 100 was shown to be unreliable.
monte_carlo_number=1000

if [ ! -f "$parameter_path/firefly_best.txt" ]; then
    echo "ERROR: $parameter_path/firefly_best.txt not found; set PARAMETER_PATH to the fitted run."
    exit 1
fi

# The runs below use the environment as it is (--no-sync), so run on its own this script checks
# it against uv.lock first; the check changes nothing, and uv's message names what differs.
# The chain, which checked it at its start, passes --environment-checked.
if [ "${1:-}" != --environment-checked ]; then
    command -v uv >/dev/null || { echo "ERROR: uv not found on PATH; see the README"; exit 1; }
    uv sync --locked --check --inexact \
        || { echo "ERROR: ${UV_PROJECT_ENVIRONMENT:-.venv} differs from uv.lock; run uv sync"
             exit 1; }
fi

for mode in "${modes[@]}"; do
    result_path="./synthetic_data_results_$mode"
    mkdir -p "$result_path"
    uv run --no-sync python -m covsyn.model.data_synthesis_main --mode "$mode" \
        --monte_carlo_number "$monte_carlo_number" \
        --result_path "$result_path" \
        --parameter_path "$parameter_path"
    echo "Done! Results are saved in: $result_path"
done
