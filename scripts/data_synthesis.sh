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
# Usage: scripts/data_synthesis.sh
# Environment:
#   PYTHON          interpreter with the CovSyn dependencies (default python3)
#   CPU_CORES       worker processes (default 24)
#   PARAMETER_PATH  directory holding firefly_best.txt (default: the name phase_d_chain.sh copies
#                   the run onto)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cpu_cores="${CPU_CORES:-24}"
modes=('spread_Taiwan_weight' 'taiwan_first_outbreak')
parameter_path="${PARAMETER_PATH:-./Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200}"
# A5: at least 1000 Monte Carlo runs; 100 was shown to be unreliable.
monte_carlo_number=1000

if [ ! -f "$parameter_path/firefly_best.txt" ]; then
    echo "ERROR: $parameter_path/firefly_best.txt not found; set PARAMETER_PATH to the fitted run."
    exit 1
fi

for mode in "${modes[@]}"; do
    result_path="./synthetic_data_results_$mode"
    mkdir -p "$result_path"
    "$PYTHON" -m covsyn.model.data_synthesis_main --mode "$mode" \
        --monte_carlo_number "$monte_carlo_number" \
        --result_path "$result_path" \
        --cpu_cores "$cpu_cores" \
        --parameter_path "$parameter_path"
    echo "Done! Results are saved in: $result_path"
done
