#!/usr/bin/env bash
# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Phase D: wait for the firefly run to finish, then synthesise the data and produce every
# validation output in one go, so the whole chain can run unattended overnight.
#
#   1. wait for the firefly tmux session to end
#   2. archive the previous synthetic data, then generate spread_Taiwan_weight and
#      taiwan_first_outbreak with 1000 Monte-Carlo runs each (decisions A4, A5, B36)
#   3. run the checklist of every post-simulation item of B14 / B17-B36
#   4. run the standard run report (show_final, check_gap, rr_exact, tw_check, show_bounds)
#   5. redraw the full validation figure set
#   6. re-measure the contact days
#   7. compare with the previous run, when PREVIOUS_CHECKS names its phaseD_checks.json
#   8. draw the todolist923 figures and check the per-case constraints
#
# Usage: scripts/phase_d_chain.sh   (start it inside its own tmux session; see launch_phase_d.sh)
# Environment: PYTHON (default python3), PREVIOUS_CHECKS (optional).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
LOG=phaseD_chain.log
# The firefly names its output directory after its settings, while the report scripts read the
# optimizer output from one fixed name. The run directory is copied onto that name below.
FIREFLY=Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200
FIREFLY_RUN="${FIREFLY_RUN:-Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.03_max_generations_120}"
FIGS=validation_figures_phaseD
SPREAD=synthetic_data_results_spread_Taiwan_weight
FIRST=synthetic_data_results_taiwan_first_outbreak

say() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
run() { "$PYTHON" -m "$@"; }

say 'waiting for the firefly session to finish'
while tmux has-session -t firefly 2>/dev/null; do sleep 120; done
if [ -f "$FIREFLY_RUN/firefly_best.txt" ]; then
    rm -rf "$FIREFLY"
    cp -R "$FIREFLY_RUN" "$FIREFLY"
    say "copied $FIREFLY_RUN to $FIREFLY for the report scripts"
fi
if [ ! -f "$FIREFLY/firefly_best.txt" ]; then
    say "ERROR: $FIREFLY/firefly_best.txt missing, stopping"
    exit 1
fi
say "firefly finished: $(grep -v 'it/s]' firefly_phaseD.log | tail -1 || true)"

# ---------------------------------------------------------------- 2. synthetic data
for mode in spread_Taiwan_weight taiwan_first_outbreak; do
    if [ -d "synthetic_data_results_$mode" ]; then
        mv "synthetic_data_results_$mode" "synthetic_data_results_${mode}_prev_$(date +%Y%m%d_%H%M)"
        say "archived the previous synthetic_data_results_$mode"
    fi
done
say 'generating synthetic data (2 modes x 1000 Monte-Carlo runs)'
bash scripts/data_synthesis.sh >> "$LOG" 2>&1 || { say 'ERROR: data synthesis failed'; exit 1; }
say 'synthetic data done'

# ---------------------------------------------------------------- 2b. age risk ratio
# B14 is accepted on this, not on the 1000-simulation spread output: that held 226
# infections over four age bands on run 6, with health care 60+ resting on a single one,
# and the interval was wider than the acceptance band (E68). 20000 index cases take about
# 30 s and give ~4300 infections. It must run BEFORE verify_phase_d, which reads the JSON.
say 'measuring the age risk ratio on 20000 index cases'
run covsyn.validation.measure_age_rr "$FIREFLY" 20000 >> phaseD_report.txt 2>&1 \
  || say 'WARNING: the age risk ratio measurement failed, B14 falls back to the spread output'

# ---------------------------------------------------------------- 3. checklist
say 'running the Phase D checklist'
run covsyn.validation.verify_phase_d "$SPREAD" "$FIRST" validation_reference/phaseD_checks.json \
    > phaseD_checks.txt 2>&1 || say 'WARNING: the checklist exited with an error'
tail -3 phaseD_checks.txt | tee -a "$LOG"

# ---------------------------------------------------------------- 4. standard run report
say 'running the standard run report'
{
  echo '=== show_final ==='   ; run covsyn.validation.show_final
  echo '=== check_gap ==='    ; run covsyn.validation.check_gap
  echo '=== rr_exact ==='     ; run covsyn.validation.rr_exact "$SPREAD"
  echo '=== tw_check ==='     ; run covsyn.validation.tw_check "$FIRST"
  echo '=== show_bounds ===' ; run covsyn.validation.show_bounds
} > phaseD_report.txt 2>&1 || say 'WARNING: a run-report step failed'
say 'run report written to phaseD_report.txt'

# ---------------------------------------------------------------- 5. figures
mkdir -p "$FIGS"
say "drawing the validation figures into $FIGS"
{
  run covsyn.figures.validate_layers               "$SPREAD" "$FIGS"
  run covsyn.figures.validate_infection            "$SPREAD" "$FIGS"
  run covsyn.figures.plot_diagnostics              "$SPREAD" "$FIGS"
  run covsyn.figures.plot_10panel                  "$SPREAD" "$FIGS"
  run covsyn.figures.plot_cheng_attackrate         "$SPREAD" "$FIGS"
  run covsyn.figures.plot_vs_notebook_literature   "$SPREAD" "$FIGS"
  run covsyn.figures.plot_validation               "$FIREFLY" "$FIGS"
  run covsyn.figures.workplace_sampling_options    "$SPREAD" "$FIGS" "$FIREFLY"
  run covsyn.figures.plot_reality_vs_covsyn        "$SPREAD" "$FIRST" "$FIGS"
  run covsyn.figures.plot_violin_reality_vs_covsyn "$SPREAD" "$FIGS"
  run covsyn.figures.compare_healthcare_municipality "$SPREAD" "$FIGS"
} >> "$LOG" 2>&1 || say 'WARNING: a figure step failed'
say "figures: $(find "$FIGS" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ') files"

# ---------------------------------------------------------------- 6. re-measure contact days
# MEAN_CONTACT_DAYS in sar_anchors.py converts the literature CUMULATIVE attack rates into
# per-day probabilities, and it is a model OUTPUT: it moved from {2.82, 3.17, 3.69, 4.16,
# 1.71} on the second run to {3.71, 4.45, 6.81, 2.23, 2.49} on the third. If it is not
# re-measured after every run the conversion silently stops holding, which is how the
# anchors came to exist in three places with three different sets of numbers (E53).
say 're-measuring the contact days for the next round of anchors'
{
  echo '=== measure_days (update MEAN_CONTACT_DAYS in sar_anchors.py with this) ==='
  run covsyn.validation.measure_days "$SPREAD"
} >> phaseD_report.txt 2>&1 || say 'WARNING: measure_days failed'
tail -8 phaseD_report.txt | tee -a "$LOG"

# ---------------------------------------------------------------- 7. compare with the previous run
if [ -n "${PREVIOUS_CHECKS:-}" ] && [ -f "$PREVIOUS_CHECKS" ]; then
  say "comparing against $PREVIOUS_CHECKS"
  run covsyn.validation.compare_phase_d_runs "$PREVIOUS_CHECKS" validation_reference/phaseD_checks.json \
    > phaseD_previous_vs_current.txt 2>&1 || say 'WARNING: the run comparison failed'
  tail -5 phaseD_previous_vs_current.txt | tee -a "$LOG"
fi

# ---------------------------------------------------------------- 8. todolist923 figures and Table A
# The figures rebuilt to the 2026-09-23 meeting (actual values against intervals, input /
# calibration / independent split) and the per-case constraint check (N2). Both read files the
# steps above wrote, so they must stay after the checklist.
say 'drawing the todolist923 figures and checking the per-case constraints'
run covsyn.figures.plot_todolist923 "$SPREAD" "$FIREFLY_RUN" validation_reference/phaseD_checks.json \
    "$FIGS/todolist923" >> "$LOG" 2>&1 || say 'WARNING: plot_todolist923 failed'
run covsyn.validation.check_constraints "$SPREAD" "$FIRST" --out constraint_check >> "$LOG" 2>&1 \
  || say 'WARNING: check_constraints failed'
tail -4 "$LOG"
say 'ALL DONE'
