#!/bin/bash
# Phase D: wait for the firefly run to finish, then synthesise the data and produce every
# validation output in one go, so the whole chain can run unattended overnight.
#
#   1. wait for the firefly tmux session to end
#   2. archive the previous synthetic data, then generate spread_Taiwan_weight and
#      taiwan_first_outbreak with 1000 Monte-Carlo runs each (decisions A4, A5, B36)
#   3. run the checklist of every post-simulation item of B14 / B17-B36
#   4. run the standard run report (show_final, check_gap, cheng_gate, rr_exact, tw_check)
#   5. redraw the full validation figure set
#
# Usage: bash phaseD_chain.sh   (start it inside its own tmux session)
set -u
cd ~/COVID19-CovSyn
PYTHON='/Users/rex/covsyn/bin/python'
LOG=phaseD_chain.log
# The firefly names its output directory after the generation count, which changed from 200
# to 120 for the third run, while show_final.py, check_gap.py, validate_layers.py and the
# other report scripts have the old name built in. The run directory is copied onto the old
# name below so none of them has to change.
FIREFLY=Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200
FIREFLY_RUN=Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.03_max_generations_120
FIGS=validation_figures_phaseD

say() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

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
say "firefly finished: $(grep -v 'it/s]' firefly_phaseD.log | tail -1)"

# ---------------------------------------------------------------- 2. synthetic data
for mode in spread_Taiwan_weight taiwan_first_outbreak; do
    if [ -d "synthetic_data_results_$mode" ]; then
        mv "synthetic_data_results_$mode" "synthetic_data_results_${mode}_prev_$(date +%Y%m%d_%H%M)"
        say "archived the previous synthetic_data_results_$mode"
    fi
done
say 'generating synthetic data (2 modes x 1000 Monte-Carlo runs)'
bash data_synthesis.sh >> "$LOG" 2>&1 || { say 'ERROR: data synthesis failed'; exit 1; }
say 'synthetic data done'

# ---------------------------------------------------------------- 3. checklist
# ---------------------------------------------------------------- 2b. age risk ratio
# B14 is accepted on this, not on the 1000-simulation spread output: that held 226
# infections over four age bands on run 6, with health care 60+ resting on a single one,
# and the interval was wider than the acceptance band (E68). 20000 index cases take about
# 30 s and give ~4300 infections. It must run BEFORE verify_phaseD.py, which reads the JSON.
say 'measuring the age risk ratio on 20000 index cases'
if ! "$PYTHON" measure_age_rr.py "$FIREFLY" 20000 >> phaseD_report.txt 2>&1; then say 'WARNING: the age risk ratio measurement failed, B14 falls back to the spread output'; fi

say 'running the Phase D checklist'
"$PYTHON" verify_phaseD.py synthetic_data_results_spread_Taiwan_weight \
    synthetic_data_results_taiwan_first_outbreak validation_reference/phaseD_checks.json \
    > phaseD_checks.txt 2>&1
tail -3 phaseD_checks.txt | tee -a "$LOG"

# ---------------------------------------------------------------- 4. standard run report
say 'running the standard run report'
{
  echo '=== show_final ==='   ; "$PYTHON" show_final.py
  echo '=== check_gap ==='    ; "$PYTHON" check_gap.py
  echo '=== rr_exact ==='     ; "$PYTHON" rr_exact.py synthetic_data_results_spread_Taiwan_weight
  echo '=== tw_check ==='     ; "$PYTHON" tw_check.py synthetic_data_results_taiwan_first_outbreak
  echo '=== show_bounds ===' ; "$PYTHON" show_bounds.py
} > phaseD_report.txt 2>&1
say 'run report written to phaseD_report.txt'

# ---------------------------------------------------------------- 5. figures
mkdir -p "$FIGS"
say "drawing the validation figures into $FIGS"
SPREAD=synthetic_data_results_spread_Taiwan_weight
{
  "$PYTHON" validate_layers.py             "$SPREAD" "$FIGS"
  "$PYTHON" validate_infection.py          "$SPREAD" "$FIGS"
  "$PYTHON" plot_diagnostics.py            "$SPREAD" "$FIGS"
  "$PYTHON" plot_10panel.py                "$SPREAD" "$FIGS"
  "$PYTHON" plot_cheng_attackrate.py       "$SPREAD" "$FIGS"
  "$PYTHON" plot_vs_notebook_literature.py "$SPREAD" "$FIGS"
  "$PYTHON" plot_validation.py             "$FIREFLY" "$FIGS"
  "$PYTHON" workplace_sampling_options.py  "$SPREAD" "$FIGS" "$FIREFLY"
  "$PYTHON" plot_reality_vs_covsyn.py       "$SPREAD" synthetic_data_results_taiwan_first_outbreak "$FIGS"
  "$PYTHON" plot_violin_reality_vs_covsyn.py "$SPREAD" "$FIGS"
  "$PYTHON" compare_healthcare_municipality.py "$SPREAD" "$FIGS"
} >> "$LOG" 2>&1
say "figures: $(ls "$FIGS"/*.png 2>/dev/null | wc -l | tr -d ' ') files"

# ---------------------------------------------------------------- 6. re-measure contact days
# MEAN_CONTACT_DAYS in sar_anchors.py converts the literature CUMULATIVE attack rates into
# per-day probabilities, and it is a model OUTPUT: it moved from {2.82, 3.17, 3.69, 4.16,
# 1.71} on the second run to {3.71, 4.45, 6.81, 2.23, 2.49} on the third. If it is not
# re-measured after every run the conversion silently stops holding, which is how the
# anchors came to exist in three places with three different sets of numbers (E53).
say 're-measuring the contact days for the next round of anchors'
{
  echo '=== measure_days (update MEAN_CONTACT_DAYS in sar_anchors.py with this) ==='
  "$PYTHON" measure_days.py "$SPREAD"
} >> phaseD_report.txt 2>&1
tail -8 phaseD_report.txt | tee -a "$LOG"

# ---------------------------------------------------------------- 7. compare with run 3
if [ -d ARCHIVE_20260928_run_phaseD9 ]; then
  say 'comparing against run 9'
  "$PYTHON" compare_phaseD_runs.py ARCHIVE_20260928_run_phaseD9/phaseD_checks.json       validation_reference/phaseD_checks.json > phaseD_run9_vs_run10.txt 2>&1
  tail -5 phaseD_run9_vs_run10.txt | tee -a "$LOG"
fi
# ---------------------------------------------------------------- 8. todolist923 figures and Table A
# The figures rebuilt to the 2026-09-23 meeting (actual values against intervals, input /
# calibration / independent split) and the per-case constraint check (N2). Both read files the
# steps above wrote, so they must stay after the checklist.
say 'drawing the todolist923 figures and checking the per-case constraints'
"$PYTHON" plot_todolist923.py "$SPREAD" "$FIREFLY_RUN" validation_reference/phaseD_checks.json "$FIGS/todolist923" >> "$LOG" 2>&1 \
  || say 'WARNING: plot_todolist923.py failed'
"$PYTHON" check_constraints.py "$SPREAD" synthetic_data_results_taiwan_first_outbreak --out constraint_check >> "$LOG" 2>&1 \
  || say 'WARNING: check_constraints.py failed'
tail -4 "$LOG"
say 'ALL DONE'
