# Phase D run 18, selected candidate 80

- `firefly_best.txt`: the selected vector of run 18 (candidate 80 of the revalidation, B56), 208 parameters, the same file as run 18's `candidates/candidate_80/firefly_best.txt`.
- `measure_days_b75.txt`: `python -m covsyn.validation.measure_days` on 400 `spread_Taiwan_weight` simulations of this vector at commit `890da02` (B75 work calendar), the source of `sar_anchors.MEAN_CONTACT_DAYS` for run 19.
- `checks_b76.txt`: the same checklist at commit `a35d951` with 200 bootstrap replicates (`verify_phase_d b75_mc1000 none b76_checks.json 200`), the sigma of B76's charged floor in `calibration/charged_targets.json`.
- `checks_b75.txt`: `python -m covsyn.validation.verify_phase_d` on 1000 `spread_Taiwan_weight` simulations of this vector at commit `9136f59` (bootstrap off), the source of `sar_anchors.COVSYN_CANDIDATE_CONTACTS` for run 19.

Reproduce from the repository root:

```bash
mkdir -p b75_spread_Taiwan_weight && \
  python -m covsyn.model.data_synthesis_main --mode spread_Taiwan_weight \
    --monte_carlo_number 400 --result_path b75_spread_Taiwan_weight \
    --cpu_cores 24 --parameter_path firefly_result/phaseD_run18 && \
  python -m covsyn.validation.measure_days b75_spread_Taiwan_weight

mkdir -p b75_mc1000 && \
  python -m covsyn.model.data_synthesis_main --mode spread_Taiwan_weight \
    --monte_carlo_number 1000 --result_path b75_mc1000 \
    --cpu_cores 24 --parameter_path firefly_result/phaseD_run18 && \
  python -m covsyn.validation.verify_phase_d b75_mc1000 none b75_checks.json 0
```
