# Phase D run 19, selected candidate 60

- `firefly_best.txt`: the selected vector of run 19 (candidate 60 of the revalidation, B56: 0 failed checks, validation cost 9.6509), 208 parameters, the same file as run 19's `selected/firefly_best.txt` on the compute Mac (two identical rows).
- `measure_days_b77.txt`: `python -m covsyn.validation.measure_days` on 400 `spread_Taiwan_weight` simulations of this vector under the B77 school calendar, the source of `sar_anchors.MEAN_CONTACT_DAYS` for run 20.
- `checks_b77.txt`: `python -m covsyn.validation.verify_phase_d` on 1000 `spread_Taiwan_weight` and 1000 `taiwan_first_outbreak` simulations of this vector under B77 (200 bootstrap replicates), the source of `sar_anchors.COVSYN_CANDIDATE_CONTACTS` for run 20; 54 checks passed (5 ok~), 0 failed.

The B77 code was the uncommitted working tree of 2026-11-02 on top of `c23b99c` (B74 recalibrated, B77 school calendar, B78 options off). Reproduce from the repository root once it is committed:

```bash
for mode in spread_Taiwan_weight taiwan_first_outbreak; do
  mkdir -p b77_$mode && \
    python -m covsyn.model.data_synthesis_main --mode $mode --monte_carlo_number 1000 \
      --result_path b77_$mode --cpu_cores 32 --parameter_path firefly_result/phaseD_run19
done
python -m covsyn.validation.measure_days b77_spread_Taiwan_weight
python -m covsyn.validation.verify_phase_d b77_spread_Taiwan_weight b77_taiwan_first_outbreak \
  validation_reference/phaseD_checks_b77.json
```
