# CovSyn Phase D code manifest (2026-09-29)

Prepared before moving the code into the thesis repository. **Nothing has been copied or committed yet.**

## Where the code has to go

`C:\Theisis\covid19-research\src\covsyn` is **not an ordinary folder**. It is a git submodule:

```
[submodule "src/covsyn"]
    path = src/covsyn
    url  = git@github.com:nordlinglab/COVID19-CovSyn.git     (pinned at 15dd3f1)
```

That is the same repository as this working directory (`origin = nordlinglab/COVID19-CovSyn`, HEAD `15dd3f1`).
So the code reaches the thesis repo by committing it here, pushing, and moving the submodule pointer:

1. In this repo: create a branch, commit the files listed below, push.
2. In `covid19-research`: `git -C src/covsyn fetch && git -C src/covsyn checkout <new commit>`, then commit the new
   submodule pointer.

Copying files into `src/covsyn` instead would put untracked files inside a submodule and leave the thesis repo pointing
at the old commit.

## State of the code

- Only **62 files** are tracked in git (the original CovSyn). **All Phase D work is uncommitted**: 11 tracked files are
  modified and about 70 scripts and documents are untracked.
- The **remote Mac holds the authoritative copy** (the runs execute there). Compared file by file:
  63 identical, **1 different**, 3 local-only, **39 remote-only**, including files the pipeline calls
  (`show_final.py`, `check_gap.py`, `show_bounds.py`, `smoke_run7.py`). The local copy is therefore incomplete.
- `Data_synthesis_main.py` differs: the local copy has 5 extra lines that also save
  `{layer}_expected_infections` (via `getattr(..., None)`). The remote copy is the one every Phase D run used.
  **Decision needed** on which one to commit.

## A. Pipeline — used by the current runs (39 files, found by following the entry points and their imports)

Entry points: `launch_run6.sh`, `phaseD_chain.sh`, `data_synthesis.sh`, the gates `verify_fast_cost.py` and
`smoke_run7.py`, and the setup scripts `apply_phaseD_parameters.py`, `sar_anchors.py`, `rebuild_school_pmf.py`.

| Group | Files |
|---|---|
| Model | `Data_synthesize.py`, `Data_synthesis_main.py`, `R0_network.py`, `rw_data_processing.py`, `plot_results.py` (the Cheng binning lives here) |
| Objective and optimizer | `firefly_optimizer.py`, `fast_cost.py`, `cost_parts.py`, `sar_anchors.py` |
| Parameter setup | `apply_phaseD_parameters.py`, `rebuild_school_pmf.py` |
| Run control | `launch_run6.sh`, `phaseD_chain.sh`, `data_synthesis.sh` |
| Gates | `verify_fast_cost.py`, `smoke_run7.py` |
| Acceptance and reports | `verify_phaseD.py`, `measure_age_rr.py`, `measure_days.py`, `rr_exact.py`, `tw_check.py`, `show_final.py`, `show_bounds.py`, `check_gap.py`, `compare_phaseD_runs.py`, `check_constraints.py` |
| Validation figures | `validate_layers.py`, `validate_infection.py`, `compare_healthcare_municipality.py`, `workplace_sampling_options.py`, `plot_10panel.py`, `plot_cheng_attackrate.py`, `plot_diagnostics.py`, `plot_reality_vs_covsyn.py`, `plot_violin_reality_vs_covsyn.py`, `plot_vs_notebook_literature.py`, `plot_validation.py`, `plot_todolist923.py`, `taiwan_reference.py` |

## B. Supporting tools — not in the pipeline, but they produced recorded findings (keep for reproducibility)

| File | Produced / used for |
|---|---|
| `extract_tracing_reference.py`, `extract_coworker_contacts.py` | build `validation_reference/*.csv` from the Taiwan tracing workbooks |
| `extract_literature_provenance.py` (local only) | `validation_reference/literature_provenance.csv` (Table D draft) |
| `probe_community_shape.py`, `probe_medical_shape.py` (remote only), `probe_tail_price.py`, `probe_tail_price_detail.py` | reachability gates (B40, B47, B50) |
| `probe_recovery_tradeoff.py`, `probe_city_ratio.py`, `probe_preonset_window.py` | findings E74, E83, E84 |
| `recover_cost_parts.py`, `test_shared_cost_parts.py` | E71 |
| `calibrate_outcome_weight.py`, `audit_bounds.py`, `rr_population_check.py` | B43, bound audits, E62 |
| `bench_fast_cost.py`, `bench_speed.py`, `bench_cost.py`, `verify_speedup.py`, `firefly_optimizer_baseline.py` | speed-up evidence (E64, E65); the baseline is the pre-speed-up reference copy |
| `plot_progress.py`, `analyze_synthetic.py`, `cost_decomp.py`, `validate_transition_times.py`, `validate_full.py`, `cheng_gate.py` | earlier diagnostics, not cited in the register; keep or drop (**your call**) |

## C. Original CovSyn files that are tracked but not used by Phase D

`parameters_for_initialization.py` (one-off generator, E53), `parameters_for_training.py`, `scoring.py`,
`transition_probability_estimation.py`, `firefly_txt_to_csv.py`, notebooks, `requirements.txt`, `README.md`, `LICENSE`.
Keep them: they are the upstream code the preprint describes.

## D. One-off scratch — recommend NOT committing

Remote only: `bins.py`, `calibrate.py`, `check.py`, `check_gap_dir.py`, `check_pen.py`, `check_rr.py`, `cp.py`,
`daily.py`, `e26.py`, `explore_tw.py`, `fix_show_final.py`, `gaps.py`, `layerday.py`, `mode_cmp.py`, `new.py`, `q678.py`,
`reconcile.py`, `rr_trial.py`, `seed.py`, `smoke.py`, `smoke_run4.py`, `smoke_run6.py`, `verify.py`, `verify2.py`,
`verify_new.py`, `wp.py`, `chain_after_firefly.sh`.
Both sides: `check_age.py`, `debug_hc_bins.py`, `debug_hc_bins2.py`, `patch_firefly_penalty.py`,
`widen_recovery_bounds.py`, `verify_changes.py`, `attack_curve_plot.py`.
Local only: `apply_course_fixes.py`, `plot_layer_sar.py`.
Also: every `*.bak_run*`, `__pycache__/`, `old/`, `old_covsyn/`, `.codex-tools/`, logs.

## E. Documents

| File | Recommendation |
|---|---|
| `covsyn_decisions.md` | commit: it is the record of why the model is what it is |
| `covsyn_flowcharts*.md`, `covsyn_age_weight_litreview*.md`, `covsyn_parameter_audit.md`, other `covsyn_*.md` | commit (review first) |
| `report_run10_todolist923/` | commit or keep outside the code repo (**your call**) |
| **`HANDOVER.md`, `.claude/skills/`** | **do NOT commit as they are**: they contain the remote Mac's IP address, port, user name and SSH key path (`HANDOVER.md` 3 places, `.claude/skills/remote-mac-ssh/SKILL.md` 11, `.claude/skills/covsyn-report/SKILL.md` 1). Commit only a scrubbed copy, or keep them out |
| `CLAUDE.md` | contains no connection details; it is agent instructions for this working copy, so keep it out unless wanted |

## F. Data and results

| Path | Recommendation |
|---|---|
| `variable/` | commit the Phase D parameter files (they are modified tracked files; needed to reproduce run 10) |
| `validation_reference/*.csv` | commit (small, derived from the public tracing workbook) |
| `firefly_result/…/firefly_best.txt` of run 10 | commit the best vector (small) so run 10 can be reproduced |
| `ARCHIVE_*`, `synthetic_data_results_*`, figures, `diagnostic_figures/` | do not commit (large, regenerable) |

Note: the thesis repo's own `src/README.md` (issue 1) wants data out of `src/`. Because `src/covsyn` is the upstream
CovSyn repository, that is a decision about the CovSyn repo layout, not about this move.
