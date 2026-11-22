# CovSyn code manifest

What each part of the repository does, as of 2026-11-22 (rework plan Tasks 1–9 merged, Task 10 in progress).
`README.md` explains installation and usage;
`docs/covsyn_decisions.md` records why each part is as it is.

The first version (2026-09-29) listed the uncommitted Phase D files before they entered git (decision B53).
That move is done, so this version describes the committed layout instead;
`git log -p -- docs/CODE_MANIFEST_phaseD.md` shows the earlier versions.

## Entry points

Each runs as `uv run <command>` from the repository root, as `pyproject.toml` `[project.scripts]` lists them.

| Command | Module | Purpose |
|---|---|---|
| `covsyn-pipeline` | `pipeline/runner.py` | Run the Phase D stages in order, and resume an interrupted run (Task 4) |
| `covsyn-optimise` | `calibration/firefly_optimizer.py` | Firefly optimisation in the `train`, `test`, `profile`, `pilot` and `production` modes |
| `covsyn-revalidation` | `calibration/revalidation.py` | Re-score a run's candidates on unseen seeds; in production, the archive's best 10 at S, with `cost_uncertainty.csv` (Task 10) |
| `covsyn-select-by-checklist` | `calibration/select_by_checklist.py` | Choose among the revalidated candidates by the acceptance checklist (B60) |
| `covsyn-initial-population` | `calibration/initial_population.py` | Build the fixed 256-row initial population (Task 7) |
| `covsyn-pilot-noise` | `calibration/pilot_noise.py` | Pilot 1: the objective's noise against S (Task 9) |
| `covsyn-pilot-size` | `calibration/pilot_size.py` | Pilot 2: population and step size at pilot 1's S (Task 9) |
| `covsyn-cores` | `runtime/hardware.py` | Print the performance cores and the worker count (Task 2) |
| `covsyn-data-synthesis` | `model/data_synthesis_main.py` | Synthesise data from a parameter vector |
| `covsyn-verify-phase-d` | `validation/verify_phase_d.py` | The acceptance checklist (B14, B17–B36) |
| `covsyn-measure-age-rr` | `validation/measure_age_rr.py` | Pooled age risk ratio (B14) |
| `covsyn-measure-days` | `validation/measure_days.py` | Days a candidate contact is met, per layer |
| `covsyn-check-constraints` | `validation/check_constraints.py` | Per-case temporal and biological constraints (N2, E48) |
| `covsyn-show-bounds` | `validation/show_bounds.py` | Layer attack-rate bounds |
| `covsyn-compare-phase-d-runs` | `validation/compare_phase_d_runs.py` | Two verification runs, check by check |
| `covsyn-plot-todolist923` | `figures/plot_todolist923.py` | Validation figures of the 2026-09-23 meeting |
| `covsyn-reproduce-wu2025` | `figures/reproduce_wu2025.py` | The CovSyn paper's data comparisons with the corrected model |

Module paths are relative to `src/covsyn/`.
A module without an entry point runs as `uv run python -m covsyn.<package>.<module>`.

## Package `src/covsyn/`

| Package | Modules |
|---|---|
| `model/` | `data_synthesize.py` (demographics, social contexts, course of disease, contacts), `data_synthesis_main.py` (one simulation and the Monte Carlo driver), `r0_network.py`, `contact_measures.py`, `school_calendar.py` (B77), `work_calendar.py` (B75) |
| `calibration/`, objective | `fast_cost.py` (the objective the optimiser runs), `cost_parts.py`, `sar_anchors.py`, `target_margins.py` (B60) |
| `calibration/`, search | `firefly_optimizer.py`, `parameter_space.py` (bounds and literature vector, Task 7), `initial_population.py` (Task 7), `seed_streams.py` (Task 7), `step_schedule.py` (Task 8) |
| `calibration/`, run size and stopping | `run_size.py` (n, G, S, Task 9), `monitoring.py` (production stopping rule, Task 9), `block_scoring.py` (resumable block scores, Task 9), `pilot_noise.py`, `pilot_size.py` |
| `calibration/`, after the run | `revalidation.py` (E87; cost uncertainty, Task 10), `select_by_checklist.py`, `parameter_uncertainty.py` (spread across the five production runs, Task 10; no command calls it yet) |
| `calibration/`, parameter setup | `apply_phase_d_parameters.py`, `rebuild_school_pmf.py` (E39) |
| `pipeline/` | `config.py` (TOML settings), `stages.py`, `runner.py`, `default.toml` (Task 4) |
| `runtime/` | `hardware.py` (Task 2), `runs.py` (run names and directories, Task 3), `measure.py`, `memory.py`, `profiling.py`, `timings.py` (Task 5) |
| `validation/` | `verify_phase_d.py`, `acceptance.py` (B57, E89), `check_constraints.py`, `measure_age_rr.py`, `measure_days.py`, `rr_exact.py`, `tw_check.py`, `show_final.py`, `show_bounds.py`, `check_gap.py`, `compare_phase_d_runs.py`, `city_effect.py` (B56), `compare_first_wave.py` (B66) |
| `figures/` | `plot_results.py` (holds the Cheng binning the objective uses), `plot_todolist923.py`, `reproduce_wu2025.py`, `validate_layers.py`, `validate_infection.py`, `compare_healthcare_municipality.py`, `workplace_sampling_options.py` (E3), `plot_10panel.py`, `plot_cheng_attackrate.py`, `plot_diagnostics.py`, `plot_reality_vs_covsyn.py`, `plot_violin_reality_vs_covsyn.py`, `plot_vs_notebook_literature.py`, `plot_validation.py` |
| `data_processing/` | `rw_data_processing.py`, `taiwan_reference.py`, `build_first_wave_seeds.py` (B66), `recompute_taiwan_targets.py`, and the `extract_*.py` scripts that build `validation_reference/` |
| `upstream/` | The preprint's code, not used by the pipeline: `parameters_for_initialization.py` (E53), `parameters_for_training.py`, `scoring.py`, `transition_probability_estimation.py`, `firefly_txt_to_csv.py` |

## Scripts `scripts/`

| Path | Purpose |
|---|---|
| `data_synthesis.sh` | Synthesise the data sets of a finished run |
| `gates/` | Checks before a run: `smoke_test.py`, `verify_fast_cost.py` (`fast_cost` equals the reference objective) |
| `probes/` | One-question tools behind findings in the register, such as reachability gates (B40, B47, B50, B62, B76) and trade-offs (B54, B58, E74) |
| `benchmarks/` | Speed-up evidence (E64, E65); `firefly_optimizer_baseline.py` is the pre-speed-up reference |
| `calibrate_*.py` | Calibrations of the first-wave seeding (B74, B78, B79) |

## Tests `tests/`

Run with `uv run pytest`.

| Files | Cover |
|---|---|
| `test_regression.py`, `test_pipeline.py`, `test_pipeline_tolerance.py` | Run 10's cost, simulation fingerprints and pipeline outputs, against `tests/fixtures/` |
| `test_task<N>_*.py` | Rework plan Tasks 2–10 |
| `test_b<NN>_*.py` and the rest | One decision or finding each |
| `make_*_fixtures.py`, `firefly_standins.py`, `pipeline_runner.py`, `regression_digest.py`, `conftest.py` | Fixture builders and shared helpers |

## Data and results

| Path | Contents |
|---|---|
| `variable/` | Model parameters, search bounds, the literature vector and `initial_population_v1.npz` |
| `data/` | Raw demographic and course-of-disease inputs |
| `validation_reference/` | Reference distributions derived from the public Taiwan tracing data |
| `results/models/<run-id>/` | A run's parameters, populations, logs and `config.json` |
| `results/tables/<run-id>/`, `results/figures/<run-id>/` | Its tables and figures |

A run id is `<YYYYMMDD-HHMMSS>_<commit7>[-dirty]_<config-name>` (`runtime/runs.py`).
Runs are not committed, except those `.gitignore` names: Phase D runs 10, 18 and 19, `legacy_gamma0131_g200`, and pilot 1's summary tables in `results/tables/20261122-075442_054e9e0_pilot-noise/`.

## Kept out of git

- `HANDOVER.md` and `.claude/`: they hold the remote host's address and key path (B53).
- `CLAUDE.md`: agent instructions for one working copy.
- `ARCHIVE_*/`, `synthetic_data_results_*/`, `logs/`, and every other run output: large and regenerable.
