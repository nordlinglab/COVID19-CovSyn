# COVID19-CovSyn

Repository for the code of the COVID-19 Data Synthesis project

## Overview

CovSyn, an innovative tool that generates comprehensive synthetic data at the individual level, encompassing demographic characteristics (age, gender, occupation), course of disease (infection dates, symptom onset, recovery dates), and contact tracing information (daily social interactions including household, school, workplace, healthcare, and municipality).

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone git@github.com:nordlinglab/COVID19-CovSyn.git
cd COVID19-CovSyn
uv sync
uv run pytest
```

`uv sync` installs Python 3.13, the dependencies pinned in `uv.lock`, and `covsyn` in editable mode, on macOS and Linux.
`uv run pytest` includes the slow regression tests, which take several minutes;
see [Tests](#tests).

## Repository layout

The code is a Python package under `src/covsyn/`.
Every command below runs from the repository root, because the modules read their inputs through paths relative to it (`./variable/`, `./data/`).

| Path | Contents |
|---|---|
| `src/covsyn/model/` | The simulation: `data_synthesize.py` (demographics, social contexts, course of disease, contacts), `data_synthesis_main.py` (one simulation and the Monte Carlo driver), `r0_network.py`, `contact_measures.py`, `school_calendar.py`, `work_calendar.py` |
| `src/covsyn/calibration/` | Fitting: `firefly_optimizer.py` (optimizer and reference objective), `fast_cost.py` (the objective the optimizer runs), `cost_parts.py`, `sar_anchors.py` (attack-rate anchors), `target_margins.py`, `apply_phase_d_parameters.py` (search bounds and seed vector), `rebuild_school_pmf.py`; `parameter_space.py`, `initial_population.py` and `seed_streams.py` (Task 7), `step_schedule.py` (Task 8), `run_size.py`, `monitoring.py`, `block_scoring.py`, `pilot_noise.py` and `pilot_size.py` (Task 9), `revalidation.py`, `select_by_checklist.py` and `parameter_uncertainty.py` (Task 10) |
| `src/covsyn/data_processing/` | Model inputs and validation references: `rw_data_processing.py`, `taiwan_reference.py`, the `extract_*.py` scripts that build `validation_reference/` |
| `src/covsyn/validation/` | Acceptance checklist and run reports: `verify_phase_d.py`, `acceptance.py`, `check_constraints.py` (per-case constraint table), `measure_age_rr.py`, `measure_days.py`, `rr_exact.py`, `tw_check.py`, `show_final.py`, `show_bounds.py`, `check_gap.py`, `compare_phase_d_runs.py`, `city_effect.py`, `compare_first_wave.py` |
| `src/covsyn/figures/` | Validation figures, including `plot_todolist923.py` |
| `src/covsyn/upstream/` | Upstream code the preprint describes, not used by the Phase D pipeline |
| `src/covsyn/pipeline/` | `covsyn-pipeline`: the Phase D stages in order, with resume, and `default.toml` |
| `src/covsyn/runtime/` | Worker count (`hardware.py`), run names and directories (`runs.py`), and timings, memory and profiling |
| `scripts/` | `data_synthesis.sh`; `gates/` (checks before a run), `probes/` and `benchmarks/` (the tools behind findings in the decision register) |
| `tests/` | Regression, pipeline and unit tests; fixtures in `tests/fixtures/` |
| `variable/`, `data/`, `validation_reference/` | Fitted parameters, raw inputs, and reference distributions |
| `results/models/<run-id>/`, `results/tables/<run-id>/`, `results/figures/<run-id>/` | Each run's parameters and logs, tables, and figures; only the runs `.gitignore` names are committed, such as Phase D run 10 and pilot 1's summaries |
| `docs/` | `covsyn_decisions.md` (every modelling decision and finding), `CODE_MANIFEST_phaseD.md` (what each file does) and Phase D notes |
| `notebooks/` | Analysis notebooks |

## Usage

Modules run as `uv run python -m covsyn.<package>.<module>`, and the optimiser and the modules with a `main` function also as entry points, such as `uv run covsyn-optimise`;
`pyproject.toml` lists them under `[project.scripts]`.
The shell scripts run every module through `uv run`.

Every module that starts worker processes uses the performance cores, which `uv run covsyn-cores` prints with how it found them;
the checklist also starts no more workers than fit in 80% of the available memory.
`--workers N` or the environment variable `COVSYN_WORKERS` overrides that count, in that order.

### 1. Search bounds and seed vector

```bash
uv run python -m covsyn.calibration.apply_phase_d_parameters
```

### 2. Checks before an optimisation run

```bash
uv run pytest
uv run python scripts/gates/verify_fast_cost.py results/models/phaseD_run10 --workers 8 --vectors 4
uv run python scripts/gates/smoke_test.py
```

The first reproduces Phase D run 10 exactly;
the second checks that the fast objective equals the reference one, and that one worker task per candidate, as the optimiser runs it, equals the fast objective;
the third evaluates the seed vector once.

### 3. Parameter optimisation

```bash
nohup uv run covsyn-pipeline --config default \
    > pipeline.out 2>&1 &
```

On macOS, prefix the command with `caffeinate -is` so the machine does not sleep.
`nohup` keeps the run going after the terminal or SSH session closes;
`tmux` and `screen` remain optional.

The pipeline runs the stages in order: optimise, revalidate, select, synthesise, check, report, figures.
The optimiser uses the synchronous firefly algorithm (Yang, 2010).
In each generation every firefly moves towards each brighter one from the positions and costs the generation starts with, and then all are scored at once, one worker task per firefly, on the generation's own block of seeds.
An elite of the k = min(5, n − 1) lowest-cost fireflies does not move and is re-scored on every block, and in generation 0, which shares the initial population's block, it keeps its initial costs.
`firefly_archive.txt` holds every vector that was in the elite, with its mean cost over the blocks it was scored on there and their number, and revalidation re-scores them with the final population.
The step size shrinks geometrically from α_0 in generation 0 to α_end in generation G_plan − 1 and stays there: `--alpha-0` (default 0.2), `--alpha-end` (default 0.001) and `--g-plan` (default the mode's generations), recorded in `config.json`, until pilot 2 chooses them;
the pipeline takes them as `alpha_0`, `alpha_end` and `g_plan` in `[optimise]`.
A run started before this schedule (Task 8) cannot be resumed.
The workers start by spawn on macOS and Linux, and the results do not depend on the number of workers.
Every run starts from the committed initial population `variable/initial_population_v1.npz`, taking its first n rows: the literature vector, the best vectors of runs 10 and 19, then a Latin hypercube.
A run whose bounds or dimension differ from the population's is refused;
`uv run covsyn-initial-population --version 2` then makes a new version, which is how the warm starts change too.
The moves are drawn from `rng_seed` (`--seed`, default 1), so two runs of one configuration are identical.
Each use takes its simulation seeds from its own stream: block b of stream s holds the seeds s × 10^7 + b × S to s × 10^7 + (b + 1) × S − 1.
The optimiser takes streams 1 to 99 (`--stream`, default 1), revalidation 200 to 299 (default 200), the synthesis the candidate is chosen on stream 301, and the reported synthesis stream 300.
The age risk ratio is measured on each checklist's stream, 301 for `checks.json` and 300 for `checks_independent.json`, into `age_rr.json` and `age_rr_independent.json`.
Stream 0 holds the seeds of earlier runs, which the regression tests re-make;
a run draws from its own streams instead, so the optimiser's, revalidation's and 301 may choose another vector, and its checklists, on 301 and 300, differ from ones made before these streams partly because their seeds differ.
A run writes into `results/models/<run-id>/`, named `<YYYYMMDD-HHMMSS>_<commit7>[-dirty]_<config-name>`, with `config.json` recording the optimiser's settings, stream, initial population, `rng_seed`, workers, machine and versions, and `pipeline.toml` the pipeline's;
its tables go into `results/tables/<run-id>/` and its figures into `results/figures/<run-id>/`.
Each finished stage adds a line to `run.log`, and the optimiser saves a checkpoint after every generation, so

```bash
nohup uv run covsyn-pipeline --resume <run-id> > pipeline.out 2>&1 &
```

continues an interrupted run where it stopped.
`--resume` takes the run id, or `results/models/<run-id>` written out as a whole path;
anything else, such as `.`, a subdirectory of the run or the place the run is stored, is refused before anything is written, because its tables and figures could then go elsewhere.
A run directory that is a link to other storage is not supported.
`src/covsyn/pipeline/default.toml` holds the settings, such as the optimisation and validation streams and the previous run's `checks.json` to compare with;
`--config` also takes another TOML file.

Two pilots size the production run (Task 9), and each writes its command, settings and results into `results/tables/<run-id>/`:

```bash
uv run covsyn-pilot-noise
uv run covsyn-pilot-size --noise results/tables/<pilot-1 run-id>
```

Each resumes with `--resume results/tables/<run-id>`;
pilot 2's must be its own directory, neither a link nor renamed, since its runs' directories are named after it.
A resume runs with `--workers`, else `COVSYN_WORKERS`, else the worker count in its `config.json`, and writes the count it runs with there;
a `config.json` that is read-only, a symbolic link, hard-linked, another user's, in a read-only directory or in a group the user cannot keep stops a resume with another count.

Pilot 1 scores the literature vector, the run-19 vector and two random rows of the initial population on 30 blocks of validation stream 201 at each S of 300, 1,000, 3,000 and 10,000, and every row on two blocks at S = 10,000 for the cost spread across the population.
It chooses the smallest S at which the largest SD of a paired difference between the vectors is at most 10% of that spread, and none when no S passes.
Pilot 2 runs `--mode pilot` for n of 24, 48 and 96 and α_0 of 0.05 and 0.2, each on optimisation streams 1 and 2, for G = ⌊(10,000 − n) / n⌋ generations with G_plan = G, at pilot 1's S.
It scores each run's 10 best distinct archive vectors on blocks 0 to 29 of validation stream 202, takes the lowest as the run's representative, and scores the representatives on blocks 30 to 59.
It chooses the setting with the lowest mean cost whose two runs agree, the mean of their block-by-block differences being at most its SD / √30, and none when no setting's runs agree.
The production run, `--mode production` or `mode = "production"` in `[optimise]`, takes pilot 2's n (`pop_size`), S (`simulations`), α_0, α_end, G_plan and run id (`pilot`), all required.
It scores each new elite vector on blocks 0 to 4 of monitoring stream 100, and stops when the best monitored vector has improved by no more than one paired standard error over 5 generations, or at G_max = 1,000 generations;
the reason goes into `run.log`, and the monitoring's cost into `monitoring.json` beside `progress.csv`.
Its revalidation re-scores the archive's 10 best distinct vectors, by ascending mean, at its S on 30 blocks of its validation stream (Task 10), and refuses another number of blocks.
The best is the lowest mean;
`cost_uncertainty.csv` gives each candidate's mean ± SE and its paired difference to the best with a 95% t interval on 29 degrees of freedom, exploratory since the best is chosen on the same blocks, and marks the candidates whose interval includes 0 as indistinguishable from the best.
The selection uses the same scores.

Each `run.log` line also records the stage's peak memory: that of its steps' own processes and of their largest worker, the steps' resident memory the pipeline samples once a second, and the least available memory.
The pipeline warns as soon as available memory falls below 10% of the total.
A stage is refused when the latest earlier run with the same `pipeline.toml` that measured the stage whole shows it needs more than the available memory: its steps' peak plus the workers times the peak per worker.
The steps' peak is a conservative addition to the plan's formula, and stopped attempts are not used, so a stage once stopped for want of memory may run out again;
without such an earlier run, the pipeline says the need cannot be estimated.
This is a conservative check at the stage's start, not a guarantee against running out of memory.
A forked worker's resident memory, on Linux, also counts the pages it shares with its parent, so a stage that would fit may be refused.
The optimiser writes `timings_evaluation.csv`, one row per evaluation, and `timings_generation.csv`, one row per generation, beside `progress.csv`;
`--stage-timings` adds the time `run_covid` spends in each stage, and a resumed run keeps the setting it started with.

### Profiling

```bash
uv run covsyn-optimise --mode profile --workers 2 --simulations 3000
```

Each worker profiles its own tasks into `results/models/<run-id>/profile/worker_<pid>.prof`;
the run merges them into `merged.prof` there and lists the 40 functions with the largest cumulative time in `results/tables/<run-id>/profile_top.csv`.
A profile has as many fireflies as workers and 2 generations;
`--simulations` sets the simulations per evaluation (default 3,000).

### 4. Data synthesis

```bash
PARAMETER_PATH=results/models/phaseD_run10 scripts/data_synthesis.sh
```

## Tests

`tests/test_regression.py` and `tests/test_pipeline.py` require every commit to reproduce the output recorded for the run 10 parameter vector: the objective and its cost parts, simulation output for fixed seeds, and every step of the post-optimisation pipeline.
Simulation output, data files and printed reports must match exactly;
the objective and the numbers in the checklist JSON match to a relative 1e-6, the normal precision difference between platforms (B91).
The other tests check the rules in `docs/covsyn_decisions.md`.
The slow tests are marked `slow`;
`-m "not slow"` skips them.
Their fixtures were made on macOS (arm64), and Linux x86-64 passes them: the objective agrees to a relative 4e-9 and the checklist's dispersion fit to 1.5e-8.

## Data Structure

### Agent Properties

#### 1. Demographic Data
- `age`: Agent's age
- `gender`: Agent's gender
- `job`: Agent's occupation

#### 2. Social Data
- `municipality`: City/county of residence
- `household_size`: Number of household members (excluding agent)
- `school_class_size`: Size of school class if applicable
- `work_group_size`: Size of work group if applicable
- `clinic_size`: Healthcare facility capacity

#### 3. Disease Progression Data
- `infection_day`: Time of infection (day 0 = simulation start)
- `latent_period`: Days until infectious
- `incubation_period`: Days until symptom onset
- `infectious_period`: Duration of infectiousness
- `monitor_isolation_period`: Days until monitored isolation
- `date_of_critically_ill`: Time of critical illness
- `date_of_death`: Time of death (if applicable)
- `date_of_recovery`: Time of recovery
- `natural_immunity_status`: Natural immunity status post-recovery
- `negative_test_date`: Timestamps of negative test results
- `negative_test_status`: Test result validity indicators
- `positive_test_date`: Time of confirmed positive test

#### 4. Contact Data
Each contact layer (household, workplace, school, healthcare, municipality) includes:
- `contacts_matrix`: Contact history matrix (contacts × monitoring period)
- `effective_contacts`: Boolean vector of infection-causing contacts
- `effective_contacts_infection_time`: Infection timestamps
- `secondary_contact_ages`: Ages of contacted individuals
- `previously_infected_index_list`: Previous infection records of contacts

### 5. Contact Network Structure
The case edge list contains:
- Source case index
- Target case index
- Infection timestamp
- Contact type/infection setting

### 6. Firefly results
### 6. Firefly results
The firefly result contains 198 parameters in the following order:

1. Contact behavior parameters (index 0 to 34):
   These parameters define contact patterns across five social settings (Household, School, Workgroup, Health care, and Municipality), with 7 parameters per setting:
   - Probability of contact
   - Consecutive daily contact probability
   - Contact probability when healthy
   - Contact probability when symptomatic
   - Steepness of logistic contact probability function
   - Phase relative to symptom onset for symptomatic
   - Phase relative to symptom onset for resuming normal social context
2. Overdispersion rate and overdispersion weight (index 35 and 36).
3. Latent period gamma distribution parameters [shape, scale] (index 37 to 38).
4. Infectious period gamma distribution parameters [shape, scale] (index 39 to 40).
5. Incubation period gamma distribution parameters [shape, scale] (index 41 to 42).
6. Period from symptom onset to monitored isolation gamma distribution parameters [shape, scale, location] (index 43 to 45).
7. Period from asymptomatic to recovered gamma distribution parameters [shape, scale, location] (index 46 to 48).
8. Period from symptom onset to critically ill gamma distribution parameters [shape, scale, location] (index 49 to 51).
9. Period from symptom to recovered gamma distribution parameters [shape, scale, location] (index 52 to 54).
10. Period from critically ill to recovered gamma distribution parameters [shape, scale, location] (index 55 to 57).
11. Period from asymptomatic to death gamma distribution parameters [shape, scale] (index 58 to 59).
12. Period from negative COVID-19 test to confirmed gamma distribution parameters [shape, scale, location] (index 60 to 62).
13. Age-related risk ratio vector of the secondary attack rate (4 values repeated across age groups: 0-19, 20-39, 40-59, and 60 above) (index 63 to 66).
14. Natural immunity rate (index 67).
15. Vaccination rate (index 68).
16. Vaccine efficacy (index 69).
17. Daily secondary attack rate vector:
    - Daily secondary attack rate vector for household layer (25 parameters, index 70 to 94)
    - Daily secondary attack rate vector for school layer (25 parameters, index 95 to 119)
    - Daily secondary attack rate vector for workplace layer (25 parameters, index 120 to 144)
    - Daily secondary attack rate vector for health care layer (25 parameters, index 145 to 169)
    - Daily secondary attack rate vector for municipality layer (25 parameters, index 170 to 194)
18. Transition probabilities:
    - Asymptomatic to recovered transition probability (index 195)
    - Symptom onset to recovered transition probability (index 196)
    - Critically ill to recovered transition probability (index 197)
