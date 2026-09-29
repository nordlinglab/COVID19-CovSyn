# COVID19-CovSyn

Repository for the code of the COVID-19 Data Synthesis project

## Overview

CovSyn, an innovative tool that generates comprehensive synthetic data at the individual level, encompassing demographic characteristics (age, gender, occupation), course of disease (infection dates, symptom onset, recovery dates), and contact tracing information (daily social interactions including household, school, workplace, healthcare, and municipality).

To run our code, we recommend you follow the following steps:

## Installation

1. Clone the repository
2. Install Miniconda
3. Create and activate the conda environment:
   ```bash
   conda create -n "covsyn" python=3.10.16
   conda activate covsyn
   ```
4. Navigate to the project directory and install dependencies:
   ```bash
   cd to CovSyn folder
   pip install -r requirements.txt --use-deprecated=legacy-resolver
   ```
    

## Repository layout

The code is a Python package under `src/covsyn/`.
Every command below runs from the repository root, because the modules read their inputs through paths relative to it (`./variable/`, `./data/`).

| Path | Contents |
|---|---|
| `src/covsyn/model/` | The simulation: `data_synthesize.py` (demographics, social contexts, course of disease, contacts), `data_synthesis_main.py` (one simulation and the Monte Carlo driver), `r0_network.py` |
| `src/covsyn/calibration/` | Fitting: `firefly_optimizer.py` (optimizer and reference objective), `fast_cost.py` (the objective the optimizer runs), `cost_parts.py`, `sar_anchors.py` (attack-rate anchors), `apply_phase_d_parameters.py` (search bounds and seed vector), `rebuild_school_pmf.py` |
| `src/covsyn/data_processing/` | Model inputs and validation references: `rw_data_processing.py`, `taiwan_reference.py`, the `extract_*.py` scripts that build `validation_reference/` |
| `src/covsyn/validation/` | Acceptance checklist and run reports: `verify_phase_d.py`, `check_constraints.py` (per-case constraint table), `measure_age_rr.py`, `measure_days.py`, `rr_exact.py`, `tw_check.py`, `show_final.py`, `show_bounds.py`, `check_gap.py`, `compare_phase_d_runs.py` |
| `src/covsyn/figures/` | Validation figures, including `plot_todolist923.py` |
| `src/covsyn/upstream/` | Upstream code the preprint describes, not used by the Phase D pipeline |
| `scripts/` | `launch_phase_d.sh`, `phase_d_chain.sh`, `data_synthesis.sh`; `gates/` (checks before a run), `probes/` and `benchmarks/` (the tools behind findings in the decision register) |
| `tests/` | Regression, pipeline and unit tests; fixtures in `tests/fixtures/` |
| `variable/`, `data/`, `validation_reference/` | Fitted parameters, raw inputs, and reference distributions |
| `firefly_result/phaseD_run10/` | Best vector, bounds and progress of Phase D run 10 |
| `docs/` | `covsyn_decisions.md` (every modelling decision and finding) and Phase D notes |
| `notebooks/` | Analysis notebooks |

## Usage

Put `src/` on the Python path, or install the package in editable mode:

```bash
export PYTHONPATH="$PWD/src"
```

Modules run as `python -m covsyn.<package>.<module>`.
The shell scripts set `PYTHONPATH` themselves and take the interpreter from `PYTHON` (default `python3`).

### 1. Search bounds and seed vector

```bash
python -m covsyn.calibration.apply_phase_d_parameters
```

### 2. Checks before an optimisation run

```bash
python -m pytest tests/
python scripts/gates/verify_fast_cost.py --workers 8 --vectors 4
python scripts/gates/smoke_test.py
```

The first reproduces Phase D run 10 exactly; the second checks that the fast objective equals the reference one; the third evaluates the seed vector once.

### 3. Parameter optimisation

```bash
PYTHON=python3 scripts/launch_phase_d.sh
```

This starts the Firefly optimizer and then `scripts/phase_d_chain.sh` in two tmux sessions.
`WARM_START` names the `firefly_best.txt` files that seed the initial population (default: run 10), and `PREVIOUS_CHECKS` the previous run's `phaseD_checks.json` for the comparison step.

### 4. Data synthesis

```bash
PARAMETER_PATH=firefly_result/phaseD_run10 scripts/data_synthesis.sh
```

## Tests

`tests/test_regression.py` and `tests/test_pipeline.py` require every commit to reproduce the output of the commit that produced Phase D run 10 (`799fb15`): the objective and its 40 cost parts, simulation output for fixed seeds, and every step of the post-optimisation pipeline.
The other tests check the rules in `docs/covsyn_decisions.md`.
The slow tests are marked `slow`; `-m "not slow"` skips them.

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
