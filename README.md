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
   conda create -n "covsyn" python=3.10.11
   conda activate covsyn
   ```
4. Navigate to the project directory and install dependencies:
   ```bash
   cd CovSyn
   pip install -r requirements.txt
   ```
    

## Usage

The simulation pipeline consists of three main stages:

### 1. Initial Parameter Setting

Run `All_parameters_for_data_synthesis.ipynb` to generate the required initial parameters and search boundaries for the Taiwan main island simulation.

### 2. Parameter Optimization (Optional)

Run the firefly optimization algorithm to find optimal simulation parameters:
```bash
python firefly_optimizer.py
```
This will generate optimized parameter files in `firefly_result/Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200/`.

### 3. Data Synthesis

Run the main data synthesis script:
```bash
./data_synthesis.sh
```
Note: We recommend Windows users to use Git Bash to run this script.

## Core Components

### Simulation Scripts
- `Data_synthesis_main.py`: Main simulation implementation
- `Data_synthesize.py`: Core simulation functions
- `data_synthesis.sh`: Simulation execution script

### Analysis Tools
- `plot_results.py`: Visualization functions for optimization and results
- `rw_data_processing.py`: Data processing utilities
- `transition_probability_estimation.py`: Disease state transition calculations
- `Course_synthesis.ipynb`: Analysis notebook for generating state transition K-M plots
- `example_data_mapping.ipynb`: Examples of mapping synthetic data to common COVID-19 simulation model formats

<!-- ### Testing
- `test_data_synthesize.py`: Unit tests for simulation functions -->

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
