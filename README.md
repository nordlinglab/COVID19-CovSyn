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

### Contact Network Structure
The case edge list contains:
- Source case index
- Target case index
- Infection timestamp
- Contact type/infection setting