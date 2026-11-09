import argparse
import concurrent.futures
import cProfile
import numpy as np
import pickle
import pstats
import queue
import random
import time
from pathlib import Path
from tqdm import tqdm
from covsyn.model.data_synthesize import *
from covsyn.model.school_calendar import SchoolCalendar
from covsyn.model.work_calendar import WorkCalendar, first_wave_day_zero, weekday_calendar
from covsyn.model.data_synthesize import (community_event_parameters, community_power_law_exponent,
                                          tracing_delay_mean, work_group_parameters)

# B66: the two first-wave scenarios and their seed tables (build_first_wave_seeds). 'all' seeds every
# imported case to 2020-04-12 with restrictions that grow over time; 'feb7' only the seeds before
# the 2020-02-07 border control, with none.
FIRST_WAVE_SEED_FILES = {'taiwan_first_outbreak': './variable/taiwan_first_wave_seeds.csv',
                         'taiwan_first_outbreak_feb7': './variable/taiwan_first_wave_seeds_feb7.csv',
                         'taiwan_first_outbreak_feb29': './variable/taiwan_first_wave_seeds_feb29.csv',
                         'taiwan_first_outbreak_mar18': './variable/taiwan_first_wave_seeds_mar18.csv',
                         'taiwan_first_outbreak_mar18_local':
                             './variable/taiwan_first_wave_seeds_mar18_local.csv'}
# B75: second entropy word of the work calendar's generator, next to the run's seed.
WORK_CALENDAR_STREAM = 75
# B78 options for the imported seeds, which infect about 3x the data in the community (run 19,
# 2026-11-02). Isolation: 'model' (the model's onset-to-isolation delay, as before) or
# 'imported_delay' (delay drawn from IMPORTED_ONSET_TO_CONFIRMATION, the default since the
# professor meeting of 2026-11-02, which chose A2 + C and dropped A1 'observed'). The
# transmission factor (option C) multiplies the infection probabilities of an imported seed
# living in the community; 1.0 is off, and scripts/calibrate_imported_factor.py calibrates it.
IMPORTED_ISOLATION_OPTIONS = ('model', 'imported_delay')
# B78 C: q calibrated under A2 with run 19 candidate 60, 400 runs (2026-11-02): with q = 1 the
# 118 imported community seeds of the 2020-04-12 scenario infect 36.2 people, q = 0.294 gives
# 12.1 (data 12). Recalibrate when the vector or the isolation changes.
# B82 (2026-11-04): recalibrated with the run 21 selected vector, and now the POST-BAN value
# only, since the 16 seeds that arrived before BORDER_CONTROL_DATE are held at 1.0. With q = 1
# the 118 community seeds infect 46.93; the relation is m(q) = A + B*q with A measured at 7.27
# (q = 0, 800 runs), so those 16 seeds alone produce 61% of the 12 infections in the data, and
# q = 0.107 gives 11.9-12.5. Steps measured at 800 runs: q = 0 -> 7.27, 0.090 -> 11.27,
# 0.104 -> 11.86, 0.120 -> 12.54. Recalibrate when the vector or the isolation changes; the
# proportional update in calibrate_imported_factor.py is biased by the constant A, so solve
# A + B*q = 12 from two low-q measurements instead of reading its last line.
IMPORTED_FACTOR = 0.107
# B82 (2026-11-04, professor meeting of 2026-11-02): the factor stands for the home quarantine,
# which did not exist before Taiwan's 2020-02-07 entry ban, so it applies only from that date.
# The professor's rule for the quarantine terms is the same one: nothing in February, the home
# quarantine probability in March, a new factor for the quarantine that replaced it. Applying one
# value to every period suppressed exactly the seeds the February scenarios run on: 16 of the
# feb7 scenario's 21 seeds arrived before the ban, while 102 of the 118 community seeds the
# factor was fitted on arrived after it. The data puts the local cases per seed 7.1x higher
# before the ban (17 from 21 seeds) than after (38 from the remaining 333), which no single
# factor can hold. The breach probability is NOT split: its five events are 2 of the 44 seeds
# quarantined before 2020-03-19 and 3 of the 78 after, which cannot tell the two periods apart.
BORDER_CONTROL_DATE = '2020-02-07'
# Onset to confirmation (days: cases) of the 210 imported cases of the first wave (to
# 2020-04-12) found in the community or in home quarantine with an onset date; mean 6.0, median
# 5 (Taiwan case data, figshare_taiwan_covid.xlsx). Airport cases are left out: they are
# isolated on arrival whatever their delay.
IMPORTED_ONSET_TO_CONFIRMATION = {1: 2, 2: 25, 3: 41, 4: 30, 5: 29, 6: 17, 7: 15, 8: 9, 9: 10,
                                  10: 11, 11: 3, 12: 3, 13: 2, 15: 3, 16: 1, 18: 2, 19: 4, 24: 1,
                                  27: 1, 30: 1}
# B69: the social-data attribute holding the other members of each shared group.
SHARED_GROUP_SIZE = {'household': 'household_size', 'school': 'school_class_size',
                     'workplace': 'work_group_size'}


def imported_factor_for(arrival_date, factor):
    """The imported transmission factor of a seed that arrived on this date (B82).

    Args:
        arrival_date: The seed's arrival date as 'YYYY-MM-DD', or None/'' when it has none
            (a local seed, which is never an imported seed and so never reads this).
        factor: The calibrated factor, which stands for the home quarantine.

    Returns:
        1.0 for an arrival before BORDER_CONTROL_DATE, when there was no screening and no
        quarantine to reduce the seed's contacts, and `factor` from that date on.
    """
    if not arrival_date:
        return factor
    return 1.0 if str(arrival_date) < BORDER_CONTROL_DATE else factor


def load_first_wave_seeds(path):
    """The seed cases of the Taiwan first wave (B66), as a list of dicts.

    Built by covsyn.data_processing.build_first_wave_seeds from the Taiwan case data: every
    imported case of the first wave and the 13 local cases of unknown source, with a fixed
    infection day (day 0 = the earliest seed infection), the ten-year age group, the gender,
    the arrival day counted from infection and the status after arrival.
    """
    import csv
    seeds = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            seeds.append({
                'case_id': int(row['case_id']),
                'seed_type': row['seed_type'],
                'age_low': float(row['age_low']) if row['age_low'] else np.nan,
                'age_high': float(row['age_high']) if row['age_high'] else np.nan,
                'gender': row['gender'] or None,
                'infection_day': int(row['infection_day']),
                'arrival_day': (int(float(row['arrival_day_from_infection']))
                                if row['arrival_day_from_infection'] else None),
                # B82: the calendar arrival date, which decides the border-control regime the
                # seed arrived under; empty for a local seed.
                'arrival_date': row.get('arrival_date') or None,
                'status': row['status'],
                # B68: a local seed infected by an imported case is traced from its
                # infector's confirmation (simulation day); empty for the other seeds.
                'source_confirmed_day': (float(row['source_confirmed_day'])
                                         if row.get('source_confirmed_day') else np.nan),
                # B73: a 'community' seed recalled into home quarantine on 2020-03-18, counted
                # from its infection; None for the others.
                'quarantine_from_day': (int(float(row['quarantine_from_day_from_infection']))
                                        if row.get('quarantine_from_day_from_infection') else None),
            })
    return seeds


def get_saveable_data(data, data_type):
    """Convert a list of data objects into a list of dictionaries containing only data attributes."""
    saveable_list = []
    data_dict = {}
    if data_type == 'demographic':
        # Save demographic data attributes
        data_dict = {
            'age': getattr(data, 'age', None),
            'gender': getattr(data, 'gender', None),
            'job': getattr(data, 'job', None)
        }
    elif data_type == 'social':
        # Save social data attributes
        data_dict = {
            'municipality': getattr(data, 'municipality', None),
            'household_size': getattr(data, 'household_size', None),
            'school_class_size': getattr(data, 'school_class_size', None),
            'work_group_size': getattr(data, 'work_group_size', None),
            'enterprise_size': getattr(data, 'enterprise_size', None),
            'clinic_size': getattr(data, 'clinic_size', None),
        }
    elif data_type == 'course_of_disease':
        data_dict = {
            'infection_day': getattr(data, 'infection_day', None),
            'latent_period': getattr(data, 'latent_period', None),
            'incubation_period': getattr(data, 'incubation_period', None),
            'infectious_period': getattr(data, 'infectious_period', None),
            'monitor_isolation_period': getattr(data, 'monitor_isolation_period', None),
            'pre_onset_window': getattr(data, 'pre_onset_window', None),
            'isolation_route': getattr(data, 'isolation_route', None),
            'date_of_critically_ill': getattr(data, 'date_of_critically_ill', None),
            'date_of_death': getattr(data, 'date_of_death', None),
            'date_of_recovery': getattr(data, 'date_of_recovery', None),
            'natural_immunity_status': getattr(data, 'natural_immunity_status', None),
            'negative_test_date': getattr(data, 'negative_test_date', None),
            'negative_test_status': getattr(data, 'negative_test_status', None),
            'positive_test_date': getattr(data, 'positive_test_date', None),
            # B66: 'imported', 'local_unknown' or (B68) 'local_from_imported' for a
            # first-wave seed, None otherwise; seed_case_id is its case number in the data.
            'seed_type': getattr(data, 'seed_type', None),
            'seed_status': getattr(data, 'seed_status', None),
            'seed_case_id': getattr(data, 'seed_case_id', None),
            # B82: the arrival date decides which border-control regime the seed arrived under,
            # so a scenario's output can be split by period.
            'seed_arrival_date': getattr(data, 'seed_arrival_date', None),
        }
    elif data_type == "contact":
        # Save contact data attributes
        data_dict = {
            'household_contacts_matrix': getattr(data, 'household_contacts_matrix', None),
            'household_effective_contacts': getattr(data, 'household_effective_contacts', None),
            'household_effective_contacts_infection_time': getattr(data, 'household_effective_contacts_infection_time', None),
            'household_secondary_contact_ages': getattr(data, 'household_secondary_contact_ages', None),
            'household_contact_ages': getattr(data, 'household_contact_ages', None),
            'household_previously_infected_index_list': getattr(data, 'household_previously_infected_index_list', None),

            'workplace_contacts_matrix': getattr(data, 'workplace_contacts_matrix', None),
            'workplace_effective_contacts': getattr(data, 'workplace_effective_contacts', None),
            'workplace_effective_contacts_infection_time': getattr(data, 'workplace_effective_contacts_infection_time', None),
            'workplace_secondary_contact_ages': getattr(data, 'workplace_secondary_contact_ages', None),
            'workplace_contact_ages': getattr(data, 'workplace_contact_ages', None),
            'workplace_previously_infected_index_list': getattr(data, 'workplace_previously_infected_index_list', None),

            'school_class_contacts_matrix': getattr(data, 'school_class_contacts_matrix', None),
            'school_effective_contacts': getattr(data, 'school_effective_contacts', None),
            'school_effective_contacts_infection_time': getattr(data, 'school_effective_contacts_infection_time', None),
            'school_secondary_contact_ages': getattr(data, 'school_secondary_contact_ages', None),
            'school_contact_ages': getattr(data, 'school_contact_ages', None),
            'school_previously_infected_index_list': getattr(data, 'school_previously_infected_index_list', None),

            'health_care_contacts_matrix': getattr(data, 'health_care_contacts_matrix', None),
            'health_care_effective_contacts': getattr(data, 'health_care_effective_contacts', None),
            'health_care_effective_contacts_infection_time': getattr(data, 'health_care_effective_contacts_infection_time', None),
            'health_care_secondary_contact_ages': getattr(data, 'health_care_secondary_contact_ages', None),
            'health_care_contact_ages': getattr(data, 'health_care_contact_ages', None),
            'health_care_previously_infected_index_list': getattr(data, 'health_care_previously_infected_index_list', None),

            'municipality_contacts_matrix': getattr(data, 'municipality_contacts_matrix', None),
            'municipality_effective_contacts': getattr(data, 'municipality_effective_contacts', None),
            'municipality_effective_contacts_infection_time': getattr(data, 'municipality_effective_contacts_infection_time', None),
            'municipality_secondary_contact_ages': getattr(data, 'municipality_secondary_contact_ages', None),
            'municipality_contact_ages': getattr(data, 'municipality_contact_ages', None),
            'municipality_previously_infected_index_list': getattr(data, 'municipality_previously_infected_index_list', None)
        }
        # E95: what the objective charges in place of the realised infection counts.
        if hasattr(data, 'expected_outcomes'):
            data_dict['expected_outcomes'] = data.expected_outcomes()
        # B54: only present when the vector carries event parameters, so pre-B54 output
        # keeps exactly its old keys.
        if getattr(data, 'municipality_event_mask', None) is not None:
            data_dict['municipality_event_mask'] = data.municipality_event_mask
            data_dict['municipality_event_expected_contacts'] = (
                data.municipality_event_expected_contacts)
            data_dict['municipality_event_expected_infections'] = getattr(
                data, 'municipality_event_expected_infections', None)
        # B75: only present when the run has a work calendar, so older output keeps its keys.
        if getattr(data, 'workplace_at_work', None) is not None:
            data_dict['workplace_at_work'] = data.workplace_at_work

    return data_dict


def write_run_metadata(result_path, **settings):
    """Record how a synthetic dataset was made next to it (todolist929 3.2: commit with data).

    Args:
        result_path: Directory of the generated data.
        **settings: Mode, contact weight, seeds and parameter path of the run.
    """
    import json
    import subprocess
    from datetime import datetime

    # The commit of the CovSyn code that ran, whatever directory the run was started from;
    # no git on the machine records None rather than failing a finished run.
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                check=False, cwd=Path(__file__).resolve().parent
                                ).stdout.strip() or None
    except OSError:
        commit = None
    record = {**settings, 'covsyn_commit': commit,
              'written': datetime.now().isoformat(timespec='seconds')}
    with open(Path(result_path) / 'run_metadata.json', 'w') as f:
        json.dump(record, f, indent=1)


def apply_contact_weight(input_P, contact_weight):
    """Scale every layer's contact parameter in place by contact_weight (the *_weight modes).

    Args:
        input_P: Parameter vector, modified in place.
        contact_weight: Factor on the contacts of every layer.
    """
    # Household, school, workplace and health care are probabilities, capped at 1.
    for index in (0, 7, 14, 21):
        input_P[index] = min(input_P[index]*contact_weight, 1)
    # Municipality: P[28] is a COUNT since B27 (mean community contacts per case), not
    # a probability, so it must not be capped at 1 -- doing so would silently cut the
    # community layer down to a single contact per case.
    # B62: with P[204] present P[28] is the shift s of P(N = n) ~ (n + s)^-alpha. Scaling s by w
    # scales the counts by about w too, since (n + w s)^-alpha = w^-alpha (n / w + s)^-alpha,
    # so the same line keeps meaning "w times the community contacts".
    input_P[28] = input_P[28]*contact_weight
    # Mass events (B54) are part of the community layer too: scaling their probability scales
    # the expected event contacts by the same factor, while the event size stays a property
    # of the event. Vectors from before B54 have no event block.
    if len(input_P) > 199:
        input_P[199] = min(input_P[199]*contact_weight, 1)


def run_covid(seed, input_P, demographic_parameters, save_file=False, result_path='.',
              mode='result', contact_weight=1.0, quarantine_breach='with_unclear',
              imported_isolation='imported_delay', imported_factor=IMPORTED_FACTOR, seed_file=None):
    """Simulate one Monte-Carlo run of the given mode.

    Args:
        seed: Random seed of the run.
        input_P: Parameter vector.
        demographic_parameters: Demographic inputs.
        save_file: Write the run's data files to result_path.
        result_path: Output directory.
        mode: Scenario (sources, population, time limit).
        contact_weight: Contact multiplier of spread_Taiwan_weight; 1 is the calibrated model.
        quarantine_breach: First-wave modes: option of QUARANTINE_BREACH ('confirmed' or
            'with_unclear'), or a number used as the breach probability itself (B74). The
            default counts the two seeds that infected someone on the way home (meeting
            2026-11-02).
        imported_isolation: First-wave modes: one of IMPORTED_ISOLATION_OPTIONS (B78).
        imported_factor: First-wave modes: transmission factor of the imported seeds living in
            the community (B78 option C); 1.0 leaves them unchanged.
        seed_file: First-wave modes: seed table to use instead of the mode's own, such as one
            with copied seeds (B79, build_first_wave_seeds.write_copied_seeds).

    Returns:
        The demographic, social, course-of-disease and contact data of every case.
    """
    # Load parameters
    # Load demographic and social data
    # demographic_parameters = np.load(
    #     './variable/demographic_parameters.npy', allow_pickle=True)
    age_p = demographic_parameters[0]
    gender_p = demographic_parameters[1]
    student_p = demographic_parameters[2]
    employment_p = demographic_parameters[3]
    job_p = demographic_parameters[4]
    family_size_dict = demographic_parameters[5]
    municipality_data = demographic_parameters[6]
    school_p = demographic_parameters[7]
    workplace_p = demographic_parameters[8]
    hospital_p = demographic_parameters[9]
    hospital_size_p = demographic_parameters[10]
    hospital_sizes = demographic_parameters[11]
    population_size = demographic_parameters[12]
    # B29 / B30 / B31: an infected person is a person, not a household, a class or a
    # company, so every social-context size distribution is person-weighted. Doing it here
    # rather than in the saved pickle keeps the raw census distributions untouched.
    family_size_dict, school_p, workplace_p = apply_person_weighting(
        family_size_dict, school_p, workplace_p)
    overdispersion_rate = input_P[35]
    overdispersion_weight = input_P[36]
    # P[198] (E35): dispersion of the community contact COUNT, drawn independently of the
    # infectiousness multiplier. Appended at the end of the vector so every existing index
    # keeps its meaning; older 198-long vectors simply fall back to the shared draw.
    community_dispersion = input_P[198] if len(input_P) > 198 else None
    # P[199..203] (B54): municipality mass events; None for older vectors.
    community_event = community_event_parameters(input_P)
    # P[204] (B62): the scale-free community count replaces B27's count and B54's events.
    community_exponent = community_power_law_exponent(input_P)
    # P[205], P[206] (B64): the work group; None keeps B31's. P[207] (B65): tracing delay.
    work_group = work_group_parameters(input_P)
    tracing_delay = tracing_delay_mean(input_P)
    latent_period_gamma = {
        'latent_period_shape': input_P[37], 'latent_period_scale': input_P[38]}
    infectious_period_gamma = {
        'infectious_period_shape': input_P[39], 'infectious_period_scale': input_P[40]}
    incubation_period_gamma = {
        'incubation_period_shape': input_P[41], 'incubation_period_scale': input_P[42]}
    symptom_to_isolation_gamma = {
        'symptom_to_confirmed_shape': input_P[43], 'symptom_to_confirmed_scale': input_P[44], 'symptom_to_confirmed_loc': input_P[45]}
    asymptomatic_to_recovered_gamma = {
        'asymptomatic_to_recovered_shape': input_P[46], 'asymptomatic_to_recovered_scale': input_P[47], 'asymptomatic_to_recovered_loc': input_P[48]}
    symptomatic_to_critically_ill_gamma = {
        'symptomatic_to_critically_ill_shape': input_P[49], 'symptomatic_to_critically_ill_scale': input_P[50], 'symptomatic_to_critically_ill_loc': input_P[51]}
    symptomatic_to_recovered_gamma = {
        'symptomatic_to_recovered_shape': input_P[52], 'symptomatic_to_recovered_scale': input_P[53], 'symptomatic_to_recovered_loc': input_P[54]}
    critically_ill_to_recovered_gamma = {
        'critically_ill_to_recovered_shape': input_P[55], 'critically_ill_to_recovered_scale': input_P[56], 'critically_ill_to_recovered_loc': input_P[57]}
    infection_to_death_gamma = {
        'infection_to_death_shape': input_P[58], 'infection_to_death_scale': input_P[59]}
    negative_to_confirmed_gamma = {
        'negative_to_confirmed_shape': input_P[60], 'negative_to_confirmed_scale': input_P[61], 'negative_to_confirmed_loc': input_P[62]}
    age_risk_ratios = input_P[63:67]
    # Contact age distribution per social layer (school/workplace are re-weighted by
    # student / employment probability instead of using the population distribution).
    layer_age_p = build_layer_age_distributions(age_p, gender_p, student_p, employment_p)
    age_risk_ratios = np.repeat(age_risk_ratios, [20, 20, 20, 41])
    natural_immunity_rate = input_P[67]
    vaccination_rate = input_P[68]
    vaccine_efficacy = input_P[69]
    attack_rate = {}
    attack_rate['household_attack_rate'] = input_P[70:95]
    attack_rate['school_attack_rate'] = input_P[95:120]
    attack_rate['workplace_attack_rate'] = input_P[120:145]
    attack_rate['health_care_attack_rate'] = input_P[145:170]
    attack_rate['municipality_attack_rate'] = input_P[170:195]
    infection_to_recovered_transition_p = input_P[195]
    symptom_to_recovered_transition_p = input_P[196]
    critically_ill_to_recovered_transition_p = input_P[197]
    transition_p = [infection_to_recovered_transition_p,
                    symptom_to_recovered_transition_p, critically_ill_to_recovered_transition_p]

    result_path = Path(result_path)
    if mode in FIRST_WAVE_SEED_FILES:
        seed_path = seed_file or FIRST_WAVE_SEED_FILES[mode]
        time_limit = 365
        # B66 (2026-10-24): the seeds are every imported case of the first wave and the 13
        # local cases of unknown source, each with its own infection day, age group, gender and
        # status after arrival (load_first_wave_seeds). They replace the 28 local confirmation
        # dates of E29, which used the confirmation date as the infection date and seeded local
        # cases with a known source.
        first_wave_seeds = load_first_wave_seeds(seed_path)
        # B74: infection probabilities of a seed in home quarantine are multiplied by this.
        breach = (float(quarantine_breach) if not isinstance(quarantine_breach, str)
                  else QUARANTINE_BREACH[quarantine_breach])
        population_size = 23008366 - len(first_wave_seeds)
        natural_immunity_rate = 1
        if imported_isolation not in IMPORTED_ISOLATION_OPTIONS:
            raise ValueError(f'imported_isolation must be one of {IMPORTED_ISOLATION_OPTIONS}')
        onset_delays = (np.repeat(list(IMPORTED_ONSET_TO_CONFIRMATION),
                                  list(IMPORTED_ONSET_TO_CONFIRMATION.values())).astype(float)
                        if imported_isolation == 'imported_delay' else None)
    if mode == 'spread_Taiwan':
        time_limit = 7*52
        number_source_cases = 1
        # infection_days = np.arange(number_source_cases)  # One case each day
        infection_days = np.zeros(number_source_cases)  # All happened in the first day.
        infection_days[0] = 0
        population_size = 23008366 - number_source_cases
        natural_immunity_rate = 1
    if mode == 'spread_Taiwan_weight':
        time_limit = 7*12
        number_source_cases = 1
        # infection_days = np.arange(number_source_cases)  # One case each day
        infection_days = np.zeros(number_source_cases)  # All happened in the first day.
        infection_days[0] = 0
        population_size = 23008366 - number_source_cases
        natural_immunity_rate = 1
        # Referee item B2 (2026-10-10): the weight is an argument (--contact_weight) so the
        # benchmark can span several transmission regimes; 1 is the calibrated Taiwan model.
        apply_contact_weight(input_P, contact_weight)

    if mode == 'spread_Taitung':
        time_limit = 365
        number_source_cases = 365
        infection_days = np.arange(number_source_cases)  # One case each day
        infection_days[0] = 0
        population_size = 213032 - number_source_cases
        natural_immunity_rate = 1
    if mode == 'spread_Taitung_outbreak_weight':
        # time_limit = 7*20  # 20 weeks
        time_limit = 7*52
        number_source_cases = 1
        infection_days = np.arange(number_source_cases)  # One case each day
        infection_days[0] = 0
        population_size = 213032 - number_source_cases
        natural_immunity_rate = 1
        contact_weight = 4
        apply_contact_weight(input_P, contact_weight)
    if mode == 'spread_Lienchiang':
        time_limit = 365*3
        number_source_cases = 1000
        infection_days = np.random.randint(0, time_limit, number_source_cases)
        infection_days[0] = 0
        population_size = 13645 - number_source_cases
        natural_immunity_rate = 1
    if mode == 'profile':
        time_limit = 1
        number_source_cases = 1
        infection_days = np.zeros(number_source_cases)
    if mode == 'result':
        time_limit = 1
        number_source_cases = 1
        infection_days = np.zeros(number_source_cases)
        # result_path = './synthetic_data_results'
    if mode == 'cheng2020':
        time_limit = 0
        number_source_cases = 100
        infection_days = np.zeros(number_source_cases)
        # result_path = './synthetic_data_results_cheng2020'
    if mode == 'test':  # The same setting as 'cheng2020' but not using multiprocessing
        time_limit = 0
        number_source_cases = 100
        infection_days = np.zeros(number_source_cases)
        # result_path = './synthetic_data_results_cheng2020'
    if mode == 'ge2021':
        time_limit = 0
        number_source_cases = 730
        infection_days = np.zeros(number_source_cases)
        # result_path = './synthetic_data_results_ge2021'
    if mode == 'Boonpatcharanon2022':
        time_limit = 150
        number_source_cases = 1
        infection_days = np.zeros(number_source_cases)
        # result_path = './synthetic_data_results_Boonpatcharanon2022'

    # Initialization
    # print('Seed: ', seed)
    np.random.seed(seed)
    random.seed(int(seed))
    # Initialization
    demographic_data_list = []
    social_data_list = []
    course_of_disease_data_list = []
    contact_data_list = []
    infection_queue = queue.PriorityQueue()
    previously_infected_list = []
    natural_immunity_status_list = []
    case_id = 1
    source_case_id = np.nan
    previously_infected_index = np.nan
    # B75: days off work. Only for vectors with the B64 work group, which is calibrated to
    # Fu 2010's days at work, so older vectors (run 10) reproduce bit for bit. The draws use
    # their own generator, seeded by the run, and leave the simulation's random stream alone.
    # A first-wave run is dated from its seed table (2020 holidays included); any other run
    # starts on a random weekday.
    if work_group is None:
        work_calendar = None
    elif mode in FIRST_WAVE_SEED_FILES:
        work_calendar = WorkCalendar(first_wave_day_zero(seed_path),
                                     np.random.default_rng([int(seed), WORK_CALENDAR_STREAM]))
    else:
        work_calendar = weekday_calendar(np.random.default_rng([int(seed), WORK_CALENDAR_STREAM]))
    # B77 (user, 2026-11-02): school is open only on school days, with the same day 0 and the same
    # vectors as the work calendar (B75): the 2020 calendar in a first-wave run, weekends off in
    # any other run, and none for the older vectors, which reproduce bit for bit.
    if work_calendar is None:
        school_days = None
    else:
        school_days = SchoolCalendar(work_calendar.day_zero, dated=mode in FIRST_WAVE_SEED_FILES)
    age = np.nan
    contact_type = np.nan
    transmission_digraph = []
    groups = {}  # B69: group id -> {'size': members including the first case, 'infected'}
    group_of = []  # B69: per case, {layer: group id}
    confirmed_count_per_day = {}  # Dictionary to store confirmed case count per day
    total_infections = 0  # Variable to store the total number of infections
    threshold_confirmed_day = np.nan
    # for infection_day in tqdm(infection_days):
    if mode in FIRST_WAVE_SEED_FILES:
        # B66: the year of birth is drawn within the case's ten-year age group (professor,
        # 2026-10-19); the seed is found again by its contact type when it leaves the queue.
        for k, seed_case in enumerate(first_wave_seeds):
            low, high = seed_case['age_low'], seed_case['age_high']
            seed_age = (random.randint(int(low), int(high)) if np.isfinite(low) and np.isfinite(high)
                        else np.nan)
            infection_queue.put((source_case_id, seed_case['infection_day'],
                                 previously_infected_index, seed_age, f'seed:{k}'))
    else:
        for infection_day in infection_days:
            infection_queue.put((source_case_id, infection_day,
                                previously_infected_index, age, contact_type))

    # Simulation
    try:  # Stop if empty queue
        while True:
            source_case_id, infection_day, previously_infected_index, age, contact_type = infection_queue.get(
                block=False)
            if infection_day <= time_limit:
                # Save contact network
                transmission_digraph = transmission_digraph + \
                    [(source_case_id, case_id, infection_day, contact_type)]
                seed_case = (first_wave_seeds[int(contact_type[5:])]
                             if isinstance(contact_type, str) and contact_type.startswith('seed:')
                             else None)
                # Preciously infected or not
                if np.isnan(previously_infected_index):  # Not previously infected
                    # Draw demographic data
                    demographic_data = Draw_demographic_data(
                        age_p, gender_p, student_p, employment_p, job_p, age,
                        gender=None if seed_case is None else seed_case['gender'])
                    demographic_data.draw_demographic_data()
                    demographic_tmp = get_saveable_data(
                        demographic_data, 'demographic')
                    demographic_data_list.append(demographic_tmp)
                else:  # Previously infected
                    demographic_data = demographic_data_list[int(
                        previously_infected_index)]
                    demographic_tmp = get_saveable_data(
                        demographic_data, 'demographic')
                    demographic_data_list.append(demographic_tmp)

                # Draw social data
                social_data = Draw_social_data(municipality_data, family_size_dict, school_p, workplace_p, demographic_data,
                                               hospital_p, hospital_size_p, hospital_sizes,
                                               work_group=work_group)
                social_data.draw_social_data()
                # B69: a case infected in a household, school class or work group joins its
                # infector's group instead of a newly drawn one, so it meets the same members,
                # some of them already infected (the infector among them). A household also
                # keeps its municipality, a work group its enterprise.
                inherited = (contact_type in SHARED_GROUP_LAYERS and not np.isnan(source_case_id))
                case_groups = {}
                group_infected = {}
                for layer, size_name in SHARED_GROUP_SIZE.items():
                    if inherited and contact_type == layer:
                        gid = group_of[int(source_case_id) - 1][layer]
                        setattr(social_data, size_name, groups[gid]['size'] - 1)
                        infector_social = social_data_list[int(source_case_id) - 1]
                        if layer == 'household':
                            social_data.municipality = infector_social['municipality']
                            social_data.municipality_size = municipality_data[social_data.municipality]
                        if layer == 'workplace':
                            social_data.enterprise_size = infector_social['enterprise_size']
                        group_infected[layer] = groups[gid]['infected'] - 1
                    else:
                        gid = len(groups)
                        groups[gid] = {'size': int(getattr(social_data, size_name)) + 1, 'infected': 1}
                    case_groups[layer] = gid
                group_of.append(case_groups)
                social_data_tmp = get_saveable_data(social_data, 'social')
                social_data_list.append(social_data_tmp)

                # Draw course of disease data
                # course_of_disease_data = False
                # while not course_of_disease_data:
                # B20 / B26: the course of disease now needs the case's age (severity is age
                # dependent) and the confirmation date of its infector (contact tracing
                # isolates this case shortly after, instead of everybody being isolated on a
                # symptom date that asymptomatic cases never have).
                case_age = demographic_tmp.get('age', None)
                case_age = np.nan if case_age is None else case_age
                if np.isnan(source_case_id):
                    source_confirmed_day = np.nan
                else:
                    source_confirmed_day = course_of_disease_data_list[
                        int(source_case_id) - 1]['positive_test_date']
                if seed_case is not None and np.isfinite(seed_case['source_confirmed_day']):
                    source_confirmed_day = seed_case['source_confirmed_day']
                # B78 option A2: the onset-to-isolation delay of an imported seed from the data.
                imported_seed = seed_case is not None and seed_case['seed_type'] == 'imported'
                seed_onset_delays = onset_delays if imported_seed else None
                course_of_disease_data = Draw_course_of_disease_data(infection_day, latent_period_gamma, infectious_period_gamma, incubation_period_gamma, symptom_to_isolation_gamma,
                                                                     asymptomatic_to_recovered_gamma, symptomatic_to_critically_ill_gamma, symptomatic_to_recovered_gamma,
                                                                     critically_ill_to_recovered_gamma, infection_to_death_gamma, negative_to_confirmed_gamma, natural_immunity_rate, transition_p,
                                                                     age=case_age, source_confirmed_day=source_confirmed_day, age_p=age_p,
                                                                     tracing_delay=tracing_delay,
                                                                     onset_to_isolation_sample=seed_onset_delays)
                # print('Infection day: ', infection_day)
                course_of_disease_data.draw_course_of_disease()
                if seed_case is not None:
                    course_of_disease_data.seed_type = seed_case['seed_type']
                    course_of_disease_data.seed_status = seed_case['status']
                    course_of_disease_data.seed_case_id = seed_case['case_id']
                    # B82: kept so a scenario's output can be split by border-control regime.
                    course_of_disease_data.seed_arrival_date = seed_case['arrival_date']
                natural_immunity_status_list.append(
                    course_of_disease_data.natural_immunity_status)
                course_of_disease_data_tmp = get_saveable_data(
                    course_of_disease_data, 'course_of_disease')
                course_of_disease_data_list.append(course_of_disease_data_tmp)

                # Append the case to the previously-infected set (B32, finding E5).
                # This used to require a recovery date, so a case that died was never added.
                # Combined with the contact loop skipping every layer while the set was
                # empty, the first case of a simulation produced no contacts at all whenever
                # it died -- about 8% of the runs. The set means "has been infected", which
                # is true whether the case recovers or dies.
                if ~np.isnan(previously_infected_index):
                    previously_infected_list.remove(previously_infected_index)
                previously_infected_list.append(case_id)

                # Draw contact data
                contact_data = Draw_contact_data(attack_rate, social_data, course_of_disease_data,
                                                 previously_infected_list, population_size, vaccine_efficacy,
                                                 vaccination_rate, natural_immunity_status_list,
                                                 overdispersion_rate, overdispersion_weight, age_risk_ratios, age_p,
                                                 layer_age_p, community_dispersion, community_event,
                                                 community_exponent,
                                                 seed_rule=(None if seed_case is None
                                                            or seed_case['arrival_day'] is None
                                                            else {'arrival_day': seed_case['arrival_day'],
                                                                  'status': seed_case['status'],
                                                                  'seed_type': seed_case['seed_type'],
                                                                  'breach': breach,
                                                                  # B82: by arrival regime
                                                                  'imported_factor':
                                                                      imported_factor_for(
                                                                          seed_case['arrival_date'],
                                                                          imported_factor),
                                                                  'quarantine_from_day':
                                                                      seed_case['quarantine_from_day']}),
                                                 group_infected=group_infected,
                                                 work_calendar=work_calendar,
                                                 school_calendar=school_days)

                _, population_size = contact_data.draw_contact_data(input_P)
                for layer in SHARED_GROUP_LAYERS:  # B69: the new infections join the group
                    groups[case_groups[layer]]['infected'] += int(
                        np.nansum(getattr(contact_data, f'{layer}_effective_contacts') or [0]))
                contact_data_tmp = get_saveable_data(contact_data, 'contact')
                contact_data_list.append(contact_data_tmp)

                # Append infection queue
                # Household
                if np.sum(contact_data.household_effective_contacts) > 0:
                    household_effective_contacts = np.array(
                        contact_data.household_effective_contacts)
                    household_effective_contacts_index = np.where(
                        household_effective_contacts == 1)[0]
                    for i in household_effective_contacts_index:
                        infection_queue.put((case_id,
                                            contact_data.household_effective_contacts_infection_time[
                                                i]+infection_day,
                                            contact_data.household_previously_infected_index_list[i],
                                            contact_data.household_secondary_contact_ages[i],
                                            'household'))

                # School
                if np.sum(contact_data.school_effective_contacts) > 0:
                    school_effective_contacts = np.array(
                        contact_data.school_effective_contacts)
                    school_effective_contacts_index = np.where(
                        school_effective_contacts == 1)[0]
                    for i in school_effective_contacts_index:
                        infection_queue.put((case_id,
                                            contact_data.school_effective_contacts_infection_time[
                                                i]+infection_day,
                                            contact_data.school_previously_infected_index_list[i],
                                            contact_data.school_secondary_contact_ages[i],
                                            'school'))

                # Workplace
                if np.sum(contact_data.workplace_effective_contacts) > 0:
                    workplace_effective_contacts = np.array(
                        contact_data.workplace_effective_contacts)
                    workplace_effective_contacts_index = np.where(
                        workplace_effective_contacts == 1)[0]
                    for i in workplace_effective_contacts_index:
                        infection_queue.put((case_id,
                                            contact_data.workplace_effective_contacts_infection_time[
                                                i]+infection_day,
                                            contact_data.workplace_previously_infected_index_list[i],
                                            contact_data.workplace_secondary_contact_ages[i],
                                            'workplace'))

                # Health care
                if np.sum(contact_data.health_care_effective_contacts) > 0:
                    health_care_effective_contacts = np.array(
                        contact_data.health_care_effective_contacts)
                    health_care_effective_contacts_index = np.where(
                        health_care_effective_contacts == 1)[0]
                    for i in health_care_effective_contacts_index:
                        infection_queue.put((case_id,
                                            contact_data.health_care_effective_contacts_infection_time[
                                                i]+infection_day,
                                            contact_data.health_care_previously_infected_index_list[i],
                                            contact_data.health_care_secondary_contact_ages[i],
                                            'health_care'))

                # Municipality
                if np.sum(contact_data.municipality_effective_contacts) > 0:
                    municipality_effective_contacts = np.array(
                        contact_data.municipality_effective_contacts)
                    municipality_effective_contacts_index = np.where(
                        municipality_effective_contacts == 1)[0]
                    for i in municipality_effective_contacts_index:
                        infection_queue.put((case_id,
                                            contact_data.municipality_effective_contacts_infection_time[
                                                i]+infection_day,
                                            contact_data.municipality_previously_infected_index_list[i],
                                            contact_data.municipality_secondary_contact_ages[i],
                                            'municipality'))

                case_id += 1
                # Update infection count per day
                confirmed_day = course_of_disease_data.positive_test_date
                if confirmed_day in confirmed_count_per_day:
                    confirmed_count_per_day[confirmed_day] += 1
                else:
                    confirmed_count_per_day[confirmed_day] = 1

                # Check if total confrimed cases exceed threshold
                daily_confirmed_threshold = 1000
                if confirmed_count_per_day[confirmed_day] >= daily_confirmed_threshold:
                    threshold_confirmed_day = confirmed_day
                    print(
                        f"Daily confirmed cases exceeded {daily_confirmed_threshold}. Stopping simulation.")
                    break
            else:
                # Drop the case
                pass
    except queue.Empty:
        # The loop ends when the queue runs dry. This used to be a bare `except:`, which
        # also swallowed real errors and returned a silently empty simulation -- exactly how
        # finding E5 stayed hidden. Anything other than an empty queue is now raised.
        infection_queue.task_done()

    # Remove individual data if the confirmed day surpass the time of the daily confirmed cases threshold
    # Create a list of indices to keep

    if np.isnan(threshold_confirmed_day):
        pass
    else:
        indices_to_keep = [i for i, course_of_disease_data in enumerate(course_of_disease_data_list)
                           if course_of_disease_data['positive_test_date'] <= threshold_confirmed_day]

        # Filter the lists based on the indices to keep
        course_of_disease_data_list = [
            course_of_disease_data_list[i] for i in indices_to_keep]
        social_data_list = [social_data_list[i] for i in indices_to_keep]
        demographic_data_list = [demographic_data_list[i]
                                 for i in indices_to_keep]
        contact_data_list = [contact_data_list[i] for i in indices_to_keep]
        transmission_digraph = [transmission_digraph[i] for i in indices_to_keep]

    if save_file == True:
        with open(result_path / f'demographic_data_{seed}.npy', 'wb') as f:
            np.save(f, demographic_data_list)
        with open(result_path / f'social_data_{seed}.npy', 'wb') as f:
            np.save(f, social_data_list)
        with open(result_path / f'course_of_disease_data_{seed}.npy', 'wb') as f:
            np.save(f, course_of_disease_data_list)
        with open(result_path / f'contact_data_{seed}.npy', 'wb') as f:
            np.save(f, contact_data_list)
        with open(result_path / f'transmission_digraph_{seed}.npy', 'wb') as f:
            np.save(f, transmission_digraph)

    return (demographic_data_list, social_data_list, course_of_disease_data_list, contact_data_list)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Data synthesis')
    parser.add_argument('--mode', type=str, default='spread_Taiwan_weight', choices=['spread_Taiwan', 'spread_Taiwan_weight', 'spread_Taitung', 'profile', 'result', 'cheng2020', 'test', 'ge2021',
                        'spread_Taitung_outbreak_weight', 'spread_Lienchiang', 'taiwan_first_outbreak',
                                 'taiwan_first_outbreak_feb7', 'taiwan_first_outbreak_feb29',
                                 'taiwan_first_outbreak_mar18', 'taiwan_first_outbreak_mar18_local'],
                        help='Mode options include: spread, spread_demo, profile, ICIDMID, result, cheng2020, test, ge2021, Boonpatcharanon2022')
    parser.add_argument('--parameter_path', type=str,
                        default='./Firefly_result/Firefly_result_pop_size_50_alpha_1_betamin_1_gamma_0.131_max_generations_200')
    parser.add_argument('--monte_carlo_number', type=int, default=100)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--cpu_cores', type=int, default=1)
    parser.add_argument('--contact_weight', type=float, default=1.0,
                        help='contact multiplier of spread_Taiwan_weight (1 = calibrated Taiwan)')
    parser.add_argument('--quarantine_breach', type=str, default='with_unclear',
                        choices=['confirmed', 'with_unclear'],
                        help='first-wave modes: breach probability of home quarantine, calibrated '
                             'to the 3 confirmed or the 5 confirmed and unclear infections (B74)')
    parser.add_argument('--imported_isolation', type=str, default='imported_delay',
                        choices=list(IMPORTED_ISOLATION_OPTIONS),
                        help='B78 option A: isolation of the imported first-wave seeds')
    parser.add_argument('--imported_factor', type=float, default=IMPORTED_FACTOR,
                        help='B78 option C: transmission factor of the imported seeds in the '
                             'community (1 = off)')
    parser.add_argument('--seed_file', type=str, default=None,
                        help='first-wave modes: seed table replacing the mode\'s own (B79)')
    parser.add_argument('--seed_start', type=int, default=0,
                        help='first Monte-Carlo seed; B60 uses 100000 for an independent set')

    args = parser.parse_args()
    mode = args.mode
    parameter_path = Path(args.parameter_path)
    mc_number = args.monte_carlo_number
    cpu_cores = args.cpu_cores
    seed_start = args.seed_start
    contact_weight = args.contact_weight
    quarantine_breach = args.quarantine_breach
    options = {'imported_isolation': args.imported_isolation,
               'imported_factor': args.imported_factor, 'seed_file': args.seed_file}
    if contact_weight != 1.0 and mode != 'spread_Taiwan_weight':
        raise SystemExit('--contact_weight applies to spread_Taiwan_weight only')
    result_path = Path(args.result_path)
    print(f'Result path: {result_path}')
    if mode == 'spread_Taitung_outbreak_weight_2' or mode == 'spread_Taitung_outbreak_weight_1p5' or mode == 'spread_Taitung_outbreak_weight_3' or mode == 'spread_Taitung':
        with open('./variable_Taitung/demographic_parameters.pkl', 'rb') as f:
            demographic_parameters = pickle.load(f)
    else:
        with open('./variable/demographic_parameters.pkl', 'rb') as f:
            demographic_parameters = pickle.load(f)

    # Load firefly parameters
    file_name = Path('./firefly_best.txt')
    with open(parameter_path/file_name, 'r') as f:
        results = [[float(num) for num in line.split(' ')] for line in f]
    results = np.array(results)
    best_index = np.argmin(results[:, -1])
    best_firefly = results[best_index, 1:-1]
    input_P = best_firefly

    # Monte_carlo
    print(f'mode: {mode}')
    if mode == 'Boonpatcharanon2022':
        mc_number = 1000
    elif mode == 'profile':
        mc_number = 1000
        result_path = Path('./profile_results/')

    if mode == 'profile':
        start_t = time.time()
        mc_number = 100
        with cProfile.Profile() as pr:
            for seed in range(mc_number):
                run_covid(seed=seed, input_P=input_P,
                          demographic_parameters=demographic_parameters, save_file=False,
                          result_path=result_path, mode=mode)
            result_name = Path('data_synthesis_profiling.prof')
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.dump_stats(
                filename=result_path/result_name)
        print("--- Done %s seconds ---" % (time.time() - start_t))
    elif mode == 'test':
        # mc_number = 10
        seeds = range(mc_number)
        results = []
        for i in range(mc_number):
            results.append(run_covid(seed=seeds[i], input_P=input_P,
                                     demographic_parameters=demographic_parameters, save_file=True,
                                     result_path=result_path, mode=mode))
        print("--- Done %s seconds ---" % (time.time() - start_time))
    else:
        full_batches = mc_number // cpu_cores
        remainder = mc_number % cpu_cores

        print(f'Full batches: {full_batches}, Remainder: {remainder}')
        for i in tqdm(range(full_batches)):
            seeds = range(seed_start + i*cpu_cores, seed_start + (i+1)*cpu_cores)
            with concurrent.futures.ProcessPoolExecutor() as executor:  # Multiprocessing
                futures = [executor.submit(run_covid, seed, input_P, demographic_parameters,
                                           save_file=True, result_path=result_path, mode=mode,
                                           contact_weight=contact_weight,
                                           quarantine_breach=quarantine_breach, **options)
                           for seed in seeds]
                # A worker's exception used to stay inside its unread future, so a failed
                # simulation left no file and no error; .result() raises it here.
                for future in futures:
                    future.result()

        # Process the remainder
        print('Processing the remainder')
        if remainder > 0:
            seeds = range(seed_start + full_batches*cpu_cores, seed_start + mc_number)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_covid, seed, input_P, demographic_parameters,
                                           save_file=True, result_path=result_path, mode=mode,
                                           contact_weight=contact_weight,
                                           quarantine_breach=quarantine_breach, **options)
                           for seed in seeds]
                for future in futures:
                    future.result()

        write_run_metadata(result_path, mode=mode, contact_weight=contact_weight,
                           quarantine_breach=(quarantine_breach if mode in FIRST_WAVE_SEED_FILES
                                              else None),
                           **options, seed_start=seed_start, monte_carlo_number=mc_number,
                           parameter_path=str(parameter_path))
        print("--- Done %s seconds ---" % (time.time() - start_time))
