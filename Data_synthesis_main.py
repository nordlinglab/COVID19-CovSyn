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
from Data_synthesize import *


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
            'clinic_size': getattr(data, 'clinic_size', None),
        }
    elif data_type == 'course_of_disease':
        data_dict = {
            'infection_day': getattr(data, 'infection_day', None),
            'latent_period': getattr(data, 'latent_period', None),
            'incubation_period': getattr(data, 'incubation_period', None),
            'infectious_period': getattr(data, 'infectious_period', None),
            'monitor_isolation_period': getattr(data, 'monitor_isolation_period', None),
            'date_of_critically_ill': getattr(data, 'date_of_critically_ill', None),
            'date_of_death': getattr(data, 'date_of_death', None),
            'date_of_recovery': getattr(data, 'date_of_recovery', None),
            'natural_immunity_status': getattr(data, 'natural_immunity_status', None),
            'negative_test_date': getattr(data, 'negative_test_date', None),
            'negative_test_status': getattr(data, 'negative_test_status', None),
            'positive_test_date': getattr(data, 'positive_test_date', None)
        }
    elif data_type == "contact":
        # Save contact data attributes
        data_dict = {
            'household_contacts_matrix': getattr(data, 'household_contacts_matrix', None),
            'household_effective_contacts': getattr(data, 'household_effective_contacts', None),
            'household_effective_contacts_infection_time': getattr(data, 'household_effective_contacts_infection_time', None),
            'household_secondary_contact_ages': getattr(data, 'household_secondary_contact_ages', None),
            'household_previously_infected_index_list': getattr(data, 'household_previously_infected_index_list', None),

            'workplace_contacts_matrix': getattr(data, 'workplace_contacts_matrix', None),
            'workplace_effective_contacts': getattr(data, 'workplace_effective_contacts', None),
            'workplace_effective_contacts_infection_time': getattr(data, 'workplace_effective_contacts_infection_time', None),
            'workplace_secondary_contact_ages': getattr(data, 'workplace_secondary_contact_ages', None),
            'workplace_previously_infected_index_list': getattr(data, 'workplace_previously_infected_index_list', None),

            'school_class_contacts_matrix': getattr(data, 'school_class_contacts_matrix', None),
            'school_effective_contacts': getattr(data, 'school_effective_contacts', None),
            'school_effective_contacts_infection_time': getattr(data, 'school_effective_contacts_infection_time', None),
            'school_secondary_contact_ages': getattr(data, 'school_secondary_contact_ages', None),
            'school_previously_infected_index_list': getattr(data, 'school_previousl', None),

            'health_care_contacts_matrix': getattr(data, 'health_care_contacts_matrix', None),
            'health_care_effective_contacts': getattr(data, 'health_care_effective_contacts', None),
            'health_care_effective_contacts_infection_time': getattr(data, 'health_care_effective_contacts_infection_time', None),
            'health_care_secondary_contact_ages': getattr(data, 'health_care_secondary_contact_ages', None),
            'health_care_previously_infected_index_list': getattr(data, 'health_care_previously_infected_index_list', None),

            'municipality_contacts_matrix': getattr(data, 'municipality_contacts_matrix', None),
            'municipality_effective_contacts': getattr(data, 'municipality_effective_contacts', None),
            'municipality_effective_contacts_infection_time': getattr(data, 'municipality_effective_contacts_infection_time', None),
            'municipality_secondary_contact_ages': getattr(data, 'municipality_secondary_contact_ages', None),
            'municipality_previously_infected_index_list': getattr(data, 'municipality_previously_infected_index_list', None)
        }

    return data_dict


def run_covid(seed, input_P, demographic_parameters, save_file=False, result_path='.', mode='result'):
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
    overdispersion_rate = input_P[35]
    overdispersion_weight = input_P[36]
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
    if mode == 'taiwan_first_outbreak':
        time_limit = 365
        with open('./variable/infection_days_list.pkl', 'rb') as file:
            infection_days_list = pickle.load(file)
        population_size = 23008366 - len(infection_days_list)
        natural_immunity_rate = 1
    if mode == 'spread_Taiwan':
        time_limit = 7*52
        number_source_cases = 1
        # infection_days = np.arange(number_source_cases)  # One case each day
        infection_days = np.zeros(number_source_cases)  # All happened in the first day.
        infection_days[0] = 0
        population_size = 23008366 - number_source_cases
        natural_immunity_rate = 1
    if mode == 'spread_Taiwan_weight':
        time_limit = 7*52
        number_source_cases = 1
        # infection_days = np.arange(number_source_cases)  # One case each day
        infection_days = np.zeros(number_source_cases)  # All happened in the first day.
        infection_days[0] = 0
        population_size = 23008366 - number_source_cases
        natural_immunity_rate = 1
        contact_weight = 3
        # Household
        input_P[0] = min(input_P[0]*contact_weight, 1)
        # School
        input_P[7] = min(input_P[7]*contact_weight, 1)
        # Workplace
        input_P[14] = min(input_P[14]*contact_weight, 1)
        # Health Care
        input_P[21] = min(input_P[21]*contact_weight, 1)
        # Municipality
        input_P[28] = min(input_P[28]*contact_weight, 1)
        # print(input_P[0], input_P[7], input_P[14], input_P[21], input_P[28])

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
        # Household
        input_P[0] = min(input_P[0]*contact_weight, 1)
        # School
        input_P[7] = min(input_P[7]*contact_weight, 1)
        # Workplace
        input_P[14] = min(input_P[14]*contact_weight, 1)
        # Health Care
        input_P[21] = min(input_P[21]*contact_weight, 1)
        # Municipality
        input_P[28] = min(input_P[28]*contact_weight, 1)
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
    age = np.nan
    contact_type = np.nan
    case_edge_list = []
    confirmed_count_per_day = {}  # Dictionary to store confirmed case count per day
    total_infections = 0  # Variable to store the total number of infections
    threshold_confirmed_day = np.nan
    # for infection_day in tqdm(infection_days):
    if mode == 'taiwan_first_outbreak':
        infection_days = infection_days_list[seed]
        # print('seed: ', seed)
        # print('infection_days: ', infection_days)
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
                case_edge_list = case_edge_list + \
                    [(source_case_id, case_id, infection_day, contact_type)]
                # Preciously infected or not
                if np.isnan(previously_infected_index):  # Not previously infected
                    # Draw demographic data
                    demographic_data = Draw_demographic_data(
                        age_p, gender_p, student_p, employment_p, job_p, age)
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
                                               hospital_p, hospital_size_p, hospital_sizes)
                social_data.draw_social_data()
                social_data_tmp = get_saveable_data(social_data, 'social')
                social_data_list.append(social_data_tmp)

                # Draw course of disease data
                # course_of_disease_data = False
                # while not course_of_disease_data:
                course_of_disease_data = Draw_course_of_disease_data(infection_day, latent_period_gamma, infectious_period_gamma, incubation_period_gamma, symptom_to_isolation_gamma,
                                                                     asymptomatic_to_recovered_gamma, symptomatic_to_critically_ill_gamma, symptomatic_to_recovered_gamma,
                                                                     critically_ill_to_recovered_gamma, infection_to_death_gamma, negative_to_confirmed_gamma, natural_immunity_rate, transition_p)
                # print('Infection day: ', infection_day)
                course_of_disease_data.draw_course_of_disease()
                natural_immunity_status_list.append(
                    course_of_disease_data.natural_immunity_status)
                course_of_disease_data_tmp = get_saveable_data(
                    course_of_disease_data, 'course_of_disease')
                course_of_disease_data_list.append(course_of_disease_data_tmp)

                # Append recovered cases to previously infected set
                if ~np.isnan(course_of_disease_data.date_of_recovery):
                    if ~np.isnan(previously_infected_index):
                        previously_infected_list.remove(
                            previously_infected_index)
                    previously_infected_list.append(case_id)

                # Draw contact data
                contact_data = Draw_contact_data(attack_rate, social_data, course_of_disease_data,
                                                 previously_infected_list, population_size, vaccine_efficacy,
                                                 vaccination_rate, natural_immunity_status_list,
                                                 overdispersion_rate, overdispersion_weight, age_risk_ratios, age_p)

                _, population_size = contact_data.draw_contact_data(input_P)
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
    except:
        # raise
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
        case_edge_list = [case_edge_list[i] for i in indices_to_keep]

    if save_file == True:
        with open(result_path / f'demographic_data_{seed}.npy', 'wb') as f:
            np.save(f, demographic_data_list)
        with open(result_path / f'social_data_{seed}.npy', 'wb') as f:
            np.save(f, social_data_list)
        with open(result_path / f'course_of_disease_data_{seed}.npy', 'wb') as f:
            np.save(f, course_of_disease_data_list)
        with open(result_path / f'contact_data_{seed}.npy', 'wb') as f:
            np.save(f, contact_data_list)
        with open(result_path / f'case_edge_list_{seed}.npy', 'wb') as f:
            np.save(f, case_edge_list)

    return (demographic_data_list, social_data_list, course_of_disease_data_list, contact_data_list)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Data synthesis')
    parser.add_argument('--mode', type=str, default='result', choices=['spread_Taiwan', 'spread_Taiwan_weight', 'spread_Taitung', 'profile', 'result', 'cheng2020', 'test', 'ge2021',
                        'spread_Taitung_outbreak_weight', 'spread_Lienchiang', 'taiwan_first_outbreak'],
                        help='Mode options include: spread, spread_demo, profile, ICIDMID, result, cheng2020, test, ge2021, Boonpatcharanon2022')
    parser.add_argument('--parameter_path', type=str,
                        default='./Firefly_result/Firefly_result_pop_size_50_alpha_1_betamin_1_gamma_0.131_max_generations_200')
    parser.add_argument('--monte_carlo_number', type=int, default=100)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--cpu_cores', type=int, default=1)

    args = parser.parse_args()
    mode = args.mode
    parameter_path = Path(args.parameter_path)
    mc_number = args.monte_carlo_number
    cpu_cores = args.cpu_cores
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
            seeds = range(i*cpu_cores, (i+1)*cpu_cores)
            with concurrent.futures.ProcessPoolExecutor() as executor:  # Multiprocessing
                results = [executor.submit(run_covid, seed, input_P, demographic_parameters, save_file=True,
                                           result_path=result_path, mode=mode)
                           for seed in seeds]

        # Process the remainder
        print('Processing the remainder')
        if remainder > 0:
            seeds = range(full_batches*cpu_cores, mc_number)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(run_covid, seed, input_P, demographic_parameters,
                                           save_file=True, result_path=result_path, mode=mode)
                           for seed in seeds]

        print("--- Done %s seconds ---" % (time.time() - start_time))
