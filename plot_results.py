import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time
import os
from lifelines import KaplanMeierFitter


def load_synthetic_data(data_path, return_len=10, memory_limit=1e9, min_case_num=0):
    '''
    Load all the saved synthetic data. Each file represent one simulation.

    Parameters
    ----------
    data_path : str, Path to data
    return_len : int or 'all', default 'all'
    memory_limit : int, default 1e9 (1 GB), maximum memory usage in bytes
    min_case_num: int, default 0, minimum number of cases required for user

    Returns
    -------
    demographic_data_list_all : list
    social_data_list_all : list
    course_of_disease_data_list_all : list
    contact_data_list_all : list
    case_edge_list_list : list
    '''
    seed = 0

    demographic_data_list_all = []
    social_data_list_all = []
    course_of_disease_data_list_all = []
    contact_data_list_all = []
    case_edge_list_all = []

    total_memory_usage = 0

    while len(demographic_data_list_all) < return_len:
        print(
            f"seed: {seed}, loaded data length: {len(demographic_data_list_all)}")
        demographic_data = np.load(
            data_path / f'demographic_data_{seed}.npy', allow_pickle=True)
        if demographic_data.shape[0] < min_case_num:
            seed += 1
            continue
        social_data = np.load(
            data_path / f'social_data_{seed}.npy', allow_pickle=True)
        course_of_disease_data = np.load(
            data_path / f'course_of_disease_data_{seed}.npy', allow_pickle=True)
        contact_data = np.load(
            data_path / f'contact_data_{seed}.npy', allow_pickle=True)
        case_edge_list = np.load(
            data_path / f'case_edge_list_{seed}.npy', allow_pickle=True)

        total_memory_usage += (demographic_data.nbytes + social_data.nbytes +
                               course_of_disease_data.nbytes + contact_data.nbytes +
                               case_edge_list.nbytes)
        if total_memory_usage > memory_limit:
            print(
                f"Memory limit exceeded: {total_memory_usage} bytes. Stopping data loading.")
            break

        demographic_data_list_all.append(demographic_data)
        social_data_list_all.append(social_data)
        course_of_disease_data_list_all.append(course_of_disease_data)
        contact_data_list_all.append(contact_data)
        case_edge_list_all.append(case_edge_list)
        seed += 1
    print(f'Loaded data length: {len(demographic_data_list_all)}')
    print(f"Total memory usage: {total_memory_usage / 1e9} GB.")
    if len(demographic_data_list_all) < return_len:
        print(
            f"Warning: Only {len(demographic_data_list_all)} simulations loaded. Too few number of cases simulated.")

    return (demographic_data_list_all, social_data_list_all, course_of_disease_data_list_all, contact_data_list_all, case_edge_list_all)

####################################################################################################
# Course data
####################################################################################################


def kmplot(course_of_disease_data_list, start_state, target_state):
    duration = np.array([])
    if (start_state == 'symptomatic') & (target_state == 'critically ill'):
        for i, data in enumerate(course_of_disease_data_list):
            duration = np.append(
                duration, data.date_of_critically_ill-data.incubation_period)

    # Remove nan
    duration = duration[~np.isnan(duration)]
    event = np.ones(len(duration))

    # Plot
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event)
    fig = kmf.plot()
    plt.xlabel('Days from %s to %s' % (start_state, target_state))
    plt.ylabel('Survival probability')
    fig.get_legend().remove()
    print(np.median(duration))
    print(np.mean(duration))

    return (fig)
####################################################################################################
# Contact data
####################################################################################################


def generate_course_and_contact_combine_data(course_of_disease_data_list, contact_data_list, layer='All'):
    # NOTE: This code drop the asymptomatic cases since in Cheng et al.'s plot, they did not include asymptomatic cases
    # Drop asymptomatic cases
    pop_index = []
    for i in range(len(course_of_disease_data_list)):
        incubation_period = course_of_disease_data_list[i]['incubation_period']
        if np.isnan(incubation_period):
            pop_index.append(i)
    course_of_disease_data_list = np.delete(
        course_of_disease_data_list, pop_index)
    contact_data_list = np.delete(contact_data_list, pop_index)

    # Calculate return array size for preallocation
    size = 0
    for j in range(len(course_of_disease_data_list)):
        contact_data = contact_data_list[j]
        if (layer == 'Household') | (layer == 'All'):
            size += len(contact_data['household_contacts_matrix'])
        if (layer == 'School') | (layer == 'All'):
            size += len(contact_data['school_class_contacts_matrix'])
        if (layer == 'Workplace') | (layer == 'All'):
            size += len(contact_data['workplace_contacts_matrix'])
        if (layer == 'Health care') | (layer == 'All'):
            size += len(contact_data['health_care_contacts_matrix'])
        if (layer == 'Municipality') | (layer == 'All'):
            size += len(contact_data['municipality_contacts_matrix'])

    # Calculate return array
    duration_array = np.ones(size)*np.nan
    infection_day_array = np.ones(size)*np.nan
    incubation_period_array = np.ones(size)*np.nan
    first_contact_day_array = np.ones(size)*np.nan
    index = 0
    for k in range(len(course_of_disease_data_list)):
        incubation_period = course_of_disease_data_list[k]['incubation_period']
        contact_data = contact_data_list[k]
        if (layer == 'Household') | (layer == 'All'):
            # Household contact
            for m, row in enumerate(contact_data['household_contacts_matrix']):
                incubation_period_array[index+m] = incubation_period
                duration_array[index +
                               m] = np.where(row)[0][-1]-np.where(row)[0][0]+1
                first_contact_day_array[index+m] = np.where(row)[0][0]
            for n, value in enumerate(contact_data['household_effective_contacts_infection_time']):
                infection_day_array[index+n] = value
            index += len(contact_data['household_contacts_matrix'])
        if (layer == 'School') | (layer == 'All'):
            # School contact
            for m, row in enumerate(contact_data['school_class_contacts_matrix']):
                incubation_period_array[index+m] = incubation_period
                duration_array[index +
                               m] = np.where(row)[0][-1]-np.where(row)[0][0]+1
                first_contact_day_array[index+m] = np.where(row)[0][0]
            for n, value in enumerate(contact_data['school_effective_contacts_infection_time']):
                infection_day_array[index+n] = value
            index += len(contact_data['school_class_contacts_matrix'])
        if (layer == 'Workplace') | (layer == 'All'):
            # Workplace contact
            for m, row in enumerate(contact_data['workplace_contacts_matrix']):
                incubation_period_array[index+m] = incubation_period
                duration_array[index +
                               m] = np.where(row)[0][-1]-np.where(row)[0][0]+1
                first_contact_day_array[index+m] = np.where(row)[0][0]
            for n, value in enumerate(contact_data['workplace_effective_contacts_infection_time']):
                infection_day_array[index+n] = value
            index += len(contact_data['workplace_contacts_matrix'])
        if (layer == 'Health care') | (layer == 'All'):
            # Health care
            for m, row in enumerate(contact_data['health_care_contacts_matrix']):
                incubation_period_array[index+m] = incubation_period
                duration_array[index +
                               m] = np.where(row)[0][-1]-np.where(row)[0][0]+1
                first_contact_day_array[index+m] = np.where(row)[0][0]
            for n, value in enumerate(contact_data['health_care_effective_contacts_infection_time']):
                infection_day_array[index+n] = value
            index += len(contact_data['health_care_contacts_matrix'])
        if (layer == 'Municipality') | (layer == 'All'):
            # Municipality contact
            for m, row in enumerate(contact_data['municipality_contacts_matrix']):
                incubation_period_array[index+m] = incubation_period
                duration_array[index +
                               m] = np.where(row)[0][-1]-np.where(row)[0][0]+1
                first_contact_day_array[index+m] = np.where(row)[0][0]
            for n, value in enumerate(contact_data['municipality_effective_contacts_infection_time']):
                infection_day_array[index+n] = value
            index += len(contact_data['municipality_contacts_matrix'])

    return (duration_array, infection_day_array, first_contact_day_array, incubation_period_array)


def generate_stack_contact_matrix(contact_data_list, layer='All'):
    contacts_list = []
    # Extract contact matrices
    if layer == 'Household':
        for i in range(len(contact_data_list)):
            contacts_list.append(
                contact_data_list[i]['household_contacts_matrix'])
    if layer == 'School':
        for i in range(len(contact_data_list)):
            contacts_list.append(
                contact_data_list[i]['school_class_contacts_matrix'])
    if layer == 'Workplace':
        for i in range(len(contact_data_list)):
            contacts_list.append(
                contact_data_list[i]['workplace_contacts_matrix'])
    if layer == 'Health care':
        for i in range(len(contact_data_list)):
            contacts_list.append(
                contact_data_list[i]['health_care_contacts_matrix'])
    if layer == 'Municipality':
        for i in range(len(contact_data_list)):
            contacts_list.append(
                contact_data_list[i]['municipality_contacts_matrix'])
    if layer == 'All':
        for i in range(len(contact_data_list)):
            contacts_list.append(
                contact_data_list[i]['household_contacts_matrix'])
            contacts_list.append(
                contact_data_list[i]['school_class_contacts_matrix'])
            contacts_list.append(
                contact_data_list[i]['workplace_contacts_matrix'])
            contacts_list.append(
                contact_data_list[i]['health_care_contacts_matrix'])
            contacts_list.append(
                contact_data_list[i]['municipality_contacts_matrix'])

    # Concatenate each source cases' matrices
    contact_date_stack_matrix = contacts_list[0]
    index = np.array([np.shape(contact_date_stack_matrix)[0]])
    for i in np.arange(1, len(contacts_list)):
        contacts_matrix = contacts_list[i]
        if np.shape(contacts_matrix)[0] == 0:  # No contact at all
            pass
        else:
            index = np.append(index, index[-1]+np.shape(contacts_matrix)[0])
            if np.shape(contacts_matrix)[1] >= np.shape(contact_date_stack_matrix)[1]:
                contact_date_stack_matrix = np.hstack([contact_date_stack_matrix, np.zeros([np.shape(contact_date_stack_matrix)[
                    0], np.shape(contacts_matrix)[1]-np.shape(contact_date_stack_matrix)[1]])])
            else:
                contacts_matrix = np.hstack([contacts_matrix, np.zeros([np.shape(contacts_matrix)[
                    0], np.shape(contact_date_stack_matrix)[1]-np.shape(contacts_matrix)[1]])])
            contact_date_stack_matrix = np.vstack(
                [contact_date_stack_matrix, contacts_matrix])

    if index[0] == 0:
        index = index[1::]

    return (contact_date_stack_matrix, index)


def calculate_relative_risk(course_of_disease_data_list, contact_data_list, layer='All'):
    # Drop asymptomatic cases
    course_of_disease_data_list_no_asymptomatic = []
    contact_data_list_no_asymptomatic = []
    for i in range(len(course_of_disease_data_list)):
        if ~np.isnan(course_of_disease_data_list[i].incubation_period):
            course_of_disease_data_list_no_asymptomatic.append(
                course_of_disease_data_list[i])
            contact_data_list_no_asymptomatic.append(contact_data_list[i])
    course_of_disease_data_list = course_of_disease_data_list_no_asymptomatic
    contact_data_list = contact_data_list_no_asymptomatic

    # Generate epidemiological parameters and contact matrix
    duration_array, infection_day_array, first_contact_day_array, incubation_period_array = \
        generate_course_and_contact_combine_data(
            course_of_disease_data_list, contact_data_list, layer=layer)
    contact_date_stack_matrix, index = generate_stack_contact_matrix(
        contact_data_list, layer=layer)
    exposure_day_array = np.array([])

    if np.sum(index) == 0:  # No contact
        pass
    else:
        # Exposure day array
        past_index = 0
        for i in index:
            exposure_day_array = np.append(exposure_day_array, np.where(
                contact_date_stack_matrix[past_index:i] == 1)[1]-incubation_period_array[i-1])
            past_index = i

        # Contact array
        daily_contact_array = np.zeros(
            int(np.max(exposure_day_array)-np.min(exposure_day_array)+1))
        contact_day_shift = np.min(exposure_day_array)
        for i in exposure_day_array:
            daily_contact_array[np.int32(i-contact_day_shift)] += 1

        # Infection array
        daily_infection_array = np.zeros(
            int(np.max(exposure_day_array)-np.min(exposure_day_array)+1))
        infection_day_shift = np.min(exposure_day_array)
        adjusted_infection_day_array = infection_day_array - incubation_period_array
        for i in adjusted_infection_day_array:
            if ~np.isnan(i):
                daily_infection_array[np.int32(i-infection_day_shift)] += 1

        # Attack rate
        attack_rate = daily_infection_array/daily_contact_array
        attack_rate[np.isnan(attack_rate)] = 0
        time = np.arange(len(daily_contact_array))+contact_day_shift

    return (time, attack_rate)


def generate_contact_day_vs_infection_day_array(course_of_disease_data_list, contact_data_list, layer='All'):
    '''
    Generate contact day vs infection day array

    Parameters
    ----------
    course_of_disease_data_list: List of course_of_disease_data object
    contact_data_list: List of contact_data object
    layer: Contact type layer. 'All', 'Household', 'School', 'Workplace', 'Health care', 'Municipality'

    Returns
    -------
    contact_day_shift: int. Day shift relative to the symptom-onset.
    exposure_day_array: Day of exposure relative to the symptom-onset. each persion might have multiple contacts in different days.
    daily_contact_array: Contact number each day. The day of index 0 in the array is the contact_day_shift.
    infection_day_shift: Day shift for daily_infection_array.
    daily_infection_array: Effective contact each day. The day of index 0  in the array is the contact_day_shift.
    daily_secondary_attack_rate: Daily secondary attack rate directly calculated by daily_infection_array/daily_contact_array
    '''

    # Drop asymptomatic cases
    course_of_disease_data_list_no_asymptomatic = []
    contact_data_list_no_asymptomatic = []
    for i in range(len(course_of_disease_data_list)):
        if ~np.isnan(course_of_disease_data_list[i]['incubation_period']):
            course_of_disease_data_list_no_asymptomatic.append(
                course_of_disease_data_list[i])
            contact_data_list_no_asymptomatic.append(contact_data_list[i])
    course_of_disease_data_list = course_of_disease_data_list_no_asymptomatic
    contact_data_list = contact_data_list_no_asymptomatic

    # Generate epidemiological parameters and contact matrix
    duration_array, infection_day_array, first_contact_day_array, incubation_period_array = \
        generate_course_and_contact_combine_data(
            course_of_disease_data_list, contact_data_list, layer=layer)
    contact_date_stack_matrix, index = generate_stack_contact_matrix(
        contact_data_list, layer=layer)
    exposure_day_array = np.array([])

    if np.sum(index) == 0:  # No contact
        print('No contacts')
        contact_day_shift = 0
        exposure_day_array = np.array([0])
        daily_contact_array = np.array([0])
        infection_day_shift = 0
        daily_infection_array = np.array([0])
        daily_secondary_attack_rate = np.array([0])
    else:
        # Exposure day array
        past_index = 0
        for i in index:
            exposure_day_array = np.append(exposure_day_array, np.where(
                contact_date_stack_matrix[past_index:i] == 1)[1]-incubation_period_array[i-1])
            past_index = i

        # Contact array
        daily_contact_array = np.zeros(
            int(np.max(exposure_day_array)-np.min(exposure_day_array)+1))
        contact_day_shift = np.min(exposure_day_array)
        for i in exposure_day_array:
            daily_contact_array[np.int32(i-contact_day_shift)] += 1

        # Infection array
        daily_infection_array = np.zeros(
            int(np.max(exposure_day_array)-np.min(exposure_day_array)+1))
        infection_day_shift = np.min(exposure_day_array)
        adjusted_infection_day_array = infection_day_array - incubation_period_array
        for i in adjusted_infection_day_array:
            if ~np.isnan(i):
                daily_infection_array[np.int32(i-infection_day_shift)] += 1

        # Attack rate
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_secondary_attack_rate = np.divide(daily_infection_array, daily_contact_array, out=np.zeros_like(
                daily_infection_array), where=daily_contact_array != 0)
        daily_secondary_attack_rate[np.isnan(daily_secondary_attack_rate)] = 0

    return (contact_day_shift, exposure_day_array, daily_contact_array, infection_day_shift, daily_infection_array, daily_secondary_attack_rate)


def stack_day_matrix(day_matrix, matrix_day_shift, day_vector, vector_day_shift):
    # vstack the day_vector to day_matrix
    if matrix_day_shift > vector_day_shift:
        shift_difference = matrix_day_shift - vector_day_shift
        if day_matrix.shape[1]+shift_difference > len(day_vector):
            day_matrix = np.hstack(
                (np.zeros((day_matrix.shape[0], shift_difference)), day_matrix))
            day_vector = np.hstack(
                (day_vector, np.zeros(day_matrix.shape[1]-len(day_vector))))
        else:
            day_matrix = np.hstack(
                (np.zeros((day_matrix.shape[0], shift_difference)), day_matrix))
            day_matrix = np.hstack((day_matrix, np.zeros(
                (day_matrix.shape[0], len(day_vector)-day_matrix.shape[1]))))
    else:
        shift_difference = vector_day_shift - matrix_day_shift
        if day_matrix.shape[1] < len(day_vector)+shift_difference:
            day_vector = np.hstack((np.zeros(shift_difference), day_vector))
            day_matrix = np.hstack(
                (day_matrix, np.zeros((day_matrix.shape[0], len(day_vector)-day_matrix.shape[1]))))
        else:
            day_vector = np.hstack(
                (np.zeros((shift_difference)), day_vector))
            day_vector = np.hstack((day_vector, np.zeros(
                (day_matrix.shape[1] - len(day_vector)))))

    # if matrix_day_shift < vector_day_shift:
    #     if day_matrix.shape[1]+matrix_day_shift < len(day_vector)+vector_day_shift:
    #         day_matrix = np.hstack((day_matrix, np.zeros(day_matrix.shape[0], len(
    #             vector_day_shift)+vector_day_shift-day_matrix.shape[1]+matrix_day_shift)))
    #         day_vector = np.hstack(
    #             (np.zeros(vector_day_shift-matrix_day_shift), day_vector))
    #     else:
    #         day_vector = np.hstack((np.zeros(vector_day_shift-matrix_day_shift), day_vector, np.zeros(
    #             len(day_vector)+vector_day_shift-day_matrix.shape[1]+matrix_day_shift)))
    # else:
    #     if day_matrix.shape[1]+matrix_day_shift < len(day_vector)+vector_day_shift:
    #         day_matrix = np.hstack((np.zeros(day_matrix.shape[0], len(
    #             vector_day_shift)+vector_day_shift-day_matrix.shape[1]+matrix_day_shift), day_matrix))
    #         day_vector = np.hstack(
    #             (np.zeros(vector_day_shift-matrix_day_shift), day_vector))
    #     else:
    #         day_vector = np.hstack((
    #             np.zeros(matrix_day_shift-vector_day_shift), day_vector,
    #             np.zeros((day_matrix.shape[1]+matrix_day_shift)-(len(day_vector)+vector_day_shift))))

    stack_day_matrix = np.vstack((day_matrix, day_vector))
    day_shift = np.min((matrix_day_shift, vector_day_shift))

    return (stack_day_matrix, day_shift)


def plot_contact_day_vs_infection_day(course_of_disease_data_list, contact_data_list, attack_y_limit, monte_carlo_number=1, num_source_cases=100, layer='All', save_fig=True):
    '''
    Plot contact day vs infection day, x-axis: Days from onset to contact, y-axis: Number of contacts
    Note: this plot shows the detail daily contact instead of just the first contact like code `plot_cheng2020_fig2`

    Parameters
    ----------
    course_of_disease_data_list: List of course_of_disease_data object
    contact_data_list: List of contact_data object
    monte_carlo_number: Number of Monte-Carlo simulations. If the monte_carlo_number is 100, then there should be 100 files (index from 0 to 99) in the synthetic data folder (e.g. synthetic_data_results_cheng2020) for each data types (case_edge_list, demographic_data, contact_data, course_of_disease_data, and social_data).
    num_source_cases: Number of source cases. This is preseted in the Data_synthesis_main.py.
    layer: Contact type layer. 'All', 'Household', 'School', 'Workplace', 'Health care', 'Municipality'
    save_fig: True for saving the pdf figure

    Returns
    -------
    None
    '''
    current_palette = seaborn.color_palette()
    contact_day_shift_all = []
    exposure_day_array_all = []
    daily_contact_array_all = []
    infection_day_shift_all = []
    daily_infection_array_all = []
    attack_rate_all = []
    min_contact_day_shift = 0
    min_infection_day_shift = 0
    max_len_daily_contact_array = 0
    max_len_daily_infection_array = 0
    # Save Monte-carlo result
    for i in range(monte_carlo_number):
        contact_day_shift, exposure_day_array, daily_contact_array, infection_day_shift, \
            daily_infection_array, attack_rate = generate_contact_day_vs_infection_day_array(
                course_of_disease_data_list[i *
                                            num_source_cases:(i+1)*num_source_cases],
                contact_data_list[i*num_source_cases:(i+1)*num_source_cases], layer=layer)
        contact_day_shift_all.append(contact_day_shift)
        exposure_day_array_all.append(exposure_day_array)
        daily_contact_array_all.append(daily_contact_array)
        infection_day_shift_all.append(infection_day_shift)
        daily_infection_array_all.append(daily_infection_array)
        attack_rate_all.append(attack_rate)
        min_contact_day_shift = min(min_contact_day_shift, contact_day_shift)
        min_infection_day_shift = min(
            min_infection_day_shift, infection_day_shift)
        max_len_daily_contact_array = max(
            max_len_daily_contact_array, len(daily_contact_array))
        max_len_daily_infection_array = max(
            max_len_daily_infection_array, len(daily_infection_array))
        start_index = min(min_contact_day_shift, min_infection_day_shift)
        end_index = max(max_len_daily_contact_array,
                        max_len_daily_infection_array)
    time_array = np.arange(start_index, end_index+1)

    contact_matrix = np.zeros([monte_carlo_number, len(time_array)])
    infection_matrix = np.zeros([monte_carlo_number, len(time_array)])

    # Assign daily contact and daily infection
    for i in range(monte_carlo_number):
        assign_contact_start_index = int(
            contact_day_shift_all[i] - start_index)
        assign_infection_start_index = int(
            infection_day_shift_all[i] - start_index)
        contact_matrix[i, assign_contact_start_index:
                       assign_contact_start_index+len(daily_contact_array_all[i])] = \
            daily_contact_array_all[i]
        infection_matrix[i, assign_infection_start_index:
                         assign_infection_start_index+len(daily_infection_array_all[i])] = \
            daily_infection_array_all[i]

    attack_rate_matrix = np.divide(infection_matrix, contact_matrix, out=np.zeros_like(
        infection_matrix), where=contact_matrix != 0)*100
    # Calculate mean and 95% CI
    mean_contact_array = np.mean(contact_matrix, axis=0)
    mean_infection_array = np.mean(infection_matrix, axis=0)
    mean_attack_rate_array = np.mean(attack_rate_matrix, axis=0)
    ci_contact_array = np.percentile(contact_matrix, [2.5, 97.5], axis=0)
    ci_infection_array = np.percentile(infection_matrix, [2.5, 97.5], axis=0)
    ci_attack_rate_array = np.percentile(
        attack_rate_matrix, [2.5, 97.5], axis=0)

    # Plot
    if np.sum(mean_contact_array) == 0:  # No contact
        print('No contact!')
        # Empty plot
        fig, ax1 = plt.subplots()
        fig.set_figheight(2)
        fig.set_figwidth(16)
        ax2 = ax1.twinx()
        ax1.set_xlim([-21, 25+1])
        ax1.set_xlabel('Days from onset to contact')
        ax1.set_ylabel('%s contacs' % layer)
        ax1.grid()
        # ax1.set_yscale('log')
        ax1.set_ylim([0.1, 1000])

        ax2.set_ylabel('Secondary attack rate')
        ax2.spines['right'].set_color(current_palette[2])
        ax2.yaxis.label.set_color(current_palette[2])
        ax2.tick_params(axis='y', colors=current_palette[2])
        ax2.set_ylim([0, 0.5])
    else:
        # Plot contact bar chart
        fig, ax1 = plt.subplots()
        fig.set_figheight(2)
        fig.set_figwidth(16)
        ax1.set_xlim([-20, 30])
        ax2 = ax1.twinx()
        ax1.bar(time_array+0.2, mean_contact_array,
                color=current_palette[0], width=0.4)
        ax1.bar(time_array-0.2, mean_infection_array,
                color=current_palette[1], width=0.4)
        ax1.set_xlabel('Days from onset to contact')
        ax1.set_ylabel(f'{layer} contacs')
        # Plot uncertainty
        # Contacts
        y_l_tmp = mean_contact_array-ci_contact_array[0, :]
        y_u_tmp = ci_contact_array[1, :]-mean_contact_array
        y_u_tmp[y_u_tmp < 0] = 0
        ax1.errorbar(time_array+0.2, mean_contact_array, yerr=[y_l_tmp, y_u_tmp], color='k',
                     fmt='.', capsize=1, markeredgewidth=1)

        # Infection
        y_l_tmp = mean_infection_array - ci_infection_array[0, :]
        y_l_tmp[y_l_tmp < 0] = 0
        y_u_tmp = ci_infection_array[1, :] - mean_infection_array
        y_u_tmp[y_u_tmp < 0] = 0
        ax1.errorbar(time_array-0.2, mean_infection_array, yerr=[y_l_tmp, y_u_tmp],
                     color='k', fmt='.', capsize=1, markeredgewidth=1)

        # Plot secondary attack rate
        ax2.plot(time_array, mean_attack_rate_array,
                 '.--', color=current_palette[2])
        y_l_tmp = mean_attack_rate_array-ci_attack_rate_array[0, :]
        y_u_tmp = ci_attack_rate_array[1, :]-mean_attack_rate_array
        y_u_tmp[y_u_tmp < 0] = 0
        ax2.errorbar(time_array, mean_attack_rate_array, yerr=[y_l_tmp, y_u_tmp],
                     color=current_palette[2], fmt='.', capsize=1, markeredgewidth=1)
        ax2.set_ylabel('Clinical attack rate, %')
        ax2.spines['right'].set_color(current_palette[2])
        ax2.yaxis.label.set_color(current_palette[2])
        ax2.tick_params(axis='y', colors=current_palette[2])
        ax2.set_ylim([0, attack_y_limit])

        if save_fig == True:
            plt.savefig('RW2022_contact_day_vs_infection_day_%s.pdf' % layer)


def create_array_cheng2020_fig2(course_of_disease_data_list, contact_data_list, layer='All'):

    duration_array, infection_day_array, first_contact_day_array, incubation_period_array = \
        generate_course_and_contact_combine_data(
            course_of_disease_data_list, contact_data_list, layer=layer)
    if len(duration_array) == 0:  # No contacts
        adjust_first_contact_day_array = np.empty((0, 6))
        contact_array = np.empty((0, 6))
        infection_map = np.empty((0, 6))
        infection_array = np.empty((0, 6))
    else:
        adjust_first_contact_day_array = first_contact_day_array-incubation_period_array

        contact_array = np.array([np.count_nonzero(adjust_first_contact_day_array < 0),
                                  np.count_nonzero((adjust_first_contact_day_array >= 0) & (
                                      adjust_first_contact_day_array <= 3)),
                                  np.count_nonzero((adjust_first_contact_day_array >= 4) & (
                                      adjust_first_contact_day_array <= 5)),
                                  np.count_nonzero((adjust_first_contact_day_array >= 6) & (
                                      adjust_first_contact_day_array <= 7)),
                                  np.count_nonzero((adjust_first_contact_day_array >= 8) & (
                                      adjust_first_contact_day_array <= 9)),
                                  np.count_nonzero(adjust_first_contact_day_array > 9)])
        infection_map = ~np.isnan(infection_day_array)
        infection_array = np.array([np.count_nonzero((adjust_first_contact_day_array < 0)*infection_map),
                                    np.count_nonzero((adjust_first_contact_day_array >= 0)*infection_map & (
                                        adjust_first_contact_day_array <= 3)*infection_map),
                                    np.count_nonzero((adjust_first_contact_day_array >= 4)*infection_map & (
                                        adjust_first_contact_day_array <= 5)*infection_map),
                                    np.count_nonzero((adjust_first_contact_day_array >= 6)*infection_map & (
                                        adjust_first_contact_day_array <= 7)*infection_map),
                                    np.count_nonzero((adjust_first_contact_day_array >= 8)*infection_map & (
                                        adjust_first_contact_day_array <= 9)*infection_map),
                                    np.count_nonzero((adjust_first_contact_day_array > 9)*infection_map)])

    return (adjust_first_contact_day_array, contact_array, infection_map, infection_array)


def plot_cheng2020_fig2(course_of_disease_data_list, contact_data_list, monte_carlo_number=1, num_source_cases=100, layer='All', save_fig=True):
    '''
    Plot contact day vs infection day, x-axis: Days from onset to contact categorized as Cheng et al. 2020, y-axis: Number of contacts

    Parameters
    ----------
    course_of_disease_data_list: List of course_of_disease_data object
    contact_data_list: List of contact_data object
    monte_carlo_number: Number of Monte-Carlo simulations. If the monte_carlo_number is 100, then there should be 100 files (index from 0 to 99) in the synthetic data folder (e.g. synthetic_data_results_cheng2020) for each data types (case_edge_list, demographic_data, contact_data, course_of_disease_data, and social_data).
    num_source_cases: Number of source cases. This is preseted in the Data_synthesis_main.py.
    layer: Contact type layer. 'All', 'Household', 'School', 'Workplace', 'Health care', 'Municipality'
    save_fig: True for saving the pdf figure

    Returns
    -------
    None
    '''

    contact_matrix = np.empty([0, 6])
    infection_matrix = np.empty([0, 6])
    attack_rate_matrix = np.empty([0, 6])
    for i in range(monte_carlo_number):
        adjust_first_contact_day_array, contact_array, infection_map, infection_array = create_array_cheng2020_fig2(
            course_of_disease_data_list[i *
                                        num_source_cases:(i+1)*num_source_cases],
            contact_data_list[i*num_source_cases:(i+1)*num_source_cases], layer=layer)

        # Attack rate
        try:
            attack_rate_0 = np.count_nonzero((adjust_first_contact_day_array < 0)
                                             * infection_map)/np.count_nonzero(adjust_first_contact_day_array < 0)
        except:
            attack_rate_0 = 0
        try:
            attack_rate_1 = np.count_nonzero((adjust_first_contact_day_array >= 0)*infection_map & (
                adjust_first_contact_day_array <= 3)*infection_map)/np.count_nonzero((adjust_first_contact_day_array >= 0) &
                                                                                     (adjust_first_contact_day_array <= 3))
        except:
            attack_rate_1 = 0
        try:
            attack_rate_2 = np.count_nonzero((adjust_first_contact_day_array >= 4)*infection_map & (
                adjust_first_contact_day_array <= 5)*infection_map)/np.count_nonzero((adjust_first_contact_day_array >= 4) &
                                                                                     (adjust_first_contact_day_array <= 5))
        except:
            attack_rate_2 = 0
        try:
            attack_rate_3 = np.count_nonzero((adjust_first_contact_day_array >= 6)*infection_map & (
                adjust_first_contact_day_array <= 7)*infection_map)/np.count_nonzero((adjust_first_contact_day_array >= 6) &
                                                                                     (adjust_first_contact_day_array <= 7))
        except:
            attack_rate_3 = 0
        try:
            attack_rate_4 = np.count_nonzero((adjust_first_contact_day_array >= 8)*infection_map & (
                adjust_first_contact_day_array <= 9)*infection_map)/np.count_nonzero((adjust_first_contact_day_array >= 8) &
                                                                                     (adjust_first_contact_day_array <= 9))
        except:
            attack_rate_4 = 0
        try:
            attack_rate_5 = np.count_nonzero((adjust_first_contact_day_array > 9)
                                             * infection_map)/np.count_nonzero(adjust_first_contact_day_array > 9)
        except:
            attack_rate_5 = 0

        attack_rate = np.array([attack_rate_0, attack_rate_1,
                                attack_rate_2, attack_rate_3, attack_rate_4, attack_rate_5])*100

        contact_matrix = np.vstack([contact_matrix, contact_array])
        infection_matrix = np.vstack([infection_matrix, infection_array])
        attack_rate_matrix = np.vstack([attack_rate_matrix, attack_rate])

    current_palette = seaborn.color_palette()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    xticks = ['<0', '0-3', '4-5', '6-7', '8-9', '>9']
    # Average contact barchart
    mean_contact_number = np.mean(contact_matrix, axis=0)
    print(mean_contact_number)
    ax1.bar(np.arange(0.5, 5.5+1, 1), mean_contact_number,
            color=current_palette[0], width=0.5)
    # Plot contact uncertainty
    contact_lb = np.percentile(contact_matrix, 5, axis=0)
    contact_ub = np.percentile(contact_matrix, 95, axis=0)
    ax1.errorbar(np.arange(0.5, 5.5+1, 1), mean_contact_number,
                 yerr=[mean_contact_number-contact_lb,
                       contact_ub-mean_contact_number],
                 color='k', fmt='.', capsize=1, markeredgewidth=1)

    # Average infection case
    mean_infection_case = np.mean(infection_matrix, axis=0)
    ax1.bar(np.arange(0, 5+1, 1), mean_infection_case,
            color=current_palette[1], width=0.5)
    # Plot infection uncertainty
    infection_lb = np.percentile(infection_matrix, 5, axis=0)
    infection_ub = np.percentile(infection_matrix, 95, axis=0)

    ax1.errorbar(np.arange(0, 5+1, 1), mean_infection_case,
                 yerr=[mean_infection_case-infection_lb,
                       np.maximum(0, infection_ub-mean_infection_case)],
                 color='k', fmt='.', capsize=1, markeredgewidth=1)

    ax1.set_xticks(np.arange(6)+0.25)
    ax1.set_xticklabels(xticks)
    ax1.set_xlabel('Days from onset to first exposure')
    ax1.set_ylabel('Counts')

    # Average attack rate
    mean_attack_rate = np.mean(attack_rate_matrix, axis=0)
    print('mean_attack_rate: ', mean_attack_rate)
    ax2.plot(np.arange(6)+0.25, mean_attack_rate,
             '.--', color=current_palette[2])
    # Plot attack rate uncertainty
    attack_lb = np.percentile(attack_rate_matrix, 2.5, axis=0)
    attack_ub = np.percentile(attack_rate_matrix, 97.5, axis=0)
    ax2.errorbar(np.arange(6)+0.25, mean_attack_rate,
                 yerr=[mean_attack_rate-attack_lb,
                       np.maximum(0, attack_ub-mean_attack_rate)],
                 color=current_palette[2], fmt='.', capsize=5, markeredgewidth=1)

    # ax2.set_ylim([0, 1.5])
    ax2.set_ylabel('Clinical attack rate, %')
    ax2.spines['right'].set_color(current_palette[2])
    ax2.yaxis.label.set_color(current_palette[2])
    ax2.tick_params(axis='y', colors=current_palette[2])

    if save_fig == True:
        plt.savefig('RW2022_synthesis_%s_contact_bar_chart.pdf' % layer)

    print('Synthetic contact number: ', sum(mean_contact_number))

    return (ax1, ax2, (mean_contact_number, contact_lb, contact_ub, mean_infection_case, infection_lb, infection_ub, mean_attack_rate, attack_lb, attack_ub))



def correct_contact_time_ge2021(duration_array, first_contact_day_array, incubation_period_array):
    adjust_first_contact_day_array = first_contact_day_array - incubation_period_array
    # The time limit of ge2021's data is -14 to 10 days from symptom-onset to exposure
    for i in range(len(duration_array)):
        first_contact_day = adjust_first_contact_day_array[i]
        if first_contact_day < -14:
            time_shift = -14 - first_contact_day
            duration_array[i] = duration_array[i] - time_shift
            adjust_first_contact_day_array[i] = -14
        elif first_contact_day > 10:
            duration_array[i] = np.nan
            adjust_first_contact_day_array[i] = np.nan
        elif (first_contact_day >= -14) & (first_contact_day <= 10) & (first_contact_day+duration_array[i]-1 > 10):
            time_shift = first_contact_day + duration_array[i]-1 - 10
            duration_array[i] = duration_array[i] - time_shift
    first_contact_day_array = adjust_first_contact_day_array + incubation_period_array

    return (duration_array, first_contact_day_array)


def plot_ge2021_efig1(course_of_disease_data_list, contact_data_list, layer='All'):
    current_palette = seaborn.color_palette("pastel")
    duration_array, infection_day_array, first_contact_day_array, incubation_period_array = generate_course_and_contact_combine_data(
        course_of_disease_data_list, contact_data_list, layer=layer)
    duration_array, first_contact_day_array = correct_contact_time_ge2021(
        duration_array, first_contact_day_array, incubation_period_array)
    # Plot
    y_bottom = 1
    y_top = np.nanmax(duration_array)
    fig = plt.figure(figsize=(16, 10))
    ax = fig.subplots()
    adjust_first_contact_day_array = first_contact_day_array-incubation_period_array
    x_left = np.nanmin(adjust_first_contact_day_array)
    x_right = np.nanmax(adjust_first_contact_day_array)

    for x in np.arange(x_left, x_right+1):
        for y in np.arange(y_bottom, y_top+1):
            x_index = np.where(adjust_first_contact_day_array == x)[0]
            y_index = np.where(duration_array == y)[0]
            index = np.intersect1d(x_index, y_index)
            number_of_infection = np.count_nonzero(
                ~np.isnan(infection_day_array[index]))
            number_of_contacts = np.count_nonzero(
                incubation_period_array[index])
            # Different color for different cases
            if (number_of_infection != 0) & (number_of_contacts != 0):
                ax.annotate(
                    text=str(int(number_of_infection))+'/' +
                    str(int(number_of_contacts)),
                    xy=(x, y), horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc=current_palette[1], ec="k"), size=10)
            elif (number_of_infection == 0) & (number_of_contacts != 0):
                ax.annotate(
                    text=str(int(number_of_infection))+'/' +
                    str(int(number_of_contacts)),
                    xy=(x, y), horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc=current_palette[2], ec="k"), size=10)
            else:
                pass
    ax.set_xlim(x_left, x_right+1)
    ax.set_ylim(y_bottom, y_top+1)
    ax.grid()
    ax.set_xlabel('Relative time of 1st contact')
    ax.set_ylabel('Duration')


def plot_ge2021_fig2(course_of_disease_data_list, contact_data_list, layer='All'):
    current_palette = seaborn.color_palette()

    # Relative risk
    time, attack_rate = calculate_relative_risk(
        course_of_disease_data_list, contact_data_list, layer=layer)
    time_index = np.where((time >= -14) & (time <= 10))
    time = time[time_index]
    attack_rate = attack_rate[time_index]
    relative_risk = attack_rate/np.mean(attack_rate)

    duration_array, infection_day_array, first_contact_day_array, incubation_period_array = generate_course_and_contact_combine_data(
        course_of_disease_data_list, contact_data_list, layer=layer)
    duration_array, first_contact_day_array = correct_contact_time_ge2021(
        duration_array, first_contact_day_array, incubation_period_array)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    adjust_first_contact_day_array = first_contact_day_array-incubation_period_array
    ax1.plot(time, relative_risk, '.', color='k')
    ax1.set_ylabel('Relative risk')
    ax1.set_xlabel(
        'Time from index patient symptom onset to index patient-contact exposure')
    ax2.yaxis.label.set_color('k')
    ax2.tick_params(axis='y', colors='k')
    for i in np.arange(np.nanmin(adjust_first_contact_day_array), np.nanmax(adjust_first_contact_day_array)+1):
        ax2.bar(i, np.count_nonzero(
            adjust_first_contact_day_array == i), color=current_palette[0])
    ax2.set_xlabel(
        'Time from index patient symptom onset to index patient–contact exposure')
    ax2.set_ylabel('No. of close contacts')
    ax2.yaxis.label.set_color(current_palette[0])
    ax2.tick_params(axis='y', colors=current_palette[0])


def plot_contact_stack_plot(contact_data_list, layer='All', save_fig=False):
    contact_date_stack_matrix, index = generate_stack_contact_matrix(
        contact_data_list, layer=layer)

    # Stack plot
    contact_date_stack_matrix[contact_date_stack_matrix == 0] = 0
    plt.figure(figsize=(16, 4))
    plt.stackplot(np.arange(np.shape(contact_date_stack_matrix)
                  [1]), contact_date_stack_matrix)
    plt.xlabel('Day since infection')
    plt.xticks(range(np.shape(contact_date_stack_matrix)[1]))
    plt.ylabel('%s contacts' % layer)
    # plt.title(layer)
    plt.grid()

    # Plot cluster line
    line_style = ['-']
    plot_index = 0
    for i in index:
        if np.sum(contact_date_stack_matrix[0:i, :]) == 0:
            pass
        else:
            plt.plot(np.sum(contact_date_stack_matrix[0:i, :], axis=0),
                     color='k', linewidth=1, linestyle=line_style[plot_index % len(line_style)])
            plot_index += 1

    if save_fig == True:
        plt.savefig('RW2022_%s_contact_stack_plot.pdf' % layer)


####################################################################################################
# Code for reproduce other papers plots using their data
####################################################################################################


def plot_cheng2020_bar_chart(layer='All', save_fig=False):
    current_palette = seaborn.color_palette()
    # Data extracted from table 3
    household_contact = np.array([100, 39, 6, 4, 2, 0])
    household_infected_contacts = np.array([4, 2, 1, 0, 0, 0])
    household_attack_rate = np.array([4, 5.1, 16.7, 0, 0, 0])
    household_lower_bound = np.array([1.6, 1.4, 3, 0, 0, 0])
    household_upper_bound = np.array([9.8, 16.8, 56.4, 49, 65.7, 100])

    non_household_contact = np.array([10, 15, 6, 10, 3, 24])
    non_household_infected_contacts = np.array([1, 3, 0, 0, 0, 0])
    non_household_attack_rate = np.array([10, 20, 0, 0, 0, 0])
    non_household_lower_bound = np.array([1.8, 7, 0, 0, 0, 0])
    non_household_upper_bound = np.array([40.4, 45.2, 39, 27.8, 56.1, 13.8])

    health_care_contact = np.array([236, 150, 38, 17, 110, 146])
    health_care_infected_contacts = np.array([2, 3, 1, 0, 0, 0])
    health_care_attack_rate = np.array([0.8, 2, 2.6, 0, 0, 0])
    health_care_lower_bound = np.array([0.2, 0.7, 0.5, 0, 0, 0])
    health_care_upper_bound = np.array([3, 5.7, 13.5, 18.4, 3.3, 2.6])

    other_contact = np.array([389, 663, 166, 88, 334, 114])
    other_infected_contacts = np.array([0, 0, 1, 0, 0, 0])
    other_attack_rate = np.array([0, 0, 0.6, 0, 0, 0])
    other_lower_bound = np.array([0, 0, 0.1, 0, 0, 0])
    other_upper_bound = np.array([1, 0.6, 3.3, 4.2, 1.1, 3.3])

    all_attack_rate = np.array([1, 0.9, 1.4, 0, 0, 0])
    all_lower_bound = np.array([0.5, 0.5, 0.5, 0, 0, 0])
    all_upper_bound = np.array([2, 1.8, 4, 3.1, 0.9, 1.3])

    # Define layer for ploting
    if (layer == 'Household'):
        contacts = household_contact
        infected_contacts = household_infected_contacts
        attack_rate = household_attack_rate
        lower_bound = household_lower_bound
        upper_bound = household_upper_bound
    elif (layer == 'Non household family'):
        contacts = non_household_contact
        infected_contacts = non_household_infected_contacts
        attack_rate = non_household_attack_rate
        lower_bound = non_household_lower_bound
        upper_bound = non_household_upper_bound
    elif (layer == 'Health care'):
        contacts = health_care_contact
        infected_contacts = health_care_infected_contacts
        attack_rate = health_care_attack_rate
        lower_bound = health_care_lower_bound
        upper_bound = health_care_upper_bound
    elif (layer == 'Others'):
        contacts = other_contact
        infected_contacts = other_infected_contacts
        attack_rate = other_attack_rate
        lower_bound = other_lower_bound
        upper_bound = other_upper_bound
    elif (layer == 'All'):
        contacts = household_contact + non_household_contact + \
            health_care_contact + other_contact
        infected_contacts = household_infected_contacts + non_household_infected_contacts + \
            health_care_infected_contacts + other_infected_contacts
        attack_rate = all_attack_rate
        lower_bound = all_lower_bound
        upper_bound = all_upper_bound
    else:
        print(
            'layers={Household, Non household family, Health care, Others, All}')
    # print('Contact number: ', sum(contacts))

    # Plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    xticks = ['<0', '0-3', '4-5', '6-7', '8-9', '>9']
    # Contacts
    ax1.bar(np.arange(0.5, 5.5+1), contacts,
            color=current_palette[0], width=0.5)

    # Infected contacts
    ax1.bar(np.arange(0, 5+1), infected_contacts,
            color=current_palette[1], width=0.5)
    ax1.set_xticks(np.arange(6)+0.25)
    ax1.set_xticklabels(xticks)
    ax1.set_xlabel('Days from onset to exposure')
    # ax1.set_ylabel('Counts')

    # Attack rate
    ax2.errorbar(np.arange(6)+0.25, attack_rate, yerr=[attack_rate-lower_bound, upper_bound-attack_rate],
                 color=current_palette[2], fmt='.--', capsize=5, markeredgewidth=1)
    ax2.set_ylabel('Clinical attack rate, %')
    ax2.spines['right'].set_color(current_palette[2])
    ax2.yaxis.label.set_color(current_palette[2])
    ax2.tick_params(axis='y', colors=current_palette[2])

    if save_fig == True:
        plt.savefig('RW2022_cheng_%s_bar_chart.pdf' % layer)

    print('Taiwan contact number: ', sum(contacts))

    return (ax1, ax2, (contacts, infected_contacts, attack_rate, lower_bound, upper_bound))
