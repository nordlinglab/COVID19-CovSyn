import numpy as np


def R0_average_effective_contact(contact_data, layer='all', method='mean'):
    household_contact_array = np.zeros(len(contact_data))
    school_effective_contact_array = np.zeros(len(contact_data))
    workplace_effective_contact_array = np.zeros(len(contact_data))
    health_care_effective_contact_array = np.zeros(len(contact_data))
    municipality_effective_contact_array = np.zeros(len(contact_data))
    for i in range(len(contact_data)):
        household_contact_array[i] = np.sum(
            contact_data[i]['household_effective_contacts'])
        school_effective_contact_array[i] = np.sum(
            contact_data[i]['school_effective_contacts'])
        workplace_effective_contact_array[i] = np.sum(
            contact_data[i]['workplace_effective_contacts'])
        health_care_effective_contact_array[i] = np.sum(
            contact_data[i]['health_care_effective_contacts'])
        municipality_effective_contact_array[i] = np.sum(
            contact_data[i]['municipality_effective_contacts'])

    all_contact_array = household_contact_array + school_effective_contact_array + \
        workplace_effective_contact_array + health_care_effective_contact_array + \
        municipality_effective_contact_array
    if layer == 'all':
        if method == 'mean':
            R0 = np.mean(all_contact_array)
        elif method == 'median':
            R0 = np.median(all_contact_array)
    elif layer == 'household':
        if method == 'mean':
            R0 = np.mean(household_contact_array)
        elif method == 'median':
            R0 = np.median(household_contact_array)
    elif layer == 'school':
        if method == 'mean':
            R0 = np.mean(school_effective_contact_array)
        elif method == 'median':
            R0 = np.median(school_effective_contact_array)
    elif layer == 'workplace':
        if method == 'mean':
            R0 = np.mean(workplace_effective_contact_array)
        elif method == 'median':
            R0 = np.median(workplace_effective_contact_array)
    elif layer == 'health_care':
        if method == 'mean':
            R0 = np.mean(health_care_effective_contact_array)
        elif method == 'median':
            R0 = np.median(health_care_effective_contact_array)
    elif layer == 'municipality':
        if method == 'mean':
            R0 = np.mean(municipality_effective_contact_array)
        elif method == 'median':
            R0 = np.median(municipality_effective_contact_array)
    # print('Saving all contact array')
    # np.save('all_contact_array.npy', all_contact_array)

    return (R0)


def R0_mc_average_effective_contact(contact_data, course_of_disease_data, time_threshold, subject_num, repeat_num=100):
    index = []
    for i in range(len(course_of_disease_data)):
        if course_of_disease_data[i].positive_test_date <= time_threshold:
            index.append(i)

    # Extract contact_data based on the index
    sub_contact_data = [contact_data[i] for i in index]
    if len(contact_data) < subject_num:
        print('Not enough data or too high subject number')
    else:
        rng = np.random.default_rng()
        R0_values = []
        for _ in range(repeat_num):
            sampled_indices = rng.choice(
                len(sub_contact_data), size=subject_num, replace=True)
            sampled_contacts = [sub_contact_data[i] for i in sampled_indices]
            R0 = R0_average_effective_contact(sampled_contacts)
            R0_values.append(R0)
        return np.mean(R0_values)

# def R0_network(edge_list):
# Sibo Todos: complete this function
    # return R0
