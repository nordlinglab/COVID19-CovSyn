import copy
import ast
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from bidict import bidict
from lifelines import KaplanMeierFitter
from numpy.random import default_rng

################################################################################
# Taiwan data related functions
################################################################################


# def clean_taiwan_data(taiwan_data_sheet, start_index=1, end_index=579):
#     # Data cleaning for pandas Taiwan COVID-19 data

#     # Correct ID 19's ICU time
#     data = taiwan_data_sheet.copy()
#     case19_icu_date = data[data.id == 19].icu.values
#     case19_index = data[data.id == 19].index
#     data.loc[case19_index, 'confirmed_date'] = case19_icu_date

#     # Set range
#     table_start_index = data[data['id'] == start_index].index.to_numpy()[
#         0]
#     table_end_index = data[data['id'] == end_index].index.to_numpy()[
#         0]
#     data = data.iloc[table_end_index:table_start_index+1]

#     # Add asymptomatic
#     data = add_asymptomatic_date(data)

#     # Replace symptomatic to death to be symptomatic to critical and directly to dead
#     for i in data.index:
#         if (data.loc[i].icu == 'C' or data.loc[i].icu == 'X'):
#             if (data.loc[i].death_date != 'C') and (data.loc[i].death_date != 'X'):
#                 print(data.loc[i].icu)
#                 print(data.loc[i].death_date)
#                 data.loc[i].icu = data.loc[i].death_date
#                 print(data.loc[i].icu)

#     return (data)

def clean_taiwan_data(taiwan_data_sheet, start_index=1, end_index=579):
    # Data cleaning for pandas Taiwan COVID-19 data

    # Clean out ID 530 since it was removed by Taiwan CDC
    if end_index >= 530:
        taiwan_data_sheet = taiwan_data_sheet[taiwan_data_sheet.id != 530]

    # Correct ID 19's ICU time
    data = taiwan_data_sheet.copy()
    case19_icu_date = data[data.id == 19].icu.values
    case19_index = data[data.id == 19].index
    data.loc[case19_index, 'confirmed_date'] = case19_icu_date

    # Set range
    table_start_index = data[data['id'] == start_index].index.to_numpy()[0]
    table_end_index = data[data['id'] == end_index].index.to_numpy()[0]
    data = data.iloc[table_end_index:table_start_index+1]

    # Add asymptomatic
    data = add_asymptomatic_date(data)

    # Replace symptomatic to death to be symptomatic to critical and directly to dead
    for i in data.index:
        if (data.loc[i, 'icu'] == 'C' or data.loc[i, 'icu'] == 'X'):
            if (data.loc[i, 'death_date'] != 'C') and (data.loc[i, 'death_date'] != 'X'):
                data.loc[i, 'icu'] = data.loc[i, 'death_date']

    return data


def add_asymptomatic_date(data):
    confirm_date = data.confirmed_date
    symptom_onset_date = data.onset_of_symptom
    infection_date = data.earliest_infection_date
    source_id = data.source_infected_case
    source_id = source_id.str.replace('ID ', '')
    asymptomatic_date = pd.DataFrame(
        {'asymptomatic_date': np.zeros(len(source_id))}, index=data.index)

    # Create array of first spread date based on infection date and source ID
    first_spread_date = pd.DataFrame(
        {'first_spread_date': np.zeros(len(source_id))}, index=data.index)
    for i in source_id.unique():
        if type(i)==str:
            first_spread_date_temp = min(infection_date[source_id == i])
            first_spread_date.loc[data[data.id ==
                                    int(i)].index[0]] = first_spread_date_temp
            first_spread_date = first_spread_date.replace(0, 'C')
    all_date = pd.concat(
        [symptom_onset_date, confirm_date, first_spread_date, infection_date], axis=1)
    # all_date = pd.concat([infection_date], axis=1)
    all_date = all_date.replace('C', np.nan)
    all_date = all_date.replace('N', np.nan)
    all_date = all_date.replace('X', np.nan)
    all_date = all_date.replace('Mid of October', np.nan)
    all_date = all_date.replace('2020-04-26 to 2020-04-27', np.nan)
    all_date = all_date.replace('2020-04-14 to 2020-04-18', np.nan)
    all_date = all_date.replace('2020-04-06, 2020-04-07', np.nan)
    all_date = all_date.replace('2020-01-28 to 2020-02-06', np.nan)

    # Find min date and assign it as asymptomatic date
    for k in all_date.index:
        all_date_row = all_date.loc[k].dropna()
        if len(all_date_row) > 0:
            asymptomatic_date.loc[k] = all_date_row.min()
    asymptomatic_date = asymptomatic_date.replace(0, 'C')

    result = pd.concat([data, asymptomatic_date], axis=1)
    return (result)


def generate_taiwan_contact_network(data):
    '''
    Generate contact network from Taiwan COVID-19 data. The network is undirected and include both the infected and uninfected contacts.

    Parameters
    ----------
    data: pandas dataframe. Taiwan COVID-19 individual subject data.

    Returns
    -------
    contact_network: networkx graph
    contact_type_dict: dictionary for effective contact type
    uninfected_contact_type_dict: dictionary for uninfected contact type
    '''
    # Initialization
    contact_network = nx.MultiGraph()
    contact_type_start_index = data.columns.get_loc('couple')
    contact_type_end_index = data.columns.get_loc('other_unknown_contact')
    uninfected_contact_type_start_index = data.columns.get_loc(
        'number_of_uninfected_contact_travel')
    uninfected_contact_type_end_index = data.columns.get_loc(
        'number_of_uninfected_contact_friend')

    # Construct contact type label
    contact_type_dict = {}
    uninfected_contact_type_dict = {}
    for i, column in enumerate(data.columns[contact_type_start_index:contact_type_end_index+1]):
        contact_type_dict[column] = i
    contact_type_dict = bidict(contact_type_dict)
    contact_type_final_index = contact_type_dict['other_unknown_contact']
    # Add uninfected contact type
    uninfected_contact_type_dict['number_of_uninfected_contact_travel'] = contact_type_dict['travel_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight'] = contact_type_dict['the_same_flight']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight_nearby_seats'] = contact_type_dict['the_same_flight_nearby_seat']
    uninfected_contact_type_dict['number_of_uninfected_contact_car'] = contact_type_dict['the_same_car']
    # New label
    uninfected_contact_type_dict['number_of_uninfected_contact_ship'] = contact_type_final_index + 1
    contact_type_final_index += 1
    uninfected_contact_type_dict['number_of_uninfected_contact_live_together'] = contact_type_dict['live_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_family'] = contact_type_dict['family']
    uninfected_contact_type_dict['number_of_uninfected_contact_coworker'] = contact_type_dict['coworker']
    uninfected_contact_type_dict['number_of_uninfected_contact_others'] = contact_type_dict['other_unknown_contact']
    uninfected_contact_type_dict['number_of_uninfected_contact_hospital'] = contact_type_dict['the_same_hospital']
    uninfected_contact_type_dict['number_of_uninfected_contact_quarantine_hotel'] = contact_type_dict['the_same_quarantine_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_hotel'] = contact_type_dict['the_same_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_school'] = contact_type_dict['the_same_school']
    uninfected_contact_type_dict['number_of_uninfected_contact_couple'] = contact_type_dict['couple']
    uninfected_contact_type_dict['number_of_uninfected_contact_panshi'] = contact_type_dict['panshi_fast_combat_support_ship']
    uninfected_contact_type_dict['number_of_uninfected_contact_friend'] = contact_type_dict['friend']
    uninfected_contact_type_dict = bidict(uninfected_contact_type_dict)

    # Construct contact network
    uninfected_id = 0
    for i, row in data.iterrows():
        row_id = row['id']

        # Loop effective contact columns
        for j in range(contact_type_start_index, contact_type_end_index+1):
            if (row[j] != 'C') and (row[j] != 'X'):  # Effective contact exist
                # Extract effective contact ID
                effective_contact_id_tmp = row[j]
                effective_contact_id_tmp = effective_contact_id_tmp.replace(
                    'ID ', '').replace(' ', '').split(',')
                effective_contact_id_tmp = np.array(
                    effective_contact_id_tmp).astype(int)
                # Extract contact type
                contact_type_tmp = row.index[j]
                contact_type_tmp = contact_type_dict[contact_type_tmp]
                # Add edge
                for effective_contact_id in effective_contact_id_tmp:
                    # Test if the contact already exist
                    try:
                        all_previous_contacts = [contact['contact_type'] for contact in contact_network['I'+str(
                            effective_contact_id)]['I'+str(row_id)].values()]
                    except:
                        all_previous_contacts = []
                    if contact_type_tmp in all_previous_contacts:  # Edge already exist
                        pass
                    else:
                        contact_network.add_edge(
                            'I'+str(row_id), 'I'+str(effective_contact_id), contact_type=contact_type_tmp)

        # Loop uninfected contact columns
        for k in np.arange(uninfected_contact_type_start_index, uninfected_contact_type_end_index+1, 2):
            if (row[k] != 'C') and (row[k] != 'X'):  # Uninfected contact exist
                # Extract uninfected contact ID
                uninfected_contact_type_tmp = row.index[k]
                uninfected_contact_type_tmp = uninfected_contact_type_dict[
                    uninfected_contact_type_tmp]
                # Extract saved linked nodes
                neighbors = []
                try:
                    for neighbor, attributes in contact_network['I'+str(row_id)].items():
                        for attribute in attributes.values():
                            if attribute['contact_type'] == uninfected_contact_type_tmp:
                                neighbors.append(neighbor)
                except:
                    pass
                count_integers = len(
                    [x for x in neighbors if isinstance(x, np.integer)])
                if count_integers == 0:
                    uninfected_id_array = np.arange(
                        uninfected_id, uninfected_id+row[k])
                    for l in uninfected_id_array:
                        contact_network.add_edge(
                            'I'+str(row_id), l, contact_type=uninfected_contact_type_tmp)
                    if (row[k+1] != 'C') and (row[k+1] != 'X'):  # Intersection exist
                        intersection = row[k+1]
                        intersection = ast.literal_eval(intersection)
                        intersection_number = intersection[0]
                        intersection_id = intersection[1::]
                        for l in uninfected_id_array[0:intersection_number]:
                            for m in intersection_id:
                                contact_network.add_edge(
                                    'I'+str(m), l, contact_type=uninfected_contact_type_tmp)
                else:
                    extra_cases = row[k] - count_integers
                    if extra_cases > 0:
                        uninfected_id_array = np.arange(
                            uninfected_id, uninfected_id+extra_cases)
                        for l in uninfected_id_array:
                            contact_network.add_edge(
                                'I'+str(row_id), l, contact_type=uninfected_contact_type_tmp)
                        if (row[k+1] != 'C') and (row[k+1] != 'X'):  # Intersection exist
                            intersection = row[k+1]
                            intersection = ast.literal_eval(intersection)
                            intersection_number = intersection[0]
                            intersection_id = intersection[1::]
                            for l in uninfected_id_array[0:intersection_number]:
                                for m in intersection_id:
                                    contact_network.add_edge(
                                        'I'+str(m), l, contact_type=uninfected_contact_type_tmp)
                uninfected_id = uninfected_id_array[-1]+1

    return (contact_network, contact_type_dict, uninfected_contact_type_dict)


def generate_taiwan_infection_contact_network(data):
    '''
    Generate contact network for the infection path from Taiwan COVID-19 data. The network is directed and include only infected cases.

    Parameters
    ----------
    data: pandas dataframe. Taiwan COVID-19 individual subject data.

    Returns
    -------
    infection_contact_network: networkx graph
    contact_type_dict: dictionary for effective contact type
    uninfected_contact_type_dict: dictionary for uninfected contact type
    '''
    # Initialization
    infection_contact_network = nx.DiGraph()
    contact_type_start_index = data.columns.get_loc('couple')
    contact_type_end_index = data.columns.get_loc('other_unknown_contact')
    uninfected_contact_type_start_index = data.columns.get_loc(
        'number_of_uninfected_contact_travel')
    uninfected_contact_type_end_index = data.columns.get_loc(
        'number_of_uninfected_contact_friend')

    # Construct contact type label
    contact_type_dict = {}
    uninfected_contact_type_dict = {}
    for i, column in enumerate(data.columns[contact_type_start_index:contact_type_end_index+1]):
        contact_type_dict[column] = i
    contact_type_dict = bidict(contact_type_dict)
    contact_type_final_index = contact_type_dict['other_unknown_contact']
    # Add uninfected contact type
    uninfected_contact_type_dict['number_of_uninfected_contact_travel'] = contact_type_dict['travel_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight'] = contact_type_dict['the_same_flight']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight_nearby_seats'] = contact_type_dict['the_same_flight_nearby_seat']
    uninfected_contact_type_dict['number_of_uninfected_contact_car'] = contact_type_dict['the_same_car']
    # New label
    uninfected_contact_type_dict['number_of_uninfected_contact_ship'] = contact_type_final_index + 1
    contact_type_final_index += 1
    uninfected_contact_type_dict['number_of_uninfected_contact_live_together'] = contact_type_dict['live_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_family'] = contact_type_dict['family']
    uninfected_contact_type_dict['number_of_uninfected_contact_coworker'] = contact_type_dict['coworker']
    uninfected_contact_type_dict['number_of_uninfected_contact_others'] = contact_type_dict['other_unknown_contact']
    uninfected_contact_type_dict['number_of_uninfected_contact_hospital'] = contact_type_dict['the_same_hospital']
    uninfected_contact_type_dict['number_of_uninfected_contact_quarantine_hotel'] = contact_type_dict['the_same_quarantine_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_hotel'] = contact_type_dict['the_same_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_school'] = contact_type_dict['the_same_school']
    uninfected_contact_type_dict['number_of_uninfected_contact_couple'] = contact_type_dict['couple']
    uninfected_contact_type_dict['number_of_uninfected_contact_panshi'] = contact_type_dict['panshi_fast_combat_support_ship']
    uninfected_contact_type_dict['number_of_uninfected_contact_friend'] = contact_type_dict['friend']
    uninfected_contact_type_dict = bidict(uninfected_contact_type_dict)

    # Construct contact network
    uninfected_id = 0
    for i, row in data.iterrows():
        row_id = row['id']
        source_id = row['source_infected_case']
        if (source_id != 'C') and (source_id != 'X'):  # Source ID available
            source_id = np.int32(source_id.replace('ID ', ''))
            # Loop effective contact columns
            for j in range(contact_type_start_index, contact_type_end_index+1):
                if (row[j] != 'C') and (row[j] != 'X'):  # Effective contact exist
                    # Extract effective contact ID
                    effective_contact_id_tmp = row[j]
                    effective_contact_id_tmp = np.array(effective_contact_id_tmp.replace(
                        'ID ', '').replace(' ', '').split(','), dtype=int)
                    if source_id in effective_contact_id_tmp:  # Effective contact from source ID
                        contact_type = row.index[j]
                        contact_type = contact_type_dict[contact_type]
                        infection_contact_network.add_edge(
                            'I'+str(source_id), 'I'+str(row_id), contact_type=contact_type)

    return (infection_contact_network, contact_type_dict, uninfected_contact_type_dict)


def generate_edge_list(taiwan_contact_network, infection_contact_network=None):
    '''
    Generate edge list (mixed graph) for Taiwanese individual subject COVID-19 data.

    Parameters
    ----------
    taiwan_contact_network: networkx graph. This network is undirected and include both infected and uninfected cases.
    infection_contact_network: networkx graph. This network is directed and include only infected cases.

    Returns
    -------
    edge_list: list containing [source_id, target_id, contact_type, 'directed/undirected']
    '''
    edge_list = []

    # Construct edge list for the undirected network first
    for source_id, target_id, contact_type in taiwan_contact_network.edges(data=True):
        contact_type = contact_type['contact_type']
        edge_list.append([source_id, target_id, contact_type, False])

    # Merge the directed and undirected network
    if infection_contact_network is not None:
        for source_id, target_id, contact_type in infection_contact_network.edges(data=True):
            contact_type = contact_type['contact_type']
            # Search the row in the edge list
            for i in range(len(edge_list)):
                if edge_list[i][0] == source_id and edge_list[i][1] == target_id:  # Normal case
                    edge_list[i][-1] = True
                elif edge_list[i][1] == source_id and edge_list[i][0] == target_id:  # Reverse case
                    edge_list[i][0], edge_list[i][1] = edge_list[i][1], edge_list[i][0]
                    edge_list[i][-1] = True
                else:
                    continue

    return (edge_list)


def write_cytoscape_file(edge_list, filename):
    # write_cytoscape_file: Create cytoscape file based on edge list. Daily information is ignored.
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'interaction', 'directed'])
        for row in edge_list:
            source = row[0]
            target = row[1]
            group = row[2]
            direct = row[-1]
            writer.writerow([source, target, group, direct])


def extract_state_data(data, start_state, end_state, exclude_state=None):
    """
    Extract data from a pandas dataframe for a given state and calculate transition days.

    Parameters
    ----------
    data: pandas DataFrame
        The pandas dataframe to extract the data from.
    start_state: str
        String header of start state such as 'onset_of_symptom'.
    end_state: str
        String header of end state such as 'recovery'
    exclude_state: str, optional (default=None)
        String header of state, such as 'icu', such that the cases with this state
        should be excluded.

    Returns
    -------
    pandas.Series
        Series containing the number of days between start_state and end_state for valid cases.
    """
    # Input validation
    required_columns = [start_state, end_state]
    if exclude_state is not None:
        required_columns.append(exclude_state)

    if not all(col in data.columns for col in required_columns):
        missing_cols = [
            col for col in required_columns if col not in data.columns]
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    # Create boolean mask for valid entries
    valid_start = data[start_state].notna()
    valid_end = data[end_state].notna()

    if exclude_state is not None:
        # Exclude cases where exclude_state is not NaN
        exclude_mask = data[exclude_state].isna()
        row_mask = valid_start & valid_end & exclude_mask
    else:
        row_mask = valid_start & valid_end

    # Calculate transition days for valid cases
    start_dates = pd.to_datetime(data.loc[row_mask, start_state])
    end_dates = pd.to_datetime(data.loc[row_mask, end_state])
    transition_days = end_dates - start_dates

    return transition_days


def convert_Taiwan_data_to_test_matrix(data):
    '''
    Convert Taiwan data to test matrix for energy distance calculation.

    Parameters
    ----------
    data: pandas dataframe
        The pandas dataframe to extract the data from.

    Returns
    -------
    data_matrix: pandas dataframe
        Each column represents
            0. age,
            1. gender,
            2. time from infection to symptomatic,
            3. time from infection to recovery,
            4. time from symptomatic to critically ill,
            5. time from symptomatic to recovery,
            6. time from critically ill to recovery,
            7. time from critically ill to death,
            8. time from first negative test to confirmed,
            9. time from symptomatic to confirmed
    '''
    data_matrix = pd.DataFrame(
        index=data.index, columns=range(10), dtype=float)

    # Age
    ages = copy.deepcopy(data.age)
    ages[ages == '0 to 9'] = 5
    ages[ages == '10 to 19'] = 15
    ages[ages == '20 to 29'] = 25
    ages[ages == '30 to 39'] = 35
    ages[ages == '40 to 49'] = 45
    ages[ages == '50 to 59'] = 55
    ages[ages == '60 to 69'] = 65
    ages[ages == '70 to 79'] = 75
    ages[ages == '80 to 89'] = 85
    ages[(ages == 'C') | (ages == 'X')] = np.nan
    data_matrix[0] = ages

    # Gender, Male=1, Female=0
    genders = copy.deepcopy(data.gender.to_numpy())
    genders[genders == 'Male'] = 1
    genders[genders == 'Female'] = 0
    genders[(genders == 'C') | (genders == 'X')] = np.nan
    data_matrix[1] = genders

    # Course of disease
    # Infection to symptomatic
    infection_to_symptomatic_days = extract_state_data(data, 'earliest_infection_date',
                                                       'onset_of_symptom', 'recovery')
    infection_to_symptomatic_days_index = infection_to_symptomatic_days.index.to_numpy()
    data_matrix.loc[infection_to_symptomatic_days_index, 2] = pd.to_timedelta(
        infection_to_symptomatic_days).dt.days.to_numpy()

    # Infection to recovered
    infection_to_recovered_days = extract_state_data(data, 'earliest_infection_date',
                                                     'recovery', 'onset_of_symptom')
    infection_to_recovered_days_index = infection_to_recovered_days.index.to_numpy()
    data_matrix.loc[infection_to_recovered_days_index,
                    3] = pd.to_timedelta(infection_to_recovered_days).dt.days.to_numpy()

    # Symptomatic to critically ill
    symptom_to_critically_ill_days = extract_state_data(
        data, 'onset_of_symptom', 'icu', 'recovery')
    symptom_to_dead_days = extract_state_data(
        data, 'onset_of_symptom', 'death_date', 'icu')
    critically_ill_to_dead_days = extract_state_data(
        data, 'icu', 'death_date', 'recovery')
    # Replace symptomatic to death to be symptomatic to critical and directly to dead
    for i in symptom_to_dead_days.index:
        symptom_to_critically_ill_days[i] = symptom_to_dead_days[i]
        critically_ill_to_dead_days[i] = pd.Timedelta(days=0)
        symptom_to_dead_days[i] = pd.NaT
    symptom_to_critically_ill_days_index = symptom_to_critically_ill_days.index.to_numpy()
    data_matrix.loc[symptom_to_critically_ill_days_index,
                    4] = pd.to_timedelta(symptom_to_critically_ill_days).dt.days.to_numpy()

    # Symptomatic to recovered
    symptom_to_recover_days = extract_state_data(
        data, 'onset_of_symptom', 'recovery', 'icu')
    symptom_to_recover_days_index = symptom_to_recover_days.index.to_numpy()
    data_matrix.loc[symptom_to_recover_days_index,
                    5] = pd.to_timedelta(symptom_to_recover_days).dt.days.to_numpy()

    # Critically ill to recovered
    critically_ill_to_recover_days = extract_state_data(
        data, 'icu', 'recovery', 'death_date')
    critically_ill_to_recover_days_index = critically_ill_to_recover_days.index.to_numpy()
    data_matrix.loc[critically_ill_to_recover_days_index,
                    6] = pd.to_timedelta(critically_ill_to_recover_days).dt.days.to_numpy()

    # Critically ill to death
    critically_ill_to_dead_days_index = critically_ill_to_dead_days.index.to_numpy()
    data_matrix.loc[critically_ill_to_dead_days_index,
                    7] = pd.to_timedelta(critically_ill_to_dead_days).dt.days.to_numpy()

    # First negative test date to confirmed date
    negative_test_date_1_to_confrimed = extract_state_data(
        data, 'negative_test_date_1', 'confirmed_date')
    negative_test_date_1_to_confrimed_index = negative_test_date_1_to_confrimed.index.to_numpy()
    data_matrix.loc[negative_test_date_1_to_confrimed_index,
                    8] = pd.to_timedelta(negative_test_date_1_to_confrimed).dt.days.to_numpy()

    # Symptomatic to confirmed
    symptomatic_to_confrimed = extract_state_data(
        data, 'onset_of_symptom', 'confirmed_date')
    symptomatic_to_confrimed_index = symptomatic_to_confrimed.index.to_numpy()
    data_matrix.loc[symptomatic_to_confrimed_index,
                    9] = pd.to_timedelta(symptomatic_to_confrimed).dt.days.to_numpy()

    return data_matrix.to_numpy(dtype=np.float32)


################################################################################
#  Synthetic data related functions
################################################################################

def transform_course_object_to_population_data(course_of_disease_data_list, contact_data_list, time_limit, detection_rate=1, death_detection_rate=1, population_size=213032):
    """
    Transforms individual-level course of disease data into aggregated population-level statistics.

    This function processes a list of individual course of disease objects and aggregates them into
    daily population-level metrics, accounting for detection rates of cases and deaths.

    Parameters
    ----------
    course_of_disease_data_list : list
        List of objects containing individual patient course of disease data.
        Each object should contain attributes for infection dates, symptoms onset,
        test results, critical illness, recovery, and death dates.

    contact_data_list: list
        List of objects containing individual patient contact data.

    time_limit : int
        Maximum number of days to include in the time series output.
        Must not exceed the original simulation time period.

    detection_rate : float, optional
        Probability of detecting/confirming a case, between 0 and 1.
        Default is 1 (100% detection).

    death_detection_rate : float, optional
        Probability of detecting/recording a death, between 0 and 1.
        Default is 1 (100% detection).

    Returns
    -------
    tuple
        Seven numpy arrays representing daily counts for:
        - Susceptible population 
        - Infected cases (new infections per day)
        - Contagious cases (new contagious individuals per day)
        - Symptomatic cases (new symptomatic cases per day)
        - Confirmed cases (new confirmed diagnoses per day)
        - Tested cases (new tests per day)
        - Suspected cases (new suspecting individuals detected from contact tracing per day)
        - Isolation cases (new isolations per day)
        - Critically ill cases (new critical cases per day)
        - Recovered cases (new recoveries per day)
        - Death cases (new deaths per day)

    Notes
    -----
    All output arrays have length time_limit + 1, with index representing days
    from the start of observation.
    """

    # Pick death cases
    death_index = np.array([])
    for i, course_of_disease_data in enumerate(course_of_disease_data_list):
        if ~np.isnan(course_of_disease_data['date_of_death']):
            if course_of_disease_data['date_of_death'] <= time_limit:
                death_detection_state = np.random.choice([True, False], size=1, p=[
                    death_detection_rate, 1-death_detection_rate])
                if death_detection_state == True:
                    death_index = np.append(death_index, i)
    # Pick recovered cases
    expected_confirmed_number = round(
        detection_rate*len(course_of_disease_data_list))
    recovered_index = np.arange(len(course_of_disease_data_list))
    recovered_index = np.array(
        [i for i in recovered_index if i not in death_index])
    add_number = round(expected_confirmed_number - len(death_index))
    index = np.sort(np.int64(np.append(death_index, np.random.choice(
        recovered_index, size=add_number, replace=False))))
    course_of_disease_data_list_tmp = course_of_disease_data_list[index]
    contact_data_list_tmp = contact_data_list[index]

    infected_dates = np.array([])
    contagious_dates = np.array([])
    symptomatic_dates = np.array([])
    confirmed_dates = np.array([])
    test_dates = np.array([])
    isolation_dates = np.array([])
    critically_ill_dates = np.array([])
    recovered_dates = np.array([])
    death_dates = np.array([])
    immune_dates = np.array([])
    suspected_dates = np.array([])
    for i, course_of_disease_data in enumerate(course_of_disease_data_list_tmp):
        infected_dates = np.append(
            infected_dates, course_of_disease_data['infection_day'])
        contagious_dates = np.append(
            contagious_dates, course_of_disease_data['infection_day']+course_of_disease_data['latent_period'])
        if ~np.isnan(course_of_disease_data['incubation_period']):
            symptomatic_dates = np.append(
                symptomatic_dates, course_of_disease_data['infection_day']+course_of_disease_data['incubation_period'])
        confirmed_dates = np.append(
            confirmed_dates, course_of_disease_data['positive_test_date'])
        test_dates = np.append(
            test_dates, course_of_disease_data['negative_test_date'])
        test_dates = np.append(
            test_dates, course_of_disease_data['positive_test_date'])
        num_suspected_cases = len(contact_data_list_tmp[i]['household_contacts_matrix'])+len(contact_data_list_tmp[i]['health_care_contacts_matrix'])+len(
            contact_data_list_tmp[i]['workplace_contacts_matrix']+len(contact_data_list_tmp[i]['school_class_contacts_matrix'])+len(contact_data_list_tmp[i]['municipality_contacts_matrix']))
        suspected_dates = np.append(suspected_dates, np.ones(
            num_suspected_cases)*course_of_disease_data['positive_test_date'])
        isolation_dates = np.append(
            isolation_dates, course_of_disease_data['infection_day']+course_of_disease_data['monitor_isolation_period'])
        if ~np.isnan(course_of_disease_data['date_of_critically_ill']):
            critically_ill_dates = np.append(
                critically_ill_dates, course_of_disease_data['date_of_critically_ill'])
        if ~np.isnan(course_of_disease_data['date_of_recovery']):
            recovered_dates = np.append(
                recovered_dates, course_of_disease_data['date_of_recovery'])
            if course_of_disease_data['natural_immunity_status']:
                immune_dates = np.append(
                    immune_dates, course_of_disease_data['date_of_recovery'])
        if ~np.isnan(course_of_disease_data['date_of_death']):
            death_dates = np.append(
                death_dates, course_of_disease_data['date_of_death'])

    test_dates = test_dates[test_dates > 0]

    # Compute the daily counts using np.bincount
    daily_infected_cases = np.bincount(
        np.int32(infected_dates), minlength=time_limit+1)[:time_limit+1]
    daily_contagious_cases = np.bincount(
        np.int32(contagious_dates), minlength=time_limit+1)[:time_limit+1]
    daily_symptomatic_cases = np.bincount(
        np.int32(symptomatic_dates), minlength=time_limit+1)[:time_limit+1]
    daily_confirmed_cases = np.bincount(
        np.int32(confirmed_dates), minlength=time_limit+1)[:time_limit+1]
    daily_tested_cases = np.bincount(
        np.int32(test_dates), minlength=time_limit+1)[:time_limit+1]
    daily_suspected_cases = np.bincount(
        np.int32(suspected_dates), minlength=time_limit+1)[:time_limit+1]
    daily_isolation_cases = np.bincount(
        np.int32(isolation_dates), minlength=time_limit+1)[:time_limit+1]
    daily_critically_ill_cases = np.bincount(
        np.int32(critically_ill_dates), minlength=time_limit+1)[:time_limit+1]
    daily_recovered_cases = np.bincount(
        np.int32(recovered_dates), minlength=time_limit+1)[:time_limit+1]
    daily_immune_cases = np.bincount(
        np.int32(immune_dates), minlength=time_limit+1)[:time_limit+1]
    daily_death_cases = np.bincount(
        np.int32(death_dates), minlength=time_limit+1)[:time_limit+1]

    daily_susceptible_population = population_size - \
        np.cumsum(daily_immune_cases) - np.cumsum(daily_death_cases)

    return (daily_susceptible_population, daily_infected_cases, daily_contagious_cases, daily_symptomatic_cases, daily_confirmed_cases, daily_tested_cases, daily_suspected_cases, daily_isolation_cases, daily_critically_ill_cases, daily_recovered_cases, daily_death_cases)


def add_asymptomatic_date_synthetic(data):
    confirm_date = data.confirmed_date
    symptom_date = data.onset_of_symptom
    infection_date = data.earliest_infection_date
    source_id = data.source_infected_case
    asymptom_date = pd.DataFrame(
        {'asymptomatic_date': np.zeros(len(source_id))}, index=data.index)

    # Create array of first spread date based on infection date and source ID
    first_spread_date = pd.DataFrame(
        {'first_spread_date': np.zeros(len(source_id))}, index=data.index)
    for i in source_id.unique():
        if i != 'C':
            first_spread_date_temp = min(infection_date[source_id == i])
            first_spread_date.loc[data[data.id ==
                                       i].index[0]] = first_spread_date_temp
            first_spread_date = first_spread_date.replace(0, 'C')
    all_date = pd.concat(
        [symptom_date, confirm_date, first_spread_date], axis=1)
    all_date = all_date.replace('C', np.nan)

    # Find min date and assign it as asymptomatic date
    for k in all_date.index:
        all_date_row = all_date.loc[k].dropna()
        if len(all_date_row) > 0:
            asymptom_date.loc[k] = all_date_row.min()
    asymptom_date = asymptom_date.replace(0, 'C')

    result = pd.concat([data, asymptom_date], axis=1)
    return (result)


def convert_synthetic_data_to_test_matrix(results, taiwan_data_matrix, number_source_cases):
    # Load data
    demographic_data_list = np.array([])
    course_of_disease_data_list = np.array([])
    contact_data_list = np.array([])
    for result in results:
        demographic_data_list = np.append(
            demographic_data_list, result.result()[0])
        course_of_disease_data_list = np.append(
            course_of_disease_data_list, result.result()[2])
        contact_data_list = np.append(contact_data_list, result.result()[3])
    # Set number of source cases
    course_of_disease_data_list = course_of_disease_data_list[0:number_source_cases]
    contact_data_list = contact_data_list[0:number_source_cases]

    # Convert my synthesis data into test matrix
    synthetic_data_matrix = np.ones(
        (number_source_cases, np.shape(taiwan_data_matrix)[1]))*np.nan
    gender_dict = {'Male': 1, 'Female': 0}
    for i in range(len(synthetic_data_matrix)):
        # Age
        age = demographic_data_list[i]['age']
        # example: 21 -> 20~25 years old -> mean value 25
        synthetic_data_matrix[i, 0] = int(np.floor(age/10)*10 + 5)

        # # Gender
        # gender = demographic_data_list[i].gender
        # synthetic_data_matrix[i, 1] = gender_dict[gender]

        # Course
        infection_day = course_of_disease_data_list[i]['infection_day']
        incubation_period = course_of_disease_data_list[i]['incubation_period']
        date_of_critically_ill = course_of_disease_data_list[i]['date_of_critically_ill']
        date_of_recovery = course_of_disease_data_list[i]['date_of_recovery']
        date_of_death = course_of_disease_data_list[i]['date_of_death']
        first_negative_test_date = course_of_disease_data_list[i]['negative_test_date'][0]
        positive_test_date = course_of_disease_data_list[i]['positive_test_date']

        # Infection to symptomatic
        synthetic_data_matrix[i, 1] = incubation_period

        # # Infection to recovered. note: shouldn't contain any symptomatic cases
        # if np.isnan(incubation_period):
        #     synthetic_data_matrix[i, 3] = date_of_recovery - infection_day

        # Symptomatic to critically ill
        synthetic_data_matrix[i, 2] = date_of_critically_ill - \
            infection_day - incubation_period

        # Symptomatic to recovered
        if np.isnan(date_of_critically_ill):
            synthetic_data_matrix[i, 3] = date_of_recovery - \
                infection_day - incubation_period

        # Critically ill to recovered
        synthetic_data_matrix[i, 4] = date_of_recovery - date_of_critically_ill

        # Critically ill to death
        synthetic_data_matrix[i, 5] = date_of_death - date_of_critically_ill

        # First negative test date to confirmed date
        synthetic_data_matrix[i, 6] = positive_test_date - \
            first_negative_test_date

        # Symptomatic to confirmed
        synthetic_data_matrix[i, 7] = positive_test_date - \
            infection_day - incubation_period

        # Size of unique infected and uninfected contacts
        # total_contacts = len(contact_data_list[i].household_contacts_matrix) + \
        #     len(contact_data_list[i].school_class_contacts_matrix) + \
        #     len(contact_data_list[i].workplace_contacts_matrix) + \
        #     len(contact_data_list[i].health_care_contacts_matrix) + \
        #     len(contact_data_list[i].municipality_contacts_matrix)
        effective_contacts = sum(contact_data_list[i]['household_effective_contacts'] +
                                 contact_data_list[i]['school_effective_contacts'] +
                                 contact_data_list[i]['workplace_effective_contacts'] +
                                 contact_data_list[i]['health_care_effective_contacts'] +
                                 contact_data_list[i]['municipality_effective_contacts'])
        # synthetic_data_matrix[i, 8] = total_contacts - effective_contacts
        # synthetic_data_matrix[i, 9] = effective_contacts
        synthetic_data_matrix[i, 8] = effective_contacts

    return (synthetic_data_matrix)


def extract_state_transition_days_synthetic(course_of_disease_data_list):
    """
    Extract the days between states such as asymptomatic, symptomatic, critically ill, recovered, and dead

    Parameters
    ----------
    course_of_disease_data_list : List of course_of_disease_data objects

    Returns
    -------
    State transition days (np array) include:
        asymptomatic_to_symptomatic_days
        asymptomatic_to_recovered_days
        symptomatic_to_critically_ill_days
        symptomatic_to_recovered_days
        critically_ill_to_recovered_days
        critically_ill_to_dead_days
    """

    asymptomatic_to_symptomatic_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    asymptomatic_to_recovered_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    symptomatic_to_critically_ill_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    symptomatic_to_recovered_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    critically_ill_to_recovered_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    critically_ill_to_dead_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    asymptomatic_to_confirmed_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    symptomatic_to_confirmed_days = np.ones(
        len(course_of_disease_data_list))*np.nan
    critically_ill_to_confirmed_days = np.ones(
        len(course_of_disease_data_list))*np.nan

    for i, course_of_disease_data in enumerate(course_of_disease_data_list):
        # Asymptomatic to symptomatic
        asymptomatic_to_symptomatic_day = course_of_disease_data['incubation_period']
        asymptomatic_to_symptomatic_days[i] = asymptomatic_to_symptomatic_day
        # Asymptomatic to recovered
        if np.isnan(course_of_disease_data.incubation_period):
            asymptomatic_to_recovered_day = course_of_disease_data['date_of_recovery'] - \
                course_of_disease_data['infection_day']
            asymptomatic_to_recovered_days[i] = asymptomatic_to_recovered_day
        # Symptomatic to critically ill
        symptomatic_to_critically_ill_day = course_of_disease_data['date_of_critically_ill'] - \
            course_of_disease_data.infection_day - course_of_disease_data.incubation_period
        symptomatic_to_critically_ill_days[i] = symptomatic_to_critically_ill_day
        # Symptomatic to recovered
        if np.isnan(course_of_disease_data.date_of_critically_ill):
            symptomatic_to_recovered_day = course_of_disease_data['date_of_recovery'] - \
                course_of_disease_data['infection_day'] - course_of_disease_data['incubation_period']
            symptomatic_to_recovered_days[i] = symptomatic_to_recovered_day
        # Critically ill to recovered
        critically_ill_to_recovered_day = course_of_disease_data['date_of_recovery'] - \
            course_of_disease_data['date_of_critically_ill']
        critically_ill_to_recovered_days[i] = critically_ill_to_recovered_day
        # Critically ill to death
        critically_ill_to_dead_day = course_of_disease_data['date_of_death'] - \
            course_of_disease_data['date_of_critically_ill']
        critically_ill_to_dead_days[i] = critically_ill_to_dead_day
        # Asymptomatic to confirmed
        asymptomatic_to_confirmed_day = course_of_disease_data['monitor_isolation_period']
        asymptomatic_to_confirmed_days[i] = asymptomatic_to_confirmed_day
        # Symptomatic to confirmed
        symptomatic_to_confirmed_day = course_of_disease_data['monitor_isolation_period'] - \
            course_of_disease_data['incubation_period']
        symptomatic_to_confirmed_days[i] = symptomatic_to_confirmed_day
        # Critically ill to confirmed
        critically_ill_to_confirmed_day = course_of_disease_data['monitor_isolation_period'] - \
            course_of_disease_data['date_of_critically_ill'] - \
            course_of_disease_data['infection_day']
        critically_ill_to_confirmed_days[i] = critically_ill_to_confirmed_day

    return (asymptomatic_to_symptomatic_days, asymptomatic_to_recovered_days,
            symptomatic_to_critically_ill_days, symptomatic_to_recovered_days,
            critically_ill_to_recovered_days, critically_ill_to_dead_days,
            asymptomatic_to_confirmed_days, symptomatic_to_confirmed_days,
            critically_ill_to_confirmed_days)


def transform_synthetic_data_to_graph(contact_data, case_edge_list):
    """
    Transform the Taiwan COVID-19 synthetic data to networkx graph data structure

    Parameters
    ----------
    contact_data : List of Data_synthesize.Draw_contact_data objects
    case_edge_list : List of case edges

    Returns
    -------
    G : Networkx graph
    Infected cases in the graph: 'I' + 'integer value'
    Uninfected cases in the graph: 'integer value'
    """
    G = nx.DiGraph()
    uninfected_node_id = 1
    for i in range(len(case_edge_list)):
        source_node = case_edge_list[i, 0]
        target_node = case_edge_list[i, 1]
        if source_node == 'nan':
            G.add_node('I'+target_node)
        else:
            G.add_edge('I'+source_node, 'I'+target_node)

        # Add the rest of the nodes in contact_data
        tmp = contact_data[i]
        num_uninfected_household_contacts = tmp['household_effective_contacts'].count(
            0)
        num_uninfected_workplace_contacts = tmp['workplace_effective_contacts'].count(
            0)
        num_uninfected_school_contacts = tmp['school_effective_contacts'].count(0)
        num_uninfected_health_care_contacts = tmp['health_care_effective_contacts'].count(
            0)
        num_uninfected_municipality_contacts = tmp['municipality_effective_contacts'].count(
            0)
        num_uninfected_contacts = num_uninfected_household_contacts + \
            num_uninfected_workplace_contacts + num_uninfected_school_contacts + \
            num_uninfected_health_care_contacts + num_uninfected_municipality_contacts
        for j in range(num_uninfected_contacts):
            G.add_edge('I'+target_node, uninfected_node_id)
            uninfected_node_id += 1

    print(f"Total number of nodes: {G.number_of_nodes()}")
    print(f"Total number of edges: {G.number_of_edges()}")

    return (G)


def generate_generation_time(course_of_disease_data_list, case_edge_list):
    """
    Generate generation time array

    Parameters
    ----------
    course_of_disease_data_list : List of course_of_disease_data objects
    case_edge_list : List of branching process information such as [source_case_id, case_id, infection_day, contact_type]

    Returns
    -------
    generation_time_array : np array of the generation time
    """
    generation_time_array = np.ones(len(course_of_disease_data_list)-1)*np.nan

    for i in range(len(generation_time_array)):
        if type(case_edge_list[i, 0]) == np.str_:
            if case_edge_list[i, 0] != 'nan':
                source_case_id = case_edge_list[i, 0]
                source_case_infection_day = course_of_disease_data_list[int(
                    source_case_id)-1]['infection_day']
                target_case_id = case_edge_list[i, 1]
                target_case_infection_day = course_of_disease_data_list[int(
                    target_case_id)-1]['infection_day']
                generation_time_array[i] = target_case_infection_day - \
                    source_case_infection_day
        else:
            if not np.isnan(case_edge_list[i, 0]):
                source_case_id = case_edge_list[i, 0]
                source_case_infection_day = course_of_disease_data_list[int(
                    source_case_id)-1]['infection_day']
                target_case_id = case_edge_list[i, 1]
                target_case_infection_day = course_of_disease_data_list[int(
                    target_case_id)-1]['infection_day']
                generation_time_array[i] = target_case_infection_day - \
                    source_case_infection_day

    generation_time_array = generation_time_array[1::]

    return (generation_time_array)


def generate_serial_interval(course_of_disease_data_list, case_edge_list):
    """
    Generate serial interval array

    Parameters
    ----------
    course_of_disease_data_list : List of course_of_disease_data objects
    case_edge_list : List of branching process information such as [source_case_id, case_id, infection_day, contact_type]

    Returns
    -------
    serial_interval_array : np array of the serial interval
    """
    serial_interval_array = np.ones(len(course_of_disease_data_list)-1)*np.nan

    for i in range(len(serial_interval_array)):
        if type(case_edge_list[i, 0]) == np.str_:
            if case_edge_list[i, 0] != 'nan':
                source_case_id = case_edge_list[i, 0]
                source_case_symptom_onset_day = course_of_disease_data_list[int(source_case_id)-1]['infection_day'] + \
                    course_of_disease_data_list[int(
                        source_case_id)-1]['incubation_period']
                target_case_id = case_edge_list[i, 1]
                target_case_symptom_onset_day = course_of_disease_data_list[int(target_case_id)-1]['infection_day'] + \
                    course_of_disease_data_list[int(
                        target_case_id)-1]['incubation_period']
                serial_interval_array[i] = target_case_symptom_onset_day - \
                    source_case_symptom_onset_day
        else:
            if not np.isnan(case_edge_list[i, 0]):
                source_case_id = case_edge_list[i, 0]
                source_case_symptom_onset_day = course_of_disease_data_list[int(source_case_id)-1]['infection_day'] + \
                    course_of_disease_data_list[int(
                        source_case_id)-1]['incubation_period']
                target_case_id = case_edge_list[i, 1]
                target_case_symptom_onset_day = course_of_disease_data_list[int(target_case_id)-1]['infection_day'] + \
                    course_of_disease_data_list[int(
                        target_case_id)-1]['incubation_period']
                serial_interval_array[i] = target_case_symptom_onset_day - \
                    source_case_symptom_onset_day

    serial_interval_array = serial_interval_array[1::]

    return (serial_interval_array)


###############################################################################
# Other functions
###############################################################################
def state_transition_plot(state_transition_days, source_state, target_state, xlim=81, save_fig=False):
    start_time = min(0, state_transition_days.min().days)
    end_time = state_transition_days.max().days
    cumulative_states_transition = np.ones((end_time-start_time)+2) *\
        len(state_transition_days[state_transition_days.notna()])
    for i in state_transition_days[state_transition_days.notna()]:
        cumulative_states_transition[i.days+np.abs(start_time)+1::] -= 1
    # plt.stairs(cumulative_states_transition/cumulative_states_transition[0],
    #            edges=np.arange(start_time, end_time+3)-1, linewidth=2)

    kmf = km_plot_transform(state_transition_days)
    plt.plot()
    fig = kmf.plot(label='%s to %s (%s)' % (
        source_state, target_state, int(cumulative_states_transition[0])))
    plt.legend(prop={'size': 20})
    plt.xlim(start_time-1, xlim)
    plt.xlabel('Day', fontsize=22)
    plt.xticks(np.arange(start_time, xlim, 10), fontsize=22)
    plt.ylabel('Proportion of cases', fontsize=22)
    plt.yticks(np.arange(0, 1+0.2, 0.2), fontsize=22)
    plt.grid()
    plt.minorticks_on()

    if save_fig == True:
        plt.savefig('RW2022_%s_to_%s.pdf' % (source_state, target_state))

    return (cumulative_states_transition)


def km_plot_transform(state_transition_days):
    """
    Plot the state transition curve to Kaplan-Meier curve and return the uncertainty.

    Parameters:
        state_transition_days (pd.Series): Days of state transition for each case.

    Returns:
        kmf (KaplanMeierFitter): KaplanMeierFitter object.
    """
    # Convert state_transition_days to timedeltas
    state_transition_days = pd.to_timedelta(state_transition_days)

    # Create duration-event table
    event = np.ones(len(state_transition_days))
    pdTable = pd.DataFrame(
        {'duration': state_transition_days.dt.days, 'event': event})

    # Fit Kaplan-Meier model
    kmf = KaplanMeierFitter()
    kmf.fit(pdTable['duration'], pdTable['event'])

    return kmf


def transform_population_data_to_sird_number(daily_confirmed_cases, daily_recovered_cases, daily_death_cases, population_size=23215015):
    """
    Transform population data such as daily confirmed cases, daily recovered cases and daily death cases to daily number of people staty in SIRD states.

    Parameters:
    -----------
    daily_confirmed_cases: np.array of daily number of confirmed (reported) cases
    population_size: Assumed population size of the country. Default = Taiwanese population size

    Returns:
    --------
    Daily number of people stay in susceptible, infecion, recovery, and death.
    """
    daily_number_of_susceptible = np.array([population_size])
    daily_number_of_infection = np.array([0])
    daily_number_of_recovery = np.array([0])
    daily_number_of_death = np.array([0])
    for i in range(len(daily_confirmed_cases)):
        daily_number_of_susceptible = np.append(
            daily_number_of_susceptible, daily_number_of_susceptible[-1]-daily_confirmed_cases[i])
        daily_number_of_infection = np.append(
            daily_number_of_infection,
            daily_number_of_infection[-1]+daily_confirmed_cases[i]-daily_recovered_cases[i]-daily_death_cases[i])
        daily_number_of_recovery = np.append(
            daily_number_of_recovery, daily_number_of_recovery[-1]+daily_recovered_cases[i])
        daily_number_of_death = np.append(
            daily_number_of_death, daily_number_of_death[-1]+daily_death_cases[i])

    return (daily_number_of_susceptible, daily_number_of_infection, daily_number_of_recovery, daily_number_of_death)


def estimate_network_R0(contact_data_list, case_number, seed):
    # Randamly pick 'case_number' cases
    rng = default_rng(seed)
    index = rng.choice(len(contact_data_list), size=case_number, replace=False)
    contact_data_list_tmp = contact_data_list[index]
    effective_contact_number_array = np.ones(len(contact_data_list_tmp))
    # course_of_disease_data_list_tme = course_of_disease_data_list[index]
    for i, contact_data in enumerate(contact_data_list_tmp):
        effective_contact_number = np.sum(contact_data.household_effective_contacts) + np.sum(contact_data.school_effective_contacts) + np.sum(
            contact_data.workplace_effective_contacts) + np.sum(contact_data.health_care_effective_contacts) + np.sum(contact_data.municipality_effective_contacts)
        effective_contact_number_array[i] = effective_contact_number

    R0 = np.mean(effective_contact_number_array)

    return (R0)
