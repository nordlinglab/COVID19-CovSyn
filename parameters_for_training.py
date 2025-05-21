import pickle
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rw_data_processing import *
from Data_synthesize import *
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

plt.style.use("./rw_visualization.mplstyle")

def get_effetive_contacts(effective_contact_network):
    case_ids = np.arange(579, 0, -1)
    remove_id = 530
    number_of_effective_contacts = np.array([])
    for case_id in case_ids:
        if case_id == remove_id:
            continue
        else:
            try:
                contact_nodes = list(effective_contact_network['I'+str(case_id)])
            except:
                contact_nodes = []
            number_of_effective_contact = len(contact_nodes)
            number_of_effective_contacts = np.append(number_of_effective_contacts, number_of_effective_contact)
    number_of_effective_contacts[number_of_effective_contacts==0] = np.nan      
    return number_of_effective_contacts

def get_processed_contact_tracing_data():
    Cheng_contact_array = np.array([[100, 39, 6, 4, 2, 0],
                                   [236, 150, 38, 17, 110, 146],
                                   [399, 678, 172, 98, 337, 138]])
    Cheng_attack_rate = np.array([[4, 5.1, 16.7, 0, 0, 0],
                                 [0.8, 2, 2.6, 0, 0, 0],
                                 [0, 0, 0.6, 0, 0, 0]])/100
    norm_weights = np.array([[1, 0.53246753, 0.15355805, 0.16734694, 0.12480974, 0.082],
                             [0.92857143, 0.52, 0.2, 0.14130435, 0.78787879, 1],
                             [0.6, 1, 0.1875, 0.14285714, 0.54545455, 0.18181818]])

    processed_contact_tracing_data = {
        'Cheng_contact_array': Cheng_contact_array,
        'Cheng_attack_rate': Cheng_attack_rate,
        'norm_weights': norm_weights
    }
    return processed_contact_tracing_data

if __name__ == "__main__":
    # Color
    current_palette = sns.color_palette()

    # Load data
    data_path = Path('./data/structured_course_of_disease_data')
    Taiwan_data_sheet = pd.read_excel(data_path/'figshare_taiwan_covid.xlsx', 
                                    sheet_name=0)
    Taiwan_data_sheet.columns = Taiwan_data_sheet.columns.str.strip().str.lower().str.\
        replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # Clean data
    data_1_to_579 = clean_taiwan_data(Taiwan_data_sheet, 1, 579)
    data_matrix = convert_Taiwan_data_to_test_matrix(data_1_to_579)

    edge_list = pd.read_excel(data_path/'figshare_taiwan_covid.xlsx', 
                                  sheet_name='Edge List')
    effective_contact_network = nx.DiGraph()
    for i in range(len(edge_list)):
        source = edge_list.iloc[i, 0]
        target = edge_list.iloc[i, 1]
        if source[0] == 'I':
            if target[0] == 'I':
                effective_contact_network.add_edge(source, target)

    number_of_effective_contacts = get_effetive_contacts(effective_contact_network)
    # Append number of effective contacts to the data matrix
    data_matrix = np.hstack([data_matrix, number_of_effective_contacts.reshape(-1, 1)])
    # Remove gender (because our model has no parameter that affect gender) and days of infection to recovered (because there were only 2 cases)
    data_matrix = np.delete(data_matrix, (1, 3), 1)
    print('Data matrix shape: ', data_matrix.shape)

    print('Number of non nan element in each column: ', np.sum(~np.isnan(data_matrix), axis=0))
    value_counts = np.sum(~np.isnan(data_matrix), axis=1)
    print('Frequency of number of non-NaN elements each subject in the Taiwan data matrix: ', np.unique(value_counts, return_counts=True))
    feature_threshold = 3
    print('feature_threshold: ', feature_threshold)
    data_matrix_small = data_matrix[value_counts >= feature_threshold, :]
    print('Data matrix small shape: ', data_matrix_small.shape)

    processed_contact_tracing_data = get_processed_contact_tracing_data()

    save_path = Path('./variable')
    print(f'Saving data in {save_path} folder')
    with open(save_path/Path('Taiwan_data_matrix.npy'), 'wb') as f:
        np.save(f, data_matrix_small)
    with open(save_path/Path('Taiwan_data_matrix_full.npy'), 'wb') as f:
        np.save(f, data_matrix)
    with open(save_path/Path('processed_contact_tracing_data.pkl'), 'wb') as f:
        pickle.dump(processed_contact_tracing_data, f)