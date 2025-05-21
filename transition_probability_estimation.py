import copy
import numpy as np


def course_data_to_state_data(course_of_disease: np.array):
    '''
    Convert course_of_disease matrix to state transition arrays

    Input
    -----
    course_of_disease -> [latent_period, incubation_period, date_of_critically_ill, date_of_recovery, date_of_death]

    Output
    ------
    State transition arrays

    Note
    ----
    asymptomatic_each_days, symptomatic_each_days, and critically_ill_each_days start at
    #       day -1 meaning the first value is the number of people who have stayed in the state.
    '''
    course = copy.copy(course_of_disease)
    asymptomatic_to_symptomatic_days = course[:, 1]
    asymptomatic_to_recovered_days = course[:, 3]
    asymptomatic_to_recovered_days[~np.isnan(
        asymptomatic_to_symptomatic_days)] = np.nan  # Replace the path from asymptomatic to systematic to recovered to nan. Only the direct path from asymptomatic to recovered is recorded.

    symptomatic_to_critically_ill_days = course_of_disease[:,
                                                           2] - course_of_disease[:, 1]
    symptomatic_to_recovered_days = course_of_disease[:,
                                                      3] - course_of_disease[:, 1]
    symptomatic_to_recovered_days[~np.isnan(course_of_disease[:, 2])] = np.nan

    critically_ill_to_recovered_days = course_of_disease[:,
                                                         3] - course_of_disease[:, 2]
    critically_ill_to_death_days = course_of_disease[:,
                                                     4] - course_of_disease[:, 2]

    # Asymptomatic
    asymptomatic_each_days = np.ones(int(np.nanmax(
        (np.nanmax(asymptomatic_to_symptomatic_days), np.nanmax(asymptomatic_to_recovered_days)))+2))\
        * len(course_of_disease)
    try:
        asymptomatic_to_symptomatic_each_days = np.zeros(
            int(np.nanmax(asymptomatic_to_symptomatic_days))+1)
    except:
        asymptomatic_to_symptomatic_each_days = []
    try:
        asymptomatic_to_recovered_each_days = np.zeros(
            int(np.nanmax(asymptomatic_to_recovered_days))+1)
    except:
        asymptomatic_to_recovered_each_days = []
    for i in asymptomatic_to_symptomatic_days:
        if ~np.isnan(i):
            asymptomatic_each_days[int(i)+1::] -= 1
            asymptomatic_to_symptomatic_each_days[int(i)] += 1
    for i in asymptomatic_to_recovered_days:
        if ~np.isnan(i):
            asymptomatic_each_days[int(i)+1::] -= 1
            asymptomatic_to_recovered_each_days[int(i)] += 1

    # Symptomatic
    try:
        symptomatic_each_days = np.ones(int(np.nanmax(
            (np.nanmax(symptomatic_to_critically_ill_days), np.nanmax(symptomatic_to_recovered_days)))+2))\
            * len(symptomatic_to_critically_ill_days[~np.isnan(course_of_disease[:, 1])])
    except:
        symptomatic_each_days = []
    try:
        symptomatic_to_critically_ill_each_days = np.zeros(
            int(np.nanmax(symptomatic_to_critically_ill_days))+1)
    except:
        symptomatic_to_critically_ill_each_days = []
    try:
        symptomatic_to_recovered_each_days = np.zeros(
            int(np.nanmax(symptomatic_to_recovered_days))+1)
    except:
        symptomatic_to_recovered_each_days = []
    for i in symptomatic_to_critically_ill_days:
        if ~np.isnan(i):
            symptomatic_each_days[int(i)+1::] -= 1
            symptomatic_to_critically_ill_each_days[int(i)] += 1
    for i in symptomatic_to_recovered_days:
        if ~np.isnan(i):
            symptomatic_each_days[int(i)+1::] -= 1
            symptomatic_to_recovered_each_days[int(i)] += 1

    # Critically ill
    try:
        critically_ill_each_days = np.ones(int(np.nanmax(
            (np.nanmax(critically_ill_to_recovered_days), np.nanmax(critically_ill_to_death_days)))+2))\
            * len(critically_ill_to_recovered_days[~np.isnan(course_of_disease[:, 2])])
    except:
        critically_ill_each_days = []
    try:
        critically_ill_to_recovered_each_days = np.zeros(
            int(np.nanmax(critically_ill_to_recovered_days))+1)
    except:
        critically_ill_to_recovered_each_days = []
    try:
        critically_ill_to_death_each_days = np.zeros(
            int(np.nanmax(critically_ill_to_death_days))+1)
    except:
        critically_ill_to_death_each_days = []
    for i in critically_ill_to_recovered_days:
        if ~np.isnan(i):
            critically_ill_each_days[int(i)+1::] -= 1
            critically_ill_to_recovered_each_days[int(i)] += 1
    for i in critically_ill_to_death_days:
        if ~np.isnan(i):
            critically_ill_each_days[int(i)+1::] -= 1
            critically_ill_to_death_each_days[int(i)] += 1

    return (asymptomatic_each_days, symptomatic_each_days, critically_ill_each_days,
            asymptomatic_to_symptomatic_each_days, asymptomatic_to_recovered_each_days,
            symptomatic_to_critically_ill_each_days, symptomatic_to_recovered_each_days,
            critically_ill_to_recovered_each_days, critically_ill_to_death_each_days)
