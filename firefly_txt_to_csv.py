import numpy as np
import pandas as pd
from pathlib import Path

def get_description():
    description = ['Evaluation']
    CONTACT_LAYERS = ['Household', 'School', 'Workplace', 'Heath care', 'Municipality']
    PARAMETERS = [
        'Probability of contact',
        'Consecutive daily contact probability',
        'Contact probability when healthy',
        'Contact probability when symptomatic',
        'Steepness of the logistic contact probability function',
        'Phase relative to symptom onset for symptomatic (days)',
        'Phase relative to symptom onset for resuming normal social context (days)'
    ]
    for contact_layer in CONTACT_LAYERS:
        for param in PARAMETERS:
            description.append(f'{contact_layer} with {param} layer')

    description = description+['Overdispersion rate', 'Overdispersion weight']
    course_var_name = [
        'Latent period shape', 'Latent period scale',
        'Infectious period shape', 'Infectious period scale',
        'Incubation period shape', 'Incubation period scale',
        'Symptom onset to monitored isolation shape', 'Symptom onset to monitored isolation scale',
        'Symptom onset to monitored isolation location', 'Asymptomatic to recovered shape',
        'Asymptomatic to recovered scale', 'Asymptomatic to recovered location',
        'Symptom onset to critically ill shape', 'Symptom onset to critically ill scale', 'Symptom onset to critically ill location',
        'Symptom onset to recovered shape', 'Symptom onset to recovered scale', 'Symptom onset to recovered location',
        'Critically ill to recovered shape', 'Critically ill to recovered scale',
        'Critically ill to recovered location', 'Asymptomatic to death shape',
        'Asymptomatic to death scale', 'Negative test to confirmed shape',
        'Negative test to confirmed scale', 'Negative test to confirmed location'
    ]
    # tmp = [f'Age risk ratio {i}' for i in range(4)]
    tmp = ['Age-related risk ratio of the secondary attack rate for age group 0-19 years old',
        'Age-related risk ratio of the secondary attack rate for age group 20-39 years old', 
        'Age-related risk ratio of the secondary attack rate for age group 40-59 years old', 
        'Age-related risk ratio of the secondary attack rate for age group 60+ years old']
    course_var_name = course_var_name + tmp
    tmp = ['Natural immunity rate', 'Vaccination rate', 'Vaccine efficacy']
    course_var_name = course_var_name + tmp
    tmp = [f'Household secondary attack rate for day {i} relative to symptom onset' for i in range(-14, 10+1)]
    course_var_name = course_var_name + tmp
    tmp = [f'School secondary attack rate for day {i} relative to symptom onset' for i in range(-14, 10+1)]
    course_var_name = course_var_name + tmp
    tmp = [f'Workplace secondary attack rate for day {i} relative to symptom onset' for i in range(-14, 10+1)]
    course_var_name = course_var_name + tmp
    tmp = [f'Health care secondary attack rate for day {i} relative to symptom onset' for i in range(-14, 10+1)]
    course_var_name = course_var_name + tmp
    tmp = [f'Municipality secondary attack rate for day {i} relative to symptom onset' for i in range(-14, 10+1)]
    course_var_name = course_var_name + tmp
    tmp = ['Asymptomatic to recovered transition probability', 'Symptom onset to recovered transition probability',
        'Critically ill to recovered transition probability']
    course_var_name = course_var_name + tmp
    description = description + course_var_name
    description.append('Loss')

    return description

if __name__ == '__main__':
    # Load the txt file
    txt_file_path = Path('firefly_result/Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200')
    bound = np.loadtxt(txt_file_path / 'bound.txt', dtype=float)
    firefly_best = np.loadtxt(txt_file_path / 'firefly_best.txt', dtype=float)
    firefly_final = np.loadtxt(txt_file_path / 'firefly_result.txt', dtype=float)
    firefly_result_first_initial_guess = np.loadtxt(txt_file_path / 'firefly_result_first_initial_guess.txt', dtype=float)
    firefly_worst = np.loadtxt(txt_file_path / 'firefly_worst.txt', dtype=float)

    description = get_description()
    data = {'description': description}

    constraint_lb = bound[0, :]
    constraint_lb = np.hstack([np.nan, constraint_lb, np.nan])
    constraint_ub = bound[1, :]
    constraint_ub = np.hstack([np.nan, constraint_ub, np.nan])
    firefly_final = np.hstack([np.ones([len(firefly_final), 1])*1103479, 
                               firefly_final]) #NOTE: This number is from the result of training process
    firefly_first_initial_guess = \
        np.hstack([np.zeros([len(firefly_result_first_initial_guess), 1]), 
                   firefly_result_first_initial_guess])

    data['Constraint lower bound'] = constraint_lb
    data['Constraint upper bound'] = constraint_ub

    seed_num = len(firefly_best)
    for i in range(seed_num):
        data[f'Best firefly {i}'] = firefly_best[i, :]
        data[f'Final firefly {i}'] = firefly_final[i, :]
        data[f'Initial guess {i}'] = firefly_first_initial_guess[i, :]
        data[f'Worst firefly {i}'] = firefly_worst[i, :]
        

    pd.DataFrame(data).to_csv(txt_file_path / 'firefly_optimal_parameters.csv', index=False)
    