import numpy as np
import random
from scipy import stats


class Draw_demographic_data:
    def __init__(self, age_p, gender_p, student_p, employment_p, job_p, age):
        self.age_p = age_p
        self.gender_p = gender_p
        self.male_student_p = student_p[0]
        self.female_student_p = student_p[1]
        self.male_employment_p = employment_p[0]
        self.female_employment_p = employment_p[1]
        self.job_list = job_p['job_list']
        self.male_part_time_job_p = job_p['part_time_job_p'][0]
        self.female_part_time_job_p = job_p['part_time_job_p'][1]
        self.male_full_time_job_p = job_p['full_time_job_p'][0]
        self.female_full_time_job_p = job_p['full_time_job_p'][1]
        self.age = age

    def draw_age(self):
        if np.isnan(self.age):
            self.age = random.choices(
                np.arange(100+1), weights=self.age_p)[0]

        return (self.age)

    def draw_gender(self):
        self.gender = random.choices(
            ['Male', 'Female'], weights=self.gender_p[self.age])[0]

        return (self.gender)

    def draw_occupation(self):
        # Decide student or not
        if self.gender == 'Male':
            student_state = random.choices(
                ['Student', 'Not student'], weights=[self.male_student_p[self.age], 1-self.male_student_p[self.age]])[0]
        elif self.gender == 'Female':
            student_state = random.choices(
                ['Student', 'Not student'], weights=[self.female_student_p[self.age], 1-self.female_student_p[self.age]])[0]
        else:
            print('Student state error')

        # Decide employment or not
        if self.gender == 'Male':
            employment_state = random.choices(['Employment', 'Unemployment'], weights=[
                self.male_employment_p[self.age], 1-self.male_employment_p[self.age]])[0]
        elif self.gender == 'Female':
            employment_state = random.choices(['Employment', 'Unemployment'], weights=[
                self.female_employment_p[self.age], 1-self.female_employment_p[self.age]])[0]
        else:
            print('Employment state error')

        # Decide occupation
        self.job = []
        if (student_state == 'Student') & (employment_state == 'Employment') & (self.gender == 'Male'):
            self.job = random.choices(
                self.job_list, weights=self.male_part_time_job_p)[0]
        elif (student_state == 'Student') & (employment_state == 'Employment') & (self.gender == 'Female'):
            self.job = random.choices(
                self.job_list, weights=self.female_part_time_job_p)[0]
        elif (student_state == 'Not student') & (employment_state == 'Employment') & (self.gender == 'Male'):
            self.job = random.choices(
                self.job_list, weights=self.male_full_time_job_p)[0]
        elif (student_state == 'Not student') & (employment_state == 'Employment') & (self.gender == 'Female'):
            self.job = random.choices(
                self.job_list, weights=self.female_full_time_job_p)[0]

        return (self.job)

    def draw_demographic_data(self):
        self.draw_age()
        self.draw_gender()
        self.draw_occupation()


class Draw_social_data:
    def __init__(self, municipality_data, family_size_dict, school_p, workplace_p, demographic_data_object, hospital_p, hospital_size_p, hospital_sizes):
        self.municipality_dict = municipality_data
        self.municipality_list = list(municipality_data.keys())
        self.municipality_p = np.fromiter(
            municipality_data.values(), dtype=int)/sum(municipality_data.values())
        self.family_size_dict = family_size_dict
        self.school_p = school_p
        self.workplace_p = workplace_p
        self.demographic_data_object = demographic_data_object
        self.hospital_p = hospital_p
        self.hospital_size_p = hospital_size_p
        self.hospital_sizes = hospital_sizes

    def draw_municipality_size(self):
        self.municipality = random.choices(
            self.municipality_list, weights=self.municipality_p)[0]
        self.municipality_size = self.municipality_dict[self.municipality]

        return (self.municipality_size)

    def draw_household_size(self):
        p = self.family_size_dict[self.municipality]
        self.household_size = random.choices(
            np.arange(len(p)), weights=p)[0]

        return (self.household_size)

    def draw_school_class_size(self):
        try:
            self.school_class_size = random.choices(np.arange(len(self.school_p[self.demographic_data_object.age])),
                                                    weights=self.school_p[self.demographic_data_object.age])[0]
        except:
            self.school_class_size = 0

        return (self.school_class_size)

    def draw_work_group_size(self):
        try:
            self.work_group_size = random.choices(np.arange(len(self.workplace_p[self.demographic_data_object.job])),
                                                  weights=self.workplace_p[self.demographic_data_object.job])[0]
        except:
            self.work_group_size = 0

        return (self.work_group_size)

    def draw_clinic_size(self):
        hospital = random.choices(
            self.hospital_p[0], weights=self.hospital_p[1].astype(float))[0]
        hospital_size_index = random.choices(np.arange(
            len(self.hospital_size_p[hospital])), weights=self.hospital_size_p[hospital])[0]
        self.clinic_size = int(np.ceil(
            self.hospital_sizes[hospital][hospital_size_index]))

        return (self.clinic_size)

    def draw_social_data(self):
        self.draw_municipality_size()
        self.draw_household_size()
        self.draw_school_class_size()
        self.draw_work_group_size()
        self.draw_clinic_size()


class Draw_course_of_disease_data:
    def __init__(self, infection_day, latent_period_gamma, infectious_period_gamma, incubation_period_gamma, symptom_to_isolation_gamma,
                 asymptomatic_to_recovered_gamma, symptomatic_to_critically_ill_gamma, symptomatic_to_recovered_gamma,
                 critically_ill_to_recovered_gamma, infection_to_death_gamma, negative_to_confirmed_gamma,
                 natural_immunity_rate, transition_p):
        self.infection_day = infection_day
        self.latent_period_shape = latent_period_gamma['latent_period_shape']
        self.latent_period_scale = latent_period_gamma['latent_period_scale']
        self.infectious_period_shape = infectious_period_gamma['infectious_period_shape']
        self.infectious_period_scale = infectious_period_gamma['infectious_period_scale']
        self.incubation_period_shape = incubation_period_gamma['incubation_period_shape']
        self.incubation_period_scale = incubation_period_gamma['incubation_period_scale']
        self.symptom_to_isolation_shape = symptom_to_isolation_gamma['symptom_to_confirmed_shape']
        self.symptom_to_isolation_scale = symptom_to_isolation_gamma['symptom_to_confirmed_scale']
        self.symptom_to_isolation_loc = symptom_to_isolation_gamma['symptom_to_confirmed_loc']
        self.asymptomatic_to_recovered_shape = asymptomatic_to_recovered_gamma[
            'asymptomatic_to_recovered_shape']
        self.asymptomatic_to_recovered_scale = asymptomatic_to_recovered_gamma[
            'asymptomatic_to_recovered_scale']
        self.asymptomatic_to_recovered_loc = asymptomatic_to_recovered_gamma[
            'asymptomatic_to_recovered_loc']
        self.symptomatic_to_critically_ill_shape = symptomatic_to_critically_ill_gamma[
            'symptomatic_to_critically_ill_shape']
        self.symptomatic_to_critically_ill_scale = symptomatic_to_critically_ill_gamma[
            'symptomatic_to_critically_ill_scale']
        self.symptomatic_to_critically_ill_loc = symptomatic_to_critically_ill_gamma[
            'symptomatic_to_critically_ill_loc']
        self.symptomatic_to_recovered_shape = symptomatic_to_recovered_gamma[
            'symptomatic_to_recovered_shape']
        self.symptomatic_to_recovered_scale = symptomatic_to_recovered_gamma[
            'symptomatic_to_recovered_scale']
        self.symptomatic_to_recovered_loc = symptomatic_to_recovered_gamma[
            'symptomatic_to_recovered_loc']
        self.critically_ill_to_recovered_shape = critically_ill_to_recovered_gamma[
            'critically_ill_to_recovered_shape']
        self.critically_ill_to_recovered_scale = critically_ill_to_recovered_gamma[
            'critically_ill_to_recovered_scale']
        self.critically_ill_to_recovered_loc = critically_ill_to_recovered_gamma[
            'critically_ill_to_recovered_loc']
        self.infection_to_death_shape = infection_to_death_gamma['infection_to_death_shape']
        self.infection_to_death_scale = infection_to_death_gamma['infection_to_death_scale']
        self.negative_to_confirmed_shape = negative_to_confirmed_gamma[
            'negative_to_confirmed_shape']
        self.negative_to_confirmed_scale = negative_to_confirmed_gamma[
            'negative_to_confirmed_scale']
        self.negative_to_confirmed_loc = negative_to_confirmed_gamma[
            'negative_to_confirmed_loc']
        self.natural_immunity_rate = natural_immunity_rate
        self.infection_to_recovered_transition_p = transition_p[0]
        self.symptom_to_recovered_transition_p = transition_p[1]
        self.critically_ill_to_recovered_transition_p = transition_p[2]

    @staticmethod
    def truncated_gamma_sample(shape, scale, loc=0, lower_bound=None, upper_bound=None, size=1):
        # Calculate the CDFs at the bounds
        lower_cdf = stats.gamma.cdf(
            lower_bound - loc, a=shape, scale=scale) if lower_bound is not None else 0
        upper_cdf = stats.gamma.cdf(
            upper_bound - loc, a=shape, scale=scale) if upper_bound is not None else 1

        # Generate uniform random numbers between lower_cdf and upper_cdf
        u = np.random.uniform(lower_cdf, upper_cdf, size=size)

        # Use the percent point function (inverse of CDF) to get the truncated samples
        samples = stats.gamma.ppf(u, a=shape, scale=scale) + loc

        return samples

    def draw_latent_period(self):
        latent_period = round(np.random.gamma(
            shape=self.latent_period_shape, scale=self.latent_period_scale))

        return (latent_period)

    def draw_infectious_period(self):
        # Input:
        # infectious_period_scale = np.random.uniform(
        #     self.infectious_period_scale_lower_bound, self.infectious_period_scale_higher_bound)
        infectious_period = round(np.random.gamma(
            shape=self.infectious_period_shape, scale=self.infectious_period_scale))

        return (infectious_period)

    def draw_incubation_period(self, lower_bound=None, upper_bound=None):
        incubation_period = round(self.truncated_gamma_sample(
            shape=self.incubation_period_shape, scale=self.incubation_period_scale,
            loc=0, lower_bound=lower_bound, upper_bound=upper_bound)[0])
        # incubation_period = round(np.random.gamma(
        #     shape=self.incubation_period_shape, scale=self.incubation_period_scale))

        return (incubation_period)

    def draw_time_from_infection_to_monitored_isolation(self):
        incubation_period_tmp = self.draw_incubation_period()
        symptom_to_isolation = round(np.random.gamma(
            shape=self.symptom_to_isolation_shape, scale=self.symptom_to_isolation_scale)
            + self.symptom_to_isolation_scale)
        time_from_infection_to_monitored_isolation = incubation_period_tmp + symptom_to_isolation

        return (time_from_infection_to_monitored_isolation)

    def draw_date_of_positive_test(self):
        positive_test_date = self.infection_day + self.monitor_isolation_period + \
            np.round(np.random.normal(loc=0, scale=0.3))

        return (positive_test_date)

    def draw_date_of_negative_test(self):
        confirmed_to_negative_test = -round(np.random.gamma(
            shape=self.negative_to_confirmed_shape, scale=self.negative_to_confirmed_scale)
            + self.negative_to_confirmed_loc)

        return (confirmed_to_negative_test)

    def draw_time_from_asymptomatic_to_recovered(self, lower_bound=None):
        asymptomatic_to_recovered_time = round(self.truncated_gamma_sample(
            shape=self.asymptomatic_to_recovered_shape, scale=self.asymptomatic_to_recovered_scale,
            loc=self.asymptomatic_to_recovered_loc,
            lower_bound=lower_bound)[0])

        return (asymptomatic_to_recovered_time)

    def draw_time_from_symptomatic_to_critically_ill(self):
        symptomatic_to_critically_ill_time = round(np.random.gamma(
            shape=self.symptomatic_to_critically_ill_shape, scale=self.symptomatic_to_critically_ill_scale)
            + self.symptomatic_to_critically_ill_loc)
        # print(
        #     f'symptomatic_to_critically_ill_time: {symptomatic_to_critically_ill_time}')
        return (symptomatic_to_critically_ill_time)

    def draw_time_from_symptomatic_to_critically_ill_new(self, lower_bound=None, upper_bound=None):
        if lower_bound > upper_bound:
            print('here')
            print(lower_bound, upper_bound)
            raise ValueError('lower_bound must be smaller than upper_bound')
        symptomatic_to_critically_ill_time = round(self.truncated_gamma_sample(
            shape=self.symptomatic_to_critically_ill_shape, scale=self.symptomatic_to_critically_ill_scale,
            loc=self.symptomatic_to_critically_ill_loc,
            lower_bound=lower_bound, upper_bound=upper_bound)[0])    # def draw_time_from_symptomatic_to_critically_ill(self):
    #     symptomatic_to_critically_ill_time = round(np.random.gamma(
    #         shape=self.symptomatic_to_critically_ill_shape, scale=self.symptomatic_to_critically_ill_scale)
    #         + self.symptomatic_to_critically_ill_loc)
        # print(
        #     f'symptomatic_to_critically_ill_time: {symptomatic_to_critically_ill_time}')
        return (symptomatic_to_critically_ill_time)

    def draw_time_from_symptomatic_to_recovered(self, lower_bound=None):
        symptomatic_to_recovered_time = round(self.truncated_gamma_sample(
            shape=self.symptomatic_to_recovered_shape, scale=self.symptomatic_to_recovered_scale,
            loc=self.symptomatic_to_recovered_loc, lower_bound=lower_bound)[0])

        return (symptomatic_to_recovered_time)

    def draw_time_from_critically_ill_to_recovered(self):
        critically_ill_to_recovered_time = round(np.random.gamma(
            shape=self.critically_ill_to_recovered_shape, scale=self.critically_ill_to_recovered_scale)
            + self.critically_ill_to_recovered_loc)

        return (critically_ill_to_recovered_time)

    def draw_time_from_infection_to_death(self):
        infection_to_death_time = round(np.random.gamma(
            shape=self.infection_to_death_shape, scale=self.infection_to_death_scale))

        return (infection_to_death_time)

    def draw_natural_immunity_status(self):
        natural_immunity_status = random.choices(
            [True, False], weights=[self.natural_immunity_rate, 1-self.natural_immunity_rate])[0]

        return (natural_immunity_status)

    def draw_course_of_disease(self):
        # Initialization
        # course_of_disease = np.ones(5)*np.nan
        self.latent_period = -1
        self.infectious_period = 0
        self.monitor_isolation_period = 0
        self.date_of_recovery = -1
        self.incubation_period = 1e10
        self.date_of_critically_ill = 1e10
        self.date_of_death = -1

        # Generate course of disease
        while not (((self.latent_period + self.infectious_period) >= self.monitor_isolation_period) &
                   ((self.latent_period) <= self.monitor_isolation_period)):
            # print('here1')
            self.latent_period = self.draw_latent_period()
            self.infectious_period = self.draw_infectious_period()
            self.monitor_isolation_period = self.draw_time_from_infection_to_monitored_isolation()

        # Draw test date result
        self.positive_test_date = self.draw_date_of_positive_test()
        self.negative_test_date = np.array(
            [self.draw_date_of_negative_test() + self.positive_test_date])
        false_negative_p = np.random.uniform(0.01, 0.3)  # Mourad2022_Discrete
        self.negative_test_status = np.array([random.choices(
            [True, False], weights=[1-false_negative_p, false_negative_p])[0]])

        while (self.negative_test_date[-1] >= 0) & (self.negative_test_status[-1] == False):
            # print('here2')
            # print('here: ', self.negative_test_date[-1])
            self.negative_test_date = np.append(self.negative_test_date, np.array(
                [self.draw_date_of_negative_test() + self.positive_test_date]))
            self.negative_test_status = np.append(self.negative_test_status, np.array([random.choices(
                [True, False], weights=[1-false_negative_p, false_negative_p])]))

        # Infection to other states
        infection_transition_decision = random.choices(
            ['transit to recovered', 'transit to symptom'],
            weights=[self.infection_to_recovered_transition_p, 1-self.infection_to_recovered_transition_p])[0]
        if infection_transition_decision == 'transit to recovered':
            lower_bound = self.latent_period+self.infectious_period
            self.date_of_recovery = self.infection_day + \
                self.draw_time_from_asymptomatic_to_recovered(lower_bound)
            # while not (self.date_of_recovery >= (self.infection_day+self.latent_period+self.infectious_period)):
            # print(
            #     f'Date of recovery: {self.date_of_recovery}, infection day: {self.infection_day}, latent period: {self.latent_period}, infectious period: {self.infectious_period}')
            # print('here3')
            # self.date_of_recovery = self.infection_day + \
            #     self.draw_time_from_asymptomatic_to_recovered()
            self.incubation_period = np.nan
            self.date_of_critically_ill = np.nan
            self.date_of_death = np.nan
        elif infection_transition_decision == 'transit to symptom':
            lower_bound = self.latent_period
            upper_bound = self.monitor_isolation_period
            self.incubation_period = self.draw_incubation_period(
                lower_bound, upper_bound)
            # while not ((self.incubation_period >= self.latent_period) &
            #            (self.incubation_period <= (self.monitor_isolation_period))):
            #     #    (self.latent_period+self.infectious_period))):
            #     # print(f'Incubation period: {self.incubation_period}, monitor isolation period: {self.monitor_isolation_period}, latent period: {self.latent_period}, infectious period: {self.infectious_period}, infection day: {self.infection_day}')
            #     print('here4')
            #     self.incubation_period = self.draw_incubation_period()

            # Symptomatic to other states
            symptom_transition_decision = random.choices(
                ['transit to recovered', 'transit to critically ill'],
                weights=[self.symptom_to_recovered_transition_p, 1-self.symptom_to_recovered_transition_p])[0]
            if symptom_transition_decision == 'transit to recovered':
                lower_bound = self.latent_period + self.infectious_period - self.incubation_period
                self.date_of_recovery = self.infection_day + self.incubation_period + \
                    self.draw_time_from_symptomatic_to_recovered(
                        lower_bound=lower_bound)
                # while not (self.date_of_recovery >= (self.infection_day+self.latent_period+self.infectious_period)):
                #     print('here5')
                #     # print('here: ', self.date_of_recovery)
                #     self.date_of_recovery = self.infection_day + self.incubation_period + \
                #         self.draw_time_from_symptomatic_to_recovered()
                self.date_of_critically_ill = np.nan
                self.date_of_death = np.nan
            elif symptom_transition_decision == 'transit to critically ill':
                # print('here')
                # while not ((self.date_of_critically_ill >= self.infection_day+self.monitor_isolation_period) &
                #            (self.date_of_critically_ill <= (self.infection_day+self.latent_period+self.infectious_period))):
                lower_bound = self.monitor_isolation_period-self.incubation_period
                upper_bound = self.latent_period+self.infectious_period-self.incubation_period
                self.date_of_critically_ill = self.infection_day + self.incubation_period + \
                    self.draw_time_from_symptomatic_to_critically_ill_new(
                        lower_bound=lower_bound, upper_bound=upper_bound)
                # while not (self.date_of_critically_ill >= self.infection_day+self.monitor_isolation_period):
                #     self.date_of_critically_ill = self.infection_day + self.incubation_period + \
                #         self.draw_time_from_symptomatic_to_critically_ill_new()
                # print('date of critically ill',
                #       self.date_of_critically_ill)
                # print('infection+monitor isolation period',
                #       self.infection_day+self.monitor_isolation_period)

                # print(
                #     f'infection {self.infection_day}, latent {self.latent_period}, incubation_period {self.incubation_period}, infectious {self.infectious_period}')
                # print(
                #     f'Data of critically ill: {self.date_of_critically_ill}, infection+latent+infectious: {self.infection_day+self.latent_period+self.infectious_period}')
                # print(self.date_of_critically_ill <= (
                #     self.infection_day+self.latent_period+self.infectious_period))
                # print(self.symptomatic_to_critically_ill_loc)
                # print(self.date_of_critically_ill >=
                #       self.infection_day+self.monitor_isolation_period)
                # print('')

                # Critically ill to other states
                critically_ill_transition_decision = random.choices(['transit to recovered', 'transit to death'],
                                                                    weights=[self.critically_ill_to_recovered_transition_p,
                                                                             1-self.critically_ill_to_recovered_transition_p])[0]
                if critically_ill_transition_decision == 'transit to recovered':
                    while not ((self.infection_day+self.latent_period+self.infectious_period) <= self.date_of_recovery):
                        # print('here7')
                        self.date_of_recovery = self.date_of_critically_ill + \
                            self.draw_time_from_critically_ill_to_recovered()
                    self.date_of_death = np.nan
                elif critically_ill_transition_decision == 'transit to death':
                    while not ((self.infection_day+self.latent_period+self.infectious_period) <= self.date_of_death):
                        # print('here8')
                        self.date_of_death = self.infection_day + \
                            self.draw_time_from_infection_to_death()
                    self.date_of_recovery = np.nan
                else:
                    print('Critically ill transition error')

            else:
                print('Symptomatic transition error')
        else:
            print('Infection transition error')

        self.natural_immunity_status = self.draw_natural_immunity_status()

        return (self.monitor_isolation_period, self.latent_period, self.incubation_period, self.infectious_period,
                self.date_of_critically_ill, self.date_of_recovery, self.date_of_death, self.positive_test_date)


class Draw_contact_data:
    def __init__(self, attack_rate, social_data_object, course_of_disease_data_object,
                 previously_infected_list, population_size, vaccine_efficacy, vaccination_rate,
                 natural_immunity_status_list, overdispersion_rate, overdispersion_weight,
                 age_risk_ratios, age_p):
        self.course_of_disease_data_object = course_of_disease_data_object
        self.social_data_object = social_data_object
        self.attack_rate = attack_rate
        self.previously_infected_list = previously_infected_list
        self.population_size = population_size
        self.vaccine_efficacy = vaccine_efficacy
        self.vaccination_rate = vaccination_rate
        self.natural_immunity_status_list = natural_immunity_status_list
        self.overdispersion_rate = overdispersion_rate
        self.overdispersion_weight = overdispersion_weight
        self.age_risk_ratios = age_risk_ratios
        self.age_p = age_p

    @ staticmethod
    def generate_logistic_contact_p(t, healthy_p, symptom_p, steepness, symptom_phase, width):
        # generate_logistic_contact_p: generate contact probability on day t
        #     healthy_p: Contact probability when healthy
        #     symptom_p: Contact probability when symptomatic
        #     steepness: Steepness of the logistic function
        #     symptom_phase: Phase relative to symptom-onset for symptomatic (days)
        #     width: Days different between symptom phase and normal phase
        normal_phase = symptom_phase + width
        logistic_p = healthy_p + (healthy_p-symptom_p) - (healthy_p-symptom_p)/(1+np.exp(-steepness*(
            t-symptom_phase))) - (healthy_p-symptom_p)/(1+np.exp(steepness*(t-normal_phase)))

        return (logistic_p)

    # def draw_social_contacts_each_day(self, social_size, p, steepness, symptom_phase, width):
    #     # p: [contact_p, contact_previous_day_p, healthy_p, symptom_p]
    #     isolation_period = self.course_of_disease_data_object.monitor_isolation_period
    #     contacts_number = np.random.binomial(n=social_size, p=p[0], size=1)[0]
    #     symptom_onset = self.course_of_disease_data_object.incubation_period

    #     if contacts_number > 0:
    #         contacts_matrix = np.zeros([contacts_number, isolation_period+1])
    #         for i in range(contacts_number):
    #             tmp = np.zeros(isolation_period+1)
    #             daily_p = np.zeros(isolation_period+1)
    #             if ~np.isnan(symptom_onset):  # symptomatic case
    #                 daily_p = self.generate_logistic_contact_p(
    #                     np.arange(isolation_period+1)-symptom_onset, p[2], p[3], steepness, symptom_phase, width)
    #             else:
    #                 daily_p[:] = np.array(p[2])

    #             normalized_daily_p = daily_p/np.sum(daily_p)
    #             index = random.choices(
    #                 np.arange(isolation_period+1), weights=normalized_daily_p)[0]
    #             # Assign 1 before daily contact draw for increasing efficiency
    #             tmp[index] = 1
    #             # Assign contacts each date
    #             for j in range(isolation_period+1):
    #                 if tmp[j] == 0:
    #                     if j > 0:
    #                         previous_contact_state = tmp[j-1]
    #                     else:
    #                         previous_contact_state = 0
    #                     # No contact in the previous day
    #                     if previous_contact_state == 0:
    #                         tmp[j] = np.random.binomial(
    #                             n=1, p=daily_p[j], size=1)
    #                     else:
    #                         tmp[j] = np.random.binomial(n=1, p=p[1], size=1)
    #             contacts_matrix[i, :] = tmp
    #     else:
    #         contacts_matrix = np.empty([0, isolation_period+1])

    #     return (contacts_matrix)

    # chatgpt optimized
    # def draw_social_contacts_each_day(self, social_size, p, steepness, symptom_phase, width):
    #     # p: [contact_p, contact_previous_day_p, healthy_p, symptom_p]
    #     isolation_period = self.course_of_disease_data_object.monitor_isolation_period
    #     contacts_number = np.random.binomial(n=social_size, p=p[0], size=1)[0]
    #     symptom_onset = self.course_of_disease_data_object.incubation_period

    #     if contacts_number > 0:
    #         daily_p = np.zeros(isolation_period+1)
    #         if not np.isnan(symptom_onset):  # symptomatic case
    #             daily_p = self.generate_logistic_contact_p(
    #                 np.arange(isolation_period+1)-symptom_onset, p[2], p[3], steepness, symptom_phase, width)
    #         else:
    #             daily_p[:] = np.array(p[2])

    #         normalized_daily_p = daily_p / np.sum(daily_p)

    #         indices = np.random.choice(
    #             np.arange(isolation_period+1), size=(contacts_number), p=normalized_daily_p)

    #         contacts_matrix = np.zeros((contacts_number, isolation_period+1))
    #         contacts_matrix[np.arange(contacts_number), indices] = 1
    #         for i in range(contacts_number):
    #             for j in range(isolation_period+1):
    #                 if contacts_matrix[i, j] == 0:
    #                     if j > 0:
    #                         previous_contact_state = contacts_matrix[i, j-1]
    #                     else:
    #                         previous_contact_state = 0

    #                     if previous_contact_state == 0:  # No contact in the previous day
    #                         contacts_matrix[i, j] = np.random.binomial(
    #                             n=1, p=daily_p[j], size=1)
    #                     else:
    #                         contacts_matrix[i, j] = np.random.binomial(
    #                             n=1, p=p[1], size=1)
    #     else:
    #         contacts_matrix = np.empty((0, isolation_period+1))

    #     return contacts_matrix

    @staticmethod
    def generate_first_contact_matrix(shape: tuple, daily_p: list):
        """
        Creates a boolean matrix representing first contact events across multiple simulations.
        
        Each row represents a single simulation, where True indicates the day of first contact.
        Uses a two-phase approach: first attempts natural probability-based assignment,
        then forces assignment for any remaining unassigned simulations.
        
        Parameters:
        -----------
        shape: Tuple of (num_simulations, num_days)
        daily_p: List of daily contact probabilities, length must equal num_days
            
        Returns:
        -------=
        np.ndarray: Boolean matrix where contacts_matrix[i,j] = True indicates
                simulation i had first contact on day j
        
        Example:
        --------
        shape = (1000, 14)  # 1000 simulations, 14 days
        daily_p = [0.3, 0.25, 0.2, ...]  # 14 probabilities
        result = generate_first_contact_matrix(shape, daily_p)
        """
        contacts_matrix = np.zeros(shape, dtype=bool)
        contacts_number = shape[0]
        isolation_period = shape[1]-1

        # Phase 1: Attempt natural probability-based assignment
        unassigned_contacts = []
        for i in range(contacts_number):
            first_contact = None
            for day in range(isolation_period+1):
                if np.random.random() < daily_p[day]:
                    first_contact = day
                    contacts_matrix[i, day] = True
                    break
            if first_contact is None:
                unassigned_contacts.append(i)

        # Phase 2: Force assignment for remaining simulations using normalized probabilities
        if unassigned_contacts:
            normalized_daily_p = daily_p / np.sum(daily_p)
            initial_contacts = np.random.choice(
                np.arange(isolation_period+1), 
                size=len(unassigned_contacts), 
                p=normalized_daily_p
            )
            contacts_matrix[unassigned_contacts, initial_contacts] = True

        return contacts_matrix


    # # Optimize by Claude3.5
    # def draw_social_contacts_each_day(self, social_size, p, steepness, symptom_phase, width):
    #     """
    #     Creates a boolean matrix representing social contact events each day across multiple individuals.

    #     Each row represents a single individual, where True indicates the day of contact.

    #     Parameters:
    #     -----------
    #     social_size: int
    #         Number of social contacts in the simulation
    #     p: list
    #         List of contact probabilities
    #     steepness: float
    #         Steepness of the logistic function
    #     symptom_phase: float
    #         Phase relative to symptom-onset for symptomatic (days)
    #     width: float
    #         Days different between symptom phase and normal phase

    #     Returns:
    #     -------=
    #     np.ndarray: Boolean matrix where contacts_matrix[i,j] = True indicates
    #             individual i had contact on day j
        

    #     """
    #     isolation_period = self.course_of_disease_data_object.monitor_isolation_period
    #     contacts_number = np.random.binomial(n=social_size, p=p[0], size=1)[0]

    #     if contacts_number == 0:
    #         return np.empty((0, isolation_period+1))

    #     symptom_onset = self.course_of_disease_data_object.incubation_period

    #     if np.isnan(symptom_onset):
    #         daily_p = np.full(isolation_period+1, p[2])
    #     else:
    #         daily_p = self.generate_logistic_contact_p(
    #             np.arange(isolation_period+1)-symptom_onset, p[2], p[3], steepness, symptom_phase, width)


    #     contacts_matrix= self.generate_first_contact_matrix((contacts_number, isolation_period+1), daily_p)

    #     # Vectorized operations for subsequent days
    #     for j in range(1, isolation_period+1):
    #         no_contact_mask = ~contacts_matrix[:, j]
    #         previous_contact = contacts_matrix[:, j-1]

    #         new_contacts = np.random.random(contacts_number) < np.where(
    #             previous_contact,
    #             p[1],  # probability if there was contact on the previous day
    #             # probability if there was no contact on the previous day
    #             daily_p[j]
    #         )

    #         contacts_matrix[no_contact_mask, j] = new_contacts[no_contact_mask]

    #     return contacts_matrix

    def draw_social_contacts_each_day(self, social_size, p, steepness, symptom_phase, width):
        """
        Simulates daily social contact patterns with dynamic probability adjustments.
        
        Generates a contact matrix where each row represents an individual's contact pattern
        over time, accounting for symptom onset, previous contacts, and isolation periods.
        
        Parameters:
        -----------
        social_size: Maximum number of potential social contacts
        p: Contact probability list
        steepness: Rate of probability change in logistic function
        symptom_phase: Time offset relative to symptom onset (days)
        width: Duration between symptom and normal phases (days)
        
        Returns:
        --------
        np.ndarray: Boolean matrix (num_contacts x num_days) where True indicates
                contact occurred on that day
        """
        isolation_period = self.course_of_disease_data_object.monitor_isolation_period
        contacts_number = np.random.binomial(n=social_size, p=p[0], size=1)[0]
        
        # Early return if no contacts sampled
        if contacts_number == 0:
            return np.empty((0, isolation_period+1))

        # Determine contact probabilities based on symptom status
        symptom_onset = self.course_of_disease_data_object.incubation_period
        
        if np.isnan(symptom_onset):
            # Asymptomatic case: constant probability
            daily_p = np.full(isolation_period+1, p[2])
        else:
            # Symptomatic case: logistic probability curve
            daily_p = self.generate_logistic_contact_p(
                np.arange(isolation_period+1)-symptom_onset,
                p[2],  # base_p
                p[3],  # peak_p
                steepness,
                symptom_phase,
                width
            )

        # Initialize contact matrix with first day probabilities
        contacts_matrix = self.generate_first_contact_matrix(
            (contacts_number, isolation_period+1),
            daily_p
        )

        # Simulate subsequent days using conditional probabilities
        for j in range(1, isolation_period+1):
            no_contact_mask = ~contacts_matrix[:, j]
            previous_contact = contacts_matrix[:, j-1]

            # Determine contact probabilities based on previous day's status
            new_contacts = np.random.random(contacts_number) < np.where(
                previous_contact,
                p[1],       # Higher probability if contact yesterday
                daily_p[j]  # Base probability if no contact yesterday
            )

            # Update only for individuals without contact yet today
            contacts_matrix[no_contact_mask, j] = new_contacts[no_contact_mask]

        return contacts_matrix

    def draw_contacts_each_day(self, P):
        # Household
        social_size = self.social_data_object.household_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[0], P[1], P[2], P[3]
        p = [contact_p, contact_previous_day_p, healthy_p, symptom_p]
        steepness, symptom_phase, recover_phase = P[4], P[5], P[6]
        self.household_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase)

        # School class
        social_size = self.social_data_object.school_class_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[7], P[8], P[9], P[10]
        p = [contact_p, contact_previous_day_p, healthy_p, symptom_p]
        steepness, symptom_phase, recover_phase = P[11], P[12], P[13]
        self.school_class_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase)

        # Workplace
        social_size = self.social_data_object.work_group_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[14], P[15], P[16], P[17]
        p = [contact_p, contact_previous_day_p, healthy_p, symptom_p]
        steepness, symptom_phase, recover_phase = P[18], P[19], P[20]
        self.workplace_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase)

        # Health care
        social_size = self.social_data_object.clinic_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[21], P[22], P[23], P[24]
        p = [contact_p, contact_previous_day_p, healthy_p, symptom_p]
        steepness, symptom_phase, recover_phase = P[25], P[26], P[27]
        # healthy_p = 0.0001
        # symptom_p = 0.5
        # symptom_phase = 8
        # recover_phase = 20
        # steepness = 10
        self.health_care_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase)

        # Municipality
        social_size = self.social_data_object.municipality_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[28], P[29], P[30], P[31]
        p = [contact_p, contact_previous_day_p, healthy_p, symptom_p]
        steepness, symptom_phase, recover_phase = P[32], P[33], P[34]
        self.municipality_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase)

    def draw_from_previously_infected_set(self):
        if self.population_size > 0:
            previously_infection_status = random.choices([True, False], weights=[
                len(self.previously_infected_list) /
                (len(self.previously_infected_list)+self.population_size),
                1-len(self.previously_infected_list)/(len(self.previously_infected_list)+self.population_size)])[0]
        else:
            previously_infection_status = True

        return (previously_infection_status)

    def draw_vaccination_status(self):
        overdispersion_state = random.choices(
            [True, False], weights=[self.overdispersion_rate, 1-self.overdispersion_rate])[0]

        return (overdispersion_state)
    
    def determine_overdispersion_state(self):
        overdispersion_state = random.choices(
            [True, False], weights=[self.overdispersion_rate, 1-self.overdispersion_rate])[0]

        return (overdispersion_state)

    def calculate_daily_secondary_attack_rate(self):
        infectious_period = self.course_of_disease_data_object.infectious_period
        incubation_period = self.course_of_disease_data_object.incubation_period
        latent_period = self.course_of_disease_data_object.latent_period
        monitor_isolation_period = self.course_of_disease_data_object.monitor_isolation_period
        adjusted_attack_rate_matrix = np.empty([0, monitor_isolation_period+1])
        layers = ['household_attack_rate', 'school_attack_rate',
                  'workplace_attack_rate', 'health_care_attack_rate', 'municipality_attack_rate']
        for layer in layers:
            attack_rate = self.attack_rate[layer]
            if np.isnan(incubation_period):  # Asymptomatic case
                attack_rate_index = np.round(
                    np.linspace(0, 24, infectious_period+1))
                # attack_rate_index = np.round(
                #     np.linspace(0, 16, infectious_period+1))
                adjusted_attack_rate = attack_rate[np.int32(attack_rate_index)]
            else:  # Symptomatic case
                attack_rate_index = np.round(
                    np.linspace(0, 14, incubation_period-latent_period+1))
                # attack_rate_index = np.round(
                #     np.linspace(0, 6, incubation_period-latent_period+1))
                attack_rate_index = np.append(attack_rate_index,
                                              np.round(np.linspace(15, 24, latent_period+infectious_period-incubation_period)))
                # attack_rate_index = np.append(attack_rate_index,
                #                               np.round(np.linspace(7, 16, latent_period+infectious_period-incubation_period)))
                adjusted_attack_rate = attack_rate[np.int32(attack_rate_index)]

            adjusted_attack_rate = np.append(
                np.zeros(self.course_of_disease_data_object.latent_period), adjusted_attack_rate)
            if monitor_isolation_period > latent_period+infectious_period:
                adjusted_attack_rate = np.append(adjusted_attack_rate, np.zeros(
                    monitor_isolation_period-(latent_period+infectious_period)))
            else:
                adjusted_attack_rate = adjusted_attack_rate[0:monitor_isolation_period+1]

            adjusted_attack_rate_matrix = np.vstack(
                [adjusted_attack_rate_matrix, adjusted_attack_rate])

        return (adjusted_attack_rate_matrix)

    # def draw_infection_status(self, adjusted_attack_rate, contact_day_vector, natural_immunity_status,
    #                           vaccine_status, secondary_contact_age):
    #     # Initialization
    #     effective_contacts_vector = np.zeros(len(contact_day_vector))
    #     infection_status = False
    #     for j in range(len(contact_day_vector)):
    #         if (contact_day_vector[j] == 1):
    #             if (natural_immunity_status == False) & (vaccine_status == False):
    #                 age_adjusted_attack_rate = self.calculate_age_adjusted_secondary_attack_rate(
    #                     secondary_contact_age, adjusted_attack_rate)
    #                 overdispersion_state = self.determine_overdispersion_state()
    #                 if overdispersion_state == True:
    #                     overdispersion_p = np.min(
    #                         [1, age_adjusted_attack_rate[j]*self.overdispersion_weight])
    #                     tmp = random.choices([True, False],
    #                                          weights=[overdispersion_p, 1-overdispersion_p])[0]
    #                 else:
    #                     tmp = random.choices([True, False],
    #                                          weights=[age_adjusted_attack_rate[j],
    #                                                   1-age_adjusted_attack_rate[j]])[0]
    #                 if tmp == True:
    #                     effective_contacts_vector[j] = 1
    #                     infection_status = True
    #                     break

    #     return (infection_status, effective_contacts_vector)

    # Claude3.5 optimize
    def draw_infection_status(self, adjusted_attack_rate, contact_day_vector, natural_immunity_status,
                              vaccine_status, secondary_contact_age):
        # Early return if immune
        if natural_immunity_status or vaccine_status:
            return False, np.zeros_like(contact_day_vector)

        # Calculate age-adjusted attack rate for all days at once
        age_adjusted_attack_rate = self.calculate_age_adjusted_secondary_attack_rate(
            secondary_contact_age, adjusted_attack_rate)

        # Apply contact day vector
        effective_attack_rate = age_adjusted_attack_rate * contact_day_vector

        # Determine overdispersion state once
        overdispersion_state = self.determine_overdispersion_state()

        if overdispersion_state:
            # Apply overdispersion to all days at once
            overdispersion_p = np.minimum(
                1, effective_attack_rate**self.overdispersion_weight)
            infection_probs = overdispersion_p
        else:
            infection_probs = effective_attack_rate

        # Generate random numbers for all days at once
        random_numbers = np.random.random(len(contact_day_vector))

        # Determine infection days
        infection_days = random_numbers < infection_probs

        # Find first infection day, if any
        first_infection_day = np.argmax(infection_days)

        if first_infection_day < len(contact_day_vector) and infection_days[first_infection_day]:
            effective_contacts_vector = np.zeros_like(contact_day_vector)
            effective_contacts_vector[first_infection_day] = 1
            return True, effective_contacts_vector
        else:
            return False, np.zeros_like(contact_day_vector)

    def calculate_age_adjusted_secondary_attack_rate(self, secondary_contact_age, adjusted_attack_rate):
        age_adjusted_attack_rate = self.age_risk_ratios[secondary_contact_age] * \
            adjusted_attack_rate
        age_adjusted_attack_rate[age_adjusted_attack_rate > 1] = 1

        return (age_adjusted_attack_rate)

    def draw_contact_data(self, P):
        self.draw_contacts_each_day(P)
        adjusted_attack_rate_matrix = self.calculate_daily_secondary_attack_rate()
        household_attack_rate = adjusted_attack_rate_matrix[0, :]
        school_attack_rate = adjusted_attack_rate_matrix[1, :]
        workplace_attack_rate = adjusted_attack_rate_matrix[2, :]
        health_care_attack_rate = adjusted_attack_rate_matrix[3, :]
        municipality_attack_rate = adjusted_attack_rate_matrix[4, :]

        self.household_previously_infected_index_list = np.array([])
        self.school_previously_infected_index_list = np.array([])
        self.workplace_previously_infected_index_list = np.array([])
        self.health_care_previously_infected_index_list = np.array([])
        self.municipality_previously_infected_index_list = np.array([])

        # Household
        self.household_effective_contacts = []
        self.household_effective_contacts_infection_time = []
        self.household_secondary_contact_ages = []
        if np.sum(self.household_contacts_matrix) > 0:
            for index, row in enumerate(self.household_contacts_matrix):
                if self.population_size > 0 and len(self.previously_infected_list):
                    previously_infection_status = self.draw_from_previously_infected_set()
                    if previously_infection_status:
                        # Pop one from infected set
                        np.random.shuffle(self.previously_infected_list)
                        # pop one
                        case_id = int(self.previously_infected_list[0])
                        self.household_previously_infected_index_list = np.append(
                            self.household_previously_infected_index_list, case_id)
                        natural_immunity_status = self.natural_immunity_status_list[case_id-1]
                    else:
                        self.household_previously_infected_index_list = np.append(
                            self.household_previously_infected_index_list, np.nan)
                        natural_immunity_status = False
                    vaccination_status = self.draw_vaccination_status()
                    secondary_contact_age = random.choices(
                        np.arange(100+1), weights=self.age_p)[0]
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        household_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age)
                    if infection_status == True:
                        self.household_effective_contacts.append(1)
                        self.household_effective_contacts_infection_time.append(
                            np.where(effective_contacts_vector == 1)[0][0])
                        self.household_secondary_contact_ages.append(
                            secondary_contact_age)
                        if previously_infection_status == True:
                            # Pop one from infected set
                            self.previously_infected_list = self.previously_infected_list[1::]
                        else:
                            # Pop one from population set
                            self.population_size -= 1
                    else:
                        self.household_effective_contacts.append(0)
                        self.household_effective_contacts_infection_time.append(
                            np.nan)
                        self.household_secondary_contact_ages.append(np.nan)
                else:
                    break

        # School
        self.school_effective_contacts = []
        self.school_effective_contacts_infection_time = []
        self.school_secondary_contact_ages = []
        if np.sum(self.school_class_contacts_matrix) > 0:
            for index, row in enumerate(self.school_class_contacts_matrix):
                if self.population_size > 0 and len(self.previously_infected_list):
                    previously_infection_status = self.draw_from_previously_infected_set()
                    if previously_infection_status == True:
                        # Pop one from infected set
                        np.random.shuffle(self.previously_infected_list)
                        # pop one
                        case_id = int(self.previously_infected_list[0])
                        self.school_previously_infected_index_list = np.append(
                            self.school_previously_infected_index_list, case_id)
                        natural_immunity_status = self.natural_immunity_status_list[case_id-1]
                    else:
                        # Pop one from population set
                        self.school_previously_infected_index_list = np.append(
                            self.school_previously_infected_index_list, np.nan)
                        natural_immunity_status = False
                    vaccination_status = self.draw_vaccination_status()
                    secondary_contact_age = random.choices(
                        np.arange(100+1), weights=self.age_p)[0]
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        school_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age)
                    if infection_status == True:
                        self.school_effective_contacts.append(1)
                        self.school_effective_contacts_infection_time.append(
                            np.where(effective_contacts_vector == 1)[0][0])
                        self.school_secondary_contact_ages.append(
                            secondary_contact_age)
                        if previously_infection_status == True:
                            self.previously_infected_list = self.previously_infected_list[1::]
                        else:
                            self.population_size -= 1
                    else:
                        self.school_effective_contacts.append(0)
                        self.school_effective_contacts_infection_time.append(
                            np.nan)
                        self.school_secondary_contact_ages.append(np.nan)
                else:
                    break

        # Workplace
        self.workplace_effective_contacts = []
        self.workplace_effective_contacts_infection_time = []
        self.workplace_secondary_contact_ages = []
        if np.sum(self.workplace_contacts_matrix) > 0:
            for index, row in enumerate(self.workplace_contacts_matrix):
                if self.population_size > 0 and len(self.previously_infected_list):
                    previously_infection_status = self.draw_from_previously_infected_set()
                    if previously_infection_status == True:
                        # Pop one from infected set
                        np.random.shuffle(self.previously_infected_list)
                        # pop one
                        case_id = int(self.previously_infected_list[0])
                        self.workplace_previously_infected_index_list = np.append(
                            self.workplace_previously_infected_index_list, case_id)
                        natural_immunity_status = self.natural_immunity_status_list[case_id-1]
                    else:
                        # Pop one from population set
                        self.workplace_previously_infected_index_list = np.append(
                            self.workplace_previously_infected_index_list, np.nan)
                        natural_immunity_status = False
                    vaccination_status = self.draw_vaccination_status()
                    secondary_contact_age = random.choices(
                        np.arange(100+1), weights=self.age_p)[0]
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        workplace_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age)
                    if infection_status == True:
                        self.workplace_effective_contacts.append(1)
                        self.workplace_effective_contacts_infection_time.append(
                            np.where(effective_contacts_vector == 1)[0][0])
                        self.workplace_secondary_contact_ages.append(
                            secondary_contact_age)
                        if previously_infection_status == True:
                            self.previously_infected_list = self.previously_infected_list[1::]
                        else:
                            self.population_size -= 1
                    else:
                        self.workplace_effective_contacts.append(0)
                        self.workplace_effective_contacts_infection_time.append(
                            np.nan)
                        self.workplace_secondary_contact_ages.append(np.nan)
                else:
                    break

        # Health care
        self.health_care_effective_contacts = []
        self.health_care_effective_contacts_infection_time = []
        self.health_care_secondary_contact_ages = []
        if np.sum(self.health_care_contacts_matrix) > 0:
            for index, row in enumerate(self.health_care_contacts_matrix):
                if self.population_size > 0 and len(self.previously_infected_list):
                    previously_infection_status = self.draw_from_previously_infected_set()
                    if previously_infection_status == True:
                        # Pop one from infected set
                        np.random.shuffle(self.previously_infected_list)
                        # pop one
                        case_id = int(self.previously_infected_list[0])
                        self.health_care_previously_infected_index_list = np.append(
                            self.health_care_previously_infected_index_list, case_id)
                        natural_immunity_status = self.natural_immunity_status_list[case_id-1]
                    else:
                        # Pop one from population set
                        self.health_care_previously_infected_index_list = np.append(
                            self.health_care_previously_infected_index_list, np.nan)
                        natural_immunity_status = False
                    vaccination_status = self.draw_vaccination_status()
                    secondary_contact_age = random.choices(
                        np.arange(100+1), weights=self.age_p)[0]
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        health_care_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age)
                    if infection_status == True:
                        self.health_care_effective_contacts.append(1)
                        self.health_care_effective_contacts_infection_time.append(
                            np.where(effective_contacts_vector == 1)[0][0])
                        self.health_care_secondary_contact_ages.append(
                            secondary_contact_age)
                        if previously_infection_status == True:
                            self.previously_infected_list = self.previously_infected_list[1::]
                        else:
                            self.population_size -= 1
                    else:
                        self.health_care_effective_contacts.append(0)
                        self.health_care_effective_contacts_infection_time.append(
                            np.nan)
                        self.health_care_secondary_contact_ages.append(np.nan)
                else:
                    break

        # Municipality
        self.municipality_effective_contacts = []
        self.municipality_effective_contacts_infection_time = []
        self.municipality_secondary_contact_ages = []
        if np.sum(self.municipality_contacts_matrix) > 0:
            for index, row in enumerate(self.municipality_contacts_matrix):
                if self.population_size > 0 and len(self.previously_infected_list):
                    previously_infection_status = self.draw_from_previously_infected_set()
                    if previously_infection_status == True:
                        # Pop one from infected set
                        np.random.shuffle(self.previously_infected_list)
                        # pop one
                        case_id = int(self.previously_infected_list[0])
                        self.municipality_previously_infected_index_list = np.append(
                            self.municipality_previously_infected_index_list, case_id)
                        natural_immunity_status = self.natural_immunity_status_list[case_id-1]
                    else:
                        # Pop one from population set
                        self.municipality_previously_infected_index_list = np.append(
                            self.municipality_previously_infected_index_list, np.nan)
                        natural_immunity_status = False
                    vaccination_status = self.draw_vaccination_status()
                    secondary_contact_age = random.choices(
                        np.arange(100+1), weights=self.age_p)[0]
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        municipality_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age)
                    if infection_status == True:
                        self.municipality_effective_contacts.append(1)
                        self.municipality_effective_contacts_infection_time.append(
                            np.where(effective_contacts_vector == 1)[0][0])
                        self.municipality_secondary_contact_ages.append(
                            secondary_contact_age)
                        if previously_infection_status == True:
                            self.previously_infected_list = self.previously_infected_list[1::]
                        else:
                            self.population_size -= 1
                    else:
                        self.municipality_effective_contacts.append(0)
                        self.municipality_effective_contacts_infection_time.append(
                            np.nan)
                        self.municipality_secondary_contact_ages.append(np.nan)
                else:
                    break

        return self.household_effective_contacts, self.population_size
