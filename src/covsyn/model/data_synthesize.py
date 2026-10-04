import numpy as np
import random
from collections.abc import Sequence
from functools import lru_cache
from scipy import stats


LAYER_NAMES = ('household', 'school', 'workplace', 'health_care', 'municipality')

# ----------------------------------------------------------------------------------------
# Phase D constants (covsyn_decisions.md B17-B36). Everything here is either taken from a
# named source or recorded as an explicit assumption; none of it is optimized by firefly.
# ----------------------------------------------------------------------------------------

# B26: days between the infector being confirmed and the contact being isolated by the
# contact-tracing team. Taiwan CDC traced and quarantined named contacts within about a day
# of confirmation (PMC7605820); drawn as Poisson(1) so it varies between cases.
TRACING_DELAY_MEAN = 1.0

# B31: the work group actually met inside an enterprise. Chen 2022 (BMJ Open 12:e055643)
# Table 5 reports the cases per workplace cluster rising only from 3 to 8 (median 4) while
# the median staff count of the enterprises runs from 16 to 650 -- a 40-fold change in the
# company against a 2.7-fold change in the circle that infects each other. That is a power
# law with exponent log(2.7)/log(40) = 0.27, which is what is used here instead of the
# earlier assumption that the circle is a company-independent log-normal: a five-person firm
# cannot contain a ten-person work group, and Taiwan's enterprises are small (half of them
# have fewer than five employees), so the flat version broke exactly where most cases are.
# The coefficient is set so that the median group matches the Taiwan tracing records of
# 2-2.5 coworker contacts per index case after the contact probability is applied.
WORK_GROUP_EXPONENT = 0.27
WORK_GROUP_COEFFICIENT = 3.0
# Spread around that mean, so two people in identical firms do not meet identical numbers.
WORK_GROUP_LOG_SIGMA = 0.6

# B23: health care contacts do not stop at isolation -- an isolated or hospitalised patient
# keeps meeting nurses and doctors, which is what Cheng 2020's "8+ days after onset" medical
# contact bin (256/697) is made of. The window is extended by at most this many days beyond
# isolation, and the attack rate on those days is multiplied by the protection factor
# because the staff are wearing PPE by then. Both are assumptions; no Taiwan source gives
# the in-hospital contact rate of a confirmed case.
# 14 days in the first Phase D run put 78.5% of medical contacts 8 or more days after onset
# against Cheng 2020's 37%; shortening it to 10 in the second run sent that share to 0.0%,
# because the optimizer simply moved the symptomatic contact peak from 9.1 to 1.2 days after
# onset and 92% of the medical contacts piled into the 0-3 day bin. The window was the wrong
# lever: it is back at 14, and the share of late medical contacts is now a measured target in
# the objective (medical_late_share), so the peak cannot be moved to avoid it.
HEALTH_CARE_POST_ISOLATION_DAYS = 14
HEALTH_CARE_POST_ISOLATION_PROTECTION = 0.25

# B20 / B33: age dependence of severity, relative to the 20-39 reference group, for the four
# age bands used everywhere else in CovSyn (0-19, 20-39, 40-59, 60+). Derived from Verity
# 2020 (Lancet Infect Dis 20:669) Table 1: the symptomatic-to-hospitalised proportions give
# the symptom->ICU gradient and IFR/hospitalisation gives the ICU->death gradient. The
# absolute level is not taken from Verity -- it stays with P[196] / P[197], which are
# calibrated to Taiwan's own cascade (B33) -- only the shape is.
SYMPTOM_TO_ICU_AGE_RR = np.array([0.05, 1.0, 2.8, 6.4])
ICU_TO_DEATH_AGE_RR = np.array([0.10, 1.0, 2.3, 12.8])
AGE_BAND_EDGES = (20, 40, 60)

# B54: municipality mass events, P[199]..P[203], appended after P[198] so every existing
# index keeps its meaning. The Gamma-Poisson count of B27 has an exponential tail
# (P(N >= 446) ~ 2e-16 at run 10), while 3 of the 38 first-wave tracing records with a
# known community count list 446, 822 and 850 contacts; those 7 records with >= 100 contacts
# hold 84% of all community contacts. An event therefore adds a heavy-tailed number of
# contacts, met once. P[203] scales their attack rate; it is locked at 1 for now, so the
# events change the contact-count distribution only.
COMMUNITY_EVENT_FIRST_INDEX = 199
COMMUNITY_EVENT_FIELDS = ('probability', 'exponent', 'min_size', 'max_size', 'risk_ratio')


def community_event_parameters(P: Sequence[float]) -> dict[str, float | int] | None:
    """Read the mass-event parameters P[199..203] (B54).

    Args:
        P: The full parameter vector.

    Returns:
        A dict keyed by COMMUNITY_EVENT_FIELDS, with integer min_size and max_size, or None
        for a vector that ends before P[199] (run 10 and earlier).

    Raises:
        ValueError: The block is partial, not finite, or out of range, so a malformed
            vector cannot silently run the pre-B54 model.
    """
    extra = len(P) - COMMUNITY_EVENT_FIRST_INDEX
    if extra <= 0:
        return None
    if extra < len(COMMUNITY_EVENT_FIELDS):
        raise ValueError(f'parameter vector has {len(P)} entries: the community event block '
                         f'P[199..203] needs all {len(COMMUNITY_EVENT_FIELDS)}')
    values = [float(P[COMMUNITY_EVENT_FIRST_INDEX + i])
              for i in range(len(COMMUNITY_EVENT_FIELDS))]
    if not np.all(np.isfinite(values)):
        raise ValueError(f'community event parameters must be finite: {values}')
    event = dict(zip(COMMUNITY_EVENT_FIELDS, values))
    event['min_size'] = int(round(event['min_size']))
    event['max_size'] = int(round(event['max_size']))
    if not 0.0 <= event['probability'] <= 1.0:
        raise ValueError(f"event probability {event['probability']} is outside [0, 1]")
    if event['exponent'] <= 0.0:
        raise ValueError(f"event size exponent {event['exponent']} must be positive")
    if not 1 <= event['min_size'] <= event['max_size']:
        raise ValueError(f"event sizes need 1 <= min_size <= max_size, got "
                         f"{event['min_size']} and {event['max_size']}")
    if event['risk_ratio'] < 0.0:
        raise ValueError(f"event risk ratio {event['risk_ratio']} must not be negative")
    return event


@lru_cache(maxsize=64)
def _event_size_cdf(exponent: float, min_size: int, max_size: int) -> np.ndarray:
    sizes = np.arange(min_size, max_size + 1)
    weights = sizes.astype(float) ** -exponent
    cdf = np.cumsum(weights) / np.sum(weights)
    # Cached and shared by every later draw, so it must not be modifiable in place.
    cdf.setflags(write=False)
    return cdf


def draw_event_size(exponent: float, min_size: int, max_size: int) -> int:
    """Draw one event size from P(S = s) proportional to s^-exponent (B54).

    Inverse-CDF sampling with a single uniform from the global numpy stream, like every
    other draw in this module.

    Args:
        exponent: Power-law exponent gamma, positive.
        min_size: Smallest event size, at least 1.
        max_size: Largest event size, at least min_size.

    Returns:
        The event size, in [min_size, max_size].
    """
    cdf = _event_size_cdf(float(exponent), int(min_size), int(max_size))
    index = int(np.searchsorted(cdf, np.random.random(), side='right'))
    return int(min_size) + min(index, len(cdf) - 1)


@lru_cache(maxsize=None)
def _capped_gamma_mean(shape, cap):
    """E[min(X, cap)] for X ~ Gamma(shape, 1/shape), which has mean 1 before capping.

    Used to rescale the capped case-level multiplier back to mean 1 so that capping does
    not quietly lower every layer's average attack rate (B17)."""
    scale = 1.0 / shape
    below = stats.gamma.cdf(cap, a=shape + 1, scale=scale)
    above = stats.gamma.sf(cap, a=shape, scale=scale)
    mean = below + cap * above
    return mean if mean > 1e-6 else 1.0


def age_band(age):
    """Index of CovSyn's four age bands (0-19, 20-39, 40-59, 60+) for a single age."""
    if age < AGE_BAND_EDGES[0]:
        return 0
    if age < AGE_BAND_EDGES[1]:
        return 1
    if age < AGE_BAND_EDGES[2]:
        return 2
    return 3


def size_weighted_pmf(pmf, first_size=1):
    """Re-weight a distribution over group sizes so that a randomly chosen PERSON, not a
    randomly chosen group, is sampled (decisions B29 household, B30 school, B31 workplace).

    pmf[i] is the probability of a group whose size is i + first_size. A person is
    first_size + i times more likely to sit in such a group, so the person-weighted
    probability is proportional to (i + first_size) * pmf[i]. Household sizes are stored as
    "number of other members", hence first_size = 1; school and workplace store the size
    itself, hence first_size = 0.
    """
    pmf = np.asarray(pmf, dtype=float)
    sizes = np.arange(len(pmf), dtype=float) + first_size
    weighted = pmf * sizes
    total = weighted.sum()
    return pmf if total <= 0 else weighted / total


_PERSON_WEIGHT_CACHE = {}


def apply_person_weighting(family_size_dict, school_p, workplace_p):
    """Person-weight every social-context size distribution (B29, B30, B31).

    Called once per simulation but the input is the same object for every simulation of an
    evaluation, so the result is cached against the identity of the inputs (which are also
    kept alive by the cache, so the identities cannot be reused by another object)."""
    key = (id(family_size_dict), id(school_p), id(workplace_p))
    cached = _PERSON_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached[1]

    # School is NOT re-weighted here. school_p is a histogram over SCHOOLS, and a school
    # holds several classes, so the weight a random student deserves is pupils = class size x
    # number of classes, not class size alone. Weighting by class size closed only two thirds
    # of the gap (18.2 -> 22.6 against the 25.7 a Taiwanese first-grader actually experiences)
    # because large schools have both bigger classes and more of them (r = 0.64), finding
    # E39. The weighting is now done at source in get_school_data(), where the number of
    # classes is still available, so doing it again here would double-count it.
    weighted = ({city: size_weighted_pmf(p, first_size=1) for city, p in family_size_dict.items()},
                school_p,
                {job: size_weighted_pmf(p, first_size=0) for job, p in workplace_p.items()})
    _PERSON_WEIGHT_CACHE.clear()
    _PERSON_WEIGHT_CACHE[key] = ((family_size_dict, school_p, workplace_p), weighted)
    return weighted


def build_layer_age_distributions(age_p, gender_p, student_p, employment_p):
    """Age distribution of the contacts met in each social layer.

    Every layer used to draw the contact age from the population distribution, so school
    contacts were not school-aged and workplace contacts were not of working age. School and
    workplace are re-weighted by the probability of being a student / being employed at that
    age; household, health care and municipality keep the population distribution because the
    input data carries no age structure for them (the household data gives sizes only).
    """
    age_p = np.asarray(age_p, dtype=float)
    age_p = age_p / np.sum(age_p)
    gender = np.asarray(gender_p, dtype=float)
    gender = gender / np.clip(gender.sum(axis=1, keepdims=True), 1e-12, None)
    student = (gender[:, 0] * np.asarray(student_p[0], dtype=float)
               + gender[:, 1] * np.asarray(student_p[1], dtype=float))
    employed = (gender[:, 0] * np.asarray(employment_p[0], dtype=float)
                + gender[:, 1] * np.asarray(employment_p[1], dtype=float))

    def reweight(w):
        w = age_p * np.asarray(w, dtype=float)
        total = np.sum(w)
        return age_p if total <= 0 else w / total

    return {'household': age_p,
            'school': reweight(student),
            'workplace': reweight(employed),
            'health_care': age_p,
            'municipality': age_p}


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
        """Two-stage workplace context (B31): the employee-weighted census gives the size of
        the ENTERPRISE the case works for, and the work group actually met is that size
        capped by a log-normal group size that does not grow with the company (Chen 2022)."""
        try:
            self.enterprise_size = random.choices(np.arange(len(self.workplace_p[self.demographic_data_object.job])),
                                                  weights=self.workplace_p[self.demographic_data_object.job])[0]
        except:
            self.enterprise_size = 0

        if self.enterprise_size > 0:
            typical = WORK_GROUP_COEFFICIENT * self.enterprise_size ** WORK_GROUP_EXPONENT
            group = int(round(np.random.lognormal(
                mean=np.log(max(typical, 1e-6)) - WORK_GROUP_LOG_SIGMA ** 2 / 2,
                sigma=WORK_GROUP_LOG_SIGMA)))
            self.work_group_size = int(min(self.enterprise_size, max(group, 1)))
        else:
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
                 natural_immunity_rate, transition_p, age=np.nan, source_confirmed_day=np.nan, age_p=None):
        self.infection_day = infection_day
        # B20 / B26: the age of the case (severity depends on it) and the day its infector
        # was confirmed (contact tracing isolates this case shortly afterwards).
        self.age = age
        self.source_confirmed_day = source_confirmed_day
        self.age_p = age_p
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
        # B20: the two severity probabilities are population averages, so the age risk
        # ratios are divided by their population-weighted mean. That keeps P[196] / P[197]
        # interpretable as "the share of all cases" while giving the age gradient.
        self.symptom_to_icu_norm = self._age_rr_norm(SYMPTOM_TO_ICU_AGE_RR)
        # The second stage is conditional on already being in the ICU, and ICU cases are far
        # older than the population, so normalising the death ratios by the POPULATION age
        # distribution would leave the average death-given-ICU well above P[197] (a Simpson
        # effect). The weights used here are therefore the age distribution of ICU cases,
        # which the model itself implies: w_a * rr_icu_a.
        self.icu_to_death_norm = self._age_rr_norm(ICU_TO_DEATH_AGE_RR,
                                                   conditional_on=SYMPTOM_TO_ICU_AGE_RR)

    def _age_rr_norm(self, risk_ratios, conditional_on=None):
        if self.age_p is None:
            return 1.0
        p = np.asarray(self.age_p, dtype=float)
        p = p / np.sum(p)
        bands = np.array([age_band(a) for a in range(len(p))])
        weights = np.array([p[bands == b].sum() for b in range(4)])
        if conditional_on is not None:
            weights = weights * np.asarray(conditional_on, dtype=float)
            weights = weights / np.sum(weights)
        norm = float(np.sum(weights * np.asarray(risk_ratios, dtype=float)))
        return norm if np.isfinite(norm) and norm > 0 else 1.0

    def age_adjusted_probability(self, base_probability, risk_ratios, norm):
        """Scale a population-average transition probability by the case's age band."""
        if np.isnan(self.age):
            return float(np.clip(base_probability, 0.0, 1.0))
        ratio = float(risk_ratios[age_band(int(self.age))]) / norm
        return float(np.clip(base_probability * ratio, 0.0, 1.0))

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

        # The truncation point can sit beyond everything this Gamma can numerically produce
        # (cdf(lower_bound) rounds to 1), and ppf(1) is +inf. It happens while the optimizer
        # explores short recovery times against a long infectious period, and it used to
        # abort the whole run with an OverflowError. The draw then collapses onto the bound
        # that caused it, which is the closest value the distribution can still represent.
        if not np.all(np.isfinite(samples)):
            fallback = lower_bound if lower_bound is not None else upper_bound
            fallback = loc if fallback is None else fallback
            samples = np.where(np.isfinite(samples), samples, fallback)

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

    def draw_pre_onset_window(self):
        """Days between becoming infectious and developing symptoms (B22 (3)).

        The incubation period used to be drawn independently and then truncated at the
        latent period, which forced a fifth of the cases to have a zero-day pre-symptomatic
        infectious window. It is now built as incubation = latent + this window, so the
        window is a quantity of its own and P[41] / P[42] describe it instead of the whole
        incubation period."""
        window = round(np.random.gamma(shape=self.incubation_period_shape,
                                       scale=self.incubation_period_scale))
        return max(int(window), 0)

    def draw_symptom_to_isolation(self):
        """Symptom onset to confirmation/isolation of a case that presents by itself."""
        return max(0, round(np.random.gamma(
            shape=self.symptom_to_isolation_shape, scale=self.symptom_to_isolation_scale)
            + self.symptom_to_isolation_loc))

    def draw_time_from_infection_to_monitored_isolation(self):
        """Kept for backward compatibility with scripts that call it directly; the course of
        disease itself no longer uses it (isolation is now decided in draw_course_of_disease
        by whichever comes first, the case's own symptoms or contact tracing, B26)."""
        return self.draw_latent_period() + self.draw_pre_onset_window() + self.draw_symptom_to_isolation()

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
        """Draw the whole course of disease for one case.

        Rewritten in Phase D (decisions B20, B22, B26, B33). Two structural changes:

        * Isolation is no longer a hypothetical onset drawn for everybody. It is now the
          earlier of (i) the case's own symptom onset plus the onset-to-confirmation delay
          and (ii) the day the contact-tracing team reaches it, which is the infector's
          confirmation date plus a short tracing delay. An asymptomatic case with no
          confirmed infector is therefore never isolated in time to cut its contacts short,
          which is the realistic behaviour; before, asymptomatic cases were isolated on a
          symptom date they never had.
        * Severity depends on age: the symptom->ICU and ICU->death probabilities are the
          calibrated population averages multiplied by the age risk ratio of the case.
        """
        self.latent_period = max(0, self.draw_latent_period())
        self.infectious_period = max(1, self.draw_infectious_period())
        infectious_end = self.latent_period + self.infectious_period

        # --- symptomatic or not -------------------------------------------------------
        asymptomatic = random.choices(
            [True, False],
            weights=[self.infection_to_recovered_transition_p,
                     1 - self.infection_to_recovered_transition_p])[0]
        if asymptomatic:
            self.incubation_period = np.nan
            self.pre_onset_window = np.nan
            own_isolation = np.inf
        else:
            # The window is capped at the infectious period so that symptoms still appear
            # while the case is shedding; the daily attack-rate profile is defined between
            # onset and the end of the infectious period and has no meaning otherwise. With
            # a window of about 2 days against an infectious period of 5-12 the cap binds
            # rarely, but it must be applied before the incubation period is formed.
            self.pre_onset_window = min(self.draw_pre_onset_window(), self.infectious_period)
            self.incubation_period = int(self.latent_period + self.pre_onset_window)
            own_isolation = self.incubation_period + self.draw_symptom_to_isolation()

        # --- isolation: own symptoms or contact tracing, whichever comes first (B26) ---
        if np.isfinite(self.source_confirmed_day):
            traced_isolation = max(0, self.source_confirmed_day
                                   + np.random.poisson(TRACING_DELAY_MEAN)
                                   - self.infection_day)
        else:
            traced_isolation = np.inf

        if np.isfinite(own_isolation) or np.isfinite(traced_isolation):
            isolation = min(own_isolation, traced_isolation)
            self.isolation_route = 'traced' if traced_isolation <= own_isolation else 'symptom'
        else:
            # Asymptomatic index case that nobody traces: it is found only once its own
            # infectious period is over. Reporting is still complete (decision B18).
            isolation = infectious_end
            self.isolation_route = 'untraced'
        self.monitor_isolation_period = int(max(0, round(isolation)))

        # --- testing ------------------------------------------------------------------
        self.positive_test_date = self.draw_date_of_positive_test()
        self.negative_test_date = np.array(
            [self.draw_date_of_negative_test() + self.positive_test_date])
        false_negative_p = np.random.uniform(0.01, 0.3)  # Mourad2022_Discrete
        self.negative_test_status = np.array([random.choices(
            [True, False], weights=[1 - false_negative_p, false_negative_p])[0]])

        while (self.negative_test_date[-1] >= 0) & (self.negative_test_status[-1] == False):
            self.negative_test_date = np.append(self.negative_test_date, np.array(
                [self.draw_date_of_negative_test() + self.positive_test_date]))
            self.negative_test_status = np.append(self.negative_test_status, np.array([random.choices(
                [True, False], weights=[1 - false_negative_p, false_negative_p])]))

        # --- outcome ------------------------------------------------------------------
        self.date_of_critically_ill = np.nan
        self.date_of_death = np.nan
        self.date_of_recovery = np.nan

        if asymptomatic:
            # date_of_recovery is the day the case is closed (B28), so it cannot fall
            # before the end of the infectious period.
            self.date_of_recovery = self.infection_day + \
                self.draw_time_from_asymptomatic_to_recovered(lower_bound=infectious_end)
        else:
            onset_day = self.infection_day + self.incubation_period
            icu_probability = self.age_adjusted_probability(
                1 - self.symptom_to_recovered_transition_p, SYMPTOM_TO_ICU_AGE_RR,
                self.symptom_to_icu_norm)
            if np.random.random() >= icu_probability:   # recovers without critical illness
                lower_bound = max(0, infectious_end - self.incubation_period)
                self.date_of_recovery = onset_day + self.draw_time_from_symptomatic_to_recovered(
                    lower_bound=lower_bound)
            else:
                lower_bound = 0
                upper_bound = max(infectious_end - self.incubation_period, lower_bound + 1)
                self.date_of_critically_ill = onset_day + \
                    self.draw_time_from_symptomatic_to_critically_ill_new(
                        lower_bound=lower_bound, upper_bound=upper_bound)
                earliest_end = max(self.infection_day + infectious_end,
                                   self.date_of_critically_ill)
                death_probability = self.age_adjusted_probability(
                    1 - self.critically_ill_to_recovered_transition_p, ICU_TO_DEATH_AGE_RR,
                    self.icu_to_death_norm)
                if np.random.random() >= death_probability:
                    self.date_of_recovery = self.date_of_critically_ill + \
                        self.draw_time_from_critically_ill_to_recovered()
                    if self.date_of_recovery < earliest_end:
                        self.date_of_recovery = earliest_end + 1
                else:
                    self.date_of_death = self.infection_day + \
                        self.draw_time_from_infection_to_death()
                    if self.date_of_death < earliest_end:
                        self.date_of_death = earliest_end + 1

        self.natural_immunity_status = self.draw_natural_immunity_status()

        return (self.monitor_isolation_period, self.latent_period, self.incubation_period, self.infectious_period,
                self.date_of_critically_ill, self.date_of_recovery, self.date_of_death, self.positive_test_date)


class Draw_contact_data:
    def __init__(self, attack_rate, social_data_object, course_of_disease_data_object,
                 previously_infected_list, population_size, vaccine_efficacy, vaccination_rate,
                 natural_immunity_status_list, overdispersion_rate, overdispersion_weight,
                 age_risk_ratios, age_p, layer_age_p=None, community_dispersion=None,
                 community_event=None):
        self.course_of_disease_data_object = course_of_disease_data_object
        self.social_data_object = social_data_object
        self.attack_rate = attack_rate
        self.previously_infected_list = previously_infected_list
        self.population_size = population_size
        self.vaccine_efficacy = vaccine_efficacy
        self.vaccination_rate = vaccination_rate
        self.natural_immunity_status_list = natural_immunity_status_list
        # B17 / B27: one infectiousness-and-activity multiplier per CASE, not one lottery
        # per contact. A per-contact lottery is thinning, which can never make the offspring
        # distribution more dispersed than Poisson (finding E15); a per-case multiplier is
        # the standard individual-reproductive-number formulation (Lloyd-Smith 2005) and is
        # what produces superspreading. The same nu also scales the community contact count
        # (B27), so a highly infectious and highly social case is one and the same person
        # and the dispersion is not modelled twice.
        # P[35] is the Gamma shape k (variance 1/k); P[36] caps the multiplier so a single
        # case cannot get an absurd value. The draw is rescaled to mean 1 so the layer
        # attack rates keep their calibrated meaning.
        self.overdispersion_shape = max(float(overdispersion_rate), 1e-3)
        self.overdispersion_cap = max(float(overdispersion_weight), 1.0)
        self.infectiousness_multiplier = self.draw_infectiousness_multiplier()
        # The community contact COUNT gets its own dispersion, drawn independently. It used
        # to share nu with the attack rate, as B27 asked, so that a highly infectious case
        # was also a highly social one. The first Phase D run showed what that costs
        # (finding E35): the optimizer drove nu to its most dispersed setting, and because
        # exposure and infectiousness were then perfectly correlated, the realised attack
        # rate of the low-rate community layer was multiplied by E[nu^2]/E[nu] (5.47%
        # against a 0.2% anchor, 60% of all infections) while the high-rate household layer
        # was pushed the other way by the cap at 1 (3.24% against a 6.62% anchor). Keeping
        # the two draws separate leaves B27's heavy tail in the contact counts and B17's
        # individual infectiousness in the attack rate, without multiplying one by the other.
        self.community_dispersion = (None if community_dispersion is None
                                     else max(float(community_dispersion), 1e-3))
        self.community_activity = self.draw_community_activity()
        # B54: dict from community_event_parameters(), or None for no mass events.
        self.community_event = community_event
        self.age_risk_ratios = age_risk_ratios
        self.age_p = age_p
        if layer_age_p is None:
            layer_age_p = {name: age_p for name in LAYER_NAMES}
        self.layer_age_p = layer_age_p
        # Per-layer population-weighted mean risk ratio. The layer attack rates are
        # calibrated as all-age averages, while the age risk ratios are relative to the
        # 20-39 reference group, so applying them directly would move each layer's average
        # by sum(p_age * ra) (~1.4 with the literature values). Dividing by that weighted
        # mean keeps every layer at its calibrated average attack rate while preserving the
        # relative age structure -- the population mean is normalised to 1, the convention
        # used by OpenABM-Covid19. It is computed per layer because the layers now have
        # different age compositions.
        ratios = np.asarray(age_risk_ratios, dtype=float)
        self.age_risk_ratio_norm = {}
        for name in LAYER_NAMES:
            weights = np.asarray(self.layer_age_p[name], dtype=float)
            weights = weights / np.sum(weights)
            norm = float(np.sum(weights * ratios))
            self.age_risk_ratio_norm[name] = norm if np.isfinite(norm) and norm > 0 else 1.0

    def draw_infectiousness_multiplier(self):
        """Case-level multiplier nu ~ Gamma(k, 1/k), capped, rescaled to mean 1 (B17)."""
        nu = np.random.gamma(shape=self.overdispersion_shape,
                             scale=1.0 / self.overdispersion_shape)
        nu = min(nu, self.overdispersion_cap)
        return nu / _capped_gamma_mean(self.overdispersion_shape, self.overdispersion_cap)

    def draw_community_activity(self):
        """How socially active this case is, for the community contact count only (B27).

        Same Gamma(k, 1/k) form as the infectiousness multiplier but an independent draw
        with its own shape, so the community contact distribution can carry the heavy tail
        the Taiwan tracing records show without also multiplying the attack rate (E35).
        Falls back to the infectiousness multiplier when no shape is supplied, which keeps
        older parameter vectors working."""
        if self.community_dispersion is None:
            return self.infectiousness_multiplier
        return np.random.gamma(shape=self.community_dispersion,
                               scale=1.0 / self.community_dispersion)

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

    def layer_end_day(self, layer):
        """Last day of the contact window of a layer, counted from the day of infection.

        Every layer stops at isolation except health care (B23): an isolated or hospitalised
        patient keeps meeting staff, so its window runs on for
        HEALTH_CARE_POST_ISOLATION_DAYS more days, bounded only by death.

        B48 (E74): the window used to be bounded by recovery as well. date_of_recovery is the
        day the case is CLOSED (B28), so run 8 shortened onset-to-recovery from 23 to 8 days to
        cut the late health care contacts off -- trading case closure (18.6 d against 20-32)
        for the medical timing shares. The window no longer depends on the closure date."""
        course = self.course_of_disease_data_object
        isolation = int(course.monitor_isolation_period)
        if layer != 'health_care':
            return isolation
        window = isolation + HEALTH_CARE_POST_ISOLATION_DAYS
        if np.isfinite(course.date_of_death):
            follow_up = int(max(0, round(course.date_of_death - course.infection_day)))
            window = min(window, max(isolation, follow_up))
        return int(window)

    def daily_contact_p(self, p: Sequence[float], steepness: float, symptom_phase: float,
                        width: float, end_day: int) -> np.ndarray:
        """Daily contact probability of one layer on days 0..end_day.

        Args:
            p: The layer's [contact_p, contact_previous_day_p, healthy_p, symptom_p].
            steepness: Steepness of the logistic curve.
            symptom_phase: Offset of the curve from symptom onset, in days.
            width: Days between the healthy and the symptomatic phase.
            end_day: Last day of the layer's window, counted from infection.

        Returns:
            Constant healthy_p for an asymptomatic case, otherwise the logistic curve from
            healthy_p to symptom_p; length end_day + 1.
        """
        symptom_onset = self.course_of_disease_data_object.incubation_period
        if np.isnan(symptom_onset):
            return np.full(end_day+1, p[2])
        return self.generate_logistic_contact_p(
            np.arange(end_day+1)-symptom_onset, p[2], p[3], steepness, symptom_phase, width)

    def draw_community_event_contacts(self, p: Sequence[float], steepness: float,
                                      symptom_phase: float, width: float, end_day: int,
                                      room: int) -> np.ndarray:
        """Draw the contacts of at most one mass event before isolation (B54).

        With probability community_event['probability'] the case attends one event whose
        size follows draw_event_size(). Everyone at the event is met once, on one day drawn
        from the layer's daily contact profile, so being symptomatic lowers the chance the
        event falls on that day exactly as it lowers ordinary contacts (B19).

        Args:
            p, steepness, symptom_phase, width: The municipality layer's contact profile,
                as for daily_contact_p().
            end_day: Last day of the municipality window, counted from infection.
            room: People of the municipality not already drawn as ordinary contacts; caps
                the event size.

        Returns:
            Boolean matrix (contacts x end_day + 1), with no rows when there is no event.
        """
        no_event = np.zeros((0, end_day+1), dtype=bool)
        event = self.community_event
        if event is None or np.random.random() >= event['probability']:
            return no_event
        size = min(draw_event_size(event['exponent'], event['min_size'], event['max_size']),
                   int(room))
        if size <= 0:
            return no_event
        daily_p = np.clip(self.daily_contact_p(p, steepness, symptom_phase, width, end_day),
                          0.0, None)
        total = np.sum(daily_p)
        weights = daily_p / total if total > 0 else np.full(end_day+1, 1.0 / (end_day+1))
        day = np.random.choice(end_day+1, p=weights)
        contacts = np.zeros((size, end_day+1), dtype=bool)
        contacts[:, day] = True
        return contacts

    def draw_social_contacts_each_day(self, social_size, p, steepness, symptom_phase, width,
                                      end_day=None, contacts_number=None):
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
        isolation_period = (self.course_of_disease_data_object.monitor_isolation_period
                            if end_day is None else int(end_day))
        if contacts_number is None:
            contacts_number = np.random.binomial(n=social_size, p=p[0], size=1)[0]

        # Early return if no contacts sampled
        if contacts_number == 0:
            return np.empty((0, isolation_period+1))

        daily_p = self.daily_contact_p(p, steepness, symptom_phase, width, isolation_period)

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
        """Draw the contact matrix of every layer for this case.

        Phase D changes:
        * B19 (a): school, workplace and municipality cannot have a HIGHER daily contact
          probability after symptom onset than before it, so the symptomatic probability is
          clamped to the healthy one. Household and health care are exempt: a sick person
          does stay home with the family and does go to a clinic.
        * B23: the health care window runs past isolation (see layer_end_day).
        * B27: the number of community contacts no longer comes from Binomial(city
          population, p) -- which made a Taipei case meet 12x more people than a Taitung one
          (finding E4) and produced no heavy tail. It is now Poisson(nu * lambda), where
          lambda = P[28] is the mean number of distinct community contacts per case and nu
          is this case's activity multiplier, giving the Gamma-Poisson (negative binomial)
          shape seen in the Taiwan tracing records.
        """
        self.layer_windows = {layer: self.layer_end_day(layer) for layer in LAYER_NAMES}

        # Household
        social_size = self.social_data_object.household_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[0], P[1], P[2], P[3]
        p = [contact_p, contact_previous_day_p, healthy_p, symptom_p]
        steepness, symptom_phase, recover_phase = P[4], P[5], P[6]
        self.household_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase,
            end_day=self.layer_windows['household'])

        # School class
        social_size = self.social_data_object.school_class_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[7], P[8], P[9], P[10]
        p = [contact_p, contact_previous_day_p, healthy_p, min(symptom_p, healthy_p)]
        steepness, symptom_phase, recover_phase = P[11], P[12], P[13]
        self.school_class_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase,
            end_day=self.layer_windows['school'])

        # Workplace
        social_size = self.social_data_object.work_group_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[14], P[15], P[16], P[17]
        p = [contact_p, contact_previous_day_p, healthy_p, min(symptom_p, healthy_p)]
        steepness, symptom_phase, recover_phase = P[18], P[19], P[20]
        self.workplace_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase,
            end_day=self.layer_windows['workplace'])

        # Health care
        social_size = self.social_data_object.clinic_size
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[21], P[22], P[23], P[24]
        p = [contact_p, contact_previous_day_p, healthy_p, symptom_p]
        steepness, symptom_phase, recover_phase = P[25], P[26], P[27]
        self.health_care_contacts_matrix = self.draw_social_contacts_each_day(
            social_size, p, steepness, symptom_phase, recover_phase,
            end_day=self.layer_windows['health_care'])

        # Municipality (community)
        contact_p, contact_previous_day_p, healthy_p, symptom_p = P[28], P[29], P[30], P[31]
        p = [contact_p, contact_previous_day_p, healthy_p, min(symptom_p, healthy_p)]
        steepness, symptom_phase, recover_phase = P[32], P[33], P[34]
        community_contacts = np.random.poisson(
            max(contact_p, 0.0) * self.community_activity)
        community_contacts = int(min(community_contacts,
                                     self.social_data_object.municipality_size))
        self.municipality_contacts_matrix = self.draw_social_contacts_each_day(
            None, p, steepness, symptom_phase, recover_phase,
            end_day=self.layer_windows['municipality'], contacts_number=community_contacts)
        # B54: event contacts go after the ordinary ones, flagged by municipality_event_mask
        # so the infection loop can apply the event risk ratio. Without event parameters
        # nothing is drawn and the mask is never set, which keeps the random stream, and so
        # every pre-B54 result, unchanged.
        if self.community_event is not None:
            event_contacts = self.draw_community_event_contacts(
                p, steepness, symptom_phase, recover_phase, self.layer_windows['municipality'],
                self.social_data_object.municipality_size - community_contacts)
            ordinary = self.municipality_contacts_matrix
            if event_contacts.shape[0] > 0:
                self.municipality_contacts_matrix = (
                    event_contacts if ordinary.shape[0] == 0
                    else np.vstack([ordinary, event_contacts.astype(ordinary.dtype)]))
            self.municipality_event_mask = np.concatenate(
                [np.zeros(ordinary.shape[0], dtype=bool),
                 np.ones(event_contacts.shape[0], dtype=bool)])

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
        vaccination_status = random.choices(
            [True, False], weights=[self.vaccination_rate, 1-self.vaccination_rate])[0]

        return (vaccination_status)
    
    def calculate_daily_secondary_attack_rate(self):
        """Daily attack rate of every layer, indexed by days since infection.

        Phase D changes:
        * each layer is cut to its own window (health care runs past isolation, B23) and the
          days after isolation carry the PPE protection factor;
        * the whole profile is scaled by this case's infectiousness multiplier nu (B17), so
          the dispersion of the offspring distribution comes from case-to-case variation
          instead of a per-contact lottery.
        Returns a dict keyed by layer name.
        """
        infectious_period = self.course_of_disease_data_object.infectious_period
        incubation_period = self.course_of_disease_data_object.incubation_period
        latent_period = self.course_of_disease_data_object.latent_period
        isolation = int(self.course_of_disease_data_object.monitor_isolation_period)
        windows = getattr(self, 'layer_windows',
                          {layer: isolation for layer in LAYER_NAMES})
        profiles = {}
        for layer in LAYER_NAMES:
            attack_rate = self.attack_rate[f'{layer}_attack_rate']
            if np.isnan(incubation_period):  # Asymptomatic case
                attack_rate_index = np.round(np.linspace(0, 24, infectious_period+1))
                adjusted_attack_rate = attack_rate[np.int32(attack_rate_index)]
            else:  # Symptomatic case
                attack_rate_index = np.round(
                    np.linspace(0, 14, incubation_period-latent_period+1))
                attack_rate_index = np.append(attack_rate_index,
                                              np.round(np.linspace(15, 24, latent_period+infectious_period-incubation_period)))
                adjusted_attack_rate = attack_rate[np.int32(attack_rate_index)]

            adjusted_attack_rate = np.append(
                np.zeros(latent_period), adjusted_attack_rate)
            end_day = int(windows[layer])
            if len(adjusted_attack_rate) < end_day + 1:
                adjusted_attack_rate = np.append(
                    adjusted_attack_rate, np.zeros(end_day + 1 - len(adjusted_attack_rate)))
            else:
                adjusted_attack_rate = adjusted_attack_rate[0:end_day+1]

            if layer == 'health_care' and end_day > isolation:
                adjusted_attack_rate[isolation+1:] *= HEALTH_CARE_POST_ISOLATION_PROTECTION

            profiles[layer] = adjusted_attack_rate * self.infectiousness_multiplier

        return profiles

    def draw_infection_status(self, adjusted_attack_rate, contact_day_vector, natural_immunity_status,
                              vaccine_status, secondary_contact_age, layer='household'):
        """Whether this contact gets infected, and on which day.

        The per-contact overdispersion lottery was removed in Phase D (B17): drawing an
        independent multiplier for every contact is thinning, which cannot raise the
        variance-to-mean ratio of the offspring count above 1, so it could never produce
        superspreading (finding E15). The multiplier now sits on the case and has already
        been applied to adjusted_attack_rate.
        """
        # Early return if immune
        if natural_immunity_status or vaccine_status:
            self.last_infection_probability = 0.0
            return False, np.zeros_like(contact_day_vector)

        age_adjusted_attack_rate = self.calculate_age_adjusted_secondary_attack_rate(
            secondary_contact_age, adjusted_attack_rate, layer)
        infection_probs = age_adjusted_attack_rate * contact_day_vector

        # The probability that this contact is infected on ANY of its contact days. Counting
        # realised infections is a Bernoulli draw on top of it, and with a case-level
        # infectiousness multiplier the realisations cluster so hard that 100 simulations
        # cannot estimate a layer's attack rate (finding E38). Recording the probability
        # itself gives the same quantity without that noise.
        self.last_infection_probability = float(1.0 - np.prod(1.0 - np.clip(infection_probs, 0.0, 1.0)))

        random_numbers = np.random.random(len(contact_day_vector))
        infection_days = random_numbers < infection_probs
        first_infection_day = np.argmax(infection_days)

        if first_infection_day < len(contact_day_vector) and infection_days[first_infection_day]:
            effective_contacts_vector = np.zeros_like(contact_day_vector)
            effective_contacts_vector[first_infection_day] = 1
            return True, effective_contacts_vector
        return False, np.zeros_like(contact_day_vector)

    def calculate_age_adjusted_secondary_attack_rate(self, secondary_contact_age, adjusted_attack_rate,
                                                     layer='household'):
        age_adjusted_attack_rate = (self.age_risk_ratios[secondary_contact_age] /
                                    self.age_risk_ratio_norm[layer]) * adjusted_attack_rate
        age_adjusted_attack_rate[age_adjusted_attack_rate > 1] = 1

        return (age_adjusted_attack_rate)

    def draw_contact_data(self, P):
        self.draw_contacts_each_day(P)
        attack_rate_profiles = self.calculate_daily_secondary_attack_rate()
        household_attack_rate = attack_rate_profiles['household']
        school_attack_rate = attack_rate_profiles['school']
        workplace_attack_rate = attack_rate_profiles['workplace']
        health_care_attack_rate = attack_rate_profiles['health_care']
        municipality_attack_rate = attack_rate_profiles['municipality']

        self.household_previously_infected_index_list = np.array([])
        self.school_previously_infected_index_list = np.array([])
        self.workplace_previously_infected_index_list = np.array([])
        self.health_care_previously_infected_index_list = np.array([])
        self.municipality_previously_infected_index_list = np.array([])

        # Household
        self.household_effective_contacts = []
        # Sum of the per-contact infection probabilities: the expected number of
        # infections in this layer, a low-variance version of the realised count (E38).
        self.household_expected_infections = 0.0
        self.household_effective_contacts_infection_time = []
        self.household_secondary_contact_ages = []
        self.household_contact_ages = []
        if np.sum(self.household_contacts_matrix) > 0:
            for index, row in enumerate(self.household_contacts_matrix):
                if self.population_size > 0:
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
                        np.arange(100+1), weights=self.layer_age_p['household'])[0]
                    self.household_contact_ages.append(secondary_contact_age)
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        household_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age,
                        'household')
                    self.household_expected_infections += self.last_infection_probability
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
        # Sum of the per-contact infection probabilities: the expected number of
        # infections in this layer, a low-variance version of the realised count (E38).
        self.school_expected_infections = 0.0
        self.school_effective_contacts_infection_time = []
        self.school_secondary_contact_ages = []
        self.school_contact_ages = []
        if np.sum(self.school_class_contacts_matrix) > 0:
            for index, row in enumerate(self.school_class_contacts_matrix):
                if self.population_size > 0:
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
                        np.arange(100+1), weights=self.layer_age_p['school'])[0]
                    self.school_contact_ages.append(secondary_contact_age)
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        school_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age,
                        'school')
                    self.school_expected_infections += self.last_infection_probability
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
        # Sum of the per-contact infection probabilities: the expected number of
        # infections in this layer, a low-variance version of the realised count (E38).
        self.workplace_expected_infections = 0.0
        self.workplace_effective_contacts_infection_time = []
        self.workplace_secondary_contact_ages = []
        self.workplace_contact_ages = []
        if np.sum(self.workplace_contacts_matrix) > 0:
            for index, row in enumerate(self.workplace_contacts_matrix):
                if self.population_size > 0:
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
                        np.arange(100+1), weights=self.layer_age_p['workplace'])[0]
                    self.workplace_contact_ages.append(secondary_contact_age)
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        workplace_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age,
                        'workplace')
                    self.workplace_expected_infections += self.last_infection_probability
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
        # Sum of the per-contact infection probabilities: the expected number of
        # infections in this layer, a low-variance version of the realised count (E38).
        self.health_care_expected_infections = 0.0
        self.health_care_effective_contacts_infection_time = []
        self.health_care_secondary_contact_ages = []
        self.health_care_contact_ages = []
        if np.sum(self.health_care_contacts_matrix) > 0:
            for index, row in enumerate(self.health_care_contacts_matrix):
                if self.population_size > 0:
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
                        np.arange(100+1), weights=self.layer_age_p['health_care'])[0]
                    self.health_care_contact_ages.append(secondary_contact_age)
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        health_care_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age,
                        'health_care')
                    self.health_care_expected_infections += self.last_infection_probability
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
        # Sum of the per-contact infection probabilities: the expected number of
        # infections in this layer, a low-variance version of the realised count (E38).
        self.municipality_expected_infections = 0.0
        self.municipality_effective_contacts_infection_time = []
        self.municipality_secondary_contact_ages = []
        self.municipality_contact_ages = []
        # B54: event contacts carry the municipality rate times P[203] (locked at 1 for now).
        event_mask = getattr(self, 'municipality_event_mask', None)
        event_risk_ratio = (self.community_event['risk_ratio']
                            if self.community_event is not None else 1.0)
        if np.sum(self.municipality_contacts_matrix) > 0:
            for index, row in enumerate(self.municipality_contacts_matrix):
                if self.population_size > 0:
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
                        np.arange(100+1), weights=self.layer_age_p['municipality'])[0]
                    self.municipality_contact_ages.append(secondary_contact_age)
                    contact_attack_rate = municipality_attack_rate
                    if event_mask is not None and event_mask[index]:
                        contact_attack_rate = municipality_attack_rate * event_risk_ratio
                    infection_status, effective_contacts_vector = self.draw_infection_status(
                        contact_attack_rate, row, natural_immunity_status, vaccination_status, secondary_contact_age,
                        'municipality')
                    self.municipality_expected_infections += self.last_infection_probability
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
