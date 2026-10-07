"""A cost function that reduces inside the worker processes instead of the parent.

Why this exists
---------------
Batching the simulations into fewer pool tasks (SIMULATIONS_PER_TASK, finding E64) took one
objective evaluation from 0.1235 s to 0.0436 s on the benchmark vector, and the real run 6
search from 0.1284 s to 0.0663 s per evaluation. What is left is no longer task dispatch: it
is (a) shipping 300 whole simulation results -- contact matrices included -- back to the
parent, and (b) the parent then walking all of them single-threaded.

Every one of those parent-side passes is per case, so each worker can do its own share:

  * the Cheng 2020 six-bin contact and infection counts are COUNTS, so they add across cases
    and across tasks, and integer addition is exact in any order;
  * the energy-distance test matrix is one row per case;
  * measure_outcomes() needs a couple of dozen scalars per index case, after which the
    cross-case parts (a variance, two percentiles, some means) are cheap.

So the workers return counts, matrix rows and scalars, and no contact matrix ever crosses the
process boundary. The read-only demographic parameters are sent once per worker through the
pool initializer rather than once per task.

Bit-identity
------------
The binning is not reimplemented here: _case_cheng_bins() calls the SAME
create_array_cheng2020_fig2() the parent used, one case at a time, and _case_matrix_row()
calls the same convert_synthetic_data_to_test_matrix(). Only the aggregation moved. Run
verify_fast_cost.py to check the total cost and every cost part against
firefly_optimizer.cost_function() on real parameter vectors; adopt this only if that reports
identical output.
"""
import copy
import traceback

import numpy as np

from covsyn.calibration import firefly_optimizer as fo
from covsyn.calibration.cost_parts import LAST as LAST_COST_PARTS
from covsyn.model.data_synthesis_main import run_covid
from covsyn.figures.plot_results import create_array_cheng2020_fig2
from covsyn.model.contact_measures import contacts_per_day_before_onset
from covsyn.data_processing.rw_data_processing import convert_synthetic_data_to_test_matrix

LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
MATRIX = {L: ('school_class_contacts_matrix' if L == 'school' else f'{L}_contacts_matrix')
          for L in LAYERS}
# The three groups the objective scores, and the create_array_cheng2020_fig2 layer names they
# are built from. 'Cheng others' is the sum of the other three layers, exactly as
# firefly_optimizer.generate_contact_result() sums them.
CHENG_GROUPS = (('Household', ('Household',)),
                ('Health care', ('Health care',)),
                ('Cheng others', ('School', 'Workplace', 'Municipality')))

_WORKER_DEMOGRAPHIC_PARAMETERS = None
_WORKER_TAIWAN_MATRIX_COLUMNS = None


def init_worker(demographic_parameters, taiwan_matrix_columns):
    """Give each worker the read-only data once, instead of once per task."""
    global _WORKER_DEMOGRAPHIC_PARAMETERS, _WORKER_TAIWAN_MATRIX_COLUMNS
    _WORKER_DEMOGRAPHIC_PARAMETERS = demographic_parameters
    _WORKER_TAIWAN_MATRIX_COLUMNS = taiwan_matrix_columns


class _OneCase:
    """The shape convert_synthetic_data_to_test_matrix() expects, holding a single case."""

    __slots__ = ('_value',)

    def __init__(self, demographic, course, contact):
        self._value = ([demographic], None, [course], [contact])

    def result(self):
        return self._value


def _case_cheng_bins(course, contact):
    """This case's contribution to the three groups' six-bin counts.

    Calls create_array_cheng2020_fig2() itself, so the bin edges, the dropping of asymptomatic
    cases and the first-contact-day arithmetic are the original code, not a copy of it.
    """
    bins = np.zeros((len(CHENG_GROUPS), 2, 6))
    one_course, one_contact = [course], [contact]
    for gi, (_group, layer_names) in enumerate(CHENG_GROUPS):
        for layer in layer_names:
            _, contact_array, _, infection_array = create_array_cheng2020_fig2(
                one_course, one_contact, layer=layer, expected_events=True)   # B55
            if contact_array.size == 6:
                bins[gi, 0] += contact_array
                bins[gi, 1] += infection_array
    return bins


def _case_matrix_row(demographic, course, contact, columns):
    """This case's row of the energy-distance test matrix, from the original converter."""
    dummy = np.empty((1, columns))
    return convert_synthetic_data_to_test_matrix([_OneCase(demographic, course, contact)],
                                                dummy, 1)[0]


def _index_case_scalars(course, contact):
    """Everything measure_outcomes() needs from one index case, as plain numbers.

    Each entry mirrors one block of firefly_optimizer.measure_outcomes(); the cross-case parts
    (the negative-binomial k, the community median and p90, the means and shares) stay in the
    parent because they are not per case.
    """
    out = {}
    offspring = 0.0
    for layer in LAYERS:
        effective = contact[f'{layer}_effective_contacts']
        effective = [] if effective is None else list(effective)
        offspring += float(np.nansum(np.asarray(effective, dtype=float))) if effective else 0.0
        out[f'candidate_{layer}'] = len(effective)
        out[f'effective_{layer}'] = sum(1 for x in effective if x == 1)
    out['offspring'] = offspring

    onset = course['incubation_period']
    for layer in LAYERS:
        # B55: ordinary contacts only, see contact_measures.contacts_per_day_before_onset.
        out[f'per_day_{layer}'] = contacts_per_day_before_onset(course, contact, layer)

    medical_early = medical_late = medical_total = 0
    if not (onset is None or np.isnan(onset)):
        matrix = np.asarray(contact['health_care_contacts_matrix'], dtype=float)
        if matrix.size:
            first = np.argmax(matrix > 0, axis=1) - onset
            medical_total = len(first)
            medical_early = int((first < 4).sum())
            medical_late = int((first >= 8).sum())
    out['medical_early'] = medical_early
    out['medical_late'] = medical_late
    out['medical_total'] = medical_total

    out['community'] = len(contact['municipality_effective_contacts'] or [])
    out['incubation_period'] = float(onset) if onset is not None else np.nan
    out['pre_onset_window'] = float(course['pre_onset_window'])
    out['positive_test_date'] = float(course['positive_test_date'])
    out['infection_day'] = float(course['infection_day'])
    out['date_of_critically_ill'] = float(course['date_of_critically_ill'])
    out['date_of_death'] = float(course['date_of_death'])
    out['date_of_recovery'] = float(course['date_of_recovery'])
    return out


def run_batch(seeds, P, columns):
    """One pool task: run several simulations and return only what the parent needs.

    Per simulation: the index case's scalars, and per CASE (in order) the Cheng bins and the
    test-matrix row. The parent concatenates these in task order, which reproduces the pooled
    order the original code built with np.append.

    The deepcopy matters. run_covid() mutates demographic_parameters, which is why
    _cost_function() deepcopied it for every evaluation and why each task then got its own
    unpickled copy. Sending it once through the pool initializer instead would make the
    worker's copy persist across tasks AND across evaluations, letting mutations accumulate;
    copying from the pristine global per task restores exactly the original semantics -- every
    task in every evaluation starts from the same untouched state -- while still shipping the
    0.21 MB object to each worker once rather than once per task.

    It is NOT what caused the first version of this module to differ from the parent-side code.
    That was a missing per-group term (the Health care bin weights, see _cost_function below);
    adding this deepcopy did not move the discrepancy by a single bit. Keep it anyway -- the
    mutation is real -- but do not read it as the fix for that bug.
    """
    demographic_parameters = copy.deepcopy(_WORKER_DEMOGRAPHIC_PARAMETERS)
    out = []
    for seed in seeds:
        demographic_list, _social, course_list, contact_list = run_covid(
            seed, P, demographic_parameters, save_file=False)
        cases = []
        for j, (course, contact) in enumerate(zip(course_list, contact_list)):
            cases.append((_case_cheng_bins(course, contact),
                          _case_matrix_row(demographic_list[j], course, contact, columns),
                          not np.isnan(course['incubation_period'])))
        index = (_index_case_scalars(course_list[0], contact_list[0])
                 if course_list and contact_list else None)
        out.append((index, cases))
    return out


def reduce_outcomes(scalars):
    """The cross-case half of measure_outcomes(), fed by _index_case_scalars().

    Every key of OUTCOME_TARGETS is always present (NaN when nothing supports it) so the
    progress CSV keeps a stable set of columns, exactly as measure_outcomes() does.
    """
    measured = {name: np.nan for name in fo.OUTCOME_TARGETS}
    if not scalars:
        return measured
    n = len(scalars)

    offspring = np.array([s['offspring'] for s in scalars], dtype=float)
    mean = offspring.mean()
    var = offspring.var(ddof=1) if n > 1 else 0.0
    if var > mean > 0:
        measured['offspring_k'] = mean ** 2 / (var - mean)
    elif mean > 0:
        measured['offspring_k'] = 100.0

    for layer in LAYERS:
        measured[f'daily_{layer}'] = float(
            np.mean([s[f'per_day_{layer}'] for s in scalars]))
        candidate = sum(s[f'candidate_{layer}'] for s in scalars)
        effective = sum(s[f'effective_{layer}'] for s in scalars)
        if candidate:
            measured[f'sar_{layer}'] = effective / candidate
        key = f'infections_per_index_{layer}'
        if key in measured:
            measured[key] = effective / n

    medical_total = sum(s['medical_total'] for s in scalars)
    if medical_total:
        measured['medical_late_share'] = sum(s['medical_late'] for s in scalars) / medical_total
        measured['medical_early_share'] = sum(s['medical_early'] for s in scalars) / medical_total

    community = np.array([s['community'] for s in scalars], dtype=float)
    measured['community_zero_share'] = float(np.mean(community == 0))
    # E73: mirrors measure_outcomes -- the charged median over every index case, the tail
    # statistics over the non-zero ones.
    measured['community_median'] = float(np.median(community))
    nonzero = community[community > 0]
    if len(nonzero) > 20:
        median, p90 = float(np.median(nonzero)), float(np.percentile(nonzero, 90))
        measured['community_p90'] = p90
        if median > 0:
            measured['community_tail_ratio'] = p90 / median

    incubation = np.array([s['incubation_period'] for s in scalars], dtype=float)
    symptomatic = ~np.isnan(incubation)
    measured['asymptomatic_share'] = float(np.mean(~symptomatic))
    if symptomatic.any():
        measured['incubation_mean'] = float(np.nanmean(incubation))

    pre_onset = np.array([s['pre_onset_window'] for s in scalars], dtype=float)
    if np.isfinite(pre_onset).any():
        measured['pre_onset_window_mean'] = float(np.nanmean(pre_onset))
        measured['pre_onset_zero_share'] = float(np.nanmean(pre_onset == 0))

    positive = np.array([s['positive_test_date'] for s in scalars], dtype=float)
    infection_day = np.array([s['infection_day'] for s in scalars], dtype=float)
    onset_to_confirmation = (positive - infection_day - incubation)[symptomatic]
    if np.isfinite(onset_to_confirmation).any():
        measured['onset_to_confirmation'] = float(np.nanmedian(onset_to_confirmation))

    critical = np.array([s['date_of_critically_ill'] for s in scalars], dtype=float)
    death = np.array([s['date_of_death'] for s in scalars], dtype=float)
    icu = ~np.isnan(critical)
    dead = ~np.isnan(death)
    if symptomatic.sum():
        measured['icu_share_of_symptom'] = float(icu.sum() / symptomatic.sum())
    if icu.sum():
        measured['death_share_of_icu'] = float(dead.sum() / icu.sum())
    measured['case_fatality'] = float(dead.mean())

    recovery = np.array([s['date_of_recovery'] for s in scalars], dtype=float)
    measured.update(fo.course_timing(recovery, positive, critical, infection_day, incubation,
                                     symptomatic))
    return measured


def cost_function(P, demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate,
                  norm_weights, seed_offset=0):
    """Same contract as firefly_optimizer.cost_function, including the 1e6 on failure (E31).

    seed_offset shifts the simulation seeds; 0, the default, is the objective the optimizer
    uses. revalidation.py scores candidates on other offsets (E87).
    """
    try:
        return _cost_function(P, demographic_parameters, executor, Cheng_contact_array,
                              Cheng_attack_rate, norm_weights, seed_offset)
    except Exception:
        traceback.print_exc()
        print('fast_cost: simulation failed for this parameter vector, charging '
              f'{fo.FAILED_EVALUATION_COST:g}', flush=True)
        LAST_COST_PARTS.clear()
        LAST_COST_PARTS.update(
            cost_contact=np.nan, cost_attack_rate=np.nan, cost_energy=np.nan,
            cost_penalty=np.nan, cost_outcome=np.nan, cost_contact_household=np.nan,
            cost_contact_healthcare=np.nan, cost_contact_others=np.nan,
            **{f'measured_{k}': np.nan for k in fo.OUTCOME_TARGETS})
        return fo.FAILED_EVALUATION_COST


def _cost_function(P, demographic_parameters, executor, Cheng_contact_array, Cheng_attack_rate,
                   norm_weights, seed_offset=0):
    source_case_number = fo.SIMULATIONS_PER_EVALUATION
    repeat_number = 1
    case_limit = source_case_number * repeat_number
    taiwan_data_matrix = np.load('./variable/Taiwan_data_matrix.npy')
    columns = taiwan_data_matrix.shape[1]

    seeds = list(range(seed_offset, seed_offset + case_limit))
    P_copy = copy.deepcopy(P)
    batches = [seeds[i:i + fo.SIMULATIONS_PER_TASK]
               for i in range(0, len(seeds), fo.SIMULATIONS_PER_TASK)]
    futures = [executor.submit(run_batch, batch, P_copy, columns) for batch in batches]

    index_scalars = []
    bins = np.zeros((len(CHENG_GROUPS), 2, 6))
    rows = []
    symptomatic_cases = 0
    for future in futures:
        for index, cases in future.result():
            if index is not None:
                index_scalars.append(index)
            for case_bins, case_row, symptomatic in cases:
                # Only the first `case_limit` POOLED cases feed the Cheng fit and the test
                # matrix: the original truncated the concatenated lists at that length, so
                # cases beyond it were never counted. Stopping here keeps that exactly.
                if len(rows) >= case_limit:
                    continue
                bins += case_bins
                rows.append(case_row)
                symptomatic_cases += symptomatic

    max_Cheng_contact = np.max(Cheng_contact_array)
    norm_Cheng_contact_array = Cheng_contact_array / max_Cheng_contact
    max_Cheng_attack_rate = np.max(Cheng_attack_rate)
    norm_Cheng_attack_rate = Cheng_attack_rate / max_Cheng_attack_rate

    contact_costs = []
    attack_rate_costs = []
    contact_scale = fo.cheng_contact_scale(symptomatic_cases)   # E86
    for i in range(len(CHENG_GROUPS)):
        contact_array = bins[i, 0].astype(float)
        infection_array = bins[i, 1].astype(float)
        norm_contact_array = contact_array / max_Cheng_contact
        attack_rate = np.divide(infection_array, contact_array,
                                out=np.zeros_like(infection_array), where=contact_array != 0)
        norm_attack_rate = attack_rate / max_Cheng_attack_rate
        norm_Cheng_data = norm_Cheng_contact_array[i]
        norm_Cheng_attack = norm_Cheng_attack_rate[i]
        # The Health care group weights its last two bins double. Missing this was the one
        # real defect the bit-identity check caught: it moved cost_contact_healthcare by 0.13
        # on every vector while leaving the other two groups exact, which is what pointed at a
        # per-group term rather than at the binning.
        if CHENG_GROUPS[i][0] == 'Health care':
            weights = np.array([1, 1, 1, 1, 2, 2])
            cost = np.sum((((norm_contact_array * contact_scale - norm_Cheng_data) * weights) ** 2))
        else:
            cost = np.sum(((norm_contact_array * contact_scale - norm_Cheng_data) ** 2))
        attack_rate_cost = np.nansum(((norm_attack_rate - norm_Cheng_attack) * norm_weights[i]) ** 2)
        contact_costs.append(cost)
        attack_rate_costs.append(attack_rate_cost)

    synthetic_data_matrix = np.ones((case_limit, columns)) * np.nan
    for i, row in enumerate(rows[:case_limit]):
        synthetic_data_matrix[i] = row
    _, energy_cost, _ = fo.estat(taiwan_data_matrix, synthetic_data_matrix, nboot=1)

    energy_weight = 1
    contact_cost = sum(contact_costs)
    attack_cost = fo.ATTACK_RATE_WEIGHT * sum(attack_rate_costs)
    penalty = fo.physiology_penalty(P)
    measured = reduce_outcomes(index_scalars)
    outcome = fo.outcome_penalty(measured)
    total_cost = contact_cost + attack_cost + energy_weight * energy_cost + penalty + outcome

    LAST_COST_PARTS.clear()
    LAST_COST_PARTS.update(
        cost_contact=float(contact_cost), cost_attack_rate=float(attack_cost),
        cost_energy=float(energy_weight * energy_cost), cost_penalty=float(penalty),
        cost_outcome=float(outcome),
        cost_contact_household=float(contact_costs[0]),
        cost_contact_healthcare=float(contact_costs[1]),
        cost_contact_others=float(contact_costs[2]),
        **{f'measured_{k}': float(v) for k, v in measured.items()})
    return total_cost
