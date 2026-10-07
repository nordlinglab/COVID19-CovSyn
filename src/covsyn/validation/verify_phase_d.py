"""Check every post-simulation item recorded against decisions B14 and B17-B36.

Each decision in covsyn_decisions.md ends with a list of things to verify once the model has
been rerun. This script runs that list against the Monte-Carlo output and prints one line per
item with its target, the measured value and a verdict, so the rerun can be accepted or
rejected on the record rather than on impressions.

Usage:
    python -m covsyn.validation.verify_phase_d [spread_dir] [first_outbreak_dir] [out_json]

spread_dir          synthetic_data_results_spread_Taiwan_weight (single seed, 84 days)
first_outbreak_dir  synthetic_data_results_taiwan_first_outbreak (28 seeds), optional
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from covsyn.model.contact_measures import contacts_per_day_before_onset

SPREAD = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight')
FIRST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('synthetic_data_results_taiwan_first_outbreak')
OUT_JSON = Path(sys.argv[3] if len(sys.argv) > 3 else 'validation_reference/phaseD_checks.json')
LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
MATRIX = {L: ('school_class_contacts_matrix' if L == 'school' else f'{L}_contacts_matrix') for L in LAYERS}

results = []


def check(decision, name, value, target, unit='', ok=None, note=''):
    """Record one checklist line. target is (lo, hi), a string, or None for information."""
    if ok is None and isinstance(target, tuple) and value is not None and np.isfinite(value):
        ok = target[0] <= value <= target[1]
    results.append({'decision': decision, 'name': name, 'value': None if value is None else float(value),
                    'target': list(target) if isinstance(target, tuple) else target,
                    'unit': unit, 'ok': None if ok is None else bool(ok), 'note': note})


def load(directory):
    """Load every Monte-Carlo file of a run, case by case."""
    files = sorted(glob.glob(str(directory / 'course_of_disease_data_*.npy')),
                   key=lambda p: int(Path(p).stem.split('_')[-1]))
    runs = []
    for f in files:
        k = int(Path(f).stem.split('_')[-1])
        try:
            course = np.load(f, allow_pickle=True)
            contact = np.load(directory / f'contact_data_{k}.npy', allow_pickle=True)
            social = np.load(directory / f'social_data_{k}.npy', allow_pickle=True)
            demo = np.load(directory / f'demographic_data_{k}.npy', allow_pickle=True)
        except (FileNotFoundError, ValueError):
            continue
        runs.append({'k': k, 'course': list(course), 'contact': list(contact),
                     'social': list(social), 'demo': list(demo)})
    return runs


def offspring_counts(contacts):
    return np.array([sum(np.nansum(np.asarray(c[f'{L}_effective_contacts'], dtype=float)) for L in LAYERS)
                     for c in contacts], dtype=float)


def negative_binomial_k(counts):
    mean, var = counts.mean(), counts.var(ddof=1)
    return mean ** 2 / (var - mean) if var > mean > 0 else np.inf


def main():
    runs = load(SPREAD)
    if not runs:
        raise SystemExit(f'no Monte-Carlo output found in {SPREAD}')
    course = [c for r in runs for c in r['course']]
    contact = [c for r in runs for c in r['contact']]
    social = [s for r in runs for s in r['social']]
    demo = [d for r in runs for d in r['demo']]
    index_course = [r['course'][0] for r in runs]
    index_contact = [r['contact'][0] for r in runs]
    index_social = [r['social'][0] for r in runs]
    index_demo = [r['demo'][0] for r in runs]
    print(f'{SPREAD}: {len(runs)} simulations, {len(course)} cases\n')

    def field(key, source, cast=float):
        return np.array([x.get(key, np.nan) for x in source], dtype=cast)

    # ---------------------------------------------------------------- B29 household
    others = field('household_size', index_social)
    check('B29', 'household: other members per case', others.mean(), (2.6, 3.0))
    check('B29', 'household: living alone', 100 * np.mean(others == 0), (11.0, 16.0), '%')

    # ---------------------------------------------------------------- B30 school
    ages = field('age', index_demo)
    classes = field('school_class_size', index_social)
    for label, lo, hi, target in [('elementary (7-12)', 7, 12, (21, 27)),
                                  ('junior high (13-15)', 13, 15, (24, 31)),
                                  ('senior high (16-18)', 16, 18, (29, 36)),
                                  ('university (19-22)', 19, 22, (75, 100))]:
        band = (ages >= lo) & (ages <= hi) & (classes > 0)
        check('B30', f'class size, {label}', classes[band].mean() if band.sum() else np.nan,
              target, note=f'n={int(band.sum())}')

    # ---------------------------------------------------------------- B31 workplace
    enterprise = field('enterprise_size', index_social)
    group = field('work_group_size', index_social)
    employed = group > 0
    check('B31', 'work group size (median)', np.median(group[employed]) if employed.any() else np.nan,
          (5, 20), note=f'n={int(employed.sum())}')
    # Chen 2022 (BMJ Open 12:e055643) found the number of cases per workplace cluster
    # essentially constant across enterprise sizes (median 4, staff medians 16 to 650), so
    # the share of staff infected fell from 17.8% to 4.3% to 1.1%. Its numerator is a whole
    # cluster over all its generations; CovSyn's work groups are drawn per case and are not
    # shared between cases, so there is no cluster to count and the absolute 17.8% is not
    # reachable here. What IS testable, and is the actual claim, is the SHAPE: the number a
    # case infects at work must not grow with the size of the company, which then makes the
    # share of staff fall roughly as 1/staff.
    bands = [('1-49 staff', 1, 49), ('50-249 staff', 50, 249), ('250+ staff', 250, 10 ** 9)]
    per_case, per_staff = {}, {}
    for label, lo, hi in bands:
        band = (enterprise >= lo) & (enterprise <= hi)
        if band.sum() < 20:
            continue
        infected = np.array([np.nansum(np.asarray(c['workplace_effective_contacts'], dtype=float))
                             for c, keep in zip(index_contact, band) if keep])
        per_case[label] = float(infected.mean())
        per_staff[label] = float(100 * infected.sum() / enterprise[band].sum())
        check('B31', f'workplace infections per case, {label}', per_case[label], None,
              note=f'n={int(band.sum())}; must not grow with company size (Chen 2022)')
        check('B31', f'workplace infections per staff, {label}', per_staff[label], None, '%',
              note='Chen 2022 whole-cluster gradient 17.8 / 4.3 / 1.1; CovSyn counts one '
                   'generation, so only the shape is comparable')
    if len(per_case) == 3:
        small, large = per_case['1-49 staff'], per_case['250+ staff']
        # Chen 2022 does NOT say the cluster is constant: its cases per cluster run from 3 to
        # 8 while the staff median runs from 16 to 650, which is a ratio of 2.7 over a 40-fold
        # change in company size. The earlier target of 0.7-1.6 read "constant" into the paper
        # and failed a model that was actually flatter than the data.
        check('B31', 'infections per case, 250+ staff / 1-49 staff',
              large / small if small else np.nan, (1.0, 3.0),
              note='Chen 2022 cases per cluster 3 -> 8 over staff 16 -> 650, i.e. 2.7x')
        small, large = per_staff['1-49 staff'], per_staff['250+ staff']
        check('B31', 'infections per staff, 1-49 staff / 250+ staff',
              small / large if large else np.nan, (4.0, 60.0),
              note='Chen 2022 observed 17.8 / 1.1 = 16x')

    # ---------------------------------------------------------------- B25 daily contacts
    targets = {'household': (1.5, 2.2), 'school': (0.9, 1.4), 'workplace': (1.2, 1.8),
               'municipality': (1.2, 1.8), 'health_care': (0.0, 0.3)}
    total = 0.0
    for L in LAYERS:
        # B55: ordinary contacts only, the same function the objective uses.
        per_day = [contacts_per_day_before_onset(k, c, L)
                   for c, k in zip(index_contact, index_course)]
        value = float(np.mean(per_day))
        total += value
        check('B25', f'contacts per day before onset, {L}', value, targets[L])
    check('B25', 'contacts per day, all layers', total, (5.0, 7.0),
          note='2020 national survey: 6.03 close contacts per day')

    # household: how often is each member actually met
    met = []
    for c in index_contact:
        m = np.asarray(c['household_contacts_matrix'], dtype=float)
        if m.size:
            met.append(m.mean())
    check('B25', 'household: share of days each member is met', 100 * np.mean(met) if met else np.nan,
          (50.0, 100.0), '%', note='was 27% before Phase D')

    # ---------------------------------------------------------------- B32 / E5
    anomalies = 0
    zero_contact = 0
    for c in contact:
        any_candidate = any(np.asarray(c[MATRIX[L]], dtype=float).size for L in LAYERS)
        any_effective = any(len(c[f'{L}_effective_contacts']) for L in LAYERS)
        if not any_candidate:
            zero_contact += 1
        elif not any_effective:
            anomalies += 1
    check('B32', 'cases with candidate contacts but no effective-contact record', anomalies, (0, 0),
          note='was the E5 signature')
    check('B32', 'cases with no contact in any layer', 100 * zero_contact / len(contact), (0.0, 4.0), '%',
          note='was 6.4% before Phase D')

    # ------------------------------------------------- B8 / B9 / B13 per-layer cumulative SAR
    # Infections per candidate contact over the whole window, which is the quantity the
    # calibration anchors are written in. Measured on index cases, matching the design of
    # the tracing studies the anchors come from.
    # Taken from sar_anchors.py so that the acceptance test and the search bounds cannot
    # state different anchors, which they did until 2026-09-25 (household was checked
    # against 4.6-10.1 while the bounds were built from a different pair of numbers again).
    from covsyn.calibration.sar_anchors import LAYER_CUMULATIVE_SAR, LAYER_INFECTIONS_PER_INDEX
    anchors = {layer: (100 * centre, 100 * lo, 100 * hi)
               for layer, (lo, centre, hi) in LAYER_CUMULATIVE_SAR.items()}
    per_index = {}
    for layer, (anchor, lo, hi) in anchors.items():
        candidate = effective = 0
        for c in index_contact:
            eff = list(c[f'{layer}_effective_contacts'] or [])
            candidate += len(eff)
            effective += sum(1 for x in eff if x == 1)
        per_index[layer] = (effective / len(index_contact), candidate / len(index_contact))
        # E56: for the three layers Cheng reports this rate is now REPORTED, not accepted on.
        # Dividing his infection count by CovSyn's own candidate-contact count made the target
        # a ratio whose denominator the search controls, and run 5 met the health care rate
        # while medical infections per index case fell further below Cheng's 0.06. The rate is
        # still worth printing -- it is what places the daily attack-rate bounds -- but the
        # acceptance test is the absolute count below.
        charged = layer not in LAYER_INFECTIONS_PER_INDEX
        check('B9', f'cumulative SAR per contact, {layer}',
              100 * effective / candidate if candidate else np.nan,
              (lo, hi) if charged else None, '%',
              note=f'calibration anchor {anchor}%; n={candidate} candidate contacts'
                   + ('' if charged else '; informational since E56, accepted on the count'))

    # E56 / E57: what one index case actually infects in each layer, which is the quantity
    # Cheng measured and the quantity the objective now charges. Every layer is printed; the
    # three Cheng reports are the ones with an acceptance interval.
    for layer, (infections, contacts) in per_index.items():
        target = LAYER_INFECTIONS_PER_INDEX.get(layer)
        cheng = {'household': 0.10, 'health_care': 0.06, 'municipality': 0.01}.get(layer)
        note = f'CovSyn candidate contacts per index {contacts:.3f}'
        if cheng is not None:
            note += (f'; Cheng 2020 {cheng:.2f} per index case'
                     f' -> CovSyn is {infections / cheng:.2f}x')
            if layer == 'municipality':
                note += '; upper capped at 3x the centre (E57), not the Poisson CI'
        else:
            note += '; Cheng has no category for this layer, reported only'
        check('B9', f'infections per index case, {layer}', infections,
              (target[0], target[2]) if target else None, '', note=note)

    # ---------------------------------------------------------------- B17 / B27 dispersion
    counts = offspring_counts(index_contact)
    k_hat = negative_binomial_k(counts)
    # Upper bound raised 0.30 -> 0.35 so that this test and the cost term agree; they used to
    # state different intervals for the same quantity (OUTCOME_TARGETS has 0.20-0.35).
    check('B17', 'offspring dispersion k', k_hat, (0.10, 0.35),
          note='Taiwan tracing NegBin MLE k = 0.29')
    check('B17', 'R (mean offspring of an index case)', counts.mean(), (0.3, 0.6),
          note='Taiwan tracing R = 0.43')
    check('B17', 'cases infecting 3 or more', 100 * np.mean(counts >= 3), (2.0, 8.0), '%',
          note='Taiwan tracing 4.3%')
    check('B17', 'largest number infected by one case', counts.max(), (5, 30),
          note='Taiwan tracing maximum 8')

    community = np.array([len(c['municipality_effective_contacts']) for c in index_contact], dtype=float)
    check('B27', 'community contacts per case, median', np.median(community), (3.0, 15.0),
          note='Taiwan tracing friend+other median 7.5')
    # The maximum over 1000 cases is a single order statistic and says almost nothing; the
    # p90 and the shape do. The tracing records have a median of 7 and a p90 of 172.
    nonzero = community[community > 0]
    # B47 made the p90 informational (its band's centre, 110, is out of the model's reach) and
    # B50 checks the ratio on the band the objective charges -- imported, not copied (lesson 5).
    from covsyn.calibration.firefly_optimizer import OUTCOME_TARGETS
    check('B27', 'community contacts per case, p90',
          np.percentile(nonzero, 90) if len(nonzero) else np.nan, None,
          note='Taiwan tracing 2020 first wave p90 = 195 (n=38); informational since B47')
    check('B50', 'community contacts, p90 / median',
          (np.percentile(nonzero, 90) / np.median(nonzero)) if len(nonzero) and np.median(nonzero) > 0 else np.nan,
          tuple(OUTCOME_TARGETS['community_tail_ratio'][:2]),
          note='Taiwan tracing 2020 first wave 26.0 (n=38), bootstrap 95% CI [5.5, 93.1] (E77)')
    check('B27', 'community contacts per case, maximum', community.max(), None,
          note='Taiwan tracing maximum 850 (informational: one order statistic)')
    # decoupled from city population (finding E4)
    city = np.array([s['municipality'] for s in index_social])
    by_city = {c: community[city == c].mean() for c in set(city) if (city == c).sum() >= 20}
    spread_ratio = (max(by_city.values()) / max(min(by_city.values()), 1e-9)) if by_city else np.nan
    check('B27', 'largest / smallest city mean community contacts', spread_ratio, (1.0, 1.6),
          note='was 12x, proportional to city population (E4)')

    # ---------------------------------------------------------------- B22 / B26 timing
    incubation = field('incubation_period', index_course)
    window = field('pre_onset_window', index_course)
    latent = field('latent_period', index_course)
    isolation = field('monitor_isolation_period', index_course)
    positive = field('positive_test_date', index_course)
    infection_day = field('infection_day', index_course)
    symptomatic = ~np.isnan(incubation)
    check('B22', 'incubation period, mean', np.nanmean(incubation), (3.9, 8.0))
    check('B22', 'pre-onset infectious window, mean', np.nanmean(window), (1.0, 3.0))
    check('B22', 'pre-onset window of zero days', 100 * np.nanmean(window == 0), (0.0, 12.0), '%',
          note='was 20% before Phase D')
    check('B22', 'latent period, mean', latent.mean(), (4.1, 5.5),
          note='B3: widened to the whole literature reported-mean range')
    onset_to_confirm = positive - infection_day - incubation
    check('B2', 'onset to confirmation, median', np.nanmedian(onset_to_confirm[symptomatic]), (5.0, 7.0),
          note='Taiwan tracing median 6 d (n=442); Ge 2021 onset to isolation 5 d; published CovSyn fits 8.17 d')
    # Re-derived for B2 (2026-09-26). The old ceiling of 10.5 was simply "what the model did
    # before Phase D", set while onset -> confirmation was fitted at 1 day. With B2 putting
    # that median at 5-7 days on top of a 5.3-day incubation, infection -> isolation has to
    # land near 11-12 days; the fourth run gave 10.939 and was marked FAIL against a bound
    # that its own decision had already invalidated.
    check('B26', 'infection to isolation, mean', isolation.mean(), (8.0, 14.0),
          note='B2: incubation about 5.3 d plus an onset-to-confirmation median of 5-7 d')
    routes = [c.get('isolation_route') for c in course]
    for route in ('symptom', 'traced', 'untraced', 'critical'):  # 'critical': B52
        check('B26', f'isolation route: {route}', 100 * routes.count(route) / len(routes), None, '%')

    # share of contacts that happen before onset (Cheng: 27.5%)
    before = after8 = totalc = 0
    med_before = med_after8 = med_early = med_total = 0
    for c, k in zip(index_contact, index_course):
        onset = k['incubation_period']
        if onset is None or np.isnan(onset):
            continue
        for L in LAYERS:
            m = np.asarray(c[MATRIX[L]], dtype=float)
            if m.size == 0:
                continue
            first = np.argmax(m > 0, axis=1) - onset
            totalc += len(first)
            before += int((first < 0).sum())
            after8 += int((first >= 8).sum())
            if L == 'health_care':
                med_total += len(first)
                med_before += int((first < 0).sum())
                med_after8 += int((first >= 8).sum())
                med_early += int((first < 4).sum())
    check('B26', 'contacts starting before symptom onset', 100 * before / max(totalc, 1), (20.0, 40.0), '%',
          note='Cheng 2020: 27.5%; was 54% before Phase D')
    check('B23', 'health care contacts starting 8+ days after onset',
          100 * med_after8 / max(med_total, 1), (20.0, 50.0), '%',
          note='Cheng 2020 medical: 256/697 = 36.7%; was ~1-3% before Phase D')
    # E59: the other end of the same distribution. Cheng's medical contacts are mostly EARLY
    # (his <0 and 0-3 bins hold 33.9% and 21.5%, so 55.4% start before day 4), and charging
    # only the late tail let run 5 reverse the shape -- 9.0% early, 50.7% late -- without any
    # acceptance item noticing. Same bin edge as firefly_optimizer's medical_early_share.
    check('B23', 'health care contacts starting before day 4',
          100 * med_early / max(med_total, 1), (40.0, 70.0), '%',
          note='Cheng 2020 medical: 33.9% before onset + 21.5% on days 0-3 = 55.4%')

    # ---------------------------------------------------------------- B28 case closure
    recovery = field('date_of_recovery', index_course) - infection_day
    check('B28', 'infection to case closure, symptomatic', np.nanmean(recovery[symptomatic]), (20.0, 32.0),
          note='Taiwan: onset to release about 25 days')
    check('B28', 'infection to case closure, asymptomatic', np.nanmean(recovery[~symptomatic]), (20.0, 32.0))

    # ---------------------------------------------------------------- B33 / B20 severity
    icu = ~np.isnan(field('date_of_critically_ill', index_course))
    dead = ~np.isnan(field('date_of_death', index_course))
    check('B33', 'asymptomatic share', 100 * np.mean(~symptomatic), (20.0, 28.0), '%',
          note='Taiwan tracing 23.7%')
    check('B33', 'symptomatic to ICU', 100 * icu.sum() / max(symptomatic.sum(), 1), (9.0, 17.0), '%',
          note='Taiwan tracing 12.7%')
    check('B33', 'ICU to death', 100 * dead.sum() / max(icu.sum(), 1), (7.0, 19.0), '%',
          note='Taiwan tracing 12.5%')
    check('B33', 'case fatality', 100 * dead.mean(), (0.7, 2.0), '%', note='Taiwan tracing 1.2%')
    band = np.digitize(ages, [20, 40, 60])
    icu_by_band = []
    for b, label in enumerate(['0-19', '20-39', '40-59', '60+']):
        m = band == b
        rate = 100 * icu[m].sum() / max((symptomatic & m).sum(), 1) if m.sum() else np.nan
        icu_by_band.append(rate)
        check('B20', f'symptomatic to ICU, {label}', rate, None, '%', note=f'n={int(m.sum())}')
    if np.isfinite(icu_by_band[1]) and icu_by_band[1] > 0:
        check('B20', 'ICU rate 60+ / 20-39', icu_by_band[3] / icu_by_band[1], (3.0, 12.0),
              note='Verity 2020 gradient')

    # ---------------------------------------------------------------- B14 age risk ratio
    contact_ages, infected_ages = [], []
    for c in index_contact:
        for L in LAYERS:
            contact_ages.extend(np.asarray(c[f'{L}_contact_ages'], dtype=float).tolist())
            infected_ages.extend([a for a in np.asarray(c[f'{L}_secondary_contact_ages'], dtype=float)
                                  if np.isfinite(a)])
    contact_ages = np.array(contact_ages)
    infected_ages = np.array(infected_ages)
    if len(contact_ages) and len(infected_ages):
        rr = []
        for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 200)]:
            denom = ((contact_ages >= lo) & (contact_ages < hi)).sum()
            numer = ((infected_ages >= lo) & (infected_ages < hi)).sum()
            rr.append(numer / denom if denom else np.nan)
        rr = np.array(rr) / rr[1] if rr[1] else np.array(rr)
        # E68 / E70: these five-layer numbers are measured on the ~1000 index cases of this run,
        # which held 226 infections over four bands on run 6 -- health care 60+ rested on a single
        # infection. They are kept as INFORMATION; the acceptance test is the high-N,
        # Cheng-comparable measurement below, from measure_age_rr.py.
        for value, label in zip(rr, ['0-19', '20-39', '40-59', '60+']):
            check('B14', f'age risk ratio, all five layers, {label}', value, None,
                  note='information: this run only, all layers pooled. Cheng has no school or '
                       'workplace category, and CovSyn school carries a 2.3% attack rate '
                       'concentrated in 0-19, so this pooling is not comparable to his (E70)')

    age_rr_file = Path('validation_reference/age_rr.json')
    if age_rr_file.exists():
        measurement = json.loads(age_rr_file.read_text())
        layers = ', '.join(measurement.get('cheng_comparable_layers', []))
        n_cases = measurement.get('n_index_cases')
        infections = measurement.get('cheng_comparable_infections')
        for label, band in measurement.get('bands', {}).items():
            lo, hi = band['target']
            ci_lo, ci_hi = band['ci']
            check('B14', f'measured age risk ratio, {label}', band['rr'], (lo, hi),
                  note=(f'{n_cases} index cases, {infections} infections over {layers} '
                        f'(the settings Cheng traced); 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]; '
                        f'{band["precision"]}; reference: '
                        + band.get('band_source', 'see measure_age_rr.py')))
        # E70 / N3: the layer-by-layer ratios are an INPUT-REPRODUCTION check -- the reference is
        # the model's own locked setting, so agreement says the sampler works, not that the model
        # matches reality. Kept apart from the acceptance rows above, which compare against
        # external estimates. todolist923 1.10 asks for exactly this split.
        locked = measurement.get('locked_input')
        if locked:
            check('B14', 'locked age risk ratio input', None,
                  '[%s]' % ', '.join('%.2f' % v for v in locked),
                  note='read from variable/course_parameters.npy; NOT a validation target -- the '
                       'rows below check that the sampler reproduces it (input reproduction, N3)')
        for layer, gap in measurement.get('layer_input_reproduction_gap', {}).items():
            check('B14', f'input reproduction, {layer}', gap, None,
                  note='largest |measured RR - locked input| over the age bands this layer has; '
                       'implementation check, not independent validation (N3)')
        for value, label in zip(measurement.get('all_layer_rr', []),
                                ['0-19', '20-39', '40-59', '60+']):
            check('B14', f'age risk ratio, all five layers (high N), {label}', value, None,
                  note='information: the same high-N measurement pooled over all five layers, '
                       'for the size of the composition effect (E70)')
    else:
        check('B14', 'high-N age risk ratio measurement', None, 'measure_age_rr.py has not run',
              note='B14 is accepted on that measurement since E68; without it only the '
                   'informational rows above are available')

    # ---------------------------------------------------------------- B36 Taiwan first wave
    if FIRST.exists():
        first_runs = load(FIRST)
        if first_runs:
            sizes = np.array([len(r['course']) for r in first_runs], dtype=float)
            deaths = np.array([sum(1 for c in r['course'] if not np.isnan(c['date_of_death']))
                               for r in first_runs], dtype=float)
            kept = sizes >= 29
            check('B36', 'first wave cases per simulation, all runs', sizes.mean(), None,
                  note=f'observed 55 reported cases; n={len(sizes)}')
            check('B36', 'first wave cases per simulation, runs with at least 1 onward case',
                  sizes[kept].mean() if kept.any() else np.nan, None,
                  note=f'the filter used in taiwan_first_outbreak.ipynb; n={int(kept.sum())}')
            check('B36', 'first wave deaths per simulation', deaths.mean(), None,
                  note='observed 3; expected about 0.6 under the B33 cascade, see B36')
            first_ages = np.array([d['age'] for r in first_runs for d in r['demo']], dtype=float)
            check('B34', 'Taiwan scenario mean case age', first_ages.mean(), (33.0, 42.0),
                  note='observed tracing mean 37.1')

    # ---------------------------------------------------------------- report
    width = max(len(r['name']) for r in results)
    current = None
    passed = failed = 0
    for r in results:
        if r['decision'] != current:
            current = r['decision']
            print(f'--- {current}')
        target = r['target']
        if isinstance(target, list):
            target_text = '[%g, %g]' % (target[0], target[1])
        else:
            target_text = '(information)'
        if r['ok'] is None:
            verdict = '    '
        elif r['ok']:
            verdict = ' ok '
            passed += 1
        else:
            verdict = 'FAIL'
            failed += 1
        value = 'n/a' if r['value'] is None or not np.isfinite(r['value']) else '%9.3f' % r['value']
        print('  %s %-*s %s%-2s  %-16s %s' % (verdict, width, r['name'], value, r['unit'],
                                              target_text, r['note']))
    print(f'\n{passed} checks passed, {failed} failed, '
          f'{len(results) - passed - failed} informational')

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump({'spread_dir': str(SPREAD), 'simulations': len(runs), 'cases': len(course),
                   'checks': results}, f, indent=1)
    print('written', OUT_JSON)


if __name__ == '__main__':
    main()
