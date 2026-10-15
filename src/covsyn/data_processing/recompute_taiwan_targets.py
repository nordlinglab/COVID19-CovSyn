"""Recompute every CovSyn calibration or acceptance value that is derived from Taiwan's 2020 data.

Each value in docs/calibration_sources.md marked as computed from data is recomputed here from
its input file, with its definition written next to the computation, so that no value rests on a
number copied from an earlier note.

Inputs (paths relative to the CovSyn root):
  data/structured_course_of_disease_data/figshare_taiwan_covid.xlsx   sheet 'Individual Data'
      the public first-wave case file (578 rows; case 530 is absent from this version)
  validation_reference/taiwan_tracing_contacts_per_case.csv           per case and layer
  validation_reference/taiwan_infection_events.csv                    infector -> infectee links
  validation_reference/taiwan_coworker_close_contacts.csv             coworker close contacts
      the three CSVs were extracted by extract_tracing_reference.py and
      extract_coworker_contacts.py from the forecasting repository's workbooks

Output: validation_reference/recomputed_targets.csv, one row per value, and the same table on stdout.

Usage: python -m covsyn.data_processing.recompute_taiwan_targets [output_csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special

from covsyn.model.contact_measures import offspring_dispersion

CASES = Path('data/structured_course_of_disease_data/figshare_taiwan_covid.xlsx')
TRACING = Path('validation_reference/taiwan_tracing_contacts_per_case.csv')
EVENTS = Path('validation_reference/taiwan_infection_events.csv')
COWORKERS = Path('validation_reference/taiwan_coworker_close_contacts.csv')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('validation_reference/recomputed_targets.csv')
WAVE = 'first_wave_2020'
REPLICATES = 20_000  # as in E77
RNG = np.random.default_rng(0)
ROWS = []


def record(item, quantity, value, n, method, data, interval=(np.nan, np.nan)):
    ROWS.append(dict(item=item, quantity=quantity, value=value, n=n, ci_low=interval[0],
                     ci_high=interval[1], method=method, data=data))


def boot(x, statistic=np.mean):
    """Percentile bootstrap 95% interval of statistic(x)."""
    x = np.asarray(x, float)
    reps = [statistic(RNG.choice(x, len(x))) for _ in range(REPLICATES)]
    return tuple(np.percentile(reps, [2.5, 97.5]))


def days(later, earlier):
    return (later - earlier).dt.days


def nb_k_mle(x):
    """Negative-binomial dispersion k by maximum likelihood: the estimator the objective and the
    acceptance check use (contact_measures.offspring_dispersion, B61)."""
    return offspring_dispersion(x)


def nb_k_moments(x):
    """Method-of-moments k, the estimator the objective and the checklist used before B61."""
    m, v = np.mean(x), np.var(x, ddof=1)
    return m ** 2 / (v - m) if v > m > 0 else np.inf


def clauset_discrete(x):
    """Discrete power-law tail (Clauset, Shalizi & Newman 2009, sections 3.1 and 3.3).

    For each candidate x_min, alpha maximises the exact discrete likelihood
    -alpha * sum(ln x) - n * ln zeta(alpha, x_min) (Hurwitz zeta), and the KS distance compares
    the empirical CDF of the tail with the model CDF 1 - zeta(alpha, x + 1) / zeta(alpha, x_min).
    x_min is the candidate with the smallest KS distance."""
    x = np.sort(np.asarray(x, float))
    best = None
    for xmin in np.unique(x)[:-1]:
        tail = x[x >= xmin]
        if len(tail) < 2:
            continue
        slog = np.log(tail).sum()
        nll = lambda a: a * slog + len(tail) * np.log(special.zeta(a, xmin))
        alpha = optimize.minimize_scalar(nll, bounds=(1.0001, 6.0), method='bounded').x
        values = np.unique(tail)
        emp = np.searchsorted(tail, values, side='right') / len(tail)
        model = 1 - special.zeta(alpha, values + 1) / special.zeta(alpha, xmin)
        ks = np.max(np.abs(emp - model))
        if best is None or ks < best[2]:
            best = (xmin, alpha, ks, len(tail))
    return best


def severity_and_timing():
    d = pd.read_excel(CASES, sheet_name='Individual Data')
    data = f'{CASES.name}, sheet Individual Data ({len(d)} rows)'
    for col in ('onset_of_symptom', 'confirmed_date', 'icu', 'recovery', 'death_date'):
        d[col] = pd.to_datetime(d[col], errors='coerce')
    n = len(d)
    no_onset = d.onset_of_symptom.isna()
    # The forecasting workbook marks "no symptom" with X; the public figshare file leaves it blank.
    symptom_x = d.symptom.isna() | d.symptom.astype(str).str.strip().str.upper().eq('X')
    symptomatic = ~no_onset
    icu, dead = d.icu.notna(), d.death_date.notna()

    record('1.7', 'asymptomatic share, broad (no onset date)', no_onset.mean(), n,
           'cases without onset_of_symptom / all cases', data)
    record('1.7', 'asymptomatic share, strict (no onset date and symptom = X)',
           (no_onset & symptom_x).mean(), n, 'cases with neither an onset date nor a symptom / all cases (E47)', data)
    record('1.8', 'ICU share of symptomatic, onset date required', (icu & symptomatic).sum() / symptomatic.sum(),
           int(symptomatic.sum()), 'ICU cases with an onset date / cases with an onset date', data)
    severe_or_onset = symptomatic | icu
    record('1.8', 'ICU share of symptomatic, ICU counted as symptomatic', icu.sum() / severe_or_onset.sum(),
           int(severe_or_onset.sum()), 'all ICU cases / (cases with an onset date + ICU cases without one); '
           'an ICU case without an onset date is taken as a missing date, not as asymptomatic', data)
    record('1.8', 'ICU cases without an onset date', int((icu & ~symptomatic).sum()), n,
           'count, reported so the ICU numerator can be checked', data)
    record('1.9', 'death share of ICU', (dead & icu).sum() / icu.sum(), int(icu.sum()),
           'ICU cases with a death date / ICU cases', data)
    record('1.9', 'deaths without an ICU date', int((dead & ~icu).sum()), n, 'count', data)
    record('1.10', 'case fatality', dead.mean(), n, 'cases with a death date / all cases', data)

    def timing(item, name, x, note):
        x = x.dropna()
        x = x[x >= 0]
        record(item, f'{name}, mean (days)', x.mean(), len(x), f'{note}; negative intervals dropped; '
               f'interval = bootstrap 95% of the mean', data, boot(x))
        record(item, f'{name}, median (days)', x.median(), len(x), note, data)

    has_recovery = d.recovery.notna()
    timing('1.25', 'confirmation -> recovery, symptomatic', days(d.recovery, d.confirmed_date)[symptomatic & ~icu & has_recovery],
           'recovery - confirmed_date, cases with an onset date and no ICU')
    timing('1.25', 'confirmation -> recovery, symptomatic incl. ICU', days(d.recovery, d.confirmed_date)[symptomatic & has_recovery],
           'recovery - confirmed_date, cases with an onset date')
    timing('1.26', 'confirmation -> recovery, asymptomatic', days(d.recovery, d.confirmed_date)[no_onset & has_recovery],
           'recovery - confirmed_date, cases without an onset date')
    timing('1.27', 'ICU -> recovery', days(d.recovery, d.icu)[icu], 'recovery - icu')
    timing('1.28', 'onset -> ICU', days(d.icu, d.onset_of_symptom)[icu], 'icu - onset_of_symptom')
    timing('1.29', 'onset -> confirmation', days(d.confirmed_date, d.onset_of_symptom), 'confirmed_date - onset_of_symptom')
    d['earliest_infection_date'] = pd.to_datetime(d.earliest_infection_date, errors='coerce')
    timing('4.8', 'infection -> confirmation', days(d.confirmed_date, d.earliest_infection_date),
           'confirmed_date - earliest_infection_date, cases with a recorded infection date')
    timing('4.8', 'infection -> onset (incubation)', days(d.onset_of_symptom, d.earliest_infection_date),
           'onset_of_symptom - earliest_infection_date, cases with both dates')

    lo = {'0 to 9': 0, '10 to 19': 10, '20 to 29': 20, '30 to 39': 30, '40 to 49': 40, '50 to 59': 50,
          '60 to 69': 60, '70 to 79': 70, '80 to 89': 80, '90 to 99': 90}
    bands = d.age.astype(str).str.strip()
    mid = bands.map({k: v + 5 for k, v in lo.items()})
    record('4.12', 'mean case age (band midpoints)', mid.mean(), int(mid.notna().sum()),
           'mean of the midpoints of the 10-year age bands; bands not of that form are left out '
           f'({sorted(set(bands[mid.isna()]))})', data, boot(mid.dropna()))


def tracing():
    t = pd.read_csv(TRACING)
    t = t[t.dataset == WAVE]
    data = f'{TRACING.name}, dataset {WAVE}'
    n_cases = int(pd.read_csv(Path('validation_reference/taiwan_case_timeline.csv')).dataset.eq(WAVE).sum())

    # Offspring, upper definition (E80): confirmed contacts per case, summed over layers, 0 if no record.
    per = t.groupby('case_id').infected.sum()
    a = np.zeros(n_cases)
    a[:len(per)] = per.values
    m = 'infected contacts per case summed over layers, cases with no record = 0 (E80 upper definition)'
    record('1.1', 'offspring k, upper definition, MLE', nb_k_mle(a), n_cases, m, data)
    record('1.1', 'offspring k, upper definition, method of moments', nb_k_moments(a), n_cases,
           m + '; the estimator used before B61', data)
    record('4.7', 'R, upper definition', a.mean(), n_cases, m, data)
    record('4.7', 'share infecting >= 3, upper definition', np.mean(a >= 3), n_cases, m, data)
    record('4.7', 'max infected, upper definition', a.max(), n_cases, m, data)

    ev = pd.read_csv(EVENTS)
    links = ev[ev.dataset == WAVE].groupby('source_id').size()
    b = np.zeros(n_cases)
    b[:len(links)] = links.values
    m = 'recorded infector -> infectee links per case, 0 otherwise (E80 lower definition)'
    record('1.1', 'offspring k, lower definition, MLE', nb_k_mle(b), n_cases, m, f'{EVENTS.name}, {WAVE}')
    record('4.7', 'R, lower definition', b.mean(), n_cases, m, f'{EVENTS.name}, {WAVE}')
    record('4.7', 'share infecting >= 3, lower definition', np.mean(b >= 3), n_cases, m, f'{EVENTS.name}, {WAVE}')
    record('4.7', 'max infected, lower definition', b.max(), n_cases, m, f'{EVENTS.name}, {WAVE}')

    # Community: municipality layer, uninfected count known, total > 0 (E77).
    c = t[(t.layer == 'municipality') & t.uninfected.notna()]
    x = c.total[c.total > 0].to_numpy(float)
    m = 'municipality-layer total contacts per index case, rows with a known uninfected count and total > 0 (E77)'
    record('1.18', 'community median', np.median(x), len(x), m + '; interval = bootstrap 95%', data, boot(x, np.median))
    p90 = lambda v: np.percentile(v, 90)
    record('1.19', 'community p90', p90(x), len(x), m + '; numpy linear percentile', data, boot(x, p90))
    ratio = lambda v: np.percentile(v, 90) / np.median(v)
    record('1.20', 'community p90 / median', ratio(x), len(x), m + '; interval = bootstrap 95% (E77)', data, boot(x, ratio))
    record('1.21', 'community zero share among rows with a known uninfected count', np.mean(c.total == 0), len(c),
           'municipality-layer rows with total = 0 / rows with a known uninfected count', data)
    xmin, alpha, ks, ntail = clauset_discrete(x)
    record('3.12', 'community tail power-law exponent (discrete)', alpha, ntail,
           f'Clauset 2009 discrete MLE above x_min, KS = {ks:.3f}', data)
    record('3.13', 'community tail x_min', xmin, len(x), 'x_min minimising the KS distance (Clauset 2009)', data)

    w = pd.read_csv(COWORKERS)
    w = w[w.dataset == WAVE]
    record('3.9', 'coworker close contacts per index case, median', w.total_coworker_close_contacts.median(), len(w),
           'infected + uninfected coworkers per index case with any coworker record', f'{COWORKERS.name}, {WAVE}')


def main():
    severity_and_timing()
    tracing()
    out = pd.DataFrame(ROWS)
    out.to_csv(OUT, index=False)
    with pd.option_context('display.max_colwidth', 70, 'display.width', 250):
        print(out[['item', 'quantity', 'value', 'n', 'ci_low', 'ci_high']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
