"""Measure the pooled age risk ratio on enough index cases for the answer to mean something.

E68, 2026-09-27. B14 was being checked on the 1000-simulation spread output, which on run 6
held 226 infections spread over four age bands -- health care 60+ rested on a SINGLE infection,
and rr_exact.py's own interval for the 0-19 ratio, (0.00, 1.18), was wider than the whole
acceptance band [0.42, 0.62]. The direction of the failure may well be real, but that test
cannot resolve the +/- 0.10 it is judged on.

The fix is sample size, not a new method: the age risk ratio is a property of the parameter
vector, so it can be measured on as many index cases as we care to simulate. One simulation is
about 0.8 ms, so 20000 index cases cost roughly 20 s of wall time on 32 workers and give ~20x
the infections. Nothing here is external evidence -- it is the model being asked whether the
age gradient locked into its input comes out the other end.

Writes validation_reference/age_rr.json, which verify_phaseD.py reads for B14.

Usage: python -m covsyn.validation.measure_age_rr [firefly_result_dir] [n_index_cases] [--workers N]
"""
import argparse
import concurrent.futures
import glob
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from covsyn.calibration import fast_cost
from covsyn.model.data_synthesis_main import run_covid

LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
# E70: the layers Cheng 2020 actually traced. His exposure settings are household, health care
# and "other" -- there is no school and no workplace category -- so his pooled age risk ratio is
# a ratio over THOSE contacts. Pooling CovSyn's five layers against it compares different
# contact mixes, and the mix dominates the answer: CovSyn's school layer carries an attack rate
# of 2.29% against 0.92% in the household and 0.22-0.23% in health care and the community, and
# it exists only in the 0-19 and 20-39 bands, so on run 6 it produced 81% of every 0-19
# infection. That alone lifts the pooled 0-19 ratio from 0.36 to 0.83 with no change at all to
# the age model. The acceptance test below therefore uses these three layers, and the five-layer
# figure is reported beside it.
CHENG_COMPARABLE = ['household', 'health_care', 'municipality']
BINS = [(0, 20), (20, 40), (40, 60), (60, 200)]
LABELS = ['0-19', '20-39', '40-59', '60+']
TARGET = [0.52, 1.00, 1.83, 1.32]
# E70 / N5, 2026-09-28. The 0-19 band is NOT Cheng 2020's 0.52 +/- 0.10 any more. Cheng's 0-19
# estimate is ONE infection among 281 contacts, and todolist923 1.13 says in terms not to invent a
# reference value where the study has none, to state explicitly that it cannot validate the
# school-age group, and to find another source; 1.11 adds that the criterion is a supported RANGE,
# not a literature mean. covsyn_age_weight_litreview.md already collected the alternatives, four
# of which measure the same quantity as Cheng's ratio -- the relative risk of infection among
# traced or household contacts, children against adults:
#
#   Zhang 2020, Science          <15  0.34 (0.24-0.49)   Hunan contact tracing + RT-PCR
#   Davies 2020, Nat. Med.       0-9  0.40 (0.25-0.57)   age-structured model fit (not a contact
#                                                        ratio, so used for context only)
#   Viner 2021, JAMA Pediatr.    <20  0.56 (0.37-0.85)   meta-analysis, 32 studies
#   Uthman 2024, PLoS One        0-19 0.58 (0.44-0.77)   household meta-analysis, WILD TYPE
#   Madewell 2020, JAMA Netw Op  child:adult 0.59        household meta-analysis, 54 studies
#
# The litreview's own synthesis is ~0.50 with the bound "tightened to [0.34, 0.77] using Zhang
# (lower) and Uthman wild-type (upper)", which is what is used here.
#
# 40-59 and 60+ keep Cheng's bands. The litreview's numbers for those bands are SUSCEPTIBILITY
# weights, which differ from a measured secondary attack rate ratio by the age-specific clinical
# fraction (its own section C does that reconciliation), and both bands currently pass against
# Cheng, so there is nothing to gain by swapping the reference.
LITERATURE_0_19 = (0.34, 0.77)
BANDS = [LITERATURE_0_19, (1.0, 1.0), (1.63, 2.03), (1.12, 1.52)]
BAND_SOURCE = [
    'Zhang 2020 0.34 / Viner 2021 0.56 / Uthman 2024 wild-type 0.58 / Madewell 2020 0.59; '
    'range [0.34, 0.77]. NOT Cheng 2020, whose 0-19 is 1 infection in 281 contacts and cannot '
    'validate this band (todolist 1.13, decision N5)',
    'reference band, fixed to 1 by construction',
    'Cheng 2020 all-infection 1.83 +/- 0.10',
    'Cheng 2020 all-infection 1.32 +/- 0.10']
# What apply_phaseD_parameters.py locks into the parameter vector, for the input-reproduction
# check. Read from disk rather than copied, because rr_exact.py and the scorecard both carried a
# hardcoded "input" of [0.50, 1.00, 1.83, 1.32] that had drifted from the real [0.39, 1, 1.9, 1.44].
LOCKED_INPUT_INDEX = slice(26, 30)


def count_batch(seeds, P):
    """Contacts and infections by age band for a batch of index cases."""
    demographic_parameters = fast_cost._WORKER_DEMOGRAPHIC_PARAMETERS
    import copy
    contacts = np.zeros((5, 4))
    infections = np.zeros((5, 4))
    for seed in seeds:
        _demo, _social, _course, contact_list = run_covid(
            seed, P, copy.deepcopy(demographic_parameters), save_file=False)
        if not contact_list:
            continue
        record = contact_list[0]                      # the index case, matching verify_phaseD
        for li, layer in enumerate(LAYERS):
            ages = record.get(layer + '_contact_ages')
            effective = record.get(layer + '_effective_contacts')
            if ages is None or effective is None or len(ages) == 0:
                continue
            a = np.asarray(ages, dtype=float)
            e = np.asarray(effective, dtype=float)
            if a.shape != e.shape:
                continue
            for bi, (lo, hi) in enumerate(BINS):
                mask = (a >= lo) & (a < hi)
                contacts[li, bi] += mask.sum()
                infections[li, bi] += (e[mask] == 1).sum()
    return contacts, infections


def wilson(successes, trials, z=1.959963984540054):
    """Wilson score interval, which behaves sensibly when the count is small."""
    if trials == 0:
        return float('nan'), float('nan')
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default=None)
    parser.add_argument('n', nargs='?', type=int, default=20000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--out', default='validation_reference/age_rr.json')
    args = parser.parse_args()

    directory = args.directory
    if directory is None:
        directory = [d for d in sorted(glob.glob('Firefly_result_pop_size_100_*'))
                     if os.path.exists(d + '/firefly_best.txt')][0]
    result = np.loadtxt(Path(directory) / 'firefly_best.txt')
    P = result[int(np.argmin(result[:, -1])), 1:-1]

    with open('./variable/demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    columns = np.load('./variable/Taiwan_data_matrix.npy').shape[1]

    print('age risk ratio on %d index cases, parameters from %s' % (args.n, directory))
    start = time.perf_counter()
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=fast_cost.init_worker, initargs=(demo, columns))
    per_task = max(args.n // (args.workers * 4), 1)
    batches = [list(range(i, min(i + per_task, args.n))) for i in range(0, args.n, per_task)]
    contacts = np.zeros((5, 4))
    infections = np.zeros((5, 4))
    for future in [executor.submit(count_batch, batch, P) for batch in batches]:
        c, i = future.result()
        contacts += c
        infections += i
    executor.shutdown()
    print('  %.1f s\n' % (time.perf_counter() - start))

    locked = np.load('./variable/course_parameters.npy')[LOCKED_INPUT_INDEX]
    print('locked age risk ratio input: %s  (read from variable/course_parameters.npy)\n'
          % np.round(locked, 3).tolist())
    print('%-14s %-28s %-26s %-26s %s'
          % ('layer', 'contacts by band', 'infections by band', 'RR', 'max |RR - input|'))
    print('-' * 118)
    layer_gaps = {}
    for li, layer in enumerate(LAYERS):
        if contacts[li].sum() == 0:
            continue
        rate = np.divide(infections[li], contacts[li], out=np.zeros(4), where=contacts[li] > 0)
        rr = rate / rate[1] if rate[1] > 0 else np.zeros(4)
        # Input reproduction (todolist 1.10 / N3): within a layer the measured ratio should
        # come back as the locked input. This is an implementation check, not validation --
        # the reference IS the model's own setting.
        comparable = [i for i in range(4) if contacts[li][i] > 0 and rr[i] > 0]
        gap = max((abs(rr[i] - locked[i]) for i in comparable), default=float('nan'))
        layer_gaps[layer] = gap
        print('%-14s %-28s %-26s %-26s %.3f'
              % (layer, np.int64(contacts[li]).tolist(), np.int64(infections[li]).tolist(),
                 np.round(rr, 2).tolist(), gap))

    cheng_rows = [LAYERS.index(layer) for layer in CHENG_COMPARABLE]
    cheng_contacts = contacts[cheng_rows].sum(axis=0)
    cheng_infections = infections[cheng_rows].sum(axis=0)

    total_contacts = contacts.sum(axis=0)
    total_infections = infections.sum(axis=0)
    rate = np.divide(total_infections, total_contacts, out=np.zeros(4), where=total_contacts > 0)
    rr = rate / rate[1] if rate[1] > 0 else np.zeros(4)
    all_layer_rr = rr.copy()
    print('-' * 100)
    print('%-14s %-28s %-26s %s'
          % ('POOLED', np.int64(total_contacts).tolist(), np.int64(total_infections).tolist(),
             np.round(rr, 3).tolist()))
    cheng_rate = np.divide(cheng_infections, cheng_contacts, out=np.zeros(4),
                           where=cheng_contacts > 0)
    cheng_rr = cheng_rate / cheng_rate[1] if cheng_rate[1] > 0 else np.zeros(4)
    print('%-14s %-28s %-26s %s'
          % ('CHENG LAYERS', np.int64(cheng_contacts).tolist(),
             np.int64(cheng_infections).tolist(), np.round(cheng_rr, 3).tolist()))
    print('  (household + health care + community: the settings Cheng 2020 traced. POOLED above')
    print('   adds school and workplace, which he has no category for at all -- finding E70.)')
    print('\ntotal infections: %d  (run 6 measured this on 226)' % int(total_infections.sum()))

    # The ratio's interval, from each band's own Wilson interval for the attack rate. The
    # reference band's uncertainty is carried through both ends, which is conservative.
    print('\n%-8s %10s %10s %-22s %-14s %s'
          % ('band', 'rate %', 'RR', '95% CI of the RR', 'band', 'verdict'))
    intervals = {}
    # The acceptance test runs on the Cheng-comparable layers (E70); the five-layer ratio is a
    # different quantity and is reported, not judged.
    rate, rr = cheng_rate, cheng_rr
    used_contacts, used_infections = cheng_contacts, cheng_infections
    ref_lo, ref_hi = wilson(used_infections[1], used_contacts[1])
    for bi, label in enumerate(LABELS):
        lo, hi = wilson(used_infections[bi], used_contacts[bi])
        rr_lo = lo / ref_hi if ref_hi > 0 else float('nan')
        rr_hi = hi / ref_lo if ref_lo > 0 else float('nan')
        target_lo, target_hi = BANDS[bi]
        verdict = 'ok' if target_lo <= rr[bi] <= target_hi else 'FAIL'
        resolvable = 'resolvable' if (rr_hi - rr_lo) < (target_hi - target_lo) * 3 or bi == 1 \
            else 'TOO NOISY'
        intervals[label] = {'rr': float(rr[bi]), 'ci': [float(rr_lo), float(rr_hi)],
                            'contacts': int(used_contacts[bi]),
                            'infections': int(used_infections[bi]),
                            'target': [target_lo, target_hi], 'ok': verdict == 'ok',
                            'precision': resolvable}
        intervals[label]['band_source'] = BAND_SOURCE[bi]
        print('%-8s %10.3f %10.3f [%8.3f, %8.3f] %-14s %s  %s'
              % (label, 100 * rate[bi], rr[bi], rr_lo, rr_hi,
                 '[%.2f, %.2f]' % (target_lo, target_hi), verdict, resolvable))
        print('         reference: %s' % BAND_SOURCE[bi])

    print('\ninput age risk ratios (locked) : %s' % np.round(locked, 3).tolist())
    print('Cheng all-infection reference  : %s  (0-19 NOT used, see above)' % TARGET)
    print('\ninput reproduction (N3): the largest |measured RR - locked input| per layer')
    for layer, gap in layer_gaps.items():
        print('  %-14s %.3f' % (layer, gap))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {'n_index_cases': args.n, 'parameters_from': str(directory),
         'total_infections': int(total_infections.sum()),
         'cheng_comparable_layers': CHENG_COMPARABLE,
         'cheng_comparable_infections': int(cheng_infections.sum()),
         'cheng_comparable_rr': [float(v) for v in cheng_rr],
         'locked_input': [float(v) for v in locked],
         'layer_input_reproduction_gap': {k: float(v) for k, v in layer_gaps.items()},
         'all_layer_rr': [float(v) for v in all_layer_rr],
         'bands': intervals}, indent=2))
    print('\nwritten %s' % out)


if __name__ == '__main__':
    main()
