"""Is 'contacts starting before symptom onset' (B26: CovSyn 59.8% against Cheng 27.5%) comparing
the same quantity?

Cheng et al. 2020 (JAMA Intern Med 180:1156, Methods p.1157): "The period of investigation started
at the date at symptom onset (and could be extended up to 4 days before symptom onset when
epidemiologically indicated) and ended at the date at COVID-19 confirmation." Their Table 1 shows a
minimum time from onset to first exposure of -4 days in every setting. verify_phaseD.py counts every
CovSyn candidate contact from the day of infection, i.e. up to the whole incubation period before
onset, which the tracing never looked at.

This recomputes the CovSyn share with Cheng's investigation window applied, on the same index cases
verify_phaseD.py uses (symptomatic index case of every simulation):
  * a contact counts only if it was met on at least one day in [onset - W, end of window];
  * its first exposure is its first contact day inside that window;
for W = 4 (Cheng's maximum extension), and for comparison W = 0, 1, 2, 3 and 'no limit'.

Usage: python probe_preonset_window.py [spread_dir]
"""
import glob
import sys
from pathlib import Path

import numpy as np

SPREAD = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight')
LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
MATRIX = {L: ('school_class_contacts_matrix' if L == 'school' else f'{L}_contacts_matrix') for L in LAYERS}
WINDOWS = [None, 4, 3, 2, 1, 0]      # days before onset the tracing looks back; None = from infection


def main():
    counts = {w: {'before': 0, 'total': 0, 'by_layer': {L: [0, 0] for L in LAYERS}} for w in WINDOWS}
    n_index = 0
    for f in sorted(glob.glob(str(SPREAD / 'contact_data_*.npy'))):
        k = f.split('_')[-1].split('.')[0]
        contact = np.load(f, allow_pickle=True)
        course = np.load(SPREAD / f'course_of_disease_data_{k}.npy', allow_pickle=True)
        if not len(contact) or not len(course):
            continue
        c, onset = contact[0], course[0]['incubation_period']
        if onset is None or np.isnan(onset):
            continue                      # Cheng's timing is relative to onset: symptomatic only
        n_index += 1
        onset = int(onset)
        for L in LAYERS:
            m = np.asarray(c[MATRIX[L]], dtype=float)
            if m.size == 0:
                continue
            m = m > 0
            for w in WINDOWS:
                start = 0 if w is None else max(0, onset - w)
                inside = m[:, start:]
                met = inside.any(axis=1)
                first = np.argmax(inside, axis=1) + start - onset
                first = first[met]
                counts[w]['before'] += int((first < 0).sum())
                counts[w]['total'] += len(first)
                counts[w]['by_layer'][L][0] += int((first < 0).sum())
                counts[w]['by_layer'][L][1] += len(first)

    print(f'{n_index} symptomatic index cases from {SPREAD}')
    print('Cheng 2020: 735 of 2,670 contacts with a known date = 27.5% first exposed before onset\n')
    print('%-24s %9s %9s %8s   %s' % ('look-back window', 'contacts', 'before', 'share',
                                      '  '.join(f'{L[:6]:>7s}' for L in LAYERS)))
    for w in WINDOWS:
        d = counts[w]
        label = 'from infection (current)' if w is None else f'onset - {w} days'
        per = '  '.join(f'{100 * b / max(t, 1):6.1f}%' for b, t in d['by_layer'].values())
        print('%-24s %9d %9d %7.1f%%   %s' % (label, d['total'], d['before'],
                                             100 * d['before'] / max(d['total'], 1), per))
    print('\nCheng 2020 by setting (Table 1, <0 / known): household 100/151 = 66.2%, '
          'family 10/68 = 14.7%, health care 236/697 = 33.9%, others 389/1754 = 22.2%')


if __name__ == '__main__':
    main()
