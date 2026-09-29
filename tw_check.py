"""Population-level check: cumulative cases/deaths produced by the Taiwan first-outbreak
mode, compared with the observed Taiwan totals and with the previous parameter set.

Phase D (decisions B36, finding E29): the run directory is now an argument, and every run is
summarised twice -- over all simulations, and over only those with at least one onward case,
which is the filter taiwan_first_outbreak.ipynb applies (min_case_num = 29) and which lifts
the mean. Both numbers have to be on the table when this is compared with the observed 55.

Usage: python tw_check.py [run_dir ...]
"""
import glob
import sys

import numpy as np

NA = (lambda x: x is None or (isinstance(x, float) and (np.isnan(x) or x <= -1 or x >= 1e9)))
SEEDS = 28


def summarise(path, label):
    files = sorted(glob.glob(path + '/course_of_disease_data_*.npy'),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    if not files:
        print('%-34s no data' % label)
        return
    cases, confirmed, deaths = [], [], []
    for f in files:
        items = list(np.load(f, allow_pickle=True))
        cases.append(len(items))
        confirmed.append(sum(1 for c in items if not NA(c.get('positive_test_date'))))
        deaths.append(sum(1 for c in items if not NA(c.get('date_of_death'))))
    cases, confirmed, deaths = np.array(cases), np.array(confirmed), np.array(deaths)
    spread = cases > SEEDS
    for keep, note in ((np.ones(len(cases), bool), 'all simulations'),
                       (spread, 'at least one onward case')):
        if not keep.any():
            continue
        print('%-34s %-24s sims=%4d  cases %6.1f (%.0f-%.0f)   confirmed %6.1f   deaths %5.2f'
              % (label, note, int(keep.sum()), cases[keep].mean(),
                 np.percentile(cases[keep], 2.5), np.percentile(cases[keep], 97.5),
                 confirmed[keep].mean(), deaths[keep].mean()))
        label = ''


paths = sys.argv[1:] or ['synthetic_data_results_taiwan_first_outbreak']
print('=== Taiwan first outbreak (28 local index cases seeded on their reported confirmation '
      'dates, 365 days; finding E29) ===')
for path in paths:
    summarise(path, path)
for path in sorted(glob.glob('synthetic_data_results_taiwan_first_outbreak_*')):
    if path not in paths:
        summarise(path, 'previous: ' + path)
print('\nObserved Taiwan first wave: 55 cumulative local confirmed cases, 3 local deaths.')
print('The 28 seeds are included in the simulated counts. Reporting is complete in the model')
print('(B18), so the simulated infection count should sit ABOVE 55, not on it; and under the')
print('B33 severity cascade the expected death count is about 0.6, not 3 (B36).')
