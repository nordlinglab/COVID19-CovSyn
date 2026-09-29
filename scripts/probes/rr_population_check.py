"""Age risk ratio measured on two populations, plus the candidate-contact counts the
sar_anchors.py anchors are derived from.

verify_phaseD.py measures B14 on the INDEX CASE of each simulation only
(index_contact = [r['contact'][0] for r in runs]); rr_exact.py measures it on EVERY case.
Same numerator definition, same bins -- different population, so the checklist and the
report print two different numbers for one acceptance item. This prints both.
"""
import glob
import sys
from pathlib import Path

import numpy as np

SYN = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight')
LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
BINS = [(0, 20), (20, 40), (40, 60), (60, 200)]
LAB = ['0-19', '20-39', '40-59', '60+']


def rr(records):
    n = np.zeros(4)
    i = np.zeros(4)
    per_layer_n = {L: 0.0 for L in LAYERS}
    for rec in records:
        for L in LAYERS:
            ages = rec.get(L + '_contact_ages')
            eff = rec.get(L + '_effective_contacts')
            if ages is None or eff is None or len(ages) == 0:
                continue
            a = np.asarray(ages, float)
            e = np.asarray(eff, float)
            if a.shape != e.shape:
                continue
            per_layer_n[L] += a.size
            for gi, (lo, hi) in enumerate(BINS):
                m = (a >= lo) & (a < hi)
                n[gi] += m.sum()
                i[gi] += (e[m] == 1).sum()
    ar = np.divide(i, n, out=np.zeros(4), where=n > 0)
    return (ar / ar[1] if ar[1] > 0 else np.zeros(4)), n, i, per_layer_n


files = sorted(glob.glob(str(SYN / 'contact_data_*.npy')),
               key=lambda p: int(Path(p).stem.split('_')[-1]))
all_cases = []
index_cases = []
for f in files:
    recs = list(np.load(f, allow_pickle=True))
    if not recs:
        continue
    all_cases += recs
    index_cases.append(recs[0])

print('%s\n  simulations %d   index cases %d   all cases %d\n'
      % (SYN, len(files), len(index_cases), len(all_cases)))

target = np.array([0.52, 1.00, 1.83, 1.32])
band = [(0.42, 0.62), (1.0, 1.0), (1.63, 2.03), (1.12, 1.52)]
for label, records in [('index cases only  (verify_phaseD B14)', index_cases),
                       ('every case         (rr_exact.py)', all_cases)]:
    r, n, i, per_layer_n = rr(records)
    print('%s' % label)
    print('  contacts by band  %s' % np.int64(n).tolist())
    print('  infections        %s' % np.int64(i).tolist())
    print('  RR                %s' % np.round(r, 3).tolist())
    verdict = ['ok' if lo <= v <= hi else 'FAIL' for v, (lo, hi) in zip(r, band)]
    print('  vs target %s -> %s\n' % (np.round(target, 2).tolist(), verdict))

# ------------------------------------------------ anchor circularity (sar_anchors.py)
_, _, _, per_layer_n = rr(index_cases)
print('candidate contacts per INDEX case (the sar_anchors.py anchor denominator)')
try:
    sys.path.insert(0, '.')
    from covsyn.calibration.sar_anchors import COVSYN_CANDIDATE_CONTACTS, CANDIDATE_CONTACTS_MEASURED_ON
    print('  in sar_anchors.py: %s' % CANDIDATE_CONTACTS_MEASURED_ON)
    for L in LAYERS:
        now = per_layer_n[L] / max(len(index_cases), 1)
        was = COVSYN_CANDIDATE_CONTACTS[L]
        print('  %-13s this run %6.3f   in sar_anchors %6.3f   change %+6.1f%%'
              % (L, now, was, 100 * (now - was) / was))
except Exception as exc:                                     # noqa: BLE001
    print('  could not import sar_anchors: %s' % exc)
    for L in LAYERS:
        print('  %-13s this run %6.3f' % (L, per_layer_n[L] / max(len(index_cases), 1)))
