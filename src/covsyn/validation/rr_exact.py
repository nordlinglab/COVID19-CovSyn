"""Age-stratified secondary attack rate measured with the EXACT denominator.

Every contact's age is now recorded in {layer}_contact_ages, aligned row-for-row with
{layer}_effective_contacts (1 = infected), so the age-specific attack rate no longer has to
be estimated from the population age distribution -- which stopped being valid once the
layers were given their own age compositions.
"""
import glob
import sys
import numpy as np

SYN = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_cheng2020_NEW'
LAYERS = ['household', 'school', 'workplace', 'health_care', 'municipality']
BINS = [(0, 19), (20, 39), (40, 59), (60, 100)]
LAB = ['0-19', '20-39', '40-59', '60+']

contact = []
index_contact = []
for f in sorted(glob.glob(SYN + '/contact_data_*.npy')):
    records = list(np.load(f, allow_pickle=True))
    contact += records
    if records:
        index_contact.append(records[0])
print('source cases: %d   (%s)\n' % (len(contact), SYN))

n_by = {L: np.zeros(4) for L in LAYERS}
i_by = {L: np.zeros(4) for L in LAYERS}
for rec in contact:
    for L in LAYERS:
        ages = rec.get(L + '_contact_ages')
        eff = rec.get(L + '_effective_contacts')
        if ages is None or eff is None or len(ages) == 0:
            continue
        a = np.asarray(ages, float)
        e = np.asarray(eff, float)
        if a.shape != e.shape:
            print('WARNING: %s length mismatch %s vs %s' % (L, a.shape, e.shape))
            continue
        for gi, (lo, hi) in enumerate(BINS):
            m = (a >= lo) & (a <= hi)
            n_by[L][gi] += m.sum()
            i_by[L][gi] += (e[m] == 1).sum()

print('%-14s %-30s %-30s %s' % ('layer', 'contacts by age group', 'attack rate % by age group', 'RR (ref 20-39)'))
print('-' * 108)
tot_n = np.zeros(4)
tot_i = np.zeros(4)
for L in LAYERS:
    n, i = n_by[L], i_by[L]
    tot_n += n
    tot_i += i
    if n.sum() == 0:
        continue
    ar = np.divide(i, n, out=np.zeros(4), where=n > 0) * 100
    rr = ar / ar[1] if ar[1] > 0 else np.zeros(4)
    print('%-14s %-30s %-30s %s'
          % (L, np.int32(n).tolist(), np.round(ar, 2).tolist(), np.round(rr, 2).tolist()))
ar = np.divide(tot_i, tot_n, out=np.zeros(4), where=tot_n > 0) * 100
rr = ar / ar[1]
print('-' * 108)
print('%-14s %-30s %-30s %s'
      % ('ALL contacts', np.int32(tot_n).tolist(), np.round(ar, 2).tolist(), np.round(rr, 2).tolist()))

# E62, 2026-09-27: this SAME acceptance item was reported here on every case and in
# verify_phaseD.py on the index case of each simulation only, and the two populations do not
# agree -- on run 5, [0.80, 1.0, 1.61, 1.15] here against [0.75, 1.0, 1.63, 1.12] there. The
# 40-59 band straddles its 1.63 threshold between them, so "B14 40-59 fixed" on run 5 was an
# artefact of which population was asked, not a result. Both are printed now. The INDEX row is
# the one to quote against the checklist: it matches Cheng's design, which traced the contacts
# of confirmed index cases.
n_idx = np.zeros(4)
i_idx = np.zeros(4)
for rec in index_contact:
    for L in LAYERS:
        ages, eff = rec.get(L + '_contact_ages'), rec.get(L + '_effective_contacts')
        if ages is None or eff is None or len(ages) == 0:
            continue
        a = np.asarray(ages, float)
        e = np.asarray(eff, float)
        if a.shape != e.shape:
            continue
        for gi, (blo, bhi) in enumerate(BINS):
            m = (a >= blo) & (a <= bhi)
            n_idx[gi] += m.sum()
            i_idx[gi] += (e[m] == 1).sum()
ar_idx = np.divide(i_idx, n_idx, out=np.zeros(4), where=n_idx > 0) * 100
rr_idx = ar_idx / ar_idx[1] if ar_idx[1] > 0 else np.zeros(4)
print('%-14s %-30s %-30s %s'
      % ('INDEX cases', np.int32(n_idx).tolist(), np.round(ar_idx, 2).tolist(),
         np.round(rr_idx, 2).tolist()))
print('-' * 108)
print('ALL contacts = every case (%d).  INDEX cases = the first case of each simulation (%d),'
      % (len(contact), len(index_contact)))
print('which is the population verify_phaseD.py checks B14 on, so quote that row.')

# E70: this used to print [0.50, 1.00, 1.83, 1.32] as "the input", which is not what is locked
# into the parameter vector -- that is [0.39, 1, 1.90, 1.44]. Read it from disk so the two cannot
# drift apart again, the same lesson as E63.
_locked = np.load('variable/course_parameters.npy')[26:30]
print('\ninput age risk ratios (locked, from variable/course_parameters.npy): %s'
      % np.round(_locked, 3).tolist())
print('reference (Cheng all-infection): [0.52, 1.00, 1.83, 1.32]')
print('  NOTE: the 0-19 figure of 0.52 rests on ONE infection among 281 contacts and is NOT used')
print('  as an acceptance band any more; B14 uses [0.34, 0.77] from Zhang 2020 / Viner 2021 /')
print('  Uthman 2024 / Madewell 2020 instead (todolist 1.13, decision N5, finding E70).')
print('target (Cheng clinical)     : [0.00, 1.00, 2.19, 1.75]   <- not reproducible, see decisions doc')
gap = rr - np.array([0.52, 1.00, 1.83, 1.32])
print('gap vs all-infection target : %s' % np.round(gap, 2).tolist())

# Monte-Carlo spread: recompute per simulation file to get a confidence interval
per_sim = []
for f in sorted(glob.glob(SYN + '/contact_data_*.npy')):
    n = np.zeros(4)
    i = np.zeros(4)
    for rec in np.load(f, allow_pickle=True):
        for L in LAYERS:
            ages, eff = rec.get(L + '_contact_ages'), rec.get(L + '_effective_contacts')
            if ages is None or eff is None or len(ages) == 0:
                continue
            a = np.asarray(ages, float)
            e = np.asarray(eff, float)
            if a.shape != e.shape:
                continue
            for gi, (lo, hi) in enumerate(BINS):
                m = (a >= lo) & (a <= hi)
                n[gi] += m.sum()
                i[gi] += (e[m] == 1).sum()
    a_ = np.divide(i, n, out=np.zeros(4), where=n > 0)
    if a_[1] > 0:
        per_sim.append(a_ / a_[1])
per_sim = np.array(per_sim)
if len(per_sim):
    lo = np.percentile(per_sim, 2.5, axis=0)
    hi = np.percentile(per_sim, 97.5, axis=0)
    print('\nRR 95%% CI over %d simulations:' % len(per_sim))
    for gi, g in enumerate(LAB):
        print('  %-6s %5.2f  (%.2f - %.2f)' % (g, rr[gi], lo[gi], hi[gi]))
