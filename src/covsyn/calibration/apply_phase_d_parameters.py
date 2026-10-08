"""Rewrite the firefly search bounds and the literature seed vector for the Phase D rerun.

Phase D implements decisions B17-B36 of covsyn_decisions.md. Several parameters changed
MEANING, so their bounds had to change with them; others only needed room the old bounds did
not give. Everything this script writes is listed below with the decision it comes from.

  P[28]        municipality contact gate -> MEAN NUMBER OF COMMUNITY CONTACTS per case
                 (B27). The community layer is no longer Binomial(city population, p), so
                 this is a count, not a probability, and the city population no longer
                 enters the contact count at all (finding E4).
  P[30]        community daily contact probability: ceiling raised, the old 0.1 could not
                 reach the 1.49 contacts/day target (B25).
  P[2], P[16]  household / workplace daily contact probability: ceiling raised for the same
                 reason (household target 1.86/day, workplace 1.49/day, B25).
  P[21], P[23] health care contact gate and daily probability: floor lowered, the layer was
                 producing 2.5 contacts/day for a person who is not yet ill (B25, E26).
  P[35]        overdispersion rate -> GAMMA SHAPE k of the case-level infectiousness
                 multiplier (B17).
  P[36]        overdispersion weight -> CAP on that multiplier (B17).
  P[41:43]     incubation Gamma -> PRE-ONSET WINDOW Gamma; incubation is now latent + window
                 (B22 (3)).
  P[43:46]     onset -> confirmation retuned to Taiwan's own median of 1 day (B26, E27).
  P[46:49]     asymptomatic -> case closure: the old floor was 33.3 days, which made the
                 target unreachable by construction (B22 (2), E23).
  P[52:58]     symptomatic / ICU -> case closure: same, floors lowered.
  P[63:67]     age risk ratios locked to the B14 trial value (B2: lb == ub, set by hand).
  P[195:198]   the severity cascade recomputed from the Taiwan tracing file (B33):
                 asymptomatic 23.7%, symptomatic->ICU 12.7%, ICU->death 12.5%.

Usage: python -m covsyn.calibration.apply_phase_d_parameters [variable_dir]
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

from covsyn.model.data_synthesize import COMMUNITY_EVENT_FIRST_INDEX

from covsyn.calibration.sar_anchors import ATTACK_RATE_SLICE, CONTACT_DAYS_MEASURED_ON, LAYERS, \
    LAYER_CUMULATIVE_SAR, MEAN_CONTACT_DAYS, attack_rate_block

VAR = Path(sys.argv[1] if len(sys.argv) > 1 else 'variable')
BACKUP = VAR.parent / 'ARCHIVE_20260921_pre_phaseD' / 'variable_pre_phaseD'
BACKUP.mkdir(parents=True, exist_ok=True)


def backup_once(name):
    """Copy a file into the archive only the first time, so re-running this script does not
    overwrite the pre-Phase-D originals with the already-patched versions (decision A4)."""
    if not (BACKUP / name).exists():
        shutil.copy(VAR / name, BACKUP / name)

# ----------------------------------------------------------------- contact layer bounds
# order per layer: [contact gate, contact-again-tomorrow, healthy daily p, symptomatic
#                   daily p, logistic steepness, symptom phase, width]
CONTACT_BOUNDS = {
    # Household: every member is a candidate (gate ~1) and the target is 1.86 contacts per
    # day, which with 2.8 other members needs a daily probability around 0.66; the old
    # ceiling of 0.5 could not reach it. Household keeps a free symptomatic probability
    # (B19 exempts it: a sick person really does stay home with the family).
    'household':    ([0.95, 0.4, 0.20, 0.01, 1, 0, 0], [1, 1, 0.95, 0.90, 20, 10, 10]),
    # School: unchanged bounds, the layer already overshoots its 1.16/day target.
    'school':       ([0.10, 0.5, 0.01, 0.001, 1, 0, 0], [1, 1, 0.50, 0.01, 10, 10, 10]),
    # Workplace: 1.49 contacts/day over the whole population means about 3/day for an
    # employed case, against a work group of about 10 people (B31).
    'workplace':    ([0.10, 0.5, 0.01, 0.001, 1, 0, 0], [1, 1, 0.70, 0.05, 10, 10, 10]),
    # Health care: a person who is not yet ill barely meets medical staff, so both the gate
    # and the healthy-day probability get a lower floor. The symptomatic probability stays
    # high -- going to a clinic when ill is the point of this layer.
    # 2026-09-27 (E59): the STEEPNESS floor (layer index 4, i.e. P[25]) comes down from 1 to
    # 0.2. Run 5 pegged it at 1, meaning the search wanted a flatter medical contact profile
    # than the bounds allowed, and the profile it did produce had Cheng's timing reversed --
    # 9% of medical contacts in his two early bins where he has 55%, and 50.7% in the late
    # tail where he has 37%. A flatter curve is how the early bins get filled, so the floor
    # must not be the thing blocking it. Charging both ends of the distribution (the new
    # medical_early_share target) is the other half of that fix.
    'health_care':  ([0.0005, 0.5, 0.0002, 0.6, 0.2, 0, 0], [0.008, 1, 0.006, 1, 10, 10, 10]),
    # Municipality: index 0 is now a COUNT (mean community contacts per case, B27). The
    # Taiwan tracing records give a median of 7.5 and a mean of 87 friend/other contacts per
    # index case, so the search runs from 1 to 40 and the heavy tail comes from the
    # case-level multiplier rather than from the mean.
    'municipality': ([1.0, 0.5, 0.01, 0.001, 1, 0, 0], [40.0, 1, 0.50, 0.50, 10, 10, 10]),
}
# Case-level infectiousness multiplier nu ~ Gamma(k, 1/k), capped (B17). k below 1 is where
# superspreading lives: Taiwan's own offspring dispersion is k = 0.29. The floor was 0.05 in
# the first Phase D run and the optimizer sat on it, producing an offspring distribution with
# k = 0.024 and a single case infecting 196 people against a Taiwan maximum of 8 (E35). The
# floor is now 0.15: still well inside the superspreading regime, but it cannot degenerate
# into "almost every case infects nobody and a handful infect hundreds". The ceiling comes
# down too, because a shape above 1.5 is effectively Poisson and worth no search effort.
# 2026-09-26: the ceiling drops again, 1.5 -> 0.8. The fourth run chose k = 1.105, which is
# already effectively Poisson -- the offspring dispersion came out at 0.428 against Taiwan's
# 0.29 and a target of 0.10-0.35, i.e. the optimizer traded superspreading away once B2 gave
# every case five more days of contact. A Gamma shape of 0.8 still allows a heavier tail than
# Poisson while leaving the whole superspreading range 0.15-0.8 searchable.
OVERDISPERSION_BOUNDS = ([0.15, 3.0], [0.8, 60.0])

# P[198], appended at the end of the vector (E35): dispersion of the community contact COUNT,
# drawn independently of the infectiousness multiplier. The Taiwan tracing records give
# friend + other contacts per index case with a median of 7.5 and a maximum of 850, which
# needs a heavy tail in the COUNT; it must not also multiply the attack rate.
# The ceiling is 1.0, not 5.0: the second run chose 4.44, which is almost Poisson, and the
# community contact distribution came out with a p90 of 18 against the 172 of the tracing
# records -- the heavy tail B27 exists for was simply absent. Below 1 the Gamma-Poisson has
# the over-dispersed shape the records show.
COMMUNITY_DISPERSION = (0.4, 0.05, 1.0)

# P[199..203] (B54): municipality mass events, (seed, lower, upper) in the order of
# data_synthesize.COMMUNITY_EVENT_FIELDS. The course block starts after the 37 contact
# parameters P[0..36], so course index 162 is P[199].
# * probability: searched in [0, 0.20], seed 0.10 (B55). Run 11 settled at 0.012 for two
#   reasons now removed: a single sampled event inflated the Cheng fit (E85; the fit now
#   bins the expected event contacts) and the fit compared 300 cases of contacts with
#   Cheng's 100 (E86). Locking it at 0.10, where run 11's other parameters give Cheng's
#   1,822 'others' contacts per 100 cases, was considered and dropped: those parameters
#   were fitted under E86, so the Cheng fit itself now decides how much of the contact
#   count comes from events. The seed 0.10 puts the tail ratio at 7.3 (target [5.5, 93.1]).
# * exponent 1.49 and min_size 21: locked to the maximum-likelihood power-law tail of the 38
#   first-wave tracing records (Clauset et al. 2009 method: k_min chosen by the KS distance,
#   0.065, 17 records in the tail), not chosen by eye (todolist929 4.3).
# * max_size 1000: locked; the largest record is 850, and the cap bounds the run time.
# * risk_ratio: locked at 1, so an event contact carries the ordinary municipality attack
#   rate and only the contact-count distribution changes, as the 2026-09-29 meeting asked.
#   The 0 infections among the 2,795 contacts of records with >= 100 community contacts
#   come from only 7 index cases; with CovSyn's case-level dispersion (run 10: P[35] = 0.177) and
#   municipality attack rate (~0.22%) they do not reject a ratio of 1 (P(0) = 0.14).
CONTACT_PARAMETER_COUNT = 37
COMMUNITY_EVENT_COURSE_INDEX = COMMUNITY_EVENT_FIRST_INDEX - CONTACT_PARAMETER_COUNT
# B59 (2026-10-08): P[199] is floored at 0.05. Run 13 settled at 0.0035, but with the community
# attack rate compensating, the expected objective is flat from 0 to 0.05 (probe_event_floor.py,
# 50 unseen 300-seed blocks per point) and 0.05 is where the tail ratio enters [5.5, 93.1].
COMMUNITY_EVENT = ((0.10, 0.05, 0.20),
                   (1.49, 1.49, 1.49),
                   (21, 21, 21),
                   (1000, 1000, 1000),
                   (1.0, 1.0, 1.0))

# The five layers' daily attack-rate bounds are no longer patched in place here. They are
# REBUILT from sar_anchors.py, which is the single source of truth for the anchors, the Ge
# 2021 profile shape and the contact-days conversion (see main()). Patching in place is what
# produced finding E51: the upper bound was rescaled down to a ceiling while the lower bound
# was only ever element-wise min'd against it, so the household daily attack rate ended up
# confined to [0.0532, 0.0553] and the only way to lower the household cumulative SAR was to
# cut contact days.

# ------------------------------------------------------------------ course of disease
# index into course_parameters (P index = 37 + i): (value, lower, upper)
COURSE_UPDATES = {
    # P[38] latent scale. The physiology target for the latent period is 4.1-4.5 days
    # (decision B1: put the latent period inside the literature reported-mean range and
    # accept that the generation time then sits above its own range), but the bounds only
    # reached 4.0 -- shape is locked at 4.0 and the scale ceiling was 1.0 -- so the target
    # was unreachable by construction and the optimizer settled at 3.53, BELOW the reported
    # range it was supposed to be inside (finding E34). The scale is now pinned to the range
    # that produces exactly the decided mean, the same way the severity cascade is pinned:
    # B1 is a decision, not something to be traded against the contact fit.
    # B3 (2026-09-25): the latent period is unpinned from [4.1, 4.5] to the whole literature
    # reported-mean range [4.1, 5.5]. E34 pinned it because the target was unreachable and
    # the optimizer traded it away to 3.53; the fix for that was to make the target
    # reachable, not to freeze the parameter, and freezing it also removed the only slack
    # the generation time had. P[37] shape stays 4.0, so scale 1.025-1.375 gives 4.1-5.5.
    1:  (1.200, 1.025, 1.375),   # P[38] latent scale -> mean 4.1 to 5.5 days
    4:  (1.5, 0.8, 4.0),      # P[41] pre-onset window shape
    5:  (1.3, 0.3, 2.5),      # P[42] pre-onset window scale   -> mean ~2 days
    # B2 (2026-09-25): room for a median of 5-7 days. The old box could reach a mean of at
    # most 3*2+1 = 7 days but was seeded at a median of 1, and all three Phase D runs stayed
    # at 1.0 -- the value that leaves almost no post-onset contact window and drives nine of
    # run 3's seventeen failures. Taiwan's own tracing file gives a median of 6 days over its
    # 442 symptomatic cases, Ge 2021 gives 5 days onset to isolation, and the published
    # CovSyn model fits a mean of 8.17 days. Seeded at a median of about 5.3 days.
    6:  (3.0, 0.5, 6.0),      # P[43] onset -> confirmation shape
    7:  (1.8, 0.3, 3.0),      # P[44] onset -> confirmation scale
    8:  (0.5, 0.0, 3.0),      # P[45] onset -> confirmation loc -> median about 5.3 days
    9:  (4.0, 1.0, 9.0),      # P[46] asymptomatic -> closure shape
    10: (3.0, 0.8, 6.0),      # P[47] asymptomatic -> closure scale
    11: (0.0, 0.0, 5.0),      # P[48] asymptomatic -> closure loc
    15: (4.0, 1.0, 9.0),      # P[52] symptomatic -> closure shape
    16: (3.0, 0.8, 6.0),      # P[53] symptomatic -> closure scale
    17: (0.0, 0.0, 5.0),      # P[54] symptomatic -> closure loc
    18: (4.0, 1.0, 9.0),      # P[55] ICU -> closure shape
    19: (3.0, 0.8, 6.0),      # P[56] ICU -> closure scale
    20: (0.0, 0.0, 6.0),      # P[57] ICU -> closure loc
}
# B2 / B14: locked by hand, lb == ub.
AGE_RISK_RATIOS = np.array([0.39, 1.0, 1.90, 1.44])
# B33, measured on the 579 cases of the Taiwan contact-tracing file.
# These three are not fitted against the simulation: each one IS the quantity it has to
# reproduce, so a measured penalty would only let the optimizer overfit the 100 fixed seeds
# the objective is evaluated on (which happen to contain 1 ICU case instead of the expected
# 9). They are constrained by narrow bounds instead, centred on Taiwan's own cascade and
# wide enough to absorb the fact that the age gradient is applied on top.
# Absolute indices, NOT negative ones: once P[198] is appended the vector grows and negative
# indices silently point at the wrong parameters -- re-running this script then overwrote the
# ICU and death probabilities with the asymptomatic share and clobbered the new parameter.
TRANSITION_P = {
    # 2026-09-27 (E58): the asymptomatic ceiling comes down 0.28 -> 0.26. Run 5 pegged P[195]
    # at 0.28 and the REALISED share still came out at 29.2%, outside B33's [20, 28]: the age
    # normalisation sits on top of this parameter, so it is not exactly the share after all,
    # and bounding the parameter at the top of the band cannot bound the share at the top of
    # the band. The share is also charged in the objective now, so this is the belt to that
    # brace rather than the only defence.
    158: (0.237, 0.200, 0.260),   # P[195] infection -> recovered = asymptomatic share 23.7%
    159: (0.873, 0.840, 0.900),   # P[196] symptom -> recovered, i.e. 10-16% go to ICU
    160: (0.875, 0.820, 0.920),   # P[197] ICU -> recovered, i.e. 8-18% die
}


def main():
    backup_once('contact_parameters.pkl')
    with open(VAR / 'contact_parameters.pkl', 'rb') as f:
        contact_parameters = pickle.load(f)
    for layer, (lb, ub) in CONTACT_BOUNDS.items():
        contact_parameters[f'{layer}_lower_bound'] = lb
        contact_parameters[f'{layer}_upper_bound'] = ub
    contact_parameters['overdispersion_lower_bound'] = OVERDISPERSION_BOUNDS[0]
    contact_parameters['overdispersion_upper_bound'] = OVERDISPERSION_BOUNDS[1]
    with open(VAR / 'contact_parameters.pkl', 'wb') as f:
        pickle.dump(contact_parameters, f)

    for name in ('course_parameters.npy', 'course_parameters_lb.npy', 'course_parameters_ub.npy'):
        backup_once(name)
    value = np.load(VAR / 'course_parameters.npy')
    lower = np.load(VAR / 'course_parameters_lb.npy')
    upper = np.load(VAR / 'course_parameters_ub.npy')

    for i, (v, lo, hi) in COURSE_UPDATES.items():
        value[i], lower[i], upper[i] = v, lo, hi
    value[26:30] = AGE_RISK_RATIOS
    lower[26:30] = AGE_RISK_RATIOS
    upper[26:30] = AGE_RISK_RATIOS
    for i, (v, lo, hi) in TRANSITION_P.items():
        value[i], lower[i], upper[i] = v, lo, hi

    # P[198] lives at the end of the course-of-disease block, so appending it here puts it
    # at the end of the whole parameter vector and every existing index keeps its meaning.
    v, lo, hi = COMMUNITY_DISPERSION
    if len(value) == 161:
        value = np.append(value, v)
        lower = np.append(lower, lo)
        upper = np.append(upper, hi)
    else:
        value[161], lower[161], upper[161] = v, lo, hi

    # P[199..203] (B54) follow P[198] for the same reason.
    for offset, (event_v, event_lo, event_hi) in enumerate(COMMUNITY_EVENT):
        i = COMMUNITY_EVENT_COURSE_INDEX + offset
        if len(value) == i:
            value = np.append(value, event_v)
            lower = np.append(lower, event_lo)
            upper = np.append(upper, event_hi)
        else:
            value[i], lower[i], upper[i] = event_v, event_lo, event_hi

    # Rebuild each layer's 25 daily attack rates from the anchors, rather than rescaling
    # whatever happens to be on disk. The arrays in variable/ descend from the ORIGINAL
    # published values and had been patched in place by earlier runs of this script, so
    # rescaling them was neither idempotent nor traceable to any anchor.
    print('  attack rates rebuilt from sar_anchors.py; contact days measured on %s'
          % CONTACT_DAYS_MEASURED_ON)
    for layer in LAYERS:
        start, stop = ATTACK_RATE_SLICE[layer]
        was_lo, was_hi = float(np.mean(lower[start:stop])), float(np.mean(upper[start:stop]))
        block, block_lo, block_hi = attack_rate_block(layer)
        value[start:stop] = block
        lower[start:stop] = block_lo
        upper[start:stop] = block_hi
        sar_lb, _sar_centre, sar_ub = LAYER_CUMULATIVE_SAR[layer]
        print('  %-13s daily [%.5f, %.5f] -> [%.5f, %.5f]  centre %.5f'
              '   (cumulative %.1f-%.1f%% over %.2f days)'
              % (layer, was_lo, was_hi, float(np.mean(block_lo)), float(np.mean(block_hi)),
                 float(np.mean(block)), 100 * sar_lb, 100 * sar_ub, MEAN_CONTACT_DAYS[layer]))

    bad = np.where((value < lower) | (value > upper))[0]
    if len(bad):
        raise SystemExit(f'seed value outside its bounds at course indices {bad.tolist()}')

    np.save(VAR / 'course_parameters.npy', value)
    np.save(VAR / 'course_parameters_lb.npy', lower)
    np.save(VAR / 'course_parameters_ub.npy', upper)

    print('contact bounds and course-of-disease bounds rewritten for Phase D')
    print('  pre-onset window mean      %.2f days' % (value[4] * value[5]))
    print('  onset -> confirmation mean %.2f days' % (value[6] * value[7] + value[8]))
    print('  asymptomatic share         %.1f%%' % (100 * value[158]))
    print('  symptomatic -> ICU         %.1f%%' % (100 * (1 - value[159])))
    print('  ICU -> death               %.1f%%' % (100 * (1 - value[160])))
    print('  age risk ratios locked at  %s' % AGE_RISK_RATIOS.tolist())
    print('  infectiousness shape k in  [%g, %g]' % (OVERDISPERSION_BOUNDS[0][0], OVERDISPERSION_BOUNDS[1][0]))
    print('  community dispersion k in  [%g, %g], seed %g' % (lo, hi, v))
    print('  parameter vector length    %d' % (37 + len(value)))
    print('  backup of the previous files in', BACKUP)


if __name__ == '__main__':
    main()
