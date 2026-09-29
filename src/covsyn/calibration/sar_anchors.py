"""Single source of truth for the five layers' secondary attack rates.

Before this module the same quantity was written down twice with different values:
`parameters_for_initialization.py` converted the literature cumulative SAR to a daily
probability using contact-day constants {household 2.00, school 2.95, workplace 4.02,
health_care 1.99, municipality 3.91}, while `apply_phaseD_parameters.py` derived the daily
CEILING from a different set {3.80, 4.98, 5.39, 4.81, 1.98}. Worse, the first of those two
never ran: `parameters_for_initialization.py` is a one-shot generator that nothing in the
Phase D pipeline imports, so the bounds actually in `variable/course_parameters_*.npy` were
the ORIGINAL published arrays patched in place. Both files now import from here.

Everything a layer's 25 daily attack rates depend on is in one place:

  GE2021_RELATIVE_RISK   the shape of the profile, from Ge 2021 Figure 2
  LAYER_CUMULATIVE_SAR   the level, one (lower, centre, upper) per layer
  MEAN_CONTACT_DAYS      how many days a candidate contact is actually met
  CEILING_SAFETY_FACTOR  how far above the upper anchor the daily ceiling is allowed

Reference: Ge Y, et al. COVID-19 Transmission Dynamics Among Close Contacts of Index
Patients With COVID-19. JAMA Intern Med. 2021;181(10):1343-1350. PDF in C:/Theisis/pdf.
"""
import numpy as np

# ---------------------------------------------------------------------- profile shape
# Ge 2021 Figure 2: adjusted relative risk of COVID-19 in a close contact by the day of
# exposure relative to the index patient's symptom onset, one bin per day from -14 to +10.
# That is 25 bins, which is why this array has 25 elements; element 14 IS symptom onset.
# Ge's own text pins three of them: the peak at day 0 (ARR 1.3, 95% CI 1.2-1.5, here
# 1.34 / 1.18 / 1.54) and the dip at days -6 and -5 (ARR 0.8, 95% CI 0.6-1.0, here
# 0.78 / 0.65 / 0.94 and 0.78 / 0.63 / 0.96).
#
# Caveat to state in the thesis: Ge measured this against ABSOLUTE days. CovSyn stretches it
# onto each case's own latent->onset and onset->end-of-infectiousness windows with linspace,
# turning an absolute-time profile into a relative-phase one. A case with a 2-day pre-onset
# window gets Ge's 14 pre-onset days compressed into 2. That is a modelling choice, not
# something Ge established. The apparent second hump at days -11/-10 also sits in the region
# Ge describes as "nonstatistically higher", i.e. it is noise being carried as signal.
GE2021_RELATIVE_RISK = np.array([
    0.86, 0.98, 1.09, 1.16, 1.16, 1.07, 0.95, 0.84, 0.78, 0.78, 0.86, 1.02, 1.18,
    1.30, 1.34, 1.33, 1.27, 1.19, 1.11, 1.05, 1.01, 0.99, 0.98, 0.99, 0.99])
GE2021_RELATIVE_RISK_LB = np.array([
    0.39, 0.63, 0.90, 0.91, 0.81, 0.75, 0.72, 0.70, 0.65, 0.63, 0.72, 0.90, 1.05,
    1.13, 1.18, 1.19, 1.15, 1.07, 0.98, 0.92, 0.89, 0.88, 0.84, 0.76, 0.69])
GE2021_RELATIVE_RISK_UB = np.array([
    1.91, 1.52, 1.31, 1.48, 1.66, 1.54, 1.26, 1.02, 0.94, 0.96, 1.04, 1.15, 1.34,
    1.49, 1.54, 1.49, 1.40, 1.32, 1.26, 1.20, 1.14, 1.11, 1.15, 1.27, 1.43])

LAYERS = ('household', 'school', 'workplace', 'health_care', 'municipality')

# Where each layer's 25 daily attack rates live inside course_parameters.
ATTACK_RATE_SLICE = {'household': (33, 58), 'school': (58, 83), 'workplace': (83, 108),
                     'health_care': (108, 133), 'municipality': (133, 158)}

# ---------------------------------------------------------------------- level per layer
# CUMULATIVE secondary attack rate per candidate contact over the whole exposure, as
# (lower, centre, upper). NOT a per-day probability -- see MEAN_CONTACT_DAYS below.
#
# E55, decided 2026-09-26: the three layers Cheng 2020 reports are anchored on INFECTIONS
# PER INDEX CASE, not on the secondary attack rate per contact. Cheng's SAR denominator is
# the contacts Taiwan's tracers chose to trace; CovSyn's is its own candidate contacts, and
# the two differ by a different factor in every layer -- CovSyn has 1.78x Cheng's household
# contacts but only 0.39x his health care contacts. The quantity that survives that
# difference is how many people one index case infects in a layer, so the anchor is
#
#       anchor = (Cheng infections per index case) / (CovSyn candidate contacts per index)
#
# with the interval from the exact Poisson 95% CI of Cheng's own infection COUNT, which is
# where the real uncertainty sits (10, 6 and 1 infections respectively). Run this module as
# a script to reprint the derivation.
#
#   household    10/100 index over 2.684 candidate contacts -> 3.73% (1.79-6.85).
#                Was 5.5% under B1, and 6.62% before that.
#   health_care   6/100 index over 2.747 -> 2.18% (0.80-4.75). This RAISES the anchor 2.5x,
#                which is the point of E55: CovSyn gives a case 2.75 medical contacts where
#                Cheng traced 6.98, so reproducing Cheng's 6 infections needs a HIGHER rate
#                per contact, not the same one. Watch the pooled age risk ratios -- this is
#                the layer whose contacts spread evenly across all four age bands.
#   municipality  1/100 index over 4.115 -> 0.243%. The Poisson lower bound of a single
#                infection is 0.006%, which would let the optimizer switch the community
#                layer off entirely, so the floor is held at 0.1% (B27 needs a tail here).
#
# school and workplace are NOT on this basis. Cheng has no school and no workplace category,
# so there is no infections-per-index-case figure to divide by, and they keep the per-contact
# literature anchors. STATE THIS ASYMMETRY IN THE THESIS: three layers are calibrated to
# Taiwan per index case, two to per-contact values whose provenance is itself unresolved.
#
#   school       B4: kept at 1-4% CUMULATIVE. The published model used the same 1-4% but as
#                a DAILY probability, which over its own 2.95 contact days is 2.96-11.5%
#                cumulative -- our target is about 3x stricter, deliberately.
#   workplace    3.4% centre. Upper bound 5%, not the 15% of the published model: at 15%
#                the optimizer pushed workplace to 7.6% cumulative, and because workplace
#                contacts concentrate in ages 20-59 that inflated the reference group and
#                dragged the pooled 60+ age risk ratio to 0.81 against an input of 1.32.
#
# CIRCULARITY, to be stated as a limitation: three of the five anchors now depend on the
# model's own candidate-contact counts, which the same optimisation moves. They are derived
# from the PREVIOUS run's counts and held constant throughout a run, never updated inside
# the loop, and re-derived between runs exactly as MEAN_CONTACT_DAYS is. If a run moves a
# layer's candidate-contact count materially, that anchor must be re-derived before the next.
#
# PROVENANCE WARNING (finding E52). Ge 2021 Table 2 is the source of the published model's
# household 0.101 and health care 0.004, and of nothing else: Ge's contact types are
# conversation, dining, enclosed space, health care, living together, multiple settings,
# shared transport and others, with the non-household non-health-care ones POOLED into
# "Other" at 62/5931 = 1.0%. There is no school and no workplace category. The published
# school 0.024, workplace 0.034 and municipality 0.002 appear in neither Ge nor Cheng.
# Cheng 2020, 100 index cases: (infections, contacts traced) per exposure setting.
CHENG2020_INDEX_CASES = 100
CHENG2020_BY_SETTING = {'household': (10, 151), 'health_care': (6, 698),
                        'municipality': (1, 1836)}
# CovSyn candidate contacts per index case, measured on the fourth Phase D run
# (verify_phaseD.py, 1000 index cases). Re-measure and update between runs.
COVSYN_CANDIDATE_CONTACTS = {'household': 2.696, 'school': 1.430, 'workplace': 1.229,
                             'health_care': 1.728, 'municipality': 4.491}
CANDIDATE_CONTACTS_MEASURED_ON = 'Phase D run 5 (rr_population_check.py, 1000 index cases)'
# How far these moved on run 5, which is what E56 is about: school -47.7%, health care -37.1%,
# workplace -12.3%, municipality +9.1%, household +0.4%. Run 4 had
# {2.684, 2.733, 1.401, 2.747, 4.115}.
ANCHOR_FLOOR = {'municipality': 0.001}


# E57, decided 2026-09-27: the Poisson 95% CI of Cheng's community count is useless as an
# acceptance band because that count is ONE infection -- the interval runs to 5.6x the centre,
# and run 5 walked the community layer out to 4.9x its anchor while still being scored "ok".
# The community upper bound is therefore a fixed multiple of the centre instead of the CI.
# The lower bound stays at a floor rather than the CI's 0.025x, because B27 needs this layer
# to keep a tail; switching the community off entirely is not an acceptable fit either.
COMMUNITY_UPPER_MULTIPLE = 3.0
PER_INDEX_FLOOR = {'municipality': 0.001}


def cheng_poisson_interval(layer):
    """(lower, centre, upper) INFECTIONS PER INDEX CASE in this layer, from Cheng 2020.

    The exact Poisson 95% CI of Cheng's own infection count, which is where the real
    uncertainty sits (10, 6 and 1 infections). Nothing about CovSyn enters this.
    """
    from scipy.stats import chi2
    infections, _traced = CHENG2020_BY_SETTING[layer]
    centre = infections / CHENG2020_INDEX_CASES
    lo = chi2.ppf(0.025, 2 * infections) / 2 / CHENG2020_INDEX_CASES
    hi = chi2.ppf(0.975, 2 * infections + 2) / 2 / CHENG2020_INDEX_CASES
    return lo, centre, hi


def infections_per_index_target(layer):
    """(lower, centre, upper) infections per index case, as the OBJECTIVE now charges it.

    E56, decided 2026-09-27. Until run 5 the three layers Cheng reports were charged as a
    per-contact rate obtained by dividing his infection count by CovSyn's OWN candidate
    contact count (see infections_per_index_anchor below). That made the target a ratio whose
    denominator the optimizer controls, and run 5 showed exactly what that buys: the health
    care anchor was raised 2.5x, the optimizer cut health care contacts by 37% in response,
    and the quantity the anchor existed to fix -- medical infections per index case -- went
    DOWN, from 0.0160 to 0.0140 against Cheng's 0.06. Charging the product instead removes
    the lever: a case either infects 0.06 people in a medical setting or it does not, however
    many contacts it has. This is HANDOVER lesson 1 (never charge a ratio without anchoring
    its level) applied to the anchors themselves.
    """
    lo, centre, hi = cheng_poisson_interval(layer)
    if layer == 'municipality':
        hi = COMMUNITY_UPPER_MULTIPLE * centre
    return max(lo, PER_INDEX_FLOOR.get(layer, 0.0)), centre, hi


def infections_per_index_anchor(layer):
    """(lower, centre, upper) PER CONTACT, from Cheng's count over CovSyn's contact count.

    Still used to place the daily attack-rate BOUNDS, which have to be written per contact
    per day. It is no longer an optimisation target (E56), so the circularity noted above now
    only affects how much headroom the search has, not what it is aiming at.
    """
    lo, per_index, hi = cheng_poisson_interval(layer)
    contacts = COVSYN_CANDIDATE_CONTACTS[layer]
    return (max(lo / contacts, ANCHOR_FLOOR.get(layer, 0.0)),
            per_index / contacts, hi / contacts)


LAYER_CUMULATIVE_SAR = {
    # per-contact literature anchors; Cheng has no category for these two
    'school':    (0.010, 0.023, 0.040),
    'workplace': (0.015, 0.034, 0.050),
}
for _layer in CHENG2020_BY_SETTING:
    LAYER_CUMULATIVE_SAR[_layer] = infections_per_index_anchor(_layer)

# What the objective and the checklist now charge for the three layers Cheng reports: the
# number of people one index case infects in that layer (E56). school and workplace stay on
# their per-contact anchors above because Cheng has no category for them.
LAYER_INFECTIONS_PER_INDEX = {layer: infections_per_index_target(layer)
                              for layer in CHENG2020_BY_SETTING}

# ---------------------------------------------------------------------- contact days
# Mean number of DAYS a candidate contact is actually met, per layer: the row sums of
# `{layer}_contacts_matrix` averaged over every candidate contact. Measured on the third
# Phase D run (`measure_days.py`, 400 simulations of spread_Taiwan_weight).
#
# This is a model OUTPUT, not a literature value, and it moves between runs (the second run
# gave 2.82 / 3.17 / 3.69 / 4.16 / 1.71 and the third 3.71 / 4.45 / 6.81 / 2.23 / 2.49 by the
# same measurement; every layer grew on the fourth run because B2 pushed confirmation, and
# with it isolation, about five days later). It is re-measured after every run and updated
# here; the run it was measured on is recorded below so the conversion can always be redone.
# Ge 2021's median first-to-last exposure duration of 3 days (IQR 0-7) is the only external
# check available and is consistent with the earlier runs.
#
# E61, 2026-09-27: four runs in, this has grown EVERY time and shows no sign of converging
# (run 2 {2.82, 3.17, 3.69, 4.16, 1.71} -> run 3 {3.71, 4.45, 6.81, 2.23, 2.49} -> run 4
# {5.09, 4.79, 9.17, 3.18, 4.18} -> run 5 {7.65, 6.35, 10.30, 5.26, 5.02}), driven by B2
# pushing confirmation later and B23 extending the medical window past isolation. Re-measuring
# once per run therefore does NOT make the conversion hold: run 5 ran on run 4's numbers, so
# its household ceiling of 0.0277/day was really 19.4% cumulative over the 7.65 days actually
# realised, not the 13.3% this module printed. The reason that is now tolerable rather than
# fatal is E56: since the objective charges infections per index case directly, this
# conversion only sets how much headroom the search has, not what it aims at.
MEAN_CONTACT_DAYS = {'household': 7.65, 'school': 6.35, 'workplace': 10.30,
                     'health_care': 5.26, 'municipality': 5.02}
CONTACT_DAYS_MEASURED_ON = 'Phase D run 5 (spread_Taiwan_weight, 400 simulations)'

# How far above the upper anchor the DAILY ceiling may sit. The ceiling is a safety net, not
# a target: the cumulative SAR itself is a measured cost term (cost_attack_rate, on 300
# fixed seeds with the expected-infection estimator since E38), so the ceiling only has to
# stop a runaway. Twice, because the rate realised after averaging over the case-level
# infectiousness multiplier comes out well below its value at nu = 1.
CEILING_SAFETY_FACTOR = 2.0


def daily_from_cumulative(cumulative_sar, layer):
    """p_daily such that 1 - (1 - p_daily)**n_days == cumulative_sar."""
    return 1.0 - (1.0 - cumulative_sar) ** (1.0 / MEAN_CONTACT_DAYS[layer])


def attack_rate_block(layer):
    """The layer's 25 daily attack rates as (value, lower, upper) profiles.

    Each profile keeps Ge's shape and is scaled so that its MEAN is the daily probability
    implied by the corresponding cumulative anchor. The centre is clipped into the box
    because Ge's lower and upper profiles have a different shape from the central one.

    The lower bound is the honest literature lower bound. It is NOT clamped against the
    ceiling: doing that (lower = minimum(lower, upper)) is what froze the household daily
    attack rate into [0.0532, 0.0553] for all three Phase D runs -- a 4%-wide interval, so
    the only way the optimizer could lower the household cumulative SAR was to cut contact
    days, which is the mechanism behind finding E51.

    The CEILING of every non-household layer is capped at the household ceiling (E61). That
    is the opposite operation from the E51 bug -- it lowers an upper bound against another
    layer's upper bound, and never touches a lower bound, so no interval is squeezed shut --
    and it is checked below: a cap that would cross a layer's own lower bound raises instead
    of silently freezing it.
    """
    sar_lb, sar, sar_ub = LAYER_CUMULATIVE_SAR[layer]
    lower = GE2021_RELATIVE_RISK_LB * (daily_from_cumulative(sar_lb, layer)
                                       / GE2021_RELATIVE_RISK_LB.mean())
    upper = _ceiling_profile(layer)
    if layer != 'household':
        upper = np.minimum(upper, _ceiling_profile('household'))
    value = GE2021_RELATIVE_RISK * (daily_from_cumulative(sar, layer)
                                    / GE2021_RELATIVE_RISK.mean())
    if np.any(lower > upper):
        raise ValueError(layer + ': lower bound above upper bound, check the anchors')
    return np.clip(value, lower, upper), lower, upper


def _ceiling_profile(layer):
    """The layer's own daily ceiling, before the cross-layer cap of attack_rate_block()."""
    _sar_lb, _sar, sar_ub = LAYER_CUMULATIVE_SAR[layer]
    return GE2021_RELATIVE_RISK_UB * (CEILING_SAFETY_FACTOR
                                      * daily_from_cumulative(sar_ub, layer)
                                      / GE2021_RELATIVE_RISK_UB.mean())


def cross_layer_ceiling_report():
    """Check that no layer is allowed a higher daily attack rate than the household.

    E61, 2026-09-27: run 5's bounds let a HEALTH CARE contact be infected at up to 0.0441 per
    day against the household's 0.0402 -- i.e. the search was permitted to make meeting a
    nurse more dangerous than living with the case. That is not a defensible model and it was
    not a decision: it fell out of dividing Cheng's 6 medical infections by a health care
    contact count that the optimizer itself had shrunk. A layer breaching this is evidence
    that its DENOMINATOR is wrong (too few contacts of that kind), not that its rate should
    be that high, so the right response is to look at the contact model, not to raise the cap.

    attack_rate_block() now caps the breach away, so this reports on the UNCAPPED ceiling --
    the cap stops the implausible fit, this says whether the anchor derivation still wants one.

    Returns a list of (layer, its uncapped ceiling, household ceiling) for every breach.
    """
    household_ceiling = float(_ceiling_profile('household').max())
    breaches = []
    for layer in LAYERS:
        if layer == 'household':
            continue
        ceiling = float(_ceiling_profile(layer).max())
        if ceiling > household_ceiling:
            breaches.append((layer, ceiling, household_ceiling))
    return breaches


if __name__ == '__main__':
    print('contact days  : %s' % CONTACT_DAYS_MEASURED_ON)
    print('contacts/index: %s\n' % CANDIDATE_CONTACTS_MEASURED_ON)
    for layer in LAYERS:
        lo, centre, hi = LAYER_CUMULATIVE_SAR[layer]
        basis = ('Cheng infections/index over CovSyn contacts/index'
                 if layer in CHENG2020_BY_SETTING else 'per-contact literature')
        print('  %-13s cumulative %6.3f%%  [%.3f, %.3f]   (%s)'
              % (layer, 100 * centre, 100 * lo, 100 * hi, basis))
    print()
    print('%-13s %6s %10s %10s %10s   %s'
          % ('layer', 'days', 'lb daily', 'centre', 'ceiling', 'cumulative at the ceiling'))
    for layer in LAYERS:
        value, lower, upper = attack_rate_block(layer)
        days = MEAN_CONTACT_DAYS[layer]
        print('%-13s %6.2f %10.5f %10.5f %10.5f   %.1f%% at nu = 1'
              % (layer, days, lower.mean(), value.mean(), upper.mean(),
                 100 * (1 - (1 - upper.mean()) ** days)))

    print('\nwhat the objective charges for the three layers Cheng reports (E56):')
    print('%-13s %10s %10s %10s   %s'
          % ('layer', 'lower', 'centre', 'upper', 'basis'))
    for layer, (lo, centre, hi) in LAYER_INFECTIONS_PER_INDEX.items():
        basis = ('Poisson 95%% CI of %d infections' % CHENG2020_BY_SETTING[layer][0])
        if layer == 'municipality':
            basis = 'upper capped at %.0fx the centre (E57), floor %.3f' % (
                COMMUNITY_UPPER_MULTIPLE, PER_INDEX_FLOOR[layer])
        print('%-13s %10.4f %10.4f %10.4f   infections per index case, %s'
              % (layer, lo, centre, hi, basis))

    breaches = cross_layer_ceiling_report()
    print('\ncross-layer ceiling check (E61): %s'
          % ('every layer is capped at or below the household' if not breaches
             else 'BREACH -- the contact model is wrong, do not raise the cap'))
    for layer, ceiling, household_ceiling in breaches:
        print('  %-13s daily ceiling %.5f > household %.5f' % (layer, ceiling, household_ceiling))
