"""Per-case check of the temporal / biological constraints of the CURRENT CovSyn model (N2, E48).

todolist923.md 1.2 / 1.3 asked for a citable constraint table and a script that checks every
simulated case against it. The professor's draft table was written for the pre-Phase-D model: its
C02 (end of latency < isolation) and C03 (isolation < end of infectiousness) were removed on
purpose by B26 -- contact tracing can isolate a case before it becomes infectious, and an untraced
asymptomatic case is isolated only when its infectious period is over. They are reported here as
DESCRIPTIVE shares, not as constraints, so the table does not claim the model enforces something it
does not (finding E48).

Every time below is in days. latent_period, incubation_period, infectious_period and
monitor_isolation_period are counted from the case's own infection; infection_day and the date_of_*
fields are absolute simulation days. The infectious window is the CLOSED interval
[latent, latent + infectious] (calculate_daily_secondary_attack_rate builds infectious + 1 daily
values), and every contact window includes the isolation day itself.

Outputs (in out_dir):
  constraint_summary.csv      one row per constraint: pass / fail / not applicable / pass rate
  constraint_violations.csv   every failing case or transmission with its values
  constraint_table.md         the thesis Table A, with the counts filled in

Usage: python check_constraints.py [data_dir ...] [--out out_dir]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEALTH_CARE_POST_ISOLATION_DAYS = 14   # Data_synthesize.py (B23 / B48)

# id, statement, rationale, applies to, kind, reference
CONSTRAINTS = [
    ('C01', 'latent >= 0', 'a case cannot become infectious before it is infected', 'all cases',
     'by construction', 'definition; CovSyn preprint supplement, state diagram'),
    ('C02', 'infectious >= 1', 'every infected case is infectious for at least one day', 'all cases',
     'by construction', 'CovSyn preprint supplement'),
    ('C03', 'latent <= incubation', 'symptoms cannot precede infectiousness in the model; incubation = '
     'latent + pre-symptomatic window (B22)', 'symptomatic', 'by construction',
     'Byrne et al. 2020 (pre-symptomatic infectious period >= 0)'),
    ('C04', 'incubation <= latent + infectious', 'symptoms appear while the case is still infectious '
     '(pre-symptomatic window capped at the infectious period)', 'symptomatic', 'by construction',
     'Byrne et al. 2020 (post-symptomatic infectious period >= 0)'),
    ('C05', 'isolation >= 0', 'a case cannot be isolated before it is infected', 'all cases',
     'by construction', 'definition'),
    ('C06', 'symptom route: isolation >= incubation', 'a case that presents by itself is isolated on or '
     'after its own symptom onset', "isolation_route == 'symptom'", 'by construction (B26)',
     'Cheng et al. 2020; Ge et al. 2021 (onset to isolation)'),
    ('C07', '|positive test - (infection + isolation)| <= 1', 'the confirming test is taken at isolation',
     'all cases', 'by construction', 'model rule; reference to be confirmed'),
    ('C08', 'closure >= end of infectiousness', 'a case is closed (released or dead) only after it stops '
     'being infectious (B28)', 'all cases', 'by construction',
     'Jian et al. 2020 (release criteria); Byrne et al. 2020'),
    ('C09', 'onset <= critical illness', 'critical illness follows symptom onset', 'ICU cases',
     'by construction', 'CovSyn preprint supplement'),
    ('C10', 'critical illness <= end of infectiousness + 1', 'the model draws ICU admission inside the '
     'infectious period', 'ICU cases', 'model rule', 'CovSyn preprint supplement; to be confirmed'),
    ('C11', 'critical illness <= closure', 'recovery or death does not precede critical illness; the '
     'model allows death on the day of ICU admission (closure >= max(end of infectiousness, ICU day))',
     'ICU cases', 'by construction', 'definition'),
    ('C12', 'exactly one outcome (recovered or dead)', 'every case ends in one absorbing state',
     'all cases', 'by construction', 'definition'),
    ('C13', 'asymptomatic: no ICU and no death', 'the asymptomatic path has no severe branch',
     'asymptomatic', 'by construction', 'CovSyn preprint supplement'),
    ('C14', 'isolation <= critical illness', 'a case in intensive care is hospitalised, so it must already '
     'be isolated; before that day it cannot keep meeting household, school or work contacts '
     '(todolist 1.2: isolation < critical state < end of infectiousness)', 'ICU cases',
     'NOT enforced by the model (E81)', 'todolist 1.2; clinical definition of ICU admission'),
    ('T01', 'infector latent <= generation interval', 'no transmission before the infector is '
     'infectious (todolist 1.2 C04)', 'transmissions', 'model rule',
     'definition of the latent period; Byrne et al. 2020'),
    ('T02', 'generation interval <= infector latent + infectious', 'no transmission after the infector '
     'stops being infectious', 'transmissions', 'model rule', 'Byrne et al. 2020'),
    ('T03', 'generation interval <= infector isolation', 'isolation stops transmission in every setting '
     'except health care', 'transmissions outside health care', 'model rule (B26)',
     'Li W et al. 2020 (0% SAR when the index case isolated); Cheng et al. 2020'),
    ('T04', 'generation interval <= infector isolation + 14', 'health-care contact continues for at most '
     '14 days after isolation (B23 / B48)', 'health-care transmissions', 'model rule', 'Cheng et al. 2020'),
    ('T05', 'every case is infected at most once', 'an infectee has one infector', 'transmissions',
     'by construction', 'definition'),
]
DESCRIPTIVE = [
    ('D01', 'isolated before becoming infectious (isolation < latent)',
     'was C02 in the 2026-09-23 draft; removed as a constraint by B26: tracing can isolate a case early'),
    ('D02', 'isolated after the end of infectiousness (isolation > latent + infectious)',
     'was C03 in the draft; removed by B26: an untraced asymptomatic case is found only afterwards'),
    ('D03', 'traced before its own symptom onset (symptomatic, isolation < incubation)',
     'contact tracing reaching a case pre-symptomatically'),
]


def finite(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def case_id(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else int(f)


def check(dirs):
    counts = {c[0]: [0, 0, 0] for c in CONSTRAINTS}     # pass, fail, not applicable
    described = {d[0]: [0, 0] for d in DESCRIPTIVE}    # true, applicable
    bad = []

    def record(cid, ok, applicable, where, values):
        if not applicable:
            counts[cid][2] += 1
            return
        counts[cid][0 if ok else 1] += 1
        if not ok:
            bad.append({'constraint': cid, **where, **values})

    for d in dirs:
        files = sorted(glob.glob(f'{d}/course_of_disease_data_*.npy'),
                       key=lambda p: int(p.split('_')[-1].split('.')[0]))
        for f in files:
            sim = int(f.split('_')[-1].split('.')[0])
            course = list(np.load(f, allow_pickle=True))
            try:
                digraph = np.load(f'{d}/transmission_digraph_{sim}.npy', allow_pickle=True)
            except FileNotFoundError:
                digraph = []
            for i, c in enumerate(course):
                where = {'dataset': Path(d).name, 'simulation': sim, 'case': i + 1}
                lat, inf, iso = float(c['latent_period']), float(c['infectious_period']), \
                    float(c['monitor_isolation_period'])
                t0 = float(c['infection_day'])
                sym = finite(c['incubation_period'])
                inc = float(c['incubation_period']) if sym else np.nan
                end = lat + inf
                icu, dead, rec = (float(c[k]) if finite(c[k]) else np.nan
                                  for k in ('date_of_critically_ill', 'date_of_death', 'date_of_recovery'))
                closure = np.nanmax([rec, dead]) if (finite(rec) or finite(dead)) else np.nan
                v = {'latent': lat, 'incubation': inc, 'infectious': inf, 'isolation': iso,
                     'infection_day': t0, 'icu': icu, 'death': dead, 'recovery': rec}
                record('C01', lat >= 0, True, where, v)
                record('C02', inf >= 1, True, where, v)
                record('C03', sym and lat <= inc, sym, where, v)
                record('C04', sym and inc <= end, sym, where, v)
                record('C05', iso >= 0, True, where, v)
                is_symptom_route = c.get('isolation_route') == 'symptom'
                record('C06', is_symptom_route and sym and iso >= inc, is_symptom_route and sym, where, v)
                pos = float(c['positive_test_date']) if finite(c['positive_test_date']) else np.nan
                record('C07', finite(pos) and abs(pos - (t0 + iso)) <= 1, True, where, {**v, 'positive_test': pos})
                record('C08', finite(closure) and closure >= t0 + end, True, where, v)
                record('C09', finite(icu) and sym and icu >= t0 + inc, finite(icu), where, v)
                record('C10', finite(icu) and icu <= t0 + end + 1, finite(icu), where, v)
                record('C11', finite(icu) and finite(closure) and icu <= closure, finite(icu), where, v)
                record('C14', finite(icu) and t0 + iso <= icu, finite(icu), where, v)
                record('C12', finite(rec) != finite(dead), True, where, v)
                record('C13', not finite(icu) and not finite(dead), not sym, where, v)
                described['D01'][1] += 1
                described['D01'][0] += iso < lat
                described['D02'][1] += 1
                described['D02'][0] += iso > end
                if sym:
                    described['D03'][1] += 1
                    described['D03'][0] += iso < inc
            by = {i + 1: c for i, c in enumerate(course)}
            seen = set()
            for edge in digraph:
                p, q = case_id(edge[0]), case_id(edge[1])
                if p is None or q is None or p not in by or q not in by:
                    continue
                layer = str(edge[3]) if len(edge) > 3 else ''
                src, dst = by[p], by[q]
                g = float(dst['infection_day']) - float(src['infection_day'])
                lat, inf, iso = float(src['latent_period']), float(src['infectious_period']), \
                    float(src['monitor_isolation_period'])
                where = {'dataset': Path(d).name, 'simulation': sim, 'case': q, 'infector': p}
                v = {'layer': layer, 'generation_interval': g, 'infector_latent': lat,
                     'infector_infectious': inf, 'infector_isolation': iso}
                record('T01', g >= lat, True, where, v)
                record('T02', g <= lat + inf, True, where, v)
                record('T03', g <= iso, layer != 'health_care', where, v)
                record('T04', g <= iso + HEALTH_CARE_POST_ISOLATION_DAYS, layer == 'health_care', where, v)
                record('T05', q not in seen, True, where, v)
                seen.add(q)
    return counts, described, bad


def main():
    args = sys.argv[1:]
    out = Path('constraint_check')
    if '--out' in args:
        k = args.index('--out')
        out = Path(args[k + 1])
        args = args[:k] + args[k + 2:]
    dirs = args or ['synthetic_data_results_spread_Taiwan_weight']
    out.mkdir(parents=True, exist_ok=True)
    counts, described, bad = check(dirs)

    rows = []
    for cid, statement, why, applies, kind, ref in CONSTRAINTS:
        p, f, na = counts[cid]
        rows.append({'id': cid, 'constraint': statement, 'rationale': why, 'applies_to': applies,
                     'kind': kind, 'reference': ref, 'passed': p, 'failed': f, 'not_applicable': na,
                     'pass_rate': (p / (p + f)) if p + f else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'constraint_summary.csv', index=False)
    pd.DataFrame(bad).to_csv(out / 'constraint_violations.csv', index=False)

    lines = ['# Table A - temporal / biological constraints of the current CovSyn model', '',
             f'Data: {", ".join(dirs)}. Generated by `check_constraints.py`.', '',
             '| ID | Constraint | Rationale | Applies to | Kind | Reference | Passed | Failed | Pass rate |',
             '|---|---|---|---|---|---|---:|---:|---:|']
    for r in rows:
        rate = '' if np.isnan(r['pass_rate']) else f"{100 * r['pass_rate']:.2f}%"
        lines.append(f"| {r['id']} | {r['constraint']} | {r['rationale']} | {r['applies_to']} | {r['kind']} "
                     f"| {r['reference']} | {r['passed']:,} | {r['failed']:,} | {rate} |")
    lines += ['', '## Descriptive shares (not constraints of the current model)', '',
              '| ID | Quantity | Share | Why it is not a constraint |', '|---|---|---:|---|']
    for did, text, why in DESCRIPTIVE:
        t, n = described[did]
        lines.append(f'| {did} | {text} | {t:,} / {n:,} = {100 * t / max(n, 1):.1f}% | {why} |')
    (out / 'constraint_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    for r in rows:
        print('%s %-48s pass %8d  fail %6d  n/a %8d' % (r['id'], r['constraint'][:48], r['passed'],
                                                       r['failed'], r['not_applicable']))
    for did, text, _ in DESCRIPTIVE:
        t, n = described[did]
        print('%s %-60s %d / %d' % (did, text[:60], t, n))
    print('violations written:', len(bad), '->', out)


if __name__ == '__main__':
    main()
