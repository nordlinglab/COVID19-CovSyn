"""Compare two Phase D verification runs check by check.

A rerun that fixes three things can quietly break a fourth. This prints the checklist of
both runs side by side and, first of all, the regressions: the checks that passed before the
change and fail after it. Those are the ones worth arguing about; everything else is either
an intended improvement or unchanged.

Usage: python compare_phaseD_runs.py old_checks.json new_checks.json
"""
import json
import sys


def load(path):
    data = json.load(open(path))
    return data, {(c['decision'], c['name']): c for c in data['checks']}


def fmt(check):
    if check is None:
        return 'not measured'
    value = check['value']
    text = 'n/a' if value is None else f'{value:.3f}'
    if check['ok'] is None:
        return f'{text} (info)'
    return f'{text} {"ok" if check["ok"] else "FAIL"}'


def main():
    old_path, new_path = sys.argv[1], sys.argv[2]
    old_data, old = load(old_path)
    new_data, new = load(new_path)
    print(f'old: {old_path}  ({old_data["simulations"]} simulations, {old_data["cases"]} cases)')
    print(f'new: {new_path}  ({new_data["simulations"]} simulations, {new_data["cases"]} cases)\n')

    keys = list(old) + [k for k in new if k not in old]
    regressions = [k for k in keys
                   if old.get(k) and new.get(k) and old[k]['ok'] is True and new[k]['ok'] is False]
    fixes = [k for k in keys
             if old.get(k) and new.get(k) and old[k]['ok'] is False and new[k]['ok'] is True]
    still = [k for k in keys
             if old.get(k) and new.get(k) and old[k]['ok'] is False and new[k]['ok'] is False]
    added = [k for k in keys if k not in old]

    def section(title, entries):
        print(f'=== {title} ({len(entries)}) ===')
        for key in entries:
            decision, name = key
            target = (new.get(key) or old[key])['target']
            target = f'[{target[0]:g}, {target[1]:g}]' if isinstance(target, list) else '(information)'
            print(f'  {decision:5s} {name:<58s} {fmt(old.get(key)):>18s} -> {fmt(new.get(key)):<18s} {target}')
        print()

    section('REGRESSIONS: passed before, fails now', regressions)
    section('fixed: failed before, passes now', fixes)
    section('still failing', still)
    section('new checks, not in the old run', added)

    old_pass = sum(1 for c in old.values() if c['ok'] is True)
    new_pass = sum(1 for c in new.values() if c['ok'] is True)
    old_fail = sum(1 for c in old.values() if c['ok'] is False)
    new_fail = sum(1 for c in new.values() if c['ok'] is False)
    print(f'passed {old_pass} -> {new_pass}   failed {old_fail} -> {new_fail}')

    # every informational quantity that moved a lot is worth a look too
    print('\n=== informational values that moved by more than 25% ===')
    for key in keys:
        a, b = old.get(key), new.get(key)
        if not a or not b or a['ok'] is not None or a['value'] in (None, 0) or b['value'] is None:
            continue
        if abs(b['value'] - a['value']) / max(abs(a['value']), 1e-9) > 0.25:
            print(f'  {key[0]:5s} {key[1]:<58s} {a["value"]:>10.3f} -> {b["value"]:<10.3f}')


if __name__ == '__main__':
    main()
