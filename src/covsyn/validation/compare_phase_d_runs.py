"""Compare two Phase D verification runs check by check.

A rerun that fixes three things can quietly break a fourth. This prints the checklist of
both runs side by side and, first of all, the regressions: the checks that passed before the
change and fail after it. Those are the ones worth arguing about; everything else is either
an intended improvement or unchanged.

Usage: python -m covsyn.validation.compare_phase_d_runs old_checks.json new_checks.json
"""
import json
import sys


def load(path):
    data = json.load(open(path))
    return data, {(c['decision'], c['name']): c for c in data['checks']}


def status(check):
    """'ok', 'within' (passed only within its 95% interval, B57), 'fail', or None."""
    if check is None or check['ok'] is None:
        return None
    if not check['ok']:
        return 'fail'
    return 'within' if check.get('within_noise') else 'ok'


def fmt(check):
    if check is None:
        return 'not measured'
    value = check['value']
    text = 'n/a' if value is None else f'{value:.3f}'
    label = {None: '(info)', 'ok': 'ok', 'within': 'ok~', 'fail': 'FAIL'}[status(check)]
    return f'{text} {label}'


def main():
    old_path, new_path = sys.argv[1], sys.argv[2]
    old_data, old = load(old_path)
    new_data, new = load(new_path)
    print(f'old: {old_path}  ({old_data["simulations"]} simulations, {old_data["cases"]} cases)')
    print(f'new: {new_path}  ({new_data["simulations"]} simulations, {new_data["cases"]} cases)\n')

    keys = list(old) + [k for k in new if k not in old]

    def moved(before, after):
        return [k for k in keys if old.get(k) and new.get(k)
                and status(old[k]) in before and status(new[k]) in after]

    regressions = moved({'ok', 'within'}, {'fail'})
    fixes = moved({'fail', 'within'}, {'ok'})
    noise_only = moved({'fail'}, {'within'})
    softened = moved({'ok'}, {'within'})
    still = moved({'fail'}, {'fail'})
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
    section('failed before, now passes only within noise (ok~, B57)', noise_only)
    section('passed before, now only within noise (ok~, B57)', softened)
    section('still failing', still)
    section('new checks, not in the old run', added)

    def count(checks, wanted):
        return sum(1 for c in checks.values() if status(c) == wanted)

    print(f'passed {count(old, "ok")} -> {count(new, "ok")}   '
          f'within noise (ok~) {count(old, "within")} -> {count(new, "within")}   '
          f'failed {count(old, "fail")} -> {count(new, "fail")}')

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
