"""Prove the E71 fix: both copies of firefly_optimizer share one LAST_COST_PARTS.

Reproduces the exact situation that lost run 7's diagnostics -- firefly_optimizer.py loaded twice,
once as the running script and once by name -- and checks that a write through either copy is
visible in the other. Before the fix the second copy had its own empty dict, so the running
script's progress_metrics.csv came out with 34 columns instead of 73.
"""
import importlib.util
import sys

import cost_parts
import firefly_optimizer


def load_second_copy():
    """Load firefly_optimizer.py again under a different name, as running it as a script does."""
    spec = importlib.util.spec_from_file_location('firefly_optimizer_second_copy',
                                                 'firefly_optimizer.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['firefly_optimizer_second_copy'] = module
    spec.loader.exec_module(module)
    return module


def main():
    second = load_second_copy()

    print('two distinct module objects        : %s'
          % (second is not firefly_optimizer))
    assert second is not firefly_optimizer, 'the second load did not produce a separate module'

    same = (firefly_optimizer.LAST_COST_PARTS is cost_parts.LAST
            and second.LAST_COST_PARTS is cost_parts.LAST)
    print('both bind the same dict object      : %s' % same)
    assert same, 'the two copies do NOT share LAST_COST_PARTS -- E71 is not fixed'

    # A write through one copy must be visible through the other, which is what the firefly loop
    # depends on: fast_cost writes, the running script reads.
    firefly_optimizer.LAST_COST_PARTS.clear()
    firefly_optimizer.LAST_COST_PARTS.update(cost_contact=1.25, measured_offspring_k=0.3)
    seen = dict(second.LAST_COST_PARTS)
    print('write through copy A, read from B   : %s' % seen)
    assert seen == {'cost_contact': 1.25, 'measured_offspring_k': 0.3}, 'the write was not shared'

    second.LAST_COST_PARTS.clear()
    print('clear through copy B, copy A sees   : %s' % dict(firefly_optimizer.LAST_COST_PARTS))
    assert not firefly_optimizer.LAST_COST_PARTS, 'the clear was not shared'

    # The constants really are identical in both copies, which is why run 7's FIT was unaffected
    # even though its logging was not.
    for name in ('SIMULATIONS_PER_EVALUATION', 'SIMULATIONS_PER_TASK', 'ATTACK_RATE_WEIGHT',
                 'OUTCOME_PENALTY_WEIGHT', 'PHYSIOLOGY_PENALTY_WEIGHT',
                 'MAX_INTERVAL_WIDTH_OVER_CENTRE'):
        a, b = getattr(firefly_optimizer, name), getattr(second, name)
        assert a == b, '%s differs between the copies: %r vs %r' % (name, a, b)
    assert firefly_optimizer.OUTCOME_TARGETS == second.OUTCOME_TARGETS, 'OUTCOME_TARGETS differ'
    print('every constant identical in both    : yes (why run 7 fitted correctly regardless)')

    print('\nE71 FIXED -- the diagnostics will be logged on the next run')


if __name__ == '__main__':
    main()
