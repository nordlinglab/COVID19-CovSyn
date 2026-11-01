"""Print the layer attack-rate bounds of variable/course_parameters*.npy, daily and cumulative.

The cumulative rate over a contact's days is 1 - (1 - daily)^days, with the days measured on the
last run (sar_anchors.MEAN_CONTACT_DAYS), the same conversion apply_phase_d_parameters uses to
build the bounds. It used to carry its own days {2.00, 2.95, 4.02, 1.99, 3.91}, the legacy
constants, so its cumulative percentages disagreed with the bounds it printed.
"""
import numpy as np

from covsyn.calibration.sar_anchors import ATTACK_RATE_SLICE, MEAN_CONTACT_DAYS


def attack_rate_rows(p: np.ndarray, lb: np.ndarray,
                     ub: np.ndarray) -> list[tuple[str, float, float, float, float, float, float]]:
    """Daily and cumulative attack rates per layer, block means of the three vectors.

    Args:
        p: Course parameter vector.
        lb: Its lower bounds.
        ub: Its upper bounds.

    Returns:
        (layer, daily, lb, ub, cumulative %, lb %, ub %) for every layer.
    """
    rows = []
    for layer, (a, b) in ATTACK_RATE_SLICE.items():
        d, dl, du = (float(np.mean(x[a:b])) for x in (p, lb, ub))
        days = MEAN_CONTACT_DAYS[layer]
        f = lambda x: 100 * (1 - (1 - x) ** days)  # noqa: E731
        rows.append((layer, d, dl, du, f(d), f(dl), f(du)))
    return rows


def main() -> None:
    """Print the bounds of variable/course_parameters*.npy in the working directory."""
    p = np.load('./variable/course_parameters.npy')
    lb = np.load('./variable/course_parameters_lb.npy')
    ub = np.load('./variable/course_parameters_ub.npy')
    print('%-13s %10s %-22s %10s %s' % ('layer', 'daily', '[lb, ub]', 'cumul %', '[lb, ub]'))
    for row in attack_rate_rows(p, lb, ub):
        print('%-13s %10.5f [%.5f, %.5f] %9.2f  [%.2f, %.2f]' % row)
    print('\nage risk ratios %s  locked=%s' % (np.round(p[26:30], 3).tolist(), bool(np.all(lb[26:30] == ub[26:30]))))
    print('latent mean %.2f [%.2f, %.2f]' % (p[0] * p[1], lb[0] * lb[1], ub[0] * ub[1]))


if __name__ == '__main__':
    main()
