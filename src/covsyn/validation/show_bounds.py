"""Print the layer attack-rate bounds of variable/course_parameters*.npy, daily and cumulative.

The cumulative rate over a contact's days is 1 - (1 - daily)^days, with the days the bounds were
built with: variable/attack_rate_contact_days.json, which apply_phase_d_parameters writes next to
the bounds, or sar_anchors.MEAN_CONTACT_DAYS for bounds built before it did (review of B76). It used to carry its own days {2.00, 2.95, 4.02, 1.99, 3.91}, the legacy
constants, so its cumulative percentages disagreed with the bounds it printed.
"""
import json
from pathlib import Path

import numpy as np

from covsyn.calibration.sar_anchors import ATTACK_RATE_SLICE, MEAN_CONTACT_DAYS

DAYS_FILE = 'attack_rate_contact_days.json'


def bounds_contact_days(directory: Path | str) -> tuple[dict[str, float], str]:
    """The contact days the bounds in `directory` were built with.

    apply_phase_d_parameters writes them next to the bounds since the review of B76; bounds
    built before that fall back to the current sar_anchors.MEAN_CONTACT_DAYS.

    Args:
        directory: The variable/ directory holding the bounds.

    Returns:
        (days per layer, where they were measured).
    """
    path = Path(directory) / DAYS_FILE
    if not path.exists():
        return MEAN_CONTACT_DAYS, 'not recorded with these bounds; current MEAN_CONTACT_DAYS'
    data = json.loads(path.read_text())
    return data['days'], data['measured_on']


def attack_rate_rows(p: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                     days: dict[str, float] | None = None
                     ) -> list[tuple[str, float, float, float, float, float, float]]:
    """Daily and cumulative attack rates per layer, block means of the three vectors.

    Args:
        p: Course parameter vector.
        lb: Its lower bounds.
        ub: Its upper bounds.
        days: Contact days per layer for the conversion; MEAN_CONTACT_DAYS when None.

    Returns:
        (layer, daily, lb, ub, cumulative %, lb %, ub %) for every layer.
    """
    rows = []
    for layer, (a, b) in ATTACK_RATE_SLICE.items():
        d, dl, du = (float(np.mean(x[a:b])) for x in (p, lb, ub))
        n_days = (MEAN_CONTACT_DAYS if days is None else days)[layer]
        f = lambda x: 100 * (1 - (1 - x) ** n_days)  # noqa: E731
        rows.append((layer, d, dl, du, f(d), f(dl), f(du)))
    return rows


def main(directory: Path | str = './variable') -> None:
    """Print the bounds in `directory` with the contact days they were built with.

    Args:
        directory: A variable/ directory, the working one by default or a run's archived copy.
    """
    directory = Path(directory)
    p = np.load(directory / 'course_parameters.npy')
    lb = np.load(directory / 'course_parameters_lb.npy')
    ub = np.load(directory / 'course_parameters_ub.npy')
    days, measured_on = bounds_contact_days(directory)
    print('contact days: %s' % measured_on)
    print('%-13s %10s %-22s %10s %s' % ('layer', 'daily', '[lb, ub]', 'cumul %', '[lb, ub]'))
    for row in attack_rate_rows(p, lb, ub, days):
        print('%-13s %10.5f [%.5f, %.5f] %9.2f  [%.2f, %.2f]' % row)
    print('\nage risk ratios %s  locked=%s' % (np.round(p[26:30], 3).tolist(), bool(np.all(lb[26:30] == ub[26:30]))))
    print('latent mean %.2f [%.2f, %.2f]' % (p[0] * p[1], lb[0] * lb[1], ub[0] * ub[1]))


def cli() -> None:
    """Print the bounds from the command line (``covsyn-show-bounds [directory]``)."""
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else './variable')


if __name__ == '__main__':
    cli()
