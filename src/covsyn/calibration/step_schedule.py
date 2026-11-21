# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The firefly step size schedule tied to the planned run length (Task 8).

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 8. The optimiser moves by it and
the pipeline checks its settings by it, so this module imports nothing of the model.
"""

from __future__ import annotations

import numpy as np

# Task 8: the step size schedule's ends until pilot 2 (Task 9) chooses them, the larger of the
# plan's two starting points for alpha_0 (decision of 2026-11-20); G_plan defaults to the run's
# generations.
DEFAULT_ALPHA_0 = 0.2
DEFAULT_ALPHA_END = 0.001
# Each optimiser mode's alpha_0 and generations, which main() runs and the pipeline checks its
# schedule against: an unset alpha_0 is the mode's, and a g_plan longer than its generations is
# warned about. Kept here only, so the command and the pipeline cannot differ; the two tables
# and OPTIMISER_MODES have the same modes (tests/test_task8_pipeline.py).
# Task 9: a pilot run's alpha_0 and generations are the pilot's, given on the command line, so
# its entries are None; a production run's alpha_0 is pilot 2's, given too, and it runs to its
# stopping rule or G_max = 1,000 generations.
MODE_ALPHA_0 = {'train': DEFAULT_ALPHA_0, 'test': DEFAULT_ALPHA_0, 'profile': DEFAULT_ALPHA_0,
                'pilot': None, 'production': None}
# Task 5: a profile runs 2 generations.
MODE_GENERATIONS = {'train': 120, 'test': 10, 'profile': 2, 'pilot': None, 'production': 1000}


class GrowingSchedule(ValueError):
    """alpha_end above alpha_0: the steps would grow, though the schedule shrinks them."""


def step_size(generation: int, alpha_0: float, alpha_end: float, g_plan: int,
              names: tuple[str, str, str] = ('alpha_0', 'alpha_end', 'G_plan')) -> float:
    """The step size of a generation under the Task 8 schedule.

    alpha_0 (alpha_end / alpha_0)^(g / (G_plan - 1)) for g < G_plan, alpha_end after; a
    one-generation plan moves with alpha_0 only at g = 0. It replaces the factor of 0.97 per
    generation, so the steps shrink over the run that was planned, whatever its length.

    The refusals name alpha_0, alpha_end and G_plan as ``names`` does, the caller's names.

    Raises:
        ValueError: A step size that is not positive, alpha_end above alpha_0 (GrowingSchedule),
            which would make the steps grow, or a plan shorter than one generation.
    """
    if not (alpha_0 > 0 and alpha_end > 0 and np.isfinite(alpha_0) and np.isfinite(alpha_end)):
        raise ValueError(f'{names[0]} and {names[1]} must be positive, not {alpha_0!r} and '
                         f'{alpha_end!r}')
    if alpha_end > alpha_0:
        raise GrowingSchedule(f'{names[1]} {alpha_end!r} is above {names[0]} {alpha_0!r}: the '
                              f'steps would grow')
    if isinstance(g_plan, bool) or not isinstance(g_plan, (int, np.integer)) or g_plan < 1:
        raise ValueError(f'{names[2]} must be a whole number of generations, at least 1, '
                         f'not {g_plan!r}')
    if generation >= g_plan:
        return alpha_end
    if g_plan == 1:
        return alpha_0
    return alpha_0 * (alpha_end / alpha_0) ** (generation / (g_plan - 1))


def checked_schedule(alpha_0: float | None, alpha_end: float | None, g_plan: int | None,
                     generations: int,
                     names: tuple[str, str, str] = ('alpha_0', 'alpha_end', 'g_plan'),
                     alpha_0_default: float | None = DEFAULT_ALPHA_0
                     ) -> tuple[float, float, int]:
    """The schedule a run moves on, each unset end at its default and G_plan at generations.

    The command line and the pipeline both check theirs here, so they refuse the same ones.

    Args:
        alpha_0: As given; None when not given.
        alpha_end: As given; None when not given.
        g_plan: As given; None when not given.
        generations: The run's generations, G_plan when none is given.
        names: How the caller names alpha_0, alpha_end and g_plan, for the refusal.
        alpha_0_default: alpha_0 when none is given, the mode's (MODE_ALPHA_0); None when the
            mode has none, so alpha_0 must be given.

    Raises:
        ValueError: As step_size; for steps that would grow, naming an end left at its default
            and the range the other must be in. G_plan has no upper bound: it takes no seed
            blocks, and the optimiser warns when it outlasts the run (decision of 2026-11-20).
    """
    if alpha_0 is None and alpha_0_default is None:
        raise ValueError(f'{names[0]} must be given: this mode has no default')
    schedule = (alpha_0_default if alpha_0 is None else alpha_0,
                DEFAULT_ALPHA_END if alpha_end is None else alpha_end,
                generations if g_plan is None else g_plan)
    try:
        step_size(0, *schedule, names=names)
    except GrowingSchedule as error:
        message = str(error)
        if alpha_0 is None:
            message += (f'; {names[0]} was left at its default, so give {names[0]} too, at '
                        f'least {names[1]}')
        elif alpha_end is None:
            message += (f'; {names[1]} was left at its default, so give {names[1]} too, at '
                        f'most {names[0]}')
        raise ValueError(message) from None
    return schedule


def plan_outlasts_run(g_plan: int, generations: int, name: str = 'G_plan') -> str | None:
    """Why to warn of a plan longer than its run, or None; the optimiser and the pipeline's rule.

    G_plan takes no seed blocks, so it has no cap, but the steps of a run that ends first stay
    above alpha_end, as a typed extra zero would make them (decision of 2026-11-20). A run of
    no generations takes no step, so it gives no warning.
    """
    if 1 <= generations < g_plan:
        return (f"{name} {g_plan} is longer than the run's {generations} generations, so the "
                f'steps do not reach alpha_end')
    return None
