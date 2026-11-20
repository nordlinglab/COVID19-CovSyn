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


def step_size(generation: int, alpha_0: float, alpha_end: float, g_plan: int) -> float:
    """The step size of a generation under the Task 8 schedule.

    alpha_0 (alpha_end / alpha_0)^(g / (G_plan - 1)) for g < G_plan, alpha_end after; a
    one-generation plan moves with alpha_0 only at g = 0. It replaces the factor of 0.97 per
    generation, so the steps shrink over the run that was planned, whatever its length.

    Raises:
        ValueError: A step size that is not positive, alpha_end above alpha_0, which would make
            the steps grow, or a plan shorter than one generation.
    """
    if not (alpha_0 > 0 and alpha_end > 0 and np.isfinite(alpha_0) and np.isfinite(alpha_end)):
        raise ValueError(f'step sizes must be positive, not {alpha_0!r} and {alpha_end!r}')
    if alpha_end > alpha_0:
        raise ValueError(f'alpha_end {alpha_end!r} is above alpha_0 {alpha_0!r}: the steps '
                         f'would grow')
    if isinstance(g_plan, bool) or not isinstance(g_plan, (int, np.integer)) or g_plan < 1:
        raise ValueError(f'G_plan must be a whole number of generations, at least 1, '
                         f'not {g_plan!r}')
    if generation >= g_plan:
        return alpha_end
    if g_plan == 1:
        return alpha_0
    return alpha_0 * (alpha_end / alpha_0) ** (generation / (g_plan - 1))


def checked_schedule(alpha_0: float | None, alpha_end: float | None, g_plan: int | None,
                     generations: int, names: tuple[str, str] = ('alpha_0', 'alpha_end'),
                     alpha_0_default: float = DEFAULT_ALPHA_0) -> tuple[float, float, int]:
    """The schedule a run moves on, each unset end at its default and G_plan at generations.

    The command line and the pipeline both check theirs here, so they refuse the same ones.

    Args:
        alpha_0: As given; None when not given.
        alpha_end: As given; None when not given.
        g_plan: As given; None when not given.
        generations: The run's generations, G_plan when none is given.
        names: How the caller names alpha_0 and alpha_end, for the refusal.
        alpha_0_default: alpha_0 when none is given, the mode's.

    Raises:
        ValueError: As step_size; when one end was left at its default, it says so and names it.
    """
    schedule = (alpha_0_default if alpha_0 is None else alpha_0,
                DEFAULT_ALPHA_END if alpha_end is None else alpha_end,
                generations if g_plan is None else g_plan)
    try:
        step_size(0, *schedule)
    except ValueError as error:
        if (alpha_0 is None) != (alpha_end is None) and schedule[1] > schedule[0]:
            unset = names[0] if alpha_0 is None else names[1]
            raise ValueError(f'{error}; {unset} was left at its default, so give it too') from None
        raise
    return schedule
