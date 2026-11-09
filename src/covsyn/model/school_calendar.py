# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whether school is open on a given day (B77).

The professor (2026-10-27): from primary to senior high school the whole class attends on a
school day and nobody on a day without school; sick leave is small enough to ignore. Without
data for universities, they are treated the same way. So school contacts happen only on days
the school is open, with no attendance draw.

Dated calendar (first-wave scenarios, day 0 in January 2020):

* Autumn term 108-1 until 2020-01-20; winter vacation from 2020-01-21 (Regulations on School
  Years, Terms and Vacations, as reported by TVBS 2019-12, https://news.tvbs.com.tw/life/1251095;
  senior high schools ended on Jan 16 with two Saturday make-up days, a difference of two school
  days that is ignored).
* Spring term 108-2 from 2020-02-25 instead of Feb 11, and the Feb 15 make-up day cancelled
  (Central Epidemic Command Center, 2020-02-02; Ministry of Education: the term closes on
  2020-07-14 instead of Jun 30; e.g. https://www.mirrormedia.mg/story/amp/20200215soc007).
* Summer vacation 2020-07-15 to 2020-08-30; autumn term 109-1 from Monday 2020-08-31 (school
  calendars of the 109 academic year, e.g. https://www.tshs.tp.edu.tw/download/1091calender.pdf).
* Weekends and the national holidays of work_calendar.HOLIDAYS_2020 are closed, except Labour
  Day (May 1), a day off for workers under the Labor Standards Act but a school day.

Undated runs (calibration) use the day of the week only: closed on Saturday and Sunday.
The school contact target (daily_school, B25/B61) is per person over all days of Fu 2010's
diaries, weekends included, so it stays an average-day target under this calendar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from covsyn.model.work_calendar import HOLIDAYS_2020

TERMS = ((date(2019, 8, 30), date(2020, 1, 20)),
         (date(2020, 2, 25), date(2020, 7, 14)),
         (date(2020, 8, 31), date(2021, 1, 19)))
SCHOOL_HOLIDAYS = HOLIDAYS_2020 - {date(2020, 5, 1)}
# Last date the dated calendar is meant for: day 364 of a first-wave run from 2020-01-09.
DATED_UNTIL = date(2021, 1, 8)


def school_open(day: date, dated: bool = True) -> bool:
    """Whether school is open on this date.

    Args:
        day: The calendar date.
        dated: Apply the 2020 terms, vacations and holidays; False uses the day of the week
            only (and so does a date after DATED_UNTIL).

    Returns:
        True on a school day.
    """
    if day.weekday() >= 5:
        return False
    if not dated or day > DATED_UNTIL:
        return True
    if day in SCHOOL_HOLIDAYS:
        return False
    return any(first <= day <= last for first, last in TERMS)


@dataclass
class SchoolCalendar:
    """The school calendar of one simulation run: simulation day 0 falls on day_zero.

    Attributes:
        day_zero: Calendar date of simulation day 0 (the same as the run's work calendar).
        dated: Dated 2020 calendar (first-wave scenarios) or weekends only (undated runs).
    """

    day_zero: date
    dated: bool = True
    _open: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), repr=False)

    def open_days(self, first_day: int, n_days: int) -> np.ndarray:
        """Whether school is open on n_days days from simulation day first_day.

        Raises:
            ValueError: If first_day or n_days is negative.
        """
        if n_days < 0:
            raise ValueError(f"n_days must be non-negative, got {n_days}")
        if first_day < 0:
            raise ValueError(f"first_day must be non-negative, got {first_day}")
        end = int(first_day) + n_days
        if end > len(self._open):
            self._open = np.array([school_open(self.day_zero + timedelta(days=k), self.dated)
                                   for k in range(max(end, 2 * len(self._open)))], dtype=bool)
        return self._open[int(first_day):end]
