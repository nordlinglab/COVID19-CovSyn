# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whether an employed case is at work on a given day (B75).

The professor (2026-10-27) asked the workplace layer to first draw whether a person is at work
today, with a probability that depends on the day of the week and on national holidays, and
only then to draw coworker contacts. A case off work meets no coworker that day.

Probabilities (user decisions, 2026-10-31):

* Monday to Friday: 69.1% (bootstrap 95% 65.3-72.7%), the share of Fu 2010's employed weekday
  diaries (jobs 3-6, n = 616) with any work contact, weighted to the 2010 population with the
  five weekdays weighted equally (covid19-research scripts/summarize_taiwan_contact_data.py
  attendance_by_day()). It counts leave, shifts and misfiled diaries as days off.
* Saturday: 1.99%, the Ministry of Labor Occupational Wage Survey of July 2020 (109), industry
  and services, by number of workers: six-day weeks 1.46% plus half of the alternating weeks
  1.06% (covid19-research data/external/mol_working_days_per_week/). Fu 2010's Saturday, 27.7%,
  agrees with the 2010 survey (28.6%), but Saturday work fell after the 2016 labour law change.
* Sunday and national holidays: 0. None of the survey's schedules includes Sunday; Fu 2010's
  Sunday (67.8%, as high as a weekday) contradicts every other source and is not used.

The holiday table covers 2020-01-01 to 2020-05-31, every day the first-wave scenarios need;
later dates fall back to the day of the week, as the make-up working days of June and September
2020 could not be confirmed from an official source.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

WEEKDAY_AT_WORK = 0.691
SATURDAY_AT_WORK = 0.0199
SUNDAY_AT_WORK = 0.0

LAST_TABLED_DAY = date(2020, 5, 31)


def _days(first: date, last: date) -> frozenset[date]:
    return frozenset(first + timedelta(days=k) for k in range((last - first).days + 1))


# Days off of 2020 (Ministry of the Interior, 109 年放假日, 2019-12-02,
# https://www.ris.gov.tw/info-liferay/app/channel/newsDetail/1575286868351; Directorate-General
# of Personnel Administration: Jan 23 adjusted off and made up on Feb 15, Apr 2 observed for
# Children's Day and Apr 3 for Tomb Sweeping Day), plus New Year and Labour Day, days off for
# workers by the Enforcement Rules of the Labor Standards Act, article 23.
HOLIDAYS_2020 = (
    _days(date(2020, 1, 1), date(2020, 1, 1))
    | _days(date(2020, 1, 23), date(2020, 1, 29))     # Lunar New Year
    | _days(date(2020, 2, 28), date(2020, 3, 1))      # Peace Memorial Day
    | _days(date(2020, 4, 2), date(2020, 4, 5))       # Children's Day and Tomb Sweeping Day
    | _days(date(2020, 5, 1), date(2020, 5, 1))       # Labour Day
)
MAKE_UP_WORKDAYS_2020 = frozenset({date(2020, 2, 15)})

# A Monday; the weekday-only calendar starts at one of the seven days after it.
REFERENCE_MONDAY = date(2020, 3, 2)


def attendance_probability(day: date, holidays: bool = True) -> float:
    """Probability that an employed person is at work on this date.

    Args:
        day: The calendar date.
        holidays: Apply the 2020 holidays and make-up working days; False uses the day of the
            week only.

    Returns:
        The probability of being at work.
    """
    if holidays and day in MAKE_UP_WORKDAYS_2020:
        return WEEKDAY_AT_WORK
    if holidays and day in HOLIDAYS_2020:
        return SUNDAY_AT_WORK
    return {5: SATURDAY_AT_WORK, 6: SUNDAY_AT_WORK}.get(day.weekday(), WEEKDAY_AT_WORK)


@dataclass
class WorkCalendar:
    """The calendar of one simulation run: simulation day 0 falls on day_zero.

    Attributes:
        day_zero: Calendar date of simulation day 0.
        rng: Generator of the attendance draws, separate from the simulation's random stream.
        holidays: Apply the 2020 holidays (dated scenarios) or not (weekday-only calendar).
    """

    day_zero: date
    rng: np.random.Generator
    holidays: bool = True

    def at_work(self, first_day: int, n_days: int) -> np.ndarray:
        """Draw, independently for each day, whether the case is at work.

        Args:
            first_day: Simulation day of the first column (the case's infection day).
            n_days: Number of consecutive days.

        Returns:
            Boolean array of length n_days.

        Raises:
            ValueError: If n_days is negative.
        """
        if n_days < 0:
            raise ValueError(f"n_days must be non-negative, got {n_days}")
        start = self.day_zero + timedelta(days=int(first_day))
        p = np.array([attendance_probability(start + timedelta(days=k), self.holidays)
                      for k in range(n_days)], dtype=float)
        return self.rng.random(n_days) < p


def weekday_calendar(rng: np.random.Generator) -> WorkCalendar:
    """Calendar of an undated run (calibration): day 0 on a random weekday, no holidays."""
    return WorkCalendar(REFERENCE_MONDAY + timedelta(days=int(rng.integers(7))), rng,
                        holidays=False)


def first_wave_day_zero(seed_path: str) -> date:
    """Calendar date of simulation day 0 of a first-wave seed table: the date of the seeds
    whose infection_day is 0 (build_first_wave_seeds counts days from the earliest seed)."""
    with open(seed_path, newline='') as f:
        for row in csv.DictReader(f):
            if int(row['infection_day']) == 0:
                return date.fromisoformat(row['infection_date'])
    raise ValueError(f"{seed_path} has no seed on day 0")
