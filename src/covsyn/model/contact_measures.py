# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contact quantities measured on simulated cases, in one place.

The contacts per day before onset were computed by four copies of the same loop (the two
objective implementations, the acceptance checklist and the comparison figure); B55 changes
the definition, so all four now call contacts_per_day_before_onset().
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import optimize, special

MATRIX_KEY = {
    "household": "household_contacts_matrix",
    "school": "school_class_contacts_matrix",
    "workplace": "workplace_contacts_matrix",
    "health_care": "health_care_contacts_matrix",
    "municipality": "municipality_contacts_matrix",
}


def contacts_per_case(contact: Mapping[str, Any]) -> int:
    """Candidate contacts of one case over all five layers, mass events included (B56).

    The quantity compared with Jian et al. 2020's 16.5 close contacts per confirmed case.
    """
    return sum(
        len(contact.get(f"{layer}_effective_contacts") or [])
        for layer in ("household", "school", "workplace", "health_care", "municipality")
    )


def contacts_in_tracing_window(
    course: Mapping[str, Any], contact: Mapping[str, Any], lead_days: int = 2
) -> int:
    """Contacts a contact tracer would list: those met at least once inside the tracing window.

    Jian et al. 2020 traced from 2 days before symptom onset to isolation (finding E7), and
    CovSyn's candidate contacts run from infection, so only this count is comparable with
    Jian's 16.5 close contacts per case (E89). An asymptomatic case has no onset; its window
    starts lead_days before isolation, the day its confirming test is taken (C07).

    Args:
        course: The case's saved course of disease.
        contact: The case's saved contact data.
        lead_days: Days before onset (or isolation) at which the window opens.

    Returns:
        The number of contacts, over all five layers, met on a day in the window.
    """
    isolation = int(course["monitor_isolation_period"])
    onset = course["incubation_period"]
    # Whole days, like the matrix columns (incubation_period is an int in the model today).
    start = (
        int(np.floor(onset)) if onset is not None and np.isfinite(onset) else isolation
    ) - lead_days
    count = 0
    for key in MATRIX_KEY.values():
        matrix = np.asarray(contact.get(key, np.zeros((0, 0))), dtype=bool)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            continue
        days = np.arange(matrix.shape[1])
        window = (days >= start) & (days <= isolation)
        count += int(np.count_nonzero(matrix[:, window].any(axis=1)))
    return count


def ordinary_contact_matrix(contact: Mapping[str, Any], layer: str) -> np.ndarray:
    """The layer's contact matrix without mass-event contacts (B54).

    Args:
        contact: One case's saved contact data.
        layer: One of the keys of MATRIX_KEY.

    Returns:
        The rows of ordinary contacts, as a float matrix (contacts x days).
    """
    matrix = np.asarray(contact[MATRIX_KEY[layer]], dtype=float)
    mask = contact.get("municipality_event_mask") if layer == "municipality" else None
    if mask is not None:
        if len(mask) != matrix.shape[0]:
            raise ValueError(
                f"municipality_event_mask has {len(mask)} entries for "
                f"{matrix.shape[0]} contact rows"
            )
        matrix = matrix[~np.asarray(mask, dtype=bool)]
    return matrix


def contacts_per_day_before_onset(
    course: Mapping[str, Any], contact: Mapping[str, Any], layer: str
) -> float:
    """Ordinary contacts per day before symptom onset, the quantity of the B25 survey target.

    The 2020 national survey records the close contacts of an ordinary day, so a mass event,
    a rare single-day gathering, is not part of it (B55). The window is the days before onset
    (at least one), or the whole contact window for an asymptomatic case.

    Args:
        course: The case's saved course of disease.
        contact: The case's saved contact data.
        layer: One of the keys of MATRIX_KEY.

    Returns:
        Contacts per day; 0 when the layer has no contact window.
    """
    matrix = ordinary_contact_matrix(contact, layer)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return 0.0
    days = _days_before_onset(course, matrix)
    return float(matrix[:, :days].sum() / max(days, 1))


def _days_before_onset(course: Mapping[str, Any], matrix: np.ndarray) -> int:
    """Number of leading matrix columns in the before-onset window (at least one day)."""
    onset = course["incubation_period"]
    return (
        matrix.shape[1]
        if (onset is None or np.isnan(onset))
        else int(min(matrix.shape[1], max(onset, 1)))
    )


def daily_contacts_before_onset(
    course: Mapping[str, Any], contact: Mapping[str, Any], layer: str
) -> np.ndarray:
    """Ordinary contacts on each day before onset: one count per day (B61).

    The same window as contacts_per_day_before_onset(), whose value is the mean of these
    counts, kept per day so the shape can be compared with a contact diary, where one
    person records one day.

    Args:
        course: The case's saved course of disease.
        contact: The case's saved contact data.
        layer: One of the keys of MATRIX_KEY.

    Returns:
        Contacts per day, one entry per day; empty when the layer has no contact window.
    """
    matrix = ordinary_contact_matrix(contact, layer)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros(0)
    days = _days_before_onset(course, matrix)
    return matrix[:, :days].sum(axis=0)


def one_day_counts(course: Mapping[str, Any], contact: Mapping[str, Any], layer: str) -> np.ndarray:
    """daily_contacts_before_onset(), with a case whose layer has no window counted as one day
    without contacts, so that it is not dropped from a pooled one-day distribution (B63)."""
    days = daily_contacts_before_onset(course, contact, layer)
    return days if len(days) else np.zeros(1)


def one_day_shape(per_case_days: Any) -> tuple[float, float]:
    """Zero share and 90th percentile of the pooled one-day counts of many cases (B63).

    A contact diary records one person's one day, so the days before onset of every index case
    are pooled before comparing with Fu 2010. Both statistics depend only on the multiset of
    days, so the two objective implementations, which collect the cases in a different order,
    get the same bits.

    Args:
        per_case_days: One array of one-day counts per case (one_day_counts()).

    Returns:
        (share of days without contacts, 90th percentile), or (nan, nan) with no days.
    """
    days = np.sort(np.concatenate([np.asarray(d, dtype=float) for d in per_case_days])) \
        if len(per_case_days) else np.zeros(0)
    if not len(days):
        return float("nan"), float("nan")
    return float(np.mean(days == 0)), float(np.percentile(days, 90))


# Returned when the counts show no over-dispersion (variance <= mean), i.e. Poisson or tighter.
NO_OVERDISPERSION_K = 100.0


def offspring_dispersion(counts: Any) -> float:
    """Negative-binomial dispersion k of offspring counts, by maximum likelihood (B61).

    Taiwan's target k = 0.29 is a maximum-likelihood estimate, so the objective and the
    acceptance check estimate k the same way instead of by the method of moments they used
    before. The mean's MLE is the sample mean, so only k is searched, on the profile
    likelihood. The counts are sorted first, so the two objective implementations, which
    collect the cases in a different order, get the same bits (E65).

    Args:
        counts: Offspring count of every index case.

    Returns:
        k; NaN when there are no counts or no case infects anyone; NO_OVERDISPERSION_K when
        the variance does not exceed the mean or the likelihood peaks at that ceiling.
    """
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    mean = float(x.mean()) if n else 0.0
    if mean <= 0:
        return float("nan")
    var = float(x.var(ddof=1)) if n > 1 else 0.0
    if not var > mean:
        return NO_OVERDISPERSION_K
    total = float(x.sum())

    def negative_log_likelihood(log_k: float) -> float:
        k = np.exp(log_k)
        return -(
            float(np.sum(special.gammaln(x + k))) - n * special.gammaln(k)
            + n * k * np.log(k / (k + mean)) + total * np.log(mean / (k + mean))
        )

    upper = np.log(NO_OVERDISPERSION_K)
    fit = optimize.minimize_scalar(negative_log_likelihood, bounds=(np.log(1e-4), upper),
                                   method="bounded", options={"xatol": 1e-10})
    k = float(np.exp(fit.x))
    return NO_OVERDISPERSION_K if k >= 0.999 * NO_OVERDISPERSION_K else k
