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

MATRIX_KEY = {'household': 'household_contacts_matrix',
              'school': 'school_class_contacts_matrix',
              'workplace': 'workplace_contacts_matrix',
              'health_care': 'health_care_contacts_matrix',
              'municipality': 'municipality_contacts_matrix'}


def contacts_per_case(contact: Mapping[str, Any]) -> int:
    """Candidate contacts of one case over all five layers, mass events included (B56).

    The quantity compared with Jian et al. 2020's 16.5 close contacts per confirmed case.
    """
    return sum(len(contact.get(f'{layer}_effective_contacts') or [])
               for layer in ('household', 'school', 'workplace', 'health_care', 'municipality'))


def contacts_in_tracing_window(course: Mapping[str, Any], contact: Mapping[str, Any],
                               lead_days: int = 2) -> int:
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
    isolation = int(course['monitor_isolation_period'])
    onset = course['incubation_period']
    # Whole days, like the matrix columns (incubation_period is an int in the model today).
    start = (int(np.floor(onset)) if onset is not None and np.isfinite(onset) else isolation) - lead_days
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
    mask = contact.get('municipality_event_mask') if layer == 'municipality' else None
    if mask is not None:
        if len(mask) != matrix.shape[0]:
            raise ValueError(f'municipality_event_mask has {len(mask)} entries for '
                             f'{matrix.shape[0]} contact rows')
        matrix = matrix[~np.asarray(mask, dtype=bool)]
    return matrix


def contacts_per_day_before_onset(course: Mapping[str, Any], contact: Mapping[str, Any],
                                  layer: str) -> float:
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
    onset = course['incubation_period']
    days = matrix.shape[1] if (onset is None or np.isnan(onset)) \
        else int(min(matrix.shape[1], max(onset, 1)))
    return float(matrix[:, :days].sum() / max(days, 1))
