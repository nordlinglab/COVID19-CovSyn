# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decision B55: how mass-event contacts enter the calibration.

1. The event probability P[199] is locked at 0.10, where CovSyn's Cheng 'others' contacts per
   100 cases match Cheng et al. (2020) (finding E85).
2. In the Cheng contact fit, the sampled event contacts are replaced by their expectation,
   P[199] x E[event size] spread over the event-day weights, so a single event of hundreds of
   people no longer dominates a 100-case objective; infections stay as sampled.
3. Contacts per day before onset (the survey target) count ordinary contacts only.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from covsyn.model import contact_measures as cm
from covsyn.model import data_synthesize as ds

EVENT = {'probability': 0.1, 'exponent': 1.49, 'min_size': 21, 'max_size': 1000,
         'risk_ratio': 1.0}


# --- expected event size and day weights ----------------------------------------------

def test_expected_event_size_is_the_power_law_mean() -> None:
    sizes = np.arange(21, 1001)
    weights = sizes ** -1.49
    expected = np.sum(sizes * weights) / np.sum(weights)
    assert ds.expected_event_size(1.49, 21, 1000, room=10**7) == pytest.approx(expected)


def test_expected_event_size_respects_the_room_left() -> None:
    assert ds.expected_event_size(1.49, 21, 1000, room=30) < 30
    assert ds.expected_event_size(1.49, 21, 1000, room=0) == 0.0


def test_expected_event_size_matches_draws() -> None:
    np.random.seed(1)
    draws = [min(ds.draw_event_size(1.49, 21, 1000), 200) for _ in range(100000)]
    assert np.mean(draws) == pytest.approx(ds.expected_event_size(1.49, 21, 1000, 200), rel=0.02)


def _owner(onset: float = np.nan) -> SimpleNamespace:
    course = SimpleNamespace(incubation_period=onset)
    owner = SimpleNamespace(course_of_disease_data_object=course,
                            generate_logistic_contact_p=ds.Draw_contact_data.generate_logistic_contact_p)
    owner.daily_contact_p = ds.Draw_contact_data.daily_contact_p.__get__(owner)
    return owner


def test_event_day_weights_are_the_normalised_daily_profile() -> None:
    weights = ds.Draw_contact_data.event_day_weights(_owner(), [5, 0.7, 0.04, 0.04], 1, 0, 1, 4)
    np.testing.assert_allclose(weights, np.full(5, 0.2))


def test_event_day_weights_fall_back_to_uniform_without_contact() -> None:
    weights = ds.Draw_contact_data.event_day_weights(_owner(), [5, 0.7, 0.0, 0.0], 1, 0, 1, 3)
    np.testing.assert_allclose(weights, np.full(4, 0.25))


# --- the simulation records the expectation -------------------------------------------

def _index_contact(vector: np.ndarray, demographic_parameters: object, seed: int) -> tuple:
    from covsyn.model.data_synthesis_main import run_covid

    _, _, courses, contacts = run_covid(seed, vector.copy(), copy.deepcopy(demographic_parameters),
                                        save_file=False, mode='result')
    return courses[0], contacts[0]


@pytest.mark.parametrize("seed", range(4))
def test_expected_event_contacts_are_saved_per_day(
        seed: int, run10_vector: np.ndarray, demographic_parameters: object) -> None:
    vector = np.concatenate([run10_vector, [0.10, 1.49, 21, 1000, 1.0]])
    _, contact = _index_contact(vector, demographic_parameters, seed)
    expected = contact['municipality_event_expected_contacts']
    assert len(expected) == contact['municipality_contacts_matrix'].shape[1]
    assert np.sum(expected) == pytest.approx(0.10 * ds.expected_event_size(1.49, 21, 1000, 10**7),
                                             abs=len(expected) * 2.0 ** -21)
    # On the 2**-20 grid, so sums over cases are exact in any order (E65).
    np.testing.assert_array_equal(expected * 2.0 ** 20, np.round(expected * 2.0 ** 20))


def test_vector_without_events_saves_no_expectation(run10_vector: np.ndarray,
                                                    demographic_parameters: object) -> None:
    _, contact = _index_contact(run10_vector, demographic_parameters, 0)
    assert 'municipality_event_expected_contacts' not in contact


# --- Cheng bins with expected event contacts ------------------------------------------

def _case(onset: float = 2.0) -> tuple[dict, dict]:
    """Two ordinary municipality contacts (days 0 and 3) and three event contacts on day 5."""
    matrix = np.zeros((5, 8), dtype=bool)
    matrix[0, 0] = matrix[1, 3] = True
    matrix[2:, 5] = True
    empty = np.zeros((0, 8), dtype=bool)
    contact = {f'{k}_contacts_matrix': empty for k in ('household', 'school_class', 'workplace',
                                                        'health_care')}
    for k in ('household', 'school', 'workplace', 'health_care'):
        contact[f'{k}_effective_contacts_infection_time'] = []
    contact['municipality_contacts_matrix'] = matrix
    contact['municipality_effective_contacts_infection_time'] = [np.nan, 4.0, 5.0, np.nan, np.nan]
    contact['municipality_event_mask'] = np.array([False, False, True, True, True])
    contact['municipality_event_expected_contacts'] = np.array(
        [0, 0, 0, 0, 0, 0, 0, 10.0])          # all expected on day 7, onset + 5
    course = {'incubation_period': onset}
    return course, contact


def test_default_binning_counts_the_sampled_event_contacts() -> None:
    from covsyn.figures.plot_results import create_array_cheng2020_fig2

    course, contact = _case()
    _, contacts, _, infections = create_array_cheng2020_fig2([course], [contact], 'Municipality')
    # days relative to onset: -2, +1, +3 (x3) -> bins <0: 1, 0-3: 4
    np.testing.assert_array_equal(contacts, [1, 4, 0, 0, 0, 0])
    np.testing.assert_array_equal(infections, [0, 2, 0, 0, 0, 0])


def test_expected_binning_replaces_event_contacts_and_keeps_infections() -> None:
    from covsyn.figures.plot_results import create_array_cheng2020_fig2

    course, contact = _case()
    _, contacts, _, infections = create_array_cheng2020_fig2([course], [contact], 'Municipality',
                                                           expected_events=True)
    # ordinary: -2 -> <0, +1 -> 0-3; expected 10 on day 7 = onset + 5 -> bin 4-5
    np.testing.assert_allclose(contacts, [1, 1, 10, 0, 0, 0])
    np.testing.assert_array_equal(infections, [0, 2, 0, 0, 0, 0])


def test_expected_binning_skips_asymptomatic_cases() -> None:
    from covsyn.figures.plot_results import create_array_cheng2020_fig2

    course, contact = _case(onset=np.nan)
    _, contacts, _, _ = create_array_cheng2020_fig2([course], [contact], 'Municipality',
                                                    expected_events=True)
    assert np.sum(contacts) == 0


def test_expected_binning_without_events_equals_the_default() -> None:
    from covsyn.figures.plot_results import create_array_cheng2020_fig2

    course, contact = _case()
    del contact['municipality_event_mask'], contact['municipality_event_expected_contacts']
    a = create_array_cheng2020_fig2([course], [contact], 'Municipality')
    b = create_array_cheng2020_fig2([course], [contact], 'Municipality', expected_events=True)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[3], b[3])


# --- contacts per day before onset ----------------------------------------------------

def test_contacts_per_day_before_onset_ignore_event_contacts() -> None:
    course, contact = _case(onset=4.0)
    # ordinary rows: day 0 and day 3, both before onset (days 0..3): 2 contacts over 4 days
    assert cm.contacts_per_day_before_onset(course, contact, 'municipality') == pytest.approx(0.5)


def test_contacts_per_day_before_onset_unchanged_for_other_layers() -> None:
    course, contact = _case(onset=4.0)
    contact['household_contacts_matrix'] = np.ones((2, 8), dtype=bool)
    assert cm.contacts_per_day_before_onset(course, contact, 'household') == pytest.approx(2.0)


def test_contacts_per_day_of_an_asymptomatic_case_use_the_whole_window() -> None:
    course, contact = _case(onset=np.nan)
    assert cm.contacts_per_day_before_onset(course, contact, 'municipality') == pytest.approx(2 / 8)


def test_contacts_per_day_with_no_contacts_is_zero() -> None:
    course, contact = _case()
    assert cm.contacts_per_day_before_onset(course, contact, 'school') == 0.0


# --- search bounds and the physiology penalty -----------------------------------------

def test_event_probability_is_locked_at_the_cheng_match() -> None:
    from covsyn.calibration import apply_phase_d_parameters as apply

    assert apply.COMMUNITY_EVENT[0] == (0.10, 0.10, 0.10)
    course_upper = np.load('variable/course_parameters_ub.npy')
    course_lower = np.load('variable/course_parameters_lb.npy')
    assert course_lower[apply.COMMUNITY_EVENT_COURSE_INDEX] == 0.10
    assert course_upper[apply.COMMUNITY_EVENT_COURSE_INDEX] == 0.10


def test_infectious_period_uses_the_reported_mean_range() -> None:
    """B5: the physiology penalty uses the literature reported-mean range, 3.45-20 days."""
    pytest.importorskip('sklearn')
    from covsyn.calibration import firefly_optimizer as fo

    P = np.ones(204)
    P[37], P[38] = 4.5, 1.0        # latent 4.5, inside [4.1, 5.5]
    P[41], P[42] = 2.0, 1.0        # pre-onset window 2, inside [1, 3]
    P[43], P[44], P[45] = 2.0, 2.0, 1.0   # onset to confirmation 5
    for infectious in (3.5, 4.22, 12.0, 19.9):
        P[39], P[40] = infectious, 1.0
        assert fo.physiology_penalty(P) == 0.0
    P[39] = 3.0
    assert fo.physiology_penalty(P) > 0.0
