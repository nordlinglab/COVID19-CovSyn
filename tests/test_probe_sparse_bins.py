# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The sparse-bin probe: each Cheng bin's share of the attack-rate cost, and pilot 1 without one.

Pilot 1's outlier (run 19, S = 10,000, block 23, cost 34.19) came from one bin, household
contacts 0 to 3 days after onset: 1 of its 2 contacts infected added 20.49 to the cost. The
probe splits the attack-rate term per bin and re-makes pilot 1's choice with a bin left out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_sparse_bins as psb  # noqa: E402

CHENG = np.array([[0.04, 0.051, 0.167, 0, 0, 0],
                  [0.008, 0.02, 0.026, 0, 0, 0],
                  [0, 0, 0.006, 0, 0, 0]])  # fmt: skip
WEIGHTS = np.array([[1.0, 0.53246753, 0.15355805, 0.16734694, 0.12480974, 0.082],
                    [0.92857143, 0.52, 0.2, 0.14130435, 0.78787879, 1.0],
                    [0.6, 1.0, 0.1875, 0.14285714, 0.54545455, 0.18181818]])  # fmt: skip


def _bins(household_contacts, household_infections):
    """Pooled (group, contacts|infections, bin) counts with the household 0-3 day bin set."""
    bins = np.zeros((3, 2, 6))
    bins[0, 0, 1], bins[0, 1, 1] = household_contacts, household_infections
    return bins


def test_one_infected_of_two_household_contacts_adds_twenty_point_five():
    """The Mac's block 23: rate 0.5 against Cheng's 0.051, weight 0.532, x 10 / 0.167^2."""
    terms = psb.attack_terms(_bins(2, 1), CHENG, WEIGHTS)
    assert terms[0, 1] == pytest.approx(20.4949, abs=1e-4)


def test_one_uninfected_contact_adds_a_quarter():
    """Blocks 17 and 43: rate 0, so the term is Cheng's own, 0.2644."""
    assert psb.attack_terms(_bins(1, 0), CHENG, WEIGHTS)[0, 1] == pytest.approx(0.2644, abs=1e-4)


def test_a_bin_without_contacts_charges_cheng_s_rate():
    """No contacts: rate 0, as the objective takes it (the household 4-5 day bin, 0.2358)."""
    assert psb.attack_terms(_bins(0, 0), CHENG, WEIGHTS)[0, 2] == pytest.approx(0.2358, abs=1e-4)


def test_the_rows_without_a_bin_lose_its_term_only():
    """Each row's cost less that bin's term; everything else as it was."""
    rows = [
        {"kind": "vector", "cost": "34.19", "term_0_1": "20.49", "term_0_2": "0.24"},
        {"kind": "spread", "cost": "14.60", "term_0_1": "0.26", "term_0_2": "0.24"},
    ]
    out = psb.without_bin(rows, 0, 1)
    assert [r["cost"] for r in out] == pytest.approx([34.19 - 20.49, 14.60 - 0.26])
    assert out[0]["kind"] == "vector" and rows[0]["cost"] == "34.19"  # input untouched


def test_no_rows_give_no_rows():
    """Nothing to take the term from."""
    assert psb.without_bin([], 0, 1) == []
