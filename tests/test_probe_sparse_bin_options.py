# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ways to handle the nearly empty household 0-3 day bin, scored on the probe's bins.csv.

Each option rewrites every evaluation's cost from its bin terms and its household 0-3 day
contacts alone, so pilot 1's choice can be made again without simulating.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_sparse_bin_options as pso  # noqa: E402


def _row(cost=100.0, term=20.0, contacts=2, infections=1):
    return {"cost": cost, "term_0_1": term, "household_0_3_contacts": contacts,
            "household_0_3_infections": infections}  # fmt: skip


def test_dropping_takes_the_term_out_of_every_cost():
    """Dropped, the bin adds nothing."""
    assert pso.rescored([_row()], "drop", None)[0]["cost"] == pytest.approx(80.0)


def test_a_threshold_drops_the_term_only_below_it():
    """Below the threshold the term goes; at it, it stays."""
    few, many = _row(contacts=4), _row(contacts=5)
    out = pso.rescored([few, many], "threshold", 5)
    assert [r["cost"] for r in out] == pytest.approx([80.0, 100.0])


def test_shrinking_toward_cheng_scales_the_term_by_the_squared_share_of_contacts():
    """With k contacts of prior, the term keeps (n / (n + k)) squared."""
    # rate (i + k c) / (n + k) lies n / (n + k) of the way from c; the term is that squared
    out = pso.rescored([_row(contacts=2)], "shrink", 2)
    assert out[0]["cost"] == pytest.approx(80.0 + 20.0 * 0.25)


def test_shrinking_a_bin_without_contacts_leaves_no_term():
    """No contacts: the shrunk rate is Cheng's, so the term is 0."""
    assert pso.rescored([_row(contacts=0, term=7.0)], "shrink", 5)[0]["cost"] == pytest.approx(93)


def test_a_failed_evaluation_keeps_its_charge():
    """A failed evaluation has no bin, so its 1e6 stays."""
    failed = _row(cost=1e6, term=0.0, contacts=-1, infections=-1)
    for option, value in (("drop", None), ("threshold", 5), ("shrink", 5)):
        assert pso.rescored([failed], option, value)[0]["cost"] == 1e6


def test_other_columns_are_kept():
    """Only the cost changes."""
    row = {**_row(), "label": "run19", "simulations": "3000"}
    out = pso.rescored([row], "drop", None)[0]
    assert (out["label"], out["simulations"]) == ("run19", "3000")


def test_an_unknown_option_is_refused():
    """An option the probe does not know is refused."""
    with pytest.raises(ValueError, match="unknown option"):
        pso.rescored([_row()], "merge", None)


def test_no_rows_give_no_rows():
    """Empty input gives empty output."""
    assert pso.rescored([], "drop", None) == []


@given(
    term=st.floats(0, 1e3),
    contacts=st.integers(0, 10**6),
    prior=st.floats(0.1, 1e3),
)
def test_shrinking_never_adds_cost_nor_takes_more_than_the_term(term, contacts, prior):
    """Shrinking keeps between none and all of the term."""
    out = pso.rescored([_row(cost=1e4, term=term, contacts=contacts)], "shrink", prior)[0]["cost"]
    assert 1e4 - term - 1e-9 <= out <= 1e4 + 1e-9


@given(contacts=st.integers(0, 10**6), threshold=st.integers(0, 10**6))
def test_a_threshold_either_keeps_or_drops_the_whole_term(contacts, threshold):
    """A threshold keeps all of the term or none."""
    out = pso.rescored([_row(contacts=contacts)], "threshold", threshold)[0]["cost"]
    assert out == pytest.approx(80.0 if contacts < threshold else 100.0)
