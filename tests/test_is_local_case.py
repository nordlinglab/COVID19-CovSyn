# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""F5: one definition of a locally acquired first-wave case, shared by every report.

B66 left out B68's 'local_from_imported' seeds while compare_first_wave counted them, so the
two reports disagreed on the local cases of the same simulation.
"""

from __future__ import annotations

import pytest

from covsyn.validation.compare_first_wave import LOCAL_SEED_TYPES, is_local_case


@pytest.mark.parametrize(
    ("course", "local"),
    [
        ({}, True),                                    # a case the simulation infected
        ({"seed_type": None}, True),
        ({"seed_type": "local_unknown"}, True),        # B66
        ({"seed_type": "local_from_imported"}, True),  # B68, the one B66 left out
        ({"seed_type": "imported"}, False),
        ({"seed_type": ""}, False),                    # not a seed type the build writes
        ({"seed_type": "Local_unknown"}, False),       # seed types are case-sensitive
    ],
)
def test_each_seed_type_is_classified(course, local):
    """Each seed type, and a case with none, is classified."""
    assert is_local_case(course) is local


def test_every_seed_type_the_build_writes_except_imported_is_local():
    """build_first_wave_seeds writes these three; only 'imported' is not local."""
    for seed_type in ("imported", "local_unknown", "local_from_imported"):
        assert is_local_case({"seed_type": seed_type}) is (seed_type != "imported")


def test_a_string_outside_the_local_types_is_never_local():
    """Property: any string that is not one of the local seed types is not local."""
    hypothesis = pytest.importorskip("hypothesis")
    strategies = pytest.importorskip("hypothesis.strategies")

    @hypothesis.given(strategies.text())
    def check(seed_type):
        assert is_local_case({"seed_type": seed_type}) is (seed_type in LOCAL_SEED_TYPES)

    check()
