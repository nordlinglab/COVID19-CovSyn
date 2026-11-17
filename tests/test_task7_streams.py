# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 7, item 4: simulation seeds from (stream, block).

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 7: block b of stream s covers
s x 10^7 + b x S to s x 10^7 + (b + 1) x S - 1. Stream 0 holds every earlier run's seeds,
1-99 optimisation, 100 the stopping rule's monitoring blocks, 200-299 validation, 300 the
reporting checklist, 301 the synthesis for candidate selection (decision: the report keeps the
independent stream 300, E97). Acceptance: the seed ranges of all streams are disjoint for
S <= 10,000 and up to 1,000 blocks per stream.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from covsyn.calibration import seed_streams as ss

USED_STREAMS = [*ss.OPTIMISATION_STREAMS, ss.MONITORING_STREAM, *ss.VALIDATION_STREAMS,
                ss.REPORTING_STREAM, ss.SELECTION_STREAM]  # fmt: skip


def test_a_block_covers_the_plans_range():
    """Block 3 of stream 2 at S = 300: 2 x 10^7 + 900 to 2 x 10^7 + 1,199."""
    assert ss.block_seeds(2, 3, 300) == range(20_000_900, 20_001_200)
    assert ss.block_start(2, 3, 300) == 20_000_900


def test_the_reserved_streams():
    """The plan's table, and the report on the independent stream (E97)."""
    assert ss.EARLIER_STREAM == 0
    assert list(ss.OPTIMISATION_STREAMS) == list(range(1, 100))
    assert ss.MONITORING_STREAM == 100
    assert list(ss.VALIDATION_STREAMS) == list(range(200, 300))
    assert (ss.REPORTING_STREAM, ss.SELECTION_STREAM) == (300, 301)
    assert ss.DEFAULT_OPTIMISATION_STREAM == 1 and ss.DEFAULT_VALIDATION_STREAM == 200


def test_every_stream_is_disjoint_at_the_largest_size():
    """S = 10,000 and 1,000 blocks fill a stream exactly, and no two streams overlap."""
    spans = [(ss.block_start(s, 0, 10_000), ss.block_seeds(s, 999, 10_000).stop)
             for s in [ss.EARLIER_STREAM, *USED_STREAMS]]  # fmt: skip
    spans.sort()
    assert all(stop <= start for (_, stop), (start, _) in zip(spans, spans[1:], strict=False))
    assert spans[0][1] - spans[0][0] == 10**7


@given(
    a=st.sampled_from(USED_STREAMS), b=st.sampled_from(USED_STREAMS),
    block_a=st.integers(0, 999), block_b=st.integers(0, 999),
    simulations=st.integers(1, 10_000),
)  # fmt: skip
def test_two_blocks_share_a_seed_only_when_they_are_one_block(a, b, block_a, block_b, simulations):
    """Any two blocks of any streams are disjoint unless they are the same block."""
    first = ss.block_seeds(a, block_a, simulations)
    second = ss.block_seeds(b, block_b, simulations)
    overlap = max(first.start, second.start) < min(first.stop, second.stop)
    assert overlap == ((a, block_a) == (b, block_b))


@pytest.mark.parametrize(
    "stream, block, simulations",
    [(-1, 0, 300), (302, 0, 300), (1, -1, 300), (1, 1000, 300), (1, 0, 0), (1, 0, 10_001),
     (1, 0, 1.5), (True, 0, 300)],
)  # fmt: skip
def test_a_block_outside_the_plans_limits_is_refused(stream, block, simulations):
    """Beyond 1,000 blocks or S = 10,000 a stream could run into the next, so it is refused."""
    with pytest.raises(ValueError):
        ss.block_seeds(stream, block, simulations)


def test_stream_0_holds_the_earlier_runs():
    """Every earlier run's seeds lie in stream 0, which no new use takes.

    The objective's 0-299, the chain's 0-999, revalidation's from 100,000 and the reporting
    seeds from 500,000.
    """
    stream_0 = range(ss.block_start(0, 0, 1), ss.block_start(1, 0, 1))
    for seed in (0, 299, 999, 100_000, 500_000, 501_000):
        assert seed in stream_0


def test_one_check_says_which_streams_a_use_may_take():
    """Every use checks its stream the same way, naming the use and its streams."""
    assert ss.require(7, ss.OPTIMISATION_STREAMS, "optimisation") == 7
    assert ss.require(300, (ss.SELECTION_STREAM, ss.REPORTING_STREAM), "synthesis") == 300
    with pytest.raises(ValueError, match=r"validation stream.*200 to 299"):
        ss.require(1, ss.VALIDATION_STREAMS, "validation")
    with pytest.raises(ValueError, match="synthesis stream.*301, 300"):
        ss.require(1, (ss.SELECTION_STREAM, ss.REPORTING_STREAM), "synthesis")
    for bad in (True, 7.0, "7", None):
        with pytest.raises(ValueError):
            ss.require(bad, ss.OPTIMISATION_STREAMS, "optimisation")
    with pytest.raises(ValueError, match="one of the optimisation streams") as error:
        ss.require(100, ss.OPTIMISATION_STREAMS, "optimisation")
    assert "a optimisation" not in str(error.value)
