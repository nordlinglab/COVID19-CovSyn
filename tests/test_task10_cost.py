# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 10, item 1: the cost's measured uncertainty.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 10: re-score the best k = 10
archive candidates on 30 blocks of a validation stream not used before, at S; report each mean
cost +- SE (SD / sqrt(30)) in cost_uncertainty.csv, with the paired difference to the best
candidate and its 95% confidence interval; the candidates whose interval includes 0 cannot be
told apart from the best.

Decisions of 2026-11-22: only the archive, its at most 10 distinct vectors by ascending mean,
ties to the one that entered the elite first (pilot 2's rule); the best is the lowest mean on
the 30 blocks, a tie to the archive's order; the interval is a paired t interval, 29 degrees of
freedom, exploratory, as the best is chosen on the same blocks; the production pipeline's
revalidation does this at its S and 30 blocks, and its results serve the selection and
cost_uncertainty.csv alike.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy import stats

from covsyn.calibration import pilot_size, seed_streams
from covsyn.calibration import revalidation as rv
from covsyn.calibration.revalidation import ARCHIVE_HEADER

DIM = 3


def _archive(path, rows):
    np.savetxt(path / "firefly_archive.txt", np.array(rows, dtype=float), header=ARCHIVE_HEADER,
               encoding="utf-8")  # fmt: skip


def test_the_archive_ranking_is_one_rule_for_pilot_2_and_task_10(tmp_path):
    """The revalidation module holds it; pilot 2 uses the same function."""
    assert pilot_size.archive_candidates is rv.archive_candidates


def test_the_candidates_can_be_the_archive_s_best_alone(tmp_path):
    """--archive-best k: no final population, no personal bests; k by ascending mean."""
    rows = [[g % 4, 1, g, g + 1, g + 2, 50.0 - g] for g in range(15)]
    _archive(tmp_path, rows)
    np.savetxt(tmp_path / "firefly_result.txt", np.zeros((2, DIM + 1)))
    np.savetxt(tmp_path / "firefly_best.txt", np.zeros((2, DIM + 2)))
    vectors, means = rv.load_candidates(tmp_path, archive_best=10)
    assert len(vectors) == 10
    np.testing.assert_array_equal(vectors[0], [14, 15, 16])
    np.testing.assert_allclose(means, [50.0 - g for g in range(14, 4, -1)])


def _rows(costs_by_candidate):
    """revalidation.csv rows from each candidate's block costs."""
    return [{"candidate": i, **{f"block_{b}": c for b, c in enumerate(costs)}}
            for i, costs in enumerate(costs_by_candidate)]  # fmt: skip


def test_each_candidate_s_mean_and_se_over_its_blocks():
    """Mean and SD / sqrt(30), the SD with ddof = 1."""
    rng = np.random.default_rng(1)
    costs = [rng.normal(10 + i, 1, 30) for i in range(4)]
    table = rv.cost_uncertainty(_rows(costs))
    for line, c in zip(table, costs, strict=True):
        assert line["mean"] == pytest.approx(np.mean(c))
        assert line["se"] == pytest.approx(np.std(c, ddof=1) / np.sqrt(30))


def test_the_paired_difference_to_the_best_has_a_t_interval_on_29_degrees_of_freedom():
    """Mean of d = candidate - best by block, +- t(0.975, 29) SD(d) / sqrt(30)."""
    rng = np.random.default_rng(2)
    shared = rng.normal(0, 3, 30)
    costs = [10 + shared + rng.normal(0, 0.5, 30), 10.2 + shared + rng.normal(0, 0.5, 30),
             14 + shared + rng.normal(0, 0.5, 30)]  # fmt: skip
    table = rv.cost_uncertainty(_rows(costs))
    best = min(range(3), key=lambda i: np.mean(costs[i]))
    for line, c in zip(table, costs, strict=True):
        d = c - costs[best]
        half = (stats.t.ppf(0.975, 29) * np.std(d, ddof=1) / np.sqrt(30)
                if line["candidate"] != best else 0.0)  # fmt: skip
        assert line["difference_to_best"] == pytest.approx(np.mean(d))
        assert line["ci_low"] == pytest.approx(np.mean(d) - half)
        assert line["ci_high"] == pytest.approx(np.mean(d) + half)
        assert line["indistinguishable"] == bool(line["ci_low"] <= 0 <= line["ci_high"])
    assert table[best]["best"] and sum(line["best"] for line in table) == 1
    assert not table[2]["indistinguishable"]  # 4 higher, with shared noise: told apart


def test_a_tie_for_best_goes_to_the_archive_s_order():
    """The earlier candidate, which ranked higher in the archive."""
    costs = [np.full(30, 5.0), np.full(30, 5.0)]
    table = rv.cost_uncertainty(_rows(costs))
    assert table[0]["best"] and not table[1]["best"]


def test_the_best_is_indistinguishable_from_itself():
    """Its difference is 0, inside its own interval."""
    costs = [np.arange(30.0), np.arange(30.0) + 1]
    assert rv.cost_uncertainty(_rows(costs))[0]["indistinguishable"]


@pytest.mark.parametrize("costs", [[[], []], [[10.0], [12.0]]])
def test_insufficient_blocks_leave_uncertainty_unknown(costs):
    """Candidates are retained when no paired interval can distinguish them."""
    table = rv.cost_uncertainty(_rows(costs))
    assert all(row["indistinguishable"] for row in table)
    assert all(np.isnan(row[key]) for row in table for key in ("se", "ci_low", "ci_high"))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_cost_uncertainty_refuses_nonfinite_blocks(value):
    """Failed evaluations are finite sentinels, so nonfinite input signals a fault."""
    rows = _rows([[10.0, 11.0], [12.0, value]])
    with pytest.raises(ValueError, match=r"candidate 1.*block_1"):
        rv.cost_uncertainty(rows)


def test_cost_uncertainty_refuses_an_empty_candidate_list():
    """No candidate means there is no best to use for pairing."""
    with pytest.raises(ValueError, match="at least one candidate"):
        rv.cost_uncertainty([])


def test_cost_uncertainty_refuses_candidates_with_different_block_keys():
    """A missing block must never be broadcast across the other candidate's costs."""
    with pytest.raises(ValueError, match="candidate 1"):
        rv.cost_uncertainty(_rows([[10.0], [12.0, 13.0]]))


def test_the_command_rescoring_the_archive_at_s_writes_cost_uncertainty(tmp_path, monkeypatch):
    """--simulations S and --archive-best k: blocks of S seeds of the given stream."""
    run = tmp_path / "run"
    run.mkdir()
    _archive(run, [[g, 1, g * 0.1, 0.2, 0.3, 9.0 - g * 0.1] for g in range(12)])
    np.savetxt(run / "firefly_result.txt", np.zeros((2, DIM + 1)))
    np.savetxt(run / "firefly_best.txt", np.zeros((2, DIM + 2)))
    seen = {}

    def scored(vector, demo, pool, cheng, seed_blocks, simulations):
        seen.setdefault("blocks", seed_blocks)
        seen["simulations"] = simulations
        totals = [float(vector[0]) + b.start % 7 for b in seed_blocks]
        return totals, {name: 0.0 for name in rv.REPORTED_OUTCOMES}

    monkeypatch.setattr(rv, "score", scored)
    out = tmp_path / "out"
    rv.main([str(run), str(out), "30", "--stream", "203", "--simulations", "3000",
             "--archive-best", "10", "--workers", "1"])  # fmt: skip
    assert seen["simulations"] == 3000
    assert seen["blocks"] == [seed_streams.block_seeds(203, b, 3000) for b in range(30)]
    with open(out / "revalidation.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 10
    with open(out / "cost_uncertainty.csv", newline="") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 10 and sum(r["best"] == "True" for r in table) == 1


def test_a_simulation_count_a_block_cannot_hold_is_refused(tmp_path):
    """S must fit one block of a stream, as everywhere else."""
    with pytest.raises(SystemExit):
        rv.main([str(tmp_path), str(tmp_path / "out"), "30", "--simulations", "10001"])


def test_the_score_runs_its_s_simulations_per_block(monkeypatch):
    """The score hands S to the objective, not the optimiser's 300."""
    from covsyn.calibration import fast_cost

    seen = []
    monkeypatch.setattr(fast_cost, "cost_function",
                        lambda *a, seed_offset=0, simulations=None: seen.append(simulations) or 1.0)
    rv.score(np.zeros(DIM), None, None, (None, None, None), [range(0, 3000)], 3000)
    assert seen == [3000]


def test_an_archive_of_no_rows_ranks_no_candidates(tmp_path):
    """A header-only archive gives no vectors, not a reshape error (code review 2026-11-22)."""
    _archive(tmp_path, np.empty((0, DIM + 3)))
    vectors, means = rv.ranked_archive(tmp_path / "firefly_archive.txt", 10)
    assert len(vectors) == 0 and len(means) == 0


def test_revalidating_an_empty_archive_stops_with_a_reason(tmp_path):
    """--archive-best on a header-only archive: no candidates, said so, nothing scored."""
    run = tmp_path / "run"
    run.mkdir()
    _archive(run, np.empty((0, DIM + 3)))
    with pytest.raises(SystemExit, match="no candidates.*archive is empty"):
        rv.main([str(run), str(tmp_path / "out"), "30", "--simulations", "3000",
                 "--archive-best", "10", "--workers", "1"])  # fmt: skip
    assert not (tmp_path / "out").exists()  # stopped before writing anything


def test_a_stream_outside_validation_stops_before_writing(tmp_path, capsys):
    """--stream 150 is refused, and no output directory is left (code review round 3)."""
    run = tmp_path / "run"
    run.mkdir()
    _archive(run, [[g, 1, g * 0.1, 0.2, 0.3, 9.0 - g * 0.1] for g in range(3)])
    with pytest.raises(SystemExit):
        rv.main([str(run), str(tmp_path / "out"), "30", "--stream", "150",
                 "--archive-best", "10", "--workers", "1"])  # fmt: skip
    assert "150" in capsys.readouterr().err  # the stream is the reason
    assert not (tmp_path / "out").exists()


def test_an_archive_from_before_task_8_stops_with_a_reason(tmp_path):
    """--archive-best needs the means of a v2 archive: one line, not a traceback."""
    run = tmp_path / "run"
    run.mkdir()
    np.savetxt(run / "firefly_archive.txt", np.ones((2, DIM + 2)))
    with pytest.raises(SystemExit, match="v2 archive"):
        rv.main([str(run), str(tmp_path / "out"), "30", "--archive-best", "10",
                 "--workers", "1"])  # fmt: skip


@pytest.mark.parametrize("blocks", ["0", "-3"])
def test_no_blocks_is_refused_before_writing(tmp_path, capsys, blocks):
    """Revalidation on no blocks would write NaN means (code review round 4)."""
    run = tmp_path / "run"
    run.mkdir()
    _archive(run, [[g, 1, g * 0.1, 0.2, 0.3, 9.0 - g * 0.1] for g in range(3)])
    with pytest.raises(SystemExit):
        rv.main([str(run), str(tmp_path / "out"), blocks, "--archive-best", "10",
                 "--workers", "1"])  # fmt: skip
    assert "blocks" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_archive_refusal_names_its_path_once(tmp_path):
    """The archive's own message holds its path (code review round 4)."""
    run = tmp_path / "run"
    run.mkdir()
    np.savetxt(run / "firefly_archive.txt", np.ones((2, DIM + 2)))
    with pytest.raises(SystemExit) as stopped:
        rv.main([str(run), str(tmp_path / "out"), "30", "--archive-best", "10",
                 "--workers", "1"])  # fmt: skip
    assert str(stopped.value).count(str(run)) == 1


def test_a_malformed_file_is_refused_naming_the_run(tmp_path):
    """A message of no file, as loadtxt's, gets the run added (code review round 5)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "firefly_archive.txt").write_text(f"# {ARCHIVE_HEADER}\nx 1 2 3 4 5\n", "utf-8")
    with pytest.raises(SystemExit) as stopped:
        rv.main([str(run), str(tmp_path / "out"), "30", "--archive-best", "10",
                 "--workers", "1"])  # fmt: skip
    assert str(stopped.value).count(str(run)) == 1


def test_a_malformed_personal_best_file_is_named(tmp_path):
    """Without --archive-best: the file that cannot be read, not only its run (round 6)."""
    run = tmp_path / "run"
    run.mkdir()
    np.savetxt(run / "firefly_result.txt", np.zeros((2, DIM + 1)))
    (run / "firefly_best.txt").write_text("0 x 1 2 3\n")
    with pytest.raises(SystemExit, match="firefly_best.txt"):
        rv.main([str(run), str(tmp_path / "out"), "30", "--workers", "1"])


def test_a_relative_run_directory_is_still_named(tmp_path, monkeypatch):
    """Run as '.', whose name is in any message: the file's path is given (round 6)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "firefly_archive.txt").write_text(f"# {ARCHIVE_HEADER}\nx 1 2 3 4 5\n", "utf-8")
    monkeypatch.chdir(run)
    with pytest.raises(SystemExit, match="firefly_archive.txt"):
        rv.main([".", str(tmp_path / "out"), "30", "--archive-best", "10", "--workers", "1"])


def test_files_of_different_widths_are_refused_naming_them(tmp_path):
    """The population and personal bests must hold vectors of one length (round 7)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "firefly_result.txt").write_text("1 2 3 4\n")
    (run / "firefly_best.txt").write_text("0 1 2 3 4 5\n")
    with pytest.raises(SystemExit, match="firefly_result.txt.*firefly_best.txt"):
        rv.main([str(run), str(tmp_path / "out"), "30", "--workers", "1"])


@pytest.mark.parametrize("name", ["firefly_result.txt", "firefly_best.txt"])
def test_a_file_of_no_rows_is_refused_naming_it(tmp_path, name):
    """An empty population or personal-best file: named, not an IndexError (round 7)."""
    run = tmp_path / "run"
    run.mkdir()
    np.savetxt(run / "firefly_result.txt", np.zeros((2, DIM + 1)))
    np.savetxt(run / "firefly_best.txt", np.zeros((2, DIM + 2)))
    (run / name).write_text("")
    with pytest.raises(SystemExit, match=f"{name}.*no rows"):
        rv.main([str(run), str(tmp_path / "out"), "30", "--workers", "1"])


def test_files_of_costs_alone_are_refused(tmp_path):
    """Vectors of no parameters cannot be simulated (round 8)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "firefly_result.txt").write_text("5.0\n")
    (run / "firefly_best.txt").write_text("0 5.0\n")
    with pytest.raises(SystemExit, match="firefly_result.txt.*no parameters"):
        rv.main([str(run), str(tmp_path / "out"), "30", "--workers", "1"])


@pytest.mark.parametrize("read", ["read_archive", "ranked_archive", "load_candidates"])
def test_other_warnings_while_reading_still_show(tmp_path, monkeypatch, read):
    """Only numpy's no-data warning is hidden, wherever a run's files are read (round 9)."""
    import warnings

    _archive(tmp_path, [[0, 1, 0.1, 0.2, 0.3, 9.0]])
    np.savetxt(tmp_path / "firefly_result.txt", np.zeros((2, DIM + 1)))
    np.savetxt(tmp_path / "firefly_best.txt", np.zeros((2, DIM + 2)))
    real = np.loadtxt

    def noisy(*args, **kwargs):
        warnings.warn("another warning", UserWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(rv.np, "loadtxt", noisy)
    arguments = {"read_archive": (tmp_path / "firefly_archive.txt",),
                 "ranked_archive": (tmp_path / "firefly_archive.txt", 10),
                 "load_candidates": (tmp_path,)}[read]  # fmt: skip
    with pytest.warns(UserWarning, match="another warning"):
        getattr(rv, read)(*arguments)
