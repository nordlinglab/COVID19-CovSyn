# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 10's summary of the five production runs, and the runner-up's comparison.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 10, items 1 and 2: the main run's
cost_uncertainty.csv, and parameter_uncertainty.csv over the five selected vectors with the
range of the main run's indistinguishable set. Decision of 2026-11-26: the runner-up of pilot 2
runs on the same five stream pairs, and each run's best validation mean is compared, run by run.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from covsyn.calibration import task10_summary as ts

DIM = 3
LOWER = np.zeros(DIM)
UPPER = np.array([1.0, 10.0, 2.0])


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(root, name, selected, candidates, uncertainty, stop="no improvement", bounds=None):
    """A production run's files as the pipeline leaves them.

    candidates: revalidation/firefly_best.txt's vectors, by candidate number;
    uncertainty: cost_uncertainty.csv's rows, (candidate, mean, se, indistinguishable, best).
    """
    run = root / name
    (run / "revalidation").mkdir(parents=True)
    (run / "selected").mkdir()
    np.savetxt(run / "bound.txt", np.vstack((LOWER, UPPER)) if bounds is None else bounds)
    costs = np.arange(len(candidates), dtype=float)
    rows = np.column_stack([np.arange(len(candidates)), candidates, costs])
    np.savetxt(run / "revalidation" / "firefly_best.txt", rows, fmt="%.7f")
    np.savetxt(
        run / "selected" / "firefly_best.txt", np.vstack([np.r_[0, selected, 1.0]] * 2), fmt="%.7f"
    )
    _write_rows(run / "revalidation" / "cost_uncertainty.csv", [
        {"candidate": c, "mean": m, "se": s, "difference_to_best": 0.0, "ci_low": 0.0,
         "ci_high": 0.0, "indistinguishable": i, "best": b}
        for c, m, s, i, b in uncertainty])  # fmt: skip
    lines = [
        {"event": "optimiser_stopped", "generation": 99, "reason": stop},
        {"stage": "optimise", "seconds": 100.0},
        {"stage": "revalidate", "seconds": 20.0},
    ]
    (run / "run.log").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return run


def _five(root, prefix="a", best_means=(10.0, 11.0, 12.0, 13.0, 14.0)):
    """Five runs; each selects (i, 2i, 0.1i); run 1's set is candidates 0 and 2 of three."""
    candidates = np.array([[0.5, 5.0, 1.0], [0.9, 9.0, 1.9], [0.4, 4.0, 0.8]])
    runs = []
    for i, mean in enumerate(best_means, start=1):
        uncertainty = [
            (0, mean, 0.5, True, True),
            (1, mean + 5, 0.5, False, False),
            (2, mean + 1, 0.5, True, False),
        ]
        runs.append(_run(root, f"{prefix}{i}", [0.1 * i, i, 0.2 * i], candidates, uncertainty))
    return runs


def _table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_parameter_uncertainty_is_over_the_five_selected_vectors(tmp_path):
    """Mean and SD per parameter of the selected vectors, against bound.txt's width."""
    out = tmp_path / "out"
    ts.summarise(_five(tmp_path), main=1, out=out)
    table = _table(out / "parameter_uncertainty.csv")
    assert len(table) == DIM
    selected = np.array([[0.1 * i, i, 0.2 * i] for i in range(1, 6)])
    for row, values, width in zip(table, selected.T, UPPER - LOWER, strict=True):
        assert float(row["mean"]) == pytest.approx(values.mean())
        assert float(row["sd_fraction"]) == pytest.approx(values.std(ddof=1) / width)


def test_the_range_comes_from_the_main_run_s_indistinguishable_candidates(tmp_path):
    """Candidates 0 and 2 of the main run, found by their number in firefly_best.txt."""
    out = tmp_path / "out"
    ts.summarise(_five(tmp_path), main=1, out=out)
    table = _table(out / "parameter_uncertainty.csv")
    assert [(float(r["set_min"]), float(r["set_max"])) for r in table] == [
        (0.4, 0.5),
        (4.0, 5.0),
        (0.8, 1.0),
    ]
    assert {r["set_size"] for r in table} == {"2"}


def test_the_candidate_number_not_the_csv_order_picks_the_vector(tmp_path):
    """cost_uncertainty.csv listed out of order still names firefly_best.txt's rows."""
    runs = _five(tmp_path)
    path = runs[0] / "revalidation" / "cost_uncertainty.csv"
    rows = _table(path)
    _write_rows(path, [rows[2], rows[0], rows[1]])
    out = tmp_path / "out"
    ts.summarise(runs, main=1, out=out)
    assert float(_table(out / "parameter_uncertainty.csv")[1]["set_min"]) == 4.0


def test_a_set_of_only_the_best_gives_its_own_values_as_the_range(tmp_path):
    """The set always holds the best; alone, its minimum is its maximum."""
    runs = _five(tmp_path)
    path = runs[0] / "revalidation" / "cost_uncertainty.csv"
    rows = _table(path)
    for row in rows[1:]:
        row["indistinguishable"] = "False"
    _write_rows(path, rows)
    out = tmp_path / "out"
    ts.summarise(runs, main=1, out=out)
    table = _table(out / "parameter_uncertainty.csv")
    assert [(float(r["set_min"]), float(r["set_max"])) for r in table] == [
        (0.5, 0.5),
        (5.0, 5.0),
        (1.0, 1.0),
    ]


def test_the_main_run_s_cost_table_and_every_run_s_are_kept(tmp_path):
    """cost_uncertainty.csv is the main run's; cost_uncertainty_run<i>.csv each run's."""
    runs = _five(tmp_path)
    out = tmp_path / "out"
    ts.summarise(runs, main=3, out=out)
    assert (out / "cost_uncertainty.csv").read_text() == (
        runs[2] / "revalidation" / "cost_uncertainty.csv"
    ).read_text()
    for i, run in enumerate(runs, start=1):
        assert (out / f"cost_uncertainty_run{i}.csv").read_text() == (
            run / "revalidation" / "cost_uncertainty.csv"
        ).read_text()


def test_the_summary_records_each_run(tmp_path):
    """Run id, best mean +- SE, set size, the optimiser's stop and each stage's seconds."""
    out = tmp_path / "out"
    ts.summarise(_five(tmp_path), main=1, out=out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["main"] == 1
    first = summary["runs"][0]
    assert first["run_id"] == "a1"
    assert (first["best_mean"], first["best_se"], first["set_size"]) == (10.0, 0.5, 2)
    assert first["stop_reason"] == "no improvement"
    assert first["stage_seconds"] == {"optimise": 100.0, "revalidate": 20.0}
    assert summary["not_identified"] == sum(
        r["not_identified"] == "True" for r in _table(out / "parameter_uncertainty.csv")
    )


def test_the_runner_up_is_compared_run_by_run(tmp_path):
    """Each pair's best means, their difference, and a t interval over the five pairs."""
    runs = _five(tmp_path, "a", (10.0, 11.0, 12.0, 13.0, 14.0))
    others = _five(tmp_path, "b", (12.0, 11.5, 15.0, 13.0, 16.0))
    out = tmp_path / "out"
    ts.summarise(runs, main=1, out=out, competitor=others)
    rows = _table(out / "comparison.csv")
    differences = np.array([2.0, 0.5, 3.0, 0.0, 2.0])
    assert [float(r["difference"]) for r in rows] == pytest.approx(differences)
    comparison = json.loads((out / "summary.json").read_text())["comparison"]
    from scipy import stats

    half = stats.t.ppf(0.975, 4) * differences.std(ddof=1) / np.sqrt(5)
    assert comparison["mean_difference"] == pytest.approx(differences.mean())
    assert (comparison["ci_low"], comparison["ci_high"]) == pytest.approx(
        (differences.mean() - half, differences.mean() + half)
    )
    assert comparison["production_lower_in"] == 4


def test_without_a_runner_up_there_is_no_comparison(tmp_path):
    """No --competitor: no comparison.csv, and no comparison in summary.json."""
    out = tmp_path / "out"
    ts.summarise(_five(tmp_path), main=1, out=out)
    assert not (out / "comparison.csv").exists()
    assert "comparison" not in json.loads((out / "summary.json").read_text())


def test_208_parameters_are_summarised(tmp_path):
    """Production's dimension, through the same files."""
    rng = np.random.default_rng(0)
    bounds = np.vstack((np.zeros(208), np.ones(208)))
    runs = [_run(tmp_path, f"r{i}", rng.random(208), rng.random((10, 208)),
                 [(c, 10.0 + c, 0.5, c < 3, c == 0) for c in range(10)], bounds=bounds)
            for i in range(1, 6)]  # fmt: skip
    out = tmp_path / "out"
    ts.summarise(runs, main=1, out=out)
    table = _table(out / "parameter_uncertainty.csv")
    assert len(table) == 208 and {r["set_size"] for r in table} == {"3"}


@pytest.mark.parametrize("count", [0, 4, 6])
def test_not_five_runs_is_refused(tmp_path, count):
    """Task 10 asks for R = 5 runs; none, four or six are refused."""
    runs = _five(tmp_path)
    runs = (runs + runs)[:count]
    with pytest.raises(ValueError, match="5 runs"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_a_runner_up_of_not_five_runs_is_refused(tmp_path):
    """The comparison pairs the runs one to one."""
    with pytest.raises(ValueError, match="5 runs"):
        ts.summarise(
            _five(tmp_path), main=1, out=tmp_path / "out", competitor=_five(tmp_path, "b")[:4]
        )


@pytest.mark.parametrize("main", [0, 6, -1])
def test_a_main_run_outside_one_to_five_is_refused(tmp_path, main):
    """The main run is numbered 1 to 5."""
    with pytest.raises(ValueError, match="main"):
        ts.summarise(_five(tmp_path), main=main, out=tmp_path / "out")


def test_the_same_run_twice_is_refused(tmp_path):
    """A run counted twice would shrink the SD."""
    runs = _five(tmp_path)
    runs[4] = runs[0]
    with pytest.raises(ValueError, match="twice"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_runs_with_other_bounds_are_refused(tmp_path):
    """The SD fraction needs one width per parameter."""
    runs = _five(tmp_path)
    np.savetxt(runs[3] / "bound.txt", np.vstack((LOWER, UPPER * 2)))
    with pytest.raises(ValueError, match="bounds"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_a_run_without_its_selection_is_named(tmp_path):
    """A run that stopped before select."""
    runs = _five(tmp_path)
    (runs[2] / "selected" / "firefly_best.txt").unlink()
    with pytest.raises(FileNotFoundError, match="a3"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_an_empty_cost_table_is_refused(tmp_path):
    """A header only: no best, no set."""
    runs = _five(tmp_path)
    path = runs[0] / "revalidation" / "cost_uncertainty.csv"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="no candidates"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_a_cost_table_without_one_best_is_refused(tmp_path):
    """Two rows marked best leave the best cost ambiguous."""
    runs = _five(tmp_path)
    path = runs[1] / "revalidation" / "cost_uncertainty.csv"
    rows = _table(path)
    rows[2]["best"] = "True"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="best"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_an_unknown_candidate_number_is_refused(tmp_path):
    """A set member firefly_best.txt does not hold."""
    runs = _five(tmp_path)
    path = runs[0] / "revalidation" / "cost_uncertainty.csv"
    rows = _table(path)
    rows[2]["candidate"] = "7"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="candidate 7"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_an_output_directory_with_files_is_not_overwritten(tmp_path):
    """An earlier summary stays."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text("{}")
    with pytest.raises(FileExistsError):
        ts.summarise(_five(tmp_path), main=1, out=out)


def test_the_command_writes_the_summary(tmp_path):
    """covsyn-task10-summary with --runs and --competitor."""
    runs = _five(tmp_path)
    others = _five(tmp_path, "b")
    out = tmp_path / "out"
    ts.main([str(out), "--main", "2", "--runs", *map(str, runs), "--competitor", *map(str, others)])
    assert json.loads((out / "summary.json").read_text())["main"] == 2
    assert (out / "comparison.csv").exists()
