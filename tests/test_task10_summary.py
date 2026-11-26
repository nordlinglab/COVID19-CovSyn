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
from hypothesis import given
from hypothesis import strategies as st

from covsyn.calibration import revalidation
from covsyn.calibration import task10_summary as ts
from covsyn.pipeline import config as pc

DIM = 3
LOWER = np.zeros(DIM)
UPPER = np.array([1.0, 10.0, 2.0])


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(
    root,
    name,
    selected,
    candidates,
    uncertainty,
    stop="no improvement",
    bounds=None,
    streams=(5, 203),
    simulations=3000,
    chosen=1,
    pop_size=24,
):
    """A production run's files as the pipeline leaves them.

    candidates: the archive's best, by candidate number, with ``selected`` put at ``chosen``,
    the checklist's choice; firefly_archive.txt holds them at full precision, in this order,
    and both firefly_best.txt files at 7 decimals, as revalidation.write_best_file writes;
    uncertainty: cost_uncertainty.csv's rows, (candidate, mean, se, indistinguishable, best).
    """
    run = root / name
    (run / "revalidation").mkdir(parents=True)
    (run / "selected").mkdir()
    np.savetxt(run / "bound.txt", np.vstack((LOWER, UPPER)) if bounds is None else bounds)
    candidates = np.array(candidates, dtype=float)
    candidates[chosen] = selected
    costs = np.arange(len(candidates), dtype=float)
    archive = np.column_stack([np.zeros(len(candidates)), np.ones(len(candidates)), candidates,
                               costs])  # fmt: skip
    np.savetxt(run / "firefly_archive.txt", archive, header=revalidation.ARCHIVE_HEADER)
    revalidation.write_best_file(run / "revalidation" / "firefly_best.txt", candidates, costs)
    revalidation.write_best_file(run / "selected" / "firefly_best.txt", candidates[chosen],
                                 costs[chosen])  # fmt: skip
    _write_rows(run / "selected" / "selection.csv",
                [{"candidate": c, "validation_mean": float(c), "chosen": c == chosen}
                 for c in range(len(candidates))])  # fmt: skip
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
    config = pc.PipelineConfig(
        mode="production",
        stream=streams[0],
        validation_stream=streams[1],
        pop_size=pop_size,
        simulations=simulations,
        pilot="pilot-2",
        alpha_0=0.4,
        alpha_end=0.001,
        g_plan=415,
        revalidation_blocks=30,
    )
    (run / "pipeline.toml").write_text(pc.to_toml(config))
    return run


def _five(root, prefix="a", best_means=(10.0, 11.0, 12.0, 13.0, 14.0), pop_size=24):
    """Five runs; each selects (0.1i, i + 0.5, 0.2i), candidate 1; the set is candidates 0, 2."""
    # No choice equals candidate 0 or 2: the archive holds distinct vectors.
    candidates = np.array([[0.5, 5.0, 1.0], [0.9, 9.0, 1.9], [0.4, 4.0, 0.8]])
    runs = []
    for i, mean in enumerate(best_means, start=1):
        uncertainty = [
            (0, mean, 0.5, True, True),
            (1, mean + 5, 0.5, False, False),
            (2, mean + 1, 0.5, True, False),
        ]
        runs.append(
            _run(
                root,
                f"{prefix}{i}",
                [0.1 * i, i + 0.5, 0.2 * i],
                candidates,
                uncertainty,
                streams=(4 + i, 202 + i),
                pop_size=pop_size,
            )
        )
    return runs


def _runner_up(root, best_means=(10.0, 11.0, 12.0, 13.0, 14.0)):
    """The runner-up's five runs, at n = 48, on the same stream pairs."""
    return _five(root, "b", best_means, pop_size=48)


def _table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_parameter_uncertainty_is_over_the_five_selected_vectors(tmp_path):
    """Mean and SD per parameter of the selected vectors, against bound.txt's width."""
    out = tmp_path / "out"
    ts.summarise(_five(tmp_path), main=1, out=out)
    table = _table(out / "parameter_uncertainty.csv")
    assert len(table) == DIM
    selected = np.array([[0.1 * i, i + 0.5, 0.2 * i] for i in range(1, 6)])
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


def test_the_vectors_are_read_at_full_precision(tmp_path):
    """firefly_best.txt keeps 7 decimals; values closer than that still have an SD."""
    candidates = np.array([[0.5, 5.0, 1.0], [0.9, 9.0, 1.9], [0.4, 4.0, 0.8]])
    uncertainty = [
        (0, 1.0, 0.5, True, True),
        (1, 6.0, 0.5, False, False),
        (2, 2.0, 0.5, False, False),
    ]
    runs = [_run(tmp_path, f"a{i}", [0.5 + i * 1e-9, i, 0.2], candidates, uncertainty,
                 streams=(4 + i, 202 + i)) for i in range(1, 6)]  # fmt: skip
    out = tmp_path / "out"
    ts.summarise(runs, main=1, out=out)
    first = _table(out / "parameter_uncertainty.csv")[0]
    assert float(first["sd"]) == pytest.approx(np.std([0.5 + i * 1e-9 for i in range(1, 6)],
                                                      ddof=1), rel=1e-6)  # fmt: skip


def test_an_archive_that_does_not_match_the_scored_candidates_is_refused(tmp_path):
    """Its rounded vectors must be firefly_best.txt's, or the candidate numbers are off."""
    runs = _five(tmp_path)
    best = runs[0] / "revalidation" / "firefly_best.txt"
    rows = np.loadtxt(best)
    rows[[0, 2]] = rows[[2, 0]]
    np.savetxt(best, rows, fmt="%.7f")
    with pytest.raises(ValueError, match="archive"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_a_selection_without_one_choice_is_refused(tmp_path):
    """Two chosen candidates leave the run's vector ambiguous."""
    runs = _five(tmp_path)
    path = runs[2] / "selected" / "selection.csv"
    rows = _table(path)
    rows[0]["chosen"] = "True"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="chosen"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


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


def test_a_stage_resumed_from_a_checkpoint_is_marked_partial(tmp_path):
    """A resumed stage's seconds understate it, so the summary says which they are.

    The pipeline writes a stage's line once it finishes, and a resumed stage times only its last
    attempt (partial in run.log).
    """
    runs = _five(tmp_path)
    log = runs[0] / "run.log"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    for line in lines:
        if line.get("stage") == "optimise":
            line["partial"] = True
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    out = tmp_path / "out"
    ts.summarise(runs, main=1, out=out)
    records = json.loads((out / "summary.json").read_text())["runs"]
    assert records[0]["partial_stages"] == ["optimise"]
    assert records[1]["partial_stages"] == []


def test_a_run_without_the_optimiser_s_stop_is_refused(tmp_path):
    """Every production run records why it stopped; without it run.log is not the run's."""
    runs = _five(tmp_path)
    log = runs[1] / "run.log"
    log.write_text("".join(line + "\n" for line in log.read_text().splitlines()
                           if "optimiser_stopped" not in line))  # fmt: skip
    with pytest.raises(ValueError, match="stop"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_the_runner_up_is_compared_run_by_run(tmp_path):
    """Each pair's best means, their difference, and a t interval over the five pairs."""
    runs = _five(tmp_path, "a", (10.0, 11.0, 12.0, 13.0, 14.0))
    others = _runner_up(tmp_path, (12.0, 11.5, 15.0, 13.0, 16.0))
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
                 [(c, 10.0 + c, 0.5, c < 3, c == 0) for c in range(10)], bounds=bounds,
                 streams=(4 + i, 202 + i))
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
            _five(tmp_path), main=1, out=tmp_path / "out", competitor=_runner_up(tmp_path)[:4]
        )


@pytest.mark.parametrize("main", [0, 6, -1])
def test_a_main_run_outside_one_to_five_is_refused(tmp_path, main):
    """The main run is numbered 1 to 5."""
    with pytest.raises(ValueError, match="main"):
        ts.summarise(_five(tmp_path), main=main, out=tmp_path / "out")


def test_a_runner_up_out_of_stream_order_is_refused(tmp_path):
    """Pair i must share its optimisation and validation streams, or the pairing is noise."""
    with pytest.raises(ValueError, match="streams"):
        ts.summarise(
            _five(tmp_path), main=1, out=tmp_path / "out", competitor=_runner_up(tmp_path)[::-1]
        )


def test_two_runs_on_one_validation_stream_are_refused(tmp_path):
    """Each run takes a stream not used before (Task 10, item 1)."""
    runs = _five(tmp_path)
    toml = runs[1] / "pipeline.toml"
    toml.write_text(toml.read_text().replace("stream = 204", "stream = 203"))
    with pytest.raises(ValueError, match="validation stream 203"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_runs_at_another_s_are_refused(tmp_path):
    """Their costs are not comparable."""
    runs = _five(tmp_path)
    toml = runs[2] / "pipeline.toml"
    toml.write_text(toml.read_text().replace("simulations = 3000", "simulations = 1000"))
    with pytest.raises(ValueError, match="S"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_a_run_without_its_pipeline_toml_is_named(tmp_path):
    """Its streams and S come from it."""
    runs = _five(tmp_path)
    (runs[3] / "pipeline.toml").unlink()
    with pytest.raises(FileNotFoundError, match="a4"):
        ts.summarise(runs, main=1, out=tmp_path / "out")


def test_runs_swapped_between_the_lists_are_refused(tmp_path):
    """Pair 2 swapped keeps its streams but mixes n = 48 into the production runs."""
    runs, others = _five(tmp_path), _runner_up(tmp_path)
    runs[1], others[1] = others[1], runs[1]
    with pytest.raises(ValueError, match="pop_size"):
        ts.summarise(runs, main=1, out=tmp_path / "out", competitor=others)


def test_a_runner_up_of_the_production_configuration_is_refused(tmp_path):
    """The runner-up is another setting; the same one would compare a setting with itself."""
    with pytest.raises(ValueError, match="same configuration"):
        ts.summarise(_five(tmp_path), main=1, out=tmp_path / "out", competitor=_five(tmp_path, "b"))


def test_a_run_in_both_lists_is_refused(tmp_path):
    """It would give a difference of exactly 0 and narrow the interval."""
    runs = _five(tmp_path)
    others = _runner_up(tmp_path)
    others[2] = runs[2]
    with pytest.raises(ValueError, match="both"):
        ts.summarise(runs, main=1, out=tmp_path / "out", competitor=others)


def test_a_runner_up_with_other_bounds_is_refused(tmp_path):
    """Another search space is not the same configuration at another n."""
    others = _runner_up(tmp_path)
    np.savetxt(others[1] / "bound.txt", np.vstack((LOWER, UPPER * 2)))
    with pytest.raises(ValueError, match="bounds"):
        ts.summarise(_five(tmp_path), main=1, out=tmp_path / "out", competitor=others)


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


def test_a_failure_while_writing_leaves_no_output(tmp_path, monkeypatch):
    """A full disk part-way leaves nothing that a second attempt would refuse."""
    runs = _five(tmp_path)
    out = tmp_path / "out"
    real = ts.shutil.copyfile
    calls = []

    def failing(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise OSError(28, "No space left on device")
        return real(source, target)

    monkeypatch.setattr(ts.shutil, "copyfile", failing)
    with pytest.raises(OSError):
        ts.summarise(runs, main=1, out=out)
    assert not out.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("out")] == []
    monkeypatch.setattr(ts.shutil, "copyfile", real)
    ts.summarise(runs, main=1, out=out)
    assert (out / "summary.json").exists()


def test_an_empty_output_directory_is_used(tmp_path):
    """Made beforehand and empty: the summary goes into it."""
    out = tmp_path / "out"
    out.mkdir()
    ts.summarise(_five(tmp_path), main=1, out=out)
    assert (out / "summary.json").exists()


def test_the_command_writes_the_summary(tmp_path):
    """covsyn-task10-summary with --runs and --competitor."""
    runs = _five(tmp_path)
    others = _runner_up(tmp_path)
    out = tmp_path / "out"
    ts.main([str(out), "--main", "2", "--runs", *map(str, runs), "--competitor", *map(str, others)])
    assert json.loads((out / "summary.json").read_text())["main"] == 2
    assert (out / "comparison.csv").exists()


@given(st.text())
def test_only_true_and_false_are_read_as_flags(text):
    """The csv module writes True and False; any other text is refused, not read as False."""
    if text in ("True", "False"):
        assert ts._true(text) is (text == "True")
    else:
        with pytest.raises(ValueError):
            ts._true(text)


@pytest.mark.parametrize("text", ["true", "1", "", " True", "False\n", "TRUE"])
def test_near_spellings_of_a_flag_are_refused(text):
    """Spellings a hand edit or another writer might give."""
    with pytest.raises(ValueError):
        ts._true(text)
