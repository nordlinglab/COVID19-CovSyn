# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 9, pilot 1: noise against simulations per evaluation.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 9: score the literature seed,
the run-19 vector and two random rows of the initial population at S in {300, 1,000, 3,000,
10,000}, 30 blocks each; report each vector's mean and SD, the SD of the paired difference
between vectors on the same block, and how the mean changes with S; choose the smallest S at
which the SD of a paired difference is at most 10% of the cost spread across the initial
population. Decisions of 2026-11-21: validation stream 201, with non-overlapping block ranges
per S; the two random rows drawn without replacement from rows 3-255 with a fixed, recorded
seed; the spread the sample SD across all 256 rows of each row's mean on blocks 43 and 44 at
S = 10,000; the largest of the six paired SDs decides; no S meeting it chooses none.
"""

from __future__ import annotations

import concurrent.futures
import csv
import itertools
import json

import numpy as np
import pytest

from covsyn.calibration import pilot_noise as pn
from covsyn.calibration import seed_streams

DIM = 4


def test_the_vectors_are_the_literature_run_19_and_two_random_rows():
    """Rows 0 and 2, then two distinct Latin-hypercube rows, the same on every call."""
    rows = pn.pilot_rows()
    assert list(rows) == ["literature", "run19", "random_a", "random_b"]
    assert (rows["literature"], rows["run19"]) == (0, 2)
    assert rows["random_a"] != rows["random_b"]
    assert {rows["random_a"], rows["random_b"]} <= set(range(3, 256))
    assert pn.pilot_rows() == rows
    assert pn.pilot_rows(pn.ROW_SEED + 1) != rows  # the seed, not a fixed pick, chooses them


def test_each_s_has_thirty_blocks_of_stream_201():
    """S in {300, 1,000, 3,000, 10,000}, 30 blocks each; two blocks at 10,000 for the spread."""
    assert pn.STREAM == 201 and set(pn.BLOCKS) == {300, 1_000, 3_000, 10_000}
    assert all(len(blocks) == 30 for blocks in pn.BLOCKS.values())
    assert (pn.SPREAD_SIMULATIONS, pn.SPREAD_BLOCKS) == (10_000, (43, 44))


def test_no_two_evaluations_of_the_pilot_share_a_seed():
    """Every S and the spread take their own seeds of stream 201.

    So no mean is paired by accident across S; within one S all vectors share each block's
    seeds.
    """
    ranges = [seed_streams.block_seeds(pn.STREAM, b, s) for s, bs in pn.BLOCKS.items() for b in bs]
    ranges += [seed_streams.block_seeds(pn.STREAM, b, pn.SPREAD_SIMULATIONS)
               for b in pn.SPREAD_BLOCKS]  # fmt: skip
    for a, b in itertools.combinations(ranges, 2):
        assert a.stop <= b.start or b.stop <= a.start


def test_the_tasks_score_every_vector_on_every_block_and_every_row_for_the_spread():
    """4 vectors x 4 S x 30 blocks, and 256 rows x 2 blocks."""
    tasks = pn.tasks(pn.pilot_rows(), rows=256)
    vectors = [t for t in tasks if t.kind == "vector"]
    spread = [t for t in tasks if t.kind == "spread"]
    assert len(vectors) == 4 * 4 * 30 and len(spread) == 256 * 2
    assert {(t.simulations, t.block) for t in vectors if t.label == "run19"} == {
        (s, b) for s, bs in pn.BLOCKS.items() for b in bs}  # fmt: skip
    for t in tasks:
        assert t.seed_offset == seed_streams.block_seeds(pn.STREAM, t.block, t.simulations).start


def _rows(costs):
    """blocks.csv rows from {(kind, label, row, S, block): cost}."""
    return [{"kind": k, "label": label, "row": r, "simulations": s, "block": b, "cost": c}
            for (k, label, r, s, b), c in costs.items()]  # fmt: skip


def _synthetic(pair_sd, spread_sd, seed=0):
    """Vector costs whose paired differences have SD near pair_sd[S]; a spread near spread_sd."""
    rng = np.random.default_rng(seed)
    labels = pn.pilot_rows()
    costs = {}
    for s, blocks in pn.BLOCKS.items():
        shared = rng.normal(20, 3, len(blocks))  # each block's seeds, common to every vector
        for i, (label, row) in enumerate(labels.items()):
            own = rng.normal(0, pair_sd[s] / np.sqrt(2), len(blocks))
            for b, c in zip(blocks, shared + i + own, strict=True):
                costs[("vector", label, row, s, b)] = float(c)
    level = rng.normal(20, spread_sd, 256)
    for row in range(256):
        for b in pn.SPREAD_BLOCKS:
            costs[("spread", "", row, pn.SPREAD_SIMULATIONS, b)] = float(level[row])
    return _rows(costs)


def test_the_summary_gives_each_vector_s_mean_and_sd_at_each_s():
    """Over its 30 blocks, the SD with ddof = 1."""
    rows = _synthetic({s: 1.0 for s in pn.BLOCKS}, 10.0)
    summary = pn.summarise(rows)
    for line in summary["vectors"]:
        costs = [r["cost"] for r in rows if r["kind"] == "vector"
                 and r["label"] == line["label"] and r["simulations"] == line["simulations"]]
        assert line["blocks"] == 30
        assert line["mean"] == pytest.approx(np.mean(costs))
        assert line["sd"] == pytest.approx(np.std(costs, ddof=1))
    assert len(summary["vectors"]) == 16


def test_the_summary_gives_the_sd_of_each_paired_difference_block_by_block():
    """Six pairs at each S, each difference taken on the same block."""
    rows = _synthetic({s: 1.0 for s in pn.BLOCKS}, 10.0)
    summary = pn.summarise(rows)
    assert len(summary["pairs"]) == 6 * 4
    line = next(p for p in summary["pairs"]
                if p["simulations"] == 300 and p["pair"] == "literature-run19")  # fmt: skip
    by = {(r["label"], r["block"]): r["cost"] for r in rows
          if r["kind"] == "vector" and r["simulations"] == 300}  # fmt: skip
    d = [by[("literature", b)] - by[("run19", b)] for b in pn.BLOCKS[300]]
    assert line["sd_difference"] == pytest.approx(np.std(d, ddof=1))
    assert line["mean_difference"] == pytest.approx(np.mean(d))


def test_the_spread_is_the_sd_across_rows_of_their_two_block_means():
    """With each single block's SD beside it, to show whether the denominator is stable."""
    rows = _synthetic({s: 1.0 for s in pn.BLOCKS}, 10.0)
    for r in rows:  # make the two blocks differ
        if r["kind"] == "spread" and r["block"] == 44:
            r["cost"] += (r["row"] % 3) - 1.0
    spread = pn.summarise(rows)["spread"]
    by = {(r["row"], r["block"]): r["cost"] for r in rows if r["kind"] == "spread"}
    means = [(by[(i, 43)] + by[(i, 44)]) / 2 for i in range(256)]
    assert spread["rows"] == 256
    assert spread["sd"] == pytest.approx(np.std(means, ddof=1))
    assert spread["sd_block_43"] == pytest.approx(np.std([by[(i, 43)] for i in range(256)], ddof=1))
    assert spread["sd_block_44"] == pytest.approx(np.std([by[(i, 44)] for i in range(256)], ddof=1))


def test_the_smallest_s_whose_largest_paired_sd_is_within_ten_percent_is_chosen():
    """The criterion and, per S, the largest paired SD, its ratio to the spread and the verdict."""
    pair_sd = {300: 5.0, 1_000: 2.5, 3_000: 0.4, 10_000: 0.2}
    choice = pn.summarise(_synthetic(pair_sd, 10.0))["choice"]
    assert choice["chosen_simulations"] == 3_000
    assert choice["threshold"] == pytest.approx(0.1 * choice["spread_sd"])
    for line in choice["per_simulations"]:
        assert line["ratio"] == pytest.approx(line["max_pair_sd"] / choice["spread_sd"])
        assert line["meets"] == (line["max_pair_sd"] <= choice["threshold"])


def test_the_largest_pair_decides_though_the_others_pass():
    """Every one of the six pairs must pass (decision of 2026-11-21)."""
    rows = _synthetic({s: 0.1 for s in pn.BLOCKS}, 10.0)
    for r in rows:  # one vector noisy at every S: its three pairs fail, the other three pass
        if r["kind"] == "vector" and r["label"] == "random_b":
            r["cost"] += 5.0 * np.sin(r["block"])
    assert pn.summarise(rows)["choice"]["chosen_simulations"] is None


def test_no_s_meeting_the_criterion_chooses_none():
    """Then the measurements go to the professor (decision of 2026-11-21)."""
    choice = pn.summarise(_synthetic({s: 5.0 for s in pn.BLOCKS}, 10.0))["choice"]
    assert choice["chosen_simulations"] is None
    assert all(not line["meets"] for line in choice["per_simulations"])


def test_a_failed_evaluation_is_counted_in_the_summary():
    """The objective charges 1e6 for a failed simulation (E31), which would swamp a mean."""
    rows = _synthetic({s: 1.0 for s in pn.BLOCKS}, 10.0)
    rows[0]["cost"] = pn.FAILED_EVALUATION_COST
    assert pn.summarise(rows)["failed_evaluations"] == 1


def _evaluate(P, seed_offset):
    """A stand-in for fast_cost.evaluate_candidate: (cost, parts, timings)."""
    return float(np.sum(P) + seed_offset % 997 / 100), {}, {}


def _threads(simulations):
    return concurrent.futures.ThreadPoolExecutor(max_workers=4)


class _Stop(Exception):
    pass


def _population():
    return np.random.default_rng(3).random((256, DIM))


def test_a_run_writes_every_evaluation_and_the_summary(tmp_path):
    """blocks.csv holds one row per evaluation; the summaries are written beside it."""
    pn.run(tmp_path, _population(), np.zeros(DIM), np.full(DIM, 2.0), _evaluate, _threads)
    with open(tmp_path / pn.BLOCKS_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 4 * 30 + 256 * 2
    choice = json.loads((tmp_path / pn.CHOICE_FILE).read_text())
    assert "chosen_simulations" in choice
    for name in (pn.VECTORS_FILE, pn.PAIRS_FILE):
        assert (tmp_path / name).is_file()


def test_the_vectors_are_scored_in_the_model_s_units(tmp_path):
    """Normalised rows become parameters by the bounds, as the optimiser's are."""
    seen = []

    def evaluate(P, seed_offset):
        seen.append(np.array(P))
        return _evaluate(P, seed_offset)

    population = _population()
    lower, upper = np.full(DIM, 1.0), np.full(DIM, 3.0)
    pn.run(tmp_path, population, lower, upper, evaluate, _threads)
    assert any(np.allclose(p, population[0] * 2.0 + 1.0) for p in seen)


def test_a_killed_run_resumes_without_scoring_anything_twice(tmp_path):
    """Each finished evaluation is in blocks.csv, so a resume skips it."""
    calls = []

    def dies(P, seed_offset):
        if len(calls) >= 200:
            raise _Stop
        calls.append(seed_offset)
        return _evaluate(P, seed_offset)

    with pytest.raises(_Stop):
        pn.run(tmp_path, _population(), np.zeros(DIM), np.ones(DIM), dies, _threads)
    done = (tmp_path / pn.BLOCKS_FILE).read_text().count("\n") - 1
    assert 0 < done <= 200
    again = []

    def counts(P, seed_offset):
        again.append(seed_offset)
        return _evaluate(P, seed_offset)

    pn.run(tmp_path, _population(), np.zeros(DIM), np.ones(DIM), counts, _threads)
    assert len(again) == 4 * 4 * 30 + 256 * 2 - done
    with open(tmp_path / pn.BLOCKS_FILE, newline="") as f:
        keys = [(r["kind"], r["row"], r["simulations"], r["block"]) for r in csv.DictReader(f)]
    assert len(keys) == len(set(keys)) == 4 * 4 * 30 + 256 * 2


def test_a_resumed_and_an_uninterrupted_run_give_the_same_summary(tmp_path):
    """The summary depends on the evaluations only, not on the order they finished in."""
    whole, killed = tmp_path / "whole", tmp_path / "killed"
    whole.mkdir(), killed.mkdir()
    pn.run(whole, _population(), np.zeros(DIM), np.ones(DIM), _evaluate, _threads)
    calls = []

    def dies(P, seed_offset):
        if len(calls) >= 300:
            raise _Stop
        calls.append(1)
        return _evaluate(P, seed_offset)

    with pytest.raises(_Stop):
        pn.run(killed, _population(), np.zeros(DIM), np.ones(DIM), dies, _threads)
    pn.run(killed, _population(), np.zeros(DIM), np.ones(DIM), _evaluate, _threads)
    for name in (pn.CHOICE_FILE, pn.VECTORS_FILE, pn.PAIRS_FILE):
        assert (killed / name).read_text() == (whole / name).read_text()


def test_the_command_records_what_it_ran(tmp_path, monkeypatch):
    """config.json: the command, the commit, the rows, the row seed, the stream and blocks."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    out = pn.main(["--results-root", str(tmp_path)])
    assert out.parent == tmp_path / "tables" and out.name.endswith("_pilot-noise")
    config = json.loads((out / pn.CONFIG_FILE).read_text())["settings"]  # runs.write_config
    assert config["command"].startswith("covsyn-pilot-noise")
    assert config["rows"] == pn.pilot_rows() and config["row_seed"] == pn.ROW_SEED
    assert config["stream"] == 201
    assert config["blocks"] == {str(s): [b.start, b.stop - 1] for s, b in pn.BLOCKS.items()}
    assert "commit" in config


def test_the_command_resumes_the_directory_it_is_given(tmp_path, monkeypatch):
    """--resume DIR continues the pilot in DIR."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    out = pn.main(["--results-root", str(tmp_path)])
    before = (out / pn.BLOCKS_FILE).read_text()
    assert pn.main(["--resume", str(out)]) == out
    assert (out / pn.BLOCKS_FILE).read_text() == before  # nothing left to score


def test_resuming_a_directory_without_a_pilot_is_refused(tmp_path):
    """No config.json of a pilot: nothing to continue."""
    with pytest.raises(SystemExit):
        pn.main(["--resume", str(tmp_path)])


def test_failed_evaluations_choose_no_s():
    """A failed evaluation's 1e6 swamps the spread, so every S would pass: none is chosen."""
    rows = _synthetic({300: 5.0, 1_000: 2.5, 3_000: 0.4, 10_000: 0.2}, 10.0)
    spread = next(r for r in rows if r["kind"] == "spread")
    spread["cost"] = pn.FAILED_EVALUATION_COST
    choice = pn.summarise(rows)["choice"]
    assert choice["chosen_simulations"] is None and choice["failed_evaluations"] == 1
    assert not any(line["meets"] for line in choice["per_simulations"])  # no S passes either


def test_a_resume_keeps_the_workers_pilot_1_started_with(tmp_path, monkeypatch):
    """As pilot 2's: --workers, else COVSYN_WORKERS, else config.json's count."""
    seen = []
    monkeypatch.setattr(pn, "_evaluator", lambda workers: seen.append(workers)
                        or (_evaluate, _threads))  # fmt: skip
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    out = pn.main(["--results-root", str(tmp_path), "--workers", "3"])
    pn.main(["--resume", str(out)])
    assert seen[-1] == 3


def test_a_read_only_config_says_so_not_that_pilot_1_is_missing(tmp_path, monkeypatch, capsys):
    """As pilot 2's: the record cannot take the new workers, and the message says that."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    out = pn.main(["--results-root", str(tmp_path), "--workers", "2"])
    (out / "config.json").chmod(0o444)
    try:
        with pytest.raises(SystemExit) as stop:
            pn.main(["--resume", str(out), "--workers", "3"])
    finally:
        (out / "config.json").chmod(0o644)
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "read-only" in err and "holds no pilot" not in err


def test_a_config_whose_rewrite_fails_is_not_called_missing(tmp_path, monkeypatch, capsys):
    """As pilot 2's: a failed write of the workers says so."""
    import errno

    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    out = pn.main(["--results-root", str(tmp_path), "--workers", "2"])

    def failing(out_dir, workers):
        raise OSError(errno.EIO, "short write", str(out_dir / "config.json"))

    monkeypatch.setattr(pn.block_scoring, "record_workers", failing)
    with pytest.raises(SystemExit) as stop:
        pn.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2
    assert "could not record" in err and "holds no pilot" not in err


def test_a_config_json_cannot_write_back_is_a_command_error(tmp_path, monkeypatch, capsys):
    """NaN reads but does not write as JSON: an exit-2 refusal, not a traceback."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    out = pn.main(["--results-root", str(tmp_path), "--workers", "2"])
    text = (out / "config.json").read_text()
    (out / "config.json").write_text(text.replace('"settings": {', '"settings": {"x": NaN, ', 1))
    with pytest.raises(SystemExit) as stop:
        pn.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "could not record" in err
    assert err.count(out.name) == 1  # the error names no file, so the message names the run


def test_a_resume_reads_config_json_as_utf_8(tmp_path, monkeypatch):
    """As pilot 2's: the read asks for UTF-8, as write_config writes."""
    import pathlib

    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    out = pn.main(["--results-root", str(tmp_path), "--workers", "2"])
    encodings = []
    real = pathlib.Path.read_text

    def recording(self, *args, **kwargs):
        if self.name == "config.json":
            encodings.append(kwargs.get("encoding", args[0] if args else None))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", recording)
    pn.main(["--resume", str(out)])
    assert encodings and set(encodings) == {"utf-8-sig"}  # UTF-8, a byte order mark skipped


def test_a_config_nested_too_deep_to_read_is_a_command_error(tmp_path):
    """As pilot 2's: exit 2, not a traceback."""
    (tmp_path / "config.json").write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(SystemExit) as stop:
        pn.main(["--resume", str(tmp_path)])
    assert stop.value.code == 2


def test_a_config_nested_too_deep_by_the_second_read_is_a_command_error(
    tmp_path, monkeypatch, capsys
):
    """As pilot 2's: the record step's RecursionError, exit 2, the run named once."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    out = pn.main(["--results-root", str(tmp_path), "--workers", "2"])

    def too_deep(out_dir, workers):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(pn.block_scoring, "record_workers", too_deep)
    with pytest.raises(SystemExit) as stop:
        pn.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and err.count(out.name) == 1
    assert "could not record the workers" in err  # the record step's, not the first read's


def test_a_config_saved_with_a_byte_order_mark_still_reads(tmp_path, monkeypatch):
    """As pilot 2's: an editor's UTF-8 with BOM does not make the pilot look missing."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    out = pn.main(["--results-root", str(tmp_path), "--workers", "2"])
    config = out / "config.json"
    config.write_bytes(b"\xef\xbb\xbf" + config.read_bytes())
    assert pn.main(["--resume", str(out), "--workers", "3"]) == out
    assert json.loads(config.read_text(encoding="utf-8-sig"))["workers"] == 3  # recorded


def test_pilot_1_warns_once_about_a_table_a_spreadsheet_rewrote(tmp_path, caplog):
    """blocks.csv with a byte order mark: one warning in a run that reads it twice."""
    import logging

    pn.run(tmp_path, _population(), np.zeros(DIM), np.ones(DIM), _evaluate, _threads)
    blocks = tmp_path / pn.BLOCKS_FILE
    blocks.write_bytes(b"\xef\xbb\xbf" + blocks.read_bytes())
    with caplog.at_level(logging.WARNING, logger=pn.block_scoring.log.name):
        pn.run(tmp_path, _population(), np.zeros(DIM), np.ones(DIM), _evaluate, _threads)
    assert caplog.text.count(str(blocks)) == 1


def test_a_mark_added_while_pilot_1_runs_is_warned_about(tmp_path, caplog, monkeypatch):
    """Saved by a spreadsheet after the first read: the summary's read, the choice's, warns."""
    import logging

    blocks = tmp_path / pn.BLOCKS_FILE
    real_summary = pn.write_summary

    def saved_meanwhile(out_dir, warn=True):  # between the scoring and the summary
        blocks.write_bytes(b"\xef\xbb\xbf" + blocks.read_bytes())
        return real_summary(out_dir, warn=warn)

    monkeypatch.setattr(pn, "write_summary", saved_meanwhile)
    with caplog.at_level(logging.WARNING, logger=pn.block_scoring.log.name):
        pn.run(tmp_path, _population(), np.zeros(DIM), np.ones(DIM), _evaluate, _threads)
    assert caplog.text.count(str(blocks)) == 1


@pytest.mark.parametrize("recorded", [True, 0, -2, "8"])
def test_a_recorded_worker_count_out_of_range_names_pilot_1(tmp_path, monkeypatch, capsys,
                                                            recorded):
    """The pilot is there; one recorded value is wrong, and the refusal says so."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    out = pn.main(["--results-root", str(tmp_path), "--workers", "2"])
    config = json.loads((out / "config.json").read_text())
    config["workers"] = recorded
    (out / "config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit) as stop:
        pn.main(["--resume", str(out)])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "pilot 1 in" in err and "config.json's workers is" in err
    assert "holds no pilot" not in err


def test_a_table_gone_during_pilot_1_stops_it_naming_the_table(tmp_path, monkeypatch):
    """Removed between the scoring and the summary: stop, naming blocks.csv."""
    real_score = pn.block_scoring.score

    def removed_after_scoring(path, *args, **kwargs):
        real_score(path, *args, **kwargs)
        path.unlink()

    monkeypatch.setattr(pn.block_scoring, "score", removed_after_scoring)
    with pytest.raises(pn.block_scoring.TableError, match=pn.BLOCKS_FILE):
        pn.run(tmp_path, _population(), np.zeros(DIM), np.ones(DIM), _evaluate, _threads)


def test_pilot_1_reports_a_table_error_as_a_refusal(tmp_path, monkeypatch, capsys):
    """Exit 2 with the table's path, not a traceback."""
    monkeypatch.setattr(pn, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.setattr(pn, "_population_and_bounds",
                        lambda: (_population(), np.zeros(DIM), np.ones(DIM)))  # fmt: skip

    def gone(*args, **kwargs):
        raise pn.block_scoring.TableError("/x/blocks.csv went missing; restore it")

    monkeypatch.setattr(pn, "run", gone)
    with pytest.raises(SystemExit) as stop:
        pn.main(["--results-root", str(tmp_path)])
    assert stop.value.code == 2 and "blocks.csv went missing" in capsys.readouterr().err
