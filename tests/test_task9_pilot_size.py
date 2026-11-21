# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Task 9, pilot 2: population size and step size, at the S of pilot 1.

From doc/plan_Nordling2026_CovSynFireflyNordlingLab.md, Task 9: n in {24, 48, 96} x alpha_0 in
{0.05, 0.2}, a budget of n x G = 10,000 evaluations per run, G_plan = G, 2 runs per setting on
different optimisation streams; re-score each run's best archive candidates on 30 blocks of
one validation stream, and choose the setting with the lowest mean validated cost whose two
runs agree within their standard errors.

Decisions of 2026-11-21: G = floor((10,000 - n) / n), counting the initial evaluation; streams
1 and 2 for every setting; at most 10 distinct archive vectors per run, by ascending archive
mean, ties by entering the elite first; validation stream 202, blocks 0-29 to pick each run's
representative (screening) and blocks 30-59 to score the representatives (confirmation); two
runs agree when the mean of their block-by-block differences d over the confirmation blocks
is at most SD(d) / sqrt(30); among those, the lowest mean of the two representatives'
confirmation means; none agreeing chooses none, and the measurements go to the professor.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json

import numpy as np
import pytest

from covsyn.calibration import pilot_size as ps
from covsyn.calibration import seed_streams

DIM = 3


def test_twelve_runs_two_streams_per_setting():
    """Each n and alpha_0 on streams 1 and 2, in a fixed order."""
    settings = ps.settings()
    assert len(settings) == 12
    assert {(s.pop_size, s.alpha_0) for s in settings} == {
        (n, a) for n in (24, 48, 96) for a in (0.05, 0.2)}  # fmt: skip
    assert all({s.stream for s in settings if (s.pop_size, s.alpha_0) == (n, a)} == {1, 2}
               for n in (24, 48, 96) for a in (0.05, 0.2))  # fmt: skip
    assert settings == ps.settings()


@pytest.mark.parametrize("n, g", [(24, 415), (48, 207), (96, 103)])
def test_the_generations_spend_the_budget_with_the_initial_evaluation(n, g):
    """The initial n and n per generation: n + n x G <= 10,000, 9,984 for each n."""
    assert ps.generations(n) == g
    assert n + n * g <= 10_000 < n + n * (g + 1)


def test_each_run_has_its_own_name_from_its_setting():
    """One models directory per run, named by its n, alpha_0 and stream."""
    names = {ps.run_name("20261121-000000_abcdef0", s) for s in ps.settings()}
    assert len(names) == 12
    assert "20261121-000000_abcdef0_pilot-size-n24-a0p05-s1" in names


def test_a_run_is_the_optimiser_s_pilot_mode_with_its_setting(tmp_path):
    """G_plan = G by the pilot mode's default; S from pilot 1."""
    setting = ps.Setting(48, 0.2, 2)
    args = ps.optimiser_args(setting, 3_000, tmp_path / "r")
    assert args[:2] == ["-m", "covsyn.calibration.firefly_optimizer"]
    flags = dict(zip(args[2::2], args[3::2], strict=True))
    assert flags == {"--mode": "pilot", "--pop-size": "48", "--generations": "207",
                     "--simulations": "3000", "--alpha-0": "0.2", "--stream": "2",
                     "--run-dir": str(tmp_path / "r")}  # fmt: skip


def _archive(path, rows):
    """firefly_archive.txt v2: (entered, blocks, params..., mean), written as the optimiser does."""
    from covsyn.calibration import firefly_optimizer as fo

    fo.write_archive(path / "firefly_archive.txt", np.array(rows, dtype=float))


def test_a_run_s_candidates_are_its_ten_best_distinct_archive_vectors(tmp_path):
    """By ascending archive mean."""
    rows = [[g % 4, 1, g, g + 1, g + 2, 50.0 - g] for g in range(15)]
    _archive(tmp_path, rows)
    vectors = ps.archive_candidates(tmp_path / "firefly_archive.txt")
    assert len(vectors) == 10
    np.testing.assert_array_equal(vectors[0], [14, 15, 16])  # mean 36, the lowest
    np.testing.assert_array_equal(vectors[-1], [5, 6, 7])


def test_a_tie_goes_to_the_vector_that_entered_the_elite_first(tmp_path):
    """Equal means: the earlier entered generation first."""
    _archive(tmp_path, [[3, 1, 0.1, 0.1, 0.1, 2.0], [1, 1, 0.2, 0.2, 0.2, 2.0],
                        [2, 1, 0.3, 0.3, 0.3, 1.0]])  # fmt: skip
    vectors = ps.archive_candidates(tmp_path / "firefly_archive.txt")
    np.testing.assert_array_equal(vectors[:, 0], [0.3, 0.2, 0.1])


def test_an_archive_of_fewer_than_ten_gives_them_all(tmp_path):
    """A short run's archive is taken whole."""
    _archive(tmp_path, [[0, 1, 0.1, 0.2, 0.3, 4.0], [1, 1, 0.3, 0.2, 0.1, 3.0]])
    assert len(ps.archive_candidates(tmp_path / "firefly_archive.txt")) == 2


def test_an_archive_from_before_task_8_is_refused(tmp_path):
    """Its rows are snapshots, not elite vectors with means: no candidates to rank."""
    np.savetxt(tmp_path / "firefly_archive.txt", np.array([[0, 0.1, 0.2, 0.3, 4.0]]))
    with pytest.raises(ValueError, match="v2"):
        ps.archive_candidates(tmp_path / "firefly_archive.txt")


def test_the_screening_and_confirmation_blocks_do_not_share_a_seed():
    """Blocks 0-29 pick the representative, blocks 30-59 score it, at any S."""
    assert ps.VALIDATION_STREAM == 202
    assert (ps.SCREEN_BLOCKS, ps.CONFIRM_BLOCKS) == (range(0, 30), range(30, 60))
    for s in (300, 3_000, 10_000):
        screen = seed_streams.block_seeds(202, 29, s)
        confirm = seed_streams.block_seeds(202, 30, s)
        assert screen.stop <= confirm.start


def _confirmed(means, spread=1.0, seed=0, offsets=None):
    """Confirmation rows: each run's representative on the 30 blocks around its mean."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 3, 30)
    rows = []
    for setting in ps.settings():
        key = (setting.pop_size, setting.alpha_0)
        mean = means[key] + (offsets or {}).get((*key, setting.stream), 0.0)
        noise = rng.normal(0, spread, 30)
        for b, c in zip(ps.CONFIRM_BLOCKS, mean + shared + noise, strict=True):
            rows.append({"run": ps.run_name("x", setting), "pop_size": setting.pop_size,
                         "alpha_0": setting.alpha_0, "stream": setting.stream, "block": b,
                         "cost": float(c)})  # fmt: skip
    return rows


def _means(best=(48, 0.2)):
    means = {(n, a): 30.0 + n / 10 + a for n in (24, 48, 96) for a in (0.05, 0.2)}
    means[best] = 10.0
    return means


def test_two_runs_agree_when_their_mean_difference_is_within_its_paired_se():
    """With d stream 1 minus stream 2 block by block: |mean(d)| <= SD(d) / sqrt(30)."""
    rows = _confirmed(_means(), spread=1.0)
    table = ps.compare_settings(rows)
    assert len(table) == 6
    for line in table:
        d = np.array(
            [
                r["cost"]
                for r in rows
                if r["stream"] == 1
                and (r["pop_size"], r["alpha_0"]) == (line["pop_size"], line["alpha_0"])
            ]
        )
        d -= np.array(
            [
                r["cost"]
                for r in rows
                if r["stream"] == 2
                and (r["pop_size"], r["alpha_0"]) == (line["pop_size"], line["alpha_0"])
            ]
        )
        assert line["mean_difference"] == pytest.approx(d.mean())
        assert line["paired_se"] == pytest.approx(d.std(ddof=1) / np.sqrt(30))
        assert line["agree"] == (abs(d.mean()) <= d.std(ddof=1) / np.sqrt(30))


def test_the_setting_s_cost_is_the_mean_of_its_two_representatives():
    """Over both representatives' 30 confirmation blocks."""
    rows = _confirmed(_means())
    for line in ps.compare_settings(rows):
        costs = [
            r["cost"]
            for r in rows
            if (r["pop_size"], r["alpha_0"]) == (line["pop_size"], line["alpha_0"])
        ]
        assert line["mean_cost"] == pytest.approx(np.mean(costs))


def test_the_lowest_agreeing_setting_is_chosen():
    """With its G as G_plan, which production repeats (Task 8)."""
    rows = _confirmed(_means(best=(48, 0.2)), spread=1.0)
    choice = ps.choose(ps.compare_settings(rows))
    agreeing = [line for line in ps.compare_settings(rows) if line["agree"]]
    if (48, 0.2) in {(line["pop_size"], line["alpha_0"]) for line in agreeing}:
        assert (choice["pop_size"], choice["alpha_0"]) == (48, 0.2)
        assert choice["g_plan"] == ps.generations(48)


def test_a_lower_setting_whose_runs_disagree_is_passed_over():
    """Its two runs differ beyond their paired SE, so it is not chosen."""
    offsets = {(48, 0.2, 1): -5.0, (48, 0.2, 2): 5.0}  # far apart: they disagree
    rows = _confirmed(_means(best=(48, 0.2)), spread=0.1, offsets=offsets)
    table = ps.compare_settings(rows)
    assert not next(line for line in table if (line["pop_size"], line["alpha_0"]) == (48, 0.2))[
        "agree"]  # fmt: skip
    choice = ps.choose(table)
    assert choice is None or (choice["pop_size"], choice["alpha_0"]) != (48, 0.2)


def test_no_setting_agreeing_chooses_none():
    """Then the measurements go to the professor (decision of 2026-11-21)."""
    offsets = {(n, a, 1): -5.0 for n in (24, 48, 96) for a in (0.05, 0.2)}
    rows = _confirmed(_means(), spread=0.1, offsets=offsets)
    assert ps.choose(ps.compare_settings(rows)) is None


# --- the whole pilot, with stand-ins for the optimiser and the objective -----------------------


def _evaluate(P, seed_offset):
    return float(np.sum(P) * 10 + (seed_offset % 101) / 50), {}, {}


def _threads(simulations):
    return concurrent.futures.ThreadPoolExecutor(max_workers=4)


class _Optimiser:
    """Stands in for covsyn-optimise: writes a finished run's archive, or fails on request."""

    def __init__(self, fail_on=None, kill_after=None):
        self.calls, self.fail_on, self.kill_after = [], fail_on, kill_after
        self.envs = []

    def __call__(self, args, output, append, env):
        flags = dict(zip(args[2::2], args[3::2], strict=False))
        self.calls.append(flags)
        if self.kill_after is not None and len(self.calls) > self.kill_after:
            raise KeyboardInterrupt
        run = ps.Path(flags.get("--run-dir") or flags["--resume"])
        if self.fail_on is not None and self.fail_on in str(run):
            return 1
        run.mkdir(parents=True, exist_ok=True)
        # a resume names only the run; its setting is the run's own
        n, a, s = (int(flags.get("--pop-size", 24)), float(flags.get("--alpha-0", 0.05)),
                   int(flags.get("--stream", 1)))  # fmt: skip
        rng = np.random.default_rng(n * 10 + s)
        rows = [[g, 1, *rng.random(DIM) * a, 5.0 + rng.random()] for g in range(12)]
        _archive(run, rows)
        (run / "config.json").write_text(json.dumps({"settings": {"pop_size": n}}))
        (run / "firefly_result.txt").write_text("done\n")
        (run / "firefly_worst.txt").write_text("done\n")  # the optimiser's last file
        self.envs.append(dict(env))
        return 0


def _pilot(tmp_path, optimiser, simulations=300, workers=2):
    out = tmp_path / "tables" / "p"
    out.mkdir(parents=True, exist_ok=True)
    return ps.run(
        out, tmp_path / "models", "p", simulations, optimiser, _evaluate, _threads, workers=workers
    ), out


def test_the_pilot_runs_every_setting_then_screens_and_confirms(tmp_path):
    """Twelve runs, then 10 candidates x 30 blocks per run, then 30 blocks per representative."""
    optimiser = _Optimiser()
    summary, out = _pilot(tmp_path, optimiser)
    assert len(optimiser.calls) == 12
    with open(out / ps.SCREEN_FILE, newline="") as f:
        screen = list(csv.DictReader(f))
    assert len(screen) == 12 * 10 * 30  # 10 candidates per run, 30 blocks
    with open(out / ps.CONFIRM_FILE, newline="") as f:
        confirm = list(csv.DictReader(f))
    assert len(confirm) == 12 * 30  # one representative per run
    assert {int(r["block"]) for r in confirm} == set(ps.CONFIRM_BLOCKS)
    choice = json.loads((out / ps.CHOICE_FILE).read_text())
    assert "chosen" in choice and choice["simulations"] == 300
    assert (out / ps.SETTINGS_FILE).is_file()


def test_each_run_s_representative_is_its_lowest_screened_candidate(tmp_path):
    """Lowest screening mean; a tie to the lower candidate index."""
    summary, out = _pilot(tmp_path, _Optimiser())
    with open(out / ps.SCREEN_FILE, newline="") as f:
        screen = list(csv.DictReader(f))
    with open(out / ps.RUNS_FILE, newline="") as f:
        runs_table = list(csv.DictReader(f))
    for line in runs_table:
        means = {}
        for r in screen:
            if r["run"] == line["run"]:
                means.setdefault(int(r["candidate"]), []).append(float(r["cost"]))
        best = min(means, key=lambda c: (np.mean(means[c]), c))
        assert int(line["representative"]) == best


def test_a_failed_run_stops_the_pilot_before_any_scoring(tmp_path):
    """A run that did not finish has no archive to score; the pilot says which."""
    with pytest.raises(RuntimeError, match="n48-a0p2-s2"):
        _pilot(tmp_path, _Optimiser(fail_on="n48-a0p2-s2"))
    assert not (tmp_path / "tables" / "p" / ps.SCREEN_FILE).exists()


def test_a_resumed_pilot_skips_finished_runs_and_resumes_checkpointed_ones(tmp_path):
    """Finished runs are kept; a run with a checkpoint continues from it."""
    with pytest.raises(KeyboardInterrupt):
        _pilot(tmp_path, _Optimiser(kill_after=3))
    # the fourth run was killed after its first checkpoint
    fourth = tmp_path / "models" / ps.run_name("p", ps.settings()[3])
    (fourth / "checkpoints").mkdir(parents=True)
    (fourth / "checkpoints" / "generation_0000.npz").write_text("x")
    optimiser = _Optimiser()
    _pilot(tmp_path, optimiser)
    assert len(optimiser.calls) == 12 - 3
    assert optimiser.calls[0].get("--resume") == str(fourth)


def test_a_run_killed_before_its_first_checkpoint_starts_again(tmp_path):
    """Its claim would refuse a fresh start, so its partial output goes, as the pipeline does."""
    first = tmp_path / "models" / ps.run_name("p", ps.settings()[0])
    first.mkdir(parents=True)
    (first / "config.json").write_text('{"claimed": true}\n')
    optimiser = _Optimiser()
    _pilot(tmp_path, optimiser)
    assert optimiser.calls[0]["--run-dir"] == str(first)


def test_a_resumed_pilot_scores_nothing_twice(tmp_path):
    """Each scored block is in its table, so a resume skips it."""
    summary, out = _pilot(tmp_path, _Optimiser())
    before = {name: (out / name).read_text() for name in (ps.SCREEN_FILE, ps.CONFIRM_FILE)}
    calls = []

    def counts(P, seed_offset):
        calls.append(1)
        return _evaluate(P, seed_offset)

    ps.run(out, tmp_path / "models", "p", 300, _Optimiser(), counts, _threads, workers=2)
    assert calls == []
    assert {name: (out / name).read_text() for name in before} == before


def test_the_runs_table_gives_each_run_s_evaluations(tmp_path):
    """To check the budget: the last checkpoint's count, n + n x G at most."""
    summary, out = _pilot(tmp_path, _Optimiser())
    with open(out / ps.RUNS_FILE, newline="") as f:
        lines = list(csv.DictReader(f))
    assert {line["generations"] for line in lines} == {"415", "207", "103"}
    assert all("evaluations" in line for line in lines)


def test_the_command_takes_s_from_pilot_1(tmp_path, monkeypatch):
    """--noise DIR: pilot 1's directory, whose choice.json gives S; config.json records both."""
    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "choice.json").write_text(json.dumps({"chosen_simulations": 3000}))
    seen = {}
    monkeypatch.setattr(ps, "run", lambda out, models, prefix, s, *rest, **kw: seen.update(s=s)
                        or {})  # fmt: skip
    monkeypatch.setattr(ps, "_evaluator", lambda workers: (_evaluate, _threads))
    out = ps.main(["--noise", str(noise), "--results-root", str(tmp_path)])
    assert seen["s"] == 3000
    config = json.loads((out / "config.json").read_text())["settings"]
    assert config["simulations"] == 3000 and config["noise"] == str(noise)


def test_the_command_refuses_a_pilot_1_that_chose_no_s(tmp_path):
    """No S met the criterion: the professor decides first (decision of 2026-11-21)."""
    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "choice.json").write_text(json.dumps({"chosen_simulations": None}))
    with pytest.raises(SystemExit):
        ps.main(["--noise", str(noise), "--results-root", str(tmp_path)])


def test_every_run_uses_the_pilot_s_workers(tmp_path):
    """--workers reaches the optimiser runs too, as COVSYN_WORKERS, resumed ones included."""
    from covsyn.runtime import hardware

    optimiser = _Optimiser()
    _pilot(tmp_path, optimiser, workers=3)
    assert optimiser.envs and all(env == {hardware.ENV_VARIABLE: "3"} for env in optimiser.envs)


def test_a_run_without_its_last_file_is_not_finished(tmp_path):
    """firefly_worst.txt comes last; a run killed before it is continued, not skipped."""
    optimiser = _Optimiser()
    _pilot(tmp_path, optimiser)
    first = tmp_path / "models" / ps.run_name("p", ps.settings()[0])
    (first / "firefly_worst.txt").unlink()
    (first / "checkpoints").mkdir()
    (first / "checkpoints" / "generation_0000.npz").write_text("x")
    again = _Optimiser()
    _pilot(tmp_path, again)
    assert [c.get("--resume") for c in again.calls] == [str(first)]


def test_a_relative_resume_runs_the_optimiser_on_whole_paths(tmp_path, monkeypatch):
    """The optimiser runs from the repository root, so a relative path would point elsewhere."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}}))
    seen = {}
    monkeypatch.setattr(ps, "run", lambda o, models, prefix, s, *rest, **kw: seen.update(
        out=o, models=models) or {})  # fmt: skip
    monkeypatch.setattr(ps, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.chdir(tmp_path)
    ps.main(["--resume", "results/tables/20261121-000000_abcdef0_pilot-size"])
    assert seen["out"].is_absolute() and seen["models"] == tmp_path / "results" / "models"


def test_the_command_refuses_a_pilot_1_with_failed_evaluations(tmp_path):
    """A failed evaluation's 1e6 is no cost, so its S is not pilot 2's to run."""
    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "choice.json").write_text(
        json.dumps({"chosen_simulations": 300, "failed_evaluations": 2})
    )
    with pytest.raises(SystemExit):
        ps.main(["--noise", str(noise), "--results-root", str(tmp_path)])


def test_the_choice_reports_the_alpha_end_the_runs_used():
    """The optimiser's default, which the pilot mode leaves in place."""
    from covsyn.calibration import step_schedule

    rows = _confirmed(_means(), spread=1.0)
    table = [dict(line, agree=True) for line in ps.compare_settings(rows)]
    assert ps.choose(table)["alpha_end"] == step_schedule.DEFAULT_ALPHA_END


def test_a_resume_keeps_the_workers_the_pilot_started_with(tmp_path, monkeypatch):
    """config.json's workers, unless --workers is given again."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 8}))
    seen = {}
    monkeypatch.setattr(ps, "run", lambda o, models, prefix, s, *rest, **kw: seen.update(kw)
                        or {})  # fmt: skip
    monkeypatch.setattr(ps, "_evaluator", lambda workers: seen.update(pools=workers)
                        or (_evaluate, _threads))  # fmt: skip
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    ps.main(["--resume", str(out)])
    assert seen["workers"] == 8 and seen["pools"] == 8
    ps.main(["--resume", str(out), "--workers", "3"])
    assert seen["workers"] == 3 and seen["pools"] == 3
    assert json.loads((out / "config.json").read_text())["workers"] == 3  # recorded
    monkeypatch.setenv("COVSYN_WORKERS", "5")  # the documented order: before the record
    ps.main(["--resume", str(out)])
    assert seen["workers"] == 5


@pytest.mark.parametrize("recorded", [True, 0, -2, "8"])
def test_a_recorded_worker_count_out_of_range_is_refused(tmp_path, monkeypatch, recorded):
    """Not a traceback, nor two different counts for the runs and the pools."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(
        json.dumps({"settings": {"simulations": 300}, "workers": recorded})
    )
    monkeypatch.delenv("COVSYN_WORKERS", raising=False)
    monkeypatch.setattr(ps, "run", lambda *a, **kw: {})
    with pytest.raises(SystemExit):
        ps.main(["--resume", str(out)])


def test_a_resume_through_dot_dot_finds_the_results_root(tmp_path, monkeypatch):
    """results/tables/other/../<run-id>: the runs are still results/models."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (tmp_path / "results" / "tables" / "other").mkdir()
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))
    seen = {}
    monkeypatch.setattr(ps, "run", lambda o, models, prefix, s, *rest, **kw: seen.update(
        models=models, prefix=prefix) or {})  # fmt: skip
    monkeypatch.setattr(ps, "_evaluator", lambda workers: (_evaluate, _threads))
    monkeypatch.chdir(tmp_path / "results" / "tables" / "other")
    ps.main(["--resume", "../20261121-000000_abcdef0_pilot-size"])
    assert seen["models"] == tmp_path / "results" / "models"
    assert seen["prefix"] == "20261121-000000_abcdef0"


def test_a_resume_of_a_link_to_the_run_directory_is_refused(tmp_path, monkeypatch):
    """Its name is not the run's, so its prefix and root would be wrong: refused, not guessed."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))
    (tmp_path / "latest").symlink_to(out)
    monkeypatch.setattr(ps, "run", lambda *a, **kw: {})
    with pytest.raises(SystemExit):
        ps.main(["--resume", str(tmp_path / "latest")])


def test_a_resume_through_a_linked_tables_directory_keeps_the_results_root(tmp_path, monkeypatch):
    """results/tables a link elsewhere: the runs are still results/models."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "20261121-000000_abcdef0_pilot-size").mkdir(parents=True)
    (elsewhere / "20261121-000000_abcdef0_pilot-size" / "config.json").write_text(
        json.dumps({"settings": {"simulations": 300}, "workers": 2})
    )
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "tables").symlink_to(elsewhere)
    seen = {}
    monkeypatch.setattr(ps, "run", lambda o, models, prefix, s, *rest, **kw: seen.update(
        models=models) or {})  # fmt: skip
    monkeypatch.setattr(ps, "_evaluator", lambda workers: (_evaluate, _threads))
    ps.main(["--resume", str(tmp_path / "results" / "tables" /
                             "20261121-000000_abcdef0_pilot-size")])  # fmt: skip
    assert seen["models"] == tmp_path / "results" / "models"


@pytest.mark.parametrize("name", ["latest_pilot-size", "20261121-000000_abcdef0_pilot-size"])
def test_a_resume_that_is_not_a_pilot_s_own_directory_is_refused(tmp_path, monkeypatch, name):
    """A hand-made name, or a link in another tables directory: its runs' place is unknown."""
    real = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    real.mkdir(parents=True)
    (real / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))
    other = tmp_path / "backup" / "tables"
    other.mkdir(parents=True)
    (other / name).symlink_to(real)
    monkeypatch.setattr(ps, "run", lambda *a, **kw: {})
    with pytest.raises(SystemExit):
        ps.main(["--resume", str(other / name)])


def test_a_config_that_cannot_be_written_is_a_command_error(tmp_path, monkeypatch, capsys):
    """Read-only config.json, new workers: refused with a message, not a traceback."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))

    def refused(out_dir, workers):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ps.block_scoring, "record_workers", refused)
    monkeypatch.setattr(ps, "run", lambda *a, **kw: {})
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "Permission denied" in err and "holds no pilot" not in err


def test_a_config_without_settings_is_named_as_such(tmp_path, monkeypatch, capsys):
    """The refusal names the missing key by its exception, not only its name."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"workers": 2}))
    with pytest.raises(SystemExit):
        ps.main(["--resume", str(out)])
    assert "KeyError" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name",
    [
        "latest_pilot-size",
        "20261121-000000_ABCDEF0_pilot-size",
        "20261121-000000_abcdef0_pilot-noise",
        "\u0662\u0660\u0662\u0666\u0661\u0661\u0662\u0661-000000_abcdef0_pilot-size",
    ],
)
def test_a_hand_named_pilot_directory_is_refused(tmp_path, monkeypatch, capsys, name):
    """A real directory, not a link, whose name runs.make_run_id would not give a pilot 2."""
    out = tmp_path / "results" / "tables" / name
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))
    monkeypatch.setattr(ps, "run", lambda *a, **kw: {})
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out)])
    assert stop.value.code == 2
    assert "is not results/tables/<run-id> of a pilot 2" in capsys.readouterr().err


from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(max_examples=200, deadline=None)
@given(
    when=st.datetimes(
        min_value=__import__("datetime").datetime(2000, 1, 1),
        max_value=__import__("datetime").datetime(9999, 12, 31),
    ),
    commit=st.one_of(st.none(), st.text("0123456789abcdef", min_size=7, max_size=40)),
    dirty=st.booleans(),
)
def test_every_pilot_2_run_id_is_recognised(when, commit, dirty):
    """Whatever time, commit and state runs.make_run_id is given."""
    from covsyn.runtime import runs

    name = runs.make_run_id("pilot-size", when=when, commit=commit or "", dirty=dirty)
    assert ps.RUN_ID.fullmatch(name)


VALID = (
    "20261121-093015_0123abc_pilot-size",
    "20261121-093015_nocommit_pilot-size",
    "20261121-093015_0123abc-dirty_pilot-size",
)
CHANGES = ["\u0663", "\uff13", "A", "F", "_", "-", " ", "x", "0", "a", "-dirty"]


def _could_make(name):
    """Whether runs.make_run_id("pilot-size", ...) can return name, read from its format.

    <YYYYMMDD-HHMMSS>_<commit[:7] or nocommit>[-dirty]_<config name>, with the stamp a
    datetime's strftime and the commit lowercase hex, as git prints it.
    """
    from datetime import datetime

    stamp, _, rest = name.partition("_")
    commit, _, config = rest.partition("_")
    try:
        if datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%Y%m%d-%H%M%S") != stamp:
            return False
    except ValueError:
        return False
    commit = commit.removesuffix("-dirty")
    hex_commit = len(commit) == 7 and all(c in "0123456789abcdef" for c in commit)
    return config == "pilot-size" and (hex_commit or commit == "nocommit")


@settings(max_examples=500, deadline=None)
@given(
    base=st.sampled_from(VALID),
    at=st.integers(min_value=0, max_value=60),
    change=st.sampled_from(CHANGES),
    how=st.sampled_from(["swap", "insert", "delete"]),
)
# Always tried: the separator in the config name, an uppercase commit, a non-ASCII digit.
@example(base=VALID[0], at=VALID[0].index("-size"), change="x", how="swap")
@example(base=VALID[0], at=VALID[0].index("_") + 1, change="A", how="swap")
@example(base=VALID[0], at=0, change="\u0663", how="swap")
def test_a_name_one_edit_off_is_refused(base, at, change, how):
    """A valid name with one character swapped, inserted (the end included) or deleted.

    Refused unless runs.make_run_id could still have made it; a stamp that is no date may pass,
    since the name check guards against hand-made names, not impossible times.
    """
    import re

    at = at % (len(base) + (how == "insert"))
    name = {
        "swap": base[:at] + change + base[at + 1 :],
        "insert": base[:at] + change + base[at:],
        "delete": base[:at] + base[at + 1 :],
    }[how]
    if _could_make(name):
        assert ps.RUN_ID.fullmatch(name), name
    elif not ps.RUN_ID.fullmatch(name):
        return
    else:  # accepted though make_run_id could not make it: only a stamp that is no date
        stamp, _, rest = name.partition("_")
        assert re.fullmatch(r"[0-9]{8}-[0-9]{6}", stamp), name
        assert _could_make("20261121-093015_" + rest), name  # the stamp is its only defect


def test_make_run_id_s_names_are_recognised_by_the_independent_reading():
    """The reference above agrees with make_run_id itself."""
    from datetime import datetime

    from covsyn.runtime import runs

    for commit, dirty in (("0123abcdef", False), ("", False), ("fedcba9876543", True)):
        name = runs.make_run_id("pilot-size", when=datetime(2026, 11, 21, 9, 30, 15),
                                commit=commit, dirty=dirty)  # fmt: skip
        assert _could_make(name) and ps.RUN_ID.fullmatch(name), name


def test_a_read_only_config_says_so_not_that_the_pilot_is_missing(tmp_path, monkeypatch, capsys):
    """The pilot is there; only its config.json cannot record the new workers."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))
    (out / "config.json").chmod(0o444)
    monkeypatch.setattr(ps, "run", lambda *a, **kw: {})
    try:
        with pytest.raises(SystemExit):
            ps.main(["--resume", str(out), "--workers", "3"])
    finally:
        (out / "config.json").chmod(0o644)
    err = capsys.readouterr().err
    assert "read-only" in err and "holds no pilot" not in err


def test_a_config_whose_rewrite_fails_is_not_called_missing(tmp_path, monkeypatch, capsys):
    """A failed write (a full disk, a file system that takes nothing) says so."""
    import errno

    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))

    def failing(out_dir, workers):
        raise OSError(errno.EIO, "short write", str(out_dir / "config.json"))

    monkeypatch.setattr(ps.block_scoring, "record_workers", failing)
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2
    assert "could not record" in err and "holds no pilot" not in err


def test_a_config_json_cannot_write_back_is_a_command_error(tmp_path, capsys):
    """NaN reads but does not write as JSON: an exit-2 refusal, not a traceback."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text('{"settings": {"simulations": 300, "x": NaN}, "workers": 2}')
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "could not record" in err
    assert out.name in err  # the error names no file, so the message names the run


def test_a_real_short_write_is_reported_once_with_config_json_kept(tmp_path, monkeypatch, capsys):
    """Through main and the real record step: exit 2, the path once, config.json unchanged."""
    import os

    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300}, "workers": 2}))
    monkeypatch.setattr(os, "write", lambda fd, data: 0)
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out), "--workers", "3"])
    monkeypatch.undo()
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "could not record" in err
    assert err.count(out.name) == 1  # the run's own name, however the OS writes paths
    assert "config.json stays as it was" in err
    assert json.loads((out / "config.json").read_text())["workers"] == 2


def test_a_resume_reads_config_json_as_utf_8_whatever_the_locale(tmp_path, monkeypatch):
    """write_config writes UTF-8; the reads say so rather than take the locale's encoding.

    The default encoding is set in C, so the test checks the reads ask for UTF-8.
    """
    import pathlib

    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(
        json.dumps({"settings": {"simulations": 300}, "workers": 2}), encoding="utf-8"
    )
    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "choice.json").write_text(json.dumps({"chosen_simulations": 300}))
    encodings: dict[str, list] = {}
    real = pathlib.Path.read_text

    def recording(self, *args, **kwargs):
        if self.name in ("config.json", "choice.json"):  # every read, main's first among them
            encodings.setdefault(self.name, []).append(kwargs.get("encoding", args[0] if args
                                                                  else None))  # fmt: skip
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", recording)
    monkeypatch.setattr(ps, "run", lambda *a, **kw: {})
    monkeypatch.setattr(ps, "_evaluator", lambda workers: (_evaluate, _threads))
    ps.main(["--resume", str(out)])
    ps.main(["--noise", str(noise), "--results-root", str(tmp_path)])
    assert set(encodings) == {"config.json", "choice.json"}
    assert all(set(reads) == {"utf-8"} for reads in encodings.values()), encodings


def test_a_config_turned_into_a_list_is_named_once(tmp_path, monkeypatch, capsys):
    """Changed between the two reads: record_workers' refusal and the pilot name it once."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(
        json.dumps({"settings": {"simulations": 300}, "workers": 2})
    )
    real = ps.block_scoring.record_workers

    def changed_meanwhile(out_dir, workers):
        (out_dir / "config.json").write_text("[1, 2]")
        return real(out_dir, workers)

    monkeypatch.setattr(ps.block_scoring, "record_workers", changed_meanwhile)
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2
    assert "not a JSON object" in err and err.count(out.name) == 1


def test_the_archive_is_read_as_utf_8_header_and_rows(tmp_path, monkeypatch):
    """Both the header check and np.loadtxt, as the optimiser writes it (Task 9 review)."""
    import builtins

    _archive(tmp_path, [[0, 1, 0.1, 0.2, 0.3, 4.0]])
    seen, opened = [], []
    real = np.loadtxt
    monkeypatch.setattr(np, "loadtxt", lambda *a, **kw: seen.append(kw.get("encoding"))
                        or real(*a, **kw))  # fmt: skip
    real_open = builtins.open

    def recording(file, mode="r", *args, **kwargs):
        if str(file).endswith("firefly_archive.txt") and "b" not in mode:
            opened.append(kwargs.get("encoding", args[1] if len(args) > 1 else None))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording)
    ps.archive_candidates(tmp_path / "firefly_archive.txt")
    monkeypatch.undo()
    assert seen == ["utf-8"] and opened and set(opened) == {"utf-8"}  # rows and header


def test_revalidation_reads_the_archive_as_utf_8(tmp_path, monkeypatch):
    """The other reader of the archive, the same way."""
    from covsyn.calibration import revalidation

    _archive(tmp_path, [[0, 1, 0.1, 0.2, 0.3, 4.0]])
    seen = []
    real = np.loadtxt
    monkeypatch.setattr(np, "loadtxt", lambda *a, **kw: seen.append(kw.get("encoding"))
                        or real(*a, **kw))  # fmt: skip
    revalidation.read_archive(tmp_path / "firefly_archive.txt")
    assert seen == ["utf-8"]


def test_the_optimiser_writes_the_archive_as_utf_8(tmp_path):
    """np.savetxt's default is latin1, which cannot hold a header the readers read as UTF-8."""
    from covsyn.calibration import firefly_optimizer as fo

    path = tmp_path / "firefly_archive.txt"
    fo.write_archive(path, np.array([[0, 1, 0.1, 0.2, 4.0]]), header="archive \u03c3 \u2265")
    assert path.read_text(encoding="utf-8").startswith("# archive \u03c3 \u2265")
    np.testing.assert_allclose(np.loadtxt(path, ndmin=2, encoding="utf-8"),
                               [[0, 1, 0.1, 0.2, 4.0]])  # fmt: skip


def test_a_config_nested_too_deep_to_read_is_a_command_error(tmp_path):
    """json.loads raises RecursionError, not ValueError: still exit 2, not a traceback."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out)])
    assert stop.value.code == 2


def test_a_config_nested_too_deep_by_the_second_read_is_a_command_error(
    tmp_path, monkeypatch, capsys
):
    """Rewritten between the reads: the record step's RecursionError, the run named once."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    (out / "config.json").write_text(
        json.dumps({"settings": {"simulations": 300}, "workers": 2})
    )

    def too_deep(out_dir, workers):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(ps.block_scoring, "record_workers", too_deep)
    with pytest.raises(SystemExit) as stop:
        ps.main(["--resume", str(out), "--workers", "3"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and err.count(out.name) == 1
    assert "could not record the workers" in err  # the record step's, not the first read's


def test_a_pilot_1_choice_nested_too_deep_is_a_command_error(tmp_path):
    """--noise reads choice.json the same way: exit 2, not a traceback."""
    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "choice.json").write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(SystemExit) as stop:
        ps.main(["--noise", str(noise), "--results-root", str(tmp_path)])
    assert stop.value.code == 2


def test_the_optimiser_s_run_writes_its_archive_through_write_archive(tmp_path, monkeypatch):
    """So the run's archive is UTF-8, as write_archive writes it."""
    from covsyn.calibration import firefly_optimizer as fo
    from tests import firefly_standins as standins

    written = []
    real = fo.write_archive
    monkeypatch.setattr(fo, "write_archive", lambda path, rows, *a, **kw: written.append(path)
                        or real(path, rows, *a, **kw))  # fmt: skip
    monkeypatch.setattr(fo, "result_path", tmp_path, raising=False)
    engine = fo.Firefly(pop_size=3, alpha_0=0.2, betamin=1, gamma=0.03, seed=1)
    engine.firefly(function=standins.objective, dim=208, lb=np.zeros(208), ub=np.ones(208),
                   demographic_parameters=None, max_generations=2, max_workers=2,
                   Cheng_contact_array=None, Cheng_attack_rate=None, norm_weights=None)  # fmt: skip
    assert written == [tmp_path / "firefly_archive.txt"]
