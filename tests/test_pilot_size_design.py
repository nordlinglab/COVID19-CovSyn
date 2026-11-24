# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pilot 2 with a design of its own: the grid, the streams and the warm starts (2026-11-24).

Pilot 2 as planned (n in {24, 48, 96} x alpha_0 in {0.05, 0.2} on streams 1 and 2, CovSyn
9792f7f) returned run 19, a row of the initial population, from 11 of its 12 runs, so its
agreement rule compared nothing. Its best setting lay at the corner of the grid (smallest n,
largest alpha_0), and two streams could not measure how much runs differ. User decision
2026-11-24: rerun on n in {12, 24, 48} x alpha_0 in {0.2, 0.4}, four streams each, without the
warm starts. Each run's representative is scored on the confirmation blocks; a setting's cost is
the mean of its runs' means, and the lowest is chosen (user decision 2026-11-24, as the OpenAI
agent advised: production uses one search, so the mean cost is what it targets; a rule that let
the SD of four runs decide could swap a better setting out). The SD between runs is reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from covsyn.calibration import pilot_size as ps

RERUN = ps.Design(pop_sizes=(12, 24, 48), alpha_0s=(0.2, 0.4), streams=(1, 2, 3, 4),
                  without_warm_starts=True)  # fmt: skip


def test_the_plan_is_the_default_design():
    """The planned grid, two streams, the warm starts kept, the agreement rule."""
    assert ps.PLAN == ps.Design((24, 48, 96), (0.05, 0.2), (1, 2), False)
    assert ps.PLAN.rule == "agreement"
    assert ps.settings() == ps.settings(ps.PLAN)


def test_more_than_two_streams_use_the_mean_rule():
    """Two streams can only agree or not; three or more are compared by their mean."""
    assert RERUN.rule == "mean"
    assert ps.Design((24,), (0.2,), (1, 2, 3), False).rule == "mean"


@pytest.mark.parametrize("bad", [
    {"pop_sizes": ()}, {"alpha_0s": ()}, {"streams": (1,)}, {"streams": (1, 1, 2)},
    {"pop_sizes": (1,)}, {"pop_sizes": (24, 24)}, {"alpha_0s": (0.0,)}, {"streams": (0, 1, 2)},
    {"streams": (1, 100, 2)}, {"alpha_0s": (float("nan"),)}, {"alpha_0s": (float("inf"),)},
])  # fmt: skip
def test_a_design_that_cannot_run_is_refused(bad):
    """Empty or repeated values, one stream, n below 2, alpha_0 not positive, no stream."""
    with pytest.raises(ValueError):
        ps.Design(**{**RERUN._asdict(), **bad}).check()


def test_the_rerun_has_twenty_four_runs_in_order():
    """Each n, each alpha_0, each stream: 3 x 2 x 4."""
    runs = ps.settings(RERUN)
    assert len(runs) == 24
    assert runs[0] == ps.Setting(12, 0.2, 1)
    assert runs[-1] == ps.Setting(48, 0.4, 4)


def test_n_of_12_runs_832_generations():
    """G = floor((10,000 - n) / n)."""
    assert ps.Setting(12, 0.2, 1).generations == 832


def test_the_optimiser_is_told_to_leave_the_warm_starts_out(tmp_path):
    """Only when the design says so."""
    setting = ps.Setting(12, 0.4, 3)
    assert "--without-warm-starts" in ps.optimiser_args(setting, 3000, tmp_path, RERUN)
    assert "--without-warm-starts" not in ps.optimiser_args(setting, 3000, tmp_path, ps.PLAN)
    args = ps.optimiser_args(setting, 3000, tmp_path, RERUN)
    assert args[args.index("--run-dir") + 1] == str(tmp_path)
    assert args[-1] == "--without-warm-starts"  # last: every flag before it keeps its value


def test_the_design_round_trips_through_its_record():
    """config.json keeps it as a dict, from which a resume rebuilds it."""
    assert ps.Design.from_record(RERUN.record()) == RERUN
    assert ps.Design.from_record(None) == ps.PLAN  # a pilot from before the design


def _rows(means, sds, k=4, blocks=30, seed=0):
    """Confirmation rows: run s of setting (n, a) has mean means[(n, a)] + its run effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for (n, a), m in means.items():
        spread = np.linspace(-1, 1, k)
        effects = spread / np.std(spread, ddof=1) * sds[(n, a)]
        for s, e in zip(range(1, k + 1), effects, strict=True):
            for b, c in enumerate(m + e + rng.normal(0, 0.01, blocks)):
                rows.append({"pop_size": n, "alpha_0": a, "stream": s, "block": 30 + b,
                             "cost": float(c)})  # fmt: skip
    return rows


def test_a_setting_s_cost_and_spread_come_from_its_runs_means():
    """Mean of the run means, and their SD with ddof 1."""
    design = ps.Design((12, 24), (0.2,), (1, 2, 3, 4), True)
    rows = _rows({(12, 0.2): 10.0, (24, 0.2): 20.0}, {(12, 0.2): 2.0, (24, 0.2): 1.0})
    table = {(line["pop_size"], line["alpha_0"]): line
             for line in ps.compare_settings(rows, design)}
    for key, sd in (((12, 0.2), 2.0), ((24, 0.2), 1.0)):
        runs = [np.mean([r["cost"] for r in rows if (r["pop_size"], r["alpha_0"]) == key
                         and r["stream"] == s]) for s in (1, 2, 3, 4)]  # fmt: skip
        assert table[key]["mean_cost"] == pytest.approx(np.mean(runs))
        assert table[key]["run_sd"] == pytest.approx(np.std(runs, ddof=1))
        assert table[key]["run_sd"] == pytest.approx(sd, rel=0.05)
        assert table[key]["runs"] == 4


def test_a_clearly_lowest_setting_is_chosen():
    """10 against 20, both with run SD 1: far beyond the SE of the difference."""
    design = ps.Design((12, 24), (0.2,), (1, 2, 3, 4), True)
    rows = _rows({(12, 0.2): 10.0, (24, 0.2): 20.0}, {(12, 0.2): 1.0, (24, 0.2): 1.0})
    choice = ps.choose(ps.compare_settings(rows, design), design)
    assert (choice["pop_size"], choice["alpha_0"]) == (12, 0.2)
    assert choice["g_plan"] == choice["generations"] == 832


def test_the_lowest_mean_wins_even_when_less_steady():
    """10 with run SD 6 against 11 with SD 1: the mean decides; the SDs are only reported."""
    design = ps.Design((12, 24), (0.2,), (1, 2, 3, 4), True)
    rows = _rows({(12, 0.2): 10.0, (24, 0.2): 11.0}, {(12, 0.2): 6.0, (24, 0.2): 1.0})
    table = ps.compare_settings(rows, design)
    choice = ps.choose(table, design)
    assert (choice["pop_size"], choice["alpha_0"]) == (12, 0.2)
    assert {line["pop_size"]: round(line["run_sd"]) for line in table} == {12: 6, 24: 1}


def test_the_planned_design_keeps_its_agreement_rule():
    """The plan's tables and choice are as before the design existed."""
    rows = [{"pop_size": n, "alpha_0": a, "stream": s, "block": b, "cost": 10.0 + n + a}
            for n in (24, 48, 96) for a in (0.05, 0.2) for s in (1, 2) for b in range(30, 60)]
    table = ps.compare_settings(rows)
    assert all("agree" in line for line in table)
    assert ps.choose(table)["pop_size"] == 24


# --- the design through the whole pilot and its command -----------------------------------------

SMALL = ps.Design((12, 24), (0.2,), (1, 2, 3), True)


def test_the_pilot_runs_the_design_s_settings_and_chooses_by_its_rule(tmp_path):
    """Six runs, each without the warm starts; the spread table and its choice."""
    from tests.test_task9_pilot_size import _evaluate, _Optimiser, _threads

    optimiser = _Optimiser()
    out = tmp_path / "tables" / "p"
    out.mkdir(parents=True)
    summary = ps.run(out, tmp_path / "models", "p", 300, optimiser, _evaluate, _threads,
                     workers=2, design=SMALL)  # fmt: skip
    assert len(optimiser.calls) == 6
    assert {(c["--pop-size"], c["--stream"]) for c in optimiser.calls} == {
        (str(n), str(s)) for n in (12, 24) for s in (1, 2, 3)}
    assert summary["design"] == SMALL.record()
    assert summary["chosen"]["pop_size"] in (12, 24)
    with open(out / ps.SETTINGS_FILE, newline="") as f:
        assert "run_sd" in f.readline()


def _noise(tmp_path):
    import json

    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "choice.json").write_text(json.dumps({"chosen_simulations": 3000}))
    return noise


def test_the_command_takes_a_design_and_records_it(tmp_path, monkeypatch):
    """--pop-sizes, --alpha-0s, --streams, --without-warm-starts; config.json keeps them."""
    import json

    from tests.test_task9_pilot_size import _evaluate, _threads

    seen = {}
    monkeypatch.setattr(ps, "run", lambda *a, **kw: seen.update(kw) or {})
    monkeypatch.setattr(ps, "_evaluator", lambda workers: (_evaluate, _threads))
    out = ps.main(["--noise", str(_noise(tmp_path)), "--results-root", str(tmp_path),
                   "--pop-sizes", "12,24,48", "--alpha-0s", "0.2,0.4", "--streams", "1,2,3,4",
                   "--without-warm-starts"])  # fmt: skip
    assert seen["design"] == RERUN
    config = json.loads((out / "config.json").read_text())["settings"]
    assert config["design"] == RERUN.record()
    assert len(config["runs"]) == 24
    seen.clear()
    ps.main(["--resume", str(out)])
    assert seen["design"] == RERUN  # a resume runs the design it started with


def test_without_options_the_command_runs_the_plan(tmp_path, monkeypatch):
    """The planned pilot is unchanged."""
    from tests.test_task9_pilot_size import _evaluate, _threads

    seen = {}
    monkeypatch.setattr(ps, "run", lambda *a, **kw: seen.update(kw) or {})
    monkeypatch.setattr(ps, "_evaluator", lambda workers: (_evaluate, _threads))
    ps.main(["--noise", str(_noise(tmp_path)), "--results-root", str(tmp_path)])
    assert seen["design"] == ps.PLAN


@pytest.mark.parametrize("args", [["--streams", "1"], ["--pop-sizes", "12,x"],
                                  ["--alpha-0s", "0.2,0.2"]])  # fmt: skip
def test_a_design_that_cannot_run_is_refused_before_anything_starts(tmp_path, args):
    """Refused by the parser; no tables directory is made."""
    with pytest.raises(SystemExit):
        ps.main(["--noise", str(_noise(tmp_path)), "--results-root", str(tmp_path), *args])
    assert not (tmp_path / "tables").exists()


def test_a_resume_refuses_a_design(tmp_path):
    """The design is the pilot's own."""
    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    with pytest.raises(SystemExit):
        ps.main(["--resume", str(out), "--streams", "1,2,3"])


@pytest.mark.parametrize("bad", [
    {"pop_sizes": (8, 24)},  # G = 1,249: more generations than a stream's 1,000 blocks
    {"pop_sizes": (255,), "without_warm_starts": True},  # 254 rows are left
    {"pop_sizes": (257,), "without_warm_starts": False},  # the population has 256
    {"alpha_0s": (0.0005,)},  # below alpha_end 0.001: the steps would grow
])  # fmt: skip
def test_a_design_the_optimiser_would_refuse_is_refused_first(bad):
    """Code review round 4: else the pilot records a design whose runs fail hours in."""
    with pytest.raises(ValueError):
        ps.Design(**{**RERUN._asdict(), **bad}).check()


def test_the_largest_population_without_warm_starts_is_accepted():
    """256 rows less the two warm starts."""
    assert ps.Design((254,), (0.2,), (1, 2, 3), True).check()


def test_alpha_0s_that_name_the_same_run_are_refused():
    """0.2 and 0.2000001 both name ...-a0p2-...: two settings would share one run."""
    with pytest.raises(ValueError, match="alpha_0"):
        ps.Design((24,), (0.2, 0.2000001), (1, 2, 3), True).check()


@pytest.mark.parametrize("drop", ["stream", "block"])
def test_the_mean_rule_needs_every_run_on_every_confirmation_block(drop):
    """A missing stream or block is an error, not a NaN that min() may or may not pick."""
    design = ps.Design((12, 24), (0.2,), (1, 2, 3, 4), True)
    rows = _rows({(12, 0.2): 10.0, (24, 0.2): 20.0}, {(12, 0.2): 1.0, (24, 0.2): 1.0})
    if drop == "stream":
        rows = [r for r in rows if not (r["pop_size"] == 12 and r["stream"] == 3)]
    else:
        rows = [r for r in rows if not (r["pop_size"] == 12 and r["stream"] == 3
                                        and r["block"] == 31)]  # fmt: skip
    with pytest.raises(ValueError, match="n12"):
        ps.compare_settings(rows, design)


def test_equal_means_go_to_the_setting_listed_first():
    """A tie goes to the smaller n, then the smaller alpha_0, the order the design lists."""
    design = ps.Design((12, 24), (0.2,), (1, 2, 3, 4), True)
    table = [{"pop_size": 12, "alpha_0": 0.2, "generations": 832, "mean_cost": 10.0,
              "run_sd": 1.0, "runs": 4},
             {"pop_size": 24, "alpha_0": 0.2, "generations": 415, "mean_cost": 10.0,
              "run_sd": 0.5, "runs": 4}]  # fmt: skip
    assert ps.choose(table, design)["pop_size"] == 12


def test_the_rows_left_come_from_the_population_file(monkeypatch):
    """The file records two warm starts; the code's list changing does not move the bound."""
    from covsyn.calibration import initial_population as ip

    monkeypatch.setattr(ip, "WARM_STARTS", ("a", "b", "c"))
    assert ps.Design((254,), (0.2,), (1, 2, 3), True).check()
    with pytest.raises(ValueError):
        ps.Design((255,), (0.2,), (1, 2, 3), True).check()


@pytest.mark.parametrize("streams", [(2,), (1, 2)])
def test_the_agreement_rule_needs_every_run_on_every_confirmation_block(streams):
    """A block missing from one stream or from both is an error, not a KeyError or a gap."""
    rows = [{"pop_size": n, "alpha_0": a, "stream": s, "block": b, "cost": 10.0 + n + a}
            for n in (24, 48, 96) for a in (0.05, 0.2) for s in (1, 2) for b in range(30, 60)
            if not (n == 48 and a == 0.2 and s in streams and b == 31)]  # fmt: skip
    with pytest.raises(ValueError, match="n48"):
        ps.compare_settings(rows)


@pytest.mark.parametrize(("bad", "words"), [
    ({"pop_sizes": (8, 24)}, ["--pop-sizes", "n = 8", "1249"]),
    ({"pop_sizes": (255,)}, ["--pop-sizes", "n = 255", "254"]),
    ({"alpha_0s": (0.0005,)}, ["--alpha-0s"]),
])  # fmt: skip
def test_a_refused_design_names_this_command_s_options(bad, words):
    """Not the optimiser's --generations or --alpha-0, which this command does not take."""
    with pytest.raises(ValueError) as refusal:
        ps.Design(**{**RERUN._asdict(), **bad}).check()
    assert all(w in str(refusal.value) for w in words), str(refusal.value)
    assert "--generations" not in str(refusal.value)


def test_a_resume_with_a_refused_recorded_design_says_so(tmp_path, capsys):
    """config.json and the runs are there; only the recorded design fails the checks."""
    import json

    out = tmp_path / "results" / "tables" / "20261121-000000_abcdef0_pilot-size"
    out.mkdir(parents=True)
    record = ps.Design((8,), (0.2,), (1, 2, 3), True)._asdict() | {"rule": "mean"}
    record = {k: list(v) if isinstance(v, tuple) else v for k, v in record.items()}
    (out / "config.json").write_text(json.dumps({"settings": {"simulations": 300,
                                                              "design": record}}))
    with pytest.raises(SystemExit):
        ps.main(["--resume", str(out)])
    err = capsys.readouterr().err
    assert "recorded design" in err and "holds no pilot" not in err


def test_a_record_of_another_rule_is_refused():
    """A pilot begun under the earlier spread rule does not resume under the mean rule."""
    record = RERUN.record() | {"rule": "spread"}
    with pytest.raises(ValueError, match="spread"):
        ps.Design.from_record(record)


def test_a_design_is_checked_from_any_directory(monkeypatch, tmp_path):
    """The population file is the repository's, wherever the command is started."""
    monkeypatch.chdir(tmp_path)
    assert RERUN.check() == RERUN


def test_an_unreadable_population_file_is_named(monkeypatch, tmp_path):
    """A missing file is refused as a design error naming the file and how to make it."""
    from covsyn.calibration import initial_population as ip

    missing = tmp_path / "initial_population_v1.npz"
    monkeypatch.setattr(ip, "population_path", lambda version, directory=None: missing)
    with pytest.raises(ValueError, match="covsyn-initial-population"):
        RERUN.check()
