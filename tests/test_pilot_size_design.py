# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pilot 2 with a design of its own: the grid, the streams and the warm starts (2026-11-24).

Pilot 2 as planned (n in {24, 48, 96} x alpha_0 in {0.05, 0.2} on streams 1 and 2, CovSyn
9792f7f) returned run 19, a row of the initial population, from 11 of its 12 runs, so its
agreement rule compared nothing. Its best setting lay at the corner of the grid (smallest n,
largest alpha_0), and two streams could not measure how much runs differ. User decision
2026-11-24: rerun on n in {12, 24, 48} x alpha_0 in {0.2, 0.4}, four streams each, without the
warm starts, and compare the settings by the spread between their runs: each run's
representative is scored on the confirmation blocks; a setting's cost is the mean of its runs'
means, its spread the SD between them; the lowest is chosen unless another lies within the SE
of their difference, sqrt(sd_a^2/k + sd_b^2/k), in which case the steadier (smaller SD) of
those within it is chosen.
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


def test_more_than_two_streams_use_the_spread_rule():
    """Two streams can only agree or not; three or more give a spread between runs."""
    assert RERUN.rule == "spread"
    assert ps.Design((24,), (0.2,), (1, 2, 3), False).rule == "spread"


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
    assert choice["within_se"] == []


def test_within_the_se_the_steadier_setting_is_chosen():
    """10 with run SD 6 against 11 with SD 1: SE sqrt(36/4 + 1/4) = 3.04 > 1, so 11 wins."""
    design = ps.Design((12, 24), (0.2,), (1, 2, 3, 4), True)
    rows = _rows({(12, 0.2): 10.0, (24, 0.2): 11.0}, {(12, 0.2): 6.0, (24, 0.2): 1.0})
    choice = ps.choose(ps.compare_settings(rows, design), design)
    assert (choice["pop_size"], choice["alpha_0"]) == (24, 0.2)
    assert choice["within_se"] == [[12, 0.2]]


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
    assert "within_se" in summary["chosen"]
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
