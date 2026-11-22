# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The sparse-bin probe: each Cheng bin's share of the attack-rate cost, and pilot 1 without one.

Pilot 1's outlier (run 19, S = 10,000, block 23, cost 34.19) came from one bin, household
contacts 0 to 3 days after onset: 1 of its 2 contacts infected added 20.49 to the cost. The
probe splits the attack-rate term per bin and re-makes pilot 1's choice with a bin left out.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "probes"))

import probe_sparse_bins as psb  # noqa: E402

CHENG = np.array([[0.04, 0.051, 0.167, 0, 0, 0],
                  [0.008, 0.02, 0.026, 0, 0, 0],
                  [0, 0, 0.006, 0, 0, 0]])  # fmt: skip
WEIGHTS = np.array([[1.0, 0.53246753, 0.15355805, 0.16734694, 0.12480974, 0.082],
                    [0.92857143, 0.52, 0.2, 0.14130435, 0.78787879, 1.0],
                    [0.6, 1.0, 0.1875, 0.14285714, 0.54545455, 0.18181818]])  # fmt: skip


def _bins(household_contacts, household_infections):
    """Pooled (group, contacts|infections, bin) counts with the household 0-3 day bin set."""
    bins = np.zeros((3, 2, 6))
    bins[0, 0, 1], bins[0, 1, 1] = household_contacts, household_infections
    return bins


def test_one_infected_of_two_household_contacts_adds_twenty_point_five():
    """The Mac's block 23: rate 0.5 against Cheng's 0.051, weight 0.532, x 10 / 0.167^2."""
    terms = psb.attack_terms(_bins(2, 1), CHENG, WEIGHTS)
    assert terms[0, 1] == pytest.approx(20.4949, abs=1e-4)


def test_one_uninfected_contact_adds_a_quarter():
    """Blocks 17 and 43: rate 0, so the term is Cheng's own, 0.2644."""
    assert psb.attack_terms(_bins(1, 0), CHENG, WEIGHTS)[0, 1] == pytest.approx(0.2644, abs=1e-4)


def test_a_bin_without_contacts_charges_cheng_s_rate():
    """No contacts: rate 0, as the objective takes it (the household 4-5 day bin, 0.2358)."""
    assert psb.attack_terms(_bins(0, 0), CHENG, WEIGHTS)[0, 2] == pytest.approx(0.2358, abs=1e-4)


def test_the_rows_without_a_bin_lose_its_term_only():
    """Each row's cost less that bin's term; everything else as it was."""
    rows = [
        {"kind": "vector", "cost": "34.19", "term_0_1": "20.49", "term_0_2": "0.24"},
        {"kind": "spread", "cost": "14.60", "term_0_1": "0.26", "term_0_2": "0.24"},
    ]
    out = psb.without_bin(rows, 0, 1)
    assert [r["cost"] for r in out] == pytest.approx([34.19 - 20.49, 14.60 - 0.26])
    assert out[0]["kind"] == "vector" and rows[0]["cost"] == "34.19"  # input untouched


def test_no_rows_give_no_rows():
    """Nothing to take the term from."""
    assert psb.without_bin([], 0, 1) == []


def test_a_simulation_that_raises_is_charged_as_pilot_1_charged_it(monkeypatch):
    """FAILED_EVALUATION_COST and no bins, not a crash of the whole re-scoring."""
    from covsyn.calibration import fast_cost
    from covsyn.calibration import firefly_optimizer as fo

    def broken(P, seed_offset, clock):
        raise RuntimeError("simulation failed")

    monkeypatch.setattr(fast_cost, "_simulate_and_score", broken)
    monkeypatch.setattr(fast_cost, "_WORKER_SIMULATIONS", 300)
    cost, attack, bins = psb.score_bins(np.zeros(3), 0)
    assert cost == fo.FAILED_EVALUATION_COST and np.isnan(attack) and bins is None


def test_an_empty_or_missing_table_needs_a_header(tmp_path):
    """A table killed before its header reached the disk is begun again."""
    path = tmp_path / "bins.csv"
    assert psb.needs_header(path)
    path.write_text("")
    assert psb.needs_header(path)
    path.write_text(",".join(psb.COLUMNS) + "\n")
    assert not psb.needs_header(path)


def test_every_evaluation_must_be_in_pilot_1_s_table():
    """Another file, or another stream's, would check nothing: name what is missing."""
    keys = [("vector", 0, 300, 0), ("vector", 2, 300, 0)]
    assert psb.missing_from_pilot(keys, {("vector", 0, 300, 0): 1.0}) == [("vector", 2, 300, 0)]
    assert psb.missing_from_pilot(keys, dict.fromkeys(keys, 1.0)) == []


def test_a_failed_simulation_prints_its_traceback(monkeypatch, capsys):
    """Charged 1e6 like pilot 1, but the reason is shown (code review round 2)."""
    from covsyn.calibration import fast_cost
    from covsyn.calibration import firefly_optimizer as fo

    def broken(P, seed_offset, clock):
        raise RuntimeError("the reason")

    monkeypatch.setattr(fast_cost, "_simulate_and_score", broken)
    cost, _, bins = psb.score_bins(np.zeros(3), 0)
    assert "the reason" in capsys.readouterr().err
    assert cost == fo.FAILED_EVALUATION_COST and bins is None


def test_a_table_with_another_header_is_refused(tmp_path):
    """A header cut short when killed would misread every row after it."""
    path = tmp_path / "bins.csv"
    path.write_text("kind,ro")
    with pytest.raises(SystemExit, match="header"):
        psb.needs_header(path)
    path.write_text(",".join(psb.COLUMNS) + "\n")
    assert not psb.needs_header(path)


def test_a_table_whose_last_row_was_cut_is_refused(tmp_path):
    """The next row would be joined onto it (code review round 3)."""
    path = tmp_path / "bins.csv"
    path.write_text(",".join(psb.COLUMNS) + "\nvector,literature,0,30")
    with pytest.raises(SystemExit, match="cut"):
        psb.needs_header(path)


def test_resumed_rows_are_checked_against_pilot_1_too():
    """A row from an earlier run, against the table given now (code review round 3)."""
    row = {"attack": "0.0", "household_0_3_contacts": "0", "household_0_3_infections": "0",
           **dict.fromkeys(psb.TERMS, "0.0")}  # fmt: skip
    done = {("vector", 0, 300, 0): {**row, "cost": "10.0"},
            ("vector", 2, 300, 0): {**row, "cost": "5.0"}}  # fmt: skip
    pilot = {("vector", 0, 300, 0): 10.0, ("vector", 2, 300, 0): 6.0}
    assert psb.mismatched(done, pilot) == [("vector", 2, 300, 0)]


def test_a_table_that_is_not_utf_8_is_refused_with_a_reason(tmp_path):
    """Its first line cannot be the header (code review round 4)."""
    path = tmp_path / "bins.csv"
    path.write_bytes(b"\xff\xfek\x00i\x00\n")
    with pytest.raises(SystemExit, match="another header"):
        psb.needs_header(path)


def test_a_header_without_its_newline_is_called_a_cut_header(tmp_path):
    """Not a cut row, which would send the user to delete the header (code review round 4)."""
    path = tmp_path / "bins.csv"
    path.write_text(",".join(psb.COLUMNS))
    with pytest.raises(SystemExit, match="its header lacks its newline"):
        psb.needs_header(path)


def test_resumed_rows_whose_terms_do_not_add_to_their_attack_cost_are_flagged():
    """Terms from an older probe would feed the choice unchecked (code review round 4)."""
    key = ("vector", 0, 300, 0)
    row = {"cost": "10.0", "attack": "3.0", "household_0_3_contacts": "0",
           "household_0_3_infections": "0",
           **dict.fromkeys(psb.TERMS, "0.0")}  # fmt: skip
    good = {**row, "term_0_1": "3.0"}
    assert psb.mismatched({key: good}, {key: 10.0}) == []
    assert psb.mismatched({key: {**row, "term_0_1": "2.0"}}, {key: 10.0}) == [key]


def _scored(**changes):
    """A resumed row as bins.csv holds it, of cost 10 and attack 3, all in term_0_1."""
    row = {c: "0" for c in psb.COLUMNS}
    row.update({"kind": "vector", "label": "literature", "cost": "10.0", "attack": "3.0",
                "term_0_1": "3.0", **changes})  # fmt: skip
    return row


KEY = ("vector", 0, 300, 0)


def test_a_failed_row_with_no_terms_is_kept():
    """Charged 1e6, attack NaN, counts -1, terms 0: as a fresh failed row is written."""
    row = _scored(cost=str(1e6), attack="nan", term_0_1="0.0", household_0_3_contacts="-1",
                  household_0_3_infections="-1")  # fmt: skip
    assert psb.mismatched({KEY: row}, {KEY: 1e6}) == []


def test_a_failed_row_with_terms_is_refused():
    """Leaving a bin out would take its term from a cost of 1e6."""
    row = _scored(cost=str(1e6), attack="nan", term_0_1="5.0", household_0_3_infections="-1")
    assert psb.mismatched({KEY: row}, {KEY: 1e6}) == [KEY]


@pytest.mark.parametrize(
    "field, value",
    [("attack", None), ("term_2_5", None), ("household_0_3_infections", ""), ("cost", "x")],
)
def test_a_short_or_unreadable_row_is_refused_not_a_traceback(field, value):
    """A row cut inside, or a cell that is no number: named, not a crash."""
    assert psb.mismatched({KEY: _scored(**{field: value})}, {KEY: 10.0}) == [KEY]


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=psb.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_rows_already_scored_are_read_by_their_key(tmp_path):
    """kind, row, S and block identify an evaluation, as in pilot 1's table."""
    path = tmp_path / "bins.csv"
    _write(path, [_scored(row="2", simulations="300", block="7")])
    assert list(psb.read_done(path)) == [("vector", 2, 300, 7)]


@pytest.mark.parametrize("field", ["row", "simulations", "block"])
def test_a_row_whose_key_cannot_be_read_is_refused(tmp_path, field):
    """Named, not a traceback (code review round 6)."""
    path = tmp_path / "bins.csv"
    _write(path, [_scored(**{field: "x"})])
    with pytest.raises(SystemExit, match="line 2"):
        psb.read_done(path)


def test_an_evaluation_held_twice_is_refused(tmp_path):
    """One of them would be dropped unchecked (code review round 6)."""
    path = tmp_path / "bins.csv"
    _write(path, [_scored(), _scored(cost="11.0")])
    with pytest.raises(SystemExit, match="twice"):
        psb.read_done(path)


@pytest.mark.parametrize("changes", [
    {"household_0_3_contacts": "5"},  # failed, but counts contacts
    {"attack": "3.0"},  # failed, but has an attack-rate cost
])  # fmt: skip
def test_a_failed_row_must_be_written_as_a_fresh_failed_row(changes):
    """Charged 1e6, attack NaN, counts -1, terms 0 (code review round 6)."""
    failed = {
        "cost": str(1e6),
        "attack": "nan",
        "term_0_1": "0.0",
        "household_0_3_contacts": "-1",
        "household_0_3_infections": "-1",
    }
    row = _scored(**{**failed, **changes})
    assert psb.mismatched({KEY: row}, {KEY: 1e6}) == [KEY]


def test_a_scored_row_with_negative_contacts_is_refused():
    """Only a failed row has the -1 placeholders."""
    assert psb.mismatched({KEY: _scored(household_0_3_contacts="-1")}, {KEY: 10.0}) == [KEY]


NUMERIC = ["cost", "attack", "household_0_3_contacts", "household_0_3_infections", *psb.TERMS]


@given(field=st.sampled_from(NUMERIC),
       text=st.text().filter(lambda t: not _number(t)))  # fmt: skip
def test_any_cell_that_is_no_number_makes_a_row_unusable(field, text):
    """Property: whatever the text, a numeric cell that is not a number refuses the row."""
    assert psb.mismatched({KEY: _scored(**{field: text})}, {KEY: 10.0}) == [KEY]


@given(costs=st.lists(st.floats(0, 100), min_size=len(psb.TERMS), max_size=len(psb.TERMS)))
def test_a_row_written_as_main_writes_it_is_usable(costs):
    """Property: terms that add up to the attack cost, and the pilot's cost, always pass."""
    row = _scored(attack=repr(sum(costs)), **dict(zip(psb.TERMS, map(repr, costs), strict=True)))
    assert psb.mismatched({KEY: row}, {KEY: 10.0}) == []


def _number(text):
    """Whether float() reads it: a cell that is a number in some column."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def test_a_row_that_is_not_utf_8_is_refused_naming_its_line(tmp_path):
    """A damaged byte after the header: named, not a traceback (round 7)."""
    path = tmp_path / "bins.csv"
    _write(path, [_scored()])
    with open(path, "ab") as f:
        f.write(b"vector,\xff\n")
    with pytest.raises(SystemExit, match="line 3"):
        psb.read_done(path)


def test_the_line_named_is_the_file_s_own_line(tmp_path):
    """After a blank line, the record count and the line differ (round 7)."""
    path = tmp_path / "bins.csv"
    _write(path, [_scored()])
    with open(path, "a", newline="") as f:
        f.write("\n")
        csv.DictWriter(f, fieldnames=psb.COLUMNS).writerow(_scored(block="x"))
    with pytest.raises(SystemExit, match="line 4"):
        psb.read_done(path)


@pytest.mark.parametrize("changes", [
    {"household_0_3_contacts": "1", "household_0_3_infections": "2"},  # more infected than met
    {"attack": "inf", "term_0_1": "inf"},  # adds up, but no cost
])  # fmt: skip
def test_a_scored_row_that_cannot_be_real_is_refused(changes):
    """Infections within contacts, and finite costs (round 7)."""
    assert psb.mismatched({KEY: _scored(**changes)}, {KEY: 10.0}) == [KEY]


@given(field=st.sampled_from(["household_0_3_contacts", "household_0_3_infections"]),
       text=st.text().filter(lambda t: not _integer(t)))  # fmt: skip
def test_a_count_that_is_no_integer_makes_a_row_unusable(field, text):
    """Property: counts are integers; '1.5' or 'nan' refuse the row (round 7)."""
    assert psb.mismatched({KEY: _scored(**{field: text})}, {KEY: 10.0}) == [KEY]


@given(field=st.sampled_from(["row", "simulations", "block"]),
       text=st.text(alphabet=st.characters(blacklist_categories=("Cs",),
                                           blacklist_characters="\r\n\x00"))
       .filter(lambda t: not _integer(t)))  # fmt: skip
def test_a_key_that_is_no_integer_is_refused_naming_its_line(tmp_path_factory, field, text):
    """Property: any key cell int() cannot read stops the resume at its line (round 7)."""
    path = tmp_path_factory.mktemp("done") / "bins.csv"
    _write(path, [_scored(**{field: text})])
    with pytest.raises(SystemExit, match="line 2"):
        psb.read_done(path)


def _integer(text):
    """Whether int() reads it."""
    try:
        int(text)
        return True
    except ValueError:
        return False


def test_a_scored_row_with_a_negative_term_is_refused():
    """Each term is a weighted square (round 8)."""
    row = _scored(term_0_1="5.0", term_0_2="-2.0")
    assert psb.mismatched({KEY: row}, {KEY: 10.0}) == [KEY]


PILOT_HEADER = "kind,label,row,simulations,block,seed_offset,cost\n"


def test_pilot_1_s_table_is_read_as_pilot_1_reads_it(tmp_path):
    """Its byte order mark skipped, a last line cut short left out (round 9)."""
    path = tmp_path / "blocks.csv"
    path.write_bytes(
        b"\xef\xbb\xbf"
        + PILOT_HEADER.encode()
        + b"vector,literature,0,300,0,0,1.5\nvector,literature,0,300,1,0,13.4"
    )
    assert psb.read_pilot(path) == {("vector", 0, 300, 0): 1.5}


def test_pilot_1_s_table_with_an_unreadable_key_is_refused_naming_its_record(tmp_path):
    """Named, not a traceback (round 8)."""
    path = tmp_path / "blocks.csv"
    path.write_text(PILOT_HEADER + "vector,literature,0,300,0,0,1.0\n"
                    "vector,literature,x,300,1,0,1.0\n", encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit, match="finished row 2 .*: its key cannot be read"):
        psb.read_pilot(path)


def test_pilot_1_s_table_with_an_unreadable_cost_says_so(tmp_path):
    """The cost is the bad cell, not the key (round 9)."""
    path = tmp_path / "blocks.csv"
    path.write_text(PILOT_HEADER + "vector,literature,0,300,0,0,abc\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="finished row 1 .*: its cost cannot be read"):
        psb.read_pilot(path)


def test_a_bad_line_is_found_with_the_csv_reader_s_line_endings(tmp_path):
    """A lone carriage return ends a line for csv; none found gives no line (round 8)."""
    path = tmp_path / "bins.csv"
    path.write_bytes(b"a\rb\r\xff\n")
    assert psb._first_bad_line(path) == 3
    path.write_bytes(b"a\nb\n")
    assert psb._first_bad_line(path) is None


@pytest.mark.parametrize("cost", ["nan", "inf", "-inf"])
def test_pilot_1_s_table_with_a_cost_that_is_not_finite_is_refused(tmp_path, cost):
    """Found before any simulation, naming the record (round 9)."""
    path = tmp_path / "blocks.csv"
    path.write_text(PILOT_HEADER + f"vector,literature,0,300,0,0,{cost}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="finished row 1 .*: its cost is not finite"):
        psb.read_pilot(path)


def test_a_missing_pilot_table_stops_before_the_output_is_made(tmp_path, monkeypatch):
    """Named, with nothing left behind (round 9)."""
    out, missing = tmp_path / "out", tmp_path / "typo_blocks.csv"
    monkeypatch.setattr(sys, "argv", ["probe", str(out), str(missing)])
    with pytest.raises(SystemExit, match="typo_blocks.csv"):
        psb.main()
    assert not out.exists()


def test_a_pilot_table_that_is_not_utf_8_is_refused_naming_its_line(tmp_path):
    """As bins.csv is (round 10)."""
    path = tmp_path / "blocks.csv"
    path.write_bytes(PILOT_HEADER.encode() + b"vector,lit\xe9,0,300,0,0,1.0\n")
    with pytest.raises(SystemExit, match="blocks.csv, line 2: not UTF-8"):
        psb.read_pilot(path)


def test_a_pilot_table_that_does_not_exist_is_said_not_to(tmp_path):
    """Not "went missing while the pilot ran": the probe never saw it (round 10)."""
    with pytest.raises(SystemExit, match="nope.csv does not exist"):
        psb.read_pilot(tmp_path / "nope.csv")


def test_the_row_named_counts_finished_rows_and_says_so(tmp_path):
    """A short row before the bad one is left out, as pilot 1 leaves it out (round 10)."""
    path = tmp_path / "blocks.csv"
    path.write_text(PILOT_HEADER + "vector,lit,0,300,0,0,1.0\nvector,lit,0,300\n"
                    "vector,lit,x,300,2,0,1.0\n", encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit, match=r"finished row 2 \(cut, short or long rows not counted\)"):
        psb.read_pilot(path)


def test_a_table_that_is_not_pilot_1_s_stops_before_the_output_is_made(tmp_path, monkeypatch):
    """Readable but lacking pilot 1's evaluations: nothing left behind (round 10)."""
    pilot = tmp_path / "blocks.csv"
    pilot.write_text(PILOT_HEADER + "vector,literature,0,300,0,0,1.0\n", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["probe", str(out), str(pilot)])
    with pytest.raises(SystemExit, match="lacks"):
        psb.main()
    assert not out.exists()


def test_a_worker_count_refused_stops_before_the_output_is_made(tmp_path, monkeypatch):
    """The inputs and the pool are set up before anything is written (round 11)."""
    pilot = tmp_path / "blocks.csv"
    pilot.write_text(PILOT_HEADER, encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["probe", str(out), str(pilot)])
    monkeypatch.setattr(psb, "missing_from_pilot", lambda keys, pilot: [])

    def refused(workers):
        raise SystemExit("workers refused")

    monkeypatch.setattr(psb.block_scoring, "objective_pools", refused)
    with pytest.raises(SystemExit, match="workers refused"):
        psb.main()
    assert not out.exists()


def test_a_pilot_table_linked_to_a_missing_file_says_so(tmp_path):
    """The link exists; its target does not (round 11)."""
    link = tmp_path / "blocks.csv"
    link.symlink_to(tmp_path / "moved" / "blocks.csv")
    with pytest.raises(SystemExit, match="is a link to .*moved.*, which does not exist"):
        psb.read_pilot(link)


def test_a_chain_of_links_names_the_file_that_is_missing(tmp_path, monkeypatch):
    """The final target, resolved from the link's own directory (round 12)."""
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot" / "b.csv").symlink_to("gone.csv")  # relative to pilot/
    link = tmp_path / "blocks.csv"
    link.symlink_to("pilot/b.csv")
    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(SystemExit) as stopped:
        psb.read_pilot(link)
    assert str(tmp_path / "pilot" / "gone.csv") in str(stopped.value)


def test_a_long_row_left_out_is_named_among_those_not_counted(tmp_path):
    """read_table leaves out a row of too many values too (round 12)."""
    path = tmp_path / "blocks.csv"
    path.write_text(PILOT_HEADER + "vector,lit,0,300,0,0,1.0,extra\n"
                    "vector,lit,x,300,2,0,1.0\n", encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit, match=r"finished row 1 \(cut, short or long rows not counted\)"):
        psb.read_pilot(path)
