"""The WP6 vote arithmetic, with no corpus, no model and no network.

The run files are written by hand here, which is exactly the state the
script is in when it reads what WP5 and WP1b left behind. The numbers are
small enough to check on paper, which is the point: the whole of this
measurement is arithmetic, and arithmetic nobody has checked is the kind
that quietly ships.
"""

from __future__ import annotations

import json

import pytest

from module_work.report_ingest_harness import measure_wp6_vote as m
from report_ingest.vote import (
    PageVote, agreement, combine, required_review_accuracy, trust_table,
)

#: One in-sample report: two pages the rules win, one vision wins, three
#: they agree on (two right, one wrong).
IN_RULES = {0: "calculation", 1: "calculation", 2: "plan", 3: "figure",
            4: "narrative", 5: "lab_test"}
IN_VISION = {0: "narrative", 1: "narrative", 2: "plan", 3: "plan",
             4: "narrative", 5: "lab_test"}
IN_HAND = {0: "calculation", 1: "calculation", 2: "plan", 3: "plan",
           4: "narrative", 5: "figure"}

#: One out-of-sample report: four pages, four disagreements, one per branch.
OUT_RULES = {0: "calculation", 1: "figure", 2: "other", 3: "lab_test"}
OUT_VISION = {0: "narrative", 1: "photos", 2: "boring_log", 3: "figure"}
OUT_HAND = {0: "calculation", 1: "photos", 2: "boring_log", 3: "lab_test"}

TRUTH = {
    "R98": {p: {"label": v, "alternates": []} for p, v in IN_HAND.items()},
    "R99": {p: {"label": v, "alternates": []} for p, v in OUT_HAND.items()},
}


def _vision_run(rid, rules, seen, rules_conf=None, vision_conf=None):
    """One saved WP5 run, as :func:`measure_wp5_vision.run_one` writes it."""
    rules_conf = rules_conf or {}
    vision_conf = vision_conf or {}
    return {
        "id": rid, "run_date": "2026-09-20", "n_pages": len(rules),
        "model": "fake-model", "mode": "sheet", "dpi": 100.0,
        "outline_context": False,
        "rules_labels": {str(k): v for k, v in rules.items()},
        "rules_confidence": {str(k): v for k, v in rules_conf.items()},
        "vision": {
            "labels": {str(k): v for k, v in seen.items()},
            "detail": [{"page": k, "label": v,
                        "confidence": vision_conf.get(k, 0.8),
                        "reason": "the title block names the firm"}
                       for k, v in seen.items()],
            "unresolved": [], "qa": [],
        },
        "cost": {"calls": 2, "input_tokens": 1300, "output_tokens": 90,
                 "dollars": 0.0},
        "error": None,
    }


@pytest.fixture()
def runs_on_disk(monkeypatch, tmp_path):
    """WP5 runs for both reports, and no WP1b run beside either."""
    vision = tmp_path / "wp5"
    vision.mkdir()
    (vision / "R98.json").write_text(json.dumps(_vision_run(
        "R98", IN_RULES, IN_VISION,
        {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.4, 4: 0.9, 5: 0.9},
        {0: 0.5, 1: 0.5, 2: 0.8, 3: 0.85, 4: 0.8, 5: 0.8})),
        encoding="utf-8")
    (vision / "R99.json").write_text(json.dumps(_vision_run(
        "R99", OUT_RULES, OUT_VISION,
        {0: 0.9, 1: 0.4, 2: 0.4, 3: 0.9},
        {0: 0.5, 1: 0.8, 2: 0.9, 3: 0.6})), encoding="utf-8")
    monkeypatch.setattr(m, "VISION_RUNS_DIR", vision)
    monkeypatch.setattr(m.wp5, "REVIEW_RUNS_DIR", tmp_path / "wp1b")
    monkeypatch.setattr(m, "OUT_DIR", tmp_path / "wp6")
    return vision


def _votes(rid, rules, vision, hand, rules_conf=None, vision_conf=None):
    blob = _vision_run(rid, rules, vision, rules_conf, vision_conf)
    return m.votes_for(rid, blob, TRUTH)[0]


# -- the inputs --------------------------------------------------------------

def test_the_saved_runs_are_read_back_by_id(runs_on_disk):
    runs = m.load_vision_runs([runs_on_disk])
    assert sorted(runs) == ["R98", "R99"]


def test_a_run_with_no_labels_is_not_a_voter(runs_on_disk, tmp_path):
    (runs_on_disk / "R97.json").write_text(
        json.dumps({"id": "R97", "error": "the page would not open"}),
        encoding="utf-8")
    assert "R97" not in m.load_vision_runs([runs_on_disk])


def test_the_first_folder_that_holds_a_report_wins(runs_on_disk, tmp_path):
    older = tmp_path / "older"
    older.mkdir()
    (older / "R99.json").write_text(json.dumps(_vision_run(
        "R99", OUT_RULES, {0: "cover", 1: "cover", 2: "cover", 3: "cover"})),
        encoding="utf-8")
    runs = m.load_vision_runs([runs_on_disk, older])
    labels = runs["R99"]["vision"]["labels"]
    assert labels["0"] == "narrative", "the newer folder was asked first"


# -- agreement ---------------------------------------------------------------

def test_the_agreement_rate_and_what_it_is_worth():
    agree = agreement(_votes("R98", IN_RULES, IN_VISION, IN_HAND))
    assert agree["pages"] == 6 and agree["agree"] == 3
    assert agree["agreement"] == pytest.approx(0.5)
    assert agree["agreed"]["correct"] == 2
    assert agree["agreed"]["accuracy"] == pytest.approx(0.6667)
    split = agree["disagreed"]
    assert (split["pages"], split["rules_correct"], split["vision_correct"]) \
        == (3, 2, 1)
    assert split["ceiling"] == pytest.approx(1.0), (
        "between them the two voters had all three")


def test_a_set_with_nothing_hand_labelled_reports_no_rate_rather_than_zero():
    rows = [PageVote("R01", 1, "plan", "figure")]
    agree = agreement(rows)
    assert agree["agreement"] == pytest.approx(0.0)
    assert agree["agreed"]["accuracy"] is None
    assert agree["disagreed"]["rules_accuracy"] is None


# -- the trust table ---------------------------------------------------------

def test_the_table_counts_only_the_pages_the_voters_split_on():
    table = trust_table(_votes("R98", IN_RULES, IN_VISION, IN_HAND))
    assert table["calculation"] == {"pages": 2, "rules": 2, "vision": 0,
                                    "winner": "rules"}
    assert table["figure"] == {"pages": 1, "rules": 0, "vision": 1,
                               "winner": "vision"}
    assert "plan" not in table, "they agreed on the plan page"
    assert "lab_test" not in table, "and on the lab_test one"


def test_a_tie_goes_to_vision_so_believing_the_rules_always_means_better():
    rows = [
        PageVote("R01", 1, "figure", "plan", hand="plan"),
        PageVote("R01", 2, "figure", "plan", hand="figure"),
    ]
    assert trust_table(rows)["figure"]["winner"] == "vision"


def test_the_table_is_learned_on_what_it_is_handed_and_nothing_else():
    """The caller keeps the promise; this pins that it is keepable."""
    in_sample = _votes("R98", IN_RULES, IN_VISION, IN_HAND)
    out_sample = _votes("R99", OUT_RULES, OUT_VISION, OUT_HAND)
    table = trust_table(in_sample)
    assert "other" not in table and "lab_test" not in table
    assert "other" in trust_table(in_sample + out_sample)


# -- the three policies ------------------------------------------------------

def test_each_policy_settles_the_out_of_sample_report_its_own_way():
    table = trust_table(_votes("R98", IN_RULES, IN_VISION, IN_HAND))
    rows = _votes("R99", OUT_RULES, OUT_VISION, OUT_HAND,
                  {0: 0.9, 1: 0.4, 2: 0.4, 3: 0.9},
                  {0: 0.5, 1: 0.8, 2: 0.9, 3: 0.6})
    chosen = {v.page: {p: combine(v, p, table)
                       for p in ("trust", "structural", "confidence")}
              for v in rows}
    # calculation: the table says the rules; structural says the rules; the
    # rules were also the more confident. All three get it right.
    assert chosen[0] == {"trust": "calculation", "structural": "calculation",
                         "confidence": "calculation"}
    # figure: the table says vision, and so does structural.
    assert chosen[1] == {"trust": "photos", "structural": "photos",
                         "confidence": "photos"}
    # other: the table never saw it, so trust falls to vision and is right;
    # structural keeps the rules and is wrong.
    assert chosen[2] == {"trust": "boring_log", "structural": "other",
                         "confidence": "boring_log"}
    # lab_test: the table never saw it either, so trust goes to vision and
    # is wrong where structural and confidence keep the rules and are right.
    assert chosen[3] == {"trust": "figure", "structural": "lab_test",
                         "confidence": "lab_test"}


def test_the_policies_are_scored_beside_the_voters(runs_on_disk):
    table = trust_table(_votes("R98", IN_RULES, IN_VISION, IN_HAND))
    rows = _votes("R99", OUT_RULES, OUT_VISION, OUT_HAND,
                  {0: 0.9, 1: 0.4, 2: 0.4, 3: 0.9},
                  {0: 0.5, 1: 0.8, 2: 0.9, 3: 0.6})
    scored = m.score_set(rows, table)
    accuracy = {name: s.accuracy for name, s in scored["scorers"].items()
                if s.n}
    assert accuracy["rules"] == pytest.approx(0.5)
    assert accuracy["vision"] == pytest.approx(0.5)
    assert accuracy["trust"] == pytest.approx(0.75)
    assert accuracy["structural"] == pytest.approx(0.75)
    assert accuracy["confidence"] == pytest.approx(1.0)
    assert "review" not in accuracy, "no WP1b run sits beside this one"
    assert [n for n, _s in m.columns(scored)] == [
        "rules", "vision", "trust", "struct", "conf"]


def test_the_review_joins_as_a_column_where_its_run_file_exists(
        runs_on_disk, tmp_path, monkeypatch):
    wp1b_runs = tmp_path / "wp1b"
    wp1b_runs.mkdir()
    final = dict(IN_RULES)
    final[3] = "plan"
    (wp1b_runs / "R98.json").write_text(json.dumps({
        "id": "R98",
        "review": {"final_labels": {str(k): v for k, v in final.items()}}}),
        encoding="utf-8")
    monkeypatch.setattr(m.wp5, "REVIEW_RUNS_DIR", wp1b_runs)
    rows = m.votes_for("R98", _vision_run("R98", IN_RULES, IN_VISION),
                       TRUTH)[0]
    scored = m.score_set(rows, {})
    assert scored["scorers"]["review"].accuracy == pytest.approx(5 / 6)
    assert "+review" in [n for n, _s in m.columns(scored)]


# -- the disagreement set ----------------------------------------------------

def test_what_a_review_of_the_splits_alone_would_have_to_reach():
    # Four hand pages, none agreed, so the review carries the whole gate.
    assert required_review_accuracy(0, 0, 4, 0.98) == pytest.approx(0.98)
    # Three agreed pages carrying two hits: the gate is out of reach even
    # with a perfect review of the other three.
    assert required_review_accuracy(3, 2, 3, 0.98) == pytest.approx(1.2933,
                                                                    abs=1e-4)
    # Nothing to review says nothing rather than a number.
    assert required_review_accuracy(6, 6, 0, 0.98) is None


# -- the scorecard -----------------------------------------------------------

def test_the_scorecard_carries_labels_and_rates_and_no_reason():
    table = trust_table(_votes("R98", IN_RULES, IN_VISION, IN_HAND))
    by_set = {
        "insample": m.score_set(_votes("R98", IN_RULES, IN_VISION, IN_HAND),
                                table),
        "oos_open": m.score_set(_votes("R99", OUT_RULES, OUT_VISION,
                                       OUT_HAND), table),
    }
    text = m.report(by_set, table, ["R98"])
    assert "agreement, and what it is worth" in text
    assert "per-label trust" in text and "believe" in text
    assert "P trust" in text and "P struct" in text and "P conf" in text
    assert "must reach" in text
    assert "the title block names the firm" not in text, (
        "a reason stays in the gitignored run file")


def test_an_empty_trust_table_says_so_rather_than_printing_nothing():
    rows = _votes("R99", OUT_RULES, OUT_VISION, OUT_HAND)
    text = m.report({"oos_open": m.score_set(rows, {})}, {}, [])
    assert "nothing learned" in text


# -- the command line --------------------------------------------------------

def test_the_script_runs_end_to_end_over_the_saved_files(runs_on_disk,
                                                         monkeypatch, capsys):
    monkeypatch.setattr(m.wp1b, "set_ids", lambda name: {
        "insample": ("R98",), "oos_open": ("R99",), "oos_blind": (),
        "checkpoint": (),
    }[name])
    monkeypatch.setattr(m.wp5, "oos_labels", lambda: TRUTH)
    assert m.main([]) == 0
    printed = capsys.readouterr().out
    assert "insample" in printed and "oos_open" in printed
    blob = json.loads((m.OUT_DIR / "R99.json").read_text(encoding="utf-8"))
    assert [r["page"] for r in blob["disagreements"]] == [0, 1, 2, 3]
    assert blob["trust_table_learned_on"] == ["R98"]


def test_no_saved_vision_run_is_an_error_that_says_what_to_run(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "VISION_RUNS_DIR", tmp_path / "empty")
    assert m.main([]) == 1
    assert "measure_wp5_vision" in capsys.readouterr().err
