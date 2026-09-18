"""The WP4 scorecard's own arithmetic, with no corpus and no model.

Pure: the table the development script prints, the open/blind split and the
per-question tally. Each one decides what a published number means, and the
one worth pinning hardest is that a reader which answers nothing prints a
recall of zero rather than an accuracy of ninety.
"""

from __future__ import annotations

import pytest

from module_work.report_ingest_harness import measure_wp4_narrative as m
from report_ingest.model import GeneralFacts, NaturalHazardFacts
from report_ingest.narrative_scoring import score_narrative


def hand(**sections):
    return {"general": sections.get("general", {}),
            "natural_hazards": sections.get("natural_hazards", {})}


def scored(report, truth, general=None, hazards=None):
    out = score_narrative(truth, general or GeneralFacts(),
                          hazards or NaturalHazardFacts(), report=report)
    out.model_calls = 1
    out.cost = {"dollars": 0.42}
    return out


def test_the_open_set_falls_back_to_the_named_default(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "TRUTH_DIR", tmp_path)

    assert list(m.open_reports(m.DEFAULT_OPEN)) == list(m.DEFAULT_OPEN)
    (tmp_path / "OPEN.txt").write_text("R15\nR28\n", encoding="utf-8")
    assert list(m.open_reports(m.DEFAULT_OPEN)) == ["R15", "R28"]


def test_the_truth_files_carry_their_report_id_from_their_name(tmp_path,
                                                              monkeypatch):
    import json

    monkeypatch.setattr(m, "TRUTH_DIR", tmp_path)
    (tmp_path / "R07.json").write_text(
        json.dumps({"general": {"boringCount": 3}}), encoding="utf-8")

    (truth,) = m.load_truth()
    assert truth["id"] == "R07"
    assert m.load_truth(only=["R99"]) == []


def test_no_hand_answers_at_all_says_where_they_should_be(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(m, "TRUTH_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="gitignored"):
        m.load_truth()


def test_the_table_prints_the_three_numbers_per_report():
    truth = hand(general={"boringCount": 4, "projectName": "Rosewood"})
    rows = [scored("R36", truth, GeneralFacts(boringCount=4,
                                              projectName="Rosewood")),
            scored("R22", truth)]

    text = m.report(rows, ("R36",))

    assert "R36" in text and "R22" in text
    assert "open" in text and "blind" in text
    assert "100% 2/2" in text                     # R36's recall
    assert "0% 0/2" in text                       # R22 answered nothing
    assert "OPEN --" in text and "BLIND --" in text and "ALL --" in text


def test_a_reader_that_says_nothing_does_not_print_ninety_percent():
    truth = hand(general={"boringCount": 4})
    text = m.report([scored("R22", truth)], ("R36",))

    recall_line = [line for line in text.splitlines()
                   if line.startswith("recall")][0]
    assert "0% 0/1" in recall_line
    # The flattering number is printed too, and labelled.
    assert any(line.startswith("agreement") for line in text.splitlines())


def test_the_per_question_table_is_opt_in():
    truth = hand(general={"boringCount": 4})
    rows = [scored("R36", truth, GeneralFacts(boringCount=4))]

    assert "Per question" not in m.report(rows, ("R36",))
    text = m.report(rows, ("R36",), fields=True)
    assert "Per question" in text and "boringCount" in text


def test_the_detail_lists_misses_on_the_open_reports_only():
    truth = hand(general={"boringCount": 4, "postName": "Somewhere"})
    rows = [scored("R36", truth, GeneralFacts(boringCount=4)),
            scored("R22", truth)]

    text = m.report(rows, ("R36",), detail=True)

    assert "R36 -- 1 miss(es)" in text
    assert "postName" in text
    assert "R22 --" not in text


def test_a_report_that_failed_is_named_and_left_out_of_the_totals():
    from report_ingest.narrative_scoring import NarrativeScore

    truth = hand(general={"boringCount": 4})
    rows = [scored("R36", truth, GeneralFacts(boringCount=4)),
            NarrativeScore(report="R19", error="no narrative work item")]

    text = m.report(rows, ("R36",))

    assert "R19" in text and "ERROR" in text
    assert "1 model call(s) over 1 report(s)" in text
