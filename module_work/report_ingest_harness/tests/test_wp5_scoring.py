"""The WP5 vision scorecard's own arithmetic, with no corpus and no model.

The run files are written by hand here, which is exactly the state a
``--reuse`` re-score is in, so the whole of the scoring and the report runs
for real: the three columns, the column that is missing where the review
never ran, and the rule that an unresolved page is scored rather than
excused.
"""

from __future__ import annotations

import json

import pytest

from module_work.report_ingest_harness import measure_wp5_vision as m


def _run(rid, rules, seen, unresolved=(), n_pages=20, calls=None, error=None):
    if error:
        return {"id": rid, "error": error}
    return {
        "id": rid, "run_date": "2026-09-17", "n_pages": n_pages,
        "model": "fake-model", "mode": "page", "dpi": 100.0,
        "outline_context": False,
        "rules_labels": {str(k): v for k, v in rules.items()},
        "vision": {
            "labels": {str(k): v for k, v in seen.items()},
            "detail": [{"page": k, "label": v, "confidence": 0.9,
                        "reason": "the title block names the firm"}
                       for k, v in seen.items()],
            "unresolved": [{"page": p, "why": "no answer"}
                           for p in unresolved],
            "qa": [],
        },
        "cost": {"calls": calls or len(seen), "input_tokens": 1300,
                 "output_tokens": 90, "dollars": 0.0},
        "error": None,
    }


OOS = {
    "R22": {0: {"label": "cover", "alternates": []},
            1: {"label": "narrative", "alternates": []},
            2: {"label": "boring_log", "alternates": []},
            3: {"label": "lab_test", "alternates": []}},
}


@pytest.fixture()
def no_review(monkeypatch, tmp_path):
    """No WP1b runs on disk, so the report prints two columns."""
    monkeypatch.setattr(m, "REVIEW_RUNS_DIR", tmp_path / "wp1b")
    return tmp_path


@pytest.fixture()
def with_review(monkeypatch, tmp_path):
    """A saved WP1b run for R22, so the report prints three."""
    runs = tmp_path / "wp1b"
    runs.mkdir()
    (runs / "R22.json").write_text(json.dumps({
        "id": "R22",
        "review": {"final_labels": {"0": "cover", "1": "narrative",
                                    "2": "boring_log", "3": "figure"}}}),
        encoding="utf-8")
    monkeypatch.setattr(m, "REVIEW_RUNS_DIR", runs)
    return runs


# -- the columns --------------------------------------------------------------

def test_the_three_columns_are_scored_on_one_set_of_hand_labels(with_review):
    runs = [_run("R22",
                 {0: "cover", 1: "narrative", 2: "boring_log", 3: "lab_test"},
                 {0: "cover", 1: "narrative", 2: "figure", 3: "lab_test"})]
    scored = m.score(runs, OOS)

    assert scored["rules"].accuracy == pytest.approx(1.0)
    assert scored["review"].accuracy == pytest.approx(0.75), (
        "the review moved a correct lab_test to figure")
    assert scored["vision"].accuracy == pytest.approx(0.75)
    assert [n for n, _s in m.columns(scored)] == ["rules", "+review", "vision"]


def test_without_a_saved_review_the_report_prints_two_columns(no_review):
    runs = [_run("R22", {0: "cover"}, {0: "cover"})]
    scored = m.score(runs, OOS)

    assert scored["review"].n == 0
    assert [n for n, _s in m.columns(scored)] == ["rules", "vision"]
    assert scored["per_report"][0]["review"] is None
    assert "+review" not in m.report(scored)


def test_an_unresolved_page_is_scored_as_other_and_not_excused(no_review):
    runs = [_run("R22",
                 {0: "cover", 1: "narrative", 2: "boring_log", 3: "lab_test"},
                 {0: "cover", 1: "narrative"}, unresolved=[2, 3])]
    scored = m.score(runs, OOS)

    assert scored["vision"].accuracy == pytest.approx(0.5)
    assert scored["per_report"][0]["unresolved"] == 2
    assert scored["per_report"][0]["labelled"] == 2


def test_a_report_that_failed_is_named_and_is_in_no_number(no_review):
    runs = [_run("R22", {}, {}, error="RuntimeError: the page would not open")]
    scored = m.score(runs, OOS)

    assert scored["vision"].n == 0
    assert scored["per_report"][0]["error"].startswith("RuntimeError")
    assert "ERROR" in m.report(scored)


# -- the scorecard ------------------------------------------------------------

def test_the_scorecard_names_no_page_and_quotes_no_reason(with_review):
    runs = [_run("R22",
                 {0: "cover", 1: "narrative", 2: "boring_log", 3: "lab_test"},
                 {0: "cover", 1: "narrative", 2: "figure", 3: "lab_test"})]
    text = m.report(m.score(runs, OOS))

    assert "R22" in text
    assert "strict accuracy" in text
    assert "P vision" in text and "R vision" in text
    assert "the title block names the firm" not in text, (
        "a reason can quote a page and stays in the gitignored run file")


def test_the_blind_set_is_a_summary_with_no_per_report_line(with_review):
    runs = [_run("R22",
                 {0: "cover", 1: "narrative", 2: "boring_log", 3: "lab_test"},
                 {0: "cover", 1: "narrative", 2: "figure", 3: "lab_test"})]
    scored = m.score(runs, OOS)

    blind = m.report(scored, blind=True)
    assert "summary only" in blind
    assert "R22" not in blind
    assert "strict accuracy" in blind
    assert "R22" in m.report(scored, blind=False)


def test_the_sets_are_the_label_scorecards_own(no_review):
    assert m.set_ids("oos_blind") == m.wp1b.OOS_BLIND
    with pytest.raises(ValueError, match="unknown set"):
        m.set_ids("everything")


# -- the run itself -----------------------------------------------------------

def test_reuse_calls_no_model_and_says_so_when_there_is_nothing_saved(
        monkeypatch, tmp_path):
    monkeypatch.setattr(m, "RUNS_DIR", tmp_path / "wp5")
    blob = m.run_one("R22", engine=None, reuse=True)
    assert blob["error"] == "no saved run to reuse"

    (tmp_path / "wp5").mkdir(exist_ok=True)
    saved = _run("R22", {0: "cover"}, {0: "cover"})
    (tmp_path / "wp5" / "R22.json").write_text(json.dumps(saved),
                                               encoding="utf-8")
    assert m.run_one("R22", engine=None, reuse=True)["id"] == "R22"


def test_a_report_runs_over_the_synthetic_document_on_a_scripted_engine(
        monkeypatch, tmp_path):
    """The real pass, the real pictures, a scripted engine: no key."""
    from planlens.document import open_document
    from planlens.testing import build_synthetic_report

    from report_ingest.tests.fake_engine import FakeEngine
    from report_ingest.vision_labels import VisionPageAnswer

    gt = build_synthetic_report()
    doc = open_document(gt.pdf, name="synthetic")
    monkeypatch.setattr(m, "RUNS_DIR", tmp_path / "wp5")
    monkeypatch.setattr(m.corpus, "open_report",
                        lambda rid, di="auto", warn=True: doc)
    monkeypatch.setattr(doc, "close", lambda: None)

    engine = FakeEngine([{"final": VisionPageAnswer(
        page=p, label="narrative", confidence=0.7, reason="prose")}
        for p in range(gt.n_pages)])
    blob = m.run_one("R22", engine)

    assert blob["error"] is None
    assert blob["n_pages"] == gt.n_pages
    assert len(blob["vision"]["labels"]) == gt.n_pages
    assert blob["rules_labels"], "the rules column is computed with no model"
    assert (tmp_path / "wp5" / "R22.json").is_file(), "the run is restartable"
    doc._doc.close()
