"""Repairs of the cited-pages check (BACKLOG EV-18, 2026-11-18).

An offline pass over saved runs found the check calling pages "never
opened" that the answer did not cite as a source: a page whose statement
only repeats what the page map said of it, a page the answer names as not
viewed, a printed chapter-page number read as a run of PDF pages, a page
whose markups were listed, a page a reference reader rendered. Each repair
here has a made-up case that must stop firing beside one that must still
fire. ``GEOTECH_CITED_PAGES`` stays off by default; with it off the answer
is returned as it was (the last tests). Made-up documents and answers.
"""

from __future__ import annotations

import itertools
import json

import pytest

from aec_agent_core import cited_pages as cp
from aec_agent_core import review_flags

_IDS = itertools.count(1)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (review_flags.ALL_ENVS + review_flags.LATER_ENVS
                + review_flags.SETTINGS_ENVS):
        monkeypatch.delenv(env, raising=False)
    cp.clear_cache()
    yield
    cp.clear_cache()


def _conv(tmp_path):
    conv = tmp_path / f"conv{next(_IDS)}"
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text("{}", encoding="utf-8")
    return conv


def _write_events(conv, calls, turn=1):
    """Tool calls as the activity log records them."""
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        for i, call in enumerate(calls):
            rid = f"r{turn}_{i}_{call['name']}"
            fh.write(json.dumps({"turn": turn, "event": "tool_start",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            fh.write(json.dumps({"turn": turn, "event": "tool_end",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid, "truncated": False,
                                 "result": json.dumps(call["result"])})
                     + "\n")


def _labels(rows):
    """A label source: the same printed numbers for every document."""
    return lambda opened, key: {p: set(ls) for p, ls in rows.items()}


OPEN = {"name": "open_document", "args": {"source": "made_up_report.pdf"},
        "result": {"handle": "doc_mu", "name": "made_up_report.pdf",
                   "n_pages": 30}}


def _read(pages):
    return {"name": "read_document",
            "args": {"handle": "doc_mu", "pages": pages},
            "result": {"handle": "doc_mu", "pages_returned": pages}}


PAGE_MAP = {"name": "document_page_map", "args": {"handle": "doc_mu"},
            "result": {"handle": "doc_mu", "n_pages": 30, "rows": [
                {"page": 3, "kind": "blank", "words": 0},
                {"page": 5, "kind": "form", "title": "Log of Boring MB-1"},
                {"page": 6, "kind": "form", "title": "Log of Boring MB-2"},
                {"page": 7, "kind": "text", "duplicate_of": 2,
                 "title": "Made-up site description"},
                {"page": 11, "kind": "text", "title": "Chapter 2 Methods"}],
                "toc": [
                    {"level": 1, "title": "Chapter 2 Methods", "page": 11},
                    {"level": 1, "title": "Appendix C Laboratory results",
                     "page": 20},
                    {"level": 1, "title": "Appendix D Site photographs",
                     "page": 26}]}}


def _opened(tmp_path, calls):
    conv = _conv(tmp_path)
    _write_events(conv, calls)
    return cp.pages_opened(str(conv), folder=str(conv / "files"))


def _check(tmp_path, text, calls, labels=None):
    return cp.check_answer(text, opened=_opened(tmp_path, calls),
                           label_source=_labels(labels or {}))


def _cited(res, key):
    return [r["cited"] for r in res.get(key, [])]


# ---------------------------------------------------------------------------
# A: what the page map said of a page is not a citation of what it says
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, cited", [
    ("PDF page 4 is blank.", "PDF page 4"),
    ("The two logs (PDF pages 6 and 7) are MB-1 and MB-2.", "PDF page 6"),
    ("PDF page 8 repeats PDF page 3 exactly.", "PDF page 8"),
    ("PDF page 8 repeats PDF page 3 exactly.", "PDF page 3"),
    ("Appendix C (PDF pages 21-26) holds the laboratory results.",
     "PDF pages 21-26"),
    ("Chapter 2 starts at PDF page 12, printed 2-1.", "PDF page 12"),
])
def test_what_the_page_map_said_is_filed_apart(tmp_path, text, cited):
    res = _check(tmp_path, text, [OPEN, PAGE_MAP, _read("0-1")],
                 labels={11: {"2-1"}})
    assert _cited(res, "never_opened") == []
    rows = [r for r in res["page_map"] if r["cited"] == cited]
    assert rows and rows[0]["matched"] and "page map" in rows[0]["why"]


@pytest.mark.parametrize("text, cited", [
    # a value the page map never gave
    ("PDF page 4 gives a groundwater depth of 3.2 m.", "PDF page 4"),
    # an identifier the page map never gave for these pages
    ("PDF pages 6 and 7 are the logs of MB-1 and MB-3.", "PDF page 6"),
    ("PDF page 8 shows the log of boring MB-7.", "PDF page 8"),
    # a finding inside the part, not the part's title
    ("PDF page 23 gives a liquid limit of 41 for sample S-4.",
     "PDF page 23"),
    # a page no structure result named at all
    ("PDF page 10 is blank.", "PDF page 10"),
])
def test_a_claim_the_page_map_did_not_make_still_fires(tmp_path, text,
                                                        cited):
    res = _check(tmp_path, text, [OPEN, PAGE_MAP, _read("0-1")])
    assert cited in _cited(res, "never_opened")


# ---------------------------------------------------------------------------
# B: a page the answer names as not opened is not a citation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "PDF pages 10 and 14 were not viewed.",
    "Not read: PDF page 10.",
    "If the stamp matters, I can look at PDF page 10.",
    "PDF page 10 (the vicinity map) was not needed.",
    "The blank-looking PDF page 10 should be checked.",
    "Want me to open PDF page 10?",
])
def test_a_page_named_as_unread_is_not_a_citation(tmp_path, text):
    res = _check(tmp_path, text, [OPEN, _read("0-1")])
    assert _cited(res, "never_opened") == []
    assert res["named_unread"] and res["named_unread"][0]["said"]


@pytest.mark.parametrize("text", [
    "The stamp is on PDF page 10.",
    "The stamp is on PDF page 10, and PDF page 14 was not viewed.",
    "PDF page 14 was not viewed, and the stamp is on PDF page 10.",
    "PDF page 10 shows that the slab was not checked for uplift.",
])
def test_a_page_cited_as_a_source_still_fires(tmp_path, text):
    res = _check(tmp_path, text, [OPEN, _read("0-1")])
    assert "PDF page 10" in _cited(res, "never_opened")


# ---------------------------------------------------------------------------
# C: printed numbers and PDF pages
# ---------------------------------------------------------------------------

def test_a_printed_number_beside_the_pdf_page_defers_to_it(tmp_path):
    calls = [OPEN, _read("12")]
    res = _check(tmp_path, "The method is on printed page 1-3 (PDF page "
                 "13).", calls)
    assert _cited(res, "never_opened") == []
    assert _cited(res, "unverified") == []
    res = _check(tmp_path, "| Item | Page | PDF page |\n|---|---|---|\n"
                 "| Method | 1-3 | 13 |", calls)
    assert _cited(res, "never_opened") == []


def test_the_pdf_page_beside_a_printed_number_still_fires(tmp_path):
    res = _check(tmp_path, "The method is on printed page 1-3 (PDF page "
                 "15).", [OPEN, _read("12")])
    assert _cited(res, "never_opened") == ["PDF page 15"]


def test_a_chapter_page_number_with_unknown_numbering_is_unverified(
        tmp_path):
    res = _check(tmp_path, "The figure is on p. 1-3.", [OPEN, _read("12")])
    assert _cited(res, "never_opened") == []
    assert _cited(res, "unverified") == ["pages 1-3"]


def test_a_run_of_pages_still_fires(tmp_path):
    res = _check(tmp_path, "The figures are on pages 1-3.",
                 [OPEN, _read("12")])
    assert _cited(res, "never_opened") == ["pages 1-3"]


def test_in_a_plainly_numbered_document_p_12_16_is_still_a_run(tmp_path):
    plain = {p: {str(p + 1)} for p in range(30)}
    res = _check(tmp_path, "The analysis is on p. 12-16.",
                 [OPEN, _read("0-1")], labels=plain)
    assert _cited(res, "never_opened") == ["pages 12-16"]


def test_a_number_said_to_be_printed_that_no_page_prints(tmp_path):
    # printed 1 to 26 on PDF pages 5 to 30
    offset = {p: {str(p - 3)} for p in range(4, 30)}
    res = _check(tmp_path, "The table is on printed page 28.",
                 [OPEN, _read("0-1")], labels=offset)
    assert _cited(res, "never_opened") == []
    assert _cited(res, "unverified") == ["page 28"]
    res = _check(tmp_path, "The table is on page 28.",
                 [OPEN, _read("0-1")], labels=offset)
    assert _cited(res, "never_opened") == ["page 28"]


def test_a_count_of_pages_read_is_not_a_citation():
    assert cp.find_citations("text pages 27 of 213 read, 186 skipped") == []
    assert cp.find_citations("the log (Page 27 of 213)")


# ---------------------------------------------------------------------------
# D: a page whose markups were listed counts as opened
# ---------------------------------------------------------------------------

MARKUPS = {"name": "document_markups", "args": {"handle": "doc_mu"},
           "result": {"handle": "doc_mu", "markups": [
               "p3 (MB-2) callout: check the made-up bearing value",
               "p4 cloud: revise the made-up note"]}}


def test_a_page_whose_markups_were_listed_counts(tmp_path):
    res = _check(tmp_path, "The callout on PDF page 4 asks to check the "
                 "bearing value.", [OPEN, MARKUPS])
    assert _cited(res, "never_opened") == []


def test_a_page_with_no_markup_listed_still_fires(tmp_path):
    res = _check(tmp_path, "PDF page 7 has a cloud.", [OPEN, MARKUPS])
    assert _cited(res, "never_opened") == ["PDF page 7"]


def test_markups_do_not_change_what_write_diggs_is_told(tmp_path):
    opened = _opened(tmp_path, [OPEN, MARKUPS])
    assert opened.status(3) != "opened"
    entries, _summary = cp.record_entries(
        {"investigations": [{"investigation_id": "MB-2", "pages": [3]}]},
        opened=opened)
    assert [e["where"] for e in entries] == ["MB-2"]


# ---------------------------------------------------------------------------
# G: a page a reference reader rendered counts as opened
# ---------------------------------------------------------------------------

FIGURE = {"name": "read_reference_figure",
          "args": {"reference": "made_up_manual", "figure_number": "4-2"},
          "result": {"source": "made_up_manual.pdf", "figure_number": "4-2",
                     "pdf_page_index": 41, "printed_page": "4-7",
                     "analysis": "a made-up chart"}}


def test_a_rendered_reference_page_counts(tmp_path):
    res = _check(tmp_path, "Figure 4-2 is on PDF page 42 (printed page "
                 "4-7).", [FIGURE])
    assert _cited(res, "never_opened") == []
    assert _cited(res, "unverified") == []


def test_a_reference_page_nothing_rendered_still_fires(tmp_path):
    res = _check(tmp_path, "The chart continues on PDF page 50.", [FIGURE])
    assert _cited(res, "never_opened") == ["PDF page 50"]


def test_beside_a_document_of_its_own_the_line_must_name_the_reference(
        tmp_path):
    calls = [OPEN, _read("0-1"), FIGURE]
    res = _check(tmp_path, "Figure 4-2 of the manual is on PDF page 42.",
                 calls)
    assert _cited(res, "never_opened") == []
    res = _check(tmp_path, "The groundwater note is on PDF page 42.", calls)
    assert _cited(res, "never_opened") == ["PDF page 42"]


# ---------------------------------------------------------------------------
# H: a run of printed labels is one citation
# ---------------------------------------------------------------------------

def test_a_run_of_printed_labels_is_one_citation(tmp_path):
    got = cp.find_citations("Chapter 4 (pp. 4-1 to 4-3) covers it.")
    assert [(c.kind, c.labels) for c in got] == [
        ("label", ["4-1", "4-2", "4-3"])]
    res = _check(tmp_path, "Chapter 4 (pp. 4-1 to 4-3) covers it.",
                 [OPEN, _read("10-11")],
                 labels={10: {"4-1"}, 11: {"4-2"}, 12: {"4-3"}})
    part = res["partly_opened"][0]
    assert part["opened"] == "2 of 3" and part["not_opened"] == \
        "PDF page 13"


# ---------------------------------------------------------------------------
# What still fires, and what the model and the user are handed
# ---------------------------------------------------------------------------

def test_a_tools_0_based_index_cited_as_the_pdf_page_still_fires(tmp_path):
    res = _check(tmp_path, "Section 6-4, PDF page 21: treatment may reduce "
                 "the swelling.", [OPEN, PAGE_MAP, _read("21")])
    assert _cited(res, "never_opened") == ["PDF page 21"]


def _turn(tmp_path, mode):
    conv = _conv(tmp_path)
    _write_events(conv, [OPEN, PAGE_MAP, _read("0-1")])
    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv), log=logged.append,
                             mode=mode, label_source=_labels({}))
    return check, logged


EXEMPT = "PDF page 4 is blank, and PDF pages 10 and 14 were not viewed."
FIRES = "The groundwater depth is on PDF page 10."


def test_off_hands_back_the_answer_as_it_was(tmp_path):
    check, logged = _turn(tmp_path, "off")
    assert check.feedback(EXEMPT) is None
    assert check.finish(EXEMPT) == EXEMPT
    assert logged[-1]["page_map"] and logged[-1]["named_unread"]
    assert "never_opened" not in logged[-1]
    assert check.feedback(FIRES) is None
    assert check.finish(FIRES) == FIRES
    assert logged[-1]["never_opened"][0]["cited"] == "PDF page 10"


def test_with_the_note_on_only_a_page_never_opened_is_noted(tmp_path):
    check, _logged = _turn(tmp_path, "note")
    assert check.finish(EXEMPT) == EXEMPT
    assert "Page check" in check.finish(FIRES)
