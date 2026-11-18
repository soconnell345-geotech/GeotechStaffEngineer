"""Fixes from the review of the EV-18 repairs of the cited-pages check
(``module_work/ev18/REVIEW.md``). Each repair of EV-18 removes firings; the
cases here are citations of a page no tool opened that the repairs had
stopped firing and that must fire. Made-up documents and answers.
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


def _write_events(conv, calls):
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        for i, call in enumerate(calls):
            rid = f"r{i}_{call['name']}"
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            fh.write(json.dumps({"turn": 1, "event": "tool_end",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid, "truncated": False,
                                 "result": json.dumps(call["result"])})
                     + "\n")


OPEN = {"name": "open_document", "args": {"source": "made_up_report.pdf"},
        "result": {"handle": "doc_mu", "name": "made_up_report.pdf",
                   "n_pages": 60}}
READ = {"name": "read_document", "args": {"handle": "doc_mu", "pages": "0-1"},
        "result": {"handle": "doc_mu", "pages_returned": "0-1"}}
PAGE_MAP = {"name": "document_page_map", "args": {"handle": "doc_mu"},
            "result": {"handle": "doc_mu", "n_pages": 60, "rows": [
                {"page": 3, "kind": "blank", "words": 0},
                {"page": 5, "kind": "form", "title": "Log of Boring MB-1"},
                {"page": 9, "kind": "table",
                 "title": "Summary of laboratory test results"},
                {"page": 14, "kind": "drawing_sheet", "sheet": "S-201",
                 "title": "Retaining wall section and drainage details"},
                {"page": 19, "kind": "text",
                 "title": "Slope stability analysis results"},
                {"page": 24, "kind": "text",
                 "title": "Foundation recommendations and bearing capacity"},
                {"page": 27, "kind": "text",
                 "title": "Groundwater observations"},
                {"page": 33, "kind": "scanned",
                 "title": "Concrete mix design submittal"}],
                "toc": [
                    {"level": 1, "title": "3 Laboratory testing", "page": 12},
                    {"level": 1, "title": "Appendix C Calculations",
                     "page": 50}]}}

#: Every page prints its viewer number.
PLAIN = {p: {str(p + 1)} for p in range(60)}


def _check(tmp_path, text, labels=None, calls=(OPEN, PAGE_MAP, READ)):
    conv = tmp_path / f"conv{next(_IDS)}"
    (conv / "files").mkdir(parents=True)
    _write_events(conv, list(calls))
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    rows = PLAIN if labels is None else labels
    return cp.check_answer(
        text, opened=opened,
        label_source=lambda _o, _k: {p: set(ls) for p, ls in rows.items()})


def _cited(res, key):
    return [r["cited"] for r in res.get(key, [])]


# ---------------------------------------------------------------------------
# 1. A finding said mostly in the page map's own words is still a finding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, cited", [
    ("The slope stability results (PDF page 20) fail.", "PDF page 20"),
    ("The slope stability analysis results on PDF page 20 are missing.",
     "PDF page 20"),
    ("PDF page 25: the foundation recommendations lack a bearing capacity.",
     "PDF page 25"),
    ("The drainage details on PDF page 15 are wrong.", "PDF page 15"),
    ("PDF page 15 has no drainage details.", "PDF page 15"),
    ("PDF page 34: the concrete mix design is rejected.", "PDF page 34"),
    ("Retaining wall section and drainage details (PDF page 15) omit the "
     "outlet pipe.", "PDF page 15"),
    ("Foundation recommendations (PDF page 25): bearing capacity of one "
     "hundred fifty kilopascals.", "PDF page 25"),
    ("Appendix C Calculations (PDF pages 51-60) contain an arithmetic "
     "error.", "PDF pages 51-60"),
    ("| Finding | Page |\n|---|---|\n| Calculation error | 52-55 |",
     "pages 52-55"),
])
def test_a_finding_in_title_words_still_fires(tmp_path, text, cited):
    res = _check(tmp_path, text)
    assert cited in _cited(res, "never_opened")
    assert cited not in _cited(res, "page_map")


@pytest.mark.parametrize("text, cited", [
    ("PDF page 4 is blank.", "PDF page 4"),
    ("PDF page 20 holds the slope stability analysis results.",
     "PDF page 20"),
    ("Appendix C (PDF pages 51-60) contains the calculations.",
     "PDF pages 51-60"),
    ("PDF page 4 is blank, and PDF page 47 was not viewed.", "PDF page 4"),
])
def test_only_the_page_maps_words_is_still_filed_apart(tmp_path, text,
                                                       cited):
    res = _check(tmp_path, text)
    assert cited in _cited(res, "page_map")
    assert cited not in _cited(res, "never_opened")


# ---------------------------------------------------------------------------
# 2. A page cited in a heading over its findings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, cited", [
    ("### PDF page 25: Foundation recommendations\n"
     "- The allowable pressure is not justified.", "PDF page 25"),
    ("**Page 25 (Foundation recommendations)**\n"
     "The allowable pressure is too high.", "page 25"),
    ("PDF page 25, foundation recommendations:\n"
     "- no settlement estimate", "PDF page 25"),
])
def test_a_page_in_a_heading_over_its_findings_still_fires(tmp_path, text,
                                                           cited):
    res = _check(tmp_path, text)
    assert cited in _cited(res, "never_opened")


def test_a_layout_list_is_still_what_the_page_map_said(tmp_path):
    res = _check(tmp_path, "Layout of the file:\n"
                 "- PDF page 4: blank\n"
                 "- PDF page 20: slope stability analysis results\n")
    assert _cited(res, "page_map") == ["PDF page 4", "PDF page 20"]
    assert "never_opened" not in res


# ---------------------------------------------------------------------------
# 3. Words that say a page was not opened, said of something else
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, cited", [
    # the cue belongs to the page before it
    ("I did not open page 9, and page 34 shows a water-cement ratio of "
     "0.6.", "page 34"),
    ("Pages 34 and 47 (47 not opened) give ratios of 0.6 and 0.5.",
     "page 34"),
    # the cue is about something that is not the page
    ("The contractor did not read the notes: page 34 gives a ratio of 0.6.",
     "page 34"),
    ("Page 34 gives 30 MPa for a test that was not read at 28 days.",
     "page 34"),
    ("Page 34 gives a ratio of 0.6, which I can read as over the limit.",
     "page 34"),
    ("Page 34 (ratio 0.6) was not checked by the contractor.", "page 34"),
    ("If you want, I can also check the wall: page 15 shows a 300 mm stem.",
     "page 15"),
])
def test_a_cue_said_of_something_else_does_not_excuse_a_page(tmp_path, text,
                                                             cited):
    res = _check(tmp_path, text)
    assert cited in _cited(res, "never_opened")
    assert cited not in _cited(res, "named_unread")


@pytest.mark.parametrize("text, cited", [
    ("PDF pages 34 and 47 were not viewed.", ["PDF page 34", "PDF page 47"]),
    ("Not read: PDF page 34, PDF page 47", ["PDF page 34", "PDF page 47"]),
    ("I did not open page 34 because the turn ended.", ["page 34"]),
    ("PDF page 34 (the mix submittal) was not needed.", ["PDF page 34"]),
])
def test_a_page_named_as_unread_is_still_filed_apart(tmp_path, text, cited):
    res = _check(tmp_path, text)
    assert _cited(res, "named_unread") == cited
    assert "never_opened" not in res


# ---------------------------------------------------------------------------
# 4. A page number beside a PDF page that is another place
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The ratio of 0.6 is on page 34; PDF page 1 is the cover.",
    "Strength 30 MPa on page 34, PDF page 2 the title.",
    "Page 34 or PDF page 2 gives the ratio 0.6.",
    "| Finding | Page | PDF page |\n|---|---|---|\n| Ratio 0.6 | 34 | 2 |",
])
def test_a_pdf_page_that_prints_another_number_is_not_its_pair(tmp_path,
                                                               text):
    # PDF pages 1 and 2 were read and print "1" and "2": not page 34
    res = _check(tmp_path, text)
    assert _cited(res, "never_opened") == ["page 34"]


def test_a_printed_number_beside_its_own_pdf_page_is_still_one_place(
        tmp_path):
    # the page map says PDF page 34 prints "32"; and knows nothing of 13
    labels = {33: {"32"}}
    res = _check(tmp_path, "The mix is on printed page 32 (PDF page 34).",
                 labels=labels)
    assert _cited(res, "never_opened") == ["PDF page 34"]
    res = _check(tmp_path, "See printed page 1-3; PDF page 13.",
                 labels=labels)
    assert _cited(res, "never_opened") == ["PDF page 13"]
    assert "unverified" not in res


# ---------------------------------------------------------------------------
# 5. "p. 12-16" in a plainly numbered report with one chapter-style label
# ---------------------------------------------------------------------------

def test_one_stray_label_does_not_make_a_run_unverifiable(tmp_path):
    # every page prints its plain number but one appendix page ("A-1")
    labels = dict(PLAIN)
    labels[58] = {"a-1"}
    res = _check(tmp_path, "The mix is on p. 12-16.", labels=labels)
    assert _cited(res, "never_opened") == ["pages 12-16"]
    assert "unverified" not in res
    res = _check(tmp_path, "| Finding | Page |\n|---|---|\n"
                 "| Ratio 0.6 | 12-16 |", labels=labels)
    assert _cited(res, "never_opened") == ["pages 12-16"]


def test_a_chapter_page_no_page_prints_is_still_unverified(tmp_path):
    # numbered by chapter throughout: "1-3" is one printed page
    labels = {p: {f"{p // 10 + 1}-{p % 10 + 1}"} for p in range(60)}
    res = _check(tmp_path, "The mix is on p. 9-3.", labels=labels)
    assert _cited(res, "unverified") == ["pages 9-3"] or \
        _cited(res, "unverified") == ["page 9-3"]
    assert "never_opened" not in res


# ---------------------------------------------------------------------------
# 6. A very long answer is checked at turn end, also with the switch off
# ---------------------------------------------------------------------------

def test_a_long_answer_of_thousands_of_citations_is_checked_in_time(
        tmp_path):
    import time
    text = "".join(
        f"Finding {i}: the wall on PDF page {1000 + i} was thinner than "
        f"drawn, about {i} mm, as page {3000 + i} says, and page "
        f"A-{i} too.\n" for i in range(2000))
    began = time.monotonic()
    res = _check(tmp_path, text)
    took = time.monotonic() - began
    assert res["n_cited"] == 6000
    # about 2 s with each citation compared only with its neighbours; about
    # 20 s when every citation was compared with every other
    assert took < 10, f"{took:.1f} s"


# ---------------------------------------------------------------------------
# 7. A reference's figure number inside a longer number does not name it
# ---------------------------------------------------------------------------

FIGURE = {"name": "read_reference_figure",
          "args": {"reference": "made_up_manual", "figure_number": "5-2"},
          "result": {"source": "made_up_manual.pdf", "pdf_page_index": 87,
                     "printed_page": "5-14", "figure_number": "5-2",
                     "reading": "made up"}}


def test_a_longer_number_does_not_name_the_rendered_reference(tmp_path):
    calls = (OPEN, READ, FIGURE)
    res = _check(tmp_path, "Table 15-2 is on PDF page 88 of the report, "
                 "value 0.4.", calls=calls)
    assert _cited(res, "never_opened") == ["PDF page 88"]
    res = _check(tmp_path, "Figure 5-2 (PDF page 88) gives 0.4.",
                 calls=calls)
    assert "never_opened" not in res
