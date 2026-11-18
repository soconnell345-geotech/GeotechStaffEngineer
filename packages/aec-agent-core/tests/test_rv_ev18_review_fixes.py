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
