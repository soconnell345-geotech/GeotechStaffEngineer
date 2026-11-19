"""Review of EV-18's viewer-number-label repair: a PDF page the page map
labels only with its viewer number, cited beside a printed label that
ANOTHER page is known to print, is still two places, so the wrong pairing
is listed; the right pairing on such a page stays quiet. Made-up document
and answers.
"""

from __future__ import annotations

import itertools
import json

import pytest

from aec_agent_core import cited_pages as cp
from aec_agent_core import review_flags

_IDS = itertools.count(1)
N = 40


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (review_flags.ALL_ENVS + review_flags.LATER_ENVS
                + review_flags.SETTINGS_ENVS):
        monkeypatch.delenv(env, raising=False)
    cp.clear_cache()
    yield
    cp.clear_cache()


def _labels():
    """Chapter 1 is printed "1-1" to "1-10" on PDF pages 11 to 20 and
    chapter 2 on PDF pages 21 to 30; every page carries its viewer number,
    and the printed number was not read on PDF pages 13 and 15."""
    rows = {p: {str(p + 1)} for p in range(N)}
    for p in range(10, 30):
        if p not in (12, 14):
            rows[p].add(f"{(p - 10) // 10 + 1}-{(p - 10) % 10 + 1}")
    return rows


def _check(tmp_path, text):
    conv = tmp_path / f"conv{next(_IDS)}"
    (conv / "files").mkdir(parents=True)
    calls = [
        ("open_document", {"source": "made_up_study.pdf"},
         {"handle": "doc_rv", "name": "made_up_study.pdf", "n_pages": N}),
        ("read_document", {"handle": "doc_rv", "pages": "12-14"},
         {"handle": "doc_rv", "pages_returned": "12-14"})]
    with open(conv / "activity.jsonl", "w", encoding="utf-8") as fh:
        for i, (name, args, result) in enumerate(calls):
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "agent": "primary", "name": name,
                                 "run_id": f"r{i}", "args": args}) + "\n")
            fh.write(json.dumps({"turn": 1, "event": "tool_end",
                                 "agent": "primary", "name": name,
                                 "run_id": f"r{i}", "truncated": False,
                                 "result": json.dumps(result)}) + "\n")
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    return cp.check_answer(text, opened=opened,
                           label_source=lambda _o, _k: _labels())


def _listed(res):
    return [(r["cited"], r.get("pages")) for key in
            ("never_opened", "partly_opened", "unverified")
            for r in res.get(key, [])]


@pytest.mark.parametrize("text, page", [
    ("The value is on PDF page 13 (printed page 1-7).", "PDF page 17"),
    ("The value is on page 2-4 (PDF p. 15).", "PDF page 24"),
], ids=["pdf-first", "printed-first"])
def test_a_printed_page_another_page_prints_is_still_listed(tmp_path, text,
                                                            page):
    """Guards the review fix: PDF page 13 was opened and is labelled only
    "13", but "1-7" is known to be printed on PDF page 17, which nobody
    opened: the wrong pairing is listed as before the repair."""
    res = _check(tmp_path, text)
    assert [p for _c, p in _listed(res)] == [page]


@pytest.mark.parametrize("text", [
    "The value is on PDF page 13 (printed page 1-3).",
    "The value is on page 1-3 (PDF p. 13).",
    "The load is on PDF page 13 (printed page 1-3) and the depth on PDF "
    "page 15 (printed page 1-5).",
])
def test_the_right_pairing_on_an_unread_label_stays_quiet(tmp_path, text):
    """Guards the repair itself against the review fix: no page is known
    to print "1-3" or "1-5", so the PDF page given beside each decides."""
    assert _listed(_check(tmp_path, text)) == []
