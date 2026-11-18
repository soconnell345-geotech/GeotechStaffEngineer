"""EV-18: a page-map label equal to a PDF page's own viewer number is no
evidence that the page prints another number, where the documents are not
plainly numbered. The pairing re-check (a page number beside a PDF page
known to print another number is checked by itself) does not split a pair
on such a label. Made-up document and answers.
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


OPEN = {"name": "open_document", "args": {"source": "made_up_manual.pdf"},
        "result": {"handle": "doc_vn", "name": "made_up_manual.pdf",
                   "n_pages": 40}}
READ = {"name": "read_document",
        "args": {"handle": "doc_vn", "pages": "0-1"},
        "result": {"handle": "doc_vn", "pages_returned": "0-1"}}

#: Every page prints its viewer number.
PLAIN = {p: {str(p + 1)} for p in range(40)}
#: Chapter 2 is printed "2-1" to "2-9" on PDF pages 22 to 30; elsewhere the
#: page map gives only the viewer number (PDF page 1 is labelled "1").
CHAPTER = {**PLAIN, **{p: {f"2-{p - 20}"} for p in range(21, 30)}}
#: No printed number known but PDF page 1's label, its viewer number.
UNKNOWN = {**{p: set() for p in range(40)}, 0: {"1"}}

PAIR_CHAPTER = "Source: PDF page 1, printed page 2-3."
PAIR_PLAIN = "Source: PDF page 1, printed page 34."


def _check(tmp_path, text, labels):
    conv = tmp_path / f"conv{next(_IDS)}"
    (conv / "files").mkdir(parents=True)
    with open(conv / "activity.jsonl", "w", encoding="utf-8") as fh:
        for i, call in enumerate((OPEN, READ)):
            rid = f"r{i}"
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid, "args": call["args"]})
                     + "\n")
            fh.write(json.dumps({"turn": 1, "event": "tool_end",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid, "truncated": False,
                                 "result": json.dumps(call["result"])})
                     + "\n")
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    return cp.check_answer(
        text, opened=opened,
        label_source=lambda _o, _k: {p: set(ls) for p, ls in labels.items()})


def _checked_alone(res):
    """The citations the check looked at by themselves (a firing or a
    printed number it cannot place)."""
    return [r["cited"] for key in ("never_opened", "unverified")
            for r in res.get(key, [])]


def test_chapter_numbered_a_viewer_number_label_keeps_the_pair(tmp_path):
    """Guards the repair: in a document numbered by chapter, a PDF page the
    page map labels only with its viewer number is not taken as printing
    another number, so the printed page beside it is not checked alone."""
    res = _check(tmp_path, PAIR_CHAPTER, CHAPTER)
    assert _checked_alone(res) == []


def test_numbering_unknown_a_viewer_number_label_keeps_the_pair(tmp_path):
    """Guards the repair where the printed numbers are not known: a label
    equal to the viewer number does not split the pair either."""
    res = _check(tmp_path, PAIR_CHAPTER, UNKNOWN)
    assert _checked_alone(res) == []


def test_plainly_numbered_a_viewer_number_label_still_splits(tmp_path):
    """Guards what the repair keeps: where pages print their viewer
    numbers, PDF page 1 printing "1" is a different place from printed
    page 34, which is checked by itself and fires."""
    res = _check(tmp_path, PAIR_PLAIN, PLAIN)
    assert any("34" in c for c in _cited(res, "never_opened"))


@pytest.mark.parametrize("labels, text", [
    ({**CHAPTER, 0: {"1-7"}}, PAIR_CHAPTER),
    ({**UNKNOWN, 0: {"iv"}}, PAIR_CHAPTER),
    ({**PLAIN, 0: {"7"}}, PAIR_PLAIN),
], ids=["chapter", "unknown", "plain"])
def test_a_label_that_truly_differs_still_splits(tmp_path, labels, text):
    """Guards fix 4 itself, in each numbering: a PDF page the page map
    labels with another number is another place, so the printed page beside
    it is checked by itself."""
    res = _check(tmp_path, text, labels)
    assert _checked_alone(res) != []


def _cited(res, key):
    return [r["cited"] for r in res.get(key, [])]
