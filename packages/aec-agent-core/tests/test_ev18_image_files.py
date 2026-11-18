"""An image file is not a document with pages (BACKLOG EV-18; finding 4 of
``module_work/b13_whole/REVIEW_B.md``).

A zoom on an uploaded picture is recorded as page 0 of a document named
after the picture. The answer check read that as a page of the documents:
"PDF page 1" of ANY document then counted as opened, and the picture's
missing page labels made every document's printed labels unknown, so the
check went quiet for the conversation. Made-up documents and answers.
"""

from __future__ import annotations

import itertools
import json

import pytest

from aec_agent_core import cited_documents as cd
from aec_agent_core import cited_pages as cp
from aec_agent_core import review_flags

_IDS = itertools.count(1)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (review_flags.ALL_ENVS + review_flags.LATER_ENVS
                + review_flags.SETTINGS_ENVS):
        monkeypatch.delenv(env, raising=False)
    cp.clear_cache()
    cd.clear_cache()
    yield
    cp.clear_cache()
    cd.clear_cache()


def _conv(tmp_path, calls):
    conv = tmp_path / f"conv{next(_IDS)}"
    (conv / "files").mkdir(parents=True)
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        for i, call in enumerate(calls):
            rid = f"r{i}_{call['name']}"
            result = call["result"]
            if not isinstance(result, str):
                result = json.dumps(result)
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            fh.write(json.dumps({"turn": 1, "event": "tool_end",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid, "truncated": False,
                                 "result": result}) + "\n")
    return conv


OPEN = {"name": "open_document", "args": {"source": "made_up_report.pdf"},
        "result": {"handle": "doc_mu", "name": "made_up_report.pdf",
                   "n_pages": 60}}
READ = {"name": "read_document", "args": {"handle": "doc_mu", "pages": "4-5"},
        "result": {"handle": "doc_mu", "pages_returned": "4-5"}}
ZOOM = {"name": "render_region",
        "args": {"source": "site.png", "page": 0, "bbox": [0, 0, 10, 10]},
        "result": "The picture shows a made-up fence line beside a ditch."}
OPEN_IMAGE = {"name": "open_document", "args": {"source": "Site.PNG"},
              "result": {"handle": "doc_pic", "name": "Site.PNG",
                         "n_pages": 1}}
ZOOM_HANDLE = {"name": "render_region",
               "args": {"handle": "doc_pic", "page": 0,
                        "bbox": [0, 0, 10, 10]},
               "result": "The picture shows a made-up fence line."}

ANSWER = "The bearing value is on PDF page 1 and the settlement on page 12."


def _check(tmp_path, text, calls):
    conv = _conv(tmp_path, calls)
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    asked = []

    def labels(_opened, key):
        asked.append(key)
        if key.lower().endswith(".png"):
            return None                # a picture has no page labels
        return {p: {str(p + 1)} for p in range(60)}

    return cp.check_answer(text, opened=opened, label_source=labels), asked


def _cited(res, key):
    return [r["cited"] for r in res.get(key, [])]


def test_a_zoom_on_a_picture_opens_no_page_of_the_report(tmp_path):
    alone, _ = _check(tmp_path, ANSWER, [OPEN, READ])
    assert _cited(alone, "never_opened") == ["PDF page 1", "page 12"]
    with_zoom, _ = _check(tmp_path, ANSWER, [OPEN, READ, ZOOM])
    assert _cited(with_zoom, "never_opened") == ["PDF page 1", "page 12"]
    assert not with_zoom.get("unverified")


def test_a_picture_does_not_make_the_printed_labels_unknown(tmp_path):
    res, asked = _check(tmp_path, "The settlement is on page 12; the bearing "
                                  "value is on page 5.", [OPEN, READ, ZOOM])
    assert asked == ["made_up_report.pdf"]
    assert _cited(res, "never_opened") == ["page 12"]


def test_a_picture_opened_by_handle_is_left_out_too(tmp_path):
    res, asked = _check(tmp_path, ANSWER,
                        [OPEN, READ, OPEN_IMAGE, ZOOM_HANDLE])
    assert asked == ["made_up_report.pdf"]
    assert _cited(res, "never_opened") == ["PDF page 1", "page 12"]


def test_only_a_picture_pdf_page_1_is_no_page_of_it(tmp_path):
    """Neighbour 1: a conversation with only an image file. The picture is
    not a document whose pages can be cited as "PDF page N"."""
    res, _ = _check(tmp_path, "The fence is shown on PDF page 1.", [ZOOM])
    assert _cited(res, "never_opened") == ["PDF page 1"]


def test_only_a_picture_page_1_is_as_with_no_document(tmp_path):
    """Neighbour 1: "page 1" with only a picture is listed as it is when
    the conversation read no document at all (not checkable)."""
    text = "The fence is shown on page 1."
    res, _ = _check(tmp_path, text, [ZOOM])
    none, _ = _check(tmp_path, text, [])
    assert _cited(none, "unverified") == ["page 1"]
    assert _cited(res, "unverified") == ["page 1"]
    assert not res.get("never_opened")


def test_the_record_of_pages_looked_at_still_holds_the_picture(tmp_path):
    """Only the answer check leaves the picture out: the record other
    readers use (the photograph-page check's attachment list) keeps it."""
    conv = _conv(tmp_path, [OPEN, READ, ZOOM])
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    assert opened.documents() == ["made_up_report.pdf", "site.png"]
    assert opened.looked("site.png") == {0}
    assert cp.without_images(opened).documents() == ["made_up_report.pdf"]


def test_cited_documents_a_look_at_a_picture_names_no_other_document(
        tmp_path):
    """Neighbour 2: the cited-documents check goes by what a result SAYS,
    not by which file was looked at, so a look at a picture is a source
    only for a document its own result names."""
    text = "Per GEC-11, the reinforcement length is 0.7 of the wall height."
    conv = _conv(tmp_path, [OPEN, READ, ZOOM])
    res = cd.check_text(text, record_folder=str(conv), user_words=[])
    assert cd.not_looked_up(res) == ["GEC-11"]
    cover = dict(ZOOM, result="The picture is a cover that reads GEC-11.")
    conv = _conv(tmp_path, [OPEN, READ, cover])
    res = cd.check_text(text, record_folder=str(conv), user_words=[])
    assert cd.not_looked_up(res) == []
