"""Review of the image-file repair of the cited-pages check (BACKLOG EV-18).

The repair leaves image files out of the documents the answer check reads.
A citation of a picture the conversation looked at ("site.png, page 1"; any
page number where pictures are all the conversation has) was then read as a
page of the report, or of no document, and listed. These tests state that
such a citation is as it was before the repair (not listed), and that the
repair itself holds on other made-up conversations. Made-up text only.
"""

from __future__ import annotations

import itertools
import json
import os

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


def _conv(tmp_path, calls):
    conv = tmp_path / f"conv{next(_IDS)}"
    (conv / "files").mkdir(parents=True)
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        for i, (name, args, result) in enumerate(calls):
            if not isinstance(result, str):
                result = json.dumps(result)
            for rec in ({"event": "tool_start", "args": args},
                        {"event": "tool_end", "truncated": False,
                         "result": result}):
                fh.write(json.dumps(dict(rec, turn=1, agent="primary",
                                         name=name, run_id=f"r{i}")) + "\n")
    return conv


def _labels(_opened, key):
    if key.lower().endswith(".pdf"):
        return {p: {str(p + 1)} for p in range(40)}
    return None                        # a picture, a bare handle


def _open(name, handle, n=40):
    return ("open_document", {"source": name},
            {"handle": handle, "name": os.path.basename(name), "n_pages": n})


def _read(handle, pages):
    return ("read_document", {"handle": handle, "pages": pages},
            {"handle": handle, "pages_returned": pages})


def _zoom(page=0, **ref):
    return ("render_region", dict(ref, page=page, bbox=[0, 0, 5, 5]),
            "A made-up view of a culvert headwall.")


REPORT = [_open("drain_memo.pdf", "doc_a"), _read("doc_a", "2-3")]
ANSWER = ("The pipe schedule is on PDF page 1 and the invert levels on "
          "page 17; see also p. 3.")


def _check(tmp_path, text, calls):
    conv = _conv(tmp_path, calls)
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    return cp.check_answer(text, opened=opened, label_source=_labels)


def _cited(res, key):
    return [r["cited"] for r in res.get(key, [])]


# -- the repair, on other conversations ---------------------------------------

@pytest.mark.parametrize("picture", [
    "yard.png", "Site.PNG", "scan.tif", "uploads/photos/yard.jpeg",
    "C:\\up\\photos\\Yard.JPG", "a.webp", "b.GIF", "c.bmp", "d.tiff"])
@pytest.mark.parametrize("first", [True, False])
def test_a_picture_in_either_order_changes_nothing_of_the_report(
        tmp_path, picture, first):
    zoom = [_zoom(source=picture)]
    res = _check(tmp_path, ANSWER, zoom + REPORT if first else REPORT + zoom)
    assert _cited(res, "never_opened") == ["PDF page 1", "page 17"]
    assert not res.get("unverified")


def test_two_reports_and_a_picture(tmp_path):
    second = [_open("culvert_calc.pdf", "doc_b"), _read("doc_b", "8")]
    res = _check(tmp_path, ANSWER,
                 REPORT + second + [_zoom(source="yard.png")])
    assert _cited(res, "never_opened") == ["PDF page 1", "page 17"]


def test_a_pdf_with_an_image_suffix_inside_its_name_is_a_document(tmp_path):
    calls = [_open("plan.png.pdf", "doc_c", 5),
             ("analyze_pdf_page", {"source": "plan.png.pdf", "page": 0},
              "A made-up plan of a yard.")]
    res = _check(tmp_path, "The plan is on PDF page 1; the legend on PDF "
                           "page 4.", calls)
    assert _cited(res, "never_opened") == ["PDF page 4"]


# -- a citation of the picture itself -----------------------------------------

@pytest.mark.parametrize("text", [
    "site.png, page 1, shows the ditch.",
    "The ditch is on page 1 of site.png.",
    "The ditch is on PDF page 1 (Site.PNG).",
])
def test_a_page_cited_of_a_named_picture_that_was_looked_at(tmp_path, text):
    res = _check(tmp_path, text, REPORT + [_zoom(source="site.png")])
    assert res["n_cited"] == 1
    assert not res.get("never_opened") and not res.get("unverified")


def test_the_named_picture_grants_nothing_in_the_next_sentence(tmp_path):
    res = _check(tmp_path, "The photo site.png shows the ditch. The pipe "
                           "schedule is on PDF page 1.",
                 REPORT + [_zoom(source="site.png")])
    assert _cited(res, "never_opened") == ["PDF page 1"]


def test_a_named_picture_nothing_looked_at_grants_nothing(tmp_path):
    res = _check(tmp_path, "site.png, page 1, shows the ditch.",
                 REPORT + [_open("site.png", "doc_p", 1)])
    assert _cited(res, "never_opened") == ["page 1"]


@pytest.mark.parametrize("text", [
    "The stamp is on page 1 of the scan.",
    "The stamp is on PDF page 1.",
    "The stamp is on p. 1 (scan.tif).",
])
def test_only_a_picture_its_page_is_the_page_looked_at(tmp_path, text):
    res = _check(tmp_path, text, [_zoom(source="scan.tif")])
    assert res["checked"] and res["n_cited"] == 1
    assert not res.get("never_opened") and not res.get("unverified")


def test_only_a_picture_a_page_it_has_not_is_still_listed(tmp_path):
    res = _check(tmp_path, "The stamp is on PDF page 2.",
                 [_zoom(source="scan.tif")])
    assert _cited(res, "never_opened") == ["PDF page 2"]


def test_note_mode_says_nothing_of_a_picture_that_was_looked_at(tmp_path):
    conv = _conv(tmp_path, REPORT + [_zoom(source="site.png")])
    events = []
    turn = cp.TurnPageCheck(record_folder=str(conv),
                            folder=str(conv / "files"), log=events.append,
                            mode="note", label_source=_labels,
                            photo_mode="off")
    answer = "The fence line leans: site.png, page 1."
    assert turn.finish(answer) == answer
    noted = turn.finish("The pipe schedule is on PDF page 1.")
    assert "Page check (by the app)" in noted and "PDF page 1" in noted
