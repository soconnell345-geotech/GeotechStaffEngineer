"""Live smoke wave 2c, second pass (module_work/live_smoke/runs/
w2c-review-sonnet/REVIEW.md): the marked copy's result.

* E9  ``remove=`` names a mark by its comment OR its visible label (F47
      Alice t2: ``remove: ["Embedment: not shown"]`` was a label's words and
      was refused, "no mark this app wrote says that").
* E10 the result says what the copy holds -- the marks this call added and
      the ones kept from before, by author (F47 Bob t3: told "a NEW file: the
      document you opened is unchanged" and ``n_written: 1``, the model said
      the document's five markups were not in the copy; they were).

Offline: planlens' synthetic review document (Reviewer A's stamp and callout,
Contractor B's cloud, its box and an arrow: the F47 set), no engine.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from funhouse_agent import document_tools  # noqa: E402
from funhouse_agent.deep.tools import make_vision_tools  # noqa: E402
from planlens.testing import build_synthetic_review_document  # noqa: E402

pytestmark = pytest.mark.skipif(
    not document_tools.has_tool("annotate_document"),
    reason="installed planlens predates annotate_document")

#: What the synthetic document carries before anything is written: the F47
#: set exactly.
THEIRS = {"Reviewer A": 2, "Contractor B": 3}


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


@pytest.fixture
def folder(tmp_path, monkeypatch):
    work = tmp_path / "conv" / "files"
    work.mkdir(parents=True)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))
    monkeypatch.delenv("GEOTECH_MARKUP_AUTHOR", raising=False)
    return work


def _setup(gt):
    tools = {t.name: t for t in make_vision_tools(
        engine=None, attachments={"set.pdf": gt.pdf})}
    handle = json.loads(tools["open_document"].invoke(
        {"source": "set.pdf"}))["handle"]
    return tools, handle


def _annotate(tools, **kwargs):
    kwargs.setdefault("check", False)
    return json.loads(tools["annotate_document"].invoke(kwargs))


def _ours_in(folder, name):
    """This app's marks and labels in a written copy: (comments, labels)."""
    doc = fitz.open(str(folder / name))
    try:
        comments, labels = [], []
        for page in doc:
            for a in page.annots() or []:
                info = a.info or {}
                if "GeotechStaffEngineer" not in (info.get("title") or ""):
                    continue
                if info.get("subject") == "Label":
                    labels.append(info.get("content"))
                else:
                    comments.append(info.get("content"))
        return comments, labels
    finally:
        doc.close()


def _three_marks(gt):
    page = gt.narrative_page
    return [
        {"kind": "box", "page": page, "bbox": [72, 300, 300, 330],
         "comment": "[#1] The pile embedment depth is missing.",
         "label": "Embedment: not shown"},
        {"kind": "box", "page": page, "bbox": [72, 360, 300, 390],
         "comment": "[#2] Which boring is this? Its location is not shown.",
         "label": "Boring?"},
        {"kind": "box", "page": page, "bbox": [72, 420, 300, 450],
         "comment": "[#3] The boring depth differs from the log.",
         "label": "Depth"},
    ]


# ---------------------------------------------------------------------------
# E9: remove by a label's words
# ---------------------------------------------------------------------------

def test_a_mark_is_removed_by_the_words_of_its_label(gt, folder):
    """F47 Alice t2: the label's words name the mark; it goes, with its
    label, and every other mark stays."""
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=_three_marks(gt))
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    remove=["Embedment: not shown"])
    assert "not_removed" not in out, out
    (gone,) = out["removed"]
    assert gone["says"].startswith("[#1]") and gone["with_attached"] >= 1
    comments, labels = _ours_in(folder, "set_marked.pdf")
    assert sorted(c[:4] for c in comments) == ["[#2]", "[#3]"]
    assert "Embedment: not shown" not in labels
    assert sorted(labels) == ["Boring?", "Depth"]


def test_words_in_one_marks_label_and_anothers_comment_are_refused(gt,
                                                                   folder):
    """"not shown" is #1's label and in #2's comment: two marks. A remover
    reading comments alone would have taken #2 without a word; both stay,
    and the refusal names both."""
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=_three_marks(gt))
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    remove=["not shown"])
    assert out["removed"] == []
    (refused,) = out["not_removed"]
    assert refused["remove"] == "not shown"
    assert refused["reason"].startswith("2 marks say that in their comment "
                                        "or label (p")
    assert "give the id" in refused["reason"]
    comments, labels = _ours_in(folder, "set_marked.pdf")
    assert sorted(c[:4] for c in comments) == ["[#1]", "[#2]", "[#3]"]
    assert sorted(labels) == ["Boring?", "Depth", "Embedment: not shown"]


def test_words_naming_nothing_are_refused_and_new_marks_still_land(gt,
                                                                   folder):
    """Nothing to take out, and a mark to add: the copy is still the base,
    so the earlier marks stay and the new one is added onto it."""
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=_three_marks(gt)[:2])
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    remove=["no such words anywhere"],
                    markups=_three_marks(gt)[2:])
    assert "error" not in out, out
    assert out["removed"] == []
    (refused,) = out["not_removed"]
    assert refused["reason"] == ("no mark this app wrote says that, in its "
                                 "comment or its label")
    comments, _labels = _ours_in(folder, "set_marked.pdf")
    assert sorted(c[:4] for c in comments) == ["[#1]", "[#2]", "[#3]"]
    assert out["in_file"]["kept_from_before"] == 5 + 2


def test_removing_by_id_and_by_comment_still_work(gt, folder):
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=_three_marks(gt))
    copy = json.loads(tools["open_document"].invoke(
        {"source": "set_marked.pdf"}))["handle"]
    listed = json.loads(tools["document_markups"].invoke(
        {"handle": copy, "author": "AI draft"}))["markups"]
    id3 = next(row.split("[", 1)[1].split("]", 1)[0] for row in listed
               if "[#3]" in row)
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    remove=[id3, "which boring"])
    assert sorted(r["says"][:4] for r in out["removed"]) == ["[#2]", "[#3]"]
    assert "not_removed" not in out


# ---------------------------------------------------------------------------
# E10: what the copy holds
# ---------------------------------------------------------------------------

def test_the_result_says_the_documents_markups_are_in_the_copy(gt, folder):
    """F47 Bob t3, exactly: one box onto a document carrying Reviewer A's
    two markups and Contractor B's three."""
    tools, handle = _setup(gt)
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    append=False, markups=[
                        {"kind": "box", "page": gt.narrative_page,
                         "bbox": [72, 300, 300, 330],
                         "comment": "Bob - checked", "label": "Bob - checked"}])
    assert out["n_written"] == 1
    assert out["in_file"] == {"markups": 6, "added_by_this_call": 1,
                              "kept_from_before": 5,
                              "kept_by_author": THEIRS}
    note = out["note"]
    assert note.startswith(
        "'set_marked.pdf' holds 6 markups: 1 added by this call and 5 kept "
        "from before (Reviewer A 2, Contractor B 3). Every markup already on "
        "the document is in the copy")
    assert "the document you opened is unchanged" in note
    assert "a NEW file" not in note
    assert "READ the skipped rows" in note          # planlens' rest is kept
    # counted off the file: a label is part of its mark, and the two hidden
    # AutoCAD SHX text squares are the drawing's text, not markups
    doc = fitz.open(str(folder / "set_marked.pdf"))
    try:
        authors = [(a.info or {}).get("title") for p in doc
                   for a in p.annots() or []
                   if a.type[1] not in ("Popup", "Link", "Widget")]
    finally:
        doc.close()
    assert authors.count("AutoCAD SHX Text") == 2
    assert len(authors) == 5 + 1 + 1 + 2      # theirs, the box, its label, CAD


def test_a_second_call_counts_this_apps_earlier_marks(gt, folder):
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=_three_marks(gt)[:2])
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    markups=_three_marks(gt)[2:])
    assert out["in_file"] == {
        "markups": 8, "added_by_this_call": 1, "kept_from_before": 7,
        "kept_by_author": {**THEIRS, "this app, earlier calls": 2}}
    assert "(Reviewer A 2, Contractor B 3, this app, earlier calls 2)" in \
        out["note"]


def test_a_removal_is_counted(gt, folder):
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=_three_marks(gt))
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    remove=["Embedment: not shown"])
    assert out["in_file"] == {
        "markups": 7, "added_by_this_call": 0, "kept_from_before": 7,
        "kept_by_author": {**THEIRS, "this app, earlier calls": 2},
        "removed_by_this_call": 1}
    assert out["note"].startswith(
        "'set_marked.pdf' holds 7 markups: 0 added by this call and 7 kept "
        "from before")
    assert "; 1 removed by this call." in out["note"]


def test_the_tool_describes_its_result(gt, folder):
    tools, _handle = _setup(gt)
    desc = tools["annotate_document"].description
    assert "in_file" in desc and "kept from before" in desc
    assert "words from its comment or its label" in desc
    assert "never compared with what is under the mark" in desc
