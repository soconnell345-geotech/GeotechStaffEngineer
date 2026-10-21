"""BACKLOG DR-32: the app's side of misspelt keys in annotate_document and
search_document (a survey after GM-58, 2026-10-16: planlens refuses
misspelt arguments and markup keys, and ``strict_tool`` refuses unknown
top-level arguments, but four app-side gaps let a key go without a word).

* A ``remove=`` entry written as an object with a key other than
  ``id``/``text`` (``{"markup_id": ...}``) removed nothing and said nothing:
  it is refused, with the valid keys and a did-you-mean.
* The model was shown ``markups`` as an array of anything: it is now shown
  planlens' own markup fields (read off the installed spec), closed to other
  keys -- while the names planlens forgives (``color``, ``box``, a nested
  ``anchor``, a kind's other name) and the app's ``pdf_page`` still pass.
* A mark with no comment was written as an empty note without a word (7
  circle marks of one Foundry task): the result says so.
* ``min_score`` given without ``fuzzy`` was dropped without a word: the
  result says so.

And one found on the way: a markup's own ``author`` replaced the app's
signature for that mark (one recorded call signed 28 marks "AI draft
review"); it is dropped with a note.

Offline: planlens' synthetic review document, no engine.
"""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from funhouse_agent import document_tools  # noqa: E402
from funhouse_agent.deep.tools import (MIN_SCORE_UNUSED_NOTE,  # noqa: E402
                                       make_vision_tools)
from planlens.testing import build_synthetic_review_document  # noqa: E402

pytestmark = pytest.mark.skipif(
    not document_tools.has_tool("annotate_document"),
    reason="installed planlens predates annotate_document")


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


def _shown(tool):
    from langchain_core.utils.function_calling import convert_to_openai_tool
    return convert_to_openai_tool(tool)["function"]["parameters"]


def _signed(path, comments):
    """``(author, comment)`` of each mark in the copy saying one of
    ``comments`` (its label left out)."""
    doc = pymupdf.open(str(path))
    try:
        return [((a.info or {}).get("title"), (a.info or {}).get("content"))
                for p in doc for a in p.annots() or []
                if (a.info or {}).get("subject") != "Label"
                and (a.info or {}).get("content") in comments]
    finally:
        doc.close()


def _box(gt, y, comment="a remark", **extra):
    return {"kind": "box", "page": gt.narrative_page,
            "bbox": [72, y, 300, y + 30], "comment": comment, **extra}


# ---------------------------------------------------------------------------
# 1. remove= objects with a key other than id / text
# ---------------------------------------------------------------------------

def test_a_remove_object_with_an_unknown_key_is_refused(gt, folder):
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=[_box(gt, 300, "[#1] first"),
                       _box(gt, 360, "[#2] second")])
    before = (folder / "set_marked.pdf").read_bytes()
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    remove=[{"markup_id": "p0.m0"}],
                    markups=[_box(gt, 420, "[#3] third")])
    assert out["error"] == (
        "annotate_document: remove[0] has unknown key(s) ['markup_id']. "
        "Did you mean: markup_id -> id? Valid keys: ['id', 'text'].")
    assert "nothing was removed or written" in out["hint"]
    # nothing taken out, nothing added
    assert (folder / "set_marked.pdf").read_bytes() == before


def test_remove_objects_with_id_or_text_still_work(gt, folder):
    tools, handle = _setup(gt)
    _annotate(tools, handle=handle, output_path="set_marked.pdf",
              markups=[_box(gt, 300, "[#1] first"),
                       _box(gt, 360, "[#2] second")])
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    remove=[{"text": "[#2] second"}])
    assert "error" not in out, out
    assert [r["says"] for r in out["removed"]] == ["[#2] second"]


@pytest.mark.parametrize("remove, ok", [
    (["p0.m1", "words"], True),
    ([{"id": "p0.m1"}, {"text": "words"}, {"id": "p0.m2", "text": "w"}],
     True),
    ("p0.m1", True),
    ([""], True),                       # the app's own "copy is the base"
    (None, True),
    ({"markup_id": "p0.m1"}, False),    # one object, not in a list
    ([{"comment": "words"}], False),
    (["p0.m1", {"id": "p0.m1", "colour": "red"}], False),
])
def test_remove_problem(remove, ok):
    problem = document_tools.remove_problem(remove)
    assert (problem is None) is ok, problem


def test_the_did_you_mean_points_words_at_text():
    problem = document_tools.remove_problem([{"comment": "words"}])
    assert "Did you mean: comment -> text?" in problem["error"]


# ---------------------------------------------------------------------------
# 2. the markups the model is shown
# ---------------------------------------------------------------------------

def _spec_items():
    from planlens.tools.specs import TOOL_SPECS
    spec = next(s for s in TOOL_SPECS if s["name"] == "annotate_document")
    return spec["parameters"]["properties"]["markups"]["items"]


def test_markups_are_shown_with_planlens_own_fields(gt):
    tools, _handle = _setup(gt)
    shown = _shown(tools["annotate_document"])["properties"]["markups"]
    items = shown["items"]
    planlens_items = _spec_items()
    assert shown["type"] == "array"
    assert items["additionalProperties"] is False
    # planlens' own fields, read off its spec; described only where the
    # tool's description does not already say it
    for name, field in planlens_items["properties"].items():
        if name == "author":
            continue
        if name in document_tools.DESCRIBED_MARKUP_FIELDS:
            assert items["properties"][name] == field, name
        else:
            bare = {k: v for k, v in field.items() if k != "description"}
            assert items["properties"][name] == bare, name
    assert "circle" in items["properties"]["kind"]["description"]
    assert "callout" in items["properties"]["bbox"]["description"]
    # the app signs every comment; pdf_page is the app's own
    assert "author" not in items["properties"]
    assert items["properties"]["pdf_page"]["type"] == "integer"
    # page or pdf_page; the comment stays required, as planlens says
    assert "page" not in items["required"]
    assert {"kind", "comment"} <= set(items["required"])


def test_the_shown_schema_stays_small_and_inline(gt):
    """Every model call carries it: a cap on its size, and no $ref / $defs
    a provider might not resolve."""
    from langchain_core.utils.function_calling import convert_to_openai_tool
    tools, _handle = _setup(gt)
    shown = json.dumps(convert_to_openai_tool(tools["annotate_document"]))
    assert len(shown) <= 7500, len(shown)
    assert "$ref" not in shown and "$defs" not in shown


def test_remove_is_shown_as_strings_and_nothing_else_changes(gt):
    tools, _handle = _setup(gt)
    props = _shown(tools["annotate_document"])["properties"]
    assert props["remove"]["items"] == {"type": "string"}
    assert set(props) == {"handle", "markups", "output_path", "append",
                          "check", "remove"}


def test_the_shown_markups_follow_the_installed_planlens(monkeypatch):
    """A field a newer planlens adds is shown with no change here; a
    planlens whose spec has no item schema leaves the list as it was."""
    from planlens.tools import specs as planlens_specs
    newer = copy.deepcopy(planlens_specs.TOOL_SPECS)
    spec = next(s for s in newer if s["name"] == "annotate_document")
    spec["parameters"]["properties"]["markups"]["items"]["properties"][
        "stamp"] = {"type": "string", "description": "A new field."}
    monkeypatch.setattr(planlens_specs, "TOOL_SPECS", newer)
    tools = {t.name: t for t in make_vision_tools(engine=None)}
    items = _shown(tools["annotate_document"])["properties"]["markups"][
        "items"]
    # shown by name and type; its words are the tool description's job
    assert items["properties"]["stamp"] == {"type": "string"}

    older = copy.deepcopy(newer)
    spec = next(s for s in older if s["name"] == "annotate_document")
    del spec["parameters"]["properties"]["markups"]["items"]
    monkeypatch.setattr(planlens_specs, "TOOL_SPECS", older)
    assert document_tools.annotate_markups_schema() is None
    tools = {t.name: t for t in make_vision_tools(engine=None)}
    markups = _shown(tools["annotate_document"])["properties"]["markups"]
    assert markups["anyOf"][0] == {"items": {}, "type": "array"}


def test_what_planlens_forgives_still_passes(gt, folder):
    """The schema is what the model is SHOWN; the call is not checked
    against it, so color/colour, box/page_bbox, a nested anchor, a kind's
    other name and pdf_page all still go on (8 Foundry runs once lost a
    turn to color)."""
    tools, handle = _setup(gt)
    page = gt.narrative_page
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    append=False, markups=[
                        {"kind": "box", "page": page, "comment": "c0",
                         "box": [72, 300, 300, 330], "color": "blue"},
                        {"kind": "box", "page": page, "comment": "c1",
                         "page_bbox": [72, 360, 300, 390], "colour": "red"},
                        {"kind": "note", "page": page, "comment": "c2",
                         "anchor": {"point": [120, 120]}},
                        {"kind": "text", "pdf_page": page + 1, "comment": "c3",
                         "point": [150, 150]},
                    ])
    assert "error" not in out, out
    assert out["n_written"] == 4 and not out.get("skipped")
    notes = {row["index"]: " ".join(row["notes"]) for row in out["adjusted"]}
    assert "'color' is not a markup field" in notes[0]
    assert "'colour' is not a markup field" in notes[1]
    assert "anchor object" in notes[2]
    assert "kind 'text' was written as a note" in notes[3]


def test_an_unknown_argument_is_still_refused(gt):
    tools, handle = _setup(gt)
    out = json.loads(tools["annotate_document"].invoke(
        {"handle": handle, "markups": [], "author": "Me"}))
    assert out["unknown_arguments"] == ["author"]


# ---------------------------------------------------------------------------
# a markup's own author (found on the way)
# ---------------------------------------------------------------------------

def test_a_markups_own_author_does_not_sign_it(gt, folder):
    tools, handle = _setup(gt)
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    append=False, markups=[
                        _box(gt, 300, "[#1] signed", author="Bob"),
                        _box(gt, 360, "[#2] plain")])
    assert out["n_written"] == 2
    (row,) = out["adjusted"]
    assert row["index"] == 0
    assert row["notes"] == [document_tools.AUTHOR_IGNORED_NOTE]
    rows = _signed(folder / "set_marked.pdf", ("[#1] signed", "[#2] plain"))
    assert sorted(c for _a, c in rows) == ["[#1] signed", "[#2] plain"]
    assert all(a == out["author"] for a, _c in rows), rows
    assert "Bob" not in {a for a, _c in rows}


# ---------------------------------------------------------------------------
# 3. marks with no comment
# ---------------------------------------------------------------------------

def test_marks_with_no_comment_are_said(gt, folder):
    tools, handle = _setup(gt)
    page = gt.narrative_page
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    append=False, markups=[
                        {"kind": "circle", "page": page,
                         "bbox": [72, 300, 300, 330]},
                        _box(gt, 360, "said"),
                        {"kind": "circle", "page": page, "comment": "  ",
                         "bbox": [72, 420, 300, 450], "label": "A"},
                        # skipped (no such page): not written, not counted
                        {"kind": "box", "page": 99, "bbox": [1, 2, 3, 4]},
                    ])
    assert out["n_written"] == 3            # written, not refused
    assert out["empty_comments"].startswith(
        "markups 0, 2 have no comment, so each was written as a mark with "
        "an empty note.")
    assert "remove=[its id]" in out["empty_comments"]


def test_a_cut_skipped_list_is_said_with_the_empty_comments():
    """planlens cuts its skipped list to fit; a mark it left off that list
    could be named as written, so the line says the list was cut."""
    marks = [{"kind": "box", "page": 0}, {"kind": "box", "page": 0}]
    full = document_tools.empty_comments_note(
        marks, {"skipped": [{"index": 1}]})
    assert full.startswith("markup 0 has no comment")
    assert "cut short" not in full
    cut = document_tools.empty_comments_note(
        marks, {"skipped": [], "skipped_truncated_after": 0})
    assert cut.startswith("markups 0, 1 have no comment")
    assert "The skipped list was cut short" in cut


def test_adjusted_rows_merge_and_sort_with_any_index():
    result = {"adjusted": [{"index": 2, "notes": ["planlens"]},
                           {"index": "x", "notes": ["odd"]}]}
    document_tools._merge_adjusted(result, [
        {"index": 2, "notes": ["app"]}, {"index": 0, "notes": ["app"]},
        {"index": None, "notes": ["none"]}])
    assert [r["index"] for r in result["adjusted"]] == [0, 2, None, "x"]
    assert result["adjusted"][1]["notes"] == ["planlens", "app"]


def test_one_mark_with_no_comment_and_none(gt, folder):
    tools, handle = _setup(gt)
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    append=False, markups=[
                        _box(gt, 300, "said"),
                        {"kind": "box", "page": gt.narrative_page,
                         "bbox": [72, 360, 300, 390]}])
    assert out["empty_comments"].startswith(
        "markup 1 has no comment, so it was written")
    out = _annotate(tools, handle=handle, output_path="set_marked.pdf",
                    append=False, markups=[_box(gt, 300, "said")])
    assert "empty_comments" not in out


# ---------------------------------------------------------------------------
# 4. search_document's min_score without fuzzy
# ---------------------------------------------------------------------------

def _search(gt, monkeypatch=None):
    tools, handle = _setup(gt)
    sent = []
    if monkeypatch is not None:
        real = document_tools.dispatch_document_tool

        def spy(name, args, **kw):
            sent.append(dict(args))
            return real(name, args, **kw)

        monkeypatch.setattr(document_tools, "dispatch_document_tool", spy)
    return tools["search_document"], handle, sent


def test_min_score_without_fuzzy_is_said(gt, folder):
    search, handle, _sent = _search(gt)
    out = json.loads(search.invoke({"handle": handle, "pattern": "EMBEDMENT",
                                    "min_score": 70}))
    assert out["n_hits"] >= 1
    assert out["min_score_unused"] == MIN_SCORE_UNUSED_NOTE
    out = json.loads(search.invoke({"handle": handle,
                                    "pattern": "EMBEDMENT"}))
    assert "min_score_unused" not in out


def test_min_score_with_fuzzy_is_handed_on(gt, folder, monkeypatch):
    if not document_tools.search_supports_fuzzy():
        pytest.skip("installed planlens predates fuzzy search")
    pytest.importorskip("rapidfuzz")
    search, handle, sent = _search(gt, monkeypatch)
    out = json.loads(search.invoke({"handle": handle, "pattern": "EMBEDMANT",
                                    "fuzzy": True, "min_score": 75}))
    assert "min_score_unused" not in out and out["n_hits"] >= 1
    assert sent[-1]["min_score"] == 75
    # left out: planlens' own default (80), as before
    search.invoke({"handle": handle, "pattern": "EMBEDMANT", "fuzzy": True})
    assert "min_score" not in sent[-1]


def test_the_model_is_still_shown_min_score_80(gt):
    tools, _handle = _setup(gt)
    shown = _shown(tools["search_document"])["properties"]["min_score"]
    assert shown == {"type": "integer", "default": 80}
