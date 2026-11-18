"""A reading helper's open items carry a text read cut at its size limit,
and drop a failed zoom the helper redid in another box (BACKLOG DR-50).

Since DR-46 the open items its tools flagged are added to a reading
helper's ``task`` result (``keep_work.KeepWork(readers=...)``). Two cases
were missed: a text read cut at its size limit was not carried, so the
main agent was not told that text was cut; and a zoom that failed and was
redone in another box stayed listed as unread. A helper with neither gets
exactly what it got before.

No model, no network; every value is made up.
"""

import json
import os
import types

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from aec_agent_core._fileio import working_dir_bound
from aec_agent_core.deep import keep_work

DOC = "made_up_set.pdf"


def _rec(steps, subagent="general-purpose"):
    """A helper record from ``steps``: (tool, args, result, status)."""
    rec = keep_work.HelperRecord("call_1", subagent, "read pages 1-8")
    for i, (tool, args, result, status) in enumerate(steps):
        cid = f"c{i}"
        rec.add_model(AIMessage(content="", tool_calls=[
            {"name": tool, "args": args, "id": cid}]))
        rec.add_result(tool, cid, result if isinstance(result, str)
                       else json.dumps(result), status)
    return rec


def _look(page, **fields):
    return ("analyze_pdf_page", {"attachment_key": DOC, "page": page},
            {"page": page, "pdf_page": page + 1, "analysis": "made up",
             **fields}, None)


def _zoom(page, bbox=None, prompt=None, result=None, status=None, **more):
    args = {"attachment_key": DOC, "page": page, **more}
    if bbox is not None:
        args["bbox"] = bbox
    if prompt is not None:
        args["prompt"] = prompt
    if result is None:
        result = {"page": page, "pdf_page": page + 1, "analysis": "made up"}
    return ("render_region", args, result, status)


BOTH = {"error": "pass bbox OR view + image_box, not both"}


# ---------------------------------------------------------------------------
# (1) a text read cut at its size limit
# ---------------------------------------------------------------------------

def _made_up_pdf(pages=4, lines=100):
    """A PDF of ``pages`` pages, each about 5,000 characters of made-up
    text."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page(width=612, height=792)
        for i in range(lines):
            page.insert_text(
                (24, 14 + i * 7.6),
                f"Made-up clause {p + 1}.{i:03d}: the quick grey fox notes "
                "nothing", fontsize=6)
    data = doc.tobytes()
    doc.close()
    return data


def test_the_real_text_read_cut_at_its_budget_is_carried_where_it_says():
    """The core's own ``read_pdf_text`` on a made-up PDF: its cut is listed
    at the page its note says to go on from, in the tool's words; reading
    on from there settles it."""
    from aec_agent_core import vision_tools
    files = {DOC: _made_up_pdf()}
    first = vision_tools._dispatch_read_pdf_text(
        {"source": DOC, "pdf_pages": "1-4"}, files)
    data = json.loads(first)
    assert data["truncated"] is True
    nxt = data["pdf_pages_returned"][-1]
    assert f"pdf_pages='{nxt}-" in data["truncated_note"]
    args = {"source": DOC, "pdf_pages": "1-4"}
    assert keep_work.open_items(_rec([("read_pdf_text", args, first,
                                       None)])) == [
        f"- {DOC} page {nxt}: a text read (read_pdf_text) was cut at its "
        "size limit, so the rest of the text it asked for was not read (the "
        "tool's note: " + data["truncated_note"].rstrip(".") + ")"]
    rest = vision_tools._dispatch_read_pdf_text(
        {"source": DOC, "pdf_pages": f"{nxt}-4"}, files)
    assert "truncated" not in json.loads(rest)
    assert keep_work.open_items(_rec([
        ("read_pdf_text", args, first, None),
        ("read_pdf_text", {"source": DOC, "pdf_pages": f"{nxt}-4"}, rest,
         None)])) == []


def _pdf_text(returned, cut_at=None, source=DOC):
    """A ``read_pdf_text`` result shaped as the tool shapes it."""
    out = {"source": source, "pages_returned": [p - 1 for p in returned],
           "pdf_pages_returned": list(returned),
           "pages": [{"page": p - 1, "pdf_page": p, "has_text_layer": True,
                      "text": "made up"} for p in returned]}
    if cut_at is not None:
        out["truncated"] = True
        out["truncated_note"] = (
            "Output hit the character budget. Request specific later pages "
            f"(e.g. pdf_pages='{cut_at}-{cut_at + 7}') to continue reading.")
    asked = f"{returned[0]}-{returned[-1] + 2}"
    return ("read_pdf_text", {"source": source, "pdf_pages": asked}, out,
            None)


def test_a_cut_text_read_stays_until_it_is_read_on_from_there():
    cut = _pdf_text([1, 2, 3], cut_at=3)
    items = keep_work.open_items(_rec([cut]))
    assert len(items) == 1 and items[0].startswith(
        f"- {DOC} page 3: a text read (read_pdf_text) was cut")
    assert "pdf_pages='3-10'" in items[0]
    # not settled: other pages read, a look at the page, another file, a
    # failed read
    assert keep_work.open_items(_rec([cut, _pdf_text([6, 7])])) == items
    assert keep_work.open_items(_rec([cut, _look(2)])) == items
    assert keep_work.open_items(_rec([
        cut, _pdf_text([3, 4], source="other.pdf")])) == items
    failed = ("read_pdf_text", {"source": DOC, "pdf_pages": "3-4"},
              {"error": "made-up failure"}, None)
    assert keep_work.open_items(_rec([cut, failed]))[1] == items[0]
    # settled: read on from page 3; cut again further on, only the new cut
    assert keep_work.open_items(_rec([cut, _pdf_text([3, 4, 5])])) == []
    again = keep_work.open_items(_rec([cut, _pdf_text([3, 4, 5],
                                                      cut_at=5)]))
    assert len(again) == 1 and again[0].startswith(f"- {DOC} page 5: ")


def test_a_cut_note_that_names_no_page_is_carried_and_stays():
    cut = _pdf_text([1, 2], cut_at=None)
    cut[2]["truncated"] = True
    cut[2]["truncated_note"] = "Output hit the character budget."
    items = keep_work.open_items(_rec([cut, _pdf_text([3, 4])]))
    assert items == [
        f"- {DOC}: a text read (read_pdf_text) was cut at its size limit, so "
        "the rest of the text it asked for was not read (the tool's note: "
        "Output hit the character budget)"]


def _read_document(args, returned=None, nxt=None, **more):
    """A planlens ``read_document`` result, as the review layer hands it on
    (pages from 0, with the viewer's ``pdf_`` copy; the cursor as given)."""
    out = {"handle": args["handle"], "text": "made up", **more}
    if returned is not None:
        out["pages_returned"] = returned
    if nxt is not None:
        out["next"] = nxt
        out["note"] = ("output reached the size limit; call read_document "
                       "again with the 'next' pages and start_line")
    return ("read_document", args, out, None)


def test_a_cut_document_read_carries_the_tool_s_own_cursor():
    cut = _read_document({"handle": "doc_1", "pages": "0-7"}, "0-4",
                         {"pages": "4-7", "start_line": 31})
    items = keep_work.open_items(_rec([cut]))
    assert items == [
        "- doc_1 page 5: a text read (read_document) was cut at its size "
        "limit, so the rest of the text it asked for was not read (the "
        "tool's cursor to go on from: next={\"pages\": \"4-7\", "
        "\"start_line\": 31})"]
    # settled by reading on from the cursor, or the page from its top
    for later, returned in (({"pages": "4-7", "start_line": 31}, "4-7"),
                            ({"pages": "4-5"}, "4-5"),
                            ({"pages": "3-5"}, "3-5")):
        assert keep_work.open_items(_rec([cut, _read_document(
            {"handle": "doc_1", **later}, returned)])) == [], later
    # not settled: later pages only, a start past the cut line, another
    # document
    assert keep_work.open_items(_rec([cut, _read_document(
        {"handle": "doc_1", "pages": "6-7"}, "6-7")])) == items
    assert keep_work.open_items(_rec([cut, _read_document(
        {"handle": "doc_1", "pages": "4-7", "start_line": 40},
        "4-7")])) == items
    assert keep_work.open_items(_rec([cut, _read_document(
        {"handle": "doc_2", "pages": "4-7", "start_line": 31},
        "4-7")])) == items


def test_a_cut_read_of_a_file_read_by_lines_is_carried():
    cut = _read_document({"handle": "notes.docx"}, nxt={"start_line": 121},
                         lines_returned="0-120")
    items = keep_work.open_items(_rec([cut]))
    assert len(items) == 1 and items[0].startswith(
        "- notes.docx: a text read (read_document) was cut")
    assert "next={\"start_line\": 121}" in items[0]
    assert keep_work.open_items(_rec([cut, _read_document(
        {"handle": "notes.docx", "start_line": 121},
        lines_returned="121-200")])) == []
    assert keep_work.open_items(_rec([cut, _read_document(
        {"handle": "notes.docx", "start_line": 150},
        lines_returned="150-200")])) == items


def test_a_part_of_a_file_read_by_offset_is_no_open_item():
    """``read_text_file`` returns what was asked for from an offset (a
    reader opens a kept file at its index's offsets): not a cut."""
    assert keep_work.open_items(_rec([
        ("read_text_file", {"path": "kept/reader_x.md", "offset": 0},
         {"path": "kept/reader_x.md", "showing": "chars 0-6,000 of 9,000",
          "truncated": True, "next_offset": 6000, "text": "made up"},
         None)])) == []


# ---------------------------------------------------------------------------
# (2) a failed zoom redone
# ---------------------------------------------------------------------------

ASK = "Read the made-up title block lettering."


def test_a_failed_zoom_redone_in_another_box_with_the_same_ask_is_settled():
    failed = _zoom(1, bbox=[0, 0, 9, 9], prompt=ASK, result=BOTH,
                   view=[0, 0, 612, 792], image_box=[0, 0, 99, 99])
    redo = _zoom(1, bbox=[10, 10, 90, 90], prompt=ASK)
    assert keep_work.open_items(_rec([failed, redo])) == []
    # the question as sent, on one line
    assert keep_work.open_items(_rec([failed, _zoom(
        1, bbox=[10, 10, 90, 90], prompt="  " + ASK.replace(" ", "\n  ")
    )])) == []
    # a zoom that failed by its tool's status, too
    errored = _zoom(1, bbox=[0, 0, 9, 9], prompt=ASK,
                    result="Error: made-up render failure", status="error")
    assert keep_work.open_items(_rec([errored, redo])) == []


def test_a_failed_zoom_stays_when_the_redo_is_not_clearly_it():
    failed = _zoom(1, bbox=[0, 0, 9, 9], prompt=ASK, result=BOTH)
    line = f"- {DOC} page 2 (a zoom): render_region failed: " \
           "pass bbox OR view + image_box, not both"
    for later in (
            _zoom(1, bbox=[10, 10, 90, 90], prompt="Read the notes."),
            _zoom(1, bbox=[10, 10, 90, 90]),          # no question
            _zoom(4, bbox=[10, 10, 90, 90], prompt=ASK),  # another page
            ("analyze_pdf_page", {"attachment_key": DOC, "page": 1,
                                  "prompt": ASK},
             {"page": 1, "pdf_page": 2, "analysis": "made up"}, None),
            _zoom(1, bbox=[10, 10, 90, 90], prompt=ASK, result=BOTH),
    ):
        assert keep_work.open_items(_rec([failed, later]))[0].startswith(
            line), later
    # a zoom that failed before the same ask came back is not settled by
    # an earlier success
    assert keep_work.open_items(_rec([
        _zoom(1, bbox=[10, 10, 90, 90], prompt=ASK), failed])) == [line]
    # a failed zoom that sent no question stays, redone without one too
    bare = _zoom(1, bbox=[0, 0, 9, 9], result=BOTH)
    assert keep_work.open_items(_rec([bare, _zoom(
        1, bbox=[10, 10, 90, 90])])) == [line]
    # the same box again still settles it (DR-46)
    assert keep_work.open_items(_rec([bare, _zoom(
        1, bbox=[0, 0, 9, 9])])) == []


def test_a_redo_settles_only_the_failure_not_its_own_flags():
    failed = _zoom(1, bbox=[0, 0, 9, 9], prompt=ASK, result=BOTH)
    redo = _zoom(1, bbox=[10, 10, 90, 90], prompt=ASK,
                 result={"page": 1, "pdf_page": 2, "analysis": "made up",
                         "edge_note": "lettering runs past the top edge"})
    assert keep_work.open_items(_rec([failed, redo])) == [
        f"- {DOC} page 2 (a zoom): lettering runs past the top edge"]


# ---------------------------------------------------------------------------
# A helper with no cut text and no failed look gets what it got before
# ---------------------------------------------------------------------------

ANSWER = "Findings: none.\nTo compare with other pages: none.\nOpen items: none."


def _run(keep, steps, answer=ANSWER, subagent="general-purpose"):
    request = types.SimpleNamespace(tool_call={
        "name": "task", "id": "task_1",
        "args": {"subagent_type": subagent, "description": "read pages 1-8"}})

    def helper(_request):
        rec = keep_work.current_helper()
        for i, (tool, args, result, status) in enumerate(steps):
            rec.add_model(AIMessage(content="", tool_calls=[
                {"name": tool, "args": args, "id": f"c{i}"}]))
            rec.add_result(tool, f"c{i}", result if isinstance(result, str)
                           else json.dumps(result), status)
        rec.add_model(AIMessage(content=answer))
        return ToolMessage(content=answer, name="task", tool_call_id="task_1")

    return keep.wrap_tool_call(request, helper).content


CLEAN = [
    _look(0), _zoom(0, bbox=[10, 10, 90, 90], prompt=ASK),
    _pdf_text([1, 2, 3]),
    _read_document({"handle": "doc_1", "pages": "0-3"}, "0-3"),
    _read_document({"handle": "notes.docx"}, lines_returned="0-40"),
    ("read_text_file", {"path": "notes.txt"},
     {"path": "notes.txt", "truncated": True, "next_offset": 6000,
      "text": "made up"}, None),
]


def test_a_clean_reader_s_result_is_exactly_as_before(tmp_path):
    """No cut text, no failed look: the answer and the kept-file note, word
    for word as DR-46 wrote them, and no open-items block."""
    assert keep_work.open_items(_rec(CLEAN)) == []
    with working_dir_bound(str(tmp_path)):
        out = _run(keep_work.KeepWork(readers={"general-purpose"}), CLEAN)
    name = out.split("the file '", 1)[1].split("'", 1)[0]
    assert name.startswith(f"kept/{keep_work.READER_STEM}_")
    chars = sum(len(r if isinstance(r, str) else json.dumps(r))
                for _t, _a, r, _s in CLEAN)
    assert out == (
        ANSWER + "\n\n"
        f"[This reading helper's whole record ({len(CLEAN)} tool results, "
        f"{chars:,} characters) is kept in the file '{name}' in the working "
        "folder. Read it with read_text_file -- its index gives the offset "
        "of each result -- to compare what it read with other pages, "
        "instead of reading the same pages again.]")
    assert keep_work.OPEN_ITEMS_HEADING not in out
    assert os.path.isfile(os.path.join(str(tmp_path), *name.split("/")))


def test_the_geotech_page_s_helpers_are_still_unchanged(tmp_path):
    """No reader named: a helper with a cut text read and a failed zoom
    comes back as it did, and nothing is kept."""
    steps = [_pdf_text([1, 2, 3], cut_at=3),
             _zoom(1, bbox=[0, 0, 9, 9], prompt=ASK, result=BOTH)]
    with working_dir_bound(str(tmp_path)):
        out = _run(keep_work.KeepWork(), steps)
    assert out == ANSWER
    assert not os.path.exists(os.path.join(str(tmp_path), keep_work.KEPT_DIR))


def test_a_reader_s_result_names_the_cut_and_the_kept_file_lists_it(
        tmp_path):
    with working_dir_bound(str(tmp_path)):
        out = _run(keep_work.KeepWork(readers={"general-purpose"}),
                   [_look(0), _pdf_text([1, 2, 3], cut_at=3),
                    _zoom(1, bbox=[0, 0, 9, 9], prompt=ASK, result=BOTH),
                    _zoom(1, bbox=[10, 10, 90, 90], prompt=ASK)])
    note = out.split(f"[{keep_work.OPEN_ITEMS_HEADING}", 1)[1]
    assert [ln for ln in note.splitlines() if ln.startswith("- ")] == [
        f"- {DOC} page 3: a text read (read_pdf_text) was cut at its size "
        "limit, so the rest of the text it asked for was not read (the "
        "tool's note: Output hit the character budget. Request specific "
        "later pages (e.g. pdf_pages='3-10') to continue reading)]"]
    name = out.split("the file '", 1)[1].split("'", 1)[0]
    with open(os.path.join(str(tmp_path), *name.split("/")),
              encoding="utf-8") as fh:
        assert f"- {DOC} page 3: a text read (read_pdf_text)" in fh.read()
