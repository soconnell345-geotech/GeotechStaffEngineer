"""A page's reading helpers keep their whole record and hand on the open
items their tools flagged (BACKLOG DR-46).

Trace review of a 40-sheet whole-set review (2026-11-13): a reading helper's
"could not check" notes and what it read for comparison with other sheets
died with it when its answer left them out. ``KeepWork(readers=...)`` keeps
every named reader's record in ``kept/``, names it in the ``task`` result,
and adds the open items its tools' own fields flagged. The geotech page
names no reader, so its helpers get what they got before.

No model, no network; every value is made up.
"""

import json
import os
import types

from langchain_core.messages import AIMessage, ToolMessage

from aec_agent_core._fileio import working_dir_bound
from aec_agent_core.deep import keep_work

DOC = "made_up_set.pdf"


def _rec(steps, subagent="general-purpose"):
    """A helper record from ``steps``: (tool, args, result, status)."""
    rec = keep_work.HelperRecord("call_1", subagent, "read pages 4-6")
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


CUT = ("tile r1c1: the vision model's answer was CUT OFF at its output "
       "limit, so the end of this reading is missing")


def test_open_items_come_from_the_tools_own_fields():
    rec = _rec([
        _look(3, tiles_not_read="tile r2c3 failed: made-up timeout; the "
                                "rest were read.",
              cut_off=CUT, legibility="! small lettering",
              found=[{"what": "QZ-7", "bbox": [1, 2, 3, 4],
                      "seen_in": ["r1c2"], "tiles_only": True},
                     {"what": "QZ-8", "bbox": [5, 6, 7, 8],
                      "seen_in": ["page", "r1c2"]}]),
        ("read_pdf_text", {"source": DOC, "pdf_pages": "5-6"},
         {"text": "", "scanned_pages": [4, 5]}, None),
        _look(4),                               # page 5 looked at after
        ("render_region", {"attachment_key": DOC, "page": 0,
                           "bbox": [0, 0, 9, 9]},
         "Error: made-up render failure", "error"),
    ])
    items = keep_work.open_items(rec)
    # one line per file and page, the most serious first
    assert len(items) == 3, items
    assert items[0].startswith(f"- {DOC} page 1 (a zoom): render_region "
                               "failed: Error: made-up render failure")
    page4 = items[1]
    assert page4.startswith(f"- {DOC} page 4: ")
    assert "part not read in close-up: tile r2c3 failed: made-up timeout" \
        in page4
    assert "cut off at its output limit (tile r1c1)" in page4
    assert page4.index("part not read") < page4.index("cut off")
    # things found only in the tiles and the legibility caution are left out
    assert "QZ-7" not in page4 and "QZ-8" not in page4
    assert "lettering" not in page4
    assert items[2] == (f"- {DOC} page 6: no text layer, so a text read saw "
                        "nothing there (only a look reads it)")


def test_a_routine_tiled_read_flags_nothing():
    """Brief 8's saved runs: 87-97 % of the readers' looks had tiles-only
    rows and about 30 % of zooms a legibility note; neither is an open
    item, so a reader that read every page whole adds nothing."""
    rows = [{"what": f"QZ-{i}", "bbox": [i, i, i + 1, i + 1],
             "seen_in": ["r1c1"], "tiles_only": True} for i in range(9)]
    assert keep_work.open_items(_rec([
        _look(p, found=rows, legibility="! small lettering",
              found_note="made up") for p in range(13)])) == []


def test_a_flag_read_again_is_settled():
    # the same look again without the flag
    assert keep_work.open_items(_rec([_look(2, cut_off=CUT), _look(2)])) == []
    # a zoom on that page does not settle a cut-off answer
    zoom = ("render_region", {"attachment_key": DOC, "page": 2,
                              "bbox": [0, 0, 9, 9]},
            {"page": 2, "pdf_page": 3, "analysis": "made up"}, None)
    items = keep_work.open_items(_rec([
        _look(2, cut_off=CUT, legibility="! small",
              found=[{"what": "QZ-1", "tiles_only": True}]), zoom]))
    assert len(items) == 1 and "cut off" in items[0]
    assert "QZ-1" not in items[0] and "reliably" not in items[0]
    # an error read again without one
    bad = ("analyze_pdf_page", {"attachment_key": DOC, "page": 2},
           {"error": "made-up failure"}, None)
    assert keep_work.open_items(_rec([bad, _look(2)])) == []
    assert keep_work.open_items(_rec([_look(2), bad])) == [
        f"- {DOC} page 3: analyze_pdf_page failed: made-up failure"]


def test_plain_text_results_flag_nothing():
    assert keep_work.open_items(_rec([
        ("read_text_file", {"path": "notes.txt"}, "cut_off: not a field",
         None)])) == []


# ---------------------------------------------------------------------------
# The task result
# ---------------------------------------------------------------------------

ANSWER = "Findings: none.\nTo compare with other pages: none.\nOpen items: none."


def _run(keep, steps, answer=ANSWER, subagent="general-purpose"):
    """One ``task`` call through ``keep`` whose helper does ``steps``."""
    request = types.SimpleNamespace(tool_call={
        "name": "task", "id": "task_1",
        "args": {"subagent_type": subagent, "description": "read pages 4-6"}})

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


def _named(folder, content):
    name = content.split("the file '", 1)[1].split("'", 1)[0]
    assert name.startswith(("kept/helper_", "kept/reader_"))
    assert name.endswith(".md")
    with open(os.path.join(folder, *name.split("/")), encoding="utf-8") as fh:
        return name, fh.read()


def test_a_reader_s_record_is_kept_named_and_its_open_items_carried(
        tmp_path):
    with working_dir_bound(str(tmp_path)), keep_work.turn_scope() as scope:
        out = _run(keep_work.KeepWork(readers={"general-purpose"}),
                   [_look(0), _look(1, cut_off=CUT)])
    assert out.startswith(ANSWER)
    name, text = _named(str(tmp_path), out)
    assert name.startswith(f"kept/{keep_work.READER_STEM}_")
    # a reader that finished is not counted as a helper that stopped
    assert scope.kept_files == []
    assert text.startswith("# What a reading helper read and wrote")
    assert keep_work.READER_KEPT_WHY in text
    assert f"## {keep_work.OPEN_ITEMS_HEADING}\n- {DOC} page 2: " in text
    note = out.split(keep_work.OPEN_ITEMS_HEADING, 1)[1]
    assert f"- {DOC} page 2: the vision answer was cut off" in note
    assert "page 1:" not in note


def test_past_the_cap_the_rest_are_counted_and_left_in_the_file(tmp_path):
    steps = [_look(p, cut_off=CUT) for p in range(40)]
    with working_dir_bound(str(tmp_path)):
        out = _run(keep_work.KeepWork(readers={"general-purpose"}), steps)
    name, text = _named(str(tmp_path), out)
    note = out.split(f"[{keep_work.OPEN_ITEMS_HEADING}", 1)[1]
    shown = [ln for ln in note.splitlines() if ln.startswith("- ")]
    assert 0 < len(shown) < 40
    assert sum(len(ln.split()) for ln in shown) \
        <= keep_work.OPEN_ITEMS_MAX_WORDS
    assert (f"{40 - len(shown)} more open item(s) are not shown here; every "
            f"one is in '{name}' under '{keep_work.OPEN_ITEMS_HEADING}'.") \
        in note
    assert sum(1 for ln in text.splitlines()
               if ln.startswith(f"- {DOC} page ")) == 40


def test_a_reader_that_gave_up_is_kept_as_before_plus_its_open_items(
        tmp_path):
    big = "x" * keep_work.KEEP_MIN_READ_CHARS
    with working_dir_bound(str(tmp_path)), keep_work.turn_scope() as scope:
        out = _run(keep_work.KeepWork(readers={"general-purpose"}),
                   [_look(0, analysis=big, cut_off=CUT)], answer="Sorry.")
    assert out.startswith("Sorry.\n\n[Kept: its answer (6 characters)")
    assert keep_work.OPEN_ITEMS_HEADING in out
    name, text = _named(str(tmp_path), out)
    assert name.startswith("kept/helper_")
    assert [os.path.basename(p) for p in scope.kept_files] == [
        name.split("/")[1]]
    assert text.startswith("# What a helper read and wrote before it stopped")


def test_the_geotech_page_s_helpers_are_unchanged(tmp_path):
    """No reader named (the geotech page's build): the same helper's result
    comes back as it did, and nothing is kept."""
    with working_dir_bound(str(tmp_path)):
        out = _run(keep_work.KeepWork(), [_look(0), _look(1, cut_off=CUT)])
    assert out == ANSWER
    assert not os.path.exists(os.path.join(str(tmp_path), keep_work.KEPT_DIR))


def test_a_reader_that_read_nothing_is_unchanged(tmp_path):
    with working_dir_bound(str(tmp_path)):
        out = _run(keep_work.KeepWork(readers={"general-purpose"}), [])
    assert out == ANSWER
    assert not os.path.exists(os.path.join(str(tmp_path), keep_work.KEPT_DIR))


def test_another_helper_of_a_page_with_a_reader_is_unchanged(tmp_path):
    with working_dir_bound(str(tmp_path)):
        out = _run(keep_work.KeepWork(readers={"general-purpose"}),
                   [_look(1, cut_off=CUT)], subagent="figure")
    assert out == ANSWER
