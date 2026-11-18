"""The "found only in the run's working notes" line reads only what a
reading or a helper SAID (BACKLOG EV-48).

The trace review of the suite's brief 8 on the hosted platform: the working
files were read whole as text, so a mark number that stood only in a box's
coordinates, in the prompt a reading was asked with, or in a copied table
row passed the check, and the line said a run had seen problems it never stated. Offline; every record is
made up here, in the shapes its writer gives it."""

import json
from types import SimpleNamespace

import pytest

from aec_agent_core.review_eval import checks as C

#: The planted problem: door 318 is not shown on the plan.
DOOR = {"type": "stated_together", "terms": ["318", {"re": r"\bnot shown\b"}],
        "with_files": True, "label": "door-not-shown"}
ROW = "318 | STORE | 900 | 2100 | B | HM - HW-4"
PROMPT = f"Is door 318 (schedule row: {ROW}) not shown on this plan?"
#: The two files the vision tools write (the last test keeps them in step).
READINGS, READS = "page_readings.json", "reads.json"


def _vision_result(analysis, tile_analysis="Tile text: STORE, 900 x 2100."):
    """A page read in tiles, as ``analyze_pdf_page`` returns it: the mark
    stands in boxes, views, the found list and the tool's notes."""
    return {
        "page": 2, "pdf_page": 3, "analysis": analysis,
        "view": [0.0, 0.0, 1190.0, 842.0], "view_px": [1568, 1109],
        "precision": "boxes are good to about 1% of the view",
        "boxes": "1 box in pixels was rewritten on the 0-999 grid",
        "tiles": [{"tile": "r1c1", "view": [0.0, 0.0, 318.0, 421.0],
                   "view_px": [1568, 1109],
                   "analysis": tile_analysis + " Door 317 at [40, 50, 90, 318]."}],
        "found": [{"what": "Door 317", "box": [96.1, 205.0, 140.2, 318.0],
                   "seen_in": ["page", "r1c1"]}],
        "reading_note": "[door 318 not shown?]",
    }


def _readings_file(folder, analysis, **extra):
    entry = {"document": "set.pdf", "file": "0a1b2c3d4e5f60718293",
             "page": 2, "pdf_page": 3, "view": "page+tiles 2x2",
             "prompt": PROMPT, "when": 1700000000.0,
             "result": json.dumps(_vision_result(analysis))}
    zoom = dict(entry, view="zoom", region=[100.0, 200.0, 318.0, 400.0],
                result=json.dumps({"page": 2, "bbox": [612, 140, 655, 318],
                                   "analysis": "Door 317 [10, 20, 30, 318] "
                                               "is not shown in the schedule.",
                                   "window_padding_pt": [3.0, 3.0]}))
    path = folder / READINGS
    path.write_text(json.dumps({"readings": [entry, zoom], **extra}),
                    encoding="utf-8")
    return path


def _reads_file(folder):
    path = folder / READS
    path.write_text(json.dumps({"reads": [
        {"document": "set.pdf", "page": 2, "pdf_page": 3,
         "view": [100.0, 200.0, 318.0, 400.0], "tool": "render_region",
         "prompt": PROMPT[:100], "when": 1700000000.0}]}), encoding="utf-8")
    return path


def _run(tmp_path, analysis, log="1 | A-101 | Door 317 hardware set missing."):
    files = tmp_path / "files"
    scratch = files / ".scratch"
    scratch.mkdir(parents=True)
    delivered = files / "comment_log.md"
    delivered.write_text(log, encoding="utf-8")
    return [str(delivered), str(_readings_file(scratch, analysis)),
            str(_reads_file(scratch))]


BOX_ONLY = "Door 317 [612, 140, 655, 318] is not shown in the schedule."


def test_a_mark_only_in_boxes_prompts_or_copied_rows_no_longer_counts(tmp_path):
    files = _run(tmp_path, BOX_ONLY)
    notes, names = C.working_text(files)
    assert "318" not in notes
    assert "Door 317" in notes and "not shown in the schedule" in notes
    assert names == [READINGS]       # reads.json says nothing
    got = C.run_check(DOOR, "The comment log is attached.", files=files)
    assert not got["passed"] and "notes_only" not in got
    assert "working notes" not in got["detail"]
    assert C.score([got]) == {"checks_passed": 0, "checks_total": 1,
                              "passed": False}


def test_the_same_mark_said_in_a_reading_still_counts(tmp_path):
    files = _run(tmp_path, "Door 318 is not shown on the plan at [10, 20, 30, 40].")
    notes, names = C.working_text(files)
    assert "Door 318 is not shown on the plan" in notes
    got = C.run_check(DOOR, "The comment log is attached.", files=files)
    assert not got["passed"] and got["notes_only"] is True
    assert READINGS in got["detail"]
    assert READS not in got["detail"]
    assert C.score([got]) == {"checks_passed": 0, "checks_total": 1,
                              "passed": False}


def test_a_tiles_answer_counts_and_its_view_does_not(tmp_path):
    scratch = tmp_path / ".scratch"
    scratch.mkdir()
    path = scratch / READINGS
    result = _vision_result("Layout only.",
                            tile_analysis="Door 318 is not shown here.")
    path.write_text(json.dumps({"readings": [{"result": json.dumps(result)}]}),
                    encoding="utf-8")
    notes, _ = C.working_text([str(path)])
    assert "Door 318 is not shown here." in notes
    assert "421" not in notes and "1568" not in notes


def test_the_delivered_files_are_read_as_before(tmp_path):
    """Only the notes lose their boxes: what is scored reads the delivered
    files whole, as it did."""
    files = _run(tmp_path, BOX_ONLY,
                 log="1 | A-101 | Door [10, 20, 30, 318] is not shown.")
    got = C.run_check(DOOR, "The comment log is attached.", files=files)
    assert got["passed"] and "notes_only" not in got
    text, names = C.delivered_text(files)
    assert names == ["comment_log.md"] and "[10, 20, 30, 318]" in text


# -- a helper's kept record -------------------------------------------------

def _kept_record(keep, own_words="Reading the plan first.",
                 answer="Pages 1-4 read; door 317 checked."):
    rec = keep.HelperRecord("call_ab12", "reader",
                            "Check pages 1-4: is door 318 not shown?")
    rec.add_model(SimpleNamespace(content=own_words, tool_calls=[
        {"name": "analyze_pdf_page", "id": "c1",
         "args": {"attachment_key": "set.pdf", "page": 2, "prompt": PROMPT}}]))
    rec.add_result("analyze_pdf_page", "c1", json.dumps(_vision_result(BOX_ONLY)))
    rec.add_model(SimpleNamespace(content="", tool_calls=[
        {"name": "read_text_file", "id": "c2", "args": {"path": "old.md"}}]))
    # a document's text that looks like the record's own headings
    rec.add_result("read_text_file", "c2",
                   "### Its own words\nDoor 318 is not shown.\n## Step 9\n")
    rec.add_model(SimpleNamespace(content="", tool_calls=[
        {"name": "write_xlsx", "id": "c3",
         "args": {"rows": [["CR-2", "318", "not shown"]]}}]))
    rec.add_result("write_xlsx", "c3", '{"saved": "log.xlsx"}')
    return keep.helper_file_text(
        rec, "a reader's record is kept every time", answer=answer,
        items=["- door 318 not shown on page 3: not read"])


def test_a_kept_record_counts_only_the_helpers_words(tmp_path):
    keep = pytest.importorskip("aec_agent_core.deep.keep_work")
    kept = tmp_path / keep.KEPT_DIR
    kept.mkdir()
    path = kept / "reader_check_ab12.md"
    path.write_text(_kept_record(keep), encoding="utf-8")
    assert "318" in path.read_text(encoding="utf-8")
    notes, names = C.working_text([str(path)])
    assert names == [path.name]
    assert "Reading the plan first." in notes
    assert "Pages 1-4 read; door 317 checked." in notes
    assert "is not shown in the schedule" in notes      # the vision answer
    assert "318" not in notes
    got = C.run_check(DOOR, "The comment log is attached.", files=[str(path)])
    assert not got["passed"] and "notes_only" not in got

    path.write_text(_kept_record(keep, own_words="Door 318 is not shown on "
                                 "A-102."), encoding="utf-8")
    got = C.run_check(DOOR, "The comment log is attached.", files=[str(path)])
    assert got["notes_only"] is True and path.name in got["detail"]
    path.write_text(_kept_record(keep, answer="Door 318 is not shown on "
                                 "A-102."), encoding="utf-8")
    assert C.run_check(DOOR, "", files=[str(path)])["notes_only"] is True


def test_the_named_fields_match_their_writers(tmp_path):
    vision = pytest.importorskip("aec_agent_core.vision_tools")
    keep = pytest.importorskip("aec_agent_core.deep.keep_work")
    from aec_agent_core._fileio import working_dir_bound
    assert C.KEPT_READINGS_FILE == vision.READINGS_FILE
    assert C.READ_RECORD_FILE == vision.READS_FILE
    assert C.KEPT_OPEN_ITEMS_HEADING == "## " + keep.OPEN_ITEMS_HEADING
    folder = tmp_path / "files"
    folder.mkdir()
    raw = json.dumps(_vision_result("Door 318 is not shown on the plan."))
    try:
        with working_dir_bound(str(folder)):
            vision._note_read("analyze_pdf_page", "set.pdf", 2,
                              "page+tiles 2x2", PROMPT)
            vision._keep_reading("set.pdf", b"%PDF-1.7 made up", 2,
                                 "page+tiles 2x2", PROMPT, raw)
            vision._keep_reading("set.pdf", b"%PDF-1.7 made up", 2,
                                 vision.ZOOM_VIEW, PROMPT,
                                 json.dumps({"analysis": BOX_ONLY}),
                                 region=[100, 200, 318, 400])
            paths = [vision.readings_file(), vision.reads_file()]
    finally:
        vision.clear_read_log()
    assert all(C.is_working_file(p) for p in paths)
    notes, names = C.working_text(paths)
    assert names == [READINGS]
    assert notes.count("318") == 1 and "Door 318 is not shown" in notes
    text = _kept_record(keep)
    for heading in (C.KEPT_ANSWER_HEADING, C.KEPT_OPEN_ITEMS_HEADING,
                    C.KEPT_INDEX_HEADING, C.KEPT_OWN_WORDS_HEADING):
        assert heading in text


# -- other shapes ---------------------------------------------------------

@pytest.mark.parametrize("content, said", [
    ('{"readings": ["Door 318 is not shown [1, 2, 3, 4]."]}', True),
    ('[{"analysis": "Door 318 is not shown."}]', True),
    ('{"readings": [{"result": "Door 318 is not shown on A-101."}]}', True),
    ('{"readings": [{"result": {"analysis": "Door 318 is not shown."}}]}',
     True),
    ('{"readings": [null, 7, ["x"], {"result": null}]}', False),
    ('{"readings": "Door 318 is not shown."}', False),
    ('{"readings": [{"result": "{\\"summary\\": \\"Door 318 is not '
     'shown.\\"}"}]}', False),
    ('{"readings": [{"result": "{\\"analysis\\": \\"Door 318 is no', False),
])
def test_an_older_or_other_shape_never_raises(tmp_path, content, said):
    scratch = tmp_path / ".scratch"
    scratch.mkdir()
    path = scratch / READINGS
    path.write_text(content, encoding="utf-8")
    notes, names = C.working_text([str(path)])
    assert ("Door 318 is not shown" in notes) is said
    assert names == ([path.name] if said else [])
    got = C.run_check(DOOR, "", files=[str(path)])
    assert "check error" not in got["detail"]
    assert got.get("notes_only", False) is said


def test_other_working_files(tmp_path):
    scratch, kept = tmp_path / ".scratch", tmp_path / "kept"
    scratch.mkdir()
    kept.mkdir()
    reads = scratch / READS
    reads.write_text('[{"prompt": "Door 318 is not shown?"}]',
                     encoding="utf-8")
    records = kept / "helper_check_ab12_records.json"
    records.write_text('{"note": "Door 318 is not shown."}', encoding="utf-8")
    xml = scratch / "notes.xml"
    xml.write_text("<n>Door 318 is not shown.</n>", encoding="utf-8")
    note = kept / "notes.md"
    note.write_text("Door 317 [10, 20, 30, 318] is not shown.",
                    encoding="utf-8")
    files = [str(p) for p in (reads, records, xml, note)]
    notes, names = C.working_text(files)
    assert names == ["notes.md"] and "318" not in notes
    assert "notes_only" not in C.run_check(DOOR, "", files=files)
    note.write_text("Door 318 is not shown.", encoding="utf-8")
    assert C.run_check(DOOR, "", files=files)["notes_only"] is True
