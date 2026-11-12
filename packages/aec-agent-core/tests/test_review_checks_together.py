"""Two things the review suite's checks learned for findings that relate two
places (BACKLOG DR-25): terms stated side by side, and a comment log
delivered as an Excel file. Offline; all made-up values."""

import pytest

from aec_agent_core.review_eval import checks as C


def _together(answer, terms, **kw):
    return C.run_check(dict({"type": "stated_together", "terms": terms},
                            **kw), answer)


def test_terms_in_one_place_pass_and_terms_apart_do_not():
    near = "Level 2 is 104.20 on the elevation and 104.50 on the section."
    got = _together(near, ["104.20", "104.50"])
    assert got["passed"] and "together within" in got["detail"]
    apart = ("Level 2 is 104.20 on the elevation. " + "Nothing else was "
             "found on the sheets in between. " * 12 +
             "The section shows 104.50.")
    got = _together(apart, ["104.20", "104.50"])
    assert not got["passed"] and "never within 300" in got["detail"]
    # the same answer passes a wider window
    assert _together(apart, ["104.20", "104.50"], window=2000)["passed"]


def test_a_missing_term_is_named():
    got = _together("Level 2 is 104.20 everywhere.", ["104.20", "104.50"])
    assert not got["passed"] and got["detail"] == "missing: ['104.50']"
    assert not _together("", ["104.20", "104.50"])["passed"]


def test_terms_may_be_alternatives_or_patterns():
    terms = [["second floor", "level 2"], {"re": r"\b104\.50?\b"}]
    assert _together("Level 2 reads 104.5 on the section.", terms)["passed"]
    assert _together("SECOND FLOOR EL. 104.50", terms)["passed"]
    assert not _together("Third floor reads 104.50.", terms)["passed"]
    # markdown, case and a hyphen between letters do not matter
    assert _together("**Roof-framing** plan: S-201",
                     ["roof framing", "s-201"])["passed"]


def test_the_order_of_the_terms_does_not_matter():
    for answer in ("109 is not scheduled", "not scheduled: door 109"):
        assert _together(answer, [{"re": r"\b109\b"},
                                  {"re": r"\bnot\b"}])["passed"]


def test_the_closest_stretch_is_the_one_measured():
    # a far first mention does not hide a near later one
    answer = ("Door 109 is on the plan. " + "Other doors are fine. " * 20 +
              "Door 109 is not in the schedule.")
    got = _together(answer, [{"re": r"\b109\b"}, {"re": r"\bnot\b"}],
                    window=40)
    assert got["passed"], got["detail"]


def test_a_workbook_a_run_delivered_is_read_with_the_answer(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Comments"
    sheet.append(["No.", "Sheet", "Comment"])
    sheet.append([1, "S-001", "Bearing 200 kPa; the report gives 150 kPa."])
    sheet.append([None, None, None])
    path = tmp_path / "log.xlsx"
    book.save(path)
    text, names = C.delivered_text([str(path)])
    assert names == ["log.xlsx"]
    assert text.split("\n") == [
        "Comments", "No. | Sheet | Comment",
        "1 | S-001 | Bearing 200 kPa; the report gives 150 kPa."]
    check = {"type": "stated_together", "terms": ["200 kpa", "150 kpa"],
             "with_files": True}
    got = C.run_check(check, "See the attached log.", files=[str(path)])
    assert got["passed"] and "log.xlsx" in got["detail"]
    # a file that is not a workbook is skipped, not an error
    bad = tmp_path / "broken.xlsx"
    bad.write_bytes(b"not a workbook")
    assert C.delivered_text([str(bad)]) == ("", [])


NOT = {"re": r"\bnot\b"}


def test_a_term_in_the_heading_counts_for_every_line_under_it():
    filler = "".join(f"- item {n}, as drawn on the plan\n" for n in range(30))
    for head in ("## On a plan but not scheduled", "**Not scheduled:**",
                 "- **Not scheduled**", "On a plan but not scheduled:"):
        answer = f"{head}\n\n{filler}- D-7\n"
        got = _together(answer, ["d-7", NOT], window=40)
        assert got["passed"], head
        assert "under the heading" in got["detail"]
    # a heading counts for its own list only
    other = ("## Not scheduled\n\n- D-1\n\n## On the plans\n\n"
             + filler + "- D-7\n")
    assert not _together(other, ["d-7", NOT], window=40)["passed"]


def test_a_table_header_row_heads_its_rows():
    rows = "".join(f"| D-{n} | A-10{n} | tagged near the stair |\n"
                   for n in range(1, 8))
    table = "| Mark | Sheet | Not in the schedule |\n|---|---|---|\n" + rows
    assert _together(table, ["d-7", NOT], window=40)["passed"]
    # the heading above a table and its header row both count
    both = "### Not scheduled\n\n| Mark | Sheet | Where |\n|---|---|---|\n" + rows
    assert _together(both, ["d-7", NOT], window=40)["passed"]
    # text after the table is no longer under its header row
    after = table + "\nOther marks seen: " + "x " * 40 + "D-9.\n"
    assert not _together(after, ["d-9", NOT], window=40)["passed"]


def test_the_terms_under_a_heading_must_still_be_together():
    filler = "".join(f"- item {n}\n" for n in range(30))
    answer = ("## Levels that differ\n\n- 104.20 on the elevation\n"
              + filler + "- 104.50 on the section\n")
    # the heading holds neither value: they are too far apart
    assert not _together(answer, ["104.20", "104.50"], window=60)["passed"]
    answer = "## Second floor, 104.20 elsewhere\n\n" + filler + "- A-301: 104.50\n"
    assert _together(answer, ["104.20", "104.50"], window=60)["passed"]


# -- what a run delivered, and what it only noted (Foundry brief 7, EV-41) --

def _run_folder(tmp_path, log_text, reading_text, helper_text=""):
    """A made-up run's working folder: a delivered log, the vision tools'
    kept page readings and a helper's kept work."""
    files = tmp_path / "files"
    (files / ".scratch").mkdir(parents=True)
    (files / "kept").mkdir()
    log = files / "comment_log.md"
    log.write_text(log_text, encoding="utf-8")
    readings = files / ".scratch" / "page_readings.json"
    readings.write_text('{"readings": ["' + reading_text + '"]}',
                        encoding="utf-8")
    out = [str(log), str(readings)]
    if helper_text:
        helper = files / "kept" / "helper_sheet_check.md"
        helper.write_text(helper_text, encoding="utf-8")
        out.append(str(helper))
    return out


LEVELS = {"type": "stated_together", "terms": ["212.40", "212.75"],
          "with_files": True, "label": "level-roof"}


def test_a_page_reading_is_not_what_the_run_delivered(tmp_path):
    files = _run_folder(
        tmp_path, "1 | Q-101 | Door 31 is not in the schedule.",
        "Q-301 shows the roof at 212.40 where Q-201 shows 212.75.")
    text, names = C.delivered_text(files)
    assert names == ["comment_log.md"] and "212.40" not in text
    notes, note_names = C.working_text(files)
    assert note_names == ["page_readings.json"] and "212.40" in notes
    got = C.run_check(LEVELS, "The comment log is attached.", files=files)
    assert not got["passed"] and got["notes_only"] is True
    assert "page_readings.json" in got["detail"]
    assert "comment_log.md" in got["detail"]
    assert C.score([got]) == {"checks_passed": 0, "checks_total": 1,
                              "passed": False}


def test_a_finding_in_the_delivered_log_passes_without_the_notes(tmp_path):
    files = _run_folder(
        tmp_path, "1 | Q-301 | Roof 212.40 here, 212.75 on Q-201.",
        "Q-301 shows the roof at 212.40 where Q-201 shows 212.75.")
    got = C.run_check(LEVELS, "The comment log is attached.", files=files)
    assert got["passed"] and "notes_only" not in got
    assert "page_readings.json" not in got["detail"]


def test_a_finding_nowhere_is_a_plain_miss(tmp_path):
    files = _run_folder(tmp_path, "1 | Q-101 | Door 31 is not scheduled.",
                        "Q-301 shows the roof at 212.40.")
    got = C.run_check(LEVELS, "The comment log is attached.", files=files)
    assert not got["passed"] and "notes_only" not in got
    assert "working notes" not in got["detail"]


def test_a_helpers_kept_work_is_not_a_produced_file(tmp_path):
    files = _run_folder(
        tmp_path, "1 | Q-101 | Door 31 is not scheduled.", "nothing",
        helper_text="Roof 212.40 on Q-301 against 212.75 on Q-201.")
    got = C.run_check(LEVELS, "The comment log is attached.", files=files)
    assert not got["passed"] and got["notes_only"] is True
    assert "helper_sheet_check.md" in got["detail"]
    made = C.run_check({"type": "file_produced", "ext": ".md",
                        "name_contains": "helper"}, "", files=files)
    assert not made["passed"]
    made = C.run_check({"type": "file_produced", "ext": ".md"}, "",
                       files=files)
    assert made["passed"] and "comment_log.md" in made["detail"]
    assert "helper_sheet_check.md" not in made["detail"].split("among")[0]


def test_a_working_file_is_told_by_its_folder_with_either_separator():
    assert C.is_working_file("runs/a/files/.scratch/reads.json")
    assert C.is_working_file("runs" + chr(92) + "a" + chr(92) + "files"
                             + chr(92) + "kept" + chr(92) + "helper.md")
    assert not C.is_working_file("runs/a/files/comment_log.docx")
    assert not C.is_working_file("kept/runs/a/files/comment_log.docx")
    assert not C.is_working_file("reads.json")


def test_the_working_folder_names_match_the_tools_that_write_them():
    vision = pytest.importorskip("aec_agent_core.vision_tools")
    keep = pytest.importorskip("aec_agent_core.deep.keep_work")
    assert C.WORKING_DIRS == (vision.SCRATCH_DIR, keep.KEPT_DIR)
