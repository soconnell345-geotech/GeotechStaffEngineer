"""The coordination tasks (BACKLOG DR-25) - offline, no model, no network.

These tasks are only worth running if the documents hold exactly the
problems the checks look for, each problem is on the page where the truth
says it is, and a check tells an answer that found a problem from one that
only mentioned the same sheets and numbers. All made-up values.
"""

import pytest

pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core.review_eval import checks as C  # noqa: E402
from aec_agent_core.review_eval import documents as D  # noqa: E402
from aec_document_review.review_eval import (  # noqa: E402
    coordination_fixture as F, score_review_suite)
from aec_document_review.review_eval.tasks import (  # noqa: E402
    COORDINATION_TASKS, OPEN_TASKS, _PLANTED_CHECKS)

BY_ID = {t.id: t for t in COORDINATION_TASKS}


def _pages(pdf: bytes):
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def _passes(task_id: str, answer: str):
    """The labels of the task's planted-problem checks ``answer`` passes
    (the false-alarm check, which an answer that names nothing passes, is
    tested on its own: :func:`_alarm`)."""
    out = []
    for check in BY_ID[task_id].checks:
        if check.get("info") or check["type"] == "no_false_alarm":
            continue
        if C.run_check(check, answer)["passed"]:
            out.append(check["label"])
    return out


def _alarm(task_id: str, answer: str):
    """The task's false-alarm check on ``answer``."""
    (check,) = [c for c in BY_ID[task_id].checks
                if c["type"] == "no_false_alarm"]
    return C.run_check(check, answer)


# ---------------------------------------------------------------------------
# The documents hold the planted problems and no others of those kinds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("floors, issue, sheets", [(2, 1, 15), (2, 0, 15),
                                                   (5, 1, 40)])
def test_the_rules_find_exactly_what_is_planted(floors, issue, sheets):
    built = F.build_drawing_set(floors, issue)
    assert len(built.sheets) == sheets
    assert F.audit(F._project(floors, issue)) == sorted(
        p.key for p in built.planted)


def test_what_is_planted_in_each_set():
    assert len(F.BASE_PLANTED) == 13
    assert len(F.LONG_PLANTED) == 19
    assert F.LONG_PLANTED[:13] == F.BASE_PLANTED
    # issue 0 is the set before its three changes: two of the problems are
    # issue 1's own (the enlarged plans' bay, the bearing pressure)
    before = {p.key for p in F.build_drawing_set(2, 0).planted}
    assert {p.key for p in F.BASE_PLANTED} - before == {
        "bay-b-c", "bearing-pressure"}


def test_every_sheet_is_a_page_with_its_number_in_the_title_block():
    for floors in (2, 5):
        built = F.build_drawing_set(floors, 1)
        texts = _pages(built.pdf)
        assert len(texts) == len(built.sheets)
        for n, (number, title) in enumerate(built.sheets, 1):
            assert built.page_of(number) == n
            lines = texts[n - 1].split("\n")
            assert number in lines, number
            assert f"SHEET {n} OF {len(built.sheets)}" in lines


def test_each_planted_problem_is_on_the_page_the_truth_names():
    built = F.build_drawing_set(2, 1)
    text = dict(zip((n for n, _t in built.sheets), _pages(built.pdf)))
    lines = {n: t.split("\n") for n, t in text.items()}
    # the index lists a sheet the set does not hold, and misnames another
    assert "A-202" in lines["G-001"] and "A-202" not in dict(built.sheets)
    assert lines["G-001"].count("ROOF FRAMING PLAN") == 2
    assert "SECOND FLOOR FRAMING PLAN" in lines["S-201"]
    # doors: 109 drawn and not scheduled, 204 scheduled and not drawn
    assert "109" in lines["A-101"] and "109" not in lines["A-601"]
    assert "204" in lines["A-601"] and "204" not in lines["A-102"]
    # levels and bays
    assert "EL. 104.20" in lines["A-201"] and "EL. 104.50" in lines["A-301"]
    assert "EL. 104.50" not in lines["A-201"]
    assert "7200" in lines["A-101"] and "7500" in lines["A-401"]
    assert "7500" not in lines["A-101"] and "7200" not in lines["A-401"]
    # notes against the manual and the memorandum
    assert "SEE SECTION 07 84 00." in text["G-002"]
    manual = "\n".join(_pages(F.build_project_manual()))
    assert "07 84 00" not in manual and "07 92 00" in manual
    assert "PRESSURE: 200 kPa." in text["S-001"]
    assert "FOOTINGS 25 MPa" in text["S-001"]
    memo = "\n".join(_pages(F.build_geotechnical_memo()))
    assert "150 kPa" in memo and "200 kPa" not in memo
    assert "at least 1.5 m" in memo and "1.2 m BELOW" in text["S-001"]
    assert "30 MPa" in manual
    # footings
    assert "F4" in lines["S-101"] and "F4" not in lines["S-501"]
    for mark in ("F1", "F2", "F3"):
        assert mark in lines["S-101"] and mark in lines["S-501"]


def test_a_callout_is_two_pieces_of_text_one_over_the_other():
    """The view number sits over the sheet number in a bubble: no text
    search finds "7/A-501", and the two are not neighbours in the page's
    text either (it is written in no helpful order, as a CAD plot's is)."""
    built = F.build_drawing_set(2, 1)
    doc = pymupdf.open(stream=built.pdf, filetype="pdf")
    try:
        page = doc[built.page_of("A-101") - 1]
        assert not page.search_for("7/A-501")
        assert "7\nA-501" not in page.get_text()
        words = page.get_text("words")
    finally:
        doc.close()
    pairs = set()
    for x0, y0, x1, y1, word, *_ in words:
        if word != "A-501":
            continue
        over = [w[4] for w in words
                if 0 < y0 - w[1] < 22 and abs((w[0] + w[2]) / 2
                                             - (x0 + x1) / 2) < 8]
        pairs.update(over)
    assert pairs == {"3", "7"}
    # sheet A-501 itself holds views 1 to 6
    views = F._project(2, 1).sheet("A-501").views
    assert views == ["1", "2", "3", "4", "5", "6"]


def test_issue_1_differs_from_issue_0_in_the_three_changes_and_the_cover():
    old, new = F.build_drawing_set(2, 0), F.build_drawing_set(2, 1)

    # every title block carries the issue line; set it aside and compare
    def body(text):
        return sorted(line for line in text.split("\n")
                      if not line.startswith("ISSUE "))
    changed = [n for (n, _t), a, b in zip(new.sheets, _pages(old.pdf),
                                          _pages(new.pdf))
               if body(a) != body(b)]
    assert changed == ["G-001", "A-401", "A-601", "S-001"]
    assert [sheet for _k, sheet, _w, _f in F.REVISIONS] == changed[1:]
    text = dict(zip((n for n, _t in new.sheets), _pages(new.pdf)))
    # the two flagged changes are in their revision blocks; the third is not
    assert "REVISED CORE LAYOUT" in text["A-401"]
    assert "REVISED DOOR HARDWARE" in text["A-601"]
    assert "REVISED" not in text["S-001"]
    assert "PRESSURE: 150 kPa." in dict(
        zip((n for n, _t in old.sheets), _pages(old.pdf)))["S-001"]


def test_the_submittal_holds_its_four_problems():
    text = " ".join("\n".join(_pages(F.build_concrete_submittal())).split())
    manual = " ".join("\n".join(_pages(F.build_project_manual())).split())
    assert "0.48" in text and "0.45" in manual
    assert "30 %" in text and "25 percent" in manual
    assert "alkali" not in text.lower() and "alkali-silica" in manual
    assert "Number of tests: 12." in text and "30 consecutive" in manual
    assert "DEVIATIONS: NONE." in text
    # 133 of 443 kg is 30 %; 213 of 443 is 0.48
    assert round(133 / 443, 2) == 0.30 and round(213 / 443, 2) == 0.48


def test_the_documents_resolve_by_their_suite_ids():
    names = {}
    for doc_id, pages in (("fixture_coord_set", 15),
                          ("fixture_coord_set_prev", 15),
                          ("fixture_coord_set_scan", 15),
                          ("fixture_coord_set_long", 40),
                          ("fixture_coord_manual", 5),
                          ("fixture_coord_memo", 4),
                          ("fixture_coord_submittal", 5)):
        name, data = D.resolve(doc_id)
        assert data[:5] == b"%PDF-" and len(_pages(data)) == pages
        names[doc_id] = name
    assert len(set(names.values())) == 7
    # an upload's name says what it is and nothing of what is wrong with it
    for name in names.values():
        for word in ("planted", "fixture", "coord", "synthetic", "error"):
            assert word not in name.lower()
    assert F.fixture("no_such_fixture") is None


def test_a_build_is_the_same_bytes_every_time():
    F.build_drawing_set.cache_clear()
    first = F.build_drawing_set(2, 1).pdf
    F.build_drawing_set.cache_clear()
    assert F.build_drawing_set(2, 1).pdf == first


# ---------------------------------------------------------------------------
# The tasks and their checks
# ---------------------------------------------------------------------------

def test_the_tasks_cover_every_planted_problem():
    assert len(COORDINATION_TASKS) == 14
    assert all(t in OPEN_TASKS for t in COORDINATION_TASKS)
    labels = {t.id: [c["label"] for c in t.checks if not c.get("info")]
              for t in COORDINATION_TASKS}
    assert labels["coord-full-review"] == [p.key for p in F.BASE_PLANTED]
    assert labels["coord-long-review"] == [p.key for p in F.LONG_PLANTED]
    assert labels["coord-submittal"] == [p.key for p in F.SUBMITTAL_PLANTED]
    assert set(_PLANTED_CHECKS) == {
        p.key for p in F.LONG_PLANTED + F.SUBMITTAL_PLANTED}
    # a comment log may come as a file: the review tasks read those too
    for task_id in ("coord-full-review", "coord-long-review"):
        assert all(c.get("with_files") for c in BY_ID[task_id].checks)
    # the callout and schedule questions also score false alarms (EV-44)
    alarms = sorted(t.id for t in COORDINATION_TASKS for c in t.checks
                    if c["type"] == "no_false_alarm")
    assert alarms == sorted(
        f"coord-{w}{k}" for w in ("", "scan-", "long-")
        for k in ("callouts", "schedules"))
    for task_id in alarms:
        assert labels[task_id][-1] == "no-false-alarm"


@pytest.mark.parametrize("planted", F.LONG_PLANTED + F.SUBMITTAL_PLANTED,
                         ids=lambda p: p.key)
def test_each_check_takes_its_own_truth_and_no_other(planted):
    """A check passes on its problem's sentence and on no other problem's:
    a log that holds twelve of the thirteen scores twelve."""
    check = _PLANTED_CHECKS[planted.key]
    assert C.run_check(check, planted.truth)["passed"]
    others = " ".join(p.truth for p in F.LONG_PLANTED + F.SUBMITTAL_PLANTED
                      if p.key != planted.key)
    assert not C.run_check(check, others)["passed"], planted.key


def test_an_answer_that_finds_nothing_scores_nothing():
    nothing = ("I checked the sheet index, the callouts, the schedules, the "
               "levels and dimensions, the notes, the project manual and "
               "the geotechnical memorandum. Everything is coordinated and "
               "I found no discrepancies. Sheets reviewed: G-001, G-002, "
               "C-101, A-101, A-102, A-201, A-301, A-401, A-501, A-601, "
               "S-001, S-101, S-201, S-202 and S-501.")
    for task in COORDINATION_TASKS:
        assert _passes(task.id, nothing) == [], task.id


def test_listing_what_is_there_is_not_finding_what_is_missing():
    # every door named, the gap between plan and schedule never stated
    listed = ("Doors on A-101: 101, 102, 103, 104, 105, 106, 107, 108 and "
              "109. Doors on A-102: 201, 202, 203, 205 and 206. The door "
              "schedule on A-601 lists 101 to 108 and 201, 202, 203, 204, "
              "205 and 206. Footing marks on S-101: F1, F2, F3 and F4. The "
              "footing schedule lists F1, F2 and F3.")
    assert _passes("coord-schedules", listed) == []
    found = ("Door 109 is drawn on A-101 with no row in the schedule. "
             "Door 204 has a schedule row but no tag on A-102. "
             "F4 on S-101 is missing from the footing schedule.")
    assert _passes("coord-schedules", found) == [
        "door-not-scheduled", "door-not-on-plan", "footing-not-scheduled"]
    # neither raises a false alarm
    for answer in (listed, found):
        assert _alarm("coord-schedules", answer)["passed"], answer


def test_both_sides_of_a_disagreement_are_wanted_side_by_side():
    # the two levels in one sentence
    assert _passes("coord-levels-dimensions",
                   "Second floor: A-201 shows 104.20, A-301 shows 104.50."
                   ) == ["level-second-floor"]
    # one side only, or the two far apart, is not the finding
    assert _passes("coord-levels-dimensions",
                   "The second floor is at EL. 104.20 throughout.") == []
    far = ("A-201 gives the second floor as 104.20. " + "The grids were "
           "checked on every plan and agree with the foundation plan. " * 8
           + "A-301 has levels of 100.00, 104.50, 108.40 and 109.30.")
    assert _passes("coord-levels-dimensions", far) == []
    # metres for millimetres is the same finding
    assert _passes("coord-levels-dimensions",
                   "Bay B-C is 7.2 m on A-101 but 7.5 m on A-401."
                   ) == ["bay-b-c"]


def test_a_section_number_is_not_a_strength():
    # "03 30 00" holds a 30; the finding needs 30 MPa said of footings
    half = ("S-001 gives footings 25 MPa, which conflicts with Section "
            "03 30 00.")
    assert "footing-concrete" not in _passes("coord-drawings-vs-manual",
                                             half)
    whole = ("S-001 note 6: footings 25 MPa. Section 03 30 00 Class A "
             "(footings): 30 MPa. Conflict.")
    assert "footing-concrete" in _passes("coord-drawings-vs-manual", whole)
    assert "footing-concrete" in _passes(
        "coord-drawings-vs-manual",
        "Footing concrete is 25 instead of 30 MPa (Section 03 30 00).")


def test_a_callout_is_recognised_however_it_is_written():
    for text in ("7/A-501 does not exist", "detail 7 on A-501 is not there",
                 "Detail 7 (sheet A-501) is not drawn",
                 "A-501 has no detail 7", "bubble 7 over A-501 leads nowhere",
                 "callout 7 / A501 cannot be found",
                 "7/A-501: sheet A-501 has details 1 to 6 only"):
        assert "callout-missing-detail" in _passes("coord-callouts", text), \
            text
    for text in ("3/A-501 and 5/A-501 do not resolve", "17/A-501 is missing",
                 "7/A-5010 is missing", "details 1 to 6 only are on A-501",
                 "7/A-501 resolves to its detail",
                 "A-501 at 1:5, not 7 sheets on"):
        assert "callout-missing-detail" not in _passes("coord-callouts",
                                                       text), text


def test_the_silent_change_wants_its_silence_said():
    told = ("S-001: allowable bearing pressure raised from 150 kPa to 200 "
            "kPa. Not clouded, and the revision block has no entry.")
    assert "bearing-pressure, and that it is not flagged" in _passes(
        "coord-revisions", told)
    only_change = "S-001: bearing pressure is now 200 kPa (was 150 kPa)."
    assert _passes("coord-revisions", only_change) == []


def test_a_comment_log_delivered_as_excel_is_read(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["No.", "Sheet", "Comment"])
    sheet.append([1, "A-101", "Door 109 is not in the door schedule."])
    sheet.append([2, "A-301", "Second floor EL. 104.50; A-201 has 104.20."])
    path = tmp_path / "comment_log.xlsx"
    book.save(path)
    checks = {c["label"]: c for c in BY_ID["coord-full-review"].checks}
    reply = "The comment log is attached as comment_log.xlsx."
    for label in ("door-not-scheduled", "level-second-floor"):
        got = C.run_check(checks[label], reply, files=[str(path)])
        assert got["passed"], got["detail"]
        assert "comment_log.xlsx" in got["detail"]
        assert not C.run_check(checks[label], reply)["passed"]
    assert not C.run_check(checks["bay-b-c"], reply,
                           files=[str(path)])["passed"]


def test_the_dry_run_resolves_the_coordination_documents(tmp_path):
    out = score_review_suite(out_dir=str(tmp_path), dry_run=True,
                             ids=["coord-"])
    text = out["results_md"]
    assert "14 tasks" in text
    assert ("fixture_coord_set_long -> "
            "north_yard_annex_full_set_issue_1.pdf") in text
    assert "MISSING" not in text.upper().replace("CALLOUT-MISSING", "")


def test_an_all_clear_that_lists_everything_is_not_a_finding():
    clear = ("Doors 101 to 109 are tagged on A-101 and I found no "
             "discrepancies with the schedule; nothing is missing. Door 204 "
             "is scheduled, no issues. Footings F1 to F4: no problems. The "
             "index lists A-202 and S-201 ROOF FRAMING PLAN, and I did not "
             "find any mismatch with the sheets.")
    assert _passes("coord-schedules", clear) == []
    assert _passes("coord-sheet-index", clear) == []
    assert _alarm("coord-schedules", clear)["passed"]


def test_not_found_on_a_plan_is_a_finding_and_not_finding_any_is_not():
    # Foundry brief 8: a delivered "was not found on the plan" failed
    found = ("Door 509 is scheduled on A-602 but was not found on the "
             "fifth-floor plan.")
    assert "door-not-on-plan-5" in _passes("coord-long-schedules", found)
    for clear in ("Door 509 is on the fifth-floor plan; I did not find any "
                  "problem with it.",
                  "Door 509: I did not see anything wrong.",
                  "Door 509 is tagged; I did not find a conflict."):
        assert "door-not-on-plan-5" not in _passes("coord-long-schedules",
                                                   clear), clear
        assert _alarm("coord-long-schedules", clear)["passed"], clear


def test_a_finding_said_in_a_heading_over_a_long_table_counts():
    rows = "".join(
        f"| {mark} | {sheet} | Tagged at grid {g}; see the plan. |\n"
        for mark, sheet, g in (
            ("Door 109", "A-101", "B/2-3"), ("Door 407", "A-104", "C-D/2"),
            ("Footing F4", "S-101", "C/2"), ("FCU-3-4", "M-103", "A-B/2-3"),
            ("Panel LP-4B", "E-104", "B-C/2-3")))
    answer = ("## On a plan but not scheduled\n\n"
              "| Mark | Sheet | Where |\n|---|---|---|\n" + rows +
              "\n## Scheduled but not on a plan\n\n"
              "- Door 204 (A-601)\n- Door 509 (A-602)\n")
    assert _passes("coord-long-schedules", answer) == [
        "door-not-scheduled", "door-not-on-plan", "door-not-scheduled-4",
        "door-not-on-plan-5", "footing-not-scheduled", "fcu-not-scheduled",
        "panel-not-scheduled"]
    # the same marks under a heading that says nothing is wrong
    fine = answer.replace("On a plan but not scheduled",
                          "Marks checked on the plans").replace(
        "Scheduled but not on a plan", "Schedule rows checked")
    assert _passes("coord-long-schedules", fine) == []


def test_the_scanned_set_is_images_with_no_text():
    data = F.build_scanned_set()
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert len(doc) == 15
        for page in doc:
            assert page.get_text().strip() == ""
            (image,) = page.get_images(full=True)
            # 34 x 22 in at 150 dpi: 6 pt lettering about 12 pixels tall
            assert (image[2], image[3]) == (5100, 3300)
    finally:
        doc.close()
    assert len(data) < 12_000_000
    # the two scanned tasks ask what their text-layer twins ask
    for scan, twin in (("coord-scan-callouts", "coord-callouts"),
                       ("coord-scan-schedules", "coord-schedules")):
        assert BY_ID[scan].question == BY_ID[twin].question
        assert BY_ID[scan].documents == ["fixture_coord_set_scan"]
        assert ([c["label"] for c in BY_ID[scan].checks if not c.get("info")]
                == [c["label"] for c in BY_ID[twin].checks
                    if not c.get("info")])


# ---------------------------------------------------------------------------
# Foundry brief 7's scorer defects (BACKLOG EV-44)
# ---------------------------------------------------------------------------

def test_a_door_number_is_not_the_front_of_a_decimal():
    # the fixture's parapet level starts with a door number
    level = ("The parapet is at EL. 109.30 on A-201; its height is not "
             "dimensioned.")
    assert "door-not-scheduled" not in _passes("coord-schedules", level)
    box = "A tag box at 204.9 mm from the sheet edge has no label."
    assert "door-not-on-plan" not in _passes("coord-schedules", box)
    for said in ("Door 109 is tagged on A-101 and is not scheduled.",
                 "Door 109. Not in the schedule."):
        assert "door-not-scheduled" in _passes("coord-schedules", said), said


def test_the_index_title_wants_both_titles_in_one_line_or_item():
    built = F.build_drawing_set(2, 1)
    listing = ("The sheet index matches the set.\n\nSheets in the set:\n"
               + "".join(f"- {n} - {t.title()}\n" for n, t in built.sheets))
    assert "index-title" not in _passes("coord-sheet-index", listing)
    for found in (
            "The index calls S-201 ROOF FRAMING PLAN; the sheet itself is "
            "titled SECOND FLOOR FRAMING PLAN.",
            "2. S-201 is misnamed in the index.\n"
            "   - Index: S-201 Roof Framing Plan\n"
            "   - Title block: S-201 Second Floor Framing Plan\n",
            "| S-201 | ROOF FRAMING PLAN | SECOND FLOOR FRAMING PLAN |",
            "**S-201 title:**\n- Index: Roof Framing Plan\n"
            "- Sheet: Second Floor Framing Plan\n"):
        assert "index-title" in _passes("coord-sheet-index", found), found


def test_a_good_callout_called_broken_is_a_false_alarm():
    wrong = ("Callouts that do not resolve:\n"
             "- 7/A-501 on A-101: A-501 has no detail 7.\n"
             "- 2/A-302 on A-102: there is no sheet A-302.\n"
             "- 3/A-501 on A-101: sheet A-501 is not in the set.\n")
    got = _alarm("coord-callouts", wrong)
    assert not got["passed"] and "3/A-501" in got["detail"]
    # the planted checks alone cannot tell
    assert _passes("coord-callouts", wrong) == [
        "callout-missing-detail", "callout-missing-sheet"]
    right = ("| Sheet | Callout | Why it does not resolve |\n|---|---|---|\n"
             "| A-101 | 7/A-501 | A-501 holds details 1 to 6 only; detail 7 "
             "is missing. |\n"
             "| A-102 | 2/A-302 | There is no sheet A-302 in the set. |\n\n"
             "Every other callout leads to its view: 1/A-301, 2/A-301, "
             "1/A-401, 2/A-401, 3/A-501 and 5/A-501 among them. Sheet A-501 "
             "holds details 1 to 6, and detail 7 is not on it.")
    for task_id in ("coord-callouts", "coord-scan-callouts"):
        got = _alarm(task_id, right)
        assert got["passed"], got["detail"]
    assert _passes("coord-callouts", right) == [
        "callout-missing-detail", "callout-missing-sheet"]
    # the 40-sheet set: its own good callouts, and its third broken one
    assert not _alarm("coord-long-callouts",
                      "4/S-501 on S-203 is broken: no sheet S-501 in this "
                      "set.")["passed"]
    ceiling = ("9/A-502 on A-114 does not resolve: A-502 holds details 1 "
               "to 8, and 1/A-502 and 8/A-502 lead to theirs.")
    got = _alarm("coord-long-callouts", right + "\n" + ceiling)
    assert got["passed"], got["detail"]


def test_a_mark_that_is_fine_called_missing_is_a_false_alarm():
    wrong = ("Door 109 is not in the schedule. Doors 201, 202, 203, 205 and "
             "206 are scheduled but not tagged on A-101.")
    got = _alarm("coord-schedules", wrong)
    assert not got["passed"]
    for door in ("201", "202", "203", "205", "206"):
        assert f"door {door}" in got["detail"]
    assert "door 109" not in got["detail"]
    # a door number that no sheet carries, said to be missing from a plan
    assert not _alarm("coord-long-schedules", "Door 409 is scheduled on "
                      "A-602 but not tagged on A-104.")["passed"]
    right = ("Door 109 is tagged on A-101 and A-401 but has no row in the "
             "schedule, which lists 101 to 108. Door 204 has a schedule row, "
             "yet A-102 tags only 201, 202, 203, 205 and 206. The S-501 "
             "schedule has rows for F1, F2 and F3 only; F4, drawn on S-101, "
             "is missing from it.\n"
             "Door 407 is not scheduled: the A-602 rows run 406 then 408. "
             "Door 509 has a row on A-602 but no tag on A-105, which carries "
             "501 to 508. LP-4A is scheduled on E-601, LP-4B is not. FCU-3-4 "
             "on M-103 is missing from M-601; FCU-3-1 to FCU-3-3 are there.")
    got = _alarm("coord-long-schedules", right)
    assert got["passed"], got["detail"]
    assert _passes("coord-long-schedules", right) == [
        "door-not-scheduled", "door-not-on-plan", "door-not-scheduled-4",
        "door-not-on-plan-5", "footing-not-scheduled", "fcu-not-scheduled",
        "panel-not-scheduled"]


def test_the_truth_texts_agree_with_the_fixture():
    long = F._project(5, 1)
    numbers = {s.number: s for s in long.sheets}
    called = [(v, t) for s in long.sheets for _k, v, t in s.callouts]
    good = [(v, t) for v, t in called
            if t in numbers and v in numbers[t].views]
    on = [s for s in long.sheets if s.callouts]
    assert (len(called), len(on), len(good)) == (39, 17, 36)
    assert (f"{len(called)} callouts on {len(on)} of them, {len(good)} of "
            f"which resolve") in BY_ID["coord-long-callouts"].truth
    # drawings against the manual: which cited sections the volume binds,
    # and which its contents list as issued separately
    manual = " ".join("\n".join(_pages(F.build_project_manual())).split())
    bound = [n for n, _t in F.MANUAL_SECTIONS if f"SECTION {n} -" in manual]
    listed = {n for n, _t in F.MANUAL_SECTIONS}
    separate = [n for n, _t in F.CITED_SECTIONS
                if n in listed and n not in bound]
    assert bound == ["01 33 00", "03 30 00", "08 71 00"]
    assert separate == ["04 20 00", "07 92 00", "08 11 13", "09 91 00"]
    truth = BY_ID["coord-drawings-vs-manual"].truth
    assert "01 33 00, 03 30 00 and 08 71 00 are bound" in truth
    assert ("04 20 00, 07 92 00, 08 11 13 and 09 91 00 are listed as issued "
            "separately") in truth
    assert "the other cited sections are in the manual" not in truth.lower()
