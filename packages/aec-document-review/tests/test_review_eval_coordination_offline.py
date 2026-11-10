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
    """The labels of the task's scored checks ``answer`` passes."""
    out = []
    for check in BY_ID[task_id].checks:
        if check.get("info"):
            continue
        if C.run_check(check, answer)["passed"]:
            out.append(check["label"])
    return out


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
