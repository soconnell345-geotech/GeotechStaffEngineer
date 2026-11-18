"""BACKLOG DR-25 (2026-11-18): the suite's specification section with a
submittal answering it, and a third issue of the drawing set - offline, no
model, no network.

The two tasks are only worth running if the documents hold exactly the
problems the checks look for, where the truth says they are, read back the
way the page reads them (its own document tools: ``open_document`` and
``read_document``, planlens through the core), and if a check tells an
answer that found a problem from one that only names the same sheets and
numbers. All made-up values.
"""

import json
import re
from collections import Counter

import pytest

pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core.document_tools import dispatch_document_tool  # noqa: E402
from aec_agent_core.review_eval import checks as C  # noqa: E402
from aec_agent_core.review_eval import documents as D  # noqa: E402
from aec_document_review.review_eval import (  # noqa: E402
    coordination_fixture as F, score_review_suite)
from aec_document_review.review_eval.tasks import (  # noqa: E402
    COORDINATION_TASKS, OPEN_TASKS, _PLANTED_CHECKS)

BY_ID = {t.id: t for t in COORDINATION_TASKS}
NEW = ("coord-spec-submittal", "coord-reissue")
ALL_PLANTED = (F.LONG_PLANTED + F.SUBMITTAL_PLANTED + F.SPEC_PLANTED +
               F.REISSUE_PLANTED)


def _pages(pdf: bytes):
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def _flat(text: str) -> str:
    return " ".join(str(text).split())


class _Reader:
    """A document read as the page reads it: opened with ``open_document``
    and read a page at a time with ``read_document``, following its cursor
    when one call does not hold the whole page."""

    def __init__(self, name: str, data: bytes):
        self.att = {name: data}
        got = json.loads(dispatch_document_tool(
            "open_document", {"source": name}, self.att))
        assert "error" not in got, got
        self.handle = got["handle"]

    def page(self, pdf_page: int) -> str:
        args = {"handle": self.handle, "pdf_pages": str(pdf_page)}
        out = []
        for _ in range(40):
            got = json.loads(dispatch_document_tool("read_document", args,
                                                    self.att))
            assert "error" not in got, got
            out.append(got["text"])
            cursor = got.get("next")
            if not cursor:
                return _flat("\n".join(out))
            args = {"handle": self.handle, "pages": cursor["pages"],
                    "start_line": cursor["start_line"]}
        raise AssertionError("read_document never reached the page's end")

    def whole(self, n_pages: int) -> str:
        return " ".join(self.page(p) for p in range(1, n_pages + 1))


def _drawn(pdf: bytes, page_no: int):
    """The paths drawn on a sheet left of its title block, as a multiset:
    a revision cloud adds to it, and nothing else in a reissue does."""
    edge = F.W - F.MARGIN - F.TB
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        return Counter(
            (tuple(round(v, 1) for v in d["rect"]), len(d["items"]))
            for d in doc[page_no - 1].get_drawings()
            if d["rect"].x1 < edge - 1)
    finally:
        doc.close()


def _token(n: str) -> str:
    """``n`` as a whole token, not part of a number, a date or an id."""
    return rf"(?<![\d.,/-]){re.escape(n)}(?![\d.,/-])"


# ---------------------------------------------------------------------------
# Issue 2 of the drawing set
# ---------------------------------------------------------------------------

def test_the_rules_find_exactly_what_is_planted_in_issue_2():
    built = F.build_drawing_set(2, 2)
    assert len(built.sheets) == 15
    assert built.planted == F.BASE_PLANTED + F.ISSUE_2_PLANTED
    assert F.audit(F._project(2, 2)) == sorted(p.key for p in built.planted)
    # issue 2 is drawn for the 15-sheet set only
    with pytest.raises(ValueError):
        F._project(5, 2)


def test_issue_2_differs_from_issue_1_on_five_sheets_only():
    one, two = F.build_drawing_set(2, 1), F.build_drawing_set(2, 2)
    assert one.sheets == two.sheets

    def body(text):
        # the title block's issue line differs on every sheet
        return sorted(line for line in text.split("\n")
                      if not line.startswith("ISSUE "))
    changed = [n for (n, _t), a, b in zip(two.sheets, _pages(one.pdf),
                                          _pages(two.pdf))
               if body(a) != body(b)]
    assert changed == ["G-001", "A-102", "A-201", "A-601", "S-501"]
    # what is drawn changes only where a cloud was added: on A-102 (an
    # entry and nothing else) and on every other sheet it is the same
    clouded = {"A-201", "A-601", "S-501"}
    for number, _title in two.sheets:
        before = _drawn(one.pdf, one.page_of(number))
        after = _drawn(two.pdf, two.page_of(number))
        if number in clouded:
            assert not before - after, number
            assert sum((after - before).values()) > 10, number
        else:
            assert before == after, number


def test_the_reissue_faults_read_back_with_the_page_tools():
    one, two = F.build_drawing_set(2, 1), F.build_drawing_set(2, 2)
    old = _Reader("drawings_issue_1.pdf", one.pdf)
    new = _Reader("drawings_issue_2.pdf", two.pdf)
    at = two.page_of

    # the parapet: raised and listed on A-201; A-301 left at EL. 109.30
    a201_old, a201_new = old.page(at("A-201")), new.page(at("A-201"))
    a301_new = new.page(at("A-301"))
    assert "EL. 109.30" in a201_old and "EL. 109.60" not in a201_old
    assert "EL. 109.60" in a201_new and "EL. 109.30" not in a201_new
    assert "PARAPET RAISED" in a201_new
    assert "EL. 109.30" in a301_new and "EL. 109.60" not in a301_new
    assert "PARAPET" not in a301_new.replace("T.O. PARAPET", "")

    # S-501: F3's reinforcing changed, and its revision block has no row
    # for issue 2 (the date is the title block's issue line alone)
    s501_old, s501_new = old.page(at("S-501")), new.page(at("S-501"))
    assert "10-20M EACH WAY" in s501_old and "12-20M" not in s501_old
    assert "12-20M EACH WAY" in s501_new and "10-20M" not in s501_new
    assert F.ISSUE_DATES[2] in s501_new and F.ISSUE_DATES[1] not in s501_new
    s501_layer = _flat(_pages(two.pdf)[at("S-501") - 1])
    assert s501_layer.count(F.ISSUE_DATES[2]) == 1     # the issue line only

    # A-102: a revision 2 entry; door 205 still tagged as in issue 1
    a102_old, a102_new = old.page(at("A-102")), new.page(at("A-102"))
    assert "DOOR 205 RELOCATED" in a102_new
    assert "DOOR 205 RELOCATED" not in a102_old
    for text in (a102_old, a102_new):
        assert re.search(_token("205"), text)

    # A-601 renumbered door 106 as 110 and listed it; A-101 still tags 106
    # (a drawing's text comes in drafting order, so marks are read as
    # tokens, not as rows)
    a601_old, a601_new = old.page(at("A-601")), new.page(at("A-601"))
    a101_new = new.page(at("A-101"))
    assert "DOOR 106 RENUMBERED 110" in a601_new
    rows_new = a601_new.replace("DOOR 106 RENUMBERED 110", "")
    assert re.search(_token("106"), a601_old)
    assert not re.search(_token("110"), a601_old)
    assert re.search(_token("110"), rows_new)
    assert not re.search(_token("106"), rows_new)
    assert re.search(_token("106"), a101_new)
    assert not re.search(_token("110"), a101_new)

    # the cover's index carries each sheet's latest revision
    cover = new.page(at("G-001"))
    assert F.ISSUE_DATES[2] in cover and "SHEET INDEX UPDATED" in cover


def test_issue_0_and_issue_1_keep_their_own_dates():
    for issue in (0, 1):
        built = F.build_drawing_set(2, issue)
        text = _flat("\n".join(_pages(built.pdf)))
        assert f"ISSUE {issue} - {F.ISSUE_DATES[issue]}" in text
        assert F.ISSUE_DATES[2] not in text


# ---------------------------------------------------------------------------
# Section 08 51 13 and the window submittal
# ---------------------------------------------------------------------------

def test_the_window_problems_read_back_with_the_page_tools():
    section_pdf = F.build_window_section()
    submittal_pdf = F.build_window_submittal()
    manual_pdf = F.build_project_manual()
    s = _Reader("section.pdf", section_pdf).whole(len(_pages(section_pdf)))
    t = _Reader("submittal.pdf", submittal_pdf).whole(
        len(_pages(submittal_pdf)))
    m = _Reader("manual.pdf", manual_pdf).whole(len(_pages(manual_pdf)))

    # the three-part form
    for part in ("PART 1 GENERAL", "PART 2 PRODUCTS", "PART 3 EXECUTION"):
        assert part in s
    assert s.index("PART 1") < s.index("PART 2") < s.index("PART 3")
    # the section is listed in the manual's contents and not bound in it
    assert "08 51 13" in m and "ALUMINUM WINDOWS" in m
    assert "SECTION 08 51 13" not in m

    # the whole-window U-value misses the limit; the centre of glass meets
    # it and is not the value the section governs
    assert "whole window (glass, glass edge and frame together): not more " \
           "than 1.60 W/(m2.K)" in s
    assert re.search(r"whole window\W+1\.90 W/\(m2\.K\)", t)
    assert re.search(r"centre of glass\W+1\.10 W/\(m2\.K\)", t)

    # the anchorage calculations are required and absent
    assert "SD-05 Design Data: anchorage calculations" in s
    assert "anchor" not in t.lower() and "design data" not in t.lower()

    # the schedule of values cites a section the manual does not have
    assert "08 80 00" in t
    assert "08 80 00" not in m and "08 80 00" not in s
    assert "07 92 00" in t and "07 92 00" in m

    # the water penetration test is required and not reported
    assert "structural, air leakage and water penetration tests of 2.1" in s
    assert "water penetration" not in t.lower()
    assert "Air leakage at 75 Pa" in t
    assert "Structural, 1.80 kPa inward" in t
    assert "Structural, 1.80 kPa outward" in t

    # the schedule of values' total is not the sum of its lines
    for _what, section, value in F.WINDOW_SOV:
        assert f"{value:,}" in t and section in t
    assert sum(v for _w, _s, v in F.WINDOW_SOV) == 174500
    assert f"{F.WINDOW_SOV_TOTAL:,}" == "178,500" and "178,500" in t
    assert "174,500" not in t

    # what complies, complies
    assert "2.2 mm" in t and "not less than 2.0 mm" in s
    assert "20 micrometres" in t and "not less than 18 micrometres" in s
    assert "0.20 L/s per m2" in t and "not more than 0.30 L/s per m2" in s
    assert "2025-09-22" in t and "2026-04-06" in t

    # no heading or line names what is wrong
    for text in (s, t):
        for word in ("MISSING", "ERROR", "DEFICIEN", "NONCOMPL",
                     "NOT COMPL", "DISCREPAN", "PLANTED", "INCORRECT"):
            assert word not in text.upper(), word


def test_the_new_documents_resolve_by_their_suite_ids():
    want = {"fixture_coord_set_2": ("north_yard_annex_drawings_issue_2.pdf",
                                    15),
            "fixture_coord_window_section": (
                "section_08_51_13_aluminum_windows.pdf", 2),
            "fixture_coord_window_submittal": (
                "submittal_08_51_13-01_aluminum_windows.pdf", 4)}
    for doc_id, (name, pages) in want.items():
        got, data = D.resolve(doc_id)
        assert got == name
        assert data[:5] == b"%PDF-" and len(_pages(data)) == pages, doc_id
        for word in ("planted", "fixture", "coord", "synthetic", "error",
                     "wrong", "missing", "fault", "reissue"):
            assert word not in got.lower()
    assert F.fixture("coord_set_2") == F.build_drawing_set(2, 2).pdf
    names = {D.DOCUMENTS[d]["name"] for d in D.DOCUMENTS
             if d.startswith("fixture_coord")}
    assert len(names) == 10


def test_the_builds_are_the_same_bytes_every_time():
    for build in (F.build_window_section, F.build_window_submittal):
        build.cache_clear()
        first = build()
        build.cache_clear()
        assert build() == first
    F.build_drawing_set.cache_clear()
    first = F.build_drawing_set(2, 2).pdf
    F.build_drawing_set.cache_clear()
    assert F.build_drawing_set(2, 2).pdf == first


# ---------------------------------------------------------------------------
# The tasks and their checks
# ---------------------------------------------------------------------------

def test_the_new_tasks_are_listed_like_the_others():
    assert [t.id for t in COORDINATION_TASKS[-2:]] == list(NEW)
    shapes = {
        "coord-spec-submittal": (
            F.SPEC_PLANTED, "check",
            ["fixture_coord_manual", "fixture_coord_window_section",
             "fixture_coord_window_submittal"]),
        "coord-reissue": (
            F.REISSUE_PLANTED, "compare",
            ["fixture_coord_set", "fixture_coord_set_2"])}
    for task_id, (planted, category, documents) in shapes.items():
        task = BY_ID[task_id]
        assert task in OPEN_TASKS
        assert task.doc_type == "coordination_set" and task.page == "review"
        assert task.category == category and task.documents == documents
        assert [c["label"] for c in task.checks] == [p.key for p in planted]
        assert {c["type"] for c in task.checks} == {"stated_together"}
        # the question names no planted value, sheet or section
        for word in ("1.90", "08 80 00", "anchorage", "water", "174",
                     "A-301", "S-501", "A-102", "106", "parapet"):
            assert word not in task.question, (task_id, word)


PERFECT_SPEC = """Review of Submittal 08 51 13-01 against Section 08 51 13:

1. Thermal performance does not comply: the data sheet's whole-window
   U-value is 1.90 W/m2K against the 1.60 W/m2K maximum of 2.1 E (the 1.10
   figure is the centre of the glass only).
2. Missing: SD-05 Design Data. No anchorage calculations were submitted.
3. Missing test: the laboratory report ET-25-0412 covers air leakage and
   the structural tests only; there is no water penetration result, which
   2.1 D and SD-06 require.
4. The schedule of values prices the glazing units under Section 08 80 00,
   which does not exist in the project manual; glazing is part of 08 51 13.
5. The schedule of values' five lines total 174,500, not the 178,500 shown.
6. The transmittal's "DEVIATIONS: NONE" is wrong given item 1.
"""

CARELESS_SPEC = """The submittal looks complete. The U-value of 1.10 W/m2K
is within the 1.60 limit, the air leakage of 0.20 meets the 0.30 limit, and
the structural tests at 1.80 kPa passed. The product data, the test report
and the schedule of values (total 178,500) are attached and cite Sections
08 51 13 and 07 92 00. The field water test is not reported, as expected
before installation. I recommend approval."""

PERFECT_REISSUE = """Issue 2 against issue 1:

- A-201: T.O. parapet raised from EL. 109.30 to 109.60, clouded and listed
  as revision 2. A-301 was not updated: the building sections still show
  the parapet at 109.30, so the two sheets disagree.
- S-501: F3 reinforcing revised from 10-20M to 12-20M each way inside a
  delta 2 cloud, but there is no revision 2 entry in S-501's revision block
  (only revision 0).
- A-102: the revision block adds revision 2, "DOOR 205 RELOCATED", yet door
  205 has not moved; nothing on the sheet changed.
- A-601: door 106 renumbered to 110, clouded and listed. The first floor
  plan A-101 still tags it 106, so the plan and the schedule do not match.
- G-001: the index and the revision block updated for the issue.
"""

CARELESS_REISSUE = """Changes in issue 2:
- A-201: T.O. parapet raised from 109.30 to 109.60, clouded and listed as
  revision 2.
- A-601: door 106 renumbered 110, clouded and listed.
- A-102: door 205 relocated, listed as revision 2.
- S-501: F3 reinforcing changed from 10-20M to 12-20M, clouded.
- G-001: index updated.
All other sheets, including A-301 and A-101, are unchanged."""

# the right changes with the wrong flags: S-501 called unclouded, A-102's
# door taken as moved, the twins never named
WRONG_REISSUE = """- A-201: parapet now EL. 109.60 (was 109.30), revision 2.
- S-501: F3 reinforcing changed to 12-20M, not clouded and not in the
  revision block.
- A-102: door 205 relocated as listed in revision 2.
- A-601: door 106 is now door 110."""


def _passed(task_id: str, answer: str):
    return [c["label"] for c in BY_ID[task_id].checks
            if C.run_check(c, answer)["passed"]]


def test_a_perfect_answer_scores_full_and_a_careless_one_nothing():
    assert _passed("coord-spec-submittal", PERFECT_SPEC) == [
        p.key for p in F.SPEC_PLANTED]
    assert _passed("coord-reissue", PERFECT_REISSUE) == [
        p.key for p in F.REISSUE_PLANTED]
    for task_id, careless in (("coord-spec-submittal", CARELESS_SPEC),
                              ("coord-reissue", CARELESS_REISSUE),
                              ("coord-reissue", WRONG_REISSUE)):
        assert _passed(task_id, careless) == [], task_id
        assert _passed(task_id, "") == [], task_id


@pytest.mark.parametrize("planted", F.SPEC_PLANTED + F.REISSUE_PLANTED,
                         ids=lambda p: p.key)
def test_each_new_check_takes_its_own_truth_and_no_other(planted):
    check = _PLANTED_CHECKS[planted.key]
    assert C.run_check(check, planted.truth)["passed"]
    others = " ".join(p.truth for p in ALL_PLANTED if p.key != planted.key)
    assert not C.run_check(check, others)["passed"], planted.key


def test_no_older_check_takes_a_new_truth():
    new = " ".join(p.truth for p in F.SPEC_PLANTED + F.REISSUE_PLANTED)
    for p in F.LONG_PLANTED + F.SUBMITTAL_PLANTED:
        assert not C.run_check(_PLANTED_CHECKS[p.key], new)["passed"], p.key


def test_a_clouded_change_is_not_an_unclouded_one():
    check = _PLANTED_CHECKS["cloud-not-listed"]
    said = ("S-501: F3 reinforcing changed from 10-20M to 12-20M; it is "
            "clouded, but the revision block has no revision 2 entry.")
    assert C.run_check(check, said)["passed"]
    for wrong in (said.replace("it is clouded", "it is not clouded"),
                  said.replace("it is clouded", "there is no cloud"),
                  said.replace("has no revision 2 entry",
                               "lists revision 2")):
        assert not C.run_check(check, wrong)["passed"], wrong


def test_the_twin_wants_the_other_sheet_left_behind():
    check = _PLANTED_CHECKS["parapet-twin"]
    told = ("A-201 raised the parapet from 109.30 to 109.60; A-301 should "
            "also have been revised and still shows 109.30.")
    assert C.run_check(check, told)["passed"]
    listed = ("A-201 raised the parapet from 109.30 to 109.60. A-301 is "
              "unchanged.")
    assert not C.run_check(check, listed)["passed"]


def test_the_dry_run_resolves_the_new_documents(tmp_path):
    out = score_review_suite(out_dir=str(tmp_path), dry_run=True,
                             ids=list(NEW))
    text = out["results_md"]
    assert "2 tasks" in text
    assert ("fixture_coord_set_2 -> north_yard_annex_drawings_issue_2.pdf"
            in text)
    assert ("fixture_coord_window_submittal -> "
            "submittal_08_51_13-01_aluminum_windows.pdf") in text
    assert "MISSING" not in text.upper().replace("SECTION-MISSING", "")
