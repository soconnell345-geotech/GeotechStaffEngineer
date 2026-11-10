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
