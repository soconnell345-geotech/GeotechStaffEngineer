"""The question list's row with one record per investigation (BACKLOG
RI-207): code's ``investigations`` reading and its comparison with the
model's list.

Offline: a made-up row and made-up records. Every key, part name, firm,
date and count here is invented.
"""

from __future__ import annotations

import pytest

from report_ingest import question_list as ql
from report_ingest.model import (BoundReport, DocumentFacts, Investigation,
                                 PageLabel, QAEntry, ReportRecord)
from report_ingest.question_list import Answer

PARTS = (ql.Part("year"), ql.Part("consultant"), ql.Part("borings"),
         ql.Part("test_pits"), ql.Part("cones"),
         ql.Part("status", ("current", "prior")))


def _row(parts=PARTS, answered_by="both", data_type="list of records"):
    return ql.Question(row=1, output="Site work", category="Made up",
                       data_type=data_type, allow_null=True,
                       prompt="List the site work.", key="site_work",
                       answered_by=answered_by, parts=tuple(parts))


def _list(row=None, arg=""):
    return ql.QuestionList(questions=[row or _row()],
                           readings={"site_work": ("investigations", arg)},
                           map_name="made-up")


def _log(name, page, kind="boring", date="", firm=""):
    return Investigation(investigation_id=name, pages=[page], kind=kind,
                         date_started=date,
                         fields={"Consultant": firm} if firm else {})


def _record(logs, bound=(), qa=()):
    return ReportRecord(
        document=DocumentFacts(report_id="R00", n_pages=30),
        page_labels=[PageLabel(page=0, label="cover")],
        investigations=list(logs), bound_documents=list(bound),
        qa=list(qa))


OWN = [_log("B-1", 4, date="3/4/2031", firm="Invented Geo"),
       _log("B-2", 5, date="3/5/2031", firm="Invented Geo"),
       _log("TP-1", 6, kind="test_pit", date="3/6/2031")]

PRIOR_DOC = BoundReport(bound_id="bound1", kind="appended_prior_report",
                        first_page=20, last_page=29, date="June 2019",
                        firm="Older Geo", counts={"investigations": 2})
PRIOR = _record([_log("BH-1", 22, date="8/9/2018", firm="Older Geo"),
                 _log("C-1", 24, kind="cpt", date="8/10/2018")])

PARTIAL = QAEntry(kind="partial", rank="open", where="document.completeness",
                  detail="this copy looks incomplete: made up")


def _code(record, row=None, arg="", **kwargs):
    return ql.code_answers(record, _list(row, arg), **kwargs)["site_work"]


def test_the_report_s_own_investigation_alone():
    got = _code(_record(OWN))
    assert (got.status, got.lower_bound) == ("answered", True)
    # No cone log beside the borings and pits, and no partial flag: 0.
    assert got.value == [{"year": "2031", "consultant": "Invented Geo",
                          "borings": "2", "test_pits": "1", "cones": "0",
                          "status": "current"}]
    assert "may be short" in got.note and "lower bound" in got.note
    assert got.pdf_pages == [5]


def test_with_one_bound_prior_report(tmp_path):
    record = _record(OWN, [PRIOR_DOC])
    got = _code(record, bound={"bound1": PRIOR})
    assert got.value == [
        {"year": "2031", "consultant": "Invented Geo", "borings": "2",
         "test_pits": "1", "cones": "0", "status": "current"},
        {"year": "2018", "consultant": "Older Geo", "borings": "1",
         "test_pits": "0", "cones": "1", "status": "prior"}]
    # The prior report's own record is found beside the parent's.
    path = tmp_path / "bound" / "bound1" / "report.record.json"
    path.parent.mkdir(parents=True)
    path.write_text(PRIOR.model_dump_json(), encoding="utf-8")
    doc = PRIOR_DOC.model_copy(update={
        "record_path": "bound/bound1/report.record.json"})
    assert _code(_record(OWN, [doc]), record_dir=tmp_path).value == got.value
    # Its record not given: listed as prior, its parts empty, and why.
    bare = _code(record)
    assert bare.value[1] == {"year": "", "consultant": "", "borings": "",
                             "test_pits": "", "cones": "", "status": "prior"}
    assert "its record holds 2 log(s) and was not given" in bare.note


def test_two_firms_on_the_logs_leave_the_consultant_empty():
    logs = [_log("B-1", 4, date="3/4/2031", firm="Invented Geo"),
            _log("B-2", 5, date="3/5/2031", firm="Made-Up Soils")]
    got = _code(_record(logs))
    assert got.value[0]["consultant"] == ""
    assert "its logs name 2 firms (Invented Geo; Made-Up Soils)" in got.note
    assert got.value[0]["year"] == "2031"


def test_a_prior_report_with_undated_logs_has_no_year():
    prior = _record([_log("BH-1", 22, firm="Older Geo"),
                     _log("BH-2", 23, firm="Older Geo")])
    got = _code(_record(OWN, [PRIOR_DOC]), bound={"bound1": prior})
    assert got.value[1]["year"] == ""          # not 2019, its report date
    assert got.value[1]["consultant"] == "Older Geo"
    assert "(June 2019) is not the year the field work was done" in got.note
    # Only borings in that record: nothing of another kind to count beside
    # is needed, the borings are; pits and cones are 0.
    assert (got.value[1]["borings"], got.value[1]["cones"]) == ("2", "0")


def test_a_partial_copy_leaves_a_kind_it_holds_none_of_empty():
    got = _code(_record(OWN, qa=[PARTIAL]))
    assert got.value[0]["borings"] == "2"
    assert got.value[0]["test_pits"] == "1"
    assert got.value[0]["cones"] == ""         # not 0: the copy is partial
    assert "cones left empty: the copy is flagged as incomplete" in got.note
    assert "a count may be short" in got.note


def test_no_log_of_its_own_is_not_an_investigation_code_lists():
    got = _code(_record([]))
    assert (got.status, got.value, got.lower_bound) == ("not said", None,
                                                        False)
    assert "holds no log of it" in got.note
    # A prior report with no log in its record is not listed either.
    doc = PRIOR_DOC.model_copy(update={"counts": {}})
    got = _code(_record([], [doc]), bound={"bound1": _record([])})
    assert got.status == "not said"
    assert "is not listed: its record holds no log" in got.note


def test_part_names_in_other_words_and_another_order():
    parts = (ql.Part("investigation_type", ("Previous study", "Current study")),
             ql.Part("number_of_cone_soundings"), ql.Part("pit_count"),
             ql.Part("boreholes"), ql.Part("geotechnical_firm"),
             ql.Part("yr_performed"), ql.Part("remarks"))
    row = _row(parts)
    got = _code(_record(OWN, [PRIOR_DOC]), row, bound={"bound1": PRIOR})
    assert got.status == "answered", got.note
    assert got.value == [
        {"investigation_type": "Current study",
         "number_of_cone_soundings": "0", "pit_count": "1", "boreholes": "2",
         "geotechnical_firm": "Invented Geo", "yr_performed": "2031",
         "remarks": ""},
        {"investigation_type": "Previous study",
         "number_of_cone_soundings": "1", "pit_count": "0", "boreholes": "1",
         "geotechnical_firm": "Older Geo", "yr_performed": "2018",
         "remarks": ""}]
    # A part that names two of the counts takes neither.
    mixed = _row((ql.Part("borings_and_test_pits"), ql.Part("year"),
                  ql.Part("status", ("current", "prior"))))
    got = _code(_record(OWN), mixed)
    assert got.value == [{"borings_and_test_pits": "", "year": "2031",
                          "status": "current"}]


def test_the_code_map_may_name_the_log_kinds_of_a_count():
    logs = OWN + [_log("MW-1", 7, kind="well", date="3/7/2031")]
    assert _code(_record(logs)).value[0]["borings"] == "2"
    got = _code(_record(logs), arg="borings=boring,well")
    assert got.value[0]["borings"] == "3"
    for wrong in ("holes=boring", "borings", "borings=shaft"):
        with pytest.raises(ValueError):
            ql.READINGS["investigations"](wrong)


def test_a_row_that_is_not_a_list_of_records_is_not_answered():
    row = _row((), data_type="string")
    assert _code(_record(OWN), row).status == "not said"


# ---------------------------------------------------------------------------
# the comparison with the model's list
# ---------------------------------------------------------------------------

def _mine(year="2031", borings="2", status="current", **more):
    return {"year": year, "consultant": "Invented Geo", "borings": borings,
            "test_pits": "1", "cones": "0", "status": status, **more}


def _model(*records):
    return Answer(value=list(records), status="answered")


def test_one_more_prior_investigation_in_the_model_s_list_agrees():
    row = _row()
    code = _code(_record(OWN))
    described = {"year": "2009", "consultant": "Someone Else",
                 "borings": "4", "test_pits": "", "cones": "",
                 "status": "prior"}
    out = ql.compare(row, _model(_mine(consultant="Invented Geo, Inc."),
                                 described), code)
    assert out.source == "both: agree"
    assert "1 investigation(s) only the report text lists" in out.note
    assert "code (may be short)" in out.note
    assert len(out.value) == 2 and out.code_value == code.value


def test_a_different_year_for_the_current_one_disagrees():
    row = _row()
    code = _code(_record(OWN))
    out = ql.compare(row, _model(_mine(year="2030")), code)
    assert out.source == "both: disagree"
    assert "no investigation in the report text's list matches" in out.note
    # A different count disagrees; a count or year the model leaves empty,
    # or writes in words with the same number, does not.
    assert ql.compare(row, _model(_mine(borings="3")),
                      code).source == "both: disagree"
    assert ql.compare(row, _model(_mine(year="", borings="2 borings")),
                      code).source == "both: agree"
    # Current against prior is not a match, and an empty list is not one.
    assert ql.compare(row, _model(_mine(status="prior")),
                      code).source == "both: disagree"
    assert ql.compare(row, _model(), code).source == "both: disagree"


def test_each_of_code_s_investigations_needs_a_record_of_its_own():
    row = _row()
    record = _record(OWN, [PRIOR_DOC, PRIOR_DOC.model_copy(update={
        "bound_id": "bound2", "first_page": 10, "last_page": 19})])
    code = _code(record, bound={"bound1": PRIOR, "bound2": PRIOR})
    prior = {"year": "2018", "consultant": "", "borings": "1",
             "test_pits": "", "cones": "1", "status": "prior"}
    assert ql.compare(row, _model(_mine(), prior),
                      code).source == "both: disagree"
    assert ql.compare(row, _model(prior, _mine(), dict(prior)),
                      code).source == "both: agree"


def test_the_lower_bound_rule_is_this_reading_s_only():
    """Another list-of-records row still compares as before: one more
    record in the model's list is a disagreement."""
    row = _row((ql.Part("title"), ql.Part("pdf_page")))
    code = Answer(value=[{"title": "A", "pdf_page": 3}], status="answered",
                  source="code")
    assert not code.lower_bound
    more = _model({"title": "A", "pdf_page": 3}, {"title": "B",
                                                  "pdf_page": 9})
    assert ql.compare(row, more, code).source == "both: disagree"


def test_the_sheet_s_note_says_the_list_may_be_short(tmp_path):
    import openpyxl

    row = _row(answered_by="code")
    rows = _list(row)
    answers = ql.code_answers(_record(OWN), rows)
    path = ql.answer_sheet(rows, answers, tmp_path / "sheet.xlsx")
    sheet = openpyxl.load_workbook(path)["Answers"]
    cells = [c.value for c in sheet[2]]
    assert cells[6] == "code" and "may be short" in cells[7]
    assert "status: current" in cells[4]
