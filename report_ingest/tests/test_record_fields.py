"""What the record keeps for the owner's question list's code rows
(BACKLOG RI-102): triage's document type, each page's printed page number,
the document's own list of appendices, the caption labels the caption count
saw, and the years and firms of the report's own investigation.

Offline: made-up records, a made-up list of rows and the synthetic report
of :mod:`report_ingest.tests.narrative_fixtures` on scripted engines. Every
question, key, value and name here is invented.
"""

from __future__ import annotations

import json

import pytest

from report_ingest import question_list as ql
from report_ingest import record_facts
from report_ingest.model import (AppendixEntry, BoundReport, DocumentFacts,
                                 DrillingDetails, FieldWork, Investigation,
                                 NarrativeFacts, PageLabel, ReportRecord)


# ---------------------------------------------------------------------------
# a record saved before the fields
# ---------------------------------------------------------------------------

#: A record as one was written before RI-102: none of the new fields.
OLD_RECORD = ReportRecord(
    document=DocumentFacts(report_id="R00", n_pages=3, workflow="standard"),
    page_labels=[PageLabel(page=n, label=lab)
                 for n, lab in enumerate(["cover", "narrative",
                                          "boring_log"])],
    narrative=NarrativeFacts(counted={"tables": 1, "figures": 2}),
    investigations=[Investigation(investigation_id="B-1", pages=[2],
                                  date_started="5/6/2031")])


def test_a_record_saved_before_loads_and_writes_byte_for_byte():
    saved = OLD_RECORD.model_dump_json(indent=2)
    blob = json.loads(saved)
    for name in ("document_type", "printed_pages", "appendices"):
        assert name not in blob["document"]
    assert "caption_labels" not in blob["narrative"]
    assert "field_work" not in blob
    again = ReportRecord.model_validate_json(saved)
    assert again.model_dump_json(indent=2) == saved
    assert again.document.printed_pages == {}
    assert again.document.appendices == []
    assert again.narrative.caption_labels == {}
    assert again.field_work is None


def test_the_new_fields_round_trip():
    record = OLD_RECORD.model_copy(deep=True)
    record.document.document_type = "geotechnical report"
    record.document.printed_pages = {1: "1", 2: "A-1"}
    record.document.appendices = [AppendixEntry(
        number="A", title="Logs", printed_page="A-1", page=2)]
    record.narrative.caption_labels = {"appendix_figures": ["figure A-1"]}
    record.field_work = FieldWork(years=[2031], dates=["5/6/2031"],
                                  firms=["Made-Up Drilling"], holes=1,
                                  pages=[2])
    text = record.model_dump_json()
    blob = json.loads(text)
    # JSON keys are text; they load back as page numbers.
    assert blob["document"]["printed_pages"] == {"1": "1", "2": "A-1"}
    again = ReportRecord.model_validate_json(text)
    assert again.document.printed_pages == {1: "1", 2: "A-1"}
    assert again == record


def test_the_exports_of_a_record_saved_before_do_not_change(tmp_path):
    from report_ingest.diggs_writer import write_diggs
    from report_ingest.writers import library_page, summary_markdown

    record = ReportRecord.model_validate_json(OLD_RECORD.model_dump_json())
    filled = record.model_copy(deep=True)
    filled.document.printed_pages = {1: "1"}
    filled.document.appendices = [AppendixEntry(number="A", title="Logs")]
    filled.narrative.caption_labels = {"figures": ["figure 1"]}
    filled.field_work = FieldWork(years=[2031], holes=1)
    # The summary and the library page read none of the new fields.
    assert summary_markdown(record) == summary_markdown(filled)
    assert library_page(record, "k") == library_page(filled, "k")
    assert write_diggs(record, document_id="d") == write_diggs(
        filled, document_id="d")


# ---------------------------------------------------------------------------
# the readings the graph stores
# ---------------------------------------------------------------------------

class _Summary:
    def __init__(self, page, printed):
        self.page, self.printed_page = page, printed


def test_printed_pages_keep_only_the_pages_that_print_one():
    summaries = [_Summary(0, None), _Summary(1, 1), _Summary(2, 2),
                 _Summary(3, "")]
    assert record_facts.printed_pages(summaries) == {1: "1", 2: "2"}
    assert record_facts.printed_pages(summaries, [2, 3]) == {2: "2"}


def test_caption_labels_keep_what_the_count_saw():
    counted = {"tables": 1, "figures": 0, "table_labels": ["table 1"],
               "figure_labels": [],
               "appendix_figure_labels": ["figure A-1", "figure B-2"]}
    assert record_facts.caption_labels(counted) == {
        "tables": ["table 1"],
        "appendix_figures": ["figure A-1", "figure B-2"]}
    assert record_facts.caption_labels(None) == {}


def _log(name, page, *, date="", contractor="", fields=None):
    return Investigation(investigation_id=name, pages=[page],
                         date_started=date,
                         drilling=DrillingDetails(contractor=contractor),
                         fields=fields or {})


def test_field_work_is_the_own_logs_years_and_firms():
    record = ReportRecord(
        investigations=[
            _log("B-1", 4, date="12/30/2030", contractor="Made-Up Drilling",
                 fields={"Drilling Company": "Made-Up Drilling",
                         "Contractor Foreman": "A. Sample", "Firm": "NA"}),
            _log("B-2", 5, date="Jan 3, 2031",
                 fields={"Consultant": "Invented Geo", "Engineer": "A. N."}),
            _log("B-3", 6),                         # prints neither
            _log("P-1", 9, date="2/2/1999",
                 fields={"Consultant": "Older Geo"})],
        bound_documents=[BoundReport(bound_id="bound1",
                                     kind="appended_prior_report",
                                     first_page=8, last_page=10)])
    work = record_facts.field_work(record)
    assert work.years == [2030, 2031]
    assert work.dates == ["12/30/2030", "Jan 3, 2031"]
    # Only a firm, company or consultant field names the firm: never the
    # driller (the contractor, a drilling company), a foreman, an "Engineer"
    # (a person as often as a firm) or "NA".
    assert work.firms == ["Invented Geo"]
    assert (work.holes, work.pages) == (2, [4, 5])
    # A two-digit year is not read; a log with nothing gives no field work.
    assert record_facts.field_work(ReportRecord(
        investigations=[_log("B-1", 1, date="3/8/31")])).years == []
    assert record_facts.field_work(ReportRecord(
        investigations=[_log("B-1", 1)])) is None


# ---------------------------------------------------------------------------
# the graph keeps them
# ---------------------------------------------------------------------------

pytest.importorskip("planlens.document.roles")


@pytest.fixture()
def run(tmp_path, monkeypatch):
    """One standard scripted run of the synthetic report (today's readers),
    the parent's logs printing a made-up date, driller and consultant."""
    from report_ingest import log_reader
    from report_ingest.graph import MODEL_FIRST_ENV, ingest_report
    from report_ingest.log_reader import ReadDrilling
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.fake_engine import FakeEngine
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    monkeypatch.setenv(MODEL_FIRST_ENV, "off")
    seed = log_reader.seed_from_grid

    def with_consultant(*args, **kwargs):
        # The header field a form reader pulls off the parent's logs.
        floor = seed(*args, **kwargs)
        if floor.investigation_id in ("B-1", "TP-1"):
            floor.fields = {**floor.fields, "Consultant": "Invented Geo"}
        return floor

    monkeypatch.setattr(log_reader, "seed_from_grid", with_consultant)
    pdf = tmp_path / "SYN.pdf"
    pdf.write_bytes(build_narrative_report().pdf)
    script = tg.full_script()
    for turn in script:
        final = turn.get("final") if isinstance(turn, dict) else None
        # The parent's two logs; the bound report's is BH-1.
        if getattr(final, "investigation_id", "") in ("B-1", "TP-1"):
            turn["final"] = final.model_copy(update={
                "date_started": "4/5/2031",
                "drilling": ReadDrilling(contractor="Made-Up Drilling")})
    out = tmp_path / "out"
    record = ingest_report(pdf, FakeEngine(script), out_dir=out,
                           report_id="SYN")
    child = ReportRecord.model_validate_json(
        (out / record.bound_documents[0].record_path).read_bytes())
    return record, child, out


def test_the_graph_keeps_what_the_run_read(run):
    record, child, _out = run
    # Triage's document type, which until RI-102 was in triage.json only.
    assert record.document.document_type == "geotechnical report"
    # planlens' printed page numbers, the parent's over the whole file and
    # the bound report's over its own pages.
    assert record.document.printed_pages
    assert set(child.document.printed_pages) <= set(range(
        child.parent.first_page, child.parent.last_page + 1))
    assert set(child.document.printed_pages.items()) <= set(
        record.document.printed_pages.items())
    # The document's own list of appendices.
    assert [a.number for a in record.document.appendices] == [
        "A", "B", "C", "D"]
    assert all(a.title for a in record.document.appendices)
    # The caption labels the narrative's caption count saw.
    assert record.narrative.caption_labels["figures"] == [
        "figure 1", "figure 2"]
    # The bound report's type is what its identity call said.
    assert child.document.document_type == "geotechnical report"
    # The years and firms of this report's own logs, not the prior's; the
    # consultant field, not the driller.
    assert record.field_work.years == [2031]
    assert record.field_work.firms == ["Invented Geo"]
    assert child.field_work is None


def test_a_bound_prior_report_s_record_holds_its_logs(run):
    """Rows 19-21: a prior report's logs are counted off its own record,
    which the question list finds beside the parent's."""
    record, child, out = run
    assert [i.investigation_id for i in child.investigations] == ["BH-1"]
    assert not [i for i in record.investigations
                if i.investigation_id == "BH-1"]
    rows = ql.QuestionList(questions=[_row("earlier_holes", "integer")],
                           readings={"earlier_holes": ("prior_hole_count",
                                                       "boring")},
                           map_name="made-up")
    without = ql.code_answers(record, rows)["earlier_holes"]
    assert without.status == "not said"
    found = ql.code_answers(record, rows, record_dir=out)["earlier_holes"]
    assert (found.status, found.value) == ("answered", 1)


# ---------------------------------------------------------------------------
# the code rows that answer now
# ---------------------------------------------------------------------------

def _row(key, data_type, options=(), parts=()):
    return ql.Question(row=1, output=key, category="Made up",
                       data_type=data_type, allow_null=True,
                       prompt=f"Give the {key}.", key=key,
                       answered_by="code", options=tuple(options),
                       parts=tuple(parts))


#: Made-up rows, one per new reading, and the page-label row whose printed
#: page part the record now fills.
ROWS = [
    _row("doc_kind", "string",
         options=("geotechnical report", "boring logs only")),
    _row("is_soils_report", "boolean"),
    _row("back_figures", "list of strings"),
    _row("back_figure_total", "integer"),
    _row("appendix_names", "list of records",
         parts=(ql.Part("number"), ql.Part("title"), ql.Part("pdf_page"))),
    _row("appendix_index", "list of records",
         parts=(ql.Part("label"), ql.Part("title"),
                ql.Part("first_pdf_page"))),
    _row("field_year", "integer"),
    _row("field_firm", "string"),
    _row("page_kinds", "dictionary",
         parts=(ql.Part("pdf_page"), ql.Part("sheet_class"),
                ql.Part("printed_page"))),
]
#: The document types a made-up yes/no row takes as yes: the code map's
#: call, not code's.
YES_TYPES = "geotechnical report|report addendum"
READINGS = {
    "doc_kind": ("document_type", ""),
    "is_soils_report": ("document_type", YES_TYPES),
    "back_figures": ("caption_labels", "appendix_figures"),
    "back_figure_total": ("caption_count", "appendix_figures"),
    "appendix_names": ("appendix_list", ""),
    "appendix_index": ("appendix_list", ""),
    "field_year": ("investigation_year", ""),
    "field_firm": ("investigation_firm", ""),
    "page_kinds": ("page_labels", ""),
}


def _list():
    return ql.QuestionList(questions=list(ROWS), readings=dict(READINGS),
                           map_name="made-up")


def _filled():
    record = ReportRecord.model_validate_json(OLD_RECORD.model_dump_json())
    record.investigations = []
    record.document.document_type = "geotechnical report"
    record.document.printed_pages = {1: "1"}
    record.document.appendices = [
        AppendixEntry(number="A", title="Boring Logs", page=2),
        AppendixEntry(number="B", title="Laboratory Data")]
    record.narrative.caption_labels = {"appendix_figures": ["figure A-1",
                                                            "figure A-2"]}
    record.field_work = FieldWork(years=[2030, 2031],
                                  dates=["12/30/2030", "1/2/2031"],
                                  firms=["Invented Geo"], holes=2,
                                  pages=[2])
    return record


def test_rows_the_record_could_not_answer_now_answer():
    before = ql.code_answers(
        ReportRecord.model_validate_json(
            OLD_RECORD.model_copy(update={"investigations": []})
            .model_dump_json()), _list())
    after = ql.code_answers(_filled(), _list())
    for key in READINGS:
        if key != "page_kinds":
            assert before[key].status == "not said", key
        assert after[key].status == "answered", (key, after[key].note)
    assert after["doc_kind"].value == "geotechnical report"
    assert after["is_soils_report"].value is True
    assert after["back_figures"].value == ["figure A-1", "figure A-2"]
    assert after["back_figure_total"].value == 2
    assert after["appendix_names"].value == [
        {"number": "A", "title": "Boring Logs", "pdf_page": 3},
        {"number": "B", "title": "Laboratory Data", "pdf_page": None}]
    assert after["appendix_index"].value == [
        {"label": "A", "title": "Boring Logs", "first_pdf_page": 3},
        {"label": "B", "title": "Laboratory Data", "first_pdf_page": None}]
    assert after["field_year"].value == 2030
    assert "the earliest of 2030, 2031" in after["field_year"].note
    assert after["field_firm"].value == "Invented Geo"
    # The page list's printed number comes off the record when the caller
    # gives none, and a caller's own still wins.
    assert before["page_kinds"].value[1]["printed_page"] == ""
    assert after["page_kinds"].value[1]["printed_page"] == "1"
    given = ql.code_answers(_filled(), _list(), printed_pages={1: "ii"})
    assert given["page_kinds"].value[1]["printed_page"] == "ii"


def test_a_record_saved_before_answers_the_year_off_its_logs():
    """field_work is not on a record saved before RI-102; the reading takes
    the same reading of its logs."""
    code = ql.code_answers(OLD_RECORD, _list())
    assert (code["field_year"].status, code["field_year"].value) == (
        "answered", 2031)
    assert code["field_firm"].status == "not said"


def test_a_document_type_off_the_row_s_list_is_not_given():
    record = _filled()
    record.document.document_type = "design memorandum"
    code = ql.code_answers(record, _list())
    assert code["doc_kind"].status == "not said"
    assert "not near one option" in code["doc_kind"].note


def test_a_yes_no_row_is_yes_only_for_the_types_the_map_names():
    """Yes-or-nothing: a type the code map does not name is not a no, and
    a yes/no row mapped with no types is not answered."""
    rows = ql.QuestionList(
        questions=[_row("is_soils_report", "boolean"),
                   _row("bare_yes_no", "boolean")],
        readings={"is_soils_report": ("document_type", YES_TYPES),
                  "bare_yes_no": ("document_type", "")},
        map_name="made-up")
    record = _filled()
    code = ql.code_answers(record, rows)
    assert code["bare_yes_no"].status == "not said"
    assert "needs the document types" in code["bare_yes_no"].note
    record.document.document_type = "partial report"
    code = ql.code_answers(record, rows)
    assert (code["is_soils_report"].status,
            code["is_soils_report"].value) == ("not said", None)
    assert "cannot show a no" in code["is_soils_report"].note


def test_the_new_readings_refuse_an_argument_they_cannot_use():
    with pytest.raises(ValueError):
        ql.READINGS["caption_labels"]("plates")
    with pytest.raises(ValueError):
        ql.READINGS["appendix_list"]("anything")
    # A yes type the triage vocabulary does not have.
    with pytest.raises(ValueError):
        ql.READINGS["document_type"]("geotechnical report|soils memo")
