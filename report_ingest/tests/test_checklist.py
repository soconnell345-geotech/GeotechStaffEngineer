"""The geotechnical report review checklist, run on a report's record.

Offline: no model, no network, no planlens. The record is the hand-built one
the reconciler and the writers are tested on
(:mod:`report_ingest.tests.record_fixtures`), reconciled, with the awkward
cases a review must catch planted in it: a summary-table value that
disagrees with its sheet, a sheet naming a hole the report does not carry,
a log page nobody read, and a stated count the logs do not bear out.

What is pinned:

* the checklist is DATA, DRAFT, its items unchanged since it moved here
  (BACKLOG RI-34; OWN-31 is the owner's markup);
* every code item runs on the record and reads what the record holds: the
  page labels, the explorations and laboratory tests, the narrative's
  answers and citations, and the reconciler's QA entries;
* a code item that cannot run on the record (none given, no page labels)
  is handed to the agent with the judgement items, never dropped;
* the tool reads a record by its path or its folder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from report_ingest import checklist as CL
from report_ingest.model import PageLabel
from report_ingest.reconciler import reconcile
from report_ingest.tests.record_fixtures import (
    atterberg, build_record, general_facts, gradation, orphan_sheet,
    page_context, summary_table,
)

#: The items as the owner's first draft has them (2026-10-08 draft 1). A
#: change to this list is the owner's markup (OWN-31), not a refactor.
DRAFT_ITEMS = {
    "completeness.explorations_have_logs": ("code", "explorations_vs_logs"),
    "completeness.pages_read": ("code", "logs_and_lab_read"),
    "completeness.summary_and_sheets": ("code", "summary_and_sheets"),
    "completeness.campaigns": ("judge", None),
    "completeness.groundwater": ("judge", None),
    "completeness.locations": ("judge", None),
    "consistency.summary_vs_sheets": ("code", "cross_checks"),
    "consistency.lab_ids_on_logs": ("code", "lab_ids_vs_logs"),
    "consistency.descriptions_vs_lab": ("judge", None),
    "consistency.text_vs_logs": ("judge", None),
    "consistency.units": ("code", "cross_checks"),
    "currency.governing_campaign": ("judge", None),
    "currency.code_editions": ("judge", None),
    "traceability.values_cite_pages": ("code", "values_cite_pages"),
    "traceability.recommendations_trace": ("judge", None),
    "plausibility.values": ("judge", None),
    "recommendations.parameters": ("judge", None),
    "recommendations.topics": ("judge", None),
}


def _record(**kw):
    """The fixture record, reconciled, with its page labels."""
    ctx = page_context()
    rec = build_record(**kw)
    rec = reconcile(rec, labels=ctx["labels"], items=ctx["items"])
    rec.page_labels = [PageLabel(page=p, label=label)
                       for p, label in sorted(ctx["labels"].items())]
    return rec


def _by_id(result):
    return {row["id"]: row for row in result["code_checks"]}


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------

def test_the_checklist_is_the_owners_draft_unchanged():
    data = CL.load_checklist()
    assert data["status"] == "DRAFT"
    assert data["version"] == "2026-10-08 draft 1"
    assert data["title"] == "Geotechnical report review checklist"
    got = {i["id"]: (i["mode"], i.get("check")) for i in data["items"]}
    assert got == DRAFT_ITEMS
    for item in data["items"]:
        assert item["text"]
        if item["mode"] == "code":
            assert item["check"] in CL.CHECKS, item["id"]
    kinds = {i["id"]: i.get("kinds") for i in data["items"]}
    assert kinds["consistency.summary_vs_sheets"] == ["disagreement",
                                                      "conflict"]
    assert kinds["consistency.units"] == ["unconverted"]
    # the record holds every reader's QA entries, so the summary item is
    # narrowed to the laboratory ones (see the test below)
    wheres = {i["id"]: i.get("where") for i in data["items"]}
    assert wheres["consistency.summary_vs_sheets"] == "lab_tests."
    # the notes say where it lives now
    assert "report_ingest/checklist.py" in data["how_it_is_used"]
    assert "funhouse_agent" not in data["how_to_edit"]


def test_the_data_file_ships_in_the_wheel():
    root = os.path.dirname(os.path.dirname(os.path.abspath(CL.__file__)))
    text = open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
    block = text[text.index("report_ingest = ["):]
    block = block[:block.index("]") + 1]
    assert "report_review_checklist.json" in block
    assert os.path.dirname(CL.DATA_FILE) == os.path.dirname(
        os.path.abspath(CL.__file__))


def test_importing_it_costs_neither_planlens_nor_langchain():
    code = ("import sys, report_ingest.checklist as c; c.load_checklist(); "
            "bad = [m for m in ('planlens', 'langchain_core', 'anthropic', "
            "'funhouse_agent') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=os.path.dirname(os.path.dirname(
                       os.path.abspath(CL.__file__))))


def test_the_counted_fields_are_the_narrative_readers():
    from report_ingest import narrative_reader as NR
    assert set(CL.EXPLORATION_FIELDS) == set(NR.DETERMINISTIC_FIELDS)
    assert set(CL.COUNTED_BY_CODE) == set(NR.DETERMINISTIC_FIELDS) | set(
        NR._COUNTED_FIELD)


# ---------------------------------------------------------------------------
# The code items on a record
# ---------------------------------------------------------------------------

def test_a_clean_record_passes_what_it_can():
    gen = general_facts()
    for field in gen.answered():
        gen.citations.setdefault(field, list(gen.citations["boringCount"]))
    gen.testPitDictionary = ["TP-1"]
    gen.citations["testPitDictionary"] = list(gen.citations["boringCount"])
    rec = _record(general=gen, lab_tests=[atterberg(), gradation(),
                                          summary_table()])
    for inv in rec.investigations:          # B-1's second sheet was read too
        if inv.investigation_id == "B-1":
            inv.pages = [7, 8]
    rec.natural_hazards.citations.update(
        {f: list(gen.citations["boringCount"])
         for f in rec.natural_hazards.answered()})
    by = _by_id(CL.run_checklist(rec))
    for item in ("completeness.explorations_have_logs",
                 "completeness.pages_read", "completeness.summary_and_sheets",
                 "consistency.summary_vs_sheets",
                 "consistency.lab_ids_on_logs", "consistency.units",
                 "traceability.values_cite_pages"):
        assert by[item]["status"] == "pass", (item, by[item])
    assert "logs 3 of 3" in by["completeness.pages_read"]["detail"]


@pytest.mark.parametrize("read", [True, False])
def test_a_bound_documents_sheets_are_checked_in_its_own_record(read):
    # BACKLOG RI-14: pages 12-13 lie in a report bound inside this one and
    # were read into ITS record, so this one holds nothing off them -- and
    # is not failed for it. A bound document that was not read is.
    from report_ingest.model import BoundReport

    rec = _record(lab_tests=[])
    for inv in rec.investigations:
        if inv.investigation_id == "B-1":
            inv.pages = [7, 8]
    rec.bound_documents = [BoundReport(bound_id="bound1", pages="12-18",
                                       first_page=12, last_page=18,
                                       read=read)]
    row = _by_id(CL.run_checklist(rec))["completeness.pages_read"]

    assert row["status"] == ("pass" if read else "fail"), row
    assert ("laboratory sheets" in row["detail"]) is (not read)


def test_the_planted_faults_are_found():
    gen = general_facts()
    gen.boringCount = 2                      # the logs show one boring
    rec = _record(general=gen, lab_tests=[
        atterberg(), gradation(), summary_table(ll_for_s1=41.0),
        orphan_sheet()])
    res = CL.run_checklist(rec)
    by = _by_id(res)

    ex = by["completeness.explorations_have_logs"]
    assert ex["status"] == "fail"
    assert any("narrative states 2 borings" in e for e in ex["entries"])
    assert "RI-24" in ex["detail"]           # what the record cannot hold

    pages = by["completeness.pages_read"]
    assert pages["status"] == "fail"
    assert "exploration logs 2 of 3" in pages["detail"]
    assert "PDF pages 9" in pages["detail"]  # page 8, B-1's second sheet

    sv = by["consistency.summary_vs_sheets"]
    assert sv["status"] == "fail"
    assert any("summary table 41" in e and "atterberg sheet 38" in e
               for e in sv["entries"])

    ids = by["consistency.lab_ids_on_logs"]
    assert ids["status"] == "fail" and any("B-9" in e
                                           for e in ids["entries"])
    sh = by["completeness.summary_and_sheets"]
    assert sh["status"] == "fail" and "B-9 S-1" in sh["detail"]
    assert by["consistency.units"]["status"] == "pass"

    lines = CL.failed_lines(rec)
    assert any(line.startswith("- Summary-table values") for line in lines)
    assert res["report"].startswith("Rosewood Terrace Development")


def test_the_summary_item_reads_only_the_summary_against_the_sheets():
    """A real record carries every reader's QA entries: a log value the grid
    and the model read differently, an SPT whose drives do not add up, a
    DIGGS read-back, one sheet read two ways. None of them is a summary
    value against its sheet."""
    from report_ingest.model import QAEntry
    rec = _record(lab_tests=[atterberg(), gradation(), summary_table()])
    rec.qa += [
        QAEntry(kind="disagreement", where="investigations[B-1]",
                detail="N-value at 5 ft: the grid read 12, the model read "
                       "21; the record carries the grid's.", pages=[7]),
        QAEntry(kind="conflict", where="B-1.spt",
                detail="5.0 ft: its drives add up to 0.6 m but the interval "
                       "is 0.45 m", pages=[7]),
        QAEntry(kind="conflict", where="diggs.roundtrip",
                detail="reading the DIGGS file back does not give the "
                       "record's own values"),
        QAEntry(kind="disagreement", where="lab_tests[atterberg B-1 5 ft]",
                detail="liquid limit: the tables read 38, the model read "
                       "39; the record carries the tables'.", pages=[12]),
    ]
    sv = _by_id(CL.run_checklist(rec))["consistency.summary_vs_sheets"]
    assert sv["status"] == "pass", sv


def test_a_sheet_of_a_kind_the_summary_does_not_carry_is_not_missing_from_it():
    """DR-21: chemistry beside a summary of index tests is not a fault."""
    from report_ingest.tests.record_fixtures import q
    from report_ingest.model import ChemicalResult, LabTest
    chem = LabTest(kind="chemical", investigation_id="B-1",
                   sample_id="S-2", depth_top=q(7.0, "ft", 13), pages=[13],
                   source_report="SYN",
                   result=ChemicalResult(kind="chemical", pH=7.2,
                                         sulfate=310.0))
    rec = _record(lab_tests=[atterberg(), gradation(), summary_table(),
                             chem])
    sh = _by_id(CL.run_checklist(rec))["completeness.summary_and_sheets"]
    assert sh["status"] == "pass"
    assert "chemical" in sh["detail"]


def test_an_exploration_count_answered_only_from_the_logs_is_not_a_check():
    gen = general_facts()
    gen.citations = {}                       # nothing the narrative stated
    rec = _record(general=gen)
    ex = _by_id(CL.run_checklist(rec))["completeness.explorations_have_logs"]
    assert ex["status"] == "unsure"
    assert "boring B-1" in ex["detail"] and "test_pit TP-1" in ex["detail"]


def test_values_with_no_page_and_answers_with_no_citation_fail():
    gen = general_facts()
    rec = _record(general=gen)
    rec.lab_tests[0].pages = []
    rec.lab_tests[0].prov = []
    row = _by_id(CL.run_checklist(rec))["traceability.values_cite_pages"]
    assert row["status"] == "fail"
    assert "laboratory test atterberg B-1 S-1" in row["detail"]
    assert "projectName" in row["detail"]
    # the counts the reader takes from the logs carry no quote by design
    assert "boringDictionary" not in row["detail"]


# ---------------------------------------------------------------------------
# What cannot run is handed over
# ---------------------------------------------------------------------------

def test_with_no_record_every_item_goes_to_the_agent():
    res = CL.run_checklist(None)
    assert res["status"] == "DRAFT"
    assert {r["status"] for r in res["code_checks"]} == {"not_run"}
    judged = {r["id"] for r in res["for_you_to_judge"]}
    assert judged == set(DRAFT_ITEMS)
    said = [r for r in res["for_you_to_judge"] if "code_said" in r]
    assert len(said) == 7 and CL.RECORD_NAME in said[0]["code_said"]


def test_a_record_without_page_labels_says_why_it_cannot_list_pages():
    ctx = page_context()
    rec = reconcile(build_record(), labels=ctx["labels"], items=ctx["items"])
    assert not rec.page_labels
    res = CL.run_checklist(rec)
    row = _by_id(res)["completeness.pages_read"]
    assert row["status"] == "not_run"
    assert "no per-page labels" in row["detail"]
    assert "boring_log 2" in row["detail"]
    assert "completeness.pages_read" in {r["id"]
                                         for r in res["for_you_to_judge"]}


def test_register_check_is_the_hook():
    data = {"title": "t", "status": "DRAFT", "items": [
        {"id": "x", "section": "S", "mode": "code", "check": "custom_hook",
         "text": "a check wired later"}]}
    res = CL.run_checklist(_record(), data)
    assert res["code_checks"][0]["status"] == "not_run"
    assert res["for_you_to_judge"][0]["id"] == "x"
    CL.register_check("custom_hook", lambda ctx, item: {
        "status": "pass", "detail": f"{len(ctx.record.investigations)}"})
    try:
        res = CL.run_checklist(_record(), data)
        assert res["code_checks"][0] == {
            "id": "x", "section": "S", "text": "a check wired later",
            "status": "pass", "detail": "2"}
        assert res["for_you_to_judge"] == []
    finally:
        CL.CHECKS.pop("custom_hook", None)


def test_a_check_that_breaks_is_judged():
    data = {"title": "t", "status": "DRAFT", "items": [
        {"id": "x", "section": "S", "mode": "code", "check": "boom",
         "text": "a check with a bug"}]}
    CL.register_check("boom", lambda ctx, item: 1 / 0)
    try:
        res = CL.run_checklist(_record(), data)
    finally:
        CL.CHECKS.pop("boom", None)
    assert res["code_checks"][0]["status"] == "not_run"
    assert "ZeroDivisionError" in res["for_you_to_judge"][0]["code_said"]


# ---------------------------------------------------------------------------
# The record by path, and the tool
# ---------------------------------------------------------------------------

def test_the_record_is_read_by_its_file_or_its_folder(tmp_path):
    rec = _record()
    path = tmp_path / CL.RECORD_NAME
    path.write_text(rec.model_dump_json(), encoding="utf-8")
    by_file = CL.run_checklist(str(path))
    by_folder = CL.run_checklist(str(tmp_path))
    assert by_file["code_checks"] == by_folder["code_checks"]
    assert by_folder["record"] == str(path)
    with pytest.raises(ValueError, match="no record"):
        CL.run_checklist(str(tmp_path / "missing"))


def test_the_record_name_is_the_writers():
    from report_ingest.writers import RECORD_NAME
    assert CL.RECORD_NAME == RECORD_NAME


def test_the_tool(tmp_path):
    pytest.importorskip("langchain_core")
    rec = _record(lab_tests=[atterberg(), summary_table(ll_for_s1=41.0)])
    (tmp_path / CL.RECORD_NAME).write_text(rec.model_dump_json(),
                                           encoding="utf-8")
    tool = CL.make_checklist_tool()
    assert tool.name == "report_checklist"
    assert set(tool.args) == {"record"}
    out = json.loads(tool.invoke({"record": str(tmp_path)}))
    assert out["status"] == "DRAFT" and out["for_you_to_judge"]
    assert _by_id(out)["consistency.summary_vs_sheets"]["status"] == "fail"
    blind = json.loads(tool.invoke({"record": ""}))
    assert {r["status"] for r in blind["code_checks"]} == {"not_run"}
    err = json.loads(tool.invoke({"record": str(tmp_path / "nope")}))
    assert "no record at" in err["error"]
    capped = CL.make_checklist_tool(900).invoke({"record": str(tmp_path)})
    assert len(capped) <= 900 and '"entries"' not in capped


def test_the_tool_names_the_ingest_tool_only_where_the_agent_has_it():
    """No pointer at a tool the agent does not have: the geotech page
    offers report_ingest only behind enable_report_ingest (OFF)."""
    pytest.importorskip("langchain_core")
    with_ingest = CL.make_checklist_tool().description
    alone = CL.make_checklist_tool(ingest_tool=False).description
    assert "report_ingest" in with_ingest and "paths.record" in with_ingest
    assert "report_ingest" not in alone and "paths.record" not in alone
    assert CL.RECORD_NAME in alone
    # and nothing the model sees without the tool points at it either
    assert "report_ingest" not in json.dumps(CL.run_checklist(None))
    # tracing shorthand an engineer does not read
    assert "reconciler" not in with_ingest


def test_the_old_ledger_call_says_what_the_checklist_reads_now():
    """Before RI-34 a caller passed (ledger, document key)."""
    with pytest.raises(TypeError, match="record"):
        CL.run_checklist(object(), "doc-key")
    with pytest.raises(TypeError, match="record"):
        CL.failed_lines(object(), "doc-key")
