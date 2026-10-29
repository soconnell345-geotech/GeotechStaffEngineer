"""A model-first page group that was never read is a failed item (BACKLOG
RI-187).

Build f's MF-2 on arm left 45 content pages in 9 groups unread -- cut off
at the output cap, or refused as no structured answer -- while its closing
failures line said "ingest none": the groups' ``skipped`` entries did not
say "could not be read", the words a run file's ``item_errors`` are read
by (RI-110), and most of one report's were in its bound document's record.
No model is called; every page and name here is made up.
"""

from __future__ import annotations

from report_ingest import cluster_scoring as cs
from report_ingest import model_first_read as mfr
from report_ingest.model import ParentReport, QAEntry, ReportRecord
from report_ingest.reconciler import _carry_unresolved


def _record(*rows):
    qa = []
    _carry_unresolved(qa, list(rows))
    return ReportRecord(qa=qa)


def test_an_unread_group_says_it_could_not_be_read():
    row = mfr._skipped("lab", [9, 10], "the reply stopped at its output "
                                       "cap, and one page cannot be split")
    assert row["refused_by"] == "budget" and row["pages"] == [9, 10]
    assert row["why"].endswith("pages 9-10 could not be read")


def test_it_is_an_item_error_of_the_report():
    record = _record(mfr._skipped("calc", [20, 21, 22], "the call gave no "
                                  "structured answer (made up)"))
    (error,) = cs._item_error_fields(record)["item_errors"]
    assert error["where"] == "calc pages 20-22"
    assert error["pages"] == [20, 21, 22]
    assert "could not be read" in error["detail"]


def test_and_of_the_report_it_is_bound_in():
    parent = ReportRecord()
    child = _record(mfr._skipped("lab", [40], "made up"))
    child.parent = ParentReport(report_id="R00", bound_id="bound1")
    (error,) = cs._item_error_fields(parent, [child])["item_errors"]
    assert error["where"] == "bound1: lab page 40"


def test_a_report_with_one_is_not_finished_and_a_resume_retries_it():
    blob = {"id": "R00", **cs._item_error_fields(
        _record(mfr._skipped("logs", [3, 4], "made up")))}
    assert cs._unfinished_reason(blob) == (
        "1 item(s) could not be read; a resume retries them")


def test_other_skipped_entries_are_not_item_errors():
    record = ReportRecord(qa=[QAEntry(
        kind="skipped", where="pages", detail="labelled made-up pages "
        "belong to no work item, so nothing read them", pages=[5])])
    assert cs._item_error_fields(record) == {}
