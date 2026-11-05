"""Laboratory tests on combined samples (BACKLOG RI-107).

A test run on two or more samples combined to get enough material is
printed as "S-3 & S-4", "S3+S4", "composite of S-3 and S-4", or as a depth
range spanning two samples. It links to EVERY sample it combined, flagged as
combined, rather than to one or none -- by the label (including a hole read
out of the label, RI-07) and by the depth. The report sheets are private, so
these are made-up sheets covering each printed form; every hole, sample and
depth here is invented.
"""

from __future__ import annotations

import pytest

from report_ingest.model import (
    AtterbergResult, GradationResult, Investigation, LabTest, Quantity,
    ReportRecord, Sample, SummaryRow, SummaryTableResult,
)
from report_ingest.reconciler import reconcile


def _m(v: float) -> Quantity:
    return Quantity(value=v, unit="m")


def _log(samples=None, hole="QB-3") -> Investigation:
    """A made-up boring with three driven samples and a bulk bag."""
    samples = samples or [
        Sample(sample_id="S-2", top=_m(1.5), bottom=_m(1.95), kind="spt"),
        Sample(sample_id="S-3", top=_m(3.0), bottom=_m(3.45), kind="spt"),
        Sample(sample_id="S-4", top=_m(4.5), bottom=_m(4.95), kind="spt"),
        Sample(sample_id="S-5", top=_m(6.0), bottom=_m(6.45), kind="spt"),
    ]
    return Investigation(investigation_id=hole, depth_unit="m",
                         total_depth=_m(9.0), samples=samples, pages=[4])


def _sheet(sample_id="", hole="QB-3", top=None, bottom=None,
           kind="gradation") -> LabTest:
    result = GradationResult() if kind == "gradation" else AtterbergResult(
        ll=41.0, pl=20.0)
    return LabTest(kind=kind, investigation_id=hole, sample_id=sample_id,
                   depth_top=None if top is None else _m(top),
                   depth_bottom=None if bottom is None else _m(bottom),
                   pages=[12], result=result)


def _linked(test: LabTest, log=None) -> LabTest:
    record = ReportRecord(investigations=[log or _log()], lab_tests=[test])
    reconcile(record)
    return record.lab_tests[0]


# ---------------------------------------------------------------------------
# each printed form
# ---------------------------------------------------------------------------

class TestByTheLabel:
    @pytest.mark.parametrize("label", [
        "S-3 & S-4", "S3+S4", "S-3 + S-4", "S-3 and S-4",
        "Composite of S-3 and S-4", "S-3/S-4 comp.", "S-3, S-4 combined",
    ])
    def test_a_label_joining_two_samples_links_both(self, label):
        test = _linked(_sheet(label, top=3.0))
        assert test.combined is True
        assert test.linked_investigation_id == "QB-3"
        assert test.linked_sample_ids == ["S-3", "S-4"]
        assert test.linked_sample_id == "S-3"

    def test_no_depth_printed_still_links_both_when_the_label_says(self):
        test = _linked(_sheet("S-4 & S-3"))
        assert test.linked_sample_ids == ["S-3", "S-4"]   # shallowest first

    def test_a_range_of_names_combined(self):
        test = _linked(_sheet("Composite S-2 to S-4", top=1.5, bottom=4.95))
        assert test.linked_sample_ids == ["S-2", "S-3", "S-4"]

    def test_two_names_with_no_combine_word_and_a_depth_meeting_both(self):
        test = _linked(_sheet("S-3, S-4", top=3.0, bottom=4.95))
        assert test.combined and test.linked_sample_ids == ["S-3", "S-4"]

    def test_the_hole_read_out_of_the_label_too(self):
        # RI-07: the sheet prints the hole only inside the specimen label.
        test = _linked(_sheet("QB-3 S-3 & S-4", hole=""))
        assert test.linked_investigation_id == "QB-3"
        assert test.linked_sample_ids == ["S-3", "S-4"]

    def test_the_record_says_so_once(self):
        record = ReportRecord(investigations=[_log()], lab_tests=[
            _sheet("S-3 & S-4", top=3.0), _sheet("S-2"),
            _sheet("S3+S5", kind="atterberg")])
        reconcile(record)
        (note,) = [e for e in record.qa if e.where == "lab_tests.combined"]
        assert note.kind == "note" and note.pages == [12]
        assert "2 laboratory result(s)" in note.detail
        assert any("QB-3 S-3 + S-4" in v for v in note.values)
        assert any("QB-3 S-3 + S-5" in v for v in note.values)


class TestByTheDepth:
    def test_a_range_spanning_two_samples_links_both(self):
        test = _linked(_sheet(top=3.0, bottom=4.95))
        assert test.combined and test.linked_sample_ids == ["S-3", "S-4"]
        assert test.linked_depth_delta_m == 0.0

    def test_a_range_held_by_one_sample_is_that_sample(self):
        log = _log([Sample(sample_id="BAG-1", top=_m(0.0), bottom=_m(1.5),
                           kind="bulk"),
                    Sample(sample_id="S-1", top=_m(0.6), bottom=_m(1.05),
                           kind="spt")])
        test = _linked(_sheet(top=0.0, bottom=1.5), log)
        assert not test.combined and test.linked_sample_ids == []
        assert test.linked_sample_id == "BAG-1"

    def test_a_short_interval_is_one_sample(self):
        test = _linked(_sheet(top=3.0, bottom=3.45))
        assert not test.combined and test.linked_sample_id == "S-3"


class TestOneSampleIsUnchanged:
    def test_a_label_naming_one_sample(self):
        test = _linked(_sheet("S-3", top=3.0))
        assert not test.combined and test.linked_sample_id == "S-3"
        assert "combined" not in test.model_dump()
        assert "linked_sample_ids" not in test.model_dump()

    def test_two_names_no_combine_word_and_a_depth_meeting_one(self):
        # RI-71 as before: the sample the printed depth meets.
        test = _linked(_sheet("S-3, S-4", top=3.0))
        assert not test.combined and test.linked_sample_id == "S-3"

    def test_reconciled_twice_a_sheet_that_changed_loses_the_flag(self):
        test = _sheet("S-3 & S-4", top=3.0)
        record = ReportRecord(investigations=[_log()], lab_tests=[test])
        reconcile(record)
        assert record.lab_tests[0].combined
        record.lab_tests[0].sample_id = "S-3"
        reconcile(record)
        assert not record.lab_tests[0].combined
        assert record.lab_tests[0].linked_sample_ids == []


class TestSummaryRows:
    def test_a_row_on_combined_samples_links_both(self):
        table = LabTest(kind="summary_table", pages=[20],
                        result=SummaryTableResult(rows=[
                            SummaryRow(investigation_id="QB-3",
                                       sample_id="S-3 & S-4",
                                       depth_top=_m(3.0)),
                            SummaryRow(investigation_id="QB-3",
                                       depth_top=_m(4.5),
                                       depth_bottom=_m(6.45)),
                            SummaryRow(investigation_id="QB-3",
                                       sample_id="S-2")]))
        record = ReportRecord(investigations=[_log()], lab_tests=[table])
        reconcile(record)
        rows = record.lab_tests[0].result.rows
        assert rows[0].combined and rows[0].linked_sample_ids == [
            "S-3", "S-4"]
        assert rows[1].combined and rows[1].linked_sample_ids == [
            "S-4", "S-5"]
        assert not rows[2].combined and rows[2].linked_sample_id == "S-2"
        assert "combined" not in rows[2].model_dump()


class TestWhatReadsIt:
    def test_the_library_page_names_every_sample(self):
        from report_ingest.writers import library_page

        record = ReportRecord(investigations=[_log()],
                              lab_tests=[_sheet("S-3 & S-4", top=3.0)])
        reconcile(record)
        assert "S-3 + S-4 (combined)" in library_page(record, "R00")

    def test_the_checklist_counts_it_linked(self):
        from report_ingest.checklist import CheckContext, check_lab_ids_vs_logs

        record = ReportRecord(investigations=[_log()],
                              lab_tests=[_sheet("S3+S4", top=3.0)])
        reconcile(record)
        got = check_lab_ids_vs_logs(CheckContext(record=record), {})
        assert got["status"] == "pass"

    def test_the_record_round_trips(self):
        test = _linked(_sheet("S-3 & S-4", top=3.0))
        back = LabTest.model_validate(test.model_dump())
        assert back.combined and back.linked_sample_ids == ["S-3", "S-4"]
