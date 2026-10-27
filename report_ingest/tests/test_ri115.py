"""RI-115: a sheet that prints its hole only inside the specimen label, its
boring left blank, and that the reconciler links, is read in the linked hole
by every reader that places a test by its hole: the lab scorer, the DIGGS
writer, the summary-table cross-check and the checklist that shares it. The
printed field stays as printed, and a sheet nothing linked stays unlinked.

Made-up values only: no hole, sample or number here is from a report.
"""

from __future__ import annotations

from report_ingest import checklist as CL
from report_ingest.diggs_writer import (
    diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
)
from report_ingest.lab_scoring import score_record
from report_ingest.model import (
    ChemicalResult, DocumentFacts, Investigation, LabTest, Layer, Project,
    Quantity, ReportRecord, Sample, SummaryRow, SummaryTableResult, lab_hole,
)
from report_ingest.reconciler import _sheets_for, reconcile

DEPTH = 1.65


def _m(value):
    return Quantity(value=value, unit="m")


def _ppm(value):
    return Quantity(value=value, unit="ppm")


def _hole(name, samples):
    return Investigation(
        investigation_id=name, kind="boring", depth_unit="m",
        total_depth=_m(12.6),
        layers=[Layer(top=_m(0.0), bottom=_m(12.6),
                      description="Grey CLAYEY SILT", uscs="ML")],
        samples=[Sample(sample_id=sid, top=_m(top), bottom=_m(top + 0.45),
                        kind="spt") for sid, top in samples],
        pages=[4], source_report="SYN")


def _logs():
    return [_hole("TH-7", [("P-31", DEPTH), ("P-32", 3.85)]),
            _hole("TH-12", [("P-17", DEPTH)])]


def _chem(label, boring="", ph=6.27, page=30):
    return LabTest(
        kind="chemical", investigation_id=boring, sample_id=label,
        depth_top=_m(DEPTH), pages=[page], source_report="SYN",
        result=ChemicalResult(pH=ph, chloride=_ppm(41.5),
                              sulfate=_ppm(263.0)))


def _record(*lab):
    return ReportRecord(document=DocumentFacts(report_id="SYN", n_pages=40),
                        project=Project(name="Made-up Depot"),
                        investigations=_logs(), lab_tests=list(lab))


def _truth(hole):
    return {"id": "chemical__R99_p31", "kind": "chemical", "depth_unit": "m",
            "tests": [{"investigation_id": hole, "depth_top": DEPTH,
                       "pH": 6.27, "chloride_ppm": 41.5,
                       "sulfate_ppm": 263}]}


def _summary(*rows):
    return LabTest(kind="summary_table", pages=[28], source_report="SYN",
                   result=SummaryTableResult(rows=list(rows)))


def _row(hole, ph):
    return SummaryRow(investigation_id=hole, sample_id="P-31",
                      depth_top=_m(DEPTH), pH=ph)


class TestTheScorer:

    def test_a_label_linked_sheet_scores_in_its_linked_hole(self):
        sheet = _chem("TH-7 1.65-2.1 m")
        record = reconcile(_record(sheet))

        assert sheet.investigation_id == ""            # as printed: blank
        assert sheet.linked_investigation_id == "TH-7"
        assert lab_hole(sheet) == "TH-7"
        score = score_record(_truth("TH-7"), record.lab_tests)
        assert (score.scores["link"].found, score.scores["link"].total) == \
            (1, 1)
        assert score.scores["index"].found == score.scores["index"].total == 3
        assert score.total.found == score.total.total

    def test_the_readers_own_answer_is_scored_on_the_printed_boring(self):
        # Nothing has linked a reader's answer yet: the scorer reads what it
        # printed, so a blank boring is no hole, as before RI-115.
        score = score_record(_truth("TH-7"), [_chem("TH-7 1.65-2.1 m")])
        assert score.scores["link"].found == 0
        assert score.scores["index"].found == 0

    def test_a_sheet_linked_to_another_hole_does_not_score_here(self):
        sheet = _chem("TH-12 1.65 m")
        record = reconcile(_record(sheet))

        assert sheet.linked_investigation_id == "TH-12"
        score = score_record(_truth("TH-7"), record.lab_tests)
        assert score.scores["link"].found == 0
        assert score.scores["index"].found == 0

    def test_a_sheet_with_no_link_stays_unlinked(self):
        sheet = _chem("QX-3 1.65 m")             # QX-3 is on no log
        record = reconcile(_record(sheet))

        assert (sheet.investigation_id, sheet.linked_investigation_id,
                lab_hole(sheet)) == ("", "", "")
        assert any("names no exploration" in e.detail for e in record.qa)
        score = score_record(_truth("QX-3"), record.lab_tests)
        assert score.scores["link"].found == 0
        assert score.scores["index"].found == 0

    def test_a_printed_boring_nothing_linked_is_still_its_hole(self):
        sheet = _chem("K-5", boring="QX-3")
        record = reconcile(_record(sheet))

        assert sheet.linked_investigation_id == ""
        assert lab_hole(sheet) == "QX-3"
        score = score_record(_truth("QX-3"), record.lab_tests)
        assert score.scores["link"].found == 1
        assert score.scores["index"].found == 3


class TestTheDiggsFile:

    def test_the_sheet_is_written_in_the_linked_hole(self):
        from subsurface_characterization import parse_diggs

        record = reconcile(_record(_chem("TH-7 1.65-2.1 m")))
        notes = []
        xml = write_diggs(record, notes=notes)

        assert notes[0].synthesised == []
        assert notes[0].lab_tests == 1
        ok, errors = diggs_schema_gate(xml)
        assert ok, errors[:2]
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs
        by_id = {inv.investigation_id: inv
                 for inv in parse_diggs(content=xml).site.investigations}
        assert "pH" in {m.parameter for m in by_id["TH-7"].measurements}
        assert "pH" not in {m.parameter for m in by_id["TH-12"].measurements}
        assert record.lab_tests[0].investigation_id == ""   # untouched

    def test_a_sheet_with_no_link_is_not_written(self):
        record = reconcile(_record(_chem("QX-3 1.65 m")))
        notes = []
        write_diggs(record, notes=notes)

        assert notes[0].lab_tests == 0
        assert notes[0].synthesised == []
        assert any("names no boring" in s for s in notes[0].skipped)


class TestTheSummaryTable:

    def test_a_row_meets_the_label_linked_sheet_of_its_hole(self):
        sheet = _chem("TH-7 1.65-2.1 m", ph=6.27)
        other = _chem("TH-12 1.65 m", ph=8.29, page=31)
        row = _row("TH-7", ph=7.63)
        record = reconcile(_record(sheet, other, _summary(row)))

        assert _sheets_for(row, [sheet, other]) == [sheet]
        (conflict,) = [e for e in record.qa if e.kind == "conflict"]
        assert conflict.where.startswith("lab_tests.pH at TH-7")
        assert conflict.values == ["summary table 7.63",
                                   "chemical sheet 6.27"]

    def test_a_row_does_not_meet_a_sheet_nothing_linked(self):
        sheet = _chem("QX-3 1.65 m")
        row = _row("TH-7", ph=7.63)
        reconcile(_record(sheet, _summary(row)))

        assert _sheets_for(row, [sheet]) == []

    def test_the_checklist_finds_the_sheet_on_the_summary(self):
        record = reconcile(_record(_chem("TH-7 1.65-2.1 m"),
                                   _summary(_row("TH-7", ph=6.27))))
        rows = {r["id"]: r for r in CL.run_checklist(record)["code_checks"]}
        sh = rows["completeness.summary_and_sheets"]

        assert sh["status"] == "pass", sh
        assert "no sheet" not in sh["detail"]
