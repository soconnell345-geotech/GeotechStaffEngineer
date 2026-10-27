"""RI-142: the checklist asks a summary-table row for a sheet of each test it
tabulates. A row joins any sheet of its own sample whatever the test (the
reconciler's match allows it, deliberately: RI-140), so a row tabulating
limits and a grading whose sample had only a chemistry sheet read as having
its sheet. Now each test the row's columns name (``reconciler.COLUMN_KINDS``
and the ``other`` headings' keywords) needs a joined sheet that reports it:
one of the kinds the column names, or one carrying the column's value. A row
naming no test keeps the old rule (any joined sheet); a test only a tied
sheet reports is the tie's, not a gap; a water content or a density no sheet
reports is a note (labs often report these on the summary table only).

Made-up values only: no hole, sample, depth or number here is from a report.
"""

from __future__ import annotations

from report_ingest import checklist as CL
from report_ingest.model import (
    AtterbergResult, CBRResult, ChemicalResult, DocumentFacts,
    GradationResult, Investigation, LabTest, Layer, Project, Quantity,
    ReportRecord, Sample, StrengthResult, SummaryRow, SummaryTableResult,
)
from report_ingest.reconciler import (
    COLUMN_KINDS, _kinds_named, _summary_matches, _tests_named, reconcile,
)


def _m(value):
    return Quantity(value=value, unit="m")


def _row(hole="QB-4", sample="T-3", top=2.40, **cells):
    return SummaryRow(investigation_id=hole, sample_id=sample,
                      depth_top=_m(top) if top is not None else None,
                      **cells)


def _sheet(kind, result, sample="T-3", top=2.40, page=30, hole="QB-4"):
    return LabTest(kind=kind, investigation_id=hole, sample_id=sample,
                   depth_top=_m(top), pages=[page], source_report="SYN",
                   result=result)


def _atterberg(sample="T-3", top=2.40, page=21, **values):
    return _sheet("atterberg", AtterbergResult(
        **(values or {"ll": 44.0, "pl": 21.0, "pi": 23.0})),
        sample=sample, top=top, page=page)


def _grading(sample="T-3", top=2.40, page=24, **values):
    return _sheet("gradation", GradationResult(
        **(values or {"fines_percent": 63.0})),
        sample=sample, top=top, page=page)


def _chemistry(sample="T-3", top=2.40, page=30):
    return _sheet("chemical", ChemicalResult(pH=6.6), sample=sample,
                  top=top, page=page)


def _summary(*rows):
    return LabTest(kind="summary_table", pages=[19], source_report="SYN",
                   result=SummaryTableResult(rows=list(rows)))


def _elsewhere(*kinds):
    """A sheet of each kind for QB-9, deep down and on no summary row: the
    record holds sheets of these kinds, so a row with none of its own has
    a gap, not a note (RI-143: a kind the record holds no sheet of at all
    is a note)."""
    results = {
        "atterberg": AtterbergResult(ll=38.0, pl=19.0, pi=19.0),
        "gradation": GradationResult(fines_percent=41.0),
        "cbr": CBRResult(cbr_percent=6.0),
        "unconfined": StrengthResult(kind="unconfined",
                                     qu=Quantity(value=95.0, unit="kPa"))}
    return [_sheet(kind, results[kind], sample="T-9", top=9.0, page=36 + i,
                   hole="QB-9") for i, kind in enumerate(kinds)]


def _record(*lab):
    """QB-4 and QB-9 logged, their one sample each far below the depths
    tested here, so nothing links a sheet to a log sample."""
    holes = [Investigation(
        investigation_id=name, kind="boring", depth_unit="m",
        total_depth=_m(15.0),
        layers=[Layer(top=_m(0.0), bottom=_m(15.0),
                      description="Brown sandy CLAY", uscs="CL")],
        samples=[Sample(sample_id="T-20", top=_m(11.0), bottom=_m(11.45),
                        kind="spt")],
        pages=[5], source_report="SYN") for name in ("QB-4", "QB-9")]
    return ReportRecord(document=DocumentFacts(report_id="SYN", n_pages=40),
                        project=Project(name="Made-up Yard"),
                        investigations=holes, lab_tests=list(lab))


def _check(*lab):
    record = reconcile(_record(*lab))
    rows = {r["id"]: r for r in CL.run_checklist(record)["code_checks"]}
    return rows["completeness.summary_and_sheets"]


LIMITS_AND_GRADING = {"ll": 44.0, "pl": 21.0, "pi": 23.0,
                      "fines_percent": 63.0}


class TestEachTestNeedsItsSheet:

    def test_a_chemistry_sheet_of_its_sample_is_not_its_limits_or_grading(
            self):
        row = _row(**LIMITS_AND_GRADING)
        chem = _chemistry()
        # The join is RI-140's, unchanged: the same sample joins them.
        assert _summary_matches([row], [chem]) == ([[chem]], [])

        sh = _check(chem, *_elsewhere("atterberg", "gradation"),
                    _summary(row))
        assert sh["status"] == "fail", sh
        assert sh["detail"].startswith(
            "1 summary row(s) have no sheet for a test they tabulate (0 of "
            "them with no sheet joined at all): 1 with no atterberg sheet, "
            "1 with no gradation sheet")
        assert ("summary rows short of a sheet: QB-4 T-3 at 2.4 m (no "
                "atterberg sheet, no gradation sheet)") in sh["detail"]

    def test_a_row_with_its_limits_sheet_lacks_only_its_grading(self):
        sh = _check(_atterberg(), *_elsewhere("gradation"),
                    _summary(_row(**LIMITS_AND_GRADING)))
        assert sh["status"] == "fail", sh
        assert "QB-4 T-3 at 2.4 m (no gradation sheet)" in sh["detail"]
        assert "no atterberg sheet" not in sh["detail"]

    def test_a_row_with_a_sheet_for_each_test_passes(self):
        sh = _check(_atterberg(), _grading(),
                    _summary(_row(**LIMITS_AND_GRADING)))
        assert sh["status"] == "pass", sh
        assert sh["detail"].startswith(
            "all 1 summary rows have a sheet for each test they tabulate")

    def test_any_kind_a_column_names_reports_it(self):
        # A qu off a UU triaxial reporting su is the qu's sheet.
        kpa = Quantity(value=118.0, unit="kPa")
        uu = _sheet("triaxial", StrengthResult(kind="triaxial",
                                               test_type="UU", su=kpa))
        assert _check(uu, _summary(_row(qu=kpa)))["status"] == "pass"
        sh = _check(_atterberg(), *_elsewhere("unconfined"),
                    _summary(_row(ll=44.0, qu=kpa)))
        assert ("QB-4 T-3 at 2.4 m (no unconfined or unconfined_rock or "
                "triaxial sheet)") in sh["detail"]

    def test_a_uscs_symbol_is_not_said_again_beside_missing_limits(self):
        # USCS is read off the limits or the grading: the grading sheet
        # reports it; with neither, the missing limits say it.
        sh = _check(_grading(), *_elsewhere("atterberg"),
                    _summary(_row(ll=44.0, uscs="CL")))
        assert "QB-4 T-3 at 2.4 m (no atterberg sheet)" in sh["detail"]
        sh = _check(_chemistry(), *_elsewhere("atterberg"),
                    _summary(_row(ll=44.0, uscs="CL")))
        assert "QB-4 T-3 at 2.4 m (no atterberg sheet)" in sh["detail"]
        assert "gradation" not in sh["detail"]

    def test_a_value_the_sheet_carries_reports_its_column(self):
        # A shear sheet printing the specimen's water content reports the
        # row's, and so does a limits sheet printing one; one printing none
        # leaves it to the note.
        shear = _sheet("direct_shear", StrengthResult(
            kind="direct_shear", phi_deg=29.0, wc=17.4))
        sh = _check(shear, _summary(_row(wc=17.4, phi_deg=29.0)))
        assert sh["status"] == "pass", sh
        assert "water content" not in sh["detail"]
        sh = _check(_atterberg(ll=44.0, water_content=17.4),
                    _summary(_row(ll=44.0, wc=17.4)))
        assert sh["status"] == "pass", sh
        assert "water content" not in sh["detail"]
        sh = _check(_atterberg(), _summary(_row(ll=44.0, wc=17.4)))
        assert sh["status"] == "pass", sh
        assert "1 summary row(s) tabulate water content with no sheet" in \
            sh["detail"]

    def test_an_other_heading_names_its_test(self):
        sh = _check(_atterberg(), *_elsewhere("cbr"),
                    _summary(_row(ll=44.0, other=[("CBR (%)", "9")])))
        assert "QB-4 T-3 at 2.4 m (no cbr sheet)" in sh["detail"]
        cbr = _sheet("cbr", CBRResult(cbr_percent=9.0), page=33)
        assert _check(_atterberg(), cbr, _summary(
            _row(ll=44.0, other=[("CBR (%)", "9")])))["status"] == "pass"


class TestWaterContentAndDensity:
    """Labs often report a water content and a density (unit weight) on the
    summary table only: one no sheet reports is a note, never a gap."""

    def test_neither_fails_nor_is_unsure(self):
        kn = Quantity(value=17.9, unit="kN/m3")
        sh = _check(_atterberg(), _summary(_row(ll=44.0, wc=16.2,
                                                dry_density=kn)))
        assert sh["status"] == "pass", sh
        assert sh["detail"].startswith(
            "all 1 summary rows have a sheet for each test they tabulate "
            "other than water content and density")
        assert ("1 summary row(s) tabulate water content or density with no "
                "sheet reporting it (water content 1, density 1): labs often "
                "report these on the summary table only, so they are not "
                "counted as missing") in sh["detail"]
        assert "have no sheet for a test" not in sh["detail"]

    def test_a_row_tabulating_only_them_with_no_sheet_passes(self):
        kn = Quantity(value=18.6, unit="kN/m3")
        sh = _check(_summary(_row(wc=21.0), _row(sample="T-4", top=3.10,
                                                 wet_density=kn)))
        assert sh["status"] == "pass", sh
        assert ("2 summary row(s) tabulate water content or density with no "
                "sheet reporting it (water content 1, density 1)") in \
            sh["detail"]

    def test_a_gap_beside_them_counts_alone(self):
        sh = _check(_atterberg(), *_elsewhere("gradation"),
                    _summary(_row(wc=16.2, **LIMITS_AND_GRADING)))
        assert sh["status"] == "fail", sh
        assert sh["detail"].startswith(
            "1 summary row(s) have no sheet for a test they tabulate (0 of "
            "them with no sheet joined at all): 1 with no gradation sheet;")
        assert "QB-4 T-3 at 2.4 m (no gradation sheet)" in sh["detail"]
        assert "moisture_content" not in sh["detail"]
        assert "1 summary row(s) tabulate water content with no sheet" in \
            sh["detail"]


class TestARowNamingNoTest:

    def test_its_exact_interval_sheet_is_its_sheet(self):
        row = _row(sample="", description="Grey silty CLAY")
        assert _check(_chemistry(sample=""), _summary(row))["status"] == \
            "pass"

    def test_with_none_it_is_listed_as_before(self):
        row = _row(sample="", description="Grey silty CLAY")
        sh = _check(_chemistry(sample="", top=5.10), _summary(row))
        assert sh["status"] == "fail", sh
        assert ("summary rows naming no test, with no sheet: QB-4 at 2.4 m"
                in sh["detail"])
        assert "short of a sheet" not in sh["detail"]


class TestTies:

    def test_a_test_only_tied_sheets_report_is_the_ties(self):
        # The grading sheet is the row's own; two specimens' limits sheets
        # are as close: unsure, not a gap.
        row = _row(sample="", **LIMITS_AND_GRADING)
        sh = _check(_grading(sample=""),
                    _atterberg(sample="T-5", top=2.30),
                    _atterberg(sample="T-6", top=2.50, page=22),
                    _summary(row))
        assert sh["status"] == "unsure", sh
        assert "no atterberg sheet" not in sh["detail"]
        assert "(2 sheets: atterberg)" in sh["detail"]

    def test_a_real_gap_beside_a_tie_fails(self):
        row = _row(sample="", other=[("CBR", "9")], **LIMITS_AND_GRADING)
        sh = _check(_grading(sample=""),
                    _atterberg(sample="T-5", top=2.30),
                    _atterberg(sample="T-6", top=2.50, page=22),
                    *_elsewhere("cbr"), _summary(row))
        assert sh["status"] == "fail", sh
        assert "QB-4 at 2.4 m (no cbr sheet)" in sh["detail"]
        assert "no atterberg sheet" not in sh["detail"]
        assert "(2 sheets: atterberg)" in sh["detail"]


class TestTheNote:

    def test_counts_by_kind_and_rows_with_a_sheet_listed_first(self):
        # Eighteen rows with no sheet at all, then one whose limits sheet
        # joined it and whose grading sheet is missing.
        bare = [_row(sample=f"T-{n}", top=1.0 + n, ll=40.0 + n)
                for n in range(30, 48)]
        partial = _row(sample="T-3", **LIMITS_AND_GRADING)
        sh = _check(_atterberg(), *_elsewhere("gradation"),
                    _summary(*bare, partial))

        assert sh["status"] == "fail", sh
        assert sh["detail"].startswith(
            "19 summary row(s) have no sheet for a test they tabulate (18 of "
            "them with no sheet joined at all): 18 with no atterberg sheet, "
            "1 with no gradation sheet")
        listed = sh["detail"].split("summary rows short of a sheet: ")[1]
        assert listed.startswith("QB-4 T-3 at 2.4 m (no gradation sheet), "
                                 "QB-4 T-30 at 31 m (no atterberg sheet)")
        assert "(and 4 more)" in listed


class TestTheColumnsMap:

    def test_the_tests_named_are_the_kinds_named(self):
        row = _row(uscs="SC", ll=31.0, wc=12.5, phi_deg=33.0,
                   dry_density=Quantity(value=17.1, unit="kN/m3"), pH=7.4,
                   other=[("Gs", "2.66"), ("Remarks", "x")])
        tests = _tests_named(row)
        assert {k for _c, kinds in tests for k in kinds} == _kinds_named(row)
        assert [c for c, _k in tests] == ["ll", "wc", "uscs", "dry_density",
                                          "phi_deg", "pH", "Gs"]
        assert ("uscs", COLUMN_KINDS["uscs"]) in tests
        assert ("Gs", ("specific_gravity",)) in tests
        assert _tests_named(_row(pH="not tested", other=[("CBR", "--")])) \
            == []
