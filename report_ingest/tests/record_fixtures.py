"""A synthetic :class:`~report_ingest.model.ReportRecord`, built by hand.

The reconciler and the writers take a record, not a PDF, so they are tested
against one written out here: two borings with layers, samples, driven records
and water levels; four laboratory tests including a summary table; and the
owner's two schemas answered. Every number is invented and every name is made
up -- what matters is that the shapes are the shapes a real reading produces,
including the awkward ones a test needs: a laboratory sheet naming a hole the
report does not carry, a summary-table row that disagrees with the sheet
beside it, and a value printed in a unit the record cannot convert.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from report_ingest.model import (
    AtterbergResult, BearingValue, Citation, DocumentFacts, GeneralFacts,
    GradationResult, Investigation, LabTest, Layer, MoistureDensityResult,
    NarrativeFacts, NaturalHazardFacts, Project, Provenance, Quantity,
    ReportRecord, SPT, Sample, SievePoint, Stratum, SummaryRow,
    SummaryTableResult, WaterLevel,
)


def q(value: float, unit: str, page: int = 7) -> Quantity:
    return Quantity(value=value, unit=unit,
                    prov=Provenance(page=page, method="grid"))


def boring(name: str = "B-1", page: int = 7) -> Investigation:
    """One boring with two layers, three samples, three drives and water."""
    return Investigation(
        investigation_id=name,
        kind="boring",
        depth_unit="ft",
        elevation=q(102.5, "ft", page),
        total_depth=q(30.0, "ft", page),
        date_started="2026-02-03",
        layers=[
            Layer(top=q(0.0, "ft", page), bottom=q(8.0, "ft", page),
                  description="Brown sandy lean CLAY, stiff, moist",
                  uscs="CL"),
            Layer(top=q(8.0, "ft", page), bottom=q(30.0, "ft", page),
                  description="Grey silty SAND, medium dense", uscs="SM"),
        ],
        samples=[
            Sample(sample_id="S-1", top=q(2.0, "ft", page),
                   bottom=q(3.5, "ft", page), kind="spt", water_content=24.1),
            Sample(sample_id="S-2", top=q(7.0, "ft", page),
                   bottom=q(8.5, "ft", page), kind="spt"),
            Sample(sample_id="S-3", top=q(14.5, "ft", page),
                   bottom=q(16.0, "ft", page), kind="spt", uscs="SM"),
        ],
        spt=[
            SPT(depth_top=q(2.0, "ft", page), blows=[3, 5, 7], n=12,
                sample_id="S-1"),
            SPT(depth_top=q(7.0, "ft", page), blows=[4, 6, 9], n=15,
                sample_id="S-2"),
            SPT(depth_top=q(14.5, "ft", page), blows=[8, 12, 14], n=26,
                sample_id="S-3"),
        ],
        water=[WaterLevel(depth=q(12.0, "ft", page), when="while_drilling")],
        pages=[page], source_report="SYN")


def pit(name: str = "TP-1", page: int = 9) -> Investigation:
    return Investigation(
        investigation_id=name, kind="test_pit", depth_unit="ft",
        total_depth=q(9.0, "ft", page),
        layers=[Layer(top=q(0.0, "ft", page), bottom=q(9.0, "ft", page),
                      description="Silty SAND fill", uscs="SM")],
        samples=[Sample(sample_id="B-1", top=q(4.0, "ft", page), kind="bulk")],
        pages=[page], source_report="SYN")


def atterberg(investigation: str = "B-1", sample: str = "S-1",
              depth_ft: float = 2.0, ll: float = 38.0, pl: float = 19.0,
              page: int = 12) -> LabTest:
    return LabTest(
        kind="atterberg", investigation_id=investigation, sample_id=sample,
        depth_top=q(depth_ft, "ft", page), standard="ASTM D4318",
        pages=[page], source_report="SYN",
        result=AtterbergResult(kind="atterberg", ll=ll, pl=pl, pi=ll - pl,
                               water_content=24.1))


def gradation(page: int = 13) -> LabTest:
    return LabTest(
        kind="gradation", investigation_id="B-1", sample_id="S-3",
        depth_top=q(14.5, "ft", page), standard="ASTM D6913",
        pages=[page], source_report="SYN",
        result=GradationResult(
            kind="gradation", fines_percent=54.0, sand_percent=41.0,
            gravel_percent=5.0,
            percent_passing=[
                SievePoint(sieve="No. 4", size=Quantity(value=4.75, unit="mm"),
                           percent_passing=98.0),
                SievePoint(sieve="No. 200",
                           size=Quantity(value=0.075, unit="mm"),
                           percent_passing=54.0)]))


def summary_table(ll_for_s1: float = 38.0, page: int = 14) -> LabTest:
    """The report's own summary of its laboratory testing."""
    return LabTest(
        kind="summary_table", pages=[page], source_report="SYN",
        result=SummaryTableResult(
            kind="summary_table",
            title="Summary of Laboratory Test Results",
            rows=[
                SummaryRow(investigation_id="B-1", sample_id="S-1",
                           depth_top=q(2.0, "ft", page), ll=ll_for_s1,
                           pl=19.0, pi=ll_for_s1 - 19.0, wc=24.1),
                SummaryRow(investigation_id="B-1", sample_id="S-3",
                           depth_top=q(14.5, "ft", page), fines_percent=54.0,
                           uscs="SM"),
            ]))


def orphan_sheet(page: int = 15) -> LabTest:
    """A sheet naming a hole that is not in this report."""
    return LabTest(
        kind="moisture_content", investigation_id="B-9", sample_id="S-1",
        depth_top=q(5.0, "ft", page), pages=[page], source_report="SYN",
        result=MoistureDensityResult(kind="moisture_content", wc=18.2))


def general_facts() -> GeneralFacts:
    return GeneralFacts(
        documentType="geotechnical report",
        quickSummary="A geotechnical investigation for a residential "
                     "development, recommending spread footings.",
        projectName="Rosewood Terrace Development",
        projectNumber="26-118",
        geotechnicalEngineerFirm="Soil & Rock Consulting Engineers",
        testingProgramSummary="One boring and one test pit, with Atterberg "
                              "limits and a gradation.",
        boringCount=1, testPitCount=1, figureCount=1,
        recommendedFoundations=["spread footings"],
        boringDictionary=["B-1"],
        bearingCapacity=["3,000 psf allowable for spread footings"],
        bearingCapacityValues=[BearingValue(
            value=Quantity(value=3000.0, unit="psf"),
            foundation_type="spread footing",
            condition="net allowable on dense residual soil",
            citation=[Citation(page=4, quote="An allowable bearing pressure "
                                             "of 3,000 psf")])],
        strataList=[Stratum(name="residual soil",
                            description="dense silty sand",
                            top=Quantity(value=8.0, unit="ft"), uscs="SM")],
        citations={"boringCount": [Citation(
            page=4, quote="one boring and one test pit were completed")]})


def hazard_facts() -> NaturalHazardFacts:
    return NaturalHazardFacts(
        liquefactionPotential="low",
        siteClass="Site Class D", siteClassNormalized="D",
        seismicCodeUsed="ASCE 7-16", asceSevenVersion="ASCE 7-16",
        asceSevenVersionNormalized="7-16",
        reportDate="14 March 2026", reportDateISO="2026-03-14",
        naturalHazardSummary="Low liquefaction potential; no other hazards "
                             "identified.",
        citations={"siteClass": [Citation(page=4,
                                          quote="Site Class D")]})


def build_record(*, investigations: Optional[List[Investigation]] = None,
                 lab_tests: Optional[List[LabTest]] = None,
                 general: Optional[GeneralFacts] = None,
                 natural_hazards: Optional[NaturalHazardFacts] = None,
                 narrative: Optional[NarrativeFacts] = None,
                 document: Optional[DocumentFacts] = None,
                 calculations: Optional[List[Any]] = None
                 ) -> ReportRecord:
    """A whole record: one boring, one test pit, four laboratory tests."""
    return ReportRecord(
        document=document or DocumentFacts(
            report_id="SYN", n_pages=22, workflow="standard",
            planlens_version="test", page_roles={"narrative": 3,
                                                 "boring_log": 2},
            model_calls=6, input_tokens=12000, output_tokens=2400),
        project=Project(name="Rosewood Terrace Development", number="26-118",
                        client="Rosewood Terrace Partners"),
        general=general if general is not None else general_facts(),
        natural_hazards=(natural_hazards if natural_hazards is not None
                         else hazard_facts()),
        narrative=narrative or NarrativeFacts(
            stated_counts={"borings": 1, "test_pits": 1},
            counted={"figures": 1, "tables": 0},
            pages=[2, 3, 4]),
        investigations=(investigations if investigations is not None
                        else [boring(), pit()]),
        lab_tests=(lab_tests if lab_tests is not None
                   else [atterberg(), gradation(), summary_table()]),
        calculations=list(calculations or ()))


def page_context() -> Dict[str, Any]:
    """The page-level facts the graph hands the reconciler."""
    labels = {0: "cover", 1: "toc", 2: "narrative", 3: "narrative",
              4: "narrative", 5: "figure", 6: "divider", 7: "boring_log",
              8: "boring_log", 9: "test_pit_log", 12: "lab_test",
              13: "lab_test", 20: "calculation", 21: "calculation"}
    items = [_Item("narrative", [2, 3, 4]), _Item("boring_log", [7, 8]),
             _Item("test_pit_log", [9]), _Item("lab_test", [12]),
             _Item("lab_test", [13])]
    return {"labels": labels, "items": items, "no_text_pages": [19, 20],
            "di_pages": [19]}


class _Item:
    """The two attributes of a planlens work item the reconciler reads."""

    def __init__(self, kind: str, pages: List[int]) -> None:
        self.kind = kind
        self.pages = pages
