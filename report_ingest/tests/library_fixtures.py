"""A small synthetic LIBRARY: six reports, one of them with a report in it.

The library layer takes a folder of written records, not a PDF, so it is
tested and measured against a folder built here. Six reports is the smallest
number that makes the cross-report questions real: two firms that appear
twice, two posts that appear twice, three property phases, a report that
answered almost nothing, a report bound inside another one, and a handful of
disagreements for a reviewer to be sent to.

EVERY NAME AND NUMBER IS INVENTED. Places, firms, posts and projects are made
up for this file; the shapes are the shapes a real reading produces and
nothing here came off a real report.

The page numbers are deliberately regular across all six, because the
retrieval measurement checks the pages an answer cites: narrative on 2-4, the
logs on 7-9, the laboratory sheets on 12-14, the calculations on 20-21, and a
bound document on 15-18.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from report_ingest.model import (
    AtterbergResult, BearingValue, BoundReport, Calculation, ChemicalResult,
    Citation, ConsolidationResult, CPTData, CPTPoint, DCPData, DCPPoint,
    DocumentFacts, GeneralFacts, GradationResult, Investigation, LabTest,
    Layer, NamedQuantity, NarrativeFacts, NaturalHazardFacts, ParentReport,
    PitDimensions, Project, Provenance, QAEntry, Quantity, ReportRecord, SPT,
    Sample, SievePoint, StrengthResult, Stratum, SummaryRow,
    SummaryTableResult, WaterLevel,
)

__all__ = ["build_library", "RECORDS", "REPORT_IDS", "build_records"]

#: The file stems, which are also the report ids the queries answer with.
REPORT_IDS = ("L01", "L02", "L03", "L04", "L05", "L06")

NARRATIVE_PAGES = [2, 3, 4]
LOG_PAGE = 7
PIT_PAGE = 9
LAB_PAGES = (12, 13, 14)
CALC_PAGES = (20, 21)


def q(value: float, unit: str, page: int = LOG_PAGE) -> Quantity:
    return Quantity(value=value, unit=unit,
                    prov=Provenance(page=page, method="grid"))


def cite(page: int, quote: str) -> List[Citation]:
    return [Citation(page=page, quote=quote)]


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------

def boring(name: str, *, page: int = LOG_PAGE, depth: float = 30.0,
           n_values: Optional[List[int]] = None,
           description: str = "Brown sandy lean CLAY, stiff, moist",
           uscs: str = "CL", water: Optional[float] = 12.0,
           elevation: float = 102.5) -> Investigation:
    """One boring: two layers, three samples, three drives and water."""
    blows = n_values or [12, 15, 26]
    return Investigation(
        investigation_id=name, kind="boring", depth_unit="ft",
        elevation=q(elevation, "ft", page), total_depth=q(depth, "ft", page),
        date_started="2026-02-03",
        layers=[
            Layer(top=q(0.0, "ft", page), bottom=q(8.0, "ft", page),
                  description=description, uscs=uscs),
            Layer(top=q(8.0, "ft", page), bottom=q(depth, "ft", page),
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
        spt=[SPT(depth_top=q(2.0, "ft", page), blows=[3, 5, 7],
                 n=blows[0], sample_id="S-1"),
             SPT(depth_top=q(7.0, "ft", page), blows=[4, 6, 9],
                 n=blows[1], sample_id="S-2"),
             SPT(depth_top=q(14.5, "ft", page), blows=[8, 12, 14],
                 n=blows[2], sample_id="S-3")],
        water=([WaterLevel(depth=q(water, "ft", page), when="while_drilling")]
               if water is not None else []),
        pages=[page, page + 1])


def test_pit(name: str = "TP-1", page: int = PIT_PAGE) -> Investigation:
    return Investigation(
        investigation_id=name, kind="test_pit", depth_unit="ft",
        total_depth=q(9.0, "ft", page),
        pit=PitDimensions(width=q(0.9, "m", page), method="backhoe",
                          prov=Provenance(page=page, method="grid")),
        layers=[Layer(top=q(0.0, "ft", page), bottom=q(9.0, "ft", page),
                      description="Silty SAND fill with brick fragments",
                      uscs="SM")],
        samples=[Sample(sample_id="BK-1", top=q(4.0, "ft", page),
                        kind="bulk")],
        pages=[page])


def cone(name: str = "CPT-1", page: int = 10) -> Investigation:
    points = [CPTPoint(depth=q(float(d), "m", page),
                       qc=q(2.0 + d * 0.8, "MPa", page),
                       fs=q(0.02 + d * 0.004, "MPa", page))
              for d in range(1, 9)]
    return Investigation(
        investigation_id=name, kind="cpt", depth_unit="m",
        total_depth=q(8.0, "m", page),
        cpt=CPTData(points=points, cone_type="10 cm2 piezocone",
                    standard="ASTM D5778", vertical_axis="depth"),
        pages=[page])


def probe(name: str = "DP-1", page: int = 11) -> Investigation:
    points = [DCPPoint(depth=q(float(d) * 0.1, "m", page), blows=3 + d)
              for d in range(1, 12)]
    return Investigation(
        investigation_id=name, kind="dcp", depth_unit="m",
        total_depth=q(1.1, "m", page),
        dcp=DCPData(points=points, hammer_mass=q(10.0, "kg", page),
                    hammer_drop=q(0.5, "m", page), test_type="DPL",
                    standard="EN ISO 22476-2",
                    increment=q(0.1, "m", page)),
        pages=[page])


def atterberg(inv: str = "B-1", sample: str = "S-1", ll: float = 38.0,
              pl: float = 19.0, page: int = LAB_PAGES[0]) -> LabTest:
    return LabTest(
        kind="atterberg", investigation_id=inv, sample_id=sample,
        depth_top=q(2.0, "ft", page), standard="ASTM D4318", pages=[page],
        result=AtterbergResult(kind="atterberg", ll=ll, pl=pl, pi=ll - pl,
                               water_content=24.1))


def gradation(page: int = LAB_PAGES[1]) -> LabTest:
    return LabTest(
        kind="gradation", investigation_id="B-1", sample_id="S-3",
        depth_top=q(14.5, "ft", page), standard="ASTM D6913", pages=[page],
        result=GradationResult(
            kind="gradation", fines_percent=54.0, sand_percent=41.0,
            gravel_percent=5.0,
            percent_passing=[
                SievePoint(sieve="No. 4", size=Quantity(value=4.75, unit="mm"),
                           percent_passing=98.0),
                SievePoint(sieve="No. 200",
                           size=Quantity(value=0.075, unit="mm"),
                           percent_passing=54.0)]))


def summary_table(ll: float = 41.0, page: int = LAB_PAGES[2]) -> LabTest:
    return LabTest(
        kind="summary_table", pages=[page],
        result=SummaryTableResult(
            kind="summary_table",
            title="Summary of Laboratory Test Results",
            rows=[SummaryRow(investigation_id="B-1", sample_id="S-1",
                             depth_top=q(2.0, "ft", page), ll=ll, pl=19.0,
                             pi=ll - 19.0, wc=24.1)]))


def triaxial(page: int = LAB_PAGES[0]) -> LabTest:
    return LabTest(
        kind="triaxial", investigation_id="B-2", sample_id="U-1",
        depth_top=q(6.0, "m", page), standard="ASTM D4767", pages=[page],
        result=StrengthResult(kind="triaxial", test_type="CU",
                              c=Quantity(value=18.0, unit="kPa"),
                              phi_deg=28.5))


def consolidation(page: int = LAB_PAGES[1]) -> LabTest:
    return LabTest(
        kind="swell_consolidation", investigation_id="B-2", sample_id="U-2",
        depth_top=q(9.0, "m", page), standard="ASTM D2435", pages=[page],
        result=ConsolidationResult(kind="swell_consolidation",
                                   test_type="oedometer", cc=0.31, cr=0.042,
                                   pc=Quantity(value=185.0, unit="kPa")))


def corrosivity(page: int = LAB_PAGES[0]) -> LabTest:
    return LabTest(
        kind="chemical", investigation_id="B-5", sample_id="S-2",
        depth_top=q(3.0, "m", page), standard="ASTM D4327", pages=[page],
        result=ChemicalResult(kind="chemical", pH=4.6, sulfate=2100.0,
                              chloride=480.0, resistivity=820.0))


def calculation(kind: str, *, program: Optional[str], method: str,
                subject: str, summary: str,
                results: List[NamedQuantity],
                page: int = CALC_PAGES[0]) -> Calculation:
    return Calculation(
        kind=kind, program=program, method=method, subject=subject,
        summary=summary, results=results, pages=[page, page + 1],
        prov=[Provenance(page=page, method="model")])


def named(name: str, value: float, unit: str) -> NamedQuantity:
    return NamedQuantity(name=name, value=Quantity(value=value, unit=unit))


# ---------------------------------------------------------------------------
# the six reports
# ---------------------------------------------------------------------------

def _document(report_id: str, n_pages: int, workflow: str = "standard",
              scan: float = 0.0) -> DocumentFacts:
    return DocumentFacts(
        report_id=report_id, n_pages=n_pages, workflow=workflow,
        scan_fraction=scan, planlens_version="test", model_calls=9,
        input_tokens=18000, output_tokens=3200)


def record_l01() -> ReportRecord:
    """A design-build investigation with footings, and one conflict."""
    general = GeneralFacts(
        documentType="geotechnical report",
        quickSummary="A geotechnical investigation for a new compound, "
                     "recommending spread footings on the dense residual "
                     "soil.",
        postName="Elmridge",
        propertyType="New embassy or consulate compound",
        projectNumber="26-118", projectName="Rosewood Terrace Compound",
        projectPhase="Design-build",
        primeContractor="Harrow Build Group",
        geotechnicalEngineerFirm="Soil & Rock Consulting Engineers",
        testingProgramSummary="One boring and one test pit, with Atterberg "
                              "limits, a gradation and a summary table.",
        boringCount=1, testPitCount=1, figureCount=2, structureCount=3,
        structureList=["chancery", "warehouse", "guard booth"],
        strata="Fill over stiff lean clay over dense silty sand.",
        recommendedFoundations=["spread footings"],
        boringDictionary=["B-1"], testPitDictionary=["TP-1"],
        bearingCapacity=["3,000 psf allowable for spread footings"],
        bearingCapacityValues=[BearingValue(
            value=Quantity(value=3000.0, unit="psf"),
            foundation_type="spread footing",
            condition="net allowable on dense residual soil",
            citation=cite(4, "An allowable bearing pressure of 3,000 psf"))],
        strataList=[Stratum(name="residual soil",
                            description="dense silty sand",
                            top=Quantity(value=8.0, unit="ft"), uscs="SM",
                            citation=cite(3, "dense silty sand below 8 ft"))],
        citations={
            "boringCount": cite(4, "one boring and one test pit were "
                                   "completed"),
            "postName": cite(2, "the Elmridge compound"),
            "recommendedFoundations": cite(4, "spread footings are "
                                              "recommended"),
            "bearingCapacity": cite(4, "An allowable bearing pressure of "
                                       "3,000 psf"),
            "projectPhase": cite(2, "design-build phase services"),
            "geotechnicalEngineerFirm": cite(2, "Soil & Rock Consulting "
                                                "Engineers"),
        })
    hazards = NaturalHazardFacts(
        liquefactionPotential="not liquefiable",
        siteClass="Site Class D", siteClassNormalized="D",
        seismicCodeUsed="ASCE 7-16", asceSevenVersion="ASCE 7-16",
        asceSevenVersionNormalized="7-16",
        earthHazardsExposed=["seismic shaking", "expansive soil"],
        soilCorrosion="no - sulfate content below the reporting limit",
        geophysicalTestingMention="no",
        reportDate="14 March 2026", reportDateISO="2026-03-14",
        naturalHazardSummary="The soils are not liquefiable and no other "
                             "hazards were identified beyond expansive "
                             "near-surface clay.",
        citations={"siteClass": cite(4, "Site Class D"),
                   "liquefactionPotential": cite(4, "the soils are not "
                                                    "liquefiable"),
                   "reportDate": cite(0, "14 March 2026")})
    return ReportRecord(
        document=_document("L01", 24),
        project=Project(name="Rosewood Terrace Compound", number="26-118",
                        client="Overseas Estates Office"),
        general=general, natural_hazards=hazards,
        narrative=NarrativeFacts(stated_counts={"borings": 1, "test_pits": 1},
                                 counted={"figures": 2},
                                 pages=list(NARRATIVE_PAGES)),
        investigations=[boring("B-1"), test_pit("TP-1")],
        lab_tests=[atterberg(), gradation(), summary_table()],
        calculations=[calculation(
            "shallow_foundation_bearing", program=None, method="Meyerhof",
            subject="chancery strip footing",
            summary="Bearing capacity of a 1.2 m strip footing at 1.0 m "
                    "depth, giving 3,000 psf net allowable.",
            results=[named("qall", 3000.0, "psf"),
                     named("FS", 3.0, "")])],
        qa=[QAEntry(kind="conflict", where="lab_tests[2].rows[0].ll",
                    detail="the summary table and the Atterberg sheet give "
                           "different liquid limits for B-1 S-1",
                    values=["38", "41"], pages=[12, 14]),
            QAEntry(kind="note", where="diggs.schema",
                    detail="the DIGGS file is valid against the bundled 2.6 "
                           "schema")])


def record_l02() -> ReportRecord:
    """A bridging report recommending piles, with a report bound inside."""
    general = GeneralFacts(
        documentType="geotechnical report",
        quickSummary="A bridging geotechnical report for a waterfront "
                     "compound, recommending driven piles through the soft "
                     "marine clay.",
        postName="Vale Harbour",
        propertyType="Existing embassy or consulate compound",
        projectNumber="25-044", projectName="Harbour Gate Compound",
        projectPhase="Bridging",
        primeAe="Calderwood Architects",
        geotechnicalEngineerFirm="Meridian Geotechnical",
        testingProgramSummary="Two borings to 30 m with undisturbed sampling, "
                              "a triaxial series and one consolidation test.",
        boringCount=3, tableCount=4, previousInvestigationCount=1,
        structureCount=2, structureList=["chancery", "seawall"],
        outsideProject="no",
        recommendedFoundations=["driven piles", "pile-supported mat"],
        boringDictionary=["B-1", "B-2"],
        bearingCapacity=["600 kN allowable axial capacity per 400 mm pile"],
        bearingCapacityValues=[BearingValue(
            value=Quantity(value=600.0, unit="kN"),
            foundation_type="driven pile",
            condition="allowable axial capacity, 400 mm precast section",
            citation=cite(4, "an allowable axial capacity of 600 kN"))],
        citations={
            "postName": cite(2, "the Vale Harbour compound"),
            "recommendedFoundations": cite(4, "driven piles are recommended"),
            "boringCount": cite(3, "three borings were drilled"),
            "projectPhase": cite(2, "bridging documents"),
            "bearingCapacity": cite(4, "an allowable axial capacity of "
                                       "600 kN"),
            "geotechnicalEngineerFirm": cite(2, "Meridian Geotechnical")})
    hazards = NaturalHazardFacts(
        liquefactionPotential="potentially liquefiable",
        siteClass="Site Class C", siteClassNormalized="C",
        seismicCodeUsed="ASCE 7-22", asceSevenVersion="ASCE 7-22",
        asceSevenVersionNormalized="7-22",
        earthHazardsExposed=["seismic shaking", "liquefaction",
                             "liquefaction-induced settlement", "flooding"],
        siteResponseMention="yes - a one-dimensional site response analysis "
                            "was run for the seawall",
        hazardAnalysisMention="yes - a probabilistic seismic hazard analysis",
        seismicParameterSummary="Ss 1.21 g, S1 0.48 g, site class C, "
                                "SDS 0.97 g, SD1 0.55 g.",
        reportDate="2 July 2025", reportDateISO="2025-07-02",
        naturalHazardSummary="The loose marine sand between 4 m and 9 m is "
                             "potentially liquefiable and settlement of up "
                             "to 75 mm is estimated.",
        citations={"siteClass": cite(4, "Site Class C"),
                   "liquefactionPotential": cite(3, "the marine sand is "
                                                    "potentially liquefiable"),
                   "hazardAnalysisMention": cite(4, "a probabilistic seismic "
                                                    "hazard analysis was run"),
                   "reportDate": cite(0, "2 July 2025")})
    return ReportRecord(
        document=_document("L02", 61),
        project=Project(name="Harbour Gate Compound", number="25-044",
                        client="Overseas Estates Office"),
        general=general, natural_hazards=hazards,
        narrative=NarrativeFacts(stated_counts={"borings": 3},
                                 found_counts={"borings": 2},
                                 pages=list(NARRATIVE_PAGES)),
        investigations=[boring("B-1", n_values=[4, 6, 9], water=3.0,
                               description="Soft grey marine CLAY",
                               uscs="CH", elevation=4.2),
                        boring("B-2", page=8, depth=30.0,
                               n_values=[5, 8, 31], water=3.5,
                               description="Loose marine SAND", uscs="SP",
                               elevation=4.0)],
        lab_tests=[triaxial(), consolidation()],
        calculations=[calculation(
            "lateral_pile", program="LPILE 2022", method="p-y",
            subject="400 mm precast pile at the seawall",
            summary="Lateral response of a single 400 mm pile under a 120 kN "
                    "head load, giving 18 mm of head deflection.",
            results=[named("head deflection", 18.0, "mm"),
                     named("maximum moment", 96.0, "kN*m")]),
            calculation(
                "liquefaction", program="CLiq", method="Boulanger & Idriss",
                subject="borings B-1 and B-2",
                summary="Liquefaction triggering for the marine sand, giving "
                        "75 mm of post-liquefaction settlement.",
                results=[named("settlement", 75.0, "mm")],
                page=CALC_PAGES[0])],
        qa=[QAEntry(kind="count_mismatch", where="narrative.boringCount",
                    detail="the narrative states three borings and two logs "
                           "were found in the appendix",
                    values=["3", "2"], pages=[3]),
            QAEntry(kind="disagreement", where="B-2.spt",
                    detail="the log grid and the page tables read different "
                           "N values at 14.5 ft",
                    values=["31", "13"], pages=[8])],
        bound_documents=[BoundReport(
            bound_id="bound1", report_id="L02-bound1",
            title="Fairhaven Annex Prior Study",
            firm="Atlas Ground Engineering", date="12 June 2014",
            kind="appended_prior_report",
            document_type="geotechnical report",
            pages="15-18", first_page=15, last_page=18, n_pages=4,
            said_by=["triage", "planlens"],
            counts={"investigations": 1, "lab_tests": 0},
            folder=os.path.join("bound", "bound1"),
            record_path=os.path.join("bound", "bound1",
                                     "report.record.json"))])


def record_l02_bound() -> ReportRecord:
    """The earlier firm's study reproduced inside L02, as its own record."""
    general = GeneralFacts(
        documentType="geotechnical report",
        quickSummary="An earlier investigation of the same waterfront "
                     "parcel, reproduced as an appendix.",
        postName="Vale Harbour", projectName="Fairhaven Annex Prior Study",
        projectNumber="14-207",
        geotechnicalEngineerFirm="Atlas Ground Engineering",
        boringCount=1, boringDictionary=["BH-101"],
        recommendedFoundations=["driven piles"],
        citations={"geotechnicalEngineerFirm": cite(15, "Atlas Ground "
                                                       "Engineering"),
                   "boringCount": cite(16, "one borehole was drilled")})
    hazards = NaturalHazardFacts(
        reportDate="12 June 2014", reportDateISO="2014-06-12",
        siteClass="Site Class D", siteClassNormalized="D",
        citations={"reportDate": cite(15, "12 June 2014")})
    return ReportRecord(
        document=_document("L02-bound1", 4, workflow="partial"),
        project=Project(name="Fairhaven Annex Prior Study", number="14-207"),
        general=general, natural_hazards=hazards,
        investigations=[boring("BH-101", page=16, depth=22.0,
                               n_values=[7, 11, 19], water=2.8,
                               description="Soft grey marine CLAY",
                               uscs="CH", elevation=4.1)],
        parent=ParentReport(report_id="L02", bound_id="bound1",
                            pages="15-18", first_page=15, last_page=18,
                            n_pages=4,
                            record_path=os.path.join(
                                os.pardir, os.pardir, "report.record.json")),
        qa=[QAEntry(kind="note", where="bound",
                    detail="read as its own record from pages 15-18 of L02")])


def record_l03() -> ReportRecord:
    """A due-diligence addendum, mostly scanned, that answered little."""
    general = GeneralFacts(
        documentType="report addendum",
        quickSummary="An addendum reporting four cone soundings on the "
                     "northern parcel.",
        postName="Elmridge", projectPhase="Technical due diligence",
        projectNumber="24-311", projectName="Larkspur Ridge Parcel",
        geotechnicalEngineerFirm="Atlas Ground Engineering",
        cptCount=4,
        citations={"cptCount": cite(3, "four cone penetration tests"),
                   "postName": cite(2, "the Elmridge compound"),
                   "projectPhase": cite(2, "technical due diligence")})
    hazards = NaturalHazardFacts(
        reportDate="20 November 2024", reportDateISO="2024-11-20",
        citations={"reportDate": cite(0, "20 November 2024")})
    return ReportRecord(
        document=_document("L03", 18, workflow="partial", scan=0.55),
        project=Project(name="Larkspur Ridge Parcel", number="24-311"),
        general=general, natural_hazards=hazards,
        investigations=[cone("CPT-1"), cone("CPT-2", page=11)],
        qa=[QAEntry(kind="unreadable", where="pages 5-9",
                    detail="five pages have no text layer and no Document "
                           "Intelligence result was supplied",
                    pages=[5, 6, 7, 8, 9]),
            QAEntry(kind="partial", where="narrative",
                    detail="the narrative pages are a scan, so the owner's "
                           "questions were answered from the cover only",
                    pages=[2])])


def record_l04() -> ReportRecord:
    """A recommendation letter with pits, a probe and a slope calculation."""
    general = GeneralFacts(
        documentType="recommendation letter",
        quickSummary="A recommendation letter for a retaining wall behind "
                     "the annex, based on four test pits and a dynamic "
                     "probe.",
        postName="Fairhaven",
        propertyType="Existing embassy or consulate compound",
        projectNumber="26-002", projectName="Northfield Annex Wall",
        projectPhase="Design-build",
        geotechnicalEngineerFirm="Stonebridge Geosciences",
        testingProgramSummary="Four test pits to 2.7 m and one dynamic probe.",
        testPitCount=4,
        structureCount=1, structureList=["retaining wall"],
        recommendedFoundations=["spread footings", "gravity retaining wall"],
        testPitDictionary=["TP-1", "TP-2", "TP-3", "TP-4"],
        bearingCapacity=["150 kPa allowable beneath the wall footing"],
        bearingCapacityValues=[BearingValue(
            value=Quantity(value=150.0, unit="kPa"),
            foundation_type="gravity wall footing",
            condition="allowable on the dense residual soil",
            citation=cite(3, "an allowable bearing pressure of 150 kPa"))],
        citations={"postName": cite(2, "the Fairhaven compound"),
                   "testPitCount": cite(3, "four test pits were excavated"),
                   "recommendedFoundations": cite(3, "a gravity retaining "
                                                     "wall is recommended"),
                   "bearingCapacity": cite(3, "an allowable bearing pressure "
                                              "of 150 kPa"),
                   "geotechnicalEngineerFirm": cite(2,
                                                    "Stonebridge Geosciences")})
    hazards = NaturalHazardFacts(
        earthHazardsExposed=["landslide", "erosion"],
        geophysicalTestingMention="no",
        soilCorrosion="unclear",
        reportDate="9 January 2026", reportDateISO="2026-01-09",
        naturalHazardSummary="The slope above the annex shows shallow "
                             "creep and surface erosion; no seismic hazard "
                             "assessment was made.",
        citations={"reportDate": cite(0, "9 January 2026"),
                   "earthHazardsExposed": cite(3, "shallow creep on the "
                                                  "slope above")})
    return ReportRecord(
        document=_document("L04", 14, workflow="standard"),
        project=Project(name="Northfield Annex Wall", number="26-002"),
        general=general, natural_hazards=hazards,
        investigations=[test_pit("TP-1"), test_pit("TP-2", page=PIT_PAGE + 1),
                        probe("DP-1")],
        lab_tests=[atterberg("TP-1", "BK-1", ll=52.0, pl=23.0)],
        calculations=[calculation(
            "slope_stability", program="SLIDE2 9.0",
            method="Bishop simplified",
            subject="the slope above the annex, static and seismic",
            summary="Circular limit-equilibrium analysis of the 12 m slope, "
                    "giving a static factor of safety of 1.43 and 1.05 "
                    "pseudo-static.",
            results=[named("FS static", 1.43, ""),
                     named("FS seismic", 1.05, "")])],
        qa=[QAEntry(kind="out_of_range", where="lab_tests[0].result.ll",
                    detail="a liquid limit of 52 on a sample logged as silty "
                           "sand is outside the range that pairing usually "
                           "gives", values=["52"], pages=[12])])


def record_l05() -> ReportRecord:
    """An older report whose finding is corrosive ground."""
    general = GeneralFacts(
        documentType="geotechnical report",
        quickSummary="An investigation of the utility corridor, whose "
                     "finding is severely corrosive ground.",
        postName="Vale Harbour",
        propertyType="Existing embassy or consulate compound",
        projectNumber="23-087", projectName="Cedar Hollow Corridor",
        projectPhase="Technical due diligence",
        geotechnicalEngineerFirm="Meridian Geotechnical",
        testingProgramSummary="One boring with corrosivity testing on two "
                              "samples.",
        boringCount=1, boringDictionary=["B-5"],
        recommendedFoundations=["spread footings"],
        citations={"postName": cite(2, "the Vale Harbour compound"),
                   "boringCount": cite(3, "a single boring was drilled"),
                   "geotechnicalEngineerFirm": cite(2,
                                                    "Meridian Geotechnical")})
    hazards = NaturalHazardFacts(
        liquefactionPotential="not liquefiable",
        siteClass="Site Class E", siteClassNormalized="E",
        seismicCodeUsed="ASCE 7-16", asceSevenVersion="ASCE 7-16",
        asceSevenVersionNormalized="7-16",
        soilCorrosion="yes - sulfate at 2,100 ppm and a pH of 4.6, which is "
                      "severely corrosive to buried concrete",
        earthHazardsExposed=["seismic shaking"],
        reportDate="30 May 2023", reportDateISO="2023-05-30",
        naturalHazardSummary="No liquefaction; the finding is severely "
                             "corrosive ground along the utility corridor.",
        citations={"siteClass": cite(4, "Site Class E"),
                   "soilCorrosion": cite(4, "severely corrosive to buried "
                                            "concrete"),
                   "reportDate": cite(0, "30 May 2023")})
    return ReportRecord(
        document=_document("L05", 31),
        project=Project(name="Cedar Hollow Corridor", number="23-087"),
        general=general, natural_hazards=hazards,
        investigations=[boring("B-5", n_values=[22, 34, 41], water=None,
                               description="Firm brown residual SILT",
                               uscs="ML", elevation=61.0)],
        lab_tests=[corrosivity()],
        qa=[QAEntry(kind="note", where="diggs.roundtrip",
                    detail="the DIGGS file reads back equal to the record")])


def record_l06() -> ReportRecord:
    """An appendix of figures: the library's reference tier."""
    general = GeneralFacts(
        documentType="report appendix or figure(s)",
        projectNumber="22-514", projectName="Quarry Bend Appendix",
        geotechnicalEngineerFirm="Soil & Rock Consulting Engineers",
        boringCount=1, boringDictionary=["B-9"],
        citations={"boringCount": cite(1, "Boring B-9")})
    hazards = NaturalHazardFacts(
        reportDate="15 September 2022", reportDateISO="2022-09-15")
    return ReportRecord(
        document=_document("L06", 9, workflow="appendix_only"),
        project=Project(name="Quarry Bend Appendix", number="22-514"),
        general=general, natural_hazards=hazards,
        investigations=[boring("B-9", n_values=[18, 25, 33], water=6.0,
                               description="Weathered SANDSTONE",
                               uscs="", elevation=88.0)],
        qa=[QAEntry(kind="skipped", where="narrative",
                    detail="an appendix has no narrative, so the owner's "
                           "questions were not asked of it")])


#: The six top-level records, by their file stem, plus the bound child.
RECORDS = {
    "L01": record_l01,
    "L02": record_l02,
    "L03": record_l03,
    "L04": record_l04,
    "L05": record_l05,
    "L06": record_l06,
}


def build_records() -> Dict[str, ReportRecord]:
    """The six records, built fresh."""
    return {name: build() for name, build in RECORDS.items()}


def build_library(root: Any, *, write_diggs: bool = False) -> Dict[str, Any]:
    """Write the six reports into ``root`` the way the ingest would.

    The bound child lands under ``L02/bound/bound1/`` with its parent's key
    folded into its own, which is exactly what the graph does, so the
    library's rebuild has the same thing to read that a real run leaves.
    Returns ``{report_id: library key}``.
    """
    from report_ingest.writers import write_outputs

    out = str(root)
    os.makedirs(out, exist_ok=True)
    db_path = os.path.join(out, "reports.db")
    keys: Dict[str, Any] = {}
    for name, build in RECORDS.items():
        record = build()
        written = write_outputs(record, os.path.join(out, name),
                                db_path=db_path,
                                write_diggs_file=write_diggs)
        keys[name] = written.key
        if name == "L02":
            child = record_l02_bound()
            written_child = write_outputs(
                child, os.path.join(out, name, "bound", "bound1"),
                db_path=db_path, write_diggs_file=write_diggs,
                parent_base=written.key)
            keys["L02-bound1"] = written_child.key
    return keys
