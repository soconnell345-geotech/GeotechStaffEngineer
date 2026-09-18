"""The synthetic report, with a narrative that actually answers the schemas.

``planlens.testing.build_synthetic_report`` builds a whole geotechnical report
whose every page role is stated by hand -- cover, contents, narrative, figure,
appendix tabs, two boring logs, a test pit log, photographs, two laboratory
sheets, an appended prior report and a calculation printout. Its narrative is
real prose about a real-shaped investigation, but it was written to exercise
page ROLES, so it never states a bearing pressure, a site class or a seismic
code: the things the owner's two query schemas are mostly about.

So this module takes that report and writes four more sentences onto its last
narrative page, in the free space below the prose that is already there. The
sentences are the ones a geotechnical report always carries and this one did
not: the recommendation with its unit, the site class, the code it came from,
the liquefaction verdict, the firm and the date. planlens is not touched --
the pages come back out of it as they went in, with one paragraph more on one
of them -- and the expected answers are stated here, beside the sentences that
say them, so a test asserts against what a reader of the page would say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

#: The page the extra sentences are written on: the third narrative page of
#: the synthetic report (pages 2, 3 and 4 are its narrative).
NARRATIVE_PAGE = 4
#: The narrative work item, as ``document_items`` reports it.
NARRATIVE_PAGES = (2, 3, 4)
#: The main body: the narrative plus the figure page that follows it, which
#: is where the one figure caption in the document sits.
BODY_PAGES = (2, 3, 4, 5)

#: What is written onto the page. Ordinary report sentences, in the trade's
#: own words; nothing here comes from any real document.
EXTRA_NARRATIVE = (
    "5.0 RECOMMENDATIONS "
    "Four borings and three test pits were completed for this study. "
    "An allowable bearing pressure of 3,000 psf is recommended for spread "
    "footings bearing on the dense residual soil, with a minimum embedment "
    "of 0.6 m below finished grade. "
    "The site is classified as Site Class D in accordance with ASCE 7-16. "
    "The liquefaction potential of the site soils is considered low. "
    "This report is dated 14 March 2026 and was prepared by Soil & Rock "
    "Consulting Engineers."
)


@dataclass
class NarrativeGT:
    """The extended report and the answers its narrative supports."""

    pdf: bytes
    narrative_pages: List[int] = field(
        default_factory=lambda: list(NARRATIVE_PAGES))
    body_pages: List[int] = field(default_factory=lambda: list(BODY_PAGES))
    #: What the narrative says, field by field, in the owner's names. These
    #: are the answers a reader of these pages should give; the fake engine's
    #: reading is built from them so a test asserts on the pipeline rather
    #: than on a model.
    general: Dict[str, Any] = field(default_factory=lambda: {
        "documentType": "geotechnical report",
        "projectName": "Rosewood Terrace Development",
        "geotechnicalEngineerFirm": "Soil & Rock Consulting Engineers",
        "boringCount": 4,
        "testPitCount": 3,
        "recommendedFoundations": ["spread footings"],
        "bearingCapacity": [
            "An allowable bearing pressure of 3,000 psf is recommended for "
            "spread footings bearing on the dense residual soil"],
    })
    natural_hazards: Dict[str, Any] = field(default_factory=lambda: {
        "siteClass": "Site Class D",
        "seismicCodeUsed": "ASCE 7-16",
        "asceSevenVersion": "ASCE 7-16",
        "liquefactionPotential": "low",
        "reportDate": "14 March 2026",
    })
    #: The bearing pressure as a number, with the unit the page prints.
    bearing_psf: float = 3000.0
    #: What ``count_captions`` must find over the body pages: the document
    #: captions Figure 1 and nothing else, while its printed list of figures
    #: names two. That disagreement is the point of the fixture.
    counted_figures: int = 1
    counted_tables: int = 0
    listed_figures: int = 2


def build_narrative_report() -> NarrativeGT:
    """The synthetic report with the four extra narrative sentences."""
    import fitz
    from planlens.testing.report_fixtures import build_synthetic_report

    base = build_synthetic_report()
    doc = fitz.open(stream=base.pdf, filetype="pdf")
    page = doc[NARRATIVE_PAGE]
    # Below the prose that is already on the page and above its footer. The
    # existing text box runs from y=120 and the prose fills roughly half of
    # it, so this sits in blank paper rather than on top of anything.
    page.insert_textbox(fitz.Rect(72, 430, 540, 700), EXTRA_NARRATIVE,
                        fontsize=10)
    pdf = doc.tobytes()
    doc.close()
    return NarrativeGT(pdf=pdf)
