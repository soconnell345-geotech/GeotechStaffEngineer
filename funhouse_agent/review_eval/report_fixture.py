"""A synthetic geotechnical report for the COVERAGE tasks of the review suite.

Built to measure one thing the 2026-10-06 field session lost: whether an agent
asked to take the data out of a report reads EVERY page that holds data
(``module_work/field_feedback/2026-10-06_geotech-report-session_v5.32.0/
FINDINGS.md`` P3 and section 8). That session's report had new boring logs
as clean vector pages, older logs as scans, and a laboratory appendix of
many sheets; the agent read 10 of 23 classification sheets, none of the
chemistry, compaction or density sheets, and skipped the newest logs because
it took "boring log" to mean "scanned page".

This report has the same SHAPE and nothing else of it: every name, place
and number below is invented, and the wording is the wording of the trade.

Layout (0-based tool pages; a PDF viewer shows each one higher)::

    0        cover
    1        contents
    2-4      narrative: the 2026 borings B-1 to B-3, the 2011 borings BH-1 to
             BH-3 whose logs are reproduced, groundwater, the testing
    5        site plan (a figure; the boring labels are text)
    6        divider  APPENDIX A-1 - BORING LOGS (2026 INVESTIGATION)
    7-8      B-1, a vector log form of two sheets
    9        B-2, vector
    10       B-3, vector
    11       divider  APPENDIX A-2 - BORING LOGS FROM THE 2011 INVESTIGATION
    12-14    BH-1, BH-2, BH-3: SCANNED - a picture of the page, no text layer
    15       divider  APPENDIX B - LABORATORY TEST RESULTS
    16       Table B-1, the summary of laboratory test results
    17-29    thirteen laboratory sheets: three Atterberg, six particle size,
             compaction, in-place density, soil chemistry, water chemistry

The summary table carries ONE error of the kind the field session's report
did: for B-2 S-3 its liquid and plastic limits are swapped (LL 20, PL 32,
where the sheet on page 18 reads LL 32, PL 20). That is what the consistency
task asks about.

Ground truth is built WITH the PDF (:func:`build_synthetic_extraction_report`)
so a check can never drift from the document it is asked of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

LETTER = (612.0, 792.0)

PROJECT = "Harbour Road Pump Station"
FIRM = "Example Geotechnical Ltd"
HEADER = f"{PROJECT} | Geotechnical Investigation Report"

#: The 0-based pages of the logs drawn as vector forms (text layer present).
VECTOR_LOG_PAGES = (7, 8, 9, 10)
#: The 0-based pages of the scanned logs (a picture, no text layer).
SCANNED_LOG_PAGES = (12, 13, 14)
#: The summary table and the thirteen laboratory sheets.
LAB_PAGES = tuple(range(16, 30))
#: Every page that holds data an extraction must read.
TARGET_PAGES = tuple(sorted(VECTOR_LOG_PAGES + SCANNED_LOG_PAGES + LAB_PAGES))

NEW_BORINGS = ("B-1", "B-2", "B-3")
OLD_BORINGS = ("BH-1", "BH-2", "BH-3")


@dataclass
class ExtractionReportGT:
    """The report and what is where in it."""

    pdf: bytes
    n_pages: int = 30
    target_pages: Tuple[int, ...] = TARGET_PAGES
    #: Pages a text tool cannot read: they must be LOOKED at.
    look_pages: Tuple[int, ...] = SCANNED_LOG_PAGES
    log_pages: Tuple[int, ...] = VECTOR_LOG_PAGES + SCANNED_LOG_PAGES
    lab_pages: Tuple[int, ...] = LAB_PAGES
    borings: Tuple[str, ...] = NEW_BORINGS + OLD_BORINGS
    #: page -> the role a reader would give it.
    roles: Dict[int, str] = field(default_factory=dict)
    #: (boring, sample) -> the values the sheets print.
    lab_values: Dict[Tuple[str, str], Dict[str, str]] = field(
        default_factory=dict)


# ---------------------------------------------------------------------------
# Content (invented)
# ---------------------------------------------------------------------------

#: rows of each 2026 log: (depth m, sample, blows, N, USCS, description).
NEW_LOGS: Dict[str, List[Tuple[str, str, str, str, str, str]]] = {
    "B-1": [
        ("1.0", "S-1", "3-4-5", "9", "SM", "Brown silty SAND, loose, moist"),
        ("2.5", "S-2", "4-5-6", "11", "CL", "Grey lean CLAY, stiff, moist"),
        ("4.0", "S-3", "5-7-8", "15", "CL", "Grey lean CLAY, stiff"),
        ("5.5", "S-4", "7-9-12", "21", "SC", "Brown clayey SAND, medium dense"),
        ("7.0", "S-5", "9-12-14", "26", "SP", "Brown poorly graded SAND, wet"),
        ("8.5", "S-6", "11-14-17", "31", "SP", "Brown poorly graded SAND, dense"),
        ("10.0", "S-7", "12-16-19", "35", "GP", "Grey sandy GRAVEL, dense"),
        ("11.5", "S-8", "14-18-22", "40", "GP", "Grey sandy GRAVEL, dense"),
        ("13.0", "S-9", "15-21-25", "46", "GP", "Grey GRAVEL with cobbles"),
        ("14.5", "S-10", "18-24-28", "52", "GP", "Grey GRAVEL, very dense"),
    ],
    "B-2": [
        ("1.0", "S-1", "2-3-3", "6", "ML", "Brown sandy SILT, soft, moist"),
        ("2.5", "S-2", "3-4-4", "8", "ML", "Brown sandy SILT, firm"),
        ("4.0", "S-3", "4-5-5", "10", "CL", "Grey lean CLAY, firm to stiff"),
        ("5.5", "S-4", "6-8-9", "17", "SC", "Grey clayey SAND, medium dense"),
        ("7.0", "S-5", "8-10-13", "23", "SM", "Brown silty SAND, medium dense"),
    ],
    "B-3": [
        ("1.0", "S-1", "4-5-7", "12", "SM", "Brown silty SAND with gravel"),
        ("2.5", "S-2", "5-6-8", "14", "CL", "Mottled lean CLAY, stiff"),
        ("4.0", "S-3", "6-8-10", "18", "SC", "Brown clayey SAND, medium dense"),
        ("5.5", "S-4", "9-13-16", "29", "SP", "Brown SAND, medium dense, wet"),
        ("7.0", "S-5", "13-19-24", "43", "GP", "Grey sandy GRAVEL, dense"),
    ],
}

NEW_LOG_HEADERS = {
    "B-1": ("12 March 2026", "4.9", "3.1", "14.9"),
    "B-2": ("13 March 2026", "5.2", "2.8", "7.4"),
    "B-3": ("14 March 2026", "4.6", "3.6", "7.4"),
}

#: rows of each 2011 log (the scans): (depth m, N, description).
OLD_LOGS: Dict[str, List[Tuple[str, str, str]]] = {
    "BH-1": [("1.5", "7", "Brown silty SAND, loose"),
             ("3.0", "12", "Grey CLAY, stiff"),
             ("4.5", "19", "Brown clayey SAND"),
             ("6.0", "33", "Grey sandy GRAVEL, dense")],
    "BH-2": [("1.5", "5", "Grey silty SAND with shell fragments, loose"),
             ("3.0", "9", "Grey silty SAND with shell fragments"),
             ("4.5", "16", "Grey CLAY, stiff"),
             ("6.0", "38", "Grey sandy GRAVEL, dense")],
    "BH-3": [("1.5", "8", "Brown sandy SILT"),
             ("3.0", "14", "Brown CLAY, stiff"),
             ("4.5", "27", "Brown SAND, medium dense"),
             ("6.0", "41", "Grey GRAVEL, very dense")],
}

#: (water depth m, total depth m) printed on each 2011 log.
OLD_LOG_HEADERS = {"BH-1": ("3.8", "6.5"), "BH-2": ("3.4", "6.5"),
                   "BH-3": ("4.1", "6.5")}

ATTERBERG = [  # (boring, sample, depth, LL, PL, PI, wc)
    ("B-1", "S-2", "2.5", "36", "19", "17", "24.0"),
    ("B-2", "S-3", "4.0", "32", "20", "12", "27.5"),
    ("B-3", "S-2", "2.5", "44", "21", "23", "22.8"),
]

GRADING = [  # (boring, sample, depth, gravel %, sand %, fines %)
    ("B-1", "S-1", "1.0", "4.2", "62.4", "33.4"),
    ("B-1", "S-4", "5.5", "8.9", "55.1", "36.0"),
    ("B-2", "S-1", "1.0", "0.6", "38.8", "60.6"),
    ("B-2", "S-5", "7.0", "6.1", "52.2", "41.7"),
    ("B-3", "S-1", "1.0", "17.3", "54.9", "27.8"),
    ("B-3", "S-4", "5.5", "11.4", "83.3", "5.3"),
]

COMPACTION = ("B-1", "BULK-1", "0.5-1.5", "1.94", "11.6")
DENSITY = ("B-2", "S-2", "2.5", "1.71", "2.05", "19.9")  # dry, wet, wc
SOIL_CHEM = [("B-1", "S-3", "4.0", "7.2", "310", "95", "4,200"),
             ("B-3", "S-3", "4.0", "6.8", "640", "460", "1,900")]
WATER_CHEM = ("B-2", "W-1", "2.8", "7.4", "1,850", "2,300")


def _lab_values() -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for b, s, d, ll, pl, pi, wc in ATTERBERG:
        out[(b, s)] = {"LL": ll, "PL": pl, "PI": pi, "wc": wc, "depth": d}
    for b, s, d, g, sa, f in GRADING:
        out.setdefault((b, s), {}).update(
            {"gravel": g, "sand": sa, "fines": f, "depth": d})
    b, s, d, mdd, omc = COMPACTION
    out[(b, s)] = {"MDD": mdd, "OMC": omc, "depth": d}
    b, s, d, dry, wet, wc = DENSITY
    out[(b, s)] = {"dry_density": dry, "wet_density": wet, "wc": wc,
                   "depth": d}
    for b, s, d, ph, so4, cl, res in SOIL_CHEM:
        out[(b, s)] = {"pH": ph, "sulfate": so4, "chloride": cl,
                       "resistivity": res, "depth": d}
    b, s, d, ph, so4, cl = WATER_CHEM
    out[(b, s)] = {"pH": ph, "sulfate": so4, "chloride": cl, "depth": d}
    return out


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _page(doc):
    return doc.new_page(width=LETTER[0], height=LETTER[1])


def _footer(p, n: int) -> None:
    p.insert_text((72, 40), HEADER, fontsize=8)
    p.insert_text((72, 770), f"Project 26-204 | Page {n}", fontsize=8)


def _cover(doc) -> None:
    import fitz
    p = _page(doc)
    p.insert_text((90, 250), "GEOTECHNICAL INVESTIGATION REPORT", fontsize=20,
                  fontname="hebo")
    p.insert_text((90, 285), PROJECT, fontsize=18)
    p.insert_textbox(fitz.Rect(90, 420, 520, 620),
                     "Prepared for:\nHarbour Road Utilities Board\n\n"
                     f"Prepared by:\n{FIRM}\n20 April 2026", fontsize=11)


CONTENTS = (
    "TABLE OF CONTENTS",
    "1.0 Introduction ........................................... 1",
    "2.0 Subsurface Exploration ................................. 2",
    "3.0 Groundwater and Laboratory Testing ..................... 3",
    "",
    "LIST OF FIGURES",
    "Figure 1  Site Plan and Boring Locations ................... 4",
    "",
    "APPENDICES",
    "Appendix A-1  Boring Logs (2026 Investigation)",
    "Appendix A-2  Boring Logs from the 2011 Investigation",
    "Appendix B  Laboratory Test Results",
)


def _contents(doc) -> None:
    p = _page(doc)
    p.insert_text((90, 90), CONTENTS[0], fontsize=16, fontname="hebo")
    y = 130
    for line in CONTENTS[1:]:
        if line:
            bold = line in ("LIST OF FIGURES", "APPENDICES")
            p.insert_text((90, y), line, fontsize=11,
                          fontname="hebo" if bold else "helv")
        y += 20


NARRATIVE = (
    ("1.0 Introduction",
     f"{FIRM} carried out a geotechnical investigation for the proposed "
     f"{PROJECT}. The station comprises a wet well about 7 m deep, a valve "
     "chamber and a single-storey control building. This report presents "
     "the subsurface conditions encountered, the results of the field and "
     "laboratory testing, and our interpretation of those results. The site "
     "lies on reclaimed land beside the estuary and was used as a storage "
     "yard until 2019. An earlier investigation of the same yard was carried "
     "out in 2011 for a warehouse that was never built; the logs of that "
     "investigation are reproduced in Appendix A-2 with the permission of the "
     "owner. "),
    ("2.0 Subsurface Exploration",
     "Three borings, B-1, B-2 and B-3, were drilled between 12 and 14 March "
     "2026 with a truck-mounted rig using hollow-stem augers. Standard "
     "penetration tests were made at 1.5 m intervals with an automatic "
     "hammer. Boring B-1 was taken to 14.9 m at the wet well; B-2 and B-3 "
     "were stopped at 7.4 m. The logs are in Appendix A-1. The three borings "
     "of the 2011 investigation, BH-1, BH-2 and BH-3, reached 6.5 m; their "
     "locations are shown on Figure 1 and their logs, reproduced from the "
     "earlier report, are in Appendix A-2. The ground comprises fill and "
     "estuarine silty sand over lean clay, over sand and dense gravel. "),
    ("3.0 Groundwater and Laboratory Testing",
     "Groundwater was met in all three 2026 borings between 2.8 m and 3.6 m "
     "below ground and a sample of it was taken from B-2 for chemical "
     "testing. The 2011 logs record water between 3.4 m and 4.1 m. "
     "Laboratory testing comprised Atterberg limits, particle size "
     "distribution, a compaction test on a bulk sample, an in-place density, "
     "and chemical tests on soil and groundwater for the exposure of buried "
     "concrete and steel. The results are summarised in Table B-1 and each "
     "test is reported on its own sheet in Appendix B. "),
)


def _narrative(doc, n: int) -> None:
    import fitz
    heading, prose = NARRATIVE[n - 1]
    p = _page(doc)
    _footer(p, n)
    p.insert_text((72, 100), heading, fontsize=14, fontname="hebo")
    p.insert_textbox(fitz.Rect(72, 120, 540, 720), prose * 2, fontsize=10)


#: Where each boring is drawn on the site plan (page points).
PLAN_POINTS = {"B-1": (220, 300), "B-2": (360, 260), "B-3": (300, 430),
               "BH-1": (160, 380), "BH-2": (420, 380), "BH-3": (250, 520)}


def _site_plan(doc) -> None:
    p = _page(doc)
    shape = p.new_shape()
    shape.draw_rect((110, 180, 500, 600))
    shape.draw_rect((250, 330, 340, 400))
    shape.finish(color=(0, 0, 0), width=1.0)
    for name, (x, y) in PLAN_POINTS.items():
        if name.startswith("BH"):
            shape.draw_rect((x - 4, y - 4, x + 4, y + 4))
        else:
            shape.draw_circle((x, y), 5)
        shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()
    for name, (x, y) in PLAN_POINTS.items():
        p.insert_text((x + 8, y + 3), name, fontsize=8)
    p.insert_text((110, 160), "SITE PLAN AND BORING LOCATIONS", fontsize=13,
                  fontname="hebo")
    p.insert_text((110, 630), "Circles: 2026 borings. Squares: 2011 borings.",
                  fontsize=8)
    p.insert_text((110, 660), "Figure 1 - Site Plan and Boring Locations",
                  fontsize=11)


def _divider(doc, title: str, contents: str) -> None:
    import fitz
    p = _page(doc)
    p.insert_text((90, 300), title, fontsize=18, fontname="hebo")
    p.insert_textbox(fitz.Rect(90, 340, 520, 520), contents, fontsize=11)


LOG_FIELDS = ("DEPTH (m)", "SAMPLE", "BLOWS", "N", "USCS",
              "MATERIAL DESCRIPTION")
_LOG_X = (72, 122, 172, 232, 262, 302, 540)


def _vector_log(doc, boring: str, rows, sheet: str) -> None:
    """A ruled log form with a text layer (the 2026 logs)."""
    p = _page(doc)
    date, elev, water, total = NEW_LOG_HEADERS[boring]
    p.insert_text((72, 50), "LOG OF BORING", fontsize=13, fontname="hebo")
    p.insert_text((400, 50), f"BORING NO. {boring}", fontsize=13,
                  fontname="hebo")
    p.insert_text((72, 68), f"Project: {PROJECT}   Sheet {sheet}", fontsize=8)
    p.insert_text((72, 80), f"Date drilled: {date}   Ground elevation: "
                            f"{elev} m   Water level: {water} m   "
                            f"Total depth: {total} m", fontsize=8)
    top, h = 96, 34
    n_rows = 16
    for x in _LOG_X:
        p.draw_line((x, top), (x, top + h * (n_rows + 1)), width=0.6)
    for i in range(n_rows + 2):
        p.draw_line((_LOG_X[0], top + h * i), (_LOG_X[-1], top + h * i),
                    width=0.6)
    for i, head in enumerate(LOG_FIELDS):
        p.insert_text((_LOG_X[i] + 2, top + 20), head, fontsize=6.5)
    for r, row in enumerate(rows):
        y = top + h * (r + 1) + 20
        for i, value in enumerate(row):
            p.insert_text((_LOG_X[i] + 2, y), value, fontsize=7)
    p.insert_text((72, 770), f"Drilled by Coastal Drilling   Logged by RK   "
                             f"{FIRM}", fontsize=7)


def _scanned_log(doc, boring: str) -> None:
    """A 2011 log as a SCAN: drawn on a scratch page, rasterised, speckled,
    and placed as a picture - no text layer at all."""
    import fitz
    scratch = fitz.open()
    p = scratch.new_page(width=LETTER[0], height=LETTER[1])
    water, total = OLD_LOG_HEADERS[boring]
    p.insert_text((72, 60), "BOREHOLE LOG", fontsize=15, fontname="hebo")
    p.insert_text((380, 60), boring, fontsize=15, fontname="hebo")
    p.insert_text((72, 84), "Warehouse Site Investigation 2011   "
                            "Earlier Consultants", fontsize=9)
    p.insert_text((72, 100), f"Water struck: {water} m   Final depth: "
                             f"{total} m", fontsize=9)
    xs = (72, 152, 212, 540)
    top, h = 120, 60
    for x in xs:
        p.draw_line((x, top), (x, top + h * 5), width=0.8)
    for i in range(6):
        p.draw_line((xs[0], top + h * i), (xs[-1], top + h * i), width=0.8)
    for i, head in enumerate(("DEPTH (m)", "SPT N", "DESCRIPTION")):
        p.insert_text((xs[i] + 3, top + 30), head, fontsize=9)
    for r, (depth, n, desc) in enumerate(OLD_LOGS[boring]):
        y = top + h * (r + 1) + 32
        p.insert_text((xs[0] + 3, y), depth, fontsize=10)
        p.insert_text((xs[1] + 3, y), n, fontsize=10)
        p.insert_text((xs[2] + 3, y), desc, fontsize=10)
    pix = p.get_pixmap(dpi=150, colorspace=fitz.csGRAY)
    scratch.close()
    # A scan's grain: a light speckle, deterministic.
    data = bytearray(pix.samples)
    for i in range(0, len(data), 997):
        data[i] = 120
    img = fitz.Pixmap(fitz.csGRAY, pix.width, pix.height, bytes(data), False)
    page = _page(doc)
    page.insert_image(page.rect, stream=img.tobytes("jpeg", jpg_quality=70))


def _lab_sheet(doc, title: str, standard: str, boring: str, sample: str,
               depth: str, rows) -> None:
    p = _page(doc)
    p.insert_text((72, 60), title, fontsize=15, fontname="hebo")
    p.insert_text((72, 82), standard, fontsize=9)
    p.insert_text((72, 110), f"Boring No. {boring}     Sample {sample}     "
                             f"Depth {depth} m", fontsize=9)
    y = 150
    for label, value in rows:
        p.insert_text((90, y), label, fontsize=10)
        p.insert_text((350, y), value, fontsize=10)
        y += 22
    p.insert_text((72, 770), "Tested by Example Materials Laboratory",
                  fontsize=8)


def _summary_table(doc) -> None:
    """Table B-1 - with B-2 S-3's limits SWAPPED (the planted error)."""
    p = _page(doc)
    p.insert_text((72, 60), "TABLE B-1  SUMMARY OF LABORATORY TEST RESULTS",
                  fontsize=13, fontname="hebo")
    heads = ("BORING", "SAMPLE", "DEPTH (m)", "wc (%)", "LL", "PL", "PI",
             "FINES (%)")
    xs = (72, 132, 192, 252, 312, 362, 412, 462)
    y = 100
    for x, head in zip(xs, heads):
        p.insert_text((x, y), head, fontsize=8, fontname="hebo")
    rows = []
    for b, s, d, ll, pl, pi, wc in ATTERBERG:
        if (b, s) == ("B-2", "S-3"):
            ll, pl = pl, ll                    # the report's own error
        rows.append((b, s, d, wc, ll, pl, pi, "-"))
    for b, s, d, _g, _sa, f in GRADING:
        rows.append((b, s, d, "-", "-", "-", "-", f))
    rows.sort(key=lambda r: (r[0], float(r[2])))
    for row in rows:
        y += 18
        for x, value in zip(xs, row):
            p.insert_text((x, y), value, fontsize=8)
    p.insert_text((72, y + 30), "See the individual test sheets for the "
                                "compaction, density and chemical tests.",
                  fontsize=8)


def build_synthetic_extraction_report() -> ExtractionReportGT:
    """The report in this module's docstring, with its ground truth."""
    import fitz
    doc = fitz.open()
    _cover(doc)                                                   # 0
    _contents(doc)                                                # 1
    for n in (1, 2, 3):                                           # 2-4
        _narrative(doc, n)
    _site_plan(doc)                                               # 5
    _divider(doc, "APPENDIX A-1 - BORING LOGS (2026 INVESTIGATION)",
             "Boring Logs B-1 (2 sheets), B-2 and B-3")           # 6
    b1 = NEW_LOGS["B-1"]
    _vector_log(doc, "B-1", b1[:5], "1 of 2")                     # 7
    _vector_log(doc, "B-1", b1[5:], "2 of 2")                     # 8
    _vector_log(doc, "B-2", NEW_LOGS["B-2"], "1 of 1")            # 9
    _vector_log(doc, "B-3", NEW_LOGS["B-3"], "1 of 1")            # 10
    _divider(doc, "APPENDIX A-2 - BORING LOGS FROM THE 2011 INVESTIGATION",
             "Borehole Logs BH-1, BH-2 and BH-3, reproduced from the "
             "Warehouse Site Investigation of 2011")              # 11
    for boring in OLD_BORINGS:                                    # 12-14
        _scanned_log(doc, boring)
    _divider(doc, "APPENDIX B - LABORATORY TEST RESULTS",
             "Table B-1 Summary of Laboratory Test Results\n"
             "Laboratory test sheets (13 sheets)")                # 15
    _summary_table(doc)                                           # 16
    for b, s, d, ll, pl, pi, wc in ATTERBERG:                     # 17-19
        _lab_sheet(doc, "ATTERBERG LIMITS", "ASTM D4318", b, s, d,
                   [("Liquid Limit, LL", ll), ("Plastic Limit, PL", pl),
                    ("Plasticity Index, PI", pi),
                    ("Natural moisture content", f"{wc} %")])
    for b, s, d, g, sa, f in GRADING:                             # 20-25
        _lab_sheet(doc, "PARTICLE SIZE DISTRIBUTION", "ASTM D6913", b, s, d,
                   [("Gravel", f"{g} %"), ("Sand", f"{sa} %"),
                    ("Fines (passing 0.075 mm)", f"{f} %")])
    b, s, d, mdd, omc = COMPACTION                                # 26
    _lab_sheet(doc, "MOISTURE-DENSITY (COMPACTION) RELATIONSHIP",
               "ASTM D1557", b, s, d,
               [("Maximum dry density", f"{mdd} Mg/m3"),
                ("Optimum moisture content", f"{omc} %")])
    b, s, d, dry, wet, wc = DENSITY                               # 27
    _lab_sheet(doc, "IN-PLACE DENSITY (DRIVE CYLINDER)", "ASTM D2937",
               b, s, d,
               [("Dry density", f"{dry} Mg/m3"),
                ("Wet density", f"{wet} Mg/m3"),
                ("Moisture content", f"{wc} %")])
    _soil_chemistry(doc)                                          # 28
    b, s, d, ph, so4, cl = WATER_CHEM                             # 29
    _lab_sheet(doc, "GROUNDWATER CHEMISTRY", "ASTM D516 / D512", b, s, d,
               [("pH", ph), ("Sulfate (SO4)", f"{so4} mg/L"),
                ("Chloride (Cl)", f"{cl} mg/L")])
    pdf = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    roles = {0: "cover", 1: "toc", 2: "narrative", 3: "narrative",
             4: "narrative", 5: "figure", 6: "divider", 11: "divider",
             15: "divider"}
    roles.update({p: "boring_log" for p in VECTOR_LOG_PAGES
                  + SCANNED_LOG_PAGES})
    roles.update({p: "lab_test" for p in LAB_PAGES})
    return ExtractionReportGT(pdf=pdf, roles=roles, lab_values=_lab_values())


def _soil_chemistry(doc) -> None:
    """One sheet, two samples: the way a chemistry laboratory reports."""
    p = _page(doc)
    p.insert_text((72, 60), "SOIL CHEMISTRY - CORROSIVITY", fontsize=15,
                  fontname="hebo")
    p.insert_text((72, 82), "pH ASTM G51, Sulfate ASTM C1580, Chloride "
                            "ASTM D512, Resistivity ASTM G57", fontsize=9)
    heads = ("BORING", "SAMPLE", "DEPTH (m)", "pH", "SULFATE (mg/kg)",
             "CHLORIDE (mg/kg)", "RESISTIVITY (ohm-cm)")
    xs = (72, 132, 190, 250, 290, 380, 470)
    for x, head in zip(xs, heads):
        p.insert_text((x, 120), head, fontsize=7, fontname="hebo")
    y = 120
    for row in SOIL_CHEM:
        y += 22
        for x, value in zip(xs, row):
            p.insert_text((x, y), value, fontsize=9)
    p.insert_text((72, 770), "Tested by Example Materials Laboratory",
                  fontsize=8)


__all__ = ["ExtractionReportGT", "build_synthetic_extraction_report",
           "TARGET_PAGES", "SCANNED_LOG_PAGES", "VECTOR_LOG_PAGES",
           "LAB_PAGES", "NEW_BORINGS", "OLD_BORINGS"]
