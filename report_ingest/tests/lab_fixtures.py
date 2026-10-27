"""Synthetic laboratory sheets, with the answers stated by hand.

Four pages, drawn with PyMuPDF in the style of ``planlens.testing``, carrying
between them what the lab reader has to cope with:

* an ATTERBERG sheet whose limits are in a results box beside a plasticity
  chart -- the tabulated-value-beats-the-plot case;
* a GRADATION sheet with a ruled percent-finer table under a grading curve,
  and Atterberg limits printed on the same sheet, so one page is two tests;
* a SUMMARY table of four specimens from three borings, one row each;
* a laboratory CERTIFICATE that lists the samples received and reports no
  result at all.

Nothing here is copied from a real sheet. The wording is the trade's; the
numbers are made up and the answers are in :class:`LabSheetGT`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

LETTER = (612.0, 792.0)


@dataclass
class LabSheetGT:
    """One synthetic sheet and what a reader should get off it."""

    pdf: bytes
    kind: str
    depth_unit: str = ""
    investigation_id: str = ""
    depth_top: float = 0.0
    #: What the sheet prints, by the record's own field names.
    values: Dict[str, Any] = field(default_factory=dict)
    #: ``(sieve, percent passing)`` where the sheet tabulates a grading.
    passing: List[Tuple[str, float]] = field(default_factory=list)
    #: One entry per row of a summary table.
    rows: List[Dict[str, Any]] = field(default_factory=list)
    #: The box of the plot, for the zoom tool's tests.
    plot_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _page():
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    return doc, page


def _text(page, x: float, y: float, text: str, size: float = 9.0,
          bold: bool = False) -> None:
    import pymupdf
    page.insert_text(pymupdf.Point(x, y), text, fontsize=size,
                     fontname="hebo" if bold else "helv")


def _grid(page, x0: float, y0: float, widths, rows,
          header: bool = True) -> Tuple[float, float, float, float]:
    """A ruled table. Returns its box.

    Ruled rather than merely aligned because a table detector is looking for
    lines, and a fixture that only sometimes reads as a table would make a
    test that only sometimes means anything.
    """
    import pymupdf
    height = 16.0
    xs = [x0]
    for width in widths:
        xs.append(xs[-1] + width)
    y = y0
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            _text(page, xs[c] + 3, y + 11, str(cell), 8.0,
                  bold=(header and r == 0))
        y += height
    for i, x in enumerate(xs):
        page.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y), width=0.6)
    for r in range(len(rows) + 1):
        page.draw_line(pymupdf.Point(xs[0], y0 + r * height),
                       pymupdf.Point(xs[-1], y0 + r * height), width=0.6)
    return (xs[0], y0, xs[-1], y)


def _plot(page, x0: float, y0: float, x1: float, y1: float,
          ticks_x, ticks_y, points) -> None:
    """A plot with labelled axes and a polyline, for the zoom tool to read."""
    import pymupdf
    page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), width=0.8)
    for value, fraction in ticks_x:
        x = x0 + (x1 - x0) * fraction
        page.draw_line(pymupdf.Point(x, y1), pymupdf.Point(x, y1 + 4), width=0.6)
        _text(page, x - 6, y1 + 13, str(value), 7.0)
    for value, fraction in ticks_y:
        y = y1 - (y1 - y0) * fraction
        page.draw_line(pymupdf.Point(x0 - 4, y), pymupdf.Point(x0, y), width=0.6)
        _text(page, x0 - 24, y + 3, str(value), 7.0)
    previous = None
    for fx, fy in points:
        point = pymupdf.Point(x0 + (x1 - x0) * fx, y1 - (y1 - y0) * fy)
        page.draw_circle(point, 2.0, width=0.8)
        if previous is not None:
            page.draw_line(previous, point, width=0.8)
        previous = point


def build_atterberg_sheet() -> LabSheetGT:
    """A plasticity chart with the limits printed in a box beside it."""
    doc, page = _page()
    _text(page, 180, 60, "ATTERBERG LIMITS RESULTS", 14, bold=True)
    _text(page, 180, 78, "ASTM D4318", 9)
    _text(page, 60, 110, "Boring: B-4")
    _text(page, 230, 110, "Sample: S-2")
    _text(page, 380, 110, "Depth: 7.5 ft")
    _text(page, 60, 126, "Tested: 14 March 2025")
    box = _grid(page, 60, 150, (110, 70, 70, 70),
                [("SAMPLE", "LL", "PL", "PI"),
                 ("B-4 S-2", "48", "22", "26")])
    _text(page, 60, 210, "Classification: LEAN CLAY (CL)")
    _text(page, 60, 226, "Natural water content: 27.4 %")
    _plot(page, 120, 300, 480, 520,
          [(0, 0.0), (20, 0.2), (40, 0.4), (60, 0.6), (80, 0.8)],
          [(0, 0.0), (20, 0.33), (40, 0.66), (60, 1.0)],
          [(0.2, 0.05), (0.48, 0.43), (0.8, 0.8)])
    _text(page, 250, 545, "PLASTICITY CHART", 9)
    pdf = doc.tobytes()
    doc.close()
    return LabSheetGT(
        pdf=pdf, kind="atterberg", depth_unit="ft",
        investigation_id="B-4", depth_top=7.5,
        values={"ll": 48.0, "pl": 22.0, "pi": 26.0, "uscs": "CL",
                "water_content": 27.4},
        plot_bbox=(120.0, 300.0, 480.0, 520.0))
    # (the grid's box is not returned: the reader is asked for values, not
    # for where the table was)


def build_gradation_sheet() -> LabSheetGT:
    """A grading curve with the percent-finer values tabulated beneath it.

    The sheet also prints Atterberg limits, because most real ones do: it is
    TWO tests on one specimen and a reader that returns one has lost half the
    page.
    """
    doc, page = _page()
    _text(page, 150, 60, "PARTICLE SIZE DISTRIBUTION", 14, bold=True)
    _text(page, 150, 78, "ASTM D6913 / D7928", 9)
    _text(page, 60, 104, "Boring: SB-11        Sample: 3        Depth: 12.0 ft")
    _plot(page, 110, 130, 500, 330,
          [(75, 0.0), (19, 0.25), (4.75, 0.5), (0.425, 0.75), (0.075, 1.0)],
          [(0, 0.0), (25, 0.25), (50, 0.5), (75, 0.75), (100, 1.0)],
          [(0.0, 1.0), (0.25, 0.98), (0.5, 0.87), (0.75, 0.61), (1.0, 0.43)])
    _text(page, 240, 352, "GRAIN SIZE IN MILLIMETRES", 8)
    _grid(page, 60, 375, (90, 90, 90),
          [("SIEVE", "SIZE (mm)", "PERCENT FINER"),
           ("3 in", "75", "100"),
           ("3/4 in", "19", "98"),
           ("No. 4", "4.75", "87"),
           ("No. 40", "0.425", "61"),
           ("No. 200", "0.075", "43")])
    _grid(page, 340, 375, (110, 80),
          [("PROPERTY", "VALUE"),
           ("Gravel (%)", "13"),
           ("Sand (%)", "44"),
           ("Fines (%)", "43"),
           ("Liquid limit", "31"),
           ("Plastic limit", "19"),
           ("Plasticity index", "12")])
    _text(page, 60, 520, "Classification: CLAYEY SAND (SC)")
    pdf = doc.tobytes()
    doc.close()
    return LabSheetGT(
        pdf=pdf, kind="gradation", depth_unit="ft",
        investigation_id="SB-11", depth_top=12.0,
        values={"gravel_percent": 13.0, "sand_percent": 44.0,
                "fines_percent": 43.0, "ll": 31.0, "pl": 19.0, "pi": 12.0,
                "uscs": "SC"},
        passing=[("3 in", 100.0), ("3/4 in", 98.0), ("No. 4", 87.0),
                 ("No. 40", 61.0), ("No. 200", 43.0)],
        plot_bbox=(110.0, 130.0, 500.0, 330.0))


def build_summary_table() -> LabSheetGT:
    """Four specimens from three borings, one row each."""
    doc, page = _page()
    _text(page, 140, 60, "SUMMARY OF LABORATORY TEST RESULTS", 13, bold=True)
    _text(page, 240, 78, "Sheet 1 of 2", 9)
    rows = [
        ("BORING", "SAMPLE", "DEPTH (m)", "WC (%)", "LL", "PL", "PI",
         "-200 (%)"),
        ("A-1", "S-1", "2.0", "18.2", "41", "20", "21", "72"),
        ("A-1", "S-4", "6.5", "24.6", "", "", "", "88"),
        ("A-2", "S-2", "3.5", "12.1", "28", "17", "11", "35"),
        ("A-3", "S-1", "1.5", "31.0", "55", "24", "31", "94"),
    ]
    _grid(page, 45, 110, (60, 60, 70, 60, 45, 45, 45, 65), rows)
    _text(page, 45, 230, "Tests by ASTM D2216, D4318 and D1140.")
    pdf = doc.tobytes()
    doc.close()
    return LabSheetGT(
        pdf=pdf, kind="summary_table", depth_unit="m",
        rows=[
            {"investigation_id": "A-1", "sample_id": "S-1", "depth_top": 2.0,
             "wc": 18.2, "ll": 41.0, "pl": 20.0, "pi": 21.0,
             "passing_200": 72.0},
            {"investigation_id": "A-1", "sample_id": "S-4", "depth_top": 6.5,
             "wc": 24.6, "passing_200": 88.0},
            {"investigation_id": "A-2", "sample_id": "S-2", "depth_top": 3.5,
             "wc": 12.1, "ll": 28.0, "pl": 17.0, "pi": 11.0,
             "passing_200": 35.0},
            {"investigation_id": "A-3", "sample_id": "S-1", "depth_top": 1.5,
             "wc": 31.0, "ll": 55.0, "pl": 24.0, "pi": 31.0,
             "passing_200": 94.0},
        ])


def build_certificate_page() -> LabSheetGT:
    """A list of the samples a laboratory received, and no result at all."""
    doc, page = _page()
    _text(page, 170, 60, "CERTIFICATE OF ANALYSIS", 14, bold=True)
    _text(page, 60, 90, "Laboratory order: 25A0123")
    _text(page, 60, 106, "Date received: 12 January 2025")
    _text(page, 60, 122, "Page 2 of 9")
    _text(page, 60, 150, "SAMPLE SUMMARY", 10, bold=True)
    _grid(page, 60, 170, (110, 100, 80, 90),
          [("CLIENT SAMPLE", "LAB ID", "MATRIX", "SAMPLED"),
           ("TP-3 at 0.5 m", "25A0123-01", "Solids", "10/01/2025"),
           ("TP-5 at 0.8 m", "25A0123-02", "Solids", "10/01/2025"),
           ("TP-9 at 1.2 m", "25A0123-03", "Solids", "10/01/2025")])
    _text(page, 60, 260, "Results for these samples are reported on the "
                         "pages that follow.")
    pdf = doc.tobytes()
    doc.close()
    return LabSheetGT(pdf=pdf, kind="other", depth_unit="m")

def build_water_content_run() -> bytes:
    """Three water content sheets, one specimen a page (RI-56).

    The first two pages print their own boring and depth; the third prints
    neither. Made-up values: 21.4, 17.9 and 25.0 %.
    """
    import pymupdf
    doc = pymupdf.open()
    for boring, depth, wc, dry in (("B-7", "3.0 ft", "21.4", "101.2"),
                                   ("B-8", "6.0 ft", "17.9", "106.5"),
                                   ("", "", "25.0", "98.3")):
        page = doc.new_page(width=LETTER[0], height=LETTER[1])
        _text(page, 180, 60, "MOISTURE CONTENT", 14, bold=True)
        if boring:
            _text(page, 60, 110, f"Boring: {boring}")
            _text(page, 380, 110, f"Depth: {depth}")
        _grid(page, 60, 150, (150, 80),
              [("Water content (%)", wc), ("Dry density (pcf)", dry)],
              header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_bare_run() -> bytes:
    """Three water content sheets that print no boring and no depth (RI-56).

    The first two hold one made-up value each (23.6 and 19.2 %); the third
    holds no value at all.
    """
    import pymupdf
    doc = pymupdf.open()
    for rows in ([("Water content, natural (%)", "23.6")],
                 [("Water content, natural (%)", "19.2")],
                 [("Remarks", "retained")]):
        page = doc.new_page(width=LETTER[0], height=LETTER[1])
        _text(page, 180, 60, "MOISTURE CONTENT", 14, bold=True)
        _grid(page, 60, 150, (170, 80), rows, header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_bare_empty_run() -> bytes:
    """Two Atterberg sheets that print no boring, no depth and no value: the
    first a remarks table, the second its title alone (RI-56)."""
    import pymupdf
    doc = pymupdf.open()
    for rows in ([("Remarks", "retained")], []):
        page = doc.new_page(width=LETTER[0], height=LETTER[1])
        _text(page, 180, 60, "ATTERBERG LIMITS", 14, bold=True)
        if rows:
            _grid(page, 60, 150, (170, 80), rows, header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_limits_trials() -> bytes:
    """A limits worksheet whose title is not read, with its trial rows: a
    mass of water in grams and the water contents of three trials (RI-56).
    Made-up values."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _text(page, 60, 110, "Boring: B-11")
    _text(page, 380, 110, "Depth: 4.0 ft")
    _grid(page, 60, 200, (170, 60, 60, 60),
          [("Moisture lost, mass (g)", "12.7", "14.1", "13.3"),
           ("Moisture content, trials (%)", "38.2", "41.5", "44.0"),
           ("Liquid limit (%)", "41", "", "")], header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_particle_density_page() -> bytes:
    """A particle density sheet whose tables hold no value (RI-56)."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _text(page, 160, 60, "Particle density by gas jar", 14, bold=True)
    _grid(page, 60, 150, (170, 80), [("Remarks", "retained")],
          header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_density_run() -> bytes:
    """Two sheets titled a density that print a compaction mould (RI-56).

    Each page prints its own boring and depth. Made-up values.
    """
    import pymupdf
    doc = pymupdf.open()
    for boring, depth, bulk, wc in (("TP-4", "1.0 m", "2.05", "14.2"),
                                    ("TP-5", "2.0 m", "2.11", "12.6")):
        page = doc.new_page(width=LETTER[0], height=LETTER[1])
        _text(page, 200, 60, "BULK DENSITY", 14, bold=True)
        _text(page, 60, 82, "Mould volume (cm3): 944")
        _text(page, 330, 82, "Rammer: 2.5 kg")
        _text(page, 60, 110, f"Boring: {boring}")
        _text(page, 380, 110, f"Depth: {depth}")
        _grid(page, 60, 150, (170, 80),
              [("Bulk density (Mg/m3)", bulk), ("Water content (%)", wc)],
              header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_grading_legend() -> bytes:
    """A grading sheet with its percent-finer table and the chart's legend
    table: a key letter, a boring number and a sample type (RI-71).
    Made-up values."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _text(page, 150, 60, "PARTICLE SIZE DISTRIBUTION", 14, bold=True)
    _text(page, 60, 104, "Boring: QB-6")
    _text(page, 380, 104, "Depth: 3.4 m")
    _grid(page, 60, 150, (90, 90),
          [("SIEVE", "PERCENT FINER"), ("No. 4", "96"), ("No. 200", "38")])
    _grid(page, 60, 260, (40, 60, 60),
          [("A", "7", "Pail"), ("C", "5", "Drum")], header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_field_density_sheet() -> bytes:
    """A field density sheet quoting the compaction it is measured against:
    the field test's own dry density and water content beside the
    compaction's maximum dry density and optimum (RI-71). Made-up values."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _text(page, 160, 60, "FIELD DENSITY, SAND CONE", 14, bold=True)
    _text(page, 60, 110, "Boring: QB-9")
    _text(page, 380, 110, "Depth: 0.65 m")
    _grid(page, 60, 150, (190, 80),
          [("Dry density (Mg/m3)", "1.794"), ("Water content (%)", "14.62"),
           ("Maximum dry density (Mg/m3)", "1.951"),
           ("Optimum water (%)", "12.38")], header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_compaction_trials() -> bytes:
    """A compaction sheet with its trial points in a table under a header
    row, one point a row, and its result in a second table (RI-71).
    Made-up values."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _text(page, 140, 60, "WATER-DENSITY RELATION (TRIALS)", 14, bold=True)
    _text(page, 60, 110, "Boring: QB-12")
    _text(page, 380, 110, "Depth: 1.35 m")
    _grid(page, 60, 150, (60, 130, 130),
          [("Point", "Water content (%)", "Dry density (Mg/m3)"),
           ("1", "8.72", "1.793"), ("2", "10.94", "1.862"),
           ("3", "13.18", "1.836")])
    _grid(page, 60, 300, (190, 80),
          [("Maximum dry density (Mg/m3)", "1.871"),
           ("Optimum water (%)", "11.46")], header=False)
    pdf = doc.tobytes()
    doc.close()
    return pdf


#: A made-up legend of four specimens (RI-149): each row's label, its
#: description and the percentages printed on its row. The values lines
#: are of different lengths on purpose: listed top to bottom as displayed,
#: a sheet scanned sideways gives them in the order of their lengths.
TURNED_LEGEND: Tuple[Tuple[str, str, str, Dict[str, float]], ...] = (
    ("2.0-3.3", "Silty fine SAND w/ Clay",
     "Silt =30%, SAND =62%, Clay =8%",
     {"silt_percent": 30.0, "sand_percent": 62.0, "clay_percent": 8.0}),
    ("5.0-6.3", "Gravelly coarse SAND w/ Silt",
     "SAND =47%, Gravel =21%, Silt =22%, Clay =10%",
     {"sand_percent": 47.0, "gravel_percent": 21.0, "silt_percent": 22.0,
      "clay_percent": 10.0}),
    ("8.0-9.3", "Clayey SILT w/ fine Sand",
     "Clay =35%, SILT =55%, Sand =10%",
     {"clay_percent": 35.0, "silt_percent": 55.0, "sand_percent": 10.0}),
    ("11.0-12.3", "Fine SAND w/ Silt",
     "SAND =88%, Silt =12%",
     {"sand_percent": 88.0, "silt_percent": 12.0}),
)
TURNED_HOLE = "QX-4"


#: Where a sheet turned by each quarter (counter-clockwise as displayed)
#: puts its top-left corner on a letter page: bottom-left for a quarter,
#: bottom-right upside down, top-right for three quarters.
TURNED_ORIGIN = {90: (24.0, 770.0), 180: (588.0, 770.0), 270: (588.0, 22.0)}


def _turned_text(page, origin: Tuple[float, float], skew: float, u: float,
                 v: float, text: str, size: float = 8.0,
                 angle: float = 0.0, quarter: int = 90) -> None:
    """``text`` placed at ``(u, v)`` on a sheet turned ``quarter`` degrees
    counter-clockwise and skewed ``skew`` degrees off it (``u`` along its
    rows, ``v`` down the sheet from its top), its own line drawn at
    ``angle`` (the sheet's own direction, ``quarter + skew``, by default):
    an optical reader gives a short line's direction a degree or so out."""
    import math

    import pymupdf
    theta = math.radians(quarter + skew)
    x = origin[0] + u * math.cos(theta) + v * math.sin(theta)
    y = origin[1] - u * math.sin(theta) + v * math.cos(theta)
    point = pymupdf.Point(x, y)
    page.insert_text(point, text, fontsize=size, fontname="helv",
                     morph=(point, pymupdf.Matrix(angle or quarter + skew)))


def build_turned_legend_page(doc, skew: float = -1.0, hole: str = TURNED_HOLE,
                             rows=TURNED_LEGEND, title: str =
                             "GRAIN SIZE DISTRIBUTION",
                             jitter: bool = True, quarter: int = 90,
                             label_codes=None) -> None:
    """One grading sheet scanned sideways onto a portrait page: its title,
    its hole, a chart's axis numbers and a legend of several specimens, the
    whole turned ``quarter`` degrees counter-clockwise (90 by default) and
    ``skew`` degrees further. With ``jitter`` each kind of line is drawn a
    little off the sheet's own direction, as an optical reader reports them.
    ``label_codes``: a code printed before each row's label (a hole's or a
    sample's), or None."""
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    origin = TURNED_ORIGIN[quarter]
    own = quarter + skew

    def put(u, v, text, size=8.0, off=0.0):
        _turned_text(page, origin, skew, u, v, text, size,
                     angle=own + (off if jitter else 0.0), quarter=quarter)

    put(220, 40, title, 12)
    put(40, 70, f"BORE HOLE NO. ({hole})", 9)
    put(40, 90, "PROJECT  Made-up Works, Sheet 1")
    for n, tick in enumerate(("0", "20", "40", "60", "80", "100")):
        put(30, 140 + 50 * n, tick, off=1.5)
    put(300, 410, "CUMULATIVE % FINER", off=0.4)
    for n, (label, words, values, _expected) in enumerate(rows):
        v = 470.0 + 13.0 * n
        if label_codes:
            label = f"{label_codes[n]} {label}"
        put(40, v, label, off=0.6)
        put(120, v, words)
        put(330, v, values, off=-0.3)


def build_turned_sheets(quarters=(90, 180, 270), skew: float = -0.8,
                        **kwargs) -> bytes:
    """One made-up grading sheet per quarter turn, each skewed ``skew``
    degrees further (RI-149)."""
    import pymupdf
    doc = pymupdf.open()
    for quarter in quarters:
        build_turned_legend_page(doc, skew=skew, quarter=quarter, **kwargs)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_portrait_form() -> bytes:
    """A made-up particle-size form printed upright with its chart turned:
    a portrait header (title, project, sample) over a landscape chart whose
    axis titles, size bands and ticks read bottom to top (RI-149). Its
    letters split about evenly between the two: a little more than half of
    them run turned on a page that is upright."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    for y, text in ((50, "PARTICLE SIZE DISTRIBUTION TEST REPORT"),
                    (70, "Project: Made-up Works, Phase 2"),
                    (84, "Client: Made-up Client"),
                    (98, "Location: QX-4   Depth: 2.0-3.3 m"),
                    (112, "Sample No.: S-1   Tested by: AB")):
        _text(page, 60, y, text, 9)

    def turned(x, y, text):
        point = pymupdf.Point(x, y)
        page.insert_text(point, text, fontsize=8, fontname="helv",
                         morph=(point, pymupdf.Matrix(90.0)))

    turned(80, 700, "PERCENT FINER BY WEIGHT")
    turned(560, 700, "GRAIN SIZE IN MILLIMETERS")
    turned(530, 700, "COBBLES  GRAVEL  SAND  SILT OR CLAY")
    turned(515, 700, "coarse  fine  coarse  medium  fine")
    turned(110, 700, "U.S. STANDARD SIEVE NUMBERS")
    for n, tick in enumerate(("100", "10", "1", "0.1", "0.01", "0.001")):
        turned(500, 690 - 80 * n, tick)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_turned_run(skews=(-1.0, 0.0)) -> bytes:
    """A summary of grading and Atterberg results and the grading sheets
    after it, all scanned sideways: page 0 the summary, then one grading
    sheet per skew (RI-149, RI-150). Made-up values."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    origin = (24.0, 770.0)
    for u, v, text, size in (
            (120, 40, "TABLE 2. GRAIN SIZE DISTRIBUTION & ATTERBERG LIMITS "
                      "TEST RESULTS", 12),
            (40, 90, f"BORE HOLE NO. ({TURNED_HOLE})", 9),
            (40, 120, "2.0-3.3   Silty fine SAND w/ Clay   30  62  8  NL  NP", 8),
            (40, 135, "5.0-6.3   Gravelly coarse SAND   21  47  22  10  NL", 8)):
        _turned_text(page, origin, -1.0, u, v, text, size)
    for skew in skews:
        build_turned_legend_page(doc, skew=skew)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_upright_legend_sheet() -> bytes:
    """The same legend on an upright sheet, each row on one printed line
    (RI-149: a legend is read the same way upright). Made-up values."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _text(page, 150, 60, "GRAIN SIZE DISTRIBUTION", 14, bold=True)
    _text(page, 60, 100, f"Bore hole: {TURNED_HOLE}")
    for n, (label, words, values, _expected) in enumerate(TURNED_LEGEND):
        y = 600.0 + 14.0 * n
        _text(page, 60, y, label, 8)
        _text(page, 120, y, words, 8)
        _text(page, 330, y, values, 8)
    pdf = doc.tobytes()
    doc.close()
    return pdf


def build_atterberg_run(n: int = 7) -> Tuple[bytes, List[Dict[str, Any]]]:
    """``n`` look-alike Atterberg sheets, one specimen a page (RI-116).

    Every page prints the same title, its own boring and depth, its limits
    as text lines and a flow curve PLOTTED with no table of its trials, so a
    reader has the curve only from the plot. Returns the PDF and, per page,
    what it prints. Made-up values.
    """
    import pymupdf
    doc = pymupdf.open()
    sheets: List[Dict[str, Any]] = []
    for i in range(n):
        ll, pl = 33.0 + 3.0 * i, 16.0 + i
        flow = [(16, round(ll + 2.1, 1)), (23, round(ll + 0.6, 1)),
                (31, round(ll - 1.0, 1))]
        sheet = {"boring": f"TB-{i + 1}", "depth_ft": 2.5 + 2.5 * i,
                 "ll": ll, "pl": pl, "flow_curve": flow}
        sheets.append(sheet)
        page = doc.new_page(width=LETTER[0], height=LETTER[1])
        _text(page, 180, 60, "ATTERBERG LIMITS", 14, bold=True)
        _text(page, 60, 110, f"Boring: {sheet['boring']}")
        _text(page, 380, 110, f"Depth: {sheet['depth_ft']} ft")
        _text(page, 60, 140, f"Liquid limit: {ll:.0f}")
        _text(page, 60, 156, f"Plastic limit: {pl:.0f}")
        _plot(page, 120, 300, 480, 520,
              [(10, 0.0), (20, 0.33), (30, 0.66), (40, 1.0)],
              [(round(ll - 5), 0.0), (round(ll + 5), 1.0)],
              [((b - 10) / 30.0, (w - (ll - 5)) / 10.0) for b, w in flow])
        _text(page, 250, 545, "FLOW CURVE", 9)
    pdf = doc.tobytes()
    doc.close()
    return pdf, sheets
