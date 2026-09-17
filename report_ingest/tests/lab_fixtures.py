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
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    return doc, page


def _text(page, x: float, y: float, text: str, size: float = 9.0,
          bold: bool = False) -> None:
    import fitz
    page.insert_text(fitz.Point(x, y), text, fontsize=size,
                     fontname="hebo" if bold else "helv")


def _grid(page, x0: float, y0: float, widths, rows,
          header: bool = True) -> Tuple[float, float, float, float]:
    """A ruled table. Returns its box.

    Ruled rather than merely aligned because a table detector is looking for
    lines, and a fixture that only sometimes reads as a table would make a
    test that only sometimes means anything.
    """
    import fitz
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
        page.draw_line(fitz.Point(x, y0), fitz.Point(x, y), width=0.6)
    for r in range(len(rows) + 1):
        page.draw_line(fitz.Point(xs[0], y0 + r * height),
                       fitz.Point(xs[-1], y0 + r * height), width=0.6)
    return (xs[0], y0, xs[-1], y)


def _plot(page, x0: float, y0: float, x1: float, y1: float,
          ticks_x, ticks_y, points) -> None:
    """A plot with labelled axes and a polyline, for the zoom tool to read."""
    import fitz
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), width=0.8)
    for value, fraction in ticks_x:
        x = x0 + (x1 - x0) * fraction
        page.draw_line(fitz.Point(x, y1), fitz.Point(x, y1 + 4), width=0.6)
        _text(page, x - 6, y1 + 13, str(value), 7.0)
    for value, fraction in ticks_y:
        y = y1 - (y1 - y0) * fraction
        page.draw_line(fitz.Point(x0 - 4, y), fitz.Point(x0, y), width=0.6)
        _text(page, x0 - 24, y + 3, str(value), 7.0)
    previous = None
    for fx, fy in points:
        point = fitz.Point(x0 + (x1 - x0) * fx, y1 - (y1 - y0) * fy)
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
