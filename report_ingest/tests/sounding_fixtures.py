"""Synthetic test pits and soundings, with the answers stated by hand.

Five pages, drawn with PyMuPDF in the style of ``planlens.testing``, carrying
between them every shape this train has to read:

* a TEST PIT form -- a depth ruler, a description column, a sampling column
  and a header whose Equipment field names a bucket, which is the only place
  a pit log in the wild states how wide it is;
* a TABULATED CONE SOUNDING -- a header block and a plain column table of
  depth, qc, fs and u2, unruled, which is how the corpus prints one;
* a PLOTTED CONE SOUNDING -- axis titles with their units, tick labels, and
  traces that exist only as strokes, so the floor can read the AXES and
  nothing else and the reader must digitise;
* a TABULATED DYNAMIC PROBE -- blows over a fixed increment against depth,
  with a printed index column, and a header split over THREE printed lines
  the way the French sheets set theirs;
* a CALCULATION RUN of four pages carrying TWO program banners, for the item
  split.

Nothing here is copied from a real sheet. The wording is the trade's; the
numbers are made up and the answers are on the ground-truth objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LETTER = (612.0, 792.0)


@dataclass
class SoundingGT:
    """One synthetic sheet and what a reader should get off it."""

    pdf: bytes
    kind: str
    investigation_id: str = ""
    depth_unit: str = "m"
    #: ``depth -> {channel: value}`` for a sounding.
    series: Dict[float, Dict[str, float]] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    #: ``role -> (low, high, tick)`` for a plotted sheet.
    axes: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    #: A pit's plan size.
    dimensions: Dict[str, Any] = field(default_factory=dict)
    layers: List[Tuple[float, float, str, str]] = field(default_factory=list)
    samples: List[Tuple[str, float, float]] = field(default_factory=list)
    water: Optional[float] = None
    fields: Dict[str, str] = field(default_factory=dict)
    plot_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    n_pages: int = 1


def _page(doc=None):
    import fitz
    doc = doc if doc is not None else fitz.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    return doc, page


def _text(page, x: float, y: float, text: str, size: float = 9.0,
          bold: bool = False) -> None:
    import fitz
    page.insert_text(fitz.Point(x, y), text, fontsize=size,
                     fontname="hebo" if bold else "helv")


def _row(page, y: float, xs, cells, size: float = 8.0,
         bold: bool = False) -> None:
    for x, cell in zip(xs, cells):
        if cell is None:
            continue
        _text(page, x, y, str(cell), size, bold)


# ---------------------------------------------------------------------------
# a test pit
# ---------------------------------------------------------------------------

def build_test_pit() -> SoundingGT:
    """A pit form with a ruler, layers, samples and a bucket in its header."""
    import fitz

    doc, page = _page()
    _text(page, 60, 50, "TEST PIT LOG", 13, bold=True)
    _text(page, 430, 50, "Test Pit Number:", 9, bold=True)
    _text(page, 530, 50, "TP-9", 13, bold=True)
    header = [
        ("Contractor:", "Northmoor Plant Hire"),
        ("Equipment:", "Backhoe with 80 cm bucket"),
        ("Method:", "Test Pit"),
        ("Date Started:", "4 Mar 2024"),
        ("Ground Surface Elevation:", "104.2 (m)"),
        ("Total Depth:", "2.60 m"),
        ("Sheet:", "1 of 1"),
    ]
    y = 78.0
    for label, value in header:
        _text(page, 60, y, label, 9, bold=True)
        _text(page, 215, y, value, 9)
        y += 14.0

    # the ruled log
    top, bottom = 210.0, 610.0
    left, right = 60.0, 560.0
    page.draw_rect(fitz.Rect(left, top, right, bottom), width=0.8)
    for x in (110.0, 340.0, 400.0, 470.0):
        page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom), width=0.6)
    page.draw_line(fitz.Point(left, top + 26), fitz.Point(right, top + 26),
                   width=0.8)
    _text(page, 64, top + 18, "DEPTH (m)", 8, bold=True)
    _text(page, 170, top + 18, "MATERIAL DESCRIPTION", 8, bold=True)
    _text(page, 345, top + 18, "SYMBOL", 8, bold=True)
    _text(page, 405, top + 18, "SAMPLE", 8, bold=True)
    _text(page, 480, top + 18, "REMARKS", 8, bold=True)

    # the ruler: 0 m at y0, 2.6 m at y1, a tick every 0.2 m
    y0, y1 = top + 40.0, bottom - 20.0
    per_m = (y1 - y0) / 2.6

    def at(depth: float) -> float:
        return y0 + depth * per_m

    depth = 0.0
    while depth <= 2.6001:
        page.draw_line(fitz.Point(left + 2, at(depth)),
                       fitz.Point(left + 12, at(depth)), width=0.5)
        if abs(depth * 5 - round(depth * 5)) < 1e-6 and \
                abs(depth - round(depth, 1)) < 1e-6 and \
                abs((depth * 10) % 5) < 1e-6:
            _text(page, left + 16, at(depth) + 3, f"{depth:.2f}", 7)
        depth = round(depth + 0.2, 2)

    layers = [
        (0.00, 0.30, "TOPSOIL; dark brown, moist", ""),
        (0.30, 1.10, "SILTY SAND; brown, moist, fine grained", "SM"),
        (1.10, 2.60, "LEAN CLAY; grey, moist, stiff", "CL"),
    ]
    for lo, hi, description, symbol in layers:
        page.draw_line(fitz.Point(110.0, at(lo)), fitz.Point(400.0, at(lo)),
                       width=0.6)
        _text(page, 118, at(lo) + 14, description, 8)
        if symbol:
            _text(page, 355, at((lo + hi) / 2.0), symbol, 8)
    page.draw_line(fitz.Point(110.0, at(2.60)), fitz.Point(400.0, at(2.60)),
                   width=0.6)

    samples = [("B-1", 0.40, 0.60), ("B-2", 1.40, 1.60)]
    for name, lo, hi in samples:
        page.draw_rect(fitz.Rect(408, at(lo), 428, at(hi)), width=0.6)
        _text(page, 432, at(lo) + 8, name, 8)
    _text(page, 476, at(2.30), "Bottom of test pit at 2.60 m", 8)

    return SoundingGT(
        pdf=doc.tobytes(), kind="test_pit", investigation_id="TP-9",
        dimensions={"width": 80.0, "unit": "cm"},
        layers=[(lo, hi, d, s) for lo, hi, d, s in layers],
        samples=samples,
        fields={"test_pit_id": "TP-9", "total_depth": "2.60 m",
                "drilling_equipment": "Backhoe with 80 cm bucket"})


# ---------------------------------------------------------------------------
# a tabulated cone sounding
# ---------------------------------------------------------------------------

#: The series the tabulated sheet prints, ``depth -> (qc, fs, u2)``.
_CPT_ROWS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.20, 1.80, 0.02, 12.0),
    (0.40, 3.40, 0.04, 18.0),
    (0.60, 6.15, 0.07, 24.0),
    (0.80, 9.20, 0.11, 31.0),
    (1.00, 12.55, 0.14, 38.0),
    (1.20, 10.90, 0.12, 44.0),
    (1.40, 8.35, 0.09, 51.0),
    (1.60, 7.10, 0.08, 57.0),
    (1.80, 5.65, 0.06, 63.0),
    (2.00, 4.20, 0.05, 70.0),
)


def build_tabulated_cpt() -> SoundingGT:
    """A cone sounding whose values are printed as an unruled column table."""
    doc, page = _page()
    _text(page, 60, 52, "CPT-4", 14, bold=True)
    for n, (label, value) in enumerate((
            ("Project number:", "2024-0117"),
            ("Reference level:", "mASL"),
            ("Date:", "04-03-24"),
            ("Ground level:", "31.40"),
            ("Cone no.:", "S15CFIIP.09912"),
            ("Test according:", "EN ISO 22476-1"))):
        _text(page, 60, 78 + n * 14, label, 9)
        _text(page, 175, 78 + n * 14, value, 9)

    xs = (80.0, 175.0, 275.0, 375.0)
    _row(page, 200, xs, ("Depth (m)", "qc (MPa)", "fs (MPa)", "u2 (kPa)"),
         9, bold=True)
    y = 216.0
    for depth, qc, fs, u2 in _CPT_ROWS:
        _row(page, y, xs, (f"{depth:.2f}", f"{qc:.2f}", f"{fs:.2f}",
                           f"{u2:.0f}"))
        y += 13.0
    return SoundingGT(
        pdf=doc.tobytes(), kind="cpt", investigation_id="CPT-4",
        series={d: {"qc": qc, "fs": fs, "u2": u2}
                for d, qc, fs, u2 in _CPT_ROWS},
        units={"qc": "MPa", "fs": "MPa", "u2": "kPa", "depth": "m"},
        fields={"cone": "S15CFIIP.09912", "standard": "EN ISO 22476-1"})


# ---------------------------------------------------------------------------
# a plotted cone sounding
# ---------------------------------------------------------------------------

def build_plotted_cpt() -> SoundingGT:
    """A cone sounding drawn as traces: axis titles, ticks and strokes.

    The floor can read the AXES off this and NOTHING else, which is the whole
    point of it: it is the sheet that makes the reader digitise.
    """
    import fitz

    doc, page = _page()
    _text(page, 60, 50, "CPT-7", 14, bold=True)
    _text(page, 60, 70, "Ground level: 18.90", 9)
    _text(page, 60, 84, "Cone no.: S10CFIP.44107", 9)

    left, right = 120.0, 520.0
    top, bottom = 140.0, 640.0
    page.draw_rect(fitz.Rect(left, top, right, bottom), width=0.9)

    _text(page, 210, top - 26, "Cone resistance (qc) in MPa", 9, bold=True)
    for n, value in enumerate((5, 10, 15, 20, 25)):
        x = left + (n + 1) * (right - left) / 6.0
        _text(page, x - 6, top - 8, str(value), 8)
        page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom),
                       width=0.3, color=(0.8, 0.8, 0.8))

    _text(page, 210, bottom + 26, "Sleeve friction (fs) in MPa", 9, bold=True)
    for n, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        x = left + (n + 1) * (right - left) / 6.0
        _text(page, x - 9, bottom + 12, f"{value:.1f}", 8)

    _text(page, 40, 400, "Depth (m)", 9, bold=True)
    for n in range(0, 11):
        y = top + n * (bottom - top) / 10.0
        _text(page, left - 26, y + 3, str(n), 8)
        page.draw_line(fitz.Point(left, y), fitz.Point(right, y),
                       width=0.3, color=(0.8, 0.8, 0.8))

    # the traces, as strokes and nothing else
    points = [(left + 20 + 14 * n % 180, top + n * 5.0) for n in range(100)]
    for a, b in zip(points, points[1:]):
        page.draw_line(fitz.Point(*a), fitz.Point(*b), width=0.7,
                       color=(0, 0, 1))
    return SoundingGT(
        pdf=doc.tobytes(), kind="cpt", investigation_id="CPT-7",
        axes={"qc": (5.0, 25.0, 5.0), "fs": (0.1, 0.5, 0.1),
              "depth": (0.0, 10.0, 1.0)},
        units={"qc": "MPa", "fs": "MPa", "depth": "m"},
        plot_bbox=(left, top, right, bottom))


# ---------------------------------------------------------------------------
# a tabulated dynamic probe
# ---------------------------------------------------------------------------

#: ``depth -> (blows, index)`` for the dynamic probe.
_DCP_ROWS: Tuple[Tuple[float, int, float], ...] = (
    (0.20, 4, 21.5), (0.40, 3, 16.1), (0.60, 5, 26.8), (0.80, 7, 37.6),
    (1.00, 9, 48.3), (1.20, 12, 64.4), (1.40, 15, 80.5), (1.60, 22, 118.1),
    (1.80, 31, 166.4), (2.00, 50, 268.4),
)


def build_tabulated_dcp() -> SoundingGT:
    """A dynamic probe whose column header runs over THREE printed lines."""
    doc, page = _page()
    _text(page, 60, 50, "DYNAMIC PROBE RECORD", 12, bold=True)
    for n, (label, value) in enumerate((
            ("Client:", "Northmoor Plant Hire"),
            ("Borehole / Date:", "DP-3 / 04-03-24"),
            ("Equipment:", "Dynamic penetrometer, 63.5 kg hammer"))):
        _text(page, 60, 76 + n * 14, label, 9)
        _text(page, 175, 76 + n * 14, value, 9)

    xs = (80.0, 185.0, 300.0)
    # The header, set over three printed lines the way the French sheets do.
    _row(page, 150, xs, ("Depth", "Number of", "Dynamic"), 9, bold=True)
    _row(page, 163, xs, (None, "blows", "resistance"), 9, bold=True)
    _row(page, 176, xs, ("[m]", None, "(MPa)"), 9, bold=True)
    y = 194.0
    for depth, blows, index in _DCP_ROWS:
        _row(page, y, xs, (f"{depth:.2f}", str(blows), f"{index:.1f}"))
        y += 13.0
    _text(page, 80, y + 14, "Refusal at 2.00 m", 8)
    return SoundingGT(
        pdf=doc.tobytes(), kind="dcp", investigation_id="DP-3",
        series={d: {"blows": float(b), "index": i} for d, b, i in _DCP_ROWS},
        units={"index": "MPa", "depth": "m"},
        fields={"hammer": "Dynamic penetrometer, 63.5 kg hammer"})


# ---------------------------------------------------------------------------
# a calculation run with two banners
# ---------------------------------------------------------------------------

def build_two_banner_run() -> bytes:
    """Four pages: two printouts of two pages each, back to back.

    Page 1 opens with an LPILE banner and page 3 with a STABL one, and the
    running header changes with them -- which is the pattern a calculation
    appendix prints and the one the item grouping used to miss.
    """
    doc = None
    for n, (banner, title, rows) in enumerate((
            ("LPILE Version 2019.11.3", "LATERAL PILE ANALYSIS",
             (("Pile diameter", "0.914", "m"),
              ("Head shear", "220.0", "kN"))),
            (None, "LATERAL PILE ANALYSIS",
             (("Maximum moment", "1180.0", "kN-m"),
              ("Head deflection", "0.0210", "m"))),
            ("STABL 5M", "SLOPE STABILITY ANALYSIS",
             (("Slope height", "8.50", "m"),
              ("Unit weight", "19.2", "kN/m3"))),
            (None, "SLOPE STABILITY ANALYSIS",
             (("Minimum factor of safety", "1.42", ""),
              ("Critical circle radius", "14.80", "m"))))):
        doc, page = _page(doc)
        y = 54.0
        if banner:
            _text(page, 60, y, banner, 10, bold=True)
        # The running header sits at the same place on every page of one
        # printout, which is what makes it a running header.
        _text(page, 60, 78.0, title, 10, bold=True)
        _text(page, 60, 96.0, f"Sheet {n + 1}", 8)
        y = 220.0
        for label, value, unit in rows:
            _text(page, 60, y, f"{label} = {value} {unit}".strip(), 9)
            y += 15.0
    return doc.tobytes()
