"""Synthetic calculation printouts, with the answers stated by hand.

Three pages, drawn with PyMuPDF in the style of ``planlens.testing``,
carrying between them the three shapes a calculation printout takes:

* a SPREADSHEET printed to PDF -- a title block, a project block whose
  labels and values are separate text runs on one printed line, a ruled
  table of inputs with the unit in the column heading, and a table whose
  LAST filled row is the answer;
* a PROGRAM PRINTOUT -- a banner naming the program and its version,
  fixed-width ``label = value unit`` lines, a block of echoed data rows that
  name nothing, and a summary;
* a PLOTTED result -- a section with its factor of safety written on it in a
  box and nowhere else, which is what a slope-stability program prints.

Nothing here is copied from a real printout. The wording is the trade's; the
numbers are made up and the answers are in :class:`CalcPageGT`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

LETTER = (612.0, 792.0)


@dataclass
class CalcPageGT:
    """One synthetic printout and what a reader should get off it."""

    pdf: bytes
    kind: str
    program: str = ""
    method: str = ""
    subject: str = ""
    #: ``label -> (value, unit)`` for what the pages state as GIVEN.
    inputs: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    #: The same, for what they state as WORKED OUT.
    results: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    #: The box of the plot, for the zoom tool's tests.
    plot_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    n_pages: int = 1


def _page(doc=None):
    import pymupdf
    doc = doc if doc is not None else pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    return doc, page


def _text(page, x: float, y: float, text: str, size: float = 9.0,
          bold: bool = False) -> None:
    import pymupdf
    page.insert_text(pymupdf.Point(x, y), text, fontsize=size,
                     fontname="hebo" if bold else "helv")


def _pair(page, y: float, label: str, value: str, x_label: float = 60.0,
          x_value: float = 230.0, size: float = 9.0) -> None:
    """A label and its value as SEPARATE runs on one printed line.

    Which is how a spreadsheet sets them, and the reason the floor groups
    text spans into bands before it looks for a pair.
    """
    _text(page, x_label, y, label, size)
    _text(page, x_value, y, value, size)


def _grid(page, x0: float, y0: float, widths, rows,
          header: bool = True) -> Tuple[float, float, float, float]:
    """A ruled table. Returns its box."""
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
    for x in xs:
        page.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y), width=0.6)
    for r in range(len(rows) + 1):
        page.draw_line(pymupdf.Point(xs[0], y0 + r * height),
                       pymupdf.Point(xs[-1], y0 + r * height), width=0.6)
    return (xs[0], y0, xs[-1], y)


def build_settlement_spreadsheet() -> CalcPageGT:
    """A settlement spreadsheet: a project block, inputs, and a total row."""
    doc, page = _page()
    _text(page, 150, 56, "Spread Footing Settlement Calculation", 13,
          bold=True)
    _text(page, 150, 74, "Schmertmann Strain Influence Method", 10)

    _pair(page, 108, "Project Name", "Riverbend Logistics Park")
    _pair(page, 122, "Job Number", "24-0117")
    _pair(page, 136, "Description", "Column footing F-3, bearing on new fill")
    _pair(page, 150, "Calculated By", "AJP")

    _grid(page, 330, 96, (150, 70),
          [("Foundation Details", ""),
           ("Footing Width B (ft)", "8.5"),
           ("Footing Length L (ft)", "8.5"),
           ("Footing Depth (ft)", "3"),
           ("Footing Bearing Pressure (ksf)", "4.00"),
           ("Depth to Water Table (ft)", "19.2")])

    _grid(page, 60, 260, (50, 90, 90, 110, 110),
          [("Layer", "Bottom (ft)", "Modulus (tsf)",
            "Layer Settlement (inches)",
            "Total Cumulative Settlement (inches)"),
           ("1", "9", "180", "0.31", "0.31"),
           ("2", "16", "420", "0.14", "0.45"),
           ("3", "28", "900", "0.07", "0.52"),
           ("4", "", "", "", ""),
           ("5", "", "", "", "")])

    _text(page, 60, 400, "Estimated total settlement = 0.52 inches")
    _text(page, 60, 416, "Differential settlement = 0.26 inches")
    _text(page, 60, 760, "Page 1 of 1", 8)
    pdf = doc.tobytes()
    doc.close()
    return CalcPageGT(
        pdf=pdf, kind="settlement", program="",
        method="Schmertmann Strain Influence Method",
        subject="Column footing F-3, bearing on new fill",
        inputs={"Footing Width B (ft)": (8.5, "ft"),
                "Footing Length L (ft)": (8.5, "ft"),
                "Footing Depth (ft)": (3.0, "ft"),
                "Footing Bearing Pressure (ksf)": (4.0, "ksf"),
                "Depth to Water Table (ft)": (19.2, "ft")},
        results={"Total Cumulative Settlement (inches)": (0.52, "in"),
                 "Differential settlement": (0.26, "in")})


def build_program_printout() -> CalcPageGT:
    """A program's own output: a banner, labelled lines, and data rows."""
    doc, page = _page()
    _text(page, 60, 46, "PILEWORKS 2024  Version 11.2.3", 11, bold=True)
    _text(page, 60, 62, "Analysis of Laterally Loaded Piles", 9)
    _text(page, 60, 88, "Project: Riverbend Logistics Park", 9)
    _text(page, 60, 102, "Load case: Service, fixed head", 9)

    lines = [
        "Pile diameter                    =     0.610 m",
        "Pile length                      =    18.000 m",
        "Modulus of elasticity            = 25000000.0 kPa",
        "Applied lateral load             =   240.000 kN",
        "Number of iterations             =         7",
        "Pile-head deflection             =     0.01840 m",
        "Maximum bending moment           =   612.400 kN-m",
        "Maximum shear force              =   240.000 kN",
        "Depth to maximum moment          =     2.750 m",
    ]
    y = 130
    for line in lines:
        _text(page, 60, y, line, 8.5)
        y += 13

    _text(page, 60, 270, "DEPTH   DEFLECTION   MOMENT   SHEAR", 8.5)
    y = 284
    for depth in range(1, 13):
        _text(page, 60,  y,
              f"{depth * 0.5:6.2f} {0.0184 / depth:12.5f} "
              f"{612.4 / depth:9.2f} {240.0 / depth:8.2f}", 8.5)
        y += 12
    _text(page, 60, 760, "Page 1 of 4", 8)
    pdf = doc.tobytes()
    doc.close()
    return CalcPageGT(
        pdf=pdf, kind="lateral_pile", program="PILEWORKS 2024 Version 11.2.3",
        method="p-y analysis", subject="Service, fixed head",
        inputs={"Pile diameter": (0.610, "m"),
                "Pile length": (18.0, "m"),
                "Applied lateral load": (240.0, "kN")},
        results={"Pile-head deflection": (0.0184, "m"),
                 "Maximum bending moment": (612.4, "kN-m"),
                 "Depth to maximum moment": (2.75, "m")})


def build_plotted_result() -> CalcPageGT:
    """A slope section whose factor of safety is printed ON the drawing."""
    import pymupdf
    doc, page = _page()
    _text(page, 200, 46, "SLOPE STABILITY - SECTION A-A", 12, bold=True)
    box = (80.0, 90.0, 520.0, 430.0)
    page.draw_rect(pymupdf.Rect(*box), width=0.8)
    page.draw_line(pymupdf.Point(90, 400), pymupdf.Point(260, 400), width=1.0)
    page.draw_line(pymupdf.Point(260, 400), pymupdf.Point(400, 250), width=1.0)
    page.draw_line(pymupdf.Point(400, 250), pymupdf.Point(510, 250), width=1.0)
    page.draw_circle(pymupdf.Point(340, 160), 3.0, width=0.8)
    page.draw_rect(pymupdf.Rect(320, 130, 372, 150), width=0.7)
    _text(page, 326, 145, "1.478", 9)
    for i, value in enumerate(("0", "10", "20", "30", "40")):
        _text(page, 90 + i * 105, 445, value, 7.0)
    _text(page, 240, 460, "DISTANCE (m)", 8.0)

    _grid(page, 80, 480, (130, 80, 70, 70),
          [("Material", "Unit weight (kN/m3)", "Cohesion (kPa)", "Phi (deg)"),
           ("Fill", "19.5", "0", "32"),
           ("Residual soil", "18.0", "8", "28")])
    _text(page, 80, 600, "SLOPEWORKS 8.1", 8)
    pdf = doc.tobytes()
    doc.close()
    return CalcPageGT(
        pdf=pdf, kind="slope_stability", program="SLOPEWORKS 8.1",
        method="limit equilibrium", subject="SECTION A-A",
        inputs={"Fill unit weight": (19.5, "kN/m3"),
                "Residual soil phi": (28.0, "deg")},
        results={"Factor of safety": (1.478, "")},
        plot_bbox=box)


def build_long_printout(n_pages: int = 16) -> CalcPageGT:
    """A printout longer than one window: banner first, the answer LAST.

    The middle pages carry a load case each and nothing a reader needs, so a
    window cut off the back of it would miss the summary -- which is the
    reason the window always keeps the first page and the last.
    """
    import pymupdf
    doc = pymupdf.open()
    for n in range(n_pages):
        _doc, page = _page(doc)
        _text(page, 60, 46, "PILEWORKS 2024  Version 11.2.3", 11, bold=True)
        if n == 0:
            _text(page, 60, 70,
                  "Pile diameter                    =     0.610 m", 9)
            _text(page, 60, 84,
                  "Applied lateral load             =   240.000 kN", 9)
        elif n == n_pages - 1:
            _text(page, 60, 70,
                  "Maximum bending moment           =   612.400 kN-m", 9)
            _text(page, 60, 84,
                  "Governing load case              =         3", 9)
        else:
            _text(page, 60, 70, f"Output Summary for Load Case No. {n}:", 9)
            _text(page, 60, 84,
                  f"Pile-head deflection             =     0.0{n:03d} m", 9)
        _text(page, 60, 760, f"Page {n + 1} of {n_pages}", 8)
    pdf = doc.tobytes()
    doc.close()
    return CalcPageGT(
        pdf=pdf, kind="lateral_pile", program="PILEWORKS 2024 Version 11.2.3",
        inputs={"Pile diameter": (0.610, "m"),
                "Applied lateral load": (240.0, "kN")},
        results={"Maximum bending moment": (612.4, "kN-m")},
        n_pages=n_pages)


def build_two_page_printout() -> CalcPageGT:
    """One printout over two pages: the banner first, the answer last."""
    import pymupdf
    doc = pymupdf.open()
    _doc, first = _page(doc)
    _text(first, 60, 46, "PILEWORKS 2024  Version 11.2.3", 11, bold=True)
    _text(first, 60, 70, "Pile diameter                    =     0.610 m", 9)
    _text(first, 60, 84, "Applied lateral load             =   240.000 kN", 9)
    _text(first, 60, 760, "Page 1 of 2", 8)
    _doc, last = _page(doc)
    _text(last, 60, 46, "PILEWORKS 2024  Version 11.2.3", 11, bold=True)
    _text(last, 60, 70, "Maximum bending moment           =   612.400 kN-m", 9)
    _text(last, 60, 760, "Page 2 of 2", 8)
    pdf = doc.tobytes()
    doc.close()
    return CalcPageGT(
        pdf=pdf, kind="lateral_pile", program="PILEWORKS 2024 Version 11.2.3",
        inputs={"Pile diameter": (0.610, "m"),
                "Applied lateral load": (240.0, "kN")},
        results={"Maximum bending moment": (612.4, "kN-m")},
        n_pages=2)


def build_repeated_label_sheet() -> CalcPageGT:
    """ONE page carrying three blocks of the same worked sheet, one per
    material, each printing the same labels with its own numbers.

    The shape RI-17 is about: a label printed three times on one page is
    three values, and a merge that keyed its slots by the label alone let
    one block's reading replace the other two.
    """
    doc, page = _page()
    _text(page, 150, 46, "Bearing Capacity by the Meyerhof Method", 12,
          bold=True)
    blocks = (("Material A", "30", "30.14", "18.40"),
              ("Material B", "20", "14.83", "6.40"),
              ("Material C", "36", "50.59", "37.75"))
    y = 90.0
    for name, phi, nc, nq in blocks:
        _text(page, 60, y, name, 10, bold=True)
        _text(page, 60, y + 16, f"Phi = {phi} degrees", 9)
        _text(page, 300, y + 16, f"Nc = {nc}", 9)
        _text(page, 300, y + 30, f"Nq = {nq}", 9)
        y += 150.0
    pdf = doc.tobytes()
    doc.close()
    return CalcPageGT(
        pdf=pdf, kind="shallow_foundation_bearing",
        method="Meyerhof",
        results={f"Nc, {name}": (float(nc), "")
                 for name, _phi, nc, _nq in blocks})


def build_repeated_label_run() -> CalcPageGT:
    """Three pages of one printout, each printing the SAME label with its
    own value -- a head load in each page's legend, for three pile sizes."""
    import pymupdf
    doc = pymupdf.open()
    loads = ("12.5", "40", "95")
    for n, load in enumerate(loads):
        _doc, page = _page(doc)
        _text(page, 60, 46, f"Lateral Analysis of Pile P-{n + 1}", 11,
              bold=True)
        _text(page, 60, 300, f"Head load = {load} kN", 9)
    pdf = doc.tobytes()
    doc.close()
    return CalcPageGT(
        pdf=pdf, kind="lateral_pile",
        inputs={f"Head load, Pile P-{n + 1}": (float(v), "kN")
                for n, v in enumerate(loads)},
        n_pages=len(loads))


def build_titled_pages(heads) -> bytes:
    """Pages whose top bands are given: ``(banner or None, [lines])`` each.

    For the run splitter, which reads only a page's head. Every page also
    carries one labelled value in its body, as a calculation page does.
    """
    doc = None
    for n, (banner, lines) in enumerate(heads):
        doc, page = _page(doc)
        y = 46.0
        if banner:
            _text(page, 60, y, banner, 10, bold=True)
            y += 20.0
        for line in lines:
            _text(page, 60, y, line, 10)
            y += 16.0
        _text(page, 60, 400, f"Value {n + 1} = {n + 1}.5 m", 9)
    pdf = doc.tobytes()
    doc.close()
    return pdf
