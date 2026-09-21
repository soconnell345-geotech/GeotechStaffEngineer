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
    import fitz
    doc = doc if doc is not None else fitz.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    return doc, page


def _text(page, x: float, y: float, text: str, size: float = 9.0,
          bold: bool = False) -> None:
    import fitz
    page.insert_text(fitz.Point(x, y), text, fontsize=size,
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
    for x in xs:
        page.draw_line(fitz.Point(x, y0), fitz.Point(x, y), width=0.6)
    for r in range(len(rows) + 1):
        page.draw_line(fitz.Point(xs[0], y0 + r * height),
                       fitz.Point(xs[-1], y0 + r * height), width=0.6)
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
    import fitz
    doc, page = _page()
    _text(page, 200, 46, "SLOPE STABILITY - SECTION A-A", 12, bold=True)
    box = (80.0, 90.0, 520.0, 430.0)
    page.draw_rect(fitz.Rect(*box), width=0.8)
    page.draw_line(fitz.Point(90, 400), fitz.Point(260, 400), width=1.0)
    page.draw_line(fitz.Point(260, 400), fitz.Point(400, 250), width=1.0)
    page.draw_line(fitz.Point(400, 250), fitz.Point(510, 250), width=1.0)
    page.draw_circle(fitz.Point(340, 160), 3.0, width=0.8)
    page.draw_rect(fitz.Rect(320, 130, 372, 150), width=0.7)
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
    import fitz
    doc = fitz.open()
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
    import fitz
    doc = fitz.open()
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
