"""Synthetic vector-PDF fixtures with planted leaders + decoy geometry.

Builds a programmatic PDF (PyMuPDF) at a realistic sheet scale (900x700 pt)
containing N genuine leader callouts (bent shaft + filled-triangle arrowhead
+ tail text) plus decoy constructs that plausibly confuse a naive geometric
heuristic: a dimension line (extension lines + a dimension line with
arrowheads at BOTH ends + a value — geometrically near-identical to a leader
arrowhead), stray unlabelled lines, and an isolated small closed shape with
no nearby line-work (a marker symbol, not an arrowhead-on-a-shaft).

Every planted leader's ground truth is returned in the SAME coordinate frame
``drawing_ir.ingest.from_pdf_vector`` produces by default
(``origin="bottom_left"`` — page points, y flipped to increase upward), so a
test can compare directly to query results without re-deriving the flip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]

PAGE_WIDTH = 900.0
PAGE_HEIGHT = 700.0

#: Arrowhead geometry (PDF points) — a fixed shape/size used for every planted
#: leader and the dimension-decoy's tick arrows, so both are "small closed
#: triangles" and the recall/precision numbers reflect the heuristic's actual
#: discriminating power, not an artificially easy size gap.
ARROW_LENGTH = 12.0
ARROW_HALF_WIDTH = 4.0


@dataclass
class LeaderGT:
    """Ground truth for one planted leader (IR bottom_left coordinates)."""
    label: str
    tail_xy: Point          # shaft's far endpoint (text end)
    bend_xy: Point
    tip_xy: Point            # arrow tip == shaft's near endpoint
    text: str
    text_pos: Point          # text insertion point (IR coords)


def _to_ir(x: float, y: float) -> Point:
    """PDF top-left-origin point -> IR bottom_left-origin point (this fixture's page height)."""
    return (x, PAGE_HEIGHT - y)


def _unit(dx: float, dy: float) -> Point:
    m = math.hypot(dx, dy)
    return (dx / m, dy / m) if m > 1e-9 else (0.0, 0.0)


def _draw_shaft(shape, pts: List[Tuple[float, float]]) -> None:
    """An OPEN polyline shaft (explicit closePath=False — see DESIGN.md note:
    PyMuPDF's Shape.finish() defaults closePath=True even for plain
    line-work, which would make a shaft geometrically indistinguishable from
    a filled arrowhead; real CAD/plot PDF exporters normally do NOT close a
    stroke-only path, so tests use the realistic, explicit form)."""
    for a, b in zip(pts, pts[1:]):
        shape.draw_line(_pt(a), _pt(b))
    shape.finish(color=(0, 0, 0), width=1.0, closePath=False)
    shape.commit()


def _draw_filled_triangle(shape, apex, base1, base2) -> None:
    shape.draw_polyline([_pt(apex), _pt(base1), _pt(base2)])
    shape.finish(color=(0, 0, 0), fill=(0, 0, 0), closePath=True)
    shape.commit()


def _arrowhead_at(tip: Tuple[float, float], direction: Point,
                  length: float = ARROW_LENGTH,
                  half_width: float = ARROW_HALF_WIDTH):
    """Triangle vertices (apex, base1, base2) for an arrowhead at ``tip``
    pointing along ``direction`` (unit vector), in PDF (top-left) coords."""
    dx, dy = direction
    px, py = -dy, dx  # perpendicular
    back = (tip[0] - length * dx, tip[1] - length * dy)
    base1 = (back[0] + half_width * px, back[1] + half_width * py)
    base2 = (back[0] - half_width * px, back[1] - half_width * py)
    return tip, base1, base2


def _pt(xy):
    import fitz
    return fitz.Point(*xy)


def _plant_leader(page, shape, tail_pdf, bend_pdf, tip_pdf, label,
                  text_gap=6.0) -> LeaderGT:
    """Draw one leader (shaft + correctly-aligned arrowhead + tail text)."""
    _draw_shaft(shape, [tail_pdf, bend_pdf, tip_pdf])
    direction = _unit(tip_pdf[0] - bend_pdf[0], tip_pdf[1] - bend_pdf[1])
    apex, base1, base2 = _arrowhead_at(tip_pdf, direction)
    ashape = page.new_shape()
    _draw_filled_triangle(ashape, apex, base1, base2)

    # Tail text: placed just past the tail point, away from the shaft
    # direction, so its insertion point sits close to (not exactly on) the
    # tail — the realistic case text_anchored_geometry/find_leaders handle.
    shaft_in_dir = _unit(bend_pdf[0] - tail_pdf[0], bend_pdf[1] - tail_pdf[1])
    text_pdf = (tail_pdf[0] - text_gap * shaft_in_dir[0] - len(label) * 0.5,
               tail_pdf[1] - text_gap * shaft_in_dir[1] + 3.0)
    page.insert_text(_pt(text_pdf), label, fontsize=9)

    return LeaderGT(
        label=label,
        tail_xy=_to_ir(*tail_pdf),
        bend_xy=_to_ir(*bend_pdf),
        tip_xy=_to_ir(*tip_pdf),
        text=label,
        text_pos=_to_ir(*text_pdf),
    )


def _plant_dimension_decoy(page, x0, x1, y_ext_top, y_dim,
                           value="12'-6\"") -> Dict[str, Any]:
    """Extension lines + a dimension line with arrowheads at BOTH ends +
    a value — geometrically a near-twin of a leader arrowhead (small filled
    triangle at a line end). The intended false-positive stress case."""
    # Extension lines (vertical).
    for x in (x0, x1):
        s = page.new_shape()
        _draw_shaft(s, [(x, y_ext_top), (x, y_dim)])
    # Dimension line (horizontal), open shaft between the two ends.
    dim_shape = page.new_shape()
    _draw_shaft(dim_shape, [(x0, y_dim), (x1, y_dim)])
    # Arrowheads at both ends, pointing OUTWARD (toward each extension line).
    left_dir = _unit(x0 - x1, 0.0)
    right_dir = _unit(x1 - x0, 0.0)
    for tip, direction in ((( x0, y_dim), left_dir), ((x1, y_dim), right_dir)):
        apex, b1, b2 = _arrowhead_at(tip, direction)
        s = page.new_shape()
        _draw_filled_triangle(s, apex, b1, b2)
    mid = ((x0 + x1) / 2.0, y_dim - 8.0)
    page.insert_text(_pt(mid), value, fontsize=8)
    return {"x0": x0, "x1": x1, "y_dim": y_dim, "value": value}


def build_synthetic_leader_pdf(
    tmp_path, n_leaders: int = 3, include_decoys: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Build a synthetic vector-PDF sheet with planted leaders + decoys.

    Returns ``(pdf_path, ground_truth)`` where ``ground_truth`` is::

        {
          "page_size": (width, height),   # PDF points
          "leaders": [LeaderGT, ...],     # planted, in IR bottom_left coords
          "n_decoy_dimensions": int,
          "n_decoy_lines": int,
          "n_decoy_shapes": int,
        }
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    labels = ["SAE", "TYP", "SEE NOTE 3", "SAE", "MATCH LINE", "EOR"]
    leaders: List[LeaderGT] = []

    # Spread leaders across the sheet, each with a distinct bend geometry so
    # the fixture is not degenerate (varying shaft length/angle).
    layout = [
        # (tail, bend, tip)
        ((80, 100), (160, 140), (230, 120)),
        ((720, 620), (650, 560), (600, 600)),
        ((120, 600), (200, 560), (260, 610)),
        ((820, 150), (760, 210), (700, 170)),
        ((450, 640), (500, 590), (560, 630)),
        ((60, 380), (140, 400), (190, 360)),
    ]
    for i in range(n_leaders):
        tail, bend, tip = layout[i % len(layout)]
        # Offset each repeat cycle so leaders don't overlap when n>len(layout).
        cyc = i // len(layout)
        off = cyc * 15.0
        tail = (tail[0] + off, tail[1] + off)
        bend = (bend[0] + off, bend[1] + off)
        tip = (tip[0] + off, tip[1] + off)
        shape = page.new_shape()
        label = labels[i % len(labels)]
        leaders.append(_plant_leader(page, shape, tail, bend, tip, label))

    n_decoy_dimensions = n_decoy_lines = n_decoy_shapes = 0
    if include_decoys:
        # Dimension decoy (arrowheads at both ends — the honest confusable).
        _plant_dimension_decoy(page, 300, 500, 60, 90)
        n_decoy_dimensions = 1

        # Stray unlabelled lines (no arrowhead, no nearby text).
        for (a, b) in [((400, 300), (500, 260)), ((550, 400), (650, 450)),
                      ((300, 500), (350, 470))]:
            s = page.new_shape()
            _draw_shaft(s, [a, b])
            n_decoy_lines += 1

        # An isolated small closed shape (marker / north-arrow-ish symbol),
        # NOT near any line endpoint -> should be rejected for lack of a shaft.
        s = page.new_shape()
        _draw_filled_triangle(s, (850, 400), (840, 415), (860, 415))
        n_decoy_shapes += 1

        # A detail-bubble decoy: circle with a number inside (keynote-style).
        page.draw_circle(fitz.Point(500, 500), 12, color=(0, 0, 0))
        page.insert_text(fitz.Point(496, 504), "3", fontsize=8)

    path = str(tmp_path / "leader_sheet.pdf")
    doc.save(path)
    doc.close()

    return path, {
        "page_size": (PAGE_WIDTH, PAGE_HEIGHT),
        "leaders": leaders,
        "n_decoy_dimensions": n_decoy_dimensions,
        "n_decoy_lines": n_decoy_lines,
        "n_decoy_shapes": n_decoy_shapes,
    }
