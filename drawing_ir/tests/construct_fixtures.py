"""Synthetic fixtures for the Phase-2 composition family.

Companion to :mod:`leader_fixtures` (which owns the leader sheet and the
shared drawing helpers reused here): one builder per construct family —
dimensions, title block, bubble callouts, revision clouds — each planting
genuine constructs PLUS decoys chosen to stress the heuristics' actual
discriminating power (a true leader on the dimension sheet, a mid-sheet
box on the title-block sheet, unlabelled/oversized circles on the bubble
sheet, a plain rectangle and a smooth circle on the cloud sheet).

Ground truth is returned in the IR ``bottom_left`` frame (same convention
as leader_fixtures) so tests compare directly to query results.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from drawing_ir.tests.leader_fixtures import (
    PAGE_HEIGHT, PAGE_WIDTH, _arrowhead_at, _draw_filled_triangle,
    _draw_shaft, _plant_leader, _pt, _to_ir, _unit,
)

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def _plant_dimension(page, p_a: Point, p_b: Point, value: str,
                     ext_len: float = 30.0, text_offset: float = 8.0
                     ) -> Dict[str, Any]:
    """A full dimension at any orientation: extension lines perpendicular to
    the shaft at both ends, arrowheads at both ends pointing outward, and the
    value text near the shaft midpoint."""
    import fitz  # noqa: F401  (fixture helpers need fitz loaded)

    axis = _unit(p_b[0] - p_a[0], p_b[1] - p_a[1])
    perp = (-axis[1], axis[0])
    # Extension lines: perpendicular, crossing the dimension line slightly.
    for p in (p_a, p_b):
        s = page.new_shape()
        _draw_shaft(s, [(p[0] - perp[0] * 5, p[1] - perp[1] * 5),
                        (p[0] + perp[0] * ext_len, p[1] + perp[1] * ext_len)])
    # The dimension shaft.
    s = page.new_shape()
    _draw_shaft(s, [p_a, p_b])
    # Arrowheads at both ends, pointing outward.
    for tip, direction in ((p_a, _unit(p_a[0] - p_b[0], p_a[1] - p_b[1])),
                           (p_b, axis)):
        apex, b1, b2 = _arrowhead_at(tip, direction)
        sh = page.new_shape()
        _draw_filled_triangle(sh, apex, b1, b2)
    mid = ((p_a[0] + p_b[0]) / 2.0 - perp[0] * text_offset,
           (p_a[1] + p_b[1]) / 2.0 - perp[1] * text_offset)
    page.insert_text(_pt(mid), value, fontsize=8)
    return {"end_a": _to_ir(*p_a), "end_b": _to_ir(*p_b), "value": value}


def build_synthetic_dimension_pdf(tmp_path, include_decoys: bool = True
                                  ) -> Tuple[str, Dict[str, Any]]:
    """A sheet with 2 genuine dimensions (one horizontal, one vertical) plus
    decoys: a TRUE leader (single arrowhead — must NOT surface as a
    dimension) and stray unlabelled lines."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    dims = [
        _plant_dimension(page, (150, 120), (400, 120), "8'-0\""),
        _plant_dimension(page, (600, 200), (600, 450), "3.5 m"),
    ]

    n_decoy_leaders = n_decoy_lines = 0
    if include_decoys:
        shape = page.new_shape()
        _plant_leader(page, shape, (100, 500), (180, 540), (250, 520), "TYP")
        n_decoy_leaders = 1
        for a, b in [((450, 600), (550, 560)), ((700, 620), (820, 650))]:
            s = page.new_shape()
            _draw_shaft(s, [a, b])
            n_decoy_lines += 1

    path = str(tmp_path / "dimension_sheet.pdf")
    doc.save(path)
    doc.close()
    return path, {
        "page_size": (PAGE_WIDTH, PAGE_HEIGHT),
        "dimensions": dims,
        "n_decoy_leaders": n_decoy_leaders,
        "n_decoy_lines": n_decoy_lines,
    }


# ---------------------------------------------------------------------------
# Title block
# ---------------------------------------------------------------------------

def build_synthetic_title_block_pdf(tmp_path) -> Tuple[str, Dict[str, Any]]:
    """A bordered sheet with a subdivided title block in the bottom-right
    corner, body content elsewhere, and a mid-sheet labelled box decoy (a
    schedule/detail box that must NOT win over the corner block)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    # Sheet border (near-page-size rectangle — must be ignored as "the sheet").
    page.draw_rect(fitz.Rect(20, 20, PAGE_WIDTH - 20, PAGE_HEIGHT - 20),
                   color=(0, 0, 0))

    # Title block: outer rect + 3 cells, bottom-right (PDF top-left coords:
    # large y = visual bottom).
    tb = (620, 560, PAGE_WIDTH - 20, PAGE_HEIGHT - 20)
    page.draw_rect(fitz.Rect(*tb), color=(0, 0, 0))
    page.draw_rect(fitz.Rect(620, 560, 880, 600), color=(0, 0, 0))
    page.draw_rect(fitz.Rect(620, 600, 880, 640), color=(0, 0, 0))
    page.draw_rect(fitz.Rect(620, 640, 750, 680), color=(0, 0, 0))
    tb_texts = ["ACME ENGINEERING", "RETAINING WALL PLAN", "SHEET S-101",
                "REV 2", "SCALE: 1\"=20'"]
    for i, t in enumerate(tb_texts):
        page.insert_text(fitz.Point(630, 578 + i * 20), t, fontsize=8)

    # Body content: lines + labels away from the corner.
    for a, b in [((100, 150), (500, 150)), ((100, 300), (500, 320)),
                 ((150, 100), (150, 400))]:
        s = page.new_shape()
        _draw_shaft(s, [a, b])
    page.insert_text(fitz.Point(250, 90), "PLAN VIEW", fontsize=10)
    page.insert_text(fitz.Point(520, 250), "EL. 412.5", fontsize=8)

    # Mid-sheet labelled box (schedule) — rectangle with text, NOT at an edge.
    page.draw_rect(fitz.Rect(300, 380, 520, 470), color=(0, 0, 0))
    page.insert_text(fitz.Point(310, 400), "PILE SCHEDULE", fontsize=8)
    page.insert_text(fitz.Point(310, 420), "P1  HP12x53", fontsize=8)

    path = str(tmp_path / "title_block_sheet.pdf")
    doc.save(path)
    doc.close()

    x0, y0, x1, y1 = tb
    return path, {
        "page_size": (PAGE_WIDTH, PAGE_HEIGHT),
        # IR frame: y flipped.
        "block_bbox_ir": (x0, PAGE_HEIGHT - y1, x1, PAGE_HEIGHT - y0),
        "texts": tb_texts,
        "n_cells": 3,
    }


# ---------------------------------------------------------------------------
# Bubble callouts
# ---------------------------------------------------------------------------

def build_synthetic_bubble_pdf(tmp_path) -> Tuple[str, Dict[str, Any]]:
    """Three genuine bubbles (keynote, grid bubble with attached grid line,
    split-circle detail callout) plus decoys: an oversized unlabelled circle
    (a tank/column in plan — bigger than any plausible bubble) and an
    unlabelled bubble-sized circle."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    bubbles: List[Dict[str, Any]] = []

    # Keynote: circle + centered number.
    page.draw_circle(fitz.Point(200, 200), 10, color=(0, 0, 0))
    page.insert_text(fitz.Point(197, 204), "3", fontsize=9)
    bubbles.append({"center": _to_ir(200, 200), "radius": 10,
                    "text": "3", "kind": "keynote"})

    # Grid bubble: circle + letter + a grid line ending on the ring.
    page.draw_circle(fitz.Point(500, 80), 12, color=(0, 0, 0))
    page.insert_text(fitz.Point(496, 84), "A", fontsize=9)
    s = page.new_shape()
    _draw_shaft(s, [(500, 92), (500, 420)])
    bubbles.append({"center": _to_ir(500, 80), "radius": 12,
                    "text": "A", "kind": "grid_bubble"})

    # Detail callout: circle + horizontal chord through center + two labels.
    page.draw_circle(fitz.Point(700, 500), 18, color=(0, 0, 0))
    s = page.new_shape()
    _draw_shaft(s, [(682, 500), (718, 500)])
    page.insert_text(fitz.Point(695, 496), "5", fontsize=9)
    page.insert_text(fitz.Point(686, 514), "S-201", fontsize=7)
    bubbles.append({"center": _to_ir(700, 500), "radius": 18,
                    "text": "5", "kind": "detail_callout"})

    # Decoys.
    page.draw_circle(fitz.Point(300, 450), 60, color=(0, 0, 0))   # oversized
    page.draw_circle(fitz.Point(80, 600), 10, color=(0, 0, 0))    # unlabelled

    path = str(tmp_path / "bubble_sheet.pdf")
    doc.save(path)
    doc.close()
    return path, {
        "page_size": (PAGE_WIDTH, PAGE_HEIGHT),
        "bubbles": bubbles,
        "n_decoy_oversized": 1,
        "n_decoy_unlabelled": 1,
    }


# ---------------------------------------------------------------------------
# Revision clouds
# ---------------------------------------------------------------------------

def _plant_cloud(page, x0: float, y0: float, x1: float, y1: float,
                 scallop: float = 40.0, bump: float = 12.0) -> None:
    """A scalloped revision cloud around a rectangle, one bezier per bump
    bulging OUTWARD (matches how CAD plots a cloud to PDF)."""
    import fitz

    # Boundary sample points, clockwise, roughly `scallop` apart.
    pts: List[Point] = []
    for x in _steps(x0, x1, scallop):
        pts.append((x, y0))
    for y in _steps(y0, y1, scallop):
        pts.append((x1, y))
    for x in _steps(x1, x0, scallop):
        pts.append((x, y1))
    for y in _steps(y1, y0, scallop):
        pts.append((x0, y))

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sh = page.new_shape()
    for a, b in zip(pts, pts[1:] + pts[:1]):
        mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        # Outward = away from the cloud center.
        ox, oy = _unit(mx - cx, my - cy)
        ctrl = (mx + ox * bump, my + oy * bump)
        sh.draw_curve(fitz.Point(*a), fitz.Point(*ctrl), fitz.Point(*b))
    sh.finish(color=(0, 0, 0), closePath=False)
    sh.commit()


def _steps(a: float, b: float, step: float) -> List[float]:
    n = max(1, int(round(abs(b - a) / step)))
    return [a + (b - a) * i / n for i in range(n)]


def build_synthetic_cloud_pdf(tmp_path) -> Tuple[str, Dict[str, Any]]:
    """A sheet with one scalloped revision cloud + one revision delta
    (labelled triangle, no shaft) plus decoys: a plain rectangle, a smooth
    circle, and a TRUE leader (whose arrowhead has a shaft and must NOT be
    proposed as a delta)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    cloud_pdf = (250, 150, 470, 290)
    _plant_cloud(page, *cloud_pdf)

    # Revision delta: isolated triangle + number.
    tri = ((520, 420), (508, 442), (532, 442))
    sh = page.new_shape()
    _draw_filled_triangle(sh, *tri)
    page.insert_text(fitz.Point(516, 438), "2", fontsize=8)

    # Decoys.
    page.draw_rect(fitz.Rect(100, 500, 200, 560), color=(0, 0, 0))
    page.draw_circle(fitz.Point(650, 200), 15, color=(0, 0, 0))
    shape = page.new_shape()
    _plant_leader(page, shape, (700, 600), (760, 560), (820, 590), "NOTE 1")

    path = str(tmp_path / "cloud_sheet.pdf")
    doc.save(path)
    doc.close()

    x0, y0, x1, y1 = cloud_pdf
    cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0
    cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0
    return path, {
        "page_size": (PAGE_WIDTH, PAGE_HEIGHT),
        "cloud_bbox_ir": (x0, PAGE_HEIGHT - y1, x1, PAGE_HEIGHT - y0),
        "delta_center_ir": _to_ir(cx, cy),
        "delta_text": "2",
        "n_decoy_shapes": 2,
        "n_decoy_leaders": 1,
    }


# ---------------------------------------------------------------------------
# Multi-page set (P2-C): a small drawing set for cross-sheet aggregation
# ---------------------------------------------------------------------------

def build_synthetic_drawing_set_pdf(tmp_path) -> Tuple[str, Dict[str, Any]]:
    """A 3-page PDF 'drawing set': page 0 with two leaders labelled 'W1' and
    one labelled 'TYP'; page 1 with one 'W1' leader and a dimension; page 2
    with title-block-ish corner text only. Ground truth = per-page 'W1'
    counts, for the find-across-set aggregation scenario."""
    import fitz

    doc = fitz.open()

    # Page 0: two W1 leaders + one TYP.
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    for (tail, bend, tip), label in [
        (((80, 100), (160, 140), (230, 120)), "W1"),
        (((720, 620), (650, 560), (600, 600)), "W1"),
        (((120, 600), (200, 560), (260, 610)), "TYP"),
    ]:
        shape = page.new_shape()
        _plant_leader(page, shape, tail, bend, tip, label)

    # Page 1: one W1 leader + one dimension.
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    shape = page.new_shape()
    _plant_leader(page, shape, (100, 200), (180, 240), (250, 220), "W1")
    _plant_dimension(page, (400, 400), (650, 400), "12'-6\"")

    # Page 2: corner text only.
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    for i, t in enumerate(["SHEET S-103", "REV 1", "AS NOTED", "ACME"]):
        page.insert_text(fitz.Point(760, 600 + i * 18), t, fontsize=8)

    path = str(tmp_path / "drawing_set.pdf")
    doc.save(path)
    doc.close()
    return path, {
        "n_pages": 3,
        "w1_counts_by_page": {0: 2, 1: 1, 2: 0},
        "leader_counts_by_page": {0: 3, 1: 1, 2: 0},
    }
