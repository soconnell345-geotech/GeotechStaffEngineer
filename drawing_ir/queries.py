"""
Query interface over a :class:`drawing_ir.results.DrawingIR`.

This is the surface an LLM (or the funhouse adapter) calls to request *slices*
of a drawing instead of interpreting pixels: spatial windows, angle bands, text
matches, layer/color groups, nearest-entity lookups, and a couple of geotech
heuristics. Every function takes a ``DrawingIR`` and returns compact, JSON-able
Python (lists/dicts of primitives) — entity *references* (id + a small summary),
never full coordinate dumps. Use ``get_entities(ir, ids)`` to pull exact
coordinates for a shortlist the query narrowed down.

The deterministic extractor owns the coordinates; these queries only slice and
summarize them. Anything labelled a *proposal* (e.g. candidate_ground_surface)
is a heuristic suggestion for the caller to confirm, never an assertion.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from drawing_ir.results import (
    Arc, Circle, DrawingIR, Entity, Line, Polyline, Region, TextItem, _r,
)

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Compact references
# ---------------------------------------------------------------------------

def _ref(e: Entity) -> Dict[str, Any]:
    """A compact, JSON-able summary of an entity (no full coordinate dump)."""
    d: Dict[str, Any] = {
        "id": e.id,
        "type": e.KIND,
        "source": e.source,
        "confidence": _r(e.confidence, 3),
    }
    if e.layer is not None:
        d["layer"] = e.layer
    if e.color is not None:
        d["color"] = e.color
    if e.bbox is not None:
        d["bbox"] = [_r(v) for v in e.bbox]
    if isinstance(e, Line):
        d["length"] = _r(e.length())
        d["angle_deg"] = _r(e.angle_deg(), 2)
    elif isinstance(e, Polyline):
        d["n_vertices"] = len(e.vertices)
        d["length"] = _r(e.length())
        d["closed"] = bool(e.closed)
    elif isinstance(e, Arc):
        d["radius"] = _r(e.radius)
        d["length"] = _r(e.length())
    elif isinstance(e, Circle):
        d["center"] = [_r(e.center[0]), _r(e.center[1])]
        d["radius"] = _r(e.radius)
    elif isinstance(e, TextItem):
        d["content"] = e.content
        d["position"] = [_r(e.position[0]), _r(e.position[1])]
    elif isinstance(e, Region):
        d["n_vertices"] = len(e.boundary)
        d["area"] = _r(e.area())
    return d


def _refs(entities) -> List[Dict[str, Any]]:
    return [_ref(e) for e in entities]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iter_segments(ir: DrawingIR):
    """Yield (entity, seg_index, p0, p1) for every Line/Polyline segment."""
    for e in ir.entities:
        if isinstance(e, Line):
            yield e, 0, tuple(e.start), tuple(e.end)
        elif isinstance(e, Polyline):
            pts = e.vertices
            for i in range(len(pts) - 1):
                yield e, i, tuple(pts[i]), tuple(pts[i + 1])
            if e.closed and len(pts) > 2:
                yield e, len(pts) - 1, tuple(pts[-1]), tuple(pts[0])


def _seg_angle(p0: Point, p1: Point) -> float:
    """Segment orientation folded to [0, 180)."""
    a = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0
    return a


def _bbox_intersects(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _bbox_contains(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def _point_seg_dist(px: float, py: float, ax: float, ay: float,
                    bx: float, by: float) -> float:
    """Shortest distance from (px, py) to segment (a)-(b)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _point_to_entity_dist(x: float, y: float, e: Entity) -> float:
    """Shortest distance from (x, y) to an entity's actual geometry.

    Point-to-segment for lines/polylines/region edges, point-to-ring for
    circles, insertion-point distance for text, sampled distance for arcs.
    """
    if isinstance(e, Line):
        return _point_seg_dist(x, y, e.start[0], e.start[1],
                               e.end[0], e.end[1])
    if isinstance(e, Polyline):
        pts = e.vertices
        if len(pts) == 1:
            return math.hypot(x - pts[0][0], y - pts[0][1])
        segs = list(zip(pts, pts[1:]))
        if e.closed and len(pts) > 2:
            segs.append((pts[-1], pts[0]))
        return min(_point_seg_dist(x, y, a[0], a[1], b[0], b[1])
                   for a, b in segs) if segs else math.inf
    if isinstance(e, Region):
        pts = e.boundary
        if len(pts) < 2:
            return math.inf
        segs = list(zip(pts, pts[1:] + pts[:1]))
        return min(_point_seg_dist(x, y, a[0], a[1], b[0], b[1])
                   for a, b in segs)
    if isinstance(e, Circle):
        return abs(math.hypot(x - e.center[0], y - e.center[1]) - e.radius)
    if isinstance(e, TextItem):
        return math.hypot(x - e.position[0], y - e.position[1])
    pts = e.points()  # arcs and any fallback: nearest sampled point
    if not pts:
        return math.inf
    return min(math.hypot(px - x, py - y) for px, py in pts)


# ---------------------------------------------------------------------------
# Spatial queries
# ---------------------------------------------------------------------------

def entities_in_bbox(ir: DrawingIR, x_min: float, y_min: float,
                     x_max: float, y_max: float,
                     mode: str = "intersect",
                     entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Entities whose bbox intersects (or is contained in) a query window.

    ``mode``: "intersect" (default) or "contain". ``entity_type`` optionally
    restricts to one kind ("line"/"polyline"/"arc"/"circle"/"text"/"region").
    """
    win = (min(x_min, x_max), min(y_min, y_max),
           max(x_min, x_max), max(y_min, y_max))
    hits = []
    for e in ir.entities:
        if entity_type and e.KIND != entity_type:
            continue
        if e.bbox is None:
            continue
        ok = (_bbox_contains(win, e.bbox) if mode == "contain"
              else _bbox_intersects(win, e.bbox))
        if ok:
            hits.append(e)
    return _refs(hits)


def nearest_entity(ir: DrawingIR, x: float, y: float,
                   entity_type: Optional[str] = None,
                   k: int = 1) -> List[Dict[str, Any]]:
    """The ``k`` entities nearest to point (x, y), closest first.

    Each result carries a ``distance`` field (shortest distance to the entity's
    geometry; 0 if the point lies on it).
    """
    scored = []
    for e in ir.entities:
        if entity_type and e.KIND != entity_type:
            continue
        scored.append((_point_to_entity_dist(x, y, e), e))
    scored.sort(key=lambda t: t[0])
    out = []
    for dist, e in scored[:max(1, k)]:
        ref = _ref(e)
        ref["distance"] = _r(dist)
        out.append(ref)
    return out


# ---------------------------------------------------------------------------
# Angle / length queries (operate on Line + Polyline segments)
# ---------------------------------------------------------------------------

def lines_by_angle(ir: DrawingIR, min_deg: float, max_deg: float
                   ) -> List[Dict[str, Any]]:
    """Line/polyline segments whose orientation (folded to [0,180)) is in band.

    Returns one entry per matching segment: ``{entity_id, type, seg_index,
    angle_deg, start, end, length}``.
    """
    lo, hi = min_deg % 180.0, max_deg % 180.0
    out = []
    for e, i, p0, p1 in _iter_segments(ir):
        ang = _seg_angle(p0, p1)
        inside = (lo <= ang <= hi) if lo <= hi else (ang >= lo or ang <= hi)
        if inside:
            out.append({
                "entity_id": e.id, "type": e.KIND, "seg_index": i,
                "angle_deg": _r(ang, 2),
                "start": [_r(p0[0]), _r(p0[1])],
                "end": [_r(p1[0]), _r(p1[1])],
                "length": _r(math.hypot(p1[0] - p0[0], p1[1] - p0[1])),
            })
    return out


def horizontal_lines(ir: DrawingIR, tol_deg: float = 2.0) -> List[Dict[str, Any]]:
    """Segments within ``tol_deg`` of horizontal (0 deg)."""
    out = []
    for e, i, p0, p1 in _iter_segments(ir):
        ang = _seg_angle(p0, p1)
        dev = min(ang, 180.0 - ang)
        if dev <= tol_deg:
            out.append({
                "entity_id": e.id, "type": e.KIND, "seg_index": i,
                "angle_deg": _r(ang, 2),
                "start": [_r(p0[0]), _r(p0[1])], "end": [_r(p1[0]), _r(p1[1])],
                "length": _r(math.hypot(p1[0] - p0[0], p1[1] - p0[1])),
            })
    return out


def vertical_lines(ir: DrawingIR, tol_deg: float = 2.0) -> List[Dict[str, Any]]:
    """Segments within ``tol_deg`` of vertical (90 deg)."""
    out = []
    for e, i, p0, p1 in _iter_segments(ir):
        ang = _seg_angle(p0, p1)
        if abs(ang - 90.0) <= tol_deg:
            out.append({
                "entity_id": e.id, "type": e.KIND, "seg_index": i,
                "angle_deg": _r(ang, 2),
                "start": [_r(p0[0]), _r(p0[1])], "end": [_r(p1[0]), _r(p1[1])],
                "length": _r(math.hypot(p1[0] - p0[0], p1[1] - p0[1])),
            })
    return out


def polylines_longer_than(ir: DrawingIR, min_length: float
                          ) -> List[Dict[str, Any]]:
    """Polylines whose total path length >= ``min_length`` (longest first)."""
    hits = [e for e in ir.entities
            if isinstance(e, Polyline) and e.length() >= min_length]
    hits.sort(key=lambda e: e.length(), reverse=True)
    return _refs(hits)


# ---------------------------------------------------------------------------
# Text queries
# ---------------------------------------------------------------------------

def text_items(ir: DrawingIR, pattern: Optional[str] = None
               ) -> List[Dict[str, Any]]:
    """Text items, optionally filtered by a regex ``pattern`` (case-insensitive).

    If ``pattern`` is not a valid regex it is treated as a literal substring.
    """
    texts = [e for e in ir.entities if isinstance(e, TextItem)]
    if pattern:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
            texts = [e for e in texts if rx.search(e.content)]
        except re.error:
            low = pattern.lower()
            texts = [e for e in texts if low in e.content.lower()]
    return _refs(texts)


def text_near(ir: DrawingIR, entity_id: str, radius: float
              ) -> List[Dict[str, Any]]:
    """Text items whose insertion point is within ``radius`` of an entity.

    Distance is measured from the text insertion point to the target entity
    (0 if the point lies inside the target's bbox). Sorted nearest first.
    """
    target = ir.by_id(entity_id)
    if target is None:
        return [{"error": f"Unknown entity_id '{entity_id}'"}]
    out = []
    for e in ir.entities:
        if not isinstance(e, TextItem):
            continue
        d = _point_to_entity_dist(e.position[0], e.position[1], target)
        if d <= radius:
            ref = _ref(e)
            ref["distance"] = _r(d)
            out.append((d, ref))
    out.sort(key=lambda t: t[0])
    return [ref for _, ref in out]


# ---------------------------------------------------------------------------
# Layer / color groups
# ---------------------------------------------------------------------------

def entities_on_layer(ir: DrawingIR, layer: str) -> List[Dict[str, Any]]:
    """Entities on a given layer (exact match)."""
    return _refs([e for e in ir.entities if e.layer == layer])


def entities_by_color(ir: DrawingIR, color: str) -> List[Dict[str, Any]]:
    """Entities of a given color (case-insensitive exact match)."""
    low = color.lower()
    return _refs([e for e in ir.entities
                  if e.color is not None and e.color.lower() == low])


# ---------------------------------------------------------------------------
# Selective coordinate retrieval
# ---------------------------------------------------------------------------

def get_entities(ir: DrawingIR, ids: List[str]) -> List[Dict[str, Any]]:
    """Full ``to_dict`` (exact coordinates) for a shortlist of entity ids."""
    out = []
    for eid in ids:
        e = ir.by_id(eid)
        if e is not None:
            out.append(e.to_dict())
        else:
            out.append({"id": eid, "error": "not found"})
    return out


# ---------------------------------------------------------------------------
# Heuristics (PROPOSALS — caller confirms)
# ---------------------------------------------------------------------------

def candidate_ground_surface(ir: DrawingIR) -> Dict[str, Any]:
    """Propose the entity most likely to be the ground surface.

    Heuristic ONLY (never an assertion): among Line/Polyline entities, pick the
    one with the widest horizontal (x) extent — a ground surface typically runs
    left-to-right across the section. Ties broken toward the upper (higher-y)
    candidate. The caller/LLM must confirm against the drawing; soil properties
    never come from a drawing.
    """
    paths = [e for e in ir.entities if isinstance(e, (Line, Polyline))]
    if not paths:
        return {"candidate": None,
                "note": "No line/polyline entities to propose from.",
                "proposal_only": True}

    page_w = ir.width or 0.0
    best = None
    best_score = None
    for e in paths:
        bb = e.bbox
        if bb is None:
            continue
        width = bb[2] - bb[0]
        mid_y = 0.5 * (bb[1] + bb[3])
        score = (width, mid_y)  # widest, then highest
        if best_score is None or score > best_score:
            best_score, best = score, e

    if best is None:
        return {"candidate": None, "note": "No usable geometry.",
                "proposal_only": True}

    bb = best.bbox
    width = bb[2] - bb[0]
    coverage = (width / page_w) if page_w > 0 else None
    ref = _ref(best)
    return {
        "candidate": ref,
        "x_range": [_r(bb[0]), _r(bb[2])],
        "y_range": [_r(bb[1]), _r(bb[3])],
        "width": _r(width),
        "page_width_coverage": _r(coverage, 3) if coverage is not None else None,
        "note": ("Longest left-to-right path — a PROPOSAL for the ground "
                 "surface. Confirm against the drawing before use."),
        "proposal_only": True,
    }


def summary_stats(ir: DrawingIR) -> Dict[str, Any]:
    """Counts by type/layer, page metadata, extent, and scale/provenance."""
    bb = ir.bbox()
    return {
        "source": ir.source,
        "n_entities": len(ir.entities),
        "counts_by_type": ir.counts_by_type(),
        "counts_by_layer": ir.counts_by_layer(),
        "page": {
            "width": _r(ir.width), "height": _r(ir.height),
            "units": ir.units, "coordinate_space": ir.coordinate_space,
            "origin": ir.origin,
        },
        "scale": _r(ir.scale, 9) if ir.scale is not None else None,
        "scale_provenance": ir.scale_provenance,
        "bbox": [_r(v) for v in bb] if bb is not None else None,
        "warnings": list(ir.warnings),
    }
