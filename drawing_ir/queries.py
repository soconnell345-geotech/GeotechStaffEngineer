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
# Endpoint-anchored queries (B4) — "what TERMINATES here?"
# ---------------------------------------------------------------------------

def _arc_endpoint(e: Arc, angle_deg: float) -> Point:
    a = math.radians(angle_deg)
    return (e.center[0] + e.radius * math.cos(a),
            e.center[1] + e.radius * math.sin(a))


def _entity_endpoints(e: Entity) -> List[Tuple[str, Point]]:
    """The named terminal points of an entity's geometry, if it has any.

    ``[("start", pt), ("end", pt)]`` for Line/Polyline/Arc/Region (Polyline and
    Region use the first/last vertex of their vertex/boundary list — labelled
    "start"/"end" even when ``closed`` is True; a closed ring's endpoints are
    less meaningful, but the labels stay consistent so callers can treat every
    result uniformly). Circle and TextItem have no endpoints (a circle is a
    closed loop with no ends; text has an insertion point, not a terminus) and
    return ``[]``.
    """
    if isinstance(e, Line):
        return [("start", tuple(e.start)), ("end", tuple(e.end))]
    if isinstance(e, Polyline):
        if not e.vertices:
            return []
        return [("start", tuple(e.vertices[0])), ("end", tuple(e.vertices[-1]))]
    if isinstance(e, Arc):
        return [("start", _arc_endpoint(e, e.start_angle)),
                ("end", _arc_endpoint(e, e.end_angle))]
    if isinstance(e, Region):
        if not e.boundary:
            return []
        return [("start", tuple(e.boundary[0])), ("end", tuple(e.boundary[-1]))]
    return []


def entities_ending_near(ir: DrawingIR, point: Point, radius: float,
                         entity_types: Optional[List[str]] = None
                         ) -> List[Dict[str, Any]]:
    """Entities with an ENDPOINT (not just bbox/full-geometry) within radius.

    Unlike :func:`entities_in_bbox`/:func:`nearest_entity` (bbox or nearest-
    point-on-geometry), this checks only an entity's TERMINAL points — the
    natural query for "what line/polyline/arc TERMINATES here" (a leader's
    arrow tip, a callout's tail, a dimension's end). ``entity_types``
    optionally restricts to a set of KINDs (e.g. ``["line", "polyline"]``);
    Circle and TextItem never match (see :func:`_entity_endpoints`).

    Returns one entry per matching ``(entity, endpoint)`` pair, closest first:
    the usual entity reference plus ``end`` ("start"|"end" — which endpoint
    matched), ``end_point`` (the matched endpoint, exact coordinates),
    ``other_end`` (the entity's OTHER endpoint — e.g. the far end a leader
    points FROM), and ``distance``.
    """
    px, py = point
    scored = []
    for e in ir.entities:
        if entity_types and e.KIND not in entity_types:
            continue
        endpoints = _entity_endpoints(e)
        if len(endpoints) < 2:
            continue
        for i, (label, pt) in enumerate(endpoints):
            d = math.hypot(px - pt[0], py - pt[1])
            if d > radius:
                continue
            _, other_pt = endpoints[1 - i]
            ref = _ref(e)
            ref["end"] = label
            ref["end_point"] = [_r(pt[0]), _r(pt[1])]
            ref["other_end"] = [_r(other_pt[0]), _r(other_pt[1])]
            ref["distance"] = _r(d)
            scored.append((d, ref))
    scored.sort(key=lambda t: t[0])
    return [ref for _, ref in scored]


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


def text_anchored_geometry(ir: DrawingIR, pattern: str,
                           radius: Optional[float] = None,
                           entity_types: Optional[List[str]] = None
                           ) -> List[Dict[str, Any]]:
    """Text matches plus any geometry that TERMINATES at/near them.

    Composes :func:`text_items` (find text matching ``pattern``) with
    :func:`entities_ending_near` around each match's insertion point — the
    "find text X -> the leader/line terminating there -> the far end it
    points at" primitive: search for any tail label (the pattern is always a
    runtime parameter — nothing here is specific to one string), read off the
    connected geometry's far endpoint as the callout's target location, then
    hand that location to a vision zoom for a description.

    ``radius`` defaults to a PER-MATCH value derived from that text item's
    own height (``3 * height`` in the IR's coordinate units) when not given,
    since text size is the natural proxy for "how close counts as touching
    this label" and it scales correctly in BOTH page space (points) and model
    space (meters). Only when the text height is unknown/zero does a fixed
    5.0-unit fallback apply (a fixed floor must not bind when height is
    known — 5.0 is a few text-heights in page points but a huge reach in
    model-space meters).

    Returns one entry per text match: ``{text, anchor, connected,
    points_at, proposal_only: True}``. ``connected`` is the full
    :func:`entities_ending_near` result (possibly empty); ``points_at`` is the
    ``other_end`` of the CLOSEST connected entity (the far endpoint — where
    the geometry is aimed FROM the text), or ``None`` if nothing connects.
    Geometric adjacency alone does not prove a leader relationship (a
    coincidental nearby line-end also matches) — this is a PROPOSAL, confirm
    against the drawing (e.g. via a vision zoom on ``points_at``) before
    treating it as fact.
    """
    matches = text_items(ir, pattern)
    out = []
    for m in matches:
        e = ir.by_id(m["id"])
        anchor = tuple(e.position) if isinstance(e, TextItem) else None
        if anchor is None:
            out.append({"text": m, "anchor": None, "connected": [],
                        "points_at": None, "proposal_only": True})
            continue
        h = e.height or 0.0
        r = radius if radius is not None else (3.0 * h if h > 0 else 5.0)
        connected = entities_ending_near(ir, anchor, r, entity_types=entity_types)
        points_at = connected[0]["other_end"] if connected else None
        out.append({
            "text": m,
            "anchor": [_r(anchor[0]), _r(anchor[1])],
            "search_radius": _r(r),
            "connected": connected,
            "points_at": points_at,
            "proposal_only": True,
        })
    return out


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


def _bbox_diag(bbox) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _centroid(pts: List[Point]) -> Point:
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def _unit_vec(dx: float, dy: float) -> Point:
    m = math.hypot(dx, dy)
    return (dx / m, dy / m) if m > 1e-12 else (0.0, 0.0)


def _default_max_arrowhead_size(ir: DrawingIR) -> float:
    """~25% of the drawing's typical (median) open shaft-segment length.

    Scaled off the drawing's own line-work rather than its overall bbox: the
    bbox can be inflated by unrelated far-away content (a title block, a
    second detail), which would make a bbox-fraction threshold too permissive
    or too tight depending on what else is on the sheet. The median length of
    Line / open-Polyline entities is a steadier proxy for "how big is a
    typical shaft here" — an arrowhead is normally a small fraction of that.
    Falls back to a bbox-diagonal fraction (drawings with no line-work at
    all), then a fixed floor.
    """
    lengths = [e.length() for e in ir.entities
              if (isinstance(e, Line)
                  or (isinstance(e, Polyline) and not e.closed))
              and e.length() > 0]
    if lengths:
        lengths.sort()
        median = lengths[len(lengths) // 2]
        return max(median * 0.25, 1.0)
    bb = ir.bbox()
    if bb is not None:
        diag = _bbox_diag(bb)
        if diag > 0:
            return max(diag * 0.03, 1.0)
    return 20.0


def _arrowhead_candidates(ir: DrawingIR, max_arrowhead_size: float):
    """Yield (entity, vertices) for small closed 3-5-vertex Polyline/Region.

    This is what a filled-triangle arrowhead becomes on ingest — verified
    empirically against ``from_pdf_vector`` (see the module docstring below
    :func:`find_leaders`): PDF-vector ingest never emits a ``Region`` (only
    Line/Polyline/TextItem), so a filled triangle arrives as a CLOSED
    Polyline with 3 vertices and a small bbox. A DXF HATCH-based arrowhead
    can arrive as a small ``Region`` instead, so both are checked (keeps this
    usable once B1's DXF LEADER ingest lands).
    """
    for e in ir.entities:
        if isinstance(e, Polyline) and e.closed and 3 <= len(e.vertices) <= 5:
            verts = e.vertices
        elif isinstance(e, Region) and 3 <= len(e.boundary) <= 5:
            verts = e.boundary
        else:
            continue
        if e.bbox is None or _bbox_diag(e.bbox) > max_arrowhead_size:
            continue
        yield e, [tuple(p) for p in verts]


def _alignment_score(u: Point, v: Point) -> Tuple[float, float]:
    """(score, angle_deg) — 1.0 at 0 deg (parallel), 0.0 at >=90 deg apart."""
    dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    angle = math.degrees(math.acos(dot))
    return max(0.0, 1.0 - angle / 90.0), angle


def _text_proximity_score(dist: Optional[float], radius: float) -> float:
    if dist is None or radius <= 0:
        return 0.0
    return max(0.0, 1.0 - dist / radius)


def _chain_simplicity_score(n_vertices: int) -> float:
    """1.0 for a plain 2-point shaft (a Line); decreasing with more bends."""
    if n_vertices <= 2:
        return 1.0
    return max(0.3, 1.0 - 0.15 * (n_vertices - 2))


def find_leaders(ir: DrawingIR, max_arrowhead_size: Optional[float] = None,
                 search_radius: Optional[float] = None,
                 text_radius: Optional[float] = None,
                 min_confidence: float = 0.0) -> List[Dict[str, Any]]:
    """PROPOSE leader constructs (bent arrow + tail text) from primitives.

    A "leader" (AutoCAD sense): a shaft (Line, or a Polyline with one or more
    bends) ending in an arrowhead at one end ("points at" something) and text
    at the other end ("tail"). Neither PDF vector nor an un-augmented DXF
    model space has a LEADER entity type for PDF; this composes one from the
    shared primitives, as a confidence-scored PROPOSAL — never asserted, same
    house pattern as :func:`candidate_ground_surface`.

    Heuristic (documented, not hidden):

    1. **Arrowhead candidates** — small closed 3-5-vertex Polyline/Region
       entities (see :func:`_arrowhead_candidates`). "Small" = bbox diagonal
       <= ``max_arrowhead_size`` (default: see
       :func:`_default_max_arrowhead_size` — ~25% of the drawing's typical
       shaft-segment length, falling back to a bbox-diagonal fraction; scales
       with the drawing instead of assuming absolute units).
    2. **Shaft** — the nearest Line/open-Polyline endpoint (excluding other
       arrowhead candidates) within ``search_radius`` (default
       ``1.5 * max_arrowhead_size``) of the candidate's centroid, via
       :func:`entities_ending_near`. ``tip_xy`` is that SHAFT endpoint (the
       deterministic source of truth), not the triangle's own apex vertex.
    3. **Alignment** — the shaft's terminal-segment direction vs. the
       arrowhead's own apex-from-base direction (apex = candidate vertex
       nearest the shaft endpoint; base = centroid of the other vertices).
       Parallel directions score high; the composed proposal is otherwise
       shape-blind (no assumption about arrowhead style beyond "small closed
       3-5-gon").
    4. **Tail text** — nearest TextItem to the shaft's FAR endpoint (the end
       away from the arrowhead), within ``text_radius`` (default
       ``4 * max_arrowhead_size``).
    5. **Confidence** = ``0.45*alignment + 0.35*text_proximity +
       0.20*chain_simplicity`` (weights chosen so a well-aligned, clearly
       labelled, simple 1-bend leader scores near 1.0; each factor is 0..1,
       see ``evidence`` in each proposal for the breakdown).

    Known false-positive source (by design, not a bug): a dimension line's
    end arrow is geometrically identical to a leader arrowhead (a small
    filled triangle at a line end), and a dimension VALUE sitting near the
    line's other end can score as "tail text" — so a true dimension can also
    surface here as a lower/moderate-confidence leader proposal. This
    heuristic composes primitives; it does not (yet) distinguish leader
    semantics from dimension semantics (that is :func:`find_dimensions`,
    Phase 2). Confirm visually (e.g. ``drawing_ir.render.render_region`` on
    ``tip_xy``) before treating a proposal as a true annotation leader.

    Returns proposals sorted by confidence (descending), each:
    ``{tip_xy, tail_xy, vertices, arrowhead_id, shaft_id, text, text_id,
    text_distance, confidence, evidence, proposal_only: True}``, filtered to
    ``confidence >= min_confidence`` (default 0 = return every candidate).
    """
    max_arrowhead_size = max_arrowhead_size or _default_max_arrowhead_size(ir)
    search_radius = (search_radius if search_radius is not None
                     else max_arrowhead_size * 1.5)
    text_radius = (text_radius if text_radius is not None
                   else max_arrowhead_size * 4.0)

    proposals = []
    for cand, verts in _arrowhead_candidates(ir, max_arrowhead_size):
        centroid = _centroid(verts)
        shaft_hits = entities_ending_near(ir, centroid, search_radius,
                                          entity_types=["line", "polyline"])
        shaft_hits = [h for h in shaft_hits if h["id"] != cand.id
                     and not (h["type"] == "polyline" and h.get("closed"))]
        if not shaft_hits:
            continue
        shaft_ref = shaft_hits[0]
        shaft = ir.by_id(shaft_ref["id"])
        shaft_pts = shaft.points()
        if len(shaft_pts) < 2:
            continue

        tip_xy = tuple(shaft_ref["end_point"])
        far_xy = tuple(shaft_ref["other_end"])
        if shaft_ref["end"] == "end":
            term_a, term_b = shaft_pts[-2], shaft_pts[-1]
        else:
            term_a, term_b = shaft_pts[1], shaft_pts[0]
        shaft_dir = _unit_vec(term_b[0] - term_a[0], term_b[1] - term_a[1])

        apex = min(verts, key=lambda p: math.hypot(p[0] - tip_xy[0],
                                                   p[1] - tip_xy[1]))
        others = [p for p in verts if p != apex] or verts
        base_mid = _centroid(others)
        arrow_dir = _unit_vec(apex[0] - base_mid[0], apex[1] - base_mid[1])

        align_score, align_deg = _alignment_score(shaft_dir, arrow_dir)

        text_hit, text_dist = None, None
        near_text = nearest_entity(ir, far_xy[0], far_xy[1], entity_type="text", k=1)
        if near_text and near_text[0]["distance"] <= text_radius:
            text_hit = near_text[0]
            text_dist = text_hit["distance"]
        text_score = _text_proximity_score(text_dist, text_radius)

        n_vertices_shaft = shaft_ref.get("n_vertices", 2)
        simplicity = _chain_simplicity_score(n_vertices_shaft)

        confidence = round(0.45 * align_score + 0.35 * text_score
                           + 0.20 * simplicity, 3)
        if confidence < min_confidence:
            continue

        proposals.append({
            "tip_xy": [_r(tip_xy[0]), _r(tip_xy[1])],
            "tail_xy": [_r(far_xy[0]), _r(far_xy[1])],
            "vertices": [[_r(p[0]), _r(p[1])] for p in shaft_pts],
            "arrowhead_id": cand.id,
            "shaft_id": shaft.id,
            "text": text_hit["content"] if text_hit else None,
            "text_id": text_hit["id"] if text_hit else None,
            "text_distance": _r(text_dist) if text_dist is not None else None,
            "confidence": confidence,
            "evidence": {
                "alignment_score": _r(align_score, 3),
                "alignment_deg": _r(align_deg, 1),
                "text_proximity_score": _r(text_score, 3),
                "chain_simplicity_score": _r(simplicity, 3),
                "n_shaft_vertices": n_vertices_shaft,
            },
            "proposal_only": True,
        })

    proposals.sort(key=lambda p: p["confidence"], reverse=True)
    return proposals


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
