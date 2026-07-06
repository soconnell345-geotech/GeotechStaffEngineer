"""
Label -> region association for PDF cross-section import (C2).

Cross-section drawings label their soil regions ("CLAY", "SAND", "FILL"), the
water table ("GWT", "PHREATIC"), and the ground surface ("EXISTING GRADE") with
text. This module attaches each text label to the polyline REGION it sits in (or
nearest to) and proposes a colour->role ``role_mapping`` for
``extract_vector_geometry(role_mapping=...)``.

Like the scale helpers (C1), the proposed mapping is a PROPOSAL — returned with
its per-label provenance and never silently applied; the caller confirms it.

A "region" is a dict ``{"color": <hex>, "points": [(x, y), ...]}`` (a coloured
polyline / closed path, as grouped by ``extract_vector_geometry``). A "label" is a
text block ``{"text": str, "x": float, "y": float}`` from ``discover_pdf_content``.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# Line-feature phrases (checked first — unambiguous).
_SURFACE_KW = ("ground surface", "existing grade", "existing ground",
               "ground line", "finished grade", "grade", "ground level")
_GWT_KW = ("gwt", "phreatic", "water table", "groundwater", "ground water",
           "water level", "piezometric")
# USCS 2-letter symbols -> soil role (whole-word, authoritative).
_USCS_SYMBOL = {
    "cl": "boundary_Clay", "ch": "boundary_Clay",
    "ml": "boundary_Silt", "mh": "boundary_Silt",
    "sp": "boundary_Sand", "sw": "boundary_Sand",
    "sm": "boundary_Sand", "sc": "boundary_Sand",
    "gp": "boundary_Gravel", "gw": "boundary_Gravel",
    "gm": "boundary_Gravel", "gc": "boundary_Gravel",
    "pt": "boundary_Peat",
}
# Soil-noun keyword -> role. When several appear the RIGHTMOST wins ("silty
# SAND" -> Sand; "sandy CLAY" -> Clay), matching the USCS primary-noun rule.
_SOIL_NOUN = {
    "clay": "boundary_Clay", "silt": "boundary_Silt",
    "sand": "boundary_Sand", "gravel": "boundary_Gravel",
    "fill": "boundary_Fill", "embankment": "boundary_Fill",
    "rock": "boundary_Rock", "bedrock": "boundary_Rock",
    "shale": "boundary_Rock", "limestone": "boundary_Rock",
    "sandstone": "boundary_Rock", "peat": "boundary_Peat",
    "organic": "boundary_Peat", "till": "boundary_Till",
    "glacial": "boundary_Till",
}

_LINE_ROLES = ("surface", "gwt")


def classify_label(text: str) -> Optional[str]:
    """Map a label's text to a role ("surface"/"gwt"/"boundary_<Name>"), or None.

    Case-insensitive. Line features (surface/gwt) are matched by phrase first;
    then a USCS 2-letter symbol (whole word) if present; otherwise the RIGHTMOST
    soil noun wins so "silty SAND" -> Sand and "sandy CLAY" -> Clay.
    """
    if not text:
        return None
    t = text.lower().strip()
    for kw in _GWT_KW:
        if kw in t:
            return "gwt"
    for kw in _SURFACE_KW:
        if kw in t:
            return "surface"
    tokens = set(t.replace("(", " ").replace(")", " ").replace(",", " ")
                 .replace("-", " ").split())
    for sym, role in _USCS_SYMBOL.items():
        if sym in tokens:
            return role
    # Rightmost soil noun wins.
    best_pos, best_role = -1, None
    for noun, role in _SOIL_NOUN.items():
        pos = t.rfind(noun)
        if pos > best_pos:
            best_pos, best_role = pos, role
    return best_role


def _bbox(points: List[Tuple[float, float]]):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(points: List[Tuple[float, float]]):
    n = len(points)
    return sum(p[0] for p in points) / n, sum(p[1] for p in points) / n


def _point_in_polygon(x: float, y: float,
                      poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon (closed polygon assumed)."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_to_segment(px, py, ax, ay, bx, by) -> float:
    """Shortest distance from (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dist_to_polyline(x: float, y: float,
                      points: List[Tuple[float, float]]) -> float:
    """Shortest distance from (x, y) to a polyline (its nearest edge)."""
    if len(points) == 1:
        return math.hypot(x - points[0][0], y - points[0][1])
    return min(_dist_point_to_segment(x, y, points[i][0], points[i][1],
                                      points[i + 1][0], points[i + 1][1])
               for i in range(len(points) - 1))


def associate_labels_to_regions(regions: List[Dict[str, Any]],
                                text_blocks: List[Dict[str, Any]]
                                ) -> List[Dict[str, Any]]:
    """Attach each classifiable text label to the region it encloses / is nearest.

    Parameters
    ----------
    regions : list of dict
        Each {"color": hex, "points": [(x, y), ...]}.
    text_blocks : list of dict
        Each {"text": str, "x": float, "y": float}.

    Returns
    -------
    list of dict
        One entry per classifiable label: {label, role, color, method
        ("enclosing"|"nearest"), distance}. Labels that classify to a role but
        have no region, or don't classify at all, are skipped.
    """
    out: List[Dict[str, Any]] = []
    for tb in text_blocks:
        role = classify_label(tb.get("text", ""))
        if role is None:
            continue
        x, y = tb.get("x", 0.0), tb.get("y", 0.0)
        is_line = role in _LINE_ROLES
        best = None
        for reg in regions:
            pts = [tuple(p) for p in reg.get("points", [])]
            if len(pts) < 2:
                continue
            # AREA labels (soil boundaries) prefer the enclosing region; LINE
            # labels (gwt/surface) go to the nearest polyline edge (a line label
            # sitting inside a soil box still belongs to the nearby line).
            if not is_line and len(pts) >= 3 and _point_in_polygon(x, y, pts):
                enclosed, d = True, 0.0
            else:
                enclosed, d = False, _dist_to_polyline(x, y, pts)
                # For a LINE label (gwt/surface), a coincident dedicated line
                # (open polyline) should win over an area box sharing that edge:
                # nudge closed-area regions back slightly on ties.
                if is_line and len(pts) >= 4 and \
                        math.isclose(pts[0][0], pts[-1][0], abs_tol=1e-6) and \
                        math.isclose(pts[0][1], pts[-1][1], abs_tol=1e-6):
                    d += 1e-3
            if best is None or d < best["distance"]:
                best = {"label": tb.get("text", ""), "role": role,
                        "color": reg.get("color"),
                        "method": "enclosing" if enclosed else "nearest",
                        "distance": round(d, 3)}
        if best is not None:
            out.append(best)
    return out


def propose_role_mapping(regions: List[Dict[str, Any]],
                         text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Propose a colour->role ``role_mapping`` from label->region association.

    Builds `{hex_color: role}` from the associations, resolving conflicts by the
    closest (enclosing beats nearest) label per colour. PROPOSAL ONLY — returned
    with `applied: False` and the per-colour provenance; confirm before passing to
    ``extract_vector_geometry(role_mapping=...)``.

    Returns
    -------
    dict
        {"role_mapping": {hex: role}, "associations": [...], "applied": False,
         "note": "..."}
    """
    assoc = associate_labels_to_regions(regions, text_blocks)
    # Best association per colour (smallest distance wins).
    best_by_color: Dict[str, Dict[str, Any]] = {}
    for a in assoc:
        c = a["color"]
        if c is None:
            continue
        if c not in best_by_color or a["distance"] < best_by_color[c]["distance"]:
            best_by_color[c] = a
    role_mapping = {c: a["role"] for c, a in best_by_color.items()}
    return {
        "role_mapping": role_mapping,
        "associations": assoc,
        "applied": False,
        "note": ("Proposed role_mapping only — NOT applied. Review the "
                 "associations and pass a confirmed role_mapping to "
                 "extract_vector_geometry(role_mapping=...)."),
    }
