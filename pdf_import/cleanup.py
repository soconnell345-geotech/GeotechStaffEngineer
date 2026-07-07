"""
Geometry cleanup for PDF cross-section import (C3).

Vector geometry extracted from PDFs is often noisy: duplicated points, tiny
gaps between polylines that should meet, hundreds of collinear points along a
straight surface, and endpoints that miss each other by a fraction of a unit.
This module cleans a polyline (or a set of polylines) BEFORE
``build_slope_geometry`` / ``build_fem_inputs``. All operations are
tolerance-parametrized and pure (no PDF dependency), so they are directly
testable.

Coordinates are (x, y)/(x, z) tuples; tolerances are in the same units.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Polyline = List[Point]


def dedupe_consecutive(points: Sequence[Point], tol: float = 1e-6) -> Polyline:
    """Drop consecutive points within ``tol`` of each other."""
    out: Polyline = []
    for p in points:
        p = (float(p[0]), float(p[1]))
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def merge_collinear(points: Sequence[Point], angle_tol_deg: float = 1.0
                    ) -> Polyline:
    """Remove interior points that lie (within ``angle_tol_deg``) on the straight
    line between their neighbours — thinning a densely-sampled straight run."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) <= 2:
        return pts
    # Collinear = the two edges point the same way (cos ~ +1). Keep the vertex
    # only when the path bends by more than angle_tol_deg (cos below the tol).
    cos_tol = math.cos(math.radians(angle_tol_deg))
    out: Polyline = [pts[0]]
    for i in range(1, len(pts) - 1):
        ax, ay = out[-1]
        bx, by = pts[i]
        cx, cy = pts[i + 1]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        if cosang < cos_tol:      # a real bend -> keep the vertex
            out.append((bx, by))
    out.append(pts[-1])
    return out


def cleanup_polyline(points: Sequence[Point], tol: float = 1e-4,
                     angle_tol_deg: float = 1.0) -> Polyline:
    """Dedupe + collinear-thin a single polyline."""
    return merge_collinear(dedupe_consecutive(points, tol), angle_tol_deg)


def snap_endpoints(polylines: Sequence[Polyline], tol: float = 1e-3
                   ) -> List[Polyline]:
    """Snap near-coincident polyline ENDPOINTS to a shared cluster centroid.

    Only endpoints (first/last vertex of each polyline) participate; interior
    vertices are untouched. Endpoints within ``tol`` are averaged so touching
    polylines share an exact vertex.
    """
    polys = [[(float(x), float(y)) for x, y in pl] for pl in polylines if pl]
    # Collect endpoint references: (poly_index, vertex_index).
    refs: List[Tuple[int, int]] = []
    for i, pl in enumerate(polys):
        refs.append((i, 0))
        refs.append((i, len(pl) - 1))
    # Cluster endpoints by proximity. Each point joins the LOWEST-index existing
    # cluster whose REPRESENTATIVE (its creating point) is within tol, else
    # starts a new cluster. A grid hash (cell = tol) makes candidate lookup O(1)
    # instead of scanning every cluster: two points within Euclidean tol have
    # cell indices differing by at most 1 per axis, so the 3x3 neighbourhood
    # holds every representative within tol; the hypot(<=tol) filter and the
    # lowest-index tie-break are unchanged, so the clustering is bit-identical to
    # the former O(n^2) scan (fuzz-verified).
    clusters: List[List[Tuple[int, int]]] = []
    reps: List[Tuple[float, float]] = []       # representative (x, y) per cluster
    inv = 1.0 / tol if tol > 0 else None
    grid: Dict[Tuple[int, int], List[int]] = {}
    offsets = ([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
               if inv is not None else [(0, 0)])

    def _cell(x: float, y: float) -> Tuple[int, int]:
        if inv is None:                        # tol <= 0: only exact coincidence
            return (x, y)
        return (math.floor(x * inv), math.floor(y * inv))

    for r in refs:
        px, py = polys[r[0]][r[1]]
        base = _cell(px, py)
        best = -1
        for dx, dy in offsets:
            for ci in grid.get((base[0] + dx, base[1] + dy), ()):
                qx, qy = reps[ci]
                if math.hypot(px - qx, py - qy) <= tol and (best == -1 or ci < best):
                    best = ci
        if best != -1:
            clusters[best].append(r)
        else:
            ci = len(clusters)
            clusters.append([r])
            reps.append((px, py))
            grid.setdefault(base, []).append(ci)
    for cl in clusters:
        if len(cl) < 2:
            continue
        cx = sum(polys[i][j][0] for i, j in cl) / len(cl)
        cy = sum(polys[i][j][1] for i, j in cl) / len(cl)
        for i, j in cl:
            polys[i][j] = (cx, cy)
    return polys


def join_polylines(polylines: Sequence[Polyline], tol: float = 1e-3
                   ) -> List[Polyline]:
    """Greedily join polylines whose endpoints coincide (within ``tol``) into
    longer chains (reversing segments as needed)."""
    remaining = [[(float(x), float(y)) for x, y in pl] for pl in polylines if len(pl) >= 2]
    chains: List[Polyline] = []
    while remaining:
        chain = remaining.pop(0)
        extended = True
        while extended:
            extended = False
            for k, seg in enumerate(remaining):
                s, e = seg[0], seg[-1]
                ce, cs = chain[-1], chain[0]

                def near(a, b):
                    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol

                if near(ce, s):
                    chain = chain + seg[1:]
                elif near(ce, e):
                    chain = chain + list(reversed(seg))[1:]
                elif near(cs, e):
                    chain = seg[:-1] + chain
                elif near(cs, s):
                    chain = list(reversed(seg))[:-1] + chain
                else:
                    continue
                remaining.pop(k)
                extended = True
                break
        chains.append(chain)
    return chains


def cleanup_geometry(
    surface_points: Optional[Sequence[Point]] = None,
    boundary_profiles: Optional[Dict[str, List[Point]]] = None,
    gwt_points: Optional[Sequence[Point]] = None,
    tol: float = 1e-4,
    snap_tol: float = 1e-3,
    angle_tol_deg: float = 1.0,
    join: bool = False,
) -> Dict[str, Any]:
    """Clean an extracted cross-section: dedupe + collinear-thin each polyline,
    optionally join broken segments, and (across the surface + boundaries) snap
    near-coincident endpoints. Returns cleaned copies + a per-item point-count
    report. Non-destructive (inputs untouched)."""
    report: Dict[str, Any] = {"before": {}, "after": {}}

    def _clean_one(pts):
        return cleanup_polyline(pts, tol, angle_tol_deg) if pts else []

    surf = _clean_one(list(surface_points) if surface_points else [])
    bounds = {name: _clean_one(pl) for name, pl in (boundary_profiles or {}).items()}
    gwt = _clean_one(list(gwt_points) if gwt_points else []) if gwt_points else None

    if snap_tol and (surf or bounds):
        keys = (["__surface__"] if surf else []) + list(bounds.keys())
        polys = ([surf] if surf else []) + [bounds[k] for k in bounds]
        snapped = snap_endpoints(polys, snap_tol)
        idx = 0
        if surf:
            surf = snapped[idx]; idx += 1
        for k in bounds:
            bounds[k] = snapped[idx]; idx += 1

    if join:
        for k in list(bounds.keys()):
            joined = join_polylines([bounds[k]], tol)
            bounds[k] = max(joined, key=len) if joined else bounds[k]

    report["before"]["surface"] = len(surface_points) if surface_points else 0
    report["after"]["surface"] = len(surf)
    report["before"]["boundaries"] = {n: len(p) for n, p in (boundary_profiles or {}).items()}
    report["after"]["boundaries"] = {n: len(p) for n, p in bounds.items()}
    if gwt_points is not None:
        report["before"]["gwt"] = len(gwt_points)
        report["after"]["gwt"] = len(gwt) if gwt else 0

    return {
        "surface_points": surf,
        "boundary_profiles": bounds,
        "gwt_points": gwt,
        "report": report,
    }
