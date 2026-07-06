"""
Vision <-> vector cross-check for PDF cross-section import (C5).

The vector extraction (``extract_vector_geometry``, exact geometry but needs a
correct role_mapping) and the vision extraction (``extract_geometry_vision``,
robust to any drawing but approximate) can disagree. ``cross_check`` compares two
``PdfParseResult``-shaped extractions feature by feature and produces a
DISCREPANCY REPORT: which features are present in both / only one, and the
vertical deviation (RMS / max) between the polylines where they overlap. Nothing
is auto-merged — the report is for the caller / user to reconcile (the
geo_project provenance-quarantine spirit).

Both inputs are dicts or PdfParseResult with ``surface_points`` [(x, z), ...],
``boundary_profiles`` {name: [(x, z), ...]} and optional ``gwt_points``.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def _as_points(seq) -> List[Point]:
    out = []
    for p in (seq or []):
        if isinstance(p, dict):
            out.append((float(p.get("x", 0.0)), float(p.get("z", 0.0))))
        else:
            out.append((float(p[0]), float(p[1])))
    return out


def _get(result, attr):
    if isinstance(result, dict):
        return result.get(attr)
    return getattr(result, attr, None)


def _interp_z(polyline: Sequence[Point], x: float) -> Optional[float]:
    """Linear-interpolate z at x along a polyline (None if x is out of range)."""
    pts = sorted(polyline, key=lambda p: p[0])
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for i in range(len(pts) - 1):
        x0, z0 = pts[i]
        x1, z1 = pts[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return (z0 + z1) / 2.0
            t = (x - x0) / (x1 - x0)
            return z0 + t * (z1 - z0)
    return pts[-1][1]


def polyline_deviation(a: Sequence[Point], b: Sequence[Point],
                       n_samples: int = 25) -> Optional[Dict[str, float]]:
    """Vertical deviation (z) between two polylines over their overlapping x-range.

    Returns {"rms", "max", "mean", "overlap_frac", "n"} or None if the polylines
    do not overlap in x (or either is empty).
    """
    a = _as_points(a)
    b = _as_points(b)
    if len(a) < 1 or len(b) < 1:
        return None
    ax = [p[0] for p in a]
    bx = [p[0] for p in b]
    lo = max(min(ax), min(bx))
    hi = min(max(ax), max(bx))
    if hi <= lo:
        return None
    devs = []
    for i in range(n_samples):
        x = lo + (hi - lo) * i / (n_samples - 1) if n_samples > 1 else lo
        za = _interp_z(a, x)
        zb = _interp_z(b, x)
        if za is not None and zb is not None:
            devs.append(abs(za - zb))
    if not devs:
        return None
    full = max(max(ax), max(bx)) - min(min(ax), min(bx))
    return {
        "rms": round((sum(d * d for d in devs) / len(devs)) ** 0.5, 4),
        "max": round(max(devs), 4),
        "mean": round(sum(devs) / len(devs), 4),
        "overlap_frac": round((hi - lo) / full, 3) if full > 0 else 1.0,
        "n": len(devs),
    }


def cross_check(vector_result, vision_result, tol: float = 0.5,
                n_samples: int = 25) -> Dict[str, Any]:
    """Compare a vector and a vision extraction; return a discrepancy report.

    Parameters
    ----------
    vector_result, vision_result : dict or PdfParseResult
        The two extractions to compare.
    tol : float
        Vertical-deviation tolerance (same units as the geometry) below which a
        feature is considered to AGREE. Default 0.5.
    n_samples : int
        Samples used for the deviation estimate. Default 25.

    Returns
    -------
    dict
        {"features": [...], "surface": {...}, "boundaries": {...}, "gwt": {...},
         "agree": bool, "note": str}. Each feature entry: {name, in_vector,
         in_vision, deviation, agrees}.
    """
    features: List[Dict[str, Any]] = []

    def _compare(name, a, b):
        in_a = bool(a)
        in_b = bool(b)
        dev = polyline_deviation(a, b, n_samples) if (in_a and in_b) else None
        agrees = bool(dev is not None and dev["max"] <= tol)
        entry = {
            "name": name,
            "in_vector": in_a,
            "in_vision": in_b,
            "n_vector": len(_as_points(a)) if in_a else 0,
            "n_vision": len(_as_points(b)) if in_b else 0,
            "deviation": dev,
            "agrees": agrees,
        }
        if in_a != in_b:
            entry["note"] = ("only in " + ("vector" if in_a else "vision"))
        elif dev is None and in_a and in_b:
            entry["note"] = "no x-overlap"
        features.append(entry)
        return entry

    surf = _compare("surface", _get(vector_result, "surface_points"),
                    _get(vision_result, "surface_points"))

    vec_b = _get(vector_result, "boundary_profiles") or {}
    vis_b = _get(vision_result, "boundary_profiles") or {}
    boundaries = {}
    for name in sorted(set(vec_b) | set(vis_b)):
        boundaries[name] = _compare(f"boundary:{name}",
                                    vec_b.get(name), vis_b.get(name))

    gwt = _compare("gwt", _get(vector_result, "gwt_points"),
                   _get(vision_result, "gwt_points"))

    # Overall agreement: every feature present in BOTH agrees within tol, and no
    # feature is present in only one extraction.
    both = [f for f in features if f["in_vector"] and f["in_vision"]]
    only_one = [f for f in features if f["in_vector"] != f["in_vision"]]
    agree = (not only_one) and all(f["agrees"] for f in both) and len(both) > 0

    return {
        "features": features,
        "surface": surf,
        "boundaries": boundaries,
        "gwt": gwt,
        "agree": agree,
        "tolerance": tol,
        "note": ("Discrepancy report only — nothing merged. Reconcile the "
                 "features flagged 'only in ...' or with deviation > tol."),
    }
