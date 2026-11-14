"""What a search's report shows the engineer, read off the stored result
(GM-140 step 10, "reporting as printed"). Nothing here computes a factor
of safety: every value is one the search stored.

* ``centre_contours``: the minimum factor of safety at each centre of a
  circular grid search, laid out on the grid for contouring, with whether
  the lowest contour closes inside the grid (D46). FHWA NHI-06-088
  Fig. 6-13, p. 6-23 (contours of the minimum factor of safety over the
  trial centres); GEC 11 Vol. II p. 9-24 ("a contour plot of lowest safety
  factors about trial failure circle centroids is recommended"); Duncan,
  Buchignani and De Wet (1987), Fig. 28: the contours should enclose the
  minimum.
* ``lowest_surfaces``: the ten lowest surfaces in order, each with its
  factor of safety, family, method and geometry (D47). FHWA NHI-06-088
  p. 6-49 ("ordered output and plotting capability for the failure surface
  of 10 minimum safety factors"); EM 1110-2-1902 4-3b(9)(b), p. 4-8 (the
  critical surface and the other significant surfaces with their factors
  of safety, and for circles the centre and radius).

The rule for "closes inside the grid": a contour at a level L encloses
the lowest centre without reaching the grid's edge when every centre on
the edge is above L and the lowest centre inside is below it; so a closed
contour exists exactly when the lowest value inside the grid is below the
lowest value on its edge. That is a reading of the contour plot the
references ask for, not a printed formula.
"""

from typing import Any, Dict, List, Optional

#: The pages the two views are drawn from.
CONTOUR_SOURCE = ("FHWA NHI-06-088 Fig. 6-13, p. 6-23; GEC 11 Vol. II "
                  "p. 9-24; Duncan, Buchignani and De Wet (1987), Fig. 28")
LOWEST_SOURCE = "FHWA NHI-06-088 p. 6-49; EM 1110-2-1902 4-3b(9)(b), p. 4-8"

_NO_FOS = 900.0     # the solver's 999.9 marker is not a factor of safety


def _axis(lo: float, hi: float, n: int) -> List[float]:
    """The grid's centre coordinates, as ``search.grid_search`` places
    them."""
    if n <= 1:
        return [round((lo + hi) / 2.0, 4)]
    return [round(lo + (hi - lo) * i / (n - 1), 4) for i in range(n)]


def _edge_name(ix: int, iy: int, nx: int, ny: int) -> str:
    sides = []
    if ix == 0:
        sides.append("left (lowest centre x)")
    if ix == nx - 1:
        sides.append("right (highest centre x)")
    if iy == 0:
        sides.append("bottom (lowest centre height)")
    if iy == ny - 1:
        sides.append("top (highest centre height)")
    return " and ".join(sides)


def centre_contours(search) -> Optional[Dict[str, Any]]:
    """The minimum FOS over the centres of a circular grid search, on the
    grid (D46), or None for a search without a centre grid.

    Returns ``{"x", "y", "fos", "minimum", "edge_minimum", "closed",
    "statement", "n_centres", "n_scored", "n_screen_only",
    "critical_centre", "source"}``: ``fos[iy][ix]`` is the minimum FOS of
    the circles about the centre (``x[ix]``, ``y[iy]``), scored by the
    method (None where the method solved no circle about it or the centre
    was not scored by it: a centre a screen alone scored is counted in
    ``n_screen_only`` and left out, so the contours compare one method).
    ``closed``: the lowest contour closes inside the grid (the lowest value
    inside is below every value on the edge); ``statement`` says so in
    words, with where to widen the grid when it does not.
    """
    cg = getattr(search, "centre_grid", None) or {}
    if not cg:
        return None
    nx, ny = int(cg["nx"]), int(cg["ny"])
    xs = _axis(cg["x_range"][0], cg["x_range"][1], nx)
    ys = _axis(cg["y_range"][0], cg["y_range"][1], ny)
    fos: List[List[Optional[float]]] = [[None] * nx for _ in range(ny)]
    n_screen = 0
    for g in getattr(search, "grid_fos", None) or []:
        node = g.get("node")
        if node is None:
            continue                    # a refinement centre, off the grid
        if g.get("method"):
            n_screen += 1               # a screen value only
            continue
        if g.get("FOS") is None or g["FOS"] >= _NO_FOS:
            continue
        ix, iy = node
        fos[iy][ix] = float(g["FOS"])

    cells = [(fos[iy][ix], ix, iy) for iy in range(ny) for ix in range(nx)
             if fos[iy][ix] is not None]
    crit = getattr(search, "critical", None)
    critical_centre = None
    if crit is not None and getattr(crit, "is_circular", False) \
            and crit.radius > 0:
        critical_centre = {
            "xc": round(crit.xc, 3), "yc": round(crit.yc, 3),
            "radius": round(crit.radius, 3),
            "FOS": None if crit.FOS is None else round(crit.FOS, 4),
            "family": getattr(search, "governing_surface", "") or "circular"}
    out: Dict[str, Any] = {
        "x": xs, "y": ys, "fos": fos, "n_centres": nx * ny,
        "n_scored": len(cells), "n_screen_only": n_screen,
        "critical_centre": critical_centre, "source": CONTOUR_SOURCE,
        "minimum": None, "edge_minimum": None, "closed": False}
    if not cells:
        out["statement"] = ("No centre of the grid has a factor of safety "
                            "by the method, so there is nothing to contour.")
        return out

    def on_edge(ix, iy):
        return ix in (0, nx - 1) or iy in (0, ny - 1)

    low = min(cells)
    inside = [c for c in cells if not on_edge(c[1], c[2])]
    edge = [c for c in cells if on_edge(c[1], c[2])]
    low_in = min(inside) if inside else None
    low_edge = min(edge) if edge else None
    out["minimum"] = {"FOS": round(low[0], 4), "xc": xs[low[1]],
                      "yc": ys[low[2]], "node": (low[1], low[2]),
                      "on_edge": on_edge(low[1], low[2])}
    out["edge_minimum"] = (None if low_edge is None else
                           {"FOS": round(low_edge[0], 4),
                            "xc": xs[low_edge[1]], "yc": ys[low_edge[2]],
                            "edge": _edge_name(low_edge[1], low_edge[2],
                                               nx, ny)})
    closed = low_in is not None and (low_edge is None
                                     or low_in[0] < low_edge[0])
    out["closed"] = closed
    where = f"({xs[low[1]]:.2f}, {ys[low[2]]:.2f})"
    if closed:
        tail = (f", below every value on the grid's edge (lowest there "
                f"{low_edge[0]:.3f})" if low_edge is not None else "")
        out["statement"] = (
            f"The lowest contour closes inside the centre grid: the lowest "
            f"centre value, {low[0]:.3f} at {where}, is inside the grid"
            f"{tail}, so a contour between the two encloses the minimum "
            f"({CONTOUR_SOURCE}).")
    elif low_in is None:
        out["statement"] = (
            f"The lowest contour does not close inside the centre grid: "
            f"the grid ({nx} x {ny}) has no centre inside its edge with a "
            f"value, so no contour can enclose the minimum ({low[0]:.3f} "
            f"at {where}). Use a grid of at least 3 x 3 centres "
            f"({CONTOUR_SOURCE}).")
    else:
        side = _edge_name(low_edge[1], low_edge[2], nx, ny)
        out["statement"] = (
            f"The lowest contour does not close inside the centre grid: "
            f"the value {low_edge[0]:.3f} at "
            f"({xs[low_edge[1]]:.2f}, {ys[low_edge[2]]:.2f}) on the grid's "
            f"{side} edge is at or below the lowest value inside the grid "
            f"({low_in[0]:.3f}), so the minimum may lie outside the grid. "
            f"Extend the centre grid (x_range / y_range) past that edge "
            f"({CONTOUR_SOURCE}).")
    if n_screen:
        out["statement"] += (
            f" {n_screen} centre(s) scored by the screen alone are left "
            f"out of the contours (not the method's values).")
    return out


def _circle_key(xc, yc, r):
    return ("c", round(float(xc), 2), round(float(yc), 2), round(float(r), 2))


def _poly_key(points):
    return ("p",) + tuple((round(float(x), 2), round(float(z), 2))
                          for x, z in points)


def lowest_surfaces(search, n: int = 10) -> List[Dict[str, Any]]:
    """The ``n`` lowest surfaces the search kept, in order of factor of
    safety (D47): each ``{"rank", "FOS", "family", "method", "xc", "yc",
    "radius"}`` for a circle or ``{..., "points"}`` for a plane or
    polyline.

    The pool is every trial of the main search scored by the method (the
    centres' best circles of a grid or entry/exit search, the polylines of
    a noncircular one, the critical circle's polyline among them) and the
    lowest surface of each other family (toe circles, circles above the
    toe, shallow planes: those families keep only their lowest surface).
    A value a screen alone gave is not in the pool. Equal surfaces are
    listed once.
    """
    methods = {}
    for rec in getattr(search, "candidate_methods", None) or []:
        have = methods.get(rec["family"])
        if have is None or rec["FOS"] < have[0]:
            methods[rec["family"]] = (rec["FOS"], rec["method"])
    named = getattr(search, "named_method", "") or ""

    def method_of(family):
        return methods.get(family, (None, named))[1]

    pool: Dict[tuple, Dict[str, Any]] = {}

    def add(key, entry):
        have = pool.get(key)
        if have is None or entry["FOS"] < have["FOS"]:
            pool[key] = entry

    for g in getattr(search, "grid_fos", None) or []:
        if g.get("method") or not g.get("R"):
            continue
        f = g.get("FOS")
        if f is None or f >= _NO_FOS:
            continue
        add(_circle_key(g["xc"], g["yc"], g["R"]),
            {"FOS": round(float(f), 4), "family": "circular",
             "method": method_of("circular"), "xc": g["xc"], "yc": g["yc"],
             "radius": g["R"]})
    for t in getattr(search, "trial_surfaces", None) or []:
        f = t.get("FOS")
        if f is None or f >= _NO_FOS or not t.get("points"):
            continue
        entry = {"FOS": round(float(f), 4), "family": "noncircular",
                 "method": method_of("noncircular"),
                 "points": [tuple(p) for p in t["points"]]}
        if t.get("from"):
            entry["note"] = t["from"]
        add(_poly_key(t["points"]), entry)
    for family, s in (getattr(search, "family_surfaces", None) or {}).items():
        entry = {"FOS": round(float(s["FOS"]), 4), "family": family,
                 "method": s.get("method") or method_of(family)}
        if "radius" in s:
            entry.update(xc=s["xc"], yc=s["yc"], radius=s["radius"])
            key = _circle_key(s["xc"], s["yc"], s["radius"])
        else:
            entry["points"] = [tuple(p) for p in s["points"]]
            key = _poly_key(s["points"])
        add(key, entry)
    ranked = sorted(pool.values(), key=lambda e: e["FOS"])[:n]
    for i, e in enumerate(ranked, 1):
        e["rank"] = i
    return [{"rank": e.pop("rank"), **e} for e in ranked]


def lowest_surfaces_rows(surfaces) -> List[List[str]]:
    """The ordered list as table rows: rank, FOS, family, method,
    geometry."""
    from slope_stability.search_supplements import describe_surface_kind
    from slope_stability.solution import METHOD_TEXT
    rows = []
    for s in surfaces:
        if "radius" in s:
            geo = (f"centre ({s['xc']:.2f}, {s['yc']:.2f}) m, "
                   f"R = {s['radius']:.2f} m")
        else:
            pts = s["points"]
            geo = ("polyline " + ", ".join(
                f"({x:.2f}, {z:.2f})" for x, z in pts))
            if s.get("note"):
                geo += f" ({s['note']})"
        rows.append([str(s["rank"]), f"{s['FOS']:.3f}",
                     describe_surface_kind(s["family"]),
                     METHOD_TEXT.get(s["method"], s["method"] or ""), geo])
    return rows
