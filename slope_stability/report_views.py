"""What a search's report shows the engineer, read off the stored result
(GM-140 step 10, "reporting as printed"). Nothing here computes a factor
of safety: every value is one the search stored.

* ``centre_contours``: the minimum factor of safety at each centre of a
  circular grid search, laid out on the grid for contouring, with whether
  the lowest contour closes inside the grid (D46). FHWA NHI-06-088
  pp. 6-22 to 6-23 and Fig. 6-13 ("A plot of minimum safety factor for
  each circle center in the form of contours can be used to define the
  location of the most critical circle"); GEC 11 Vol. II p. 9-24 ("A
  contour plot of lowest safety factor values about the trial failure
  circle centroids is recommended"); Duncan, Buchignani and De Wet (1987),
  Fig. 28: the contours should enclose the minimum.
* ``lowest_surfaces``: the ten lowest surfaces in order, each with its
  factor of safety, family, method and geometry (D47). FHWA NHI-06-088
  p. 6-49 ("ordered output and plotting capability for the failure surface
  of 10 minimum safety factors"); EM 1110-2-1902 4-3b(9)(b), p. 4-8 (the
  critical surface and the other significant surfaces with their factors
  of safety, and for circles the centre and radius).

The rule for "closes inside the grid" (the orchestrator's ruling on the
step's review, D1 / Q2): a contour at the level midway between the lowest
value inside the grid and the lowest value on its edge encloses the
lowest centre without reaching the edge when the lowest value inside is
below every edge value and every centre next to the centres below that
level (diagonal neighbours included, as a contour is traced through the
grid's cells) has a value; where such a centre has none, whether it
closes cannot be judged, and the statement says how many centres have a
value. That is a reading of the contour plot the references ask for, not
a printed formula.

The values contoured (the ruling's Q1): the method's own minimum at each
centre where it has values enough for that rule; for a screened search
(Spencer, Morgenstern-Price or GLE named, whose values exist only at the
centres re-scored), otherwise the screen's minimum at each centre, by
Bishop's simplified method, labelled as the screen's, with the named
method's lowest surface marked beside it (FHWA Fig. 6-13 is a tool for
locating the critical circle, which the screen does).

The pool of the ordered list (the ruling's Q3): every family's ten lowest
scored surfaces (``SearchResult.family_lowest``), by the method that
scored the family, more than one circle about a centre allowed, each
surface once to 0.01 m, never a value a screen alone gave.
"""

from typing import Any, Dict, List, Optional

#: The pages the two views are drawn from.
CONTOUR_SOURCE = ("FHWA NHI-06-088 Fig. 6-13, p. 6-23; GEC 11 Vol. II "
                  "p. 9-24; Duncan, Buchignani and De Wet (1987), Fig. 28")
LOWEST_SOURCE = "FHWA NHI-06-088 p. 6-49; EM 1110-2-1902 4-3b(9)(b), p. 4-8"

#: The words the report labels the screen's contours with (Q1).
SCREEN_TEXT = "Bishop's simplified method (the screen)"

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


def _method_text(method: str) -> str:
    from slope_stability.solution import METHOD_TEXT
    text = METHOD_TEXT.get(method, method or "the method")
    return text[0].lower() + text[1:] if text.startswith("The ") else text


def _main_method(search) -> str:
    """The method that scored the main search's centres."""
    best = None
    for rec in getattr(search, "candidate_methods", None) or []:
        if rec["family"] == "circular" and (best is None
                                            or rec["FOS"] < best["FOS"]):
            best = rec
    if best is not None:
        return best["method"]
    return getattr(search, "named_method", "") or ""


def _judge(fos, nx: int, ny: int) -> Dict[str, Any]:
    """The D1 rule on one grid of values (None where a centre has none):
    ``{"verdict", "low", "low_in", "low_edge", "level"}``, verdict
    'none' (no value), 'no_inside' (no centre inside the edge: a grid
    under 3 x 3), 'open' (an edge value at or below the lowest inside),
    'closed', or 'unjudged' (a centre next to the lowest values, or every
    edge or inside centre, has no value)."""
    cells = [(fos[iy][ix], ix, iy) for iy in range(ny) for ix in range(nx)
             if fos[iy][ix] is not None]
    out = {"verdict": "none", "low": None, "low_in": None, "low_edge": None,
           "level": None}
    if not cells:
        return out

    def on_edge(ix, iy):
        return ix in (0, nx - 1) or iy in (0, ny - 1)

    inside = [c for c in cells if not on_edge(c[1], c[2])]
    edge = [c for c in cells if on_edge(c[1], c[2])]
    out.update(low=min(cells), low_in=min(inside) if inside else None,
               low_edge=min(edge) if edge else None)
    if nx < 3 or ny < 3:
        out["verdict"] = "no_inside"
    elif not inside or not edge:
        out["verdict"] = "unjudged"
    elif out["low_edge"][0] <= out["low_in"][0]:
        out["verdict"] = "open"
    else:
        level = 0.5 * (out["low_in"][0] + out["low_edge"][0])
        out["level"] = level
        ok = True
        for v, ix, iy in inside:
            if v >= level:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    jx, jy = ix + dx, iy + dy
                    if (0 <= jx < nx and 0 <= jy < ny
                            and fos[jy][jx] is None):
                        ok = False
        out["verdict"] = "closed" if ok else "unjudged"
    return out


def _grids(search, nx: int, ny: int):
    """(the method's values, the screen's values, n screen only, the
    screen's method): the per-centre minima on the grid."""
    by_method = [[None] * nx for _ in range(ny)]
    by_screen = [[None] * nx for _ in range(ny)]
    n_screen = 0
    screen = ""
    for g in getattr(search, "grid_fos", None) or []:
        node = g.get("node")
        if node is None:
            continue                    # a refinement centre, off the grid
        ix, iy = node
        f = g.get("FOS")
        if g.get("method"):
            n_screen += 1               # a screen value only
            screen = screen or g["method"]
            if f is not None and f < _NO_FOS:
                by_screen[iy][ix] = float(f)
            continue
        if f is not None and f < _NO_FOS:
            by_method[iy][ix] = float(f)
        sf = g.get("screen_FOS")
        if sf is not None and sf < _NO_FOS:
            by_screen[iy][ix] = float(sf)
            screen = screen or g.get("screen_method", "")
    return by_method, by_screen, n_screen, screen


def contour_levels(contours: Optional[Dict[str, Any]]) -> List[float]:
    """The contour levels the figure draws: four between the lowest value
    and 1.3 times it, and, where the lowest contour closes inside the
    grid, the level the statement judges it at (midway between the lowest
    inside and the lowest edge value); each kept only where a cell of the
    grid (four neighbouring centres, all with a value) spans it, so that a
    level kept is drawn. Empty: no contour can be drawn."""
    if not contours or not contours.get("n_scored"):
        return []
    fos = contours["fos"]
    ny, nx = len(fos), len(fos[0]) if fos else 0
    vals = [v for row in fos for v in row if v is not None]
    if len(vals) < 4 or nx < 2 or ny < 2 or max(vals) <= min(vals):
        return []
    lo = min(vals)
    hi = min(max(vals), lo * 1.3)
    levels = [lo + (hi - lo) * i / 4.0 for i in range(1, 5)]
    if contours.get("verdict") == "closed" and contours.get("level"):
        # the closed contour the statement judges, midway between the
        # lowest inside and the lowest edge value, drawn with the others
        levels = sorted(set(levels) | {round(contours["level"], 4)})
    spans = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            c = [fos[iy][ix], fos[iy][ix + 1], fos[iy + 1][ix],
                 fos[iy + 1][ix + 1]]
            if all(v is not None for v in c):
                spans.append((min(c), max(c)))
    return [L for L in levels if any(a < L < b for a, b in spans)]


def no_contour_reason(contours: Optional[Dict[str, Any]]) -> str:
    """Why no contour is drawn (``contour_levels`` empty), as a clause:
    the search has no centre grid, or its centres have too few values."""
    if contours is None:
        return ("the search has no grid of circle centres (an entry and "
                "exit or a noncircular search)")
    k, n, by = contours["n_scored"], contours["n_centres"], \
        contours["values_by"]
    if not k:
        return f"no centre of the grid has a factor of safety by {by}"
    fos = contours["fos"]
    full = any(all(v is not None for v in (fos[iy][ix], fos[iy][ix + 1],
                                           fos[iy + 1][ix],
                                           fos[iy + 1][ix + 1]))
               for iy in range(len(fos) - 1) for ix in range(len(fos[0]) - 1))
    if not full:
        return (f"{k} of {n} centres of the grid have a factor of safety by "
                f"{by}, and no four neighbouring centres all have one")
    return (f"the values at the {k} centres with a factor of safety by {by} "
            f"do not vary across any cell of four neighbouring centres")


def centre_contours(search) -> Optional[Dict[str, Any]]:
    """The minimum FOS over the centres of a circular grid search, on the
    grid (D46), or None for a search without a centre grid.

    Returns ``{"x", "y", "fos", "basis", "values_by", "minimum",
    "edge_minimum", "closed", "verdict", "verdict_text", "statement",
    "n_centres", "n_scored", "n_by_method", "n_screen_only",
    "critical_centre", "named_lowest", "source"}``: ``fos[iy][ix]`` is
    the minimum FOS of the circles about the centre (``x[ix]``,
    ``y[iy]``) contoured, by the method that scored the main search
    (``basis`` 'method'), or, for a screened search whose method has too
    few values for the rule (module docstring), by the screen (``basis``
    'screen'); ``values_by`` names the method in words. ``closed``: True
    (the lowest contour closes inside the grid), False (it does not: an
    edge value is at or below the lowest inside, or the grid has no
    centre inside its edge) or None (it cannot be judged); ``verdict``
    gives which ('closed', 'open', 'no_inside', 'unjudged', 'none'),
    ``verdict_text`` one sentence saying so and ``statement`` the whole
    statement, with where to widen the grid when it does not close and
    where the governing surface's centre lies outside the grid.
    ``named_lowest``: with the screen's contours, the lowest surface by
    the method named (``lowest_surfaces``' form), to be marked.
    """
    cg = getattr(search, "centre_grid", None) or {}
    if not cg:
        return None
    nx, ny = int(cg["nx"]), int(cg["ny"])
    xs = _axis(cg["x_range"][0], cg["x_range"][1], nx)
    ys = _axis(cg["y_range"][0], cg["y_range"][1], ny)
    by_method, by_screen, n_screen, screen = _grids(search, nx, ny)
    method = _main_method(search)
    n_by_method = sum(v is not None for row in by_method for v in row)
    n_by_screen = sum(v is not None for row in by_screen for v in row)
    j = _judge(by_method, nx, ny)
    basis, fos, values_by = "method", by_method, _method_text(method)
    if j["verdict"] in ("none", "unjudged") and n_by_screen:
        js = _judge(by_screen, nx, ny)
        # the screen's values only where they can be judged: a screen as
        # short of values as the method shows nothing more
        if js["verdict"] not in ("none", "unjudged"):
            basis, fos, j = "screen", by_screen, js
            values_by = (SCREEN_TEXT if (screen or "bishop") == "bishop"
                         else f"{_method_text(screen)} (the screen)")

    crit = getattr(search, "critical", None)
    critical_centre = None
    if crit is not None and getattr(crit, "is_circular", False) \
            and crit.radius > 0:
        critical_centre = {
            "xc": round(crit.xc, 3), "yc": round(crit.yc, 3),
            "radius": round(crit.radius, 3),
            "FOS": None if crit.FOS is None else round(crit.FOS, 4),
            "family": getattr(search, "governing_surface", "") or "circular"}
    n_scored = sum(v is not None for row in fos for v in row)
    out: Dict[str, Any] = {
        "x": xs, "y": ys, "fos": fos, "basis": basis,
        "values_by": values_by, "method_text": _method_text(method),
        "n_centres": nx * ny, "n_scored": n_scored,
        "n_by_method": n_by_method, "n_screen_only": n_screen,
        "critical_centre": critical_centre, "named_lowest": None,
        "source": CONTOUR_SOURCE, "minimum": None, "edge_minimum": None,
        "closed": False, "verdict": j["verdict"], "level": j["level"]}

    low, low_in, low_edge = j["low"], j["low_in"], j["low_edge"]
    if low is not None:
        out["minimum"] = {"FOS": round(low[0], 4), "xc": xs[low[1]],
                          "yc": ys[low[2]], "node": (low[1], low[2]),
                          "on_edge": (low[1] in (0, nx - 1)
                                      or low[2] in (0, ny - 1))}
    if low_edge is not None:
        out["edge_minimum"] = {
            "FOS": round(low_edge[0], 4), "xc": xs[low_edge[1]],
            "yc": ys[low_edge[2]],
            "edge": _edge_name(low_edge[1], low_edge[2], nx, ny)}

    def at(c):
        return f"({xs[c[1]]:.2f}, {ys[c[2]]:.2f})"

    v = j["verdict"]
    if v == "none":
        out["closed"] = None
        text = (f"No centre of the grid has a factor of safety by "
                f"{values_by}, so there is nothing to contour.")
        out["verdict_text"] = text
        out["statement"] = text
    elif v == "closed":
        out["closed"] = True
        text = (f"The lowest contour closes inside the centre grid: the "
                f"lowest centre value by {values_by}, {low[0]:.3f} at "
                f"{at(low)}, is inside the grid, below every value on its "
                f"edge (lowest there {low_edge[0]:.3f}), and every centre "
                f"next to the centres below {j['level']:.3f} (midway) has "
                f"a value.")
        out["verdict_text"] = text
        out["statement"] = (text[:-1] + f", so a contour between the two "
                            f"encloses the minimum ({CONTOUR_SOURCE}).")
    elif v == "no_inside":
        text = (f"The lowest contour does not close inside the centre "
                f"grid: the grid ({nx} x {ny}) has no centre inside its "
                f"edge, so no contour can enclose the minimum "
                f"({low[0]:.3f} by {values_by} at {at(low)}).")
        out["verdict_text"] = text
        out["statement"] = (text + f" Use a grid of at least 3 x 3 centres "
                            f"({CONTOUR_SOURCE}).")
    elif v == "open":
        side = _edge_name(low_edge[1], low_edge[2], nx, ny)
        text = (f"The lowest contour does not close inside the centre "
                f"grid: the value {low_edge[0]:.3f} by {values_by} at "
                f"{at(low_edge)} on the grid's {side} edge is at or below "
                f"the lowest value inside the grid ({low_in[0]:.3f}), so "
                f"the minimum may lie outside the grid.")
        out["verdict_text"] = text
        out["statement"] = (text + f" Extend the centre grid (x_range / "
                            f"y_range) past that edge ({CONTOUR_SOURCE}).")
    else:                                   # unjudged
        out["closed"] = None
        n_val = n_scored
        if low_in is None:
            why = "no centre inside the grid's edge has one"
        elif low_edge is None:
            why = "no centre on the grid's edge has one"
        else:
            why = (f"a centre next to the lowest values (below "
                   f"{j['level']:.3f}, midway between the lowest inside, "
                   f"{low_in[0]:.3f} at {at(low_in)}, and the lowest on the "
                   f"edge, {low_edge[0]:.3f}) has none")
        text = (f"Whether the lowest contour closes inside the centre grid "
                f"cannot be judged: {n_val} of {nx * ny} centres have a "
                f"factor of safety by {values_by}, and {why}.")
        out["verdict_text"] = text
        out["statement"] = text[:-1] + f" ({CONTOUR_SOURCE})."

    if basis == "screen":
        named = getattr(search, "named_method", "") or method
        out["statement"] = (
            f"{_method_text(method)[0].upper()}{_method_text(method)[1:]} "
            f"has a factor of safety at {n_by_method} of {nx * ny} centres "
            f"(those it re-scored), too few to judge the contours by; the "
            f"contours are of the screen's minimum at each centre, by "
            f"{values_by}, and the lowest surface by "
            f"{_method_text(named)} is marked. " + out["statement"])
        out["named_lowest"] = _named_lowest(search, named)
    elif n_screen:
        out["statement"] += (
            f" {n_screen} centre(s) scored by the screen alone are left "
            f"out of the contours (not the method's values).")
    if critical_centre is not None:
        x_lo, x_hi = sorted(cg["x_range"])
        y_lo, y_hi = sorted(cg["y_range"])
        tol = 1e-6
        if not (x_lo - tol <= critical_centre["xc"] <= x_hi + tol
                and y_lo - tol <= critical_centre["yc"] <= y_hi + tol):
            from slope_stability.search_supplements import \
                describe_surface_kind
            line = (f"The governing surface's centre, "
                    f"({critical_centre['xc']:.2f}, "
                    f"{critical_centre['yc']:.2f}), lies outside the centre "
                    f"grid (x {x_lo:.2f} to {x_hi:.2f}, y {y_lo:.2f} to "
                    f"{y_hi:.2f}): the "
                    f"{describe_surface_kind(critical_centre['family'])} "
                    f"family found it, not the grid.")
            out["outside_line"] = line
            out["statement"] += " " + line
    return out


def _circle_key(xc, yc, r):
    return ("c", round(float(xc), 2), round(float(yc), 2), round(float(r), 2))


def _poly_key(points):
    return ("p",) + tuple((round(float(x), 2), round(float(z), 2))
                          for x, z in points)


def _key_of(s):
    if "radius" in s:
        return _circle_key(s["xc"], s["yc"], s["radius"])
    return _poly_key(s["points"])


def _pool(search) -> Dict[tuple, Dict[str, Any]]:
    """Every surface the ordered list draws from, once each (module
    docstring): each family's lowest scored surfaces by the method that
    scored it (``family_lowest``), and, for a family without them, its
    reported surface; for a search that kept no family's list (one not
    run through ``search_critical_surface``), the main search's own
    trials scored by the method and each family's reported surface."""
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
            if have is not None and have.get("note") \
                    and not entry.get("note"):
                entry["note"] = have["note"]
            pool[key] = entry

    def entry_of(s, family):
        e = {"FOS": round(float(s["FOS"]), 4), "family": family,
             "method": s.get("method") or method_of(family)}
        if "radius" in s:
            e.update(xc=round(float(s["xc"]), 4), yc=round(float(s["yc"]), 4),
                     radius=round(float(s["radius"]), 4))
            if s.get("arc"):
                e["arc"] = tuple(round(float(v), 4) for v in s["arc"])
        else:
            e["points"] = [(round(float(x), 4), round(float(z), 4))
                           for x, z in s["points"]]
        return e

    # what a trial is noted as (the critical circle's polyline)
    notes = {}
    for t in getattr(search, "trial_surfaces", None) or []:
        if t.get("from") and t.get("points"):
            notes[_poly_key(t["points"])] = t["from"]

    kept = getattr(search, "family_lowest", None) or {}
    families = getattr(search, "family_surfaces", None) or {}
    if kept:
        for family, surfaces in kept.items():
            for s in surfaces:
                e = entry_of(s, family)
                key = _key_of(e)
                if key in notes:
                    e["note"] = notes[key]
                add(key, e)
        for family, s in families.items():
            if family not in kept:
                e = entry_of(s, family)
                add(_key_of(e), e)
        return pool

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
    for family, s in families.items():
        e = entry_of(s, family)
        add(_key_of(e), e)
    return pool


def _named_lowest(search, named: str) -> Optional[Dict[str, Any]]:
    """The lowest surface in the pool by the method ``named``."""
    best = None
    for e in _pool(search).values():
        if e["method"] == named and (best is None or e["FOS"] < best["FOS"]):
            best = e
    return None if best is None else dict(best)


def lowest_surfaces(search, n: int = 10) -> List[Dict[str, Any]]:
    """The ``n`` lowest surfaces the search scored, in order of factor of
    safety (D47): each ``{"rank", "FOS", "family", "method", "xc", "yc",
    "radius"[, "arc"]}`` for a circle (``arc``: its two ends, where the
    trial was one arc of a circle meeting the ground more than twice or a
    toe-circle family's arc) or ``{..., "points"}`` for a plane or
    polyline.

    The pool (the ruling on the review's Q3): every family's ten lowest
    scored surfaces, by the method that scored the family (more than one
    circle about a centre allowed), each surface once to 0.01 m; a value
    a screen alone gave is not in it. The geometry is as scored (to
    0.0001 m), so each surface listed can be analysed again at it.
    """
    ranked = sorted(_pool(search).values(), key=lambda e: e["FOS"])[:n]
    return [{"rank": i, **e} for i, e in enumerate(ranked, 1)]


def lowest_surfaces_rows(surfaces) -> List[List[str]]:
    """The ordered list as table rows: rank, FOS, family, method,
    geometry (to 0.001 m, so that each can be analysed again)."""
    from slope_stability.search_supplements import describe_surface_kind
    from slope_stability.solution import METHOD_TEXT
    rows = []
    for s in surfaces:
        if "radius" in s:
            geo = (f"centre ({s['xc']:.3f}, {s['yc']:.3f}) m, "
                   f"R = {s['radius']:.3f} m")
            if s.get("arc"):
                geo += (f", arc from x = {s['arc'][0]:.3f} to "
                        f"{s['arc'][1]:.3f} m")
        else:
            pts = s["points"]
            geo = ("polyline " + ", ".join(
                f"({x:.3f}, {z:.3f})" for x, z in pts))
            if s.get("note"):
                geo += f" ({s['note']})"
        rows.append([str(s["rank"]), f"{s['FOS']:.3f}",
                     describe_surface_kind(s["family"]),
                     METHOD_TEXT.get(s["method"], s["method"] or ""), geo])
    return rows
