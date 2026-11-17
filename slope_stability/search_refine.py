"""The noncircular search finished as the print describes one (GM-140 step
7a; the account's D12, D43 and the noncircular part of D9; GM-117).

A noncircular search samples polylines (random, differential evolution,
particle swarm) or builds wedges along the layers. The print finishes such
a search in a way none of the samplers does:

* EM 1110-2-1902 (2003) 4-2e, p. 4-2, and 4-2e(1)(b), pp. 4-3 and 4-6:
  more than one local minimum may exist; run independently started
  searches and compare them. C-1g(4), p. C-11: a number of separate
  searches. FHWA NHI-06-088 p. 6-22, point 1: more than one local minimum.
* EM C-1g(3), p. C-10, Fig. C-8 (Celestino and Duncan 1981): hold all the
  points of the surface but one, shift that point a small distance in two
  directions, move the surface, and repeat until the factor of safety
  stops falling and the moves become minimal.
* EM 4-2e(2)(b), p. 4-7: the final shift increments are no more than 10
  to 25 % of the thickness of the thinnest stratum the surface may pass
  through.
* EM 4-2e(2), p. 4-7: the initial surface should match the estimated
  critical surface; (a) a trial surface may become too steep at the toe
  unless its orientation is restricted.
* EM 4-2e(1)(a), p. 4-3: the search must not jump from one face to
  another.
* EM C-1f(2), p. C-6, and C-1g(2), pp. C-8 to C-10, Fig. C-7b: a wedge
  suits a long linear segment through weak material; the wedge search
  varies the two ends of the central block's base, the active wedge at
  45 + phi'_D/2 and the passive at 45 - phi'_D/2, tan phi'_D = tan phi'/F.
  FHWA p. 6-44: try several positions of the wedges.
* EM C-10b, p. C-34, Eq. C-32: the passive exit inclination,
  alpha = -(45 - phi'/2).
* EM 4-2d(2), p. 4-2: the output names the most critical surface and the
  other surfaces analysed.

This module holds the pieces: the thinnest stratum and the final
increment, the wedge starts, the one-point-at-a-time refinement with its
two restrictions (the toe and the faces), and `refinement_stage`, which the
noncircular searches of `search.py` call at their end with their own
scorer. What the print does not give is a stated call, named in the
result's ``refinement`` record and in ``choices`` (`refinement_choices`).
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Final shift increment as a fraction of the thinnest stratum (call 2):
#: EM 4-2e(2)(b), p. 4-7, prints "no more than 10 to 25 percent"; 10 %
#: meets both ends of the printed range.
FINAL_FRACTION = 0.10
#: The printed range of that fraction.
FINAL_FRACTION_RANGE = (0.0, 0.25)
#: Starting shift increment, as a fraction of the slope's height (a call:
#: no print gives one). The increment is halved from it down to the final
#: increment; it is never raised to meet the final one.
START_FRACTION_H = 0.10
#: The final increment is no more than this share of the starting one (the
#: fix round's ruling 9(a), the module's call: EM 4-2e(2)(b), p. 4-7,
#: prints a maximum only, "no more than 10 to 25 percent"), so a slope of
#: thick strata still halves at least twice. Other form: exactly the
#: fraction of the thinnest stratum (as first built).
FINAL_OF_START = 0.25
#: Vertices of the surface the refinement shifts (a call: no print gives a
#: count). A start with fewer gets points added on its longest segments,
#: which leaves the surface as it is; a circle's refinement starts from the
#: polyline of its slice-base points on this many minus one slices, while
#: the circle as it stands is that polyline on the search's own slices
#: (ruling 9(b)).
N_VERTICES = 11
#: Positions of each end of a wedge's central block over the windows' span
#: (call 3's count; no print gives one; FHWA p. 6-44 asks for several),
#: after which the best wedge's two ends are halved down to the final
#: increment (ruling 10).
WEDGE_POSITIONS = 21
#: Passes of the wedge angles on the trial's F (a call): the first with
#: F = 1, each next one with the F the pass before gave; the lowest pass
#: is kept.
WEDGE_F_PASSES = 3

#: A shift is taken only when it lowers F by more than the solver's
#: tolerance (differences below it are not differences).
_SAMPLES = 60


# ---------------------------------------------------------------------------
# The thinnest stratum and the final increment (EM 4-2e(2)(b), p. 4-7)
# ---------------------------------------------------------------------------

def thinnest_stratum(geom, x_lo: float, x_hi: float,
                     z_floor: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The stratum the surface may pass through that is thinnest, with its
    least vertical thickness between ``x_lo`` and ``x_hi`` and above
    ``z_floor`` (the search's depth: the bottom of the deepest layer when
    None). A stratum is a layer where the layers listed before it leave it
    ground (``SlopeGeometry.column_at``); its thickness at x is its top,
    cut at the ground, less its bottom, cut at the floor, taken only where
    the ground does not cut it (a layer the ground cuts at every x of the
    span, a surface layer, takes its greatest thickness below the ground).
    Returns {layer, thickness, x} or None where no layer is in the span."""
    if z_floor is None:
        z_floor = min(L.bottom_elevation for L in geom.soil_layers)
    lo, hi = min(x_lo, x_hi), max(x_lo, x_hi)
    whole: Dict[int, Tuple[float, float]] = {}      # least uncut, x
    cut: Dict[int, Tuple[float, float]] = {}        # greatest cut, x
    for i in range(_SAMPLES + 1):
        x = lo + (hi - lo) * i / _SAMPLES
        zg = geom.ground_elevation_at(x)
        for layer, z_bot, z_top in geom.column_at(x, z_high=math.inf):
            bot = max(z_bot, z_floor)
            top = min(z_top, zg)
            t = top - bot
            if t <= 1e-9:
                continue
            key = id(layer)
            if z_top <= zg + 1e-9:
                if key not in whole or t < whole[key][0]:
                    whole[key] = (t, x)
            elif key not in cut or t > cut[key][0]:
                cut[key] = (t, x)
    by_id = {id(L): L for L in geom.soil_layers}
    best = None
    for key in set(whole) | set(cut):
        t, x = whole[key] if key in whole else cut[key]
        if best is None or t < best[0]:
            best = (t, x, by_id[key])
    if best is None:
        return None
    return {"layer": best[2].name, "thickness": round(best[0], 4),
            "x": round(best[1], 3)}


def slope_height(geom, x_lo: float, x_hi: float) -> float:
    """The ground's relief between ``x_lo`` and ``x_hi`` (the height the
    starting increment is a fraction of)."""
    zs = [geom.ground_elevation_at(min(x_lo, x_hi)
                                   + abs(x_hi - x_lo) * i / _SAMPLES)
          for i in range(_SAMPLES + 1)]
    return max(zs) - min(zs)


# ---------------------------------------------------------------------------
# Faces and the toe (EM 4-2e(1)(a), p. 4-3; 4-2e(2)(a), p. 4-7; Eq. C-32)
# ---------------------------------------------------------------------------

def _face_spans(geom) -> List[Tuple[float, float, float]]:
    """(x_low, x_high, x_crest) of each face group (`search_ranges`)."""
    from slope_stability.search_ranges import face_groups
    out = []
    for g in face_groups(geom):
        xt, xk = g["toe"][0], g["crest"][0]
        out.append((min(xt, xk), max(xt, xk), xk))
    return out


def end_allowed(spans, x0: float, x_new: float) -> bool:
    """May an end of a surface that started at ``x0`` move to ``x_new``
    without jumping faces (EM 4-2e(1)(a), p. 4-3)? Not across the crest of
    any face group from its start, and, for an end that started on a face
    group, not onto another face group. An end on a flat may move onto the
    face whose toe bounds its flat (crossing a crest is refused anyway)."""
    for _lo, _hi, xk in spans:
        a, b = x0 - xk, x_new - xk
        if (a > 1e-9 and b < -1e-9) or (a < -1e-9 and b > 1e-9):
            return False
    home = [s for s in spans if s[0] - 1e-9 <= x0 <= s[1] + 1e-9]
    if home:
        for s in spans:
            if s in home:
                continue
            if s[0] + 1e-9 < x_new < s[1] - 1e-9:
                return False
    return True


def _toe_index(geom, pts) -> int:
    """0 or -1: the end of the surface with the lower ground, its toe end
    (the high-x end on a tie)."""
    return 0 if pts[0][1] < pts[-1][1] - 1e-9 else -1


def exit_angle(pts, toe: int) -> float:
    """Inclination, degrees above the horizontal, of the segment that
    reaches the ground at the toe end ``toe`` (0 or -1)."""
    (x0, z0), (x1, z1) = (pts[0], pts[1]) if toe == 0 else (pts[-1], pts[-2])
    return math.degrees(math.atan2(z0 - z1, abs(x0 - x1) or 1e-12))


def passive_exit_limit(geom, pts, toe: int) -> Tuple[float, float]:
    """(limit in degrees, phi' of the exit material): EM Eq. C-32, p. C-34,
    alpha = -(45 - phi'/2), with the phi' of the material at the middle of
    the toe-end segment (45 degrees where phi' = 0)."""
    (x0, z0), (x1, z1) = (pts[0], pts[1]) if toe == 0 else (pts[-1], pts[-2])
    lay = geom.layer_at_point(0.5 * (x0 + x1), 0.5 * (z0 + z1))
    phi = 0.0 if lay is None else float(lay.shear_strength_params[1])
    return 45.0 - phi / 2.0, phi


# ---------------------------------------------------------------------------
# Starts
# ---------------------------------------------------------------------------

def with_vertices(pts: Sequence[Tuple[float, float]], n: int = N_VERTICES
                  ) -> List[Tuple[float, float]]:
    """The polyline ``pts`` with at least ``n`` vertices: points added at
    the middle of its longest segment, one at a time, so the surface is
    the same. A polyline with more than ``n`` keeps its own."""
    out = [(float(x), float(z)) for x, z in pts]
    while len(out) < n:
        k = max(range(len(out) - 1),
                key=lambda i: math.hypot(out[i + 1][0] - out[i][0],
                                         out[i + 1][1] - out[i][1]))
        (xa, za), (xb, zb) = out[k], out[k + 1]
        out.insert(k + 1, (0.5 * (xa + xb), 0.5 * (za + zb)))
    return out


def circle_start(geom, crit, n: int = N_VERTICES - 1
                 ) -> Optional[List[Tuple[float, float]]]:
    """A circle (a result with xc, yc, radius) as the polyline of its
    slice-base points on ``n`` slices (``N_VERTICES - 1``: the start the
    refinement shifts; the search's ``n_slices``: the circle as it stands,
    step 6's circle-check polyline), or None where it cannot be sliced
    so."""
    from slope_stability.search import circle_as_polyline
    try:
        return list(circle_as_polyline(geom, crit, n).points)
    except (ValueError, ZeroDivisionError, RuntimeError, AttributeError):
        return None


class _Circle:
    """Enough of a result for `circle_as_polyline` (centre and radius),
    with the circle's own F where the circular search gave it."""
    is_circular = True

    def __init__(self, xc, yc, radius, x_entry=None, x_exit=None, arc=None,
                 FOS=None):
        self.xc, self.yc, self.radius = xc, yc, radius
        self.x_entry, self.x_exit = x_entry, x_exit
        self.arc = arc
        self.slip_points = None
        self.FOS = FOS


def circle_starts_of(circular_result) -> List[Tuple[str, Any]]:
    """The locally critical circles of a circular search's result, as
    (label, circle) pairs, each circle with its own F (``FOS``): the
    critical circle, its separate grid minima (``local_minima``) and the
    critical circles of the toe and above-toe families where they ran
    (``family_surfaces``), lowest first, a circle listed once.

    A grid minimum the refinement walk started from is the circle the walk
    ended on (``refined_centre``), with its F; one it did not start from is
    its grid node's circle, with the F the record gives. A None F (a walk
    that found no solution, or a grid valued by the Bishop screen) is no
    value: the circle is kept, sorted last, with F None (the fix round's
    D1: a None F crashed the sort under a time budget)."""
    if circular_result is None:
        return []
    out: List[Tuple[float, str, Any]] = []
    seen = set()

    def add(f, label, xc, yc, r, circle=None):
        if r is None or r <= 0:
            return
        key = (round(xc, 2), round(yc, 2), round(r, 2))
        if key in seen:
            return
        seen.add(key)
        c = circle if circle is not None else _Circle(xc, yc, r)
        c.FOS = None if f is None else float(f)
        out.append((math.inf if f is None else float(f), label, c))

    crit = getattr(circular_result, "critical", None)
    if crit is not None and (getattr(crit, "radius", 0) or 0) > 0:
        add(crit.FOS, "critical circle", crit.xc, crit.yc, crit.radius,
            _Circle(crit.xc, crit.yc, crit.radius, crit.x_entry, crit.x_exit,
                    getattr(crit, "arc", None)))
    for k, m in enumerate(getattr(circular_result, "local_minima", []) or []):
        rc = m.get("refined_centre")
        if m.get("refined") and rc and m.get("FOS") is not None:
            add(m["FOS"], f"grid minimum {k + 1}", rc["xc"], rc["yc"],
                rc["radius"])
        elif m.get("radius"):
            f = m.get("FOS")
            if f is None and not m.get("values_by"):
                f = m.get("FOS_grid")       # the node's own F, by the method
            add(f, f"grid minimum {k + 1}", m["xc"], m["yc"], m["radius"])
    fams = getattr(circular_result, "family_surfaces", {}) or {}
    for fam in ("toe_circle", "circle_above_toe"):
        s = fams.get(fam)
        if s and s.get("radius"):
            add(s.get("FOS"), fam.replace("_", " "), s["xc"], s["yc"],
                s["radius"])
    out.sort(key=lambda t: t[0])
    return [(label, c) for _f, label, c in out]


def _wedge_leg(geom, x0: float, z0: float, angle_deg: float, direction: int
               ) -> Optional[Tuple[float, float]]:
    """Where a straight leg from (x0, z0), rising at ``angle_deg`` above the
    horizontal toward -x (direction -1) or +x (+1), first meets the ground;
    None where it does not within the profile."""
    x_min, x_max = geom.x_range
    t = math.tan(math.radians(angle_deg))
    reach = (x0 - x_min) if direction < 0 else (x_max - x0)
    if reach <= 0:
        return None

    def above(s):       # leg above the ground s along x
        x = x0 + direction * s
        return z0 + t * s - geom.ground_elevation_at(x)

    n = 200
    prev = 0.0
    if above(0.0) >= 0:
        return None
    for i in range(1, n + 1):
        s = reach * i / n
        if above(s) >= 0:
            lo, hi = prev, s
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if above(mid) < 0:
                    lo = mid
                else:
                    hi = mid
            x = x0 + direction * hi
            return (x, geom.ground_elevation_at(x))
        prev = s
    return None


def _phi_above(geom, x: float, z_top: float) -> float:
    """phi' of the material just above ``z_top`` at x (the material a
    wedge leg first enters from the layer below it)."""
    lay = geom.layer_at_point(x, z_top + 1e-3)
    return 0.0 if lay is None else float(lay.shear_strength_params[1])


def _wedge_angle(phi: float, f: float, active: bool) -> float:
    """EM C-1g(2), pp. C-8 to C-10: 45 + phi'_D/2 (active) or 45 -
    phi'_D/2 (passive), tan phi'_D = tan phi'/F; 45 where phi' = 0."""
    phi_d = math.degrees(math.atan(math.tan(math.radians(phi))
                                   / max(f, 1e-6)))
    return 45.0 + phi_d / 2.0 if active else 45.0 - phi_d / 2.0


def toe_at_low_x(geom, windows) -> bool:
    """True when the toe side of the search is its low-x end (the entry
    window's ground lower than the exit window's)."""
    (e0, e1), (x0, x1) = windows
    return (geom.ground_elevation_at(0.5 * (e0 + e1))
            < geom.ground_elevation_at(0.5 * (x0 + x1)))


def _wedge_mid(geom, layer, x: float, z_floor: float
               ) -> Optional[Tuple[float, float]]:
    """(z of the mid-line, z of the layer's top) of ``layer`` at x: the
    mid-line between its bottom and its top cut at the ground, as
    `thinnest_stratum` reads the band (the fix round's D6); None where it
    holds no ground there within the search's depth."""
    from slope_stability.search import _layer_band_at
    band = _layer_band_at(geom, layer, x)
    if band is None:
        return None
    zg = geom.ground_elevation_at(x)
    top = min(band[1], zg)
    if top - band[0] <= 1e-6:
        return None
    z_mid = 0.5 * (band[0] + top)
    if not z_floor - 1e-9 <= z_mid < zg - 1e-6:
        return None
    return z_mid, band[1]


def wedge_starts(geom, windows, score: Callable, *,
                 windows_given: Tuple[bool, bool] = (True, True),
                 n_positions: int = WEDGE_POSITIONS,
                 z_floor: Optional[float] = None,
                 d_final: Optional[float] = None,
                 tol: float = 1e-4,
                 time_up: Callable[[], bool] = lambda: False,
                 on_stop: Optional[Callable] = None
                 ) -> List[Dict[str, Any]]:
    """A wedge along each layer within the search's depth (EM C-1f(2),
    p. C-6; C-1g(2), pp. C-8 to C-10; FHWA pp. 6-43 to 6-44).

    The central block's base runs on the layer's mid-line at each x, the
    mid-line between its bottom and its top cut at the ground (`_wedge_mid`;
    the fix round's D6); its two ends take ``n_positions`` positions each
    over the windows' span. The wedge at the toe side rises to the ground
    at 45 - phi'_D/2 (passive), the other at 45 + phi'_D/2 (active), with
    the phi' of the material just above the layer at the base end, on the
    trial's F (`WEDGE_F_PASSES`). Then the best wedge's two ends are
    halved, one end at a time, the inclinations fixed, from half the
    positions' spacing down to ``d_final`` (the refinement's final
    increment; ruling 10), a shift taken where F falls by more than
    ``tol``. A wedge whose end on the ground falls outside a window the
    caller gave is not taken (``windows_given``; call 5: automatic windows
    bound only the samplers). ``score(points)`` gives F (or the search's
    not-scored mark, >= 900).

    The stage is a stage of the time budget (``time_up``): where it stops,
    ``on_stop(done, of, t0, left)`` is told the low-end positions scored
    of all and what was left (the layers and positions not built), never
    a silent stop.

    Returns one record per layer: {layer, FOS, points, n_scored} and, for
    a wedge, its ``ends`` (the central block's two ends), its
    ``inclinations_deg`` (low-x and high-x legs) and ``end_levels`` (the
    halving's increments); FOS None and points None with ``why`` where no
    wedge along the layer was scored."""
    (e_lo, e_hi), (x_lo, x_hi) = windows
    lo, hi = min(e_lo, x_lo), max(e_hi, x_hi)
    if z_floor is None:
        z_floor = min(L.bottom_elevation for L in geom.soil_layers)
    low_is_toe = toe_at_low_x(geom, windows)
    spacing = (hi - lo) / (n_positions - 1)
    xs = [lo + spacing * i for i in range(n_positions)]
    at = {x: i for i, x in enumerate(xs)}
    t0 = time.perf_counter()
    plan = []
    for layer in geom.soil_layers:
        mids = {}
        for x in xs:
            m = _wedge_mid(geom, layer, x, z_floor)
            if m is not None:
                mids[x] = m
        plan.append((layer, mids))
    rows_of = sum(len(m) for _layer, m in plan)
    rows_done = 0
    left: List[str] = []            # what the time budget left
    out: List[Dict[str, Any]] = []

    def outside(p0, p1) -> bool:
        return ((windows_given[0]
                 and not e_lo - 1e-9 <= p0[0] <= e_hi + 1e-9)
                or (windows_given[1]
                    and not x_lo - 1e-9 <= p1[0] <= x_hi + 1e-9))

    for layer, mids in plan:
        rec: Dict[str, Any] = {"layer": layer.name, "FOS": None,
                               "points": None, "n_scored": 0}
        out.append(rec)
        if not mids:
            rec["why"] = ("the layer holds no ground below the ground "
                          "surface between the windows within the search's "
                          "depth")
            continue
        keys = sorted(mids)
        if left:
            rec["why"] = ("not built: the time budget stopped the wedge "
                          "stage before this layer (GM-123)")
            left.append(f"the wedges along {layer.name} (all {len(keys)} "
                        f"low-end positions)")
            continue
        n = {"pairs": 0, "no_leg": 0, "outside": 0, "unscored": 0}
        best = None
        cut_here = False

        def build(xa, xb, za, zb, a_low, a_high, keys=keys, mids=mids):
            p0 = _wedge_leg(geom, xa, za, a_low, -1)
            p1 = _wedge_leg(geom, xb, zb, a_high, +1)
            if p0 is None or p1 is None:
                return None, "no_leg"
            if outside(p0, p1):
                return None, "outside"
            return ([p0, (xa, za)]
                    + [(x, mids[x][0]) for x in keys
                       if xa + 1e-9 < x < xb - 1e-9]
                    + [(xb, zb), p1]), None

        for i, xa in enumerate(keys):
            if time_up():
                cut_here = True
                left.append(f"the wedges along {layer.name} with the low end "
                            f"from x = {xa:.2f} on ({len(keys) - i} of "
                            f"{len(keys)} low-end positions)")
                break
            rows_done += 1
            for xb in keys[i + 1:]:
                if (at[xb] - at[xa] + 1
                        != sum(1 for x in keys if xa <= x <= xb)):
                    continue        # the layer leaves the ground between
                n["pairs"] += 1
                f_trial = 1.0
                for _ in range(WEDGE_F_PASSES):
                    a_low = _wedge_angle(_phi_above(geom, xa, mids[xa][1]),
                                         f_trial, active=not low_is_toe)
                    a_high = _wedge_angle(_phi_above(geom, xb, mids[xb][1]),
                                          f_trial, active=low_is_toe)
                    pts, why = build(xa, xb, mids[xa][0], mids[xb][0],
                                     a_low, a_high)
                    if pts is None:
                        n[why] += 1
                        break
                    f = score(pts)
                    rec["n_scored"] += 1
                    if f >= 900:
                        n["unscored"] += 1
                        break
                    if best is None or f < best[0]:
                        best = (f, xa, xb, a_low, a_high, pts)
                    if abs(f - f_trial) < 1e-3:
                        break
                    f_trial = f
        levels: List[float] = []
        if best is not None and d_final is not None and not left:
            # ruling 10: the best wedge's two ends halved, one at a time,
            # the inclinations fixed, down to the final increment
            f_b, xa, xb, a_low, a_high, pts_b = best
            d = max(0.5 * spacing, d_final)
            while True:
                levels.append(round(d, 6))
                moved = True
                while moved and not cut_here:
                    moved = False
                    for end in (0, 1):
                        if time_up():
                            cut_here = True
                            break
                        trial = []
                        for s in (+1, -1):
                            na = xa + s * d if end == 0 else xa
                            nb = xb + s * d if end == 1 else xb
                            if not (lo - 1e-9 <= na and na < nb - 1e-9
                                    and nb <= hi + 1e-9):
                                continue
                            ma = _wedge_mid(geom, layer, na, z_floor)
                            mb = _wedge_mid(geom, layer, nb, z_floor)
                            if ma is None or mb is None or any(
                                    x not in mids for x in xs if na < x < nb):
                                continue
                            pts, _why = build(na, nb, ma[0], mb[0], a_low,
                                              a_high)
                            if pts is None:
                                continue
                            f = score(pts)
                            rec["n_scored"] += 1
                            if f < 900:
                                trial.append((f, na, nb, pts))
                        if trial:
                            f, na, nb, pts = min(trial, key=lambda q: q[0])
                            if f < f_b - tol:
                                f_b, xa, xb, pts_b = f, na, nb, pts
                                moved = True
                if cut_here or d <= d_final + 1e-12:
                    break
                d = max(0.5 * d, d_final)
            if cut_here:
                left.append(f"the halving of the ends of the wedge along "
                            f"{layer.name} (stopped at the increment "
                            f"{d:.3g} m)")
            best = (f_b, xa, xb, a_low, a_high, pts_b)
        if best is not None:
            f_b, xa, xb, a_low, a_high, pts_b = best
            rec.update(FOS=f_b, points=pts_b,
                       ends=(round(xa, 3), round(xb, 3)),
                       inclinations_deg=(round(a_low, 2), round(a_high, 2)))
            if levels:
                rec["end_levels"] = levels
            if cut_here:
                rec["note"] = ("the time budget stopped the wedge stage in "
                               "this layer: its wedge is the lowest of the "
                               "positions scored (GM-123)")
        else:
            if n["pairs"] == 0 and not cut_here:
                why = ("no two end positions hold the layer's ground "
                       "continuously between them")
            else:
                why = (f"no wedge along it was scored: of {n['pairs']} "
                       f"pairs of end positions, {n['no_leg']} gave a wedge "
                       f"leg that does not reach the ground within the "
                       f"profile, {n['outside']} ended on the ground outside "
                       f"a window you gave, and {n['unscored']} wedges were "
                       f"not scored (the method has no solution on them, or "
                       f"a screen refused them)")
            if cut_here:
                why += "; the time budget stopped the wedge stage in this layer (GM-123)"
            rec["why"] = why
    if left and on_stop is not None:
        on_stop(rows_done, rows_of, t0, left)
    return out


# ---------------------------------------------------------------------------
# The refinement (EM C-1g(3), p. C-10, Fig. C-8)
# ---------------------------------------------------------------------------

def refine_polyline(geom, pts, score: Callable, *, d_start: float,
                    d_final: float, windows, windows_given: Tuple[bool, bool],
                    tol: float = 1e-4,
                    time_up: Callable[[], bool] = lambda: False,
                    f_start: Optional[float] = None) -> Dict[str, Any]:
    """Shift the points of the polyline ``pts`` one at a time, all others
    held (EM C-1g(3), p. C-10, Fig. C-8): interior points up and down, the
    end points along the ground. The surface moves to the better of a
    point's two shifts where it lowers F by more than ``tol``; passes over
    the points repeat until none does, then the increment is halved, from
    ``d_start`` down to ``d_final`` (EM 4-2e(2)(b), p. 4-7), the last level
    at ``d_final``.

    A shift is not taken where it would:
    * carry an end across a crest from its start, or from one face group
      onto another (EM 4-2e(1)(a), p. 4-3; `end_allowed`);
    * make the toe-end segment steeper than EM Eq. C-32's passive
      inclination with the exit material's phi', where the start's own
      exit is not steeper still, then steeper than the start's exit (EM
      4-2e(2)(a), p. 4-7; call 4);
    * carry an end outside a window the caller gave (call 5: automatic
      windows bound only the samplers);
    * take an interior point above the ground or below the bottom of the
      deepest layer, or put two points out of order in x.

    Returns {FOS, points, F_start, n_scored, levels, stopped (the time
    budget ended it), toe_limit_deg, exit_phi}."""
    pts = [(float(x), float(z)) for x, z in pts]
    spans = _face_spans(geom)
    toe = _toe_index(geom, pts)
    c32, phi = passive_exit_limit(geom, pts, toe)
    limit = max(c32, exit_angle(pts, toe))
    x_min, x_max = geom.x_range
    z_base = min(L.bottom_elevation for L in geom.soil_layers)
    x_starts = (pts[0][0], pts[-1][0])
    (e_lo, e_hi), (x_lo, x_hi) = windows
    n_scored = 0
    if f_start is None:
        f_start = score(pts)
        n_scored += 1
    best = f_start
    stopped = False

    def ok(cand, i) -> bool:
        if any(cand[k + 1][0] - cand[k][0] <= 1e-6
               for k in range(len(cand) - 1)):
            return False
        if i in (0, len(cand) - 1):
            j = 0 if i == 0 else 1
            x_new = cand[i][0]
            if not (x_min <= x_new <= x_max):
                return False
            if not end_allowed(spans, x_starts[j], x_new):
                return False
            if windows_given[j]:
                lo_w, hi_w = (e_lo, e_hi) if j == 0 else (x_lo, x_hi)
                if not (lo_w - 1e-9 <= x_new <= hi_w + 1e-9):
                    return False
        else:
            x, z = cand[i]
            if z >= geom.ground_elevation_at(x) - 1e-6 or z < z_base:
                return False
        if exit_angle(cand, toe) > limit + 1e-9:
            return False
        return True

    levels = []
    d = max(d_start, d_final)
    while True:
        levels.append(round(d, 6))
        moved = True
        while moved and not stopped:
            moved = False
            for i in range(len(pts)):
                if time_up():
                    stopped = True
                    break
                trial = []
                for s in (+1, -1):
                    cand = list(pts)
                    x, z = cand[i]
                    if i in (0, len(pts) - 1):
                        xn = x + s * d
                        cand[i] = (xn, geom.ground_elevation_at(
                            min(max(xn, x_min), x_max)))
                    else:
                        cand[i] = (x, z + s * d)
                    if not ok(cand, i):
                        continue
                    f = score(cand)
                    n_scored += 1
                    if f < 900:
                        trial.append((f, cand))
                if trial:
                    f, cand = min(trial, key=lambda t: t[0])
                    if f < best - tol:
                        best, pts, moved = f, cand, True
        if stopped or d <= d_final + 1e-12:
            break
        d = max(0.5 * d, d_final)
    return {"FOS": best, "points": pts, "F_start": f_start,
            "n_scored": n_scored, "levels": levels, "stopped": stopped,
            "toe_limit_deg": round(limit, 3),
            "eq_c32_deg": round(c32, 3), "exit_phi": phi}


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

def refinement_stage(geom, *, score: Callable, flags_of: Callable,
                     own_best: Optional[Sequence], windows,
                     windows_given: Tuple[bool, bool],
                     n_slices: int = 30,
                     circle_starts: Optional[List[Tuple[str, Any]]] = None,
                     circle_note: Optional[str] = None,
                     seed_surface: Optional[Sequence] = None,
                     wedges: bool = True, refine: bool = True,
                     final_fraction: float = FINAL_FRACTION,
                     tol: float = 1e-4,
                     time_up: Callable[[], bool] = lambda: False,
                     record_stop: Optional[Callable] = None,
                     span_of: Optional[Callable] = None,
                     own_label: str = "the search's own best surface",
                     wedge_only: bool = False) -> Dict[str, Any]:
    """Every start scored as it stands and refined separately, the lowest
    result first (EM 4-2e(1)(b), pp. 4-3 and 4-6; 4-2e(2), p. 4-7; C-1g(4),
    p. C-11; FHWA p. 6-22).

    The starts: the search's own best surface (``own_best``, its points),
    each locally critical circle (``circle_starts``, (label, circle) from
    `circle_starts_of`), a wedge along each layer (`wedge_starts`, when
    ``wedges``), and the caller's surface (``seed_surface``).

    Every start is first scored as it stands (the fix round's ruling 9(b);
    a circle as the polyline of its slice-base points on the search's
    ``n_slices``, step 6's circle-check polyline), then refined while the
    time budget lasts (a circle from its polyline on ``N_VERTICES - 1``
    slices). A start's minimum is the lower of the two, labelled which
    (``minimum``: "as it stands" or "refined"); a start the budget left is
    listed not refined with its value as it stands. With ``refine`` False
    nothing is shifted: the starts are scored as they are (``wedge_only``:
    the wedge stage alone, Slide2's search without optimisation).

    ``score(points[, start_span])`` gives F; with ``span_of`` (the span of
    a polyline as the span screen measures it) each start's own span is
    passed as ``start_span``, and the scorer takes the lesser of the
    module's span floor and it (ruling 8). ``flags_of(points)`` gives the
    flags of the printed checks on the surface. ``record_stop(stage, done,
    of, unit, t0, covered, not_covered)`` records a stage the budget
    stopped (the wedge stage, the refinement).

    Returns the record the search puts on its result as ``refinement``:
    {final_increment (stratum, thickness, fraction, increment and the rule
    that set it), start_increment, n_vertices, wedges (one per layer),
    local_minima (one per start, lowest first: start, F_start (as it
    stands), F_circle (a circle start's own F), F_refined, FOS, minimum,
    points, flags, refined, levels, n_scored), circle_starts (the note
    where there are none), unrefined (starts the time budget left), best
    (points, FOS, start), n_scored}."""
    t0 = time.perf_counter()
    (e_lo, e_hi), (x_lo, x_hi) = windows
    lo, hi = min(e_lo, x_lo), max(e_hi, x_hi)
    z_floor = min(L.bottom_elevation for L in geom.soil_layers)
    stratum = thinnest_stratum(geom, lo, hi, z_floor)
    h = slope_height(geom, lo, hi)
    if not 0.0 < final_fraction <= FINAL_FRACTION_RANGE[1]:
        raise ValueError(
            f"refine_fraction must be above 0 and at most "
            f"{FINAL_FRACTION_RANGE[1]} (EM 1110-2-1902 4-2e(2)(b), p. 4-7: "
            f"no more than 10 to 25 percent of the thinnest stratum), got "
            f"{final_fraction}")
    # ruling 9(a): the starting increment is never raised to meet the final
    # one; the final increment is the lesser of the fraction of the
    # thinnest stratum and a quarter of the starting increment
    d_start = max(START_FRACTION_H * h, 1e-3)
    d_stratum = (final_fraction * stratum["thickness"] if stratum
                 else final_fraction * max(h, 1.0))
    d_quarter = FINAL_OF_START * d_start
    d_final = max(min(d_stratum, d_quarter), 1e-4)
    rec: Dict[str, Any] = {
        "final_increment": {
            "stratum": None if stratum is None else stratum["layer"],
            "thickness": None if stratum is None else stratum["thickness"],
            "at_x": None if stratum is None else stratum["x"],
            "fraction": final_fraction,
            "increment": round(d_final, 4),
            "fraction_of_stratum": round(d_stratum, 4),
            "quarter_of_start": round(d_quarter, 4),
            "governed_by": ("the fraction of the thinnest stratum"
                            if d_stratum <= d_quarter else
                            "a quarter of the starting increment"),
            "basis": ("EM 1110-2-1902 4-2e(2)(b), p. 4-7: final shift "
                      "increments no more than 10 to 25 percent of the "
                      "thickness of the thinnest stratum the surface may "
                      "pass through; it prints a maximum only, so the "
                      "lesser of that fraction and a quarter of the "
                      "starting increment is the module's call")},
        "start_increment": round(d_start, 4),
        "n_vertices": N_VERTICES,
        "refined": bool(refine),
        "wedges": [], "local_minima": [], "unrefined": [],
    }
    n_scored = 0
    starts: List[Dict[str, Any]] = []
    if own_best is not None and not wedge_only:
        starts.append({"label": own_label,
                       "stands": [(float(x), float(z)) for x, z in own_best]})
    if circle_starts:
        if not wedge_only:
            for label, c in circle_starts:
                starts.append({"label": label, "circle": True,
                               "F_circle": getattr(c, "FOS", None),
                               "stands": circle_start(geom, c, n_slices),
                               "reduced": circle_start(geom, c)})
    elif circle_note:
        rec["circle_starts"] = circle_note
    if wedges:
        def on_stop(done, of, tw, left):
            rec["wedge_stage_stopped"] = list(left)
            if record_stop is not None:
                record_stop("wedge stage", done, of, "low-end positions", tw,
                            "the layers in order, each over its grid of "
                            "end positions", "; ".join(left))
        ws = wedge_starts(geom, windows, score, z_floor=z_floor,
                          windows_given=windows_given, d_final=d_final,
                          tol=tol, time_up=time_up, on_stop=on_stop)
        for w in ws:
            n_scored += w["n_scored"]
            entry = {
                "layer": w["layer"], "n_scored": w["n_scored"],
                "FOS": None if w["FOS"] is None else round(w["FOS"], 4),
                "points": None if w["points"] is None else
                [(round(x, 3), round(z, 3)) for x, z in w["points"]]}
            for k in ("ends", "inclinations_deg", "end_levels", "why",
                      "note"):
                if k in w:
                    entry[k] = w[k]
            rec["wedges"].append(entry)
            if w["points"] is not None:
                starts.append({"label": f"wedge along {w['layer']}",
                               "stands": w["points"]})
    if seed_surface is not None and not wedge_only:
        starts.append({"label": "the caller's surface (seed_surface)",
                       "stands": [(float(x), float(z))
                                  for x, z in seed_surface]})

    # every start scored as it stands first (ruling 9(b); D3), each with
    # its own span floor (ruling 8)
    for s in starts:
        base = s.get("reduced") if s.get("circle") else s.get("stands")
        s["from"] = (None if base is None else
                     with_vertices(base) if refine else list(base))
        spans = []
        if span_of is not None:
            for p in (s.get("stands"), s["from"] if refine else None):
                v = None if p is None else span_of(p)
                if v is not None:
                    spans.append(v)
        span = min(spans) if spans else None
        s["score"] = (score if span is None else
                      (lambda p, _span=span: score(p, start_span=_span)))
        f = math.inf
        if s.get("stands") is not None:
            f = s["score"](s["stands"])
            n_scored += 1
        s["f_stands"] = f if f < 900 else None

    minima = []
    for s in starts:
        r = None
        why = None
        if refine and s["from"] is not None:
            if time_up():
                why = ("not refined: the time budget (GM-123); its value is "
                       "the start's as it stands")
                rec["unrefined"].append(s["label"])
            else:
                f1 = s["score"](s["from"])
                n_scored += 1
                if f1 < 900:
                    r = refine_polyline(geom, s["from"], s["score"],
                                        d_start=d_start, d_final=d_final,
                                        windows=windows,
                                        windows_given=windows_given, tol=tol,
                                        time_up=time_up, f_start=f1)
                    n_scored += r["n_scored"]
                    if r["stopped"]:
                        rec["unrefined"].append(
                            s["label"] + " (refinement cut short)")
                else:
                    why = ("not refined: the refinement's start is not "
                           "scored (the method has no solution on it, or a "
                           "screen refused it)")
        elif refine:
            why = "not refined: the start could not be built as a polyline"
        m: Dict[str, Any] = {
            "start": s["label"],
            "F_start": (None if s["f_stands"] is None
                        else round(s["f_stands"], 4))}
        if s.get("circle"):
            m["F_circle"] = (None if s["F_circle"] is None
                             else round(s["F_circle"], 4))
        cands = []
        if s["f_stands"] is not None:
            cands.append((s["f_stands"], "as it stands", s["stands"]))
        if r is not None:
            cands.append((r["FOS"], "refined", r["points"]))
        if not cands:
            m.update(FOS=None, points=None, refined=False,
                     note="the start is not scored (the method has no "
                          "solution on it, or a screen refused it)")
            if why:
                m["why_unrefined"] = why
            minima.append(m)
            continue
        f, which, pts = min(cands, key=lambda q: q[0])
        m.update(FOS=f, minimum=which, points=pts, refined=r is not None)
        if r is not None:
            m.update(F_refined=round(r["FOS"], 4), levels=r["levels"],
                     n_scored=r["n_scored"],
                     toe_limit_deg=r["toe_limit_deg"],
                     eq_c32_deg=r["eq_c32_deg"])
            if s.get("circle"):
                m["F_refine_start"] = round(r["F_start"], 4)
        elif why:
            m["why_unrefined"] = why
        minima.append(m)
    if rec["unrefined"] and record_stop is not None:
        record_stop("noncircular refinement",
                    len([m for m in minima if m.get("refined")]),
                    len(starts), "starts", t0,
                    "every start scored as it stands; the starts refined "
                    "first, in order",
                    "starts left unrefined: " + "; ".join(rec["unrefined"]))
    scored = [m for m in minima if m["FOS"] is not None]
    scored.sort(key=lambda m: m["FOS"])
    # the headline from the unrounded value and points (the list below is
    # rounded for the record)
    rec["best_exact"] = (None if not scored else
                         {"FOS": scored[0]["FOS"],
                          "points": list(scored[0]["points"])})
    for m in scored:
        m["flags"] = list(flags_of(m["points"]))
        m["FOS"] = round(m["FOS"], 4)
        m["points"] = [(round(x, 4), round(z, 4)) for x, z in m["points"]]
    rec["local_minima"] = scored + [m for m in minima if m["FOS"] is None]
    rec["best"] = scored[0] if scored else None
    rec["n_scored"] = n_scored
    # (no run time in the record, so a result does not change from run to
    # run; a search cut short has it in search_ranges['time_budget'])
    return rec


def refinement_choices(rec: Optional[Dict[str, Any]], *, refine: bool,
                       fraction: Optional[float]) -> List[Dict[str, Any]]:
    """The ``choices`` entries of a noncircular search's refinement (GM-140
    step 7a): the refinement itself, the final increment, the starting
    increment and vertices, the starts, the wedges, the toe restriction,
    the windows and the shape."""
    from geotech_common.choices import choice, how_given
    out = [choice(
        "noncircular refinement", "on" if refine else "off",
        how_given(refine, True, name="refine"),
        "EM 1110-2-1902 C-1g(3), p. C-10, Fig. C-8 (Celestino and Duncan "
        "1981): hold all points but one, shift that point a small distance "
        "in two directions, move the surface, repeat until F stops falling "
        "and the moves are minimal; C-1g(4), p. C-11, and 4-2e(1)(b), pp. "
        "4-3 and 4-6: independently started searches, compared",
        ["off (refine=False): no shifts; the starts are scored as they are "
         "(the weak-layer search: the wedge stage alone, Slide2's search "
         "without optimisation)" if refine else "on (refine=True)"])]
    if not rec or not rec.get("final_increment"):
        return out
    fi = rec["final_increment"]
    stratum = (f"{fi['fraction']:g} x the thinnest stratum "
               f"({fi['stratum']}, {fi['thickness']} m least vertical "
               f"thickness within the windows): "
               f"{fi.get('fraction_of_stratum')} m"
               if fi.get("stratum") is not None else
               f"{fi['fraction']:g} x the ground's relief (no stratum in "
               f"the windows): {fi.get('fraction_of_stratum')} m")
    out.append(choice(
        "final shift increment of the noncircular refinement",
        f"{fi['increment']} m, the lesser of {stratum} and a quarter of the "
        f"starting increment, {fi.get('quarter_of_start')} m (here "
        f"{fi.get('governed_by')})",
        how_given(fraction, None, name="refine_fraction"), fi["basis"],
        ["refine_fraction up to 0.25, the print's upper end (the default "
         "0.10 is the module's call: accuracy first, and it meets both ends "
         "of the printed range)",
         "exactly the fraction of the thinnest stratum, the starting "
         "increment raised to meet it where the strata are thick (as first "
         "built: one level of shifts on a slope of thick strata)"]))
    out.append(choice(
        "starting increment and vertices of the noncircular refinement",
        f"start {rec['start_increment']} m (0.10 x the ground's relief in "
        f"the windows, never raised to meet the final increment), halved "
        f"to the final increment; {rec['n_vertices']} vertices on the "
        f"surface the refinement shifts",
        "module",
        "not in print: EM C-1g(3) gives the method, not the increments or "
        "the vertex count. With points closer together than the increment "
        "a one-point shift is only a kink, so a refinement of a 30-point "
        "polyline would cost about three times as much and mostly stand "
        "still",
        ["other starting increments and vertex counts",
         "refine a circle's polyline on the search's own slices itself"]))
    out.append(choice(
        "starts of the noncircular refinement",
        "the search's own best surface; each locally critical circle of "
        "the circular search run for circle_check (its critical circle, "
        "its grid minima as the walk left them, and the toe and above-toe "
        "families' critical circles); a wedge along each layer within the "
        "search's depth; the caller's seed_surface. Every start is scored "
        "as it stands before it is refined, and its minimum is the lower "
        "of the two, labelled which (local_minima: minimum); a circle as "
        "it stands is the polyline of its slice-base points on the "
        "search's own slices, and its refinement starts from that polyline "
        "on ten slices",
        "module",
        "EM 4-2e(2), p. 4-7: the initial surface should match the estimated "
        "critical surface; 4-2e(1)(b), pp. 4-3 and 4-6, and C-1g(4), p. "
        "C-11: separate searches; FHWA NHI-06-088 p. 6-22",
        ["circle_check=False: no circle starts",
         "a start's minimum its refined value alone (as first built: a "
         "circle start could end above its own circle)"]))
    out.append(choice(
        "wedge starts",
        f"one per layer within the search's depth; the central block on "
        f"the mid-line between the layer's bottom and its top cut at the "
        f"ground, its ends at {WEDGE_POSITIONS} positions each over the "
        f"windows, then the best wedge's two ends halved, one at a time, "
        f"the inclinations fixed, down to the final increment; wedges at "
        f"45 +/- phi'_D/2 on the trial's F (F = 1 first, then the F each "
        f"pass gave, {WEDGE_F_PASSES} passes at most), phi' of the material "
        f"just above the layer at the base end",
        "module",
        "EM 1110-2-1902 C-1f(2), p. C-6; C-1g(2), pp. C-8 to C-10, Fig. "
        "C-7b; FHWA NHI-06-088 pp. 6-43 to 6-44. Which layers, how many "
        "positions, the halving of the ends and how F is updated are the "
        "module's calls",
        ["only layers weaker than the one above (FHWA p. 6-33), compared "
         "at a normal stress no print gives",
         "an inclination band about C-1g(2)'s values (Fig. C-7b), not "
         "built"]))
    out.append(choice(
        "span screen inside the noncircular refinement",
        "the lesser of the module's span floor (half the span of the "
        "windows you gave, else half the slope's height) and the start's "
        "own span, for every start scored as it stands and every shift of "
        "its refinement; the samplers and the wedge stage keep the "
        "module's floor",
        "module",
        "not in print: the span floor is the module's own screen (D22); a "
        "start shorter than it, such as a short critical circle, is scored "
        "and refined where it lies",
        ["the module's floor alone (as first built: it refused a critical "
         "circle's start shorter than half the slope's height)"]))
    out.append(choice(
        "toe of the noncircular refinement",
        "no shift makes the toe-end segment steeper than EM Eq. C-32's "
        "passive inclination, 45 - phi'/2 with the exit material's phi', "
        "nor, where the start's own exit is steeper, steeper than the "
        "start's exit",
        "module",
        "EM 4-2e(2)(a), p. 4-7, asks for the orientation at the toe to be "
        "restricted and prints no limit for a search; EM C-10b, p. C-34, "
        "Eq. C-32 gives the passive inclination: the limit is the module's "
        "call",
        ["no restriction, leaving the printed checks to flag a steep "
         "exit"]))
    out.append(choice(
        "windows and faces of the noncircular refinement",
        "automatic windows bound only the samplers; the shifts and the "
        "wedge ends are free along the ground on the start's side; windows "
        "the caller gives bound everything; no end crosses a crest from "
        "its start or moves onto another face group",
        "module",
        "EM 4-2e(1)(a), p. 4-3: the search must not jump from one face to "
        "another. The windows: confirmed by the owner, 2026-11-15 (OWN-143 "
        "part 42, call 5)",
        ["automatic windows also bound the shifts"]))
    out.append(choice(
        "shape of a refined noncircular surface",
        "no shape rule: the printed acceptability checks judge the surface",
        "module",
        "no shape rule found in print (EM p. 4-7 prints the toe and "
        "increment rules only)",
        ["a concave-up shape as the module's own rule"]))
    return out


def grid_minima_record(lattice, nx: int, ny: int, refined_at, *,
                       refine: bool, refine_ran: bool, screened: bool,
                       n_starts: int, within: float) -> List[Dict[str, Any]]:
    """The separate minima of a centre grid (``SearchResult.local_minima``;
    EM 1110-2-1902 4-2d(2), p. 4-2, and 4-2e(1)(b), pp. 4-3 and 4-6),
    lowest first: each minimum's centre, radius and grid FOS (the Bishop
    screen's where the grid was screened), and whether the refinement walk
    started from it; one it did not start from says why (the run-time caps
    of GM-123: at most ``n_starts`` starts, those within ``within`` of the
    lowest grid FOS, or the time budget). A record only: no value of the
    circular search changes."""
    from slope_stability.search import _grid_minima, _FOS_MAX
    minima = _grid_minima([[p[0] for p in col] for col in lattice], nx, ny)
    if not minima:
        return []
    grid_best = minima[0][0]
    out = []
    for k, (f0, ix, iy) in enumerate(minima):
        _f, x0, y0, r0 = lattice[ix][iy]
        m: Dict[str, Any] = {"start": f"grid minimum {k + 1}",
                             "FOS_grid": round(f0, 4),
                             "xc": round(x0, 3), "yc": round(y0, 3),
                             "radius": round(r0, 3), "node": (ix, iy)}
        if screened:
            m["values_by"] = "the Bishop screen"
        if (ix, iy) in refined_at:
            f, xc, yc, r = refined_at[(ix, iy)]
            m.update(refined=True,
                     FOS=None if f >= _FOS_MAX else round(f, 4),
                     refined_centre={"xc": round(xc, 3), "yc": round(yc, 3),
                                     "radius": round(r, 3)})
        else:
            m["refined"] = False
            m["FOS"] = None if screened else round(f0, 4)
            if not refine:
                m["why_unrefined"] = "refine=False"
            elif not refine_ran:
                m["why_unrefined"] = "the time budget (GM-123)"
            elif k >= n_starts:
                m["why_unrefined"] = (
                    f"the run-time cap: at most {n_starts} grid minima are "
                    f"refined (_REFINE_STARTS; GM-123)")
            elif f0 > (1.0 + within) * grid_best:
                m["why_unrefined"] = (
                    f"the run-time cap: only minima within "
                    f"{within:.0%} of the lowest grid FOS are refined "
                    f"(_REFINE_WITHIN; GM-123)")
            else:
                m["why_unrefined"] = "the time budget (GM-123)"
        out.append(m)
    return out
