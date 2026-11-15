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
#: increment.
START_FRACTION_H = 0.10
#: Vertices of a refined surface (a call: no print gives a count). A start
#: with fewer gets points added on its longest segments, which leaves the
#: surface as it is; a circle start is the polyline of its slice-base
#: points on this many minus one slices.
N_VERTICES = 11
#: Positions of each end of a wedge's central block over the windows' span
#: (call 3's count; no print gives one; FHWA p. 6-44 asks for several).
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


def circle_start(geom, crit) -> Optional[List[Tuple[float, float]]]:
    """A circle (a result with xc, yc, radius) as the polyline of its
    slice-base points on ``N_VERTICES - 1`` slices, or None where it cannot
    be sliced so."""
    from slope_stability.search import circle_as_polyline
    try:
        return list(circle_as_polyline(geom, crit, N_VERTICES - 1).points)
    except (ValueError, ZeroDivisionError, RuntimeError, AttributeError):
        return None


class _Circle:
    """Enough of a result for `circle_as_polyline` (centre and radius)."""
    is_circular = True

    def __init__(self, xc, yc, radius, x_entry=None, x_exit=None, arc=None):
        self.xc, self.yc, self.radius = xc, yc, radius
        self.x_entry, self.x_exit = x_entry, x_exit
        self.arc = arc
        self.slip_points = None


def circle_starts_of(circular_result) -> List[Tuple[str, Any]]:
    """The locally critical circles of a circular search's result, as
    (label, circle) pairs: its separate grid minima (``local_minima``) and
    the critical circles of the toe and above-toe families where they ran
    (``family_surfaces``), lowest first, a circle listed once."""
    if circular_result is None:
        return []
    out: List[Tuple[float, str, Any]] = []
    seen = set()

    def add(f, label, xc, yc, r):
        key = (round(xc, 2), round(yc, 2), round(r, 2))
        if key in seen or r is None or r <= 0:
            return
        seen.add(key)
        out.append((f, label, _Circle(xc, yc, r)))

    crit = getattr(circular_result, "critical", None)
    if crit is not None and getattr(crit, "radius", 0) > 0:
        c = _Circle(crit.xc, crit.yc, crit.radius, crit.x_entry, crit.x_exit,
                    getattr(crit, "arc", None))
        key = (round(crit.xc, 2), round(crit.yc, 2), round(crit.radius, 2))
        seen.add(key)
        out.append((crit.FOS, "critical circle", c))
    for k, m in enumerate(getattr(circular_result, "local_minima", []) or []):
        if m.get("radius"):
            add(m.get("FOS", math.inf), f"grid minimum {k + 1}", m["xc"],
                m["yc"], m["radius"])
    fams = getattr(circular_result, "family_surfaces", {}) or {}
    for fam in ("toe_circle", "circle_above_toe"):
        s = fams.get(fam)
        if s and s.get("radius"):
            add(s.get("FOS", math.inf), fam.replace("_", " "), s["xc"],
                s["yc"], s["radius"])
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


def wedge_starts(geom, windows, score: Callable, *,
                 windows_given: Tuple[bool, bool] = (True, True),
                 n_positions: int = WEDGE_POSITIONS,
                 z_floor: Optional[float] = None,
                 time_up: Callable[[], bool] = lambda: False
                 ) -> List[Dict[str, Any]]:
    """A wedge along each layer within the search's depth (EM C-1f(2),
    p. C-6; C-1g(2), pp. C-8 to C-10; FHWA pp. 6-43 to 6-44).

    The central block's base runs on the layer's mid-line where the layer
    is at each x (`search._layer_band_at`); its two ends take
    ``n_positions`` positions each over the windows' span. The wedge at the
    toe side rises to the ground at 45 - phi'_D/2 (passive), the other at
    45 + phi'_D/2 (active), with the phi' of the material just above the
    base end, on the trial's F (`WEDGE_F_PASSES`). A wedge whose end on
    the ground falls outside a window the caller gave is not taken
    (``windows_given``; call 5: automatic windows bound only the
    samplers). ``score(points)``
    gives F (or the search's not-scored mark, >= 900).

    Returns one record per layer that holds ground in the span:
    {layer, FOS, points, n_scored} (FOS None and points None where no wedge
    along it was scored)."""
    from slope_stability.search import _layer_band_at
    (e_lo, e_hi), (x_lo, x_hi) = windows
    lo, hi = min(e_lo, x_lo), max(e_hi, x_hi)
    if z_floor is None:
        z_floor = min(L.bottom_elevation for L in geom.soil_layers)
    low_is_toe = toe_at_low_x(geom, windows)
    xs = [lo + (hi - lo) * i / (n_positions - 1) for i in range(n_positions)]
    out = []
    for layer in geom.soil_layers:
        mids = {}
        for x in xs:
            band = _layer_band_at(geom, layer, x)
            if band is None:
                continue
            z_mid = 0.5 * (band[0] + band[1])
            if z_floor - 1e-9 <= z_mid < geom.ground_elevation_at(x) - 1e-6:
                mids[x] = (z_mid, band[1])
        if not mids:
            continue
        rec = {"layer": layer.name, "FOS": None, "points": None,
               "n_scored": 0}
        keys = sorted(mids)
        for i, xa in enumerate(keys):
            if time_up():
                break
            for xb in keys[i + 1:]:
                # the central block on the mid-line at every grid x between
                inner = [x for x in keys if xa <= x <= xb]
                if len(inner) != xs.index(xb) - xs.index(xa) + 1:
                    continue        # the layer leaves the ground between
                f_trial = 1.0
                for _ in range(WEDGE_F_PASSES):
                    a_low = _wedge_angle(_phi_above(geom, xa, mids[xa][1]),
                                         f_trial, active=not low_is_toe)
                    a_high = _wedge_angle(_phi_above(geom, xb, mids[xb][1]),
                                          f_trial, active=low_is_toe)
                    p0 = _wedge_leg(geom, xa, mids[xa][0], a_low, -1)
                    p1 = _wedge_leg(geom, xb, mids[xb][0], a_high, +1)
                    if p0 is None or p1 is None:
                        break
                    if ((windows_given[0]
                         and not e_lo - 1e-9 <= p0[0] <= e_hi + 1e-9)
                            or (windows_given[1] and
                                not x_lo - 1e-9 <= p1[0] <= x_hi + 1e-9)):
                        break
                    pts = ([p0] + [(x, mids[x][0]) for x in inner] + [p1])
                    f = score(pts)
                    rec["n_scored"] += 1
                    if f >= 900:
                        break
                    if rec["FOS"] is None or f < rec["FOS"]:
                        rec["FOS"], rec["points"] = f, pts
                    if abs(f - f_trial) < 1e-3:
                        break
                    f_trial = f
        out.append(rec)
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
                     circle_starts: Optional[List[Tuple[str, Any]]] = None,
                     circle_note: Optional[str] = None,
                     seed_surface: Optional[Sequence] = None,
                     wedges: bool = True, refine: bool = True,
                     final_fraction: float = FINAL_FRACTION,
                     tol: float = 1e-4,
                     time_up: Callable[[], bool] = lambda: False,
                     record_stop: Optional[Callable] = None,
                     own_label: str = "the search's own best surface",
                     wedge_only: bool = False) -> Dict[str, Any]:
    """Every start refined separately, the lowest result first (EM
    4-2e(1)(b), pp. 4-3 and 4-6; 4-2e(2), p. 4-7; C-1g(4), p. C-11; FHWA
    p. 6-22).

    The starts: the search's own best surface (``own_best``, its points),
    each locally critical circle (``circle_starts``, (label, circle) from
    `circle_starts_of`), a wedge along each layer (`wedge_starts`, when
    ``wedges``), and the caller's surface (``seed_surface``). With
    ``refine`` False nothing is shifted: the starts are scored as they are
    (``wedge_only``: the wedge stage alone, Slide2's search without
    optimisation). ``score(points)`` gives F; ``flags_of(points)`` the
    flags of the printed checks on the surface.

    Returns the record the search puts on its result as ``refinement``:
    {final_increment (stratum, thickness, fraction, increment), start_
    increment, n_vertices, wedges (one per layer), local_minima (one per
    start, lowest first: start, F_start, FOS, points, flags, refined,
    levels, n_scored), circle_starts (the note where there are none),
    unrefined (starts the time budget left), best (points, FOS, start),
    n_scored}."""
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
    d_final = (final_fraction * stratum["thickness"] if stratum
               else final_fraction * max(h, 1.0))
    d_final = max(d_final, 1e-4)
    d_start = max(START_FRACTION_H * h, d_final)
    rec: Dict[str, Any] = {
        "final_increment": {
            "stratum": None if stratum is None else stratum["layer"],
            "thickness": None if stratum is None else stratum["thickness"],
            "at_x": None if stratum is None else stratum["x"],
            "fraction": final_fraction,
            "increment": round(d_final, 4),
            "basis": ("EM 1110-2-1902 4-2e(2)(b), p. 4-7: final shift "
                      "increments no more than 10 to 25 percent of the "
                      "thickness of the thinnest stratum the surface may "
                      "pass through")},
        "start_increment": round(d_start, 4),
        "n_vertices": N_VERTICES,
        "refined": bool(refine),
        "wedges": [], "local_minima": [], "unrefined": [],
    }
    n_scored = 0
    starts: List[Tuple[str, List[Tuple[float, float]]]] = []
    if own_best is not None and not wedge_only:
        starts.append((own_label, list(own_best)))
    if circle_starts:
        for label, c in circle_starts:
            p = circle_start(geom, c)
            if p is not None:
                starts.append((label, p))
    elif circle_note:
        rec["circle_starts"] = circle_note
    if wedges:
        ws = wedge_starts(geom, windows, score, z_floor=z_floor,
                          windows_given=windows_given, time_up=time_up)
        for w in ws:
            n_scored += w["n_scored"]
            rec["wedges"].append({
                "layer": w["layer"], "n_scored": w["n_scored"],
                "FOS": None if w["FOS"] is None else round(w["FOS"], 4),
                "points": None if w["points"] is None else
                [(round(x, 3), round(z, 3)) for x, z in w["points"]]})
            if w["points"] is not None:
                starts.append((f"wedge along {w['layer']}", w["points"]))
    if seed_surface is not None:
        starts.append(("the caller's surface (seed_surface)",
                       [(float(x), float(z)) for x, z in seed_surface]))
    if wedge_only:
        starts = [s for s in starts if s[0].startswith("wedge along")]

    minima = []
    for k, (label, pts) in enumerate(starts):
        if refine:
            pts = with_vertices(pts)
        if refine and time_up():
            rec["unrefined"].append(label)
            continue
        f0 = score(pts)
        n_scored += 1
        if f0 >= 900:
            minima.append({"start": label, "F_start": None, "FOS": None,
                           "points": None, "refined": False,
                           "note": "the start is not scored (the method "
                                   "has no solution on it, or a screen "
                                   "refused it)"})
            continue
        if refine:
            r = refine_polyline(geom, pts, score, d_start=d_start,
                                d_final=d_final, windows=windows,
                                windows_given=windows_given, tol=tol,
                                time_up=time_up, f_start=f0)
            n_scored += r["n_scored"]
            if r["stopped"]:
                rec["unrefined"].append(label + " (refinement cut short)")
            m = {"start": label, "F_start": round(f0, 4),
                 "FOS": r["FOS"], "points": r["points"], "refined": True,
                 "levels": r["levels"], "n_scored": r["n_scored"],
                 "toe_limit_deg": r["toe_limit_deg"],
                 "eq_c32_deg": r["eq_c32_deg"]}
        else:
            m = {"start": label, "F_start": round(f0, 4), "FOS": f0,
                 "points": pts, "refined": False}
        minima.append(m)
    if rec["unrefined"] and record_stop is not None:
        record_stop(len([m for m in minima if m.get("refined")]),
                    len(starts), t0, rec["unrefined"])
    scored = [m for m in minima if m["FOS"] is not None]
    scored.sort(key=lambda m: m["FOS"])
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
    out.append(choice(
        "final shift increment of the noncircular refinement",
        f"{fi['increment']} m = {fi['fraction']:g} x the thinnest stratum "
        f"({fi['stratum']}, {fi['thickness']} m least vertical thickness "
        f"within the windows)",
        how_given(fraction, None, name="refine_fraction"), fi["basis"],
        ["refine_fraction up to 0.25, the print's upper end (the default "
         "0.10 is the module's call: accuracy first, and it meets both ends "
         "of the printed range)"]))
    out.append(choice(
        "starting increment and vertices of the noncircular refinement",
        f"start {rec['start_increment']} m (0.10 x the ground's relief in "
        f"the windows), halved to the final increment; "
        f"{rec['n_vertices']} vertices",
        "module",
        "not in print: EM C-1g(3) gives the method, not the increments or "
        "the vertex count", ["other starting increments and vertex counts"]))
    out.append(choice(
        "starts of the noncircular refinement",
        "the search's own best surface; each locally critical circle of "
        "the circular search run for circle_check (its grid minima and the "
        "toe and above-toe families' critical circles); a wedge along each "
        "layer within the search's depth; the caller's seed_surface",
        "module",
        "EM 4-2e(2), p. 4-7: the initial surface should match the estimated "
        "critical surface; 4-2e(1)(b), pp. 4-3 and 4-6, and C-1g(4), p. "
        "C-11: separate searches; FHWA NHI-06-088 p. 6-22",
        ["circle_check=False: no circle starts"]))
    out.append(choice(
        "wedge starts",
        f"one per layer within the search's depth; the central block on "
        f"the layer's mid-line, its ends at {WEDGE_POSITIONS} positions "
        f"each over the windows; wedges at 45 +/- phi'_D/2 on the trial's "
        f"F (F = 1 first, then the F each pass gave, {WEDGE_F_PASSES} "
        f"passes at most), phi' of the material just above the base end",
        "module",
        "EM 1110-2-1902 C-1f(2), p. C-6; C-1g(2), pp. C-8 to C-10, Fig. "
        "C-7b; FHWA NHI-06-088 pp. 6-43 to 6-44. Which layers, how many "
        "positions and how F is updated are the module's calls",
        ["only layers weaker than the one above (FHWA p. 6-33), compared "
         "at a normal stress no print gives"]))
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
