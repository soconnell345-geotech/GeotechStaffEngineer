"""
Supplementary trial-surface families for the critical-surface search.

The centre-grid and entry/exit searches sample circles inside windows
that were set from the MODEL extent until GM-57 (entry in the first 40 % of
the model width, exit in the last 40 %, centres at least 1 unit above the
crest) and are now set from the slope (`search_ranges.py`). Two failure
mechanisms fall outside that sampling on ordinary slopes and the search then
reports a factor of safety that is too HIGH (BACKLOG GM-13, GM-14, and
GM-122 for the third):

1. **Toe circles** (Taylor 1937; Duncan, Wright & Brandon 2014, Ch. 6-7).
   A circle through the toe of the slope whose other end is close above it.
   On a steep slope in a short model the critical toe circle exited on the
   crest well inside the automatic exit window's lower bound, and its centre
   lies below the crest, under the bottom of the automatic centre grid.
   Steep phi = 0 slope, H = 5 m at 45 deg, su = 40 kPa: grid 2.78 (2.348
   with the ranges from the slope) against a toe circle of 2.346. (Before
   GM-134 a toe circle closed where the arc turns vertical read 2.287 there,
   a slicing error at that end; 2.379 now, 2.384 on 10,240 slices.)

2. **Infinite slope / shallow planar (translational) sliding** in
   cohesionless soil (Duncan, Wright & Brandon 2014, Ch. 14; Skempton &
   DeLory 1957). With c' = 0 the critical failure is a shallow slide
   parallel to the slope face, FOS -> tan(phi') / tan(beta) for dry
   conditions as the slide gets shallower, independent of depth. Every
   circle the grid tries is deeper and stronger. Slide2 #86 (c' = 0,
   phi' = 37 deg fill at 1.25H:1V, no supports): grid 1.57 against 0.942.

3. **Circles passing above the toe** (FHWA NHI-06-088 p. 6-38: "If there
   is a weak layer above the toe of the slope, a circle passing above the
   toe of the slope may be more critical. Similarly, if there is water
   outside the toe of the slope, a circle passing above the water may be
   more critical"). The automatic windows meet at the middle of the face
   and the main search's radii run from just below the highest ground to
   the base, so a circle that comes out on the face is scored only if it
   also crosses the middle of the face. Until GM-122 such circles were
   reached as circles crossing the ground four times, scored across the
   air in front of the toe (1.14 on the model below); those are now
   refused, and this family starts circles on each face at 0.1 to 0.9 of
   its height, refines the best start's height and scores them like the
   toe circles. A 10 m face at 45 deg whose upper 3.5 m is c' = 3 kPa,
   phi' = 22 deg over c' = 200 kPa, phi' = 40 deg: main search 6.33, toe
   circle 6.31, circle passing above the toe 0.936 (Bishop; Spencer
   0.935), from the face where the upper layer's base comes out.

`apply_supplements` scores both circle families with the search's own limit
equilibrium method and the planar surfaces by force equilibrium (Janbu's
simplified method; the infinite slope is a force-equilibrium result, see
`check_infinite_slope`), and folds them into the `SearchResult`: the lowest
surface becomes `critical`, `governing_surface` names its kind,
`candidates` lists each family's minimum, `infinite_slope_checks` records
every cohesionless face, and `warnings` say plainly where the search cannot
be sure. Explicit (caller-given) entry/exit windows and centre ranges are
honoured; the automatic windows are a heuristic and do not restrict the
supplementary families.

No trial surface can pass the end of the ground profile, so a governing
surface near an end is also checked against circles about nearby centres
on the ground carried on level past the end (`_profile_end_warning`, GM-63;
noncircular surfaces since GM-87). Duncan Ex 1 (12.2 m 2:1 slope in clay,
su 28.7 kPa, 24.4 m of clay below the toe) drawn with 12.2 m of level
ground beyond the crest and the toe: the search gives 0.727 (0.726 with the
refined grid) from one end of the profile to the other, while Taylor's
critical circle (FHWA NHI-06-088 Fig. 6-15, D = 3) enters 3.2 H behind the
crest and comes out 2.7 H beyond the toe at 0.679.
"""

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface, circle_of)
from slope_stability.results import SearchResult, SlopeStabilityResult
from geotech_common.water import GAMMA_W as _GAMMA_W

# A ground-surface segment flatter than this is not a slope face.
_MIN_FACE_DEG = 5.0

# Toe-circle family: the circle's other end is placed at these multiples of
# the face's characteristic size (max of its horizontal length and height)
# from the toe, densest near the toe where the small circles are.
_EXIT_FACTORS = (0.15, 0.3, 0.5, 0.75, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0)
# Centre offset along the chord's perpendicular bisector, as a ratio of the
# chord: 0.02 is nearly a semicircle, 4 is a flat arc.
_T_RATIO_RANGE = (0.02, 4.0)
_N_T = 10
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
# Only the tallest faces get a toe-circle family (runtime bound).
_MAX_TOE_FACES = 3
# Circles passing above the toe (GM-122 follow-up): the lower end on the
# face at these fractions of its height above the toe, the other end
# upslope as for the toe circle; the best start's height is then refined.
# A caller-supplied per-surface solve (rapid drawdown, two or three GLE
# solves a surface) gets two starts and no refinement of the height.
_ABOVE_TOE_HEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_ABOVE_TOE_HEIGHTS_FOS_FN = (0.35, 0.7)
# Rigorous methods scan the toe circles with Bishop, then re-optimize this
# many of the best circles with the method itself (Janbu too, in the circles
# passing above the toe only: `search_toe_circles`).
_SCREENED_METHODS = ("spencer", "morgenstern_price", "gle")
_N_RESCORE = 3
# A circle the rigorous method finds no solution on (large point loads:
# GM-60) does not count toward _N_RESCORE; the next one down the Bishop
# shortlist is tried instead, up to this many in all.
_MAX_RESCORE_TRIES = 12

# Infinite-slope family: shallow planar surfaces parallel to the face, at a
# vertical depth of 1/_PLANAR_DEPTH_SLICES of a slice width. The short end
# ramps that close the surface to the ground (1.5 times the depth long) are
# slices of their own, since every corner of a polyline is a slice boundary
# (GM-131), and they move the FOS off the closed form by about their share
# of the slide's weight, 0.75 / (_PLANAR_DEPTH_SLICES * n_slices) each end:
# at a tenth of a slice width deep a drawdown face read 0.7 % above the
# closed form on 20 slices, at a twentieth 0.4 %. Thinner is not better
# under a full pool: at a fortieth the stage-1 solve of a submerged face
# drifts up, and Corps 2-stage read 2.8 % above its closed form, (gamma' /
# gamma) tan(phi') / tan(beta), against 1.5 % at a twentieth. The slide
# spans the whole outcrop, each half and each third, so a face wet over
# part of its length is caught.
_PLANAR_DEPTH_SLICES = 20.0
_PLANAR_SPLITS = (1, 2, 3)
# The method the planar surfaces are scored with (see check_infinite_slope).
PLANAR_METHOD = "janbu"
# Note on a face whose reinforcement leaves no shallow planar slide able to
# fail (the solver caps the FOS); such a face needs no infinite-slope warning.
HELD_NOTE = ("every shallow planar slide is held by the reinforcement "
             "crossing the face (no net driving force is left, so there is "
             "no factor of safety)")

# Surface-family labels (the references' names).
CIRCULAR = "circular"
NONCIRCULAR = "noncircular"
TOE_CIRCLE = "toe_circle"
ABOVE_TOE = "circle_above_toe"
INFINITE_SLOPE = "infinite_slope"

_KIND_TEXT = {
    CIRCULAR: "circular (search)",
    NONCIRCULAR: "noncircular (search)",
    TOE_CIRCLE: "toe circle",
    ABOVE_TOE: "circle passing above the toe",
    INFINITE_SLOPE: "infinite slope (shallow planar, translational)",
}


def describe_surface_kind(kind: str) -> str:
    """Plain-language name of a `governing_surface` value."""
    return _KIND_TEXT.get(kind, kind)


# ---------------------------------------------------------------------------
# Slope faces
# ---------------------------------------------------------------------------

def slope_faces(geom: SlopeGeometry,
                min_deg: float = _MIN_FACE_DEG) -> List[Dict[str, Any]]:
    """Maximal runs of consecutive ground segments inclined the same way.

    Each face is a dict with the toe (its LOW end) and crest (its HIGH end),
    ``rises`` (+1 when the face rises toward +x, -1 toward -x), the face
    ``height`` and ``length`` (horizontal). A segment flatter than
    ``min_deg`` ends a face, so each bench of a benched slope is its own
    face with its own toe.
    """
    pts = geom.surface_points
    faces: List[Dict[str, Any]] = []
    cur = None
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        ang = math.degrees(math.atan2(z1 - z0, x1 - x0))
        direction = 1 if ang >= min_deg else (-1 if ang <= -min_deg else 0)
        if direction != 0 and cur is not None and cur["rises"] == direction:
            cur["x_end"], cur["z_end"] = x1, z1
            continue
        if cur is not None:
            faces.append(cur)
            cur = None
        if direction != 0:
            cur = {"rises": direction, "x_start": x0, "z_start": z0,
                   "x_end": x1, "z_end": z1}
    if cur is not None:
        faces.append(cur)

    out = []
    for f in faces:
        if f["rises"] > 0:
            toe, crest = (f["x_start"], f["z_start"]), (f["x_end"], f["z_end"])
        else:
            toe, crest = (f["x_end"], f["z_end"]), (f["x_start"], f["z_start"])
        out.append({
            "rises": f["rises"], "toe": toe, "crest": crest,
            "height": abs(crest[1] - toe[1]),
            "length": abs(crest[0] - toe[0]),
        })
    return out


# ---------------------------------------------------------------------------
# Toe circles
# ---------------------------------------------------------------------------

def _centre_admissible(geom, xc, yc, x_range, y_range) -> bool:
    """Centre above the ground at its own x (the slip mass is the region
    between the ground and the LOWER arc) and inside any caller-given
    centre ranges."""
    if yc <= geom.ground_elevation_at(xc):
        return False
    if x_range is not None and not (x_range[0] <= xc <= x_range[1]):
        return False
    if y_range is not None and not (y_range[0] <= yc <= y_range[1]):
        return False
    return True


def _golden_min(f, a, b, tol, max_iter=30):
    """Golden-section minimum of a unimodal f on [a, b]: (x, f(x), n_calls)."""
    n = 0
    x1 = b - _GOLDEN * (b - a)
    x2 = a + _GOLDEN * (b - a)
    f1, f2 = f(x1), f(x2)
    n += 2
    for _ in range(max_iter):
        if b - a < tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - _GOLDEN * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + _GOLDEN * (b - a)
            f2 = f(x2)
        n += 1
    return (x1, f1, n) if f1 < f2 else (x2, f2, n)


def search_toe_circles(geom: SlopeGeometry,
                       method: str = "bishop",
                       n_slices: int = 30,
                       tol: float = 1e-4,
                       x_entry_range: Optional[Tuple[float, float]] = None,
                       x_exit_range: Optional[Tuple[float, float]] = None,
                       x_range: Optional[Tuple[float, float]] = None,
                       y_range: Optional[Tuple[float, float]] = None,
                       fos_fn=None,
                       above_toe: bool = True,
                       ) -> Dict[str, Any]:
    """Critical circle through the toe of each slope face, and critical
    circle passing above it.

    For each face (the tallest ``_MAX_TOE_FACES``), circles pass through the
    toe and a second ground point upslope of it. That point is scanned at
    ``_EXIT_FACTORS`` of the face size (densest near the toe), the centre
    along the chord's perpendicular bisector at ``_N_T`` offsets, and both
    are refined by golden section. Every circle is scored by the search's
    own `_compute_fos`, so the same admissibility guards apply. Spencer /
    Morgenstern-Price / GLE scan with Bishop (the same critical circles to
    about 1 % and some 30 times cheaper) and re-optimize the best
    ``_N_RESCORE`` circles with the method itself; the reported FOS is
    always the method's own. A circle on which the rigorous method has no
    solution (a large support force can leave a small circle with none)
    is passed over for the next one down the list.

    ``x_entry_range`` / ``x_exit_range`` / ``x_range`` / ``y_range`` are
    CALLER-GIVEN constraints only (pass None for automatic ones).
    ``fos_fn`` substitutes a per-surface FOS (`_compute_fos`'s hook, used
    by `search_rapid_drawdown`); every circle is then scored with it and
    there is no Bishop screen.

    With ``above_toe`` the same scan starts again from points on each face
    at ``_ABOVE_TOE_HEIGHTS`` of its height above the toe: circles passing
    above the toe, which come out on the face (FHWA NHI-06-088 p. 6-38:
    "If there is a weak layer above the toe of the slope, a circle passing
    above the toe of the slope may be more critical. Similarly, if there is
    water outside the toe of the slope, a circle passing above the water
    may be more critical"). The main search's windows ask a surface to
    cross the middle of the face (`search_ranges`) and its radii are sized
    from the base, so it does not reach them (GM-122 follow-up). Janbu
    scans it with Bishop as well: its steep entries are where Janbu's
    solve falls with the slice count. Like every family, both leave out a
    circle with m_alpha < 0.2 on a slice (`_compute_fos`; Slide2's check,
    its error -112). The circles through the toe treat a reverse-curvature
    circle as the geometry's ``reverse_curvature`` says, as the main search
    does; the circles passing above the toe leave such circles out. Made
    the same either way, a pinned case moved away from its published
    value on a circle Slide2 would keep (GM-127; DESIGN.md).

    Returns a dict: ``FOS`` (999.9 if none admissible), ``xc``, ``yc``,
    ``radius``, ``toe`` (x, z), ``n_evaluated``, ``n_scored`` (the circles
    scored by family, ``TOE_CIRCLE`` and ``ABOVE_TOE``); with ``above_toe``,
    ``above_toe`` holds the critical circle passing above the toe (``FOS``,
    ``xc``, ``yc``, ``radius``, ``start`` (x, z) on the face, ``height``,
    its fraction of the face height above the toe, and ``face_toe``), and
    ``faces`` the faces covered with the heights started from.
    """
    from slope_stability.search import (
        _compute_fos, _check_entry_exit, _FOS_MAX, _time_up, _record_not_run,
        _scoring_family, _dips_below_base,
     _remember_arc)

    scan_method = ("bishop" if method in _SCREENED_METHODS and fos_fn is None
                   else method)
    # The circles passing above the toe are screened with Bishop for Janbu
    # too: they can enter the face steeply, where Janbu's solve falls with
    # the slice count or fails (Slide2 #85's clay face: Bishop and Spencer
    # 1.22, Janbu 0.86 on 20 slices, 0.70 on 30, none on 40 or more).
    family_scan = {TOE_CIRCLE: scan_method,
                   ABOVE_TOE: ("bishop" if method in _SCREENED_METHODS
                               + ("janbu",) and fos_fn is None else method)}
    faces = sorted(slope_faces(geom), key=lambda f: -f["height"])
    faces = faces[:_MAX_TOE_FACES]
    x_min, x_max = geom.x_range
    windows_given = x_entry_range is not None or x_exit_range is not None
    counter = [0]
    # circles scored, by family: what a result's choices say was tried
    scored = {TOE_CIRCLE: 0, ABOVE_TOE: 0}
    # A caller-supplied per-surface solve (rapid drawdown: two or three GLE
    # solves a surface, no cheap screen) gets a coarser scan.
    n_t, u_tol, e_iter = (_N_T, 0.01, 12) if fos_fn is None else (7, 0.04, 6)
    u_lo = math.log(_T_RATIO_RANGE[0])
    u_hi = math.log(_T_RATIO_RANGE[1])
    us = [u_lo + (u_hi - u_lo) * i / (n_t - 1) for i in range(n_t)]
    u_step = (u_hi - u_lo) / (n_t - 1)

    def toe_arc(ch, xc, yc, r):
        """The family's trial surface on the circle through the chord's two
        ends (GM-343, "Where a circle ends"): the arc of that circle whose
        downslope end is the chord's start (the toe, or the point on the
        face), carrying its two ends, or None where the circle has no such
        arc (it is in the air above the start, or runs on below it). The
        arc ends at the toe: none is carried under the ground in front of
        it. Before, the radius was shrunk by a millionth so that the arc
        passed just above the toe, and the 200-point sampling stepped over
        the gap, so a circle that came back under the ground in front of
        the toe was read as one surface from there (Taylor's circle for a
        vertical cut: 2.676 for 0.851)."""
        slip = CircularSlipSurface(xc, yc, r)
        try:
            arcs = slip.arcs(geom)
        except ValueError:
            return None
        tol = ch.get("tol", 1e-6 * max(1.0, r))
        for a in arcs:
            low = a.x_start if ch["s"] > 0 else a.x_end
            if abs(low - ch["toe"][0]) <= tol:
                if len(arcs) > 1:
                    _remember_arc(slip, a)
                return a
        return None

    def circle_fos(ch, xc, yc, r, fos_method, family=TOE_CIRCLE):
        if _time_up():          # the search's time budget is spent (GM-123)
            _record_not_run("part of the toe-circle and circle-above-the-"
                            "toe families")
            return _FOS_MAX
        if not _centre_admissible(geom, xc, yc, x_range, y_range):
            return _FOS_MAX
        slip = toe_arc(ch, xc, yc, r)
        if slip is None:
            return _FOS_MAX
        if windows_given and not _check_entry_exit(
                geom, slip, x_entry_range, x_exit_range):
            return _FOS_MAX
        counter[0] += 1
        scored[family] += 1
        # (m_alpha < 0.2 on a slice: left out and counted there, GM-127;
        # no solution: not scored, counted under this family, GM-140 step 2)
        with _scoring_family(family):
            return _compute_fos(geom, slip, fos_method, n_slices, tol=tol,
                                fos_fn=fos_fn)

    def chord(toe, s, offset):
        """The chord from `toe` (the toe, or a point on the face above it)
        to the ground point `offset` upslope."""
        xt, zt = toe
        xe = xt + s * offset
        ze = geom.ground_elevation_at(xe)
        (xl, zl), (xr, zr) = sorted([(xt, zt), (xe, ze)])
        dx, dz = xr - xl, zr - zl
        c = math.hypot(dx, dz)
        if c < 1e-9:
            return None
        return {"toe": toe, "c": c, "mx": 0.5 * (xl + xr),
                "mz": 0.5 * (zl + zr), "px": -dz / c, "pz": dx / c,
                "xl": xl, "xr": xr, "s": s}

    def below_base(ch, xc, yc, r):
        """True when the circle reaches below the bottom of the deepest
        layer: the one test every family uses (`search._dips_below_base`,
        GM-128's rule, measured on the surface between its ends), on the
        circle as `circle_fos` builds it (GM-140 step 3, D19). Before, this
        family had a test of its own, on the arc between the chord ends
        with a tolerance of a millionth of the chord, so the families
        admitted slightly different circles. Where the toe sits on the
        base, a circle through it that dips under the base pivots about the
        toe into ground the model does not have (Slide2 #85)."""
        slip = toe_arc(ch, xc, yc, r)
        return slip is not None and _dips_below_base(geom, slip)

    def circ(ch, u):
        """Circle through both chord ends, its centre a distance
        c*exp(u) up the perpendicular bisector."""
        t = ch["c"] * math.exp(u)
        return (ch["mx"] + t * ch["px"], ch["mz"] + t * ch["pz"],
                math.hypot(0.5 * ch["c"], t))

    def chord_best(ch, fos_method, bracket=None):
        """(fos, u) of the best circle on chord `ch`: a scan over `us`
        then golden section, or golden section inside `bracket`."""
        if ch is None:
            return (_FOS_MAX, 0.0)

        def f(u):
            xc, yc, r = circ(ch, u)
            if below_base(ch, xc, yc, r):
                return _FOS_MAX
            if ch.get("above") and not passes_above_toe(ch, xc, yc, r):
                return _FOS_MAX
            return circle_fos(ch, xc, yc, r, fos_method,
                              ABOVE_TOE if ch.get("above") else TOE_CIRCLE)

        if bracket is None:
            fs = [f(u) for u in us]
            k = min(range(n_t), key=lambda i: fs[i])
            u_best, f_best = us[k], fs[k]
            if f_best >= _FOS_MAX:
                return (f_best, u_best)
            bracket = (us[max(k - 1, 0)], us[min(k + 1, n_t - 1)])
        else:
            u_best, f_best = 0.5 * (bracket[0] + bracket[1]), _FOS_MAX
        u_g, f_g, _ = _golden_min(f, bracket[0], bracket[1], tol=u_tol)
        if f_g < f_best:
            u_best, f_best = u_g, f_g
        return (f_best, u_best)

    def face_point(face, h):
        """The point on the face `h` of its height above the toe (on a
        vertical face, the point at that height)."""
        (xt, zt), (xk, zk) = face["toe"], face["crest"]
        z = zt + h * (zk - zt)
        a, b = sorted((xt, xk))
        for _ in range(60):
            m = 0.5 * (a + b)
            if (geom.ground_elevation_at(m) < z) == (xt < xk):
                a = m
            else:
                b = m
        return (0.5 * (a + b), z)

    def passes_above_toe(ch, xc, yc, r):
        """True when the circle's downslope end is the chord's start on the
        face (one that comes out lower is a toe or base circle) and it is
        not a reverse-curvature circle closed by a crack: the family's
        circles keep their centre above the ground at both ends (the
        circles through the toe follow ``reverse_curvature``; GM-127)."""
        slip = toe_arc(ch, xc, yc, r)
        if slip is None:
            return True     # no such arc: not scored (circle_fos)
        ends = slip.find_entry_exit(geom)
        return slip.reverse_curvature_crack(geom, *ends) is None

    heights = ((_ABOVE_TOE_HEIGHTS if fos_fn is None
                else _ABOVE_TOE_HEIGHTS_FOS_FN) if above_toe else ())
    # (fos with the family's scan method, chord, u, height above the toe as
    # a fraction of the face's): through the toe, and passing above it
    shortlists = {TOE_CIRCLE: [], ABOVE_TOE: []}
    covered = []
    for face in faces:
        s = face["rises"]
        size = max(face["length"], face["height"])
        if size <= 0:
            continue
        covered.append({"toe": face["toe"], "crest": face["crest"],
                        "height": face["height"]})
        starts = [(TOE_CIRCLE, face["toe"], 0.0)] + [
            (ABOVE_TOE, face_point(face, h), h) for h in heights]

        def chord_at(family, start, o):
            ch = chord(start, s, o)
            if ch is not None and family == ABOVE_TOE:
                ch.update(above=True, s=s, tol=0.005 * size,
                          face_toe=face["toe"])
            return ch

        def offsets_from(start):
            reach = abs((x_max if s > 0 else x_min) - start[0])
            offsets = sorted({min(size * f, reach) for f in _EXIT_FACTORS})
            return [o for o in offsets if o > 1e-6 * size]

        # the coarse scan from every start; then, from the best start of
        # each family on this face only, the start's height refined (golden
        # section, the far end held where it is) and the far end refined
        best_start = {}
        for family, start, h in starts:
            offsets = offsets_from(start)
            if not offsets:
                continue

            coarse = []
            for o in offsets:
                ch = chord_at(family, start, o)
                fos, u = chord_best(ch, family_scan[family])
                coarse.append((fos, ch, u, h))
            shortlists[family].extend(coarse)
            j = min(range(len(offsets)), key=lambda i: coarse[i][0])
            if (family not in best_start
                    or coarse[j][0] < best_start[family][0]):
                best_start[family] = (coarse[j][0], start, h, offsets, j)
        above_best = best_start.get(ABOVE_TOE)
        if (above_best is not None and above_best[0] < _FOS_MAX
                and fos_fn is None):
            f0, start, h, offsets, j = above_best
            k = heights.index(h)
            lo = heights[k - 1] if k > 0 else 0.0
            hi = heights[k + 1] if k + 1 < len(heights) else 1.0
            x_far = start[0] + s * offsets[j]
            cache = {}

            def g(hh):
                st = face_point(face, hh)
                ch = chord_at(ABOVE_TOE, st, abs(x_far - st[0]))
                cache[hh] = chord_best(ch, family_scan[ABOVE_TOE]) + (ch,)
                return cache[hh][0]

            h_g, f_g, _ = _golden_min(g, lo, hi, tol=0.005, max_iter=e_iter)
            fos, u, ch = cache[h_g]
            if ch is not None and fos < f0:
                shortlists[ABOVE_TOE].append((fos, ch, u, h_g))
                start, h = ch["toe"], h_g
                offsets = offsets_from(start)
                j = min(range(len(offsets)),
                        key=lambda i: abs(offsets[i] - abs(x_far - start[0])))
                best_start[ABOVE_TOE] = (fos, start, h, offsets, j)
        for family, (f0, start, h, offsets, j) in best_start.items():
            if f0 < _FOS_MAX and len(offsets) > 1:
                cache = {}

                def g(o):
                    ch = chord_at(family, start, o)
                    cache[o] = chord_best(ch, family_scan[family]) + (ch,)
                    return cache[o][0]

                o_g, f_g, _ = _golden_min(
                    g, offsets[max(j - 1, 0)],
                    offsets[min(j + 1, len(offsets) - 1)],
                    tol=0.005 * size, max_iter=e_iter)
                fos, u, ch = cache[o_g]
                shortlists[family].append((fos, ch, u, h))

    def ranked(family):
        """The family's shortlist best first, the best re-optimized with the
        method itself when it was scanned with Bishop."""
        shortlist = [c for c in shortlists[family] if c[0] < _FOS_MAX]
        shortlist.sort(key=lambda c: c[0])
        if family_scan[family] != method:
            rescored = []
            n_ok = 0
            for _fos, ch, u, h in shortlist[:_MAX_RESCORE_TRIES]:
                fos, u2 = chord_best(ch, method, (u - u_step, u + u_step))
                rescored.append((fos, ch, u2, h))
                n_ok += fos < _FOS_MAX
                if n_ok >= _N_RESCORE:
                    break
            shortlist = sorted(rescored, key=lambda c: c[0])
        return shortlist

    shortlist = ranked(TOE_CIRCLE)
    above = ranked(ABOVE_TOE) if heights else []

    best = {"FOS": _FOS_MAX, "xc": 0.0, "yc": 0.0, "radius": 0.0,
            "toe": None, "n_evaluated": counter[0]}
    if shortlist and shortlist[0][0] < _FOS_MAX:
        fos, ch, u, _h = shortlist[0]
        xc, yc, r = circ(ch, u)
        arc = toe_arc(ch, xc, yc, r)
        best.update(FOS=fos, xc=xc, yc=yc, radius=r, toe=ch["toe"],
                    arc=None if arc is None else (arc.x_start, arc.x_end))
    if heights:
        best["faces"] = [dict(f, heights=list(heights)) for f in covered]
        best["above_toe"] = {"FOS": _FOS_MAX}
        if above and above[0][0] < _FOS_MAX:
            fos, ch, u, h = above[0]
            xc, yc, r = circ(ch, u)
            arc = toe_arc(ch, xc, yc, r)
            best["above_toe"] = {"FOS": fos, "xc": xc, "yc": yc,
                                 "radius": r,
                                 "arc": (None if arc is None
                                         else (arc.x_start, arc.x_end)),
                                 "start": ch["toe"], "height": h,
                                 "face_toe": ch["face_toe"]}
    best["n_evaluated"] = counter[0]
    best["n_scored"] = dict(scored)
    return best


# ---------------------------------------------------------------------------
# Infinite slope (shallow planar) in cohesionless soil
# ---------------------------------------------------------------------------

def _is_cohesionless(layer: Optional[SlopeSoilLayer]) -> bool:
    """Drained Mohr-Coulomb soil with c' = 0 and phi' > 0: its strength
    vanishes at the ground surface, so the shallowest sliding is critical."""
    return (layer is not None
            and layer.strength_model == "mohr_coulomb"
            and layer.analysis_mode != "undrained"
            and layer.c_prime <= 1e-9
            and layer.phi > 0.0)


def cohesionless_face_runs(geom: SlopeGeometry,
                           min_deg: float = _MIN_FACE_DEG,
                           n_samples: int = 60) -> List[Dict[str, Any]]:
    """Stretches of each inclined ground segment where a cohesionless layer
    is the material at the surface.

    Each run is a dict: ``x_from``, ``x_to``, ``slope_angle_deg`` (beta,
    positive), ``gradient`` (dz/dx of the segment), ``layer`` (name),
    ``phi_deg``, ``gamma_sat``. A face crossing several layers gives one
    run per cohesionless outcrop.
    """
    pts = geom.surface_points
    x_min, x_max = geom.x_range
    eps = 1e-6 * max(x_max - x_min, 1.0)
    runs: List[Dict[str, Any]] = []
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        m = (z1 - z0) / (x1 - x0)
        beta = math.degrees(math.atan(abs(m)))
        if beta < min_deg:
            continue
        cur = None
        for k in range(n_samples):
            xa = x0 + (x1 - x0) * k / n_samples
            xb = x0 + (x1 - x0) * (k + 1) / n_samples
            xm = 0.5 * (xa + xb)
            layer = geom.layer_at_point(xm, geom.ground_elevation_at(xm) - eps)
            if _is_cohesionless(layer):
                if cur is not None and cur["_layer"] is layer:
                    cur["x_to"] = xb
                    continue
                if cur is not None:
                    runs.append(cur)
                cur = {"x_from": xa, "x_to": xb, "slope_angle_deg": beta,
                       "gradient": m, "layer": layer.name,
                       "phi_deg": layer.phi, "gamma_sat": layer.gamma_sat,
                       "_layer": layer}
            elif cur is not None:
                runs.append(cur)
                cur = None
        if cur is not None:
            runs.append(cur)
    for r in runs:
        r.pop("_layer", None)
    return runs


def _planar_surface(geom: SlopeGeometry, xa: float, xb: float,
                    n_slices: int) -> Optional[PolylineSlipSurface]:
    """Planar slip surface parallel to the (single-segment) face between
    ``xa`` and ``xb``, a twentieth of a slice width deep, with short end
    ramps up to the ground: the ramp at one end flatter than the face, at
    the other steeper, so the surface stays concave. Each ramp is a slice
    of its own (GM-131) and every other slice base lies parallel to the
    face."""
    if xb <= xa or n_slices < 1:
        return None
    dv = (xb - xa) / (n_slices * _PLANAR_DEPTH_SLICES)    # vertical depth
    ramp = 1.5 * dv
    eps = 1e-6 * max(xb - xa, 1.0)

    def above(x):
        # a point a hair above the ground at x, on whichever side is higher
        return max(geom.ground_elevation_at(x - eps),
                   geom.ground_elevation_at(x),
                   geom.ground_elevation_at(x + eps)) + eps

    pts = [
        (xa - eps, above(xa)),
        (xa + ramp, geom.ground_elevation_at(xa + ramp) - dv),
        (xb - ramp, geom.ground_elevation_at(xb - ramp) - dv),
        (xb + eps, above(xb)),
    ]
    try:
        return PolylineSlipSurface(pts)
    except ValueError:
        return None


# The infinite slope with c' = 0 and the water at the face's surface, by
# the reading the run used (GM-140 step 5a review): (formula, condition).
_SEEPAGE_FORMS = {
    "water_table": (
        "(gamma'/gamma_sat) tan(phi')/tan(beta)",
        "a face with a water table at its surface and seepage parallel to "
        "it (water_line_kind 'water_table'; FHWA NHI-06-088 Sec. 6.3.2, "
        "p. 6-8)"),
    "piezometric": (
        "(gamma_sat cos^2(beta) - gamma_w) tan(phi') / (gamma_sat "
        "sin(beta) cos(beta))",
        "a face with a piezometric line at its surface (water_line_kind "
        "'piezometric', the default: the full head, u = gamma_w z)"),
    "rapid_drawdown": (
        "(gamma'/gamma) tan(phi')/tan(beta)",
        "a face saturated to the surface with seepage parallel to it (as "
        "after a rapid drawdown)"),
}


def _seepage_closed_form(closed: float, beta_deg: float, gsat: float,
                         reading: str) -> float:
    """The infinite slope's FOS for c' = 0 with the water at the face's
    surface, by the reading of the water line (``_SEEPAGE_FORMS``).
    ``closed`` is the dry form tan(phi')/tan(beta).

    - 'water_table' (u = gamma_w z cos^2(beta)): (gamma'/gamma_sat)
      tan(phi')/tan(beta), FHWA NHI-06-088 Sec. 6.3.2, p. 6-8.
    - 'piezometric' (the full head, u = gamma_w z), the same statics:
      (gamma_sat cos^2(beta) - gamma_w) tan(phi') / (gamma_sat sin(beta)
      cos(beta)), that is the dry form x (1 - gamma_w / (gamma_sat
      cos^2(beta))).
    - 'rapid_drawdown' (a drawdown search, whose pools are level):
      (gamma'/gamma) tan(phi')/tan(beta), the saturated face a rapid
      drawdown leaves, as before.
    """
    if gsat <= 0:
        return 0.0
    if reading == "piezometric":
        c2 = math.cos(math.radians(beta_deg)) ** 2
        return closed * max(gsat * c2 - _GAMMA_W, 0.0) / (gsat * c2)
    return closed * max(gsat - _GAMMA_W, 0.0) / gsat


def check_infinite_slope(geom: SlopeGeometry,
                         method: str = PLANAR_METHOD,
                         n_slices: int = 30,
                         tol: float = 1e-4,
                         x_entry_range: Optional[Tuple[float, float]] = None,
                         x_exit_range: Optional[Tuple[float, float]] = None,
                         fos_fn=None,
                         ) -> Dict[str, Any]:
    """Shallow planar (infinite-slope) surfaces on every cohesionless face.

    For each run from `cohesionless_face_runs`, shallow planar surfaces
    parallel to the face (`_planar_surface`; over the whole run, each half
    and each third) are scored by the search's own `_compute_fos` -- so pore
    pressure, ponded water, surcharge, seismic load and reinforcement are
    treated exactly as for any other surface. (Reinforcement anchored at the
    face crosses a shallow planar slide near its head, so supports such as
    tiebacks keep this family from governing.)

    ``method`` defaults to Janbu's simplified method (`PLANAR_METHOD`). The
    infinite slope is a force-equilibrium result (Duncan, Wright & Brandon
    2014, Ch. 14: the interslice forces on a slice of an infinite slope are
    equal and opposite), and Janbu's method -- force equilibrium with no
    interslice shear, f0 -> 1 as the slide gets shallow -- reproduces the
    closed form with pore pressure and kh alike. Moment equilibrium carries
    no information on a thin planar slide: the rigorous GLE often finds no
    F_m = F_f crossing there and rejects the surface, and Spencer at 1 %
    depth ran 6 % above the closed form on a 2:1 sand face.

    Caller-given entry/exit windows clip the run; a run they exclude is
    reported, not evaluated. Surfaces are not evaluated where a tension
    crack is modelled (the crack model would strip the strength of half of
    a shallow surface); the closed-form value is still reported.

    Returns a dict: ``FOS`` (999.9 if nothing evaluated), ``slip``
    (PolylineSlipSurface or None), ``checks`` (one dict per run, see
    `SearchResult.infinite_slope_checks`), ``n_evaluated``.
    """
    from slope_stability.search import (
        _compute_fos, _FOS_MAX, trial_reinforcement, _scoring_family,
    )

    best_fos, best_slip = _FOS_MAX, None
    checks: List[Dict[str, Any]] = []
    n_eval = 0
    reading = ("rapid_drawdown" if fos_fn is not None
               else getattr(geom, "water_line_kind", "piezometric"))
    for run in cohesionless_face_runs(geom):
        beta = run["slope_angle_deg"]
        closed = (math.tan(math.radians(run["phi_deg"]))
                  / math.tan(math.radians(beta)))
        seep = _seepage_closed_form(closed, beta, run["gamma_sat"], reading)
        check = {
            "x_from": round(run["x_from"], 3), "x_to": round(run["x_to"], 3),
            "slope_angle_deg": round(beta, 3), "layer": run["layer"],
            "phi_deg": run["phi_deg"],
            "FOS_dry_closed_form": round(closed, 4),
            "FOS_seepage_closed_form": round(seep, 4),
            "seepage_reading": reading,
            "FOS": None, "note": "",
        }
        checks.append(check)

        xa, xb = run["x_from"], run["x_to"]
        lo_x, hi_x = (xa, xb)
        # The left end of the slide must be in the entry window and the
        # right end in the exit window, when the caller gave windows.
        if x_entry_range is not None:
            if hi_x < x_entry_range[0] or lo_x > x_entry_range[1]:
                check["note"] = "excluded by x_entry_range"
                continue
            lo_x = min(max(lo_x, x_entry_range[0]), x_entry_range[1])
        if x_exit_range is not None:
            if hi_x < x_exit_range[0] or lo_x > x_exit_range[1]:
                check["note"] = "excluded by x_exit_range"
                continue
            hi_x = max(min(hi_x, x_exit_range[1]), x_exit_range[0])
        if hi_x - lo_x <= 1e-6 * max(xb - xa, 1e-9):
            check["note"] = "excluded by the entry/exit windows"
            continue
        if geom.tension_crack_depth > 0:
            check["note"] = ("not evaluated: a tension crack is modelled "
                             "(closed form only)")
            continue

        width = hi_x - lo_x
        spans = [(lo_x + width * i / k, lo_x + width * (i + 1) / k)
                 for k in _PLANAR_SPLITS for i in range(k)]
        run_best, run_slip = _FOS_MAX, None
        held = False
        for (a, b) in spans:
            slip = _planar_surface(geom, a, b, n_slices)
            if slip is None:
                continue
            n_eval += 1
            with _scoring_family(INFINITE_SLOPE):
                fos = _compute_fos(geom, slip, method, n_slices, tol=tol,
                                   window_span=(b - a), fos_fn=fos_fn)
            if fos < run_best:
                run_best, run_slip = fos, slip
            elif fos >= _FOS_MAX and trial_reinforcement(geom, slip):
                held = True
        if run_best < _FOS_MAX:
            check["FOS"] = round(run_best, 4)
            if run_best < best_fos:
                best_fos, best_slip = run_best, run_slip
        elif held:
            check["note"] = (HELD_NOTE)
        else:
            check["note"] = "no admissible shallow planar surface"

    return {"FOS": best_fos, "slip": best_slip, "checks": checks,
            "n_evaluated": n_eval}


# ---------------------------------------------------------------------------
# Folding the families into a search result
# ---------------------------------------------------------------------------

def _make_result(geom, slip, fos, method_label, n_slices) -> Optional[
        SlopeStabilityResult]:
    from slope_stability.slices import build_slices
    try:
        from slope_stability.cracks import geometry_at_fos
        x_entry, x_exit = slip.find_entry_exit(geom)
        n = len(build_slices(geometry_at_fos(geom, slip, fos), slip,
                             n_slices))
    except ValueError:
        return None
    kw = {}
    if getattr(slip, "is_circular", True):
        kw = dict(xc=slip.xc, yc=slip.yc, radius=slip.radius)
        if slip.is_bound:
            # which arc of its circle it is, where the circle was split
            # (GM-343)
            from slope_stability.slip_surface import describe_arc
            found = slip.meetings(geom)
            if len(found.arcs) > 1:
                kw["arc"] = describe_arc(slip, found)
    else:
        kw = dict(slip_points=list(slip.points))
    return SlopeStabilityResult(
        FOS=fos, method=method_label, x_entry=x_entry, x_exit=x_exit,
        n_slices=n, has_seismic=geom.kh > 0, kh=geom.kh, **kw)


def _attach_reinforcement(geom, crit) -> None:
    """List the reinforcement crossing the critical surface on the result,
    in `analyze_slope`'s format, so a reinforced search says which elements
    its FOS counted (the FOS itself already includes them: GM-53)."""
    from slope_stability.search import trial_reinforcement
    if crit is None:
        return
    slip = (circle_of(crit)
            if crit.is_circular else PolylineSlipSurface(crit.slip_points))
    from slope_stability.search import _has_reinforcement
    from slope_stability.supports import (resolve_supports,
                                          not_counted_warning)
    if not _has_reinforcement(geom):
        crit.reinforcements = None
        return
    try:
        x_entry, x_exit = slip.find_entry_exit(geom)
        supports = resolve_supports(geom, slip, x_entry, x_exit)
    except ValueError:
        return
    crit.reinforcements = [
        {"kind": f.kind, "index": f.index,
         "x_m": round(f.x, 3), "z_m": round(f.z, 3),
         "T_kN_per_m": round(f.T, 2), "controlled_by": f.controlled_by,
         "support_convention": "passive" if f.passive else "active",
         **({"note": f.note} if f.note else {})}
        for f in supports.forces] or None
    # what counts nothing on this surface, and what was not checked
    # (GM-140 step 8)
    crit.supports_not_counted = supports.not_counted or None
    for w in supports.notes + [not_counted_warning(supports.not_counted)]:
        if w and w not in crit.warnings:
            crit.warnings.append(w)


def _crack_statement(geom, crit, method: str, n_slices: int, tol: float,
                     fos_fn=None) -> Optional[str]:
    """The crack the governing surface was solved with (GM-140 step 4,
    ``slope_stability.cracks``): put on the result (``tension_crack``, the
    depth used) and returned as one sentence, or None without a crack. A
    reverse-curvature circle closed by a crack says so there."""
    if crit is None:
        return None
    from slope_stability import cracks
    from slope_stability.search import _compute_fos
    slip = (circle_of(crit)
            if crit.is_circular else PolylineSlipSurface(crit.slip_points))
    try:
        if cracks.may_need_solution(geom, slip):
            _, crack = cracks.solve_with_crack(
                geom, slip,
                lambda g: _compute_fos(g, slip, method, n_slices, tol=tol,
                                       fos_fn=fos_fn, m_alpha_check=False,
                                       inside_model=False))
        else:
            crack = cracks.given_outcome(geom, slip, crit.x_entry,
                                         crit.x_exit)
    except ValueError:
        return None
    if crack is None or crack.discarded or crack.depth <= 0:
        return None
    had = list(crit.warnings)
    cracks.apply_to_result(crit, crack)
    crit.warnings[:] = had          # said once, in the search's warnings
    return crack.statement()


# Below this share of their force acting through the moment about the centre
# (arm / R), the supports crossing a Bishop / Fellenius critical circle get a
# warning: more of their force then lies in horizontal equilibrium, which
# those methods do not satisfy, than in the moment they count (GM-59).
_SHORT_ARM_SHARE = 0.5


def _support_arm_warning(geom, crit, method: str, n_slices: int,
                         tol: float, spencer_fn=None,
                         in_search: bool = True) -> Optional[str]:
    """Warn when supports cross a Bishop / Fellenius critical circle close
    to the level of its centre, so the method counts little of them.

    Bishop's simplified method satisfies moment equilibrium about the circle
    centre and vertical force equilibrium of each slice, not horizontal
    force equilibrium (Bishop 1955; Wright & Duncan 1991); a support counts
    through its moment about the centre (its component along the slip
    surface times R) and its vertical component. The warning gives the
    share counted and what Spencer, which satisfies all three, finds on the
    same circle.

    ``spencer_fn`` (``analyze_slope``, ``in_search=False``) gives Spencer's
    FOS on the circle, or None for no solution; by default the search's own
    `_compute_fos` scores it."""
    if (crit is None or not crit.is_circular
            or method not in ("bishop", "fellenius")):
        return None
    from slope_stability.search import trial_reinforcement, _compute_fos, \
        _FOS_MAX
    from slope_stability.slices import build_slices
    from slope_stability.reinforcement import support_moment
    slip = circle_of(crit)
    try:
        from slope_stability.cracks import geometry_at_fos
        forces = trial_reinforcement(geom, slip)
        slices = build_slices(geometry_at_fos(geom, slip, crit.FOS), slip,
                              n_slices)
    except ValueError:
        return None
    if not forces or not slices:
        return None
    grav = sum((s.weight + s.surcharge_force) * (s.x_mid - slip.xc)
               for s in slices)
    d_sign = 1.0 if grav >= 0 else -1.0
    t_sum = sum(f.T for f in forces)
    if t_sum <= 0:
        return None
    m_sum = sum(d_sign * support_moment(f, slip.xc, slip.yc, d_sign)
                for f in forces)
    share = m_sum / (slip.radius * t_sum)
    if share >= _SHORT_ARM_SHARE:
        return None
    arm = m_sum / t_sum
    if spencer_fn is None:
        # this circle's own Spencer value, not a search (GM-127)
        spencer = _compute_fos(geom, slip, "spencer", n_slices, tol=tol,
                               m_alpha_check=False)
    else:
        spencer = spencer_fn()
        spencer = _FOS_MAX if spencer is None else spencer
    if spencer < _FOS_MAX:
        rigorous = (f"Spencer's method, which satisfies all three, gives "
                    f"{spencer:.3f} on this circle")
    else:
        rigorous = ("Spencer's method, which satisfies all three, finds no "
                    "equilibrium solution on this circle (no lambda gives "
                    "moment and force factors of safety that agree)")
    name = "Bishop's simplified method" if method == "bishop" else \
        "The Ordinary Method of Slices"
    which = "the critical circle" if in_search else "this circle"
    return (f"{name} counts a support through its moment about the circle "
            f"centre (and its vertical component), not through horizontal "
            f"force equilibrium, which it does not satisfy (Duncan, Wright "
            f"and Brandon 2014, Eq. 6.78 and p. 93: the method neglects "
            f"what a horizontal force adds to the normal stress on the "
            f"slip surface, and care is needed where large horizontal "
            f"forces contribute; Wright & Duncan 1991). That is the method "
            f"as printed: nothing is adjusted. The {len(forces)} "
            f"support(s) crossing {which} "
            f"({t_sum:.1f} kN/m) act with an average arm of "
            f"{arm:.2f} m about its centre on a {slip.radius:.2f} m radius: "
            f"{100 * share:.0f} % of their force counts. {rigorous}. Check "
            f"the slope with method='spencer' or 'gle'.")


def no_solution_warning(result: SearchResult) -> Optional[str]:
    """Say so when the governing family left trial surfaces without a
    solution because the method did not converge on them (GM-140 step 2),
    or None. UFC 3-220-10 p. 362: "Critical failure surfaces can be missed
    as a result of non-convergence. The engineer should be able to identify
    when convergence issues are present." EM 1110-2-1902 C-10d (pp. C-37 to
    C-38) lists what to do about non-convergence. Trials with no net
    driving moment or force are counted (``no_solution``) and not warned
    of: such a surface is held, not a critical surface missed.

    Where no trial surface had a solution the search could score (no
    critical surface, so no governing family), it says that instead, with
    the counts by family, method and reason (the step 2 review's D5)."""
    from slope_stability.solution import METHOD_TEXT
    if result.critical is None and result.no_solution:
        n = sum(r["n"] for r in result.no_solution)

        def reasons(r):
            parts = [f"{r[k]} {text}" for k, text in (
                ("not_converged", "did not converge"),
                ("no_driving", "with no net driving moment or force"),
                ("with_supports", "crossed by supports, which the "
                 "Ordinary Method of Slices has no printed form for"),
                ("below_0_05", "solved below 0.05")) if r.get(k)]
            return (f"{r['family']}, "
                    f"{METHOD_TEXT.get(r['method'], r['method'])}: "
                    f"{r['n']} ({', '.join(parts)})")

        by = "; ".join(reasons(r) for r in result.no_solution)
        return (f"No trial surface had a solution by the method that the "
                f"search could score, so it reports no critical surface "
                f"and no factor of safety: {n} trial surface(s) were left "
                f"without one ({by}). Nothing stood in for the method "
                f"(UFC 3-220-10 p. 362: critical failure surfaces can be "
                f"missed as a result of non-convergence). Check with "
                f"another method (Bishop's simplified method on circles), "
                f"other search ranges, or a tension crack where the top of "
                f"the surface is in tension (EM 1110-2-1902 C-10c, "
                f"p. C-36).")
    lost = [r for r in result.no_solution
            if r["family"] == result.governing_surface
            and r.get("not_converged")]
    if not lost:
        return None
    n = sum(r["not_converged"] for r in lost)
    by = "; ".join(
        f"{METHOD_TEXT.get(r['method'], r['method'])}: {r['not_converged']}"
        for r in lost)
    return (f"The governing family ("
            f"{describe_surface_kind(result.governing_surface)}) left {n} "
            f"trial surface(s) without a solution ({by}): the method did "
            f"not converge on them, so they were not scored and nothing "
            f"stood in for the method. A lower surface among them cannot "
            f"be ruled out (UFC 3-220-10 p. 362: critical failure surfaces "
            f"can be missed as a result of non-convergence). no_solution "
            f"counts them for every family and method. Check with another "
            f"method (Bishop's simplified method on circles), or a tension "
            f"crack where the top of the surface is in tension "
            f"(EM 1110-2-1902 C-10c, p. C-36).")


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _edge_warnings(geom, crit, windows, windows_auto, grid, grid_auto,
                   ) -> List[str]:
    """Warnings when the governing search surface sits on the boundary of
    the region searched (the true minimum may lie beyond it)."""
    out = []
    if crit is None:
        return out
    x_min, x_max = geom.x_range
    width = max(x_max - x_min, 1e-9)
    for name, rng, end_x in (("entry", windows[0], crit.x_entry),
                             ("exit", windows[1], crit.x_exit)):
        if rng is None:
            continue
        # Within a tenth of the window (a grid search lands within about a
        # grid step of a constrained optimum), or 2 % of the model width.
        tol = max(0.1 * abs(rng[1] - rng[0]), 0.02 * width)
        for bound in rng:
            # A window bound on the model edge is a real limit, not a choice.
            if _near(bound, x_min, 1e-9 * width) or \
                    _near(bound, x_max, 1e-9 * width):
                continue
            if _near(end_x, bound, tol):
                kind = "automatic" if windows_auto[0 if name == "entry" else 1] \
                    else "given"
                out.append(
                    f"The critical surface's {name} point (x = {end_x:.2f}) "
                    f"lies on the edge of the {kind} x_{name}_range "
                    f"({rng[0]:.2f} to {rng[1]:.2f}); a lower factor of "
                    f"safety may lie outside it. Widen the range to check.")
                break
    if grid is not None and crit.is_circular:
        (x_lo, x_hi), (y_lo, y_hi) = grid
        dx = 1e-6 * max(x_hi - x_lo, 1.0)
        dy = 1e-6 * max(y_hi - y_lo, 1.0)
        on_edge = (_near(crit.xc, x_lo, dx) or _near(crit.xc, x_hi, dx)
                   or _near(crit.yc, y_lo, dy) or _near(crit.yc, y_hi, dy))
        if on_edge:
            kind = "automatic" if grid_auto else "given"
            out.append(
                f"The critical circle's centre ({crit.xc:.2f}, {crit.yc:.2f}) "
                f"lies on the edge of the {kind} centre grid (x {x_lo:.2f} to "
                f"{x_hi:.2f}, y {y_lo:.2f} to {y_hi:.2f}); a lower factor of "
                f"safety may lie outside it. Widen x_range / y_range to check.")
    return out


# A governing circle whose entry or exit lies within this fraction of the
# model width of an end of the ground profile is checked against circles
# that carry on past the end (GM-63) ...
_PROFILE_END_NEAR = 0.1
# ... and the warning needs one at least this much lower.
_PROFILE_END_DROP = 0.01
# A governing surface within this fraction of what the infinite slope gives
# (the planar family's FOS or a closed form), no deeper than this fraction of
# its length and lying on the cohesionless face, is itself the shallow slide
# on a c' = 0 face, and is said so.
_PLANAR_TIE = 0.02
_SHALLOW_RATIO = 0.05


def _depth_ratio(geom: SlopeGeometry, crit) -> float:
    """Greatest vertical depth of a result's surface below the ground,
    as a fraction of its horizontal length (inf when it cannot be told)."""
    span = crit.x_exit - crit.x_entry
    slip = _surface_of(crit)
    if span <= 0 or slip is None:
        return math.inf
    deepest = 0.0
    for i in range(1, 100):
        x = crit.x_entry + span * i / 100.0
        z = slip.slip_elevation_at(x)
        if z is not None:
            deepest = max(deepest, geom.ground_elevation_at(x) - z)
    return deepest / span


def _infinite_slope_match(checks, crit, fos: float):
    """(run, text) when ``crit`` lies on a cohesionless face run and its FOS
    is within ``_PLANAR_TIE`` of what the infinite slope gives there: the
    closed form for a dry face, tan(phi') / tan(beta), or with the water at
    the surface the form for the reading the run used (``seepage_reading``,
    ``_SEEPAGE_FORMS``), in a drawdown search (gamma'/gamma)
    tan(phi') / tan(beta) (Duncan, Wright & Brandon 2014, Ch. 14;
    `infinite_slope_fos`) -- the saturated face a rapid drawdown leaves
    (Pilarcitos: a circle 9 ft long and an inch deep at 1.075 against
    1.076) -- or else the planar surfaces' FOS. The closed form is named
    first, as the value to check by hand. None when nothing matches."""
    for c in checks:
        if not (crit.x_entry >= c["x_from"] - 1e-6
                and crit.x_exit <= c["x_to"] + 1e-6):
            continue
        refs = []
        refs.append((c["FOS_dry_closed_form"],
                     f"tan(phi')/tan(beta) = {c['FOS_dry_closed_form']:.3f} "
                     f"for a dry face"))
        seep = c.get("FOS_seepage_closed_form")
        if seep:
            formula, condition = _SEEPAGE_FORMS[
                c.get("seepage_reading", "rapid_drawdown")]
            refs.append((seep, f"{formula} = {seep:.3f} for {condition}"))
        if c["FOS"] is not None:
            refs.append((c["FOS"], f"the planar surfaces parallel to the "
                                   f"face score {c['FOS']:.3f}"))
        for value, text in refs:
            if value > 0 and abs(value - fos) <= _PLANAR_TIE * value:
                return c, text
    return None


def _extended_profile(geom: SlopeGeometry, length: float) -> SlopeGeometry:
    """The same model with the ground carried on level for ``length`` past
    each end of the surface profile."""
    pts = list(geom.surface_points)
    pts = ([(pts[0][0] - length, pts[0][1])] + pts
           + [(pts[-1][0] + length, pts[-1][1])])
    return dataclasses.replace(geom, surface_points=pts)


def _circle_through(p1, p2, p3) -> Optional[Tuple[float, float, float]]:
    """(xc, yc, R) of the circle through three points, None if collinear."""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9 * max(1.0, (x3 - x1) ** 2):
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    xc = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    yc = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return xc, yc, math.hypot(x1 - xc, y1 - yc)


def _profile_end_warning(geom, crit, method: str, n_slices: int, tol: float,
                         fos_fn=None) -> Optional[str]:
    """Warning when the governing surface is held back by the end of the
    ground profile (GM-63; noncircular surfaces since GM-87).

    No trial surface can enter or come out past the end of the profile, so
    a surface that would is never tried. When the governing surface ends
    near an end, circles about nine centres around its own (spacing a
    quarter of its radius) are scored on the same model with the ground
    carried on level past both ends; the warning is given only when one of
    them crosses an end of the original profile and is at least 1 % lower.
    A noncircular surface is given the centre and radius of the circle
    through its two ends and its lowest point, and the circles are scored
    with the method that scored it (Spencer when Bishop was asked for, as
    `search_critical_surface` does); a noncircular mechanism that would
    carry on past the end (along a weak layer, say) is not itself checked.
    Rigorous methods are scanned with Bishop and the best circle re-scored
    with the method itself. A rapid-drawdown search (``fos_fn``) is scored
    with its own FOS about its own centre only, as each of its surfaces
    costs a multi-stage solve.
    """
    from slope_stability.search import optimize_radius, _compute_fos, _FOS_MAX
    if crit is None:
        return None
    if getattr(crit, "is_circular", False):
        xc0, yc0, r0 = crit.xc, crit.yc, crit.radius
        what = "circle"
    elif crit.slip_points and len(crit.slip_points) >= 3:
        pts = list(crit.slip_points)
        fit = _circle_through(pts[0], min(pts[1:-1], key=lambda p: p[1]),
                              pts[-1])
        if fit is None:
            return None
        xc0, yc0, r0 = fit
        what = "surface"
        if method == "bishop":
            method = "spencer"
    else:
        return None
    x_min, x_max = geom.x_range
    width = max(x_max - x_min, 1e-9)
    if (crit.x_entry - x_min > _PROFILE_END_NEAR * width
            and x_max - crit.x_exit > _PROFILE_END_NEAR * width):
        return None
    ext = _extended_profile(geom, width)
    scan = method
    if fos_fn is None and method.lower() not in ("bishop", "fellenius"):
        scan = "bishop"
    step = 0.25 * r0
    offsets = ((0, 0),) if fos_fn is not None else \
        tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1))
    best = None
    for i, j in offsets:
        xc, yc = xc0 + i * step, yc0 + j * step
        r, f = optimize_radius(ext, xc, yc, method=scan,
                               n_slices=n_slices, tol=tol, fos_fn=fos_fn)
        if f < _FOS_MAX and (best is None or f < best[0]):
            best = (f, xc, yc, r)
    if best is None:
        return None
    f, xc, yc, r = best
    from slope_stability.search import _governing_circle
    slip = _governing_circle(ext, xc, yc, r, scan, n_slices, tol, fos_fn)
    try:
        x_in, x_out = slip.find_entry_exit(ext)
    except ValueError:
        return None
    eps = 1e-6 * width
    past = []
    if x_in < x_min - eps:
        past.append(f"the left end (x = {x_min:.2f})")
    if x_out > x_max + eps:
        past.append(f"the right end (x = {x_max:.2f})")
    if not past:
        return None
    if scan != method:
        f = _compute_fos(ext, slip, method, n_slices, tol=tol)
        if f >= _FOS_MAX:
            return None
    if f > (1.0 - _PROFILE_END_DROP) * crit.FOS:
        return None
    return (
        f"The critical {what} ends near the end of the ground surface given "
        f"(x = {x_min:.2f} to {x_max:.2f}), and no trial surface can pass "
        f"the end of the profile. With the ground carried on level past "
        f"{' and '.join(past)}, a circle crossing it gives FOS {f:.3f} "
        f"against the reported {crit.FOS:.3f}. If the ground continues "
        f"beyond the profile, extend it and search again. Undrained "
        f"(phi = 0) slopes over a deep foundation reach farthest: the "
        f"critical circle passes below the toe, tangent to the firm base, "
        f"and comes out about 1.2 to 1.4 times the foundation's depth below "
        f"the toe beyond it (Taylor's nH; FHWA NHI-06-088 Fig. 6-15).")


def _families_scored(result: SearchResult, search_kind: str,
                     candidates: Dict[str, float], n_main: int, toe, planar,
                     *, windows_given: bool, time_up: bool) -> None:
    """Set ``result.families_scored`` (the families run that gave an
    admissible surface, the ones in ``candidates``) and
    ``result.families_not_scored`` (each family run that gave none, and
    why): what the search's choices say it scored (GM-125)."""
    held_none = "none of the surfaces tried held"
    why = {}
    if search_kind not in candidates:
        why[search_kind] = (held_none if n_main else
                            "the time budget ran out first" if time_up
                            else "no surface was tried")
    if toe is not None:
        tried = toe.get("n_scored") or {}
        for family in (TOE_CIRCLE, ABOVE_TOE):
            if family in candidates:
                continue
            why[family] = (held_none if tried.get(family) else
                           "the time budget ran out first" if time_up else
                           "none within the windows and ranges given"
                           if windows_given else "none on this slope")
    if planar is not None and INFINITE_SLOPE not in candidates:
        why[INFINITE_SLOPE] = (
            held_none if planar["n_evaluated"] else
            "not evaluated on the faces of c' = 0 soil "
            "(infinite_slope_checks)" if planar["checks"] else
            "no face of c' = 0 soil")
    runs = [search_kind] + ([TOE_CIRCLE, ABOVE_TOE] if toe is not None
                            else []) + ([INFINITE_SLOPE] if planar is not None
                                        else [])
    result.families_scored = [k for k in runs if k not in why]
    result.families_not_scored = why


def apply_supplements(geom: SlopeGeometry,
                      result: SearchResult,
                      *,
                      search_kind: str,
                      method: str,
                      n_slices: int,
                      tol: float,
                      x_entry_range=None, x_exit_range=None,
                      x_range=None, y_range=None,
                      windows_used=(None, None), windows_auto=(True, True),
                      grid_used=None, grid_auto=True,
                      toe_circles: bool = True,
                      infinite_slope: bool = True,
                      fos_fn=None, auto_ranges=None) -> SearchResult:
    """Score the toe-circle and infinite-slope families and fold them into
    ``result`` (modified in place and returned).

    ``x_entry_range`` / ``x_exit_range`` / ``x_range`` / ``y_range`` are the
    CALLER-GIVEN constraints (None where the search chose its own).
    ``windows_used`` / ``grid_used`` are what the main search actually used,
    for the edge warnings and ``result.search_ranges``; ``auto_ranges`` (a
    `search_ranges.AutoRanges`) says how the automatic ones were set and
    where the model's end cut them. ``toe_circles`` applies to circular
    searches only. ``fos_fn`` (the rapid-drawdown search's per-surface FOS)
    scores both families when given; the planar surfaces are then labelled
    with ``method`` instead of Janbu.
    """
    from slope_stability.search import _FOS_MAX, _time_up
    planar_method = PLANAR_METHOD if fos_fn is None else method
    n_main = result.n_surfaces_evaluated

    main = result.critical
    candidates: Dict[str, float] = {}
    if main is not None and main.FOS < _FOS_MAX:
        # the solver's 999.9 marker is not an FOS (GM-104)
        candidates[search_kind] = round(main.FOS, 4)
    best_kind, best = search_kind, main
    best_fos = main.FOS if main is not None else _FOS_MAX
    warnings_out: List[str] = []

    toe = None
    if toe_circles and search_kind == CIRCULAR:
        toe = search_toe_circles(
            geom, method=method, n_slices=n_slices, tol=tol,
            x_entry_range=x_entry_range, x_exit_range=x_exit_range,
            x_range=x_range, y_range=y_range, fos_fn=fos_fn)
        result.n_surfaces_evaluated += toe["n_evaluated"]
        if toe["FOS"] < _FOS_MAX:
            candidates[TOE_CIRCLE] = round(toe["FOS"], 4)
            if toe["FOS"] < best_fos:
                r = _make_result(
                    geom, CircularSlipSurface(toe["xc"], toe["yc"],
                                              toe["radius"],
                                              *(toe.get("arc") or ())),
                    toe["FOS"], method.capitalize(), n_slices)
                if r is not None:
                    best_kind, best, best_fos = TOE_CIRCLE, r, toe["FOS"]
        above = toe.get("above_toe") or {}
        if above.get("FOS", _FOS_MAX) < _FOS_MAX:
            candidates[ABOVE_TOE] = round(above["FOS"], 4)
            if above["FOS"] < best_fos:
                r = _make_result(
                    geom, CircularSlipSurface(above["xc"], above["yc"],
                                              above["radius"],
                                              *(above.get("arc") or ())),
                    above["FOS"], method.capitalize(), n_slices)
                if r is not None:
                    best_kind, best, best_fos = ABOVE_TOE, r, above["FOS"]

    planar = None
    if infinite_slope:
        planar = check_infinite_slope(
            geom, method=planar_method, n_slices=n_slices, tol=tol,
            x_entry_range=x_entry_range, x_exit_range=x_exit_range,
            fos_fn=fos_fn)
        result.n_surfaces_evaluated += planar["n_evaluated"]
        result.infinite_slope_checks = planar["checks"]
        if planar["FOS"] < _FOS_MAX:
            candidates[INFINITE_SLOPE] = round(planar["FOS"], 4)
            if planar["FOS"] < best_fos:
                r = _make_result(geom, planar["slip"], planar["FOS"],
                                 planar_method.capitalize(), n_slices)
                if r is not None:
                    best_kind, best, best_fos = INFINITE_SLOPE, r, planar["FOS"]

    _families_scored(result, search_kind, candidates, n_main, toe, planar,
                     windows_given=any(r is not None for r in (
                         x_entry_range, x_exit_range, x_range, y_range)),
                     time_up=_time_up())

    main_fos = main.FOS if main is not None else None
    main_text = describe_surface_kind(search_kind)
    infinite_slope_found = False
    if best_kind == TOE_CIRCLE:
        xt, zt = toe["toe"]
        other = (best.x_exit if abs(best.x_entry - xt) <= abs(best.x_exit - xt)
                 else best.x_entry)
        was = (f"{main_fos:.3f}" if main_fos is not None
               else "no admissible surface")
        warnings_out.append(
            f"A toe circle governs: the circle through the toe at "
            f"(x = {xt:.2f}, z = {zt:.2f}) and the ground at x = {other:.2f} "
            f"gives FOS {best_fos:.3f}, below the {main_text} minimum "
            f"({was}). The main search's "
            + ("entry/exit windows, centre grid or grid spacing"
               if grid_used is not None else
               "entry/exit windows or the spacing of its entry/exit pairs")
            + " kept it from this circle; the toe circle is reported.")
    elif best_kind == ABOVE_TOE:
        above = toe["above_toe"]
        xs, zs = above["start"]
        other = (best.x_exit if abs(best.x_entry - xs) <= abs(best.x_exit - xs)
                 else best.x_entry)
        was = (f"{main_fos:.3f}" if main_fos is not None
               else "no admissible surface")
        # (the family keeps to caller-given windows, so a circle outside
        # the windows used is outside an automatic one)
        inside = all(w is None or w[0] - 1e-6 <= v <= w[1] + 1e-6
                     for w, v in zip(windows_used,
                                     (best.x_entry, best.x_exit)))
        kept = (("its centre grid or radii missed this circle"
                 if grid_used is not None else
                 "its entry/exit pairs missed this circle") if inside else
                "the circle ends outside its automatic entry/exit windows, "
                "which meet at the middle of the face")
        xt, zt = above["face_toe"]
        warnings_out.append(
            f"A circle passing above the toe governs: the circle from the "
            f"face at (x = {xs:.2f}, z = {zs:.2f}), {zs - zt:.2f} m above "
            f"the toe at x = {xt:.2f}, to the ground at x = {other:.2f} "
            f"gives FOS {best_fos:.3f}, below the {main_text} minimum "
            f"({was}). The main search did not reach it: {kept}. "
            f"The circle passing above the toe is reported (FHWA NHI-06-088 "
            f"p. 6-38).")
    elif best_kind == INFINITE_SLOPE:
        run = min((c for c in planar["checks"] if c["FOS"] is not None),
                  key=lambda c: c["FOS"])
        was = (f"{main_fos:.3f}" if main_fos is not None
               else "no admissible surface")
        warnings_out.append(
            f"Cohesionless soil (c' = 0, phi' = {run['phi_deg']:g} deg) "
            f"forms the slope face between x = {run['x_from']:.2f} and "
            f"{run['x_to']:.2f} (beta = {run['slope_angle_deg']:.1f} deg): "
            f"the critical failure is a shallow planar slide parallel to the "
            f"face (infinite slope; tan(phi')/tan(beta) = "
            f"{run['FOS_dry_closed_form']:.3f} for a dry face). The reported "
            f"surface is a planar slide at a shallow depth with FOS "
            f"{best_fos:.3f}, "
            + ("scored by force equilibrium (Janbu's simplified method, as "
               "the infinite slope is) whatever method the search used"
               if fos_fn is None else
               "scored with the search's own per-surface solve")
            + f"; the {main_text} minimum was {was}. Shallower slides tend "
            f"to this value, so it does not depend on the search grid.")
    elif (planar is not None and best is not None
          and _depth_ratio(geom, best) <= _SHALLOW_RATIO):
        # The governing surface may itself be the shallow slide on a c' = 0
        # face (GM-57: the windows set from the slope let the main search
        # reach it). Only a slide that is shallow for its length, lies on
        # the cohesionless face and scores what that mechanism gives is
        # called so (a deep circle can score about the same and is not).
        match = _infinite_slope_match(planar["checks"], best, best_fos)
        if match is not None:
            run, ref = match
            infinite_slope_found = True
            warnings_out.append(
                f"The governing surface (FOS {best_fos:.3f}) is a shallow "
                f"slide on the face of cohesionless soil (c' = 0, phi' = "
                f"{run['phi_deg']:g} deg) between x = {run['x_from']:.2f} "
                f"and {run['x_to']:.2f} (beta = "
                f"{run['slope_angle_deg']:.1f} deg): the infinite slope "
                f"mechanism; {ref}. Shallower slides tend to this value, "
                f"so it does not depend on the search grid.")

    if planar is not None:
        for c in planar["checks"]:
            if (c["FOS"] is None and c["note"] != HELD_NOTE
                    and c["FOS_dry_closed_form"] < best_fos):
                warnings_out.append(
                    f"Cohesionless soil (c' = 0, phi' = {c['phi_deg']:g} deg) "
                    f"forms the slope face between x = {c['x_from']:.2f} and "
                    f"{c['x_to']:.2f}, where the dry infinite-slope factor of "
                    f"safety tan(phi')/tan(beta) = "
                    f"{c['FOS_dry_closed_form']:.3f} is below the reported "
                    f"{best_fos:.3f}, but no shallow planar surface was "
                    f"evaluated there ({c['note']}). The infinite slope may "
                    f"govern.")

    # (A main-search surface that is the infinite-slope mechanism gets no
    # edge warning: shallower slides only tend to the same value.)
    if best_kind in (CIRCULAR, NONCIRCULAR) and not infinite_slope_found:
        warnings_out.extend(_edge_warnings(
            geom, best, windows_used, windows_auto,
            grid_used if search_kind == CIRCULAR else None, grid_auto))
    elif best_kind in (TOE_CIRCLE, ABOVE_TOE):
        # Only the caller's own windows bound the toe-circle family.
        given = (x_entry_range, x_exit_range)
        warnings_out.extend(_edge_warnings(
            geom, best, given, (False, False), None, grid_auto))

    # The lowest surface the search found but refused, its solution not
    # holding (GM-57 review): say so, so the user can judge.
    ref = getattr(result, "refused_lowest", None)
    if ref is not None and (best is None or ref["FOS"] < best_fos):
        tail = ("The reported surface is the lowest that held."
                if best is not None else
                "No other surface held, so none is reported.")
        warnings_out.append(
            f"The lowest surface found, FOS {ref['FOS']:.3f} (x = "
            f"{ref['x_entry']:.2f} to {ref['x_exit']:.2f}), did not hold: "
            f"{ref['reason']}, so it is not an equilibrium solution of the "
            f"surface and is not reported. {tail}")

    # A governing circle held back by the end of the ground profile (GM-63);
    # failing a lower circle past the end, a main-search surface that ends
    # where the model's end cut an automatic range short (GM-57)
    if best is not None and best_kind != INFINITE_SLOPE:
        w = _profile_end_warning(geom, best, method, n_slices, tol,
                                 fos_fn=fos_fn)
        if not w and best_kind in (CIRCULAR, NONCIRCULAR):
            w = _clipped_range_warning(geom, best, windows_used,
                                       windows_auto, auto_ranges)
        if w:
            warnings_out.append(w)

    # The governing circle closed by a crack at its own centre's level, and
    # supports a moment-only method counts little of (GM-59).
    if best is not None and best_kind != INFINITE_SLOPE:
        w = _crack_statement(geom, best, method, n_slices, tol, fos_fn)
        if w:
            warnings_out.append(w)
        if fos_fn is None:
            w = _support_arm_warning(geom, best, method, n_slices, tol)
            if w:
                warnings_out.append(w)

    _attach_reinforcement(geom, best)
    result.critical = best
    result.governing_surface = best_kind if best is not None else ""
    result.candidates = candidates
    result.warnings = warnings_out
    result.search_ranges = _ranges_record(
        windows_used, grid_used if search_kind == CIRCULAR else None,
        dict(x_entry_range=x_entry_range, x_exit_range=x_exit_range,
             x_range=x_range, y_range=y_range), auto_ranges)
    if toe is not None and toe.get("faces"):
        # What the toe-circle families covered: the faces, and the heights
        # above each toe (fractions of the face height) their circles'
        # lower ends started from; the best start's height is refined
        # between its neighbours unless a per-surface solve was given
        rr = lambda p: [round(p[0], 3), round(p[1], 3)]   # noqa: E731
        result.search_ranges["faces_covered"] = [
            {"toe": rr(f["toe"]), "crest": rr(f["crest"]),
             "lower_end_heights_above_toe": [0.0] + f["heights"],
             "height_refined": fos_fn is None}
            for f in toe["faces"]]
    result.governing_location = _locate(geom, best)
    return result


#: The main search of each ``surface_type`` (search_critical_surface), as a
#: result's ``choices`` names it.
SEARCH_TEXT = {
    "circular": "circular, centre grid",
    "entry_exit": "circular, entry and exit windows",
    "noncircular": "noncircular, random polylines",
    "noncircular_de": "noncircular, differential evolution",
    "pso": "noncircular, particle swarm",
    "weak_layer": "noncircular, drawn toward the weakest layers",
}

#: How to ask for a surface family a search did not try.
_TRY_FAMILY = {
    CIRCULAR: "a circular search (surface_type 'circular')",
    NONCIRCULAR: "a noncircular search (surface_type 'noncircular_de')",
    TOE_CIRCLE: "a circular search",
    ABOVE_TOE: "a circular search",
    INFINITE_SLOPE: "a search with it (infinite_slope=True)",
}

#: Each surface family and each automatic range in plain words.
_FAMILY_TEXT = {CIRCULAR: "circular", NONCIRCULAR: "noncircular",
                TOE_CIRCLE: "toe circle",
                ABOVE_TOE: "circle passing above the toe",
                INFINITE_SLOPE: "infinite slope"}
_RANGE_TEXT = {"x_entry_range": "entry window", "x_exit_range": "exit window",
               "x_range": "centre x range", "y_range": "centre height range"}


def note_budget(result: SearchResult, budget, *, method: str = "",
                screened: bool = False, can_screen: bool = False) -> None:
    """Say in ``result`` what a search cut short by its time budget covered
    (GM-123): a warning first in ``warnings`` and
    ``search_ranges['time_budget']``. Nothing when the search finished."""
    if budget is None or not budget.stopped:
        return
    parts = [f"Cut short by its time budget: the reported FOS may be too "
             f"high (unconservative), since a lower surface may lie in what "
             f"the search did not cover. The budget was {budget.seconds:g} s "
             f"(the main search may use 80 % of it); the search ran "
             f"{budget.elapsed():.0f} s in all."]
    for st in budget.stops:
        text = (f" The {st['stage']} stopped after {st['scored']} of "
                f"{st['of']} {st['unit']} in {st['time_s']:.0f} s")
        if st["projected_s"] is not None:
            text += (f"; all {st['of']} would take about "
                     f"{st['projected_s']:.0f} s")
        text += f". Covered: {st['covered']}."
        if st["not_covered"]:
            text += f" Not scored: {st['not_covered']}."
        parts.append(text)
    if budget.not_run:
        parts.append(" Not run: " + "; ".join(budget.not_run) + ".")
    from slope_stability.search import TIME_BUDGET_MAX_S
    projected = max((st["projected_s"] or 0 for st in budget.stops),
                    default=0)
    raise_it = f"raise time_budget_s (at most {TIME_BUDGET_MAX_S:g} s)"
    if projected > TIME_BUDGET_MAX_S:
        raise_it = (f"raise time_budget_s (at most {TIME_BUDGET_MAX_S:g} s, "
                    f"which alone cannot cover this search: about "
                    f"{projected:.0f} s projected)")
    levers = [raise_it,
              "use fewer points (nx, ny) or trials, or fewer slices",
              "narrow the windows"]
    if (can_screen and not screened
            and method.lower() in _SCREENED_METHODS):
        levers.insert(0, "screen it with Bishop's simplified method "
                         "(bishop_screen: every circle scored with Bishop, "
                         f"the lowest re-scored with {method}), which covers "
                         "the whole range in a fraction of the time")
    parts.append(" To cover it all, any of: " + "; ".join(levers) + ".")
    result.warnings.insert(0, "".join(parts))
    result.search_ranges["time_budget"] = {
        "time_budget_s": budget.seconds,
        "elapsed_s": round(budget.elapsed(), 1),
        "stopped": [dict(st) for st in budget.stops],
        "not_run": list(budget.not_run)}


def m_alpha_confirmation(geom: SlopeGeometry, crit, method: str,
                         n_slices: int, tol: float = 1e-4
                         ) -> Optional[Dict[str, Any]]:
    """The governing surface's FOS and least m_alpha at the search's slice
    count and at twice that count (GM-127): two solves of one surface. The
    check turns on the slices at the surface's steep end, and a FOS that
    moves with the slicing is worth saying. ``refused_fine`` says why there
    is no FOS on twice the slices: "geometry" (the finer slicing cannot be
    built, ``why_fine`` gives the reason; a circle touching the model's
    base is kept at every count since GM-128) or "no_solution"
    (the method does not converge there). None for the Ordinary Method of
    Slices (no m_alpha) or no governing surface."""
    from slope_stability.search import (
        _compute_fos, _FOS_MAX, _new_reject_stats)
    from slope_stability.slices import build_slices
    if crit is None or method == "fellenius":
        return None
    if getattr(crit, "is_circular", False):
        slip = circle_of(crit)
    elif crit.slip_points and len(crit.slip_points) >= 2:
        slip = PolylineSlipSurface(points=[tuple(p) for p in crit.slip_points])
    else:
        return None
    out: Dict[str, Any] = {"n_slices": n_slices, "n_slices_fine": 2 * n_slices}
    for key, n in (("", n_slices), ("_fine", 2 * n_slices)):
        info: Dict[str, Any] = {}
        refused = why = None
        try:
            from slope_stability.cracks import geometry_at_fos
            if len(build_slices(geometry_at_fos(geom, slip), slip, n)) < 3:
                refused, why = "geometry", "fewer than three slices"
        except ValueError as exc:
            refused, why = "geometry", str(exc)
        f = _FOS_MAX
        if refused is None:
            stats = _new_reject_stats()
            f = _compute_fos(geom, slip, method, n, tol=tol,
                             m_alpha_check=False, info=info,
                             reject_stats=stats)
            if f >= _FOS_MAX:
                refused = ("geometry" if stats["geometry"] or stats["jagged"]
                           else "no_solution")
        out["fos" + key] = None if f >= _FOS_MAX else f
        out["m_alpha_min" + key] = info.get("m_alpha_min")
        out["refused" + key] = refused
        out["why" + key] = why
    return out


# ---------------------------------------------------------------------------
# The printed acceptability checks in a search (GM-140 step 3)
# ---------------------------------------------------------------------------

def _method_key(label: str, default: str) -> str:
    """The method a result's label names ('Bishop', 'Morgenstern-Price',
    'Gle', ...), as `solution.METHODS` spells it."""
    from slope_stability.solution import METHODS
    key = (label or "").lower().replace("-", "_").replace(" ", "_")
    return key if key in METHODS else default


def critical_acceptability(geom: SlopeGeometry, result: SearchResult,
                           scored: str, n_slices: int, tol: float = 1e-4,
                           m_alpha_rule: str = "frictional") -> None:
    """Put on a search's result what the printed acceptability checks find
    (GM-140 step 3, item 4):

    * ``critical.acceptability`` and ``critical_flags``: the findings on
      the governing surface, solved again by the method that scored it at
      the search's slice count (which check, which slice, what value);
    * for a Bishop circle on which the Ordinary Method of Slices value was
      taken (Duncan 1996, p. 580), both values and ``method_used`` on
      ``critical``;
    * ``lowest_unflagged``: when the governing surface carries a flag, the
      lowest factor of safety among the surfaces the search scored by the
      same method that carry none, with the family and the surface; or the
      statement that it scored none.

    The headline stays the lowest solved surface."""
    from slope_stability.acceptability import surface_findings, flags
    from slope_stability.search import trial_reinforcement
    from slope_stability.slices import build_slices
    from slope_stability.solution import solve_method
    crit = result.critical
    if crit is None or crit.FOS is None:
        return
    method = _method_key(crit.method, scored)
    try:
        slip = _surface_of(crit)
        if slip is None:
            return
        slices = build_slices(geom, slip, n_slices)
        sol = solve_method(slices, slip, method, "half_sine", tol,
                           trial_reinforcement(geom, slip))
    except (ValueError, ZeroDivisionError, RuntimeError):
        return
    if not sol.solved:
        return
    crit.acceptability = surface_findings(geom, slip, slices, sol,
                                          rule=m_alpha_rule)
    if method == "bishop":
        crit.FOS_bishop = sol.fos_bishop
        crit.FOS_fellenius = sol.fos_oms
        if sol.oms_taken:
            crit.method_used = ("Ordinary Method of Slices value (Bishop's "
                                "value is below it; Duncan 1996, p. 580)")
    result.critical_flags = flags(crit.acceptability)
    if not result.critical_flags:
        return
    best = None
    for rec in result.trial_flags:
        low = rec.get("lowest_unflagged")
        if rec["method"] != method or not low:
            continue
        if best is None or low["FOS"] < best["FOS"]:
            best = dict(low, family=rec["family"], method=method)
    n_scored = sum(r["n_scored"] for r in result.trial_flags
                   if r["method"] == method)
    if best is None:
        result.lowest_unflagged = {
            "FOS": None, "method": method,
            "note": (f"none of the {n_scored} surfaces the search scored "
                     f"with this method is without a flag")}
    else:
        result.lowest_unflagged = best


def report_acceptability(geom: SlopeGeometry, result: SearchResult,
                         scored: str, n_slices: int, tol: float = 1e-4,
                         m_alpha_rule: str = "frictional",
                         family: Optional[str] = None) -> None:
    """The one place a search's result gets the printed acceptability
    checks' report (GM-140 step 3, item 4; its review's defect 2): the
    governing surface's findings and flags, the lowest scored surface with
    no flag beside it (`critical_acceptability`), the warning when the
    governing surface carries a flag (`acceptability_warning`), and a
    comparison with the Ordinary Method of Slices value that was not made
    on the governing circle, and why. ``family`` names the search's own
    trials in ``trial_flags`` (a search called on its own counts them as
    'main')."""
    if family is not None:
        for rec in result.trial_flags:
            if rec.get("family") == "main":
                rec["family"] = family
    critical_acceptability(geom, result, scored, n_slices, tol,
                           m_alpha_rule=m_alpha_rule)
    w = acceptability_warning(result)
    if w:
        result.warnings.append(w)
    if result.critical is not None:
        result.warnings.extend(
            f["text"] for f in (result.critical.acceptability or [])
            if f["check"] == "bishop_against_oms" and not f.get("compared"))


def headline_report(result: SearchResult, low_m_alpha: str = "flag"
                    ) -> Dict[str, Any]:
    """What the agent's tool result says of the headline in one sentence
    (``summary``) and, when the headline carries a flag, the option that
    leaves flagged surfaces out of the search (``refuse_option``): the
    headline (the lowest solved surface), its flags, and the lowest scored
    surface with no flag, or that none was found. Neither number alone is
    safe, so a flagged headline is never given alone (GM-140 step 3,
    item 4; its review's ruling 5)."""
    from slope_stability.acceptability import flags_text
    crit = result.critical
    if crit is None or crit.FOS is None:
        return {"summary": ("No surface held, so the search reports no "
                            "factor of safety; choices and warnings say "
                            "why.")}
    where = (_FAMILY_TEXT.get(result.governing_surface,
                              result.governing_surface)
             or ("circular" if getattr(crit, "is_circular", True)
                 else "noncircular"))
    head = (f"The lowest solved surface has FOS {crit.FOS:.3f} ({where}, "
            f"{crit.method})")
    names = list(result.critical_flags or [])
    if not names:
        return {"summary": head + " and meets every printed acceptability "
                                  "check made on it."}
    others = [n for n in names if n != "low_m_alpha"]
    if "low_m_alpha" in names and low_m_alpha != "refuse":
        option = ("low_m_alpha='refuse' leaves surfaces with m_alpha below "
                  "0.2 out of the search")
        if others:
            option += (f"; no option leaves out surfaces with the other "
                       f"flags ({flags_text(others)})")
    else:
        option = ("no option leaves surfaces with these flags out of the "
                  "search (low_m_alpha='refuse' leaves out only surfaces "
                  "with m_alpha below 0.2)")
    low = result.lowest_unflagged or {}
    if low.get("FOS") is not None:
        beside = (f"the lowest surface the search scored with no flag has "
                  f"FOS {low['FOS']:.3f}")
    else:
        beside = ("none found: no surface the search scored with this "
                  "method is without a flag")
    return {"summary": (f"{head} and is flagged ({flags_text(names)}); "
                        f"{beside}; {option}; give the user both."),
            "refuse_option": option}


def acceptability_warning(result: SearchResult) -> Optional[str]:
    """What a search says when its governing surface carries a flag of the
    printed acceptability checks: the flags, that the surface is kept, the
    lowest surface without a flag beside it, and where the detail is."""
    from slope_stability.acceptability import flags_text
    names = result.critical_flags
    if not names or result.critical is None:
        return None
    low = result.lowest_unflagged or {}
    text = (f"The governing surface (FOS {result.critical.FOS:.3f}) does "
            f"not meet {len(names)} of the printed acceptability checks: "
            f"{flags_text(names)}. It is kept and reported, flagged; "
            f"critical.acceptability names the slices and the values. ")
    if low.get("FOS") is not None:
        text += (f"The lowest surface the search scored that carries no "
                 f"flag has FOS {low['FOS']:.3f} ("
                 f"{_FAMILY_TEXT.get(low.get('family'), low.get('family'))}"
                 f"; lowest_unflagged). ")
    elif low.get("note"):
        text += low["note"][0].upper() + low["note"][1:] + ". "
    text += ("Examine the surface before using its factor of safety "
             "(Whitman and Bailey 1967, p. 487; EM 1110-2-1902 C-10a, "
             "pp. C-33 to C-34).")
    if "low_m_alpha" in names:
        rem = next((f.get("remedies") for f in result.critical.acceptability
                    if f["check"] == "low_m_alpha"), "")
        text += ((" " + rem if rem else "")
                 + " low_m_alpha='refuse' leaves surfaces with m_alpha "
                   "below 0.2 out of the search instead.")
    return text


def acceptability_choices(result: SearchResult, method: Optional[str],
                          low_m_alpha: str, negative_friction_term: str
                          ) -> List[Dict[str, Any]]:
    """The ``choices`` entries of the printed acceptability checks in a
    search: how flagged surfaces were treated, the Bishop against Ordinary
    Method of Slices comparison, the negative frictional term, and the
    noncircular screens (the module's own)."""
    from geotech_common.choices import MODULE, choice, how_given
    from slope_stability.acceptability import CHECK_TEXT
    out = []
    n_scored = sum(r["n_scored"] for r in result.trial_flags)
    n_flagged = sum(r["n_flagged"] for r in result.trial_flags)
    by = {}
    for r in result.trial_flags:
        for c, n in r["by_check"].items():
            by[c] = by.get(c, 0) + n
    counts = "; ".join(f"{CHECK_TEXT.get(c, c)}: {n}" for c, n in by.items())
    out.append(choice(
        "surfaces that do not meet a printed acceptability check",
        f"kept and flagged: {n_flagged} of {n_scored} scored trials"
        + (f" ({counts})" if counts else ""), MODULE,
        "every solved surface that is one sliding mass is scored; the "
        "checks' findings are reported, not acted on (Whitman and Bailey "
        "1967, p. 487: the user is warned and must examine the result; EM "
        "1110-2-1902 C-10a, pp. C-33 to C-34: forces examined, no tension "
        "on slice sides or bases, the line of thrust within the mass). "
        "trial_flags counts them by family; critical_flags and "
        "lowest_unflagged say what the governing surface carries and the "
        "lowest surface without a flag"))
    if (method or "").lower() == "bishop":
        n_taken = by.get("bishop_below_oms", 0)
        out.append(choice(
            "Bishop's value against the Ordinary Method of Slices value",
            f"the Ordinary Method of Slices value taken on {n_taken} "
            f"scored circle(s)" if n_taken else
            "computed for every circle; Bishop's value was never the lower",
            MODULE,
            "Duncan (1996, p. 580): calculate the Ordinary Method of "
            "Slices value for each circle when Bishop's method is used; "
            "where Bishop's is below it, the Ordinary Method of Slices "
            "value is the better answer. Not made on circles with "
            "supports or under a seismic coefficient (Duncan, Wright and "
            "Brandon 2014, p. 93, print no Ordinary Method of Slices "
            "equation with known forces)"))
    if (method or "").lower() in ("bishop", "fellenius"):
        out.append(choice(
            "frictional term where the pore-water force exceeds the weight",
            "as printed, negative" if negative_friction_term == "as_printed"
            else "set to zero",
            how_given(negative_friction_term, "as_printed",
                      name="negative_friction_term"),
            "Bishop (1955) Eqs. 9 and 13, p. 10 write W cos(alpha) - u l "
            "and W - u b with no lower limit; Whitman and Bailey (1967, "
            "p. 487) ask for a warning, and the result names the slices",
            ["set to zero where negative, the module's earlier rule, no "
             "printed source (negative_friction_term='zero')"
             if negative_friction_term == "as_printed" else
             "as printed (negative_friction_term='as_printed', the "
             "default)"]))
    if result.screen_refusals:
        text = {"slice_count": "too few slices",
                "span": "too short a span",
                "jagged_below_1_5": "FOS below 1.5 on a jagged surface "
                                    "(Janbu's simplified method, the "
                                    "Ordinary Method of Slices, or a "
                                    "caller's own per-surface solve)"}
        parts = [f"{text.get(k, k)}: {rec['n']}"
                 for k, rec in result.screen_refusals.items()]
        out.append(choice(
            "noncircular trial surfaces refused by the module's screens",
            "; ".join(parts), MODULE,
            "numerical stand-ins of the module's own, not found in print "
            "(EM 1110-2-1902 C-10a prints the checks on forces and the "
            "line of thrust instead), applied before any solve. The span "
            "screen is kept because it refuses surfaces the printed checks "
            "pass (short surfaces confined to part of the window); each "
            "refusal is counted here (screen_refusals). The screens on "
            "agreement between moment axes and between slicings were "
            "removed in GM-140 step 3, and the screen on a FOS below 1.5 "
            "on a jagged surface for Spencer, Morgenstern-Price and GLE: "
            "every surface they refused in the measured Spencer searches "
            "carried a flag of the printed checks, and is now scored, kept "
            "and flagged. The jagged screen stays for Janbu's simplified "
            "method and the Ordinary Method of Slices, which no check of "
            "C-10a covers, so a jagged surface it refuses there would carry "
            "no flag (ACADS 1(a), 200 random trials: 29 Janbu and 42 "
            "Ordinary Method of Slices surfaces), and for a caller's own "
            "per-surface solve (rapid drawdown)"))
    return out


_SLIDE2_CHECK = ("Slide2 reports such a surface as invalid with its 'Check "
                 "malpha < 0.2' setting on (its error -112)")


def m_alpha_choice(n_left_out: int, method: Optional[str],
                   confirm: Optional[Dict[str, Any]] = None,
                   rule: str = "frictional", action: str = "refuse",
                   n_flagged: int = 0) -> Dict[str, Any]:
    """The ``choices`` entry for the surfaces with m_alpha < 0.2 on a
    slice: kept and flagged (``action`` 'flag', the default of a search
    since GM-140 step 3; ``n_flagged`` of the scored trials) or left out
    (``action`` 'refuse', GM-127; ``n_left_out``), which slices the check
    looked at (``rule``, `methods.M_ALPHA_RULES`) and the governing surface
    on twice the slices."""
    from geotech_common.choices import choice, how_given
    how = how_given(rule, "frictional", name="m_alpha_rule")
    if action == "flag" and (method or "").lower() != "fellenius":
        return _m_alpha_flag_choice(method, confirm, rule, n_flagged)
    if (method or "").lower() == "fellenius":
        return choice(
            "slip surfaces left out for m_alpha", "none", how,
            "the Ordinary Method of Slices has no m_alpha (its base normal "
            "force is W cos(a)), so no surface is left out for it")
    where = ("any slice" if rule == "every_slice"
             else "a slice with phi' > 0")
    basis = ("m_alpha = cos(a) (1 + tan(a) tan(phi) / F), at the method's "
             "own F, is the denominator of the normal force on a slice "
             "base; below 0.2 the factor of safety may be incorrect or "
             "misleading (Whitman & Bailey 1967), and " + _SLIDE2_CHECK
             + ", circular or not")
    if rule == "every_slice":
        basis += "; every slice is checked, as Slide2 checks"
    else:
        basis += ("; slices with phi' = 0 are not checked: there m_alpha = "
                  "cos(a) does not depend on F and the shear strength is "
                  "c l whatever the normal force, so leaving such a surface "
                  "out would only raise the minimum")
    c = confirm or {}
    if c.get("fos") is not None:
        basis += (f"; on {c['n_slices']} slices the governing surface's FOS "
                  f"is {c['fos']:.3f}"
                  + (f" and its least m_alpha {c['m_alpha_min']:.2f}"
                     if c.get("m_alpha_min") is not None else ""))
        if c.get("fos_fine") is not None:
            basis += (f"; on {c['n_slices_fine']} slices, {c['fos_fine']:.3f}"
                      + (f" and {c['m_alpha_min_fine']:.2f}"
                         if c.get("m_alpha_min_fine") is not None else ""))
        elif c.get("refused_fine") == "geometry":
            basis += (f"; on {c['n_slices_fine']} slices it is not "
                      f"admissible ({c.get('why_fine') or 'its geometry'})")
        else:
            basis += (f"; on {c['n_slices_fine']} slices its method has no "
                      f"solution")
    alternatives = ["the FOS of one such surface, with a warning: "
                    "analyze_slope on that surface"]
    alternatives.append(
        "slices with phi' > 0 only (m_alpha_rule='frictional', the default)"
        if rule == "every_slice" else
        "every slice, phi' = 0 ones included, as Slide2 checks "
        "(m_alpha_rule='every_slice')")
    return choice(
        "slip surfaces left out for m_alpha",
        (f"{n_left_out} left out, with m_alpha < 0.2 on {where}"
         if n_left_out else f"none (m_alpha < 0.2 on {where} leaves a "
                            f"surface out)"),
        how, basis, alternatives)


def _m_alpha_flag_choice(method, confirm, rule: str,
                         n_flagged: int) -> Dict[str, Any]:
    """`m_alpha_choice` for a search that keeps and flags (OWN-102)."""
    from geotech_common.choices import choice, how_given
    where = ("any slice" if rule == "every_slice"
             else "a slice with phi' > 0")
    basis = ("m_alpha = cos(a) + sin(a) tan(phi') / F, at the method's own "
             "F, is the denominator of the normal force on a slice base. "
             "Whitman and Bailey (1967, p. 487, Eq. 5) print a warning and "
             "an instruction to examine the result where it is below 0.2 "
             "on any slice, not a rejection, so the surface is scored and "
             "the result says which surface and slice (trial_flags, "
             "critical.acceptability)")
    if rule == "every_slice":
        basis += "; every slice is looked at"
    else:
        basis += ("; slices with phi' = 0 are not looked at: there m_alpha "
                  "= cos(a) does not depend on F and the shear strength is "
                  "c l whatever the normal force (the module's own "
                  "reasoning, no printed source)")
    c = confirm or {}
    if c.get("fos") is not None:
        basis += (f"; on {c['n_slices']} slices the governing surface's FOS "
                  f"is {c['fos']:.3f}"
                  + (f" and its least m_alpha {c['m_alpha_min']:.2f}"
                     if c.get("m_alpha_min") is not None else ""))
        if c.get("fos_fine") is not None:
            basis += (f"; on {c['n_slices_fine']} slices, {c['fos_fine']:.3f}"
                      + (f" and {c['m_alpha_min_fine']:.2f}"
                         if c.get("m_alpha_min_fine") is not None else ""))
        elif c.get("refused_fine") == "geometry":
            basis += (f"; on {c['n_slices_fine']} slices it is not "
                      f"admissible ({c.get('why_fine') or 'its geometry'})")
        else:
            basis += (f"; on {c['n_slices_fine']} slices its method has no "
                      f"solution")
    alternatives = [
        "left out of the search and counted, as the module did before "
        "GM-140 step 3 and as Slide2 does with its 'Check malpha < 0.2' "
        "setting on (low_m_alpha='refuse')",
        "slices with phi' > 0 only (m_alpha_rule='frictional', the default)"
        if rule == "every_slice" else
        "every slice, phi' = 0 ones included (m_alpha_rule='every_slice')"]
    return choice(
        "slip surfaces with m_alpha below 0.2",
        f"kept and flagged: {n_flagged} scored trial(s) with m_alpha < 0.2 "
        f"on {where}",
        how_given(rule, "frictional", name="m_alpha_rule"), basis,
        alternatives)


def finer_slices_warning(confirm: Optional[Dict[str, Any]]) -> Optional[str]:
    """A warning when the governing surface's FOS moves by more than 1 % on
    twice the slices, or its method has no solution there (GM-127), or
    None. A finer slicing that cannot be built (a circle just touching the
    base) says nothing about the FOS and is not warned of."""
    c = confirm or {}
    f1, f2 = c.get("fos"), c.get("fos_fine")
    if f1 is None:
        return None
    n1, n2 = c["n_slices"], c["n_slices_fine"]
    if f2 is None:
        if c.get("refused_fine") != "no_solution":
            return None
        return (f"On {n2} slices, twice the search's {n1}, the governing "
                f"surface's method finds no solution: its FOS of {f1:.3f} "
                f"on {n1} slices depends on the slicing. Check it with "
                f"analyze_slope at other slice counts, or with another "
                f"method.")
    move = (f2 - f1) / f1
    if abs(move) <= 0.01:
        return None
    text = (f"On {n2} slices, twice the search's {n1}, the governing "
            f"surface's FOS is {f2:.3f} against {f1:.3f} ("
            f"{'+' if move > 0 else ''}{100 * move:.1f} %): it depends on "
            f"the slicing there")
    m2 = c.get("m_alpha_min_fine")
    if m2 is not None and m2 < 0.2:
        text += (f", and m_alpha falls to {m2:.2f} on a slice, so a search "
                 f"on {n2} slices would leave it out (" + _SLIDE2_CHECK
                 + "). A surface that comes out no steeper than the Rankine "
                   "passive angle, 45 deg - phi'/2 (Duncan, Wright & "
                   "Brandon 2014), avoids such slices.")
    else:
        text += (". Report it with that spread, or search again with more "
                 "slices (n_slices).")
    return text


def m_alpha_none_warning(result: SearchResult) -> Optional[str]:
    """When a search reports no surface and left some out for m_alpha < 0.2
    (GM-127), say so, with what else it refused; else None."""
    n = result.n_rejected_m_alpha
    if result.critical is not None or not n:
        return None
    others = [f"{k} {what}" for k, what in (
        (result.n_rejected_nonconverged, "had no converged solution"),
        (result.n_rejected_geometry, "were not admissible slip surfaces"),
        (result.n_rejected_ground_crossings,
         "left the ground and came back into it")) if k]
    return (f"No surface is reported: none of the trial surfaces held. {n} "
            f"were left out because m_alpha = cos(a) (1 + tan(a) tan(phi) "
            f"/ F) fell below 0.2 on a slice (Slide2's check, its error "
            f"-112; such surfaces usually come up steeply through the "
            f"passive zone)"
            + ("; " + "; ".join(others) if others else "")
            + ". Run a circular search (surface_type 'circular' or "
              "'entry_exit'), widen the entry and exit windows, or analyse "
              "a chosen surface with analyze_slope, which reports its FOS "
              "with a warning.")


def search_choices(result: SearchResult, *, surface_type: str,
                   method: Optional[str] = None,
                   scored_with: Optional[str] = None,
                   toe_circles: bool = True, infinite_slope: bool = True,
                   surface_types=tuple(SEARCH_TEXT), budget=None,
                   screened: bool = False,
                   screen_not_applied: Optional[str] = None,
                   m_alpha: Optional[Dict[str, Any]] = None,
                   m_alpha_rule: str = "frictional",
                   low_m_alpha: str = "flag",
                   negative_friction_term: str = "as_printed",
                   ) -> List[Dict[str, Any]]:
    """What a critical-surface search chose (GM-115), as its ``choices``
    (``geotech_common.choices``): the search, the method it scored with
    (when one was named), the surface families that gave an admissible
    surface (``result.families_scored``; a family it ran that gave none,
    with why), those it does not try, and how its ranges were set. The ranges,
    the governing surface's place and a refused lower surface are not
    repeated: the entry points at ``search_ranges``, and at
    ``governing_location`` and ``refused_lowest`` where the result has
    them (GM-125). With
    ``budget`` (GM-123): the time budget, and the Bishop screen when it was
    applied (``screened``) or asked for and not applied
    (``screen_not_applied``, the reason). The surfaces left out for m_alpha
    < 0.2 (GM-127) under ``m_alpha_rule``, with the governing surface on
    twice the slices (``m_alpha``, `m_alpha_confirmation`)."""
    from geotech_common.choices import GIVEN, MODULE, choice, how_given
    out = [choice(
        "search",
        f"{SEARCH_TEXT.get(surface_type, surface_type)} ('{surface_type}')",
        how_given(surface_type, "circular", name="surface_type"), "",
        [f"{SEARCH_TEXT.get(t, t)} ('{t}')" for t in surface_types
         if t != surface_type])]
    if method is not None:
        used = scored_with or method
        if used.lower() != method.lower():
            why = ("Bishop's simplified method is for circles; noncircular "
                   "surfaces are scored with Spencer"
                   if method.lower() == "bishop" else
                   f"this search scores with {used} whatever method is named")
            out.append(choice("FOS method of the search", used, MODULE, why))
        else:
            out.append(choice("FOS method of the search", used,
                              how_given(method, "bishop",
                                        name="method")))
        if screened:
            basis = (f"every circle of the main search scored with Bishop's "
                     f"simplified method, the lowest re-scored with {used}: "
                     f"the reported FOS is {used}'s (bishop_screen)")
            rec = getattr(budget, "screen", None) or {}
            if rec.get("dropped"):
                from slope_stability.search import _SCREEN_WITHIN
                basis += (f"; {rec['rescored']} re-scored, the most it "
                          f"re-scores: {rec['dropped']} more within "
                          f"{100 * _SCREEN_WITHIN:g} % of the screen's "
                          f"lowest were not")
            out.append(choice(
                "Bishop screen of the main search", True, GIVEN, basis,
                [f"no screen: every circle scored with {used}"]))
        elif screen_not_applied:
            out.append(choice(
                "Bishop screen of the main search", False, GIVEN,
                f"asked for (bishop_screen) but not applied: "
                f"{screen_not_applied}"))
    kind = CIRCULAR if surface_type in ("circular", "entry_exit") \
        else NONCIRCULAR
    # The families this search runs, and of those the ones it scored a
    # surface of (apply_supplements, GM-125): a choice says only what the
    # run did.
    runs = [kind]
    if toe_circles and kind == CIRCULAR:
        runs += [TOE_CIRCLE, ABOVE_TOE]
    if infinite_slope:
        runs.append(INFINITE_SLOPE)
    unscored = dict(getattr(result, "families_not_scored", None) or {})
    tried = [k for k in runs if k not in unscored]
    missed = [k for k in (CIRCULAR, NONCIRCULAR, TOE_CIRCLE, ABOVE_TOE,
                          INFINITE_SLOPE)
              if k not in runs]
    out.append(choice(
        "surface families scored",
        ", ".join(_FAMILY_TEXT[k] for k in tried) or "none", MODULE,
        ("the lowest FOS of each is reported (candidates), and the one that "
         "governs (governing_surface)" if tried else "no surface held")
        + ("; circles through each face's toe and passing above it, their "
           "lower end on the face from the toe up (search_ranges, "
           "faces_covered); the circles passing above the toe leave out "
           "any with reverse curvature"
           if ABOVE_TOE in tried else "")
        + ("; the infinite slope where c' = 0 soil forms the face"
           if INFINITE_SLOPE in tried else "")
        + "".join(f"; {_FAMILY_TEXT[k]}: none scored, {unscored[k]}"
                  for k in runs if k in unscored),
        [f"not tried: {_FAMILY_TEXT[k]}, by "
         + ("a search with them (toe_circles=True)"
            if kind == CIRCULAR and k in (TOE_CIRCLE, ABOVE_TOE)
            else _TRY_FAMILY[k])
         for k in missed]))
    sr = result.search_ranges or {}
    auto = list(sr.get("automatic") or [])
    # Point only at the fields the result carries (GM-125).
    where = ("the result says where the reported surface lies "
             "(governing_location)" if result.governing_location
             else "no surface is reported")
    if result.refused_lowest:
        where += (" and gives" if result.governing_location else
                  "; the result gives") + \
            " a lower surface that did not hold (refused_lowest)"
    if auto:
        given = [n for n in ("x_entry_range", "x_exit_range", "x_range",
                             "y_range") if n in sr and n not in auto]
        basis = ("from the slope's toe, crest, height and depth to the "
                 "base (FHWA NHI-06-088 Figs. 6-16, 6-19)"
                 if sr.get("basis") == "slope" else str(sr.get("basis", "")))
        out.append(choice(
            "search ranges",
            ", ".join(_RANGE_TEXT.get(n, n) for n in auto)
            + (" set by the module; given: "
               + ", ".join(_RANGE_TEXT.get(n, n) for n in given)
               if given else ""), MODULE,
            f"{basis}; the ranges used are in the result (search_ranges); "
            f"{where}",
            ["entry and exit windows given, for a failure in part of the "
             "slope (x_entry_range, x_exit_range)"]))
    elif sr:
        out.append(choice("search ranges", "given", GIVEN, where))
    n_low = sum(r["by_check"].get("low_m_alpha", 0)
                for r in result.trial_flags)
    out.append(m_alpha_choice(result.n_rejected_m_alpha,
                              scored_with or method, m_alpha,
                              m_alpha_rule, action=low_m_alpha,
                              n_flagged=n_low))
    from slope_stability.search import _split_so_far, _outside_so_far
    split = _split_so_far()
    if split:
        from geotech_common.choices import MODULE as _MODULE, choice as _ch
        n_out = sum(_outside_so_far().values())
        out.append(_ch(
            "circles that meet the ground at more than two points",
            f"split at those points, each arc inside the windows scored as "
            f"a trial surface of its own: {sum(split.values())} circle(s)"
            + (f"; {n_out} arc(s) outside the windows, not scored "
               f"(arcs_outside_windows)" if n_out else ""),
            _MODULE,
            "the module's rule (GM-343, 2026-11-13): no reference in hand "
            "prints one for a circle that crosses the ground more than "
            "twice (Slide2 refuses it). An arc is a slip surface between "
            "two points where the circle meets the ground; an arc with no "
            "net driving moment has no factor of safety and is counted "
            "(no_solution, no_driving); no arc is carried under the "
            "ground in front of a toe. The toe-circle and circle-above-"
            "the-toe families score only the arc that ends at the toe or "
            "face point. Circles ending on a face: FHWA "
            "NHI-06-088 p. 6-38; exits at every height of the face: GEC "
            "11 p. 9-6. circles_split counts them and critical.arc names "
            "the governing arc"))
    out.extend(acceptability_choices(result, scored_with or method,
                                     low_m_alpha, negative_friction_term))
    if budget is not None:
        out.append(budget_choice(budget))
    return out


def budget_choice(budget) -> Dict[str, Any]:
    """The ``choices`` entry for a search's time budget (GM-123)."""
    from geotech_common.choices import choice, how_given
    secs = budget.seconds
    # No run time here, so a search's choices do not change from run to
    # run; a search cut short has it in search_ranges['time_budget'].
    if budget.stopped:
        basis = ("the search stopped at the budget and reports the lowest "
                 "surface in what it covered (warnings; search_ranges, "
                 "time_budget)")
    elif secs is None:
        basis = ""
    else:
        basis = "the search finished inside it"
    return choice("time budget of the search",
                  "no limit" if secs is None else f"{secs:g} s",
                  how_given(secs, None, name="time_budget_s"), basis,
                  ["another budget (time_budget_s)" if secs is not None
                   else "a time budget (time_budget_s)"])


def _surface_of(crit):
    """The slip surface of a result, or None."""
    if getattr(crit, "is_circular", False) and crit.radius > 0:
        return circle_of(crit)
    if crit.slip_points and len(crit.slip_points) >= 2:
        return PolylineSlipSurface(points=list(crit.slip_points))
    return None


def _locate(geom: SlopeGeometry, crit, n: int = 60) -> Dict[str, Any]:
    """Where the governing surface lies (``SearchResult.governing_location``):
    its ends, its lowest point, its greatest depth below the ground (what
    tells a sliver from a slide), the layers along its base in order, and its
    position against the slope's toe and crest (the tallest face group, in
    multiples of its height H), so the engineer or the agent can judge what
    the search covered. Empty when there is no surface."""
    from slope_stability.search_ranges import face_groups
    if crit is None:
        return {}
    slip = _surface_of(crit)
    if slip is None:
        return {}
    x0, x1 = crit.x_entry, crit.x_exit
    layers: List[str] = []
    z_low = math.inf
    depth = 0.0
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        z = slip.slip_elevation_at(x)
        if z is None:
            continue
        z_low = min(z_low, z)
        depth = max(depth, geom.ground_elevation_at(x) - z)
        if i in (0, n):
            continue                  # the ends are on the ground surface
        lay = geom.layer_at_point(x, z)
        if lay is None:
            if z >= min(L.bottom_at(x) for L in geom.soil_layers):
                continue
            name = "below the deepest layer"
        else:
            name = lay.name
        if not layers or layers[-1] != name:
            layers.append(name)
    out: Dict[str, Any] = {"x_entry": round(x0, 3), "x_exit": round(x1, 3),
                           "z_lowest": round(z_low, 3),
                           "depth": round(depth, 3), "layers": layers}
    groups = face_groups(geom)
    if groups and math.isfinite(z_low):
        g = max(groups, key=lambda g: g["height"])
        (xt, zt), (xk, _) = g["toe"], g["crest"]
        s, h = g["rises"], g["height"]
        toe_end, crest_end = (x0, x1) if s > 0 else (x1, x0)
        out["slope"] = {
            "toe": [round(xt, 3), round(zt, 3)],
            "crest": [round(xk, 3), round(g["crest"][1], 3)],
            "height": round(h, 3),
            # + beyond the toe (away from the face), - up the face
            "toe_side_end_beyond_toe_H": round(s * (xt - toe_end) / h, 3),
            # + behind the crest, - down the face
            "crest_side_end_behind_crest_H": round(s * (crest_end - xk) / h,
                                                   3),
            "lowest_below_toe_H": round((zt - z_low) / h, 3),
        }
    return out


def _ranges_record(windows, grid, given, auto) -> Dict[str, Any]:
    """The ranges the main search used, which of them it set itself, and
    how (``SearchResult.search_ranges``)."""
    r = lambda v: [round(v[0], 3), round(v[1], 3)]   # noqa: E731
    rec: Dict[str, Any] = {}
    if windows[0] is not None:
        rec["x_entry_range"] = r(windows[0])
    if windows[1] is not None:
        rec["x_exit_range"] = r(windows[1])
    if grid is not None:
        rec["x_range"], rec["y_range"] = r(grid[0]), r(grid[1])
    names = ("x_entry_range", "x_exit_range") + (
        ("x_range", "y_range") if grid is not None else ())
    rec["automatic"] = [n for n in names if given.get(n) is None]
    if auto is not None and rec["automatic"]:
        rec.update(auto.to_dict())
    return rec


def _clipped_range_warning(geom, crit, windows, windows_auto,
                           auto) -> Optional[str]:
    """Warning when the governing surface ends at an end of the ground
    profile that cut an automatic window short (GM-57): the slope's own
    reach (its height and the depth to the base below the toe) runs past
    the model. Given only when the GM-63 check found no circle at least
    1 % lower on the ground carried on level past the end, so it says
    what that check does not cover."""
    if crit is None or auto is None or not auto.clipped:
        return None
    x_min, x_max = geom.x_range
    width = max(x_max - x_min, 1e-9)
    hits = []
    for key, end_x, i, edge in (("entry_lo", crit.x_entry, 0, x_min),
                                ("exit_hi", crit.x_exit, 1, x_max)):
        rng = windows[i]
        if key not in auto.clipped or not windows_auto[i] or rng is None:
            continue
        tol = max(0.1 * abs(rng[1] - rng[0]), 0.02 * width)
        if _near(end_x, edge, tol):
            hits.append((end_x, edge, auto.clipped[key]))
    if not hits:
        return None
    s = max(auto.slopes, key=lambda g: g["height"])
    parts = [f"x = {end_x:.2f}, near the end of the ground surface at "
             f"x = {edge:.2f} (the slope's reach runs to x = {want:.2f})"
             for end_x, edge, want in hits]
    return (
        f"The critical surface ends at {' and '.join(parts)}: the model "
        f"ends before the automatic entry / exit range would (a slope "
        f"{s['height']:.2f} high with its base {s['depth_to_base']:.2f} "
        f"below the toe), so no surface past the end was tried. Circles on "
        f"the ground carried on level past the end are no lower; if the "
        f"ground beyond the profile rises, falls or is a different soil, "
        f"extend the profile and search again.")
