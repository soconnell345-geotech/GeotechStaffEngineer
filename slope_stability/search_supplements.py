"""
Supplementary trial-surface families for the critical-surface search.

The centre-grid and entry/exit searches sample circles inside windows set
from the MODEL extent (entry in the first 40 % of the model width, exit in
the last 40 %, centres at least 1 unit above the crest). Two failure
mechanisms fall outside that sampling on ordinary slopes and the search then
reports a factor of safety that is too HIGH (BACKLOG GM-13, GM-14):

1. **Toe circles** (Taylor 1937; Duncan, Wright & Brandon 2014, Ch. 6-7).
   A circle through the toe of the slope whose other end is close above it.
   On a steep slope in a short model the critical toe circle exits on the
   crest well inside the automatic exit window's lower bound, and its centre
   lies below the crest, under the bottom of the automatic centre grid.
   Steep phi = 0 slope, H = 5 m at 45 deg, su = 40 kPa: grid 2.78 against a
   true toe circle of 2.287.

2. **Infinite slope / shallow planar (translational) sliding** in
   cohesionless soil (Duncan, Wright & Brandon 2014, Ch. 14; Skempton &
   DeLory 1957). With c' = 0 the critical failure is a shallow slide
   parallel to the slope face, FOS -> tan(phi') / tan(beta) for dry
   conditions as the slide gets shallower, independent of depth. Every
   circle the grid tries is deeper and stronger. Slide2 #86 (c' = 0,
   phi' = 37 deg fill at 1.25H:1V, no supports): grid 1.57 against 0.942.

`apply_supplements` scores the toe circles with the search's own limit
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
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.results import SearchResult, SlopeStabilityResult

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
# Rigorous methods scan the toe circles with Bishop, then re-optimize this
# many of the best circles with the method itself.
_SCREENED_METHODS = ("spencer", "morgenstern_price", "gle")
_N_RESCORE = 3
# A circle the rigorous method finds no solution on (large point loads:
# GM-60) does not count toward _N_RESCORE; the next one down the Bishop
# shortlist is tried instead, up to this many in all.
_MAX_RESCORE_TRIES = 12

# Infinite-slope family: shallow planar surfaces parallel to the face, at a
# vertical depth of 1/_PLANAR_DEPTH_SLICES of a slice width. The short end
# ramps that close the surface to the ground then lie inside the first and
# last slices, short of their midpoints, so every slice base is parallel to
# the face. (Deeper surfaces with ramps a slice or more long are mis-sliced:
# a slice whose midpoint lands on a 45 deg ramp swung Janbu 5 % either side
# of the closed form.) The slide spans the whole outcrop, each half and
# each third, so a face wet over part of its length is caught.
_PLANAR_DEPTH_SLICES = 10.0
_PLANAR_SPLITS = (1, 2, 3)
# The method the planar surfaces are scored with (see check_infinite_slope).
PLANAR_METHOD = "janbu"
# Note on a face whose reinforcement leaves no shallow planar slide able to
# fail (the solver caps the FOS); such a face needs no infinite-slope warning.
HELD_NOTE = ("every shallow planar slide is held by the reinforcement "
             "crossing the face (FOS above the 999.9 cap)")

# Surface-family labels (the references' names).
CIRCULAR = "circular"
NONCIRCULAR = "noncircular"
TOE_CIRCLE = "toe_circle"
INFINITE_SLOPE = "infinite_slope"

_KIND_TEXT = {
    CIRCULAR: "circular (search)",
    NONCIRCULAR: "noncircular (search)",
    TOE_CIRCLE: "toe circle",
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
                       ) -> Dict[str, Any]:
    """Critical circle through the toe of each slope face.

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

    Returns a dict: ``FOS`` (999.9 if none admissible), ``xc``, ``yc``,
    ``radius``, ``toe`` (x, z), ``n_evaluated``.
    """
    from slope_stability.search import _compute_fos, _check_entry_exit, _FOS_MAX

    scan_method = ("bishop" if method in _SCREENED_METHODS and fos_fn is None
                   else method)
    faces = sorted(slope_faces(geom), key=lambda f: -f["height"])
    faces = faces[:_MAX_TOE_FACES]
    x_min, x_max = geom.x_range
    windows_given = x_entry_range is not None or x_exit_range is not None
    counter = [0]
    # A caller-supplied per-surface solve (rapid drawdown: two or three GLE
    # solves a surface, no cheap screen) gets a coarser scan.
    n_t, u_tol, e_iter = (_N_T, 0.01, 12) if fos_fn is None else (7, 0.04, 6)
    u_lo = math.log(_T_RATIO_RANGE[0])
    u_hi = math.log(_T_RATIO_RANGE[1])
    us = [u_lo + (u_hi - u_lo) * i / (n_t - 1) for i in range(n_t)]
    u_step = (u_hi - u_lo) / (n_t - 1)

    def circle_fos(xc, yc, r, fos_method):
        if not _centre_admissible(geom, xc, yc, x_range, y_range):
            return _FOS_MAX
        # A hair inside the toe so the arc crosses the ground just above
        # it (a circle exactly through a model-edge toe finds one crossing).
        slip = CircularSlipSurface(xc, yc, r * (1.0 - 1e-6))
        if windows_given and not _check_entry_exit(
                geom, slip, x_entry_range, x_exit_range):
            return _FOS_MAX
        counter[0] += 1
        return _compute_fos(geom, slip, fos_method, n_slices, tol=tol,
                            fos_fn=fos_fn)

    def chord(toe, s, offset):
        """The chord from the toe to the ground point `offset` upslope."""
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
                "xl": xl, "xr": xr}

    def below_base(ch, xc, yc, r):
        """True when the arc between the chord ends dips below the bottom of
        the deepest layer. `build_slices` tolerates a tangential sliver or
        two there, but a toe sitting ON a rigid base would let every toe
        circle through it shed its toe slices that way and score low
        (Slide2 #85: 0.83 against the base-tangent 0.91). The critical toe
        circle can at most touch the base."""
        x_low = min(max(xc, ch["xl"]), ch["xr"])
        z_low = yc - math.sqrt(max(r * r - (x_low - xc) ** 2, 0.0))
        base = min(L.bottom_at(x_low) for L in geom.soil_layers)
        return z_low < base - 1e-6 * max(ch["c"], 1.0)

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
            return circle_fos(xc, yc, r, fos_method)

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

    shortlist = []          # (fos with scan_method, chord, u)
    for face in faces:
        toe, s = face["toe"], face["rises"]
        size = max(face["length"], face["height"])
        if size <= 0:
            continue
        reach = abs((x_max if s > 0 else x_min) - toe[0])
        offsets = sorted({min(size * f, reach) for f in _EXIT_FACTORS})
        offsets = [o for o in offsets if o > 1e-6 * size]
        if not offsets:
            continue

        coarse = []
        for o in offsets:
            ch = chord(toe, s, o)
            fos, u = chord_best(ch, scan_method)
            coarse.append((fos, ch, u))
        shortlist.extend(coarse)
        j = min(range(len(offsets)), key=lambda i: coarse[i][0])
        if coarse[j][0] < _FOS_MAX and len(offsets) > 1:
            cache = {}

            def g(o):
                ch = chord(toe, s, o)
                cache[o] = chord_best(ch, scan_method) + (ch,)
                return cache[o][0]

            o_g, f_g, _ = _golden_min(
                g, offsets[max(j - 1, 0)],
                offsets[min(j + 1, len(offsets) - 1)],
                tol=0.005 * size, max_iter=e_iter)
            fos, u, ch = cache[o_g]
            shortlist.append((fos, ch, u))

    shortlist = [c for c in shortlist if c[0] < _FOS_MAX]
    shortlist.sort(key=lambda c: c[0])
    if scan_method != method:
        rescored = []
        n_ok = 0
        for _fos, ch, u in shortlist[:_MAX_RESCORE_TRIES]:
            fos, u2 = chord_best(ch, method, (u - u_step, u + u_step))
            rescored.append((fos, ch, u2))
            n_ok += fos < _FOS_MAX
            if n_ok >= _N_RESCORE:
                break
        shortlist = sorted(rescored, key=lambda c: c[0])

    best = {"FOS": _FOS_MAX, "xc": 0.0, "yc": 0.0, "radius": 0.0,
            "toe": None, "n_evaluated": counter[0]}
    if shortlist and shortlist[0][0] < _FOS_MAX:
        fos, ch, u = shortlist[0]
        xc, yc, r = circ(ch, u)
        best.update(FOS=fos, xc=xc, yc=yc, radius=r * (1.0 - 1e-6),
                    toe=ch["toe"])
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
    ``phi_deg``. A face crossing several layers gives one run per
    cohesionless outcrop.
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
                       "phi_deg": layer.phi, "_layer": layer}
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
    ``xa`` and ``xb``, a tenth of a slice width deep, with short end ramps up
    to the ground: the ramp at one end flatter than the face, at the other
    steeper, so the surface stays concave. The ramps (0.15 of a slice
    width) end before the first and last slices' midpoints, so every slice
    base lies parallel to the face."""
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
        _compute_fos, _FOS_MAX, trial_reinforcement,
    )

    best_fos, best_slip = _FOS_MAX, None
    checks: List[Dict[str, Any]] = []
    n_eval = 0
    for run in cohesionless_face_runs(geom):
        beta = run["slope_angle_deg"]
        closed = (math.tan(math.radians(run["phi_deg"]))
                  / math.tan(math.radians(beta)))
        check = {
            "x_from": round(run["x_from"], 3), "x_to": round(run["x_to"], 3),
            "slope_angle_deg": round(beta, 3), "layer": run["layer"],
            "phi_deg": run["phi_deg"],
            "FOS_dry_closed_form": round(closed, 4),
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
        x_entry, x_exit = slip.find_entry_exit(geom)
        n = len(build_slices(geom, slip, n_slices))
    except ValueError:
        return None
    kw = {}
    if getattr(slip, "is_circular", True):
        kw = dict(xc=slip.xc, yc=slip.yc, radius=slip.radius)
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
    slip = (CircularSlipSurface(crit.xc, crit.yc, crit.radius)
            if crit.is_circular else PolylineSlipSurface(crit.slip_points))
    try:
        forces = trial_reinforcement(geom, slip)
    except ValueError:
        return
    crit.reinforcements = [
        {"kind": f.kind, "index": f.index,
         "x_m": round(f.x, 3), "z_m": round(f.z, 3),
         "T_kN_per_m": round(f.T, 2), "controlled_by": f.controlled_by,
         "support_convention": "passive" if f.passive else "active"}
        for f in (forces or [])] or None


def _reverse_curvature_warning(geom, crit, in_search: bool = True
                               ) -> Optional[str]:
    """Say so when the governing circle (``in_search``) or the circle given
    to ``analyze_slope`` is a reverse-curvature surface closed by a crack
    (``CircularSlipSurface.reverse_curvature_crack``)."""
    if crit is None or not crit.is_circular or crit.radius <= 0:
        return None
    crack = CircularSlipSurface(crit.xc, crit.yc, crit.radius) \
        .reverse_curvature_crack(geom, crit.x_entry, crit.x_exit)
    if crack is None:
        return None
    if in_search:
        whose = "The critical circle's"
        other = ("reverse_curvature='discard' drops such circles from the "
                 "search (Slide2's other choice).")
    else:
        whose = "The circle's"
        other = ("Slide2's other choice is to discard such circles: "
                 "reverse_curvature='discard' does so in a search; a circle "
                 "given to analyze_slope is evaluated either way.")
    return (f"{whose} centre (z = {crit.yc:.2f}) lies below the "
            f"ground where the circle ends (x = {crack['x']:.2f}, ground "
            f"z = {crack['z_ground']:.2f}): the circle would turn back above "
            f"its own centre there, so the surface is closed by a vertical, "
            f"dry crack {crack['depth']:.2f} m deep that carries no shear. "
            f"Slide2 calls this a reverse-curvature surface and closes it the "
            f"same way by default; its factor of safety depends on that crack "
            f"being able to open. {other}")


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
    slip = CircularSlipSurface(crit.xc, crit.yc, crit.radius)
    try:
        forces = trial_reinforcement(geom, slip)
        slices = build_slices(geom, slip, n_slices)
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
        spencer = _compute_fos(geom, slip, "spencer", n_slices, tol=tol)
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
            f"force equilibrium, which it does not satisfy (Wright & Duncan "
            f"1991). The {len(forces)} support(s) crossing {which} "
            f"({t_sum:.1f} kN/m) act with an average arm of "
            f"{arm:.2f} m about its centre on a {slip.radius:.2f} m radius: "
            f"{100 * share:.0f} % of their force counts. {rigorous}. Check "
            f"the slope with method='spencer' or 'gle'.")


def _legacy_critical_warning(geom, crit, method: str, n_slices: int,
                             tol: float) -> Optional[str]:
    """Label a critical circle a rigorous search scored with the legacy
    approximate solver (GM-81).

    `_compute_fos` requires the GLE to converge on a noncircular surface
    and on a circle supports cross, and rejects the surface otherwise; on
    a circle no support crosses it takes `spencer_fos` /
    `morgenstern_price_fos`, which fall back to the legacy approximate
    solver. When the governing circle is one of those, its result says so
    (``method_solved`` False, ``method_used``) and so does the search."""
    if crit is None or not crit.is_circular or method not in _SCREENED_METHODS:
        return None
    from slope_stability.search import trial_reinforcement
    from slope_stability.slices import build_slices
    from slope_stability.methods import solve_rigorous
    slip = CircularSlipSurface(crit.xc, crit.yc, crit.radius)
    try:
        if trial_reinforcement(geom, slip):
            return None              # the search required convergence
        slices = build_slices(geom, slip, n_slices)
    except ValueError:
        return None
    spencer = method == "spencer"
    sol = solve_rigorous(slices, slip, "constant" if spencer else "half_sine",
                         tol)
    if sol.solved:
        return None
    name = "Spencer" if spencer else "Morgenstern-Price"
    crit.method_solved = False
    crit.method_used = f"legacy approximate {name} solver"
    crit.methods_without_solution = [name]
    phrase = "Spencer's method" if spencer else "The Morgenstern-Price method"
    return (f"{phrase} finds no equilibrium solution on the critical circle: "
            f"{sol.reason}. The search scores a circle no support crosses "
            f"with the legacy approximate solver when that happens, so the "
            f"FOS reported, {crit.FOS:.3f}, is that solver's value, not a "
            f"converged {name} solution.")


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
    slip = CircularSlipSurface(xc, yc, r)
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
                      fos_fn=None) -> SearchResult:
    """Score the toe-circle and infinite-slope families and fold them into
    ``result`` (modified in place and returned).

    ``x_entry_range`` / ``x_exit_range`` / ``x_range`` / ``y_range`` are the
    CALLER-GIVEN constraints (None where the search chose its own).
    ``windows_used`` / ``grid_used`` are what the main search actually used,
    for the edge warnings. ``toe_circles`` applies to circular searches only.
    ``fos_fn`` (the rapid-drawdown search's per-surface FOS) scores both
    families when given; the planar surfaces are then labelled with
    ``method`` instead of Janbu.
    """
    from slope_stability.search import _FOS_MAX
    planar_method = PLANAR_METHOD if fos_fn is None else method

    main = result.critical
    candidates: Dict[str, float] = {}
    if main is not None:
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
                                              toe["radius"]),
                    toe["FOS"], method.capitalize(), n_slices)
                if r is not None:
                    best_kind, best, best_fos = TOE_CIRCLE, r, toe["FOS"]

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

    main_fos = main.FOS if main is not None else None
    main_text = describe_surface_kind(search_kind)
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
            f"({was}). The main search's entry/exit windows, centre grid or "
            f"grid spacing kept it from this circle; the toe circle is "
            f"reported.")
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

    if best_kind in (CIRCULAR, NONCIRCULAR):
        warnings_out.extend(_edge_warnings(
            geom, best, windows_used, windows_auto,
            grid_used if search_kind == CIRCULAR else None, grid_auto))
    elif best_kind == TOE_CIRCLE:
        # Only the caller's own windows bound the toe-circle family.
        given = (x_entry_range, x_exit_range)
        warnings_out.extend(_edge_warnings(
            geom, best, given, (False, False), None, grid_auto))

    # A governing circle held back by the end of the ground profile (GM-63)
    if best is not None and best_kind != INFINITE_SLOPE:
        w = _profile_end_warning(geom, best, method, n_slices, tol,
                                 fos_fn=fos_fn)
        if w:
            warnings_out.append(w)

    # The governing circle closed by a crack at its own centre's level, and
    # supports a moment-only method counts little of (GM-59).
    if best is not None and best_kind != INFINITE_SLOPE:
        w = _reverse_curvature_warning(geom, best)
        if w:
            warnings_out.append(w)
        if fos_fn is None:
            w = _support_arm_warning(geom, best, method, n_slices, tol)
            if w:
                warnings_out.append(w)
            # A circle scored by the legacy approximate solver (GM-81)
            w = _legacy_critical_warning(geom, best, method, n_slices, tol)
            if w:
                warnings_out.append(w)

    _attach_reinforcement(geom, best)
    result.critical = best
    result.governing_surface = best_kind if best is not None else ""
    result.candidates = candidates
    result.warnings = warnings_out
    return result
