"""
Critical slip surface search algorithms.

Algorithms:
1. Grid search — 2D grid of trial centers (xc, yc), optimizes radius at each.
2. Random noncircular — random polyline generation with Spencer's method.
3. PSO noncircular — Particle Swarm Optimization over control points.
4. Weak-layer biased — SNIFF-inspired search biasing toward weak layers.
5. Entry/exit region — generates trial arcs from entry/exit zones.

References:
    Duncan, Wright & Brandon (2014) — Chapter 14
    Kennedy & Eberhart (1995) — Particle Swarm Optimization
    Borselli (2002) — SNIFF Random Search concept
"""

import math
import random
import warnings
from typing import List, Tuple, Optional, Dict, Any, Callable

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import (
    fellenius_fos, bishop_fos, spencer_fos, morgenstern_price_fos,
)
from slope_stability.gle import janbu_fos
from slope_stability.results import SearchResult, SlopeStabilityResult


_FOS_MAX = 999.9


def _window_span(x_entry_range, x_exit_range) -> Optional[float]:
    """Characteristic span of a surface bracketed by the entry/exit windows.

    The centre-to-centre distance between the entry and exit windows — a real
    critical surface spans roughly this. Used to anchor the noncircular span
    floor (``_noncircular_admissible``) to the actual problem rather than the
    full model width. Returns None if either window is missing.
    """
    if x_entry_range is None or x_exit_range is None:
        return None
    entry_c = 0.5 * (x_entry_range[0] + x_entry_range[1])
    exit_c = 0.5 * (x_exit_range[0] + x_exit_range[1])
    return abs(exit_c - entry_c)


def _new_reject_stats() -> Dict[str, int]:
    """Fresh per-search rejection tally (see ``_compute_fos``)."""
    return {"geometry": 0, "nonconverged": 0, "jagged": 0}


def _bump(stats: Optional[Dict[str, int]], key: str) -> None:
    if stats is not None:
        stats[key] += 1


def _rejection_kwargs(stats: Dict[str, int], n_evaluated: int) -> Dict[str, int]:
    """SearchResult counter kwargs + a ``warnings.warn`` when a majority of the
    trial surfaces were rejected (an otherwise-silent under-resolution)."""
    total_rej = stats["geometry"] + stats["nonconverged"] + stats["jagged"]
    if n_evaluated > 0 and total_rej > 0.5 * n_evaluated:
        warnings.warn(
            f"Noncircular search rejected {total_rej}/{n_evaluated} trial "
            f"surfaces ({stats['geometry']} geometry, {stats['nonconverged']} "
            f"non-converged, {stats['jagged']} jagged). The critical surface may "
            f"be under-resolved -- widen the entry/exit windows, raise n_trials, "
            f"or check the geometry.")
    return {
        "n_rejected_geometry": stats["geometry"],
        "n_rejected_nonconverged": stats["nonconverged"],
        "n_rejected_jagged": stats["jagged"],
    }


def _noncircular_admissible(slip, slices, geom, n_slices_requested: int,
                            window_span: Optional[float] = None) -> bool:
    """Reject grossly-degenerate noncircular trial surfaces (SS-6, geometry).

    Cheap geometric sanity checks the team-lead's list calls for — they weed
    out sliver / non-spanning surfaces before the (more expensive) rigorous
    solve. The *jagged-but-spanning* zig-zags that used to win searches with a
    spurious low FOS are caught separately and more reliably by REQUIRING the
    rigorous GLE to converge (see ``_rigorous_noncircular_fos``): the GLE
    reports ``converged=False`` on those, whereas valid smooth surfaces
    converge. This geometric check is deliberately lenient so the random /
    weak-layer generators keep exploring.

    * **Too few surviving slices** — a surface that self-clips or barely enters
      the ground leaves a tiny sliver mass (build_slices drops below-model
      slices); not a usable failure surface.
    * **Too short a horizontal span** — the span floor is anchored to the
      caller's entry/exit WINDOW extent (``window_span``: a real surface roughly
      spans the window) when supplied, else to the local slope HEIGHT (0.5*H).
      The old floor of 15% of the FULL model width silently rejected a legitimate
      localized failure in a wide model — a wide right-of-way or a long tailwater
      bench blew the floor up far past a real slide length.
    """
    if len(slices) < max(5, n_slices_requested // 3):
        return False
    span = slices[-1].x_right - slices[0].x_left
    if window_span is not None and window_span > 0:
        min_span = 0.5 * window_span
    else:
        zs = [p[1] for p in geom.surface_points]
        slope_height = (max(zs) - min(zs)) if zs else 0.0
        min_span = 0.5 * slope_height
    if min_span > 0 and span < min_span:
        return False
    return True


# A suspiciously LOW FOS on a jagged (non-concave) noncircular surface is a
# solver artifact, not a real critical surface. We geometry-check only such
# low-FOS surfaces, so the random/weak-layer generators keep exploring the many
# high-FOS jagged trials (which harmlessly lose the search) while a spurious low
# value can never win.
_LOW_FOS_JAGGED_GATE = 1.5
_JAG_MAX_SEGMENT_DEG = 80.0     # near-vertical segment
_JAG_MAX_DROP_DEG = 25.0        # concavity reversal from the running-max angle


def _is_jagged(slip) -> bool:
    """True if the polyline is non-concave / spiky (kinematically inadmissible).

    A real slip surface is concave-up: its segment inclination increases
    roughly monotonically from the driving end to the resisting end. A surface
    with a near-vertical segment, a non-monotonic x, or a segment angle that
    DROPS more than ``_JAG_MAX_DROP_DEG`` below the running maximum (a
    back-and-forth zig-zag / bump) is flagged.
    """
    pts = getattr(slip, "points", None)
    if not pts or len(pts) < 3:
        return False
    max_seg = math.radians(_JAG_MAX_SEGMENT_DEG)
    max_drop = math.radians(_JAG_MAX_DROP_DEG)
    running_max = -math.inf
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dz = pts[i + 1][1] - pts[i][1]
        if dx <= 0:
            return True
        ang = math.atan2(dz, dx)
        if abs(ang) > max_seg or ang < running_max - max_drop:
            return True
        running_max = max(running_max, ang)
    return False


def _noncircular_axis_point(slices, geom) -> Optional[Tuple[float, float]]:
    """A deterministic, robust moment axis for the noncircular convergence gate.

    Midpoint of the surviving span, half a span above the highest ground point —
    the same centroid-above-surface heuristic the validated composite-surface
    path passes explicitly (test_validation uses the parent circle centre). This
    pins the axis for the convergence check instead of leaving it to gle's
    internal least-squares fit, which can land absurdly far away on a
    near-straight surface and spuriously fail convergence. A converged GLE FOS is
    axis-independent (both moment and force equilibrium are enforced), so this
    does not move the reported FOS of valid surfaces.
    """
    if not slices:
        return None
    x_lo = slices[0].x_left
    x_hi = slices[-1].x_right
    span = max(x_hi - x_lo, 1.0)
    z_top = max(p[1] for p in geom.surface_points) if geom is not None else \
        max(s.z_base for s in slices)
    return (0.5 * (x_lo + x_hi), z_top + 0.5 * span)


def _rigorous_noncircular_fos(slices, slip, f_interslice: str,
                              tol: float, geom=None) -> Optional[float]:
    """Rigorous GLE FOS for a noncircular surface, REQUIRING convergence.

    Returns None when the GLE does not converge, so the caller REJECTS the
    surface instead of silently accepting the legacy Spencer approximation —
    which returns a spurious low FOS (~0.05-0.2) on jagged/degenerate surfaces
    that a search would then wrongly pick as critical. The rigorous GLE reports
    ``converged=False`` on those surfaces at every slice count (verified on the
    ACADS-4 zig-zag: GLE constant 0.15 / half-sine 0.22, both non-converged,
    vs a Fellenius 8.2 — the methods disagree because the surface is
    kinematically inadmissible). Valid smooth surfaces converge and return the
    same value as before.

    The moment axis for the convergence check is tried in two steps: first gle's
    internal least-squares fit (``axis_point=None`` — byte-identical to the
    historical behaviour, so a surface that already converged is unchanged), and
    only if THAT fails to converge, a robust pinned axis
    (``_noncircular_axis_point``, when ``geom`` is supplied). The second pass
    rescues a valid surface whose least-squares centre lands absurdly far away
    (spuriously failing convergence) without ever removing a convergence the
    first pass already found.
    """
    from slope_stability.gle import gle_fos
    axes = [None]
    if geom is not None:
        pinned = _noncircular_axis_point(slices, geom)
        if pinned is not None:
            axes.append(pinned)
    for axis_point in axes:
        try:
            res = gle_fos(slices, slip, f_interslice=f_interslice,
                          tol=min(tol, 1e-4) * 0.1, axis_point=axis_point)
            if res.converged and 0.05 < res.fos < _FOS_MAX:
                return res.fos
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return None


def _compute_fos(geom: SlopeGeometry,
                 slip,
                 method: str,
                 n_slices: int,
                 tol: float = 1e-4,
                 window_span: Optional[float] = None,
                 reject_stats: Optional[Dict[str, int]] = None,
                 fos_fn: Optional[Callable] = None) -> float:
    """Safely compute FOS, returning _FOS_MAX on any error.

    ``window_span`` anchors the noncircular span floor to the caller's entry/exit
    window (see ``_noncircular_admissible``). ``reject_stats`` (optional) is a
    mutable tally the generators pass so that silent noncircular rejections
    (geometry / non-converged / jagged) are counted rather than lost.

    ``fos_fn`` (optional) substitutes a caller-supplied per-surface FOS
    evaluator ``fos_fn(geom, slip) -> float`` for the built-in method dispatch,
    while KEEPING the same geometry-admissibility, non-convergence, and jagged
    guards. This is how ``search_rapid_drawdown`` reuses every search loop
    (grid / entry-exit / random / DE) with the rapid-drawdown FOS substituted per
    surface instead of an ordinary drained/undrained solve. Default ``None``
    leaves the method dispatch byte-identical.
    """
    try:
        slices = build_slices(geom, slip, n_slices)
        if len(slices) < 3:
            _bump(reject_stats, "geometry")
            return _FOS_MAX

        # Noncircular guard (SS-6): reject kinematically-degenerate trial
        # surfaces and REQUIRE the rigorous GLE to converge. This kills the
        # weak-layer/random-polyline search pathology where a jagged surface
        # makes the rigorous solver diverge, the legacy Spencer fallback
        # returns a spurious low FOS (~0.05-0.2), and the search wrongly picks
        # it as critical. Circular surfaces are untouched (their GLE converges;
        # bishop/grid/entry-exit paths are byte-identical below).
        if not getattr(slip, 'is_circular', True):
            if not _noncircular_admissible(slip, slices, geom, n_slices,
                                           window_span):
                _bump(reject_stats, "geometry")
                return _FOS_MAX
            if fos_fn is not None:
                fos = fos_fn(geom, slip)
            elif method in ("morgenstern_price", "gle"):
                fos = _rigorous_noncircular_fos(slices, slip, "half_sine", tol,
                                                geom)
            elif method == "fellenius":
                fos = fellenius_fos(slices, slip)
            elif method == "janbu":
                fos, _u, _f0 = janbu_fos(slices, slip,
                                         tol=min(tol, 1e-4) * 0.1)
            else:  # spencer / bishop-fallback / default
                fos = _rigorous_noncircular_fos(slices, slip, "constant", tol,
                                                geom)
            if fos is None or fos < 0.05 or math.isnan(fos) or math.isinf(fos):
                _bump(reject_stats, "nonconverged")
                return _FOS_MAX
            # Spurious-low-FOS guard: a low FOS on a jagged surface is a solver
            # artifact (SS-6), not a real critical surface — reject it.
            if fos < _LOW_FOS_JAGGED_GATE and _is_jagged(slip):
                _bump(reject_stats, "jagged")
                return _FOS_MAX
            return fos

        if fos_fn is not None:
            fos = fos_fn(geom, slip)
        elif method == "fellenius":
            fos = fellenius_fos(slices, slip)
        elif method == "bishop":
            if not getattr(slip, 'is_circular', True):
                # Bishop can't handle noncircular, fall back to Spencer
                fos, _ = spencer_fos(slices, slip, tol=tol)
            else:
                fos = bishop_fos(slices, slip, tol=tol)
        elif method == "spencer":
            fos, _ = spencer_fos(slices, slip, tol=tol)
        elif method in ("morgenstern_price", "gle"):
            fos, _ = morgenstern_price_fos(slices, slip, tol=tol)
        elif method == "janbu":
            fos, _, _ = janbu_fos(slices, slip, tol=min(tol, 1e-4) * 0.1)
        else:
            fos = bishop_fos(slices, slip, tol=tol)

        # Guard against non-convergence producing negative or absurd values
        # FOS < 0.05 is physically meaningless and indicates numerical issues
        if fos is None or fos < 0.05 or math.isnan(fos) or math.isinf(fos):
            return _FOS_MAX
        return fos
    except (ValueError, ZeroDivisionError, RuntimeError):
        return _FOS_MAX


def _check_entry_exit(geom, slip, x_entry_range, x_exit_range):
    """Check if slip surface entry/exit are within allowed x-ranges.

    Returns True if acceptable, False if should be rejected.
    """
    if x_entry_range is None and x_exit_range is None:
        return True
    try:
        x_entry, x_exit = slip.find_entry_exit(geom)
    except ValueError:
        return False

    if x_entry_range is not None:
        if x_entry < x_entry_range[0] or x_entry > x_entry_range[1]:
            return False
    if x_exit_range is not None:
        if x_exit < x_exit_range[0] or x_exit > x_exit_range[1]:
            return False
    return True


def optimize_radius(geom: SlopeGeometry,
                    xc: float, yc: float,
                    method: str = "bishop",
                    n_slices: int = 30,
                    r_min: Optional[float] = None,
                    r_max: Optional[float] = None,
                    n_radii: int = 20,
                    tol: float = 1e-4,
                    x_entry_range: Optional[Tuple[float, float]] = None,
                    x_exit_range: Optional[Tuple[float, float]] = None,
                    fos_fn: Optional[Callable] = None,
                    ) -> Tuple[float, float]:
    """Find the radius that minimizes FOS for a given circle center.

    Uses a coarse scan followed by golden-section refinement.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    xc, yc : float
        Circle center coordinates.
    method : str
        'fellenius', 'bishop', or 'spencer'. Default 'bishop'.
    n_slices : int
        Number of slices.
    r_min, r_max : float, optional
        Radius search bounds. Auto-computed if None.
    n_radii : int
        Number of trial radii in coarse scan.
    tol : float
        Convergence tolerance. Default 1e-4.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range. Surfaces outside are rejected.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range. Surfaces outside are rejected.

    Returns
    -------
    (R_opt, FOS_min) : tuple
        Optimal radius and minimum FOS.
    """
    # Auto-compute radius bounds
    if r_min is None or r_max is None:
        zs = [p[1] for p in geom.surface_points]
        z_max_ground = max(zs)

        dist_to_ground = yc - z_max_ground
        if dist_to_ground <= 0:
            dist_to_ground = 0.5
        if r_min is None:
            r_min = max(dist_to_ground + 0.5, 1.0)

        min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
        max_dist = math.sqrt((xc - geom.surface_points[0][0])**2 +
                             (yc - min_layer_bot)**2)
        if r_max is None:
            r_max = max(max_dist, r_min + 5.0)

    if r_min >= r_max:
        r_max = r_min + 5.0

    # Coarse scan
    best_r = r_min
    best_fos = _FOS_MAX

    for i in range(n_radii):
        r = r_min + (r_max - r_min) * i / max(n_radii - 1, 1)
        slip = CircularSlipSurface(xc, yc, r)

        # Entry/exit range check
        if not _check_entry_exit(geom, slip, x_entry_range, x_exit_range):
            continue

        fos = _compute_fos(geom, slip, method, n_slices, tol=tol, fos_fn=fos_fn)
        if fos < best_fos:
            best_fos = fos
            best_r = r

    # Golden-section refinement around best_r
    golden = (math.sqrt(5) - 1) / 2
    margin = (r_max - r_min) / n_radii
    a = max(r_min, best_r - 2 * margin)
    b = min(r_max, best_r + 2 * margin)

    for _ in range(20):
        if b - a < 0.01:
            break
        r1 = b - golden * (b - a)
        r2 = a + golden * (b - a)

        slip1 = CircularSlipSurface(xc, yc, r1)
        slip2 = CircularSlipSurface(xc, yc, r2)

        ok1 = _check_entry_exit(geom, slip1, x_entry_range, x_exit_range)
        ok2 = _check_entry_exit(geom, slip2, x_entry_range, x_exit_range)

        f1 = _compute_fos(geom, slip1, method, n_slices, tol=tol,
                          fos_fn=fos_fn) if ok1 else _FOS_MAX
        f2 = _compute_fos(geom, slip2, method, n_slices, tol=tol,
                          fos_fn=fos_fn) if ok2 else _FOS_MAX

        if f1 < f2:
            b = r2
            if f1 < best_fos:
                best_fos = f1
                best_r = r1
        else:
            a = r1
            if f2 < best_fos:
                best_fos = f2
                best_r = r2

    return (best_r, best_fos)


def grid_search(geom: SlopeGeometry,
                x_range: Tuple[float, float],
                y_range: Tuple[float, float],
                nx: int = 10,
                ny: int = 10,
                method: str = "bishop",
                n_slices: int = 30,
                tol: float = 1e-4,
                x_entry_range: Optional[Tuple[float, float]] = None,
                x_exit_range: Optional[Tuple[float, float]] = None,
                fos_fn: Optional[Callable] = None,
                ) -> SearchResult:
    """Grid search for the critical circular slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_range : (float, float)
        (x_min, x_max) for circle center search.
    y_range : (float, float)
        (y_min, y_max) for circle center search.
    nx, ny : int
        Grid resolution. Default 10x10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        Convergence tolerance. Default 1e-4.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range.

    Returns
    -------
    SearchResult
    """
    best_fos = _FOS_MAX
    best_xc = 0.0
    best_yc = 0.0
    best_r = 0.0
    grid_fos = []
    n_evaluated = 0

    x_lo, x_hi = x_range
    y_lo, y_hi = y_range

    for ix in range(nx):
        xc = x_lo + (x_hi - x_lo) * ix / max(nx - 1, 1) if nx > 1 else (x_lo + x_hi) / 2
        for iy in range(ny):
            yc = y_lo + (y_hi - y_lo) * iy / max(ny - 1, 1) if ny > 1 else (y_lo + y_hi) / 2

            r_opt, fos_opt = optimize_radius(
                geom, xc, yc, method=method, n_slices=n_slices, tol=tol,
                x_entry_range=x_entry_range, x_exit_range=x_exit_range,
                fos_fn=fos_fn,
            )
            n_evaluated += 1

            grid_fos.append({
                "xc": round(xc, 2),
                "yc": round(yc, 2),
                "R": round(r_opt, 2),
                "FOS": round(fos_opt, 4),
            })

            if fos_opt < best_fos:
                best_fos = fos_opt
                best_xc = xc
                best_yc = yc
                best_r = r_opt

    # Build result for the critical surface
    critical = None
    if best_fos < _FOS_MAX:
        slip = CircularSlipSurface(best_xc, best_yc, best_r)
        try:
            x_entry, x_exit = slip.find_entry_exit(geom)
            slices = build_slices(geom, slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.capitalize(),
                xc=best_xc,
                yc=best_yc,
                radius=best_r,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
    )


def search_noncircular(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_trials: int = 500,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    method: str = "spencer",
    fos_fn: Optional[Callable] = None,
) -> SearchResult:
    """Search for the critical noncircular slip surface using random polylines.

    Generates random polyline slip surfaces within the specified entry/exit
    ranges and evaluates Spencer's FOS for each.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range (where slip enters ground).
    x_exit_range : (float, float)
        Allowed exit x-coordinate range (where slip exits ground).
    n_trials : int
        Number of random trial surfaces. Default 500.
    n_points : int
        Number of polyline vertices (including endpoints). Default 5.
    n_slices : int
        Number of slices per analysis. Default 30.
    tol : float
        Convergence tolerance. Default 1e-4.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SearchResult
    """
    if seed is not None:
        random.seed(seed)

    x_lo_entry, x_hi_entry = x_entry_range
    x_lo_exit, x_hi_exit = x_exit_range
    window_span = _window_span(x_entry_range, x_exit_range)

    # Get ground surface elevation range for depth generation
    z_min_ground = min(z for _, z in geom.surface_points)
    z_max_ground = max(z for _, z in geom.surface_points)
    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)

    best_fos = _FOS_MAX
    best_slip = None
    n_evaluated = 0
    grid_fos = []
    trial_surfaces = []
    reject_stats = _new_reject_stats()

    for _ in range(n_trials):
        # Random entry and exit x-coordinates
        x_entry = random.uniform(x_lo_entry, x_hi_entry)
        x_exit = random.uniform(x_lo_exit, x_hi_exit)

        if x_exit <= x_entry + 1.0:
            continue

        # Ground elevation at entry and exit
        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)

        # Generate intermediate points
        n_interior = max(n_points - 2, 1)
        points = [(x_entry, z_entry)]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)
            # Random depth below ground surface (but above lowest layer)
            max_depth = z_ground - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            depth = random.uniform(0.5, max_depth * 0.9)
            z_pt = z_ground - depth
            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))

        try:
            slip = PolylineSlipSurface(points=points)
            fos = _compute_fos(geom, slip, method, n_slices, tol=tol,
                               window_span=window_span,
                               reject_stats=reject_stats, fos_fn=fos_fn)
            n_evaluated += 1

            grid_fos.append({
                "xc": round(points[len(points) // 2][0], 2),
                "yc": round(points[len(points) // 2][1], 2),
                "R": 0.0,
                "FOS": round(fos, 4),
            })
            if fos < 900:
                trial_surfaces.append({
                    "FOS": round(fos, 4),
                    "points": [(round(px, 3), round(pz, 3))
                               for px, pz in points],
                })

            if fos < best_fos:
                best_fos = fos
                best_slip = slip
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    # Build result
    critical = None
    if best_fos < _FOS_MAX and best_slip is not None:
        try:
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.replace("_", "-").title(),
                xc=0.0,
                yc=0.0,
                radius=0.0,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        trial_surfaces=trial_surfaces,
        **_rejection_kwargs(reject_stats, n_evaluated),
    )


# ── PSO Noncircular Search ────────────────────────────────────────────────

def search_pso(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_particles: int = 30,
    n_iterations: int = 50,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> SearchResult:
    """Particle Swarm Optimization for critical noncircular slip surface.

    Each particle represents a polyline slip surface defined by control
    point coordinates. The swarm evolves toward the minimum FOS surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range.
    x_exit_range : (float, float)
        Allowed exit x-coordinate range.
    n_particles : int
        Swarm size. Default 30.
    n_iterations : int
        Maximum iterations. Default 50.
    n_points : int
        Number of polyline vertices (including endpoints). Default 5.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.
    seed : int, optional
        Random seed for reproducibility.
    w : float
        Inertia weight. Default 0.7.
    c1 : float
        Cognitive (personal best) coefficient. Default 1.5.
    c2 : float
        Social (global best) coefficient. Default 1.5.

    Returns
    -------
    SearchResult

    References
    ----------
    Kennedy & Eberhart (1995) — Particle Swarm Optimization
    """
    if seed is not None:
        random.seed(seed)

    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
    n_interior = max(n_points - 2, 1)
    window_span = _window_span(x_entry_range, x_exit_range)
    reject_stats = _new_reject_stats()

    # Particle = [x_entry, x_exit, depth_frac_1, ..., depth_frac_n_interior]
    dim = 2 + n_interior

    def _random_particle():
        x_entry = random.uniform(*x_entry_range)
        x_exit = random.uniform(*x_exit_range)
        depths = [random.uniform(0.05, 0.95) for _ in range(n_interior)]
        return [x_entry, x_exit] + depths

    def _particle_to_slip(pos):
        x_entry = max(x_entry_range[0], min(x_entry_range[1], pos[0]))
        x_exit = max(x_exit_range[0], min(x_exit_range[1], pos[1]))
        if x_exit <= x_entry + 1.0:
            return None

        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)
        points = [(x_entry, z_entry)]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)
            max_depth = z_ground - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            depth_frac = max(0.05, min(0.95, pos[2 + j]))
            z_pt = z_ground - depth_frac * max_depth
            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))
        try:
            return PolylineSlipSurface(points=points)
        except (ValueError, IndexError):
            return None

    def _evaluate(pos):
        slip = _particle_to_slip(pos)
        if slip is None:
            return _FOS_MAX
        return _compute_fos(geom, slip, "spencer", n_slices, tol=tol,
                            window_span=window_span, reject_stats=reject_stats)

    # Initialize swarm
    positions = [_random_particle() for _ in range(n_particles)]
    velocities = [[random.uniform(-0.3, 0.3) for _ in range(dim)]
                  for _ in range(n_particles)]
    fitness = [_evaluate(p) for p in positions]

    pbest_pos = [list(p) for p in positions]
    pbest_fit = list(fitness)

    gbest_idx = min(range(n_particles), key=lambda i: fitness[i])
    gbest_pos = list(positions[gbest_idx])
    gbest_fit = fitness[gbest_idx]

    n_evaluated = n_particles
    grid_fos = []
    trial_surfaces = []

    for i in range(n_particles):
        slip = _particle_to_slip(positions[i])
        if slip and fitness[i] < _FOS_MAX:
            mid = slip.points[len(slip.points) // 2]
            grid_fos.append({
                "xc": round(mid[0], 2),
                "yc": round(mid[1], 2),
                "R": 0.0,
                "FOS": round(fitness[i], 4),
            })
            trial_surfaces.append({
                "FOS": round(fitness[i], 4),
                "points": [(round(px, 3), round(pz, 3))
                           for px, pz in slip.points],
            })

    for iteration in range(n_iterations):
        w_iter = w - (w - 0.4) * iteration / max(n_iterations - 1, 1)

        for i in range(n_particles):
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                velocities[i][d] = (
                    w_iter * velocities[i][d]
                    + c1 * r1 * (pbest_pos[i][d] - positions[i][d])
                    + c2 * r2 * (gbest_pos[d] - positions[i][d])
                )
                if d < 2:
                    velocities[i][d] = max(-5.0, min(5.0, velocities[i][d]))
                else:
                    velocities[i][d] = max(-0.3, min(0.3, velocities[i][d]))
                positions[i][d] += velocities[i][d]

            positions[i][0] = max(x_entry_range[0], min(x_entry_range[1], positions[i][0]))
            positions[i][1] = max(x_exit_range[0], min(x_exit_range[1], positions[i][1]))
            for d in range(2, dim):
                positions[i][d] = max(0.05, min(0.95, positions[i][d]))

            fos = _evaluate(positions[i])
            n_evaluated += 1
            fitness[i] = fos

            slip = _particle_to_slip(positions[i])
            if slip and fos < _FOS_MAX:
                mid = slip.points[len(slip.points) // 2]
                grid_fos.append({
                    "xc": round(mid[0], 2),
                    "yc": round(mid[1], 2),
                    "R": 0.0,
                    "FOS": round(fos, 4),
                })
                trial_surfaces.append({
                    "FOS": round(fos, 4),
                    "points": [(round(px, 3), round(pz, 3))
                               for px, pz in slip.points],
                })

            if fos < pbest_fit[i]:
                pbest_fit[i] = fos
                pbest_pos[i] = list(positions[i])

            if fos < gbest_fit:
                gbest_fit = fos
                gbest_pos = list(positions[i])

    critical = None
    if gbest_fit < _FOS_MAX:
        best_slip = _particle_to_slip(gbest_pos)
        if best_slip is not None:
            try:
                x_entry, x_exit = best_slip.find_entry_exit(geom)
                slices = build_slices(geom, best_slip, n_slices)
                critical = SlopeStabilityResult(
                    FOS=gbest_fit,
                    method="Spencer",
                    xc=0.0,
                    yc=0.0,
                    radius=0.0,
                    x_entry=x_entry,
                    x_exit=x_exit,
                    n_slices=len(slices),
                    has_seismic=geom.kh > 0,
                    kh=geom.kh,
                    slip_points=list(best_slip.points),
                )
            except ValueError:
                pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        trial_surfaces=trial_surfaces,
        **_rejection_kwargs(reject_stats, n_evaluated),
    )


# ── Weak-Layer Biased Search (SNIFF-inspired) ────────────────────────────

def search_weak_layer_biased(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_trials: int = 500,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
) -> SearchResult:
    """Weak-layer biased random search inspired by SSAP's SNIFF algorithm.

    Biases random surface generation toward passing through soil layers
    with the lowest shear strength. Surfaces are weighted to preferentially
    route through weak zones.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range.
    x_exit_range : (float, float)
        Allowed exit x-coordinate range.
    n_trials : int
        Number of random trial surfaces. Default 500.
    n_points : int
        Number of polyline vertices. Default 5.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SearchResult

    References
    ----------
    Borselli (2002) — SNIFF Random Search concept (SSAP2010)
    """
    if seed is not None:
        random.seed(seed)

    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
    window_span = _window_span(x_entry_range, x_exit_range)
    reject_stats = _new_reject_stats()

    # Compute weakness score for each layer: inverse of shear strength
    layer_weakness = []
    for layer in geom.soil_layers:
        c, phi = layer.shear_strength_params
        strength = c + 50.0 * math.tan(math.radians(phi))
        if strength <= 0:
            strength = 0.1
        layer_weakness.append(1.0 / strength)

    total_weakness = sum(layer_weakness)
    if total_weakness > 0:
        layer_weights = [lw / total_weakness for lw in layer_weakness]
    else:
        layer_weights = [1.0 / len(geom.soil_layers)] * len(geom.soil_layers)

    best_fos = _FOS_MAX
    best_slip = None
    n_evaluated = 0
    grid_fos = []
    trial_surfaces = []

    for _ in range(n_trials):
        x_entry = random.uniform(*x_entry_range)
        x_exit = random.uniform(*x_exit_range)
        if x_exit <= x_entry + 1.0:
            continue

        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)

        n_interior = max(n_points - 2, 1)
        points = [(x_entry, z_entry)]

        target_idx = _weighted_choice(layer_weights)
        target_layer = geom.soil_layers[target_idx]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)

            if random.random() < 0.7:
                layer_mid = (target_layer.top_elevation + target_layer.bottom_elevation) / 2.0
                thickness = target_layer.top_elevation - target_layer.bottom_elevation
                noise = random.gauss(0, max(thickness / 4.0, 0.1))
                z_pt = layer_mid + noise
                z_pt = max(min_layer_bot, min(z_ground - 0.3, z_pt))
            else:
                max_depth = z_ground - min_layer_bot
                if max_depth <= 0:
                    max_depth = 1.0
                depth = random.uniform(0.5, max_depth * 0.9)
                z_pt = z_ground - depth

            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))

        try:
            slip = PolylineSlipSurface(points=points)
            fos = _compute_fos(geom, slip, "spencer", n_slices, tol=tol,
                               window_span=window_span,
                               reject_stats=reject_stats)
            n_evaluated += 1

            grid_fos.append({
                "xc": round(points[len(points) // 2][0], 2),
                "yc": round(points[len(points) // 2][1], 2),
                "R": 0.0,
                "FOS": round(fos, 4),
            })
            if fos < 900:
                trial_surfaces.append({
                    "FOS": round(fos, 4),
                    "points": [(round(px, 3), round(pz, 3))
                               for px, pz in points],
                })

            if fos < best_fos:
                best_fos = fos
                best_slip = slip
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    critical = None
    if best_fos < _FOS_MAX and best_slip is not None:
        try:
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method="Spencer",
                xc=0.0,
                yc=0.0,
                radius=0.0,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        trial_surfaces=trial_surfaces,
        **_rejection_kwargs(reject_stats, n_evaluated),
    )


def _weighted_choice(weights):
    """Weighted random selection, returns index."""
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1


# ── Entry/Exit Region Search ─────────────────────────────────────────────

def search_entry_exit(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_entry: int = 10,
    n_exit: int = 10,
    method: str = "bishop",
    n_slices: int = 30,
    tol: float = 1e-4,
    fos_fn: Optional[Callable] = None,
) -> SearchResult:
    """Entry/exit region search for circular slip surfaces.

    Instead of searching on a (xc, yc) grid, generates trial arcs
    connecting entry and exit points on the ground surface. For each
    entry/exit pair, finds the optimal circle passing through both points.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        x-range where slip enters the ground (typically upslope).
    x_exit_range : (float, float)
        x-range where slip exits the ground (typically at toe).
    n_entry : int
        Number of trial entry points. Default 10.
    n_exit : int
        Number of trial exit points. Default 10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.

    Returns
    -------
    SearchResult
    """
    best_fos = _FOS_MAX
    best_xc = 0.0
    best_yc = 0.0
    best_r = 0.0
    grid_fos = []
    n_evaluated = 0

    entry_xs = [x_entry_range[0] + (x_entry_range[1] - x_entry_range[0]) * i / max(n_entry - 1, 1)
                for i in range(n_entry)] if n_entry > 1 else [(x_entry_range[0] + x_entry_range[1]) / 2]
    exit_xs = [x_exit_range[0] + (x_exit_range[1] - x_exit_range[0]) * i / max(n_exit - 1, 1)
               for i in range(n_exit)] if n_exit > 1 else [(x_exit_range[0] + x_exit_range[1]) / 2]

    for x_en in entry_xs:
        z_en = geom.ground_elevation_at(x_en)
        for x_ex in exit_xs:
            if x_ex <= x_en + 1.0:
                continue
            z_ex = geom.ground_elevation_at(x_ex)

            # Circle center lies on perpendicular bisector of chord
            mid_x = (x_en + x_ex) / 2.0
            mid_z = (z_en + z_ex) / 2.0
            dx = x_ex - x_en
            dz = z_ex - z_en
            chord_len = math.sqrt(dx**2 + dz**2)
            if chord_len < 0.1:
                continue

            perp_x = -dz / chord_len
            perp_z = dx / chord_len
            half_chord = chord_len / 2.0

            t_min = 0.5
            t_max = 3.0 * chord_len
            n_t = 15

            local_best_fos = _FOS_MAX
            local_best_t = t_min

            for it in range(n_t):
                t = t_min + (t_max - t_min) * it / max(n_t - 1, 1)
                xc = mid_x + t * perp_x
                yc = mid_z + t * perp_z
                r = math.sqrt(half_chord**2 + t**2)

                if yc < mid_z:
                    continue

                slip = CircularSlipSurface(xc, yc, r)
                fos = _compute_fos(geom, slip, method, n_slices, tol=tol,
                                   fos_fn=fos_fn)
                n_evaluated += 1

                if fos < local_best_fos:
                    local_best_fos = fos
                    local_best_t = t

            # Golden-section refinement
            margin = (t_max - t_min) / n_t
            a = max(t_min, local_best_t - 2 * margin)
            b = min(t_max, local_best_t + 2 * margin)
            golden = (math.sqrt(5) - 1) / 2

            for _ in range(15):
                if b - a < 0.05:
                    break
                t1 = b - golden * (b - a)
                t2 = a + golden * (b - a)

                xc1 = mid_x + t1 * perp_x
                yc1 = mid_z + t1 * perp_z
                r1 = math.sqrt(half_chord**2 + t1**2)

                xc2 = mid_x + t2 * perp_x
                yc2 = mid_z + t2 * perp_z
                r2 = math.sqrt(half_chord**2 + t2**2)

                f1 = _compute_fos(geom, CircularSlipSurface(xc1, yc1, r1),
                                  method, n_slices, tol=tol, fos_fn=fos_fn)
                f2 = _compute_fos(geom, CircularSlipSurface(xc2, yc2, r2),
                                  method, n_slices, tol=tol, fos_fn=fos_fn)
                n_evaluated += 2

                if f1 < f2:
                    b = t2
                    if f1 < local_best_fos:
                        local_best_fos = f1
                        local_best_t = t1
                else:
                    a = t1
                    if f2 < local_best_fos:
                        local_best_fos = f2
                        local_best_t = t2

            t_final = local_best_t
            xc_f = mid_x + t_final * perp_x
            yc_f = mid_z + t_final * perp_z
            r_f = math.sqrt(half_chord**2 + t_final**2)

            grid_fos.append({
                "xc": round(xc_f, 2),
                "yc": round(yc_f, 2),
                "R": round(r_f, 2),
                "FOS": round(local_best_fos, 4),
            })

            if local_best_fos < best_fos:
                best_fos = local_best_fos
                best_xc = xc_f
                best_yc = yc_f
                best_r = r_f

    critical = None
    if best_fos < _FOS_MAX:
        slip = CircularSlipSurface(best_xc, best_yc, best_r)
        try:
            x_entry, x_exit = slip.find_entry_exit(geom)
            slices = build_slices(geom, slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.capitalize(),
                xc=best_xc,
                yc=best_yc,
                radius=best_r,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
    )


# ── Differential-Evolution Noncircular Refinement ───────────────────────

def search_de(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_points: int = 7,
    method: str = "spencer",
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    maxiter: int = 30,
    popsize: int = 15,
    seed_surface=None,
    n_seed_trials: int = 150,
    fos_fn: Optional[Callable] = None,
) -> SearchResult:
    """Noncircular critical-surface optimization via differential evolution.

    Optimizes a polyline slip surface parameterized as
    [x_entry, x_exit, depth_frac_1..k] (k = n_points - 2 interior
    vertices at evenly spaced x, each a fraction of the local depth to
    the lowest layer bottom). The parameterization enforces monotonic x
    and keeps vertices inside the ground; convex bumps (a vertex above
    the chord of its neighbours by more than 2% of the slope height)
    are penalized to keep surfaces kinematically reasonable.

    Seeded with the best surface from a short random search (or a
    user-provided ``seed_surface``) so DE refines rather than starts
    cold.

    Parameters
    ----------
    geom : SlopeGeometry
    x_entry_range, x_exit_range : (float, float)
        Entry/exit x windows on the ground surface.
    n_points : int
        Total polyline vertices (>= 4 recommended for DE). Default 7.
    method : str
        FOS method for evaluation. Default 'spencer'.
    n_slices, tol, seed : usual meanings.
    maxiter, popsize : int
        scipy differential_evolution budget. Default 30 / 15.
    seed_surface : PolylineSlipSurface, optional
        Starting surface; its vertices are resampled onto this
        parameterization. Default: best of a short random search.
    n_seed_trials : int
        Trials for the seeding random search. Default 150.

    Returns
    -------
    SearchResult

    References
    ----------
    Storn & Price (1997) — Differential Evolution.
    Cheng et al. (2007) — heuristic optimization of noncircular surfaces.
    """
    from scipy.optimize import differential_evolution
    import numpy as np

    rng = np.random.default_rng(seed)
    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
    z_max_ground = max(z for _, z in geom.surface_points)
    slope_h = z_max_ground - min(z for _, z in geom.surface_points)
    n_interior = max(n_points - 2, 1)
    dim = 2 + n_interior
    window_span = _window_span(x_entry_range, x_exit_range)
    reject_stats = _new_reject_stats()

    def _to_slip(v):
        x_en = float(min(max(v[0], x_entry_range[0]), x_entry_range[1]))
        x_ex = float(min(max(v[1], x_exit_range[0]), x_exit_range[1]))
        if x_ex <= x_en + 1.0:
            return None
        pts = [(x_en, geom.ground_elevation_at(x_en))]
        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x = x_en + frac * (x_ex - x_en)
            zg = geom.ground_elevation_at(x)
            max_depth = zg - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            df = float(min(max(v[2 + j], 0.02), 0.95))
            pts.append((x, zg - df * max_depth))
        pts.append((x_ex, geom.ground_elevation_at(x_ex)))
        try:
            return PolylineSlipSurface(points=pts)
        except ValueError:
            return None

    def _objective(v):
        slip = _to_slip(v)
        if slip is None:
            return _FOS_MAX
        fos = _compute_fos(geom, slip, method, n_slices, tol=tol,
                           window_span=window_span, reject_stats=reject_stats,
                           fos_fn=fos_fn)
        # Convexity penalty: vertex above the chord of its neighbours
        pts = slip.points
        pen = 0.0
        bump_tol = 0.02 * max(slope_h, 1.0)
        for i in range(1, len(pts) - 1):
            x0, z0 = pts[i - 1]
            x1, z1 = pts[i + 1]
            t = (pts[i][0] - x0) / (x1 - x0) if x1 > x0 else 0.5
            z_chord = z0 + t * (z1 - z0)
            if pts[i][1] > z_chord + bump_tol:
                pen += 10.0 * (pts[i][1] - z_chord - bump_tol) / max(slope_h, 1.0)
        return fos + pen

    # ---- seeding -----------------------------------------------------------
    if seed_surface is None and n_seed_trials > 0:
        seed_res = search_noncircular(
            geom, x_entry_range, x_exit_range, n_trials=n_seed_trials,
            n_points=n_points, n_slices=n_slices, tol=tol, seed=seed,
            method=method, fos_fn=fos_fn,
        )
        if seed_res.critical is not None and seed_res.critical.slip_points:
            seed_surface = PolylineSlipSurface(
                points=seed_res.critical.slip_points)

    x0 = None
    if seed_surface is not None:
        pts = seed_surface.points
        x_en, x_ex = pts[0][0], pts[-1][0]
        x_en = min(max(x_en, x_entry_range[0]), x_entry_range[1])
        x_ex = min(max(x_ex, x_exit_range[0]), x_exit_range[1])
        v = [x_en, x_ex]
        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x = x_en + frac * (x_ex - x_en)
            z_slip = seed_surface.slip_elevation_at(x)
            zg = geom.ground_elevation_at(x)
            max_depth = zg - min_layer_bot
            df = 0.3 if (z_slip is None or max_depth <= 0) else \
                min(max((zg - z_slip) / max_depth, 0.02), 0.95)
            v.append(df)
        x0 = v

    bounds = ([x_entry_range, x_exit_range]
              + [(0.02, 0.95)] * n_interior)

    # init population: latin hypercube + seed row
    init = "latinhypercube"
    if x0 is not None:
        pop = np.empty((max(popsize, 5) * 1, dim))
        for i in range(pop.shape[0]):
            for d, (lo, hi) in enumerate(bounds):
                pop[i, d] = rng.uniform(lo, hi)
        pop[0, :] = x0
        init = pop

    result = differential_evolution(
        _objective, bounds, init=init, maxiter=maxiter,
        popsize=popsize, seed=seed, tol=1e-6, polish=False,
        mutation=(0.4, 1.0), recombination=0.8,
    )

    n_evaluated = int(result.nfev)
    best_slip = _to_slip(result.x)
    best_fos = float(result.fun)

    critical = None
    if best_slip is not None and best_fos < _FOS_MAX:
        # objective may include penalty; recompute clean FOS
        clean = _compute_fos(geom, best_slip, method, n_slices, tol=tol,
                             window_span=window_span, fos_fn=fos_fn)
        try:
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=clean,
                method=method.replace("_", "-").title(),
                xc=0.0, yc=0.0, radius=0.0,
                x_entry=x_entry, x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0, kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=[],
        **_rejection_kwargs(reject_stats, n_evaluated),
    )
