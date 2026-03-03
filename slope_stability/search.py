"""
Critical slip surface search — circular grid search and noncircular random search.

For circular: grid of trial centers (xc, yc), optimizes radius R at each.
For noncircular: random polyline generation with Spencer's method.

References:
    Duncan, Wright & Brandon (2014) — Chapter 14
"""

import math
import random
import warnings
from typing import List, Tuple, Optional, Dict, Any

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import fellenius_fos, bishop_fos, spencer_fos
from slope_stability.results import SearchResult, SlopeStabilityResult


_FOS_MAX = 999.9


def _compute_fos(geom: SlopeGeometry,
                 slip,
                 method: str,
                 n_slices: int,
                 tol: float = 1e-4) -> float:
    """Safely compute FOS, returning _FOS_MAX on any error."""
    try:
        slices = build_slices(geom, slip, n_slices)
        if len(slices) < 3:
            return _FOS_MAX

        if method == "fellenius":
            fos = fellenius_fos(slices, slip)
        elif method == "bishop":
            if not getattr(slip, 'is_circular', True):
                # Bishop can't handle noncircular, fall back to Spencer
                fos, _ = spencer_fos(slices, slip, tol=tol)
            else:
                fos = bishop_fos(slices, slip, tol=tol)
        elif method == "spencer":
            fos, _ = spencer_fos(slices, slip, tol=tol)
        else:
            fos = bishop_fos(slices, slip, tol=tol)

        # Guard against non-convergence producing negative or absurd values
        # FOS < 0.05 is physically meaningless and indicates numerical issues
        if fos < 0.05 or math.isnan(fos) or math.isinf(fos):
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

        fos = _compute_fos(geom, slip, method, n_slices, tol=tol)
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

        f1 = _compute_fos(geom, slip1, method, n_slices, tol=tol) if ok1 else _FOS_MAX
        f2 = _compute_fos(geom, slip2, method, n_slices, tol=tol) if ok2 else _FOS_MAX

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

    # Get ground surface elevation range for depth generation
    z_min_ground = min(z for _, z in geom.surface_points)
    z_max_ground = max(z for _, z in geom.surface_points)
    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)

    best_fos = _FOS_MAX
    best_slip = None
    n_evaluated = 0
    grid_fos = []

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
            fos = _compute_fos(geom, slip, "spencer", n_slices, tol=tol)
            n_evaluated += 1

            grid_fos.append({
                "xc": round(points[len(points) // 2][0], 2),
                "yc": round(points[len(points) // 2][1], 2),
                "R": 0.0,
                "FOS": round(fos, 4),
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
                method="Spencer",
                xc=0.0,
                yc=0.0,
                radius=0.0,
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
