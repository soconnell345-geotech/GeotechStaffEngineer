"""
Critical slip surface search by grid-center optimization.

For each trial center (xc, yc), optimizes radius R to find the
minimum FOS. Then reports the global minimum across all centers.

References:
    Duncan, Wright & Brandon (2014) — Chapter 14
"""

import math
import warnings
from typing import List, Tuple, Optional, Dict, Any

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import fellenius_fos, bishop_fos, spencer_fos
from slope_stability.results import SearchResult, SlopeStabilityResult


_FOS_MAX = 999.9


def _compute_fos(geom: SlopeGeometry,
                 slip: CircularSlipSurface,
                 method: str,
                 n_slices: int) -> float:
    """Safely compute FOS, returning _FOS_MAX on any error."""
    try:
        slices = build_slices(geom, slip, n_slices)
        if len(slices) < 3:
            return _FOS_MAX

        # Compute nail contributions if nails are defined
        nail_contribs = None
        if geom.nails:
            from slope_stability.nails import compute_all_nail_contributions
            nail_contribs = compute_all_nail_contributions(
                geom.nails, slip.xc, slip.yc, slip.radius
            )

        if method == "fellenius":
            return fellenius_fos(slices, slip, nail_contributions=nail_contribs)
        elif method == "bishop":
            return bishop_fos(slices, slip, nail_contributions=nail_contribs)
        elif method == "spencer":
            fos, _ = spencer_fos(slices, slip, nail_contributions=nail_contribs)
            return fos
        else:
            return bishop_fos(slices, slip, nail_contributions=nail_contribs)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return _FOS_MAX


def optimize_radius(geom: SlopeGeometry,
                    xc: float, yc: float,
                    method: str = "bishop",
                    n_slices: int = 30,
                    r_min: Optional[float] = None,
                    r_max: Optional[float] = None,
                    n_radii: int = 20) -> Tuple[float, float]:
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

    Returns
    -------
    (R_opt, FOS_min) : tuple
        Optimal radius and minimum FOS.
    """
    # Auto-compute radius bounds
    if r_min is None or r_max is None:
        # Minimum: center must be above ground, radius reaches ground
        zs = [p[1] for p in geom.surface_points]
        z_min_ground = min(zs)
        z_max_ground = max(zs)

        # R_min: circle must at least reach the ground surface below center
        dist_to_ground = yc - z_max_ground
        if dist_to_ground <= 0:
            # Center is at or below ground — need larger radius
            dist_to_ground = 0.5
        if r_min is None:
            r_min = max(dist_to_ground + 0.5, 1.0)

        # R_max: circle should not extend far below lowest layer
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
        fos = _compute_fos(geom, slip, method, n_slices)
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
        f1 = _compute_fos(geom, slip1, method, n_slices)
        f2 = _compute_fos(geom, slip2, method, n_slices)

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
                n_slices: int = 30) -> SearchResult:
    """Grid search for the critical slip surface.

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
                geom, xc, yc, method=method, n_slices=n_slices,
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
