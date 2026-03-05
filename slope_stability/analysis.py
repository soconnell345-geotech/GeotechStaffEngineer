"""
Top-level slope stability analysis orchestrator.

Provides analyze_slope() and search_critical_surface() as the
primary entry points, following the project pattern of analyze_*()
functions returning result dataclasses.

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
"""

import math
from typing import Optional, Tuple

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices, compute_slice_forces
from slope_stability.methods import fellenius_fos, bishop_fos, spencer_fos
from slope_stability.search import grid_search, search_noncircular
from slope_stability.results import (
    SlopeStabilityResult, SliceData, SearchResult,
)


def analyze_slope(geom: SlopeGeometry,
                  xc: float = None,
                  yc: float = None,
                  radius: float = None,
                  slip_surface=None,
                  method: str = "bishop",
                  n_slices: int = 30,
                  tol: float = 1e-4,
                  include_slice_data: bool = False,
                  compare_methods: bool = False,
                  ) -> SlopeStabilityResult:
    """Run slope stability analysis for a specified slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Complete slope definition.
    xc : float, optional
        Circle center x-coordinate (m). Used with yc/radius for circular.
    yc : float, optional
        Circle center z-coordinate / elevation (m).
    radius : float, optional
        Circle radius (m).
    slip_surface : CircularSlipSurface or PolylineSlipSurface, optional
        Explicit slip surface object. If provided, xc/yc/radius are ignored.
    method : str
        'fellenius', 'bishop', or 'spencer'. Default 'bishop'.
    n_slices : int
        Number of slices. Default 30.
    tol : float
        Convergence tolerance for iterative methods. Default 1e-4.
    include_slice_data : bool
        If True, include per-slice breakdown in results.
    compare_methods : bool
        If True, compute FOS for all three methods.
        For noncircular surfaces, only Fellenius and Spencer are compared
        (Bishop requires circular).

    Returns
    -------
    SlopeStabilityResult
    """
    # Build slip surface from either explicit object or xc/yc/radius
    if slip_surface is not None:
        slip = slip_surface
    else:
        if xc is None or yc is None or radius is None:
            raise ValueError(
                "Either provide slip_surface or all of xc, yc, radius"
            )
        slip = CircularSlipSurface(xc, yc, radius)

    is_circular = getattr(slip, 'is_circular', True)
    x_entry, x_exit = slip.find_entry_exit(geom)
    slices = build_slices(geom, slip, n_slices)

    # For result: store circle params if circular, zeros otherwise
    r_xc = slip.xc if is_circular else 0.0
    r_yc = slip.yc if is_circular else 0.0
    r_radius = slip.radius if is_circular else 0.0
    r_slip_points = None if is_circular else list(slip.points)

    # For noncircular, force Spencer (Bishop doesn't work)
    if not is_circular and method == "bishop":
        method = "spencer"

    # Primary FOS
    theta_spencer = None
    if method == "fellenius":
        fos = fellenius_fos(slices, slip)
        method_name = "Fellenius"
    elif method == "spencer":
        fos, theta = spencer_fos(slices, slip, tol=tol)
        theta_spencer = theta
        method_name = "Spencer"
    else:
        fos = bishop_fos(slices, slip, tol=tol)
        method_name = "Bishop"

    # Comparison FOS values
    fos_fellenius = None
    fos_bishop = None
    if compare_methods:
        fos_fellenius = fellenius_fos(slices, slip)
        if is_circular:
            fos_bishop = bishop_fos(slices, slip, tol=tol)
        if theta_spencer is None:
            fos_sp, theta = spencer_fos(slices, slip, tol=tol)
            theta_spencer = theta

    # Slice data for plotting
    slice_data = None
    if include_slice_data:
        slice_data = []
        for s in slices:
            sf = compute_slice_forces(s)
            dl = s.base_length
            sigma_n = sf.N_prime / dl if dl > 0 else 0.0
            tau_mob = sf.S_mobilized / dl if dl > 0 else 0.0
            tau_avail = sf.T_available / dl if dl > 0 else 0.0
            slice_data.append(SliceData(
                x_mid=s.x_mid,
                z_top=s.z_top,
                z_base=s.z_base,
                width=s.width,
                height=s.height,
                alpha_deg=math.degrees(s.alpha),
                weight=s.weight,
                pore_pressure=s.pore_pressure,
                c=s.c,
                phi=s.phi,
                base_length=s.base_length,
                normal_stress_kPa=sigma_n,
                shear_stress_kPa=tau_mob,
                shear_resistance_kPa=tau_avail,
            ))

    return SlopeStabilityResult(
        FOS=fos,
        method=method_name,
        xc=r_xc,
        yc=r_yc,
        radius=r_radius,
        x_entry=x_entry,
        x_exit=x_exit,
        theta_spencer=theta_spencer,
        FOS_fellenius=fos_fellenius,
        FOS_bishop=fos_bishop,
        n_slices=len(slices),
        has_seismic=geom.kh > 0,
        kh=geom.kh,
        slice_data=slice_data,
        slip_points=r_slip_points,
    )


def search_critical_surface(
    geom: SlopeGeometry,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    nx: int = 10,
    ny: int = 10,
    method: str = "bishop",
    n_slices: int = 30,
    tol: float = 1e-4,
    surface_type: str = "circular",
    x_entry_range: Tuple[float, float] = None,
    x_exit_range: Tuple[float, float] = None,
    n_trials: int = 500,
    n_points: int = 5,
    seed: Optional[int] = None,
) -> SearchResult:
    """Search for the critical slip surface (minimum FOS).

    Auto-computes search bounds from slope geometry if not provided.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_range : (float, float), optional
        Circle center x search range (circular only).
    y_range : (float, float), optional
        Circle center y search range (circular only).
    nx, ny : int
        Grid resolution. Default 10x10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        Convergence tolerance for iterative methods. Default 1e-4.
    surface_type : str
        'circular' (default) or 'noncircular'.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range.
    n_trials : int
        Number of random trials for noncircular search. Default 500.
    n_points : int
        Number of polyline vertices for noncircular search. Default 5.
    seed : int, optional
        Random seed for noncircular search reproducibility.

    Returns
    -------
    SearchResult
    """
    if surface_type == "noncircular":
        # Auto-compute entry/exit ranges from slope geometry if not provided
        x_min = geom.surface_points[0][0]
        x_max = geom.surface_points[-1][0]
        if x_entry_range is None:
            x_entry_range = (x_min, x_min + (x_max - x_min) * 0.4)
        if x_exit_range is None:
            x_exit_range = (x_min + (x_max - x_min) * 0.6, x_max)

        return search_noncircular(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_trials=n_trials,
            n_points=n_points,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
        )

    # Circular search
    if x_range is None:
        x_min = geom.surface_points[0][0]
        x_max = geom.surface_points[-1][0]
        x_range = (x_min, x_max)

    if y_range is None:
        z_min = min(z for _, z in geom.surface_points)
        z_max = max(z for _, z in geom.surface_points)
        slope_height = z_max - z_min
        y_range = (z_max + 1.0, z_max + 2.0 * slope_height)

    result = grid_search(geom, x_range, y_range, nx, ny, method, n_slices,
                         tol=tol, x_entry_range=x_entry_range,
                         x_exit_range=x_exit_range)

    return result
