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
from slope_stability.slices import build_slices
from slope_stability.methods import fellenius_fos, bishop_fos, spencer_fos
from slope_stability.search import grid_search
from slope_stability.results import (
    SlopeStabilityResult, SliceData, SearchResult,
)


def analyze_slope(geom: SlopeGeometry,
                  xc: float,
                  yc: float,
                  radius: float,
                  method: str = "bishop",
                  n_slices: int = 30,
                  FOS_required: float = 1.5,
                  include_slice_data: bool = False,
                  compare_methods: bool = False,
                  ) -> SlopeStabilityResult:
    """Run slope stability analysis for a specified slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Complete slope definition.
    xc : float
        Circle center x-coordinate (m).
    yc : float
        Circle center z-coordinate / elevation (m).
    radius : float
        Circle radius (m).
    method : str
        'fellenius', 'bishop', or 'spencer'. Default 'bishop'.
    n_slices : int
        Number of slices. Default 30.
    FOS_required : float
        Minimum required FOS for pass/fail. Default 1.5.
    include_slice_data : bool
        If True, include per-slice breakdown in results.
    compare_methods : bool
        If True, compute FOS for all three methods.

    Returns
    -------
    SlopeStabilityResult
    """
    slip = CircularSlipSurface(xc, yc, radius)
    x_entry, x_exit = slip.find_entry_exit(geom)
    slices = build_slices(geom, slip, n_slices)

    # Compute nail contributions if nails are defined
    nail_contribs = None
    if geom.nails:
        from slope_stability.nails import compute_all_nail_contributions
        nail_contribs = compute_all_nail_contributions(geom.nails, xc, yc, radius)

    # Primary FOS
    theta_spencer = None
    if method == "fellenius":
        fos = fellenius_fos(slices, slip, nail_contributions=nail_contribs)
        method_name = "Fellenius"
    elif method == "spencer":
        fos, theta = spencer_fos(slices, slip, nail_contributions=nail_contribs)
        theta_spencer = theta
        method_name = "Spencer"
    else:
        fos = bishop_fos(slices, slip, nail_contributions=nail_contribs)
        method_name = "Bishop"

    # Comparison FOS values
    fos_fellenius = None
    fos_bishop = None
    if compare_methods:
        fos_fellenius = fellenius_fos(slices, slip, nail_contributions=nail_contribs)
        fos_bishop = bishop_fos(slices, slip, nail_contributions=nail_contribs)
        if theta_spencer is None:
            fos_sp, theta = spencer_fos(slices, slip, nail_contributions=nail_contribs)
            theta_spencer = theta

    # Slice data for plotting
    slice_data = None
    if include_slice_data:
        slice_data = [
            SliceData(
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
            )
            for s in slices
        ]

    # Nail summary fields
    n_nails_active = 0
    nail_resisting_kN_per_m = 0.0
    if nail_contribs:
        from slope_stability.nails import total_nail_resisting
        n_nails_active = len(nail_contribs)
        nail_resisting_kN_per_m = total_nail_resisting(nail_contribs)

    return SlopeStabilityResult(
        FOS=fos,
        method=method_name,
        xc=xc,
        yc=yc,
        radius=radius,
        x_entry=x_entry,
        x_exit=x_exit,
        is_stable=fos >= FOS_required,
        FOS_required=FOS_required,
        theta_spencer=theta_spencer,
        FOS_fellenius=fos_fellenius,
        FOS_bishop=fos_bishop,
        n_slices=len(slices),
        has_seismic=geom.kh > 0,
        kh=geom.kh,
        slice_data=slice_data,
        n_nails_active=n_nails_active,
        nail_resisting_kN_per_m=nail_resisting_kN_per_m,
    )


def search_critical_surface(
    geom: SlopeGeometry,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    nx: int = 10,
    ny: int = 10,
    method: str = "bishop",
    n_slices: int = 30,
    FOS_required: float = 1.5,
) -> SearchResult:
    """Search for the critical slip surface (minimum FOS).

    Auto-computes search bounds from slope geometry if not provided.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_range : (float, float), optional
        Circle center x search range.
    y_range : (float, float), optional
        Circle center y search range.
    nx, ny : int
        Grid resolution. Default 10x10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    FOS_required : float
        Required FOS for pass/fail.

    Returns
    -------
    SearchResult
    """
    # Auto-compute search bounds
    if x_range is None:
        x_min = geom.surface_points[0][0]
        x_max = geom.surface_points[-1][0]
        x_range = (x_min, x_max)

    if y_range is None:
        z_min = min(z for _, z in geom.surface_points)
        z_max = max(z for _, z in geom.surface_points)
        slope_height = z_max - z_min
        y_range = (z_max + 1.0, z_max + 2.0 * slope_height)

    result = grid_search(geom, x_range, y_range, nx, ny, method, n_slices)

    # Set pass/fail on critical result
    if result.critical is not None:
        result.critical.FOS_required = FOS_required
        result.critical.is_stable = result.critical.FOS >= FOS_required

    return result
