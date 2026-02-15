"""
Slice discretization for method of slices.

Cuts the soil mass between the slip surface entry and exit into
vertical slices and computes all geometric and force quantities
for each slice.

References:
    Duncan, Wright & Brandon (2014) — Chapter 6
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from geotech_common.water import GAMMA_W
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface


@dataclass
class Slice:
    """Data for a single vertical slice.

    All forces per unit width of slope (kN/m).

    Attributes
    ----------
    x_left : float
        x-coordinate of left side of slice.
    x_right : float
        x-coordinate of right side of slice.
    x_mid : float
        x-coordinate of slice midpoint.
    width : float
        Slice width b (m).
    z_top : float
        Ground surface elevation at slice midpoint.
    z_base : float
        Slip surface elevation at slice midpoint.
    height : float
        Slice height h = z_top - z_base (m).
    alpha : float
        Inclination of slice base to horizontal (radians).
    base_length : float
        Length of slice base dl = width / cos(alpha) (m).
    weight : float
        Total weight W of the slice (kN/m).
    pore_pressure : float
        Average pore pressure u on slice base (kPa).
    c : float
        Cohesion (c' or cu) at slice base (kPa).
    phi : float
        Friction angle (phi' or 0) at slice base (degrees).
    surcharge_force : float
        Vertical surcharge force on slice (kN/m).
    seismic_force : float
        Horizontal seismic force = kh * W (kN/m).
    z_centroid : float
        Elevation of slice centroid (for seismic moment arms).
    """
    x_left: float = 0.0
    x_right: float = 0.0
    x_mid: float = 0.0
    width: float = 0.0
    z_top: float = 0.0
    z_base: float = 0.0
    height: float = 0.0
    alpha: float = 0.0
    base_length: float = 0.0
    weight: float = 0.0
    pore_pressure: float = 0.0
    c: float = 0.0
    phi: float = 0.0
    surcharge_force: float = 0.0
    seismic_force: float = 0.0
    z_centroid: float = 0.0


def build_slices(geom: SlopeGeometry,
                 slip: CircularSlipSurface,
                 n_slices: int = 30) -> List[Slice]:
    """Discretize the sliding mass into vertical slices.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope geometry with surface, layers, and water table.
    slip : CircularSlipSurface
        Trial slip surface.
    n_slices : int
        Number of slices. Default 30.

    Returns
    -------
    list of Slice
    """
    if n_slices < 3:
        raise ValueError(f"Need at least 3 slices, got {n_slices}")

    x_entry, x_exit = slip.find_entry_exit(geom)
    dx = (x_exit - x_entry) / n_slices

    slices = []
    for i in range(n_slices):
        x_left = x_entry + i * dx
        x_right = x_left + dx
        x_mid = (x_left + x_right) / 2.0
        width = dx

        z_top = geom.ground_elevation_at(x_mid)
        z_base = slip.slip_elevation_at(x_mid)
        if z_base is None:
            continue

        height = z_top - z_base
        if height <= 0:
            continue

        alpha = slip.tangent_angle_at(x_mid)
        cos_alpha = math.cos(alpha)
        if abs(cos_alpha) < 1e-10:
            cos_alpha = 1e-10
        base_length = width / abs(cos_alpha)

        # GWT at slice midpoint
        gwt_elev = geom.gwt_elevation_at(x_mid)

        # Weight through multiple layers
        weight = _compute_slice_weight(geom, x_mid, z_top, z_base, width, gwt_elev)

        # Pore pressure at base
        pore_pressure = _pore_pressure_at_base(z_base, gwt_elev)

        # Strength parameters from layer at base midpoint
        base_layer = geom.layer_at_elevation(z_base)
        if base_layer is not None:
            c, phi = base_layer.shear_strength_params
        else:
            # Fallback: use bottom-most layer
            c, phi = geom.soil_layers[-1].shear_strength_params

        # Surcharge
        surcharge_force = geom.surcharge_at(x_mid) * width

        # Seismic horizontal force
        seismic_force = geom.kh * weight

        # Centroid elevation (for seismic moment arm)
        z_centroid = z_base + height / 2.0

        slices.append(Slice(
            x_left=x_left,
            x_right=x_right,
            x_mid=x_mid,
            width=width,
            z_top=z_top,
            z_base=z_base,
            height=height,
            alpha=alpha,
            base_length=base_length,
            weight=weight,
            pore_pressure=pore_pressure,
            c=c,
            phi=phi,
            surcharge_force=surcharge_force,
            seismic_force=seismic_force,
            z_centroid=z_centroid,
        ))

    return slices


def _compute_slice_weight(geom: SlopeGeometry,
                          x_mid: float,
                          z_top: float,
                          z_base: float,
                          width: float,
                          gwt_elev: Optional[float]) -> float:
    """Compute weight of a single slice through multiple soil layers.

    Traverses layers from top to bottom within the slice height,
    using gamma above GWT and gamma_sat below GWT.
    """
    weight = 0.0

    # Collect layers that overlap with (z_base, z_top)
    for layer in geom.soil_layers:
        # Clip layer to slice elevation range
        lay_top = min(layer.top_elevation, z_top)
        lay_bot = max(layer.bottom_elevation, z_base)

        if lay_top <= lay_bot:
            continue

        layer_thickness = lay_top - lay_bot

        if gwt_elev is not None:
            # Part above GWT uses gamma, part below uses gamma_sat
            above_gwt = max(0.0, lay_top - max(gwt_elev, lay_bot))
            below_gwt = max(0.0, min(gwt_elev, lay_top) - lay_bot)
            weight += (above_gwt * layer.gamma + below_gwt * layer.gamma_sat) * width
        else:
            weight += layer_thickness * layer.gamma * width

    # If no layers cover the slice (shouldn't happen normally), estimate
    if weight == 0.0 and z_top > z_base:
        fallback_layer = geom.soil_layers[0]
        weight = (z_top - z_base) * fallback_layer.gamma * width

    return weight


def _pore_pressure_at_base(z_base: float,
                           gwt_elev: Optional[float]) -> float:
    """Compute pore water pressure at the midpoint of the slice base.

    u = gamma_w * (z_gwt - z_base) if z_base < z_gwt, else 0.
    """
    if gwt_elev is None:
        return 0.0
    depth_below_gwt = gwt_elev - z_base
    if depth_below_gwt <= 0:
        return 0.0
    return GAMMA_W * depth_below_gwt
