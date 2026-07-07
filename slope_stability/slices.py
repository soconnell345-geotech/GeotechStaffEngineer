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
from slope_stability.geometry import (
    SlopeGeometry, SlopeSoilLayer, build_pore_pressure_interpolator,
)


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
    in_tension_crack: bool = False
    crack_water_force: float = 0.0
    crack_water_z: float = 0.0
    pond_force: float = 0.0      # vertical pond water weight (also in weight)
    pond_hforce: float = 0.0     # signed horizontal pond thrust, +x (kN/m)
    pond_z: float = 0.0          # elevation of pond thrust line of action


@dataclass
class SliceForces:
    """Computed forces on a single slice (Fellenius decomposition).

    All forces per unit width of slope (kN/m).

    Attributes
    ----------
    W : float
        Total weight including surcharge (kN/m).
    N_prime : float
        Effective normal force on base = W*cos(alpha) - U (kN/m).
        Clamped to zero if tensile.
    S_mobilized : float
        Driving shear = W*sin(alpha) (kN/m).
    T_available : float
        Available shear resistance = c*dl + N'*tan(phi) (kN/m).
    U : float
        Pore water force on base = u*dl (kN/m).
    surcharge : float
        Vertical surcharge force (kN/m).
    seismic : float
        Horizontal seismic force (kN/m).
    alpha_deg : float
        Base inclination angle (degrees).
    """
    W: float
    N_prime: float
    S_mobilized: float
    T_available: float
    U: float
    surcharge: float
    seismic: float
    alpha_deg: float


def compute_slice_forces(s: Slice) -> SliceForces:
    """Compute Fellenius-style forces acting on a single slice.

    Parameters
    ----------
    s : Slice
        Slice with geometry and strength data from build_slices().

    Returns
    -------
    SliceForces
        All computed forces on the slice.
    """
    alpha_rad = s.alpha
    alpha_deg = math.degrees(alpha_rad)

    W = s.weight + s.surcharge_force
    U = s.pore_pressure * s.base_length
    N_prime = W * math.cos(alpha_rad) - U
    if N_prime < 0.0:
        N_prime = 0.0

    S_mobilized = W * math.sin(alpha_rad)
    phi_rad = math.radians(s.phi)
    T_available = s.c * s.base_length + N_prime * math.tan(phi_rad)

    return SliceForces(
        W=W,
        N_prime=N_prime,
        S_mobilized=S_mobilized,
        T_available=T_available,
        U=U,
        surcharge=s.surcharge_force,
        seismic=s.seismic_force,
        alpha_deg=alpha_deg,
    )


def build_slices(geom: SlopeGeometry,
                 slip,
                 n_slices: int = 30) -> List[Slice]:
    """Discretize the sliding mass into vertical slices.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope geometry with surface, layers, and water table.
    slip : CircularSlipSurface or PolylineSlipSurface
        Trial slip surface (duck-typed — must have slip_elevation_at,
        tangent_angle_at, and find_entry_exit methods).
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

    # Discrete pore-pressure field (flow-net / TIN): build the interpolator once
    # per call for the hot path; None when the piezometric-line / ru model is used.
    pp_interp = (build_pore_pressure_interpolator(geom.pore_pressure_points)
                 if geom.pore_pressure_points is not None else None)

    # Tension crack: compute the crack base elevation at the crest end of the
    # slip surface (entry = low-x by default, or exit = high-x). Only slices on
    # that half of the surface can be in the crack.
    crack_base_elev = None
    x_midpoint = (x_entry + x_exit) / 2.0
    crack_side = geom.tension_crack_side
    crack_truncate = geom.tension_crack_model == "truncation"
    if geom.tension_crack_depth > 0:
        x_crest = x_entry if crack_side == "entry" else x_exit
        crack_base_elev = geom.ground_elevation_at(x_crest) - geom.tension_crack_depth

    slices = []
    below_base_x = []   # x of slices whose base dips below ALL soil layers
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

        # Slip surface below all soil layers = cutting through empty
        # space. Edge slivers are skipped; an INTERIOR hole makes the
        # trial surface invalid (SS-5) — previously the middle slices
        # were silently dropped and the remaining fragments produced a
        # meaningless (often absurdly low) FOS that could win searches.
        if geom.layer_at_point(x_mid, z_base) is None:
            min_layer_bot = min(L.bottom_at(x_mid) for L in geom.soil_layers)
            if z_base < min_layer_bot:
                below_base_x.append(x_mid)
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

        # Ponded water (auto-detected): GWT above the ground surface means
        # standing water on the slice. The hydrostatic pressure on the
        # submerged ground-surface segment resolves into
        #   - a vertical component = weight of the water column
        #     (added to the slice weight), and
        #   - a signed horizontal component on inclined ground
        #     Fx = gamma_w * (d_l^2 - d_r^2)/2  (d = pool depth at the
        #     slice edges, clipped at 0), pushing toward the soil — the
        #     classic external-water buttress.
        # The pore pressure at the base already uses the full head to the
        # pool surface, so with both components a fully submerged slope
        # reproduces the buoyant-weight equivalence (boundary water
        # pressures + internal u == buoyancy).
        pond_weight = 0.0
        pond_hforce = 0.0
        pond_z = z_top
        if gwt_elev is not None and gwt_elev > z_top:
            pond_weight = GAMMA_W * (gwt_elev - z_top) * width
        if gwt_elev is not None:
            d_l = max(gwt_elev - geom.ground_elevation_at(x_left), 0.0)
            d_r = max(gwt_elev - geom.ground_elevation_at(x_right), 0.0)
            if d_l > 0.0 or d_r > 0.0:
                pond_hforce = 0.5 * GAMMA_W * (d_l * d_l - d_r * d_r)
                if abs(pond_hforce) > 1e-12:
                    # line of action: centroid of the trapezoidal pressure
                    # diagram on the projected vertical face
                    depth = (2.0 * (d_l ** 3 - d_r ** 3)
                             / (3.0 * (d_l * d_l - d_r * d_r)))
                    pond_z = gwt_elev - depth

        # Pore pressure at base
        pore_pressure = _pore_pressure_at_base(z_base, gwt_elev)

        # Strength parameters from layer at base midpoint
        base_layer = geom.layer_at_point(x_mid, z_base)
        if base_layer is None:
            # Fallback: use bottom-most layer
            base_layer = geom.soil_layers[-1]
        # Apply Ru if no GWT pore pressure and layer has Ru > 0
        if pp_interp is None and pore_pressure <= 0 and base_layer.ru > 0:
            # u = Ru * gamma * h  where h = overburden depth
            overburden_h = z_top - z_base
            pore_pressure = base_layer.ru * base_layer.gamma * overburden_h

        # Discrete pore-pressure field overrides the piezometric-line / ru value
        # at the slice base (the ponded-water buttress above still uses gwt_elev).
        if pp_interp is not None:
            pore_pressure = pp_interp(x_mid, z_base)

        # Strength model evaluation (SHANSEP / Hoek-Brown need stress
        # estimates at the base: Fellenius normal and vertical effective)
        if base_layer.strength_model == "mohr_coulomb":
            c, phi = base_layer.shear_strength_params
        else:
            sigma_v = (weight / width if width > 0 else 0.0) - pore_pressure
            alpha_cos = math.cos(alpha)
            dl = width / max(abs(alpha_cos), 1e-10)
            sigma_n = (weight * alpha_cos / dl if dl > 0 else 0.0) \
                - pore_pressure
            c, phi = base_layer.strength_at(sigma_n, sigma_v)

        # Tension crack: slices on the CREST half whose base is above the crack
        # bottom either lose shear resistance (strength model — open crack face)
        # or are removed from the sliding mass (truncation model — the mass ends
        # at the vertical crack face). The x-position guard scopes it to the
        # crest side (entry = low-x default, exit = high-x).
        in_crack = False
        if crack_base_elev is not None and z_base >= crack_base_elev:
            on_crest_side = (x_mid <= x_midpoint if crack_side == "entry"
                             else x_mid >= x_midpoint)
            if on_crest_side:
                if crack_truncate:
                    continue           # truncate the mass at the crack face
                in_crack = True
                c = 0.0
                phi = 0.0

        # Surcharge
        surcharge_force = geom.surcharge_at(x_mid) * width

        # Seismic horizontal force — applied to the SOIL mass only
        # (hydrodynamic effects on ponded water are not modeled)
        seismic_force = geom.kh * weight

        # add pond water weight after seismic so kh acts on soil only
        weight = weight + pond_weight

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
            in_tension_crack=in_crack,
            pond_force=pond_weight,
            pond_hforce=pond_hforce,
            pond_z=pond_z,
        ))

    # Reject surfaces that dip below the soil model between valid slices
    # (SS-5). A surface deeper than the lowest layer bottom must be
    # re-specified (smaller circle, or model the deeper material).
    if below_base_x and slices:
        x_first = slices[0].x_mid
        x_last = slices[-1].x_mid
        if any(x_first < xb < x_last for xb in below_base_x):
            raise ValueError(
                "Slip surface passes below the bottom of the deepest soil "
                f"layer between x={x_first:.2f} and x={x_last:.2f}. "
                "Use a shallower trial surface or extend the soil layers."
            )

    # Tension crack water force: hydrostatic thrust on the retained face,
    # F_w = 0.5 * gamma_w * z_w^2 acting at z_w/3 above the crack base (treated
    # downstream as an always-driving magnitude). Apply it to the retained slice
    # adjacent to the crack face: the first non-cracked slice for an entry-side
    # crack, the last non-cracked for an exit-side crack (for truncation the
    # cracked slices are already gone, so this is the boundary retained slice).
    if crack_base_elev is not None and geom.tension_crack_water_depth > 0 and slices:
        z_w = geom.tension_crack_water_depth
        crack_water_f = 0.5 * GAMMA_W * z_w ** 2
        crack_water_elev = crack_base_elev + z_w / 3.0
        target = None
        ordered = slices if crack_side == "entry" else list(reversed(slices))
        for s in ordered:
            if not s.in_tension_crack:
                target = s
                break
        if target is None:
            target = ordered[-1]
        target.crack_water_force = crack_water_f
        target.crack_water_z = crack_water_elev

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
        lay_top = min(layer.top_at(x_mid), z_top)
        lay_bot = max(layer.bottom_at(x_mid), z_base)

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

    ASSUMPTION (SS-3): the piezometric head is taken as the full vertical
    distance from the slice base to the GWT surface (hydrostatic, no-flow
    conditions). For seepage parallel to an inclined phreatic surface the
    rigorous head is reduced (factor ~cos^2(beta) of the phreatic slope,
    or per a flownet), so this slightly OVERESTIMATES u — conservative —
    on slopes with steeply inclined water tables. Use the layer ``ru``
    coefficient as the alternative pore-pressure model when a calibrated
    value is available.
    """
    if gwt_elev is None:
        return 0.0
    depth_below_gwt = gwt_elev - z_base
    if depth_below_gwt <= 0:
        return 0.0
    return GAMMA_W * depth_below_gwt
