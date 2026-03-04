"""
Convert DXF parse results into analysis inputs.

Step 3 of the discover-then-parse workflow. Takes a DxfParseResult (coordinates)
plus user-supplied soil properties and assembles either:
  - SlopeGeometry for slope_stability analysis
  - fem2d input dicts for FEM analysis

DXF provides geometry only — soil strength parameters (gamma, phi, c') and
stiffness parameters (E, nu) must always come from the user.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from dxf_import.results import DxfParseResult
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.nails import SoilNail


@dataclass
class SoilPropertyAssignment:
    """User-supplied soil properties for a named layer.

    Parameters
    ----------
    name : str
        Must match a soil_boundaries value from LayerMapping, or
        "Surface" for the topmost layer (above first boundary).
    gamma : float
        Total unit weight (kN/m3).
    gamma_sat : float, optional
        Saturated unit weight (kN/m3). If None, uses gamma.
    phi : float
        Effective friction angle (degrees).
    c_prime : float
        Effective cohesion (kPa).
    cu : float
        Undrained shear strength (kPa).
    analysis_mode : str
        'drained' or 'undrained'.
    """
    name: str = ""
    gamma: float = 18.0
    gamma_sat: Optional[float] = None
    phi: float = 0.0
    c_prime: float = 0.0
    cu: float = 0.0
    analysis_mode: str = "drained"


def build_slope_geometry(
    parse_result: DxfParseResult,
    soil_properties: List[SoilPropertyAssignment],
    nail_defaults: Optional[Dict[str, float]] = None,
) -> SlopeGeometry:
    """Assemble a SlopeGeometry from DXF coordinates and soil properties.

    Parameters
    ----------
    parse_result : DxfParseResult
        Output from parse_dxf_geometry().
    soil_properties : list of SoilPropertyAssignment
        One entry per soil layer. Names must match boundary soil names.
        Include a "Surface" entry for the topmost layer if boundary
        profiles define the bottom of the top layer.
    nail_defaults : dict, optional
        Default nail parameters: {bar_diameter, drill_hole_diameter, fy,
        bond_stress, spacing_h, inclination}. Applied to all nails
        from DXF LINE entities.

    Returns
    -------
    SlopeGeometry
        Ready for analyze_slope() or search_critical_surface().

    Raises
    ------
    ValueError
        If soil property names don't match boundary names, or geometry invalid.
    """
    if not parse_result.surface_points:
        raise ValueError("No surface points in parse result")

    if not soil_properties:
        raise ValueError("At least one SoilPropertyAssignment is required")

    # Build property lookup
    prop_lookup = {sp.name: sp for sp in soil_properties}

    # Determine layer structure
    # The surface profile is the top of the first (topmost) layer.
    # Each boundary profile defines the bottom of one layer and top of the next.
    # Order boundaries by descending max elevation.
    boundary_names = list(parse_result.boundary_profiles.keys())
    if boundary_names:
        boundary_names.sort(
            key=lambda n: -max(z for _, z in parse_result.boundary_profiles[n])
        )

    surface_pts = parse_result.surface_points
    surface_z_max = max(z for _, z in surface_pts)
    surface_z_min = min(z for _, z in surface_pts)

    soil_layers = []

    if not boundary_names:
        # Single layer: surface top to surface bottom
        name = soil_properties[0].name
        sp = soil_properties[0]
        soil_layers.append(SlopeSoilLayer(
            name=name,
            top_elevation=surface_z_max,
            bottom_elevation=surface_z_min - 5.0,  # extend below surface
            gamma=sp.gamma,
            gamma_sat=sp.gamma_sat,
            phi=sp.phi,
            c_prime=sp.c_prime,
            cu=sp.cu,
            analysis_mode=sp.analysis_mode,
        ))
    else:
        # Multi-layer: surface → boundary1 → boundary2 → ...
        # Top layer: from surface max to first boundary max
        first_boundary = parse_result.boundary_profiles[boundary_names[0]]
        first_boundary_z_max = max(z for _, z in first_boundary)

        # Find the "Surface" or topmost property
        top_name = "Surface"
        if top_name not in prop_lookup:
            # Use the first property that isn't a boundary name
            for sp in soil_properties:
                if sp.name not in boundary_names:
                    top_name = sp.name
                    break
            else:
                top_name = soil_properties[0].name

        if top_name in prop_lookup:
            sp = prop_lookup[top_name]
            soil_layers.append(SlopeSoilLayer(
                name=top_name,
                top_elevation=surface_z_max,
                bottom_elevation=first_boundary_z_max,
                gamma=sp.gamma,
                gamma_sat=sp.gamma_sat,
                phi=sp.phi,
                c_prime=sp.c_prime,
                cu=sp.cu,
                analysis_mode=sp.analysis_mode,
            ))
            # Attach first boundary polyline as bottom
            first_boundary_sorted = sorted(first_boundary, key=lambda p: p[0])
            soil_layers[-1].bottom_boundary_points = first_boundary_sorted

        # Middle and bottom layers
        for i, bname in enumerate(boundary_names):
            boundary_pts = parse_result.boundary_profiles[bname]
            boundary_z_max = max(z for _, z in boundary_pts)
            if i + 1 < len(boundary_names):
                next_boundary = parse_result.boundary_profiles[
                    boundary_names[i + 1]
                ]
                bottom_z = max(z for _, z in next_boundary)
            else:
                bottom_z = min(z for _, z in boundary_pts) - 2.0

            if bname not in prop_lookup:
                raise ValueError(
                    f"No SoilPropertyAssignment for boundary '{bname}'. "
                    f"Available: {sorted(prop_lookup.keys())}"
                )
            sp = prop_lookup[bname]
            soil_layers.append(SlopeSoilLayer(
                name=bname,
                top_elevation=boundary_z_max,
                bottom_elevation=bottom_z,
                gamma=sp.gamma,
                gamma_sat=sp.gamma_sat,
                phi=sp.phi,
                c_prime=sp.c_prime,
                cu=sp.cu,
                analysis_mode=sp.analysis_mode,
            ))
            # Attach polyline boundary if available
            boundary_pts_sorted = sorted(boundary_pts, key=lambda p: p[0])
            soil_layers[-1].bottom_boundary_points = boundary_pts_sorted

    # --- GWT ---
    gwt_points = parse_result.gwt_points

    # --- Nails ---
    nails = None
    if parse_result.nail_lines:
        defaults = nail_defaults or {}
        nails = []
        for nl in parse_result.nail_lines:
            dx = nl["x_tip"] - nl["x_head"]
            dz = nl["z_tip"] - nl["z_head"]
            length = math.sqrt(dx ** 2 + dz ** 2)
            # Inclination: degrees below horizontal (positive = downward)
            if length > 1e-6:
                inclination = math.degrees(math.atan2(-dz, dx))
            else:
                inclination = defaults.get("inclination", 15.0)

            nails.append(SoilNail(
                x_head=nl["x_head"],
                z_head=nl["z_head"],
                length=length,
                inclination=inclination,
                bar_diameter=defaults.get("bar_diameter", 25.0),
                drill_hole_diameter=defaults.get("drill_hole_diameter", 150.0),
                fy=defaults.get("fy", 420.0),
                bond_stress=defaults.get("bond_stress", 100.0),
                spacing_h=defaults.get("spacing_h", 1.5),
            ))

    return SlopeGeometry(
        surface_points=surface_pts,
        soil_layers=soil_layers,
        gwt_points=gwt_points,
        nails=nails if nails else None,
    )


# ---------------------------------------------------------------------------
# FEM conversion
# ---------------------------------------------------------------------------

@dataclass
class FEMSoilPropertyAssignment:
    """Soil properties for FEM analysis (extends SoilPropertyAssignment with stiffness).

    Parameters
    ----------
    name : str
        Must match a boundary_profiles key or "Surface" for the topmost layer.
    gamma : float
        Total unit weight (kN/m3).
    phi : float
        Friction angle (degrees).
    c : float
        Cohesion (kPa).
    E : float
        Young's modulus (kPa).
    nu : float
        Poisson's ratio.
    psi : float
        Dilatancy angle (degrees).
    model : str
        Constitutive model: 'mc', 'hs', or 'elastic'.
    hs_params : dict, optional
        For HS model: {E50_ref, Eur_ref, m, p_ref, R_f}.
    """
    name: str = ""
    gamma: float = 18.0
    phi: float = 0.0
    c: float = 0.0
    E: float = 30000.0
    nu: float = 0.3
    psi: float = 0.0
    model: str = "mc"
    hs_params: Optional[Dict[str, float]] = None


def build_fem_inputs(
    parse_result: DxfParseResult,
    soil_properties: List[FEMSoilPropertyAssignment],
) -> Dict[str, Any]:
    """Convert DXF parse result to fem2d input format.

    Parameters
    ----------
    parse_result : DxfParseResult
        Output from parse_dxf_geometry().
    soil_properties : list of FEMSoilPropertyAssignment
        One entry per soil layer. Names must match boundary soil names.
        Include a "Surface" entry for the topmost layer.

    Returns
    -------
    dict with keys:
        'surface_points' : list of (x, z) tuples — ground surface profile.
        'soil_layers' : list of dicts — each with name, top_elevation,
            bottom_elevation, E, nu, c, phi, psi, gamma, model, hs_params.
        'gwt' : (M, 2) numpy array or None — GWT polyline for fem2d.
        'boundary_polylines' : list of (M, 2) numpy arrays — for
            assign_layers_by_polylines().

    Raises
    ------
    ValueError
        If no surface points, no properties, or missing boundary match.
    """
    if not parse_result.surface_points:
        raise ValueError("No surface points in parse result")

    if not soil_properties:
        raise ValueError("At least one FEMSoilPropertyAssignment is required")

    prop_lookup = {sp.name: sp for sp in soil_properties}

    # Sort boundaries by descending max elevation (same logic as build_slope_geometry)
    boundary_names = list(parse_result.boundary_profiles.keys())
    if boundary_names:
        boundary_names.sort(
            key=lambda n: -max(z for _, z in parse_result.boundary_profiles[n])
        )

    surface_pts = parse_result.surface_points
    surface_z_max = max(z for _, z in surface_pts)
    surface_z_min = min(z for _, z in surface_pts)

    soil_layers = []
    boundary_polylines = []

    if not boundary_names:
        # Single layer
        sp = soil_properties[0]
        layer_dict = _fem_layer_dict(
            sp, top_elev=surface_z_max,
            bottom_elev=surface_z_min - 5.0,
        )
        soil_layers.append(layer_dict)
    else:
        # Multi-layer
        first_boundary = parse_result.boundary_profiles[boundary_names[0]]
        first_boundary_z_max = max(z for _, z in first_boundary)

        # Top layer
        top_name = "Surface"
        if top_name not in prop_lookup:
            for sp in soil_properties:
                if sp.name not in boundary_names:
                    top_name = sp.name
                    break
            else:
                top_name = soil_properties[0].name

        if top_name in prop_lookup:
            sp = prop_lookup[top_name]
            soil_layers.append(_fem_layer_dict(
                sp, top_elev=surface_z_max,
                bottom_elev=first_boundary_z_max,
            ))

        # Middle and bottom layers
        for i, bname in enumerate(boundary_names):
            boundary_pts = parse_result.boundary_profiles[bname]
            boundary_z_max = max(z for _, z in boundary_pts)

            # Store polyline for assign_layers_by_polylines
            sorted_pts = sorted(boundary_pts, key=lambda p: p[0])
            boundary_polylines.append(
                np.array(sorted_pts, dtype=float)
            )

            if i + 1 < len(boundary_names):
                next_boundary = parse_result.boundary_profiles[
                    boundary_names[i + 1]
                ]
                bottom_z = max(z for _, z in next_boundary)
            else:
                bottom_z = min(z for _, z in boundary_pts) - 2.0

            if bname not in prop_lookup:
                raise ValueError(
                    f"No FEMSoilPropertyAssignment for boundary '{bname}'. "
                    f"Available: {sorted(prop_lookup.keys())}"
                )
            sp = prop_lookup[bname]
            soil_layers.append(_fem_layer_dict(
                sp, top_elev=boundary_z_max, bottom_elev=bottom_z,
            ))

    # GWT
    gwt = None
    if parse_result.gwt_points:
        sorted_gwt = sorted(parse_result.gwt_points, key=lambda p: p[0])
        gwt = np.array(sorted_gwt, dtype=float)

    return {
        "surface_points": list(surface_pts),
        "soil_layers": soil_layers,
        "gwt": gwt,
        "boundary_polylines": boundary_polylines,
    }


def _fem_layer_dict(
    sp: FEMSoilPropertyAssignment,
    top_elev: float,
    bottom_elev: float,
) -> Dict[str, Any]:
    """Build a single fem2d-format layer dict from a property assignment."""
    d = {
        "name": sp.name,
        "top_elevation": top_elev,
        "bottom_elevation": bottom_elev,
        "gamma": sp.gamma,
        "phi": sp.phi,
        "c": sp.c,
        "E": sp.E,
        "nu": sp.nu,
        "psi": sp.psi,
        "model": sp.model,
    }
    if sp.hs_params:
        d["hs_params"] = dict(sp.hs_params)
    return d
