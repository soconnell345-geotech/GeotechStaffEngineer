"""
Anchored sheet pile wall analysis (single anchor).

Determines required embedment depth, anchor force, and maximum moment
using the free earth support method.

All units are SI: kN, kPa, meters.

References:
    USACE EM 1110-2-2504, Chapter 5
    USS Steel Sheet Piling Design Manual, Chapter 4
    Das, "Principles of Foundation Engineering", Chapter 9
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sheet_pile.earth_pressure import rankine_Ka, rankine_Kp  # kept for backward compat
from sheet_pile.cantilever import (
    WallSoilLayer, _get_soil_at_depth, _cumulative_stress,
    _cumulative_stress_passive, _effective_gamma, _compute_Ka_Kp,
)
from geotech_common.water import GAMMA_W


@dataclass
class AnchoredWallResult:
    """Results from an anchored sheet pile wall analysis."""
    embedment_depth: float = 0.0
    total_wall_length: float = 0.0
    anchor_force: float = 0.0
    max_moment: float = 0.0
    max_moment_depth: float = 0.0
    anchor_depth: float = 0.0
    excavation_depth: float = 0.0
    FOS_passive: float = 1.5

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ANCHORED SHEET PILE WALL RESULTS",
            "=" * 60,
            "",
            f"  Excavation depth:   {self.excavation_depth:.2f} m",
            f"  Anchor depth:       {self.anchor_depth:.2f} m from top",
            f"  Required embedment: {self.embedment_depth:.2f} m",
            f"  Total wall length:  {self.total_wall_length:.2f} m",
            f"  FOS on passive:     {self.FOS_passive:.2f}",
            "",
            f"  Anchor force:       {self.anchor_force:.1f} kN/m",
            f"  Max bending moment: {self.max_moment:.1f} kN-m/m",
            f"  at depth:           {self.max_moment_depth:.2f} m below top",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self):
        return {
            "embedment_depth_m": round(self.embedment_depth, 3),
            "total_wall_length_m": round(self.total_wall_length, 3),
            "anchor_force_kN_per_m": round(self.anchor_force, 1),
            "max_moment_kNm_per_m": round(self.max_moment, 1),
            "max_moment_depth_m": round(self.max_moment_depth, 2),
            "anchor_depth_m": round(self.anchor_depth, 2),
            "FOS_passive": round(self.FOS_passive, 2),
        }


def analyze_anchored(
    excavation_depth: float,
    anchor_depth: float,
    soil_layers: List[WallSoilLayer],
    gwt_depth_active: Optional[float] = None,
    gwt_depth_passive: Optional[float] = None,
    surcharge: float = 0.0,
    FOS_passive: float = 1.5,
    gamma_w: float = GAMMA_W,
    pressure_method: str = "rankine",
) -> AnchoredWallResult:
    """Analyze an anchored sheet pile wall (free earth support method).

    1. Sum moments about the anchor to find embedment D
    2. Horizontal equilibrium gives anchor force T
    3. Compute maximum moment between anchor and excavation line

    Parameters
    ----------
    excavation_depth : float
        Depth of excavation H (m).
    anchor_depth : float
        Depth of anchor from top of wall (m). Must be < excavation_depth.
    soil_layers : list of WallSoilLayer
        Soil layers from top to bottom.
    gwt_depth_active : float, optional
        GWT depth on active side (m from top).
    gwt_depth_passive : float, optional
        GWT depth on passive side (m from top).
    surcharge : float, optional
        Uniform surcharge on retained side (kPa). Default 0.
    FOS_passive : float, optional
        Factor of safety on passive resistance. Default 1.5.
    gamma_w : float, optional
        Unit weight of water. Default 9.81.

    Returns
    -------
    AnchoredWallResult
    """
    if excavation_depth <= 0:
        raise ValueError(f"Excavation depth must be positive, got {excavation_depth}")
    if anchor_depth <= 0 or anchor_depth >= excavation_depth:
        raise ValueError(
            f"Anchor depth ({anchor_depth}m) must be between 0 and "
            f"excavation depth ({excavation_depth}m)"
        )
    if gwt_depth_passive is None:
        gwt_depth_passive = gwt_depth_active

    H = excavation_depth
    n_points = 500

    # Find embedment by summing moments about the anchor
    D_found = None
    for D_trial in np.linspace(0.5, 4 * H, 300):
        total_length = H + D_trial
        dz = total_length / n_points

        moment_driving = 0.0
        moment_resisting = 0.0

        for i in range(n_points):
            z = (i + 0.5) * dz
            arm = z - anchor_depth  # moment arm from anchor

            layer = _get_soil_at_depth(z, soil_layers)
            Ka, Kp = _compute_Ka_Kp(layer.friction_angle, pressure_method)

            if z <= H:
                sigma_v = surcharge + _cumulative_stress(z, soil_layers, gwt_depth_active, gamma_w)
                pa = Ka * sigma_v - 2 * layer.cohesion * math.sqrt(Ka)
                pa = max(pa, 0)
                force = pa * dz
                if gwt_depth_active is not None and z > gwt_depth_active:
                    force += gamma_w * (z - gwt_depth_active) * dz
                moment_driving += force * arm
            else:
                z_below = z - H
                sigma_v_a = surcharge + _cumulative_stress(z, soil_layers, gwt_depth_active, gamma_w)
                pa = Ka * sigma_v_a - 2 * layer.cohesion * math.sqrt(Ka)
                pa = max(pa, 0)

                sigma_v_p = _cumulative_stress_passive(z_below, H, soil_layers, gwt_depth_passive, gamma_w)
                pp = Kp * sigma_v_p + 2 * layer.cohesion * math.sqrt(Kp)
                pp_reduced = pp / FOS_passive

                net_active = pa * dz
                net_passive = pp_reduced * dz

                # Water
                u_active = gamma_w * max(0, z - (gwt_depth_active or 1e10))
                u_passive = gamma_w * max(0, z_below - max(0, (gwt_depth_passive or 1e10) - H))
                net_active += (u_active - u_passive) * dz

                moment_driving += net_active * arm
                moment_resisting += net_passive * arm

        if moment_resisting >= moment_driving and D_found is None:
            D_found = D_trial
            break

    if D_found is None:
        D_found = 4 * H
        warnings.warn("Anchored wall embedment did not converge")

    D_design = D_found  # No increase for anchored walls (free earth support)
    total_wall = H + D_design

    # Compute anchor force from horizontal equilibrium
    total_active_force = 0.0
    total_passive_force = 0.0
    dz = total_wall / n_points

    for i in range(n_points):
        z = (i + 0.5) * dz
        layer = _get_soil_at_depth(z, soil_layers)
        Ka, Kp = _compute_Ka_Kp(layer.friction_angle, pressure_method)

        if z <= H:
            sigma_v = surcharge + _cumulative_stress(z, soil_layers, gwt_depth_active, gamma_w)
            pa = Ka * sigma_v - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            total_active_force += pa * dz
            if gwt_depth_active is not None and z > gwt_depth_active:
                total_active_force += gamma_w * (z - gwt_depth_active) * dz
        else:
            z_below = z - H
            sigma_v_a = surcharge + _cumulative_stress(z, soil_layers, gwt_depth_active, gamma_w)
            pa = Ka * sigma_v_a - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            total_active_force += pa * dz

            sigma_v_p = _cumulative_stress_passive(z_below, H, soil_layers, gwt_depth_passive, gamma_w)
            pp = Kp * sigma_v_p + 2 * layer.cohesion * math.sqrt(Kp)
            total_passive_force += (pp / FOS_passive) * dz

            u_a = gamma_w * max(0, z - (gwt_depth_active or 1e10))
            u_p = gamma_w * max(0, z_below - max(0, (gwt_depth_passive or 1e10) - H))
            total_active_force += (u_a - u_p) * dz

    anchor_force = total_active_force - total_passive_force
    anchor_force = max(anchor_force, 0)

    # Max moment (between anchor and excavation base, approximately)
    max_moment = 0.0
    max_moment_depth = 0.0
    shear = -anchor_force  # anchor provides upward reaction in moment sense
    moment = 0.0

    for i in range(n_points):
        z = (i + 0.5) * dz
        layer = _get_soil_at_depth(z, soil_layers)
        Ka, Kp = _compute_Ka_Kp(layer.friction_angle, pressure_method)

        net_p = 0.0
        if z <= H:
            sigma_v = surcharge + _cumulative_stress(z, soil_layers, gwt_depth_active, gamma_w)
            pa = Ka * sigma_v - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            net_p = pa
            if gwt_depth_active is not None and z > gwt_depth_active:
                net_p += gamma_w * (z - gwt_depth_active)

        shear += net_p * dz
        moment += shear * dz
        if abs(moment) > abs(max_moment) and z <= H:
            max_moment = abs(moment)
            max_moment_depth = z

    return AnchoredWallResult(
        embedment_depth=D_design,
        total_wall_length=total_wall,
        anchor_force=anchor_force,
        max_moment=max_moment,
        max_moment_depth=max_moment_depth,
        anchor_depth=anchor_depth,
        excavation_depth=H,
        FOS_passive=FOS_passive,
    )
