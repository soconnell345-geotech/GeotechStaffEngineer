"""
Cantilever sheet pile wall analysis.

Determines required embedment depth and maximum moment for a cantilever
wall using the free earth support method (simplified).

All units are SI: kN, kPa, meters.

References:
    USACE EM 1110-2-2504, Chapter 4
    USS Steel Sheet Piling Design Manual, Chapter 3
    Das, "Principles of Foundation Engineering", Chapter 9
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from sheet_pile.earth_pressure import (
    rankine_Ka, rankine_Kp, coulomb_Ka, coulomb_Kp,
    active_pressure, passive_pressure, tension_crack_depth,
)
from geotech_common.water import GAMMA_W


@dataclass
class WallSoilLayer:
    """Soil layer for sheet pile wall analysis.

    Parameters
    ----------
    thickness : float
        Layer thickness (m).
    unit_weight : float
        Total unit weight (kN/m³).
    friction_angle : float
        Drained friction angle (degrees).
    cohesion : float, optional
        Cohesion (kPa). Default 0.
    description : str, optional
        Layer description.
    """
    thickness: float
    unit_weight: float
    friction_angle: float = 30.0
    cohesion: float = 0.0
    description: str = ""

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.friction_angle < 0 or self.friction_angle > 50:
            raise ValueError(f"Friction angle must be 0-50, got {self.friction_angle}")
        if self.cohesion < 0:
            raise ValueError(f"Cohesion must be non-negative, got {self.cohesion}")
        if self.cohesion == 0 and self.friction_angle == 0:
            raise ValueError("Soil must have c > 0 or phi > 0")


@dataclass
class CantileverWallResult:
    """Results from a cantilever sheet pile wall analysis."""
    embedment_depth: float = 0.0
    total_wall_length: float = 0.0
    max_moment: float = 0.0
    max_moment_depth: float = 0.0
    FOS_passive: float = 1.0
    excavation_depth: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  CANTILEVER SHEET PILE WALL RESULTS",
            "=" * 60,
            "",
            f"  Excavation depth:   {self.excavation_depth:.2f} m",
            f"  Required embedment: {self.embedment_depth:.2f} m",
            f"  Total wall length:  {self.total_wall_length:.2f} m",
            f"  FOS on passive:     {self.FOS_passive:.2f}",
            "",
            f"  Max bending moment: {self.max_moment:.1f} kN-m/m",
            f"  at depth:           {self.max_moment_depth:.2f} m below top",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def plot_wall_diagram(self, ax=None, show=True):
        """Plot schematic wall diagram with excavation and embedment zones.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        H = self.excavation_depth
        D = self.embedment_depth
        # Wall line
        ax.plot([0, 0], [0, H + D], 'k-', linewidth=3, label='Wall')
        # Excavation zone
        ax.fill_betweenx([0, H], -0.5, 0, color='lightyellow', alpha=0.5)
        ax.annotate('Excavation', xy=(-0.25, H / 2), ha='center',
                    fontsize=9, fontstyle='italic')
        # Embedment zone
        ax.fill_betweenx([H, H + D], -0.5, 0.5, color='tan', alpha=0.4)
        ax.annotate('Embedment', xy=(0.25, H + D / 2), ha='center',
                    fontsize=9, fontstyle='italic')
        # Retained soil
        ax.fill_betweenx([0, H + D], 0, 0.5, color='burlywood', alpha=0.3,
                         label='Retained Soil')
        # Excavation level
        ax.axhline(y=H, color='blue', linestyle='--', linewidth=1,
                   label=f'Excavation ({H:.1f} m)')
        # Max moment location
        if self.max_moment_depth > 0:
            ax.plot(0, self.max_moment_depth, 'ro', markersize=8,
                    label=f'Max Moment ({self.max_moment:.0f} kN-m/m)')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Sheet Pile Wall Diagram",
                              "", "Depth (m)")
        ax.legend(fontsize=8, loc='lower right')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self):
        return {
            "embedment_depth_m": round(self.embedment_depth, 3),
            "total_wall_length_m": round(self.total_wall_length, 3),
            "max_moment_kNm_per_m": round(self.max_moment, 1),
            "max_moment_depth_m": round(self.max_moment_depth, 2),
            "FOS_passive": round(self.FOS_passive, 2),
            "excavation_depth_m": round(self.excavation_depth, 2),
        }


def analyze_cantilever(
    excavation_depth: float,
    soil_layers: List[WallSoilLayer],
    gwt_depth_active: Optional[float] = None,
    gwt_depth_passive: Optional[float] = None,
    surcharge: float = 0.0,
    FOS_passive: float = 1.5,
    gamma_w: float = GAMMA_W,
    pressure_method: str = "rankine",
) -> CantileverWallResult:
    """Analyze a cantilever sheet pile wall.

    Uses the simplified free earth support method:
    1. Compute active and passive pressure diagrams
    2. Sum moments about the base to find required embedment D
    3. Compute maximum moment

    Parameters
    ----------
    excavation_depth : float
        Depth of excavation H (m).
    soil_layers : list of WallSoilLayer
        Soil layers from top to bottom (both retained and embedded sides).
    gwt_depth_active : float, optional
        GWT depth on the active (retained) side (m from top of wall).
    gwt_depth_passive : float, optional
        GWT depth on the passive (excavation) side (m from top of wall).
        If None, same as active side.
    surcharge : float, optional
        Uniform surcharge on the retained side (kPa). Default 0.
    FOS_passive : float, optional
        Factor of safety applied to passive resistance. Default 1.5.
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default 9.81.
    pressure_method : str, optional
        "rankine" (default) or "coulomb".

    Returns
    -------
    CantileverWallResult
    """
    if excavation_depth <= 0:
        raise ValueError(f"Excavation depth must be positive, got {excavation_depth}")

    if gwt_depth_passive is None:
        gwt_depth_passive = gwt_depth_active

    # For a simple single-layer case, we use analytical formulas
    # For multi-layer, we use numerical integration
    n_points = 200
    D_max = excavation_depth * 3  # initial guess: embedment up to 3H
    total_length = excavation_depth + D_max
    dz = total_length / n_points

    # Find embedment by iteration
    D_converged = _find_embedment(
        excavation_depth, soil_layers, gwt_depth_active, gwt_depth_passive,
        surcharge, FOS_passive, gamma_w, pressure_method, n_points=500
    )

    # Apply 20-40% increase per USACE guidance for simplified method
    D_design = D_converged * 1.2

    total_wall = excavation_depth + D_design

    # Compute maximum moment
    max_moment, max_moment_depth = _compute_max_moment(
        excavation_depth, D_design, soil_layers,
        gwt_depth_active, gwt_depth_passive,
        surcharge, FOS_passive, gamma_w, pressure_method, n_points=500
    )

    return CantileverWallResult(
        embedment_depth=D_design,
        total_wall_length=total_wall,
        max_moment=max_moment,
        max_moment_depth=max_moment_depth,
        FOS_passive=FOS_passive,
        excavation_depth=excavation_depth,
    )


def _compute_Ka_Kp(phi_deg: float, method: str = "rankine"):
    """Compute Ka, Kp using the specified method."""
    if method == "coulomb":
        return coulomb_Ka(phi_deg), coulomb_Kp(phi_deg)
    return rankine_Ka(phi_deg), rankine_Kp(phi_deg)


def _get_soil_at_depth(depth: float, soil_layers: List[WallSoilLayer]) -> WallSoilLayer:
    """Get the soil layer at a given depth from top of wall."""
    z = 0
    for layer in soil_layers:
        if z + layer.thickness > depth:
            return layer
        z += layer.thickness
    return soil_layers[-1]


def _effective_gamma(depth: float, layer: WallSoilLayer,
                     gwt_depth: Optional[float], gamma_w: float) -> float:
    """Get effective unit weight at depth."""
    if gwt_depth is not None and depth > gwt_depth:
        return layer.unit_weight - gamma_w
    return layer.unit_weight


def _find_embedment(excavation_depth, soil_layers, gwt_active, gwt_passive,
                    surcharge, FOS_passive, gamma_w,
                    pressure_method="rankine", n_points=500):
    """Find embedment depth by summing moments about the base."""
    H = excavation_depth

    # Try embedment depths from 0.5m to 4*H
    for D_trial in np.linspace(0.5, 4 * H, 200):
        total_length = H + D_trial
        dz = total_length / n_points

        # Compute net moment about the wall base
        moment_active = 0.0
        moment_passive = 0.0

        sigma_v_active = surcharge
        sigma_v_passive = 0.0

        for i in range(n_points):
            z = (i + 0.5) * dz  # depth from top of wall
            arm = total_length - z  # moment arm from base

            layer = _get_soil_at_depth(z, soil_layers)
            Ka, Kp = _compute_Ka_Kp(layer.friction_angle, pressure_method)

            if z <= H:
                # Above excavation: only active pressure
                gamma_eff = _effective_gamma(z, layer, gwt_active, gamma_w)
                sigma_v_at_z = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
                pa = Ka * sigma_v_at_z - 2 * layer.cohesion * math.sqrt(Ka)
                pa = max(pa, 0)  # no tension
                moment_active += pa * dz * arm

                # Water pressure on active side
                if gwt_active is not None and z > gwt_active:
                    u_active = gamma_w * (z - gwt_active)
                    moment_active += u_active * dz * arm
            else:
                # Below excavation: active + net passive
                z_below = z - H  # depth below excavation

                # Active side
                sigma_v_at_z = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
                pa = Ka * sigma_v_at_z - 2 * layer.cohesion * math.sqrt(Ka)
                pa = max(pa, 0)
                moment_active += pa * dz * arm

                # Passive side (with FOS)
                layer_passive = _get_soil_at_depth(z, soil_layers)
                sigma_v_passive_z = _cumulative_stress_passive(z_below, H, soil_layers, gwt_passive, gamma_w)
                pp = Kp * sigma_v_passive_z + 2 * layer_passive.cohesion * math.sqrt(Kp)
                pp_reduced = pp / FOS_passive
                moment_passive += pp_reduced * dz * arm

                # Differential water pressure
                if gwt_active is not None and z > gwt_active:
                    u_active = gamma_w * (z - gwt_active)
                else:
                    u_active = 0
                if gwt_passive is not None and z > gwt_passive:
                    u_passive = gamma_w * (z_below - max(0, gwt_passive - H))
                else:
                    u_passive = 0
                net_water = u_active - u_passive
                if net_water > 0:
                    moment_active += net_water * dz * arm

        if moment_passive >= moment_active:
            return D_trial

    warnings.warn("Embedment depth did not converge; using maximum trial depth")
    return 4 * H


def _cumulative_stress(z: float, soil_layers: List[WallSoilLayer],
                       gwt_depth: Optional[float], gamma_w: float) -> float:
    """Compute cumulative vertical effective stress at depth z."""
    sigma_v = 0.0
    depth = 0.0
    for layer in soil_layers:
        if depth >= z:
            break
        dz = min(layer.thickness, z - depth)
        if gwt_depth is not None and depth + dz > gwt_depth:
            # Part above GWT, part below
            above = max(0, gwt_depth - depth)
            below = dz - above
            sigma_v += layer.unit_weight * above
            sigma_v += (layer.unit_weight - gamma_w) * below
        else:
            sigma_v += layer.unit_weight * dz
        depth += layer.thickness
    return sigma_v


def _cumulative_stress_passive(z_below_exc: float, excavation_depth: float,
                               soil_layers: List[WallSoilLayer],
                               gwt_depth: Optional[float],
                               gamma_w: float) -> float:
    """Compute vertical effective stress on the passive side below excavation.

    Unlike _cumulative_stress which starts from the ground surface,
    this starts accumulating from the excavation depth downward using
    the soil layers that actually exist at that depth.

    Parameters
    ----------
    z_below_exc : float
        Depth below the excavation line (m).
    excavation_depth : float
        Depth of excavation from top of wall (m).
    soil_layers : list of WallSoilLayer
        Soil layers from top to bottom.
    gwt_depth : float or None
        GWT depth on the passive side (m from top of wall).
    gamma_w : float
        Unit weight of water.
    """
    # Absolute depth from the top of the wall
    z_abs = excavation_depth + z_below_exc
    # GWT depth below the excavation line (passive side reference)
    gwt_below_exc = None
    if gwt_depth is not None:
        gwt_below_exc = max(0.0, gwt_depth - excavation_depth)

    # Walk through soil layers to find where the excavation depth falls,
    # then accumulate stress from there
    sigma_v = 0.0
    depth = 0.0
    for layer in soil_layers:
        layer_top = depth
        layer_bot = depth + layer.thickness
        depth = layer_bot

        if layer_bot <= excavation_depth:
            continue  # skip layers entirely above excavation

        # The portion of this layer that is below the excavation
        start = max(layer_top, excavation_depth)
        end = min(layer_bot, z_abs)
        if start >= end:
            continue

        # Depth below excavation for this segment
        seg_top_below = start - excavation_depth
        seg_bot_below = end - excavation_depth

        dz_seg = seg_bot_below - seg_top_below
        if gwt_below_exc is not None and seg_bot_below > gwt_below_exc:
            above = max(0.0, gwt_below_exc - seg_top_below)
            below = dz_seg - above
            sigma_v += layer.unit_weight * above
            sigma_v += (layer.unit_weight - gamma_w) * below
        else:
            sigma_v += layer.unit_weight * dz_seg

        if layer_bot >= z_abs:
            break

    # If z_abs extends beyond all defined layers, use the last layer
    if depth < z_abs and soil_layers:
        last = soil_layers[-1]
        remaining = z_abs - max(depth, excavation_depth)
        if remaining > 0:
            if gwt_below_exc is not None:
                depth_below = max(depth, excavation_depth) - excavation_depth
                if depth_below < gwt_below_exc:
                    above = min(remaining, gwt_below_exc - depth_below)
                    below = remaining - above
                    sigma_v += last.unit_weight * above
                    sigma_v += (last.unit_weight - gamma_w) * below
                else:
                    sigma_v += (last.unit_weight - gamma_w) * remaining
            else:
                sigma_v += last.unit_weight * remaining

    return sigma_v


def _compute_max_moment(excavation_depth, embedment, soil_layers,
                        gwt_active, gwt_passive, surcharge,
                        FOS_passive, gamma_w,
                        pressure_method="rankine", n_points=500):
    """Compute maximum bending moment and its location."""
    H = excavation_depth
    total_length = H + embedment
    dz = total_length / n_points

    # Compute shear force along the wall; moment is max where shear = 0
    shear = 0.0
    moment = 0.0
    max_moment = 0.0
    max_moment_depth = 0.0

    for i in range(n_points):
        z = (i + 0.5) * dz
        layer = _get_soil_at_depth(z, soil_layers)
        Ka, Kp = _compute_Ka_Kp(layer.friction_angle, pressure_method)

        net_pressure = 0.0
        if z <= H:
            sigma_v = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
            pa = Ka * sigma_v - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            net_pressure = pa
            if gwt_active is not None and z > gwt_active:
                net_pressure += gamma_w * (z - gwt_active)
        else:
            z_below = z - H
            sigma_v_a = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
            pa = Ka * sigma_v_a - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)

            sigma_v_p = _cumulative_stress_passive(z_below, H, soil_layers, gwt_passive, gamma_w)
            pp = Kp * sigma_v_p + 2 * layer.cohesion * math.sqrt(Kp)
            pp_reduced = pp / FOS_passive

            net_pressure = pa - pp_reduced

            # Water
            u_active = gamma_w * max(0, z - (gwt_active or 1e10))
            u_passive = gamma_w * max(0, z_below - max(0, (gwt_passive or 1e10) - H))
            net_pressure += u_active - u_passive

        shear += net_pressure * dz
        moment += shear * dz

        if abs(moment) > abs(max_moment):
            max_moment = abs(moment)
            max_moment_depth = z

    return max_moment, max_moment_depth
