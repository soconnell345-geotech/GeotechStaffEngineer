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
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from sheet_pile.earth_pressure import rankine_Ka, rankine_Kp  # kept for backward compat
from sheet_pile.cantilever import (
    WallSoilLayer, _get_soil_at_depth, _cumulative_stress,
    _cumulative_stress_passive, _effective_gamma, _layer_Ka_Kp,
    _with_log_spiral_source, _log_spiral_choices,
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
    #: The shear/moment integration's own points, top to toe: [z, p_a,
    #: p_p/FOS, net water pressure, net pressure p, shear V, moment M]
    #: (kPa, kN/m, kN-m/m), with T applied at the anchor. Not in to_dict;
    #: read by get_tables.
    profile: List[list] = field(default_factory=list, repr=False)
    #: What the analysis chose (GM-232: the log-spiral passive Kp's source,
    #: with UFC 3-220-20 p. 260 on tied-back walls).
    choices: list = field(default_factory=list)

    def get_tables(self) -> list:
        """The pressure diagram along the wall with the shear and bending
        moment integrated from it (the anchor reaction at its depth), and
        the design summary (CSV only; the calc package prints it). Values
        are this result's own, per metre of wall.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        if self.profile:
            tables.append(CalcTable(
                title="Pressure, shear and moment along the wall",
                columns=[("Depth z", "m", ".3f"),
                         ("Active pressure p_a", "kPa", ".2f"),
                         ("Passive pressure p_p/FOS", "kPa", ".2f"),
                         ("Net water pressure", "kPa", ".2f"),
                         ("Net pressure p", "kPa", ".2f"),
                         ("Shear V", "kN/m", ".2f"),
                         ("Bending moment M", "kN-m/m", ".2f")],
                rows=[list(r) for r in self.profile],
                note=(f"Free earth support, wall {self.total_wall_length:.2f}"
                      f" m long, excavation {self.excavation_depth:.2f} m, "
                      f"anchor at {self.anchor_depth:.2f} m carrying T = "
                      f"{self.anchor_force:.1f} kN/m; passive divided by "
                      f"FOS = {self.FOS_passive:g}; V and M integrated from "
                      f"the top at the midpoint of each of "
                      f"{len(self.profile)} steps; the maximum |M| is "
                      f"{self.max_moment:.1f} kN-m/m at "
                      f"{self.max_moment_depth:.2f} m"),
                name="wall_pressures",
            ))
        tables.append(CalcTable(
            title="Wall design summary",
            columns=["Quantity", "Symbol", "Value", "Unit"],
            rows=[
                ["Excavation depth", "H", self.excavation_depth, "m"],
                ["Anchor depth", "a", self.anchor_depth, "m"],
                ["Embedment", "D", self.embedment_depth, "m"],
                ["Total wall length", "H + D", self.total_wall_length, "m"],
                ["Factor of safety on passive", "FOS", self.FOS_passive, ""],
                ["Anchor force", "T", self.anchor_force, "kN/m"],
                ["Maximum bending moment", "M_max", self.max_moment,
                 "kN-m/m"],
                ["Depth of maximum moment", "z_Mmax", self.max_moment_depth,
                 "m"],
            ],
            note="Anchored sheet pile wall, free earth support",
            name="design_summary", printed=False,
        ))
        return tables

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
            **({"choices": self.choices} if self.choices else {}),
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
    log_spiral_source: str = "ufc_3_220_20",
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
    pressure_method : str, optional
        "rankine" (default), "coulomb" or "log_spiral" (Coulomb active and
        the log-spiral passive Kp).
    log_spiral_source : str, optional
        With pressure_method "log_spiral", the printed source of the
        passive Kp; see ``analyze_cantilever``.

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
    pressure_method = _with_log_spiral_source(pressure_method,
                                              log_spiral_source)

    H = excavation_depth
    n_points = 500

    # Find embedment by summing moments about the anchor (free earth support).
    # Coarse-grid bracket + 1 mm bisection: the previous first-grid-crossing
    # return overshot the true root by up to one grid step (the Das 9.14
    # "grid quantization" ±3 %), and the 0.5 m grid start acted as an
    # artificial embedment floor.
    def _net_moment(D_trial):
        total_length = H + D_trial
        dz = total_length / n_points

        moment_driving = 0.0
        moment_resisting = 0.0

        for i in range(n_points):
            z = (i + 0.5) * dz
            arm = z - anchor_depth  # moment arm from anchor

            layer = _get_soil_at_depth(z, soil_layers)
            Ka, Kp = _layer_Ka_Kp(layer, pressure_method)

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
                u_active = gamma_w * max(0, z - (1e10 if gwt_depth_active is None else gwt_depth_active))
                u_passive = gamma_w * max(0, z_below - max(0, (1e10 if gwt_depth_passive is None else gwt_depth_passive) - H))
                net_active += (u_active - u_passive) * dz

                moment_driving += net_active * arm
                moment_resisting += net_passive * arm

        return moment_resisting - moment_driving

    def _bisect(lo, hi):
        for _ in range(60):
            if hi - lo <= 1e-3:
                break
            mid = (lo + hi) / 2.0
            if _net_moment(mid) >= 0:
                hi = mid
            else:
                lo = mid
        return hi

    D_found = None
    D_prev = None
    for D_trial in np.linspace(0.5, 4 * H, 300):
        if _net_moment(D_trial) >= 0:
            D_found = _bisect(0.01 if D_prev is None else D_prev, D_trial)
            break
        D_prev = D_trial

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
        Ka, Kp = _layer_Ka_Kp(layer, pressure_method)

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

            u_a = gamma_w * max(0, z - (1e10 if gwt_depth_active is None else gwt_depth_active))
            u_p = gamma_w * max(0, z_below - max(0, (1e10 if gwt_depth_passive is None else gwt_depth_passive) - H))
            total_active_force += (u_a - u_p) * dz

    anchor_force = total_active_force - total_passive_force
    anchor_force = max(anchor_force, 0)

    # Shear and moment down the whole wall from the same pressures, with the
    # anchor reaction T applied AT the anchor depth (GM-69). Until 2026-10-14
    # the shear started at -T at the top of the wall, which made every moment
    # below the anchor too large by T x anchor_depth, and the pressures below
    # the excavation level were left out. Midpoint pressures, as above; the
    # reaction enters the step that contains the anchor at its own depth.
    profile = []
    max_moment = 0.0
    max_moment_depth = 0.0
    shear = 0.0
    moment = 0.0
    anchored = False

    for i in range(n_points):
        z = (i + 0.5) * dz
        z_end = (i + 1) * dz
        layer = _get_soil_at_depth(z, soil_layers)
        Ka, Kp = _layer_Ka_Kp(layer, pressure_method)

        pp_reduced = 0.0
        water = 0.0
        if z <= H:
            sigma_v = surcharge + _cumulative_stress(z, soil_layers, gwt_depth_active, gamma_w)
            pa = Ka * sigma_v - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            if gwt_depth_active is not None and z > gwt_depth_active:
                water = gamma_w * (z - gwt_depth_active)
        else:
            z_below = z - H
            sigma_v_a = surcharge + _cumulative_stress(z, soil_layers, gwt_depth_active, gamma_w)
            pa = Ka * sigma_v_a - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            sigma_v_p = _cumulative_stress_passive(z_below, H, soil_layers, gwt_depth_passive, gamma_w)
            pp = Kp * sigma_v_p + 2 * layer.cohesion * math.sqrt(Kp)
            pp_reduced = pp / FOS_passive
            u_a = gamma_w * max(0, z - (1e10 if gwt_depth_active is None else gwt_depth_active))
            u_p = gamma_w * max(0, z_below - max(0, (1e10 if gwt_depth_passive is None else gwt_depth_passive) - H))
            water = u_a - u_p
        net_p = pa - pp_reduced + water

        shear += net_p * dz
        moment += shear * dz
        if not anchored and z_end >= anchor_depth:
            shear -= anchor_force
            moment -= anchor_force * (z_end - anchor_depth)
            anchored = True
        if abs(moment) > abs(max_moment):
            max_moment = abs(moment)
            max_moment_depth = z
        profile.append([z, pa, pp_reduced, water, net_p, shear, moment])

    return AnchoredWallResult(
        embedment_depth=D_design,
        total_wall_length=total_wall,
        anchor_force=anchor_force,
        max_moment=max_moment,
        max_moment_depth=max_moment_depth,
        anchor_depth=anchor_depth,
        excavation_depth=H,
        FOS_passive=FOS_passive,
        profile=profile,
        choices=_log_spiral_choices(pressure_method, soil_layers, H,
                                    tied_back=True),
    )
