"""
Combined drilled shaft capacity analysis.

Auto-selects the appropriate method for each soil layer:
- Alpha method for cohesive layers (clay/silt)
- Beta method for cohesionless layers (sand/gravel)
- Rock socket method for rock layers

Applies GEC-10 exclusion zones:
- Top 1.5m excluded from side resistance
- Bottom 1 diameter excluded from side resistance (cohesive only)
- Permanently cased zone excluded

All units are SI: kN, m, kPa.

References:
    FHWA GEC-10 (FHWA-NHI-10-016), Chapters 13-14
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from drilled_shaft.shaft import DrillShaft
from drilled_shaft.soil_profile import ShaftSoilProfile, ShaftSoilLayer
from drilled_shaft.side_resistance import (
    alpha_cohesive, side_resistance_cohesive,
    beta_cohesionless, side_resistance_cohesionless,
    side_resistance_rock,
)
from drilled_shaft.end_bearing import (
    end_bearing_cohesive, end_bearing_cohesionless, end_bearing_rock,
)
from drilled_shaft.results import DrillShaftResult


@dataclass
class DrillShaftAnalysis:
    """Drilled shaft axial capacity analysis.

    Parameters
    ----------
    shaft : DrillShaft
        Shaft geometry.
    soil : ShaftSoilProfile
        Layered soil profile.
    factor_of_safety : float, optional
        Factor of safety. Default 2.5.

    Examples
    --------
    >>> shaft = DrillShaft(diameter=1.0, length=20.0)
    >>> soil = ShaftSoilProfile(layers=[
    ...     ShaftSoilLayer(5, 'cohesionless', 18.0, phi=32),
    ...     ShaftSoilLayer(10, 'cohesive', 17.0, cu=75),
    ...     ShaftSoilLayer(5, 'rock', 22.0, qu=5000),
    ... ])
    >>> analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
    >>> result = analysis.compute()
    """
    shaft: DrillShaft = None
    soil: ShaftSoilProfile = None
    factor_of_safety: float = 2.5

    def __post_init__(self):
        if self.shaft is None:
            raise ValueError("Shaft must be provided")
        if self.soil is None:
            raise ValueError("Soil profile must be provided")
        if self.shaft.length > self.soil.total_thickness:
            warnings.warn(
                f"Shaft length ({self.shaft.length}m) exceeds soil profile "
                f"({self.soil.total_thickness}m); using full profile depth"
            )

    def compute(self) -> DrillShaftResult:
        """Run the drilled shaft capacity analysis.

        Returns
        -------
        DrillShaftResult
            Complete results with side resistance, end bearing,
            and per-layer breakdown.
        """
        shaft = self.shaft
        L = shaft.length
        D = shaft.diameter

        # Exclusion zones
        top_exclusion = max(1.5, shaft.casing_depth)
        # Bottom exclusion: 1 diameter for cohesive only (applied per-layer)

        total_Qs = 0.0
        Qs_clay = 0.0
        Qs_sand = 0.0
        Qs_rock = 0.0
        layer_results = []

        current_depth = 0.0
        for layer in self.soil.layers:
            layer_top = current_depth
            layer_bottom = current_depth + layer.thickness
            current_depth = layer_bottom

            if layer_top >= L:
                break

            # Portion of this layer within shaft length
            z_top = layer_top
            z_bottom = min(layer_bottom, L)

            # Apply top exclusion zone
            z_top_eff = max(z_top, top_exclusion)

            # Apply bottom exclusion for cohesive only (bottom 1*D)
            z_bottom_eff = z_bottom
            if layer.soil_type == "cohesive":
                bottom_excl_start = L - D
                z_bottom_eff = min(z_bottom, bottom_excl_start)

            effective_thickness = z_bottom_eff - z_top_eff
            if effective_thickness <= 0:
                layer_results.append({
                    "depth_top_m": round(z_top, 2),
                    "depth_bottom_m": round(z_bottom, 2),
                    "soil_type": layer.soil_type,
                    "description": layer.description,
                    "method": "excluded",
                    "side_resistance_kN": 0.0,
                    "fs_kPa": 0.0,
                })
                continue

            z_center = (z_top_eff + z_bottom_eff) / 2
            sigma_v = self.soil.effective_stress_at_depth(z_center)

            if layer.soil_type == "cohesive":
                alpha = alpha_cohesive(layer.cu)
                Qs_layer = side_resistance_cohesive(
                    layer.cu, shaft.perimeter, effective_thickness, alpha
                )
                fs = alpha * layer.cu
                Qs_clay += Qs_layer
                method_used = f"alpha={alpha:.3f}"

            elif layer.soil_type == "cohesionless":
                beta = beta_cohesionless(z_center)
                Qs_layer = side_resistance_cohesionless(
                    sigma_v, beta, shaft.perimeter, effective_thickness
                )
                fs = min(beta * sigma_v, 200.0)
                Qs_sand += Qs_layer
                method_used = f"beta={beta:.3f}"

            else:  # rock
                perimeter = shaft.socket_perimeter
                Qs_layer = side_resistance_rock(
                    layer.qu, perimeter, effective_thickness
                )
                fs = 1.0 * 1.0 * (layer.qu ** 0.5)
                Qs_rock += Qs_layer
                method_used = "rock socket"

            total_Qs += Qs_layer
            layer_results.append({
                "depth_top_m": round(z_top, 2),
                "depth_bottom_m": round(z_bottom, 2),
                "effective_top_m": round(z_top_eff, 2),
                "effective_bottom_m": round(z_bottom_eff, 2),
                "soil_type": layer.soil_type,
                "description": layer.description,
                "method": method_used,
                "side_resistance_kN": round(Qs_layer, 1),
                "fs_kPa": round(fs, 1),
                "sigma_v_kPa": round(sigma_v, 1),
            })

        # End bearing
        tip_layer = self.soil.layer_at_depth(L - 0.01)
        sigma_v_tip = self.soil.effective_stress_at_depth(L)
        L_over_D = L / D

        if tip_layer.soil_type == "cohesive":
            Qt = end_bearing_cohesive(tip_layer.cu, shaft.tip_area, L_over_D)
        elif tip_layer.soil_type == "cohesionless":
            N60 = tip_layer.N60 if tip_layer.N60 > 0 else 15.0
            Qt = end_bearing_cohesionless(N60, shaft.tip_area, D)
        else:  # rock
            Qt = end_bearing_rock(tip_layer.qu, shaft.tip_area, tip_layer.RQD)

        Q_ultimate = total_Qs + Qt
        Q_allowable = Q_ultimate / self.factor_of_safety

        return DrillShaftResult(
            Q_ultimate=Q_ultimate,
            Q_skin=total_Qs,
            Q_tip=Qt,
            Q_allowable=Q_allowable,
            Q_side_clay=Qs_clay,
            Q_side_sand=Qs_sand,
            Q_side_rock=Qs_rock,
            factor_of_safety=self.factor_of_safety,
            shaft_diameter=D,
            shaft_length=L,
            method="GEC-10",
            layer_breakdown=layer_results,
            sigma_v_tip=sigma_v_tip,
        )

    def capacity_vs_depth(self, depth_min: float = 3.0,
                          depth_max: float = None,
                          n_points: int = 20) -> List[dict]:
        """Compute capacity at multiple shaft lengths for optimization.

        Parameters
        ----------
        depth_min : float, optional
            Minimum shaft length (m). Default 3.0.
        depth_max : float, optional
            Maximum shaft length (m). Default: soil profile depth.
        n_points : int, optional
            Number of evaluation points. Default 20.

        Returns
        -------
        list of dict
            Each dict has 'depth_m', 'Q_ultimate_kN', 'Q_skin_kN', 'Q_tip_kN'.
        """
        if depth_max is None:
            depth_max = min(self.shaft.length, self.soil.total_thickness)

        depths = np.linspace(depth_min, depth_max, n_points)
        results = []

        original_length = self.shaft.length
        for d in depths:
            self.shaft.length = float(d)
            try:
                r = self.compute()
                results.append({
                    "depth_m": round(d, 2),
                    "Q_ultimate_kN": round(r.Q_ultimate, 1),
                    "Q_skin_kN": round(r.Q_skin, 1),
                    "Q_tip_kN": round(r.Q_tip, 1),
                })
            except Exception:
                pass
        self.shaft.length = original_length
        return results
