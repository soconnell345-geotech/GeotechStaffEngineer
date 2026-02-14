"""
Combined axial pile capacity analysis.

Auto-selects the appropriate method for each soil layer:
- Nordlund for cohesionless layers (sand/gravel)
- Tomlinson alpha for cohesive layers (clay/silt)

Also supports the beta (effective stress) method for any soil type.

All units are SI: kN, m, kPa.

References:
    FHWA GEC-12 (FHWA-NHI-16-009), Chapters 7-8
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from axial_pile.pile_types import PileSection
from axial_pile.soil_profile import AxialSoilProfile, AxialSoilLayer
from axial_pile.tomlinson import skin_friction_cohesive, end_bearing_cohesive
from axial_pile.nordlund import skin_friction_cohesionless, end_bearing_cohesionless
from axial_pile.beta_method import (
    beta_from_phi, skin_friction_beta, end_bearing_beta, Nt_from_phi,
)
from axial_pile.results import AxialPileResult


@dataclass
class AxialPileAnalysis:
    """Axial pile capacity analysis for a single driven pile.

    Parameters
    ----------
    pile : PileSection
        Pile cross-section properties.
    soil : AxialSoilProfile
        Layered soil profile.
    pile_length : float
        Embedded pile length (m).
    method : str, optional
        Analysis method:
        - "auto" (default): Nordlund for sand, Tomlinson for clay
        - "beta": Effective stress (beta) method for all layers
    factor_of_safety : float, optional
        Factor of safety. Default 2.5.
    include_uplift : bool, optional
        If True, also compute uplift (tension) capacity. Default False.

    Examples
    --------
    >>> pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
    >>> soil = AxialSoilProfile(layers=[
    ...     AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
    ...     AxialSoilLayer(5, 'cohesive', 17.0, cohesion=50),
    ... ])
    >>> analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15)
    >>> result = analysis.compute()
    """
    pile: PileSection = None
    soil: AxialSoilProfile = None
    pile_length: float = 0.0
    method: str = "auto"
    factor_of_safety: float = 2.5
    include_uplift: bool = False

    def __post_init__(self):
        if self.pile is None:
            raise ValueError("Pile section must be provided")
        if self.soil is None:
            raise ValueError("Soil profile must be provided")
        if self.pile_length <= 0:
            raise ValueError(f"Pile length must be positive, got {self.pile_length}")
        if self.pile_length > self.soil.total_thickness:
            warnings.warn(
                f"Pile length ({self.pile_length}m) exceeds soil profile "
                f"({self.soil.total_thickness}m); using full profile depth"
            )

    def compute(self) -> AxialPileResult:
        """Run the axial capacity analysis.

        Returns
        -------
        AxialPileResult
            Complete results with skin friction, end bearing, and per-layer breakdown.
        """
        n_segments = 50  # discretize pile into segments
        dz = self.pile_length / n_segments
        depths = np.linspace(dz / 2, self.pile_length - dz / 2, n_segments)

        total_Qs = 0.0
        layer_results = []
        current_depth = 0.0

        for layer in self.soil.layers:
            layer_top = current_depth
            layer_bottom = current_depth + layer.thickness
            current_depth = layer_bottom

            if layer_top >= self.pile_length:
                break

            # Portion of this layer that intersects the pile
            z_top = layer_top
            z_bottom = min(layer_bottom, self.pile_length)
            thickness_in_pile = z_bottom - z_top
            if thickness_in_pile <= 0:
                continue

            z_center = (z_top + z_bottom) / 2
            sigma_v = self.soil.effective_stress_at_depth(z_center)

            if self.method == "beta":
                phi = layer.friction_angle if layer.soil_type == "cohesionless" else 25.0
                beta = beta_from_phi(phi)
                Qs_layer = skin_friction_beta(
                    sigma_v, beta, self.pile.perimeter, thickness_in_pile
                )
                method_used = "beta"
            elif layer.soil_type == "cohesionless":
                pile_mat = "concrete" if "concrete" in self.pile.pile_type else "steel"
                Qs_layer = skin_friction_cohesionless(
                    layer.friction_angle, sigma_v,
                    self.pile.perimeter, thickness_in_pile,
                    pile_material=pile_mat,
                    delta_phi_ratio=layer.delta_phi_ratio,
                )
                method_used = "nordlund"
            else:
                pile_type = "concrete" if "concrete" in self.pile.pile_type else "steel"
                Qs_layer = skin_friction_cohesive(
                    layer.cohesion, self.pile.perimeter,
                    thickness_in_pile, pile_type=pile_type,
                )
                method_used = "tomlinson"

            total_Qs += Qs_layer
            layer_results.append({
                "depth_top_m": round(z_top, 2),
                "depth_bottom_m": round(z_bottom, 2),
                "soil_type": layer.soil_type,
                "description": layer.description,
                "method": method_used,
                "skin_friction_kN": round(Qs_layer, 1),
                "sigma_v_kPa": round(sigma_v, 1),
            })

        # End bearing
        tip_layer = self.soil.layer_at_depth(self.pile_length - 0.01)
        sigma_v_tip = self.soil.effective_stress_at_depth(self.pile_length)

        if self.method == "beta":
            phi_tip = tip_layer.friction_angle if tip_layer.soil_type == "cohesionless" else 25.0
            Nt = Nt_from_phi(phi_tip)
            Qt = end_bearing_beta(sigma_v_tip, Nt, self.pile.tip_area)
        elif tip_layer.soil_type == "cohesionless":
            Qt = end_bearing_cohesionless(
                tip_layer.friction_angle, sigma_v_tip,
                self.pile.tip_area, self.pile_length, self.pile.width
            )
        else:
            Qt = end_bearing_cohesive(tip_layer.cohesion, self.pile.tip_area)

        Q_ultimate = total_Qs + Qt
        Q_allowable = Q_ultimate / self.factor_of_safety

        # Uplift (tension) capacity: skin friction only, typically 75% of compression
        Q_uplift = None
        if self.include_uplift:
            Q_uplift = 0.75 * total_Qs  # conservative estimate

        return AxialPileResult(
            Q_ultimate=Q_ultimate,
            Q_skin=total_Qs,
            Q_tip=Qt,
            Q_allowable=Q_allowable,
            Q_uplift=Q_uplift,
            factor_of_safety=self.factor_of_safety,
            pile_length=self.pile_length,
            pile_name=self.pile.name,
            method=self.method,
            layer_breakdown=layer_results,
            sigma_v_tip=sigma_v_tip,
        )

    def capacity_vs_depth(self, depth_min: float = 3.0,
                          depth_max: Optional[float] = None,
                          n_points: int = 20) -> List[dict]:
        """Compute capacity at multiple pile lengths for optimization.

        Parameters
        ----------
        depth_min : float, optional
            Minimum pile length to evaluate (m). Default 3.0.
        depth_max : float, optional
            Maximum pile length. Default: soil profile depth.
        n_points : int, optional
            Number of evaluation points. Default 20.

        Returns
        -------
        list of dict
            Each dict has 'depth_m', 'Q_ultimate_kN', 'Q_skin_kN', 'Q_tip_kN'.
        """
        if depth_max is None:
            depth_max = min(self.pile_length, self.soil.total_thickness)

        depths = np.linspace(depth_min, depth_max, n_points)
        results = []

        original_length = self.pile_length
        for d in depths:
            self.pile_length = float(d)
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
        self.pile_length = original_length
        return results
