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
from dataclasses import dataclass, field, replace
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
    cohesive_phi : float, optional
        Effective friction angle (degrees) assumed for COHESIVE layers when
        ``method="beta"`` (used for both skin friction and end bearing).
        Default 25.0 — a typical drained phi' for clay per GEC-12 Table 7-9
        guidance; override per-project when drained clay strength is known.
        Ignored for cohesionless layers (which use their own
        ``friction_angle``) and for the "auto" method.
    uplift_skin_fraction : float, optional
        Fraction of the OUTSIDE skin friction credited in tension
        (rule-of-thumb tension reduction). Default 0.75.
    pile_weight : float, optional
        Pile self-weight (kN) added to the uplift capacity. Default None
        (self-weight NOT credited — conservative). Supply the buoyant
        weight below the water table if you want it included.

    Notes
    -----
    Skin friction is integrated layer-by-layer using the midpoint rule
    (unit friction evaluated at the mid-depth of each layer segment within
    the pile, times segment thickness). Because the effective stress
    profile is piecewise linear, this is EXACT within any segment of
    constant gamma — segments are additionally split at the groundwater
    table when it falls inside a layer, so the kink in sigma_v' there does
    not bias the integral.

    Uplift (when ``include_uplift=True``) is the rule-of-thumb
    ``uplift_skin_fraction * Q_skin_outside`` (+ ``pile_weight`` if given).
    Inside (plug) friction of open-ended pipe piles is EXCLUDED from
    uplift: mobilizing it in tension requires the soil-plug weight, which
    this estimate does not model. For final tension design use a dedicated
    method (e.g. Nordlund with tension Kd, GEC-12 Section 7.2.3.2).

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
    cohesive_phi: float = 25.0
    uplift_skin_fraction: float = 0.75
    pile_weight: Optional[float] = None
    head_depth: float = 0.0

    def __post_init__(self):
        if self.pile is None:
            raise ValueError("Pile section must be provided")
        if self.soil is None:
            raise ValueError("Soil profile must be provided")
        if self.pile_length <= 0:
            raise ValueError(f"Pile length must be positive, got {self.pile_length}")
        if self.head_depth < 0:
            raise ValueError(
                f"head_depth must be >= 0 (depth of the pile head below the "
                f"ground surface), got {self.head_depth}"
            )
        if self.head_depth + self.pile_length > self.soil.total_thickness:
            warnings.warn(
                f"Pile tip (head_depth {self.head_depth}m + pile_length "
                f"{self.pile_length}m) exceeds soil profile "
                f"({self.soil.total_thickness}m); using full profile depth"
            )

    @property
    def _tip_depth(self) -> float:
        """Absolute depth of the pile tip below the ground surface (m).

        With the default ``head_depth = 0`` this equals ``pile_length``, so
        every existing result is unchanged.
        """
        return self.head_depth + self.pile_length

    def compute(self) -> AxialPileResult:
        """Run the axial capacity analysis.

        Returns
        -------
        AxialPileResult
            Complete results with skin friction, end bearing, and per-layer breakdown.
        """
        total_Qs = 0.0
        layer_results = []
        current_depth = 0.0
        tip_depth = self._tip_depth

        for layer in self.soil.layers:
            layer_top = current_depth
            layer_bottom = current_depth + layer.thickness
            current_depth = layer_bottom

            if layer_top >= tip_depth:
                break

            # Portion of this layer that intersects the pile shaft. Layers
            # above the pile head (head_depth) contribute no skin friction;
            # with the default head_depth = 0 this clips at [0, pile_length].
            z_top = max(layer_top, self.head_depth)
            z_bottom = min(layer_bottom, tip_depth)
            if z_bottom - z_top <= 0:
                continue

            Qs_layer, sigma_v, method_used = self._layer_skin_friction(
                layer, z_top, z_bottom, self.pile.perimeter
            )

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

        # Outside skin friction (before any inside/plug friction is added)
        Qs_outside = total_Qs

        # End bearing
        tip_layer = self.soil.layer_at_depth(tip_depth - 0.01)
        sigma_v_tip = self.soil.effective_stress_at_depth(tip_depth)

        # Toe/end-bearing friction angle: a cohesionless tip layer may carry a
        # separate ``toe_friction_angle`` (GEC-12 design-limit toe phi); when
        # it is unset, ``toe_phi`` falls back to the shaft friction_angle, so
        # the single-phi behaviour is preserved exactly.
        if self.method == "beta":
            phi_tip = (tip_layer.toe_phi
                       if tip_layer.soil_type == "cohesionless"
                       else self.cohesive_phi)
            Nt = Nt_from_phi(phi_tip)
            Qt = end_bearing_beta(sigma_v_tip, Nt, self.pile.tip_area)
        elif tip_layer.soil_type == "cohesionless":
            Qt = end_bearing_cohesionless(
                tip_layer.toe_phi, sigma_v_tip,
                self.pile.tip_area, self.pile_length, self.pile.width
            )
        else:
            Qt = end_bearing_cohesive(tip_layer.cohesion, self.pile.tip_area)

        # Open-ended pipe pile plugging analysis (GEC-12 Section 7.2.1.4)
        # Governing capacity = lesser of plugged and unplugged
        if not self.pile.closed_end and self.pile.tip_area_plugged is not None:
            Qt_unplugged = Qt  # already computed with annulus tip area

            # Compute inside skin friction for unplugged case
            # Uses same unit friction as outside, applied to inner perimeter
            Qs_inside = self._compute_inside_skin_friction()
            Q_unplugged = total_Qs + Qt_unplugged + Qs_inside

            # Plugged case: full tip area, outside skin friction only
            if self.method == "beta":
                Qt_plugged = end_bearing_beta(
                    sigma_v_tip, Nt, self.pile.tip_area_plugged
                )
            elif tip_layer.soil_type == "cohesionless":
                Qt_plugged = end_bearing_cohesionless(
                    tip_layer.toe_phi, sigma_v_tip,
                    self.pile.tip_area_plugged, self.pile_length, self.pile.width
                )
            else:
                Qt_plugged = end_bearing_cohesive(
                    tip_layer.cohesion, self.pile.tip_area_plugged
                )
            Q_plugged = total_Qs + Qt_plugged

            # Governing = lesser of plugged and unplugged
            if Q_plugged <= Q_unplugged:
                Q_ultimate = Q_plugged
                Qt = Qt_plugged
            else:
                Q_ultimate = Q_unplugged
                Qt = Qt_unplugged
                total_Qs = total_Qs + Qs_inside
        else:
            Q_ultimate = total_Qs + Qt

        Q_allowable = Q_ultimate / self.factor_of_safety

        # Uplift (tension) capacity: rule-of-thumb fraction of the OUTSIDE
        # skin friction only (inside/plug friction excluded — mobilizing it
        # in tension requires the soil-plug weight, not modeled here), plus
        # optional pile self-weight. See class Notes.
        Q_uplift = None
        if self.include_uplift:
            Q_uplift = self.uplift_skin_fraction * Qs_outside
            if self.pile_weight is not None:
                Q_uplift += self.pile_weight

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

    def _layer_skin_friction(self, layer: AxialSoilLayer, z_top: float,
                             z_bottom: float, perimeter: float
                             ) -> Tuple[float, float, str]:
        """Skin friction for one layer segment of the pile.

        Integrates with the midpoint rule, splitting the segment at the
        groundwater table when it falls strictly inside: sigma_v' is
        piecewise linear with a kink at the GWT, so midpoint x thickness
        on each side of the kink is exact (unit friction is linear in
        sigma_v' for the Nordlund and beta methods; the Tomlinson alpha
        method does not use sigma_v').

        Returns
        -------
        (Qs, sigma_v_mid, method_used) : tuple
            Segment skin friction (kN), effective stress at the full
            segment midpoint (kPa, for reporting), and the method label.
        """
        # Sub-segment boundaries: split at the GWT if inside the segment
        gwt = self.soil.gwt_depth
        if gwt is not None and z_top < gwt < z_bottom:
            cuts = [z_top, gwt, z_bottom]
        else:
            cuts = [z_top, z_bottom]

        Qs = 0.0
        method_used = ""
        for za, zb in zip(cuts[:-1], cuts[1:]):
            seg_thickness = zb - za
            sigma_v = self.soil.effective_stress_at_depth((za + zb) / 2)

            if self.method == "beta":
                phi = (layer.friction_angle
                       if layer.soil_type == "cohesionless"
                       else self.cohesive_phi)
                beta = beta_from_phi(phi)
                Qs += skin_friction_beta(
                    sigma_v, beta, perimeter, seg_thickness
                )
                method_used = "beta"
            elif layer.soil_type == "cohesionless":
                pile_mat = "concrete" if "concrete" in self.pile.pile_type else "steel"
                Qs += skin_friction_cohesionless(
                    layer.friction_angle, sigma_v,
                    perimeter, seg_thickness,
                    pile_material=pile_mat,
                    delta_phi_ratio=layer.delta_phi_ratio,
                )
                method_used = "nordlund"
            else:
                pile_type = "concrete" if "concrete" in self.pile.pile_type else "steel"
                Qs += skin_friction_cohesive(
                    layer.cohesion, perimeter,
                    seg_thickness, pile_type=pile_type,
                )
                method_used = "tomlinson"

        sigma_v_mid = self.soil.effective_stress_at_depth((z_top + z_bottom) / 2)
        return Qs, sigma_v_mid, method_used

    def _compute_inside_skin_friction(self) -> float:
        """Compute inside skin friction for open-ended pipe pile (unplugged).

        Uses same unit friction as outside, applied to inner perimeter
        over the embedded length.

        Returns
        -------
        float
            Inside skin friction (kN).
        """
        inner_perim = self.pile.inner_perimeter
        if inner_perim is None or inner_perim <= 0:
            return 0.0

        Qs_inside = 0.0
        current_depth = 0.0
        tip_depth = self._tip_depth

        for layer in self.soil.layers:
            layer_top = current_depth
            layer_bottom = current_depth + layer.thickness
            current_depth = layer_bottom

            if layer_top >= tip_depth:
                break

            z_top = max(layer_top, self.head_depth)
            z_bottom = min(layer_bottom, tip_depth)
            if z_bottom - z_top <= 0:
                continue

            Qs_layer, _, _ = self._layer_skin_friction(
                layer, z_top, z_bottom, inner_perim
            )
            Qs_inside += Qs_layer

        return Qs_inside

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

        # Use a copy per trial depth rather than temporarily mutating
        # self.pile_length (re-entrant / thread-safe).
        for d in depths:
            try:
                trial = replace(self, pile_length=float(d))
                r = trial.compute()
                results.append({
                    "depth_m": round(d, 2),
                    "Q_ultimate_kN": round(r.Q_ultimate, 1),
                    "Q_skin_kN": round(r.Q_skin, 1),
                    "Q_tip_kN": round(r.Q_tip, 1),
                })
            except Exception:
                pass
        return results
