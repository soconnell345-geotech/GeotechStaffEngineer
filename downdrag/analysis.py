"""
Downdrag (negative skin friction) analysis using the neutral plane method.

Implements the Fellenius unified method (2004/2006) for pile downdrag:
1. Force equilibrium to find the neutral plane depth.
2. Settlement compatibility (soil settlement = pile settlement at NP).
3. Structural and geotechnical limit state checks.

Supports fill placement and groundwater drawdown as settlement triggers.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
- Fellenius, B.H. (2004). "Unified design of piled foundations with emphasis
  on settlement analysis." ASCE GSP 125.
- AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7.
- UFC 3-220-20, 16 Jan 2025, Chapter 6, Eqs 6-51 through 6-53, 6-80.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from downdrag.soil import DowndragSoilProfile, DowndragSoilLayer
from downdrag.results import DowndragResult


@dataclass
class DowndragAnalysis:
    """Downdrag analysis using the Fellenius unified neutral plane method.

    Parameters
    ----------
    soil : DowndragSoilProfile
        Soil profile with strength and consolidation parameters.
    pile_length : float
        Embedded pile length (m).
    pile_diameter : float
        Pile diameter (m). Used to compute perimeter if not provided.
    pile_perimeter : float, optional
        Pile perimeter (m). If None, computed as pi*D.
    pile_area : float, optional
        Pile cross-sectional area (m^2). If None, computed as pi/4*D^2.
    pile_E : float
        Pile Young's modulus (kPa). Default 200e6 (steel).
    pile_unit_weight : float
        Pile material unit weight (kN/m^3). Default 24.0 (concrete).
        For steel pipe piles, use ~78.5 * (area_ratio).
    Q_dead : float
        Dead load at pile head (kN). Only dead load causes downdrag.
    structural_capacity : float, optional
        Factored structural resistance of pile (kN). For limit state check.
    allowable_settlement : float, optional
        Allowable pile settlement (m). For serviceability check.
    fill_thickness : float
        Thickness of new fill placed at ground surface (m). Default 0.
    fill_unit_weight : float
        Unit weight of fill material (kN/m^3). Default 19.0.
    gw_drawdown : float
        Groundwater drawdown from original GWT (m). Default 0.
    Nt : float, optional
        Toe bearing capacity factor. If None, estimated from tip layer phi.
    n_sublayers : int
        Number of sublayers per soil layer for discretization. Default 10.

    Examples
    --------
    >>> from downdrag import DowndragAnalysis, DowndragSoilProfile, DowndragSoilLayer
    >>> layers = [
    ...     DowndragSoilLayer(thickness=3.0, soil_type="cohesionless",
    ...         unit_weight=19.0, phi=30.0, description="Fill"),
    ...     DowndragSoilLayer(thickness=10.0, soil_type="cohesive",
    ...         unit_weight=17.0, cu=30.0, settling=True,
    ...         Cc=0.3, Cr=0.05, e0=1.0, description="Soft clay"),
    ...     DowndragSoilLayer(thickness=7.0, soil_type="cohesionless",
    ...         unit_weight=20.0, phi=35.0, description="Dense sand"),
    ... ]
    >>> soil = DowndragSoilProfile(layers=layers, gwt_depth=2.0)
    >>> analysis = DowndragAnalysis(
    ...     soil=soil, pile_length=18.0, pile_diameter=0.3,
    ...     Q_dead=500.0, fill_thickness=3.0, fill_unit_weight=19.0)
    >>> result = analysis.compute()
    >>> print(result.summary())
    """
    soil: DowndragSoilProfile
    pile_length: float
    pile_diameter: float
    pile_perimeter: Optional[float] = None
    pile_area: Optional[float] = None
    pile_E: float = 200e6
    pile_unit_weight: float = 24.0
    Q_dead: float = 0.0
    structural_capacity: Optional[float] = None
    allowable_settlement: Optional[float] = None
    fill_thickness: float = 0.0
    fill_unit_weight: float = 19.0
    gw_drawdown: float = 0.0
    Nt: Optional[float] = None
    n_sublayers: int = 10

    def __post_init__(self):
        if self.pile_length <= 0:
            raise ValueError(
                f"Pile length must be positive, got {self.pile_length}"
            )
        if self.pile_diameter <= 0:
            raise ValueError(
                f"Pile diameter must be positive, got {self.pile_diameter}"
            )
        if self.pile_perimeter is None:
            self.pile_perimeter = math.pi * self.pile_diameter
        if self.pile_area is None:
            self.pile_area = math.pi / 4.0 * self.pile_diameter**2
        if self.Q_dead < 0:
            raise ValueError(
                f"Dead load must be non-negative, got {self.Q_dead}"
            )
        if self.fill_thickness < 0:
            raise ValueError(
                f"fill_thickness must be non-negative, got {self.fill_thickness}"
            )
        if self.gw_drawdown < 0:
            raise ValueError(
                f"gw_drawdown must be non-negative, got {self.gw_drawdown}"
            )

    def compute(self) -> DowndragResult:
        """Run the downdrag analysis.

        Returns
        -------
        DowndragResult
            Analysis results including neutral plane depth, dragload,
            settlement, and limit state checks.
        """
        # Discretize the pile into sublayers
        z_nodes, dz = self._discretize()
        n = len(z_nodes)

        # Compute effective stress at each node
        sigma_v = np.array([
            self.soil.effective_stress_at_depth(z) for z in z_nodes
        ])

        # Compute unit skin friction at each node
        fs = self._compute_skin_friction(z_nodes, sigma_v)

        # Compute stress change at each node (from fill and/or GW drawdown)
        delta_sigma = self._compute_stress_change(z_nodes)

        # Compute soil settlement profile (cumulative from surface)
        soil_settlement = self._compute_soil_settlement(z_nodes, dz,
                                                         sigma_v, delta_sigma)

        # Compute toe resistance
        toe_resistance = self._compute_toe_resistance(sigma_v[-1])

        # Find neutral plane by force equilibrium
        z_np, drag_from_top, resist_from_tip = self._find_neutral_plane(
            z_nodes, dz, fs, toe_resistance
        )

        # Compute dragload and positive resistance
        dragload = self._compute_dragload(z_nodes, dz, fs, z_np)
        pile_weight_to_np = self.pile_unit_weight * self.pile_area * z_np
        max_pile_load = self.Q_dead + dragload + pile_weight_to_np

        positive_skin = self._compute_positive_resistance(z_nodes, dz, fs, z_np)
        total_resistance = positive_skin + toe_resistance

        # Compute axial load distribution along pile
        axial_load = self._compute_axial_load_distribution(z_nodes, dz, fs,
                                                            toe_resistance)

        # Compute pile settlement at neutral plane
        elastic_short = self._compute_elastic_shortening(z_nodes, dz,
                                                          axial_load, z_np)
        toe_settle = self._compute_toe_settlement(sigma_v[-1], delta_sigma[-1],
                                                     toe_resistance)
        pile_settlement = elastic_short + toe_settle

        # Settlement at the neutral plane from the soil profile
        # Interpolate soil settlement at z_np
        soil_settle_at_np = float(np.interp(z_np, z_nodes, soil_settlement))

        # Use the larger of pile settlement and soil settlement at NP
        # as the controlling settlement (they should be close if compatible)
        settlement = max(pile_settlement, soil_settle_at_np)

        # Limit state checks

        # Structural: UFC Eq 6-80 LRFD factored demand
        #   1.25*Q_dead + 1.10*(Q_np - Q_dead) <= P_r
        # where Q_np = max_pile_load (total load at neutral plane)
        structural_ok = None
        structural_demand = None
        if self.structural_capacity is not None:
            drag_force = max_pile_load - self.Q_dead  # dragload + pile weight
            structural_demand = 1.25 * self.Q_dead + 1.10 * drag_force
            structural_ok = bool(structural_demand <= self.structural_capacity)

        geotechnical_ok = None
        if total_resistance > 0:
            # Per Fellenius/AASHTO/UFC: dragload is NOT included in
            # geotechnical check — it cancels at the neutral plane
            geotechnical_ok = bool(self.Q_dead <= total_resistance)

        settlement_ok = None
        if self.allowable_settlement is not None:
            settlement_ok = bool(settlement <= self.allowable_settlement)

        return DowndragResult(
            neutral_plane_depth=z_np,
            dragload=dragload,
            max_pile_load=max_pile_load,
            Q_dead=self.Q_dead,
            pile_weight_to_np=pile_weight_to_np,
            positive_skin_friction=positive_skin,
            toe_resistance=toe_resistance,
            total_resistance=total_resistance,
            pile_settlement=pile_settlement,
            elastic_shortening=elastic_short,
            toe_settlement=toe_settle,
            soil_settlement_at_np=soil_settle_at_np,
            z=z_nodes,
            axial_load=axial_load,
            soil_settlement_profile=soil_settlement,
            unit_skin_friction=fs,
            structural_ok=structural_ok,
            structural_demand=structural_demand,
            geotechnical_ok=geotechnical_ok,
            settlement_ok=settlement_ok,
            pile_length=self.pile_length,
            pile_diameter=self.pile_diameter,
        )

    # ── Private helper methods ────────────────────────────────────────────

    def _discretize(self):
        """Create depth nodes along the pile.

        Returns
        -------
        z_nodes : numpy.ndarray
            Depth array from 0 to pile_length.
        dz : float
            Sublayer thickness.
        """
        total_nodes = max(
            int(self.pile_length / 0.25),  # ~0.25 m spacing
            self.n_sublayers * len(self.soil.layers),
            50,
        )
        z_nodes = np.linspace(0, self.pile_length, total_nodes + 1)
        dz = z_nodes[1] - z_nodes[0]
        return z_nodes, dz

    def _compute_skin_friction(self, z_nodes: np.ndarray,
                                sigma_v: np.ndarray) -> np.ndarray:
        """Compute unit skin friction (kPa) at each depth.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        sigma_v : numpy.ndarray
            Effective vertical stress at each node.

        Returns
        -------
        numpy.ndarray
            Unit skin friction fs (kPa) at each node.
        """
        fs = np.zeros(len(z_nodes))
        for i, z in enumerate(z_nodes):
            try:
                layer = self.soil.layer_at_depth(z)
            except ValueError:
                continue

            if layer.soil_type == "cohesionless":
                beta = layer.beta if layer.beta is not None else 0.3
                fs[i] = beta * sigma_v[i]
            else:
                alpha = layer.alpha if layer.alpha is not None else 1.0
                fs[i] = alpha * layer.cu

        return fs

    def _compute_stress_change(self, z_nodes: np.ndarray) -> np.ndarray:
        """Compute stress change at each depth from fill and/or GW drawdown.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.

        Returns
        -------
        numpy.ndarray
            Stress change delta_sigma (kPa) at each node.
        """
        delta_sigma = np.zeros(len(z_nodes))

        # Fill placement: uniform 1-D stress increase
        if self.fill_thickness > 0:
            delta_sigma += self.fill_thickness * self.fill_unit_weight

        # Groundwater drawdown: increase in effective stress
        if self.gw_drawdown > 0:
            original_gwt = self.soil.gwt_depth
            new_gwt = original_gwt + self.gw_drawdown
            for i, z in enumerate(z_nodes):
                if z > original_gwt and z <= new_gwt:
                    # This zone was below GWT, now above: full drawdown effect
                    delta_sigma[i] += self.soil.gamma_w * (z - original_gwt)
                elif z > new_gwt:
                    # Below new GWT: constant effect = full drawdown
                    delta_sigma[i] += self.soil.gamma_w * self.gw_drawdown

        return delta_sigma

    def _compute_soil_settlement(self, z_nodes: np.ndarray, dz: float,
                                  sigma_v: np.ndarray,
                                  delta_sigma: np.ndarray) -> np.ndarray:
        """Compute cumulative soil settlement profile.

        Settlement is accumulated from the bottom of the settling zone
        upward: S(z) = sum of sublayer settlements from z downward to
        the bottom of the settling zone.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        sigma_v : numpy.ndarray
            Initial effective stress at each node.
        delta_sigma : numpy.ndarray
            Stress change at each node.

        Returns
        -------
        numpy.ndarray
            Soil settlement (m) at each depth. Settlement at the surface
            is the total settlement; it decreases with depth.
        """
        n = len(z_nodes)
        sublayer_settlement = np.zeros(n)

        for i in range(n):
            z = z_nodes[i]
            try:
                layer = self.soil.layer_at_depth(z)
            except ValueError:
                continue

            if not layer.settling or delta_sigma[i] <= 0 or sigma_v[i] <= 0:
                continue

            if layer.soil_type == "cohesive":
                # Clay settlement: Eq 6-53 using modified compression indices
                sigma_p = (layer.sigma_p
                           if layer.sigma_p is not None else sigma_v[i])
                sublayer_settlement[i] = _settlement_clay(
                    H=dz, C_ec=layer.C_ec, C_er=layer.C_er,
                    sigma_v0=sigma_v[i], sigma_p=sigma_p,
                    delta_sigma=delta_sigma[i],
                )
            else:
                # Coarse-grained elastic settlement: Eq 6-54
                if layer.E_s is not None and layer.E_s > 0:
                    sublayer_settlement[i] = _settlement_sand_elastic(
                        H=dz, nu_s=layer.nu_s, E_s=layer.E_s,
                        delta_sigma=delta_sigma[i],
                    )

        # Cumulate from bottom upward: settlement at depth z is the sum
        # of all sublayer settlements below z
        cumulative = np.zeros(n)
        cumulative[-1] = sublayer_settlement[-1]
        for i in range(n - 2, -1, -1):
            cumulative[i] = cumulative[i + 1] + sublayer_settlement[i]

        return cumulative

    def _find_neutral_plane(self, z_nodes: np.ndarray, dz: float,
                             fs: np.ndarray,
                             toe_resistance: float):
        """Find neutral plane depth by force equilibrium.

        The neutral plane is where the cumulative load from the top
        (dead load + pile weight + dragload) equals the cumulative
        resistance from the bottom (toe + positive friction).

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction at each node (kPa).
        toe_resistance : float
            Toe bearing capacity (kN).

        Returns
        -------
        z_np : float
            Neutral plane depth (m).
        drag_from_top : numpy.ndarray
            Cumulative load from top at each node.
        resist_from_tip : numpy.ndarray
            Cumulative resistance from tip at each node.
        """
        n = len(z_nodes)
        perimeter = self.pile_perimeter
        pile_weight_per_m = self.pile_unit_weight * self.pile_area

        # Cumulative load from the top (dead load + pile weight + negative friction)
        drag_from_top = np.zeros(n)
        drag_from_top[0] = self.Q_dead
        for i in range(1, n):
            drag_from_top[i] = (drag_from_top[i - 1]
                                + pile_weight_per_m * dz
                                + fs[i] * perimeter * dz)

        # Cumulative resistance from the tip (toe + positive friction upward)
        resist_from_tip = np.zeros(n)
        resist_from_tip[-1] = toe_resistance
        for i in range(n - 2, -1, -1):
            resist_from_tip[i] = (resist_from_tip[i + 1]
                                  + fs[i + 1] * perimeter * dz)

        # Find crossing point: where drag_from_top = resist_from_tip
        diff = drag_from_top - resist_from_tip
        z_np = z_nodes[-1]  # default to pile tip

        for i in range(n - 1):
            if diff[i] <= 0 and diff[i + 1] > 0:
                # Linear interpolation for crossing
                frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
                z_np = z_nodes[i] + frac * dz
                break

        return z_np, drag_from_top, resist_from_tip

    def _compute_dragload(self, z_nodes: np.ndarray, dz: float,
                           fs: np.ndarray, z_np: float) -> float:
        """Compute negative skin friction (dragload) above neutral plane.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction at each node (kPa).
        z_np : float
            Neutral plane depth (m).

        Returns
        -------
        float
            Dragload (kN), positive value.
        """
        dragload = 0.0
        for i in range(len(z_nodes)):
            if z_nodes[i] >= z_np:
                break
            # Partial sublayer at NP boundary
            z_top = z_nodes[i]
            z_bot = min(z_nodes[i] + dz, z_np)
            thickness = z_bot - z_top
            if thickness > 0:
                dragload += fs[i] * self.pile_perimeter * thickness
        return dragload

    def _compute_positive_resistance(self, z_nodes: np.ndarray, dz: float,
                                      fs: np.ndarray, z_np: float) -> float:
        """Compute positive skin friction below neutral plane.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction at each node (kPa).
        z_np : float
            Neutral plane depth (m).

        Returns
        -------
        float
            Positive shaft resistance (kN).
        """
        positive = 0.0
        for i in range(len(z_nodes)):
            if z_nodes[i] < z_np:
                continue
            z_top = max(z_nodes[i], z_np)
            z_bot = z_nodes[i] + dz
            if z_bot > self.pile_length:
                z_bot = self.pile_length
            thickness = z_bot - z_top
            if thickness > 0:
                positive += fs[i] * self.pile_perimeter * thickness
        return positive

    def _compute_toe_resistance(self, sigma_v_tip: float) -> float:
        """Compute toe bearing resistance.

        Parameters
        ----------
        sigma_v_tip : float
            Effective vertical stress at pile tip (kPa).

        Returns
        -------
        float
            Toe resistance (kN).
        """
        tip_area = self.pile_area

        # Get tip layer
        try:
            tip_layer = self.soil.layer_at_depth(self.pile_length)
        except ValueError:
            return 0.0

        if self.Nt is not None:
            Nt = self.Nt
        elif tip_layer.phi > 0:
            Nt = _Nt_from_phi(tip_layer.phi)
        elif tip_layer.cu > 0:
            Nt = 9.0  # Nc for deep clay
        else:
            Nt = 0.0

        if tip_layer.soil_type == "cohesive":
            return Nt * tip_layer.cu * tip_area
        else:
            return Nt * sigma_v_tip * tip_area

    def _compute_axial_load_distribution(self, z_nodes: np.ndarray,
                                          dz: float, fs: np.ndarray,
                                          toe_resistance: float) -> np.ndarray:
        """Compute axial load distribution along the pile.

        Above NP: load increases (dead load + weight + dragload).
        Below NP: load decreases (positive friction removes load).
        This uses the force-from-top approach, which naturally produces
        the correct distribution.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction (kPa).
        toe_resistance : float
            Toe resistance (kN).

        Returns
        -------
        numpy.ndarray
            Axial load Q(z) at each node (kN).
        """
        n = len(z_nodes)
        Q = np.zeros(n)
        pile_weight_per_m = self.pile_unit_weight * self.pile_area

        # We build the load distribution from equilibrium:
        # Above NP: friction adds load (negative skin friction)
        # Below NP: friction removes load (positive resistance)
        # The crossing is the neutral plane (max load).
        # Simple approach: use the from-top accumulation (drag_from_top)
        Q[0] = self.Q_dead
        for i in range(1, n):
            Q[i] = Q[i - 1] + pile_weight_per_m * dz + fs[i] * self.pile_perimeter * dz

        # However, below the NP the friction should be subtracting.
        # The _find_neutral_plane method already found the NP.
        # Re-do: build from both ends and use the minimum envelope.
        Q_from_top = np.zeros(n)
        Q_from_top[0] = self.Q_dead
        for i in range(1, n):
            Q_from_top[i] = (Q_from_top[i - 1]
                             + pile_weight_per_m * dz
                             + fs[i] * self.pile_perimeter * dz)

        Q_from_bot = np.zeros(n)
        Q_from_bot[-1] = toe_resistance
        for i in range(n - 2, -1, -1):
            Q_from_bot[i] = (Q_from_bot[i + 1]
                             + fs[i + 1] * self.pile_perimeter * dz)

        # The actual axial load is the minimum of both curves at each depth
        # (above NP: from_top governs; below NP: from_bot governs)
        Q = np.minimum(Q_from_top, Q_from_bot)
        return Q

    def _compute_elastic_shortening(self, z_nodes: np.ndarray, dz: float,
                                     axial_load: np.ndarray,
                                     z_np: float) -> float:
        """Compute elastic shortening of the pile above the neutral plane.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        axial_load : numpy.ndarray
            Axial load distribution Q(z) (kN).
        z_np : float
            Neutral plane depth (m).

        Returns
        -------
        float
            Elastic shortening (m).
        """
        AE = self.pile_area * self.pile_E
        if AE <= 0:
            return 0.0

        shortening = 0.0
        for i in range(len(z_nodes) - 1):
            if z_nodes[i] >= z_np:
                break
            # Average load in this segment
            Q_avg = 0.5 * (axial_load[i] + axial_load[i + 1])
            seg_len = min(z_nodes[i + 1], z_np) - z_nodes[i]
            if seg_len > 0:
                shortening += Q_avg * seg_len / AE

        return shortening

    def _compute_toe_settlement(self, sigma_v_tip: float,
                                 delta_sigma_tip: float,
                                 toe_resistance: float) -> float:
        """Estimate settlement of the bearing stratum below the pile tip.

        Uses the equivalent footing concept (UFC Eqs 6-49/6-50) with
        2V:1H stress distribution (Eq 6-51) into the bearing stratum.
        The equivalent footing width B' = pile_diameter (single pile).
        Settlement is computed for sublayers within an influence zone
        of 3*B' below the pile tip.

        Parameters
        ----------
        sigma_v_tip : float
            Effective stress at pile tip (kPa).
        delta_sigma_tip : float
            Stress change at pile tip from fill/GW (kPa).
        toe_resistance : float
            Toe bearing resistance (kN) for stress distribution below tip.

        Returns
        -------
        float
            Estimated toe settlement (m).
        """
        # Equivalent footing dimensions (single pile: B' = L' = diameter)
        B_prime = self.pile_diameter
        L_prime = self.pile_diameter

        # Influence zone: 3*B' below pile tip
        influence_depth = 3.0 * B_prime
        n_sub = max(int(influence_depth / 0.25), 10)
        dz_sub = influence_depth / n_sub

        total_settle = 0.0
        for j in range(n_sub):
            z_below_tip = (j + 0.5) * dz_sub  # midpoint depth below tip
            z_abs = self.pile_length + z_below_tip

            # Get the layer at this depth
            try:
                layer = self.soil.layer_at_depth(z_abs)
            except ValueError:
                break

            # Stress change from pile load using 2V:1H (Eq 6-51)
            denom = (B_prime + z_below_tip) * (L_prime + z_below_tip)
            delta_sigma_pile = toe_resistance / denom if denom > 0 else 0.0

            # Additional stress from fill/GW at this depth
            delta_sigma_total = delta_sigma_pile + delta_sigma_tip

            if delta_sigma_total <= 0:
                continue

            # Effective stress at this depth
            sigma_v0 = self.soil.effective_stress_at_depth(z_abs)
            if sigma_v0 <= 0:
                continue

            if layer.soil_type == "cohesive" and layer.C_ec is not None:
                sigma_p = (layer.sigma_p
                           if layer.sigma_p is not None else sigma_v0)
                total_settle += _settlement_clay(
                    H=dz_sub, C_ec=layer.C_ec, C_er=layer.C_er,
                    sigma_v0=sigma_v0, sigma_p=sigma_p,
                    delta_sigma=delta_sigma_total,
                )
            elif (layer.soil_type == "cohesionless"
                  and layer.E_s is not None and layer.E_s > 0):
                total_settle += _settlement_sand_elastic(
                    H=dz_sub, nu_s=layer.nu_s, E_s=layer.E_s,
                    delta_sigma=delta_sigma_total,
                )

        return total_settle


# ── Module-level helper functions ─────────────────────────────────────────

def _settlement_clay(H: float, C_ec: float, C_er: float,
                     sigma_v0: float, sigma_p: float,
                     delta_sigma: float) -> float:
    """Settlement of clay using modified compression indices (UFC Eq 6-53).

    Uses modified compression indices C_ec and C_er (= Cc/(1+e0) and
    Cr/(1+e0) respectively). Three cases:

    1. NC (sigma_v0 >= sigma_p): Sc = C_ec * H * log10(sigma_final/sigma_v0)
    2. OC stays OC (sigma_final <= sigma_p): Sc = C_er * H * log10(...)
    3. OC → NC: recompression to sigma_p, then virgin compression beyond

    Parameters
    ----------
    H : float
        Sublayer thickness (m).
    C_ec : float
        Modified compression index = Cc/(1+e0).
    C_er : float
        Modified recompression index = Cr/(1+e0).
    sigma_v0 : float
        Initial effective vertical stress (kPa).
    sigma_p : float
        Preconsolidation pressure (kPa).
    delta_sigma : float
        Stress change (kPa).

    Returns
    -------
    float
        Settlement (m).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-53.
    """
    if delta_sigma <= 0 or sigma_v0 <= 0:
        return 0.0

    sigma_final = sigma_v0 + delta_sigma

    # Treat as NC if sigma_v0 is within 5% of sigma_p
    is_NC = abs(sigma_p - sigma_v0) / sigma_v0 < 0.05

    if is_NC:
        return C_ec * H * math.log10(sigma_final / sigma_v0)
    elif sigma_final <= sigma_p:
        return C_er * H * math.log10(sigma_final / sigma_v0)
    else:
        Sc_oc = C_er * H * math.log10(sigma_p / sigma_v0)
        Sc_nc = C_ec * H * math.log10(sigma_final / sigma_p)
        return Sc_oc + Sc_nc


def _settlement_sand_elastic(H: float, nu_s: float, E_s: float,
                             delta_sigma: float) -> float:
    """Elastic settlement of coarse-grained soil (UFC Eq 6-54).

    .. math::
        delta_s = H * (1+nu_s)*(1-2*nu_s) / ((1-nu_s)*E_s) * delta_sigma

    Parameters
    ----------
    H : float
        Sublayer thickness (m).
    nu_s : float
        Poisson's ratio (dimensionless).
    E_s : float
        Young's modulus of soil (kPa).
    delta_sigma : float
        Stress change (kPa).

    Returns
    -------
    float
        Settlement (m).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-54.
    """
    if E_s <= 0 or delta_sigma <= 0:
        return 0.0
    return H * (1.0 + nu_s) * (1.0 - 2.0 * nu_s) / ((1.0 - nu_s) * E_s) * delta_sigma


def _consolidation_settlement(H: float, e0: float, Cc: float, Cr: float,
                               sigma_v0: float, sigma_p: float,
                               delta_sigma: float) -> float:
    """Legacy wrapper: convert traditional Cc/Cr/e0 to modified indices.

    Kept for backward compatibility with tests using the traditional API.
    Delegates to _settlement_clay().
    """
    C_ec = Cc / (1.0 + e0) if e0 > 0 else 0.0
    C_er = Cr / (1.0 + e0) if e0 > 0 else 0.0
    return _settlement_clay(H, C_ec, C_er, sigma_v0, sigma_p, delta_sigma)


def _Nt_from_phi(phi_deg: float) -> float:
    """Estimate toe bearing capacity factor Nt from friction angle.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nt.

    References
    ----------
    Fellenius (1991), FHWA GEC-12 Table 7-9.
    """
    if phi_deg <= 0:
        return 3.0
    elif phi_deg <= 20:
        return 3.0 + (phi_deg / 20) * 7
    elif phi_deg <= 28:
        return 10 + (phi_deg - 20) * 2.5
    elif phi_deg <= 33:
        return 30 + (phi_deg - 28) * 8
    elif phi_deg <= 38:
        return 70 + (phi_deg - 33) * 16
    else:
        return 150 + (phi_deg - 38) * 20
