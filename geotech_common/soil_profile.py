"""
Unified soil profile for all geotechnical analysis modules.

Provides a single canonical representation of the subsurface that every
module consumes. Adapter methods translate the universal profile into each
module's specific input format.

All units are SI:
    Depth: meters (m), measured from ground surface (positive downward)
    Stress/Pressure: kilopascals (kPa)
    Unit weight: kN/m3
    Angles: degrees
    Lengths: meters (m)

References:
    - Terzaghi, Peck & Mesri (1996) — Soil Mechanics in Engineering Practice
    - NAVFAC DM-7.01 (1986) — Soil Mechanics
    - Kulhawy & Mayne (1990) — EPRI EL-6800
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any

from geotech_common.water import GAMMA_W, pore_pressure, effective_unit_weight
from geotech_common.soil_properties import spt_to_phi, spt_to_cu
from geotech_common.validation import (
    check_positive, check_non_negative, check_range, check_unit_weight,
)


# ---------------------------------------------------------------------------
# SoilLayer
# ---------------------------------------------------------------------------

@dataclass
class SoilLayer:
    """A single layer in the subsurface profile.

    Provide whatever parameters are available from field and lab data.
    Missing parameters can be estimated via SoilProfile.fill_missing_from_correlations().

    Parameters
    ----------
    top_depth : float
        Depth to top of layer (m), 0 = ground surface.
    bottom_depth : float
        Depth to bottom of layer (m).
    description : str
        Soil description, e.g. "Soft gray clay (CH)".

    Classification
    --------------
    uscs : str, optional
        USCS symbol: CH, CL, SP, SM, GP, GW, ML, MH, OL, OH, PT, etc.
    is_cohesive : bool, optional
        True = clay/silt, False = sand/gravel.  If None, inferred from uscs.
    is_rock : bool
        True if this layer is rock.

    Strength
    --------
    cu : float, optional
        Undrained shear strength (kPa).
    phi : float, optional
        Effective friction angle (degrees).
    c_prime : float, optional
        Effective cohesion (kPa).
    qu : float, optional
        Unconfined compressive strength — for rock (kPa).
    RQD : float, optional
        Rock Quality Designation (%).

    Index and Physical
    ------------------
    gamma : float, optional
        Total unit weight (kN/m3).
    gamma_sat : float, optional
        Saturated unit weight (kN/m3).
    e0 : float, optional
        Initial void ratio.
    wn : float, optional
        Natural water content (%).
    LL : float, optional
        Liquid limit (%).
    PL : float, optional
        Plastic limit (%).
    PI : float, optional
        Plasticity index (%).

    Consolidation
    -------------
    Cc : float, optional
        Compression index.
    Cr : float, optional
        Recompression index.
    Cv : float, optional
        Coefficient of consolidation (m2/year).
    sigma_p : float, optional
        Preconsolidation pressure (kPa).
    C_alpha : float, optional
        Secondary compression index.

    Stiffness
    ---------
    Es : float, optional
        Elastic modulus (kPa).
    eps50 : float, optional
        Strain at 50% of max stress (for p-y curves).
    k_py : float, optional
        p-y subgrade reaction modulus (kN/m3).

    Field Test Data
    ---------------
    N_spt : float, optional
        Average SPT N-value (blows/ft), uncorrected.
    N60 : float, optional
        Energy-corrected SPT N.
    N160 : float, optional
        Overburden + energy corrected (N1)60.
    qc : float, optional
        CPT tip resistance (kPa).
    fs : float, optional
        CPT sleeve friction (kPa).
    """

    # Required
    top_depth: float
    bottom_depth: float
    description: str

    # Classification
    uscs: Optional[str] = None
    is_cohesive: Optional[bool] = None
    is_rock: bool = False

    # Strength
    cu: Optional[float] = None
    phi: Optional[float] = None
    c_prime: Optional[float] = None
    qu: Optional[float] = None
    RQD: Optional[float] = None

    # Index / physical
    gamma: Optional[float] = None
    gamma_sat: Optional[float] = None
    e0: Optional[float] = None
    wn: Optional[float] = None
    LL: Optional[float] = None
    PL: Optional[float] = None
    PI: Optional[float] = None

    # Consolidation
    Cc: Optional[float] = None
    Cr: Optional[float] = None
    Cv: Optional[float] = None
    sigma_p: Optional[float] = None
    C_alpha: Optional[float] = None

    # Stiffness
    Es: Optional[float] = None
    eps50: Optional[float] = None
    k_py: Optional[float] = None

    # Field test data
    N_spt: Optional[float] = None
    N60: Optional[float] = None
    N160: Optional[float] = None
    qc: Optional[float] = None
    fs: Optional[float] = None

    # Tracking which values were estimated (not measured)
    _estimated_fields: Dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.bottom_depth <= self.top_depth:
            raise ValueError(
                f"bottom_depth ({self.bottom_depth}) must be > top_depth ({self.top_depth})"
            )
        if self.top_depth < 0:
            raise ValueError(f"top_depth must be >= 0, got {self.top_depth}")
        if self.gamma is not None:
            check_unit_weight(self.gamma, "gamma")
        if self.gamma_sat is not None:
            check_unit_weight(self.gamma_sat, "gamma_sat")
        if self.gamma is not None and self.gamma_sat is not None:
            if self.gamma_sat < self.gamma:
                warnings.warn(
                    f"Layer '{self.description}': gamma_sat ({self.gamma_sat}) < "
                    f"gamma ({self.gamma}) — saturated weight should be >= total weight"
                )
        if self.phi is not None:
            check_range(self.phi, "phi", 0, 50)
        if self.cu is not None:
            check_non_negative(self.cu, "cu")
        if self.is_cohesive is None and self.uscs is not None:
            self.is_cohesive = _infer_cohesive_from_uscs(self.uscs)

    @property
    def thickness(self) -> float:
        """Layer thickness (m)."""
        return self.bottom_depth - self.top_depth

    @property
    def mid_depth(self) -> float:
        """Depth to midpoint of layer (m)."""
        return (self.top_depth + self.bottom_depth) / 2.0

    def mark_estimated(self, field_name: str, correlation: str) -> None:
        """Record that a field was estimated rather than measured.

        Parameters
        ----------
        field_name : str
            Name of the estimated field (e.g. "phi", "cu").
        correlation : str
            Description of the correlation used (e.g. "Peck (1974) from N60=25").
        """
        self._estimated_fields[field_name] = correlation

    def is_estimated(self, field_name: str) -> bool:
        """Check if a field value was estimated from correlations."""
        return field_name in self._estimated_fields

    def get_estimation_log(self) -> Dict[str, str]:
        """Return dict of {field_name: correlation_description} for all estimated values."""
        return dict(self._estimated_fields)


# ---------------------------------------------------------------------------
# GroundwaterCondition
# ---------------------------------------------------------------------------

@dataclass
class GroundwaterCondition:
    """Groundwater levels for the site.

    Parameters
    ----------
    depth : float
        Depth to groundwater table below ground surface (m).
    is_artesian : bool
        True if piezometric head is above ground surface.
    artesian_head : float, optional
        Piezometric head above ground surface if artesian (m).
    """

    depth: float
    is_artesian: bool = False
    artesian_head: Optional[float] = None

    def __post_init__(self):
        check_non_negative(self.depth, "Groundwater depth")
        if self.is_artesian and self.artesian_head is None:
            raise ValueError("artesian_head required when is_artesian=True")
        if self.is_artesian and self.artesian_head is not None:
            check_positive(self.artesian_head, "artesian_head")


# ---------------------------------------------------------------------------
# SoilProfile
# ---------------------------------------------------------------------------

@dataclass
class SoilProfile:
    """Complete subsurface profile for a location.

    Parameters
    ----------
    layers : List[SoilLayer]
        Soil layers, sorted top to bottom.  Must be continuous (no gaps/overlaps).
    groundwater : GroundwaterCondition
        Groundwater conditions at the site.
    ground_elevation : float, optional
        Elevation of ground surface (m, relative to datum).
    location_name : str, optional
        Site or location identifier.
    boring_id : str, optional
        Boring or CPT sounding identifier.
    """

    layers: List[SoilLayer]
    groundwater: GroundwaterCondition
    ground_elevation: Optional[float] = None
    location_name: Optional[str] = None
    boring_id: Optional[str] = None

    def __post_init__(self):
        if not self.layers:
            raise ValueError("SoilProfile must have at least one layer")
        # Sort layers by top_depth
        self.layers = sorted(self.layers, key=lambda l: l.top_depth)
        # Validate continuity
        for i in range(len(self.layers) - 1):
            if abs(self.layers[i].bottom_depth - self.layers[i + 1].top_depth) > 1e-6:
                raise ValueError(
                    f"Gap or overlap between layer {i} (bottom={self.layers[i].bottom_depth}) "
                    f"and layer {i+1} (top={self.layers[i+1].top_depth})"
                )

    @property
    def total_depth(self) -> float:
        """Total depth of the profile (m)."""
        return self.layers[-1].bottom_depth

    # ── Layer lookup ──────────────────────────────────────────────────

    def layer_at_depth(self, z: float) -> SoilLayer:
        """Return the soil layer at depth z (m below ground surface).

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        SoilLayer
            The layer containing depth z.

        Raises
        ------
        ValueError
            If z is outside the profile depth range.
        """
        if z < self.layers[0].top_depth - 1e-9:
            raise ValueError(f"Depth {z} is above the profile (top={self.layers[0].top_depth})")
        for layer in self.layers:
            if layer.top_depth - 1e-9 <= z <= layer.bottom_depth + 1e-9:
                return layer
        raise ValueError(
            f"Depth {z} is below the profile (bottom={self.layers[-1].bottom_depth})"
        )

    def layers_in_range(self, z_top: float, z_bottom: float) -> List[SoilLayer]:
        """Return all layers that intersect the depth range [z_top, z_bottom].

        Parameters
        ----------
        z_top : float
            Top of range (m).
        z_bottom : float
            Bottom of range (m).

        Returns
        -------
        List[SoilLayer]
            Layers overlapping the specified range.
        """
        result = []
        for layer in self.layers:
            if layer.bottom_depth > z_top + 1e-9 and layer.top_depth < z_bottom - 1e-9:
                result.append(layer)
        return result

    # ── Stress calculations ───────────────────────────────────────────

    def total_stress_at_depth(self, z: float) -> float:
        """Compute total vertical stress at depth z.

        Integrates gamma*dz through each layer from ground surface to depth z.
        Uses gamma_sat below the water table if available, otherwise gamma.

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        float
            Total vertical stress sigma_v (kPa).
        """
        if z <= 0:
            return 0.0
        gwt = self.groundwater.depth
        sigma_v = 0.0
        z_prev = 0.0

        for layer in self.layers:
            if z_prev >= z:
                break
            z_top_layer = max(layer.top_depth, z_prev)
            z_bot_layer = min(layer.bottom_depth, z)

            if z_bot_layer <= z_top_layer:
                continue

            gamma = self._get_gamma_for_stress(layer, z_top_layer, z_bot_layer, gwt)
            sigma_v += gamma * (z_bot_layer - z_top_layer)
            z_prev = z_bot_layer

        return sigma_v

    def pore_pressure_at_depth(self, z: float) -> float:
        """Compute pore water pressure at depth z.

        Hydrostatic below the water table.  For artesian conditions, uses the
        artesian head.

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        float
            Pore water pressure u (kPa).
        """
        gwt = self.groundwater.depth
        if self.groundwater.is_artesian and self.groundwater.artesian_head is not None:
            # Piezometric surface is above ground by artesian_head
            depth_below_piezo = z + self.groundwater.artesian_head
            return max(0.0, GAMMA_W * depth_below_piezo)
        return pore_pressure(z - gwt)

    def effective_stress_at_depth(self, z: float) -> float:
        """Compute effective vertical stress at depth z.

        sigma_v' = sigma_v - u

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        float
            Effective vertical stress sigma_v' (kPa).
        """
        return self.total_stress_at_depth(z) - self.pore_pressure_at_depth(z)

    def effective_unit_weight_at_depth(self, z: float) -> float:
        """Return effective unit weight at depth z.

        Above the water table: gamma (total).
        Below the water table: gamma_sat - gamma_w (buoyant).

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        float
            Effective unit weight (kN/m3).
        """
        layer = self.layer_at_depth(z)
        gwt = self.groundwater.depth
        if z >= gwt:
            gamma_sat = layer.gamma_sat if layer.gamma_sat is not None else layer.gamma
            if gamma_sat is None:
                raise ValueError(
                    f"Layer '{layer.description}' at depth {z}m has no unit weight defined"
                )
            return effective_unit_weight(gamma_sat)
        else:
            if layer.gamma is None:
                raise ValueError(
                    f"Layer '{layer.description}' at depth {z}m has no unit weight defined"
                )
            return layer.gamma

    def _get_gamma_for_stress(self, layer: SoilLayer,
                              z_top: float, z_bot: float,
                              gwt: float) -> float:
        """Get the appropriate unit weight for a depth interval within a layer.

        For intervals fully above GWT: use gamma.
        For intervals fully below GWT: use gamma_sat (or gamma if gamma_sat not set).
        For intervals spanning GWT: weighted average.
        """
        gamma_dry = layer.gamma
        gamma_wet = layer.gamma_sat if layer.gamma_sat is not None else layer.gamma

        if gamma_dry is None and gamma_wet is None:
            raise ValueError(
                f"Layer '{layer.description}' has no unit weight (gamma or gamma_sat)"
            )
        if gamma_dry is None:
            gamma_dry = gamma_wet
        if gamma_wet is None:
            gamma_wet = gamma_dry

        if z_bot <= gwt:
            # Entirely above GWT
            return gamma_dry
        elif z_top >= gwt:
            # Entirely below GWT
            return gamma_wet
        else:
            # Spans the GWT — weighted average
            h_above = gwt - z_top
            h_below = z_bot - gwt
            h_total = z_bot - z_top
            return (gamma_dry * h_above + gamma_wet * h_below) / h_total

    # ── Validation ────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Check the profile for internal consistency and flag issues.

        Returns
        -------
        List[str]
            Warning strings prefixed with severity: INFO, WARNING, or CRITICAL.
            Empty list means no issues found.
        """
        warnings_list: List[str] = []

        # Layer continuity (already enforced in __post_init__ but recheck)
        if self.layers[0].top_depth > 0.1:
            warnings_list.append(
                f"WARNING: Profile starts at depth {self.layers[0].top_depth}m, "
                "not at ground surface (0m)"
            )

        for i, layer in enumerate(self.layers):
            prefix = f"Layer {i} ('{layer.description}', {layer.top_depth}-{layer.bottom_depth}m)"

            # Unit weight ranges
            if layer.gamma is not None:
                if layer.gamma < 14.0:
                    warnings_list.append(
                        f"WARNING: {prefix}: gamma={layer.gamma} kN/m3 is low "
                        "(typical 14-24 kN/m3, organic soil?)"
                    )
                elif layer.gamma > 24.0:
                    warnings_list.append(
                        f"WARNING: {prefix}: gamma={layer.gamma} kN/m3 is high "
                        "(typical 14-24 kN/m3)"
                    )

            if layer.gamma is None and layer.gamma_sat is None:
                warnings_list.append(
                    f"WARNING: {prefix}: no unit weight defined — "
                    "stress calculations will fail"
                )

            # Friction angle
            if layer.phi is not None:
                if layer.phi > 45:
                    warnings_list.append(
                        f"WARNING: {prefix}: phi={layer.phi}deg is very high — verify"
                    )
                if layer.phi < 20 and layer.is_cohesive is False:
                    warnings_list.append(
                        f"WARNING: {prefix}: phi={layer.phi}deg is low for granular soil"
                    )

            # Cohesion
            if layer.cu is not None and layer.cu > 500:
                warnings_list.append(
                    f"WARNING: {prefix}: cu={layer.cu} kPa is very high — "
                    "verify (typical soft clay <50, stiff <200)"
                )

            # SPT
            if layer.N_spt is not None:
                if layer.N_spt > 100:
                    warnings_list.append(
                        f"INFO: {prefix}: N_spt={layer.N_spt} — practical refusal, "
                        "may indicate rock or very dense material"
                    )

            # Consolidation parameters
            if layer.Cc is not None and layer.Cc > 1.0:
                warnings_list.append(
                    f"WARNING: {prefix}: Cc={layer.Cc} is very high — "
                    "typical for organic soils only"
                )

            if layer.Cr is not None and layer.Cc is not None:
                if layer.Cr > layer.Cc:
                    warnings_list.append(
                        f"CRITICAL: {prefix}: Cr={layer.Cr} > Cc={layer.Cc} — "
                        "recompression index must be less than compression index"
                    )

            # Preconsolidation vs. current stress
            if layer.sigma_p is not None and layer.gamma is not None:
                sigma_v_mid = self.effective_stress_at_depth(layer.mid_depth)
                if sigma_v_mid > 0 and layer.sigma_p < sigma_v_mid * 0.9:
                    warnings_list.append(
                        f"WARNING: {prefix}: sigma_p={layer.sigma_p:.0f} kPa < "
                        f"sigma_v'={sigma_v_mid:.0f} kPa — soil appears underconsolidated"
                    )
                if sigma_v_mid > 0:
                    ocr = layer.sigma_p / sigma_v_mid
                    if ocr > 10:
                        warnings_list.append(
                            f"INFO: {prefix}: OCR={ocr:.1f} is very high — verify"
                        )

            # Consistency checks between parameters
            if layer.cu is not None and layer.phi is not None and layer.phi > 0:
                warnings_list.append(
                    f"INFO: {prefix}: both cu and phi specified — "
                    "ensure correct analysis type (total stress vs effective stress)"
                )

            if layer.N_spt is not None and layer.cu is not None:
                # Check N_spt vs cu consistency
                expected_cu = 6.25 * layer.N_spt
                if layer.cu > 0 and abs(layer.cu - expected_cu) / layer.cu > 1.0:
                    warnings_list.append(
                        f"INFO: {prefix}: cu={layer.cu} kPa vs N_spt={layer.N_spt} "
                        f"(Terzaghi-Peck would give ~{expected_cu:.0f} kPa) — "
                        "verify with lab data"
                    )

            if layer.N_spt is not None and layer.phi is not None:
                expected_phi = spt_to_phi(layer.N_spt)
                if abs(layer.phi - expected_phi) > 8:
                    warnings_list.append(
                        f"INFO: {prefix}: phi={layer.phi}deg vs N_spt={layer.N_spt} "
                        f"(Peck would give ~{expected_phi:.0f}deg) — verify"
                    )

            # Void ratio
            if layer.e0 is not None:
                if layer.e0 > 3.0:
                    warnings_list.append(
                        f"WARNING: {prefix}: e0={layer.e0} is very high — "
                        "typical for highly organic/peat soils only"
                    )
                if layer.e0 < 0.2:
                    warnings_list.append(
                        f"WARNING: {prefix}: e0={layer.e0} is very low — "
                        "verify (typical sand ~0.5-0.8, clay ~0.6-1.5)"
                    )

            # RQD
            if layer.RQD is not None:
                if layer.RQD < 0 or layer.RQD > 100:
                    warnings_list.append(
                        f"CRITICAL: {prefix}: RQD={layer.RQD}% must be 0-100"
                    )

        # GW depth vs profile
        gwt = self.groundwater.depth
        if gwt > self.total_depth:
            warnings_list.append(
                f"INFO: Groundwater at {gwt}m is below profile bottom "
                f"({self.total_depth}m) — entire profile is above GWT"
            )

        return warnings_list

    # ── Fill missing from correlations ────────────────────────────────

    def fill_missing_from_correlations(self) -> List[str]:
        """Estimate missing parameters from available data using standard correlations.

        Uses well-established empirical relationships to fill gaps in the soil
        data.  Every estimated value is flagged via SoilLayer.mark_estimated()
        so the LLM agent can disclose what is measured vs. estimated.

        Returns
        -------
        List[str]
            Log of all estimations performed, e.g.:
            "Layer 0 'Soft clay': estimated cu=31.3 kPa from N60=5 (Terzaghi & Peck)"
        """
        log: List[str] = []

        for i, layer in enumerate(self.layers):
            prefix = f"Layer {i} '{layer.description}'"

            # --- Infer is_cohesive from USCS if not set ---
            if layer.is_cohesive is None and layer.uscs is not None:
                layer.is_cohesive = _infer_cohesive_from_uscs(layer.uscs)

            # --- N60 from N_spt (assume 60% energy if not specified) ---
            if layer.N60 is None and layer.N_spt is not None:
                layer.N60 = layer.N_spt  # Common assumption: hammer delivers ~60% energy
                layer.mark_estimated("N60", f"Assumed N60 = N_spt = {layer.N_spt}")
                log.append(f"{prefix}: assumed N60 = N_spt = {layer.N_spt}")

            # --- PI from LL and PL ---
            if layer.PI is None and layer.LL is not None and layer.PL is not None:
                layer.PI = layer.LL - layer.PL
                layer.mark_estimated("PI", f"PI = LL - PL = {layer.LL} - {layer.PL}")
                log.append(f"{prefix}: computed PI = {layer.PI:.0f}%")

            # --- phi from N60 (granular soils) ---
            if (layer.phi is None and layer.N60 is not None
                    and not layer.is_rock
                    and (layer.is_cohesive is False or layer.is_cohesive is None)):
                # Only for granular soils or unknown soil type with no cu
                if layer.cu is None:
                    phi_est = spt_to_phi(layer.N60)
                    layer.phi = phi_est
                    corr = f"Peck (1974) from N60={layer.N60}"
                    layer.mark_estimated("phi", corr)
                    log.append(f"{prefix}: estimated phi={phi_est:.1f}deg ({corr})")

            # --- cu from N60 (cohesive soils) ---
            if (layer.cu is None and layer.N60 is not None
                    and not layer.is_rock
                    and layer.is_cohesive is True):
                cu_est = spt_to_cu(layer.N60)
                layer.cu = cu_est
                corr = f"Terzaghi & Peck from N60={layer.N60}"
                layer.mark_estimated("cu", corr)
                log.append(f"{prefix}: estimated cu={cu_est:.1f} kPa ({corr})")

            # --- gamma from USCS and N_spt ---
            if layer.gamma is None and not layer.is_rock:
                gamma_est = _estimate_gamma(layer)
                if gamma_est is not None:
                    layer.gamma = gamma_est
                    layer.mark_estimated("gamma", "Typical value from soil type/N_spt")
                    log.append(f"{prefix}: estimated gamma={gamma_est:.1f} kN/m3")

            # --- gamma_sat from gamma and e0 ---
            if layer.gamma_sat is None and layer.gamma is not None:
                if layer.e0 is not None:
                    # gamma_sat = (Gs + e) / (1 + e) * gamma_w
                    # Approximate Gs = gamma * (1+e) / (Gs_assumed * gamma_w)
                    # Simpler: gamma_sat = gamma + n * gamma_w where n = e/(1+e)
                    n = layer.e0 / (1.0 + layer.e0)
                    gamma_sat_est = layer.gamma + n * GAMMA_W * (1 - layer.gamma / (2.65 * GAMMA_W))
                    # Simplified: just use gamma_sat ~ gamma + 1-3 kN/m3
                    # Better formula: gamma_sat = (Gs*gamma_w + e*gamma_w) / (1+e)
                    Gs = 2.65  # assumed
                    gamma_sat_est = (Gs + layer.e0) * GAMMA_W / (1.0 + layer.e0)
                    layer.gamma_sat = gamma_sat_est
                    layer.mark_estimated("gamma_sat",
                                         f"From e0={layer.e0}, Gs=2.65 assumed")
                    log.append(
                        f"{prefix}: estimated gamma_sat={gamma_sat_est:.1f} kN/m3 "
                        f"from e0={layer.e0}"
                    )
                else:
                    # Rule of thumb: gamma_sat ~ gamma + 1 to 2 kN/m3
                    gamma_sat_est = layer.gamma + 1.5
                    if gamma_sat_est > 24.0:
                        gamma_sat_est = layer.gamma + 0.5
                    layer.gamma_sat = gamma_sat_est
                    layer.mark_estimated("gamma_sat",
                                         f"Approximate: gamma + 1.5 kN/m3")
                    log.append(
                        f"{prefix}: estimated gamma_sat={gamma_sat_est:.1f} kN/m3 "
                        "(gamma + 1.5)"
                    )

            # --- Cc from LL (Terzaghi & Peck) ---
            if layer.Cc is None and layer.LL is not None and layer.is_cohesive is True:
                Cc_est = 0.009 * (layer.LL - 10)
                if Cc_est > 0:
                    layer.Cc = Cc_est
                    layer.mark_estimated("Cc", f"Terzaghi: Cc = 0.009*(LL-10), LL={layer.LL}")
                    log.append(
                        f"{prefix}: estimated Cc={Cc_est:.3f} "
                        f"(Terzaghi, LL={layer.LL})"
                    )

            # --- Cr from Cc (rule of thumb) ---
            if layer.Cr is None and layer.Cc is not None and layer.is_cohesive is True:
                Cr_est = layer.Cc / 6.0  # Typical Cr/Cc ratio ~ 1/5 to 1/10
                layer.Cr = Cr_est
                layer.mark_estimated("Cr", f"Rule of thumb: Cr ≈ Cc/6 = {layer.Cc:.3f}/6")
                log.append(f"{prefix}: estimated Cr={Cr_est:.4f} (Cc/6)")

            # --- Es from N_spt or cu ---
            if layer.Es is None and not layer.is_rock:
                Es_est = _estimate_Es(layer)
                if Es_est is not None:
                    layer.Es = Es_est
                    layer.mark_estimated("Es", "Empirical from N60 or cu")
                    log.append(f"{prefix}: estimated Es={Es_est:.0f} kPa")

            # --- eps50 from cu for cohesive soils (for lateral pile p-y) ---
            if (layer.eps50 is None and layer.cu is not None
                    and layer.is_cohesive is True):
                eps50_est = _estimate_eps50(layer.cu)
                layer.eps50 = eps50_est
                layer.mark_estimated("eps50", f"Typical value for cu={layer.cu:.0f} kPa")
                log.append(f"{prefix}: estimated eps50={eps50_est:.4f}")

        return log

    # ── String representation ─────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the profile."""
        lines = []
        if self.location_name:
            lines.append(f"Site: {self.location_name}")
        if self.boring_id:
            lines.append(f"Boring: {self.boring_id}")
        lines.append(f"GWT: {self.groundwater.depth}m below ground surface")
        if self.groundwater.is_artesian:
            lines.append(f"  Artesian head: {self.groundwater.artesian_head}m above ground")
        lines.append(f"Total depth: {self.total_depth}m")
        lines.append("")
        lines.append(f"{'Depth (m)':>12}  {'Description':<30}  {'Key Parameters'}")
        lines.append("-" * 80)
        for layer in self.layers:
            depth_str = f"{layer.top_depth:.1f}-{layer.bottom_depth:.1f}"
            params = []
            if layer.gamma is not None:
                params.append(f"gamma={layer.gamma:.1f}")
            if layer.cu is not None:
                est = "*" if layer.is_estimated("cu") else ""
                params.append(f"cu={layer.cu:.0f}{est}")
            if layer.phi is not None:
                est = "*" if layer.is_estimated("phi") else ""
                params.append(f"phi={layer.phi:.0f}{est}")
            if layer.N_spt is not None:
                params.append(f"N={layer.N_spt:.0f}")
            if layer.N60 is not None and layer.N_spt is None:
                params.append(f"N60={layer.N60:.0f}")
            param_str = ", ".join(params)
            lines.append(f"{depth_str:>12}  {layer.description:<30}  {param_str}")
        lines.append("")
        lines.append("* = estimated from correlation")
        return "\n".join(lines)

    # ── Adapter: Bearing Capacity ─────────────────────────────────────

    def to_bearing_capacity_input(self, footing_depth: float) -> Dict[str, Any]:
        """Convert profile to bearing_capacity module input format.

        Returns a dict with keys that can be unpacked to create
        BearingSoilProfile and its SoilLayer objects.

        The adapter picks the layer at footing depth as layer1, and the
        next layer below (if any) as layer2.  For cohesive layers with
        cu > 0, it uses total-stress (phi=0) analysis; for granular layers
        it uses effective-stress (c=0, phi) analysis.

        Parameters
        ----------
        footing_depth : float
            Depth of footing base below ground surface (m).

        Returns
        -------
        dict
            Keys: "layer1" (dict), "layer2" (dict or None), "gwt_depth" (float).
            Each layer dict has: cohesion, friction_angle, unit_weight,
            thickness (for layer1 if layer2 exists), description.
        """
        layer = self.layer_at_depth(footing_depth)
        layer1 = self._layer_to_bearing(layer)

        # Find next layer below
        layer2_dict = None
        layers_below = self.layers_in_range(layer.bottom_depth,
                                            layer.bottom_depth + 0.01)
        # Actually get the next layer
        for lyr in self.layers:
            if lyr.top_depth >= layer.bottom_depth - 1e-6 and lyr is not layer:
                layer2_dict = self._layer_to_bearing(lyr)
                # Set layer1 thickness (distance from footing to layer boundary)
                layer1["thickness"] = layer.bottom_depth - footing_depth
                break

        return {
            "layer1": layer1,
            "layer2": layer2_dict,
            "gwt_depth": self.groundwater.depth,
        }

    def _layer_to_bearing(self, layer: "SoilLayer") -> Dict[str, Any]:
        """Convert a unified SoilLayer to bearing_capacity SoilLayer dict."""
        gamma = layer.gamma if layer.gamma is not None else 18.0

        if layer.is_cohesive is True and layer.cu is not None and layer.cu > 0:
            # Total-stress (undrained): phi=0, c=cu
            return {
                "cohesion": layer.cu,
                "friction_angle": 0.0,
                "unit_weight": gamma,
                "thickness": None,
                "description": layer.description,
            }
        elif layer.phi is not None and layer.phi > 0:
            c_prime = layer.c_prime if layer.c_prime is not None else 0.0
            return {
                "cohesion": c_prime,
                "friction_angle": layer.phi,
                "unit_weight": gamma,
                "thickness": None,
                "description": layer.description,
            }
        else:
            # Fallback: use whatever is available
            c = layer.cu if layer.cu is not None else 0.0
            phi = layer.phi if layer.phi is not None else 0.0
            if c == 0 and phi == 0:
                raise ValueError(
                    f"Layer '{layer.description}' has neither cu nor phi — "
                    "cannot create bearing capacity input. "
                    "Run fill_missing_from_correlations() first."
                )
            return {
                "cohesion": c,
                "friction_angle": phi,
                "unit_weight": gamma,
                "thickness": None,
                "description": layer.description,
            }

    # ── Adapter: Settlement ───────────────────────────────────────────

    def to_settlement_input(self, footing_depth: float,
                            footing_width: float,
                            footing_length: Optional[float] = None,
                            influence_depth_factor: float = 2.0,
                            ) -> Dict[str, Any]:
        """Convert profile to settlement module input format.

        Builds ConsolidationLayer dicts for layers below the footing, and
        provides elastic modulus for immediate settlement.

        Parameters
        ----------
        footing_depth : float
            Depth of footing base (m).
        footing_width : float
            Footing width B (m).
        footing_length : float, optional
            Footing length L (m). If None, assumed square (L=B).
        influence_depth_factor : float
            Depth of influence as multiple of B below footing. Default 2.0.

        Returns
        -------
        dict
            Keys: "q_overburden" (float, kPa), "B" (float), "L" (float),
            "Es_immediate" (float or None, kPa),
            "consolidation_layers" (list of dicts with thickness,
            depth_to_center, e0, Cc, Cr, sigma_v0, sigma_p, description).
        """
        L = footing_length if footing_length is not None else footing_width
        q_overburden = self.effective_stress_at_depth(footing_depth)
        influence_depth = footing_depth + influence_depth_factor * footing_width

        # Build consolidation layers for cohesive soil below footing
        consol_layers = []
        Es_values = []

        for layer in self.layers:
            # Only consider portions below footing base and within influence zone
            z_top = max(layer.top_depth, footing_depth)
            z_bot = min(layer.bottom_depth, influence_depth)
            if z_bot <= z_top:
                continue

            thickness = z_bot - z_top
            depth_to_center = ((z_top + z_bot) / 2.0) - footing_depth
            sigma_v0 = self.effective_stress_at_depth((z_top + z_bot) / 2.0)

            # Elastic modulus for immediate settlement
            if layer.Es is not None:
                Es_values.append((thickness, layer.Es))

            # Consolidation layers (only for cohesive soils with Cc)
            if (layer.is_cohesive is True and
                    layer.Cc is not None and layer.Cc > 0):
                e0 = layer.e0 if layer.e0 is not None else 0.8
                Cr = layer.Cr if layer.Cr is not None else layer.Cc / 6.0
                sigma_p = layer.sigma_p  # None = normally consolidated

                consol_layers.append({
                    "thickness": thickness,
                    "depth_to_center": depth_to_center,
                    "e0": e0,
                    "Cc": layer.Cc,
                    "Cr": Cr,
                    "sigma_v0": sigma_v0,
                    "sigma_p": sigma_p,
                    "description": layer.description,
                })

        # Weighted average Es for immediate settlement
        Es_immediate = None
        if Es_values:
            total_h = sum(h for h, _ in Es_values)
            Es_immediate = sum(h * es for h, es in Es_values) / total_h

        return {
            "q_overburden": q_overburden,
            "B": footing_width,
            "L": L,
            "Es_immediate": Es_immediate,
            "consolidation_layers": consol_layers,
        }

    # ── Adapter: Axial Pile ───────────────────────────────────────────

    def to_axial_pile_input(self, pile_length: float) -> Dict[str, Any]:
        """Convert profile to axial_pile module input format.

        Builds AxialSoilLayer dicts for each layer the pile passes through,
        clipped to pile length.

        Parameters
        ----------
        pile_length : float
            Pile embedment length (m).

        Returns
        -------
        dict
            Keys: "layers" (list of dicts with thickness, soil_type,
            unit_weight, friction_angle, cohesion, description),
            "gwt_depth" (float or None).
        """
        axial_layers = []

        for layer in self.layers:
            z_top = layer.top_depth
            z_bot = min(layer.bottom_depth, pile_length)
            if z_top >= pile_length:
                break
            thickness = z_bot - z_top
            if thickness <= 0:
                continue

            gamma = layer.gamma if layer.gamma is not None else 18.0

            if layer.is_cohesive is True and layer.cu is not None and layer.cu > 0:
                axial_layers.append({
                    "thickness": thickness,
                    "soil_type": "cohesive",
                    "unit_weight": gamma,
                    "friction_angle": 0.0,
                    "cohesion": layer.cu,
                    "description": layer.description,
                })
            elif layer.phi is not None and layer.phi > 0:
                axial_layers.append({
                    "thickness": thickness,
                    "soil_type": "cohesionless",
                    "unit_weight": gamma,
                    "friction_angle": layer.phi,
                    "cohesion": 0.0,
                    "description": layer.description,
                })
            else:
                raise ValueError(
                    f"Layer '{layer.description}' has neither cu nor phi — "
                    "cannot create axial pile input. "
                    "Run fill_missing_from_correlations() first."
                )

        return {
            "layers": axial_layers,
            "gwt_depth": self.groundwater.depth,
        }

    # ── Adapter: Lateral Pile ─────────────────────────────────────────

    def to_lateral_pile_input(self, pile_length: float,
                              pile_diameter: float,
                              loading: str = "static",
                              ) -> Dict[str, Any]:
        """Convert profile to lateral_pile module input format.

        Automatically selects p-y curve model for each layer:
        - Soft/medium clay (cu < 100, below GWT): SoftClayMatlock
        - Stiff clay (cu >= 100, below GWT): StiffClayBelowWT
        - Stiff clay (above GWT): StiffClayAboveWT
        - Sand: SandAPI
        - Rock: WeakRock

        Parameters
        ----------
        pile_length : float
            Pile embedded length (m).
        pile_diameter : float
            Pile diameter (m).
        loading : str
            "static" or "cyclic". Default "static".

        Returns
        -------
        dict
            Keys: "layers" (list of dicts with top, bottom, py_model_type,
            py_model_params, description).
            The caller must instantiate the actual p-y model objects
            from the returned type+params.
        """
        gwt = self.groundwater.depth
        lateral_layers = []

        for layer in self.layers:
            z_top = layer.top_depth
            z_bot = min(layer.bottom_depth, pile_length)
            if z_top >= pile_length:
                break
            if z_bot <= z_top:
                continue

            gamma_eff = self._get_effective_gamma(layer, z_top, z_bot, gwt)
            model_info = self._select_py_model(layer, gamma_eff, gwt,
                                               z_top, z_bot, loading)

            lateral_layers.append({
                "top": z_top,
                "bottom": z_bot,
                "py_model_type": model_info["type"],
                "py_model_params": model_info["params"],
                "description": layer.description,
            })

        return {"layers": lateral_layers}

    def _get_effective_gamma(self, layer: "SoilLayer",
                             z_top: float, z_bot: float,
                             gwt: float) -> float:
        """Get effective (buoyant) unit weight for a layer interval."""
        gamma = layer.gamma if layer.gamma is not None else 18.0
        gamma_sat = layer.gamma_sat if layer.gamma_sat is not None else gamma

        mid = (z_top + z_bot) / 2.0
        if mid >= gwt:
            return gamma_sat - GAMMA_W
        else:
            return gamma

    def _select_py_model(self, layer: "SoilLayer",
                         gamma_eff: float, gwt: float,
                         z_top: float, z_bot: float,
                         loading: str) -> Dict[str, Any]:
        """Select appropriate p-y model and parameters for a layer."""
        mid = (z_top + z_bot) / 2.0
        below_gwt = mid >= gwt

        if layer.is_rock:
            qu = layer.qu if layer.qu is not None else 1000.0
            Er = 200.0 * qu  # Default estimate
            gamma_r = layer.gamma if layer.gamma is not None else 22.0
            rqd = layer.RQD if layer.RQD is not None else 50.0
            return {
                "type": "WeakRock",
                "params": {"qu": qu, "Er": Er, "gamma_r": gamma_r,
                           "RQD": rqd, "loading": loading},
            }

        if layer.is_cohesive is True and layer.cu is not None:
            eps50 = layer.eps50 if layer.eps50 is not None else _estimate_eps50(layer.cu)

            if layer.cu < 100:
                # Soft/medium clay
                return {
                    "type": "SoftClayMatlock",
                    "params": {"c": layer.cu, "gamma": gamma_eff,
                               "eps50": eps50, "J": 0.5,
                               "loading": loading},
                }
            else:
                # Stiff clay
                k_py = layer.k_py if layer.k_py is not None else _estimate_k_stiff_clay(layer.cu)
                if below_gwt:
                    return {
                        "type": "StiffClayBelowWT",
                        "params": {"c": layer.cu, "gamma": gamma_eff,
                                   "eps50": eps50, "ks": k_py,
                                   "loading": loading},
                    }
                else:
                    return {
                        "type": "StiffClayAboveWT",
                        "params": {"c": layer.cu, "gamma": gamma_eff,
                                   "eps50": eps50,
                                   "loading": loading},
                    }

        # Granular soil
        if layer.phi is not None and layer.phi > 0:
            k = layer.k_py if layer.k_py is not None else _estimate_k_sand(layer.phi, below_gwt)
            return {
                "type": "SandAPI",
                "params": {"phi": layer.phi, "gamma": gamma_eff,
                           "k": k, "loading": loading},
            }

        raise ValueError(
            f"Layer '{layer.description}' has neither cu nor phi — "
            "cannot select p-y model. Run fill_missing_from_correlations() first."
        )

    # ── Adapter: Sheet Pile ───────────────────────────────────────────

    def to_sheet_pile_input(self, excavation_depth: float) -> Dict[str, Any]:
        """Convert profile to sheet_pile module input format.

        Builds WallSoilLayer dicts for all layers from ground surface
        through the profile.

        Parameters
        ----------
        excavation_depth : float
            Excavation depth (m).

        Returns
        -------
        dict
            Keys: "layers" (list of dicts with thickness, unit_weight,
            friction_angle, cohesion, description),
            "excavation_depth" (float).
        """
        wall_layers = []

        for layer in self.layers:
            gamma = layer.gamma if layer.gamma is not None else 18.0
            phi = layer.phi if layer.phi is not None else 0.0
            c = 0.0

            if layer.is_cohesive is True and layer.cu is not None:
                c = layer.cu
                if phi == 0:
                    phi = 0.0  # Total stress analysis
            elif layer.c_prime is not None:
                c = layer.c_prime

            # Need at least phi or c > 0
            if phi == 0 and c == 0:
                raise ValueError(
                    f"Layer '{layer.description}' has neither phi nor cu/c' — "
                    "cannot create sheet pile input. "
                    "Run fill_missing_from_correlations() first."
                )

            wall_layers.append({
                "thickness": layer.thickness,
                "unit_weight": gamma,
                "friction_angle": phi,
                "cohesion": c,
                "description": layer.description,
            })

        return {
            "layers": wall_layers,
            "excavation_depth": excavation_depth,
        }

    # ── Adapter: Pile Group ───────────────────────────────────────────

    def to_pile_group_input(self, pile_length: float,
                            pile_diameter: float) -> Dict[str, Any]:
        """Convert profile to pile_group module input format.

        Provides soil parameters needed for group efficiency and
        p-multiplier calculations.

        Parameters
        ----------
        pile_length : float
            Pile embedment length (m).
        pile_diameter : float
            Pile diameter (m).

        Returns
        -------
        dict
            Keys: "average_phi" (float, degrees — weighted by thickness),
            "average_cu" (float, kPa — weighted by thickness for cohesive layers),
            "gwt_depth" (float), "pile_length" (float),
            "pile_diameter" (float), "is_cohesive" (bool — dominant soil type).
        """
        total_phi_thickness = 0.0
        total_phi_weight = 0.0
        total_cu_thickness = 0.0
        total_cu_weight = 0.0
        cohesive_thickness = 0.0
        granular_thickness = 0.0

        for layer in self.layers:
            z_top = layer.top_depth
            z_bot = min(layer.bottom_depth, pile_length)
            if z_top >= pile_length:
                break
            h = z_bot - z_top
            if h <= 0:
                continue

            if layer.is_cohesive is True and layer.cu is not None:
                total_cu_weight += layer.cu * h
                total_cu_thickness += h
                cohesive_thickness += h
            if layer.phi is not None and layer.phi > 0:
                total_phi_weight += layer.phi * h
                total_phi_thickness += h
                granular_thickness += h

        avg_phi = total_phi_weight / total_phi_thickness if total_phi_thickness > 0 else 0
        avg_cu = total_cu_weight / total_cu_thickness if total_cu_thickness > 0 else 0

        return {
            "average_phi": avg_phi,
            "average_cu": avg_cu,
            "gwt_depth": self.groundwater.depth,
            "pile_length": pile_length,
            "pile_diameter": pile_diameter,
            "is_cohesive": cohesive_thickness > granular_thickness,
        }

    # ── Adapter: Drilled Shaft ─────────────────────────────────────────

    def to_drilled_shaft_input(self, shaft_length: float) -> Dict[str, Any]:
        """Convert profile to drilled_shaft module input format.

        Builds ShaftSoilLayer dicts for each layer the shaft passes through,
        clipped to shaft length.  Soil type is classified as "cohesive",
        "cohesionless", or "rock" based on available parameters.

        Parameters
        ----------
        shaft_length : float
            Total shaft embedment length (m).

        Returns
        -------
        dict
            Keys: "layers" (list of dicts with thickness, soil_type,
            unit_weight, cu, phi, N60, qu, RQD, description),
            "gwt_depth" (float or None).
        """
        shaft_layers = []

        for layer in self.layers:
            z_top = layer.top_depth
            z_bot = min(layer.bottom_depth, shaft_length)
            if z_top >= shaft_length:
                break
            thickness = z_bot - z_top
            if thickness <= 0:
                continue

            gamma = layer.gamma if layer.gamma is not None else 18.0
            soil_type = self._classify_soil_type(layer)

            cu = layer.cu if layer.cu is not None else 0.0
            phi = layer.phi if layer.phi is not None else 0.0
            N60 = layer.N60 if layer.N60 is not None else 0.0
            qu = layer.qu if layer.qu is not None else 0.0
            RQD = layer.RQD if layer.RQD is not None else 100.0

            # Validate minimum requirements by soil type
            if soil_type == "cohesive" and cu <= 0:
                raise ValueError(
                    f"Layer '{layer.description}' classified as cohesive but cu <= 0. "
                    "Run fill_missing_from_correlations() first."
                )
            if soil_type == "cohesionless" and phi <= 0:
                raise ValueError(
                    f"Layer '{layer.description}' classified as cohesionless but phi <= 0. "
                    "Run fill_missing_from_correlations() first."
                )
            if soil_type == "rock" and qu <= 0:
                raise ValueError(
                    f"Layer '{layer.description}' classified as rock but qu <= 0."
                )

            shaft_layers.append({
                "thickness": thickness,
                "soil_type": soil_type,
                "unit_weight": gamma,
                "cu": cu,
                "phi": phi,
                "N60": N60,
                "qu": qu,
                "RQD": RQD,
                "description": layer.description,
            })

        return {
            "layers": shaft_layers,
            "gwt_depth": self.groundwater.depth,
        }

    @staticmethod
    def _classify_soil_type(layer: "SoilLayer") -> str:
        """Classify a SoilLayer as cohesive/cohesionless/rock for drilled shaft."""
        if layer.is_rock:
            return "rock"
        if layer.is_cohesive is True:
            return "cohesive"
        if layer.is_cohesive is False:
            return "cohesionless"
        # Infer from available data
        if layer.cu is not None and layer.cu > 0:
            return "cohesive"
        if layer.phi is not None and layer.phi > 0:
            return "cohesionless"
        raise ValueError(
            f"Layer '{layer.description}' cannot be classified — "
            "set is_cohesive, is_rock, or provide cu/phi."
        )

    # ── Adapter: Retaining Wall ────────────────────────────────────────

    def to_retaining_wall_input(self, wall_height: float,
                                 surcharge: float = 0.0,
                                 ) -> Dict[str, Any]:
        """Convert profile to retaining_walls module input format.

        Uses a weighted average of the soil behind the wall (from surface
        to wall_height) as the backfill, and the layer at wall_height as the
        foundation soil.

        Parameters
        ----------
        wall_height : float
            Retained height of the wall (m).
        surcharge : float, optional
            Uniform surcharge on backfill (kPa). Default 0.

        Returns
        -------
        dict
            Keys: "gamma_backfill" (float, kN/m³),
            "phi_backfill" (float, degrees), "c_backfill" (float, kPa),
            "gamma_foundation" (float, kN/m³),
            "phi_foundation" (float, degrees),
            "c_foundation" (float, kPa),
            "surcharge" (float, kPa).
        """
        # Weighted average of backfill properties (surface to wall_height)
        total_gamma_h = 0.0
        total_phi_h = 0.0
        total_c_h = 0.0
        total_h = 0.0

        for layer in self.layers:
            z_top = layer.top_depth
            z_bot = min(layer.bottom_depth, wall_height)
            if z_top >= wall_height:
                break
            h = z_bot - z_top
            if h <= 0:
                continue

            gamma = layer.gamma if layer.gamma is not None else 18.0
            total_gamma_h += gamma * h
            total_h += h

            if layer.is_cohesive is True and layer.cu is not None:
                total_c_h += layer.cu * h
                # For undrained: phi=0, c=cu
            elif layer.phi is not None and layer.phi > 0:
                total_phi_h += layer.phi * h
                if layer.c_prime is not None:
                    total_c_h += layer.c_prime * h
            else:
                raise ValueError(
                    f"Layer '{layer.description}' has neither cu nor phi — "
                    "cannot create retaining wall input. "
                    "Run fill_missing_from_correlations() first."
                )

        if total_h <= 0:
            raise ValueError("No layers within wall height range.")

        gamma_backfill = total_gamma_h / total_h
        phi_backfill = total_phi_h / total_h
        c_backfill = total_c_h / total_h

        # Foundation soil: layer at or just below wall base
        foundation_layer = self.layer_at_depth(wall_height)
        if foundation_layer is None:
            foundation_layer = self.layers[-1]

        gamma_fdn = foundation_layer.gamma if foundation_layer.gamma is not None else 18.0
        if foundation_layer.phi is not None and foundation_layer.phi > 0:
            phi_fdn = foundation_layer.phi
            c_fdn = foundation_layer.c_prime if foundation_layer.c_prime is not None else 0.0
        elif foundation_layer.is_cohesive is True and foundation_layer.cu is not None:
            phi_fdn = 0.0
            c_fdn = foundation_layer.cu
        else:
            phi_fdn = phi_backfill
            c_fdn = c_backfill

        return {
            "gamma_backfill": round(gamma_backfill, 1),
            "phi_backfill": round(phi_backfill, 1),
            "c_backfill": round(c_backfill, 1),
            "gamma_foundation": round(gamma_fdn, 1),
            "phi_foundation": round(phi_fdn, 1),
            "c_foundation": round(c_fdn, 1),
            "surcharge": surcharge,
        }

    # ── Adapter: Seismic Geotechnical ──────────────────────────────────

    def to_seismic_input(self,
                         amax_g: float = 0.0,
                         magnitude: float = 7.5,
                         ) -> Dict[str, Any]:
        """Convert profile to seismic_geotech module input format.

        Provides data for site classification (N-bar, su-bar for top 30m)
        and liquefaction evaluation (per-layer N160, fines content, gamma).

        Note: Vs30-based classification requires shear wave velocity data
        which is not stored on SoilLayer.  If Vs data is available, compute
        Vs30 externally using seismic_geotech.site_class.compute_vs30().

        Parameters
        ----------
        amax_g : float, optional
            Peak ground acceleration (fraction of g). Default 0 (classification only).
        magnitude : float, optional
            Earthquake magnitude for liquefaction. Default 7.5.

        Returns
        -------
        dict
            Keys:
            - "site_classification": dict with "thicknesses", "N_values",
              "su_values" (lists for N-bar / su-bar in top 30m)
            - "liquefaction": dict with "depths", "N160", "FC", "gamma"
              (parallel lists for each granular layer below GWT)
            - "amax_g" (float), "magnitude" (float), "gwt_depth" (float)
        """
        # ── Site classification data (top 30m) ──
        n_thicknesses = []
        n_values = []
        su_thicknesses = []
        su_values = []

        for layer in self.layers:
            z_top = layer.top_depth
            z_bot = min(layer.bottom_depth, 30.0)
            if z_top >= 30.0:
                break
            h = z_bot - z_top
            if h <= 0:
                continue

            # N-bar: use N60 (or N_spt) for granular and mixed soils
            N = layer.N60 if layer.N60 is not None else layer.N_spt
            if N is not None and N > 0 and not layer.is_rock:
                n_thicknesses.append(h)
                n_values.append(N)

            # su-bar: use cu for cohesive layers
            if (layer.is_cohesive is True and
                    layer.cu is not None and layer.cu > 0):
                su_thicknesses.append(h)
                su_values.append(layer.cu)

        # ── Liquefaction data (granular layers below GWT) ──
        liq_depths = []
        liq_N160 = []
        liq_FC = []
        liq_gamma = []
        gwt = self.groundwater.depth

        for layer in self.layers:
            # Only evaluate granular layers below or crossing GWT
            if layer.is_rock:
                continue
            if layer.is_cohesive is True:
                continue

            z_top = max(layer.top_depth, gwt)
            z_bot = layer.bottom_depth
            if z_top >= z_bot:
                continue  # entirely above GWT

            # Use midpoint of saturated portion
            z_mid = (z_top + z_bot) / 2.0
            gamma = layer.gamma if layer.gamma is not None else 18.0

            # N160 — prefer explicit, then fall back to N60
            N160 = layer.N160
            if N160 is None:
                N160 = layer.N60 if layer.N60 is not None else layer.N_spt
            if N160 is None or N160 <= 0:
                continue  # Can't evaluate without SPT data

            # Fines content: use PI as proxy if available, else assume 5%
            FC = 5.0  # default for "clean" sand
            if layer.PI is not None and layer.PI > 0:
                # Rough estimate: higher PI → more fines
                FC = min(layer.PI * 2.0, 100.0)
            elif layer.uscs is not None:
                # Infer from USCS: SM/ML have more fines
                uscs_upper = layer.uscs.upper()
                if uscs_upper in ("SM", "GM", "ML"):
                    FC = 25.0
                elif uscs_upper in ("SC", "GC", "MH"):
                    FC = 40.0
                elif uscs_upper in ("SP", "GP", "SW", "GW"):
                    FC = 5.0

            liq_depths.append(z_mid)
            liq_N160.append(float(N160))
            liq_FC.append(FC)
            liq_gamma.append(gamma)

        return {
            "site_classification": {
                "n_thicknesses": n_thicknesses,
                "N_values": n_values,
                "su_thicknesses": su_thicknesses,
                "su_values": su_values,
            },
            "liquefaction": {
                "depths": liq_depths,
                "N160": liq_N160,
                "FC": liq_FC,
                "gamma": liq_gamma,
            },
            "amax_g": amax_g,
            "magnitude": magnitude,
            "gwt_depth": self.groundwater.depth,
        }


# ---------------------------------------------------------------------------
# SoilProfileBuilder
# ---------------------------------------------------------------------------

class SoilProfileBuilder:
    """Build a SoilProfile from common input formats.

    Examples
    --------
    From SPT boring log::

        builder = SoilProfileBuilder(gwt_depth=2.0)
        builder.add_spt_layer(0, 3, "Fill - silty sand", N=8, uscs="SM")
        builder.add_spt_layer(3, 8, "Soft gray clay (CH)", N=3, uscs="CH")
        builder.add_spt_layer(8, 15, "Medium dense sand (SP)", N=22, uscs="SP")
        profile = builder.build()

    From simple table::

        profile = SoilProfileBuilder.from_table(
            gwt_depth=3.0,
            layers=[
                {"top": 0, "bottom": 5, "desc": "Stiff clay",
                 "cu": 75, "gamma": 18.5},
                {"top": 5, "bottom": 12, "desc": "Dense sand",
                 "phi": 36, "gamma": 19.5},
            ]
        )
    """

    def __init__(self, gwt_depth: float = 0.0,
                 location_name: Optional[str] = None,
                 boring_id: Optional[str] = None,
                 ground_elevation: Optional[float] = None,
                 is_artesian: bool = False,
                 artesian_head: Optional[float] = None):
        self._gwt_depth = gwt_depth
        self._location_name = location_name
        self._boring_id = boring_id
        self._ground_elevation = ground_elevation
        self._is_artesian = is_artesian
        self._artesian_head = artesian_head
        self._layers: List[SoilLayer] = []

    def add_layer(self, layer: SoilLayer) -> "SoilProfileBuilder":
        """Add a pre-built SoilLayer."""
        self._layers.append(layer)
        return self

    def add_spt_layer(self, top: float, bottom: float, description: str,
                      N: float, uscs: Optional[str] = None,
                      gamma: Optional[float] = None,
                      **kwargs) -> "SoilProfileBuilder":
        """Add a layer from SPT boring log data.

        Parameters
        ----------
        top, bottom : float
            Depth range (m).
        description : str
            Soil description.
        N : float
            SPT blow count (assumed N60 unless otherwise noted).
        uscs : str, optional
            USCS classification symbol.
        gamma : float, optional
            Total unit weight (kN/m3).
        **kwargs
            Additional SoilLayer fields (cu, phi, e0, LL, etc.).
        """
        layer = SoilLayer(
            top_depth=top,
            bottom_depth=bottom,
            description=description,
            N_spt=N,
            N60=N,  # Assume 60% energy ratio for standard practice
            uscs=uscs,
            gamma=gamma,
            **kwargs,
        )
        self._layers.append(layer)
        return self

    def add_cpt_layer(self, top: float, bottom: float, description: str,
                      qc: float, fs: float,
                      uscs: Optional[str] = None,
                      gamma: Optional[float] = None,
                      **kwargs) -> "SoilProfileBuilder":
        """Add a layer from CPT data.

        Parameters
        ----------
        top, bottom : float
            Depth range (m).
        description : str
            Soil description.
        qc : float
            CPT tip resistance (kPa).
        fs : float
            CPT sleeve friction (kPa).
        uscs : str, optional
            USCS classification.
        gamma : float, optional
            Total unit weight (kN/m3).
        **kwargs
            Additional SoilLayer fields.
        """
        layer = SoilLayer(
            top_depth=top,
            bottom_depth=bottom,
            description=description,
            qc=qc,
            fs=fs,
            uscs=uscs,
            gamma=gamma,
            **kwargs,
        )
        self._layers.append(layer)
        return self

    def build(self, fill_correlations: bool = False) -> SoilProfile:
        """Build the SoilProfile.

        Parameters
        ----------
        fill_correlations : bool
            If True, automatically run fill_missing_from_correlations()
            after building.

        Returns
        -------
        SoilProfile
        """
        gw = GroundwaterCondition(
            depth=self._gwt_depth,
            is_artesian=self._is_artesian,
            artesian_head=self._artesian_head,
        )
        profile = SoilProfile(
            layers=self._layers,
            groundwater=gw,
            ground_elevation=self._ground_elevation,
            location_name=self._location_name,
            boring_id=self._boring_id,
        )
        if fill_correlations:
            profile.fill_missing_from_correlations()
        return profile

    @staticmethod
    def from_table(gwt_depth: float, layers: List[Dict[str, Any]],
                   location_name: Optional[str] = None,
                   boring_id: Optional[str] = None,
                   fill_correlations: bool = False) -> SoilProfile:
        """Build a SoilProfile from a simple list of dicts.

        Each dict must have keys: "top", "bottom", "desc".
        Optional keys map to SoilLayer fields: "cu", "phi", "gamma",
        "N_spt", "N60", "uscs", "e0", "LL", "PL", "Cc", "Cr", etc.

        Parameters
        ----------
        gwt_depth : float
            Groundwater depth (m).
        layers : List[dict]
            Layer definitions.
        location_name : str, optional
            Site name.
        boring_id : str, optional
            Boring ID.
        fill_correlations : bool
            If True, auto-fill missing parameters.

        Returns
        -------
        SoilProfile
        """
        # Map short keys to SoilLayer field names
        key_map = {
            "top": "top_depth",
            "bottom": "bottom_depth",
            "desc": "description",
            "description": "description",
        }
        soil_layers = []
        for lyr in layers:
            kwargs = {}
            for k, v in lyr.items():
                mapped = key_map.get(k, k)
                kwargs[mapped] = v
            soil_layers.append(SoilLayer(**kwargs))

        builder = SoilProfileBuilder(
            gwt_depth=gwt_depth,
            location_name=location_name,
            boring_id=boring_id,
        )
        for sl in soil_layers:
            builder.add_layer(sl)
        return builder.build(fill_correlations=fill_correlations)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_COHESIVE_USCS = {"CH", "CL", "MH", "ML", "OL", "OH", "PT", "CL-ML"}
_GRANULAR_USCS = {"GW", "GP", "GM", "GC", "SW", "SP", "SM", "SC"}


def _infer_cohesive_from_uscs(uscs: str) -> Optional[bool]:
    """Infer whether a soil is cohesive from its USCS symbol."""
    uscs_upper = uscs.upper().strip()
    if uscs_upper in _COHESIVE_USCS:
        return True
    if uscs_upper in _GRANULAR_USCS:
        return False
    return None


def _estimate_gamma(layer: SoilLayer) -> Optional[float]:
    """Estimate total unit weight from soil type and/or N_spt.

    Typical values (kN/m3):
        Soft clay:   16-18
        Stiff clay:  18-20
        Loose sand:  16-18
        Medium sand: 18-19
        Dense sand:  19-21
        Gravel:      20-22
    """
    if layer.is_cohesive is True:
        if layer.cu is not None:
            if layer.cu < 25:
                return 16.0
            elif layer.cu < 50:
                return 17.0
            elif layer.cu < 100:
                return 18.5
            else:
                return 19.5
        if layer.N60 is not None:
            if layer.N60 < 4:
                return 16.0
            elif layer.N60 < 8:
                return 17.5
            elif layer.N60 < 15:
                return 18.5
            else:
                return 19.5
        return 17.5  # Generic cohesive default

    elif layer.is_cohesive is False:
        if layer.N60 is not None:
            if layer.N60 < 10:
                return 17.0
            elif layer.N60 < 30:
                return 18.5
            elif layer.N60 < 50:
                return 19.5
            else:
                return 21.0
        if layer.phi is not None:
            if layer.phi < 30:
                return 17.0
            elif layer.phi < 35:
                return 18.5
            elif layer.phi < 40:
                return 19.5
            else:
                return 21.0
        return 18.5  # Generic granular default

    # Unknown soil type — use a middle value
    if layer.N60 is not None:
        if layer.N60 < 10:
            return 17.0
        elif layer.N60 < 30:
            return 18.5
        else:
            return 20.0

    return None  # Cannot estimate


def _estimate_Es(layer: SoilLayer) -> Optional[float]:
    """Estimate elastic modulus from N60 or cu.

    Correlations:
        Sand: Es ≈ 500*(N60+15) kPa (Webb, 1969) or 600*N60 to 1200*N60
        Clay: Es ≈ 250*cu to 500*cu (soft to stiff)
    """
    if layer.is_cohesive is False and layer.N60 is not None:
        # Moderate correlation for sand
        return 600.0 * (layer.N60 + 6)  # ~Es = 600*(N60+6) kPa
    if layer.is_cohesive is True and layer.cu is not None:
        if layer.cu < 50:
            return 250.0 * layer.cu  # Soft clay
        else:
            return 400.0 * layer.cu  # Stiff clay
    if layer.N60 is not None:
        return 500.0 * (layer.N60 + 15)  # Generic
    return None


def _estimate_eps50(cu: float) -> float:
    """Estimate eps50 (strain at 50% strength) from undrained shear strength.

    Based on Matlock (1970) and typical laboratory data:
        cu < 25 kPa  (very soft):  eps50 ~ 0.020
        cu 25-50 kPa (soft):       eps50 ~ 0.010
        cu 50-100 kPa (medium):    eps50 ~ 0.007
        cu 100-200 kPa (stiff):    eps50 ~ 0.005
        cu > 200 kPa (very stiff): eps50 ~ 0.004
    """
    if cu < 25:
        return 0.020
    elif cu < 50:
        return 0.010
    elif cu < 100:
        return 0.007
    elif cu < 200:
        return 0.005
    else:
        return 0.004


def _estimate_k_sand(phi: float, below_gwt: bool) -> float:
    """Estimate initial modulus of subgrade reaction k for sand (kN/m3).

    Based on API RP2A Table 6.8-1 (Reese, 1974):
        phi=25-28: k ~ 5400 (submerged) / 6800 (dry)
        phi=29-30: k ~ 11000 / 16300
        phi=31-32: k ~ 22000 / 24400
        phi=33-36: k ~ 33200 / 35300
        phi=37-40: k ~ 50000 / 56800
    """
    if below_gwt:
        if phi < 29:
            return 5400.0
        elif phi < 31:
            return 11000.0
        elif phi < 33:
            return 22000.0
        elif phi < 37:
            return 33200.0
        else:
            return 50000.0
    else:
        if phi < 29:
            return 6800.0
        elif phi < 31:
            return 16300.0
        elif phi < 33:
            return 24400.0
        elif phi < 37:
            return 35300.0
        else:
            return 56800.0


def _estimate_k_stiff_clay(cu: float) -> float:
    """Estimate initial modulus of subgrade reaction for stiff clay (kN/m3).

    Based on Reese et al. (1975) recommendations:
        cu 100-200 kPa: ks ~ 135,000 kN/m3
        cu 200-400 kPa: ks ~ 270,000 kN/m3
        cu > 400 kPa:   ks ~ 540,000 kN/m3
    """
    if cu < 200:
        return 135_000.0
    elif cu < 400:
        return 270_000.0
    else:
        return 540_000.0
