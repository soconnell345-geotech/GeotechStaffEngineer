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
from geotech_common.soil_properties import (
    spt_overburden_correction, spt_to_cu, spt_to_phi,
)
from geotech_common.validation import (
    check_positive, check_non_negative, check_range, check_unit_weight,
)

#: The undrained shear strength (kPa) from which ``to_lateral_pile_input``
#: gives a clay a stiff clay p-y criterion: this module's own value
#: (GM-217; neither GEC 9 nor COM624P prints a boundary).
STIFF_CLAY_CU = 100.0


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
    note : str, optional
        How the depth was chosen when the source gave none (a stated
        assumption, not a measurement).
    """

    depth: float
    is_artesian: bool = False
    artesian_head: Optional[float] = None
    note: str = ""

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

    def _n1_60_at_mid(self, layer: SoilLayer, N60: float):
        """(N1,60, sigma'v in kPa) at the layer's mid-depth, C_N by
        spt_overburden_correction's default (Liao & Whitman); (None, None)
        when the effective stress there cannot be worked out (a unit
        weight missing) or is not positive."""
        try:
            sv = self.effective_stress_at_depth(layer.mid_depth)
        except ValueError:
            return None, None
        if sv <= 0:
            return None, None
        return spt_overburden_correction(N60, sv)["N1"], sv

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
                expected_cu = spt_to_cu(layer.N_spt)
                if layer.cu > 0 and abs(layer.cu - expected_cu) / layer.cu > 1.0:
                    warnings_list.append(
                        f"INFO: {prefix}: cu={layer.cu} kPa vs N_spt={layer.N_spt} "
                        f"(Terzaghi-Peck would give ~{expected_cu:.0f} kPa) — "
                        "verify with lab data"
                    )

            if layer.N_spt is not None and layer.phi is not None:
                # Peck et al. (1974) is drawn on N1,60, so the check needs
                # the effective stress at mid-depth (skipped without it).
                N60 = layer.N60 if layer.N60 is not None else layer.N_spt
                N1_60, _ = self._n1_60_at_mid(layer, N60)
                expected_phi = None if N1_60 is None else spt_to_phi(N1_60)
                if expected_phi is not None and abs(layer.phi - expected_phi) > 8:
                    warnings_list.append(
                        f"INFO: {prefix}: phi={layer.phi}deg vs N_spt={layer.N_spt} "
                        f"(Peck et al. 1974 would give ~{expected_phi:.0f}deg "
                        f"at N1,60 = {N1_60:.0f}) — verify"
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

        # --- phi from N1,60 (granular soils, or unknown type with no cu),
        # once every unit weight is in: Peck et al. (1974) is drawn on
        # N1,60 (UFC 3-220-10 Fig 8-4), so N60 is corrected for the
        # effective stress at mid-depth first (Liao & Whitman, the
        # spt_overburden_correction default) ---
        for i, layer in enumerate(self.layers):
            if (layer.phi is not None or layer.N60 is None or layer.is_rock
                    or layer.is_cohesive is True or layer.cu is not None):
                continue
            prefix = f"Layer {i} '{layer.description}'"
            N1_60, sv = self._n1_60_at_mid(layer, layer.N60)
            if N1_60 is None:
                log.append(
                    f"{prefix}: phi not estimated: N1,60 needs the effective "
                    "stress at mid-depth, and a layer down to it has no unit "
                    "weight")
                continue
            phi_est = spt_to_phi(N1_60)
            layer.phi = phi_est
            corr = (f"Peck et al. (1974) from N1,60 = {N1_60:.1f} (N60 = "
                    f"{layer.N60:g}, C_N by Liao & Whitman at sigma'v = "
                    f"{sv:.0f} kPa)")
            layer.mark_estimated("phi", corr)
            log.append(f"{prefix}: estimated phi={phi_est:.1f}deg ({corr})")

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
                              n_cycles: Optional[float] = None,
                              free_water_at_surface: Optional[bool] = None,
                              stiff_clay_cu: float = STIFF_CLAY_CU,
                              ) -> Dict[str, Any]:
        """Convert profile to lateral_pile module input format.

        Selects a p-y criterion for each layer:
        - Rock: WeakRock
        - Clay with cu below ``stiff_clay_cu`` (100 kPa): SoftClayMatlock
        - Stiff clay (cu ``stiff_clay_cu`` and up) with free water: StiffClayBelowWT,
          Reese et al. (1975), "stiff clay with free water" (GEC 9 App. A.2)
        - Other stiff clay: StiffClayAboveWT, Welch & Reese (1972), "stiff
          clay with no free water" (GEC 9 App. A.3), above the water table
          and below it where free water is not at the ground surface
        - Sand: SandAPI

        GEC 9 p. 219 limits Reese et al. (1975) to "situations where stiff
        clay is exposed to free water at or near the ground surface": "In
        conditions where the groundwater surface is at depth and free water
        is not present at or near the ground surface, the Welch and Reese
        criterion is more appropriate, even below groundwater. Similarly,
        stiff clay strata at depth below a sand stratum would normally not
        be subject to degradation due to free water (unless scour removed
        the overlying sand)." Before GM-215 every stiff clay below the water
        table took Reese et al. (1975).

        Parameters
        ----------
        pile_length : float
            Pile embedded length (m).
        pile_diameter : float
            Pile diameter (m).
        loading : str
            "static" or "cyclic". Default "static".
        n_cycles : float, optional
            N, the number of times the design lateral load will be applied
            (COM624P p. 338; GEC 9 p. 226), at least 1. Required with
            loading="cyclic" when a stiff clay layer takes Welch & Reese
            (the references give no default), refused with "static".
        free_water_at_surface : bool, optional
            Whether the stiff clay is exposed to free water at or near the
            ground surface (GEC 9 p. 219). None (default) reads it from the
            profile: yes with the groundwater at the ground surface (depth
            0), except for stiff clay below a sand stratum; no with the
            groundwater at depth. "Near" the surface is the engineer's
            call: True gives every stiff clay layer below the water table
            Reese et al. (1975), False gives every one Welch & Reese. Given
            with no stiff clay below the water table, the choices say it
            was not used (GM-217).
        stiff_clay_cu : float, optional
            The undrained shear strength (kPa) from which a clay layer
            takes a stiff clay criterion; below it, Matlock's (1970) soft
            clay. Default 100, this module's own value (GM-217): GEC 9
            prints soft clays as 250-500 psf (12.0-23.9 kPa, p. 216) and
            stiff as 1,000-2,000 psf (47.9-95.8 kPa, p. 219), and neither
            it nor COM624P prints a boundary.

        Returns
        -------
        dict
            "layers": list of dicts with top, bottom, py_model_type,
            py_model_params, description; the caller instantiates the p-y
            model objects from the returned type+params. "choices": each
            clay layer's criterion, the eps50 and k estimated for it, the
            soft/stiff clay boundary and N (``geotech_common.choices``).
        """
        if not stiff_clay_cu > 0:
            raise ValueError(
                f"stiff_clay_cu must be a strength above 0 kPa, got "
                f"{stiff_clay_cu}")
        if n_cycles is not None and loading != "cyclic":
            raise ValueError(
                f"n_cycles ({n_cycles:g}) applies to loading='cyclic' "
                "only; the static curve takes no number of cycles")
        if n_cycles is not None and n_cycles < 1:
            raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
        gwt = self.groundwater.depth
        lateral_layers, choices = [], []
        sand_above = False
        welch_reese_cyclic = []
        clays = []                # (span, cu) of each clay layer (GM-217)
        free_water_used = False   # a stiff clay below the water table

        for layer in self.layers:
            z_top = layer.top_depth
            z_bot = min(layer.bottom_depth, pile_length)
            if z_top >= pile_length:
                break
            if z_bot <= z_top:
                continue

            gamma_eff = self._get_effective_gamma(layer, z_top, z_bot, gwt)
            model_info = self._select_py_model(
                layer, gamma_eff, gwt, z_top, z_bot, loading,
                free_water_at_surface, sand_above, stiff_clay_cu)
            if not layer.is_rock and layer.is_cohesive is True \
                    and layer.cu is not None:
                clays.append((f"{z_top:g}-{z_bot:g} m", layer.cu))
                free_water_used = free_water_used or (
                    layer.cu >= stiff_clay_cu and (z_top + z_bot) / 2.0 >= gwt)
            if (model_info["type"] == "StiffClayAboveWT"
                    and loading == "cyclic"):
                if n_cycles is None:
                    raise ValueError(
                        f"layer '{layer.description}' {z_top:g}-{z_bot:g} m "
                        "takes Welch & Reese (StiffClayAboveWT): "
                        + _N_CYCLES_NEEDED + " (give n_cycles)")
                model_info["params"]["n_cycles"] = n_cycles
                welch_reese_cyclic.append(f"{z_top:g}-{z_bot:g} m")
            sand_above = sand_above or model_info["type"] == "SandAPI"
            choices.extend(model_info["choices"])

            lateral_layers.append({
                "top": z_top,
                "bottom": z_bot,
                "py_model_type": model_info["type"],
                "py_model_params": model_info["params"],
                "description": layer.description,
            })

        if clays:
            choices.append(_stiff_clay_cu_choice(stiff_clay_cu, clays))
        if free_water_at_surface is not None and not free_water_used:
            choices.append(_free_water_not_used_choice(free_water_at_surface))
        if n_cycles is not None:
            choices.append(_n_cycles_choice(n_cycles, welch_reese_cyclic))
        return {"layers": lateral_layers, "choices": choices}

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
                         loading: str,
                         free_water: Optional[bool] = None,
                         sand_above: bool = False,
                         stiff_clay_cu: float = STIFF_CLAY_CU,
                         ) -> Dict[str, Any]:
        """Select a p-y model and its parameters for a layer, with what was
        chosen for it ("choices"): a clay layer's criterion and the eps50
        and k estimated for it, a sand layer's k. A clay is stiff from
        ``stiff_clay_cu``."""
        from geotech_common.choices import MODULE, choice
        mid = (z_top + z_bot) / 2.0
        below_gwt = mid >= gwt
        span = f"{z_top:g}-{z_bot:g} m"

        if layer.is_rock:
            # Weak rock, Reese (1997) as printed in FHWA GEC 9 (2018) App.
            # A.5, p. 231 (GM-161): the criterion needs the uniaxial
            # compressive strength of the intact rock and the deformation
            # modulus of the rock mass E_m, "which could be obtained from a
            # pressuremeter or dilatometer test". Neither has a printed
            # default, so both are required (before 2026-10-20 qu took 1000
            # kPa and E_r 200 qu, with no source).
            missing = [n for n, v in (("qu", layer.qu), ("Es", layer.Es))
                       if v is None]
            if missing:
                raise ValueError(
                    f"rock layer '{layer.description}' {span}: give "
                    + " and ".join(
                        {"qu": "qu (uniaxial compressive strength of the "
                               "intact rock, kPa)",
                         "Es": "Es (the deformation modulus of the rock "
                               "mass E_m, kPa)"}[n] for n in missing)
                    + " for the weak-rock p-y criterion (Reese 1997, FHWA "
                    "GEC 9 App. A.5, p. 231: E_m from a pressuremeter or "
                    "dilatometer test). Neither manual prints a default; "
                    "FHWA GEC 10 (2018) p. 9-31 reports load tests on "
                    "shafts in decomposed quartzite (RQD 0) modelled with "
                    "qu 180 psi, E_ir 27,000 psi (150 qu) and krm 0.0004, "
                    "which 'might represent a reasonable lower bound' "
                    "where hard but decomposed rock is known to be present "
                    "and the investigation gives little strength "
                    "information.")
            qu = float(layer.qu)
            Er = float(layer.Es)
            gamma_r = layer.gamma if layer.gamma is not None else 22.0
            rock_choices = [choice(
                f"rock mass modulus E_m {span}", Er, "given",
                "the layer's Es, taken as E_m of Reese (1997), FHWA GEC 9 "
                "App. A.5 Eq. A-35, p. 231")]
            if layer.RQD is not None:
                rqd = float(layer.RQD)
            else:
                rqd = 100.0
                rock_choices.append(choice(
                    f"RQD {span} (%)", rqd, MODULE,
                    "not given; FHWA GEC 9 App. A.5, p. 231: alpha_r = "
                    "0.33 at RQD 100 % rising linearly to 1.0 at RQD 0, so "
                    "RQD 100 takes the smallest printed alpha_r (the lower "
                    "ultimate resistance), the same default as the "
                    "lateral_pile tool",
                    ["the RQD from the core logs"]))
            return {
                "type": "WeakRock",
                "params": {"qu": qu, "Er": Er, "gamma_r": gamma_r,
                           "RQD": rqd, "loading": loading},
                "choices": rock_choices,
            }

        if layer.is_cohesive is True and layer.cu is not None:
            choices = []
            stiff = layer.cu >= stiff_clay_cu
            if layer.eps50 is not None:
                eps50 = layer.eps50
            else:
                eps50 = _estimate_eps50(layer.cu)
            if layer.eps50 is None or layer.is_estimated("eps50"):
                choices.append(choice(
                    f"eps50 {span}", eps50, MODULE,
                    f"estimated from cu {layer.cu:g} kPa: "
                    + _eps50_basis(layer.cu, stiff=stiff),
                    ["eps50 from the layer's own stress-strain curves"]))

            if not stiff:
                # Soft/medium clay
                choices.insert(0, choice(
                    f"p-y criterion {span}", "Matlock (1970) soft clay "
                    "(SoftClayMatlock)", MODULE,
                    f"cu {layer.cu:g} kPa, below stiff_clay_cu "
                    f"{stiff_clay_cu:g} kPa (see the soft/stiff clay "
                    "boundary)",
                    ["Welch & Reese (1972), stiff clay with no free water "
                     "(StiffClayAboveWT)", "Reese et al. (1975), stiff "
                     "clay with free water (StiffClayBelowWT)"]))
                return {
                    "type": "SoftClayMatlock",
                    "params": {"c": layer.cu, "gamma": gamma_eff,
                               "eps50": eps50, "J": 0.5,
                               "loading": loading},
                    "choices": choices,
                }
            # Stiff clay: with free water, Reese et al. (1975); otherwise
            # Welch & Reese (1972), below the water table too (GEC 9 p. 219,
            # GM-215).
            with_water, how, why = _stiff_clay_free_water(
                below_gwt, gwt, free_water, sand_above)
            if not with_water and layer.k_py is not None:
                why += (f" The layer's k_py ({layer.k_py:g} kN/m3) is not "
                        "used: Welch & Reese has no initial straight line.")
            welch = "Welch & Reese (1972), stiff clay with no free water " \
                    "(StiffClayAboveWT)"
            reese = "Reese et al. (1975), stiff clay with free water " \
                    "(StiffClayBelowWT)"
            choices.insert(0, choice(
                f"p-y criterion {span}", reese if with_water else welch,
                how, f"cu {layer.cu:g} kPa, stiff clay from stiff_clay_cu "
                f"{stiff_clay_cu:g} kPa; " + why,
                [welch, reese] if below_gwt else []))
            if with_water:
                # The initial line p = (k x) y takes ks for static loading
                # and kc for cyclic (COM624P Table 3.2; GEC 9 Table A-1),
                # each refused with the other loading (GM-208); a given
                # k_py is the k for the loading asked for.
                k_name = "kc" if loading == "cyclic" else "ks"
                if layer.k_py is not None:
                    k_py = layer.k_py
                else:
                    k_py = _estimate_k_stiff_clay(layer.cu, loading)
                    choices.append(choice(
                        f"k of the initial line, {k_name}, {span}",
                        f"{k_py:g} kN/m3", MODULE,
                        f"COM624P Table 3.2 (GEC 9 Table A-1) for cu "
                        f"{layer.cu:g} kPa: {k_name} for {loading} "
                        "loading by the average undrained shear strength "
                        "to five pile diameters, taken as the layer's cu",
                        ["k_py given on the layer"]))
                return {
                    "type": "StiffClayBelowWT",
                    "params": {"c": layer.cu, "gamma": gamma_eff,
                               "eps50": eps50, k_name: k_py,
                               "loading": loading},
                    "choices": choices,
                }
            return {
                "type": "StiffClayAboveWT",
                "params": {"c": layer.cu, "gamma": gamma_eff,
                           "eps50": eps50,
                           "loading": loading},
                "choices": choices,
            }

        # Granular soil
        if layer.phi is not None and layer.phi > 0:
            sand_choices = []
            if layer.k_py is not None:
                k = layer.k_py
            else:
                k = _estimate_k_sand(layer.phi, below_gwt)
                where = "below" if below_gwt else "above"
                sand_choices.append(choice(
                    f"k of the API sand curve {span}", round(k), MODULE,
                    f"API RP 2A-WSD Fig. 6.8.7-1 (p. 70) at phi' "
                    f"{layer.phi:g} deg, sand {where} the water table"
                    + ("; phi' is off the chart's curves, so their end value"
                       if not 28.83 <= layer.phi <= 40.0 else ""),
                    ["k_py from a lateral load test or the engineer's own "
                     "reading", "GEC 9 Table A-3 by relative density for "
                     "Reese sand: loose / medium / dense 5,429 / 16,287 / "
                     "33,931 kN/m3 submerged, 6,786 / 24,430 / 61,076 above "
                     "the water"]))
            return {
                "type": "SandAPI",
                "params": {"phi": layer.phi, "gamma": gamma_eff,
                           "k": k, "loading": loading},
                "choices": sand_choices,
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

    # ── Adapter: SOE (Support of Excavation) ────────────────────────

    def to_soe_input(self, excavation_depth: float,
                     support_depths: Optional[List[float]] = None,
                     support_type: str = "strut",
                     support_spacing: float = 3.0,
                     surcharge: float = 10.0) -> Dict[str, Any]:
        """Convert profile to soe module input format.

        Builds SOEWallLayer dicts for all layers and optional SupportLevel
        dicts for a multi-level braced excavation.

        Parameters
        ----------
        excavation_depth : float
            Excavation depth (m).
        support_depths : list of float, optional
            Depths (m) of support levels. None = cantilever (no supports).
        support_type : str
            Default support type ("strut", "anchor", "raker").
        support_spacing : float
            Horizontal spacing of supports (m).
        surcharge : float
            Surface surcharge (kPa), default 10.

        Returns
        -------
        dict
            Keys: "excavation_depth", "layers" (list of SOEWallLayer dicts),
            "supports" (list of SupportLevel dicts), "surcharge",
            "gwt_depth".
        """
        soe_layers = []
        for layer in self.layers:
            gamma = layer.gamma if layer.gamma is not None else 18.0
            phi = layer.phi if layer.phi is not None else 0.0
            c = 0.0
            soil_type = "sand"

            if layer.is_cohesive is True and layer.cu is not None:
                c = layer.cu
                if phi == 0:
                    phi = 0.0  # Total stress analysis
                # Classify clay type based on cu
                if c < 25.0:
                    soil_type = "soft_clay"
                else:
                    soil_type = "stiff_clay"
            elif layer.c_prime is not None:
                c = layer.c_prime

            # Need at least phi or c > 0
            if phi == 0 and c == 0:
                raise ValueError(
                    f"Layer '{layer.description}' has neither phi nor cu/c' — "
                    "cannot create SOE input. "
                    "Run fill_missing_from_correlations() first."
                )

            soe_layers.append({
                "thickness": layer.thickness,
                "unit_weight": gamma,
                "friction_angle": phi,
                "cohesion": c,
                "soil_type": soil_type,
                "description": layer.description,
            })

        supports = []
        if support_depths:
            for depth in sorted(support_depths):
                supports.append({
                    "depth": depth,
                    "support_type": support_type,
                    "spacing": support_spacing,
                })

        return {
            "excavation_depth": excavation_depth,
            "layers": soe_layers,
            "supports": supports,
            "surcharge": surcharge,
            "gwt_depth": self.gwt_depth,
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
            The wall's total height H (m), as the retaining_walls module
            takes it: from the underside of the base slab to the top of the
            stem (H = base thickness + stem height; GM-100). It is used here
            as the depth below the ground surface behind the wall: the soil
            above it is averaged as the backfill, the layer at it is the
            foundation soil. A wall retaining 4.0 m on a 0.5 m footing is
            4.5.
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

    # ── Adapter: Slope Stability ──────────────────────────────────────

    def to_slope_stability_input(
        self,
        ground_elevation: float = 0.0,
    ) -> Dict[str, Any]:
        """Convert profile to slope_stability module input format.

        Maps SoilLayer depth-based data to elevation-based SlopeSoilLayer dicts.
        Depth-to-elevation conversion: elevation = ground_elevation - depth.

        For cohesive layers with cu > 0: analysis_mode = "undrained" (phi=0).
        For granular layers with phi > 0: analysis_mode = "drained" (cu=0).

        Parameters
        ----------
        ground_elevation : float
            Elevation of ground surface (m). Default 0.0.

        Returns
        -------
        dict
            Keys:
            - "soil_layers" : list of dicts suitable for SlopeSoilLayer(**d)
            - "gwt_elevation" : float or None
        """
        slope_layers = []

        for layer in self.layers:
            top_elev = ground_elevation - layer.top_depth
            bot_elev = ground_elevation - layer.bottom_depth
            gamma = layer.gamma if layer.gamma is not None else 18.0
            gamma_sat = layer.gamma_sat if layer.gamma_sat is not None else gamma

            if layer.is_cohesive is True and layer.cu is not None and layer.cu > 0:
                slope_layers.append({
                    "name": layer.description,
                    "top_elevation": top_elev,
                    "bottom_elevation": bot_elev,
                    "gamma": gamma,
                    "gamma_sat": gamma_sat,
                    "phi": 0.0,
                    "c_prime": 0.0,
                    "cu": layer.cu,
                    "analysis_mode": "undrained",
                })
            elif layer.phi is not None and layer.phi > 0:
                c_prime = layer.c_prime if layer.c_prime is not None else 0.0
                slope_layers.append({
                    "name": layer.description,
                    "top_elevation": top_elev,
                    "bottom_elevation": bot_elev,
                    "gamma": gamma,
                    "gamma_sat": gamma_sat,
                    "phi": layer.phi,
                    "c_prime": c_prime,
                    "cu": 0.0,
                    "analysis_mode": "drained",
                })
            else:
                raise ValueError(
                    f"Layer '{layer.description}' has neither cu nor phi — "
                    "cannot create slope stability input"
                )

        gwt_elevation = None
        if self.groundwater.depth < self.total_depth:
            gwt_elevation = ground_elevation - self.groundwater.depth

        return {
            "soil_layers": slope_layers,
            "gwt_elevation": gwt_elevation,
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
    """Estimate eps50 (the strain at half the maximum principal-stress
    difference) from the undrained shear strength.

    From 0.5 ton/ft^2 up, COM624P Table 3.3 (FHWA-SA-91-048 p. 331; GEC 9
    Table A-2), stiff clays, by the average undrained shear strength:

        0.5-1 ton/ft^2 (47.9-95.8 kPa):    0.007
        1-2 ton/ft^2 (95.8-191.5 kPa):     0.005
        2-4 ton/ft^2 (191.5-383 kPa):      0.004

    stronger clay taking the last value. Below it, COM624P Table 3.1
    (p. 324, Matlock 1970) by consistency: soft 0.020, medium 0.010. Table
    3.1 prints no strengths; the 25 kPa between soft and medium is this
    estimate's. Before GM-215 the Table 3.3 edges were 50, 100 and 200 kPa.
    """
    if cu < 25:
        return 0.020
    elif cu < 47.9:
        return 0.010
    elif cu < 95.8:
        return 0.007
    elif cu < 191.5:
        return 0.005
    else:
        return 0.004


def _eps50_basis(cu: float, stiff: bool = True) -> str:
    """Where :func:`_estimate_eps50`'s value for ``cu`` comes from, for a
    choice's basis; ``stiff`` is whether the layer takes a stiff clay
    criterion."""
    if cu < 47.9:
        return ("COM624P Table 3.1 (Matlock 1970) by consistency: soft "
                "0.020, medium 0.010, stiff 0.005; the table prints no "
                "strengths, and 25 kPa between soft and medium is this "
                "estimate's")
    text = ("COM624P Table 3.3 (GEC 9 Table A-2), stiff clays, by the "
            "average undrained shear strength: 0.007, 0.005 and 0.004 for "
            "0.5-1, 1-2 and 2-4 ton/ft^2 (47.9-95.8, 95.8-191.5 and "
            "191.5-383 kPa)")
    if cu >= 383.0:
        text += "; the table stops at 383 kPa"
    if not stiff:
        text += ("; the layer takes Matlock's soft clay criterion, whose "
                 "Table 3.1 gives eps50 by consistency only (stiff 0.005)")
    return text


#: GEC 9 p. 219 on where Reese et al. (1975) applies (GM-215).
_GEC9_FREE_WATER = (
    "GEC 9 p. 219: the reduction of Reese et al. (1975) \"is only "
    "appropriate for situations where stiff clay is exposed to free water "
    "at or near the ground surface\"; \"In conditions where the "
    "groundwater surface is at depth and free water is not present at or "
    "near the ground surface, the Welch and Reese criterion is more "
    "appropriate, even below groundwater.\"")


def _stiff_clay_free_water(below_gwt: bool, gwt: float,
                           free_water: Optional[bool],
                           sand_above: bool):
    """Whether a stiff clay layer takes the criterion for stiff clay with
    free water, Reese et al. (1975), or the one with no free water, Welch
    & Reese (1972), by GEC 9 p. 219 (GM-215): ``(with_water, how, basis)``.
    ``free_water`` is the caller's ``free_water_at_surface`` (None: read
    from the profile); ``sand_above`` whether a sand stratum lies above."""
    from geotech_common.choices import MODULE, how_given
    if not below_gwt:
        return False, MODULE, (
            "above the water table: Welch & Reese (COM624P pp. 336-340; "
            "GEC 9 App. A.3)")
    if free_water is not None:
        return bool(free_water), how_given(
            free_water, None, name="free_water_at_surface"), (
            f"free_water_at_surface={bool(free_water)} given; "
            + _GEC9_FREE_WATER)
    if gwt > 0:
        return False, MODULE, (
            f"the groundwater is at {gwt:g} m, taken as at depth: "
            + _GEC9_FREE_WATER + " Whether free water can stand at or "
            "near the ground surface is the engineer's call "
            "(free_water_at_surface=True gives Reese et al. 1975).")
    if sand_above:
        return False, MODULE, (
            "the groundwater is at the ground surface, but a sand stratum "
            "lies above this clay: GEC 9 p. 219, \"stiff clay strata at "
            "depth below a sand stratum would normally not be subject to "
            "degradation due to free water (unless scour removed the "
            "overlying sand)\". Where scour can remove the sand, "
            "free_water_at_surface=True gives Reese et al. (1975).")
    return True, MODULE, (
        "the groundwater is at the ground surface: " + _GEC9_FREE_WATER
        + " (free_water_at_surface=False gives Welch & Reese.)")


#: What a cyclic stiff clay layer with no free water needs (GM-202); the
#: same words as ``lateral_pile.py_curves.StiffClayAboveWT`` refuses with.
_N_CYCLES_NEEDED = (
    "cyclic loading of stiff clay with no free water (Welch & Reese, "
    "StiffClayAboveWT) needs n_cycles, the number of times the design "
    "lateral load will be applied: the curve softens with log N (y_c = "
    "y_s + y50 C log N, COM624P Eq. 3.29, GEC 9 Eq. A-21) and the "
    "references give no default")


#: Where GEC 9 prints the clays the soft and stiff clay criteria were
#: developed on (GM-217).
_GEC9_CLAY_RANGES = (
    "GEC 9 p. 216: soft clays (Matlock 1970) \"are those with undrained "
    "shear strength (Su) ranging from 250 to 500 psf\" (12.0-23.9 kPa); "
    "p. 219: stiff clays (Reese et al. 1975) \"are those with undrained "
    "shear strength (Su) ranging from 1,000 to 2,000 psf\" (47.9-95.8 "
    "kPa). Neither page prints a boundary between them, nor does COM624P, "
    "whose stiff clay Tables 3.2 and 3.3 start at 0.5 ton/ft^2 (47.9 kPa)")


def _stiff_clay_cu_choice(stiff_clay_cu: float, clays: List[tuple]) -> dict:
    """The undrained shear strength from which a clay takes a stiff clay
    criterion (GM-217), with what the printed edges would change in this
    profile. ``clays``: (span, cu) of each clay layer."""
    from geotech_common.choices import choice, how_given

    def changes(edge: float) -> str:
        moved = [f"{span} (cu {cu:g} kPa)" for span, cu in clays
                 if (cu >= edge) != (cu >= stiff_clay_cu)]
        if not moved:
            return "no layer here changes"
        side = ("a stiff clay criterion" if edge < stiff_clay_cu
                else "Matlock's soft clay")
        return ", ".join(moved) + " would take " + side

    alternatives = []
    if stiff_clay_cu != 47.9:
        alternatives.append(
            f"stiff_clay_cu 47.9 kPa (1,000 psf), where GEC 9's stiff clays "
            f"and COM624P Tables 3.2 and 3.3 start: {changes(47.9)}")
    if stiff_clay_cu != STIFF_CLAY_CU:
        alternatives.append(
            f"stiff_clay_cu {STIFF_CLAY_CU:g} kPa, this module's default: "
            f"{changes(STIFF_CLAY_CU)}")
    return choice(
        "soft/stiff clay boundary",
        f"stiff clay from cu {stiff_clay_cu:g} kPa (stiff_clay_cu); below "
        "it Matlock (1970) soft clay",
        how_given(stiff_clay_cu, STIFF_CLAY_CU, name="stiff_clay_cu"),
        _GEC9_CLAY_RANGES + f". {STIFF_CLAY_CU:g} kPa is this module's own "
        "value, above GEC 9's stiff range; a clay between the two ranges "
        "is in neither",
        alternatives)


def _free_water_not_used_choice(free_water: bool) -> dict:
    """``free_water_at_surface`` given with no stiff clay below the water
    table, where it picks the criterion (GM-217)."""
    from geotech_common.choices import choice, how_given
    return choice(
        "free water at or near the ground surface",
        f"{bool(free_water)}, not used",
        how_given(free_water, None, name="free_water_at_surface"),
        "free_water_at_surface picks Reese et al. (1975) or Welch & Reese "
        "(1972) for stiff clay below the water table (GEC 9 p. 219); no "
        "layer here is stiff clay below the water table")


def _n_cycles_choice(n_cycles: float, spans: List[str]) -> dict:
    """The number of cycles N given to ``to_lateral_pile_input`` (GM-209),
    and the Welch & Reese layers that take it."""
    from geotech_common.choices import choice, how_given
    what = "number of cycles of load application N"
    how = how_given(n_cycles, None, name="n_cycles")
    if not spans:
        return choice(what, f"{n_cycles:g}, not used", how,
                      "no stiff clay layer takes Welch & Reese; the other "
                      "cyclic curves take no N")
    factor = 1.0 + 0.6 * math.log10(n_cycles)
    return choice(
        what, f"{n_cycles:g} (y50 x {factor:.3g}), {', '.join(spans)}", how,
        "Welch & Reese (1972): y_c = y_s + y50 C log N, C = 9.6 (p/pu)^4 "
        "(COM624P Eqs. 3.28-3.29, p. 338; GEC 9 Eqs. A-20, A-21); the "
        "references give no default N",
        ["N = 1: the static curve, the stiffest",
         "more cycles: a softer curve, more deflection and moment"])


#: API RP 2A-WSD, 21st ed. with Supplement 3 (2007), Fig. 6.8.7-1, printed
#: p. 70: k (lb/in^3) against phi', read off the figure's vector curves at
#: each whole degree (GM-287). The chart's phi' scale is marked 28, 29, 30,
#: 36, 40, 45 on its gridlines; between two marks phi' is taken as linear
#: across the chart. Each curve starts at its first drawn point (phi' 28.83
#: below the water table, 28.88 above) and ends at 40.
_API_SAND_K = {
    True: ([28.83, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
           [8.7, 10.1, 34.1, 42.3, 51.2, 60.7, 70.9, 82.0, 94.1, 106.9,
            121.1, 137.4, 156.6]),
    False: ([28.88, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            [8.7, 10.1, 46.6, 60.8, 76.9, 94.9, 114.7, 136.2, 159.8, 185.0,
             212.6, 243.1, 277.3]),
}


def _estimate_k_sand(phi: float, below_gwt: bool) -> float:
    """k for API sand (kN/m3) from API RP 2A-WSD Fig. 6.8.7-1 (p. 70) by
    phi', below or above the water table, linear between the whole degrees
    read off the chart. Off the curves' ends (phi' below about 28.9 or above
    40) the end value is used: no page prints one there.

    Before GM-287 this was a step table cited as "API RP2A Table 6.8-1",
    which no held page prints (phi' 36 below the water table: 33,200 against
    25,500 here; 40 above it: 56,800 against 75,300).
    """
    phis, ks = _API_SAND_K[bool(below_gwt)]
    if phi <= phis[0]:
        k = ks[0]
    elif phi >= phis[-1]:
        k = ks[-1]
    else:
        i = next(j for j in range(1, len(phis)) if phi <= phis[j])
        t = (phi - phis[i - 1]) / (phis[i] - phis[i - 1])
        k = ks[i - 1] + t * (ks[i] - ks[i - 1])
    return k * 271.447


#: COM624P Table 3.2 (FHWA-SA-91-048 p. 331; GEC 9 Table A-1), Reese,
#: Cox & Koop (1975): representative k for stiff clays by the average
#: undrained shear strength, 0.5-1, 1-2 and 2-4 ton/ft^2 (upper bounds
#: 95.8, 191.5 kPa): ks 500 / 1,000 / 2,000 and kc 200 / 400 / 800
#: lb/in^3, in kN/m^3 (1 lb/in^3 = 271.447 kN/m^3).
_STIFF_CLAY_K = {"static": (135_700.0, 271_400.0, 542_900.0),
                 "cyclic": (54_300.0, 108_600.0, 217_200.0)}


def _estimate_k_stiff_clay(cu: float, loading: str = "static") -> float:
    """k of the initial line p = (k x) y for stiff clay below the water
    table (kN/m3): ks for static loading, kc for cyclic, from COM624P
    Table 3.2 (GEC 9 Table A-1) by the undrained shear strength:

        cu below 95.8 kPa (0.5-1 ton/ft^2):  ks 135,700, kc 54,300
        cu 95.8-191.5 kPa (1-2 ton/ft^2):    ks 271,400, kc 108,600
        cu 191.5 kPa and up (2-4 ton/ft^2):  ks 542,900, kc 217,200

    The table's strength is the average to five pile diameters; the
    layer's cu stands for it here. The table stops at 4 ton/ft^2
    (383 kPa); stronger clay takes its last column. Before GM-208 the
    bands were one column low (cu 100-200 kPa took 135,000) and cyclic
    loading took ks.
    """
    ks_or_kc = _STIFF_CLAY_K["cyclic" if loading == "cyclic" else "static"]
    if cu < 95.8:
        return ks_or_kc[0]
    elif cu < 191.5:
        return ks_or_kc[1]
    else:
        return ks_or_kc[2]
