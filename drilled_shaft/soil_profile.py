"""
Soil profile definition for drilled shaft capacity analysis.

Layered soil with per-layer properties for alpha, beta, and rock
socket methods.

All units are SI: kPa, kN/m³, degrees, meters.

References:
    FHWA GEC-10 (FHWA-NHI-18-024), Sec. 10.3.1 (idealized geomaterial layer
    profiles) and Sec. 10.3.3 (geomaterial properties)
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from geotech_common.water import GAMMA_W
from drilled_shaft.side_resistance import SIGMA_P_CORRELATIONS


@dataclass
class ShaftSoilLayer:
    """Single soil layer for drilled shaft capacity analysis.

    Parameters
    ----------
    thickness : float
        Layer thickness (m).
    soil_type : str
        "cohesive", "cohesionless", or "rock".
    unit_weight : float
        Total unit weight (kN/m³).
    cu : float, optional
        Undrained shear strength (kPa). Required for cohesive.
    phi : float, optional
        Drained friction angle (degrees). Used for cohesionless; given, it
        is used, and (N1)60 is not (GM-244).
    N60 : float, optional
        Energy-corrected SPT blow count. Used for sand end bearing.
    qu : float, optional
        Unconfined compressive strength for rock (kPa).
    RQD : float, optional
        Rock Quality Designation (%). Default 100.
    N1_60 : float, optional
        Overburden-corrected SPT blow count (N1)60. Only used by the GEC-10
        rational beta method (``beta_method="rational"``), for phi' =
        27.5 + 9.2*log10[(N1)60] where no phi is given. Default 0.0
        (unused).
    sigma_v_ref : float, optional
        Reference effective vertical stress (kPa) for the OCR in the rational
        beta method — the historical / pre-scour maximum stress the layer has
        experienced. Default 0.0 -> use the current profile sigma'v (the normal
        no-scour case). Set it higher than the current sigma'v to model a
        scour-reduced design stress (GEC-10 Appendix A: OCR uses the no-scour
        sigma'v = 4,645 psf while fs uses the scoured sigma'v = 2,266 psf).
    OCR : float, optional
        Direct overconsolidation-ratio override for the rational beta method.
        Default 0.0 -> compute OCR = sigma'p/sigma_v_ref from N60. If > 0 it is
        used as given (N60/sigma_v_ref then not needed for beta).
    sigma_p_correlation : str, optional
        How the rational beta method estimates sigma'p from N60 (GEC-10
        p. 10-11): "clean_sand" (default; Eq. 10-11, m = 0.6, clean
        quartzitic sands), "silty_sand" (Eq. 10-11, m = 0.8, silty sands to
        sandy silts) or "gravel" (Eq. 10-12, sigma'p/pa = 0.15 N60).
    description : str, optional
        Layer description.
    GSI : float, optional
        Geological Strength Index of a rock layer's rock mass, 0 to 100
        (GEC-10 p. 10-23; Table 10-1 lists it among the rock's properties).
        With ``m_i``, the Hoek-Brown base of a fractured rock mass (Eqs.
        10-24 to 10-29) when the base is in this layer (GM-251). Rock
        only.
    m_i : float, optional
        Hoek-Brown constant m of the intact rock, by rock type (GEC-10 p.
        10-24, which points to Table 9-10 of GEC 5). Rock only.
    """
    thickness: float
    soil_type: str
    unit_weight: float
    cu: float = 0.0
    phi: float = 0.0
    N60: float = 0.0
    qu: float = 0.0
    RQD: float = 100.0
    N1_60: float = 0.0
    sigma_v_ref: float = 0.0
    OCR: float = 0.0
    sigma_p_correlation: str = "clean_sand"
    description: str = ""
    GSI: Optional[float] = None
    m_i: Optional[float] = None

    def __post_init__(self):
        self.soil_type = self.soil_type.lower()
        if self.soil_type not in ("cohesive", "cohesionless", "rock"):
            raise ValueError(
                f"soil_type must be 'cohesive', 'cohesionless', or 'rock', "
                f"got '{self.soil_type}'"
            )
        if self.sigma_p_correlation not in SIGMA_P_CORRELATIONS:
            raise ValueError(
                "sigma_p_correlation must be one of "
                f"{sorted(SIGMA_P_CORRELATIONS)}, got "
                f"{self.sigma_p_correlation!r}")
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.unit_weight <= 0:
            raise ValueError(f"Unit weight must be positive, got {self.unit_weight}")

        if self.soil_type == "cohesive" and self.cu <= 0:
            raise ValueError("Cohesive soil must have cu > 0")
        if self.soil_type == "cohesionless" and self.phi <= 0 \
                and self.N1_60 <= 0:
            raise ValueError(
                "Cohesionless soil must have friction angle (phi) > 0, or "
                "(N1)60 (N1_60) for phi' = 27.5 + 9.2 log (N1)60 (GEC-10 "
                "Example 10-1)")
        if self.soil_type == "rock" and self.qu <= 0:
            raise ValueError("Rock must have qu (UCS) > 0")
        # GM-251: the Hoek-Brown parameters of a rock mass
        given = [n for n in ("GSI", "m_i") if getattr(self, n) is not None]
        if given and self.soil_type != "rock":
            raise ValueError(
                f"{' and '.join(given)} describe a rock mass (GEC-10 Eqs. "
                "10-24 to 10-29, the Hoek-Brown base): only a rock layer "
                f"takes them, and this layer is {self.soil_type}")
        if self.GSI is not None and not 0 <= self.GSI <= 100:
            raise ValueError(
                f"GSI must be 0 to 100 (GEC-10 p. 10-23), got {self.GSI}")
        if self.m_i is not None and not self.m_i > 0:
            raise ValueError(f"m_i must be positive, got {self.m_i}")


@dataclass
class ShaftSoilProfile:
    """Layered soil profile for drilled shaft capacity.

    Parameters
    ----------
    layers : list of ShaftSoilLayer
        Soil layers from top to bottom.
    gwt_depth : float, optional
        Depth to groundwater table (m). If None, no groundwater.
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default 9.81.
    """
    layers: List[ShaftSoilLayer] = field(default_factory=list)
    gwt_depth: Optional[float] = None
    gamma_w: float = GAMMA_W

    def __post_init__(self):
        if not self.layers:
            raise ValueError("At least one soil layer must be provided")

    @property
    def total_thickness(self) -> float:
        """Total depth of all layers (m)."""
        return sum(layer.thickness for layer in self.layers)

    def effective_stress_at_depth(self, z: float) -> float:
        """Compute effective vertical stress at depth z.

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        float
            Effective vertical stress sigma_v' (kPa).
        """
        sigma_v = 0.0
        depth_accumulated = 0.0

        for layer in self.layers:
            layer_top = depth_accumulated
            layer_bottom = depth_accumulated + layer.thickness

            if z <= layer_top:
                break

            z_in_layer = min(z, layer_bottom) - layer_top

            if self.gwt_depth is None or layer_bottom <= self.gwt_depth:
                # Entirely above GWT or no GWT
                if self.gwt_depth is not None and layer_top >= self.gwt_depth:
                    gamma_eff = layer.unit_weight - self.gamma_w
                else:
                    gamma_eff = layer.unit_weight
                sigma_v += gamma_eff * z_in_layer
            elif layer_top >= self.gwt_depth:
                # Entirely below GWT
                sigma_v += (layer.unit_weight - self.gamma_w) * z_in_layer
            else:
                # GWT passes through this layer
                above_gwt = self.gwt_depth - layer_top
                below_gwt = min(z, layer_bottom) - self.gwt_depth
                if above_gwt > 0:
                    sigma_v += layer.unit_weight * min(above_gwt, z_in_layer)
                if below_gwt > 0:
                    sigma_v += (layer.unit_weight - self.gamma_w) * below_gwt

            depth_accumulated = layer_bottom

        return sigma_v

    def average_effective_stress(self, z1: float, z2: float) -> float:
        """Average effective vertical stress (kPa) over depths z1 to z2 (m).

        GEC-10 (2018) Eq. 10-5 takes sigma'v as the 'average vertical
        effective stress over the depth interval Delta z'. The stress is
        linear between layer boundaries and the water table, so the
        trapezoidal rule over those points is exact; with one unit weight
        over the interval it is the stress at mid-depth.
        """
        if z2 <= z1:
            return self.effective_stress_at_depth(z1)
        points = {z1, z2}
        depth = 0.0
        for layer in self.layers:
            depth += layer.thickness
            if z1 < depth < z2:
                points.add(depth)
        if self.gwt_depth is not None and z1 < self.gwt_depth < z2:
            points.add(self.gwt_depth)
        z = sorted(points)
        s = [self.effective_stress_at_depth(d) for d in z]
        area = sum((z[i + 1] - z[i]) * (s[i] + s[i + 1]) / 2
                   for i in range(len(z) - 1))
        return area / (z2 - z1)

    def layer_at_depth(self, z: float) -> Optional[ShaftSoilLayer]:
        """Get the soil layer at a given depth.

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        ShaftSoilLayer or None
        """
        depth = 0.0
        for layer in self.layers:
            if depth <= z < depth + layer.thickness:
                return layer
            depth += layer.thickness
        return self.layers[-1] if z >= depth else None
