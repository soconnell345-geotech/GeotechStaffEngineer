"""
Soil profile definition for drilled shaft capacity analysis.

Layered soil with per-layer properties for alpha, beta, and rock
socket methods.

All units are SI: kPa, kN/m³, degrees, meters.

References:
    FHWA GEC-10, Chapters 13-14
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from geotech_common.water import GAMMA_W


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
        Drained friction angle (degrees). Used for cohesionless.
    N60 : float, optional
        Energy-corrected SPT blow count. Used for sand end bearing.
    qu : float, optional
        Unconfined compressive strength for rock (kPa).
    RQD : float, optional
        Rock Quality Designation (%). Default 100.
    description : str, optional
        Layer description.
    """
    thickness: float
    soil_type: str
    unit_weight: float
    cu: float = 0.0
    phi: float = 0.0
    N60: float = 0.0
    qu: float = 0.0
    RQD: float = 100.0
    description: str = ""

    def __post_init__(self):
        self.soil_type = self.soil_type.lower()
        if self.soil_type not in ("cohesive", "cohesionless", "rock"):
            raise ValueError(
                f"soil_type must be 'cohesive', 'cohesionless', or 'rock', "
                f"got '{self.soil_type}'"
            )
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.unit_weight <= 0:
            raise ValueError(f"Unit weight must be positive, got {self.unit_weight}")

        if self.soil_type == "cohesive" and self.cu <= 0:
            raise ValueError("Cohesive soil must have cu > 0")
        if self.soil_type == "cohesionless" and self.phi <= 0:
            raise ValueError("Cohesionless soil must have friction angle (phi) > 0")
        if self.soil_type == "rock" and self.qu <= 0:
            raise ValueError("Rock must have qu (UCS) > 0")


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
