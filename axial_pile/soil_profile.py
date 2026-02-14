"""
Soil profile definition for axial pile capacity analysis.

Layered soil with per-layer properties for Nordlund, Tomlinson,
and Beta methods.

All units are SI: kPa, kN/m³, degrees, meters.

References:
    FHWA GEC-12, Chapters 7-8 (FHWA-NHI-16-009)
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from geotech_common.water import GAMMA_W


@dataclass
class AxialSoilLayer:
    """Single soil layer for axial pile capacity analysis.

    Parameters
    ----------
    thickness : float
        Layer thickness (m).
    soil_type : str
        "cohesionless" (sand/gravel) or "cohesive" (clay/silt).
        Determines which method is used for that layer.
    unit_weight : float
        Total unit weight (kN/m³).
    friction_angle : float, optional
        Drained friction angle phi (degrees). Required for cohesionless.
    cohesion : float, optional
        Undrained shear strength cu (kPa). Required for cohesive.
    delta_phi_ratio : float, optional
        Ratio of pile-soil friction angle to soil friction angle (delta/phi).
        Default: 0.8 for steel, 1.0 for concrete/timber.
        Can be overridden per layer.
    description : str, optional
        Layer description.
    """
    thickness: float
    soil_type: str
    unit_weight: float
    friction_angle: float = 0.0
    cohesion: float = 0.0
    delta_phi_ratio: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        self.soil_type = self.soil_type.lower()
        if self.soil_type not in ("cohesionless", "cohesive"):
            raise ValueError(
                f"soil_type must be 'cohesionless' or 'cohesive', got '{self.soil_type}'"
            )
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.unit_weight <= 0:
            raise ValueError(f"Unit weight must be positive, got {self.unit_weight}")

        if self.soil_type == "cohesionless":
            if self.friction_angle <= 0:
                raise ValueError("Cohesionless soil must have friction_angle > 0")
            if self.friction_angle > 50:
                raise ValueError(f"Friction angle must be <= 50 degrees, got {self.friction_angle}")
        else:
            if self.cohesion <= 0:
                raise ValueError("Cohesive soil must have cohesion (cu) > 0")


@dataclass
class AxialSoilProfile:
    """Layered soil profile for axial pile capacity.

    Parameters
    ----------
    layers : list of AxialSoilLayer
        Soil layers from top to bottom.
    gwt_depth : float, optional
        Depth to groundwater table from ground surface (m).
        If None, no groundwater effect.
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default 9.81.
    """
    layers: List[AxialSoilLayer] = field(default_factory=list)
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

            if self.gwt_depth is None or layer_top >= self.gwt_depth:
                # Below GWT or no GWT: use effective weight if below gwt
                if self.gwt_depth is not None and layer_top >= self.gwt_depth:
                    gamma_eff = layer.unit_weight - self.gamma_w
                else:
                    gamma_eff = layer.unit_weight
                sigma_v += gamma_eff * z_in_layer
            elif layer_bottom <= self.gwt_depth:
                # Entirely above GWT
                sigma_v += layer.unit_weight * z_in_layer
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

    def layer_at_depth(self, z: float) -> Optional[AxialSoilLayer]:
        """Get the soil layer at a given depth.

        Parameters
        ----------
        z : float
            Depth below ground surface (m).

        Returns
        -------
        AxialSoilLayer or None
        """
        depth = 0.0
        for layer in self.layers:
            if depth <= z < depth + layer.thickness:
                return layer
            depth += layer.thickness
        return self.layers[-1] if z >= depth else None
