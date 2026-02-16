"""
Soil layer and profile definitions for downdrag analysis.

Defines soil layers with both strength parameters (for skin friction)
and consolidation parameters (for settlement), plus a soil profile
that computes effective stress at any depth.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
- AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7.
- UFC 3-220-20, Chapter 6.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class DowndragSoilLayer:
    """A soil layer for downdrag analysis.

    Combines strength parameters (for skin friction computation) with
    consolidation parameters (for settlement computation in settling layers).

    Parameters
    ----------
    thickness : float
        Layer thickness (m).
    soil_type : str
        "cohesionless" or "cohesive".
    unit_weight : float
        Total unit weight above GWT, or saturated unit weight below GWT (kN/m^3).
    phi : float
        Effective friction angle (degrees). Required for cohesionless.
    cu : float
        Undrained shear strength (kPa). Required for cohesive.
    beta : float, optional
        Override beta coefficient for skin friction. If None, computed
        from phi using Fellenius (1991): beta = (1-sin(phi))*tan(phi).
    alpha : float, optional
        Override alpha coefficient for cohesive skin friction.
        If None, defaults to 1.0 (conservative for soft clay).
    Cc : float
        Compression index. Required if settling=True and cohesive.
    Cr : float
        Recompression index. Required if settling=True and cohesive.
    e0 : float
        Initial void ratio. Required if settling=True and cohesive.
    C_ec : float, optional
        Modified compression index = Cc/(1+e0). If provided, overrides Cc/e0.
        Used in UFC Eq 6-53.
    C_er : float, optional
        Modified recompression index = Cr/(1+e0). If provided, overrides Cr/e0.
        Used in UFC Eq 6-53.
    sigma_p : float, optional
        Preconsolidation pressure (kPa). If None, layer is normally consolidated.
    E_s : float, optional
        Young's modulus of soil (kPa). Required for coarse-grained settling layers.
        Used in UFC Eq 6-54.
    nu_s : float
        Poisson's ratio of soil. Default 0.3. Used for coarse-grained settlement
        per UFC Eq 6-54.
    settling : bool
        True if this layer settles due to fill placement or GW drawdown.
    description : str
        Layer description.
    """
    thickness: float
    soil_type: str
    unit_weight: float
    phi: float = 0.0
    cu: float = 0.0
    beta: Optional[float] = None
    alpha: Optional[float] = None
    Cc: float = 0.0
    Cr: float = 0.0
    e0: float = 0.0
    C_ec: Optional[float] = None
    C_er: Optional[float] = None
    sigma_p: Optional[float] = None
    E_s: Optional[float] = None
    nu_s: float = 0.3
    settling: bool = False
    description: str = ""

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(
                f"Layer thickness must be positive, got {self.thickness}"
            )
        valid_types = ("cohesionless", "cohesive")
        if self.soil_type not in valid_types:
            raise ValueError(
                f"soil_type must be one of {valid_types}, got '{self.soil_type}'"
            )
        if self.unit_weight <= 0:
            raise ValueError(
                f"Unit weight must be positive, got {self.unit_weight}"
            )
        if self.soil_type == "cohesionless" and self.phi <= 0:
            raise ValueError(
                f"Cohesionless layer must have phi > 0, got {self.phi}"
            )
        if self.soil_type == "cohesive" and self.cu <= 0:
            raise ValueError(
                f"Cohesive layer must have cu > 0, got {self.cu}"
            )
        if self.settling:
            if self.soil_type == "cohesive":
                # Cohesive: need Cc/e0 or C_ec/C_er
                has_traditional = self.Cc > 0 and self.e0 > 0
                has_modified = (self.C_ec is not None and self.C_ec > 0)
                if not has_traditional and not has_modified:
                    raise ValueError(
                        "Cohesive settling layer must have (Cc > 0 and e0 > 0)"
                        " or C_ec > 0"
                    )
                if self.Cr < 0:
                    raise ValueError(
                        f"Cr must be non-negative, got {self.Cr}"
                    )
                # Derive modified indices from traditional if not provided
                if self.C_ec is None and has_traditional:
                    self.C_ec = self.Cc / (1.0 + self.e0)
                if self.C_er is None and has_traditional:
                    self.C_er = self.Cr / (1.0 + self.e0)
            else:
                # Cohesionless: need E_s for elastic settlement (Eq 6-54)
                if self.E_s is None or self.E_s <= 0:
                    raise ValueError(
                        "Cohesionless settling layer must have E_s > 0"
                    )

        # Compute beta from phi if not provided
        if self.beta is None and self.phi > 0:
            phi_rad = math.radians(self.phi)
            self.beta = (1.0 - math.sin(phi_rad)) * math.tan(phi_rad)

        # Default alpha for cohesive layers
        if self.alpha is None and self.soil_type == "cohesive":
            self.alpha = 1.0


@dataclass
class DowndragSoilProfile:
    """Soil profile for downdrag analysis.

    A stack of layers from the ground surface downward, with a
    groundwater table depth.

    Parameters
    ----------
    layers : list of DowndragSoilLayer
        Soil layers from surface downward.
    gwt_depth : float
        Depth to groundwater table from ground surface (m). Default 0 (at surface).
    gamma_w : float
        Unit weight of water (kN/m^3). Default 9.81.
    """
    layers: List[DowndragSoilLayer]
    gwt_depth: float = 0.0
    gamma_w: float = 9.81

    def __post_init__(self):
        if not self.layers:
            raise ValueError("At least one soil layer is required")
        if self.gwt_depth < 0:
            raise ValueError(
                f"GWT depth must be non-negative, got {self.gwt_depth}"
            )

    @property
    def total_depth(self) -> float:
        """Total depth of all layers (m)."""
        return sum(layer.thickness for layer in self.layers)

    def effective_stress_at_depth(self, z: float) -> float:
        """Compute effective vertical stress at depth z.

        Parameters
        ----------
        z : float
            Depth from ground surface (m).

        Returns
        -------
        float
            Effective vertical stress (kPa).
        """
        sigma_total = 0.0
        u = 0.0
        depth = 0.0

        for layer in self.layers:
            top = depth
            bot = depth + layer.thickness

            if z <= top:
                break

            z_in_layer = min(z, bot) - top

            # Total stress contribution from this layer
            sigma_total += layer.unit_weight * z_in_layer

            depth = bot

        # Pore water pressure (hydrostatic below GWT)
        if z > self.gwt_depth:
            u = self.gamma_w * (z - self.gwt_depth)

        return sigma_total - u

    def layer_at_depth(self, z: float) -> DowndragSoilLayer:
        """Return the layer containing depth z.

        Parameters
        ----------
        z : float
            Depth from ground surface (m).

        Returns
        -------
        DowndragSoilLayer

        Raises
        ------
        ValueError
            If z is below all layers.
        """
        depth = 0.0
        for layer in self.layers:
            bot = depth + layer.thickness
            if z <= bot:
                return layer
            depth = bot
        raise ValueError(
            f"Depth {z} m is below the soil profile (total depth {self.total_depth} m)"
        )

    def get_layer_boundaries(self) -> List[float]:
        """Return cumulative depth to the bottom of each layer.

        Returns
        -------
        list of float
            Depths to bottom of each layer (m).
        """
        boundaries = []
        depth = 0.0
        for layer in self.layers:
            depth += layer.thickness
            boundaries.append(depth)
        return boundaries
