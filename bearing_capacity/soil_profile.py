"""
Soil profile definitions for bearing capacity analysis.

Supports one- or two-layer soil systems with groundwater at any elevation.
Each layer is defined by shear strength parameters (c, phi) and unit weight.

All units are SI: kPa, kN/m³, degrees, meters.

References:
    FHWA GEC-6, Chapter 6 (FHWA-IF-02-054)
    Meyerhof & Hanna (1978) — Two-layer bearing capacity
"""

import warnings
from dataclasses import dataclass
from typing import Optional

from geotech_common.water import GAMMA_W
from geotech_common.validation import (
    check_positive, check_non_negative, check_friction_angle, check_cohesion,
    check_unit_weight,
)


@dataclass
class SoilLayer:
    """Single soil layer for bearing capacity analysis.

    Parameters
    ----------
    cohesion : float
        Cohesion c (kPa). Use c = cu (undrained shear strength) for
        undrained (phi=0) analysis.
    friction_angle : float
        Drained friction angle phi (degrees). Use 0 for undrained
        total-stress analysis in clay.
    unit_weight : float
        Total unit weight gamma (kN/m³).
    thickness : float, optional
        Layer thickness (m). None means semi-infinite (bottom layer).
    description : str, optional
        Layer description (e.g., "Medium dense sand").
    """
    cohesion: float = 0.0
    friction_angle: float = 0.0
    unit_weight: float = 18.0
    thickness: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        check_non_negative(self.cohesion, "Cohesion")
        check_cohesion(self.cohesion)
        if self.friction_angle < 0 or self.friction_angle > 50:
            raise ValueError(
                f"Friction angle must be 0-50 degrees, got {self.friction_angle}"
            )
        check_unit_weight(self.unit_weight)
        if self.cohesion == 0 and self.friction_angle == 0:
            raise ValueError("Soil must have cohesion > 0 or friction_angle > 0 (or both)")
        if self.thickness is not None:
            check_positive(self.thickness, "Layer thickness")

    @property
    def is_cohesive(self) -> bool:
        """True if this is a purely cohesive (phi=0) layer."""
        return self.friction_angle == 0.0 and self.cohesion > 0

    @property
    def is_cohesionless(self) -> bool:
        """True if this is a purely cohesionless (c=0) layer."""
        return self.cohesion == 0.0 and self.friction_angle > 0


@dataclass
class BearingSoilProfile:
    """Soil profile for bearing capacity analysis (1 or 2 layers).

    Parameters
    ----------
    layer1 : SoilLayer
        Upper soil layer (immediately below the footing base).
    layer2 : SoilLayer, optional
        Lower soil layer. If provided, enables two-layer analysis.
    gwt_depth : float, optional
        Depth to groundwater table from ground surface (m).
        If None, groundwater is assumed to be very deep (no effect).
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default 9.81.

    Notes
    -----
    For two-layer analysis, layer1.thickness must be specified.
    The groundwater table affects:
    - Overburden pressure q at the footing base
    - Effective unit weight gamma' below the water table
    """
    layer1: SoilLayer = None
    layer2: Optional[SoilLayer] = None
    gwt_depth: Optional[float] = None
    gamma_w: float = GAMMA_W

    def __post_init__(self):
        if self.layer1 is None:
            raise ValueError("At least one soil layer (layer1) must be provided")
        if self.layer2 is not None and self.layer1.thickness is None:
            raise ValueError(
                "layer1.thickness must be specified for two-layer analysis"
            )
        if self.gwt_depth is not None and self.gwt_depth < 0:
            raise ValueError(f"GWT depth must be non-negative, got {self.gwt_depth}")

    @property
    def is_two_layer(self) -> bool:
        """True if this is a two-layer soil profile."""
        return self.layer2 is not None

    def effective_unit_weight(self, depth: float, layer: SoilLayer) -> float:
        """Get effective unit weight at a given depth, accounting for groundwater.

        Parameters
        ----------
        depth : float
            Depth below ground surface (m).
        layer : SoilLayer
            The soil layer at this depth.

        Returns
        -------
        float
            Effective unit weight (kN/m³). Equal to total unit weight if
            above GWT, or buoyant unit weight if below GWT.
        """
        if self.gwt_depth is None or depth <= self.gwt_depth:
            return layer.unit_weight
        return layer.unit_weight - self.gamma_w

    def overburden_pressure(self, footing_depth: float) -> float:
        """Compute effective overburden pressure q at the footing base.

        Parameters
        ----------
        footing_depth : float
            Depth of footing base below ground surface (m).

        Returns
        -------
        float
            Effective overburden pressure q (kPa).
        """
        if footing_depth <= 0:
            return 0.0

        q = 0.0
        # For simplicity, assume layer1 extends from ground surface
        # (bearing capacity convention: the profile starts at ground surface)
        gamma = self.layer1.unit_weight
        if self.gwt_depth is None or self.gwt_depth >= footing_depth:
            # GWT below footing base or absent
            q = gamma * footing_depth
        elif self.gwt_depth <= 0:
            # GWT at or above ground surface
            gamma_eff = gamma - self.gamma_w
            q = gamma_eff * footing_depth
        else:
            # GWT between ground surface and footing base
            q = gamma * self.gwt_depth + (gamma - self.gamma_w) * (footing_depth - self.gwt_depth)
        return q

    def gamma_below_footing(self, footing_depth: float) -> float:
        """Get effective unit weight to use for the 0.5*gamma*B*Ng term.

        Per Vesic (1973) and FHWA GEC-6: use effective unit weight of
        soil within depth B below the footing base.

        Parameters
        ----------
        footing_depth : float
            Depth of footing base below ground surface (m).

        Returns
        -------
        float
            Effective unit weight for bearing capacity (kN/m³).
        """
        layer = self.layer1
        if self.gwt_depth is None or self.gwt_depth >= footing_depth:
            # GWT below footing — could be within B below
            # Conservative: use total weight (GWT is deeper)
            return layer.unit_weight
        else:
            # GWT at or above footing base — use buoyant weight
            return layer.unit_weight - self.gamma_w
