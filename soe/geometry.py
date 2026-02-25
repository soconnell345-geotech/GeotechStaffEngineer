"""
Input geometry dataclasses for support of excavation design.

All units SI: meters, kPa, kN/m³, degrees.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SOEWallLayer:
    """A single soil layer for SOE wall design.

    Parameters
    ----------
    thickness : float
        Layer thickness (m).
    unit_weight : float
        Total unit weight (kN/m³).
    friction_angle : float
        Drained friction angle (degrees).
    cohesion : float
        Undrained shear strength cu for clay, or effective cohesion c'
        for sand (kPa). Default 0.
    soil_type : str
        One of "sand", "soft_clay", "stiff_clay". Controls apparent
        pressure diagram selection.
    description : str
        Optional layer description.
    """
    thickness: float
    unit_weight: float
    friction_angle: float = 30.0
    cohesion: float = 0.0
    soil_type: str = "sand"
    description: str = ""


@dataclass
class SupportLevel:
    """A single level of lateral support (strut, anchor, or raker).

    Parameters
    ----------
    depth : float
        Depth from top of wall to the support (m).
    support_type : str
        One of "strut", "anchor", "raker".
    spacing : float
        Horizontal center-to-center spacing (m). Default 3.0.
    angle_deg : float
        Inclination from horizontal (degrees). 0 = horizontal strut,
        positive = inclined downward (typical for anchors). Default 0.
    preload_kN_per_m : float
        Lock-off / preload per meter of wall (kN/m). Default 0.
    """
    depth: float
    support_type: str = "strut"
    spacing: float = 3.0
    angle_deg: float = 0.0
    preload_kN_per_m: float = 0.0


@dataclass
class ExcavationGeometry:
    """Complete excavation geometry for SOE analysis.

    Parameters
    ----------
    excavation_depth : float
        Total depth of excavation H (m).
    soil_layers : list
        List of SOEWallLayer from ground surface downward.
    support_levels : list
        List of SupportLevel, sorted by depth. Empty for cantilever.
    surcharge : float
        Uniform surface surcharge (kPa). Default 10 (construction).
    gwt_depth : float or None
        Groundwater depth from surface (m). None = no water.
    excavation_width : float
        Plan width of excavation B (m). Used for basal heave.
        0 = infinitely long (strip condition).
    """
    excavation_depth: float
    soil_layers: List[SOEWallLayer] = field(default_factory=list)
    support_levels: List[SupportLevel] = field(default_factory=list)
    surcharge: float = 10.0
    gwt_depth: Optional[float] = None
    excavation_width: float = 0.0

    def validate(self):
        """Check geometry for common errors."""
        if self.excavation_depth <= 0:
            raise ValueError("excavation_depth must be positive")
        if not self.soil_layers:
            raise ValueError("At least one soil layer is required")

        total_thickness = sum(layer.thickness for layer in self.soil_layers)
        if total_thickness < self.excavation_depth:
            raise ValueError(
                f"Total soil layer thickness ({total_thickness:.1f} m) must be "
                f">= excavation depth ({self.excavation_depth:.1f} m)"
            )

        for layer in self.soil_layers:
            if layer.thickness <= 0:
                raise ValueError("All layer thicknesses must be positive")
            if layer.unit_weight <= 0:
                raise ValueError("All layer unit weights must be positive")
            if layer.soil_type not in ("sand", "soft_clay", "stiff_clay"):
                raise ValueError(
                    f"soil_type must be 'sand', 'soft_clay', or 'stiff_clay', "
                    f"got '{layer.soil_type}'"
                )

        for i, sup in enumerate(self.support_levels):
            if sup.depth <= 0 or sup.depth >= self.excavation_depth:
                raise ValueError(
                    f"Support {i} depth ({sup.depth} m) must be between "
                    f"0 and excavation_depth ({self.excavation_depth} m)"
                )

        # Sort support levels by depth
        depths = [s.depth for s in self.support_levels]
        if depths != sorted(depths):
            raise ValueError("Support levels must be sorted by depth")

    def weighted_avg_unit_weight(self) -> float:
        """Weighted average unit weight over excavation depth."""
        total_gamma_h = 0.0
        remaining = self.excavation_depth
        for layer in self.soil_layers:
            h = min(layer.thickness, remaining)
            total_gamma_h += layer.unit_weight * h
            remaining -= h
            if remaining <= 0:
                break
        return total_gamma_h / self.excavation_depth

    def weighted_avg_cu(self) -> float:
        """Weighted average undrained shear strength over excavation depth."""
        total_cu_h = 0.0
        remaining = self.excavation_depth
        for layer in self.soil_layers:
            h = min(layer.thickness, remaining)
            total_cu_h += layer.cohesion * h
            remaining -= h
            if remaining <= 0:
                break
        return total_cu_h / self.excavation_depth
