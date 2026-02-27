"""
Results containers for wind load calculations per ASCE 7-22.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VelocityPressureResult:
    """Results from velocity pressure computation (ASCE 7-22 Section 26.10).

    Attributes
    ----------
    qz_Pa : float
        Velocity pressure at height z (Pa).
    qz_kPa : float
        Velocity pressure at height z (kPa).
    V_m_s : float
        Basic wind speed (m/s).
    z_m : float
        Height above ground (m).
    exposure_category : str
        Exposure category ('B', 'C', or 'D').
    Kz : float
        Velocity pressure exposure coefficient.
    Kzt : float
        Topographic factor.
    Kd : float
        Wind directionality factor.
    Ke : float
        Ground elevation factor.
    """

    qz_Pa: float = 0.0
    qz_kPa: float = 0.0
    V_m_s: float = 0.0
    z_m: float = 0.0
    exposure_category: str = "C"
    Kz: float = 0.0
    Kzt: float = 1.0
    Kd: float = 0.85
    Ke: float = 1.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 60,
            "  VELOCITY PRESSURE (ASCE 7-22 Section 26.10)",
            "=" * 60,
            "",
            f"  Basic wind speed V = {self.V_m_s:.1f} m/s",
            f"  Height z = {self.z_m:.2f} m",
            f"  Exposure category: {self.exposure_category}",
            "",
            "  Coefficients:",
            f"    Kz  = {self.Kz:.4f}  (velocity pressure exposure)",
            f"    Kzt = {self.Kzt:.4f}  (topographic factor)",
            f"    Kd  = {self.Kd:.4f}  (directionality factor)",
            f"    Ke  = {self.Ke:.4f}  (ground elevation factor)",
            "",
            f"  Velocity pressure qz = {self.qz_Pa:.1f} Pa ({self.qz_kPa:.3f} kPa)",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for LLM agent consumption."""
        return {
            "qz_Pa": round(self.qz_Pa, 2),
            "qz_kPa": round(self.qz_kPa, 4),
            "V_m_s": round(self.V_m_s, 2),
            "z_m": round(self.z_m, 3),
            "exposure_category": self.exposure_category,
            "Kz": round(self.Kz, 4),
            "Kzt": round(self.Kzt, 4),
            "Kd": round(self.Kd, 4),
            "Ke": round(self.Ke, 4),
        }


@dataclass
class FreestandingWallWindResult:
    """Results from freestanding wall / fence wind analysis (ASCE 7-22 Ch 29.3).

    Attributes
    ----------
    velocity_pressure_Pa : float
        Velocity pressure qh at top of wall (Pa).
    velocity_pressure_kPa : float
        Velocity pressure qh at top of wall (kPa).
    wind_pressure_Pa : float
        Design wind pressure p = qh * G * Cf (Pa).
    wind_pressure_kPa : float
        Design wind pressure (kPa).
    force_per_unit_length_kN_m : float
        Horizontal force per unit length f = p * s (kN/m).
    total_force_kN : float
        Total horizontal force F = f * B (kN).
    overturning_moment_kNm_per_m : float
        Overturning moment per unit length about base (kN*m/m).
    Kz : float
        Velocity pressure exposure coefficient at wall top.
    Kzt : float
        Topographic factor.
    Kd : float
        Wind directionality factor.
    Ke : float
        Ground elevation factor.
    G : float
        Gust-effect factor (0.85 for rigid structures).
    Cf : float
        Net force coefficient from Figure 29.3-1.
    B_over_s : float
        Wall length-to-height ratio.
    V_m_s : float
        Basic wind speed (m/s).
    wall_height_m : float
        Wall/fence height s (m).
    wall_length_m : float
        Wall/fence length B (m).
    exposure_category : str
        Exposure category.
    clearance_height_m : float
        Clearance from ground to bottom of wall (m).
    solidity_ratio : float
        Ratio of solid area to gross area (1.0 for solid wall).
    """

    velocity_pressure_Pa: float = 0.0
    velocity_pressure_kPa: float = 0.0
    wind_pressure_Pa: float = 0.0
    wind_pressure_kPa: float = 0.0
    force_per_unit_length_kN_m: float = 0.0
    total_force_kN: float = 0.0
    overturning_moment_kNm_per_m: float = 0.0

    # Coefficients
    Kz: float = 0.0
    Kzt: float = 1.0
    Kd: float = 0.85
    Ke: float = 1.0
    G: float = 0.85
    Cf: float = 0.0
    B_over_s: float = 0.0

    # Inputs echo
    V_m_s: float = 0.0
    wall_height_m: float = 0.0
    wall_length_m: float = 0.0
    exposure_category: str = "C"
    clearance_height_m: float = 0.0
    solidity_ratio: float = 1.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        wall_type = "FENCE" if self.solidity_ratio < 1.0 else "FREESTANDING WALL"
        lines = [
            "=" * 60,
            f"  {wall_type} WIND LOAD ANALYSIS (ASCE 7-22 Ch 29.3)",
            "=" * 60,
            "",
            f"  Wind speed V = {self.V_m_s:.1f} m/s",
            f"  Wall height s = {self.wall_height_m:.2f} m",
            f"  Wall length B = {self.wall_length_m:.2f} m",
            f"  B/s ratio = {self.B_over_s:.2f}",
            f"  Exposure: {self.exposure_category}",
        ]

        if self.clearance_height_m > 0:
            lines.append(f"  Clearance = {self.clearance_height_m:.2f} m")
        if self.solidity_ratio < 1.0:
            lines.append(f"  Solidity ratio = {self.solidity_ratio:.2f}")

        lines.extend([
            "",
            "  Coefficients:",
            f"    Kz  = {self.Kz:.4f}",
            f"    Kzt = {self.Kzt:.4f}",
            f"    Kd  = {self.Kd:.4f}",
            f"    Ke  = {self.Ke:.4f}",
            f"    G   = {self.G:.4f}",
            f"    Cf  = {self.Cf:.4f}",
            "",
            "  Results:",
            f"    Velocity pressure qh = {self.velocity_pressure_Pa:.1f} Pa"
            f" ({self.velocity_pressure_kPa:.3f} kPa)",
            f"    Wind pressure p      = {self.wind_pressure_Pa:.1f} Pa"
            f" ({self.wind_pressure_kPa:.3f} kPa)",
            f"    Force per length f   = {self.force_per_unit_length_kN_m:.3f} kN/m",
            f"    Total force F        = {self.total_force_kN:.2f} kN",
            f"    Overturning moment M = {self.overturning_moment_kNm_per_m:.3f} kN*m/m",
            "",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for LLM agent consumption."""
        return {
            "velocity_pressure_Pa": round(self.velocity_pressure_Pa, 2),
            "velocity_pressure_kPa": round(self.velocity_pressure_kPa, 4),
            "wind_pressure_Pa": round(self.wind_pressure_Pa, 2),
            "wind_pressure_kPa": round(self.wind_pressure_kPa, 4),
            "force_per_unit_length_kN_m": round(self.force_per_unit_length_kN_m, 4),
            "total_force_kN": round(self.total_force_kN, 3),
            "overturning_moment_kNm_per_m": round(self.overturning_moment_kNm_per_m, 4),
            "Kz": round(self.Kz, 4),
            "Kzt": round(self.Kzt, 4),
            "Kd": round(self.Kd, 4),
            "Ke": round(self.Ke, 4),
            "G": round(self.G, 4),
            "Cf": round(self.Cf, 4),
            "B_over_s": round(self.B_over_s, 2),
            "V_m_s": round(self.V_m_s, 2),
            "wall_height_m": round(self.wall_height_m, 3),
            "wall_length_m": round(self.wall_length_m, 3),
            "exposure_category": self.exposure_category,
            "clearance_height_m": round(self.clearance_height_m, 3),
            "solidity_ratio": round(self.solidity_ratio, 3),
        }
