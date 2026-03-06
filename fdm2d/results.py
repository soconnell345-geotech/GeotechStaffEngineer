"""
Result container for 2D explicit FDM analysis.

Follows the fem2d FEMResult pattern:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption

All units SI: meters, kPa, kN, degrees.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FDMResult:
    """Results from a 2D explicit FDM analysis.

    Attributes
    ----------
    analysis_type : str
        Type of analysis ('gravity', 'foundation').
    n_gridpoints : int
    n_zones : int
    max_displacement_m : float
        Maximum displacement magnitude (m).
    max_displacement_x_m : float
    max_displacement_y_m : float
    max_sigma_xx_kPa : float
    max_sigma_yy_kPa : float
    max_tau_xy_kPa : float
    min_sigma_yy_kPa : float
        Maximum compressive vertical stress (kPa).
    converged : bool
    n_timesteps : int
    final_force_ratio : float
    warnings : list of str
    """
    analysis_type: str = "gravity"
    n_gridpoints: int = 0
    n_zones: int = 0
    max_displacement_m: float = 0.0
    max_displacement_x_m: float = 0.0
    max_displacement_y_m: float = 0.0
    max_sigma_xx_kPa: float = 0.0
    max_sigma_yy_kPa: float = 0.0
    max_tau_xy_kPa: float = 0.0
    min_sigma_yy_kPa: float = 0.0
    converged: bool = True
    n_timesteps: int = 0
    final_force_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # Raw arrays (not serialized to dict)
    nodes: Optional[np.ndarray] = field(default=None, repr=False)
    zones: Optional[np.ndarray] = field(default=None, repr=False)
    displacements: Optional[np.ndarray] = field(default=None, repr=False)
    stresses: Optional[np.ndarray] = field(default=None, repr=False)
    velocities: Optional[np.ndarray] = field(default=None, repr=False)
    force_ratio_history: Optional[List[float]] = field(default=None,
                                                       repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  2D EXPLICIT FDM ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Analysis type: {self.analysis_type}",
            f"  Grid: {self.n_gridpoints} gridpoints, {self.n_zones} zones",
            f"  Converged: {self.converged}",
            f"  Timesteps: {self.n_timesteps}",
            f"  Final force ratio: {self.final_force_ratio:.2e}",
            "",
            f"  Max displacement: {self.max_displacement_m:.6f} m",
            f"    ux_max: {self.max_displacement_x_m:.6f} m",
            f"    uy_max: {self.max_displacement_y_m:.6f} m",
            "",
            f"  Stress range:",
            f"    sigma_xx: {self.max_sigma_xx_kPa:.1f} kPa",
            f"    sigma_yy: {self.min_sigma_yy_kPa:.1f} to "
            f"{self.max_sigma_yy_kPa:.1f} kPa",
            f"    tau_xy max: {self.max_tau_xy_kPa:.1f} kPa",
        ]
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "n_gridpoints": self.n_gridpoints,
            "n_zones": self.n_zones,
            "converged": self.converged,
            "n_timesteps": self.n_timesteps,
            "final_force_ratio": float(f"{self.final_force_ratio:.4e}"),
            "max_displacement_m": round(self.max_displacement_m, 6),
            "max_displacement_x_m": round(self.max_displacement_x_m, 6),
            "max_displacement_y_m": round(self.max_displacement_y_m, 6),
            "max_sigma_xx_kPa": round(self.max_sigma_xx_kPa, 2),
            "max_sigma_yy_kPa": round(self.max_sigma_yy_kPa, 2),
            "min_sigma_yy_kPa": round(self.min_sigma_yy_kPa, 2),
            "max_tau_xy_kPa": round(self.max_tau_xy_kPa, 2),
            "warnings": self.warnings,
        }
