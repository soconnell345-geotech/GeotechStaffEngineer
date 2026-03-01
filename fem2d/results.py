"""
Result containers for 2D FEM analysis.

Each dataclass stores analysis outputs and provides:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption

All units SI: meters, kPa, kN, degrees.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FEMResult:
    """Results from a 2D FEM analysis.

    Attributes
    ----------
    analysis_type : str
        Type of analysis ('elastic', 'elastoplastic', 'srm').
    n_nodes : int
    n_elements : int
    max_displacement_m : float
        Maximum displacement magnitude (m).
    max_displacement_x_m : float
    max_displacement_y_m : float
    max_sigma_xx_kPa : float
    max_sigma_yy_kPa : float
    max_tau_xy_kPa : float
    min_sigma_yy_kPa : float
        Maximum compressive vertical stress (kPa).
    FOS : float or None
        Factor of safety (SRM only).
    n_yielded_elements : int
        Number of elements that yielded (MC only).
    converged : bool
    warnings : list of str
    """
    analysis_type: str = "elastic"
    n_nodes: int = 0
    n_elements: int = 0
    max_displacement_m: float = 0.0
    max_displacement_x_m: float = 0.0
    max_displacement_y_m: float = 0.0
    max_sigma_xx_kPa: float = 0.0
    max_sigma_yy_kPa: float = 0.0
    max_tau_xy_kPa: float = 0.0
    min_sigma_yy_kPa: float = 0.0
    FOS: Optional[float] = None
    n_yielded_elements: int = 0
    converged: bool = True
    n_srf_trials: int = 0
    warnings: List[str] = field(default_factory=list)

    # Raw arrays (not serialized to dict)
    nodes: Optional[np.ndarray] = field(default=None, repr=False)
    elements: Optional[np.ndarray] = field(default=None, repr=False)
    displacements: Optional[np.ndarray] = field(default=None, repr=False)
    stresses: Optional[np.ndarray] = field(default=None, repr=False)
    strains: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  2D FEM ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Analysis type: {self.analysis_type}",
            f"  Mesh: {self.n_nodes} nodes, {self.n_elements} elements",
            f"  Converged: {self.converged}",
            "",
            f"  Max displacement: {self.max_displacement_m:.4f} m",
            f"    ux_max: {self.max_displacement_x_m:.4f} m",
            f"    uy_max: {self.max_displacement_y_m:.4f} m",
            "",
            f"  Stress range:",
            f"    sigma_xx: {self.max_sigma_xx_kPa:.1f} kPa",
            f"    sigma_yy: {self.min_sigma_yy_kPa:.1f} to "
            f"{self.max_sigma_yy_kPa:.1f} kPa",
            f"    tau_xy max: {self.max_tau_xy_kPa:.1f} kPa",
        ]
        if self.FOS is not None:
            lines.extend([
                "",
                f"  Factor of Safety (SRM): {self.FOS:.3f}",
                f"  SRF trials: {self.n_srf_trials}",
            ])
        if self.n_yielded_elements > 0:
            lines.append(f"  Yielded elements: {self.n_yielded_elements}")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "analysis_type": self.analysis_type,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "converged": self.converged,
            "max_displacement_m": round(self.max_displacement_m, 6),
            "max_displacement_x_m": round(self.max_displacement_x_m, 6),
            "max_displacement_y_m": round(self.max_displacement_y_m, 6),
            "max_sigma_xx_kPa": round(self.max_sigma_xx_kPa, 2),
            "max_sigma_yy_kPa": round(self.max_sigma_yy_kPa, 2),
            "min_sigma_yy_kPa": round(self.min_sigma_yy_kPa, 2),
            "max_tau_xy_kPa": round(self.max_tau_xy_kPa, 2),
            "warnings": self.warnings,
        }
        if self.FOS is not None:
            d["FOS"] = self.FOS
            d["n_srf_trials"] = self.n_srf_trials
        if self.n_yielded_elements > 0:
            d["n_yielded_elements"] = self.n_yielded_elements
        return d
