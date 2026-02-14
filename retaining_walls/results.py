"""
Results containers for retaining wall analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class CantileverWallResult:
    """Results from cantilever retaining wall analysis.

    Attributes
    ----------
    FOS_sliding : float
        Factor of safety against sliding.
    FOS_overturning : float
        Factor of safety against overturning.
    FOS_bearing : float
        Factor of safety against bearing failure (q_allowable/q_toe).
    passes_sliding : bool
        True if FOS_sliding >= required (typically 1.5).
    passes_overturning : bool
        True if FOS_overturning >= required (typically 2.0).
    passes_bearing : bool
        True if bearing is adequate.
    q_toe : float
        Maximum bearing pressure at toe (kPa).
    q_heel : float
        Bearing pressure at heel (kPa).
    eccentricity : float
        Eccentricity of resultant from center of base (m).
    in_middle_third : bool
        True if resultant is within middle third of base.
    wall_height : float
        Wall height (m).
    base_width : float
        Base width (m).
    """
    FOS_sliding: float = 0.0
    FOS_overturning: float = 0.0
    FOS_bearing: float = 0.0
    passes_sliding: bool = False
    passes_overturning: bool = False
    passes_bearing: bool = False
    q_toe: float = 0.0
    q_heel: float = 0.0
    eccentricity: float = 0.0
    in_middle_third: bool = False
    wall_height: float = 0.0
    base_width: float = 0.0

    def summary(self) -> str:
        def _status(passes):
            return "OK" if passes else "FAIL"

        lines = [
            "=" * 60,
            "  CANTILEVER RETAINING WALL RESULTS",
            "=" * 60,
            "",
            f"  Wall height:  {self.wall_height:.2f} m",
            f"  Base width:   {self.base_width:.2f} m",
            "",
            f"  Sliding:      FOS = {self.FOS_sliding:.2f}  [{_status(self.passes_sliding)}]",
            f"  Overturning:  FOS = {self.FOS_overturning:.2f}  [{_status(self.passes_overturning)}]",
            f"  Bearing:      FOS = {self.FOS_bearing:.2f}  [{_status(self.passes_bearing)}]",
            "",
            f"  q_toe  = {self.q_toe:.1f} kPa",
            f"  q_heel = {self.q_heel:.1f} kPa",
            f"  Eccentricity = {self.eccentricity:.3f} m"
            f"  ({'middle third' if self.in_middle_third else 'OUTSIDE middle third'})",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "FOS_sliding": round(self.FOS_sliding, 3),
            "FOS_overturning": round(self.FOS_overturning, 3),
            "FOS_bearing": round(self.FOS_bearing, 3),
            "passes_sliding": self.passes_sliding,
            "passes_overturning": self.passes_overturning,
            "passes_bearing": self.passes_bearing,
            "q_toe_kPa": round(self.q_toe, 1),
            "q_heel_kPa": round(self.q_heel, 1),
            "eccentricity_m": round(self.eccentricity, 3),
            "in_middle_third": self.in_middle_third,
            "wall_height_m": self.wall_height,
            "base_width_m": self.base_width,
        }


@dataclass
class MSEWallResult:
    """Results from MSE wall analysis.

    Attributes
    ----------
    FOS_sliding : float
        External FOS against sliding.
    FOS_overturning : float
        External FOS against overturning.
    FOS_bearing : float
        External FOS against bearing failure.
    passes_external : bool
        True if all external checks pass.
    internal_results : list of dict
        Per-level internal stability check results.
    all_pass_internal : bool
        True if all internal checks pass.
    wall_height : float
        Wall height (m).
    reinforcement_length : float
        Reinforcement length (m).
    """
    FOS_sliding: float = 0.0
    FOS_overturning: float = 0.0
    FOS_bearing: float = 0.0
    passes_external: bool = False
    internal_results: List[Dict[str, Any]] = field(default_factory=list)
    all_pass_internal: bool = False
    wall_height: float = 0.0
    reinforcement_length: float = 0.0

    @property
    def n_levels(self) -> int:
        return len(self.internal_results)

    def summary(self) -> str:
        def _status(passes):
            return "OK" if passes else "FAIL"

        lines = [
            "=" * 60,
            "  MSE WALL RESULTS",
            "=" * 60,
            "",
            f"  Wall height:       {self.wall_height:.2f} m",
            f"  Reinforcement L:   {self.reinforcement_length:.2f} m",
            "",
            "  EXTERNAL STABILITY:",
            f"    Sliding:     FOS = {self.FOS_sliding:.2f}",
            f"    Overturning: FOS = {self.FOS_overturning:.2f}",
            f"    Bearing:     FOS = {self.FOS_bearing:.2f}",
            f"    External:    [{_status(self.passes_external)}]",
            "",
            f"  INTERNAL STABILITY ({self.n_levels} levels):",
        ]

        for r in self.internal_results:
            status = "OK" if r.get("passes", False) else "FAIL"
            lines.append(
                f"    z={r['depth_m']:.1f}m: "
                f"Tmax={r['Tmax_kN_per_m']:.1f}, "
                f"FOS_po={r['FOS_pullout']:.2f}  [{status}]"
            )

        lines.extend([
            f"    Internal:    [{_status(self.all_pass_internal)}]",
            "",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "FOS_sliding": round(self.FOS_sliding, 3),
            "FOS_overturning": round(self.FOS_overturning, 3),
            "FOS_bearing": round(self.FOS_bearing, 3),
            "passes_external": self.passes_external,
            "all_pass_internal": self.all_pass_internal,
            "n_levels": self.n_levels,
            "wall_height_m": self.wall_height,
            "reinforcement_length_m": self.reinforcement_length,
            "internal_results": self.internal_results,
        }
