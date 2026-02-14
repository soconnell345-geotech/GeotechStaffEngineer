"""
Results container for drilled shaft capacity analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class DrillShaftResult:
    """Results from a drilled shaft capacity analysis.

    Attributes
    ----------
    Q_ultimate : float
        Ultimate axial capacity (kN) = Q_skin + Q_tip.
    Q_skin : float
        Total side resistance (kN).
    Q_tip : float
        End bearing (kN).
    Q_allowable : float
        Allowable capacity Q_ultimate / FS (kN).
    Q_side_clay : float
        Side resistance from cohesive layers (kN).
    Q_side_sand : float
        Side resistance from cohesionless layers (kN).
    Q_side_rock : float
        Side resistance from rock socket (kN).
    factor_of_safety : float
        Factor of safety applied.
    shaft_diameter : float
        Shaft diameter (m).
    shaft_length : float
        Shaft length (m).
    method : str
        Analysis method description.
    layer_breakdown : list of dict
        Per-layer side resistance breakdown.
    sigma_v_tip : float
        Effective stress at shaft tip (kPa).
    """
    Q_ultimate: float = 0.0
    Q_skin: float = 0.0
    Q_tip: float = 0.0
    Q_allowable: float = 0.0
    Q_side_clay: float = 0.0
    Q_side_sand: float = 0.0
    Q_side_rock: float = 0.0
    factor_of_safety: float = 2.5
    shaft_diameter: float = 0.0
    shaft_length: float = 0.0
    method: str = "GEC-10"
    layer_breakdown: Optional[List[Dict[str, Any]]] = None
    sigma_v_tip: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 60,
            "  DRILLED SHAFT CAPACITY RESULTS",
            "=" * 60,
            "",
            f"  Shaft diameter: {self.shaft_diameter:.3f} m",
            f"  Shaft length:   {self.shaft_length:.1f} m",
            f"  Method: {self.method}",
            "",
            f"  Side resistance (Qs): {self.Q_skin:>10,.1f} kN"
            f"  ({self._pct(self.Q_skin)}%)",
        ]

        if self.Q_side_clay > 0:
            lines.append(f"    Clay:    {self.Q_side_clay:>10,.1f} kN")
        if self.Q_side_sand > 0:
            lines.append(f"    Sand:    {self.Q_side_sand:>10,.1f} kN")
        if self.Q_side_rock > 0:
            lines.append(f"    Rock:    {self.Q_side_rock:>10,.1f} kN")

        lines.extend([
            f"  End bearing  (Qb):    {self.Q_tip:>10,.1f} kN"
            f"  ({self._pct(self.Q_tip)}%)",
            f"  {'-'*44}",
            f"  Ultimate capacity:    {self.Q_ultimate:>10,.1f} kN",
            f"  Factor of safety:     {self.factor_of_safety:>10.1f}",
            f"  Allowable capacity:   {self.Q_allowable:>10,.1f} kN",
        ])

        if self.layer_breakdown:
            lines.extend(["", "  Layer Breakdown:"])
            for layer in self.layer_breakdown:
                lines.append(
                    f"    {layer['depth_top_m']:.1f}-{layer['depth_bottom_m']:.1f}m: "
                    f"Qs={layer['side_resistance_kN']:.1f} kN "
                    f"({layer['soil_type']}, {layer['method']})"
                )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _pct(self, component: float) -> str:
        if self.Q_ultimate > 0:
            return f"{100 * component / self.Q_ultimate:.0f}"
        return "0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for LLM agent consumption."""
        d = {
            "Q_ultimate_kN": round(self.Q_ultimate, 1),
            "Q_skin_kN": round(self.Q_skin, 1),
            "Q_tip_kN": round(self.Q_tip, 1),
            "Q_allowable_kN": round(self.Q_allowable, 1),
            "Q_side_clay_kN": round(self.Q_side_clay, 1),
            "Q_side_sand_kN": round(self.Q_side_sand, 1),
            "Q_side_rock_kN": round(self.Q_side_rock, 1),
            "factor_of_safety": self.factor_of_safety,
            "shaft_diameter_m": self.shaft_diameter,
            "shaft_length_m": self.shaft_length,
            "method": self.method,
            "sigma_v_tip_kPa": round(self.sigma_v_tip, 1),
        }
        return d
