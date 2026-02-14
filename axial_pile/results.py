"""
Results container for axial pile capacity analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class AxialPileResult:
    """Results from an axial pile capacity analysis.

    Attributes
    ----------
    Q_ultimate : float
        Ultimate axial capacity (kN) = Q_skin + Q_tip.
    Q_skin : float
        Total skin friction (kN).
    Q_tip : float
        End bearing (kN).
    Q_allowable : float
        Allowable capacity Q_ultimate / FS (kN).
    Q_uplift : float, optional
        Uplift (tension) capacity (kN).
    factor_of_safety : float
        Factor of safety applied.
    pile_length : float
        Pile embedment length (m).
    pile_name : str
        Pile section designation.
    method : str
        Analysis method used.
    layer_breakdown : list of dict
        Per-layer skin friction breakdown.
    sigma_v_tip : float
        Effective stress at pile tip (kPa).
    """
    Q_ultimate: float = 0.0
    Q_skin: float = 0.0
    Q_tip: float = 0.0
    Q_allowable: float = 0.0
    Q_uplift: Optional[float] = None
    factor_of_safety: float = 2.5
    pile_length: float = 0.0
    pile_name: str = ""
    method: str = ""
    layer_breakdown: Optional[List[Dict[str, Any]]] = None
    sigma_v_tip: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 60,
            "  AXIAL PILE CAPACITY RESULTS",
            "=" * 60,
            "",
            f"  Pile: {self.pile_name}",
            f"  Embedded length: {self.pile_length:.1f} m",
            f"  Method: {self.method}",
            "",
            f"  Skin friction (Qs):  {self.Q_skin:>10,.1f} kN"
            f"  ({self._pct(self.Q_skin)}%)",
            f"  End bearing  (Qt):   {self.Q_tip:>10,.1f} kN"
            f"  ({self._pct(self.Q_tip)}%)",
            f"  {'-'*44}",
            f"  Ultimate capacity:   {self.Q_ultimate:>10,.1f} kN",
            f"  Factor of safety:    {self.factor_of_safety:>10.1f}",
            f"  Allowable capacity:  {self.Q_allowable:>10,.1f} kN",
        ]

        if self.Q_uplift is not None:
            lines.append(f"  Uplift capacity:     {self.Q_uplift:>10,.1f} kN")

        if self.layer_breakdown:
            lines.extend(["", "  Layer Breakdown:"])
            for layer in self.layer_breakdown:
                lines.append(
                    f"    {layer['depth_top_m']:.1f}-{layer['depth_bottom_m']:.1f}m: "
                    f"Qs={layer['skin_friction_kN']:.1f} kN "
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
            "factor_of_safety": self.factor_of_safety,
            "pile_length_m": self.pile_length,
            "pile_name": self.pile_name,
            "method": self.method,
            "sigma_v_tip_kPa": round(self.sigma_v_tip, 1),
        }
        if self.Q_uplift is not None:
            d["Q_uplift_kN"] = round(self.Q_uplift, 1)
        return d
