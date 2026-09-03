"""Result dataclasses for the concrete properties agent."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RCSectionResult:
    """Rectangular RC section analysis results.

    Attributes
    ----------
    b_mm, h_mm : float
        Section width and overall depth.
    fc_MPa, fy_MPa : float
        Concrete cylinder strength and bar yield strength.
    ec_MPa : float
        Concrete elastic modulus used (default ACI 318: 4700*sqrt(f'c)).
    as_bot_mm2, as_top_mm2 : float
        Total bottom / top steel areas.
    d_eff_mm : float
        Effective depth to the bottom-steel centroid.
    gross_area_mm2 : float
        Gross transformed-free concrete+steel area.
    ixx_gross_mm4 : float
        Gross (uncracked, transformed) second moment of area.
    ixx_cracked_mm4 : float or None
        Cracked transformed second moment of area (sagging).
    m_cr_kNm : float
        Cracking moment (sagging).
    mn_pos_kNm : float
        Ultimate (nominal) sagging moment capacity, bottom steel in tension.
    mn_neg_kNm : float or None
        Ultimate hogging capacity (top steel in tension); None if no top
        steel was specified.
    interaction : list of (n_kN, m_kNm)
        N-M interaction diagram points (compression positive), if requested.
    """
    b_mm: float = 0.0
    h_mm: float = 0.0
    fc_MPa: float = 0.0
    fy_MPa: float = 0.0
    ec_MPa: float = 0.0
    as_bot_mm2: float = 0.0
    as_top_mm2: float = 0.0
    d_eff_mm: float = 0.0
    gross_area_mm2: float = 0.0
    ixx_gross_mm4: float = 0.0
    ixx_cracked_mm4: Optional[float] = None
    m_cr_kNm: float = 0.0
    mn_pos_kNm: float = 0.0
    mn_neg_kNm: Optional[float] = None
    interaction: List[Tuple[float, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  RC RECTANGULAR SECTION ANALYSIS",
            "=" * 60,
            f"  Section:            {self.b_mm:.0f} x {self.h_mm:.0f} mm, "
            f"f'c = {self.fc_MPa:.0f} MPa, fy = {self.fy_MPa:.0f} MPa",
            f"  Steel (bot / top):  {self.as_bot_mm2:,.0f} / "
            f"{self.as_top_mm2:,.0f} mm^2, d = {self.d_eff_mm:.0f} mm",
            f"  Ixx gross:          {self.ixx_gross_mm4:,.3e} mm^4",
        ]
        if self.ixx_cracked_mm4 is not None:
            lines.append(f"  Ixx cracked:        {self.ixx_cracked_mm4:,.3e} mm^4")
        lines += [
            f"  M_cr:               {self.m_cr_kNm:,.1f} kN*m",
            f"  Mn (sagging):       {self.mn_pos_kNm:,.1f} kN*m",
        ]
        if self.mn_neg_kNm is not None:
            lines.append(f"  Mn (hogging):       {self.mn_neg_kNm:,.1f} kN*m")
        if self.interaction:
            n_vals = [p[0] for p in self.interaction]
            m_vals = [p[1] for p in self.interaction]
            lines.append(
                f"  N-M interaction:    {len(self.interaction)} points, "
                f"N in [{min(n_vals):,.0f}, {max(n_vals):,.0f}] kN, "
                f"M_max = {max(m_vals):,.1f} kN*m")
        lines.append("  (Nominal capacities — apply code phi factors separately.)")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "b_mm": float(self.b_mm),
            "h_mm": float(self.h_mm),
            "fc_MPa": float(self.fc_MPa),
            "fy_MPa": float(self.fy_MPa),
            "ec_MPa": float(self.ec_MPa),
            "as_bot_mm2": float(self.as_bot_mm2),
            "as_top_mm2": float(self.as_top_mm2),
            "d_eff_mm": float(self.d_eff_mm),
            "gross_area_mm2": float(self.gross_area_mm2),
            "ixx_gross_mm4": float(self.ixx_gross_mm4),
            "m_cr_kNm": float(self.m_cr_kNm),
            "mn_pos_kNm": float(self.mn_pos_kNm),
            "note": "Nominal capacities - apply code phi factors separately.",
        }
        if self.ixx_cracked_mm4 is not None:
            d["ixx_cracked_mm4"] = float(self.ixx_cracked_mm4)
        if self.mn_neg_kNm is not None:
            d["mn_neg_kNm"] = float(self.mn_neg_kNm)
        if self.interaction:
            d["interaction_n_kN"] = [round(float(p[0]), 2) for p in self.interaction]
            d["interaction_m_kNm"] = [round(float(p[1]), 2) for p in self.interaction]
        return d
