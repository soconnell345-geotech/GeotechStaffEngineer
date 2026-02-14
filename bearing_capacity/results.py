"""
Results container for bearing capacity analysis.

Stores all computed factors and provides summary output and optional plotting.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class BearingCapacityResult:
    """Results from a bearing capacity analysis.

    Attributes
    ----------
    q_ultimate : float
        Ultimate bearing capacity qult (kPa).
    q_allowable : float
        Allowable bearing capacity qult/FS (kPa).
    q_net : float
        Net ultimate bearing capacity qult - q (kPa).
    factor_of_safety : float
        Factor of safety applied.

    Nc, Nq, Ngamma : float
        Bearing capacity factors.
    sc, sq, sgamma : float
        Shape factors.
    dc, dq, dgamma : float
        Depth factors.
    ic, iq, igamma : float
        Load inclination factors.
    bc, bq, bgamma : float
        Base inclination factors.
    gc, gq, ggamma : float
        Ground inclination factors.

    q_overburden : float
        Effective overburden pressure at footing base (kPa).
    gamma_eff : float
        Effective unit weight below footing (kN/m³).
    B_eff : float
        Effective footing width used in computation (m).
    L_eff : float
        Effective footing length used in computation (m).

    term_cohesion : float
        Cohesion term contribution to qult (kPa).
    term_overburden : float
        Overburden term contribution to qult (kPa).
    term_selfweight : float
        Self-weight term contribution to qult (kPa).

    ngamma_method : str
        Method used for Ngamma.
    factor_method : str
        Method used for correction factors.

    is_two_layer : bool
        True if this was a two-layer analysis.
    q_upper_layer : float, optional
        qult of upper layer (two-layer analysis only).
    q_lower_layer : float, optional
        qult of lower layer (two-layer analysis only).
    """
    # Primary results
    q_ultimate: float = 0.0
    q_allowable: float = 0.0
    q_net: float = 0.0
    factor_of_safety: float = 3.0

    # Bearing capacity factors
    Nc: float = 0.0
    Nq: float = 0.0
    Ngamma: float = 0.0

    # Shape factors
    sc: float = 1.0
    sq: float = 1.0
    sgamma: float = 1.0

    # Depth factors
    dc: float = 1.0
    dq: float = 1.0
    dgamma: float = 1.0

    # Inclination factors
    ic: float = 1.0
    iq: float = 1.0
    igamma: float = 1.0

    # Base inclination factors
    bc: float = 1.0
    bq: float = 1.0
    bgamma: float = 1.0

    # Ground inclination factors
    gc: float = 1.0
    gq: float = 1.0
    ggamma: float = 1.0

    # Intermediate values
    q_overburden: float = 0.0
    gamma_eff: float = 0.0
    B_eff: float = 0.0
    L_eff: float = 0.0

    # Term breakdown
    term_cohesion: float = 0.0
    term_overburden: float = 0.0
    term_selfweight: float = 0.0

    # Method info
    ngamma_method: str = "vesic"
    factor_method: str = "vesic"

    # Two-layer info
    is_two_layer: bool = False
    q_upper_layer: Optional[float] = None
    q_lower_layer: Optional[float] = None

    def summary(self) -> str:
        """Return a formatted summary string.

        Returns
        -------
        str
            Human-readable summary of the bearing capacity results.
        """
        lines = [
            "=" * 60,
            "  BEARING CAPACITY ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Ultimate bearing capacity  qult = {self.q_ultimate:,.1f} kPa",
            f"  Factor of safety           FS   = {self.factor_of_safety:.1f}",
            f"  Allowable bearing capacity qall = {self.q_allowable:,.1f} kPa",
            f"  Net ultimate capacity      qnet = {self.q_net:,.1f} kPa",
            "",
            f"  Effective footing: B' = {self.B_eff:.3f} m, L' = {self.L_eff:.3f} m",
            f"  Overburden pressure: q = {self.q_overburden:.1f} kPa",
            f"  Effective unit weight: gamma = {self.gamma_eff:.2f} kN/m³",
            "",
            "  Bearing Capacity Factors:",
            f"    Nc = {self.Nc:.2f}   Nq = {self.Nq:.2f}   Ng = {self.Ngamma:.2f}"
            f"  ({self.ngamma_method})",
            "",
            "  Correction Factors ({0}):".format(self.factor_method),
            f"    Shape:       sc={self.sc:.3f}  sq={self.sq:.3f}  sg={self.sgamma:.3f}",
            f"    Depth:       dc={self.dc:.3f}  dq={self.dq:.3f}  dg={self.dgamma:.3f}",
            f"    Inclination: ic={self.ic:.3f}  iq={self.iq:.3f}  ig={self.igamma:.3f}",
            f"    Base tilt:   bc={self.bc:.3f}  bq={self.bq:.3f}  bg={self.bgamma:.3f}",
            f"    Ground:      gc={self.gc:.3f}  gq={self.gq:.3f}  gg={self.ggamma:.3f}",
            "",
            "  Term Breakdown:",
            f"    Cohesion term:    {self.term_cohesion:,.1f} kPa"
            f"  ({self._pct(self.term_cohesion)}%)",
            f"    Overburden term:  {self.term_overburden:,.1f} kPa"
            f"  ({self._pct(self.term_overburden)}%)",
            f"    Self-weight term: {self.term_selfweight:,.1f} kPa"
            f"  ({self._pct(self.term_selfweight)}%)",
        ]

        if self.is_two_layer:
            lines.extend([
                "",
                "  Two-Layer Analysis:",
                f"    Upper layer qult: {self.q_upper_layer:,.1f} kPa",
                f"    Lower layer qult: {self.q_lower_layer:,.1f} kPa",
            ])

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _pct(self, term: float) -> str:
        if self.q_ultimate > 0:
            return f"{100 * term / self.q_ultimate:.0f}"
        return "0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for LLM agent consumption.

        Returns
        -------
        dict
            All result fields as a flat dictionary.
        """
        return {
            "q_ultimate_kPa": round(self.q_ultimate, 2),
            "q_allowable_kPa": round(self.q_allowable, 2),
            "q_net_kPa": round(self.q_net, 2),
            "factor_of_safety": self.factor_of_safety,
            "Nc": round(self.Nc, 3),
            "Nq": round(self.Nq, 3),
            "Ngamma": round(self.Ngamma, 3),
            "sc": round(self.sc, 4),
            "sq": round(self.sq, 4),
            "sgamma": round(self.sgamma, 4),
            "dc": round(self.dc, 4),
            "dq": round(self.dq, 4),
            "dgamma": round(self.dgamma, 4),
            "ic": round(self.ic, 4),
            "iq": round(self.iq, 4),
            "igamma": round(self.igamma, 4),
            "bc": round(self.bc, 4),
            "bq": round(self.bq, 4),
            "bgamma": round(self.bgamma, 4),
            "gc": round(self.gc, 4),
            "gq": round(self.gq, 4),
            "ggamma": round(self.ggamma, 4),
            "q_overburden_kPa": round(self.q_overburden, 2),
            "gamma_eff_kNm3": round(self.gamma_eff, 3),
            "B_eff_m": round(self.B_eff, 4),
            "L_eff_m": round(self.L_eff, 4),
            "term_cohesion_kPa": round(self.term_cohesion, 2),
            "term_overburden_kPa": round(self.term_overburden, 2),
            "term_selfweight_kPa": round(self.term_selfweight, 2),
            "ngamma_method": self.ngamma_method,
            "factor_method": self.factor_method,
            "is_two_layer": self.is_two_layer,
        }
