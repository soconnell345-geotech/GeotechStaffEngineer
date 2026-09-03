"""Result dataclasses for the PyNite agent."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MemberResult:
    """Per-member envelope results (kN, kN*m, m).

    Sign convention: values carry PyNite's local-axis signs (a UDL in -Y on
    a simply-supported beam gives negative Mz at midspan = sagging);
    ``*_abs`` fields are magnitudes for design use.
    """
    name: str = ""
    max_moment_kNm: float = 0.0
    min_moment_kNm: float = 0.0
    moment_abs_kNm: float = 0.0
    max_shear_kN: float = 0.0
    min_shear_kN: float = 0.0
    shear_abs_kN: float = 0.0
    max_axial_kN: float = 0.0
    min_axial_kN: float = 0.0
    max_deflection_m: float = 0.0
    min_deflection_m: float = 0.0
    deflection_abs_mm: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "max_moment_kNm": round(float(self.max_moment_kNm), 4),
            "min_moment_kNm": round(float(self.min_moment_kNm), 4),
            "moment_abs_kNm": round(float(self.moment_abs_kNm), 4),
            "max_shear_kN": round(float(self.max_shear_kN), 4),
            "min_shear_kN": round(float(self.min_shear_kN), 4),
            "shear_abs_kN": round(float(self.shear_abs_kN), 4),
            "max_axial_kN": round(float(self.max_axial_kN), 4),
            "min_axial_kN": round(float(self.min_axial_kN), 4),
            "max_deflection_m": float(self.max_deflection_m),
            "min_deflection_m": float(self.min_deflection_m),
            "deflection_abs_mm": round(float(self.deflection_abs_mm), 3),
        }


@dataclass
class FrameResult:
    """Frame analysis results.

    Attributes
    ----------
    n_nodes, n_members : int
    reactions : dict
        node -> {"FX_kN", "FY_kN", "FZ_kN", "MX_kNm", "MY_kNm", "MZ_kNm"}
        for supported nodes.
    members : list of MemberResult
    max_deflection_mm : float
        Largest member deflection magnitude in the model.
    """
    n_nodes: int = 0
    n_members: int = 0
    reactions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    members: List[MemberResult] = field(default_factory=list)
    max_deflection_mm: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  FRAME ANALYSIS (linear elastic)",
            "=" * 60,
            f"  Nodes / members:    {self.n_nodes} / {self.n_members}",
        ]
        for node, r in self.reactions.items():
            comps = ", ".join(f"{k[:2]}={v:,.2f}" for k, v in r.items()
                              if abs(v) > 1e-9)
            lines.append(f"  Reaction @{node}:     {comps or '0'}")
        for m in self.members:
            lines.append(
                f"  {m.name}: |M|max={m.moment_abs_kNm:,.2f} kN*m, "
                f"|V|max={m.shear_abs_kN:,.2f} kN, "
                f"defl={m.deflection_abs_mm:,.2f} mm")
        lines.append(f"  Max deflection:     {self.max_deflection_mm:,.2f} mm")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_members": self.n_members,
            "reactions": {
                n: {k: round(float(v), 4) for k, v in r.items()}
                for n, r in self.reactions.items()},
            "members": [m.to_dict() for m in self.members],
            "max_deflection_mm": round(float(self.max_deflection_mm), 3),
        }


@dataclass
class ContinuousBeamResult:
    """Continuous-beam convenience results (kN, kN*m, m/mm)."""
    n_spans: int = 0
    span_lengths_m: List[float] = field(default_factory=list)
    support_reactions_kN: List[float] = field(default_factory=list)
    support_moments_kNm: List[float] = field(default_factory=list)
    span_max_sagging_kNm: List[float] = field(default_factory=list)
    span_max_deflection_mm: List[float] = field(default_factory=list)
    max_sagging_kNm: float = 0.0
    max_hogging_kNm: float = 0.0
    max_deflection_mm: float = 0.0
    frame: Optional[FrameResult] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  CONTINUOUS BEAM ANALYSIS",
            "=" * 60,
            f"  Spans:              {self.n_spans} "
            f"({', '.join(f'{s:g} m' for s in self.span_lengths_m)})",
            "  Support reactions:  " +
            ", ".join(f"{r:,.2f}" for r in self.support_reactions_kN) + " kN",
            "  Support moments:    " +
            ", ".join(f"{m:,.2f}" for m in self.support_moments_kNm) + " kN*m",
            f"  Max sagging M:      {self.max_sagging_kNm:,.2f} kN*m",
            f"  Max hogging M:      {self.max_hogging_kNm:,.2f} kN*m",
            f"  Max deflection:     {self.max_deflection_mm:,.3f} mm",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_spans": self.n_spans,
            "span_lengths_m": [float(s) for s in self.span_lengths_m],
            "support_reactions_kN": [round(float(r), 4)
                                     for r in self.support_reactions_kN],
            "support_moments_kNm": [round(float(m), 4)
                                    for m in self.support_moments_kNm],
            "span_max_sagging_kNm": [round(float(m), 4)
                                     for m in self.span_max_sagging_kNm],
            "span_max_deflection_mm": [round(float(d), 3)
                                       for d in self.span_max_deflection_mm],
            "max_sagging_kNm": round(float(self.max_sagging_kNm), 4),
            "max_hogging_kNm": round(float(self.max_hogging_kNm), 4),
            "max_deflection_mm": round(float(self.max_deflection_mm), 3),
        }
