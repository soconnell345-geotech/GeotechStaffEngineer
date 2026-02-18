"""
Result containers for ground improvement analyses.

Each dataclass stores analysis outputs and provides:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption

All units SI: kPa, meters, years.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class AggregatePierResult:
    """Results from aggregate pier / rammed aggregate pier analysis.

    Attributes
    ----------
    area_replacement_ratio : float
        as = Ac/A, column area over tributary area.
    stress_concentration_ratio : float
        n = sigma_c / sigma_s.
    composite_modulus_kPa : float
        E_comp = Es * [1 + as*(n-1)].
    settlement_reduction_factor : float
        SRF = 1 / (1 + as*(n-1)).
    improved_bearing_kPa : float
        q_improved after pier installation.
    unreinforced_bearing_kPa : float
        q_unreinforced (input echo).
    settlement_improved_mm : float
        S_improved = SRF * S_unreinforced.
    settlement_unreinforced_mm : float
        S_unreinforced (input echo).
    column_diameter_m : float
        Pier column diameter.
    column_spacing_m : float
        Center-to-center spacing.
    pattern : str
        'triangular' or 'square'.
    """
    area_replacement_ratio: float = 0.0
    stress_concentration_ratio: float = 0.0
    composite_modulus_kPa: float = 0.0
    settlement_reduction_factor: float = 0.0
    improved_bearing_kPa: float = 0.0
    unreinforced_bearing_kPa: float = 0.0
    settlement_improved_mm: float = 0.0
    settlement_unreinforced_mm: float = 0.0
    column_diameter_m: float = 0.0
    column_spacing_m: float = 0.0
    pattern: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  AGGREGATE PIER ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Column diameter:          {self.column_diameter_m:.3f} m",
            f"  Column spacing:           {self.column_spacing_m:.2f} m ({self.pattern})",
            f"  Area replacement ratio:   {self.area_replacement_ratio:.4f}",
            f"  Stress concentration (n): {self.stress_concentration_ratio:.1f}",
            "",
            f"  Composite modulus:        {self.composite_modulus_kPa:.0f} kPa",
            f"  Settlement reduction:     {self.settlement_reduction_factor:.3f}",
            "",
        ]
        if self.unreinforced_bearing_kPa > 0:
            lines.append(f"  Bearing capacity:")
            lines.append(f"    Unreinforced:           {self.unreinforced_bearing_kPa:.1f} kPa")
            lines.append(f"    Improved:               {self.improved_bearing_kPa:.1f} kPa")
            lines.append("")
        if self.settlement_unreinforced_mm > 0:
            lines.append(f"  Settlement:")
            lines.append(f"    Unreinforced:           {self.settlement_unreinforced_mm:.1f} mm")
            lines.append(f"    Improved:               {self.settlement_improved_mm:.1f} mm")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area_replacement_ratio": round(self.area_replacement_ratio, 4),
            "stress_concentration_ratio": round(self.stress_concentration_ratio, 2),
            "composite_modulus_kPa": round(self.composite_modulus_kPa, 1),
            "settlement_reduction_factor": round(self.settlement_reduction_factor, 4),
            "improved_bearing_kPa": round(self.improved_bearing_kPa, 1),
            "unreinforced_bearing_kPa": round(self.unreinforced_bearing_kPa, 1),
            "settlement_improved_mm": round(self.settlement_improved_mm, 1),
            "settlement_unreinforced_mm": round(self.settlement_unreinforced_mm, 1),
            "column_diameter_m": round(self.column_diameter_m, 3),
            "column_spacing_m": round(self.column_spacing_m, 2),
            "pattern": self.pattern,
        }


@dataclass
class WickDrainResult:
    """Results from PVD (wick drain) analysis.

    Attributes
    ----------
    drain_spacing_m : float
        Center-to-center drain spacing.
    pattern : str
        'triangular' or 'square'.
    equivalent_drain_diameter_m : float
        dw of the wick drain.
    influence_diameter_m : float
        de = influence zone diameter.
    spacing_ratio_n : float
        n = de / dw.
    F_n : float
        Barron/Hansbo drain function F(n).
    Ur_percent : float
        Radial degree of consolidation at analysis time.
    Uv_percent : float
        Vertical degree of consolidation at analysis time.
    U_total_percent : float
        Combined degree of consolidation.
    time_years : float
        Analysis time.
    ch_m2_per_year : float
        Horizontal coefficient of consolidation (input echo).
    cv_m2_per_year : float
        Vertical coefficient of consolidation (input echo).
    time_settlement_curve : list or None
        List of (time_years, U_total_percent) tuples.
    """
    drain_spacing_m: float = 0.0
    pattern: str = ""
    equivalent_drain_diameter_m: float = 0.0
    influence_diameter_m: float = 0.0
    spacing_ratio_n: float = 0.0
    F_n: float = 0.0
    Ur_percent: float = 0.0
    Uv_percent: float = 0.0
    U_total_percent: float = 0.0
    time_years: float = 0.0
    ch_m2_per_year: float = 0.0
    cv_m2_per_year: float = 0.0
    time_settlement_curve: Optional[List[Tuple[float, float]]] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  WICK DRAIN (PVD) ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Drain spacing:            {self.drain_spacing_m:.2f} m ({self.pattern})",
            f"  Drain diameter (dw):      {self.equivalent_drain_diameter_m:.4f} m",
            f"  Influence diameter (de):  {self.influence_diameter_m:.3f} m",
            f"  Spacing ratio (n=de/dw):  {self.spacing_ratio_n:.1f}",
            f"  Drain function F(n):      {self.F_n:.3f}",
            "",
            f"  At t = {self.time_years:.2f} years:",
            f"    Radial consolidation:   {self.Ur_percent:.1f}%",
            f"    Vertical consolidation: {self.Uv_percent:.1f}%",
            f"    Combined consolidation: {self.U_total_percent:.1f}%",
            "",
            f"  ch = {self.ch_m2_per_year:.2f} m²/yr",
            f"  cv = {self.cv_m2_per_year:.2f} m²/yr",
            "=" * 60,
        ]
        return "\n".join(lines)

    def plot_consolidation(self, ax=None, show=True, **kwargs):
        """Plot degree of consolidation vs time.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.time_settlement_curve:
            raise ValueError("No time-consolidation data available.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        times = [t for t, u in self.time_settlement_curve]
        u_vals = [u for t, u in self.time_settlement_curve]
        ax.plot(times, u_vals, 'b-o', markersize=3, linewidth=2, **kwargs)
        ax.axhline(y=self.U_total_percent, color='r', linestyle='--',
                   linewidth=1,
                   label=f'U at t={self.time_years:.2f} yr ({self.U_total_percent:.1f}%)')
        setup_engineering_plot(ax, "Wick Drain Consolidation",
                              "Time (years)", "Degree of Consolidation (%)")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "drain_spacing_m": round(self.drain_spacing_m, 2),
            "pattern": self.pattern,
            "equivalent_drain_diameter_m": round(self.equivalent_drain_diameter_m, 4),
            "influence_diameter_m": round(self.influence_diameter_m, 3),
            "spacing_ratio_n": round(self.spacing_ratio_n, 1),
            "F_n": round(self.F_n, 3),
            "Ur_percent": round(self.Ur_percent, 1),
            "Uv_percent": round(self.Uv_percent, 1),
            "U_total_percent": round(self.U_total_percent, 1),
            "time_years": round(self.time_years, 3),
            "ch_m2_per_year": round(self.ch_m2_per_year, 3),
            "cv_m2_per_year": round(self.cv_m2_per_year, 3),
        }
        if self.time_settlement_curve is not None:
            d["time_settlement_curve"] = [
                {"time_years": round(t, 3), "U_total_percent": round(u, 1)}
                for t, u in self.time_settlement_curve
            ]
        return d


@dataclass
class SurchargeResult:
    """Results from surcharge preloading analysis.

    Attributes
    ----------
    surcharge_kPa : float
        Applied surcharge pressure.
    target_U_percent : float
        Target degree of consolidation.
    time_to_target_years : float
        Time to reach target U.
    settlement_at_target_mm : float
        Settlement when target U is reached.
    settlement_ultimate_mm : float
        Ultimate settlement under surcharge.
    uses_wick_drains : bool
        Whether wick drains are included.
    wick_drain_result : WickDrainResult or None
        Drain details if drains are used.
    equivalent_sigma_p_kPa : float
        Induced preconsolidation pressure from surcharge.
    time_settlement_curve : list or None
        List of (time_years, settlement_mm) tuples.
    """
    surcharge_kPa: float = 0.0
    target_U_percent: float = 0.0
    time_to_target_years: float = 0.0
    settlement_at_target_mm: float = 0.0
    settlement_ultimate_mm: float = 0.0
    uses_wick_drains: bool = False
    wick_drain_result: Optional[WickDrainResult] = None
    equivalent_sigma_p_kPa: float = 0.0
    time_settlement_curve: Optional[List[Tuple[float, float]]] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SURCHARGE PRELOADING ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Surcharge applied:        {self.surcharge_kPa:.1f} kPa",
            f"  Target consolidation:     {self.target_U_percent:.0f}%",
            f"  Time to target:           {self.time_to_target_years:.2f} years",
            f"  Settlement at target:     {self.settlement_at_target_mm:.1f} mm",
            f"  Ultimate settlement:      {self.settlement_ultimate_mm:.1f} mm",
            f"  Uses wick drains:         {'Yes' if self.uses_wick_drains else 'No'}",
        ]
        if self.equivalent_sigma_p_kPa > 0:
            lines.append(f"  Equiv. sigma_p':          {self.equivalent_sigma_p_kPa:.1f} kPa")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def plot_consolidation(self, ax=None, show=True, **kwargs):
        """Plot settlement vs time.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.time_settlement_curve:
            raise ValueError("No time-settlement data available.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        times = [t for t, s in self.time_settlement_curve]
        settlements = [s for t, s in self.time_settlement_curve]
        ax.plot(times, settlements, 'b-o', markersize=3, linewidth=2, **kwargs)
        ax.axhline(y=self.settlement_ultimate_mm, color='gray', linestyle=':',
                   linewidth=1, label=f'Ultimate ({self.settlement_ultimate_mm:.0f} mm)')
        ax.axhline(y=self.settlement_at_target_mm, color='r', linestyle='--',
                   linewidth=1,
                   label=f'Target U={self.target_U_percent:.0f}% ({self.settlement_at_target_mm:.0f} mm)')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Surcharge Settlement vs Time",
                              "Time (years)", "Settlement (mm)")
        ax.legend(fontsize=8)
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "surcharge_kPa": round(self.surcharge_kPa, 1),
            "target_U_percent": round(self.target_U_percent, 1),
            "time_to_target_years": round(self.time_to_target_years, 3),
            "settlement_at_target_mm": round(self.settlement_at_target_mm, 1),
            "settlement_ultimate_mm": round(self.settlement_ultimate_mm, 1),
            "uses_wick_drains": self.uses_wick_drains,
            "equivalent_sigma_p_kPa": round(self.equivalent_sigma_p_kPa, 1),
        }
        if self.wick_drain_result is not None:
            d["wick_drain_details"] = self.wick_drain_result.to_dict()
        if self.time_settlement_curve is not None:
            d["time_settlement_curve"] = [
                {"time_years": round(t, 3), "settlement_mm": round(s, 1)}
                for t, s in self.time_settlement_curve
            ]
        return d


@dataclass
class VibroResult:
    """Results from vibro-compaction feasibility assessment.

    Attributes
    ----------
    is_feasible : bool
        Whether vibro-compaction is feasible.
    fines_content_percent : float
        Fines content of the target soil.
    initial_N_spt : float
        Current SPT N value.
    target_N_spt : float
        Desired SPT N value after treatment.
    recommended_spacing_m : float
        Estimated probe spacing (0 if infeasible).
    probe_pattern : str
        'triangular' or 'square'.
    reasons : list of str
        Go/no-go reasons and notes.
    """
    is_feasible: bool = False
    fines_content_percent: float = 0.0
    initial_N_spt: float = 0.0
    target_N_spt: float = 0.0
    recommended_spacing_m: float = 0.0
    probe_pattern: str = ""
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "FEASIBLE" if self.is_feasible else "NOT FEASIBLE"
        lines = [
            "=" * 60,
            "  VIBRO-COMPACTION FEASIBILITY RESULTS",
            "=" * 60,
            "",
            f"  Assessment:               {status}",
            f"  Fines content:            {self.fines_content_percent:.1f}%",
            f"  Initial N_spt:            {self.initial_N_spt:.0f}",
            f"  Target N_spt:             {self.target_N_spt:.0f}",
        ]
        if self.is_feasible and self.recommended_spacing_m > 0:
            lines.append(f"  Recommended spacing:      {self.recommended_spacing_m:.2f} m ({self.probe_pattern})")
        lines.append("")
        if self.reasons:
            lines.append("  Notes:")
            for r in self.reasons:
                lines.append(f"    - {r}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "fines_content_percent": round(self.fines_content_percent, 1),
            "initial_N_spt": round(self.initial_N_spt, 0),
            "target_N_spt": round(self.target_N_spt, 0),
            "recommended_spacing_m": round(self.recommended_spacing_m, 2),
            "probe_pattern": self.probe_pattern,
            "reasons": self.reasons,
        }


@dataclass
class FeasibilityResult:
    """Results from ground improvement feasibility evaluation.

    Attributes
    ----------
    applicable_methods : list of str
        Methods that are feasible for this site.
    not_applicable : list of dict
        Methods that are not feasible, with reasons.
        Each dict has 'method' and 'reason' keys.
    recommendations : list of str
        Prioritized recommendations.
    preliminary_sizing : dict
        Preliminary sizing estimates for applicable methods.
    soil_description : str
        Brief description of the soil conditions.
    design_problem : str
        Brief description of the design problem.
    """
    applicable_methods: List[str] = field(default_factory=list)
    not_applicable: List[Dict[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    preliminary_sizing: Dict[str, Any] = field(default_factory=dict)
    soil_description: str = ""
    design_problem: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  GROUND IMPROVEMENT FEASIBILITY EVALUATION",
            "=" * 60,
            "",
            f"  Soil conditions: {self.soil_description}",
            f"  Design problem:  {self.design_problem}",
            "",
        ]
        if self.applicable_methods:
            lines.append("  APPLICABLE METHODS:")
            for m in self.applicable_methods:
                lines.append(f"    + {m}")
            lines.append("")
        if self.not_applicable:
            lines.append("  NOT APPLICABLE:")
            for item in self.not_applicable:
                lines.append(f"    - {item['method']}: {item['reason']}")
            lines.append("")
        if self.recommendations:
            lines.append("  RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
            lines.append("")
        if self.preliminary_sizing:
            lines.append("  PRELIMINARY SIZING:")
            for method, sizing in self.preliminary_sizing.items():
                lines.append(f"    {method}:")
                if isinstance(sizing, dict):
                    for k, v in sizing.items():
                        lines.append(f"      {k}: {v}")
                else:
                    lines.append(f"      {sizing}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applicable_methods": self.applicable_methods,
            "not_applicable": self.not_applicable,
            "recommendations": self.recommendations,
            "preliminary_sizing": self.preliminary_sizing,
            "soil_description": self.soil_description,
            "design_problem": self.design_problem,
        }
