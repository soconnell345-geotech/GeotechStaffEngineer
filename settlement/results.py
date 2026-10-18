"""
Results container for settlement analysis.

Stores computed settlements and provides summary output and
optional time-settlement plotting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np


@dataclass
class SettlementResult:
    """Results from a settlement analysis.

    Attributes
    ----------
    immediate : float
        Immediate (elastic) settlement (m).
    consolidation : float
        Primary consolidation settlement (m).
    secondary : float
        Secondary compression settlement (m).
    total : float
        Total settlement = immediate + consolidation + secondary (m).
    consolidation_layers : list of dict, optional
        Per-layer consolidation settlement breakdown.
    time_settlement_curve : list of tuple, optional
        (time_years, settlement_m) pairs for plotting; settlement is
        immediate + primary consolidation at that time (Terzaghi 1-D).
    cv : float, optional
        Coefficient of consolidation used for the time curve (m^2/year).
    Hdr : float, optional
        Drainage path length used for the time curve (m).
    t50_years, t90_years : float, optional
        Times to 50 % and 90 % average degree of consolidation (years).
    choices : list of dict
        What the analysis chose (GM-115, ``geotech_common.choices``): the
        immediate-settlement method, the stress distribution under the
        footing, the drainage path and the start of secondary compression.
    warnings : list of str
        What the user must be told, e.g. that secondary settlement starts
        at an assumed t1 of 1 year.

    The terms the analysis used, as it used them (GM-143; the calc
    package prints these rather than working them out again):

    B, L : float
        The footing's width and length (m); L is None for a strip taken
        as infinitely long (no L given).
    footing_shape : str
    Iw : float, optional
        The elastic influence factor used (elastic method).
    schmertmann_terms : dict, optional
        Schmertmann's C1, C2, C3, z_peak, z_max, Iz_0, sigma_vp, Iz_peak,
        sum_Iz_dz_over_Es, strip and Se (``immediate.schmertmann_terms``).
    consolidation_terms : list of dict, optional
        Per consolidation layer, unrounded: delta_sigma, sigma_final (kPa),
        case and Sc (m) (``consolidation.consolidation_layer_terms``).
    secondary_terms : dict, optional
        t1_years, t2_years, t1_from, C_alpha, H_total, e0_given, layers
        (thickness, e0, Ss) and Ss (m).
    """
    immediate: float = 0.0
    consolidation: float = 0.0
    secondary: float = 0.0
    total: float = 0.0
    immediate_method: str = ""
    consolidation_method: str = ""
    stress_method: str = ""
    consolidation_layers: Optional[List[Dict[str, Any]]] = None
    time_settlement_curve: Optional[List[Tuple[float, float]]] = None
    cv: Optional[float] = None
    Hdr: Optional[float] = None
    t50_years: Optional[float] = None
    t90_years: Optional[float] = None
    choices: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    B: Optional[float] = None
    L: Optional[float] = None
    footing_shape: str = ""
    Iw: Optional[float] = None
    schmertmann_terms: Optional[Dict[str, Any]] = None
    consolidation_terms: Optional[List[Dict[str, Any]]] = None
    secondary_terms: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        """Return a formatted summary string.

        Returns
        -------
        str
            Human-readable summary of settlement results.
        """
        lines = [
            "=" * 60,
            "  SETTLEMENT ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Immediate settlement:     {self.immediate*1000:8.1f} mm"
            f"  ({self._pct(self.immediate)}%)",
            f"  Consolidation settlement: {self.consolidation*1000:8.1f} mm"
            f"  ({self._pct(self.consolidation)}%)",
            f"  Secondary settlement:     {self.secondary*1000:8.1f} mm"
            f"  ({self._pct(self.secondary)}%)",
            f"  {'-'*44}",
            f"  TOTAL settlement:         {self.total*1000:8.1f} mm",
        ]

        if self.immediate_method:
            lines.append(f"\n  Immediate method: {self.immediate_method}")
        if self.stress_method:
            lines.append(f"  Stress distribution: {self.stress_method}")

        if self.consolidation_layers:
            lines.append("\n  Consolidation Layer Breakdown:")
            for i, layer in enumerate(self.consolidation_layers):
                lines.append(
                    f"    Layer {i+1}: Sc={layer['settlement_mm']:.1f} mm, "
                    f"delta_sigma={layer['delta_sigma_kPa']:.1f} kPa, "
                    f"OCR={layer.get('OCR', 1.0):.1f}"
                )

        from geotech_common.choices import choices_lines
        lines.extend(choices_lines(self.choices))
        if self.warnings:
            lines.extend(["", "  Warnings:"])
            lines.extend(f"    - {w}" for w in self.warnings)
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _pct(self, component: float) -> str:
        if self.total > 0:
            return f"{100 * component / self.total:.0f}"
        return "0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for LLM agent consumption.

        Returns
        -------
        dict
            All result fields as a flat dictionary.
        """
        d = {
            "immediate_mm": round(self.immediate * 1000, 2),
            "consolidation_mm": round(self.consolidation * 1000, 2),
            "secondary_mm": round(self.secondary * 1000, 2),
            "total_mm": round(self.total * 1000, 2),
            "immediate_method": self.immediate_method,
            "stress_method": self.stress_method,
        }
        if self.consolidation_layers:
            d["consolidation_layers"] = self.consolidation_layers
        # The time curve (live smoke G6, SE-4: computed, documented, and
        # dropped here, so t90 was worked out by hand).
        if self.time_settlement_curve is not None:
            if self.cv is not None:
                d["cv_m2_per_year"] = self.cv
            if self.Hdr is not None:
                d["Hdr_m"] = round(self.Hdr, 4)
            if self.t50_years is not None:
                d["t50_years"] = round(self.t50_years, 4)
            if self.t90_years is not None:
                d["t90_years"] = round(self.t90_years, 4)
            points = []
            for t, s in self.time_settlement_curve:
                p = {"time_years": round(t, 4),
                     "settlement_mm": round(s * 1000, 2)}
                if self.consolidation > 0:
                    u = (s - self.immediate) / self.consolidation * 100.0
                    p["U_percent"] = round(min(max(u, 0.0), 100.0), 1)
                points.append(p)
            d["time_settlement_curve"] = points
            d["time_settlement_basis"] = (
                "settlement_mm = immediate + U(t) x primary consolidation "
                "(Terzaghi 1-D average degree of consolidation, time from "
                "load application; secondary compression not included)")
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    def time_curve_rows(self) -> List[list]:
        """``[time_years, U_percent, settlement_m]`` for each point of the
        time curve; U as :meth:`to_dict` gives it, (S - S_i) / S_c."""
        rows = []
        for t, s in self.time_settlement_curve or []:
            u = None
            if self.consolidation > 0:
                u = min(max((s - self.immediate) / self.consolidation
                            * 100.0, 0.0), 100.0)
            rows.append([t, u, s])
        return rows

    def get_tables(self) -> list:
        """Settle3-style tables: consolidation settlement per layer, the
        settlement against time (when a cv was given), and the settlement
        summary (CSV only; the calc package prints it in its summary).
        Values are this result's own, in its units (the per-layer
        settlement is in mm, as the module reports it).

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        if self.consolidation_layers:
            tables.append(CalcTable(
                title="Consolidation settlement per layer",
                columns=["Layer", ("Depth to centre z", "m", ".2f"),
                         ("Thickness H", "m", ".2f"), "Description",
                         ("Stress increase Δσ", "kPa", ".1f"),
                         "OCR", ("Settlement S_c", "mm", ".2f")],
                rows=[[i, lb["depth_m"], lb["thickness_m"],
                       lb.get("description", ""), lb["delta_sigma_kPa"],
                       lb["OCR"], lb["settlement_mm"]]
                      for i, lb in enumerate(self.consolidation_layers, 1)],
                note=(f"Primary consolidation by the "
                      f"{self.consolidation_method or 'e-log(p)'} method; "
                      f"Δσ at each layer centre by the "
                      f"{self.stress_method} method"),
                name="consolidation_per_layer",
            ))
        if self.time_settlement_curve:
            note = ("Terzaghi 1-D consolidation: S = S_i + U(t) S_c, time "
                    "from load application; secondary compression not "
                    "included")
            if self.cv is not None and self.Hdr is not None:
                note += (f"; cv = {self.cv:g} m²/year, H_dr = "
                         f"{self.Hdr:g} m")
            tables.append(CalcTable(
                title="Settlement against time",
                columns=[("Time t", "years"),
                         ("Average degree of consolidation U", "%", ".1f"),
                         ("Settlement S", "m")],
                rows=self.time_curve_rows(), note=note,
                name="time_settlement",
            ))
        summary = [
            ["Immediate settlement", "S_i", self.immediate, "m"],
            ["Primary consolidation settlement", "S_c", self.consolidation,
             "m"],
            ["Secondary compression settlement", "S_s", self.secondary, "m"],
            ["Total settlement", "S", self.total, "m"],
        ]
        if self.t50_years is not None:
            summary.append(["Time to 50 % consolidation", "t50",
                            self.t50_years, "years"])
        if self.t90_years is not None:
            summary.append(["Time to 90 % consolidation", "t90",
                            self.t90_years, "years"])
        tables.append(CalcTable(
            title="Settlement summary",
            columns=["Quantity", "Symbol", "Value", "Unit"], rows=summary,
            note=(f"Immediate: {self.immediate_method or 'none'}; stress "
                  f"distribution: {self.stress_method or 'none'}"),
            name="settlement_summary", printed=False,
        ))
        return tables

    def plot_time_settlement(self, ax=None):
        """Plot settlement vs time curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if self.time_settlement_curve is None:
            raise ValueError("No time-settlement data available. "
                             "Run analysis with time_rate parameters.")

        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        times = [t for t, s in self.time_settlement_curve]
        settlements = [s * 1000 for t, s in self.time_settlement_curve]

        ax.plot(times, settlements, 'b-', linewidth=2)
        ax.invert_yaxis()
        setup_engineering_plot(
            ax,
            title="Settlement vs Time",
            xlabel="Time (years)",
            ylabel="Settlement (mm)"
        )
        return ax
