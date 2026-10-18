"""
Results container for axial pile capacity analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

#: Side-resistance method labels as GEC-12 names them.
_METHOD_NAMES = {"nordlund": "Nordlund", "tomlinson": "Tomlinson α",
                 "beta": "β"}


@dataclass
class AxialPileResult:
    """Results from an axial pile capacity analysis.

    Attributes
    ----------
    Q_ultimate : float
        Ultimate axial capacity (kN) = Q_skin + Q_tip - plug_weight.
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
    warnings : list of str
        Things the user must be told about the result, e.g. a toe friction
        angle outside the chart the toe limit is read from.
    choices : list of dict
        What the analysis chose (GM-115, ``geotech_common.choices``): the
        side and toe resistance methods ``method="auto"`` resolved to by
        soil type, an open pipe's plugged or unplugged case, Nordlund's
        delta/phi, the limits applied and the judgments GEC-12 leaves to
        the engineer.
    plug_weight : float
        Soil plug weight Wp (kN) subtracted when an open pipe's unplugged
        case governs (GEC-12 Eq. 7-89); 0 otherwise.
    toe_details : dict
        The toe resistance's factors: for a Nordlund toe alpha_t, N'q, D/b,
        sigma'p, q_L and which limit governed; the unit toe resistance
        q_p and the toe area for every method.
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
    warnings: List[str] = field(default_factory=list)
    choices: List[Dict[str, Any]] = field(default_factory=list)
    plug_weight: float = 0.0
    toe_details: Dict[str, Any] = field(default_factory=dict)

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
        ]
        if self.plug_weight:
            lines.append(f"  Soil plug (Wp):     -{self.plug_weight:>10,.1f} kN")
        lines += [
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

        from geotech_common.choices import choices_lines
        lines.extend(choices_lines(self.choices))

        if self.warnings:
            lines.extend(["", "  Warnings:"])
            lines.extend(f"    - {w}" for w in self.warnings)

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
        if self.plug_weight:
            d["plug_weight_kN"] = round(self.plug_weight, 1)
        if self.toe_details:
            d["toe_details"] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.toe_details.items()}
        if self.warnings:
            d["warnings"] = list(self.warnings)
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d

    def get_tables(self) -> list:
        """APILE-style tables: side resistance per layer, the load in the
        pile at the ultimate load (the load-transfer curve), and the
        capacity summary (CSV only; the calc package prints it in its
        Capacity section). Values are this result's own, in its units.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        layers = self.layer_breakdown or []
        if layers:
            rows, cum = [], 0.0
            for lb in layers:
                cum += lb["skin_friction_kN"]
                rows.append([
                    lb["depth_top_m"], lb["depth_bottom_m"], lb["soil_type"],
                    _METHOD_NAMES.get(lb["method"], lb["method"]),
                    lb["sigma_v_kPa"], lb.get("unit_side_resistance_kPa"),
                    lb["skin_friction_kN"], cum,
                ])
            note = ("Side resistance of each soil layer along the shaft "
                    "(GEC-12 static methods: Nordlund for cohesionless, "
                    "Tomlinson α for cohesive, or β throughout); "
                    "σ'v at the segment mid-depth; f_s = Q_s / "
                    "(perimeter × segment length)")
            inside = self.Q_skin - cum
            has_inside = abs(inside) > 1e-6 * max(abs(self.Q_skin), 1.0)
            if has_inside:
                note += (f"; Q_s = {self.Q_skin:.1f} kN also includes "
                         f"{inside:.1f} kN of side resistance inside the "
                         "open pipe (unplugged case governs)")
            if any("eq_7_90_factor" in lb for lb in layers):
                note += ("; the cohesive layers carry 0.8 Ca (Tomlinson's "
                         "Eq. 7-90 for a small open pipe in clay)")
            tables.append(CalcTable(
                title="Side resistance per layer",
                columns=[("Top", "m", ".2f"), ("Bottom", "m", ".2f"),
                         "Soil type", "Method",
                         ("σ'v at mid-depth", "kPa", ".1f"),
                         ("Unit side resistance f_s", "kPa", ".2f"),
                         ("Side resistance Q_s", "kN", ".1f"),
                         ("Cumulative Q_s", "kN", ".1f")],
                rows=rows, note=note, name="side_resistance",
            ))

            # Load in the pile with the ultimate load at the head: Q_ult
            # less the side resistance above each depth (APILE's load
            # distribution at failure).
            head = layers[0]["depth_top_m"]
            load_rows = [[head, 0.0, self.Q_ultimate]]
            cum = 0.0
            for lb in layers:
                cum += lb["skin_friction_kN"]
                load_rows.append([lb["depth_bottom_m"], cum,
                                  self.Q_ultimate - cum])
            load_note = (f"Q(z) = Q_ult - Σ Q_s above z, with Q_ult = "
                         f"{self.Q_ultimate:.1f} kN at the head; ")
            load_note += (
                f"at the toe it is the toe resistance Q_t = "
                f"{self.Q_tip:.1f} kN plus the {inside:.1f} kN of side "
                "resistance inside the open pipe" if has_inside else
                f"at the toe it is the toe resistance Q_t = "
                f"{self.Q_tip:.1f} kN")
            if self.plug_weight:
                load_note += (f", less the soil plug weight Wp = "
                              f"{self.plug_weight:.1f} kN (Eq. 7-89)")
            tables.append(CalcTable(
                title="Load transfer at the ultimate load",
                columns=[("Depth z", "m", ".2f"),
                         ("Side resistance above z", "kN", ".1f"),
                         ("Axial load in pile Q", "kN", ".1f")],
                rows=load_rows, note=load_note, name="load_transfer",
            ))

        summary = [
            ["Side resistance", "Q_s", self.Q_skin, "kN"],
            ["Toe resistance", "Q_t", self.Q_tip, "kN"],
        ]
        if self.plug_weight:
            summary.append(["Soil plug weight (subtracted)", "W_p",
                            self.plug_weight, "kN"])
        summary += [
            ["Ultimate capacity", "Q_ult", self.Q_ultimate, "kN"],
            ["Factor of safety", "FS", self.factor_of_safety, ""],
            ["Allowable capacity", "Q_all", self.Q_allowable, "kN"],
            ["Effective vertical stress at the toe", "σ'v,toe",
             self.sigma_v_tip, "kPa"],
        ]
        if self.Q_uplift is not None:
            summary.append(["Uplift capacity", "Q_up", self.Q_uplift, "kN"])
        tables.append(CalcTable(
            title="Capacity summary",
            columns=["Quantity", "Symbol", "Value", "Unit"],
            rows=summary,
            note=f"Pile {self.pile_name}, embedded length "
                 f"{self.pile_length:g} m, method '{self.method}'",
            name="capacity_summary", printed=False,
        ))
        return tables

    def plot_load_transfer(self, ax=None, show=True, **kwargs):
        """Plot cumulative load transfer vs depth.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        show : bool
            Whether to call plt.show(). Default True.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.layer_breakdown:
            raise ValueError("No layer breakdown data available.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))

        depths = [0.0]
        loads = [0.0]
        cumulative = 0.0
        for layer in self.layer_breakdown:
            depths.append(layer['depth_top_m'])
            loads.append(cumulative)
            cumulative += layer['skin_friction_kN']
            depths.append(layer['depth_bottom_m'])
            loads.append(cumulative)
        # Tip resistance
        depths.append(self.pile_length)
        loads.append(cumulative + self.Q_tip)

        ax.plot(loads, depths, 'b-', linewidth=2, **kwargs)
        ax.axhline(y=self.pile_length, color='brown', linestyle='--',
                   linewidth=0.8, label='Pile Tip')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Axial Load Transfer",
                              "Cumulative Capacity (kN)", "Depth (m)")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax
