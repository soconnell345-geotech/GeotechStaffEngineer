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
    base_diameter : float
        Diameter (m) of the base the end bearing acts on: the bell's, the
        rock socket's with the tip in rock (GEC-10 Example 10-4, GM-221),
        else the shaft's.
    base_area : float
        Base area (m²), pi/4 times base_diameter squared.
    method : str
        Analysis method description.
    layer_breakdown : list of dict
        Per-layer side resistance breakdown.
    sigma_v_tip : float
        Effective stress at shaft tip (kPa).
    base_steps : dict
        Each step of the unit base resistance (GM-244): the tip's soil
        type, the method, its factors and ``qb`` (kPa).
    choices : list of dict
        What the analysis chose for the user (GM-115,
        ``geotech_common.choices``), e.g. the depth beta of a cohesionless
        side layer given no N60 (GM-148).
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
    base_diameter: float = 0.0
    base_area: float = 0.0
    method: str = "GEC-10"
    layer_breakdown: Optional[List[Dict[str, Any]]] = None
    sigma_v_tip: float = 0.0
    base_steps: Optional[Dict[str, Any]] = None
    choices: List[Dict[str, Any]] = field(default_factory=list)

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

        from geotech_common.choices import choices_lines
        lines.extend(choices_lines(self.choices))
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
            "base_diameter_m": round(self.base_diameter, 4),
            "base_area_m2": round(self.base_area, 4),
            "method": self.method,
            "sigma_v_tip_kPa": round(self.sigma_v_tip, 1),
        }
        if self.layer_breakdown:
            d["side_resistance_by_layer"] = self.layer_breakdown
            d["side_resistance_note"] = (
                "Per-layer side resistance. 'method' gives the mobilized "
                "coefficient (alpha=... for cohesive, beta=... for "
                "cohesionless, rock socket); 'fs_kPa' is the unit side "
                "resistance. effective_top_m/effective_bottom_m reflect the "
                "zones that carry no side resistance: the top zone or the "
                "casing (unless top_exclusion is given, the top 1.5 m in "
                "cohesive soil and none in cohesionless soil or rock) and, "
                "where bottom_exclusion is given, that much above the base "
                "in cohesive soil. Under the GEC-10 (2018) beta, "
                "'beta_steps' gives each cohesionless layer's phi', "
                "sigma'p, OCR, Ko, Kp, K/Ko and beta; 'alpha_steps' each "
                "cohesive layer's su, the su the alpha used (su(CIUC) under "
                "GEC-10's Eq. 10-18), su/pa and alpha."
            )
        if self.base_steps:
            d["base_steps"] = dict(self.base_steps)
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d

    def get_tables(self) -> list:
        """SHAFT-style tables: side resistance per layer (with the GEC-10
        exclusion zones), the load in the shaft at the ultimate load, and
        the capacity summary (CSV only; the calc package prints it in its
        Capacity section). Values are this result's own, in its units: the
        per-layer values are rounded to 0.1 as the module reports them.

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
                cum += lb["side_resistance_kN"]
                rows.append([
                    lb["depth_top_m"], lb["depth_bottom_m"],
                    lb.get("effective_top_m"), lb.get("effective_bottom_m"),
                    lb["soil_type"], lb["method"], lb.get("sigma_v_kPa"),
                    lb.get("fs_kPa"), lb["side_resistance_kN"], cum])
            tables.append(CalcTable(
                title="Side resistance per layer",
                columns=[("Top", "m", ".2f"), ("Bottom", "m", ".2f"),
                         ("Effective top", "m", ".2f"),
                         ("Effective bottom", "m", ".2f"), "Soil type",
                         "Method", ("σ'v", "kPa", ".1f"),
                         ("Unit side resistance f_s", "kPa", ".1f"),
                         ("Side resistance Q_s", "kN", ".1f"),
                         ("Cumulative Q_s", "kN", ".1f")],
                rows=rows,
                note=("GEC-10 side resistance: α method in cohesive "
                      "layers, β method in cohesionless layers, rock "
                      "socket in rock; 'Effective' top and bottom leave out "
                      "the zones that carry none (the top zone or the "
                      "casing; bottom_exclusion above the base in cohesive "
                      "layers, where given)"),
                name="side_resistance",
            ))

            head = layers[0]["depth_top_m"]
            load_rows = [[head, 0.0, self.Q_ultimate]]
            cum = 0.0
            for lb in layers:
                cum += lb["side_resistance_kN"]
                load_rows.append([lb["depth_bottom_m"], cum,
                                  self.Q_ultimate - cum])
            tables.append(CalcTable(
                title="Load transfer at the ultimate load",
                columns=[("Depth z", "m", ".2f"),
                         ("Side resistance above z", "kN", ".1f"),
                         ("Axial load in shaft Q", "kN", ".1f")],
                rows=load_rows,
                note=(f"Q(z) = Q_ult - Σ Q_s above z, with Q_ult = "
                      f"{self.Q_ultimate:.1f} kN at the head; at the base "
                      f"it is the base resistance Q_b = {self.Q_tip:.1f} kN "
                      "(within the 0.1 kN rounding of the layer values)"),
                name="load_transfer",
            ))

        summary = [
            ["Side resistance, cohesive layers", "Q_s,clay",
             self.Q_side_clay, "kN"],
            ["Side resistance, cohesionless layers", "Q_s,sand",
             self.Q_side_sand, "kN"],
            ["Side resistance, rock socket", "Q_s,rock",
             self.Q_side_rock, "kN"],
            ["Side resistance", "Q_s", self.Q_skin, "kN"],
            ["Base resistance", "Q_b", self.Q_tip, "kN"],
            ["Ultimate capacity", "Q_ult", self.Q_ultimate, "kN"],
            ["Factor of safety", "FS", self.factor_of_safety, ""],
            ["Allowable capacity", "Q_all", self.Q_allowable, "kN"],
            ["Effective vertical stress at the base", "σ'v,base",
             self.sigma_v_tip, "kPa"],
        ]
        tables.append(CalcTable(
            title="Capacity summary",
            columns=["Quantity", "Symbol", "Value", "Unit"], rows=summary,
            note=(f"Shaft D = {self.shaft_diameter:g} m, L = "
                  f"{self.shaft_length:g} m, {self.method}"),
            name="capacity_summary", printed=False,
        ))
        return tables

    def plot_load_transfer(self, ax=None, show=True, **kwargs):
        """Plot cumulative side resistance vs depth.

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
            cumulative += layer['side_resistance_kN']
            depths.append(layer['depth_bottom_m'])
            loads.append(cumulative)
        # Base resistance
        depths.append(self.shaft_length)
        loads.append(cumulative + self.Q_tip)

        ax.plot(loads, depths, 'b-', linewidth=2, **kwargs)
        ax.axhline(y=self.shaft_length, color='brown', linestyle='--',
                   linewidth=0.8, label='Shaft Base')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Drilled Shaft Load Transfer",
                              "Cumulative Capacity (kN)", "Depth (m)")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax
