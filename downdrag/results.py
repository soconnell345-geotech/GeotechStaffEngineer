"""
Results container for downdrag analysis.

Provides a clean interface to access analysis results with summary()
for human-readable output and to_dict() for JSON serialization.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DowndragResult:
    """Container for downdrag analysis results.

    Attributes
    ----------
    neutral_plane_depth : float
        Depth of the neutral plane from pile head (m).
    dragload : float
        Negative skin friction above the neutral plane (kN).
    max_pile_load : float
        Maximum axial load in pile at neutral plane (kN).
        Equals Q_dead + dragload + pile_weight_to_np.
    Q_dead : float
        Applied dead load at pile head (kN).
    pile_weight_to_np : float
        Pile self-weight from head to neutral plane in the load curve (kN):
        zero unless ``include_pile_weight``.
    positive_skin_friction : float
        Positive shaft resistance below neutral plane (kN).
    toe_resistance : float
        Toe bearing resistance (kN).
    total_resistance : float
        Total resistance below NP = positive_skin + toe (kN).
    pile_settlement : float
        Pile head settlement (m): the settlement at the neutral plane plus
        the elastic compression of the pile above it (UFC 3-220-20 Fig
        6-19, §6-5.8 item 6).
    elastic_shortening : float
        Elastic shortening of pile above NP (m).
    toe_settlement : float
        Settlement of the soil below the pile toe (m) at the headline toe
        force: the settling layers under the fill / drawdown down to the
        bottom of the soil profile, plus the toe load's compression down to
        the ``influence_depth_criterion`` depth (``below_toe``).
    soil_settlement_at_np : float
        Soil settlement at the neutral plane depth (m), from the original
        ground.
    z : numpy.ndarray
        Depth array along pile (m).
    axial_load : numpy.ndarray
        Axial load distribution Q(z) along pile (kN).
    soil_settlement_profile : numpy.ndarray
        Soil settlement at each depth (m).
    unit_skin_friction : numpy.ndarray
        Unit skin friction fs(z) along pile (kPa).
    structural_ok : bool or None
        True if structural check passes. None if not checked.
    structural_demand : float or None
        LRFD factored demand (kN) per UFC Eq 6-80:
        1.25*Q_dead + 1.10*(Q_np - Q_dead). None if not checked.
    geotechnical_ok : bool or None
        True if geotechnical check passes. None if not checked.
    settlement_ok : bool or None
        True if settlement is within allowable. None if not checked.
    pile_length : float
        Pile length (m).
    pile_diameter : float
        Pile diameter (m).
    neutral_plane_method : str
        The ``neutral_plane_method`` value of the method that gave the
        headline: ``"force_equilibrium"`` (the load and resistance curves
        intersect), ``"settlement_compatibility"`` (pile and soil settle
        equally, toe force compatible with the toe penetration; what
        ``auto`` uses when the curves do not intersect),
        ``"bearing_layer_top"``, ``"pile_toe"``, ``"endo"`` (chosen by the
        caller) or ``"none"`` (the dead load exceeds the total
        resistance).
    neutral_plane_basis : str
        How the headline neutral plane was located, with its source.
    toe_force_mobilized : float or None
        Toe force carried at the neutral plane (kN): the full toe
        resistance where the curves intersect with the toe fully
        mobilized, less when it is not.
    warnings : list of str
        Anything the engineer must know about the result.
    neutral_plane_requested : str
        The ``neutral_plane_method`` asked for (``"auto"`` by default).
    toe_mobilization : float
        Fraction of the toe resistance used where the curves intersect.
    skin_friction_stress : str
        ``"final"`` (default) or ``"initial"`` effective stress for beta
        friction and the toe in sand.
    include_pile_weight : bool
        False (default): the pile's own weight is left out of the load
        curve, as UFC 3-220-20 Fig 6-26 and FHWA GEC-12 §7.3.6.1 do. True:
        Q_np and the drag force carry the weight of the pile above the
        neutral plane.
    neutral_plane_comparison : list of dict
        Every method on the same inputs, one row each: name (as the cited
        reference words it), method, toe_mobilization, source,
        assumptions, applies, used, reason, and (when it applies) the
        neutral plane depth, Q_np, drag load, drag force, toe force, soil
        and pile settlement at the plane, elastic compression above it and
        the pile head settlement.
    methods_range : str
        One sentence: the spread of the neutral plane and the drag force
        over the methods that apply.
    headline_method : str
        Name of the row that gave the headline numbers.
    influence_depth_criterion : str
        ``"bearing_stress"`` (UFC 3-220-20 §6-5.8.4.3, default) or
        ``"effective_stress"`` (FHWA GEC-12 §7.3.5.3): where the toe load
        stops being counted below the toe.
    below_toe : dict
        How deep the settlement below the toe was counted and why it
        stopped: the criterion, the toe force and the equivalent footing's
        bearing stress, ``toe_load_depth_below_toe_m``,
        ``toe_load_counted_to_m`` and ``toe_load_stopped_by``
        (``"criterion"``, ``"profile_end"`` or ``"no_toe_load"``),
        ``fill_counted_to_m`` (the bottom of the soil profile; None with
        no fill or drawdown), the settlement under the fill / drawdown and
        under the toe load (mm), ``depth_m`` (the deepest point counted),
        ``other_criterion`` (the same with the other criterion) and
        ``basis`` (one paragraph saying all of it).
    """
    neutral_plane_depth: float
    dragload: float
    max_pile_load: float
    Q_dead: float
    pile_weight_to_np: float
    positive_skin_friction: float
    toe_resistance: float
    total_resistance: float
    pile_settlement: float
    elastic_shortening: float
    toe_settlement: float
    soil_settlement_at_np: float
    z: np.ndarray
    axial_load: np.ndarray
    soil_settlement_profile: np.ndarray
    unit_skin_friction: np.ndarray
    structural_ok: Optional[bool]
    structural_demand: Optional[float]
    geotechnical_ok: Optional[bool]
    settlement_ok: Optional[bool]
    pile_length: float
    pile_diameter: float
    neutral_plane_method: str = "force_equilibrium"
    neutral_plane_basis: str = ""
    toe_force_mobilized: Optional[float] = None
    warnings: list = field(default_factory=list)
    neutral_plane_requested: str = "auto"
    toe_mobilization: float = 1.0
    skin_friction_stress: str = "final"
    neutral_plane_comparison: list = field(default_factory=list)
    methods_range: str = ""
    headline_method: str = ""
    include_pile_weight: bool = False
    influence_depth_criterion: str = "bearing_stress"
    below_toe: dict = field(default_factory=dict)

    @property
    def drag_force(self) -> float:
        """Drag force DF = Q_np - Q_dead (kN), as UFC 3-220-20 Eq 6-80 and
        FHWA GEC-12 Eq 7-70 define it (it also carries the pile weight
        above the neutral plane when ``include_pile_weight``)."""
        return self.max_pile_load - self.Q_dead

    @property
    def pile_weight_text(self) -> str:
        """Whether the load curve carries the pile's own weight, in plain
        words naming the references."""
        if self.include_pile_weight:
            return ("included in the load curve (include_pile_weight=True); "
                    "UFC 3-220-20 Fig 6-26 and FHWA GEC-12 §7.3.6.1 leave it "
                    "out")
        return ("left out of the load curve, as UFC 3-220-20 Fig 6-26 and "
                "FHWA GEC-12 §7.3.6.1 do")

    @property
    def pile_weight_basis(self) -> str:
        """``pile_weight_text`` as a sentence: "Pile's own weight left out
        of the load curve, as ..."."""
        return "Pile's own weight " + self.pile_weight_text

    def drag_force_note(self) -> str:
        """The sentence defining the drag load, the drag force and the pile
        settlement under every methods table."""
        weight = ("here it includes the pile weight above the plane"
                  if self.include_pile_weight else
                  "the pile's own weight is left out, as UFC Fig 6-26 and "
                  "GEC-12 §7.3.6.1 do")
        return ("Drag load = negative skin friction above the neutral plane "
                "(UFC 3-220-20 §6-7.4 step 4). Drag force = Q_np - Q_dead "
                f"(UFC Eq 6-80, GEC-12 Eq 7-70); {weight}. Pile settlement "
                "= settlement at the neutral plane + elastic compression "
                "above it (UFC Fig 6-19).")

    def below_toe_rounded(self) -> dict:
        """``below_toe`` rounded for display / JSON."""
        def rnd(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = rnd(v)
                elif isinstance(v, float):
                    out[k] = round(v, 2 if k.endswith(("_mm", "_kN", "_kPa"))
                                   else 3)
                else:
                    out[k] = v
            return out
        return rnd(self.below_toe)

    def comparison_rows(self) -> list:
        """The comparison rows rounded for display / JSON."""
        out = []
        for r in self.neutral_plane_comparison:
            row = dict(r)
            for key, nd in (("neutral_plane_depth_m", 3),
                            ("max_pile_load_kN", 1), ("dragload_kN", 1),
                            ("drag_force_kN", 1), ("toe_force_kN", 1),
                            ("soil_settlement_at_np_mm", 2),
                            ("pile_settlement_at_np_mm", 2),
                            ("elastic_compression_above_np_mm", 2),
                            ("pile_settlement_mm", 2)):
                if row.get(key) is not None:
                    row[key] = round(float(row[key]), nd)
            out.append(row)
        return out

    def methods_table(self) -> list:
        """The methods side by side as text lines (name and numbers on one
        line, the source indented on the next)."""
        if not self.neutral_plane_comparison:
            return []
        lines = [
            f"  {'Method':<62}{'NP':>7}{'Q_np':>8}{'Drag':>8}{'Drag':>8}"
            f"{'Pile':>9}",
            f"  {'':<62}{'':>7}{'':>8}{'load':>8}{'force':>8}{'settl.':>9}",
            f"  {'':<62}{'(m)':>7}{'(kN)':>8}{'(kN)':>8}{'(kN)':>8}"
            f"{'(mm)':>9}",
        ]
        for r in self.neutral_plane_comparison:
            mark = "* " if r.get("used") else "  "
            if r["applies"]:
                lines.append(
                    f"{mark}{r['name']:<62}{r['neutral_plane_depth_m']:>7.2f}"
                    f"{r['max_pile_load_kN']:>8.0f}{r['dragload_kN']:>8.0f}"
                    f"{r['drag_force_kN']:>8.0f}"
                    f"{r['pile_settlement_mm']:>9.1f}")
                if r.get("reason"):
                    lines.append(f"      Note: {r['reason']}")
            else:
                lines.append(f"{mark}{r['name']:<62}  not applicable")
                lines.append(f"      {r['reason']}")
            lines.append(f"      {r['source']}")
        if any(r.get("used") for r in self.neutral_plane_comparison):
            lines.append("* gives the headline numbers below.")
        return lines

    def summary(self) -> str:
        """Return a text summary of key results."""
        stress = ("final (sigma'v0 + fill / drawdown change; UFC 3-220-20 "
                  "§6-7.4 step 2)" if self.skin_friction_stress == "final"
                  else "initial (sigma'v0)")
        lines = [
            "Downdrag Analysis Results (neutral plane method)",
            "=" * 48,
            f"Pile length:            {self.pile_length:.2f} m",
            f"Pile diameter:          {self.pile_diameter:.3f} m",
            f"Dead load (Q_dead):     {self.Q_dead:.1f} kN",
            f"Effective stress for skin friction: {stress}",
            f"Pile's own weight: {self.pile_weight_text}",
        ]
        table = self.methods_table()
        if table:
            lines += ["", "--- Neutral Plane by Each Method (same inputs) ---",
                      *table]
            if self.methods_range:
                lines += ["", self.methods_range]
            lines.append(self.drag_force_note())
        headline = self.headline_method or self.neutral_plane_method
        lines += [
            "",
            f"--- Headline: {headline} ---",
            f"Neutral plane depth:    {self.neutral_plane_depth:.2f} m",
            f"Drag load:              {self.dragload:.1f} kN",
            f"Pile weight to NP:      {self.pile_weight_to_np:.1f} kN",
            f"Max pile load at NP:    {self.max_pile_load:.1f} kN",
            f"Drag force (Q_np - Q_dead): {self.drag_force:.1f} kN",
            "",
            "--- Resistance Below NP ---",
            f"Positive skin friction: {self.positive_skin_friction:.1f} kN",
            f"Toe resistance:         {self.toe_resistance:.1f} kN",
            *([f"Toe force mobilized:    {self.toe_force_mobilized:.1f} kN"]
              if self.toe_force_mobilized is not None else []),
            f"Total resistance:       {self.total_resistance:.1f} kN",
            "",
            "--- Settlement ---",
            f"Pile head settlement:   {self.pile_settlement * 1000:.2f} mm",
            f"  Elastic shortening above NP: "
            f"{self.elastic_shortening * 1000:.2f} mm",
            f"  Toe settlement:       {self.toe_settlement * 1000:.2f} mm",
            *([f"  Counted below the toe to {self.below_toe['depth_m']:.2f} m"]
              if self.below_toe else []),
            f"Soil settlement at NP:  {self.soil_settlement_at_np * 1000:.2f} mm",
        ]
        if self.below_toe.get("basis"):
            lines += ["", "--- Settlement Below the Toe ---",
                      self.below_toe["basis"]]

        if any(x is not None for x in [self.structural_ok,
                                         self.geotechnical_ok,
                                         self.settlement_ok]):
            lines.append("")
            lines.append("--- Limit State Checks ---")
            if self.structural_ok is not None:
                status = "PASS" if self.structural_ok else "FAIL"
                lines.append(f"Structural (Eq 6-80):   {status}")
                if self.structural_demand is not None:
                    lines.append(
                        f"  LRFD demand:          {self.structural_demand:.1f} kN"
                    )
            if self.geotechnical_ok is not None:
                status = "PASS" if self.geotechnical_ok else "FAIL"
                lines.append(f"Geotechnical:           {status}")
            if self.settlement_ok is not None:
                status = "PASS" if self.settlement_ok else "FAIL"
                lines.append(f"Settlement:             {status}")

        if self.neutral_plane_basis:
            lines += ["", "--- How the Headline Neutral Plane Was Located ---",
                      f"Requested: neutral_plane_method="
                      f"'{self.neutral_plane_requested}'",
                      self.neutral_plane_basis]
        if self.warnings:
            lines += ["", "--- Warnings ---"]
            lines += [f"* {w}" for w in self.warnings]

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export results as a dictionary (for JSON serialization)."""
        d = {
            'neutral_plane_depth_m': self.neutral_plane_depth,
            'neutral_plane_method': self.neutral_plane_method,
            'headline_method': self.headline_method,
            'neutral_plane_basis': self.neutral_plane_basis,
            'methods_range': self.methods_range,
            'neutral_plane_comparison': self.comparison_rows(),
            'dragload_kN': self.dragload,
            'drag_force_kN': self.drag_force,
            'max_pile_load_kN': self.max_pile_load,
            'Q_dead_kN': self.Q_dead,
            'pile_weight_to_np_kN': self.pile_weight_to_np,
            'positive_skin_friction_kN': self.positive_skin_friction,
            'toe_resistance_kN': self.toe_resistance,
            'toe_force_mobilized_kN': self.toe_force_mobilized,
            'total_resistance_kN': self.total_resistance,
            'pile_settlement_m': self.pile_settlement,
            'elastic_shortening_m': self.elastic_shortening,
            'toe_settlement_m': self.toe_settlement,
            'below_toe': self.below_toe_rounded(),
            'soil_settlement_at_np_m': self.soil_settlement_at_np,
            'pile_length_m': self.pile_length,
            'pile_diameter_m': self.pile_diameter,
            'z_m': self.z.tolist(),
            'axial_load_kN': self.axial_load.tolist(),
            'soil_settlement_mm': (self.soil_settlement_profile * 1000).tolist(),
            'unit_skin_friction_kPa': self.unit_skin_friction.tolist(),
        }
        if self.structural_ok is not None:
            d['structural_ok'] = self.structural_ok
            if self.structural_demand is not None:
                d['structural_demand_kN'] = self.structural_demand
        if self.geotechnical_ok is not None:
            d['geotechnical_ok'] = self.geotechnical_ok
        if self.settlement_ok is not None:
            d['settlement_ok'] = self.settlement_ok
        d['warnings'] = list(self.warnings)
        d['neutral_plane_requested'] = self.neutral_plane_requested
        d['toe_mobilization'] = self.toe_mobilization
        d['skin_friction_stress'] = self.skin_friction_stress
        d['include_pile_weight'] = self.include_pile_weight
        d['pile_weight_basis'] = self.pile_weight_basis
        d['influence_depth_criterion'] = self.influence_depth_criterion
        return d

    def get_tables(self) -> list:
        """The load, unit skin friction and soil settlement at every depth
        of the pile (printed), and as CSV only (the calc package prints its
        own versions) the neutral plane by each method and the result
        summary. Values are this result's own, in its units.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        if len(self.z):
            tables.append(CalcTable(
                title="Load, skin friction and settlement along the pile",
                columns=[("Depth z", "m", ".2f"),
                         ("Axial load in pile Q", "kN", ".1f"),
                         ("Unit skin friction f_s", "kPa", ".2f"),
                         ("Soil settlement", "m")],
                rows=[[float(self.z[i]), float(self.axial_load[i]),
                       float(self.unit_skin_friction[i]),
                       float(self.soil_settlement_profile[i])]
                      for i in range(len(self.z))],
                note=(f"Headline: {self.headline_method or self.neutral_plane_method}"
                      f", neutral plane at {self.neutral_plane_depth:.2f} m "
                      f"with Q_np = {self.max_pile_load:.1f} kN (computed at "
                      "the plane itself, which may fall between the "
                      "tabulated depths); f_s is the magnitude, acting "
                      "downward (negative skin friction) above the neutral "
                      "plane and upward below it; the soil settlement is "
                      f"from the original ground; f_s from "
                      f"{self.skin_friction_stress} effective stress"),
                name="load_and_settlement",
            ))
        cmp_rows = self.neutral_plane_comparison or []
        if cmp_rows:
            tables.append(CalcTable(
                title="Neutral plane by each method",
                columns=["Method", "Reference", "Applies", "Headline",
                         ("Neutral plane depth z_np", "m"),
                         ("Maximum load Q_np", "kN"), ("Drag load", "kN"),
                         ("Drag force Q_np - Q_dead", "kN"),
                         ("Toe force", "kN"),
                         ("Soil settlement at the plane", "mm"),
                         ("Pile head settlement", "mm"), "Notes"],
                rows=[[r["name"], r["source"], r["applies"], r["used"],
                       r.get("neutral_plane_depth_m"),
                       r.get("max_pile_load_kN"), r.get("dragload_kN"),
                       r.get("drag_force_kN"), r.get("toe_force_kN"),
                       r.get("soil_settlement_at_np_mm"),
                       r.get("pile_settlement_mm"),
                       r.get("reason") or r.get("assumptions", "")]
                      for r in cmp_rows],
                note=(self.methods_range or
                      "Every neutral-plane method on the same inputs"),
                name="neutral_plane_methods", printed=False,
            ))
        tables.append(CalcTable(
            title="Downdrag summary",
            columns=["Quantity", "Symbol", "Value", "Unit"],
            rows=[
                ["Neutral plane depth", "z_np", self.neutral_plane_depth,
                 "m"],
                ["Dead load at the head", "Q_dead", self.Q_dead, "kN"],
                ["Drag load (negative skin friction above the plane)",
                 "Q_n", self.dragload, "kN"],
                ["Pile weight above the plane", "W_np",
                 self.pile_weight_to_np, "kN"],
                ["Maximum load in the pile", "Q_np", self.max_pile_load,
                 "kN"],
                ["Drag force", "Q_np - Q_dead", self.drag_force, "kN"],
                ["Positive shaft resistance below the plane", "R_s",
                 self.positive_skin_friction, "kN"],
                ["Toe resistance", "R_t", self.toe_resistance, "kN"],
                ["Toe force at the plane", "Q_t",
                 self.toe_force_mobilized, "kN"],
                ["Pile head settlement", "s_head", self.pile_settlement, "m"],
                ["Elastic shortening above the plane", "δ",
                 self.elastic_shortening, "m"],
                ["Soil settlement at the plane", "s_np",
                 self.soil_settlement_at_np, "m"],
            ],
            note=self.neutral_plane_basis or "",
            name="downdrag_summary", printed=False,
        ))
        return tables

    def plot_axial_load(self, ax=None, show=True, **kwargs):
        """Plot axial load Q(z) vs depth with neutral plane.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(self.axial_load, self.z, 'b-', linewidth=2, **kwargs)
        ax.axhline(y=self.neutral_plane_depth, color='r', linestyle='--',
                   linewidth=1.5,
                   label=f'Neutral Plane ({self.neutral_plane_depth:.1f} m)')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Axial Load Distribution",
                              "Axial Load Q(z) (kN)", "Depth (m)")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_settlement(self, ax=None, show=True, **kwargs):
        """Plot soil settlement profile vs depth with pile settlement.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(self.soil_settlement_profile * 1000, self.z, 'g-',
                linewidth=2, label='Soil settlement', **kwargs)
        ax.axvline(x=self.pile_settlement * 1000, color='b', linestyle='--',
                   linewidth=1.5,
                   label=f'Pile settlement ({self.pile_settlement*1000:.1f} mm)')
        ax.axhline(y=self.neutral_plane_depth, color='r', linestyle=':',
                   linewidth=1, label='Neutral Plane')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Settlement Profile",
                              "Settlement (mm)", "Depth (m)")
        ax.legend(fontsize=8)
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_neutral_plane(self, show=True):
        """Plot combined neutral plane diagram (load + settlement).

        Returns
        -------
        tuple of (fig, axes)
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        self.plot_axial_load(ax=ax1, show=False)
        self.plot_settlement(ax=ax2, show=False)
        ax2.set_ylabel('')
        fig.suptitle('Downdrag Neutral Plane Analysis', fontsize=13,
                     fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, (ax1, ax2)
