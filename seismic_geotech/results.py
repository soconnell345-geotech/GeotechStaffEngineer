"""
Results containers for seismic geotechnical analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class SiteClassResult:
    """AASHTO/NEHRP site classification results.

    Attributes
    ----------
    site_class : str
        Site class letter (A-F).
    Fpga : float
        Site coefficient for PGA.
    Fa : float
        Site coefficient for short-period (0.2s).
    Fv : float
        Site coefficient for long-period (1.0s).
    Ss : float
        Spectral acceleration at 0.2s (g).
    S1 : float
        Spectral acceleration at 1.0s (g).
    vs30 : float, optional
        Average shear wave velocity (m/s).
    n_bar : float, optional
        Average SPT N value.
    su_bar : float, optional
        Average undrained strength (kPa).
    """
    site_class: str = ""
    Fpga: float = 0.0
    Fa: float = 0.0
    Fv: float = 0.0
    Ss: float = 0.0
    S1: float = 0.0
    vs30: Optional[float] = None
    n_bar: Optional[float] = None
    su_bar: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SEISMIC SITE CLASSIFICATION",
            "=" * 60,
            "",
            f"  Site Class: {self.site_class}",
            f"  Ss = {self.Ss:.3f} g,  S1 = {self.S1:.3f} g",
            "",
            f"  Fpga = {self.Fpga:.3f}",
            f"  Fa   = {self.Fa:.3f}",
            f"  Fv   = {self.Fv:.3f}",
            "",
            f"  SDS  = {self.Fa * self.Ss:.3f} g  (Fa * Ss)",
            f"  SD1  = {self.Fv * self.S1:.3f} g  (Fv * S1)",
        ]
        if self.vs30 is not None:
            lines.append(f"  Vs30 = {self.vs30:.0f} m/s")
        if self.n_bar is not None:
            lines.append(f"  N-bar = {self.n_bar:.0f}")
        if self.su_bar is not None:
            lines.append(f"  su-bar = {self.su_bar:.0f} kPa")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "site_class": self.site_class,
            "Fpga": self.Fpga,
            "Fa": self.Fa,
            "Fv": self.Fv,
            "Ss_g": self.Ss,
            "S1_g": self.S1,
            "SDS_g": round(self.Fa * self.Ss, 4),
            "SD1_g": round(self.Fv * self.S1, 4),
        }
        if self.vs30 is not None:
            d["vs30_m_per_s"] = self.vs30
        if self.n_bar is not None:
            d["n_bar"] = self.n_bar
        if self.su_bar is not None:
            d["su_bar_kPa"] = self.su_bar
        return d


@dataclass
class SeismicEarthPressureResult:
    """Mononobe-Okabe seismic earth pressure results.

    Attributes
    ----------
    KAE : float
        Seismic active earth pressure coefficient.
    KPE : float
        Seismic passive earth pressure coefficient.
    PAE_total : float
        Total seismic active force (kN/m).
    PA_static : float
        Static active force (kN/m).
    delta_PAE : float
        Seismic increment (kN/m).
    height_of_application : float
        Height of seismic increment above base (m).
    phi : float
        Friction angle used (degrees).
    delta : float
        Wall friction angle used (degrees).
    kh : float
        Horizontal seismic coefficient.
    kv : float
        Vertical seismic coefficient.
    choices : list of dict
        What the calculation chose (GM-115, ``geotech_common.choices``;
        ``mononobe_okabe.earth_pressure_choices``).
    warnings : list of str
        What the user must be told, e.g. that the M-O equation has no
        solution (``mononobe_okabe.mo_no_solution_warning``).
    """
    KAE: float = 0.0
    KPE: float = 0.0
    PAE_total: float = 0.0
    PA_static: float = 0.0
    delta_PAE: float = 0.0
    height_of_application: float = 0.0
    phi: float = 0.0
    delta: float = 0.0
    kh: float = 0.0
    kv: float = 0.0
    choices: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SEISMIC EARTH PRESSURE (MONONOBE-OKABE)",
            "=" * 60,
            "",
            f"  phi = {self.phi:.1f}°, delta = {self.delta:.1f}°",
            f"  kh = {self.kh:.3f}, kv = {self.kv:.3f}",
            "",
            f"  KAE = {self.KAE:.4f}",
            f"  KPE = {self.KPE:.4f}",
            "",
            f"  PAE (total):      {self.PAE_total:>10.2f} kN/m",
            f"  PA  (static):     {self.PA_static:>10.2f} kN/m",
            f"  delta-PAE:        {self.delta_PAE:>10.2f} kN/m",
            f"  Applied at:       {self.height_of_application:.2f} m above base",
        ]
        from geotech_common.choices import choices_lines
        lines.extend(choices_lines(self.choices))
        if self.warnings:
            lines.extend(["", "  Warnings:"])
            lines.extend(f"    - {w}" for w in self.warnings)
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "KAE": round(self.KAE, 4),
            "KPE": round(self.KPE, 4),
            "PAE_total_kN_per_m": round(self.PAE_total, 2),
            "PA_static_kN_per_m": round(self.PA_static, 2),
            "delta_PAE_kN_per_m": round(self.delta_PAE, 2),
            "height_of_application_m": round(self.height_of_application, 2),
            "phi_deg": self.phi,
            "delta_deg": self.delta,
            "kh": self.kh,
            "kv": self.kv,
        }
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d


@dataclass
class LiquefactionResult:
    """Liquefaction evaluation results.

    Attributes
    ----------
    layer_results : list of dict
        Per-layer evaluation (depth, CSR, CRR, FOS, liquefiable).
    amax_g : float
        Peak ground acceleration (g).
    magnitude : float
        Earthquake magnitude.
    gwt_depth : float
        Groundwater depth (m).
    choices : list of dict
        What the evaluation chose (GM-115, ``geotech_common.choices``;
        ``liquefaction.liquefaction_choices``).
    """
    layer_results: List[Dict[str, Any]] = field(default_factory=list)
    amax_g: float = 0.0
    magnitude: float = 7.5
    gwt_depth: float = 0.0
    choices: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_liquefiable(self) -> int:
        return sum(1 for r in self.layer_results if r.get("liquefiable", False))

    @property
    def min_FOS(self) -> float:
        if not self.layer_results:
            return 99.9
        return min(r["FOS_liq"] for r in self.layer_results)

    @property
    def critical_depth(self) -> Optional[float]:
        if not self.layer_results:
            return None
        min_r = min(self.layer_results, key=lambda r: r["FOS_liq"])
        return min_r["depth_m"]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  LIQUEFACTION EVALUATION",
            "=" * 60,
            "",
            f"  PGA = {self.amax_g:.3f}g, M = {self.magnitude:.1f}",
            f"  GWT depth = {self.gwt_depth:.1f} m",
            "",
            f"  {'Depth':>6} {'N160cs':>6} {'CSR':>7} {'CRR':>7} {'FOS':>7} {'Liq?':>5}",
            f"  {'-'*40}",
        ]
        for r in self.layer_results:
            liq = "YES" if r["liquefiable"] else "no"
            lines.append(
                f"  {r['depth_m']:6.1f} {r['N160cs']:6.1f} "
                f"{r['CSR']:7.4f} {r['CRR']:7.4f} "
                f"{r['FOS_liq']:7.3f} {liq:>5}"
            )
        lines.extend([
            "",
            f"  Liquefiable layers: {self.n_liquefiable} of {len(self.layer_results)}",
            f"  Minimum FOS: {self.min_FOS:.3f}",
        ])
        if self.critical_depth is not None:
            lines.append(f"  Critical depth: {self.critical_depth:.1f} m")
        from geotech_common.choices import choices_lines
        lines.extend(choices_lines(self.choices))
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "amax_g": self.amax_g,
            "magnitude": self.magnitude,
            "gwt_depth_m": self.gwt_depth,
            "n_liquefiable": self.n_liquefiable,
            "min_FOS_liq": round(self.min_FOS, 3),
            "critical_depth_m": self.critical_depth,
            "layer_results": self.layer_results,
        }
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d

    def get_tables(self) -> list:
        """The triggering evaluation at every depth: (N1)60, fines,
        stresses, CSR, CRR and the factor of safety, as this result holds
        them (the module rounds them).

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        if not self.layer_results:
            return []
        # rd and MSF since GM-115; a result built before has none.
        factors = all("rd" in r and "MSF" in r for r in self.layer_results)
        columns = [("Depth z", "m", ".2f"), "(N1)60", "(N1)60cs",
                   ("Fines content FC", "%"), ("σv", "kPa"),
                   ("σ'v", "kPa")]
        if factors:
            columns += [("rd", "", ".4f"), ("MSF", "", ".3f")]
        columns += [("CSR (M 7.5)", "", ".4f"), ("CRR (M 7.5)", "", ".4f"),
                    ("FS_liq", "", ".3f"), "Liquefiable"]
        return [CalcTable(
            title="Liquefaction triggering at each depth",
            columns=columns,
            rows=[[r["depth_m"], r["N160"], r["N160cs"], r["FC_pct"],
                   r["sigma_v_kPa"], r["sigma_v_eff_kPa"]]
                  + ([r["rd"], r["MSF"]] if factors else [])
                  + [r["CSR"], r["CRR"], r["FOS_liq"], r["liquefiable"]]
                  for r in self.layer_results],
            note=(f"NCEER / Youd et al. (2001) SPT simplified procedure: "
                  f"CSR = 0.65 (amax/g)(σv/σ'v) rd / MSF with "
                  f"amax = {self.amax_g:g} g, M = {self.magnitude:g}, rd by "
                  "Liao & Whitman (1986); CRR from the NCEER clean-sand "
                  "curve; FS = CRR/CSR; groundwater at "
                  f"{self.gwt_depth:g} m"),
            name="liquefaction",
        )]

    def plot_fos_profile(self, ax=None, show=True, **kwargs):
        """Plot liquefaction FOS vs depth.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.layer_results:
            raise ValueError("No layer results available.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        depths = [r['depth_m'] for r in self.layer_results]
        fos_vals = [r['FOS_liq'] for r in self.layer_results]
        colors = ['red' if r['liquefiable'] else 'green'
                  for r in self.layer_results]
        ax.scatter(fos_vals, depths, c=colors, zorder=5, s=50, **kwargs)
        ax.plot(fos_vals, depths, 'k-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
                   label='FOS = 1.0')
        ax.invert_yaxis()
        setup_engineering_plot(
            ax,
            f"Liquefaction FOS (PGA={self.amax_g:.3f}g, M={self.magnitude:.1f})",
            "Factor of Safety", "Depth (m)")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_csr_crr(self, ax=None, show=True, **kwargs):
        """Plot CSR and CRR vs depth.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.layer_results:
            raise ValueError("No layer results available.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        depths = [r['depth_m'] for r in self.layer_results]
        csr = [r['CSR'] for r in self.layer_results]
        crr = [r['CRR'] for r in self.layer_results]
        ax.plot(csr, depths, 'r-o', linewidth=1.5, markersize=5, label='CSR')
        ax.plot(crr, depths, 'g-s', linewidth=1.5, markersize=5, label='CRR')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "CSR vs CRR Profile",
                              "Cyclic Stress Ratio", "Depth (m)")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax
