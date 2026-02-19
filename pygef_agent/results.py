"""
Result dataclasses for pygef agent.

Two result types:
- CPTParseResult: Parsed CPT data with arrays in kPa and standard columns
- BoreParseResult: Parsed borehole data with layer descriptions
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CPTParseResult:
    """Result of parsing a CPT file.

    All pressures/resistances converted from MPa to kPa.

    Attributes — Scalars
    --------------------
    n_points : int
        Number of data points.
    alias : str
        Test ID or filename.
    final_depth_m : float
        Final penetration depth (m).
    predrilled_depth_m : float
        Pre-excavated depth (m).
    gwl_m : float or None
        Groundwater level (m below surface).
    x : float or None
        X coordinate (easting).
    y : float or None
        Y coordinate (northing).
    srs_name : str
        Spatial reference system.

    Attributes — Arrays
    -------------------
    depth_m : ndarray
        Depth below ground (m).
    q_c_kPa : ndarray
        Cone tip resistance (kPa).
    f_s_kPa : ndarray
        Sleeve friction (kPa). Empty if not in file.
    u_2_kPa : ndarray
        Pore pressure u2 (kPa). Empty if not in file.
    Rf_pct : ndarray
        Friction ratio (%). Empty if not in file.
    available_columns : list
        Column names present in the original file.
    """

    # Scalars
    n_points: int = 0
    alias: str = ""
    final_depth_m: float = 0.0
    predrilled_depth_m: float = 0.0
    gwl_m: float = None
    x: float = None
    y: float = None
    srs_name: str = ""

    # Arrays
    depth_m: np.ndarray = field(default_factory=lambda: np.array([]))
    q_c_kPa: np.ndarray = field(default_factory=lambda: np.array([]))
    f_s_kPa: np.ndarray = field(default_factory=lambda: np.array([]))
    u_2_kPa: np.ndarray = field(default_factory=lambda: np.array([]))
    Rf_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    available_columns: list = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary of results."""
        gwl_str = f", GWL={self.gwl_m:.1f}m" if self.gwl_m is not None else ""
        return (
            f"CPT '{self.alias}': {self.n_points} pts, "
            f"depth=0-{self.final_depth_m:.1f}m{gwl_str}"
        )

    def to_dict(self) -> dict:
        """Return JSON-serializable dict of scalar results + data arrays."""
        d = {
            "n_points": self.n_points,
            "alias": self.alias,
            "final_depth_m": round(self.final_depth_m, 2),
            "predrilled_depth_m": round(self.predrilled_depth_m, 2),
            "gwl_m": round(self.gwl_m, 2) if self.gwl_m is not None else None,
            "x": self.x,
            "y": self.y,
            "srs_name": self.srs_name,
            "available_columns": self.available_columns,
        }
        # Include data arrays for Foundry transfer
        if len(self.depth_m) > 0:
            d["depth_m"] = [round(v, 3) for v in self.depth_m.tolist()]
        if len(self.q_c_kPa) > 0:
            d["q_c_kPa"] = [round(v, 1) for v in self.q_c_kPa.tolist()]
        if len(self.f_s_kPa) > 0:
            d["f_s_kPa"] = [round(v, 2) for v in self.f_s_kPa.tolist()]
        if len(self.u_2_kPa) > 0:
            d["u_2_kPa"] = [round(v, 2) for v in self.u_2_kPa.tolist()]
        if len(self.Rf_pct) > 0:
            d["Rf_pct"] = [round(v, 2) for v in self.Rf_pct.tolist()]
        return d

    def to_liquepy_inputs(self) -> dict:
        """Convert to dict suitable for liquepy_agent.analyze_cpt_liquefaction().

        Returns dict with keys: depth, q_c, f_s, u_2, gwl.
        All in kPa (consistent with liquepy convention).
        """
        d = {
            "depth": self.depth_m,
            "q_c": self.q_c_kPa,
            "f_s": self.f_s_kPa if len(self.f_s_kPa) > 0 else np.zeros_like(self.depth_m),
            "u_2": self.u_2_kPa if len(self.u_2_kPa) > 0 else np.zeros_like(self.depth_m),
        }
        if self.gwl_m is not None:
            d["gwl"] = self.gwl_m
        return d

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_qc(self, ax=None, show=True, **kwargs):
        """Plot cone resistance vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 8))
        ax.plot(self.q_c_kPa / 1000, self.depth_m, 'b-', linewidth=1, **kwargs)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        setup_engineering_plot(ax, f'CPT: {self.alias}',
                               'Cone Resistance, qc (MPa)', 'Depth (m)')
        if show:
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Multi-panel plot: qc, fs, Rf."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        n_panels = 1
        if len(self.f_s_kPa) > 0:
            n_panels += 1
        if len(self.Rf_pct) > 0:
            n_panels += 1
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 8), sharey=True)
        if n_panels == 1:
            axes = [axes]

        idx = 0
        # qc
        axes[idx].plot(self.q_c_kPa / 1000, self.depth_m, 'b-', linewidth=1)
        axes[idx].invert_yaxis()
        axes[idx].set_xlim(left=0)
        setup_engineering_plot(axes[idx], 'qc', 'qc (MPa)', 'Depth (m)')
        idx += 1

        # fs
        if len(self.f_s_kPa) > 0:
            axes[idx].plot(self.f_s_kPa / 1000, self.depth_m, 'r-', linewidth=1)
            axes[idx].set_xlim(left=0)
            setup_engineering_plot(axes[idx], 'fs', 'fs (MPa)', 'Depth (m)')
            idx += 1

        # Rf
        if len(self.Rf_pct) > 0:
            axes[idx].plot(self.Rf_pct, self.depth_m, 'g-', linewidth=1)
            axes[idx].set_xlim(left=0)
            setup_engineering_plot(axes[idx], 'Rf', 'Friction Ratio (%)', 'Depth (m)')

        fig.suptitle(self.summary(), fontsize=10)
        fig.tight_layout()
        if show:
            plt.show()
        return axes


@dataclass
class BoreParseResult:
    """Result of parsing a borehole file.

    Attributes — Scalars
    --------------------
    n_layers : int
        Number of soil layers.
    alias : str
        Borehole ID or filename.
    final_depth_m : float
        Total bore depth (m).
    gwl_m : float or None
        Groundwater level (m below surface).
    x : float or None
        X coordinate.
    y : float or None
        Y coordinate.
    srs_name : str
        Spatial reference system.

    Attributes — Arrays/Lists
    -------------------------
    top_m : ndarray
        Upper boundary of each layer (m).
    bottom_m : ndarray
        Lower boundary of each layer (m).
    soil_name : list
        Geotechnical soil name for each layer.
    soil_code : list
        Soil code for each layer (if available).
    """

    # Scalars
    n_layers: int = 0
    alias: str = ""
    final_depth_m: float = 0.0
    gwl_m: float = None
    x: float = None
    y: float = None
    srs_name: str = ""

    # Arrays/Lists
    top_m: np.ndarray = field(default_factory=lambda: np.array([]))
    bottom_m: np.ndarray = field(default_factory=lambda: np.array([]))
    soil_name: list = field(default_factory=list)
    soil_code: list = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary of results."""
        gwl_str = f", GWL={self.gwl_m:.1f}m" if self.gwl_m is not None else ""
        return (
            f"Bore '{self.alias}': {self.n_layers} layers, "
            f"depth=0-{self.final_depth_m:.1f}m{gwl_str}"
        )

    def to_dict(self) -> dict:
        """Return JSON-serializable dict."""
        return {
            "n_layers": self.n_layers,
            "alias": self.alias,
            "final_depth_m": round(self.final_depth_m, 2),
            "gwl_m": round(self.gwl_m, 2) if self.gwl_m is not None else None,
            "x": self.x,
            "y": self.y,
            "srs_name": self.srs_name,
            "layers": [
                {
                    "top_m": round(float(self.top_m[i]), 2),
                    "bottom_m": round(float(self.bottom_m[i]), 2),
                    "soil_name": self.soil_name[i] if i < len(self.soil_name) else "",
                    "soil_code": self.soil_code[i] if i < len(self.soil_code) else "",
                }
                for i in range(self.n_layers)
            ],
        }

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_profile(self, ax=None, show=True, **kwargs):
        """Plot borehole soil profile."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 8))
        for i in range(self.n_layers):
            top = float(self.top_m[i])
            bot = float(self.bottom_m[i])
            name = self.soil_name[i] if i < len(self.soil_name) else ""
            ax.barh(y=(top + bot) / 2, width=1, height=bot - top,
                    align='center', edgecolor='black', linewidth=0.5)
            ax.text(0.5, (top + bot) / 2, name, ha='center', va='center',
                    fontsize=7)
        ax.invert_yaxis()
        setup_engineering_plot(ax, f'Bore: {self.alias}', '', 'Depth (m)')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if show:
            plt.show()
        return ax
