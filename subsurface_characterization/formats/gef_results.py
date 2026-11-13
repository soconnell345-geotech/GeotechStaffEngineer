"""
Result dataclasses for the GEF / BRO-XML format adapter (pygef-backed).

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
    missing_values : dict
        {column: count} of readings the file does not have (null, void or
        non-finite), for each of q_c_kPa, f_s_kPa, u_2_kPa and Rf_pct that
        the file carries. A missing reading is NaN in its array and null in
        ``to_dict``; it is never 0.0.
    notes : list of str
        Anything the reader has to say about how the file was read.
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
    missing_values: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def _missing_counts(self) -> dict:
        """{column: count of non-finite readings}, counted from the arrays
        themselves so a result built by hand reports the same way."""
        counts = {}
        for name in ("q_c_kPa", "f_s_kPa", "u_2_kPa", "Rf_pct"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if len(arr) > 0:
                counts[name] = int((~np.isfinite(arr)).sum())
        return counts

    def summary(self) -> str:
        """One-line summary of results."""
        gwl_str = f", GWL={self.gwl_m:.1f}m" if self.gwl_m is not None else ""
        missing = {k: v for k, v in self._missing_counts().items() if v}
        miss_str = ""
        if missing:
            miss_str = "; missing readings: " + ", ".join(
                f"{k} {v}" for k, v in missing.items())
        return (
            f"CPT '{self.alias}': {self.n_points} pts, "
            f"depth=0-{self.final_depth_m:.1f}m{gwl_str}{miss_str}"
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
        # Include data arrays for Foundry transfer. A missing reading is
        # null, so it cannot be taken for a measured zero.
        def _col(arr, ndigits):
            return [round(v, ndigits) if np.isfinite(v) else None
                    for v in np.asarray(arr, dtype=float).tolist()]

        if len(self.depth_m) > 0:
            d["depth_m"] = _col(self.depth_m, 3)
        if len(self.q_c_kPa) > 0:
            d["q_c_kPa"] = _col(self.q_c_kPa, 1)
        if len(self.f_s_kPa) > 0:
            d["f_s_kPa"] = _col(self.f_s_kPa, 2)
        if len(self.u_2_kPa) > 0:
            d["u_2_kPa"] = _col(self.u_2_kPa, 2)
        if len(self.Rf_pct) > 0:
            d["Rf_pct"] = _col(self.Rf_pct, 2)
        counts = self._missing_counts()
        d["missing_values"] = counts
        warnings = list(self.notes)
        if any(counts.values()):
            warnings.append(
                "Readings the file does not have are null in the arrays "
                "(not 0, and not interpolated): " + ", ".join(
                    f"{k} {v} of {self.n_points}"
                    for k, v in counts.items() if v) + ".")
        if warnings:
            d["warnings"] = warnings
        return d

    def to_liquepy_inputs(self) -> dict:
        """Convert to dict suitable for liquepy_agent.analyze_cpt_liquefaction().

        Returns dict with keys: depth, q_c, f_s, u_2, gwl.
        All in kPa (consistent with liquepy convention).

        A depth whose q_c is missing, or whose f_s or u_2 is missing in a
        column the file carries, is LEFT OUT (a triggering calculation
        cannot run on a reading that is not there, and a zero in its place
        would be read as very loose ground). ``rows_dropped`` says how many
        and ``rows_dropped_reason`` why; both are absent when none were.
        A file with no f_s or u_2 column at all gets zeros for it, as
        before, and ``columns_assumed_zero`` names the column.
        """
        depth = np.asarray(self.depth_m, dtype=float)
        keep = np.ones(len(depth), dtype=bool)
        cols = {"q_c": self.q_c_kPa, "f_s": self.f_s_kPa, "u_2": self.u_2_kPa}
        assumed_zero = []
        for key, arr in cols.items():
            arr = np.asarray(arr, dtype=float)
            if len(arr) == len(depth) and len(arr) > 0:
                keep &= np.isfinite(arr)
            elif key != "q_c":
                assumed_zero.append(key)
        d = {"depth": depth[keep]}
        for key, arr in cols.items():
            arr = np.asarray(arr, dtype=float)
            d[key] = (arr[keep] if len(arr) == len(depth) and len(arr) > 0
                      else np.zeros(int(keep.sum())))
        if self.gwl_m is not None:
            d["gwl"] = self.gwl_m
        n_dropped = int((~keep).sum())
        if n_dropped:
            d["rows_dropped"] = n_dropped
            d["rows_dropped_reason"] = (
                "q_c, f_s or u_2 is missing at these depths in the file")
        if assumed_zero:
            d["columns_assumed_zero"] = assumed_zero
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
