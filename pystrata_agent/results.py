"""
Results container for pystrata equivalent-linear site response analyses.

Provides:
  - summary() -> str: human-readable text output
  - to_dict() -> dict: JSON-serializable dict for LLM agents
  - plot_*() methods: matplotlib visualizations
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np


#: The periods whose spectral ordinates ``to_dict`` reports (s). The full
#: spectra (500 periods, 0.01 to 10 s) stay on the result object.
REPORTED_PERIODS_S = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0,
                      5.0)

#: Most rows ``depth_profile`` gives: tool results are kept under 8000
#: characters.
PROFILE_MAX_ROWS = 20


@dataclass
class EQLSiteResponseResult:
    """Results from 1D equivalent-linear (or linear elastic) site response.

    Attributes
    ----------
    analysis_type : str
        "equivalent_linear" or "linear_elastic".
    total_depth_m : float
        Total soil profile depth above bedrock (m).
    n_layers : int
        Number of soil layers (excluding bedrock half-space).
    motion_name : str
        Name of the input ground motion.
    pga_input_g : float
        Peak ground acceleration of input motion (g).
    pga_surface_g : float
        Peak ground acceleration at surface (g).
    amplification_factor : float
        PGA amplification (surface / input).
    n_iterations : int or None
        Number of EQL iterations run (wave solves counted; 0 for linear),
        or None when it could not be counted.
    converged : bool
        Whether EQL iterations converged.
    time : numpy.ndarray
        Time array (s).
    surface_accel_g : numpy.ndarray
        Surface acceleration time history (g).
    depths : numpy.ndarray
        Depth array for profile outputs (m, from surface).
    max_strain_pct : numpy.ndarray
        Max shear strain profile (%).
    max_accel_g : numpy.ndarray
        Max acceleration profile (g).
    initial_Vs : numpy.ndarray
        Initial (small-strain) shear wave velocity profile (m/s).
    compatible_Vs : numpy.ndarray
        Strain-compatible shear wave velocity profile (m/s).
    periods : numpy.ndarray
        Spectral periods (s).
    Sa_surface_g : numpy.ndarray
        Surface response spectrum (g, 5% damping).
    Sa_input_g : numpy.ndarray
        Input response spectrum (g, 5% damping).
    depths_accel, depths_vs : numpy.ndarray
        The depths of ``max_accel_g`` and of the two Vs profiles (m), as
        pystrata gives them; ``depths`` is the strain profile's.
    water_table_depth_m : float or None
        The water table behind the Darendeli/Menq confining stress (None:
        none taken in the profile).
    k0 : float
        The at-rest coefficient in that stress.
    layer_stresses : list of dict
        One row per Darendeli or Menq layer: the stresses at its
        mid-depth and the mean effective stress its curves were built for
        (``eql_site_response.curve_layer_stresses``).
    choices : list of dict
        What the run chose (``geotech_common.choices``).
    warnings : list of str
        What the reader should know (e.g. no water table given).
    """
    analysis_type: str = "equivalent_linear"
    total_depth_m: float = 0.0
    n_layers: int = 0
    motion_name: str = ""
    pga_input_g: float = 0.0
    pga_surface_g: float = 0.0
    amplification_factor: float = 0.0
    n_iterations: Optional[int] = 0
    converged: bool = True
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    surface_accel_g: np.ndarray = field(default_factory=lambda: np.array([]))
    depths: np.ndarray = field(default_factory=lambda: np.array([]))
    max_strain_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    max_accel_g: np.ndarray = field(default_factory=lambda: np.array([]))
    initial_Vs: np.ndarray = field(default_factory=lambda: np.array([]))
    compatible_Vs: np.ndarray = field(default_factory=lambda: np.array([]))
    periods: np.ndarray = field(default_factory=lambda: np.array([]))
    Sa_surface_g: np.ndarray = field(default_factory=lambda: np.array([]))
    Sa_input_g: np.ndarray = field(default_factory=lambda: np.array([]))
    depths_accel: np.ndarray = field(default_factory=lambda: np.array([]))
    depths_vs: np.ndarray = field(default_factory=lambda: np.array([]))
    water_table_depth_m: Optional[float] = None
    k0: float = 0.5
    layer_stresses: list = field(default_factory=list)
    choices: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        atype = ("EQUIVALENT-LINEAR" if self.analysis_type == "equivalent_linear"
                 else "LINEAR ELASTIC")
        lines = [
            "=" * 60,
            f"  1D {atype} SITE RESPONSE RESULTS",
            "=" * 60,
            "",
            f"  Profile depth:        {self.total_depth_m:.1f} m",
            f"  Number of layers:     {self.n_layers}",
            f"  Input motion:         {self.motion_name}",
            "",
            f"  PGA (input):          {self.pga_input_g:.3f} g",
            f"  PGA (surface):        {self.pga_surface_g:.3f} g",
            f"  Amplification:        {self.amplification_factor:.2f}",
            "",
        ]
        if self.analysis_type == "equivalent_linear":
            lines.append(
                f"  EQL iterations:       {self.n_iterations}"
                f"  ({'converged' if self.converged else 'NOT converged'})")
        if len(self.max_strain_pct) > 0:
            lines.append(
                f"  Max shear strain:     {np.max(self.max_strain_pct):.4f} %")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "analysis_type": self.analysis_type,
            "total_depth_m": self.total_depth_m,
            "n_layers": self.n_layers,
            "motion_name": self.motion_name,
            "pga_input_g": round(self.pga_input_g, 4),
            "pga_surface_g": round(self.pga_surface_g, 4),
            "amplification_factor": round(self.amplification_factor, 3),
            "n_iterations": self.n_iterations,
            "converged": self.converged,
        }
        if len(self.max_strain_pct) > 0:
            d["max_shear_strain_pct"] = round(
                float(np.max(self.max_strain_pct)), 4)
        # The spectra and the profiles with depth (GM-325: computed and
        # never returned).
        if len(self.periods) > 1 and len(self.Sa_surface_g) == len(self.periods):
            T = np.asarray(self.periods, dtype=float)
            at = [p for p in REPORTED_PERIODS_S if T[0] <= p <= T[-1]]
            logT = np.log(T)
            d["periods_s"] = list(at)
            d["Sa_surface_g"] = [round(float(v), 4) for v in
                                 np.interp(np.log(at), logT, self.Sa_surface_g)]
            if len(self.Sa_input_g) == len(T):
                d["Sa_input_g"] = [round(float(v), 4) for v in
                                   np.interp(np.log(at), logT, self.Sa_input_g)]
            k = int(np.argmax(self.Sa_surface_g))
            d["Sa_surface_max_g"] = round(float(self.Sa_surface_g[k]), 4)
            d["period_at_Sa_surface_max_s"] = float(f"{T[k]:.3g}")
        rows, note = self.depth_profile()
        if rows:
            d["depth_profile"] = rows
            if note:
                d["depth_profile_note"] = note
        d["water_table_depth_m"] = self.water_table_depth_m
        d["K0"] = self.k0
        if self.layer_stresses:
            d["layer_stresses"] = [dict(r) for r in self.layer_stresses]
        if self.warnings:
            d["warnings"] = list(self.warnings)
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d

    def depth_profile(self, max_rows: int = PROFILE_MAX_ROWS):
        """The response with depth, as rows a tool result can carry.

        One row per sub-layer of the discretised profile (the half-space
        left out): ``top_m`` and ``bottom_m``, ``max_strain_pct`` (peak
        shear strain at the sub-layer's middle), ``max_accel_g`` (peak
        acceleration at its top), ``initial_Vs_m_per_s`` and
        ``compatible_Vs_m_per_s``. More than ``max_rows`` sub-layers are
        merged, neighbours together, about equally many in each row: the
        row then carries the LARGEST strain and acceleration and the
        thickness-weighted (travel-time) Vs of its sub-layers.

        Returns
        -------
        (rows, note) : (list of dict, str or None)
        """
        tops = np.asarray(self.depths_vs, dtype=float).ravel()
        n = len(tops) - 1          # the last depth is the half-space's top
        if n < 1 or len(self.depths_accel) < n + 1:
            return [], None
        bottoms = np.asarray(self.depths_accel, dtype=float).ravel()[1:n + 1]
        tops = tops[:n]
        thick = bottoms - tops
        # pystrata gives the strain at each sub-layer's MIDDLE, after a
        # leading point at the surface, so the strain samples are matched
        # to the sub-layers by depth, not by position in the array.
        strain_all = np.asarray(self.max_strain_pct, dtype=float).ravel()
        z_strain = np.asarray(self.depths, dtype=float).ravel()
        if len(z_strain) != len(strain_all) or len(strain_all) == 0:
            return [], None
        mids = 0.5 * (tops + bottoms)
        nearest = np.argmin(np.abs(z_strain[None, :] - mids[:, None]), axis=1)
        if np.any(np.abs(z_strain[nearest] - mids) > 0.5 * thick + 1e-9):
            return [], None            # no strain sample inside a sub-layer
        strain = strain_all[nearest]
        accel = np.asarray(self.max_accel_g, dtype=float).ravel()
        # pystrata writes the two Vs profiles for a step plot: the first
        # sub-layer's value twice, then one value per sub-layer, against
        # the depths of the sub-layer tops and the top of the half-space
        # (pystrata.output.InitialVelProfile). So sub-layer i is entry
        # i + 1; a profile not of that form is not returned.
        vs0 = np.asarray(self.initial_Vs, dtype=float).ravel()
        vs1 = np.asarray(self.compatible_Vs, dtype=float).ravel()
        if len(accel) < n or len(vs0) != n + 1 or len(vs1) != n + 1:
            return [], None
        if vs0[0] != vs0[1] or vs1[0] != vs1[1]:
            return [], None
        vs0, vs1 = vs0[1:], vs1[1:]

        def vs_avg(v, idx):
            # the velocity that gives the same travel time over the row
            t = float(np.sum(thick[idx] / v[idx]))
            return round(float(np.sum(thick[idx]) / t), 1) if t > 0 else None

        chunks = (np.array_split(np.arange(n), max_rows) if n > max_rows
                  else [np.array([i]) for i in range(n)])
        rows = []
        for idx in chunks:
            rows.append({
                "top_m": round(float(tops[idx[0]]), 2),
                "bottom_m": round(float(bottoms[idx[-1]]), 2),
                "max_strain_pct": round(float(np.max(strain[idx])), 4),
                "max_accel_g": round(float(np.max(accel[idx])), 4),
                "initial_Vs_m_per_s": vs_avg(vs0, idx),
                "compatible_Vs_m_per_s": vs_avg(vs1, idx),
            })
        note = None
        if n > max_rows:
            note = (
                f"{n} sub-layers merged into {len(rows)} rows to keep the "
                "result small: each row carries the largest strain and "
                "acceleration of its sub-layers and their travel-time "
                "average Vs.")
        return rows, note

    def plot_surface_motion(self, ax=None, show=True, **kwargs):
        """Plot surface acceleration time history."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.time, self.surface_accel_g, '-', linewidth=0.5, **kwargs)
        setup_engineering_plot(ax, 'Surface Acceleration Time History',
                              'Time (s)', 'Acceleration (g)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_response_spectra(self, ax=None, show=True, **kwargs):
        """Plot surface vs input response spectra."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        if len(self.Sa_input_g) > 0:
            ax.loglog(self.periods, self.Sa_input_g, '--', label='Input',
                      linewidth=1.0)
        if len(self.Sa_surface_g) > 0:
            ax.loglog(self.periods, self.Sa_surface_g, '-', label='Surface',
                      linewidth=1.5)
        ax.legend()
        setup_engineering_plot(ax, 'Response Spectra (5% damping)',
                              'Period (s)', 'Spectral Acceleration (g)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_profile(self, ax=None, show=True, **kwargs):
        """Plot max acceleration and max strain vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        else:
            return ax

        if len(self.max_accel_g) > 0:
            axes[0].plot(self.max_accel_g, self.depths, '-o', markersize=3)
            axes[0].invert_yaxis()
            setup_engineering_plot(axes[0], 'Max Acceleration',
                                  'PGA (g)', 'Depth (m)')

        if len(self.max_strain_pct) > 0:
            axes[1].plot(self.max_strain_pct, self.depths, '-o', markersize=3)
            axes[1].invert_yaxis()
            setup_engineering_plot(axes[1], 'Max Shear Strain',
                                  'Strain (%)', 'Depth (m)')

        plt.tight_layout()
        if show:
            plt.show()
        return axes

    def plot_Vs_profile(self, ax=None, show=True, **kwargs):
        """Plot initial vs strain-compatible shear wave velocity."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        if len(self.initial_Vs) > 0:
            ax.plot(self.initial_Vs, self.depths, '--', label='Initial Vs',
                    linewidth=1.5)
        if len(self.compatible_Vs) > 0:
            ax.plot(self.compatible_Vs, self.depths, '-', label='Compatible Vs',
                    linewidth=1.5)
        ax.invert_yaxis()
        ax.legend()
        setup_engineering_plot(ax, 'Shear Wave Velocity Profile',
                              'Vs (m/s)', 'Depth (m)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Create a 2x2 composite plot of all site response results.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of Axes
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        self.plot_surface_motion(ax=axes[0, 0], show=False)
        self.plot_response_spectra(ax=axes[0, 1], show=False)

        # Bottom-left: max accel and strain profiles
        if len(self.max_accel_g) > 0:
            axes[1, 0].plot(self.max_accel_g, self.depths, '-o', markersize=3)
            axes[1, 0].invert_yaxis()
            axes[1, 0].set_xlabel('PGA (g)')
            axes[1, 0].set_ylabel('Depth (m)')
            axes[1, 0].set_title('Max Acceleration Profile')
            axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: Vs profile
        self.plot_Vs_profile(ax=axes[1, 1], show=False)

        atype = ("EQL" if self.analysis_type == "equivalent_linear"
                 else "Linear")
        fig.suptitle(
            f'{atype} Site Response: {self.total_depth_m:.0f}m profile, '
            f'PGA={self.pga_surface_g:.3f}g',
            fontsize=12, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes
