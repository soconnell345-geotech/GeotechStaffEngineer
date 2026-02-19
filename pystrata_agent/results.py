"""
Results container for pystrata equivalent-linear site response analyses.

Provides:
  - summary() -> str: human-readable text output
  - to_dict() -> dict: JSON-serializable dict for LLM agents
  - plot_*() methods: matplotlib visualizations
"""

from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np


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
    n_iterations : int
        Number of EQL iterations to converge (0 for linear).
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
    """
    analysis_type: str = "equivalent_linear"
    total_depth_m: float = 0.0
    n_layers: int = 0
    motion_name: str = ""
    pga_input_g: float = 0.0
    pga_surface_g: float = 0.0
    amplification_factor: float = 0.0
    n_iterations: int = 0
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
        return d

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
