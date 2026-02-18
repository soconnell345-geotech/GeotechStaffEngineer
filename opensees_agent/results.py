"""
Results containers for OpenSees agent analyses.

Each result dataclass provides:
  - summary() -> str: human-readable text output
  - to_dict() -> dict: JSON-serializable dict for LLM agents
  - plot_*() methods: matplotlib visualizations
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np


# ===========================================================================
# PM4Sand Undrained Cyclic DSS Result
# ===========================================================================

@dataclass
class PM4SandDSSResult:
    """Results from PM4Sand undrained cyclic direct simple shear analysis.

    Attributes
    ----------
    Dr : float
        Relative density used.
    sigma_v_kPa : float
        Initial vertical effective stress (kPa).
    CSR_applied : float
        Applied cyclic stress ratio.
    K0 : float
        Lateral earth pressure coefficient.
    n_cycles_to_liq : float
        Number of cycles to reach ru_threshold (inf if no liquefaction).
    liquefied : bool
        True if ru exceeded the threshold.
    max_ru : float
        Peak excess pore pressure ratio achieved.
    max_shear_strain_pct : float
        Peak shear strain (%).
    time : numpy.ndarray
        Time array (s).
    shear_stress_kPa : numpy.ndarray
        Shear stress history (kPa).
    shear_strain_pct : numpy.ndarray
        Shear strain history (%).
    vert_eff_stress_kPa : numpy.ndarray
        Vertical effective stress history (kPa).
    ru : numpy.ndarray
        Excess pore pressure ratio history.
    """
    Dr: float = 0.0
    sigma_v_kPa: float = 0.0
    CSR_applied: float = 0.0
    K0: float = 0.5
    n_cycles_to_liq: float = float('inf')
    liquefied: bool = False
    max_ru: float = 0.0
    max_shear_strain_pct: float = 0.0
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_stress_kPa: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_strain_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    vert_eff_stress_kPa: np.ndarray = field(default_factory=lambda: np.array([]))
    ru: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PM4SAND UNDRAINED CYCLIC DSS RESULTS",
            "=" * 60,
            "",
            f"  Relative density:     Dr = {self.Dr:.2f}",
            f"  Vertical stress:      σ'v = {self.sigma_v_kPa:.1f} kPa",
            f"  Cyclic stress ratio:  CSR = {self.CSR_applied:.3f}",
            f"  K0:                   {self.K0:.2f}",
            "",
            f"  Liquefied:            {'YES' if self.liquefied else 'NO'}",
            f"  Cycles to liq:        {self.n_cycles_to_liq:.1f}" if self.liquefied
            else f"  Max cycles run:       {len(self.time)} steps (no liquefaction)",
            f"  Max ru:               {self.max_ru:.3f}",
            f"  Max shear strain:     {self.max_shear_strain_pct:.3f} %",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Dr": self.Dr,
            "sigma_v_kPa": self.sigma_v_kPa,
            "CSR_applied": self.CSR_applied,
            "K0": self.K0,
            "liquefied": self.liquefied,
            "n_cycles_to_liq": self.n_cycles_to_liq if self.liquefied else None,
            "max_ru": round(self.max_ru, 4),
            "max_shear_strain_pct": round(self.max_shear_strain_pct, 4),
        }

    def plot_stress_strain(self, ax=None, show=True, **kwargs):
        """Plot shear stress vs shear strain hysteresis loops."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.shear_strain_pct, self.shear_stress_kPa, '-',
                linewidth=0.8, **kwargs)
        setup_engineering_plot(ax, 'Cyclic Stress-Strain Response',
                              'Shear Strain (%)', 'Shear Stress (kPa)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_stress_path(self, ax=None, show=True, **kwargs):
        """Plot shear stress vs vertical effective stress (stress path)."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.vert_eff_stress_kPa, self.shear_stress_kPa, '-',
                linewidth=0.8, **kwargs)
        setup_engineering_plot(ax, 'Effective Stress Path',
                              "Vertical Effective Stress, σ'v (kPa)",
                              'Shear Stress, τ (kPa)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_pore_pressure(self, ax=None, show=True, **kwargs):
        """Plot excess pore pressure ratio vs time."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.time, self.ru, '-', linewidth=1.0, **kwargs)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                   label='ru = 1.0')
        ax.set_ylim(-0.1, 1.2)
        setup_engineering_plot(ax, 'Pore Pressure Generation',
                              'Time (s)', 'Excess Pore Pressure Ratio, ru')
        ax.legend(fontsize=9)
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Create a 2x2 composite plot of all DSS results.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of Axes
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        self.plot_stress_strain(ax=axes[0, 0], show=False)
        self.plot_stress_path(ax=axes[0, 1], show=False)
        self.plot_pore_pressure(ax=axes[1, 0], show=False)

        # Bottom-right: strain vs time
        axes[1, 1].plot(self.time, self.shear_strain_pct, '-', linewidth=0.8)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Shear Strain (%)')
        axes[1, 1].set_title('Shear Strain vs Time')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(
            f'PM4Sand Cyclic DSS: Dr={self.Dr:.2f}, CSR={self.CSR_applied:.3f}, '
            f"σ'v={self.sigma_v_kPa:.0f} kPa",
            fontsize=12, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes


# ===========================================================================
# BNWF Lateral Pile Result
# ===========================================================================

@dataclass
class BNWFPileResult:
    """Results from BNWF laterally-loaded pile analysis.

    Attributes
    ----------
    pile_length : float
        Embedded pile length (m).
    pile_diameter : float
        Pile diameter (m).
    lateral_force_kN : float
        Applied lateral load (kN).
    moment_kNm : float
        Applied moment (kN-m).
    axial_force_kN : float
        Applied axial load (kN).
    z : numpy.ndarray
        Depth array (m, from top of pile).
    deflection_m : numpy.ndarray
        Lateral deflection profile (m).
    moment_profile_kNm : numpy.ndarray
        Bending moment profile (kN-m).
    shear_profile_kN : numpy.ndarray
        Shear force profile (kN).
    soil_reaction_kN_per_m : numpy.ndarray
        Soil spring reaction (kN/m).
    y_top_m : float
        Pile head deflection (m).
    rotation_top_rad : float
        Pile head rotation (rad).
    max_moment_kNm : float
        Maximum bending moment (kN-m).
    max_moment_depth_m : float
        Depth of maximum moment (m).
    max_deflection_m : float
        Maximum deflection (m).
    converged : bool
        Whether the analysis converged.
    """
    pile_length: float = 0.0
    pile_diameter: float = 0.0
    lateral_force_kN: float = 0.0
    moment_kNm: float = 0.0
    axial_force_kN: float = 0.0
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    deflection_m: np.ndarray = field(default_factory=lambda: np.array([]))
    moment_profile_kNm: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_profile_kN: np.ndarray = field(default_factory=lambda: np.array([]))
    soil_reaction_kN_per_m: np.ndarray = field(default_factory=lambda: np.array([]))
    y_top_m: float = 0.0
    rotation_top_rad: float = 0.0
    max_moment_kNm: float = 0.0
    max_moment_depth_m: float = 0.0
    max_deflection_m: float = 0.0
    converged: bool = True

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  BNWF LATERAL PILE ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Pile length:          {self.pile_length:.1f} m",
            f"  Pile diameter:        {self.pile_diameter:.3f} m",
            f"  Lateral load:         {self.lateral_force_kN:.1f} kN",
            f"  Applied moment:       {self.moment_kNm:.1f} kN-m",
            f"  Axial load:           {self.axial_force_kN:.1f} kN",
            "",
            f"  Head deflection:      {self.y_top_m * 1000:.2f} mm",
            f"  Head rotation:        {self.rotation_top_rad * 1000:.3f} mrad",
            f"  Max deflection:       {self.max_deflection_m * 1000:.2f} mm",
            f"  Max moment:           {self.max_moment_kNm:.1f} kN-m"
            f"  at depth {self.max_moment_depth_m:.1f} m",
            f"  Converged:            {'YES' if self.converged else 'NO'}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pile_length_m": self.pile_length,
            "pile_diameter_m": self.pile_diameter,
            "lateral_force_kN": self.lateral_force_kN,
            "moment_kNm": self.moment_kNm,
            "axial_force_kN": self.axial_force_kN,
            "y_top_mm": round(self.y_top_m * 1000, 3),
            "rotation_top_mrad": round(self.rotation_top_rad * 1000, 4),
            "max_deflection_mm": round(self.max_deflection_m * 1000, 3),
            "max_moment_kNm": round(self.max_moment_kNm, 2),
            "max_moment_depth_m": round(self.max_moment_depth_m, 2),
            "converged": self.converged,
        }

    def plot_deflection(self, ax=None, show=True, **kwargs):
        """Plot lateral deflection vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        ax.plot(self.deflection_m * 1000, self.z, '-', linewidth=1.5, **kwargs)
        ax.invert_yaxis()
        setup_engineering_plot(ax, 'Lateral Deflection',
                              'Deflection (mm)', 'Depth (m)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_moment(self, ax=None, show=True, **kwargs):
        """Plot bending moment vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        ax.plot(self.moment_profile_kNm, self.z, '-', linewidth=1.5, **kwargs)
        ax.invert_yaxis()
        setup_engineering_plot(ax, 'Bending Moment',
                              'Moment (kN·m)', 'Depth (m)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_shear(self, ax=None, show=True, **kwargs):
        """Plot shear force vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        ax.plot(self.shear_profile_kN, self.z, '-', linewidth=1.5, **kwargs)
        ax.invert_yaxis()
        setup_engineering_plot(ax, 'Shear Force',
                              'Shear (kN)', 'Depth (m)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Create a 1x3 composite plot of deflection, moment, and shear.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of Axes
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(14, 8))

        self.plot_deflection(ax=axes[0], show=False)
        self.plot_moment(ax=axes[1], show=False)
        self.plot_shear(ax=axes[2], show=False)

        fig.suptitle(
            f'BNWF Pile: L={self.pile_length:.0f}m, D={self.pile_diameter:.2f}m, '
            f'V={self.lateral_force_kN:.0f}kN',
            fontsize=12, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes


# ===========================================================================
# 1D Site Response Result
# ===========================================================================

@dataclass
class SiteResponseResult:
    """Results from 1D effective-stress site response analysis.

    Attributes
    ----------
    total_depth_m : float
        Total soil profile depth (m).
    n_layers : int
        Number of soil layers.
    motion_name : str
        Name of the input ground motion.
    pga_input_g : float
        Peak ground acceleration of input motion (g).
    pga_surface_g : float
        Peak ground acceleration at surface (g).
    amplification_factor : float
        PGA amplification (surface / input).
    time : numpy.ndarray
        Time array (s).
    surface_accel_g : numpy.ndarray
        Surface acceleration time history (g).
    depths : numpy.ndarray
        Depth array (m).
    max_strain_pct : numpy.ndarray
        Max shear strain profile (%).
    max_accel_g : numpy.ndarray
        Max acceleration profile (g).
    max_pore_pressure_ratio : numpy.ndarray
        Max excess pore pressure ratio profile.
    periods : numpy.ndarray
        Spectral periods (s).
    Sa_surface_g : numpy.ndarray
        Surface response spectrum (g).
    Sa_input_g : numpy.ndarray
        Input response spectrum (g).
    """
    total_depth_m: float = 0.0
    n_layers: int = 0
    motion_name: str = ""
    pga_input_g: float = 0.0
    pga_surface_g: float = 0.0
    amplification_factor: float = 0.0
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    surface_accel_g: np.ndarray = field(default_factory=lambda: np.array([]))
    depths: np.ndarray = field(default_factory=lambda: np.array([]))
    max_strain_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    max_accel_g: np.ndarray = field(default_factory=lambda: np.array([]))
    max_pore_pressure_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    periods: np.ndarray = field(default_factory=lambda: np.array([]))
    Sa_surface_g: np.ndarray = field(default_factory=lambda: np.array([]))
    Sa_input_g: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  1D EFFECTIVE-STRESS SITE RESPONSE RESULTS",
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
        if len(self.max_strain_pct) > 0:
            lines.append(f"  Max shear strain:     {np.max(self.max_strain_pct):.3f} %")
        if len(self.max_pore_pressure_ratio) > 0:
            lines.append(f"  Max ru:               {np.max(self.max_pore_pressure_ratio):.3f}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "total_depth_m": self.total_depth_m,
            "n_layers": self.n_layers,
            "motion_name": self.motion_name,
            "pga_input_g": round(self.pga_input_g, 4),
            "pga_surface_g": round(self.pga_surface_g, 4),
            "amplification_factor": round(self.amplification_factor, 3),
        }
        if len(self.max_strain_pct) > 0:
            d["max_shear_strain_pct"] = round(float(np.max(self.max_strain_pct)), 4)
        if len(self.max_pore_pressure_ratio) > 0:
            d["max_ru"] = round(float(np.max(self.max_pore_pressure_ratio)), 4)
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
            return ax  # Can't do subplots on a single provided axis

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
