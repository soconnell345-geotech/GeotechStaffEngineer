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
# 1D Site Response Result
# ===========================================================================

#: Periods (s) at which ``SiteResponseResult.to_dict`` gives the 5 %
#: damped surface and input spectra (GM-260).
REPORTED_PERIODS_S = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0)


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
        Peak ground acceleration at surface (g): the raw peak of the
        computed record, not filtered.
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
        Peak excess pore pressure ratio at each element centre: the
        excess pore pressure over the initial vertical effective stress,
        (u - u0) / sigma'v0 (GM-79); 0 where there is no pore water. Not
        capped at 1.
    periods : numpy.ndarray
        Spectral periods (s).
    Sa_surface_g : numpy.ndarray
        Surface response spectrum (g).
    Sa_input_g : numpy.ndarray
        Input response spectrum (g).
    analysis_type : str
        "nonlinear effective stress" or "linear elastic".
    site_period_s : float
        Site period 4H / Vs (Vs averaged by travel time), the fundamental
        period of the soil column on rigid rock (Kramer 1996, Sec. 7.2).
    initial_sigma_v_eff_kPa : numpy.ndarray
        Vertical effective stress at each element centre after the
        gravity stages (kPa), the denominator of ru.
    solver_steps : dict
        How the dynamic steps converged (GM-79): the count by algorithm
        at the full step, the steps that needed a fallback algorithm and
        the steps taken in substeps (with their times).
    rayleigh_freqs_hz : list of float
        The two frequencies at which the Rayleigh damping equals the
        target damping.
    n_elements : int
        Elements in the column.
    layer_curves : list of dict
        Each layer's stress-strain curve (GM-80): its name, the strength
        and mean effective stress it was fitted at, G/Gmax and the curve's
        damping at 0.001-1 % strain, and the Masing damping the fitted
        backbone gives.
    choices : list of dict
        What the analysis chose (GM-115, ``geotech_common.choices``).
    pga_time_s : float
        Time of the surface PGA (s).
    pga_lowpass_hz : float or None
        Corner of the low-pass behind the two values below (GM-264).
    pga_surface_lowpass_g, pga_input_lowpass_g : float or None
        Peak of the surface and the input record low-passed at
        ``pga_lowpass_hz`` (fourth-order Butterworth, forward and
        backward). ``pga_surface_g`` stays the raw peak.
    pga_share_above_lowpass : float or None
        At the raw peak's instant, the share of it carried by motion
        above ``pga_lowpass_hz`` (the record less the low-passed record).
    pga_input_share_above_lowpass : float or None
        The same share for the input record at its own peak.
    pga_set_above_lowpass : bool
        ``pga_share_above_lowpass`` is over one half: motion above the
        corner sets the raw PGA.
    Sa_surface_0_01s_g : float or None
        5 % damped spectral acceleration of the surface record at 0.01 s.
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
    analysis_type: str = "nonlinear effective stress"
    site_period_s: float = 0.0
    initial_sigma_v_eff_kPa: np.ndarray = field(
        default_factory=lambda: np.array([]))
    solver_steps: dict = field(default_factory=dict)
    rayleigh_freqs_hz: list = field(default_factory=list)
    n_elements: int = 0
    layer_curves: list = field(default_factory=list)
    choices: list = field(default_factory=list)
    pga_time_s: float = 0.0
    pga_lowpass_hz: Optional[float] = None
    pga_surface_lowpass_g: Optional[float] = None
    pga_input_lowpass_g: Optional[float] = None
    pga_share_above_lowpass: Optional[float] = None
    pga_input_share_above_lowpass: Optional[float] = None
    pga_set_above_lowpass: bool = False
    Sa_surface_0_01s_g: Optional[float] = None

    def pga_note(self) -> str:
        """What sets the surface PGA, when it is motion above the
        low-pass corner (GM-264); else empty."""
        if not self.pga_set_above_lowpass:
            return ""
        fc = self.pga_lowpass_hz
        return (
            f"The surface PGA, {self.pga_surface_g:.3f} g at "
            f"{self.pga_time_s:.2f} s, is set by motion above {fc:g} Hz: "
            f"at that instant {round(self.pga_share_above_lowpass * 100)} % "
            f"of it is above {fc:g} Hz (over half, the tool's line), against "
            f"{round(self.pga_input_share_above_lowpass * 100)} % at the "
            f"input's own peak. In a sand that dilates it can be a short "
            f"spike whose height changes with the element size; the "
            f"spectrum at 0.2 s and longer changes much less. The record "
            f"low-passed at {fc:g} Hz peaks at "
            f"{self.pga_surface_lowpass_g:.3f} g (the input, low-passed "
            f"the same way, {self.pga_input_lowpass_g:.3f} g); Sa at "
            f"0.01 s is {self.Sa_surface_0_01s_g:.3f} g.")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  1D EFFECTIVE-STRESS SITE RESPONSE RESULTS",
            "=" * 60,
            "",
            f"  Analysis:             {self.analysis_type}",
            f"  Profile depth:        {self.total_depth_m:.1f} m",
            f"  Number of layers:     {self.n_layers}",
            f"  Site period (4H/Vs):  {self.site_period_s:.3f} s",
            f"  Input motion:         {self.motion_name} (rock outcrop)",
            "",
            f"  PGA (input):          {self.pga_input_g:.3f} g",
            f"  PGA (surface):        {self.pga_surface_g:.3f} g",
            f"  Amplification:        {self.amplification_factor:.2f}",
        ]
        if self.pga_surface_lowpass_g is not None:
            fc = self.pga_lowpass_hz
            lines.extend([
                f"  PGA low-passed at {fc:g} Hz (surface / input): "
                f"{self.pga_surface_lowpass_g:.3f} / "
                f"{self.pga_input_lowpass_g:.3f} g",
                f"  Sa at 0.01 s (surface): {self.Sa_surface_0_01s_g:.3f} g",
                f"  Share of the PGA above {fc:g} Hz (surface / input): "
                f"{round(self.pga_share_above_lowpass * 100)} / "
                f"{round(self.pga_input_share_above_lowpass * 100)} %",
            ])
            note = self.pga_note()
            if note:
                lines.append(f"  {note}")
        lines.append("")
        if len(self.max_strain_pct) > 0:
            lines.append(f"  Max shear strain:     {np.max(self.max_strain_pct):.3f} %")
        if len(self.max_pore_pressure_ratio) > 0:
            ru = float(np.max(self.max_pore_pressure_ratio))
            # ru is not capped (GM-79): above 1 is stated, not hidden
            lines.append(f"  Max ru:               {ru:.3f}" + (
                " (above 1: the pore pressure above the initial total "
                "vertical stress)" if ru > 1.0 else ""))
        if self.solver_steps:
            st = self.solver_steps
            lines.append(
                f"  Dynamic steps:        {st.get('n_steps', 0)} "
                f"({st.get('fallback_steps', 0)} by a fallback algorithm, "
                f"{st.get('substepped_steps', 0)} in substeps)")
        if self.choices:
            from geotech_common.choices import choices_lines
            lines.append("")
            lines.extend(choices_lines(self.choices))
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "analysis_type": self.analysis_type,
            "total_depth_m": self.total_depth_m,
            "n_layers": self.n_layers,
            "site_period_s": round(self.site_period_s, 4),
            "motion_name": self.motion_name,
            "pga_input_g": round(self.pga_input_g, 4),
            "pga_surface_g": round(self.pga_surface_g, 4),
            "amplification_factor": round(self.amplification_factor, 3),
        }
        if self.pga_surface_lowpass_g is not None:
            # The raw PGA's alternatives, and whether motion above the
            # low-pass corner sets it (GM-264)
            d.update({
                "pga_time_s": round(self.pga_time_s, 4),
                "pga_lowpass_hz": self.pga_lowpass_hz,
                "pga_surface_lowpass_g": round(self.pga_surface_lowpass_g, 4),
                "pga_input_lowpass_g": round(self.pga_input_lowpass_g, 4),
                "pga_share_above_lowpass": round(
                    self.pga_share_above_lowpass, 3),
                "pga_input_share_above_lowpass": round(
                    self.pga_input_share_above_lowpass, 3),
                "pga_set_above_lowpass": self.pga_set_above_lowpass,
                "Sa_surface_0_01s_g": round(self.Sa_surface_0_01s_g, 4),
            })
            if self.pga_set_above_lowpass:
                d["pga_note"] = self.pga_note()
        if len(self.max_strain_pct) > 0:
            d["max_shear_strain_pct"] = round(float(np.max(self.max_strain_pct)), 4)
        if len(self.max_pore_pressure_ratio) > 0:
            d["max_ru"] = round(float(np.max(self.max_pore_pressure_ratio)), 4)
        if len(self.periods) > 0 and len(self.Sa_surface_g) > 0:
            # The 5 % spectra at a few periods (GM-260): in a sand that
            # dilates the surface PGA is a short spike the element size
            # sets, while Sa at 0.2 s and longer moves little with it
            T = np.log(np.asarray(self.periods, dtype=float))
            at = np.log(REPORTED_PERIODS_S)
            d["periods_s"] = list(REPORTED_PERIODS_S)
            d["Sa_surface_g"] = [round(float(v), 4) for v in
                                 np.interp(at, T, self.Sa_surface_g)]
            if len(self.Sa_input_g) > 0:
                d["Sa_input_g"] = [round(float(v), 4) for v in
                                   np.interp(at, T, self.Sa_input_g)]
        if self.solver_steps:
            d["solver_steps"] = dict(self.solver_steps)
        if self.rayleigh_freqs_hz:
            d["rayleigh_freqs_hz"] = [round(f, 4)
                                      for f in self.rayleigh_freqs_hz]
        if self.n_elements:
            d["n_elements"] = self.n_elements
        if self.layer_curves:
            d["layer_curves"] = [dict(c) for c in self.layer_curves]
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
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
