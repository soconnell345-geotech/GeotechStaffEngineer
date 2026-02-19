"""
Results containers for seismic signal processing analyses.

Provides:
  - summary() -> str: human-readable text output
  - to_dict() -> dict: JSON-serializable dict for LLM agents
  - plot_*() methods: matplotlib visualizations
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

import numpy as np


@dataclass
class ResponseSpectrumResult:
    """Results from Nigam-Jennings response spectrum analysis.

    Attributes
    ----------
    motion_name : str
        Name of the input ground motion.
    n_points : int
        Number of points in the acceleration record.
    duration_s : float
        Total duration of the record (s).
    dt_s : float
        Time step (s).
    pga_g : float
        Peak ground acceleration (g).
    pgv_m_per_s : float
        Peak ground velocity (m/s).
    pgd_m : float
        Peak ground displacement (m).
    damping : float
        Spectral damping ratio (decimal).
    periods : numpy.ndarray
        Spectral periods (s).
    Sa_g : numpy.ndarray
        Spectral acceleration (g).
    time : numpy.ndarray
        Time array (s).
    accel_g : numpy.ndarray
        Input acceleration time history (g).
    """
    motion_name: str = ""
    n_points: int = 0
    duration_s: float = 0.0
    dt_s: float = 0.0
    pga_g: float = 0.0
    pgv_m_per_s: float = 0.0
    pgd_m: float = 0.0
    damping: float = 0.05
    periods: np.ndarray = field(default_factory=lambda: np.array([]))
    Sa_g: np.ndarray = field(default_factory=lambda: np.array([]))
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    accel_g: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  RESPONSE SPECTRUM (Nigam-Jennings)",
            "=" * 60,
            "",
            f"  Motion:               {self.motion_name}",
            f"  Duration:             {self.duration_s:.1f} s  ({self.n_points} pts, dt={self.dt_s:.4f} s)",
            f"  Damping:              {self.damping:.1%}",
            "",
            f"  PGA:                  {self.pga_g:.4f} g",
            f"  PGV:                  {self.pgv_m_per_s:.4f} m/s",
            f"  PGD:                  {self.pgd_m:.6f} m",
            "",
        ]
        if len(self.Sa_g) > 0:
            lines.append(f"  Peak Sa:              {np.max(self.Sa_g):.4f} g")
            idx = np.argmax(self.Sa_g)
            lines.append(f"  Peak Sa period:       {self.periods[idx]:.3f} s")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_name": self.motion_name,
            "n_points": self.n_points,
            "duration_s": round(self.duration_s, 2),
            "dt_s": round(self.dt_s, 6),
            "pga_g": round(self.pga_g, 4),
            "pgv_m_per_s": round(self.pgv_m_per_s, 4),
            "pgd_m": round(self.pgd_m, 6),
            "damping": self.damping,
        }

    def plot_spectrum(self, ax=None, show=True, **kwargs):
        """Plot spectral acceleration vs period (log-log)."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(self.periods, self.Sa_g, '-', linewidth=1.5, **kwargs)
        setup_engineering_plot(ax, f'Response Spectrum ({self.damping:.0%} damping)',
                              'Period (s)', 'Spectral Acceleration (g)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_time_history(self, ax=None, show=True, **kwargs):
        """Plot acceleration time history."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.time, self.accel_g, '-', linewidth=0.5, **kwargs)
        setup_engineering_plot(ax, 'Acceleration Time History',
                              'Time (s)', 'Acceleration (g)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Create a 2x1 composite plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of Axes
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        self.plot_time_history(ax=axes[0], show=False)
        self.plot_spectrum(ax=axes[1], show=False)
        fig.suptitle(f'Response Spectrum: {self.motion_name}, PGA={self.pga_g:.3f}g',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes


@dataclass
class IntensityMeasuresResult:
    """Results from earthquake intensity measure calculations.

    Attributes
    ----------
    motion_name : str
        Name of the input ground motion.
    n_points : int
        Number of points in the acceleration record.
    duration_s : float
        Total duration of the record (s).
    dt_s : float
        Time step (s).
    pga_g : float
        Peak ground acceleration (g).
    pgv_m_per_s : float
        Peak ground velocity (m/s).
    pgd_m : float
        Peak ground displacement (m).
    arias_intensity_m_per_s : float
        Total Arias intensity (m/s).
    significant_duration_s : float
        Significant duration (s).
    sig_dur_start : float
        Husid start fraction (e.g. 0.05 for D5-95).
    sig_dur_end : float
        Husid end fraction (e.g. 0.95 for D5-95).
    cav_m_per_s : float
        Cumulative Absolute Velocity (m/s).
    bracketed_duration_s : float
        Bracketed duration (s).
    arias_cumulative : numpy.ndarray
        Cumulative Arias intensity array (m/s).
    time : numpy.ndarray
        Time array (s).
    accel_g : numpy.ndarray
        Input acceleration time history (g).
    """
    motion_name: str = ""
    n_points: int = 0
    duration_s: float = 0.0
    dt_s: float = 0.0
    pga_g: float = 0.0
    pgv_m_per_s: float = 0.0
    pgd_m: float = 0.0
    arias_intensity_m_per_s: float = 0.0
    significant_duration_s: float = 0.0
    sig_dur_start: float = 0.05
    sig_dur_end: float = 0.95
    cav_m_per_s: float = 0.0
    bracketed_duration_s: float = 0.0
    arias_cumulative: np.ndarray = field(default_factory=lambda: np.array([]))
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    accel_g: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        start_pct = int(self.sig_dur_start * 100)
        end_pct = int(self.sig_dur_end * 100)
        lines = [
            "=" * 60,
            "  EARTHQUAKE INTENSITY MEASURES",
            "=" * 60,
            "",
            f"  Motion:               {self.motion_name}",
            f"  Duration:             {self.duration_s:.1f} s  ({self.n_points} pts)",
            "",
            f"  PGA:                  {self.pga_g:.4f} g",
            f"  PGV:                  {self.pgv_m_per_s:.4f} m/s",
            f"  PGD:                  {self.pgd_m:.6f} m",
            "",
            f"  Arias Intensity:      {self.arias_intensity_m_per_s:.4f} m/s",
            f"  Significant Duration: {self.significant_duration_s:.2f} s  (D{start_pct}-{end_pct})",
            f"  CAV:                  {self.cav_m_per_s:.4f} m/s",
            f"  Bracketed Duration:   {self.bracketed_duration_s:.2f} s",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_name": self.motion_name,
            "pga_g": round(self.pga_g, 4),
            "pgv_m_per_s": round(self.pgv_m_per_s, 4),
            "pgd_m": round(self.pgd_m, 6),
            "arias_intensity_m_per_s": round(self.arias_intensity_m_per_s, 4),
            "significant_duration_s": round(self.significant_duration_s, 2),
            "cav_m_per_s": round(self.cav_m_per_s, 4),
            "bracketed_duration_s": round(self.bracketed_duration_s, 2),
        }

    def plot_arias(self, ax=None, show=True, **kwargs):
        """Plot cumulative Arias intensity vs time."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.time, self.arias_cumulative, '-', linewidth=1.0, **kwargs)
        setup_engineering_plot(ax, 'Cumulative Arias Intensity',
                              'Time (s)', 'Arias Intensity (m/s)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_time_history(self, ax=None, show=True, **kwargs):
        """Plot acceleration time history."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.time, self.accel_g, '-', linewidth=0.5, **kwargs)
        setup_engineering_plot(ax, 'Acceleration Time History',
                              'Time (s)', 'Acceleration (g)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Create a 2x1 composite plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of Axes
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        self.plot_time_history(ax=axes[0], show=False)
        self.plot_arias(ax=axes[1], show=False)
        fig.suptitle(f'Intensity Measures: {self.motion_name}, '
                     f'Ia={self.arias_intensity_m_per_s:.3f} m/s',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes


@dataclass
class RotDSpectrumResult:
    """Results from rotated spectral acceleration analysis (RotD50/RotD100).

    Attributes
    ----------
    motion_a_name : str
        Name of component A ground motion.
    motion_b_name : str
        Name of component B ground motion.
    n_points : int
        Number of points in the records.
    dt_s : float
        Time step (s).
    pga_a_g : float
        Peak ground acceleration of component A (g).
    pga_b_g : float
        Peak ground acceleration of component B (g).
    damping : float
        Spectral damping ratio (decimal).
    percentiles : list
        Percentiles computed (e.g. [0, 50, 100]).
    periods : numpy.ndarray
        Spectral periods (s).
    rotd0 : numpy.ndarray
        RotD0 — minimum orientation spectral acceleration (g).
    rotd50 : numpy.ndarray
        RotD50 — median orientation spectral acceleration (g).
    rotd100 : numpy.ndarray
        RotD100 — maximum orientation spectral acceleration (g).
    """
    motion_a_name: str = ""
    motion_b_name: str = ""
    n_points: int = 0
    dt_s: float = 0.0
    pga_a_g: float = 0.0
    pga_b_g: float = 0.0
    damping: float = 0.05
    percentiles: List[int] = field(default_factory=lambda: [0, 50, 100])
    periods: np.ndarray = field(default_factory=lambda: np.array([]))
    rotd0: np.ndarray = field(default_factory=lambda: np.array([]))
    rotd50: np.ndarray = field(default_factory=lambda: np.array([]))
    rotd100: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ROTATED SPECTRAL ACCELERATION (RotD)",
            "=" * 60,
            "",
            f"  Component A:          {self.motion_a_name}",
            f"  Component B:          {self.motion_b_name}",
            f"  Record length:        {self.n_points} pts, dt={self.dt_s:.4f} s",
            f"  Damping:              {self.damping:.1%}",
            "",
            f"  PGA (comp A):         {self.pga_a_g:.4f} g",
            f"  PGA (comp B):         {self.pga_b_g:.4f} g",
            "",
        ]
        if len(self.rotd50) > 0:
            lines.append(f"  Peak RotD50:          {np.max(self.rotd50):.4f} g")
        if len(self.rotd100) > 0:
            lines.append(f"  Peak RotD100:         {np.max(self.rotd100):.4f} g")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "motion_a_name": self.motion_a_name,
            "motion_b_name": self.motion_b_name,
            "n_points": self.n_points,
            "dt_s": round(self.dt_s, 6),
            "pga_a_g": round(self.pga_a_g, 4),
            "pga_b_g": round(self.pga_b_g, 4),
            "damping": self.damping,
        }
        if len(self.rotd50) > 0:
            d["peak_rotd50_g"] = round(float(np.max(self.rotd50)), 4)
        if len(self.rotd100) > 0:
            d["peak_rotd100_g"] = round(float(np.max(self.rotd100)), 4)
        return d

    def plot_rotd(self, ax=None, show=True, **kwargs):
        """Plot RotD0/RotD50/RotD100 spectra overlay."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        if len(self.rotd0) > 0:
            ax.loglog(self.periods, self.rotd0, '--', label='RotD0',
                      linewidth=1.0, color='C0')
        if len(self.rotd50) > 0:
            ax.loglog(self.periods, self.rotd50, '-', label='RotD50',
                      linewidth=1.5, color='C1')
        if len(self.rotd100) > 0:
            ax.loglog(self.periods, self.rotd100, '-', label='RotD100',
                      linewidth=1.5, color='C2')
        ax.legend()
        setup_engineering_plot(ax, f'Rotated Response Spectra ({self.damping:.0%} damping)',
                              'Period (s)', 'Spectral Acceleration (g)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Create RotD spectrum plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : Axes
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_rotd(ax=ax, show=False)
        fig.suptitle(f'RotD Spectra: {self.motion_a_name} + {self.motion_b_name}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax


@dataclass
class SignalProcessingResult:
    """Results from signal processing (filtering + baseline correction).

    Attributes
    ----------
    motion_name : str
        Name of the input ground motion.
    n_points : int
        Number of points in the record.
    dt_s : float
        Time step (s).
    bandpass_hz : list
        Bandpass frequencies [f_low, f_high] (Hz), or empty if not applied.
    baseline_order : int
        Polynomial order for baseline correction (-1 if not applied).
    pga_original_g : float
        Peak ground acceleration of original record (g).
    pga_processed_g : float
        Peak ground acceleration of processed record (g).
    pgv_processed_m_per_s : float
        Peak ground velocity of processed record (m/s).
    pgd_processed_m : float
        Peak ground displacement of processed record (m).
    time : numpy.ndarray
        Time array (s).
    accel_original_g : numpy.ndarray
        Original acceleration time history (g).
    accel_processed_g : numpy.ndarray
        Processed acceleration time history (g).
    velocity_m_per_s : numpy.ndarray
        Velocity time history from processed record (m/s).
    displacement_m : numpy.ndarray
        Displacement time history from processed record (m).
    """
    motion_name: str = ""
    n_points: int = 0
    dt_s: float = 0.0
    bandpass_hz: List[float] = field(default_factory=list)
    baseline_order: int = -1
    pga_original_g: float = 0.0
    pga_processed_g: float = 0.0
    pgv_processed_m_per_s: float = 0.0
    pgd_processed_m: float = 0.0
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    accel_original_g: np.ndarray = field(default_factory=lambda: np.array([]))
    accel_processed_g: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity_m_per_s: np.ndarray = field(default_factory=lambda: np.array([]))
    displacement_m: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SIGNAL PROCESSING RESULTS",
            "=" * 60,
            "",
            f"  Motion:               {self.motion_name}",
            f"  Record length:        {self.n_points} pts, dt={self.dt_s:.4f} s",
            "",
        ]
        if self.bandpass_hz:
            lines.append(f"  Bandpass filter:      {self.bandpass_hz[0]:.2f} - {self.bandpass_hz[1]:.2f} Hz")
        if self.baseline_order >= 0:
            lines.append(f"  Baseline correction:  order {self.baseline_order}")
        lines.extend([
            "",
            f"  PGA (original):       {self.pga_original_g:.4f} g",
            f"  PGA (processed):      {self.pga_processed_g:.4f} g",
            f"  PGV (processed):      {self.pgv_processed_m_per_s:.4f} m/s",
            f"  PGD (processed):      {self.pgd_processed_m:.6f} m",
            "",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_name": self.motion_name,
            "n_points": self.n_points,
            "dt_s": round(self.dt_s, 6),
            "bandpass_hz": self.bandpass_hz if self.bandpass_hz else None,
            "baseline_order": self.baseline_order if self.baseline_order >= 0 else None,
            "pga_original_g": round(self.pga_original_g, 4),
            "pga_processed_g": round(self.pga_processed_g, 4),
            "pgv_processed_m_per_s": round(self.pgv_processed_m_per_s, 4),
            "pgd_processed_m": round(self.pgd_processed_m, 6),
        }

    def plot_comparison(self, ax=None, show=True, **kwargs):
        """Plot original vs processed acceleration time histories."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.time, self.accel_original_g, '-', linewidth=0.5,
                alpha=0.5, label='Original', color='C0')
        ax.plot(self.time, self.accel_processed_g, '-', linewidth=0.5,
                label='Processed', color='C1')
        ax.legend()
        setup_engineering_plot(ax, 'Acceleration Comparison',
                              'Time (s)', 'Acceleration (g)')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_vel_disp(self, ax=None, show=True, **kwargs):
        """Plot velocity and displacement time histories (2x1 subplots)."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        else:
            return ax

        axes[0].plot(self.time, self.velocity_m_per_s, '-', linewidth=0.5)
        setup_engineering_plot(axes[0], 'Velocity', 'Time (s)', 'Velocity (m/s)')

        axes[1].plot(self.time, self.displacement_m, '-', linewidth=0.5)
        setup_engineering_plot(axes[1], 'Displacement', 'Time (s)', 'Displacement (m)')

        plt.tight_layout()
        if show:
            plt.show()
        return axes

    def plot_all(self, show=True):
        """Create a 2x2 composite plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Top-left: original accel
        axes[0, 0].plot(self.time, self.accel_original_g, '-', linewidth=0.5)
        setup_engineering_plot(axes[0, 0], 'Original',
                              'Time (s)', 'Acceleration (g)')

        # Top-right: processed accel
        axes[0, 1].plot(self.time, self.accel_processed_g, '-', linewidth=0.5)
        setup_engineering_plot(axes[0, 1], 'Processed',
                              'Time (s)', 'Acceleration (g)')

        # Bottom-left: velocity
        axes[1, 0].plot(self.time, self.velocity_m_per_s, '-', linewidth=0.5)
        setup_engineering_plot(axes[1, 0], 'Velocity',
                              'Time (s)', 'Velocity (m/s)')

        # Bottom-right: displacement
        axes[1, 1].plot(self.time, self.displacement_m, '-', linewidth=0.5)
        setup_engineering_plot(axes[1, 1], 'Displacement',
                              'Time (s)', 'Displacement (m)')

        fig.suptitle(f'Signal Processing: {self.motion_name}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes
