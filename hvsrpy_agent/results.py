"""
Result dataclasses for hvsrpy agent.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class HvsrResult:
    """Results from HVSR analysis.

    Attributes
    ----------
    f0_hz : float
        Resonant frequency (Hz) — mean of per-window peak frequencies.
    A0 : float
        Peak HVSR amplitude at f0.
    T0_s : float
        Site period (s) = 1 / f0.
    f0_std_hz : float
        Standard deviation of f0 across windows.
    A0_std : float
        Standard deviation of A0 across windows.
    n_windows : int
        Total number of time windows.
    n_valid_windows : int
        Number of windows passing rejection criteria.
    window_length_s : float
        Window length used (seconds).
    distribution : str
        Statistical distribution used ('lognormal' or 'normal').
    smoothing_operator : str
        Smoothing operator name.
    horizontal_method : str
        Method used to combine horizontal components.
    sesame_reliability : list
        SESAME (2004) reliability criteria results [3 bools].
    sesame_clarity : list
        SESAME (2004) clarity criteria results [6 bools].
    frequency : np.ndarray or None
        Frequency vector (Hz).
    mean_curve : np.ndarray or None
        Mean HVSR curve.
    std_curve : np.ndarray or None
        Standard deviation curve.
    upper_curve : np.ndarray or None
        Mean + 1 std curve.
    lower_curve : np.ndarray or None
        Mean - 1 std curve.
    """
    f0_hz: float = 0.0
    A0: float = 0.0
    T0_s: float = 0.0
    f0_std_hz: float = 0.0
    A0_std: float = 0.0
    n_windows: int = 0
    n_valid_windows: int = 0
    window_length_s: float = 0.0
    distribution: str = "lognormal"
    smoothing_operator: str = "konno_and_ohmachi"
    horizontal_method: str = "geometric_mean"
    sesame_reliability: List[float] = field(default_factory=list)
    sesame_clarity: List[float] = field(default_factory=list)
    frequency: Optional[np.ndarray] = None
    mean_curve: Optional[np.ndarray] = None
    std_curve: Optional[np.ndarray] = None
    upper_curve: Optional[np.ndarray] = None
    lower_curve: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  HVSR ANALYSIS RESULTS",
            "=" * 60,
            f"  Resonant frequency (f0): {self.f0_hz:.3f} Hz",
            f"  Site period (T0):        {self.T0_s:.3f} s",
            f"  Peak amplitude (A0):     {self.A0:.2f}",
            f"  f0 std deviation:        {self.f0_std_hz:.4f} Hz",
            f"  A0 std deviation:        {self.A0_std:.2f}",
            f"  Windows: {self.n_valid_windows}/{self.n_windows} valid",
            f"  Window length:           {self.window_length_s:.1f} s",
            f"  Distribution:            {self.distribution}",
            f"  Smoothing:               {self.smoothing_operator}",
            f"  Horizontal method:       {self.horizontal_method}",
        ]
        if self.sesame_reliability:
            rel_pass = int(sum(self.sesame_reliability))
            lines.append(f"  SESAME reliability:      {rel_pass}/3")
        if self.sesame_clarity:
            cla_pass = int(sum(self.sesame_clarity))
            lines.append(f"  SESAME clarity:          {cla_pass}/6")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "f0_hz": float(self.f0_hz),
            "A0": float(self.A0),
            "T0_s": float(self.T0_s),
            "f0_std_hz": float(self.f0_std_hz),
            "A0_std": float(self.A0_std),
            "n_windows": int(self.n_windows),
            "n_valid_windows": int(self.n_valid_windows),
            "window_length_s": float(self.window_length_s),
            "distribution": self.distribution,
            "smoothing_operator": self.smoothing_operator,
            "horizontal_method": self.horizontal_method,
            "sesame_reliability": [float(x) for x in self.sesame_reliability],
            "sesame_clarity": [float(x) for x in self.sesame_clarity],
            "sesame_reliability_pass": int(sum(self.sesame_reliability)) if self.sesame_reliability else 0,
            "sesame_clarity_pass": int(sum(self.sesame_clarity)) if self.sesame_clarity else 0,
        }
        if self.frequency is not None:
            d["frequency_hz"] = [float(x) for x in self.frequency]
        if self.mean_curve is not None:
            d["mean_curve"] = [float(x) for x in self.mean_curve]
        if self.std_curve is not None:
            d["std_curve"] = [float(x) for x in self.std_curve]
        return d

    def plot_hvsr(self, ax=None, show=True, **kwargs):
        """Plot mean HVSR curve with +/- 1 std bands."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        setup_engineering_plot(ax, "HVSR Curve", "Frequency (Hz)", "H/V Ratio")

        if self.frequency is not None and self.mean_curve is not None:
            ax.semilogx(self.frequency, self.mean_curve,
                        color='black', linewidth=1.5, label='Mean', **kwargs)
            if self.upper_curve is not None and self.lower_curve is not None:
                ax.fill_between(self.frequency, self.lower_curve,
                                self.upper_curve, alpha=0.3, color='gray',
                                label=r'$\pm$1 std')
            # Mark f0
            ax.axvline(self.f0_hz, color='red', linestyle='--', alpha=0.7,
                       label=f'f0 = {self.f0_hz:.2f} Hz')
            ax.legend(fontsize=8)

        if show:
            plt.show()
        return ax
