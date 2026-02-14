"""
Results container for settlement analysis.

Stores computed settlements and provides summary output and
optional time-settlement plotting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np


@dataclass
class SettlementResult:
    """Results from a settlement analysis.

    Attributes
    ----------
    immediate : float
        Immediate (elastic) settlement (m).
    consolidation : float
        Primary consolidation settlement (m).
    secondary : float
        Secondary compression settlement (m).
    total : float
        Total settlement = immediate + consolidation + secondary (m).
    consolidation_layers : list of dict, optional
        Per-layer consolidation settlement breakdown.
    time_settlement_curve : list of tuple, optional
        (time_years, settlement_m) pairs for plotting.
    """
    immediate: float = 0.0
    consolidation: float = 0.0
    secondary: float = 0.0
    total: float = 0.0
    immediate_method: str = ""
    consolidation_method: str = ""
    stress_method: str = ""
    consolidation_layers: Optional[List[Dict[str, Any]]] = None
    time_settlement_curve: Optional[List[Tuple[float, float]]] = None

    def summary(self) -> str:
        """Return a formatted summary string.

        Returns
        -------
        str
            Human-readable summary of settlement results.
        """
        lines = [
            "=" * 60,
            "  SETTLEMENT ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Immediate settlement:     {self.immediate*1000:8.1f} mm"
            f"  ({self._pct(self.immediate)}%)",
            f"  Consolidation settlement: {self.consolidation*1000:8.1f} mm"
            f"  ({self._pct(self.consolidation)}%)",
            f"  Secondary settlement:     {self.secondary*1000:8.1f} mm"
            f"  ({self._pct(self.secondary)}%)",
            f"  {'-'*44}",
            f"  TOTAL settlement:         {self.total*1000:8.1f} mm",
        ]

        if self.immediate_method:
            lines.append(f"\n  Immediate method: {self.immediate_method}")
        if self.stress_method:
            lines.append(f"  Stress distribution: {self.stress_method}")

        if self.consolidation_layers:
            lines.append("\n  Consolidation Layer Breakdown:")
            for i, layer in enumerate(self.consolidation_layers):
                lines.append(
                    f"    Layer {i+1}: Sc={layer['settlement_mm']:.1f} mm, "
                    f"delta_sigma={layer['delta_sigma_kPa']:.1f} kPa, "
                    f"OCR={layer.get('OCR', 1.0):.1f}"
                )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _pct(self, component: float) -> str:
        if self.total > 0:
            return f"{100 * component / self.total:.0f}"
        return "0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for LLM agent consumption.

        Returns
        -------
        dict
            All result fields as a flat dictionary.
        """
        return {
            "immediate_mm": round(self.immediate * 1000, 2),
            "consolidation_mm": round(self.consolidation * 1000, 2),
            "secondary_mm": round(self.secondary * 1000, 2),
            "total_mm": round(self.total * 1000, 2),
            "immediate_method": self.immediate_method,
            "stress_method": self.stress_method,
        }

    def plot_time_settlement(self, ax=None):
        """Plot settlement vs time curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if self.time_settlement_curve is None:
            raise ValueError("No time-settlement data available. "
                             "Run analysis with time_rate parameters.")

        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        times = [t for t, s in self.time_settlement_curve]
        settlements = [s * 1000 for t, s in self.time_settlement_curve]

        ax.plot(times, settlements, 'b-', linewidth=2)
        ax.invert_yaxis()
        setup_engineering_plot(
            ax,
            title="Settlement vs Time",
            xlabel="Time (years)",
            ylabel="Settlement (mm)"
        )
        return ax
