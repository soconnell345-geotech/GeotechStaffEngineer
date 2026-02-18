"""
Results container for downdrag analysis.

Provides a clean interface to access analysis results with summary()
for human-readable output and to_dict() for JSON serialization.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DowndragResult:
    """Container for downdrag analysis results.

    Attributes
    ----------
    neutral_plane_depth : float
        Depth of the neutral plane from pile head (m).
    dragload : float
        Negative skin friction above the neutral plane (kN).
    max_pile_load : float
        Maximum axial load in pile at neutral plane (kN).
        Equals Q_dead + dragload + pile_weight_to_np.
    Q_dead : float
        Applied dead load at pile head (kN).
    pile_weight_to_np : float
        Pile self-weight from head to neutral plane (kN).
    positive_skin_friction : float
        Positive shaft resistance below neutral plane (kN).
    toe_resistance : float
        Toe bearing resistance (kN).
    total_resistance : float
        Total resistance below NP = positive_skin + toe (kN).
    pile_settlement : float
        Pile settlement at neutral plane (m).
    elastic_shortening : float
        Elastic shortening of pile above NP (m).
    toe_settlement : float
        Settlement of bearing stratum below pile tip (m).
    soil_settlement_at_np : float
        Soil settlement at the neutral plane depth (m).
    z : numpy.ndarray
        Depth array along pile (m).
    axial_load : numpy.ndarray
        Axial load distribution Q(z) along pile (kN).
    soil_settlement_profile : numpy.ndarray
        Soil settlement at each depth (m).
    unit_skin_friction : numpy.ndarray
        Unit skin friction fs(z) along pile (kPa).
    structural_ok : bool or None
        True if structural check passes. None if not checked.
    structural_demand : float or None
        LRFD factored demand (kN) per UFC Eq 6-80:
        1.25*Q_dead + 1.10*(Q_np - Q_dead). None if not checked.
    geotechnical_ok : bool or None
        True if geotechnical check passes. None if not checked.
    settlement_ok : bool or None
        True if settlement is within allowable. None if not checked.
    pile_length : float
        Pile length (m).
    pile_diameter : float
        Pile diameter (m).
    """
    neutral_plane_depth: float
    dragload: float
    max_pile_load: float
    Q_dead: float
    pile_weight_to_np: float
    positive_skin_friction: float
    toe_resistance: float
    total_resistance: float
    pile_settlement: float
    elastic_shortening: float
    toe_settlement: float
    soil_settlement_at_np: float
    z: np.ndarray
    axial_load: np.ndarray
    soil_settlement_profile: np.ndarray
    unit_skin_friction: np.ndarray
    structural_ok: Optional[bool]
    structural_demand: Optional[float]
    geotechnical_ok: Optional[bool]
    settlement_ok: Optional[bool]
    pile_length: float
    pile_diameter: float

    def summary(self) -> str:
        """Return a text summary of key results."""
        lines = [
            "Downdrag Analysis Results (Fellenius Unified Method)",
            "=" * 52,
            f"Pile length:            {self.pile_length:.2f} m",
            f"Pile diameter:          {self.pile_diameter:.3f} m",
            f"Dead load (Q_dead):     {self.Q_dead:.1f} kN",
            "",
            "--- Neutral Plane ---",
            f"Neutral plane depth:    {self.neutral_plane_depth:.2f} m",
            f"Dragload:               {self.dragload:.1f} kN",
            f"Pile weight to NP:      {self.pile_weight_to_np:.1f} kN",
            f"Max pile load at NP:    {self.max_pile_load:.1f} kN",
            "",
            "--- Resistance Below NP ---",
            f"Positive skin friction: {self.positive_skin_friction:.1f} kN",
            f"Toe resistance:         {self.toe_resistance:.1f} kN",
            f"Total resistance:       {self.total_resistance:.1f} kN",
            "",
            "--- Settlement ---",
            f"Pile settlement:        {self.pile_settlement * 1000:.2f} mm",
            f"  Elastic shortening:   {self.elastic_shortening * 1000:.2f} mm",
            f"  Toe settlement:       {self.toe_settlement * 1000:.2f} mm",
            f"Soil settlement at NP:  {self.soil_settlement_at_np * 1000:.2f} mm",
        ]

        if any(x is not None for x in [self.structural_ok,
                                         self.geotechnical_ok,
                                         self.settlement_ok]):
            lines.append("")
            lines.append("--- Limit State Checks ---")
            if self.structural_ok is not None:
                status = "PASS" if self.structural_ok else "FAIL"
                lines.append(f"Structural (Eq 6-80):   {status}")
                if self.structural_demand is not None:
                    lines.append(
                        f"  LRFD demand:          {self.structural_demand:.1f} kN"
                    )
            if self.geotechnical_ok is not None:
                status = "PASS" if self.geotechnical_ok else "FAIL"
                lines.append(f"Geotechnical:           {status}")
            if self.settlement_ok is not None:
                status = "PASS" if self.settlement_ok else "FAIL"
                lines.append(f"Settlement:             {status}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export results as a dictionary (for JSON serialization)."""
        d = {
            'neutral_plane_depth_m': self.neutral_plane_depth,
            'dragload_kN': self.dragload,
            'max_pile_load_kN': self.max_pile_load,
            'Q_dead_kN': self.Q_dead,
            'pile_weight_to_np_kN': self.pile_weight_to_np,
            'positive_skin_friction_kN': self.positive_skin_friction,
            'toe_resistance_kN': self.toe_resistance,
            'total_resistance_kN': self.total_resistance,
            'pile_settlement_m': self.pile_settlement,
            'elastic_shortening_m': self.elastic_shortening,
            'toe_settlement_m': self.toe_settlement,
            'soil_settlement_at_np_m': self.soil_settlement_at_np,
            'pile_length_m': self.pile_length,
            'pile_diameter_m': self.pile_diameter,
            'z_m': self.z.tolist(),
            'axial_load_kN': self.axial_load.tolist(),
            'soil_settlement_mm': (self.soil_settlement_profile * 1000).tolist(),
            'unit_skin_friction_kPa': self.unit_skin_friction.tolist(),
        }
        if self.structural_ok is not None:
            d['structural_ok'] = self.structural_ok
            if self.structural_demand is not None:
                d['structural_demand_kN'] = self.structural_demand
        if self.geotechnical_ok is not None:
            d['geotechnical_ok'] = self.geotechnical_ok
        if self.settlement_ok is not None:
            d['settlement_ok'] = self.settlement_ok
        return d

    def plot_axial_load(self, ax=None, show=True, **kwargs):
        """Plot axial load Q(z) vs depth with neutral plane.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(self.axial_load, self.z, 'b-', linewidth=2, **kwargs)
        ax.axhline(y=self.neutral_plane_depth, color='r', linestyle='--',
                   linewidth=1.5,
                   label=f'Neutral Plane ({self.neutral_plane_depth:.1f} m)')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Axial Load Distribution",
                              "Axial Load Q(z) (kN)", "Depth (m)")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_settlement(self, ax=None, show=True, **kwargs):
        """Plot soil settlement profile vs depth with pile settlement.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(self.soil_settlement_profile * 1000, self.z, 'g-',
                linewidth=2, label='Soil settlement', **kwargs)
        ax.axvline(x=self.pile_settlement * 1000, color='b', linestyle='--',
                   linewidth=1.5,
                   label=f'Pile settlement ({self.pile_settlement*1000:.1f} mm)')
        ax.axhline(y=self.neutral_plane_depth, color='r', linestyle=':',
                   linewidth=1, label='Neutral Plane')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Settlement Profile",
                              "Settlement (mm)", "Depth (m)")
        ax.legend(fontsize=8)
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_neutral_plane(self, show=True):
        """Plot combined neutral plane diagram (load + settlement).

        Returns
        -------
        tuple of (fig, axes)
        """
        from geotech_common.plotting import get_pyplot
        plt = get_pyplot()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        self.plot_axial_load(ax=ax1, show=False)
        self.plot_settlement(ax=ax2, show=False)
        ax2.set_ylabel('')
        fig.suptitle('Downdrag Neutral Plane Analysis', fontsize=13,
                     fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, (ax1, ax2)
