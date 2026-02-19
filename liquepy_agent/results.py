"""
Result dataclasses for liquepy agent.

Two result types:
- CPTLiquefactionResult: Full triggering + post-triggering analysis
- FieldCorrelationsResult: Vs, Dr, su/σv', permeability from CPT
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CPTLiquefactionResult:
    """Result of CPT-based liquefaction triggering analysis (B&I 2014).

    Attributes — Scalars
    --------------------
    n_points : int
        Number of CPT data points.
    gwl_m : float
        Groundwater level (m below surface).
    pga_g : float
        Peak ground acceleration (g).
    m_w : float
        Moment magnitude.
    i_c_limit : float
        I_c limit for liquefiable material.
    lpi : float
        Liquefaction Potential Index (Iwasaki et al.).
    lsn : float
        Liquefaction Severity Number.
    ldi_m : float
        Lateral Displacement Index (m).
    min_fos : float
        Minimum factor of safety in liquefiable zone.
    max_settlement_mm : float
        Maximum 1D volumetric settlement (mm).

    Attributes — Arrays
    -------------------
    depth : ndarray
        Depth (m).
    factor_of_safety : ndarray
        Factor of safety against liquefaction.
    csr : ndarray
        Cyclic stress ratio.
    crr : ndarray
        Cyclic resistance ratio.
    q_c1n_cs : ndarray
        Clean sand equivalent normalized cone resistance.
    i_c : ndarray
        Soil behavior type index.
    fines_content : ndarray
        Estimated fines content (%).
    sigma_v : ndarray
        Total vertical stress (kPa).
    sigma_veff : ndarray
        Effective vertical stress (kPa).
    volumetric_strain : ndarray
        Post-liquefaction volumetric strain (decimal).
    shear_strain : ndarray
        Post-liquefaction shear strain (decimal).
    relative_density : ndarray
        Estimated relative density (decimal, 0-1).
    """

    # Scalars
    n_points: int = 0
    gwl_m: float = 0.0
    pga_g: float = 0.0
    m_w: float = 7.5
    i_c_limit: float = 2.6
    lpi: float = 0.0
    lsn: float = 0.0
    ldi_m: float = 0.0
    min_fos: float = 0.0
    max_settlement_mm: float = 0.0

    # Arrays
    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    factor_of_safety: np.ndarray = field(default_factory=lambda: np.array([]))
    csr: np.ndarray = field(default_factory=lambda: np.array([]))
    crr: np.ndarray = field(default_factory=lambda: np.array([]))
    q_c1n_cs: np.ndarray = field(default_factory=lambda: np.array([]))
    i_c: np.ndarray = field(default_factory=lambda: np.array([]))
    fines_content: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_v: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_veff: np.ndarray = field(default_factory=lambda: np.array([]))
    volumetric_strain: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_strain: np.ndarray = field(default_factory=lambda: np.array([]))
    relative_density: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        """One-line summary of results."""
        return (
            f"CPT Liquefaction (B&I 2014): {self.n_points} pts, "
            f"PGA={self.pga_g:.2f}g, Mw={self.m_w:.1f}, "
            f"LPI={self.lpi:.1f}, LSN={self.lsn:.1f}, "
            f"min FoS={self.min_fos:.2f}"
        )

    def to_dict(self) -> dict:
        """Return JSON-serializable dict of scalar results."""
        return {
            "n_points": self.n_points,
            "gwl_m": round(self.gwl_m, 2),
            "pga_g": round(self.pga_g, 3),
            "m_w": round(self.m_w, 1),
            "i_c_limit": round(self.i_c_limit, 2),
            "lpi": round(self.lpi, 2),
            "lsn": round(self.lsn, 2),
            "ldi_m": round(self.ldi_m, 4),
            "min_fos": round(self.min_fos, 3),
            "max_settlement_mm": round(self.max_settlement_mm, 1),
        }

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_fos(self, ax=None, show=True, **kwargs):
        """Plot factor of safety vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 8))
        ax.plot(self.factor_of_safety, self.depth, 'b-', linewidth=1.5, **kwargs)
        ax.axvline(x=1.0, color='r', linestyle='--', linewidth=1, label='FoS = 1.0')
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlim(left=0)
        setup_engineering_plot(ax, 'Liquefaction Factor of Safety',
                               'Factor of Safety', 'Depth (m)')
        if show:
            plt.show()
        return ax

    def plot_csr_crr(self, ax=None, show=True, **kwargs):
        """Plot CSR and CRR vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 8))
        ax.plot(self.csr, self.depth, 'r-', linewidth=1.5, label='CSR')
        ax.plot(self.crr, self.depth, 'b-', linewidth=1.5, label='CRR')
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlim(left=0)
        setup_engineering_plot(ax, 'CSR vs CRR',
                               'Cyclic Stress / Resistance Ratio', 'Depth (m)')
        if show:
            plt.show()
        return ax

    def plot_ic(self, ax=None, show=True, **kwargs):
        """Plot soil behavior type index (Ic) vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 8))
        ax.plot(self.i_c, self.depth, 'g-', linewidth=1.5, **kwargs)
        ax.axvline(x=self.i_c_limit, color='r', linestyle='--', linewidth=1,
                    label=f'Ic limit = {self.i_c_limit}')
        ax.invert_yaxis()
        ax.legend()
        setup_engineering_plot(ax, 'Soil Behavior Type Index',
                               'Soil Behavior Type Index, Ic', 'Depth (m)')
        if show:
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Multi-panel plot: FoS, CSR/CRR, Ic, strains."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True)

        # FoS
        self.plot_fos(ax=axes[0], show=False)

        # CSR/CRR
        self.plot_csr_crr(ax=axes[1], show=False)

        # Ic
        self.plot_ic(ax=axes[2], show=False)

        # Strains
        if len(self.volumetric_strain) > 0:
            axes[3].plot(self.volumetric_strain * 100, self.depth, 'b-',
                         linewidth=1.5, label='Vol. strain (%)')
        if len(self.shear_strain) > 0:
            axes[3].plot(self.shear_strain * 100, self.depth, 'r-',
                         linewidth=1.5, label='Shear strain (%)')
        axes[3].invert_yaxis()
        axes[3].legend()
        setup_engineering_plot(axes[3], 'Post-Liquefaction Strains',
                               'Strain (%)', 'Depth (m)')

        fig.suptitle(self.summary(), fontsize=10)
        fig.tight_layout()
        if show:
            plt.show()
        return axes


@dataclass
class FieldCorrelationsResult:
    """Result of CPT field correlation analysis.

    Attributes — Scalars
    --------------------
    n_points : int
        Number of CPT data points.
    gwl_m : float
        Groundwater level (m below surface).
    vs_method : str
        Shear wave velocity correlation method used.

    Attributes — Arrays
    -------------------
    depth : ndarray
        Depth (m).
    vs_m_per_s : ndarray
        Shear wave velocity (m/s).
    relative_density : ndarray
        Relative density (decimal, 0-1).
    su_ratio : ndarray
        Undrained strength ratio su/σv'.
    permeability_cm_per_s : ndarray
        Estimated permeability (cm/s).
    i_c : ndarray
        Soil behavior type index.
    """

    # Scalars
    n_points: int = 0
    gwl_m: float = 0.0
    vs_method: str = ""

    # Arrays
    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    vs_m_per_s: np.ndarray = field(default_factory=lambda: np.array([]))
    relative_density: np.ndarray = field(default_factory=lambda: np.array([]))
    su_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    permeability_cm_per_s: np.ndarray = field(default_factory=lambda: np.array([]))
    i_c: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        """One-line summary of results."""
        vs_range = ""
        if len(self.vs_m_per_s) > 0:
            valid = self.vs_m_per_s[np.isfinite(self.vs_m_per_s)]
            if len(valid) > 0:
                vs_range = f", Vs={valid.min():.0f}-{valid.max():.0f} m/s"
        return (
            f"Field Correlations: {self.n_points} pts, "
            f"Vs method={self.vs_method}{vs_range}"
        )

    def to_dict(self) -> dict:
        """Return JSON-serializable dict of scalar results."""
        d = {
            "n_points": self.n_points,
            "gwl_m": round(self.gwl_m, 2),
            "vs_method": self.vs_method,
        }
        if len(self.vs_m_per_s) > 0:
            valid = self.vs_m_per_s[np.isfinite(self.vs_m_per_s)]
            if len(valid) > 0:
                d["vs_min_m_per_s"] = round(float(valid.min()), 1)
                d["vs_max_m_per_s"] = round(float(valid.max()), 1)
                d["vs_avg_m_per_s"] = round(float(valid.mean()), 1)
        if len(self.relative_density) > 0:
            valid = self.relative_density[np.isfinite(self.relative_density)]
            if len(valid) > 0:
                d["dr_min"] = round(float(valid.min()), 3)
                d["dr_max"] = round(float(valid.max()), 3)
        return d

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_vs(self, ax=None, show=True, **kwargs):
        """Plot shear wave velocity vs depth."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 8))
        ax.plot(self.vs_m_per_s, self.depth, 'b-', linewidth=1.5, **kwargs)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        setup_engineering_plot(ax, f'Vs Profile ({self.vs_method})',
                               'Shear Wave Velocity, Vs (m/s)', 'Depth (m)')
        if show:
            plt.show()
        return ax

    def plot_all(self, show=True):
        """Multi-panel plot: Vs, Dr, Ic."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True)

        # Vs
        self.plot_vs(ax=axes[0], show=False)

        # Dr
        if len(self.relative_density) > 0:
            axes[1].plot(self.relative_density, self.depth, 'g-', linewidth=1.5)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1)
        setup_engineering_plot(axes[1], 'Relative Density',
                               'Relative Density, Dr', 'Depth (m)')

        # Ic
        if len(self.i_c) > 0:
            axes[2].plot(self.i_c, self.depth, 'r-', linewidth=1.5)
        axes[2].invert_yaxis()
        setup_engineering_plot(axes[2], 'Ic Profile',
                               'Soil Behavior Type Index, Ic', 'Depth (m)')

        fig.suptitle(self.summary(), fontsize=10)
        fig.tight_layout()
        if show:
            plt.show()
        return axes
