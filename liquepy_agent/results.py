"""
Result dataclasses for liquepy agent.

Three result types:
- CPTLiquefactionResult: Full CPT triggering + post-triggering analysis
- SPTLiquefactionResult: SPT-based B&I-2014 triggering (per-layer FoS, CRR/CSR)
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
        Minimum factor of safety over the readings named by
        ``min_fos_basis``.
    min_fos_basis : str
        Which readings ``min_fos`` is taken over: the susceptible ones
        (below the water table with Ic under the limit), or every reading
        when there is none (GM-324; ``warnings`` then says so).
    n_susceptible : int
        Readings below the water table with Ic under the limit.
    n_liquefiable : int
        Susceptible readings with a factor of safety below 1.0.
    max_settlement_mm : float
        Maximum 1D volumetric settlement (mm).
    warnings : list of str
        What the reader should know about the scalars above.

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
    choices : list of dict
        What the evaluation chose (GM-115, ``geotech_common.choices``).
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
    min_fos_basis: str = ""
    n_susceptible: int = 0
    n_liquefiable: int = 0
    max_settlement_mm: float = 0.0
    warnings: list = field(default_factory=list)

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
    choices: list = field(default_factory=list)

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
        d = {
            "n_points": self.n_points,
            "gwl_m": round(self.gwl_m, 2),
            "pga_g": round(self.pga_g, 3),
            "m_w": round(self.m_w, 1),
            "i_c_limit": round(self.i_c_limit, 2),
            "lpi": round(self.lpi, 2),
            "lsn": round(self.lsn, 2),
            "ldi_m": round(self.ldi_m, 4),
            "min_fos": round(self.min_fos, 3),
            "min_fos_basis": self.min_fos_basis,
            "n_susceptible": int(self.n_susceptible),
            "n_liquefiable": int(self.n_liquefiable),
            "max_settlement_mm": round(self.max_settlement_mm, 1),
        }
        if self.warnings:
            d["warnings"] = list(self.warnings)
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d

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
class SPTLiquefactionResult:
    """Result of SPT-based liquefaction triggering (Boulanger & Idriss 2014).

    Per-layer triggering evaluation using the B&I-2014 SPT procedure assembled
    from liquepy's tested building blocks (CRR from (N1)60cs, rd, K_sigma, MSF).

    Attributes — Scalars
    --------------------
    n_layers : int
        Number of evaluation layers.
    gwt_depth_m : float
        Groundwater table depth (m below surface).
    amax_g : float
        Peak ground acceleration (g).
    m_w : float
        Moment magnitude.
    min_fos : float
        Minimum factor of safety against liquefaction.
    n_liquefiable : int
        Number of layers with FoS < 1.0.

    Attributes — Arrays
    -------------------
    depth : ndarray
        Layer mid-depth (m).
    n1_60 : ndarray
        Corrected SPT blow count (N1)60.
    n1_60cs : ndarray
        Clean-sand equivalent (N1)60cs.
    fines_content : ndarray
        Fines content (%).
    sigma_v : ndarray
        Total vertical stress (kPa).
    sigma_veff : ndarray
        Effective vertical stress (kPa).
    csr : ndarray
        Cyclic stress ratio (B&I-2014).
    crr : ndarray
        Cyclic resistance ratio (magnitude- and overburden-corrected).
    factor_of_safety : ndarray
        Factor of safety against liquefaction (CRR/CSR).
    liquefiable : ndarray
        Boolean per layer (FoS < 1.0).
    rd, msf, k_sigma : ndarray
        The stress reduction factor inside CSR, and the magnitude scaling
        factor and overburden correction inside CRR, per layer (GM-115).
    choices : list of dict
        What the evaluation chose (``geotech_common.choices``).
    """

    # Scalars
    n_layers: int = 0
    gwt_depth_m: float = 0.0
    amax_g: float = 0.0
    m_w: float = 7.5
    min_fos: float = 0.0
    n_liquefiable: int = 0

    # Arrays
    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    n1_60: np.ndarray = field(default_factory=lambda: np.array([]))
    n1_60cs: np.ndarray = field(default_factory=lambda: np.array([]))
    fines_content: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_v: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_veff: np.ndarray = field(default_factory=lambda: np.array([]))
    csr: np.ndarray = field(default_factory=lambda: np.array([]))
    crr: np.ndarray = field(default_factory=lambda: np.array([]))
    factor_of_safety: np.ndarray = field(default_factory=lambda: np.array([]))
    liquefiable: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    rd: np.ndarray = field(default_factory=lambda: np.array([]))
    msf: np.ndarray = field(default_factory=lambda: np.array([]))
    k_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    choices: list = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary of results."""
        return (
            f"SPT Liquefaction (B&I 2014): {self.n_layers} layers, "
            f"amax={self.amax_g:.2f}g, Mw={self.m_w:.1f}, "
            f"min FoS={self.min_fos:.2f}, "
            f"{self.n_liquefiable} liquefiable"
        )

    def layer_results(self) -> list:
        """Per-layer results as a list of JSON-safe dicts."""
        factors = (len(self.rd) == len(self.msf) == len(self.k_sigma)
                   == self.n_layers)
        out = []
        for i in range(self.n_layers):
            row = {
                "depth_m": round(float(self.depth[i]), 2),
                "N1_60": round(float(self.n1_60[i]), 1),
                "N1_60cs": round(float(self.n1_60cs[i]), 1),
                "FC_pct": round(float(self.fines_content[i]), 1),
                "sigma_v_kPa": round(float(self.sigma_v[i]), 1),
                "sigma_v_eff_kPa": round(float(self.sigma_veff[i]), 1),
            }
            if factors:
                row.update({"rd": round(float(self.rd[i]), 4),
                            "MSF": round(float(self.msf[i]), 3),
                            "K_sigma": round(float(self.k_sigma[i]), 3)})
            row.update({
                "CSR": round(float(self.csr[i]), 4),
                "CRR": round(float(self.crr[i]), 4),
                "FOS_liq": round(float(self.factor_of_safety[i]), 3),
                "liquefiable": bool(self.liquefiable[i]),
            })
            out.append(row)
        return out

    def to_dict(self) -> dict:
        """Return JSON-serializable dict of scalar + per-layer results."""
        d = {
            "method": "bi2014",
            "input_type": "SPT",
            "n_layers": self.n_layers,
            "gwt_depth_m": round(self.gwt_depth_m, 2),
            "amax_g": round(self.amax_g, 3),
            "m_w": round(self.m_w, 1),
            "min_fos": round(self.min_fos, 3),
            "n_liquefiable": self.n_liquefiable,
            "layer_results": self.layer_results(),
        }
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d


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
        Estimated permeability, as liquepy's Robertson and Cabal (2012)
        function gives it. The unit in this name is not checked against
        a printed page (no reference in the library prints the relation),
        so the tool does not return it (GM-324).
    i_c : ndarray
        Soil behavior type index.
    """

    # Scalars
    n_points: int = 0
    gwl_m: float = 0.0
    vs_method: str = ""
    #: Ic below it: sand-like (the Dr range); at or above it: clay-like
    #: (the su/sigma'v range).
    i_c_limit: float = 2.6

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

    #: The profile's columns: result key, the array it comes from and the
    #: decimals kept. Permeability is left out: its unit is not checked
    #: against a printed page.
    PROFILE_COLUMNS = (
        ("Vs_m_per_s", "vs_m_per_s", 1),
        ("Dr", "relative_density", 3),
        ("su_ratio", "su_ratio", 3),
        ("I_c", "i_c", 3),
    )

    def profile(self, max_rows: int = 40):
        """The correlations by depth, as rows a tool result can carry
        (GM-324: the tool was told to return the profiles and returned
        only their ranges).

        Each reading stands for the depth halfway to its neighbours (the
        first and last half a spacing beyond their reading, the first not
        above the ground surface), as the CPT liquefaction result's
        ``depth_profile`` does. Up to ``max_rows`` readings give one row
        each; a longer sounding is summarised in bands of whole readings,
        about equally many in each: every value is the band's average over
        its finite readings.

        Returns
        -------
        (rows, note) : (list of dict, str or None)
            Rows of ``top_m``, ``bottom_m`` and the keys of
            :data:`PROFILE_COLUMNS` (None where no reading in the row is
            finite), in depth order; ``note`` says how the bands were made
            (None for one row per reading).
        """
        z = np.asarray(self.depth, dtype=float).ravel()
        n = len(z)
        if n == 0:
            return [], None
        order = np.argsort(z, kind="stable")
        z = z[order]
        edges = np.empty(n + 1)
        if n == 1:
            edges[:] = z[0]
        else:
            edges[0] = max(0.0, z[0] - 0.5 * (z[1] - z[0]))
            edges[1:-1] = 0.5 * (z[:-1] + z[1:])
            edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
        cols = []
        for key, attr, nd in self.PROFILE_COLUMNS:
            a = np.asarray(getattr(self, attr), dtype=float).ravel()
            cols.append((key, a[order] if len(a) == n else None, nd))

        def value(a, idx, nd):
            if a is None:
                return None
            v = a[idx]
            v = v[np.isfinite(v)]
            return round(float(np.mean(v)), nd) if len(v) else None

        chunks = (np.array_split(np.arange(n), max_rows) if n > max_rows
                  else [np.array([i]) for i in range(n)])
        rows = []
        for idx in chunks:
            row = {"top_m": round(float(edges[idx[0]]), 2),
                   "bottom_m": round(float(edges[idx[-1] + 1]), 2)}
            for key, a, nd in cols:
                row[key] = value(a, idx, nd)
            rows.append(row)
        note = None
        if n > max_rows:
            widest = max(r["bottom_m"] - r["top_m"] for r in rows)
            note = (
                f"{n} readings summarised in {len(rows)} depth bands, up to "
                f"about {widest:.2g} m thick, to keep the result small. "
                "Each value is the band's average over its readings.")
        return rows, note

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict: the ranges and the profile
        with depth (:meth:`profile`)."""
        d = {
            "n_points": self.n_points,
            "gwl_m": round(self.gwl_m, 2),
            "vs_method": self.vs_method,
        }

        def finite(a):
            a = np.asarray(a, dtype=float)
            return a[np.isfinite(a)]

        valid = finite(self.vs_m_per_s)
        if len(valid) > 0:
            d["vs_min_m_per_s"] = round(float(valid.min()), 1)
            d["vs_max_m_per_s"] = round(float(valid.max()), 1)
            d["vs_avg_m_per_s"] = round(float(valid.mean()), 1)
        # Dr is a sand correlation and su/sigma'v a clay one: each range is
        # taken over the readings of its own soil type by Ic, null when
        # there is none (GM-324 review: an su ratio maximum came from a
        # sand reading and a Dr minimum from a clay band).
        ic = np.asarray(self.i_c, dtype=float)

        def of_type(a, sand):
            a = np.asarray(a, dtype=float)
            if len(a) != len(ic):
                return a[:0]
            kind = ic < self.i_c_limit if sand else ic >= self.i_c_limit
            return a[np.isfinite(a) & np.isfinite(ic) & kind]

        for key, attr, sand in (("dr", "relative_density", True),
                                ("su_ratio", "su_ratio", False)):
            if len(np.asarray(getattr(self, attr))) == 0:
                continue
            valid = of_type(getattr(self, attr), sand)
            d[f"{key}_min"] = (round(float(valid.min()), 3) if len(valid)
                               else None)
            d[f"{key}_max"] = (round(float(valid.max()), 3) if len(valid)
                               else None)
        valid = finite(self.i_c)
        if len(valid) > 0:
            d["i_c_min"] = round(float(valid.min()), 3)
            d["i_c_max"] = round(float(valid.max()), 3)
        rows, note = self.profile()
        d["profile"] = rows
        if note:
            d["profile_note"] = note
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
