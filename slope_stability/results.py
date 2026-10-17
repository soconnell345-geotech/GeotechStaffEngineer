"""
Result containers for slope stability analysis.

Each dataclass stores analysis outputs and provides:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption

All units SI: meters, kPa, kN/m, degrees.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


def fos_text(fos: Optional[float], fmt: str = ".3f") -> str:
    """A factor of safety for display: the number, or 'no solution' when
    the method found no equilibrium solution (FOS None, GM-81)."""
    return "no solution" if fos is None else format(fos, fmt)


@dataclass
class SliceData:
    """Per-slice data for detailed output and plotting.

    Attributes
    ----------
    x_mid : float
        x-coordinate of slice midpoint (m).
    z_top : float
        Ground surface elevation at midpoint (m).
    z_base : float
        Slip surface elevation at midpoint (m).
    width : float
        Slice width b (m).
    height : float
        Slice height h (m).
    alpha_deg : float
        Base inclination angle (degrees).
    weight : float
        Total slice weight W (kN/m).
    pore_pressure : float
        Pore water pressure at base u (kPa).
    c : float
        Cohesion at base (kPa).
    phi : float
        Friction angle at base (degrees).
    base_length : float
        Length of slice base dl (m).
    normal_stress_kPa : float
        Effective normal stress on base = N' / dl (kPa).
    shear_stress_kPa : float
        Mobilized shear stress on base = S_mobilized / dl (kPa).
        Positive = driving (base slopes downhill), negative = resisting.
    shear_resistance_kPa : float
        Available shear resistance on base = T_available / dl (kPa).
    N_eff_kN : float
        Effective normal FORCE on the base N' (kN/m). From the rigorous
        GLE solution for Spencer/M-P/GLE; Fellenius decomposition
        (W*cos(a) - u*l) otherwise.
    S_mob_kN : float
        Mobilized shear force on the base (kN/m).
    U_base_kN : float
        Pore-water force on the base u*l (kN/m).
    E_left_kN, E_right_kN : float or None
        Interslice normal force on the left/right boundary (kN/m).
        Only available from the rigorous GLE engine (Spencer / M-P /
        GLE methods); None otherwise.
    X_left_kN, X_right_kN : float or None
        Interslice shear force on the left/right boundary (kN/m).
        Only available from the rigorous GLE engine; None otherwise.
    """
    x_mid: float = 0.0
    z_top: float = 0.0
    z_base: float = 0.0
    width: float = 0.0
    height: float = 0.0
    alpha_deg: float = 0.0
    weight: float = 0.0
    pore_pressure: float = 0.0
    c: float = 0.0
    phi: float = 0.0
    base_length: float = 0.0
    normal_stress_kPa: float = 0.0
    shear_stress_kPa: float = 0.0
    shear_resistance_kPa: float = 0.0
    in_tension_crack: bool = False
    N_eff_kN: float = 0.0
    S_mob_kN: float = 0.0
    U_base_kN: float = 0.0
    E_left_kN: Optional[float] = None
    E_right_kN: Optional[float] = None
    X_left_kN: Optional[float] = None
    X_right_kN: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "x_mid_m": round(self.x_mid, 3),
            "z_top_m": round(self.z_top, 3),
            "z_base_m": round(self.z_base, 3),
            "width_m": round(self.width, 3),
            "height_m": round(self.height, 3),
            "alpha_deg": round(self.alpha_deg, 2),
            "weight_kN_per_m": round(self.weight, 1),
            "pore_pressure_kPa": round(self.pore_pressure, 1),
            "c_kPa": round(self.c, 1),
            "phi_deg": round(self.phi, 1),
            "base_length_m": round(self.base_length, 3),
            "normal_stress_kPa": round(self.normal_stress_kPa, 2),
            "shear_stress_kPa": round(self.shear_stress_kPa, 2),
            "shear_resistance_kPa": round(self.shear_resistance_kPa, 2),
            "N_eff_kN_per_m": round(self.N_eff_kN, 2),
            "S_mob_kN_per_m": round(self.S_mob_kN, 2),
            "U_base_kN_per_m": round(self.U_base_kN, 2),
        }
        if self.E_left_kN is not None:
            d["E_left_kN_per_m"] = round(self.E_left_kN, 2)
            d["E_right_kN_per_m"] = round(self.E_right_kN, 2)
            d["X_left_kN_per_m"] = round(self.X_left_kN, 2)
            d["X_right_kN_per_m"] = round(self.X_right_kN, 2)
        if self.in_tension_crack:
            d["in_tension_crack"] = True
        return d


@dataclass
class SlopeStabilityResult:
    """Results from slope stability analysis.

    Attributes
    ----------
    FOS : float or None
        Factor of safety. None when the method finds no equilibrium
        solution on the surface and nothing may stand in for it (supports
        cross the surface: the legacy approximate solver ignores them).
        See ``method_solved`` and ``warnings``.
    method : str
        Analysis method ('Fellenius', 'Bishop', 'Spencer', 'Morgenstern-Price').
    method_solved : bool
        False when the method (Spencer / Morgenstern-Price / GLE) finds no
        equilibrium solution on this surface: its rigorous GLE solution
        does not converge (GM-81).
    method_used : str or None
        What produced ``FOS``: '' (default) for the method itself; the
        legacy approximate solver's name when it stood in (a surface no
        support crosses); None when nothing did (``FOS`` is None).
    methods_without_solution : list of str
        The rigorous methods ('Spencer', 'Morgenstern-Price') whose FOS
        field on this result holds no converged solution (the headline or a
        compare_methods value): the legacy approximate solver's value, or
        None.
    warnings : list of str
        Plain statements of what the engineer must check on this surface: a
        method with no equilibrium solution, a reverse-curvature circle
        closed by a crack, supports a Bishop / Ordinary Method of Slices
        circle counts little of (GM-59, GM-81).
    xc : float
        Circle center x-coordinate (m). 0 for noncircular.
    yc : float
        Circle center z-coordinate / elevation (m). 0 for noncircular.
    radius : float
        Circle radius (m). 0 for noncircular.
    x_entry : float
        Slip surface entry x-coordinate (m).
    x_exit : float
        Slip surface exit x-coordinate (m).
    theta_spencer : float or None
        Interslice force angle from Spencer (degrees).
    FOS_fellenius : float or None
        Fellenius FOS for comparison.
    FOS_bishop : float or None
        Bishop FOS for comparison (circular only).
    FOS_spencer : float or None
        Spencer FOS for comparison.
    FOS_morgenstern_price : float or None
        Morgenstern-Price FOS for comparison.
    lambda_mp : float or None
        Morgenstern-Price interslice force scaling factor.
    n_slices : int
        Number of slices used.
    has_seismic : bool
        Whether seismic load was applied.
    kh : float
        Horizontal seismic coefficient.
    slice_data : list of SliceData or None
        Per-slice breakdown (force table: W, N', S_mob, u*l, E/X at the
        slice boundaries for rigorous methods, alpha).
    thrust_line : list of (float, float) or None
        Line of thrust (x, elevation) at each slice boundary from the
        rigorous GLE solution (Spencer / M-P / GLE methods only).
    slip_points : list of (float, float) or None
        Noncircular slip surface points. None for circular.
    tension_crack_depth : float
        Depth of tension crack (m). 0 = no crack.
    tension_crack_water_depth : float
        Depth of water in tension crack (m). 0 = dry.
    """
    FOS: float = 0.0
    method: str = ""
    xc: float = 0.0
    yc: float = 0.0
    radius: float = 0.0
    x_entry: float = 0.0
    x_exit: float = 0.0
    theta_spencer: Optional[float] = None
    FOS_fellenius: Optional[float] = None
    FOS_bishop: Optional[float] = None
    FOS_spencer: Optional[float] = None
    FOS_morgenstern_price: Optional[float] = None
    lambda_mp: Optional[float] = None
    FOS_janbu: Optional[float] = None
    FOS_janbu_uncorrected: Optional[float] = None
    janbu_f0: Optional[float] = None
    reinforcements: Optional[List[Dict[str, Any]]] = None
    n_slices: int = 0
    has_seismic: bool = False
    kh: float = 0.0
    slice_data: Optional[List[SliceData]] = None
    thrust_line: Optional[List[Tuple[float, float]]] = None
    slip_points: Optional[List[Tuple[float, float]]] = None
    tension_crack_depth: float = 0.0
    tension_crack_water_depth: float = 0.0
    method_solved: bool = True
    method_used: Optional[str] = ""
    methods_without_solution: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_circular(self) -> bool:
        """True if this result represents a circular slip surface."""
        return self.radius > 0

    @property
    def is_stable(self) -> Optional[bool]:
        """True if FOS >= 1.0; None when there is no FOS."""
        if self.FOS is None:
            return None
        return self.FOS >= 1.0

    @property
    def produced_by(self) -> Optional[str]:
        """What produced ``FOS`` (``method_used``, or ``method`` itself)."""
        if self.method_used == "":
            return self.method
        return self.method_used

    def _fos_line(self) -> str:
        if self.FOS is None:
            return (f"none: {self.method} finds no equilibrium solution "
                    f"on this surface (see WARNING)")
        if not self.method_solved:
            return (f"{self.FOS:.3f}  from the {self.produced_by}; "
                    f"{self.method} has no solution here (see WARNING)")
        return f"{self.FOS:.3f}"

    def _no_solution_mark(self, method: str) -> str:
        return ("  (legacy approximate solver, no solution)"
                if method in self.methods_without_solution else "")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SLOPE STABILITY ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Method:           {self.method}",
            f"  Factor of Safety: {self._fos_line()}",
            "",
        ]
        if self.is_circular:
            lines.extend([
                f"  Critical Circle:",
                f"    Center (xc, yc) = ({self.xc:.2f}, {self.yc:.2f}) m",
                f"    Radius R        = {self.radius:.2f} m",
            ])
        else:
            lines.append(f"  Slip Surface:     Noncircular ({len(self.slip_points or [])} points)")
        lines.extend([
            f"    Entry x         = {self.x_entry:.2f} m",
            f"    Exit x          = {self.x_exit:.2f} m",
            "",
            f"  Slices: {self.n_slices}",
        ])
        if self.has_seismic:
            lines.append(f"  Seismic kh:       {self.kh:.3f}")
        if self.reinforcements:
            lines.append(f"  Reinforcement:    {len(self.reinforcements)} "
                         f"element(s) crossing, total "
                         f"{sum(r['T_kN_per_m'] for r in self.reinforcements):.1f} kN/m")
        if self.tension_crack_depth > 0:
            lines.append(f"  Tension crack:    {self.tension_crack_depth:.2f} m deep")
            if self.tension_crack_water_depth > 0:
                lines.append(f"  Crack water:      {self.tension_crack_water_depth:.2f} m")
        if self.FOS_fellenius is not None:
            lines.append(f"  FOS (Fellenius):  {self.FOS_fellenius:.3f}")
        if self.FOS_bishop is not None and self.method != "Bishop":
            lines.append(f"  FOS (Bishop):     {self.FOS_bishop:.3f}")
        if self.FOS_spencer is not None and self.method != "Spencer":
            lines.append(f"  FOS (Spencer):    {self.FOS_spencer:.3f}"
                         + self._no_solution_mark("Spencer"))
        if self.theta_spencer is not None:
            lines.append(f"  Spencer theta:    {self.theta_spencer:.2f} deg")
        if self.FOS_morgenstern_price is not None and self.method != "Morgenstern-Price":
            lines.append(f"  FOS (M-P):        {self.FOS_morgenstern_price:.3f}"
                         + self._no_solution_mark("Morgenstern-Price"))
        if self.lambda_mp is not None:
            lines.append(f"  M-P lambda:       {self.lambda_mp:.3f}")
        if self.FOS_janbu is not None and self.method != "Janbu":
            lines.append(f"  FOS (Janbu corr): {self.FOS_janbu:.3f}")
        if self.FOS_janbu_uncorrected is not None:
            lines.append(f"  FOS (Janbu unc.): {self.FOS_janbu_uncorrected:.3f}"
                         + (f"  (f0={self.janbu_f0:.3f})" if self.janbu_f0 else ""))
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def plot_slip_surface_stresses(self, ax=None, show=True, **kwargs):
        """Plot contact stresses along the slip surface.

        Shows effective normal stress, mobilized shear stress, and
        available shear resistance vs distance along the slip surface.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.slice_data:
            raise ValueError("No slice data available. "
                             "Use include_slice_data=True in analyze_slope().")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        # Cumulative arc length along slip surface
        dist = []
        cumulative = 0.0
        for sd in self.slice_data:
            cumulative += sd.base_length
            dist.append(cumulative - sd.base_length / 2.0)

        sigma_n = [sd.normal_stress_kPa for sd in self.slice_data]
        tau_mob = [sd.shear_stress_kPa for sd in self.slice_data]
        tau_avail = [sd.shear_resistance_kPa for sd in self.slice_data]

        ax.plot(dist, sigma_n, 'b-o', markersize=3, linewidth=1.5,
                label="$\\sigma'_n$ (effective normal)")
        ax.plot(dist, tau_mob, 'r--s', markersize=3, linewidth=1.5,
                label="$\\tau_{mob}$ (mobilized shear)")
        ax.plot(dist, tau_avail, 'g-^', markersize=3, linewidth=1.5,
                label="$\\tau_{avail}$ (shear resistance)")

        ax.legend(fontsize=9)
        setup_engineering_plot(ax, f"Contact Stresses Along Slip Surface (FOS={fos_text(self.FOS)})",
                              "Distance Along Slip Surface (m)", "Stress (kPa)")
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "FOS": round(self.FOS, 4) if self.FOS is not None else None,
            "is_stable": self.is_stable,
            "method": self.method,
            "method_solved": self.method_solved,
            "method_used": self.produced_by,
            "surface_type": "circular" if self.is_circular else "noncircular",
            "xc_m": round(self.xc, 2),
            "yc_m": round(self.yc, 2),
            "radius_m": round(self.radius, 2),
            "x_entry_m": round(self.x_entry, 2),
            "x_exit_m": round(self.x_exit, 2),
            "n_slices": self.n_slices,
            "has_seismic": self.has_seismic,
            "kh": self.kh,
        }
        if self.tension_crack_depth > 0:
            d["tension_crack_depth_m"] = round(self.tension_crack_depth, 2)
            d["tension_crack_water_depth_m"] = round(self.tension_crack_water_depth, 2)
        if self.slip_points is not None:
            d["slip_points"] = [
                (round(x, 3), round(z, 3)) for x, z in self.slip_points
            ]
        if self.FOS_fellenius is not None:
            d["FOS_fellenius"] = round(self.FOS_fellenius, 4)
        if self.FOS_bishop is not None:
            d["FOS_bishop"] = round(self.FOS_bishop, 4)
        if self.FOS_spencer is not None:
            d["FOS_spencer"] = round(self.FOS_spencer, 4)
        if self.theta_spencer is not None:
            d["theta_spencer_deg"] = round(self.theta_spencer, 2)
        if self.FOS_morgenstern_price is not None:
            d["FOS_morgenstern_price"] = round(self.FOS_morgenstern_price, 4)
        if self.lambda_mp is not None:
            d["lambda_mp"] = round(self.lambda_mp, 4)
        if self.FOS_janbu is not None:
            d["FOS_janbu_corrected"] = round(self.FOS_janbu, 4)
        if self.FOS_janbu_uncorrected is not None:
            d["FOS_janbu_uncorrected"] = round(self.FOS_janbu_uncorrected, 4)
        if self.janbu_f0 is not None:
            d["janbu_f0"] = round(self.janbu_f0, 4)
        if self.reinforcements:
            d["reinforcements"] = self.reinforcements
            d["n_reinforcements_active"] = len(self.reinforcements)
            d["total_reinforcement_kN_per_m"] = round(
                sum(r["T_kN_per_m"] for r in self.reinforcements), 2)
        if self.slice_data is not None:
            d["slice_data"] = [s.to_dict() for s in self.slice_data]
        if self.thrust_line is not None:
            d["thrust_line"] = [
                (round(x, 3), round(z, 3)) for x, z in self.thrust_line
            ]
        if self.methods_without_solution:
            d["methods_without_solution"] = list(self.methods_without_solution)
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    def get_tables(self) -> list:
        """SLIDE-style tables for this slip surface: the stresses along it
        (printed), and as CSV only (the calc package prints its own
        versions): every slice quantity, the factor of safety by method,
        the slip surface, the line of thrust and the reinforcement forces.
        Values are this result's own, unrounded, per metre of slope.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        sd = self.slice_data or []
        if sd:
            rigorous = sd[0].E_left_kN is not None
            dist, cum = [], 0.0
            for s in sd:
                cum += s.base_length
                dist.append(cum - s.base_length / 2.0)
            tables.append(CalcTable(
                title="Stresses on the slip surface",
                columns=["Slice", ("x at mid-slice", "m", ".2f"),
                         ("Distance along the surface", "m", ".2f"),
                         ("Base angle α", "deg", ".1f"),
                         ("Effective normal stress σ'n", "kPa", ".2f"),
                         ("Mobilized shear stress τ_mob", "kPa", ".2f"),
                         ("Shear strength τ_f", "kPa", ".2f")],
                rows=[[i, s.x_mid, d, s.alpha_deg, s.normal_stress_kPa,
                       s.shear_stress_kPa, s.shear_resistance_kPa]
                      for i, (s, d) in enumerate(zip(sd, dist), 1)],
                note=(f"{self.method}, FOS = {fos_text(self.FOS)}; σ'n = N'/l, "
                      "τ_mob = S_mob/l, τ_f the shear strength "
                      "available on the base; distance measured along the "
                      "slip surface from its upslope end"),
                name="slip_surface_stresses",
            ))
            columns = ["Slice", ("x at mid-slice", "m"), ("Ground surface z",
                       "m"), ("Slip surface z", "m"), ("Width b", "m"),
                       ("Height h", "m"), ("Base angle α", "deg"),
                       ("Weight W", "kN/m"), ("Pore pressure u", "kPa"),
                       ("Cohesion c", "kPa"), ("Friction angle φ", "deg"),
                       ("Base length l", "m"),
                       ("Effective normal stress σ'n", "kPa"),
                       ("Mobilized shear stress τ_mob", "kPa"),
                       ("Shear strength τ_f", "kPa"),
                       ("Effective normal force N'", "kN/m"),
                       ("Mobilized shear force S_mob", "kN/m"),
                       ("Pore-water force U = u l", "kN/m")]
            if rigorous:
                columns += [("Interslice normal force E left", "kN/m"),
                            ("Interslice normal force E right", "kN/m"),
                            ("Interslice shear force X left", "kN/m"),
                            ("Interslice shear force X right", "kN/m")]
            columns.append("In tension crack")
            rows = []
            for i, s in enumerate(sd, 1):
                row = [i, s.x_mid, s.z_top, s.z_base, s.width, s.height,
                       s.alpha_deg, s.weight, s.pore_pressure, s.c, s.phi,
                       s.base_length, s.normal_stress_kPa, s.shear_stress_kPa,
                       s.shear_resistance_kPa, s.N_eff_kN, s.S_mob_kN,
                       s.U_base_kN]
                if rigorous:
                    row += [s.E_left_kN, s.E_right_kN, s.X_left_kN,
                            s.X_right_kN]
                row.append(s.in_tension_crack)
                rows.append(row)
            tables.append(CalcTable(
                title="Forces on each slice",
                columns=columns, rows=rows,
                note=(f"{self.method}, FOS = {fos_text(self.FOS)}, "
                      f"{len(sd)} slices; " +
                      ("N', S_mob, E and X from the converged rigorous GLE "
                       "solution" if rigorous else
                       "N' and S_mob from the Fellenius decomposition (" +
                       ("the method has no converged solution on this "
                        "surface" if not self.method_solved else
                        "the method does not resolve interslice forces")
                       + ")")),
                name="slice_forces", printed=False,
            ))

        by_method = [
            ("Ordinary method of slices (Fellenius)", self.FOS_fellenius, ""),
            ("Bishop simplified", self.FOS_bishop, ""),
            ("Janbu simplified, uncorrected", self.FOS_janbu_uncorrected,
             ""),
            ("Janbu simplified, corrected", self.FOS_janbu,
             f"f0 = {self.janbu_f0:.3f}" if self.janbu_f0 else ""),
            ("Spencer", self.FOS_spencer,
             f"θ = {self.theta_spencer:.2f} deg"
             if self.theta_spencer is not None else ""),
            ("Morgenstern-Price", self.FOS_morgenstern_price,
             f"λ = {self.lambda_mp:.3f}"
             if self.lambda_mp is not None else ""),
        ]
        rows = [[name, fos, extra + ("; " if extra else "")
                 + "legacy approximate solver: no converged solution"
                 if name in self.methods_without_solution else extra]
                for name, fos, extra in by_method if fos is not None]
        if rows:
            tables.append(CalcTable(
                title="Factor of safety by method",
                columns=["Method", ("FOS", "", ".3f"),
                         "Interslice force parameter"],
                rows=rows,
                note=f"The same slip surface; the headline is {self.method}",
                name="fos_by_method", printed=False,
            ))

        if self.slip_points:
            tables.append(CalcTable(
                title="Slip surface",
                columns=["Point", ("x", "m"), ("z", "m")],
                rows=[[i, x, z] for i, (x, z) in
                      enumerate(self.slip_points, 1)],
                note="The noncircular slip surface's vertices",
                name="slip_surface", printed=False,
            ))
        elif self.is_circular:
            tables.append(CalcTable(
                title="Slip surface",
                columns=[("Centre x", "m"), ("Centre z", "m"),
                         ("Radius R", "m"), ("Entry x", "m"),
                         ("Exit x", "m")],
                rows=[[self.xc, self.yc, self.radius, self.x_entry,
                       self.x_exit]],
                note="The circular slip surface",
                name="slip_surface", printed=False,
            ))
        if self.thrust_line:
            tables.append(CalcTable(
                title="Line of thrust",
                columns=[("x", "m"), ("z", "m")],
                rows=[[x, z] for x, z in self.thrust_line],
                note=("Where the interslice normal force E acts on each "
                      "slice boundary (rigorous GLE solution)"),
                name="thrust_line", printed=False,
            ))
        if self.reinforcements:
            tables.append(CalcTable(
                title="Reinforcement forces",
                columns=["Type", "Element", ("Crossing x", "m"),
                         ("Crossing z", "m"), ("Force T", "kN/m"),
                         "Controlled by"],
                rows=[[r.get("kind"), r.get("index"), r.get("x_m"),
                       r.get("z_m"), r.get("T_kN_per_m"),
                       r.get("controlled_by")] for r in self.reinforcements],
                note=("Mobilized force of each element where it crosses "
                      "this slip surface"),
                name="reinforcement_forces", printed=False,
            ))
        return tables


@dataclass
class SearchResult:
    """Results from a critical surface search.

    Attributes
    ----------
    critical : SlopeStabilityResult
        Result for the critical (minimum FOS) surface.
    n_surfaces_evaluated : int
        Total number of trial surfaces evaluated.
    grid_fos : list of dict
        Grid of {xc, yc, R, FOS} for all evaluated surfaces.
    trial_surfaces : list of dict
        Noncircular trial surfaces stored for plotting:
        {"FOS": float, "points": [(x, z), ...]}. Circular searches
        leave this empty (trials are reconstructable from grid_fos).
    n_rejected_geometry, n_rejected_nonconverged, n_rejected_jagged : int
        Noncircular trial surfaces rejected before scoring, by reason
        (degenerate geometry / non-converged rigorous GLE / jagged low-FOS
        artifact). Zero for circular searches. A majority-rejected search also
        emits a ``warnings.warn`` — see ``search._rejection_kwargs``.
    n_rejected_ground_crossings : int
        Trial surfaces, circular or not, the search refused because they
        are below the ground over more than one stretch: they leave the
        ground and come back into it, so the soil above them is not one
        sliding mass (GM-122, ``slip_surface.GroundCrossingError``). A
        search that runs others (``search_critical_surface``) counts theirs
        and its supplementary families' too.
    n_rejected_m_alpha : int
        Trial surfaces, circular or not, the search left out because m_alpha
        = cos(a) (1 + tan(a) tan(phi) / F) fell below 0.2 on a slice at the
        method's own F (GM-127; Slide2's check, its error -112). Counted as
        ``n_rejected_ground_crossings`` is.
    governing_surface : str
        Which kind of surface governs ``critical`` (set by
        ``search_critical_surface``): 'circular' / 'noncircular' (the main
        search), 'toe_circle', 'circle_above_toe' (a circle passing above
        the toe, its lower end on the face) or 'infinite_slope' (shallow
        planar sliding in cohesionless soil). See ``search_supplements``.
        ``search_ranges['faces_covered']`` gives the faces the two circle
        families started from. Empty when the
        search was called directly (grid_search etc.).
    candidates : dict
        Minimum FOS of each surface family tried, keyed as above.
    warnings : list of str
        Plain statements where the search cannot be sure of its answer
        (a supplementary family undercut the main search, the critical
        surface sits on the edge of the searched region, a cohesionless
        face could not be checked, the governing circle is closed by a
        reverse-curvature crack, has supports Bishop counts little of, or
        was scored by the legacy approximate solver).
    infinite_slope_checks : list of dict
        One entry per stretch of slope face where cohesionless soil is at
        the surface: x range, slope angle beta, layer, phi', the dry
        closed-form tan(phi')/tan(beta) and the FOS of the shallow planar
        surfaces evaluated there (None if not evaluated, with a note).
    search_ranges : dict
        The entry / exit windows and centre ranges the main search used
        (x_entry_range, x_exit_range, and x_range / y_range for a centre
        grid), ``automatic`` naming those it set itself, and for those how
        (GM-57, ``search_ranges.py``): ``basis``, the ``slopes`` they were
        set from (toe, crest, height, depth to the base, reach beyond the
        toe and behind the crest) and ``clipped`` (a bound the model's end
        cut short: where the slope called for it). ``time_budget`` when the
        search's time budget stopped it (GM-123): the budget, the time it
        ran, each stage ``stopped`` (what it scored, of how many, what it
        covered and did not, the run time projected for all of it) and
        what was ``not_run``.
    governing_location : dict
        Where the governing surface lies (GM-57): ``x_entry`` / ``x_exit``,
        ``z_lowest``, the ``layers`` along its base in order, and ``slope``:
        the tallest face's toe, crest and height H, and the surface's ends
        and lowest point against them in multiples of H, and its greatest
        ``depth`` below the ground.
        With ``search_ranges`` it says what the search covered.
    refused_lowest : dict or None
        The lowest noncircular surface a search refused because its
        rigorous solution did not hold (about one moment axis only, or at
        one slicing only; `search._solution_fails`): ``FOS``, ``reason``,
        ``x_entry`` / ``x_exit`` and ``points``. A warning says so when it
        is below the reported FOS. None when nothing was refused.
    choices : list of dict
        What the search chose (GM-115, ``geotech_common.choices``;
        ``search_supplements.search_choices``): the search, the method it
        scored with, the surface families it tried and those it did not,
        how its ranges were set (pointing at ``search_ranges``,
        ``governing_location`` and ``refused_lowest``), its time budget
        and any Bishop screen (GM-123).
    """
    critical: Optional[SlopeStabilityResult] = None
    n_surfaces_evaluated: int = 0
    grid_fos: List[Dict[str, float]] = field(default_factory=list)
    trial_surfaces: List[Dict[str, Any]] = field(default_factory=list)
    n_rejected_geometry: int = 0
    n_rejected_nonconverged: int = 0
    n_rejected_jagged: int = 0
    n_rejected_ground_crossings: int = 0
    n_rejected_m_alpha: int = 0
    governing_surface: str = ""
    candidates: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    infinite_slope_checks: List[Dict[str, Any]] = field(default_factory=list)
    search_ranges: Dict[str, Any] = field(default_factory=dict)
    governing_location: Dict[str, Any] = field(default_factory=dict)
    refused_lowest: Optional[Dict[str, Any]] = None
    choices: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  CRITICAL SURFACE SEARCH RESULTS",
            "=" * 60,
            "",
            f"  Surfaces evaluated: {self.n_surfaces_evaluated}",
        ]
        if self.n_rejected_ground_crossings:
            lines.append(f"  Refused, not one sliding mass (leave the ground "
                         f"and come back into it): "
                         f"{self.n_rejected_ground_crossings}")
        if self.n_rejected_m_alpha:
            lines.append(f"  Left out, m_alpha < 0.2 on a slice (Slide2's "
                         f"check): {self.n_rejected_m_alpha}")
        if self.critical:
            lines.append(f"  Minimum FOS:       {self.critical.FOS:.3f}")
            lines.append(f"  Method:            {self.critical.method}")
            if self.critical.is_circular:
                lines.append(
                    f"  Critical circle:   "
                    f"({self.critical.xc:.2f}, {self.critical.yc:.2f}), "
                    f"R={self.critical.radius:.2f} m"
                )
            else:
                lines.append(
                    f"  Critical surface:  Noncircular "
                    f"({len(self.critical.slip_points or [])} points)"
                )
        if self.governing_surface:
            from slope_stability.search_supplements import describe_surface_kind
            lines.append(f"  Governing surface: "
                         f"{describe_surface_kind(self.governing_surface)}")
            for kind, fos in self.candidates.items():
                lines.append(f"    {describe_surface_kind(kind):<48s} "
                             f"{fos:.3f}")
        gl = self.governing_location
        if gl.get("layers"):
            lines.append(f"  Governing surface: x {gl['x_entry']:.2f} to "
                         f"{gl['x_exit']:.2f}, lowest z {gl['z_lowest']:.2f}, "
                         f"at most {gl.get('depth', 0.0):.2f} deep, "
                         f"through {', '.join(gl['layers'])}")
        sr = self.search_ranges
        if sr.get("automatic"):
            ranges = ", ".join(
                f"{n} {sr[n][0]:.2f} to {sr[n][1]:.2f}"
                for n in sr["automatic"] if n in sr)
            lines.append(f"  Automatic ranges:  {ranges}")
            for key, x in sr.get("clipped", {}).items():
                end = "left" if key == "entry_lo" else "right"
                lines.append(f"    cut at the {end} end of the profile "
                             f"(the slope's reach: x = {x:.2f})")
        from geotech_common.choices import choices_lines
        lines.extend(choices_lines(self.choices))
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def plot_fos_contour(self, ax=None, show=True, **kwargs):
        """Plot FOS at each trial circle center as a scatter/contour.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.grid_fos:
            raise ValueError("No grid FOS data available for plotting.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        xc_vals = [g['xc'] for g in self.grid_fos]
        yc_vals = [g['yc'] for g in self.grid_fos]
        fos_vals = [g['FOS'] for g in self.grid_fos]
        sc = ax.scatter(xc_vals, yc_vals, c=fos_vals, cmap='RdYlGn',
                        edgecolors='black', linewidth=0.5, s=60, **kwargs)
        plt.colorbar(sc, ax=ax, label='Factor of Safety')
        if self.critical:
            ax.plot(self.critical.xc, self.critical.yc, 'r*', markersize=15,
                    markeredgecolor='black', label=f'Critical (FOS={self.critical.FOS:.3f})')
            ax.legend()
        setup_engineering_plot(ax, "Critical Surface Search",
                              "Circle Center X (m)", "Circle Center Y (m)")
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self) -> Dict[str, Any]:
        d = {"n_surfaces_evaluated": self.n_surfaces_evaluated}
        n_rej = (self.n_rejected_geometry + self.n_rejected_nonconverged
                 + self.n_rejected_jagged + self.n_rejected_ground_crossings
                 + self.n_rejected_m_alpha)
        if n_rej:
            d["n_rejected"] = {
                "geometry": self.n_rejected_geometry,
                "nonconverged": self.n_rejected_nonconverged,
                "jagged": self.n_rejected_jagged,
                "ground_crossings": self.n_rejected_ground_crossings,
                "m_alpha": self.n_rejected_m_alpha,
            }
        if self.critical:
            d["critical"] = self.critical.to_dict()
        if self.governing_surface:
            d["governing_surface"] = self.governing_surface
            d["candidates"] = dict(self.candidates)
        if self.infinite_slope_checks:
            d["infinite_slope_checks"] = [dict(c) for c in
                                          self.infinite_slope_checks]
        if self.search_ranges:
            d["search_ranges"] = dict(self.search_ranges)
        if self.governing_location:
            d["governing_location"] = dict(self.governing_location)
        if self.refused_lowest:
            d["refused_lowest"] = dict(self.refused_lowest)
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    def get_tables(self) -> list:
        """The search record, CSV only: every trial surface scored (its
        centre, or the midpoint of a noncircular surface, the radius and
        the FOS, as the search stored them), and the lowest FOS of each
        surface family tried.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        if self.grid_fos:
            tables.append(CalcTable(
                title="Trial surfaces",
                columns=["Trial", ("Centre or midpoint x", "m"),
                         ("Centre or midpoint z", "m"),
                         ("Radius R (0 = noncircular)", "m"), ("FOS", "")],
                rows=[[i, g.get("xc"), g.get("yc"), g.get("R"),
                       g.get("FOS")]
                      for i, g in enumerate(self.grid_fos, 1)],
                note=(f"{self.n_surfaces_evaluated} surfaces evaluated, in "
                      "the order the search scored them (values as the "
                      "search rounded them)"),
                name="trial_surfaces", printed=False,
            ))
        if self.candidates:
            from slope_stability.search_supplements import (
                describe_surface_kind)
            tables.append(CalcTable(
                title="Lowest FOS of each surface family",
                columns=["Surface family", ("FOS", "", ".3f"), "Governs"],
                rows=[[describe_surface_kind(k), fos,
                       k == self.governing_surface]
                      for k, fos in self.candidates.items()],
                note="The families the search tried and the minimum of each",
                name="surface_families", printed=False,
            ))
        return tables


@dataclass
class InfiniteSlopeResult:
    """Infinite-slope (planar, translational) factor-of-safety result.

    Closed-form FOS for a slip plane parallel to an infinite slope surface
    at depth z, with the effective-stress strength on that plane. All units
    SI (m, kPa, kN/m3, degrees).
    """
    FOS: float
    slope_angle: float          # beta, degrees
    depth: float                # z, m (slip-plane depth below surface)
    phi: float                  # deg
    c: float                    # kPa
    gamma: float                # kN/m3
    water_condition: str        # 'dry' | 'seepage_parallel' | 'ru'
    normal_stress: float = 0.0  # sigma_n on the slip plane, kPa
    shear_stress: float = 0.0   # tau (driving) on the slip plane, kPa
    pore_pressure: float = 0.0  # u on the slip plane, kPa
    ru: float = 0.0

    def summary(self) -> str:
        return "\n".join([
            "=" * 56,
            "  INFINITE SLOPE (planar) FACTOR OF SAFETY",
            "=" * 56,
            f"  Slope angle beta:  {self.slope_angle:.2f} deg",
            f"  Slip-plane depth:  {self.depth:.3f} m",
            f"  c' / phi':         {self.c:.2f} kPa / {self.phi:.1f} deg",
            f"  Water condition:   {self.water_condition}"
            + (f" (ru={self.ru:.3f})" if self.water_condition == "ru" else ""),
            f"  sigma_n / u:       {self.normal_stress:.2f} / "
            f"{self.pore_pressure:.2f} kPa",
            f"  tau (driving):     {self.shear_stress:.2f} kPa",
            f"  FOS:               {self.FOS:.3f}",
            "=" * 56,
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "FOS": round(self.FOS, 4),
            "slope_angle_deg": round(self.slope_angle, 3),
            "depth_m": round(self.depth, 4),
            "phi_deg": round(self.phi, 3),
            "c_kPa": round(self.c, 3),
            "gamma_kN_m3": round(self.gamma, 3),
            "water_condition": self.water_condition,
            "ru": round(self.ru, 4),
            "normal_stress_kPa": round(self.normal_stress, 3),
            "shear_stress_kPa": round(self.shear_stress, 3),
            "pore_pressure_kPa": round(self.pore_pressure, 3),
        }
