"""
Results containers for retaining wall analysis.

A check that was not run is never reported as a pass (GM-100): its FOS and
its pass flag are None, ``checks_not_run`` names it, and the summary says
"not checked". A check with no driving force (zero active thrust) passes,
with its FOS None and the words "no driving force" instead of a number.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

#: Said wherever the cantilever bearing check was not run.
BEARING_NOT_CHECKED = ("not checked (no q_allowable given); the bearing "
                       "pressures q_toe / q_heel and the eccentricity check "
                       "are still reported")

#: The names of the bearing pressure compared with q_allowable (GM-288).
_BEARING_Q_NAME = {"uniform": "q_eq", "linear": "q_max"}


def _round(value, ndigits):
    return None if value is None else round(value, ndigits)


@dataclass
class CantileverWallResult:
    """Results from cantilever retaining wall analysis.

    Attributes
    ----------
    FOS_sliding : float or None
        Factor of safety against sliding. None when there is no driving
        force (the active thrust is zero): the check passes.
    FOS_overturning : float or None
        Factor of safety against overturning. None when there is no
        overturning moment (zero active thrust): the check passes.
    FOS_bearing : float or None
        q_allowable / the bearing pressure ``q_bearing``. None when the
        bearing check was NOT RUN (no q_allowable given).
    passes_sliding : bool
        True if FOS_sliding >= required (typically 1.5).
    passes_overturning : bool
        True if FOS_overturning >= required (typically 2.0).
    passes_bearing : bool or None
        True if the bearing pressure ``q_bearing`` is within q_allowable.
        None when the bearing check was not run.
    q_toe : float
        Bearing pressure at the toe (kPa), on the bearing check's loads.
    q_heel : float
        Bearing pressure at the heel (kPa), on the bearing check's loads.
    eccentricity : float
        Eccentricity of resultant from center of base (m), on the loads the
        overturning check counts (GM-157: without a surcharge over the heel
        that is not a resisting force). The eccentricity for bearing is
        ``bearing_terms["eccentricity"]``.
    in_middle_third : bool
        True if resultant is within middle third of base (e <= B/6).
    wall_height : float
        Total wall height H (m), from the underside of the base slab to the
        top of the stem.
    base_width : float
        Base width (m).
    FOS_sliding_required, FOS_overturning_required : float
        The required factors of safety the checks were run against.
    q_allowable : float or None
        The allowable bearing pressure given (kPa), or None.
    founded_on_rock : bool
        The wall is founded on rock: the eccentricity limit is B/4, not B/6
        (GM-288, ``cantilever.ECCENTRICITY_BASIS``).
    bearing_pressure : str or None
        The bearing pressure compared with q_allowable (GM-288,
        ``cantilever.BEARING_PRESSURE_BASIS``): "uniform", q_eq = V / B'
        over B' = B - 2e, or "linear", the larger edge pressure. None on a
        result built by hand (the larger of q_toe and q_heel).
    q_bearing : float or None
        That pressure (kPa); None on a result built by hand.

    The terms each check used, unrounded, as it used them (GM-143; the
    calc package prints these rather than working them out again; None
    on a result built by hand):

    earth_pressure : dict or None
        method, Ka, H_active, q_live, q_dead (kPa), Pa_earth, Pa_surcharge,
        Pa_cohesion, Pa (kN/m), z_Pa (m above the base), delta_deg, Ph, Pv,
        and the live load's share Pa_live, Pv_live.
    vertical_forces : list of dict or None
        Each vertical force on the base, in the checks' order: component,
        kind, W (kN/m), x (arm from the toe, m), M, and the checks that
        count it: resists (sliding, overturning, eccentricity) and bearing
        (GM-157).
    sliding_terms, overturning_terms, bearing_terms : dict or None
        The checks' terms (``cantilever._sliding_terms`` and the others):
        V, V_left_out, delta_b, ca, Pp, resisting, driving; M_stabilizing,
        M_overturning, V, x_resultant, eccentricity (the eccentricity
        check); V, M_stab, M_over, x_resultant, eccentricity (for bearing),
        distribution, B_eff, q_toe, q_heel, q_max.
    choices : list of dict
        The sliding check's base friction delta_b (GM-173,
        ``cantilever.base_friction_choice``) and how the surcharges over
        the heel were counted, when the wall carries any (GM-157,
        ``cantilever.heel_surcharge_choices``; ``geotech_common.choices``).
    """
    FOS_sliding: Optional[float] = 0.0
    FOS_overturning: Optional[float] = 0.0
    FOS_bearing: Optional[float] = None
    passes_sliding: bool = False
    passes_overturning: bool = False
    passes_bearing: Optional[bool] = None
    q_toe: float = 0.0
    q_heel: float = 0.0
    eccentricity: float = 0.0
    in_middle_third: bool = False
    wall_height: float = 0.0
    base_width: float = 0.0
    FOS_sliding_required: float = 1.5
    FOS_overturning_required: float = 2.0
    q_allowable: Optional[float] = None
    founded_on_rock: bool = False
    bearing_pressure: Optional[str] = None
    q_bearing: Optional[float] = None
    earth_pressure: Optional[Dict[str, Any]] = None
    vertical_forces: Optional[List[Dict[str, Any]]] = None
    sliding_terms: Optional[Dict[str, Any]] = None
    overturning_terms: Optional[Dict[str, Any]] = None
    bearing_terms: Optional[Dict[str, Any]] = None
    choices: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def eccentricity_bearing(self) -> Optional[float]:
        """The eccentricity for bearing (m) when it differs from the
        eccentricity check's (GM-157), else None."""
        b = self.bearing_terms
        if not b:
            return None
        e = round(b["eccentricity"], 3)
        return None if e == round(self.eccentricity, 3) else e

    @property
    def bearing_checked(self) -> bool:
        """True when the bearing check was run (a q_allowable was given)."""
        return self.FOS_bearing is not None

    @property
    def eccentricity_limit(self) -> float:
        """The eccentricity limit (m): B/6 on soil, B/4 on rock (GM-288)."""
        return self.base_width / (4.0 if self.founded_on_rock else 6.0)

    @property
    def within_eccentricity_limit(self) -> bool:
        """True when |e| is within :attr:`eccentricity_limit` (GM-288)."""
        return abs(self.eccentricity) <= self.eccentricity_limit + 1e-12

    def _q_bearing(self) -> float:
        """The bearing pressure compared with q_allowable (kPa)."""
        return (self.q_bearing if self.q_bearing is not None
                else max(self.q_toe, self.q_heel))

    def _q_bearing_name(self) -> str:
        return _BEARING_Q_NAME.get(self.bearing_pressure, "q_max")

    @property
    def checks_failed(self) -> List[str]:
        """The checks that were run and FAILED: 'sliding', 'overturning',
        'bearing', 'eccentricity' (resultant outside the middle third, or
        the middle half on rock)."""
        failed = []
        if self.passes_sliding is False:
            failed.append("sliding")
        if self.passes_overturning is False:
            failed.append("overturning")
        if self.bearing_checked and self.passes_bearing is False:
            failed.append("bearing")
        if not self.within_eccentricity_limit:
            failed.append("eccentricity")
        return failed

    @property
    def checks_not_run(self) -> List[str]:
        """The checks that were NOT run: 'bearing' when no q_allowable."""
        return [] if self.bearing_checked else ["bearing"]

    @property
    def all_checks_pass(self) -> Optional[bool]:
        """True only when every check was run and passed; False when any
        check failed; None when none failed but a check was not run."""
        if self.checks_failed:
            return False
        if self.checks_not_run:
            return None
        return True

    def _check_words(self, name: str) -> str:
        """One check in plain words, with its numbers."""
        if name == "sliding":
            if self.FOS_sliding is None:
                return "sliding (no driving force: the active thrust is zero)"
            rel = ">=" if self.passes_sliding else "<"
            return (f"sliding (FOS {self.FOS_sliding:.2f} {rel} "
                    f"{self.FOS_sliding_required:.2f} required)")
        if name == "overturning":
            if self.FOS_overturning is None:
                return ("overturning (no driving force: the active thrust "
                        "is zero)")
            rel = ">=" if self.passes_overturning else "<"
            return (f"overturning (FOS {self.FOS_overturning:.2f} {rel} "
                    f"{self.FOS_overturning_required:.2f} required)")
        if name == "bearing":
            if not self.bearing_checked:
                return "bearing (no q_allowable given)"
            q_max = self._q_bearing()
            if self.q_allowable is not None and q_max > 0:
                rel = "<=" if self.passes_bearing else ">"
                return (f"bearing ({self._q_bearing_name()} {q_max:.1f} kPa "
                        f"{rel} q_allowable {self.q_allowable:.1f} kPa, FOS "
                        f"{self.FOS_bearing:.2f})")
            if self.q_allowable is not None:
                return "bearing (the resultant falls outside the base)"
            return f"bearing (FOS {self.FOS_bearing:.2f})"
        # eccentricity
        ok = self.within_eccentricity_limit
        rel = "<=" if ok else ">"
        where = "within" if ok else "outside"
        part, frac = (("middle half", "B/4") if self.founded_on_rock
                      else ("middle third", "B/6"))
        return (f"eccentricity (resultant {where} the {part}: "
                f"e = {abs(self.eccentricity):.3f} m {rel} {frac} = "
                f"{self.eccentricity_limit:.3f} m)")

    @property
    def verdict(self) -> str:
        """The result in plain words: failed checks first, with their
        numbers, then those that pass and those not run."""
        names = ("sliding", "overturning", "bearing", "eccentricity")
        failed = self.checks_failed
        not_run = self.checks_not_run
        passed = [n for n in names if n not in failed and n not in not_run]
        parts = []
        if failed:
            parts.append("FAILS: " + "; ".join(
                self._check_words(n) for n in failed) + ".")
        else:
            parts.append("No check failed.")
        if passed:
            parts.append("Passes: " + "; ".join(
                self._check_words(n) for n in passed) + ".")
        if not_run:
            parts.append("Not checked: " + "; ".join(
                self._check_words(n) for n in not_run) + ".")
        return " ".join(parts)

    def summary(self) -> str:
        def _status(passes):
            return "OK" if passes else "FAIL"

        if self.FOS_sliding is None:
            sliding = "no driving force (the active thrust is zero)  [OK]"
        else:
            sliding = (f"FOS = {self.FOS_sliding:.2f}  "
                       f"[{_status(self.passes_sliding)}]  "
                       f"(required {self.FOS_sliding_required:.2f})")
        if self.FOS_overturning is None:
            overturning = "no driving force (the active thrust is zero)  [OK]"
        else:
            overturning = (f"FOS = {self.FOS_overturning:.2f}  "
                           f"[{_status(self.passes_overturning)}]  "
                           f"(required {self.FOS_overturning_required:.2f})")
        if not self.bearing_checked:
            bearing = BEARING_NOT_CHECKED
        else:
            bearing = (f"FOS = {self.FOS_bearing:.2f}  "
                       f"[{_status(self.passes_bearing)}]")
            q_max = self._q_bearing()
            if self.q_allowable is not None and q_max > 0:
                bearing += (f"  (q_allowable {self.q_allowable:.1f} kPa / "
                            f"{self._q_bearing_name()} {q_max:.1f} kPa)")
            elif self.q_allowable is not None:
                bearing += "  (the resultant falls outside the base)"
        failed = self.checks_failed
        lines = [
            "=" * 60,
            "  CANTILEVER RETAINING WALL RESULTS",
            "=" * 60,
            "",
            f"  Wall height:  {self.wall_height:.2f} m "
            "(underside of base to top of stem)",
            f"  Base width:   {self.base_width:.2f} m",
            "",
            f"  Sliding:      {sliding}",
            f"  Overturning:  {overturning}",
            f"  Bearing:      {bearing}",
            "",
            f"  q_toe  = {self.q_toe:.1f} kPa",
            f"  q_heel = {self.q_heel:.1f} kPa",
            f"  Eccentricity = {self.eccentricity:.3f} m"
            f"  ({self._ecc_words()})",
        ]
        if self.eccentricity_bearing is not None:
            lines.append(f"  Eccentricity for bearing = "
                         f"{self.eccentricity_bearing:.3f} m (on the "
                         "bearing check's loads)")
        lines.extend([
            "",
            "  Failed checks: " + (", ".join(failed) if failed else "none"),
        ])
        if self.checks_not_run:
            lines.append("  Not checked:   " + ", ".join(self.checks_not_run))
        if self.choices:
            from geotech_common.choices import choices_lines
            lines.extend(choices_lines(self.choices))
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _ecc_words(self) -> str:
        part = "middle half" if self.founded_on_rock else "middle third"
        return part if self.within_eccentricity_limit else f"OUTSIDE {part}"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "verdict": self.verdict,
            "checks_failed": self.checks_failed,
            "checks_not_run": self.checks_not_run,
            "all_checks_pass": self.all_checks_pass,
            "FOS_sliding": _round(self.FOS_sliding, 3),
            "FOS_overturning": _round(self.FOS_overturning, 3),
            "FOS_bearing": _round(self.FOS_bearing, 3),
            "FOS_sliding_required": self.FOS_sliding_required,
            "FOS_overturning_required": self.FOS_overturning_required,
            "passes_sliding": self.passes_sliding,
            "passes_overturning": self.passes_overturning,
            "passes_bearing": self.passes_bearing,
            "bearing_checked": self.bearing_checked,
            "q_allowable_kPa": self.q_allowable,
            "q_toe_kPa": round(self.q_toe, 1),
            "q_heel_kPa": round(self.q_heel, 1),
            "eccentricity_m": round(self.eccentricity, 3),
            "in_middle_third": self.in_middle_third,
            "eccentricity_limit_m": round(self.eccentricity_limit, 3),
            "wall_height_m": self.wall_height,
            "base_width_m": self.base_width,
        }
        if self.founded_on_rock:
            d["founded_on_rock"] = True
        if self.bearing_pressure is not None:
            d["bearing_pressure"] = self.bearing_pressure
            d["q_bearing_kPa"] = _round(self.q_bearing, 1)
        if self.eccentricity_bearing is not None:
            d["eccentricity_bearing_m"] = self.eccentricity_bearing
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d

    def plot_stability_summary(self, ax=None, show=True):
        """Plot FOS bar chart for sliding, overturning, and bearing. A check
        with no FOS (not run, or no driving force) gets no bar, only its
        words.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Sliding', 'Overturning', 'Bearing']
        values = [self.FOS_sliding, self.FOS_overturning, self.FOS_bearing]
        passes = [self.passes_sliding, self.passes_overturning,
                  self.passes_bearing]
        words = ["no driving force", "no driving force", "not checked"]
        colors = ['green' if p else 'red' for p in passes]
        bars = ax.bar(labels, [0.0 if v is None else v for v in values],
                      color=colors, alpha=0.7, edgecolor='black')
        for bar, val, word in zip(bars, values, words):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    word if val is None else f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1,
                   label='FOS = 1.0')
        setup_engineering_plot(ax, "Retaining Wall Stability",
                              "", "Factor of Safety")
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()
        return ax


@dataclass
class MSEWallResult:
    """Results from MSE wall analysis.

    Attributes
    ----------
    FOS_sliding : float
        External FOS against sliding.
    FOS_overturning : float
        External FOS against overturning.
    FOS_bearing : float or None
        External FOS against bearing failure (q_allowable / sigma_v). None
        when the bearing check was NOT RUN (no q_allowable given).
    passes_external : bool or None
        True only when every external check was run and passed; False when
        one failed; None when none failed but bearing was not checked.
    internal_results : list of dict
        Per-level internal stability check results.
    all_pass_internal : bool
        True if all internal checks pass.
    wall_height : float
        Wall height (m).
    reinforcement_length : float
        Reinforcement length (m).
    passes_sliding, passes_overturning : bool or None
        Each external check's verdict (None when not recorded).
    passes_bearing : bool or None
        None when the bearing check was not run.
    q_allowable : float or None
        The allowable bearing pressure given (kPa), or None.
    sigma_v : float or None
        Meyerhof bearing pressure W / (L - 2e) (kPa).
    FOS_sliding_required, FOS_overturning_required : float
        The required factors of safety (GEC-11: 1.5 and 2.0).
    backslope : dict or None
        With a sloping backfill, the GEC-11 basis used: Kab, the pressure
        plane height h, the thrust F_T and its components, the slope fill
        weight V2, and S_eq / sigma_2 / H1 for internal stability. None for
        a level backfill.
    external_terms : dict or None
        The ASD external-stability terms, unrounded, as the checks used
        them (GM-143, GM-164; ``mse._external_terms``): each vertical force
        and the checks that count it (``resists``: sliding and overturning;
        ``bearing``), the thrusts, V, M_stabilizing, V_bearing, M_bearing,
        Pa, M_overturning, delta_b, the bearing eccentricity e_B, L_eff and
        sigma_v. The calc package prints these. None on a result built by
        hand.
    choices : list of dict
        How a live-load surcharge over the reinforced zone was counted on
        the ASD path, when the wall carries one (GM-164,
        ``geotech_common.choices``; ``mse.live_load_choices``).
    """
    FOS_sliding: float = 0.0
    FOS_overturning: float = 0.0
    FOS_bearing: Optional[float] = None
    passes_external: Optional[bool] = False
    internal_results: List[Dict[str, Any]] = field(default_factory=list)
    all_pass_internal: bool = False
    wall_height: float = 0.0
    reinforcement_length: float = 0.0
    # Optional AASHTO/GEC-11 LRFD external-stability CDR set (from
    # check_external_stability_lrfd); populated only when analyze_mse_wall is
    # called with lrfd_external=True. None -> the ASD FOS path only.
    external_lrfd: Optional[Dict[str, Any]] = None
    passes_sliding: Optional[bool] = None
    passes_overturning: Optional[bool] = None
    passes_bearing: Optional[bool] = None
    q_allowable: Optional[float] = None
    sigma_v: Optional[float] = None
    FOS_sliding_required: float = 1.5
    FOS_overturning_required: float = 2.0
    backslope: Optional[Dict[str, Any]] = None
    external_terms: Optional[Dict[str, Any]] = None
    choices: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_levels(self) -> int:
        return len(self.internal_results)

    @property
    def bearing_checked(self) -> bool:
        """True when the external bearing check was run (q_allowable given)."""
        return self.FOS_bearing is not None

    @property
    def failed_levels(self) -> List[float]:
        """Depths (m) of the reinforcement levels that fail internally."""
        return [r.get("depth_m") for r in self.internal_results
                if not r.get("passes", False)]

    @property
    def checks_failed(self) -> List[str]:
        """The checks that were run and FAILED: 'sliding', 'overturning',
        'bearing', 'internal' (pullout, rupture or Le under 3 ft at some
        level)."""
        failed = []
        if self.passes_sliding is False:
            failed.append("sliding")
        if self.passes_overturning is False:
            failed.append("overturning")
        if self.bearing_checked and self.passes_bearing is False:
            failed.append("bearing")
        if (self.passes_external is False
                and self.passes_sliding is None
                and self.passes_overturning is None):
            failed.append("external")   # per-check verdicts not recorded
        if self.failed_levels:
            failed.append("internal")
        return failed

    @property
    def checks_not_run(self) -> List[str]:
        """The checks that were NOT run: 'bearing' when no q_allowable."""
        return [] if self.bearing_checked else ["bearing"]

    @property
    def all_checks_pass(self) -> Optional[bool]:
        """True only when every check was run and passed; False when any
        check failed; None when none failed but a check was not run."""
        if self.checks_failed:
            return False
        if self.checks_not_run:
            return None
        return True

    @property
    def verdict(self) -> str:
        """The result in plain words: failed checks first, then not run."""
        words = {
            "sliding": (f"sliding (FOS {self.FOS_sliding:.2f}, "
                        f"{self.FOS_sliding_required:.2f} required)"),
            "overturning": (f"overturning (FOS {self.FOS_overturning:.2f}, "
                            f"{self.FOS_overturning_required:.2f} required)"),
            "bearing": ("bearing (no q_allowable given)"
                        if not self.bearing_checked
                        else f"bearing (FOS {self.FOS_bearing:.2f})"),
            "external": "external stability",
            "internal": ("internal stability (pullout, rupture or Le under "
                         "3 ft at z = "
                         + ", ".join(f"{z:.2f}" for z in self.failed_levels)
                         + " m)"),
        }
        failed = self.checks_failed
        parts = [("FAILS: " + "; ".join(words[n] for n in failed) + ".")
                 if failed else "No check failed."]
        if self.checks_not_run:
            parts.append("Not checked: " + "; ".join(
                words[n] for n in self.checks_not_run) + ".")
        return " ".join(parts)

    def summary(self) -> str:
        def _status(passes):
            if passes is None:
                return "NOT ALL CHECKED"
            return "OK" if passes else "FAIL"

        def _check(passes):
            return "" if passes is None else f"  [{'OK' if passes else 'FAIL'}]"

        if self.bearing_checked:
            bearing = (f"FOS = {self.FOS_bearing:.2f}"
                       f"{_check(self.passes_bearing)}")
        else:
            bearing = ("not checked (no q_allowable given); sigma_v is "
                       "still reported")
        lines = [
            "=" * 60,
            "  MSE WALL RESULTS",
            "=" * 60,
            "",
            f"  Wall height:       {self.wall_height:.2f} m",
            f"  Reinforcement L:   {self.reinforcement_length:.2f} m",
        ]
        if self.backslope is not None:
            b = self.backslope
            crest = b.get("broken_backslope_height_m")
            lines.extend([
                f"  Backfill slope:    {b['backfill_slope_deg']:.1f} deg"
                + (f" (broken backslope, crest {crest:.2f} m above the top)"
                   if crest else ""),
                f"  Kab = {b['Kab']:.4f} (delta = "
                f"{b['interface_angle_deg']:.1f} deg), h = {b['h_m']:.2f} m, "
                f"F_T = {b['FT_kN_per_m']:.1f} kN/m",
            ])
        lines.extend([
            "",
            "  EXTERNAL STABILITY:",
            f"    Sliding:     FOS = {self.FOS_sliding:.2f}"
            f"{_check(self.passes_sliding)}",
            f"    Overturning: FOS = {self.FOS_overturning:.2f}"
            f"{_check(self.passes_overturning)}",
            f"    Bearing:     {bearing}",
        ])
        if self.sigma_v is not None:
            lines.append(f"    sigma_v:     {self.sigma_v:.1f} kPa")
        lines.extend([
            f"    External:    [{_status(self.passes_external)}]",
            "",
            f"  INTERNAL STABILITY ({self.n_levels} levels):",
        ])

        for r in self.internal_results:
            status = "OK" if r.get("passes", False) else "FAIL"
            # the level's own reinforcement when given per level (GM-266)
            name = f" {r['reinforcement']}" if "reinforcement" in r else ""
            rupture = (f", FOS_ru={r['FOS_rupture']:.2f}"
                       if "FOS_rupture" in r else "")
            lines.append(
                f"    z={r['depth_m']:.1f}m{name}: "
                f"Tmax={r['Tmax_kN_per_m']:.1f}, "
                f"FOS_po={r['FOS_pullout']:.2f}{rupture}  [{status}]"
            )

        failed = self.checks_failed
        lines.extend([
            f"    Internal:    [{_status(self.all_pass_internal)}]",
            "",
            "  Failed checks: " + (", ".join(failed) if failed else "none"),
        ])
        if self.checks_not_run:
            lines.append("  Not checked:   " + ", ".join(self.checks_not_run))
        if self.choices:
            from geotech_common.choices import choices_lines
            lines.extend(choices_lines(self.choices))
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "verdict": self.verdict,
            "checks_failed": self.checks_failed,
            "checks_not_run": self.checks_not_run,
            "all_checks_pass": self.all_checks_pass,
            "FOS_sliding": round(self.FOS_sliding, 3),
            "FOS_overturning": round(self.FOS_overturning, 3),
            "FOS_bearing": _round(self.FOS_bearing, 3),
            "passes_sliding": self.passes_sliding,
            "passes_overturning": self.passes_overturning,
            "passes_bearing": self.passes_bearing,
            "bearing_checked": self.bearing_checked,
            "q_allowable_kPa": self.q_allowable,
            "sigma_v_kPa": self.sigma_v,
            "passes_external": self.passes_external,
            "all_pass_internal": self.all_pass_internal,
            "n_levels": self.n_levels,
            "wall_height_m": self.wall_height,
            "reinforcement_length_m": self.reinforcement_length,
            "internal_results": self.internal_results,
        }
        if self.external_lrfd is not None:
            d["external_lrfd"] = self.external_lrfd
        if self.backslope is not None:
            d["backslope"] = self.backslope
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        return d
