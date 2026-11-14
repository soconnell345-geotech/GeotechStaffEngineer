"""
Rapid-drawdown slope stability as EM 1110-2-1902 (31 Oct 2003) Appendix G
prints it: the Corps of Engineers' 1970 two-stage procedure, Lowe and
Karafiath's two-stage procedure and the improved three-stage procedure of
Duncan, Wright and Wong.

When a reservoir is drawn down faster than the low-permeability zones of an
embankment can drain, those zones respond undrained: their strength is the
undrained strength set by the stresses they were consolidated under before
the drawdown, while the water load on the upstream face is gone.

The rule (BACKLOG GM-140 step 9): each stage is computed as Appendix G prints
it, each stage's factor of safety is reported, and the answer is the factor
of safety of the last stage computed.

First stage (G-3a, p. G-2)
    Effective-stress analysis with the pore pressures before drawdown. On
    each slice base

        sigma'_c = N' / dl                                        (Eq. G-1)
        tau_c    = S / dl                                         (Eq. G-2)

Strengths for the second stage (G-3b, pp. G-4 to G-8), ``lowe_karafiath``
and ``duncan_3stage``
    The strength of a soil consolidated isotropically (K_c = 1), from the R
    envelope's c_R and phi_R (p. G-6):

        d   = c_R cos(phi_R) cos(phi') / (1 - sin(phi_R))         (Eq. G-5)
        psi = arctan[ sin(phi_R) cos(phi') / (1 - sin(phi_R)) ]   (Eq. G-6)
        tau_ff(K_c = 1) = d + sigma'_c tan(psi)                   (Eq. G-9)

    or d and psi as measured (G-3b(1)(a) and (b), pp. G-4 to G-5; a layer's
    ``R_d`` and ``R_psi``). The strength of a soil consolidated at failure:

        tau_ff(K_c = K_f) = c' + sigma'_c tan(phi')               (Eq. G-10)
        K_f = (1 + sin phi') / (1 - sin phi')   where c' = 0      (Eq. G-7)
        K_f by Eq. G-8 where c' > 0 (``_Kf``)

    The slice's consolidation stress ratio, held between 1 and K_f
    (Eq. G-11, p. G-7; G-3b(2)(c)):

        K_c = [sigma'_c + tau_c (sin phi' + 1) / cos phi']
              / [sigma'_c + tau_c (sin phi' - 1) / cos phi']

    and the second-stage strength (Eq. G-12, p. G-8):

        tau_ff = [(K_f - K_c) tau_ff(K_c = 1) + (K_c - 1) tau_ff(K_c = K_f)]
                 / (K_f - 1)

Strength for the second stage, ``corps_2stage`` (G-2, pp. G-1 to G-2)
    The lower of the R envelope, c_R + sigma'_c tan(phi_R) with c_R and
    phi_R as entered, and the effective-stress envelope. No third stage.

Second stage (G-3c, p. G-8)
    Solved with tau_ff as cohesion, phi = 0 and zero pore pressure on the
    slices that do not drain freely; c', phi' and the pore pressures after
    drawdown on the free-draining ones; the water left against the slope is
    an external load. For ``lowe_karafiath`` and ``corps_2stage`` its factor
    of safety is the answer.

Drained check (G-3d, p. G-8), ``duncan_3stage``

        sigma    = N / dl, N from the second-stage computation    (Eq. G-13)
        sigma'_d = sigma - u, u after drawdown                    (Eq. G-14)
        s_d      = c' + sigma'_d tan(phi')                        (Eq. G-15)

    Where s_d is not below tau_ff on any slice, the second stage's factor
    of safety is the answer.

Third stage (G-3e, pp. G-8 to G-9)
    "The same conditions as for the second-stage computations" except that
    the slices whose drained strength is lower take c', phi' and the pore
    pressure after drawdown. Its factor of safety is the answer.

Water (G-4b, p. G-9; Table 2-1, p. 2-2)
    Before drawdown: the seepage line when one is given
    (``stage1_phreatic_points``), otherwise the level pool of the manual's
    own example, named in the result as that assumption. After drawdown:
    the line given (``drawdown_phreatic_points``), otherwise the level
    lowered pool, named.

A stage with no solution gives no number: ``FOS`` is None and the result
says which stage and why. Free-draining layers (no R envelope) keep their
effective strength in every stage. All units SI (m, kPa, kN/m3, degrees).

The manual is the owner's copy and is in no repository; the equations above
were read from its pages. Its worked example (G-4, pp. G-9 to G-22;
Figs. G-5, G-6, G-10b, G-11b) is pinned in
``slope_stability/tests/test_rapid_drawdown_em_appendix_g.py``.

References
----------
US Army Corps of Engineers (2003). EM 1110-2-1902, Slope Stability,
    Appendix G, pp. G-1 to G-23; Table 2-1, p. 2-2; C-4 (Modified Swedish
    Method).
Lowe, J., Karafiath, L. (1960). "Stability of earth dams upon drawdown." Proc.
    1st Pan-American Conf. on Soil Mech. and Found. Eng., Vol. 2, 537-552.
Duncan, J.M., Wright, S.G., Wong, K.S. (1990). "Slope stability during rapid
    drawdown." H. Bolton Seed Memorial Symposium, Vol. 2, 253-272.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from slope_stability.cracks import (
    counts_reverse_discards as _counts_reverse_discards,
    crack_choices as _crack_choices, discards_so_far as _discards_so_far,
)
from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices, slice_breaks
from slope_stability.gle import gle_fos
from slope_stability.results import SearchResult
from slope_stability.search import (
    _counts_refusals, _ordered_windows, _count_m_alpha_refusal,
    m_alpha_rule_in_force, _applies_m_alpha_rule, low_m_alpha_in_force,
    _note_trial_flags,
)
from slope_stability.methods import M_ALPHA_MIN, m_alpha_min as _m_alpha_min


_METHODS = ("corps_2stage", "duncan_3stage", "lowe_karafiath")
_SURFACE_TYPES = ("circular", "entry_exit", "noncircular", "noncircular_de")

# Procedures whose second-stage strength is Eq. G-12's, between the K_c = 1
# and K_c = K_f strengths. The Corps 1970 procedure uses its composite
# envelope instead. Only ``duncan_3stage`` has the drained check and the
# third stage.
_KC_INTERP_METHODS = ("duncan_3stage", "lowe_karafiath")
_STAGE3_METHODS = ("duncan_3stage",)

_FOS_NO_SOLUTION = 999.9        # the solvers' marker, never a result

#: What a caller who still names the removed input is told (EM G-3d).
STAGE3_NORMAL_RULE = (
    "stage3_effective_normal is no longer an input. The drained check of "
    "the three-stage procedure takes the normal force on each slice base "
    "from the second-stage computation: sigma = N / dl (EM 1110-2-1902 "
    "Eq. G-13, p. G-8), sigma'_d = sigma - u with the pore pressure after "
    "drawdown (Eq. G-14) and s_d = c' + sigma'_d tan(phi') (Eq. G-15). The "
    "two earlier bases ('fellenius', an Ordinary Method of Slices normal, "
    "and 'gle', a drained solution of the drawn-down slope) are not the "
    "manual's and were removed. Leave the input out.")

#: The method of slices the stages are solved by, and what the manual's
#: example of the 1970 procedure used (EM G-5, p. G-9).
CORPS_EXAMPLE_NOTE = (
    "The manual's worked example of the Corps of Engineers' 1970 procedure "
    "(EM 1110-2-1902 G-5, pp. G-9 to G-15: 3.49, then 1.35) was computed by "
    "the Modified Swedish Method with the side forces at a fixed "
    "inclination, 'for consistency with the original procedure'; the manual "
    "recommends Spencer's Method, which this result uses, and which gives a "
    "lower value on that example (1.24).")

_STRENGTH_COLUMNS = (
    "x_left", "x_right", "sigma_c", "tau_c", "Kc", "Kf", "tau_ff_Kc1",
    "tau_ff_Kcf", "tau_ff", "N_stage2", "u_after", "sigma_d", "s_d",
    "strength_used")


def _r(v, nd=3):
    return None if v is None else round(v, nd)


@dataclass
class RapidDrawdownResult:
    """Rapid drawdown on one slip surface, stage by stage.

    ``FOS`` is the factor of safety of the last stage computed
    (``answer_stage``): the second for the Corps 1970 and the Lowe and
    Karafiath procedures, and for the three-stage procedure where no
    slice's drained strength is below its undrained strength; otherwise the
    third. It is None where a stage has no solution (``no_solution`` says
    which and why).

    The per-slice lists follow the slices from low x to high x and are the
    columns of EM 1110-2-1902 Figs. G-10b and G-11b; a free-draining slice
    carries None in the undrained columns.
    """
    FOS: Optional[float] = None
    method: str = ""
    drawdown_from: float = 0.0
    drawdown_to: float = 0.0
    n_slices: int = 0
    n_undrained_slices: int = 0
    n_drained_substituted: int = 0     # slices on c', phi' in the third stage
    f_interslice: str = "constant"
    # each stage (None: not computed, or no solution)
    stage1_fos: Optional[float] = None
    stage2_fos: Optional[float] = None
    stage3_fos: Optional[float] = None
    answer_stage: Optional[int] = None
    no_solution: str = ""
    stage_method: str = ""             # the method of slices of the stages
    water_before: str = ""
    water_after: str = ""
    # per slice (kPa; N in kN/m)
    x_left: List[float] = field(default_factory=list)
    x_right: List[float] = field(default_factory=list)
    sigma_fc: List[float] = field(default_factory=list)     # sigma'_c, Eq. G-1
    tau_fc: List[float] = field(default_factory=list)       # tau_c, Eq. G-2
    Kc: List[Optional[float]] = field(default_factory=list)      # Eq. G-11
    Kf: List[Optional[float]] = field(default_factory=list)      # Eq. G-7/G-8
    tau_ff_kc1: List[Optional[float]] = field(default_factory=list)  # G-9
    tau_ff_kcf: List[Optional[float]] = field(default_factory=list)  # G-10
    tau_ff: List[Optional[float]] = field(default_factory=list)      # G-12
    N_stage2: List[Optional[float]] = field(default_factory=list)    # G-13's N
    u_after: List[float] = field(default_factory=list)
    sigma_d: List[Optional[float]] = field(default_factory=list)     # G-14
    s_d: List[Optional[float]] = field(default_factory=list)         # G-15
    strength_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Which arc of its circle the surface is, where the circle meets the
    #: ground at more than two points (GM-343; `analysis.lowest_arc`).
    arc: Optional[Dict[str, Any]] = None
    #: Statements that are not warnings (what the manual's example of the
    #: 1970 procedure was computed by).
    notes: List[str] = field(default_factory=list)
    #: What was chosen (``geotech_common.choices``): the procedure, the
    #: method of slices, the strengths, the water before and after
    #: drawdown, and (GM-140 step 5a, ``pore_pressure.water_statements``)
    #: what the pore pressure on the bases was taken as: the reading of a
    #: sloping line before drawdown, or after it where one is given, and
    #: none on total-stress layers.
    choices: List[Dict[str, Any]] = field(default_factory=list)
    #: Where a layer was given ground a layer listed before it also covers
    #: (`SlopeGeometry.layer_overlaps`; the first listed holds it).
    layer_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    #: The least m_alpha over the slices of the last stage computed at its
    #: FOS, each slice with the strength it was solved with (GM-127;
    #: `methods.m_alpha_values`).
    m_alpha_min: float = 1.0
    #: The least m_alpha of the first stage at its own FOS (GM-130: it was
    #: not checked). Reported, with a warning below 0.2 on a given
    #: surface; a search's refusal still reads the last stage (step 3 of
    #: GM-140 replaces that refusal, OWN-102).
    m_alpha_min_stage1: float = 1.0

    # the names earlier callers read
    @property
    def stage1_converged(self) -> bool:
        return self.stage1_fos is not None

    @property
    def last_stage_converged(self) -> bool:
        """The last stage computed has a solution (stage 2 for Lowe and
        Karafiath, for the Corps procedure and for a three-stage run no
        slice needed a third stage for; stage 3 otherwise). It was named
        ``stage3_converged``, which said true beside ``stage3_FOS`` null
        where no third stage was computed (GM-140 step 9 review)."""
        return self.FOS is not None

    def slice_table(self) -> List[Dict[str, Any]]:
        """One row per slice, low x to high x: the columns of EM
        1110-2-1902 Figs. G-10b and G-11b."""
        cols = (self.x_left, self.x_right, self.sigma_fc, self.tau_fc,
                self.Kc, self.Kf, self.tau_ff_kc1, self.tau_ff_kcf,
                self.tau_ff, self.N_stage2, self.u_after, self.sigma_d,
                self.s_d, self.strength_used)
        n = max((len(c) for c in cols), default=0)
        return [{name: (col[i] if i < len(col) else None)
                 for name, col in zip(_STRENGTH_COLUMNS, cols)}
                for i in range(n)]

    def summary(self) -> str:
        def f(v):
            return "not computed" if v is None else f"{v:.3f}"
        lines = [
            "=" * 60,
            f"  RAPID DRAWDOWN: {self.method}",
            "=" * 60,
            f"  Drawdown:              el {self.drawdown_from:g} -> {self.drawdown_to:g}",
            f"  Water before:          {self.water_before}",
            f"  Water after:           {self.water_after}",
            f"  Stages solved by:      {self.stage_method}",
            f"  Stage 1 FOS:           {f(self.stage1_fos)}",
            f"  Stage 2 FOS:           {f(self.stage2_fos)}",
            f"  Stage 3 FOS:           {f(self.stage3_fos)}",
            f"  Slices (undrained):    {self.n_slices} ({self.n_undrained_slices})",
            f"  Slices drained in stage 3: {self.n_drained_substituted}",
        ]
        if self.FOS is None:
            lines.append(f"  DRAWDOWN FOS:          none: {self.no_solution}")
        else:
            lines.append(f"  DRAWDOWN FOS:          {self.FOS:.3f} "
                         f"(stage {self.answer_stage})")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "FOS": _r(self.FOS, 4),
            "method": self.method,
            "answer_stage": self.answer_stage,
            "stage1_FOS": _r(self.stage1_fos, 4),
            "stage2_FOS": _r(self.stage2_fos, 4),
            "stage3_FOS": _r(self.stage3_fos, 4),
            "stage_method": self.stage_method,
            "drawdown_from": self.drawdown_from,
            "drawdown_to": self.drawdown_to,
            "water_before": self.water_before,
            "water_after": self.water_after,
            "n_slices": self.n_slices,
            "n_undrained_slices": self.n_undrained_slices,
            "n_drained_substituted": self.n_drained_substituted,
            "stage1_converged": self.stage1_converged,
            "last_stage_converged": self.last_stage_converged,
            "m_alpha_min_stage1": round(self.m_alpha_min_stage1, 3),
            "m_alpha_min": round(self.m_alpha_min, 3),
        }
        if self.FOS is None:
            d["no_solution"] = self.no_solution
        if self.x_left:
            d["slices"] = {
                "columns": {
                    "x_left": "slice edge, low-x side (m)",
                    "x_right": "slice edge, high-x side (m)",
                    "sigma_c": "sigma'_c = N'/dl, first stage (kPa; Eq. G-1)",
                    "tau_c": "tau_c = S/dl, first stage (kPa; Eq. G-2)",
                    "Kc": "consolidation stress ratio (Eq. G-11)",
                    "Kf": "stress ratio at failure (Eq. G-7 or G-8)",
                    "tau_ff_Kc1": "strength at K_c = 1 (kPa; Eq. G-9)",
                    "tau_ff_Kcf": "strength at K_c = K_f (kPa; Eq. G-10)",
                    "tau_ff": "second-stage strength (kPa; Eq. G-12, or "
                              "the Corps composite envelope)",
                    "N_stage2": "normal force on the base, second stage "
                                "(kN/m; Eq. G-13)",
                    "u_after": "pore pressure after drawdown (kPa)",
                    "sigma_d": "sigma'_d = N/dl - u (kPa; Eq. G-14)",
                    "s_d": "drained strength (kPa; Eq. G-15)",
                    "strength_used": "in the last stage computed: "
                                     "'undrained', 'drained' or "
                                     "'free-draining'",
                },
                "x_left": [_r(v) for v in self.x_left],
                "x_right": [_r(v) for v in self.x_right],
                "sigma_c": [_r(v, 2) for v in self.sigma_fc],
                "tau_c": [_r(v, 2) for v in self.tau_fc],
                "Kc": [_r(v) for v in self.Kc],
                "Kf": [_r(v) for v in self.Kf],
                "tau_ff_Kc1": [_r(v, 2) for v in self.tau_ff_kc1],
                "tau_ff_Kcf": [_r(v, 2) for v in self.tau_ff_kcf],
                "tau_ff": [_r(v, 2) for v in self.tau_ff],
                "N_stage2": [_r(v, 1) for v in self.N_stage2],
                "u_after": [_r(v, 2) for v in self.u_after],
                "sigma_d": [_r(v, 2) for v in self.sigma_d],
                "s_d": [_r(v, 2) for v in self.s_d],
                "strength_used": list(self.strength_used),
            }
        if self.choices:
            d["choices"] = [dict(c) for c in self.choices]
        if self.layer_overlaps:
            d["layer_overlaps"] = [dict(o) for o in self.layer_overlaps]
        if self.arc:
            d["arc"] = dict(self.arc)
        if self.notes:
            d["notes"] = list(self.notes)
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d


@dataclass
class RapidDrawdownSearchResult:
    """Critical-surface search under rapid-drawdown strengths.

    Composes the ordinary rich search result (the critical surface, the FOS
    grid, and the noncircular rejection diagnostics — ``search``) with the
    stage-level rapid-drawdown detail recomputed on that critical surface
    (``drawdown``: sigma'_fc / tau_fc / Kc / tau_ff per slice, stage-1 FOS,
    stage-3 substitutions). ``FOS`` is the search minimum.
    """
    FOS: float = 0.0
    method: str = ""
    surface_type: str = ""
    drawdown_from: float = 0.0
    drawdown_to: float = 0.0
    n_surfaces_evaluated: int = 0
    search: Optional[SearchResult] = None
    drawdown: Optional[RapidDrawdownResult] = None

    @property
    def critical(self):
        """The critical ``SlopeStabilityResult`` (or None)."""
        return self.search.critical if self.search is not None else None

    @property
    def no_surface_found(self) -> bool:
        """True when the search found no admissible surface: ``FOS`` then
        holds the solver's 999.9 marker, which is not a factor of safety and
        is never reported as one (BACKLOG GM-104)."""
        return self.critical is None

    def summary(self) -> str:
        crit = self.critical
        lines = [
            "=" * 60,
            f"  RAPID-DRAWDOWN CRITICAL-SURFACE SEARCH — {self.method}",
            "=" * 60,
            f"  Drawdown:            el {self.drawdown_from:g} -> {self.drawdown_to:g}",
            f"  Surface type:        {self.surface_type}",
            f"  Surfaces evaluated:  {self.n_surfaces_evaluated}",
            (f"  MINIMUM FOS:         {self.FOS:.3f}"
             if not self.no_surface_found else
             "  MINIMUM FOS:         none: no admissible surface was found "
             "in the search ranges (widen them, or check the geometry)"),
        ]
        if crit is not None:
            if crit.is_circular:
                lines.append(
                    f"  Critical circle:     ({crit.xc:.2f}, {crit.yc:.2f}), "
                    f"R={crit.radius:.2f}")
            else:
                lines.append(
                    f"  Critical surface:    noncircular "
                    f"({len(crit.slip_points or [])} pts)")
            lines.append(f"    entry/exit x:      {crit.x_entry:.2f} / "
                         f"{crit.x_exit:.2f}")
        if self.drawdown is not None:
            dd = self.drawdown
            for k, v in ((1, dd.stage1_fos), (2, dd.stage2_fos),
                         (3, dd.stage3_fos)):
                lines.append(f"  Stage {k} FOS:         "
                             + ("not computed" if v is None else f"{v:.3f}"))
            lines.append(f"  Drained in stage 3:  "
                         f"{dd.n_drained_substituted}")
        if self.search is not None and self.search.governing_surface:
            from slope_stability.search_supplements import describe_surface_kind
            lines.append(f"  Governing surface:   "
                         f"{describe_surface_kind(self.search.governing_surface)}")
            for w in self.search.warnings:
                lines.append(f"  WARNING: {w}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "FOS": None if self.no_surface_found else round(self.FOS, 4),
            "method": self.method,
            "surface_type": self.surface_type,
            "drawdown_from": self.drawdown_from,
            "drawdown_to": self.drawdown_to,
            "n_surfaces_evaluated": self.n_surfaces_evaluated,
        }
        if self.no_surface_found:
            d["no_surface_found"] = True
            d["note"] = ("No admissible slip surface was found in the search "
                         "ranges, so there is no factor of safety to report; "
                         "widen the entry/exit ranges or check the geometry.")
        if self.search is not None:
            d["search"] = self.search.to_dict()
        if self.drawdown is not None:
            d["drawdown_detail"] = self.drawdown.to_dict()
        return d


def _pool_geometry(geom: SlopeGeometry, water_elevation: float) -> SlopeGeometry:
    """Copy the geometry with the reservoir phreatic/pool surface at
    ``water_elevation`` (horizontal), spanning the full model width. The
    module's ponded-water handling supplies the external water load where the
    pool is above the ground, and the hydrostatic pore pressure below it."""
    g = copy.deepcopy(geom)
    x0 = geom.surface_points[0][0]
    x1 = geom.surface_points[-1][0]
    g.gwt_points = [(x0 - 1.0, water_elevation), (x1 + 1.0, water_elevation)]
    return g

def _base_layer(geom: SlopeGeometry, s):
    lay = geom.layer_at_point(s.x_mid, s.z_base)
    return lay if lay is not None else geom.soil_layers[-1]


def _undrained_in_drawdown(lay) -> bool:
    """A layer that does not drain freely: it carries an R envelope
    (``R_phi``) or the measured d and psi (``R_psi``)."""
    return (lay.R_phi is not None
            or getattr(lay, "R_psi", None) is not None)


def _Kf(phi_deg: float, c_prime: float = 0.0, sigma_fc: float = 0.0) -> float:
    """Failure principal-stress ratio Kf (EM 1110-2-1902 App. G, printed p. G-7).

    c' = 0: closed form Eq. G-7, Kf = (1 + sin phi')/(1 - sin phi').
    c' > 0: stress-dependent Eq. G-8,
        Kf = (sigma'_fc + c' cos phi')(1 + sin phi')
             / ((sigma'_fc - c' cos phi')(1 - sin phi')),
    which exceeds the G-7 value at finite stress (larger Kf -> smaller
    interpolation weight w -> stage-2 strength closer to the R envelope, the
    conservative direction) and converges to G-7 as sigma'_fc >> c'. For
    sigma'_fc <= c' cos phi' the ratio is unbounded (soil cannot reach the
    failure ratio at that stress) — capped at 1e6, which drives w -> 0.
    Verified against the printed equations 2026-07-18
    (module_work/wiki_verification/le_method_originals.md follow-up ticket).
    """
    sinphi = math.sin(math.radians(phi_deg))
    if sinphi >= 1.0:
        return 1e6
    K7 = (1.0 + sinphi) / (1.0 - sinphi)
    if c_prime <= 0.0 or sigma_fc <= 0.0:
        return K7
    ccos = c_prime * math.cos(math.radians(phi_deg))
    den = (sigma_fc - ccos) * (1.0 - sinphi)
    if den <= 1e-9:
        return 1e6
    return (sigma_fc + ccos) * (1.0 + sinphi) / den


def r_envelope_d_psi(R_c: float, R_phi_deg: float,
                     phi_deg: float) -> Tuple[float, float]:
    """The intercept d and slope angle psi (degrees) of the K_c = 1
    strength envelope, tau_ff against sigma'_c, from the R envelope's c_R
    and phi_R (EM 1110-2-1902 p. G-6):

        d   = c_R cos(phi_R) cos(phi') / (1 - sin(phi_R))         (Eq. G-5)
        psi = arctan[ sin(phi_R) cos(phi') / (1 - sin(phi_R)) ]   (Eq. G-6)

    The manual's example: c_R = 1,200 psf and phi_R = 16 deg with
    phi' = 30 deg give d = 1,379 psf and psi = 18.2 deg (Fig. G-5).
    """
    sr = math.sin(math.radians(R_phi_deg))
    cr = math.cos(math.radians(R_phi_deg))
    cp = math.cos(math.radians(phi_deg))
    if sr >= 1.0:
        raise ValueError("R_phi must be below 90 degrees")
    d = R_c * cr * cp / (1.0 - sr)
    psi = math.degrees(math.atan(sr * cp / (1.0 - sr)))
    return d, psi


def _layer_d_psi(lay) -> Tuple[float, float]:
    """A layer's d and psi: as measured where the layer gives them
    (``R_d``, ``R_psi``; EM G-3b(1)(a) and (b), pp. G-4 to G-5), otherwise
    from its c_R and phi_R by Eqs. G-5 and G-6."""
    psi = getattr(lay, "R_psi", None)
    if psi is not None:
        return float(getattr(lay, "R_d", 0.0) or 0.0), float(psi)
    return r_envelope_d_psi(lay.R_c, lay.R_phi, lay.phi)


def consolidation_stress_ratio(sigma_c: float, tau_c: float, phi_deg: float,
                               Kf: float) -> float:
    """The consolidation stress ratio K_c of a slice base (EM 1110-2-1902
    Eq. G-11, p. G-7; repeated as Eq. G-24, p. G-17), held between 1 and
    K_f (G-3b(2)(c), p. G-7):

        K_c = [sigma'_c + tau_c (sin phi' + 1) / cos phi']
              / [sigma'_c + tau_c (sin phi' - 1) / cos phi']

    The denominator is the minor principal consolidation stress times a
    positive factor; where it is not above zero the stress ratio is beyond
    K_f and K_f is returned (the upper bound).
    """
    sp = math.sin(math.radians(phi_deg))
    cp = math.cos(math.radians(phi_deg))
    if cp <= 1e-12:
        return Kf
    num = sigma_c + tau_c * (sp + 1.0) / cp
    den = sigma_c + tau_c * (sp - 1.0) / cp
    if den <= 1e-12:
        return Kf
    return max(1.0, min(num / den, Kf))


def second_stage_strength(Kc: float, Kf: float, tau_kc1: float,
                          tau_kcf: float) -> float:
    """The undrained strength for the second stage (EM 1110-2-1902
    Eq. G-12, p. G-8):

        tau_ff = [(K_f - K_c) tau_ff(K_c = 1) + (K_c - 1) tau_ff(K_c = K_f)]
                 / (K_f - 1)
    """
    if Kf <= 1.0:
        return tau_kc1
    return ((Kf - Kc) * tau_kc1 + (Kc - 1.0) * tau_kcf) / (Kf - 1.0)


def drained_check(N: float, dl: float, u_after: float, c_prime: float,
                  phi_deg: float) -> Tuple[float, float]:
    """The drained strength of a slice base after drawdown (EM 1110-2-1902
    G-3d, p. G-8), as (sigma'_d, s_d):

        sigma    = N / dl, N from the second-stage computation    (Eq. G-13)
        sigma'_d = sigma - u, u after drawdown                    (Eq. G-14)
        s_d      = c' + sigma'_d tan(phi')                        (Eq. G-15)

    Where sigma'_d is negative s_d = c': the manual prints s_d = 0.00 for
    such a slice of its example, where c' = 0 (Fig. G-11b, slice 1); for
    c' above zero this is the module's reading of it.
    """
    sig_d = N / dl - u_after
    s_d = c_prime + max(sig_d, 0.0) * math.tan(math.radians(phi_deg))
    return sig_d, s_d


# ── the stages' method of slices ───────────────────────────────────────────

@dataclass
class StageSolution:
    """One stage solved on one set of slices: its factor of safety (None
    where the method has no solution, with ``reason``) and, per slice in
    the slices' own order, the total normal force N on the base, the
    effective normal force N' = N - u dl and the mobilized shear S
    (kN/m)."""
    fos: Optional[float] = None
    reason: str = ""
    method: str = ""
    N: List[float] = field(default_factory=list)
    N_eff: List[float] = field(default_factory=list)
    S: List[float] = field(default_factory=list)
    theta_deg: Optional[float] = None       # side-force inclination

    @property
    def solved(self) -> bool:
        return self.fos is not None


def _usable(fos) -> bool:
    return (fos is not None and math.isfinite(fos)
            and 0.0 < fos < _FOS_NO_SOLUTION)


def _spencer_solution(slices, slip, f_interslice, tol, reinf) -> StageSolution:
    """Spencer's method (``f_interslice='constant'``) or the GLE with
    another side-force function: the rigorous engine's own converged
    solution, or no number."""
    name = ("Spencer's method" if f_interslice == "constant"
            else f"Morgenstern-Price (GLE, f(x) = {f_interslice})")
    try:
        res = gle_fos(slices, slip, f_interslice=f_interslice,
                      tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return StageSolution(reason=f"{name} stopped with an error ({exc})",
                             method=name)
    if not (res.converged and _usable(res.fos)):
        why = "; ".join(w for w in res.warnings
                        if not w.startswith("base tension"))
        return StageSolution(
            reason=f"{name} has no converged solution"
                   + (f" ({why})" if why else ""), method=name)
    return StageSolution(
        fos=res.fos, method=name, N=list(res.base_normal),
        N_eff=list(res.base_normal_eff), S=list(res.shear_mobilized),
        theta_deg=(math.degrees(math.atan(res.lam))
                   if f_interslice == "constant" else None))


def fixed_inclination_solution(slices, slip, theta_deg: float = 0.0,
                               equilibrium: str = "force",
                               tol: float = 1e-4,
                               reinf_forces=None) -> StageSolution:
    """The factor of safety with the side forces at a fixed, constant
    inclination ``theta_deg`` to the horizontal, by one equilibrium
    condition.

    ``equilibrium='force'`` is the force-equilibrium solution: the
    Modified Swedish Method of EM 1110-2-1902 C-4, which the manual's
    example of the Corps 1970 drawdown procedure uses with the side forces
    at 19.4 degrees (G-5, pp. G-9 to G-15). It is the GLE engine's
    force-equilibrium factor of safety at lambda = tan(theta) with a
    constant side-force function, found by its direct march
    (``gle._GLESystem._solve_marched``). ``equilibrium='moment'`` with
    ``theta_deg=0`` is the Simplified Bishop Method on a circle (the GLE's
    moment factor of safety at lambda = 0), which the manual's example of
    the improved procedure uses (pp. G-15 to G-20).

    A check method of the library, used to reproduce the manual's examples;
    it is not one of the agent's tools.
    """
    from slope_stability.gle import (
        _GLESystem, _normalize, _fit_axis_point, _assign_reinforcement)
    if equilibrium not in ("force", "moment"):
        raise ValueError("equilibrium must be 'force' or 'moment'")
    if equilibrium == "moment" and theta_deg != 0.0:
        raise ValueError(
            "equilibrium='moment' is offered at theta_deg = 0 only (the "
            "Simplified Bishop Method)")
    circular = getattr(slip, "is_circular", True)
    if equilibrium == "moment" and not circular:
        raise ValueError(
            "the Simplified Bishop Method needs a circular slip surface")
    name = ("the Simplified Bishop Method" if equilibrium == "moment" else
            "force equilibrium with side forces at a fixed "
            f"{theta_deg:g} deg (the Modified Swedish Method, EM C-4)")
    if len(slices) < 3:
        return StageSolution(reason="fewer than 3 slices", method=name)
    ns, mirrored = _normalize(slices)
    axis = (((-slip.xc if mirrored else slip.xc), slip.yc) if circular
            else _fit_axis_point(ns))
    reinf_n = _assign_reinforcement(ns, reinf_forces, mirrored)
    sys_ = _GLESystem(ns, axis, "constant", min(tol, 1e-4) * 0.1,
                      reinf=reinf_n)
    lam = math.tan(math.radians(theta_deg))
    n = len(ns)
    try:
        if equilibrium == "force":
            st = sys_._solve_marched(lam, 1.3)
            if st is None:
                return StageSolution(
                    reason=f"{name} has no solution with F > 0", method=name)
            F, V, N = st[1], st[3], list(st[4])
        else:
            st = sys_.solve_for_lambda(0.0, max_iter=400)
            if st is None:
                return StageSolution(
                    reason=f"{name} has no solution (the driving moment is "
                           "not positive)", method=name)
            F, V = st[0], [0.0] * (n + 1)
            res_m, den_m = sys_._moment_terms(F, V)
            if den_m <= 1e-9 or abs(res_m / den_m - F) > 1e-3:
                return StageSolution(
                    reason=f"{name} did not converge", method=name)
            N = [sys_._normal(s, F, 0.0) for s in ns]
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return StageSolution(reason=f"{name} stopped with an error ({exc})",
                             method=name)
    if not _usable(F):
        return StageSolution(reason=f"{name} has no usable factor of safety",
                             method=name)
    N_eff = [N[i] - s.u * s.length for i, s in enumerate(ns)]
    S = [(s.c * s.length + max(N_eff[i], 0.0) * s.tan_phi) / F
         for i, s in enumerate(ns)]
    if mirrored:
        N, N_eff, S = N[::-1], N_eff[::-1], S[::-1]
    return StageSolution(fos=F, method=name, N=N, N_eff=N_eff, S=S,
                         theta_deg=theta_deg)


def _stage_solver(stage_solver, f_interslice, tol, reinf):
    """The callable (slices, slip) -> StageSolution of ``stage_solver``:
    'gle' (Spencer or the GLE with ``f_interslice``), 'bishop', or
    ('force', theta_deg)."""
    if stage_solver in (None, "gle", "spencer"):
        return lambda sl, slip: _spencer_solution(sl, slip, f_interslice,
                                                  tol, reinf)
    if stage_solver == "bishop":
        return lambda sl, slip: fixed_inclination_solution(
            sl, slip, 0.0, "moment", tol, reinf)
    if (isinstance(stage_solver, (tuple, list)) and len(stage_solver) == 2
            and stage_solver[0] == "force"):
        theta = float(stage_solver[1])
        return lambda sl, slip: fixed_inclination_solution(
            sl, slip, theta, "force", tol, reinf)
    raise ValueError(
        "stage_solver must be 'gle', 'bishop' or ('force', theta_deg), got "
        f"{stage_solver!r}")


def _same_slices(a, b) -> bool:
    if len(a) != len(b):
        return False
    span = max(abs(a[-1].x_right - a[0].x_left), 1e-9) if a else 1.0
    return all(abs(s.x_left - t.x_left) <= 1e-7 * span
               and abs(s.x_right - t.x_right) <= 1e-7 * span
               for s, t in zip(a, b))


def _drawdown_choices(method, f_interslice, stage_solver, before_given,
                      after_given, drawdown_from, drawdown_to,
                      stage_method) -> List[Dict[str, Any]]:
    from geotech_common.choices import choice, how_given, MODULE
    names = {
        "duncan_3stage": "the improved three-stage procedure (Duncan, "
                         "Wright and Wong; method 'duncan_3stage')",
        "lowe_karafiath": "Lowe and Karafiath's two-stage procedure "
                          "(method 'lowe_karafiath')",
        "corps_2stage": "the Corps of Engineers' 1970 two-stage procedure "
                        "(method 'corps_2stage')",
    }
    out = [choice(
        "rapid-drawdown procedure", names[method],
        how_given(method, "duncan_3stage", name="method"),
        basis="EM 1110-2-1902 Appendix G: the improved procedure, G-3, "
              "pp. G-2 to G-9; the 1970 procedure, G-2, pp. G-1 to G-2",
        alternatives=list(names.values()))]
    out.append(choice(
        "method of slices for every stage", stage_method,
        (how_given(f_interslice, "constant", name="f_interslice")
         if stage_solver in (None, "gle", "spencer") else MODULE),
        basis="Spencer's Method is the one the manual recommends (EM "
              "1110-2-1902 G-5, p. G-9); its examples use the Simplified "
              "Bishop Method (pp. G-15 to G-20) and the Modified Swedish "
              "Method (pp. G-9 to G-15, G-20 to G-22)"))
    if method in _KC_INTERP_METHODS:
        out.append(choice(
            "strength of soil consolidated at K_c = 1",
            "d + sigma'_c tan(psi), d and psi from the R envelope's c_R "
            "and phi_R by Eqs. G-5 and G-6, or as measured where a layer "
            "gives R_d and R_psi",
            MODULE,
            basis="EM 1110-2-1902 Eqs. G-5, G-6 and G-9, pp. G-6 to G-7; "
                  "G-3b(1), pp. G-4 to G-5"))
    else:
        out.append(choice(
            "second-stage strength",
            "the lower of the R envelope (c_R and phi_R as entered) and "
            "the effective-stress envelope, at sigma'_c",
            MODULE, basis="EM 1110-2-1902 G-2, pp. G-1 to G-2"))
    if method in _STAGE3_METHODS:
        out.append(choice(
            "drained check before the third stage",
            "s_d = c' + sigma'_d tan(phi') with sigma'_d = N/dl - u, N "
            "from the second stage and u after drawdown; s_d = c' where "
            "sigma'_d is negative",
            MODULE,
            basis="EM 1110-2-1902 Eqs. G-13 to G-15, p. G-8; the manual "
                  "prints s_d = 0 for a slice with negative sigma'_d and "
                  "c' = 0 (Fig. G-11b), read here as s_d = c'"))
    out.append(choice(
        "water before drawdown",
        ("the seepage line given (stage1_phreatic_points)" if before_given
         else f"a level pool at elevation {drawdown_from:g} m, hydrostatic "
              "below it"),
        how_given(None if not before_given else True, None,
                  name="stage1_phreatic_points"),
        basis="EM 1110-2-1902 Table 2-1, p. 2-2 (steady seepage before "
              "drawdown); a level piezometric line at the pool is the "
              "assumption of the manual's own example, 'for simplicity' "
              "(G-4b, p. G-9)",
        alternatives=["the seepage line given (stage1_phreatic_points)",
                      "a level pool at drawdown_from_elevation"]))
    out.append(choice(
        "water after drawdown",
        ("the line given (drawdown_phreatic_points)" if after_given
         else f"a level pool at elevation {drawdown_to:g} m, hydrostatic "
              "below it"),
        how_given(None if not after_given else True, None,
                  name="drawdown_phreatic_points"),
        basis="EM 1110-2-1902 G-4b, p. G-9: a level piezometric line at "
              "the lowered pool in the manual's example; seepage analyses "
              "would often be appropriate",
        alternatives=["the line given (drawdown_phreatic_points)",
                      "a level pool at drawdown_to_elevation"]))
    return out


_SLOPING_LINE = "pore pressure under the sloping water line"


def _stage_water_statements(g_full, g_draw, after_given):
    """(choices, warnings) on the pore pressure of the bases (GM-140 step
    5a): every stage's slices take it from the same function
    (``pore_pressure.base_pore_pressure``), so a sloping line is read as
    ``geom.water_line_kind`` says. Stage 1's statements in full, with its
    warnings; for a line given after drawdown (``drawdown_phreatic_points``,
    GM-140 step 9) the reading of that line where it slopes. The grid, ru
    and the total-stress layers are the same soils and inputs in every
    stage and are said once, by stage 1."""
    from slope_stability.pore_pressure import water_statements
    stage1 = "Stage 1 (before drawdown): "
    found, notes = water_statements(g_full)
    choices = [dict(c, what=stage1 + c["what"]) for c in found]
    warnings = [stage1 + n for n in notes]
    if after_given:
        after, _ = water_statements(g_draw)
        choices += [dict(c, what="After drawdown: " + c["what"])
                    for c in after if c["what"] == _SLOPING_LINE]
    return choices, warnings


def rapid_drawdown_fos(geom: SlopeGeometry,
                       drawdown_from_elevation: float,
                       drawdown_to_elevation: float,
                       xc: float = None, yc: float = None, radius: float = None,
                       slip_surface=None,
                       method: str = "duncan_3stage",
                       f_interslice: str = "constant",
                       n_slices: int = 50,
                       tol: float = 1e-4,
                       stage1_phreatic_points: Optional[List] = None,
                       stage3_effective_normal: Optional[str] = None,
                       drawdown_phreatic_points: Optional[List] = None,
                       stage_solver=None,
                       extra_boundaries=(),
                       ) -> RapidDrawdownResult:
    """Rapid drawdown on a specified slip surface, as EM 1110-2-1902
    Appendix G prints it (see the module docstring for every equation and
    its page).

    Parameters
    ----------
    geom : SlopeGeometry
        Embankment geometry. A layer that does not drain freely carries the
        R envelope (``R_c``, ``R_phi``) beside its effective ``c_prime`` and
        ``phi``, or, for the Lowe and Karafiath and three-stage procedures,
        the measured ``R_d`` and ``R_psi`` (the K_c = 1 strength envelope's
        intercept and slope, tau_ff against sigma'_c). A layer with neither
        is free-draining and keeps its effective strength in every stage.
    drawdown_from_elevation, drawdown_to_elevation : float
        Reservoir surface elevation before and after drawdown (m).
    xc, yc, radius / slip_surface :
        The slip surface: a circle or an explicit surface.
    method : str
        'duncan_3stage' (default): the improved procedure, EM G-3; its
        answer is the third stage's factor of safety, or the second's where
        no slice's drained strength is below its undrained strength.
        'lowe_karafiath': the same first and second stages, no drained
        check; its answer is the second stage's. 'corps_2stage': the Corps
        of Engineers' 1970 procedure, EM G-2, with the composite envelope
        from c_R and phi_R as entered; its answer is the second stage's.
    f_interslice : str
        Side-force function of the GLE engine that solves every stage
        ('constant', the default, is Spencer's method).
    n_slices, tol : usual meanings. One set of slices serves every stage:
        the slices are cut at the water crossings before and after
        drawdown alike, and a surface on which the two cuts differ is
        refused (GM-135).
    stage1_phreatic_points : list of (x, z), optional
        The seepage line before drawdown (EM Table 2-1, p. 2-2). Left out,
        a level pool at ``drawdown_from_elevation`` with hydrostatic
        pressures below it: the assumption of the manual's own example
        (G-4b, p. G-9), named in the result. The line should start at the
        pool level on the upstream face, so the pool still loads the face.
    drawdown_phreatic_points : list of (x, z), optional
        The piezometric line after drawdown. Left out, a level pool at
        ``drawdown_to_elevation``, named in the result. The pore pressures
        after drawdown act on free-draining slices in the second stage, in
        the drained check (Eq. G-14) and on the drained slices of the third
        stage.
    stage3_effective_normal :
        Removed. Any value is refused with the rule (``STAGE3_NORMAL_RULE``):
        the drained check takes N from the second stage, Eq. G-13.
    stage_solver, extra_boundaries :
        Library check arguments, not on the agent's tools. ``stage_solver``
        solves every stage by another method of slices: 'bishop' (the
        Simplified Bishop Method of the manual's example, circles only) or
        ('force', theta_deg) (force equilibrium at a fixed side-force
        inclination, the Modified Swedish Method of its other examples;
        ``fixed_inclination_solution``). ``extra_boundaries`` adds slice
        edges, to cut the manual's own twelve slices.

    Returns
    -------
    RapidDrawdownResult
        ``FOS`` is None where a stage has no solution (``no_solution``).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
    if stage3_effective_normal is not None:
        raise ValueError(STAGE3_NORMAL_RULE)
    if drawdown_to_elevation >= drawdown_from_elevation:
        raise ValueError("drawdown_to_elevation must be below drawdown_from_elevation")

    if slip_surface is not None:
        slip = slip_surface
    else:
        if xc is None or yc is None or radius is None:
            raise ValueError("provide slip_surface or all of xc, yc, radius")
        slip = CircularSlipSurface(xc, yc, radius)

    for lay in geom.soil_layers:
        if (method == "corps_2stage" and lay.R_phi is None
                and getattr(lay, "R_psi", None) is not None):
            raise ValueError(
                f"Layer '{lay.name}' gives R_d and R_psi only. The Corps of "
                "Engineers' 1970 procedure (method 'corps_2stage') uses the "
                "R envelope's c_R and phi_R as entered (EM 1110-2-1902 G-2, "
                "pp. G-1 to G-2): give R_c and R_phi, or use "
                "'lowe_karafiath' or 'duncan_3stage'.")

    # Where a circle ends (GM-343): a circle that meets the ground at more
    # than two points is split there; each arc is put through the stages
    # as a surface of its own and the result is the lowest, named in
    # ``arc`` and the warnings.
    from slope_stability.analysis import (split_arcs, lowest_arc,
                                          checked_surface)
    slip = checked_surface(geom, slip)
    arcs = split_arcs(geom, slip)
    if arcs:
        return lowest_arc(
            geom, slip, arcs,
            lambda arc: rapid_drawdown_fos(
                geom, drawdown_from_elevation, drawdown_to_elevation,
                slip_surface=arc, method=method,
                f_interslice=f_interslice, n_slices=n_slices, tol=tol,
                stage1_phreatic_points=stage1_phreatic_points,
                drawdown_phreatic_points=drawdown_phreatic_points,
                stage_solver=stage_solver,
                extra_boundaries=extra_boundaries),
            what="drawdown factor of safety")

    # The surface's crack as printed (GM-140 step 4, ``cracks``): one crack
    # for every stage, its depth, where it depends on the factor of safety,
    # taken at the drawdown factor of safety itself. The geometry it fixes
    # (``_crack_fixed``) is carried into both water conditions below, so
    # every stage is cut on the one set of slices with the one crack.
    from slope_stability import cracks
    if cracks.may_need_solution(geom, slip):
        plan = cracks.plan_crack(geom, slip)
        if plan is not None and (plan.needs_solution or plan.discard):
            res, crack = cracks.solve_with_crack(
                geom, slip,
                lambda g: rapid_drawdown_fos(
                    g, drawdown_from_elevation, drawdown_to_elevation,
                    slip_surface=slip, method=method,
                    f_interslice=f_interslice, n_slices=n_slices, tol=tol,
                    stage1_phreatic_points=stage1_phreatic_points,
                    drawdown_phreatic_points=drawdown_phreatic_points,
                    stage_solver=stage_solver,
                    extra_boundaries=extra_boundaries),
                fos_of=lambda r: r.FOS, plan=plan)
            if crack.discarded:
                raise cracks.refusal_error(crack)
            if crack.depth > 0:
                res.warnings.append(crack.statement())
            return res

    # Reinforcement crossings on this surface, resolved as analyze_slope does
    # and applied at every stage (GM-53). None without reinforcement, which
    # leaves every stage solve unchanged.
    from slope_stability.search import trial_reinforcement
    reinf = trial_reinforcement(geom, slip)
    solve = _stage_solver(stage_solver, f_interslice, tol, reinf)

    # ── Water before and after drawdown (EM Table 2-1; G-4b) ────────────────
    before_given = stage1_phreatic_points is not None
    after_given = drawdown_phreatic_points is not None
    if before_given:
        g_full = copy.deepcopy(geom)
        g_full.gwt_points = [tuple(p) for p in stage1_phreatic_points]
        water_before = "the seepage line given (stage1_phreatic_points)"
    else:
        g_full = _pool_geometry(geom, drawdown_from_elevation)
        water_before = (
            f"a level pool at elevation {drawdown_from_elevation:g} m with "
            "hydrostatic pressures below it: assumed, as in the manual's "
            "own example (EM 1110-2-1902 G-4b, p. G-9), no seepage line "
            "being given")
    if after_given:
        g_draw = copy.deepcopy(geom)
        g_draw.gwt_points = [tuple(p) for p in drawdown_phreatic_points]
        water_after = "the line given (drawdown_phreatic_points)"
    else:
        g_draw = _pool_geometry(geom, drawdown_to_elevation)
        water_after = (
            f"a level pool at elevation {drawdown_to_elevation:g} m with "
            "hydrostatic pressures below it: assumed, as in the manual's "
            "own example (EM 1110-2-1902 G-4b, p. G-9), no line after "
            "drawdown being given")

    # ── One set of slices for every stage (GM-134, GM-135) ──────────────────
    x_ends = slip.find_entry_exit(geom)
    extra = [float(x) for x in extra_boundaries]
    breaks_full = slice_breaks(g_full, slip, *x_ends)
    breaks_draw = slice_breaks(g_draw, slip, *x_ends)
    sl1 = build_slices(g_full, slip, n_slices,
                       extra_boundaries=breaks_draw + extra)
    sl2 = build_slices(g_draw, slip, n_slices,
                       extra_boundaries=breaks_full + extra)
    if not _same_slices(sl1, sl2):
        raise ValueError(
            "Rapid drawdown needs one set of slices for every stage: the "
            f"slices before drawdown ({len(sl1)}) and after it ({len(sl2)}) "
            "differ on this surface (a slice was dropped in one water "
            "condition), so sigma'_c and tau_ff cannot be carried from "
            "slice to slice. Use another trial surface or slice count.")
    n = len(sl1)

    warnings: List[str] = []
    stage_method = None
    res = RapidDrawdownResult(
        method=method, drawdown_from=drawdown_from_elevation,
        drawdown_to=drawdown_to_elevation, n_slices=n,
        f_interslice=f_interslice, water_before=water_before,
        water_after=water_after, warnings=warnings,
        layer_overlaps=geom.layer_overlaps(),
        x_left=[s.x_left for s in sl1], x_right=[s.x_right for s in sl1],
        u_after=[s.pore_pressure for s in sl2])

    def _finish():
        res.stage_method = stage_method or ""
        res.choices = _drawdown_choices(
            method, f_interslice, stage_solver, before_given, after_given,
            drawdown_from_elevation, drawdown_to_elevation, res.stage_method)
        # The pore pressure on the bases (GM-140 step 5a): the reading of
        # a sloping line, named; stage 1's warnings, once.
        water, notes = _stage_water_statements(g_full, g_draw, after_given)
        res.choices += water
        warnings.extend(notes)
        if (method == "corps_2stage"
                and stage_solver in (None, "gle", "spencer")):
            res.notes.append(CORPS_EXAMPLE_NOTE)
        return res

    # ── First stage (G-3a): sigma'_c and tau_c on every base ────────────────
    s1 = solve(sl1, slip)
    stage_method = s1.method
    if not s1.solved:
        res.no_solution = (
            "the first stage (before drawdown) has no solution: "
            f"{s1.reason}. No factor of safety is given: the "
            "consolidation stresses of Eqs. G-1 and G-2 come from this "
            "stage's own solution, and no estimate stands in for it.")
        warnings.append(res.no_solution)
        return _finish()
    res.stage1_fos = s1.fos
    res.m_alpha_min_stage1 = _m_alpha_min(sl1, s1.fos, slip, "gle",
                                          m_alpha_rule_in_force())
    if res.m_alpha_min_stage1 < M_ALPHA_MIN:
        from slope_stability.analysis import m_alpha_statement
        w = m_alpha_statement(sl1, slip, "spencer", s1.fos,
                              given="rapid_drawdown_fos")
        if w:
            warnings.append("First stage (before drawdown): " + w)
    sigma_c = [s1.N_eff[i] / sl1[i].base_length for i in range(n)]   # G-1
    tau_c = [abs(s1.S[i]) / sl1[i].base_length for i in range(n)]    # G-2
    res.sigma_fc, res.tau_fc = sigma_c, tau_c

    # ── Strengths for the second stage (G-3b; G-2 for the 1970 procedure) ───
    layers = [_base_layer(geom, s) for s in sl2]
    undrained = [_undrained_in_drawdown(lay) for lay in layers]
    res.n_undrained_slices = sum(undrained)
    tension: List[int] = []
    for i in range(n):
        lay = layers[i]
        if not undrained[i]:
            for col in (res.Kc, res.Kf, res.tau_ff_kc1, res.tau_ff_kcf,
                        res.tau_ff):
                col.append(None)
            continue
        sc = sigma_c[i]
        if sc < 0.0:
            # A base in tension before drawdown: the manual prints no rule.
            # sigma'_c is taken as zero and the lower envelope strength
            # used; the slice is named in a warning.
            tension.append(i)
            sc = 0.0
        t_kcf = lay.c_prime + sc * math.tan(math.radians(lay.phi))   # G-10
        if method in _KC_INTERP_METHODS:
            d, psi = _layer_d_psi(lay)
            t_kc1 = d + sc * math.tan(math.radians(psi))             # G-9
            Kf = _Kf(lay.phi, lay.c_prime, sc)                       # G-7/8
            if i in tension:
                Kc, t = None, min(t_kc1, t_kcf)
            else:
                Kc = consolidation_stress_ratio(sc, tau_c[i], lay.phi, Kf)
                t = second_stage_strength(Kc, Kf, t_kc1, t_kcf)      # G-12
        else:
            # Corps 1970: composite of the R envelope as entered and the
            # effective-stress envelope (G-2)
            t_kc1 = lay.R_c + sc * math.tan(math.radians(lay.R_phi))
            Kc = Kf = None
            t = min(t_kc1, t_kcf)
        res.Kc.append(Kc)
        res.Kf.append(Kf)
        res.tau_ff_kc1.append(t_kc1)
        res.tau_ff_kcf.append(t_kcf)
        res.tau_ff.append(max(t, 0.0))
    if tension:
        warnings.append(
            f"{len(tension)} slice(s) have a negative effective normal "
            "stress on the base before drawdown (first at x = "
            f"{sl1[tension[0]].x_mid:.2f}). EM 1110-2-1902 Appendix G "
            "prints no rule for this; sigma'_c was taken as zero there and "
            "the lower of the two envelope strengths used.")

    def _undrained_slices(drained_idx=()):
        """The after-drawdown slices with tau_ff as cohesion, phi = 0 and
        no pore pressure on the undrained ones (G-3c); c', phi' and the
        pore pressure after drawdown on the free-draining ones and on
        ``drained_idx`` (G-3e)."""
        out = [copy.copy(s) for s in sl2]
        for i, s in enumerate(out):
            if undrained[i] and i not in drained_idx:
                s.c = res.tau_ff[i]
                s.phi = 0.0
                s.pore_pressure = 0.0
        return out

    # ── Second stage (G-3c) ─────────────────────────────────────────────────
    st2 = _undrained_slices()
    s2 = solve(st2, slip)
    last_slices = st2
    res.strength_used = ["undrained" if u else "free-draining"
                         for u in undrained]
    if not s2.solved:
        res.no_solution = ("the second stage (after drawdown, undrained "
                           f"strengths) has no solution: {s2.reason}")
        warnings.append(res.no_solution)
        return _finish()
    res.stage2_fos = s2.fos
    res.N_stage2 = list(s2.N)
    res.FOS, res.answer_stage = s2.fos, 2

    # ── Drained check (G-3d) and third stage (G-3e) ─────────────────────────
    if method in _STAGE3_METHODS:
        drained = set()
        for i in range(n):
            if not undrained[i]:
                res.sigma_d.append(None)
                res.s_d.append(None)
                continue
            lay = layers[i]
            sig_d, s_d = drained_check(                       # G-13 to G-15
                s2.N[i], sl2[i].base_length, sl2[i].pore_pressure,
                lay.c_prime, lay.phi)
            res.sigma_d.append(sig_d)
            res.s_d.append(s_d)
            if s_d < res.tau_ff[i]:
                drained.add(i)
        res.n_drained_substituted = len(drained)
        if drained:
            st3 = _undrained_slices(drained)
            s3 = solve(st3, slip)
            last_slices = st3
            for i in drained:
                res.strength_used[i] = "drained"
            if not s3.solved:
                res.FOS, res.answer_stage = None, None
                res.no_solution = (
                    "the third stage (drained strength on "
                    f"{len(drained)} slice(s)) has no solution: {s3.reason}")
                warnings.append(res.no_solution)
                return _finish()
            res.stage3_fos = s3.fos
            res.FOS, res.answer_stage = s3.fos, 3

    # m_alpha on the slices of the last stage computed at its FOS, each
    # slice with the strength it was solved with (GM-127). A search leaves a
    # surface below 0.2 out; on this given surface it is said.
    res.m_alpha_min = _m_alpha_min(last_slices, res.FOS, slip, "gle",
                                   m_alpha_rule_in_force())
    if res.m_alpha_min < M_ALPHA_MIN:
        from slope_stability.analysis import m_alpha_statement
        w = m_alpha_statement(last_slices, slip, "spencer", res.FOS,
                              given="rapid_drawdown_fos")
        if w:
            warnings.append("Drawn-down stage: " + w)
    return _finish()


@_counts_refusals
@_counts_reverse_discards
@_applies_m_alpha_rule
def search_rapid_drawdown(geom: SlopeGeometry,
                          drawdown_from_elevation: float,
                          drawdown_to_elevation: float,
                          method: str = "corps_2stage",
                          surface_type: str = "circular",
                          x_range=None, y_range=None,
                          nx: int = 10, ny: int = 10,
                          x_entry_range=None, x_exit_range=None,
                          n_trials: int = 500, n_points: int = 5,
                          seed: Optional[int] = None,
                          n_slices: int = 50,
                          tol: float = 1e-4,
                          f_interslice: str = "constant",
                          stage1_phreatic_points: Optional[List] = None,
                          stage3_effective_normal: Optional[str] = None,
                          toe_circles: bool = True,
                          infinite_slope: bool = True,
                          refine: bool = True,
                          time_budget_s: Optional[float] = None,
                          drawdown_phreatic_points: Optional[List] = None,
                          low_m_alpha: str = "flag",
                          ) -> RapidDrawdownSearchResult:
    """Find the critical slip surface UNDER RAPID-DRAWDOWN strengths.

    ``rapid_drawdown_fos`` evaluates ONE specified surface; this searches for the
    minimum-FOS surface with the *drawdown* strength substituted per trial. It is
    a thin composition: it reuses the module's existing search machinery
    (``grid_search`` / ``search_entry_exit`` / ``search_noncircular`` /
    ``search_de``) with the drawdown FOS wired in through their ``fos_fn`` hook,
    so the grid/entry-exit/random/DE loops, the radius optimisation, the
    entry/exit filtering, and the noncircular degenerate/jagged guards are all
    shared with the ordinary search — no search internals are duplicated here.

    Every trial surface is scored by the same procedure as
    ``rapid_drawdown_fos`` (EM 1110-2-1902 Appendix G, stage by stage), and a
    trial on which a stage has no solution is not scored. The three
    procedures are exposed via ``method``, and both circular and noncircular
    searches via ``surface_type``. The returned object carries the ordinary rich
    ``SearchResult`` plus the stage-level drawdown detail recomputed on the
    winning surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Base embankment geometry (NO reservoir load applied — the drawdown
        analysis adds the full/drawn-down pools itself). Low-permeability layers
        carry the R-envelope (``R_c``/``R_phi``); ``R_phi is None`` = free-draining.
    drawdown_from_elevation, drawdown_to_elevation : float
        Reservoir surface elevation before / after drawdown (m).
    method : str
        Drawdown stage method: 'corps_2stage' (default), 'lowe_karafiath', or
        'duncan_3stage'. See ``rapid_drawdown_fos``.
    surface_type : str
        'circular' (default, centre-grid + radius optimisation), 'entry_exit'
        (circular arcs between entry/exit windows), 'noncircular' (random
        polylines), or 'noncircular_de' (differential-evolution refinement).
    x_range, y_range : (float, float), optional
        Circle-centre search window (circular only; auto from geometry if None).
    nx, ny : int
        Grid resolution (circular) or entry/exit sampling (entry_exit). Default 10.
    x_entry_range, x_exit_range : (float, float), optional
        Entry/exit x windows (auto from geometry if None). The entry is the
        surface's low-x end and the exit its high-x end, whichever way the
        slope faces; windows given the other way round are swapped and the
        search's ``warnings`` says so (GM-124).
    n_trials, n_points, seed : int / int / int
        Random / DE noncircular controls (see ``search_noncircular``/``search_de``).
    n_slices, tol, f_interslice :
        Passed through to each per-surface ``rapid_drawdown_fos`` solve.
    stage1_phreatic_points : list of (x, z), optional
        The seepage line before drawdown (default None: a level pool, the
        assumption of the manual's own example, named in the result).
    drawdown_phreatic_points : list of (x, z), optional
        The piezometric line after drawdown (default None: a level lowered
        pool, named in the result). See ``rapid_drawdown_fos``.
    stage3_effective_normal :
        Removed; any value is refused with the rule (EM Eq. G-13). See
        ``rapid_drawdown_fos``.
    toe_circles, infinite_slope : bool
        Also score circles through each slope toe and circles passing above
        it (circular / entry_exit) and shallow planar surfaces on faces of cohesionless soil, with the
        drawdown FOS, and report the lowest -- as ``search_critical_surface``
        does (GM-54; see ``search_supplements``). Caller-given ranges are
        honoured; the automatic ones do not restrict these families. The
        result's ``search`` carries ``governing_surface``, ``candidates`` and
        ``warnings``. Default True.
    refine : bool
        Refine the lowest centre of the circular grid by a local compass
        search on the centre (GM-87; ``search.grid_search``). With a
        per-surface solve this costly only the lowest grid centre is
        refined, three step halvings, at most 12 centres. Default True.
    low_m_alpha : str
        A trial surface with m_alpha < 0.2 on a drawn-down slice at the
        drawdown FOS: 'flag' (default), kept and counted as flagged in
        ``search.trial_flags`` (Whitman and Bailey 1967, p. 487, print a
        warning, not a rejection; GM-140 step 3); 'refuse', left out and
        counted in ``search.n_rejected_m_alpha``, as before.
    time_budget_s : float or None
        Time budget of the whole search, in seconds, as
        ``search_critical_surface`` (GM-123): the main search scores coarse
        first and stops at 80 % of it, the supplementary families at it,
        and the result says what was covered. Default None, no limit;
        the agent's tools give ``search.SEARCH_TIME_BUDGET_S`` (180 s).

    Returns
    -------
    RapidDrawdownSearchResult

    Notes
    -----
    Each trial surface costs two or three LE solves (the stages), so a circular
    grid search is markedly heavier than an ordinary search — prefer a modest
    ``nx*ny`` (or ``surface_type='entry_exit'``) and bound the entry/exit windows.
    The per-surface work also deep-copies the geometry twice (full / drawn-down
    pools) inside ``rapid_drawdown_fos``; this is an O(n_surfaces) hotspot noted
    for a future pass, not optimised here.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
    if surface_type not in _SURFACE_TYPES:
        raise ValueError(
            f"surface_type must be one of {_SURFACE_TYPES}, got '{surface_type}'")
    if drawdown_to_elevation >= drawdown_from_elevation:
        raise ValueError(
            "drawdown_to_elevation must be below drawdown_from_elevation")
    if stage3_effective_normal is not None:
        raise ValueError(STAGE3_NORMAL_RULE)

    from slope_stability.search import (
        grid_search, search_entry_exit, search_noncircular, search_de, _FOS_MAX,
        search_budget, _count_no_solution,
    )

    def _dd_fos(g: SlopeGeometry, slip) -> float:
        """Per-surface rapid-drawdown FOS for the search loop."""
        try:
            r = rapid_drawdown_fos(
                g, drawdown_from_elevation, drawdown_to_elevation,
                slip_surface=slip, method=method, f_interslice=f_interslice,
                n_slices=n_slices, tol=tol,
                stage1_phreatic_points=stage1_phreatic_points,
                drawdown_phreatic_points=drawdown_phreatic_points)
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeError):
            return _FOS_MAX
        # A stage with no solution gives no number, so the trial is not
        # scored (GM-140 step 9), and it is counted under the procedure's
        # name in the search's ``no_solution``, as every search counts the
        # trials its method has no solution on (GM-140 step 2; the search
        # leaves a caller's own scoring to count its own). Two further
        # guards of the search, kept:
        #   (1) a large near-flat circle self-clips to a handful of slices
        #       whose solution is physically meaningless (a full-dam-
        #       spanning arc with 6 slices gave stage-1 FOS 0.18);
        #   (2) an implausibly low first-stage FOS: a real reservoir slope
        #       is stable at full pool, so stage-1 << 1 signals a degenerate
        #       consolidation solve, not a real drawdown mechanism.
        if r.FOS is None:
            _count_no_solution(method)
            return _FOS_MAX
        if r.n_slices < max(8, n_slices // 2):
            return _FOS_MAX
        if r.stage1_fos < 0.5:
            return _FOS_MAX
        # m_alpha < 0.2 on a drawn-down slice at the drawdown FOS: kept and
        # counted as flagged, as every search does by default (GM-140 step
        # 3, OWN-102; Whitman and Bailey 1967, p. 487, print a warning, not
        # a rejection); left out and counted under ``low_m_alpha='refuse'``
        # (GM-127).
        low = r.m_alpha_min < M_ALPHA_MIN
        if low and low_m_alpha_in_force() == "refuse":
            _count_m_alpha_refusal()
            return _FOS_MAX
        _note_trial_flags("rapid_drawdown", r.FOS,
                          ["low_m_alpha"] if low else [], slip)
        return r.FOS

    x_entry_range, x_exit_range, window_note = _ordered_windows(
        x_entry_range, x_exit_range)
    given = dict(x_entry_range=x_entry_range, x_exit_range=x_exit_range,
                 x_range=x_range, y_range=y_range)
    windows_auto = (x_entry_range is None, x_exit_range is None)
    grid_auto = x_range is None and y_range is None

    # Ranges not given are set from the slope (as search_critical_surface).
    from slope_stability.search_ranges import resolve_ranges
    x_entry_range, x_exit_range, x_range, y_range, auto = resolve_ranges(
        geom, x_entry_range, x_exit_range, x_range, y_range)

    with search_budget(time_budget_s) as budget:
        search = _drawdown_search(
            geom, surface_type, x_range, y_range, nx, ny, x_entry_range,
            x_exit_range, n_trials, n_points, seed, n_slices, tol, _dd_fos,
            refine, budget, windows_auto=windows_auto, grid_auto=grid_auto,
            toe_circles=toe_circles, infinite_slope=infinite_slope,
            auto=auto, given=given, window_note=window_note)

    # Recompute the stage-level detail on the winning surface.
    dd = None
    crit = search.critical
    if crit is not None:
        if crit.is_circular:
            from slope_stability.slip_surface import circle_of
            crit_slip = circle_of(crit)     # the governing arc's two ends
        else:
            crit_slip = PolylineSlipSurface(points=list(crit.slip_points))
        dd = rapid_drawdown_fos(
            geom, drawdown_from_elevation, drawdown_to_elevation,
            slip_surface=crit_slip, method=method, f_interslice=f_interslice,
            n_slices=n_slices, tol=tol,
            stage1_phreatic_points=stage1_phreatic_points,
            drawdown_phreatic_points=drawdown_phreatic_points)

    return RapidDrawdownSearchResult(
        FOS=(crit.FOS if crit is not None else _FOS_MAX),
        method=method, surface_type=surface_type,
        drawdown_from=drawdown_from_elevation,
        drawdown_to=drawdown_to_elevation,
        n_surfaces_evaluated=search.n_surfaces_evaluated,
        search=search, drawdown=dd,
    )


def _drawdown_search(geom, surface_type, x_range, y_range, nx, ny,
                     x_entry_range, x_exit_range, n_trials, n_points, seed,
                     n_slices, tol, _dd_fos, refine, budget, *, windows_auto,
                     grid_auto, toe_circles, infinite_slope, auto, given,
                     window_note=None):
    """The search of `search_rapid_drawdown` under its time budget: the
    main search, its supplementary families and what it chose."""
    from slope_stability.search import (
        grid_search, search_entry_exit, search_noncircular, search_de,
    )
    if surface_type == "circular":
        search = grid_search(
            geom, x_range, y_range, nx, ny, method="spencer",
            n_slices=n_slices, tol=tol, x_entry_range=x_entry_range,
            x_exit_range=x_exit_range, fos_fn=_dd_fos, refine=refine)
    elif surface_type == "entry_exit":
        search = search_entry_exit(
            geom, x_entry_range, x_exit_range, n_entry=nx, n_exit=ny,
            method="spencer", n_slices=n_slices, tol=tol, fos_fn=_dd_fos)
    elif surface_type == "noncircular":
        search = search_noncircular(
            geom, x_entry_range, x_exit_range, n_trials=n_trials,
            n_points=n_points, n_slices=n_slices, tol=tol, seed=seed,
            method="spencer", fos_fn=_dd_fos)
    else:  # noncircular_de
        search = search_de(
            geom, x_entry_range, x_exit_range, n_points=max(n_points, 5),
            method="spencer", n_slices=n_slices, tol=tol, seed=seed,
            fos_fn=_dd_fos)

    # Toe circles and shallow planar surfaces, scored with the drawdown FOS.
    from slope_stability.search_supplements import (
        apply_supplements, note_budget, CIRCULAR, NONCIRCULAR,
    )
    budget.supplements()
    circular = surface_type in ("circular", "entry_exit")
    search = apply_supplements(
        geom, search, search_kind=CIRCULAR if circular else NONCIRCULAR,
        method="spencer", n_slices=n_slices, tol=tol,
        windows_used=(x_entry_range, x_exit_range),
        windows_auto=windows_auto,
        grid_used=(x_range, y_range) if surface_type == "circular" else None,
        grid_auto=grid_auto, toe_circles=toe_circles,
        infinite_slope=infinite_slope, fos_fn=_dd_fos, auto_ranges=auto,
        **given)
    # What the search chose (GM-115); every trial is the drawdown solve.
    # The surfaces left out for m_alpha < 0.2 on a drawn-down slice so far,
    # the supplementary families' included (GM-127).
    from slope_stability.search_supplements import search_choices
    from slope_stability.search import _m_alpha_refusals_so_far
    search.n_rejected_m_alpha = _m_alpha_refusals_so_far()
    from slope_stability.search import _flag_tally_so_far, flag_records
    search.trial_flags = flag_records(
        _flag_tally_so_far(),
        rename={"main": CIRCULAR if circular else NONCIRCULAR})
    search.choices = search_choices(
        search, surface_type=surface_type, toe_circles=toe_circles,
        infinite_slope=infinite_slope,
        surface_types=("circular", "entry_exit", "noncircular",
                       "noncircular_de"), budget=budget,
        low_m_alpha=low_m_alpha_in_force())
    search.choices += _crack_choices(
        geom, _discards_so_far(),
        circles=surface_type in ("circular", "entry_exit") or toe_circles)
    if window_note:
        search.warnings.insert(0, window_note)
    note_budget(search, budget)     # a search cut short says so first
    return search
