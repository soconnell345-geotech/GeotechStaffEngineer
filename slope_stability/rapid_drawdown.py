"""
Rapid-drawdown slope stability — USACE 2-stage, Lowe & Karafiath, and
Duncan-Wright-Wong 3-stage.

When a reservoir is drawn down faster than the low-permeability zones of an
embankment can drain, those zones respond UNDRAINED: the pore pressures set up
by the pre-drawdown steady seepage persist while the stabilizing water load on
the upstream face is removed. The available strength is the undrained strength
mobilized from the pre-drawdown CONSOLIDATION stresses, not the (higher) drained
strength that the post-drawdown effective stresses would imply.

All procedures share STAGE 1 — an effective-stress ("S") analysis at the FULL
pool that gives the consolidation stress on each slice base: the effective
normal sigma'_fc and (for the Kc methods) the shear tau_fc. They differ only in
how STAGE 2 turns those consolidation stresses into an undrained strength
tau_ff, and whether a third (drained-substitution) stage is run. Every method is
wired to the rigorous GLE / Spencer engine by overriding each low-permeability
slice's base strength to c = tau_ff, phi = 0 and re-solving at the DRAWN-DOWN
pool. Three methods are implemented (``method=``):

``corps_2stage`` — USACE / Army Corps 2-stage (EM 1110-2-1902, 1970):
  Stage 2  Undrained strength of each low-permeability slice =
           min( R-envelope(sigma'_fc), drained-envelope(sigma'_fc) )
           (the "combined" envelope — the R strength capped by the effective
           strength so negative-pore-pressure over-strength is not used).
           No third stage.

``lowe_karafiath`` — Lowe & Karafiath 2-stage (1960):
  Stage 2  Undrained strength interpolated with the consolidation stress ratio
           Kc = sigma'_1c/sigma'_3c between two envelopes plotted as tau_ff vs
           sigma'_fc: the Kc=1 (isotropically-consolidated / R) envelope (lower
           bound) and the Kc=Kf (effective-stress / drained) envelope (upper
           bound).  tau_ff = tau_R + (Kc-1)/(Kf-1) * (tau_drained - tau_R).
           No third stage. This is exactly the Duncan-Wright-Wong procedure
           below WITHOUT its stage 3 (Duncan, Wright & Wong 1990 note the DWW
           method is Lowe & Karafiath's with the drained-strength check added).

``duncan_3stage`` — Duncan, Wright & Wong 3-stage (1990; Duncan, Wright &
  Brandon 2014, Ch. 9):
  Stage 2  the SAME Kc-interpolated undrained strength as Lowe & Karafiath.
  Stage 3  For each slice the post-drawdown DRAINED strength is computed with
           the drawn-down pore pressures; where it is LESS than the stage-2
           undrained strength it is substituted, and the FOS re-computed.

Free-draining layers (``R_phi is None``) keep their effective strength in every
stage. All units SI (m, kPa, kN/m3, degrees).

References
----------
US Army Corps of Engineers (1970). EM 1110-2-1902, Stability of Earth and
    Rock-Fill Dams, Appendix G.
Lowe, J., Karafiath, L. (1960). "Stability of earth dams upon drawdown." Proc.
    1st Pan-American Conf. on Soil Mech. and Found. Eng., Vol. 2, 537-552.
Duncan, J.M., Wright, S.G., Wong, K.S. (1990). "Slope stability during rapid
    drawdown." H. Bolton Seed Memorial Symposium, Vol. 2, 253-272.
Duncan, Wright & Brandon (2014). Soil Strength and Slope Stability, Ch. 9.

Provenance (VERIFIED-PRIMARY, 2026-07-18 wiki-verification): EM 1110-2-1902
(31 Oct 2003, in the owner's library) App. G confirmed first-hand — Eq. G-12
(p. G-8) is an exact algebraic identity with the coded s_R + w*(s_D - s_R)
interpolation, and the Corps 2-stage composite rule "lower bound of the R and S
envelopes" (p. G-2) matches min(s_R, s_D) exactly. The wave-2 flagged
limitation (Kf used the c'=0 Eq. G-7 form even when c' > 0) was FIXED the same
day: _Kf now implements the printed stress-dependent Eq. G-8 for c' > 0
(p. G-7), reducing exactly to G-7 when c' = 0 — so every published validation
case (all effective-c'=0) is byte-identical, while c' > 0 layers now follow
the EM (conservative direction). Ledger:
module_work/wiki_verification/le_method_originals.md.
"""

import bisect
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices, slice_breaks
from slope_stability.gle import gle_fos
from slope_stability.results import SearchResult
from slope_stability.search import (
    _counts_refusals, _ordered_windows, _count_m_alpha_refusal,
    m_alpha_rule_in_force,
)
from slope_stability.methods import M_ALPHA_MIN, m_alpha_min as _m_alpha_min


_METHODS = ("corps_2stage", "duncan_3stage", "lowe_karafiath")
_SURFACE_TYPES = ("circular", "entry_exit", "noncircular", "noncircular_de")

# Methods whose stage-2 undrained strength is the Kc-interpolated value between
# the Kc=1 (R / IC-U) and Kc=Kf (drained) envelopes (Lowe & Karafiath 1960; also
# the Duncan-Wright-Wong stage-2). The rest use the Corps combined R/effective
# envelope. Only ``duncan_3stage`` runs the third (drained-substitution) stage.
_KC_INTERP_METHODS = ("duncan_3stage", "lowe_karafiath")
_STAGE3_METHODS = ("duncan_3stage",)


def _nearest_index(sorted_xs: List[float], x: float) -> int:
    """Index of the value in ASCENDING ``sorted_xs`` nearest ``x``.

    Bit-for-bit equivalent to ``min(range(len(sorted_xs)),
    key=lambda k: abs(sorted_xs[k] - x))`` — including the tie rule (``min``
    returns the FIRST/lowest index, i.e. the LEFT neighbour) — but O(log n)
    instead of O(n). Used to map the stage-3 slices back onto the stage-1
    slices by x_mid (the two slice lists are each built left-to-right, so their
    x_mid are ascending).
    """
    n = len(sorted_xs)
    pos = bisect.bisect_left(sorted_xs, x)
    if pos == 0:
        return 0
    if pos >= n:
        return n - 1
    left = pos - 1
    # sorted_xs[left] < x <= sorted_xs[pos]; on a tie pick the LEFT (lower index),
    # matching min's first-index rule since left < pos.
    if x - sorted_xs[left] <= sorted_xs[pos] - x:
        return left
    return pos


@dataclass
class RapidDrawdownResult:
    """Rapid-drawdown factor-of-safety result."""
    FOS: float = 0.0
    method: str = ""
    drawdown_from: float = 0.0
    drawdown_to: float = 0.0
    n_slices: int = 0
    n_undrained_slices: int = 0
    n_drained_substituted: int = 0     # stage-3 substitutions (3-stage)
    f_interslice: str = "constant"
    # per-slice diagnostics
    sigma_fc: List[float] = field(default_factory=list)     # consolidation eff. normal, kPa
    tau_fc: List[float] = field(default_factory=list)       # consolidation shear, kPa
    Kc: List[float] = field(default_factory=list)
    tau_ff: List[float] = field(default_factory=list)        # stage-2 undrained strength, kPa
    stage1_fos: float = 0.0
    stage1_converged: bool = True
    stage3_converged: bool = True
    warnings: List[str] = field(default_factory=list)
    #: The least m_alpha over the drawn-down slices at the drawdown FOS, with
    #: each slice's own drawdown strength (GM-127; `methods.m_alpha_values`).
    #: The search leaves out a surface below 0.2, as Slide2 does (-112).
    m_alpha_min: float = 1.0

    def summary(self) -> str:
        return "\n".join([
            "=" * 60,
            f"  RAPID DRAWDOWN — {self.method}",
            "=" * 60,
            f"  Drawdown:              el {self.drawdown_from:g} -> {self.drawdown_to:g}",
            f"  Stage-1 (full-pool) FOS: {self.stage1_fos:.3f}",
            f"  Slices (undrained):    {self.n_slices} ({self.n_undrained_slices})",
            f"  Drained substitutions: {self.n_drained_substituted}",
            f"  DRAWDOWN FOS:          {self.FOS:.3f}",
            "=" * 60,
        ])

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "FOS": round(self.FOS, 4),
            "method": self.method,
            "drawdown_from": self.drawdown_from,
            "drawdown_to": self.drawdown_to,
            "stage1_FOS": round(self.stage1_fos, 4),
            "n_slices": self.n_slices,
            "n_undrained_slices": self.n_undrained_slices,
            "n_drained_substituted": self.n_drained_substituted,
            "stage1_converged": self.stage1_converged,
            "stage3_converged": self.stage3_converged,
        }
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
            lines.append(f"  Stage-1 (full) FOS:  {self.drawdown.stage1_fos:.3f}")
            lines.append(f"  Drained subst.:      "
                         f"{self.drawdown.n_drained_substituted}")
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


def _fellenius_effective_normal(s) -> float:
    """Fellenius / OMS estimate of the effective normal stress on a slice
    base, W cos(alpha) / l - u, with the free water ponded on the ground
    above the slice taken out of both terms.

    Ponded water is a hydrostatic load: it raises the total normal stress
    and the pore pressure on the base by the same amount and changes no
    effective stress (FHWA NHI-06-088 p. 6-13: total weights with the
    boundary water forces, or submerged weights, give the same result).
    The OMS estimate leaves out the slice's side forces, which carry that
    balance, so with the pond's weight in W and its head in u it counted
    the pond's pressure in full on the base but its weight only at
    cos^2(alpha). Under a deep pool on an inclined base sigma' came out
    negative and was clamped to zero, and the stage-3 drained substitution
    then gave no strength at all: the true Pilarcitos section drawn down,
    a toe circle under the remaining pool, DWW 0.052 (2.048 with the
    rigorous 'gle' normal). Without a pond the value is the historical one.
    """
    q = s.pond_force / s.width if s.width > 0 else 0.0
    return max((s.weight - s.pond_force) * math.cos(s.alpha) / s.base_length
               - (s.pore_pressure - q), 0.0)


def _base_layer(geom: SlopeGeometry, s):
    lay = geom.layer_at_point(s.x_mid, s.z_base)
    return lay if lay is not None else geom.soil_layers[-1]


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


def _consolidation_Kc(sigma_fc: float, tau_fc: float, alpha: float,
                      phi_deg: float, c_prime: float = 0.0) -> float:
    """Effective consolidation stress ratio Kc = sigma'_1c/sigma'_3c for a slice.

    The base plane is inclined at ``alpha`` to the horizontal; assuming the
    major principal consolidation stress is vertical (gravity / K0 field), the
    base-plane stresses relate to the principal stresses by
        sigma'_fc = p + q*cos(2*alpha),   tau_fc = q*sin(2*alpha)
    with p = (s1+s3)/2, q = (s1-s3)/2. Solving for the principal stresses gives
    Kc, clamped to [1, Kf]. Near a horizontal base (sin 2alpha -> 0) the ratio
    is indeterminate from the plane stresses, so the at-rest value 1/(1-sin phi')
    is used.
    """
    sinphi = math.sin(math.radians(phi_deg))
    Kf = _Kf(phi_deg, c_prime, sigma_fc)
    s2a = math.sin(2.0 * alpha)
    if abs(s2a) < 0.05 or sigma_fc <= 0:
        Kc = 1.0 / (1.0 - sinphi) if sinphi < 1 else Kf
        return max(1.0, min(Kc, Kf))
    q = abs(tau_fc / s2a)
    p = sigma_fc - (tau_fc / s2a) * math.cos(2.0 * alpha)
    s1, s3 = p + q, p - q
    if s3 <= 1e-6:
        return Kf
    return max(1.0, min(s1 / s3, Kf))


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
                       stage3_effective_normal: str = "fellenius"
                       ) -> RapidDrawdownResult:
    """Rapid-drawdown factor of safety on a specified slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Embankment geometry. Low-permeability layers must carry the R-envelope
        (``R_c``, ``R_phi``) in addition to the effective ``c_prime``/``phi``;
        layers with ``R_phi is None`` are free-draining (drained throughout).
    drawdown_from_elevation, drawdown_to_elevation : float
        Reservoir surface elevation before and after drawdown (m).
    xc, yc, radius / slip_surface :
        The (specified) slip surface. Provide a circle or an explicit surface.
    method : str
        'duncan_3stage' (default), 'lowe_karafiath', or 'corps_2stage'.
        'lowe_karafiath' uses the same Kc-interpolated stage-2 undrained strength
        as 'duncan_3stage' but omits stage 3 (the drained-strength check), so it
        is never more conservative than the 3-stage; 'corps_2stage' uses the
        combined min(R, drained) envelope. See the module docstring.
    f_interslice : str
        Interslice function for the GLE engine ('constant' = Spencer).
    n_slices, tol : usual meanings.
    stage1_phreatic_points : list of (x, z), optional
        Steady-seepage phreatic surface used for the STAGE-1 consolidation
        stresses only. Default ``None`` reproduces the conservative
        no-through-seepage bound: a flat pool at ``drawdown_from_elevation``
        with hydrostatic pore pressure ``u = gamma_w * (pool - z)`` everywhere.
        When a low-permeability embankment has an established steady-state
        seepage regime under full pool, the phreatic surface DECLINES through
        the dam (head dissipates from the upstream face toward the downstream),
        so the true stage-1 pore pressures are LOWER and the consolidation
        effective stresses (hence the mobilized undrained strengths) HIGHER.
        Slide2 / EM 1110-2-1902 use this steady-seepage field. Supplying the
        flow-net (or a Casagrande) phreatic line here reproduces that condition;
        the drawn-down stage still applies the flat drawdown pool. The line
        should start at the pool elevation on the upstream face so the external
        reservoir load on the submerged face is preserved.
    stage3_effective_normal : str, optional
        How the STAGE-3 (3-stage only) post-drawdown DRAINED effective normal
        stress ``sigma'_post`` — the stress the drained substitution strength
        ``c' + sigma'_post*tan phi'`` is evaluated at — is estimated. Only used by
        ``method='duncan_3stage'``; ignored by the 2-stage method (which has no
        stage 3).

        * ``'fellenius'`` (**default**) — the Fellenius / OMS per-slice estimate
          ``sigma'_post = W*cos(alpha)/l - u`` on the drawn-down geometry, with
          water ponded above a slice taken out of both W and u (a pond changes
          no effective stress; see ``_fellenius_effective_normal``). Without a
          pond it is the historical value, byte-for-byte.
        * ``'gle'`` — the RIGOROUS effective normal from a GLE solve of the
          drawn-down slope with the drained strengths (the same
          ``base_normal_eff`` basis stage 1 uses for ``sigma'_fc``). The Fellenius
          estimate neglects interslice forces and systematically UNDER-predicts
          N' under the drawn-down pore pressures, which spuriously fires the
          stage-3 drained substitution (a lower drained strength is substituted on
          slices where the true drained strength actually exceeds the undrained).
          Using the rigorous normal — consistent with stage 1 — removes those
          spurious substitutions. Falls back to ``'fellenius'`` if the drained GLE
          solve does not converge. For Slide2 #96 this lifts the 3-stage FOS from
          1.235 to 1.31 (flat stage-1) / 1.273 to 1.37 (seepage stage-1), closing
          most of the residual to the published 1.443 (see VALIDATION.md V-038).

    Returns
    -------
    RapidDrawdownResult
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
    if stage3_effective_normal not in ("fellenius", "gle"):
        raise ValueError("stage3_effective_normal must be 'fellenius' or 'gle', "
                         f"got '{stage3_effective_normal}'")
    if drawdown_to_elevation >= drawdown_from_elevation:
        raise ValueError("drawdown_to_elevation must be below drawdown_from_elevation")

    if slip_surface is not None:
        slip = slip_surface
    else:
        if xc is None or yc is None or radius is None:
            raise ValueError("provide slip_surface or all of xc, yc, radius")
        slip = CircularSlipSurface(xc, yc, radius)

    # Reinforcement crossings on this surface, resolved as analyze_slope does
    # and applied at every stage (GM-53). None without reinforcement, which
    # leaves every stage solve unchanged.
    from slope_stability.search import trial_reinforcement
    reinf = trial_reinforcement(geom, slip)

    # ── Stage 1: full-pool effective-stress analysis -> consolidation stresses
    if stage1_phreatic_points is not None:
        g_full = copy.deepcopy(geom)
        g_full.gwt_points = [tuple(p) for p in stage1_phreatic_points]
    else:
        g_full = _pool_geometry(geom, drawdown_from_elevation)
    g_draw = _pool_geometry(geom, drawdown_to_elevation)
    # One slicing for both stages (GM-134): each stage's water-table
    # crossings are slice boundaries in the other too, so the stage-2
    # strengths map slice for slice onto the drawn-down slices.
    x_ends = slip.find_entry_exit(geom)
    breaks_full = slice_breaks(g_full, slip, *x_ends)
    breaks_draw = slice_breaks(g_draw, slip, *x_ends)
    warnings: List[str] = []
    sl1 = build_slices(g_full, slip, n_slices, extra_boundaries=breaks_draw)
    res1 = gle_fos(sl1, slip, f_interslice=f_interslice,
                   tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf)
    stage1_converged = bool(res1.converged)
    if not res1.converged:
        warnings.append(
            "Stage 1 (full-pool consolidation) GLE did not converge; "
            "consolidation stresses fell back to a Fellenius effective-normal "
            "estimate.")
        # fall back to a Fellenius effective-normal estimate if GLE stalls
        sigma_fc = [_fellenius_effective_normal(s) for s in sl1]
        tau_fc = [abs(s.weight * math.sin(s.alpha)) / s.base_length for s in sl1]
        stage1_fos = res1.fos
    else:
        sigma_fc = [max(res1.base_normal_eff[i] / sl1[i].base_length, 0.0)
                    for i in range(len(sl1))]
        tau_fc = [abs(res1.shear_mobilized[i]) / sl1[i].base_length
                  for i in range(len(sl1))]
        stage1_fos = res1.fos

    # ── Stage 2: undrained strength per slice from the consolidation stresses
    tau_ff: List[Optional[float]] = []
    Kc_list: List[float] = []
    n_und = 0
    for i, s in enumerate(sl1):
        lay = _base_layer(g_full, s)
        if lay.R_phi is None:
            tau_ff.append(None)          # free-draining: keep drained strength
            Kc_list.append(1.0)
            continue
        n_und += 1
        sfc = sigma_fc[i]
        tanphi = math.tan(math.radians(lay.phi))
        tanphiR = math.tan(math.radians(lay.R_phi))
        s_R = lay.R_c + sfc * tanphiR                    # Kc=1 (R / IC-U) envelope
        s_D = lay.c_prime + sfc * tanphi                 # Kc=Kf (drained) envelope
        if method not in _KC_INTERP_METHODS:      # corps_2stage
            s_und = min(s_R, s_D)
            Kc_list.append(1.0)
        else:  # duncan_3stage / lowe_karafiath (Kc-interpolated stage-2)
            # Interpolate between the Kc=1 (R / IC-U) envelope and the Kc=Kf
            # (drained) envelope. The R envelope is the lower bound as-plotted;
            # over-strength from its cohesion intercept at low sigma' is caught
            # by the STAGE-3 drained substitution (not by capping here — capping
            # here would erase the interpolation range for a c'=0 soil). Lowe &
            # Karafiath uses this SAME stage-2 strength but omits stage 3.
            Kc = _consolidation_Kc(sfc, tau_fc[i], s.alpha, lay.phi,
                                   lay.c_prime)
            # Kf per EM App. G: Eq. G-7 closed form for c'=0; stress-dependent
            # Eq. G-8 for c'>0 (see _Kf; larger Kf -> conservative w).
            Kf = _Kf(lay.phi, lay.c_prime, sfc)
            w = (Kc - 1.0) / (Kf - 1.0) if Kf > 1.0 else 0.0
            w = max(0.0, min(w, 1.0))
            s_und = s_R + w * (s_D - s_R)
            Kc_list.append(Kc)
        tau_ff.append(max(s_und, 0.0))

    # ── Stage 3: drawn-down analysis with the undrained strengths (+ drained
    #    substitution for the 3-stage method).
    sl3 = build_slices(g_draw, slip, n_slices, extra_boundaries=breaks_full)

    # Optional rigorous drawn-down effective normal for the stage-3 drained
    # substitution (see stage3_effective_normal). Solved with the DRAINED
    # strengths still on the slices (before the undrained override below), so it
    # is the same base_normal_eff basis stage 1 uses for sigma'_fc.
    sig_post_gle: Optional[List[float]] = None
    if method in _STAGE3_METHODS and stage3_effective_normal == "gle":
        try:
            res3d = gle_fos(sl3, slip, f_interslice=f_interslice,
                            tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf)
            if res3d.converged:
                sig_post_gle = [
                    max(res3d.base_normal_eff[j] / sl3[j].base_length, 0.0)
                    for j in range(len(sl3))]
            else:
                warnings.append(
                    "stage3_effective_normal='gle': drawn-down drained GLE did "
                    "not converge; fell back to the Fellenius effective-normal "
                    "estimate for the stage-3 drained substitution.")
        except (ValueError, ZeroDivisionError, OverflowError):
            sig_post_gle = None

    # Both stages are cut at the same boundaries (GM-134), so this maps slice
    # for slice; the nearest x_mid stays as the guard should a stage drop a
    # sliver the other keeps. sl1 is
    # built left-to-right so its x_mid are ascending -> O(log n) bisect lookup
    # (bit-identical to the previous O(n) min-scan, tie rule included).
    sl1_xmid = [s1.x_mid for s1 in sl1]
    n_sub = 0
    for j, s in enumerate(sl3):
        i = _nearest_index(sl1_xmid, s.x_mid)
        if tau_ff[i] is None:
            continue                     # free-draining: keep drained c'/phi'
        s_und = tau_ff[i]
        if method in _STAGE3_METHODS:
            # post-drawdown drained strength (low-pool effective normal stress)
            lay = _base_layer(g_draw, s)
            if sig_post_gle is not None:
                sig_post = sig_post_gle[j]
            else:
                sig_post = _fellenius_effective_normal(s)
            s_drained_post = lay.c_prime + sig_post * math.tan(math.radians(lay.phi))
            if s_drained_post < s_und:
                s_und = s_drained_post
                n_sub += 1
        s.c = s_und
        s.phi = 0.0
        s.pore_pressure = 0.0            # total-stress (phi=0): u does not act

    res3 = gle_fos(sl3, slip, f_interslice=f_interslice,
                   tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf)
    fos = res3.fos
    stage3_converged = bool(res3.converged)
    if not res3.converged:
        warnings.append(
            "Stage 3 (drawn-down) GLE did not converge; the reported drawdown "
            "FOS is the last iterate and may be approximate.")
    # m_alpha on the drawn-down slices at the drawdown FOS, each slice with
    # the strength it was solved with (GM-127). A search leaves a surface
    # below 0.2 out; on this given surface it is said.
    m_min = _m_alpha_min(sl3, fos, slip, "gle", m_alpha_rule_in_force())
    if m_min < M_ALPHA_MIN:
        from slope_stability.analysis import m_alpha_statement
        w = m_alpha_statement(sl3, slip, "spencer", fos,
                              given="rapid_drawdown_fos")
        if w:
            warnings.append("Drawn-down stage: " + w)

    return RapidDrawdownResult(
        FOS=fos, method=method,
        drawdown_from=drawdown_from_elevation,
        drawdown_to=drawdown_to_elevation,
        n_slices=len(sl3), n_undrained_slices=n_und,
        n_drained_substituted=n_sub, f_interslice=f_interslice,
        sigma_fc=sigma_fc, tau_fc=tau_fc, Kc=Kc_list,
        tau_ff=[t for t in tau_ff if t is not None],
        stage1_fos=stage1_fos,
        stage1_converged=stage1_converged,
        stage3_converged=stage3_converged,
        warnings=warnings,
        m_alpha_min=m_min,
    )


@_counts_refusals
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
                          stage3_effective_normal: str = "fellenius",
                          toe_circles: bool = True,
                          infinite_slope: bool = True,
                          refine: bool = True,
                          time_budget_s: Optional[float] = None,
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

    Both stage methods are exposed via ``method`` ('corps_2stage' /
    'duncan_3stage'), and both circular and noncircular searches via
    ``surface_type``. The returned object carries the ordinary rich
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
        Steady-seepage stage-1 phreatic surface (default None = flat full pool).
    stage3_effective_normal : str
        Stage-3 drained-substitution normal basis, 'fellenius' (default) or
        'gle'. See ``rapid_drawdown_fos``.
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
    Each trial surface costs several LE solves (stage-1 + stage-3, plus the
    stage-3 drained solve when ``stage3_effective_normal='gle'``), so a circular
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

    from slope_stability.search import (
        grid_search, search_entry_exit, search_noncircular, search_de, _FOS_MAX,
        search_budget,
    )

    def _dd_fos(g: SlopeGeometry, slip) -> float:
        """Per-surface rapid-drawdown FOS for the search loop."""
        try:
            r = rapid_drawdown_fos(
                g, drawdown_from_elevation, drawdown_to_elevation,
                slip_surface=slip, method=method, f_interslice=f_interslice,
                n_slices=n_slices, tol=tol,
                stage1_phreatic_points=stage1_phreatic_points,
                stage3_effective_normal=stage3_effective_normal)
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeError):
            return _FOS_MAX
        # Reject degenerate surfaces so a spurious low FOS cannot win the search
        # (mirrors the noncircular rigour gate). Three ways a drawdown solve can
        # go bad on a trial surface a search will happily generate:
        #   (1) either stage GLE fails to converge (stage-1 then falls back to a
        #       Fellenius estimate that can be absurd);
        #   (2) a large near-flat circle self-clips to a handful of slices whose
        #       GLE converges to a physically meaningless value (a full-dam-
        #       spanning arc with 6 slices gave stage-1 FOS 0.18 -> drawdown 0.18);
        #   (3) an implausibly low stage-1 (full-pool, drained) FOS — a real
        #       reservoir slope is stable at full pool, so stage-1 << 1 signals a
        #       degenerate consolidation solve, not a real drawdown mechanism.
        # These guards are scoped to the drawdown search wrapper; single-surface
        # rapid_drawdown_fos is unchanged.
        if not (r.stage1_converged and r.stage3_converged):
            return _FOS_MAX
        if r.n_slices < max(8, n_slices // 2):
            return _FOS_MAX
        if r.stage1_fos < 0.5:
            return _FOS_MAX
        # m_alpha < 0.2 on a drawn-down slice at the drawdown FOS: left out
        # and counted, as every search does (GM-127; Slide2's -112)
        if r.m_alpha_min < M_ALPHA_MIN:
            _count_m_alpha_refusal()
            return _FOS_MAX
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
            crit_slip = CircularSlipSurface(crit.xc, crit.yc, crit.radius)
        else:
            crit_slip = PolylineSlipSurface(points=list(crit.slip_points))
        dd = rapid_drawdown_fos(
            geom, drawdown_from_elevation, drawdown_to_elevation,
            slip_surface=crit_slip, method=method, f_interslice=f_interslice,
            n_slices=n_slices, tol=tol,
            stage1_phreatic_points=stage1_phreatic_points,
            stage3_effective_normal=stage3_effective_normal)

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
    search.choices = search_choices(
        search, surface_type=surface_type, toe_circles=toe_circles,
        infinite_slope=infinite_slope,
        surface_types=("circular", "entry_exit", "noncircular",
                       "noncircular_de"), budget=budget)
    if window_note:
        search.warnings.insert(0, window_note)
    note_budget(search, budget)     # a search cut short says so first
    return search
