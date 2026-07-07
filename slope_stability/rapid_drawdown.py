"""
Rapid-drawdown slope stability — USACE 2-stage and Duncan-Wright-Wong 3-stage.

When a reservoir is drawn down faster than the low-permeability zones of an
embankment can drain, those zones respond UNDRAINED: the pore pressures set up
by the pre-drawdown steady seepage persist while the stabilizing water load on
the upstream face is removed. The available strength is the undrained strength
mobilized from the pre-drawdown CONSOLIDATION stresses, not the (higher) drained
strength that the post-drawdown effective stresses would imply.

Two established procedures are implemented, both wired to the rigorous GLE /
Spencer engine by overriding each slice's base strength:

USACE / Army Corps 2-stage (EM 1110-2-1902, 1970; Lowe & Karafiath):
  Stage 1  Effective-stress ("S") analysis with the FULL pool -> effective
           normal stress sigma'_fc on each slice base (the consolidation stress).
  Stage 2  Undrained strength of each low-permeability slice =
           min( R-envelope(sigma'_fc), drained-envelope(sigma'_fc) )
           (the "combined" envelope — the R strength capped by the effective
           strength so negative-pore-pressure over-strength is not used).
           The factor of safety is then computed with the DRAWN-DOWN pool.

Duncan, Wright & Wong 3-stage (1990; Duncan, Wright & Brandon 2014, Ch. 9):
  Stage 1  as above, plus the shear stress tau_fc on each base.
  Stage 2  Undrained strength interpolated with the consolidation stress ratio
           Kc = sigma'_1c/sigma'_3c between two envelopes plotted as tau_ff vs
           sigma'_fc:  the Kc=1 (isotropically-consolidated / R) envelope (lower
           bound) and the Kc=Kf (effective-stress / drained) envelope (upper
           bound).  tau_ff = tau_R + (Kc-1)/(Kf-1) * (tau_drained - tau_R).
  Stage 3  For each slice the post-drawdown DRAINED strength is computed with
           the drawn-down pore pressures; where it is LESS than the stage-2
           undrained strength it is substituted, and the FOS re-computed.

Free-draining layers (``R_phi is None``) keep their effective strength in every
stage. All units SI (m, kPa, kN/m3, degrees).

References
----------
US Army Corps of Engineers (1970). EM 1110-2-1902, Stability of Earth and
    Rock-Fill Dams, Appendix G.
Duncan, J.M., Wright, S.G., Wong, K.S. (1990). "Slope stability during rapid
    drawdown." H. Bolton Seed Memorial Symposium, Vol. 2, 253-272.
Duncan, Wright & Brandon (2014). Soil Strength and Slope Stability, Ch. 9.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.gle import gle_fos
from slope_stability.results import SearchResult


_METHODS = ("corps_2stage", "duncan_3stage")
_SURFACE_TYPES = ("circular", "entry_exit", "noncircular", "noncircular_de")


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

    def summary(self) -> str:
        crit = self.critical
        lines = [
            "=" * 60,
            f"  RAPID-DRAWDOWN CRITICAL-SURFACE SEARCH — {self.method}",
            "=" * 60,
            f"  Drawdown:            el {self.drawdown_from:g} -> {self.drawdown_to:g}",
            f"  Surface type:        {self.surface_type}",
            f"  Surfaces evaluated:  {self.n_surfaces_evaluated}",
            f"  MINIMUM FOS:         {self.FOS:.3f}",
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
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "FOS": round(self.FOS, 4),
            "method": self.method,
            "surface_type": self.surface_type,
            "drawdown_from": self.drawdown_from,
            "drawdown_to": self.drawdown_to,
            "n_surfaces_evaluated": self.n_surfaces_evaluated,
        }
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


def _consolidation_Kc(sigma_fc: float, tau_fc: float, alpha: float,
                      phi_deg: float) -> float:
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
    Kf = (1.0 + sinphi) / (1.0 - sinphi) if sinphi < 1 else 1e6
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
        'duncan_3stage' (default) or 'corps_2stage'.
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
          ``sigma'_post = W*cos(alpha)/l - u`` on the drawn-down geometry. This is
          the historical behaviour and is preserved byte-for-byte as the default.
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

    # ── Stage 1: full-pool effective-stress analysis -> consolidation stresses
    if stage1_phreatic_points is not None:
        g_full = copy.deepcopy(geom)
        g_full.gwt_points = [tuple(p) for p in stage1_phreatic_points]
    else:
        g_full = _pool_geometry(geom, drawdown_from_elevation)
    warnings: List[str] = []
    sl1 = build_slices(g_full, slip, n_slices)
    res1 = gle_fos(sl1, slip, f_interslice=f_interslice,
                   tol=min(tol, 1e-4) * 0.1)
    stage1_converged = bool(res1.converged)
    if not res1.converged:
        warnings.append(
            "Stage 1 (full-pool consolidation) GLE did not converge; "
            "consolidation stresses fell back to a Fellenius effective-normal "
            "estimate.")
        # fall back to a Fellenius effective-normal estimate if GLE stalls
        sigma_fc = [max(s.weight * math.cos(s.alpha) / s.base_length
                        - s.pore_pressure, 0.0) for s in sl1]
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
        if method == "corps_2stage":
            s_und = min(s_R, s_D)
            Kc_list.append(1.0)
        else:  # duncan_3stage
            # Interpolate between the Kc=1 (R / IC-U) envelope and the Kc=Kf
            # (drained) envelope. The R envelope is the lower bound as-plotted;
            # over-strength from its cohesion intercept at low sigma' is caught
            # by the STAGE-3 drained substitution (not by capping here — capping
            # here would erase the interpolation range for a c'=0 soil).
            Kc = _consolidation_Kc(sfc, tau_fc[i], s.alpha, lay.phi)
            sinphi = math.sin(math.radians(lay.phi))
            Kf = (1.0 + sinphi) / (1.0 - sinphi) if sinphi < 1 else 1e6
            w = (Kc - 1.0) / (Kf - 1.0) if Kf > 1.0 else 0.0
            w = max(0.0, min(w, 1.0))
            s_und = s_R + w * (s_D - s_R)
            Kc_list.append(Kc)
        tau_ff.append(max(s_und, 0.0))

    # ── Stage 3: drawn-down analysis with the undrained strengths (+ drained
    #    substitution for the 3-stage method).
    g_draw = _pool_geometry(geom, drawdown_to_elevation)
    sl3 = build_slices(g_draw, slip, n_slices)

    # Optional rigorous drawn-down effective normal for the stage-3 drained
    # substitution (see stage3_effective_normal). Solved with the DRAINED
    # strengths still on the slices (before the undrained override below), so it
    # is the same base_normal_eff basis stage 1 uses for sigma'_fc.
    sig_post_gle: Optional[List[float]] = None
    if method == "duncan_3stage" and stage3_effective_normal == "gle":
        try:
            res3d = gle_fos(sl3, slip, f_interslice=f_interslice,
                            tol=min(tol, 1e-4) * 0.1)
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

    # slice count can differ if the pool changes which slivers survive; align by
    # nearest x_mid so the stage-2 strengths map onto the stage-3 slices.
    n_sub = 0
    for j, s in enumerate(sl3):
        i = min(range(len(sl1)), key=lambda k: abs(sl1[k].x_mid - s.x_mid))
        if tau_ff[i] is None:
            continue                     # free-draining: keep drained c'/phi'
        s_und = tau_ff[i]
        if method == "duncan_3stage":
            # post-drawdown drained strength (low-pool effective normal stress)
            lay = _base_layer(g_draw, s)
            if sig_post_gle is not None:
                sig_post = sig_post_gle[j]
            else:
                sig_post = max(s.weight * math.cos(s.alpha) / s.base_length
                               - s.pore_pressure, 0.0)
            s_drained_post = lay.c_prime + sig_post * math.tan(math.radians(lay.phi))
            if s_drained_post < s_und:
                s_und = s_drained_post
                n_sub += 1
        s.c = s_und
        s.phi = 0.0
        s.pore_pressure = 0.0            # total-stress (phi=0): u does not act

    res3 = gle_fos(sl3, slip, f_interslice=f_interslice,
                   tol=min(tol, 1e-4) * 0.1)
    fos = res3.fos
    stage3_converged = bool(res3.converged)
    if not res3.converged:
        warnings.append(
            "Stage 3 (drawn-down) GLE did not converge; the reported drawdown "
            "FOS is the last iterate and may be approximate.")

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
    )


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
                          stage3_effective_normal: str = "fellenius"
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
        Drawdown stage method: 'corps_2stage' (default) or 'duncan_3stage'.
    surface_type : str
        'circular' (default, centre-grid + radius optimisation), 'entry_exit'
        (circular arcs between entry/exit windows), 'noncircular' (random
        polylines), or 'noncircular_de' (differential-evolution refinement).
    x_range, y_range : (float, float), optional
        Circle-centre search window (circular only; auto from geometry if None).
    nx, ny : int
        Grid resolution (circular) or entry/exit sampling (entry_exit). Default 10.
    x_entry_range, x_exit_range : (float, float), optional
        Entry/exit x windows (auto from geometry if None).
    n_trials, n_points, seed : int / int / int
        Random / DE noncircular controls (see ``search_noncircular``/``search_de``).
    n_slices, tol, f_interslice :
        Passed through to each per-surface ``rapid_drawdown_fos`` solve.
    stage1_phreatic_points : list of (x, z), optional
        Steady-seepage stage-1 phreatic surface (default None = flat full pool).
    stage3_effective_normal : str
        Stage-3 drained-substitution normal basis, 'fellenius' (default) or
        'gle'. See ``rapid_drawdown_fos``.

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
        return r.FOS

    # Auto entry/exit windows from the geometry (mirrors search_critical_surface).
    x_min_geo = geom.surface_points[0][0]
    x_max_geo = geom.surface_points[-1][0]
    if x_entry_range is None:
        x_entry_range = (x_min_geo, x_min_geo + (x_max_geo - x_min_geo) * 0.4)
    if x_exit_range is None:
        x_exit_range = (x_min_geo + (x_max_geo - x_min_geo) * 0.6, x_max_geo)

    if surface_type == "circular":
        if x_range is None:
            x_range = (x_min_geo, x_max_geo)
        if y_range is None:
            z_min = min(z for _, z in geom.surface_points)
            z_max = max(z for _, z in geom.surface_points)
            slope_h = z_max - z_min
            y_range = (z_max + 1.0, z_max + 2.0 * slope_h)
        search = grid_search(
            geom, x_range, y_range, nx, ny, method="spencer",
            n_slices=n_slices, tol=tol, x_entry_range=x_entry_range,
            x_exit_range=x_exit_range, fos_fn=_dd_fos)
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
