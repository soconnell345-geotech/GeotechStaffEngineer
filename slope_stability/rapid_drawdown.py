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
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.gle import gle_fos


_METHODS = ("corps_2stage", "duncan_3stage")


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
                       stage1_phreatic_points: Optional[List] = None
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

    Returns
    -------
    RapidDrawdownResult
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
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
    # slice count can differ if the pool changes which slivers survive; align by
    # nearest x_mid so the stage-2 strengths map onto the stage-3 slices.
    n_sub = 0
    for s in sl3:
        i = min(range(len(sl1)), key=lambda k: abs(sl1[k].x_mid - s.x_mid))
        if tau_ff[i] is None:
            continue                     # free-draining: keep drained c'/phi'
        s_und = tau_ff[i]
        if method == "duncan_3stage":
            # post-drawdown drained strength (low-pool effective normal stress)
            lay = _base_layer(g_draw, s)
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
