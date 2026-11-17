"""
Newmark seismic sliding-block analysis for slopes.

Three pieces, all operating on a SPECIFIED slip surface:

1. ``yield_acceleration`` — the yield (critical) seismic coefficient ky: the
   horizontal pseudo-static coefficient kh that brings the factor of safety of a
   given surface to exactly 1.0. Found by bisection on the module's pseudo-static
   FOS (the same seismic_force = kh*W path validated by V-033 / Loukidis #62-#63).
   The yield acceleration is ay = ky * g.

2. ``newmark_displacement`` — the permanent downslope displacement of a rigid
   block on the surface, by double integration of an earthquake acceleration
   time history against ky (Newmark 1965). The block slides only while the
   relative velocity is positive; it never slides back upslope, so displacement
   accumulates monotonically. Two polarity conventions are offered
   (``polarity=``):
     * ``"downslope"`` (default, standard Newmark 1965 / Jibson 2007) — the
       SIGNED record is used and the block is driven only when the ground
       acceleration exceeds ay in the destabilizing (downslope-positive)
       direction; the reverse polarity decelerates the block. Only one polarity
       of a symmetric record contributes, so it gives about HALF the rectified
       displacement.
     * ``"rectified"`` — the absolute acceleration is used, so both polarities of
       the record drive the single downslope block. A conservative,
       orientation-independent option (see DESIGN.md); use when the sign of the
       record relative to the slope's downslope direction is unknown.

3. ``newmark_jibson2007`` — Jibson's (2007) regression estimate of Newmark
   displacement from the critical-acceleration ratio ky/amax, an independent
   cross-check that needs no time history.

4. ``bray_travasarou_2007`` — Bray & Travasarou's (2007) simplified semi-empirical
   estimate of earthquake-induced DEVIATORIC slope displacement from a nonlinear
   fully-coupled stick-slip sliding-block model. A period-dependent cross-check
   that captures the sliding-mass flexibility (via the initial fundamental period
   Ts) and the spectral demand at the degraded period, which the rigid-block
   Jibson/integration models do not. Two parts: the probability of negligible
   (<= ~1 cm) displacement P(D=0), and the median non-zero displacement.

References
----------
Newmark, N.M. (1965). "Effects of earthquakes on dams and embankments."
    Geotechnique 15(2), 139-160.
Jibson, R.W. (2007). "Regression models for estimating coseismic landslide
    displacement." Engineering Geology 91(2-4), 209-218. (Eq. 6.)
Bray, J.D. & Travasarou, T. (2007). "Simplified procedure for estimating
    earthquake-induced deviatoric slope displacements." J. Geotech. Geoenviron.
    Eng. 133(4), 381-392. (Also Bray 2007, Ch. 14, Eqs. 14.3-14.5.)
Duncan, Wright & Brandon (2014). Soil Strength and Slope Stability, Ch. 10.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface

_G = 9.80665   # standard gravity, m/s^2


@dataclass
class YieldAccelerationResult:
    """Yield (critical) seismic coefficient for a specified surface.

    ``ky`` and ``ay`` are None when the method has no solution on the
    surface at a trial kh the bisection needs (GM-140 step 2): there is no
    number. ``reason`` then says why and at which kh, ``no_solution_kh`` is
    that kh, and ``ky_bounds`` the last (lower, upper) kh at which the
    method did solve with F above and below 1 (None where there is none;
    an upper bound of None: the method never solved with F <= 1).
    ``fos_static`` is None when the method has no solution at kh = 0.
    """
    ky: Optional[float] = 0.0      # yield seismic coefficient (fraction of g)
    ay: Optional[float] = 0.0      # yield acceleration = ky*g (m/s^2)
    fos_static: Optional[float] = 0.0   # pseudo-static FOS at kh = 0
    method: str = "spencer"
    converged: bool = False
    n_iter: int = 0
    reason: str = ""
    no_solution_kh: Optional[float] = None
    ky_bounds: Optional[tuple] = None
    #: Which arc of its circle the surface is, where the circle meets the
    #: ground at more than two points (GM-343; `analysis.lowest_arc`).
    arc: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ky": None if self.ky is None else round(self.ky, 5),
            "ay_m_s2": None if self.ay is None else round(self.ay, 5),
            "FOS_static": (None if self.fos_static is None
                           else round(self.fos_static, 4)),
            "method": self.method,
            "converged": self.converged,
        }
        if self.arc:
            d["arc"] = dict(self.arc)
        if self.reason:
            d["reason"] = self.reason
            d["no_solution_kh"] = (None if self.no_solution_kh is None
                                   else round(self.no_solution_kh, 5))
            if self.ky_bounds is not None:
                d["ky_bounds"] = [None if b is None else round(b, 5)
                                  for b in self.ky_bounds]
        return d


@dataclass
class NewmarkResult:
    """Newmark permanent-displacement result."""
    displacement: float = 0.0      # permanent downslope displacement (m)
    ky: float = 0.0
    method: str = "integration"    # 'integration' or 'jibson2007'
    amax: float = 0.0              # peak ground acceleration used (fraction of g)
    n_exceedances: int = 0         # times the record exceeds ay (integration)
    duration: float = 0.0          # s (integration)
    polarity: str = "downslope"    # 'downslope' or 'rectified' (integration)
    sigma_log10: Optional[float] = None   # Jibson dispersion (log10 cm)

    @property
    def displacement_cm(self) -> float:
        return self.displacement * 100.0

    def regression_range_cm(self) -> Optional[tuple]:
        """For a regression estimate: the displacement one standard
        deviation either side of the median (``sigma_log10``), in cm."""
        if self.sigma_log10 is None or self.displacement <= 0:
            return None
        f = 10.0 ** self.sigma_log10
        return (self.displacement_cm / f, self.displacement_cm * f)

    def summary(self) -> str:
        line = (f"Newmark displacement ({self.method}): "
                f"{self.displacement_cm:.3f} cm  (ky={self.ky:g}"
                + (f", amax={self.amax:g}g" if self.amax else "") + ")")
        rng = self.regression_range_cm()
        if rng:
            line += (f"\n  A regression median: one standard deviation "
                     f"({self.sigma_log10:.3f} log10 units, Jibson 2007) "
                     f"either side is {rng[0]:.3f} to {rng[1]:.3f} cm; a "
                     f"screening estimate, not a site-specific analysis.")
        return line

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "displacement_m": round(self.displacement, 5),
            "displacement_cm": round(self.displacement_cm, 3),
            "ky": round(self.ky, 5),
            "method": self.method,
        }
        if self.method == "integration":
            d["n_exceedances"] = self.n_exceedances
            d["duration_s"] = round(self.duration, 3)
            d["polarity"] = self.polarity
        else:
            d["amax_g"] = round(self.amax, 5)
        rng = self.regression_range_cm()
        if rng:
            # GM-17: a regression's median alone reads as more certain than
            # it is (Jibson 2007 gives the dispersion with the equation)
            d["sigma_log10"] = self.sigma_log10
            d["displacement_cm_minus_1sigma"] = round(rng[0], 3)
            d["displacement_cm_plus_1sigma"] = round(rng[1], 3)
            d["note"] = ("Median of a regression (Jibson 2007, Eq. 6): one "
                         "standard deviation is a factor of "
                         f"{10.0 ** self.sigma_log10:.1f} either way. A "
                         "screening estimate; integrate a record "
                         "(newmark_displacement) for a site-specific value.")
        return d


class _NoSolutionAt(Exception):
    """The method has no solution at a trial kh of the bisection."""

    def __init__(self, kh: float, reason: str):
        super().__init__(reason)
        self.kh, self.reason = kh, reason


def _method_fos(slices, slip, method: str, tol: float, reinf):
    """The method's own solution on pre-built slices (a
    ``solution.MethodSolution``), from the solver ``analyze_slope`` uses
    (``solution.solve_method``; Morgenstern-Price / GLE with the half-sine
    function). Lets ``yield_acceleration`` reuse one discretization across
    the bisection instead of rebuilding the slices per FOS.

    Before GM-140 step 2 Spencer and Morgenstern-Price took the old
    approximate solver's value at any trial kh where the rigorous solution
    did not converge, so a yield coefficient named for Spencer could rest
    on another solver's factor of safety.
    """
    from slope_stability.solution import solve_method
    return solve_method(slices, slip, method, "half_sine", tol, reinf)


def yield_acceleration(geom: SlopeGeometry,
                       xc: float = None, yc: float = None, radius: float = None,
                       slip_surface=None,
                       method: str = "spencer",
                       n_slices: int = 50,
                       tol: float = 1e-4,
                       kh_max: float = 2.0,
                       bisection_tol: float = 1e-4) -> YieldAccelerationResult:
    """Yield seismic coefficient ky for a specified slip surface.

    ky is the horizontal pseudo-static coefficient at which the factor of safety
    of the surface equals 1.0 (Newmark's critical acceleration, ay = ky*g). The
    FOS is a monotonically decreasing function of kh, so a bisection on the
    module's seismic FOS gives ky directly.

    Parameters
    ----------
    geom, xc/yc/radius or slip_surface, method, n_slices, tol :
        As for ``analyze_slope`` — the surface is FIXED and only kh is varied.
        ``geom.kh`` is overridden internally (the caller's geometry is untouched).
    kh_max : float
        Upper bound on the kh bracket (default 2.0). If the surface is still
        stable at kh_max the result is returned non-converged with ky=kh_max.
        Where the method has no solution at kh_max, the upper end is halved
        until it has one.

    Returns
    -------
    YieldAccelerationResult
        ``ky`` None, with the ``reason``, where the method has no solution
        at a trial kh (GM-140 step 2): every trial kh is solved by the
        method itself. An unknown ``method`` is refused.
    """
    from slope_stability.solution import check_method
    check_method(method)
    if slip_surface is not None:
        slip = slip_surface
    else:
        if xc is None or yc is None or radius is None:
            raise ValueError("provide slip_surface or all of xc, yc, radius")
        slip = CircularSlipSurface(xc, yc, radius)

    # Where a circle ends (GM-343): a circle that meets the ground at more
    # than two points is split there; the yield coefficient is found on
    # each arc as a surface of its own and the result is the lowest,
    # named in ``arc``.
    from slope_stability.analysis import (split_arcs, lowest_arc,
                                          checked_surface)
    slip = checked_surface(geom, slip)
    arcs = split_arcs(geom, slip)
    if arcs:
        return lowest_arc(
            geom, slip, arcs,
            lambda arc: yield_acceleration(
                geom, slip_surface=arc, method=method, n_slices=n_slices,
                tol=tol, kh_max=kh_max, bisection_tol=bisection_tol),
            value=lambda r: r.ky, what="yield coefficient")

    # Build the FIXED discretization ONCE and reuse it across the bisection: the
    # only kh-dependent slice quantity is the seismic force (kh * soil weight),
    # so capture the exact per-slice soil weight (the seismic force at kh=1, since
    # 1.0*w == w exactly in IEEE-754) and overwrite just that per kh — instead of
    # rebuilding the slices for every FOS call. The reinforcement crossings and
    # entry/exit are kh-independent. FOS is bit-identical to analyze_slope().
    from slope_stability.slices import build_slices
    from slope_stability.reinforcement import compute_reinforcement_forces
    g_ref = copy.copy(geom)
    g_ref.kh = 1.0
    x_entry, x_exit = slip.find_entry_exit(g_ref)
    reinf = compute_reinforcement_forces(g_ref, slip, x_entry, x_exit) or None
    # The surface's crack as printed (GM-140 step 4, ``cracks``), one crack
    # through the bisection. Rankine's depth with the developed strengths
    # is taken at F = 1, the state the yield coefficient is solved for
    # (c_D = c, phi_D = phi); the depth of minimum factor of safety is
    # scanned on the surface without a seismic coefficient.
    from slope_stability import cracks
    if cracks.may_need_solution(g_ref, slip):
        def _static(g):
            sl = build_slices(g, slip, n_slices)
            for s in sl:
                s.seismic_force = 0.0
            return _method_fos(sl, slip, method, tol, reinf)
        _, crack = cracks.solve_with_crack(g_ref, slip, _static,
                                           fos_for_depth=1.0)
        if crack is not None and crack.discarded:
            err = cracks.refusal_error(crack)
            raise type(err)("No yield coefficient: " + crack.reason + ".",
                            crack)
        g_ref = cracks.geometry_of(g_ref, crack)
    ref_slices = build_slices(g_ref, slip, n_slices)
    soil_weights = [s.seismic_force for s in ref_slices]

    def _fos(kh: float) -> float:
        for s, sw in zip(ref_slices, soil_weights):
            s.seismic_force = kh * sw
        sol = _method_fos(ref_slices, slip, method, tol, reinf)
        if not sol.solved:
            raise _NoSolutionAt(kh, sol.reason)
        return sol.fos

    from slope_stability.solution import METHOD_TEXT
    fos_static = None
    lo = hi = None          # kh with a solution and F > 1 / F <= 1
    n = 0
    try:
        fos_static = _fos(0.0)
        if fos_static <= 1.0:
            # already at/over the limit with no seismic load -> ky = 0
            return YieldAccelerationResult(
                ky=0.0, ay=0.0, fos_static=fos_static, method=method,
                converged=True, n_iter=0)

        lo = 0.0
        # The upper end of the bracket. kh_max (2.0 by default) is a
        # numerical device, far above any yield coefficient, and a method
        # may have no solution there; it is then halved until the method
        # solves (down to the bisection tolerance), so the bracket's ends
        # are both the method's own solutions.
        top, first = kh_max, None
        while True:
            try:
                fos_hi = _fos(top)
                break
            except _NoSolutionAt as exc:
                first = first or exc
                top *= 0.5
                if top <= bisection_tol:
                    raise first
        if fos_hi > 1.0:
            if first is None:
                # surface never yields within the bracket
                return YieldAccelerationResult(
                    ky=kh_max, ay=kh_max * _G, fos_static=fos_static,
                    method=method, converged=False, n_iter=0)
            # still stable at the highest kh the method solves: the yield
            # point lies above it, where the method has no solution
            lo = top
            raise _NoSolutionAt(2.0 * top, first.reason)
        hi = top

        while hi - lo > bisection_tol and n < 100:
            mid = 0.5 * (lo + hi)
            f = _fos(mid)
            if f > 1.0:
                lo = mid
            else:
                hi = mid
            n += 1
    except _NoSolutionAt as exc:
        # A method's own solution at every trial kh, or no number (GM-140
        # step 2; UFC 3-220-10 p. 362): the bisection cannot tell which
        # side of F = 1 this kh lies on, and no other solver is asked.
        bounds = (lo, hi) if lo is not None else None
        if bounds and hi is not None:
            where = (f" The method solved with F > 1 at kh = {lo:.4f} and "
                     f"F <= 1 at kh = {hi:.4f}, so ky lies between them.")
        elif bounds and lo > 0:
            where = (f" The method solved with F > 1 at kh = {lo:.4f}, so "
                     f"ky is above it.")
        else:
            where = ""
        return YieldAccelerationResult(
            ky=None, ay=None, fos_static=fos_static, method=method,
            converged=False, n_iter=n, no_solution_kh=exc.kh,
            ky_bounds=bounds,
            reason=(f"{METHOD_TEXT.get(method, method)} has no solution on "
                    f"this surface at kh = {exc.kh:.4f}: {exc.reason}. No "
                    f"yield coefficient is reported: every trial kh is "
                    f"solved by the method itself.{where}"))
    ky = 0.5 * (lo + hi)
    return YieldAccelerationResult(
        ky=ky, ay=ky * _G, fos_static=fos_static, method=method,
        converged=True, n_iter=n)


def newmark_displacement(ky: float,
                         accel: Sequence[float],
                         dt: float,
                         accel_in_g: bool = False,
                         polarity: str = "downslope",
                         g: float = _G) -> NewmarkResult:
    """Permanent downslope displacement by Newmark double integration.

    Parameters
    ----------
    ky : float
        Yield seismic coefficient (fraction of g); yield acceleration ay = ky*g.
    accel : sequence of float
        Ground acceleration time history, equally spaced at ``dt``. In m/s^2 by
        default, or in g if ``accel_in_g=True``. The record's sign convention is
        downslope-positive (a positive value drives the block downslope) when
        ``polarity="downslope"``.
    dt : float
        Time step (s).
    accel_in_g : bool
        If True, ``accel`` is in units of g (multiplied by g internally).
    polarity : {"downslope", "rectified"}
        Which polarity of the record drives the block:

        * ``"downslope"`` (default) — the STANDARD Newmark (1965) / Jibson
          (2007) single-block treatment: the SIGNED record is integrated and the
          block accelerates only when the ground acceleration exceeds ay in the
          destabilizing (downslope-positive) direction; the opposite polarity
          decelerates it. Only the destabilizing polarity of a symmetric record
          contributes.
        * ``"rectified"`` — the ABSOLUTE record is integrated, so both polarities
          drive the single downslope block. A conservative, orientation-
          independent option (about twice the downslope displacement for a
          symmetric record); use when the record's sign relative to the slope's
          downslope direction is unknown. See DESIGN.md.
    g : float
        Gravity (m/s^2). Default 9.80665.

    Returns
    -------
    NewmarkResult
        ``displacement`` in metres; ``displacement_cm`` property in cm.

    Notes
    -----
    Rigid-block integration with no upslope rebound: the relative velocity is
    clamped at zero, so displacement is monotonic. Trapezoidal integration of the
    piecewise-linear relative velocity is EXACT for a rectangular pulse
    (D = ap*(ap-ay)*T^2/(2*ay)), which is the integrator's closed-form check.

    References: Newmark (1965); Jibson (2007); Duncan-Wright-Brandon (2014),
    Ch. 10.
    """
    if ky < 0:
        raise ValueError("ky must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if polarity not in ("downslope", "rectified"):
        raise ValueError("polarity must be 'downslope' or 'rectified'")
    ay = ky * g
    scale = g if accel_in_g else 1.0

    v = 0.0          # relative velocity (m/s)
    d = 0.0          # cumulative relative displacement (m)
    n_exc = 0
    amax = 0.0
    for a_raw in accel:
        a = a_raw * scale
        amax = max(amax, abs(a))
        # destabilizing (downslope) driving acceleration
        drive = abs(a) if polarity == "rectified" else a
        if drive > ay or v > 0.0:
            if drive > ay:
                n_exc += 1
            v_prev = v
            v = v_prev + (drive - ay) * dt
            if v < 0.0:
                v = 0.0
            d += 0.5 * (v_prev + v) * dt      # trapezoidal
    return NewmarkResult(
        displacement=d, ky=ky, method="integration",
        amax=amax / g, n_exceedances=n_exc, duration=len(accel) * dt,
        polarity=polarity)


def newmark_jibson2007(ky: float, amax: float) -> NewmarkResult:
    """Jibson (2007) regression estimate of Newmark displacement.

    Eq. 6 of Jibson (2007), the critical-acceleration-ratio model:

        log10 D = 0.215 + log10[ (1 - ky/amax)^2.341 * (ky/amax)^-1.438 ]

    with D in centimetres and a dispersion of +/- 0.510 log10 units. Both ``ky``
    (critical acceleration) and ``amax`` (peak ground acceleration) are in the
    SAME units (fractions of g), so only their ratio enters. Requires
    0 < ky < amax (no displacement is predicted once ky >= amax).

    Returns
    -------
    NewmarkResult
        ``displacement`` in metres (``displacement_cm`` in cm), method
        'jibson2007', with ``sigma_log10`` = 0.510.
    """
    if amax <= 0:
        raise ValueError("amax must be positive")
    if ky <= 0:
        raise ValueError("ky must be positive for the Jibson ratio model")
    ratio = ky / amax
    if ratio >= 1.0:
        return NewmarkResult(displacement=0.0, ky=ky, method="jibson2007",
                             amax=amax, sigma_log10=0.510)
    log_d = 0.215 + 2.341 * math.log10(1.0 - ratio) - 1.438 * math.log10(ratio)
    d_cm = 10.0 ** log_d
    return NewmarkResult(displacement=d_cm / 100.0, ky=ky, method="jibson2007",
                         amax=amax, sigma_log10=0.510)


def _std_normal_cdf(x: float) -> float:
    """Standard-normal CDF (self-contained, via erf)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class BrayTravasarou2007Result:
    """Bray & Travasarou (2007) seismic slope-displacement estimate."""
    displacement: float = 0.0      # median deviatoric displacement (m)
    p_zero: float = 0.0            # P(D=0): prob. of negligible (<= ~1 cm) displ.
    ky: float = 0.0               # yield coefficient
    ts: float = 0.0               # initial fundamental period of the mass (s)
    sa_1p5ts: float = 0.0         # 5%-damped spectral accel at 1.5*Ts (g)
    magnitude: float = 0.0        # moment magnitude Mw
    ln_d_cm: float = 0.0          # ln of the median displacement in cm
    sigma_ln: float = 0.66        # dispersion of ln(D) (Bray & Travasarou 2007)
    rigid: bool = False           # Ts < 0.05 s branch used
    method: str = "bray_travasarou_2007"

    @property
    def displacement_cm(self) -> float:
        return self.displacement * 100.0

    @property
    def d16_cm(self) -> float:
        """16th-percentile displacement (median / e^sigma), cm."""
        return self.displacement_cm * math.exp(-self.sigma_ln)

    @property
    def d84_cm(self) -> float:
        """84th-percentile displacement (median * e^sigma), cm."""
        return self.displacement_cm * math.exp(self.sigma_ln)

    def summary(self) -> str:
        return (f"Bray & Travasarou (2007): median D = {self.displacement_cm:.2f} "
                f"cm (16-84%: {self.d16_cm:.2f}-{self.d84_cm:.2f}); "
                f"P(D=0) = {self.p_zero:.3f}  "
                f"(ky={self.ky:g}, Ts={self.ts:g}s, Sa(1.5Ts)={self.sa_1p5ts:g}g, "
                f"M={self.magnitude:g})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "displacement_m": round(self.displacement, 5),
            "displacement_cm": round(self.displacement_cm, 3),
            "displacement_16pct_cm": round(self.d16_cm, 3),
            "displacement_84pct_cm": round(self.d84_cm, 3),
            "p_zero": round(self.p_zero, 5),
            "ky": round(self.ky, 5),
            "ts_s": round(self.ts, 4),
            "sa_1p5ts_g": round(self.sa_1p5ts, 5),
            "magnitude": round(self.magnitude, 3),
            "sigma_ln": self.sigma_ln,
            "rigid": self.rigid,
        }


def bray_travasarou_2007(ky: float, ts: float, sa_1p5ts: float,
                         magnitude: float,
                         rigid: Optional[bool] = None) -> BrayTravasarou2007Result:
    """Bray & Travasarou (2007) simplified seismic slope-displacement estimate.

    A fully-coupled stick-slip sliding-block regression (688 records) that, unlike
    the rigid-block Jibson / integration models, captures the sliding-mass
    flexibility through its initial fundamental period ``ts`` and the spectral
    demand at the degraded period ``1.5*ts``. Two published equations
    (Bray 2007, Ch. 14):

    P(D=0) = 1 - Phi( -1.76 - 3.22 ln(ky) - 0.484 Ts ln(ky)
                      + 3.52 ln(Sa(1.5Ts)) )                          (Eq. 14.4)

    ln(D[cm]) = a0 - 2.83 ln(ky) - 0.333 (ln ky)^2
                + 0.566 ln(ky) ln(Sa) + 3.04 ln(Sa) - 0.244 (ln Sa)^2
                + [1.5 Ts]  + 0.278 (M - 7)                          (Eq. 14.5)

    with a0 = -1.10 for a flexible mass (Ts >= 0.05 s) and a0 = -0.22 with the
    ``1.5 Ts`` term dropped for a rigid mass (Ts < 0.05 s); dispersion of ln(D) is
    sigma = 0.66 (so D_16/D_84 = median * e^-/+0.66).

    Parameters
    ----------
    ky : float
        Yield (critical) seismic coefficient (fraction of g), > 0.
    ts : float
        Initial fundamental period of the potential sliding mass (s), >= 0.
        For a flexible mass Ts ~ 4H/Vs (deep, wide) or 2.6H/Vs (surface);
        Ts = 0 is a rigid block.
    sa_1p5ts : float
        5%-damped elastic spectral acceleration at the degraded period 1.5*Ts (g),
        > 0. For a rigid block (Ts -> 0) this is the peak ground acceleration.
    magnitude : float
        Moment magnitude Mw.
    rigid : bool, optional
        Force the rigid (True) / flexible (False) branch. Default None auto-selects
        rigid when ts < 0.05 s.

    Returns
    -------
    BrayTravasarou2007Result
        ``displacement`` (median, m), ``p_zero`` = P(D=0), plus the 16/84%
        displacements and the inputs.

    Notes
    -----
    Anchor (Bray 2007 Ch. 14 worked example): ky=0.14, Ts=0.4 s, Sa(1.5Ts)=0.48 g,
    M=7.2 -> ln(D)=2.29 (D~9.9 cm), P(D=0)=0.01. Validation V-062bt.

    Provenance (VERIFIED-PRIMARY, 2026-07-18 wiki-verification): every coefficient
    of the P(D=0) model (Eq. 3), the flexible median-D model (Eq. 5, a0=-1.10,
    sigma=0.66), and the RIGID branch (Eq. 6: a0=-0.22, 1.5*Ts term dropped,
    Sa(1.5Ts)->PGA, threshold Ts<0.05 s strict) was confirmed digit-for-digit
    against the original ASCE paper (Bray & Travasarou 2007, JGGE 133(4), pp.
    386-387) AND the Bray (2007) chapter PDF; both published (flexible) worked
    examples reproduce to printed precision, and the rigid branch matches a hand
    evaluation of printed Eq. 6 (no published rigid numeric example exists in the
    literature). Full ledger: module_work/wiki_verification/bray_travasarou_2007.md.
    """
    if ky <= 0:
        raise ValueError("ky must be positive (fraction of g)")
    if sa_1p5ts <= 0:
        raise ValueError("sa_1p5ts must be positive (g)")
    if ts < 0:
        raise ValueError("ts must be non-negative (s)")

    lnky = math.log(ky)
    lnsa = math.log(sa_1p5ts)

    # P(D=0) — probability of negligible displacement (Eq. 14.4).
    p_arg = -1.76 - 3.22 * lnky - 0.484 * ts * lnky + 3.52 * lnsa
    p_zero = 1.0 - _std_normal_cdf(p_arg)

    # Median non-zero displacement (Eq. 14.5); D in cm.
    is_rigid = (ts < 0.05) if rigid is None else bool(rigid)
    a0 = -0.22 if is_rigid else -1.10
    ts_term = 0.0 if is_rigid else 1.5 * ts
    ln_d = (a0 - 2.83 * lnky - 0.333 * lnky ** 2 + 0.566 * lnky * lnsa
            + 3.04 * lnsa - 0.244 * lnsa ** 2 + ts_term
            + 0.278 * (magnitude - 7.0))
    d_cm = math.exp(ln_d)

    return BrayTravasarou2007Result(
        displacement=d_cm / 100.0, p_zero=p_zero, ky=ky, ts=ts,
        sa_1p5ts=sa_1p5ts, magnitude=magnitude, ln_d_cm=ln_d, rigid=is_rigid)
