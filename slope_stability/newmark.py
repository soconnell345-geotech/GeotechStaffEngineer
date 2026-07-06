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
   time history against ky (Newmark 1965). The classic downslope-only assumption
   is used: the block slides only while the ground acceleration drives it in the
   destabilizing (downslope) direction and the relative velocity is positive; it
   never slides back upslope, so displacement accumulates monotonically. A
   time history that is symmetric about zero is therefore rectified to its
   destabilizing polarity (both polarities of a real record are handled by taking
   the absolute acceleration — the standard single-block treatment).

3. ``newmark_jibson2007`` — Jibson's (2007) regression estimate of Newmark
   displacement from the critical-acceleration ratio ky/amax, an independent
   cross-check that needs no time history.

References
----------
Newmark, N.M. (1965). "Effects of earthquakes on dams and embankments."
    Geotechnique 15(2), 139-160.
Jibson, R.W. (2007). "Regression models for estimating coseismic landslide
    displacement." Engineering Geology 91(2-4), 209-218. (Eq. 6.)
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
    """Yield (critical) seismic coefficient for a specified surface."""
    ky: float = 0.0                # yield seismic coefficient (fraction of g)
    ay: float = 0.0                # yield acceleration = ky*g (m/s^2)
    fos_static: float = 0.0        # pseudo-static FOS at kh = 0
    method: str = "spencer"
    converged: bool = False
    n_iter: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ky": round(self.ky, 5),
            "ay_m_s2": round(self.ay, 5),
            "FOS_static": round(self.fos_static, 4),
            "method": self.method,
            "converged": self.converged,
        }


@dataclass
class NewmarkResult:
    """Newmark permanent-displacement result."""
    displacement: float = 0.0      # permanent downslope displacement (m)
    ky: float = 0.0
    method: str = "integration"    # 'integration' or 'jibson2007'
    amax: float = 0.0              # peak ground acceleration used (fraction of g)
    n_exceedances: int = 0         # times the record exceeds ay (integration)
    duration: float = 0.0          # s (integration)
    sigma_log10: Optional[float] = None   # Jibson dispersion (log10 cm)

    @property
    def displacement_cm(self) -> float:
        return self.displacement * 100.0

    def summary(self) -> str:
        return (f"Newmark displacement ({self.method}): "
                f"{self.displacement_cm:.3f} cm  (ky={self.ky:g}"
                + (f", amax={self.amax:g}g" if self.amax else "") + ")")

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
        else:
            d["amax_g"] = round(self.amax, 5)
        return d


def _fos_at_kh(geom: SlopeGeometry, slip, method: str, kh: float,
               n_slices: int, tol: float) -> float:
    from slope_stability.analysis import analyze_slope
    g = copy.copy(geom)          # shallow copy: only kh is overridden
    g.kh = kh
    res = analyze_slope(g, slip_surface=slip, method=method,
                        n_slices=n_slices, tol=tol)
    return res.FOS


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

    Returns
    -------
    YieldAccelerationResult
    """
    if slip_surface is not None:
        slip = slip_surface
    else:
        if xc is None or yc is None or radius is None:
            raise ValueError("provide slip_surface or all of xc, yc, radius")
        slip = CircularSlipSurface(xc, yc, radius)

    fos_static = _fos_at_kh(geom, slip, method, 0.0, n_slices, tol)
    if fos_static <= 1.0:
        # already at/over the limit with no seismic load -> ky = 0
        return YieldAccelerationResult(
            ky=0.0, ay=0.0, fos_static=fos_static, method=method,
            converged=True, n_iter=0)

    lo, hi = 0.0, kh_max
    fos_hi = _fos_at_kh(geom, slip, method, hi, n_slices, tol)
    if fos_hi > 1.0:
        # surface never yields within the bracket
        return YieldAccelerationResult(
            ky=kh_max, ay=kh_max * _G, fos_static=fos_static, method=method,
            converged=False, n_iter=0)

    n = 0
    while hi - lo > bisection_tol and n < 100:
        mid = 0.5 * (lo + hi)
        f = _fos_at_kh(geom, slip, method, mid, n_slices, tol)
        if f > 1.0:
            lo = mid
        else:
            hi = mid
        n += 1
    ky = 0.5 * (lo + hi)
    return YieldAccelerationResult(
        ky=ky, ay=ky * _G, fos_static=fos_static, method=method,
        converged=True, n_iter=n)


def newmark_displacement(ky: float,
                         accel: Sequence[float],
                         dt: float,
                         accel_in_g: bool = False,
                         g: float = _G) -> NewmarkResult:
    """Permanent downslope displacement by Newmark double integration.

    Parameters
    ----------
    ky : float
        Yield seismic coefficient (fraction of g); yield acceleration ay = ky*g.
    accel : sequence of float
        Ground acceleration time history, equally spaced at ``dt``. In m/s^2 by
        default, or in g if ``accel_in_g=True``.
    dt : float
        Time step (s).
    accel_in_g : bool
        If True, ``accel`` is in units of g (multiplied by g internally).
    g : float
        Gravity (m/s^2). Default 9.80665.

    Returns
    -------
    NewmarkResult
        ``displacement`` in metres; ``displacement_cm`` property in cm.

    Notes
    -----
    Downslope-only rigid-block integration: at each step the block slides only
    while |a_ground| exceeds ay and the relative velocity is positive; the
    relative velocity is clamped at zero (no upslope rebound), so displacement
    is monotonic. The absolute value of the record is used so both polarities of
    a symmetric record drive the (single) downslope block — the standard
    single-block treatment (Jibson 2007; Duncan-Wright-Brandon 2014, Ch. 10).
    """
    if ky < 0:
        raise ValueError("ky must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")
    ay = ky * g
    scale = g if accel_in_g else 1.0

    v = 0.0          # relative velocity (m/s)
    d = 0.0          # cumulative relative displacement (m)
    n_exc = 0
    amax = 0.0
    for a_raw in accel:
        a = abs(a_raw) * scale
        amax = max(amax, a)
        if a > ay or v > 0.0:
            if a > ay:
                n_exc += 1
            v_prev = v
            v = v_prev + (a - ay) * dt
            if v < 0.0:
                v = 0.0
            d += 0.5 * (v_prev + v) * dt      # trapezoidal
    return NewmarkResult(
        displacement=d, ky=ky, method="integration",
        amax=amax / g, n_exceedances=n_exc, duration=len(accel) * dt)


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
