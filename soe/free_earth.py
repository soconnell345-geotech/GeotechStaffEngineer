"""
Free-earth-support solvers for SOE walls: layered soil, effective stress, water.

These replace the single-layer routines used until 2026-09-15, which took
every property from the FIRST soil layer and used wrong moment arms (the
cantilever put the passive resultant at H + D/3 about the wall base; the
braced embedment started the active triangle at zero at the support and put
the passive resultant at D/3 below the excavation instead of 2D/3). On a
6.3 m cantilever in uniform sand they gave 2.5 m of embedment where 6.9 m is
required. Field feedback 2026-09-15 (Nairobi SOE re-run), item N2.

Both procedures follow the Caltrans Trenching and Shoring Manual (2025):

* **Cantilever walls, Simplified Method (Sec 7-5.02).** Active and passive
  pressures to a point O at depth D0 below the excavation; moments about O
  give D0; D = 1.2 D0 (AASHTO 3.11.5.6 -- the increase accounts for rotation
  below O and is not a factor of safety); the net horizontal force at O is
  reported so a positive value (more push than resistance) can be flagged.
* **Braced / anchored walls, free earth support about the lowest support
  (Sec 8-4.02, hinge method).** Embedment D from MR = FS x MD (FS = 1.3 in
  the manual); the support load T from MR = MD at D' and horizontal
  equilibrium; shear and moment along the wall from that FS = 1 body.
  Reproduces Example 8-1: D = 6.09 ft, D' = 4.89 ft, T = 14,254 lb/ft,
  M = 22,494 ft-lb/ft at the anchor.

Pressures per metre of wall (kPa), z measured down from the top of the wall:

    retained side      pa = max(0, Ka*s'v - 2c*sqrt(Ka))       (s'v includes q)
    excavation side    pp = Kp*s'v,exc + 2c*sqrt(Kp)           (z > H only)

with Rankine Ka/Kp of the layer at each depth (or explicit overrides). Water
is hydrostatic on each side: retained side from ``gwt_retained``, excavation
side from max(``gwt_excavation``, H) -- the excavation is never flooded above
its own base -- so the net water pressure below the base is carried as a
driving pressure. Above the excavation a caller may supply its own pressure
(an apparent-pressure envelope) in place of the Rankine active pressure.

All units SI: m, kPa, kN/m, kN.m/m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from geotech_common.water import GAMMA_W
from soe.earth_pressure import rankine_Ka, rankine_Kp

#: Layers are extended below the deepest given layer, with that layer's
#: properties, down to this depth (m).
_Z_FAR = 1000.0
#: Points per integration segment (segments break at layer boundaries,
#: the excavation level and the water tables).
_N_SEG = 129


class LayeredProfile:
    """Soil, surcharge and water on both sides of an excavation wall.

    Parameters
    ----------
    layers : sequence
        Objects with ``thickness``, ``unit_weight``, ``friction_angle``,
        ``cohesion`` (``SOEWallLayer``), top down. The last layer continues
        below its stated thickness.
    H : float
        Excavation depth (m).
    surcharge : float
        Uniform surcharge on the retained side (kPa).
    gwt_retained : float, optional
        Water table depth on the retained side (m). None = dry.
    gwt_excavation : float, optional
        Water table depth on the excavation side (m); clipped to be no
        shallower than H. Defaults to ``gwt_retained``.
    Ka, Kp : float, optional
        Coefficients applied to every layer instead of Rankine (e.g. a
        log-spiral Kp).
    """

    def __init__(self, layers: Sequence, H: float, surcharge: float = 0.0,
                 gwt_retained: Optional[float] = None,
                 gwt_excavation: Optional[float] = None,
                 gamma_w: float = GAMMA_W,
                 Ka: Optional[float] = None, Kp: Optional[float] = None):
        if H <= 0:
            raise ValueError("Excavation depth H must be positive")
        if not layers:
            raise ValueError("At least one soil layer is required")
        self.H = float(H)
        self.q = float(surcharge or 0.0)
        self.gamma_w = float(gamma_w)
        self.gwt_r = None if gwt_retained is None else float(gwt_retained)
        if gwt_excavation is None:
            gwt_excavation = self.gwt_r
        self.gwt_e = (None if gwt_excavation is None
                      else max(float(gwt_excavation), self.H))
        thick = np.array([float(l.thickness) for l in layers])
        if np.any(thick <= 0):
            raise ValueError("All layer thicknesses must be positive")
        self.bottoms = np.cumsum(thick)
        self.gamma = np.array([float(l.unit_weight) for l in layers])
        self.c = np.array([float(getattr(l, "cohesion", 0.0) or 0.0)
                           for l in layers])
        phis = [float(getattr(l, "friction_angle", 0.0) or 0.0) for l in layers]
        self.Ka = np.array([float(Ka) if Ka is not None else rankine_Ka(p)
                            for p in phis])
        self.Kp = np.array([float(Kp) if Kp is not None else rankine_Kp(p)
                            for p in phis])
        self._sv_r = self._stress_table(0.0, self.gwt_r, self.q)
        self._sv_e = self._stress_table(self.H, self.gwt_e, 0.0)

    # -- vertical effective stress: exact piecewise-linear tables ----------
    def _layer(self, z):
        idx = np.searchsorted(self.bottoms, z, side="right")
        return np.minimum(idx, len(self.bottoms) - 1)

    def _stress_table(self, z0: float, gwt: Optional[float], s0: float):
        pts = {z0, _Z_FAR}
        pts.update(float(b) for b in self.bottoms if z0 < b < _Z_FAR)
        if gwt is not None and z0 < gwt < _Z_FAR:
            pts.add(float(gwt))
        zs = sorted(pts)
        sig = [s0]
        for a, b in zip(zs[:-1], zs[1:]):
            g = self.gamma[int(self._layer(0.5 * (a + b)))]
            if gwt is not None and a >= gwt:
                g -= self.gamma_w
            sig.append(sig[-1] + g * (b - a))
        return np.array(zs), np.array(sig)

    def sv_retained(self, z):
        return np.interp(z, *self._sv_r)

    def sv_excavation(self, z):
        z = np.asarray(z, dtype=float)
        return np.where(z > self.H, np.interp(z, *self._sv_e), 0.0)

    # -- pressures ----------------------------------------------------------
    def u_retained(self, z):
        z = np.asarray(z, dtype=float)
        if self.gwt_r is None:
            return np.zeros_like(z)
        return self.gamma_w * np.maximum(0.0, z - self.gwt_r)

    def u_excavation(self, z):
        z = np.asarray(z, dtype=float)
        if self.gwt_e is None:
            return np.zeros_like(z)
        return self.gamma_w * np.maximum(0.0, z - self.gwt_e)

    def active(self, z):
        """Rankine active earth pressure, no tension (kPa)."""
        i = self._layer(z)
        Ka = self.Ka[i]
        return np.maximum(0.0, Ka * self.sv_retained(z)
                          - 2.0 * self.c[i] * np.sqrt(Ka))

    def passive(self, z):
        """Rankine passive earth pressure on the excavation side (kPa)."""
        z = np.asarray(z, dtype=float)
        i = self._layer(z)
        Kp = self.Kp[i]
        pp = Kp * self.sv_excavation(z) + 2.0 * self.c[i] * np.sqrt(Kp)
        return np.where(z > self.H, pp, 0.0)

    def rankine_drive(self, z):
        """Active earth pressure plus net water pressure (kPa)."""
        return self.active(z) + self.u_retained(z) - self.u_excavation(z)

    def breaks(self) -> List[float]:
        out = [float(b) for b in self.bottoms] + [self.H]
        out += [w for w in (self.gwt_r, self.gwt_e) if w is not None]
        return out


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------

def _grid(a: float, b: float, breaks: Sequence[float], n: int = _N_SEG):
    """Sample points per segment, nudged inside so a boundary point takes the
    properties of its own segment."""
    pts = [a] + sorted(x for x in set(breaks) if a < x < b) + [b]
    segs = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        z = np.linspace(lo, hi, n)
        eps = 1e-9 * (hi - lo)
        z[0] += eps
        z[-1] -= eps
        segs.append(z)
    return segs


def _integrate(f: Callable, a: float, b: float, breaks: Sequence[float]) -> float:
    if b <= a:
        return 0.0
    return float(sum(np.trapezoid(f(z), z) for z in _grid(a, b, breaks)))


def _first_root(f: Callable[[float], float], lo: float, hi: float,
                n_scan: int = 160, tol: float = 1e-5):
    """Smallest x in [lo, hi] where f turns non-negative. Returns (x, found)."""
    xs = np.linspace(lo, hi, n_scan)
    prev = None
    for x in xs:
        if f(x) >= 0.0:
            if prev is None:
                return float(x), True
            a, b = prev, float(x)
            while b - a > tol:
                m = 0.5 * (a + b)
                if f(m) >= 0.0:
                    b = m
                else:
                    a = m
            return b, True
        prev = float(x)
    return float(hi), False


def _shear_moment(q: Callable, top: float, bottom: float, breaks,
                  point_loads: Sequence = ()):
    """Shear V(z) and moment M(z) for distributed load q (push toward the
    excavation positive) plus point loads [(depth, force)] resisting it."""
    zs, qs = [], []
    for z in _grid(top, bottom, list(breaks) + [d for d, _ in point_loads]):
        zs.append(z)
        qs.append(q(z))
    z = np.concatenate(zs)
    qv = np.concatenate(qs)
    V = np.concatenate([[0.0], np.cumsum(0.5 * (qv[1:] + qv[:-1]) * np.diff(z))])
    for depth, force in point_loads:
        V = V - force * (z >= depth)
    M = np.concatenate([[0.0], np.cumsum(0.5 * (V[1:] + V[:-1]) * np.diff(z))])
    return z, V, M


# ---------------------------------------------------------------------------
# solutions
# ---------------------------------------------------------------------------

@dataclass
class CantileverSolution:
    """Caltrans Simplified Method result (per metre of wall)."""
    D0: float
    D: float
    embedment_increase: float
    net_force_at_O: float
    max_moment: float
    max_moment_depth: float
    max_shear: float
    converged: bool
    notes: List[str] = field(default_factory=list)


def solve_cantilever(profile: LayeredProfile, FS_passive: float = 1.5,
                     embedment_increase: float = 1.2) -> CantileverSolution:
    """Cantilever wall by the Caltrans Simplified Method (Sec 7-5.02).

    Passive pressure is divided by ``FS_passive``; D = ``embedment_increase``
    x D0. Moment and shear come from the pressure diagram to point O.
    """
    if FS_passive <= 0:
        raise ValueError("FS_passive must be positive")
    if embedment_increase < 1.0:
        raise ValueError("embedment_increase must be >= 1.0")
    H = profile.H
    brk = profile.breaks()
    drive = profile.rankine_drive

    def resist(z):
        return profile.passive(z) / FS_passive

    def net(D):
        O = H + D
        md = _integrate(lambda z: drive(z) * (O - z), 0.0, O, brk)
        mr = _integrate(lambda z: resist(z) * (O - z), H, O, brk + [O])
        return mr - md

    notes: List[str] = []
    D0, ok = _first_root(net, 1e-3, 6.0 * H)
    if not ok:
        notes.append(f"No moment balance within 6H = {6 * H:.1f} m of embedment; "
                     "the wall cannot be a cantilever in this soil.")
    O = H + D0
    F_drive = _integrate(drive, 0.0, O, brk)
    F_resist = _integrate(resist, H, O, brk + [O])
    net_force = F_drive - F_resist
    if net_force > 1e-6 * max(F_drive, 1.0):
        notes.append("Net horizontal force at O is toward the excavation "
                     f"({net_force:.1f} kN/m): increase embedment (Caltrans "
                     "7-5.02 step 4).")
    z, V, M = _shear_moment(lambda zz: drive(zz) - resist(zz), 0.0, O, brk)
    k = int(np.argmax(np.abs(M)))
    return CantileverSolution(
        D0=D0, D=D0 * embedment_increase, embedment_increase=embedment_increase,
        net_force_at_O=float(net_force), max_moment=float(abs(M[k])),
        max_moment_depth=float(z[k]), max_shear=float(np.max(np.abs(V))),
        converged=ok, notes=notes)


@dataclass
class SupportSolution:
    """Free earth support about a support level (per metre of wall)."""
    D: float
    D_prime: float
    support_load: float
    max_moment: float
    max_moment_depth: float
    zero_shear_depth: Optional[float]
    moment_at_zero_shear: Optional[float]
    max_shear: float
    converged: bool
    notes: List[str] = field(default_factory=list)


def solve_about_support(profile: LayeredProfile, pivot: float,
                        body_top: float = 0.0, FS: float = 1.3,
                        above: Optional[Callable] = None) -> SupportSolution:
    """Embedment and support load by free earth support about ``pivot``.

    Parameters
    ----------
    profile : LayeredProfile
    pivot : float
        Depth of the (lowest) support (m), 0 < pivot < H.
    body_top : float
        Top of the free body: 0 for a single support (the whole wall), the
        lowest support depth for a multi-level wall analysed by the hinge
        method (moments at the supports above are taken as zero).
    FS : float
        Embedment from MR = FS x MD (Caltrans uses 1.3).
    above : callable, optional
        Pressure p(z) (kPa) to use for z <= H instead of the Rankine active
        + water pressure -- an apparent-pressure envelope, with any surcharge
        and water the caller wants on it. Below H the Rankine active, net
        water and passive pressures always apply.
    """
    H = profile.H
    if not 0.0 < pivot < H:
        raise ValueError("pivot must lie between 0 and H")
    if not 0.0 <= body_top <= pivot:
        raise ValueError("body_top must lie between 0 and the pivot")
    if FS <= 0:
        raise ValueError("FS must be positive")
    brk = profile.breaks() + [pivot, body_top]

    def drive(z):
        z = np.asarray(z, dtype=float)
        below = profile.rankine_drive(z)
        if above is None:
            return below
        return np.where(z <= H, above(z), below)

    def moments(D):
        bot = H + D
        md = _integrate(lambda z: drive(z) * (z - pivot), body_top, bot, brk)
        mr = _integrate(lambda z: profile.passive(z) * (z - pivot), H, bot,
                        brk + [bot])
        return md, mr

    notes: List[str] = []
    D, ok = _first_root(lambda d: (lambda m: m[1] - FS * m[0])(moments(d)),
                        1e-3, 6.0 * H)
    Dp, ok_p = _first_root(lambda d: (lambda m: m[1] - m[0])(moments(d)),
                           1e-3, 6.0 * H)
    if not (ok and ok_p):
        notes.append(f"No moment balance within 6H = {6 * H:.1f} m of embedment.")
    bot = H + Dp
    T = (_integrate(drive, body_top, bot, brk)
         - _integrate(profile.passive, H, bot, brk + [bot]))
    z, V, M = _shear_moment(lambda zz: drive(zz) - profile.passive(zz),
                            body_top, bot, brk, point_loads=[(pivot, T)])
    k = int(np.argmax(np.abs(M)))
    zs_depth = zs_moment = None
    below = np.nonzero(z > pivot)[0]
    if below.size > 1:
        sign = np.sign(V[below])
        flips = np.nonzero(sign[1:] != sign[:-1])[0]
        if flips.size:
            j = below[flips[0]]
            # linear interpolation of the zero crossing
            z0 = z[j] - V[j] * (z[j + 1] - z[j]) / (V[j + 1] - V[j])
            zs_depth = float(z0)
            zs_moment = float(abs(np.interp(z0, z, M)))
    return SupportSolution(
        D=D, D_prime=Dp, support_load=float(T), max_moment=float(abs(M[k])),
        max_moment_depth=float(z[k]), zero_shear_depth=zs_depth,
        moment_at_zero_shear=zs_moment, max_shear=float(np.max(np.abs(V))),
        converged=ok and ok_p, notes=notes)


# ---------------------------------------------------------------------------
# bridges from ExcavationGeometry
# ---------------------------------------------------------------------------

def profile_for(geometry, Ka: Optional[float] = None,
                Kp: Optional[float] = None) -> LayeredProfile:
    """The LayeredProfile of an ``ExcavationGeometry``."""
    return LayeredProfile(
        geometry.soil_layers, geometry.excavation_depth,
        surcharge=geometry.surcharge, gwt_retained=geometry.gwt_depth,
        gwt_excavation=getattr(geometry, "gwt_depth_excavation", None),
        Ka=Ka, Kp=Kp)


def apparent_pressure_profile(geometry):
    """Design pressure above the excavation for a braced wall.

    The apparent-pressure envelope plus the surcharge and water pressures
    that FHWA GEC-4 Sec 5.2.4 says to add explicitly ("Water pressures and
    surcharge pressures should be added explicitly to the diagram"). Sand
    envelopes are effective-stress (0.65 Ka gamma' H, gamma' below the water
    table), so hydrostatic water on the retained side is added; clay
    envelopes are total-stress, so water is not added again. Surcharge adds
    K*q over the excavated height, K = the envelope Ka for sand and 1.0 for
    clay (undrained, phi = 0).

    Returns ``(envelope, p)``: the ``select_apparent_pressure`` dict
    (with ``surcharge_pressure_kPa`` and ``water_pressure_added`` added) and
    a vectorized ``p(z)`` in kPa, zero outside 0 <= z <= H.
    """
    from soe.earth_pressure import select_apparent_pressure

    H = float(geometry.excavation_depth)
    q = float(geometry.surcharge or 0.0)
    gwt = geometry.gwt_depth
    ap = dict(select_apparent_pressure(geometry.soil_layers, H, q,
                                       gwt_depth=gwt))
    shape, p_max = ap["shape"], float(ap["max_pressure_kPa"])
    sand = ap["type"] == "sand"
    ps = (float(ap.get("Ka", 1.0)) if sand else 1.0) * q
    add_water = bool(sand and gwt is not None and gwt < H)
    ap["surcharge_pressure_kPa"] = round(ps, 3)
    ap["water_pressure_added"] = add_water

    def p(z):
        z = np.asarray(z, dtype=float)
        inside = (z >= 0.0) & (z <= H)
        if shape == "trapezoidal":
            env = np.where(z <= 0.25 * H, p_max * z / (0.25 * H),
                           np.where(z <= 0.75 * H, p_max,
                                    p_max * (H - z) / (0.25 * H)))
        else:
            env = np.full_like(z, p_max)
        total = env + ps
        if add_water:
            total = total + GAMMA_W * np.maximum(0.0, z - gwt)
        return np.where(inside, total, 0.0)

    return ap, p


__all__ = ["LayeredProfile", "CantileverSolution", "SupportSolution",
           "solve_cantilever", "solve_about_support", "profile_for",
           "apparent_pressure_profile"]
