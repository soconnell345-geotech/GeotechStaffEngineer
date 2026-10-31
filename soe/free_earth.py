"""
Free-earth-support solvers for SOE walls: layered soil, effective stress, water.

These replace the single-layer routines used until 2026-09-15, which took
every property from the FIRST soil layer and used wrong moment arms (the
cantilever put the passive resultant at H + D/3 about the wall base; the
braced embedment started the active triangle at zero at the support and put
the passive resultant at D/3 below the excavation instead of 2D/3). On a
6.3 m cantilever in uniform sand they gave 2.5 m of embedment where 6.9 m is
required. Field feedback 2026-09-15 (an SOE re-run session), item N2.

Both procedures follow the Caltrans Trenching and Shoring Manual (2025):

* **Cantilever walls, Simplified Method (Sec 7-5.02).** Active and passive
  pressures to a point O at depth D0 below the excavation; moments about O
  give D0; D = 1.2 D0 (AASHTO 3.11.5.6 -- the increase accounts for rotation
  below O and is not a factor of safety); the net horizontal force at O is
  reported so a positive value (more push than resistance) can be flagged.
  The shear and moment come from the same body at FS = 1.0 (Example 7-3,
  p. 7-41), since GM-290.
* **Braced / anchored walls, free earth support about the lowest support
  (Sec 8-4.02, hinge method).** Embedment D from MR = FS x MD (FS = 1.3 in
  the manual); the support load T from MR = MD at D' and horizontal
  equilibrium; shear and moment along the wall from that FS = 1 body.
  Reproduces Example 8-1: D = 6.09 ft, D' = 4.89 ft, T = 14,254 lb/ft,
  M = 22,494 ft-lb/ft at the anchor.

Pressures per metre of wall (kPa), z measured down from the top of the wall:

    retained side      pa = max(0, Ka*s'v - 2c*sqrt(Ka))       (s'v includes q)
    excavation side    pp = Kp*s'v,exc + 2c*sqrt(Kp)           (z > H only)

with Rankine Ka/Kp of the layer at each depth (or explicit overrides; or,
with ``passive_theory="log_spiral"``, the log-spiral Kp of
:func:`passive_by_layer`, GM-242). Water
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

from geotech_common.numpy_compat import trapezoid
from geotech_common.water import GAMMA_W
from soe.earth_pressure import rankine_Ka, rankine_Kp

#: Layers are extended below the deepest given layer, with that layer's
#: properties, down to this depth (m).
_Z_FAR = 1000.0
#: Points per integration segment (segments break at layer boundaries,
#: the excavation level and the water tables).
_N_SEG = 129


def _per_layer(K, phis, rankine):
    """A coefficient for each layer: Rankine's from phi where ``K`` is None,
    ``K`` itself where it is a number, else ``K``'s own entry per layer."""
    if K is None:
        return [rankine(p) for p in phis]
    if np.ndim(K) == 0:
        return [float(K)] * len(phis)
    if len(K) != len(phis):
        raise ValueError(f"{len(K)} coefficients for {len(phis)} layers")
    return [rankine(p) if k is None else float(k) for k, p in zip(K, phis)]


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
    Ka, Kp : float or sequence, optional
        Coefficients applied instead of Rankine's: one for every layer, or
        one per layer (e.g. the log-spiral Kp of :func:`passive_by_layer`).
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
        self.Ka = np.array(_per_layer(Ka, phis, rankine_Ka))
        self.Kp = np.array(_per_layer(Kp, phis, rankine_Kp))
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
    return float(sum(trapezoid(f(z), z) for z in _grid(a, b, breaks)))


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

#: Depth searched for a moment balance, as a multiple of H.
_SEARCH_H = 6.0


def _diagram(z, p, V, M) -> dict:
    """A solution's diagram as plain lists (for a result and its tables)."""
    return {"z": [float(v) for v in z], "p_net": [float(v) for v in p],
            "V": [float(v) for v in V], "M": [float(v) for v in M]}


def _no_balance_cantilever(H: float) -> str:
    return (f"No moment balance about the point of rotation O within "
            f"{_SEARCH_H:g}H = {_SEARCH_H * H:.1f} m of embedment: the wall "
            "cannot stand as a cantilever in this soil (the passive resistance "
            "below the excavation never balances the driving pressures, "
            "Caltrans T&S 7-5.02). Consider supports (a braced or anchored "
            "wall) or a deeper or stiffer stratum. Embedment, moment and shear "
            "are not reported.")


def _no_balance_support(H: float, FS: float, at_fs: bool, at_one: bool) -> str:
    if not at_one:
        what = ("MR = MD is never reached: the passive resistance below the "
                "excavation cannot hold the toe of the wall")
        lost = ("D' (FS = 1), the support load from below, the moment and the "
                "shear are not reported"
                + (" (D at FS < 1 is)." if at_fs else ", nor is the embedment."))
    else:
        what = (f"MR = {FS:g} x MD is never reached (MR = MD is): the passive "
                "resistance below the excavation cannot hold the toe of the "
                "wall with that margin")
        lost = (f"The embedment at FS = {FS:g} is not reported; D' (FS = 1), "
                "the support load and the moments are.")
    return (f"No moment balance about the support within {_SEARCH_H:g}H = "
            f"{_SEARCH_H * H:.1f} m below the excavation (free earth support, "
            f"Caltrans T&S 8-4.02). {what}. Consider another support level, a "
            "lower support, or a deeper or stiffer stratum. " + lost)


@dataclass
class CantileverSolution:
    """Caltrans Simplified Method result (per metre of wall).

    With no moment balance within 6H (``converged`` False) D0, D and every
    quantity taken from the pressure diagram to O are None, and the first
    note says why.
    """
    D0: Optional[float]
    D: Optional[float]
    embedment_increase: float
    net_force_at_O: Optional[float]
    max_moment: Optional[float]
    max_moment_depth: Optional[float]
    max_shear: Optional[float]
    converged: bool
    notes: List[str] = field(default_factory=list)
    #: The net pressure, shear and moment of the FS = 1.0 body, top of the
    #: wall to its point O (``D0_FS1`` below the excavation): {"z", "p_net",
    #: "V", "M"} (lists; m, kPa, kN/m, kN-m/m). None without a moment
    #: balance.
    diagram: Optional[dict] = None
    #: D0 at FS = 1.0, the body the moment and shear come from (GM-290).
    D0_FS1: Optional[float] = None
    #: The maximum moment (and its depth) of the passive/FS body to O, as
    #: UFC 3-220-20 Fig. 4-41 (p. 242) takes it, for the choice (GM-290).
    max_moment_factored: Optional[float] = None
    max_moment_factored_depth: Optional[float] = None


def solve_cantilever(profile: LayeredProfile, FS_passive: float = 1.5,
                     embedment_increase: float = 1.2) -> CantileverSolution:
    """Cantilever wall by the Caltrans Simplified Method (Sec 7-5.02).

    Passive pressure is divided by ``FS_passive`` for D0 (moments about O);
    D = ``embedment_increase`` x D0. The moment and shear come from the same
    body at FS = 1.0, to its own point O: Example 7-3 (p. 7-41) takes the
    embedment at FS 1.3 and then "Calculate embedment depth using a factor
    of safety (FS) equal to 1.0 for the purposes of calculating the shear
    and moments" (Mmax at zero shear, p. 7-42). Until GM-290 they came from
    the passive/FS body, as UFC 3-220-20 Fig. 4-41 step 5 (p. 242) and its
    chart, Fig. 4-42, take them; that moment is kept as
    ``max_moment_factored``. When no D0 within 6H balances the moments the
    result carries None for the embedment, moment and shear,
    ``converged=False`` and a warning note.
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

    def net(D, fs=FS_passive):
        O = H + D
        md = _integrate(lambda z: drive(z) * (O - z), 0.0, O, brk)
        mr = _integrate(lambda z: profile.passive(z) / fs * (O - z), H, O,
                        brk + [O])
        return mr - md

    D0, ok = _first_root(net, 1e-3, _SEARCH_H * H)
    if not ok:
        return CantileverSolution(
            D0=None, D=None, embedment_increase=embedment_increase,
            net_force_at_O=None, max_moment=None, max_moment_depth=None,
            max_shear=None, converged=False,
            notes=[_no_balance_cantilever(H)])
    notes: List[str] = []
    O = H + D0
    F_drive = _integrate(drive, 0.0, O, brk)
    F_resist = _integrate(resist, H, O, brk + [O])
    net_force = F_drive - F_resist
    if net_force > 1e-6 * max(F_drive, 1.0):
        notes.append("Net horizontal force at O is toward the excavation "
                     f"({net_force:.1f} kN/m): increase embedment (Caltrans "
                     "7-5.02 step 4).")
    zf, Vf, Mf = _shear_moment(lambda zz: drive(zz) - resist(zz), 0.0, O, brk)
    kf = int(np.argmax(np.abs(Mf)))
    # The shear and moment at FS = 1.0 (Caltrans Example 7-3): the same
    # method's body to its own O
    D1, ok1 = (D0, True) if FS_passive == 1.0 else _first_root(
        lambda d: net(d, 1.0), 1e-3, _SEARCH_H * H)
    if ok1:
        O1 = H + D1
        z, V, M = _shear_moment(lambda zz: drive(zz) - profile.passive(zz),
                                0.0, O1, brk)
        p_net = drive(z) - profile.passive(z)
    else:
        D1 = None
        z, V, M, p_net = zf, Vf, Mf, drive(zf) - resist(zf)
        notes.append("No moment balance at FS = 1.0 within 6H: the moment "
                     "and shear are the passive/FS body's.")
    k = int(np.argmax(np.abs(M)))
    return CantileverSolution(
        D0=D0, D=D0 * embedment_increase, embedment_increase=embedment_increase,
        net_force_at_O=float(net_force), max_moment=float(abs(M[k])),
        max_moment_depth=float(z[k]), max_shear=float(np.max(np.abs(V))),
        converged=True, notes=notes,
        diagram=_diagram(z, p_net, V, M), D0_FS1=D1,
        max_moment_factored=float(abs(Mf[kf])),
        max_moment_factored_depth=float(zf[kf]))


@dataclass
class SupportSolution:
    """Free earth support about a support level (per metre of wall).

    ``D`` is None when MR = FS x MD has no root within 6H; ``D_prime`` and
    everything taken from the FS = 1 body (support load, moments, shear) are
    None when MR = MD has none. ``converged`` is False in either case and the
    first note says which.
    """
    D: Optional[float]
    D_prime: Optional[float]
    support_load: Optional[float]
    max_moment: Optional[float]
    max_moment_depth: Optional[float]
    zero_shear_depth: Optional[float]
    moment_at_zero_shear: Optional[float]
    max_shear: Optional[float]
    converged: bool
    notes: List[str] = field(default_factory=list)
    #: The FS = 1 body's net pressure, shear and moment, body_top to the
    #: wall's tip: {"z", "p_net", "V", "M"}; the support load acts at the
    #: pivot. None without an FS = 1 balance.
    diagram: Optional[dict] = None


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
                        1e-3, _SEARCH_H * H)
    Dp, ok_p = _first_root(lambda d: (lambda m: m[1] - m[0])(moments(d)),
                           1e-3, _SEARCH_H * H)
    if not (ok and ok_p):
        notes.append(_no_balance_support(H, FS, ok, ok_p))
    if not ok:
        D = None
    if not ok_p:
        return SupportSolution(
            D=D, D_prime=None, support_load=None, max_moment=None,
            max_moment_depth=None, zero_shear_depth=None,
            moment_at_zero_shear=None, max_shear=None, converged=False,
            notes=notes)
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
        converged=ok and ok_p, notes=notes,
        diagram=_diagram(z, drive(z) - profile.passive(z), V, M))


# ---------------------------------------------------------------------------
# bridges from ExcavationGeometry
# ---------------------------------------------------------------------------

#: The theories of the passive pressure below the excavation (GM-242):
#: Rankine's Kp, no wall friction (the default, as the Caltrans Trenching
#: and Shoring Manual's examples take it), or the log-spiral Kp of
#: ``geotech_common.log_spiral`` with wall friction.
PASSIVE_THEORIES = ("rankine", "log_spiral")

#: Why the passive pressure takes Rankine's Kp by default (GM-242), for the
#: choice, in the Caltrans Trenching and Shoring Manual's words.
PASSIVE_BASIS = (
    "The Caltrans Trenching and Shoring Manual, which these solvers follow "
    "(Secs. 7-5.02 and 8-4.02), takes Rankine's where its examples have no "
    "wall friction: 'since the wall friction (delta) is zero, use Rankine "
    "earth pressure theory to calculate the active and passive earth "
    "pressure coefficients' (pp. 7-18, 7-27 and 9-12), each time with the "
    "note 'Rankine theory tends to underestimate the passive earth "
    "pressure. It is recommended to use the Log-Spiral model to compute the "
    "passive earth force' (p. 7-27; 'Log-Spiral-Rankine Model', pp. 7-18 "
    "and 9-13); Example 8-1 takes the log-spiral Kp at delta 15 deg on phi "
    "30 deg (p. 8-12); and Sec. 4-6 (p. 4-28) says Rankine's theory "
    "'should not be used to calculate the passive earth pressure forces "
    "for a shoring system because it does not account for wall friction'. "
    "The module keeps Rankine's, the smaller Kp, as its default; "
    "passive_theory 'log_spiral' takes the log-spiral Kp with wall "
    "friction. At delta = 0 the two are the same Kp")

#: Each layer's wall friction on the passive side when it is left out
#: (GM-242), for the choice.
PASSIVE_FRICTION_LEFT_OUT = (
    "Left out, a layer below the excavation with no cohesion takes "
    "delta/phi' = 0.66 (UFC 3-220-20 p. 191, 'a common assumption', the "
    "wall friction its Figs. 4-12 and 4-13 are drawn for); "
    "log_spiral_source 'caquot_kerisel_1948' or 'dm7_2_eq_4_12' needs it "
    "given. A layer with cohesion takes none, so Rankine's Kp, as UFC "
    "3-220-20's anchored bulkhead procedure does (Fig. 4-37 step 1, p. "
    "237): 'Passive pressures for clean coarse grained soils include wall "
    "friction (see Table 4-1). For active or passive pressures in all "
    "other soil types, ignore wall friction.' Judgment, for the engineer "
    "to confirm: a layer with no cohesion is read as a clean coarse-grained "
    "soil; for a steel sheet pile Table 4-1 (p. 184) gives 17 deg against "
    "clean sand (the sheet_pile tool's default; less than 0.66 phi' for "
    "phi' above 26 deg) and 22 deg against clean gravel. A wall friction "
    "given "
    "is used in every layer below the excavation, at most its phi'")


def check_passive(passive_theory: str = "rankine",
                  log_spiral_source: Optional[str] = None,
                  passive_wall_friction_deg: Optional[float] = None) -> None:
    """The passive theory's inputs checked, and refused where they are not
    used (GM-242)."""
    from geotech_common import log_spiral as _ls
    if passive_theory not in PASSIVE_THEORIES:
        raise ValueError(f"passive_theory must be one of "
                         f"{list(PASSIVE_THEORIES)}, got {passive_theory!r}")
    if log_spiral_source is not None \
            and log_spiral_source not in _ls.SOURCES:
        raise ValueError(f"log_spiral_source must be one of "
                         f"{list(_ls.SOURCES)}, got {log_spiral_source!r}")
    if passive_theory == "rankine":
        unused = [n for n, v in (
            ("log_spiral_source", log_spiral_source),
            ("passive_wall_friction_deg", passive_wall_friction_deg))
            if v is not None]
        if unused:
            raise ValueError(
                f"{' and '.join(unused)}: for the log-spiral passive Kp "
                "below the excavation (passive_theory 'log_spiral'); "
                "Rankine's Kp takes no wall friction. Give passive_theory "
                "'log_spiral', or leave "
                f"{'it' if len(unused) == 1 else 'them'} out.")
    if passive_wall_friction_deg is not None \
            and not 0.0 <= passive_wall_friction_deg < 90.0:
        raise ValueError(f"passive_wall_friction_deg must be in [0, 90) "
                         f"deg, got {passive_wall_friction_deg:g}")
    if passive_theory == "log_spiral" and passive_wall_friction_deg is None \
            and log_spiral_source in _ls.LEVEL_FRONT_NEEDS_DELTA:
        raise ValueError(_ls.LEVEL_FRONT_NEEDS_DELTA[
            log_spiral_source].format(name="passive_wall_friction_deg"))


def passive_by_layer(layers: Sequence, H: float,
                     passive_theory: str = "rankine",
                     log_spiral_source: Optional[str] = None,
                     passive_wall_friction_deg: Optional[float] = None
                     ) -> List[dict]:
    """Each layer's passive coefficient (GM-242): ``Kp``, ``delta`` (deg),
    ``came`` and ``below`` (whether any of the layer is below the excavation
    level ``H``, where the passive pressure acts).

    "rankine": Rankine's Kp, delta 0 (``came`` "rankine"). "log_spiral",
    in a layer below the excavation: the log-spiral Kp of
    ``log_spiral_source`` (default UFC 3-220-20's, Eq. 4-12 for the vertical
    face and level ground in front of the wall) at the wall friction given,
    at most the layer's phi' ("given"), or left out delta/phi' = 0.66 with
    no cohesion ("source"; refused with the 1948 chart and Eq. 4-12 by
    name) and none with cohesion ("cohesion",
    :data:`PASSIVE_FRICTION_LEFT_OUT`); phi' = 0 gives Kp = 1 ("phi_0").
    With delta = 0 the log-spiral Kp is Rankine's, and Rankine's is taken.
    A layer wholly above the excavation keeps Rankine's (``came``
    "above")."""
    from geotech_common import log_spiral as _ls
    check_passive(passive_theory, log_spiral_source,
                  passive_wall_friction_deg)
    source = log_spiral_source or _ls.DEFAULT_SOURCE
    out, top = [], 0.0
    for layer in layers:
        phi = float(getattr(layer, "friction_angle", 0.0) or 0.0)
        c = float(getattr(layer, "cohesion", 0.0) or 0.0)
        bottom = top + float(layer.thickness)
        below = bottom > H or layer is layers[-1]
        top = bottom
        row = {"Kp": rankine_Kp(phi), "delta": 0.0, "below": below,
               "came": "rankine"}
        out.append(row)
        if passive_theory == "rankine":
            continue
        if not below:
            row["came"] = "above"
        elif phi <= 0:
            row.update(Kp=1.0, came="phi_0")
        elif passive_wall_friction_deg is not None:
            row.update(delta=min(float(passive_wall_friction_deg), phi),
                       came="given")
        elif c > 0:
            row["came"] = "cohesion"
        else:
            row.update(delta=_ls.level_front_Kp(phi, None, source)
                       ["delta_deg"], came="source")
        if row["delta"] > 0:
            row["Kp"] = _ls.level_front_Kp(phi, row["delta"], source)["Kp"]
    return out


def profile_for(geometry, Ka: Optional[float] = None,
                Kp: Optional[float] = None,
                passive_theory: str = "rankine",
                log_spiral_source: Optional[str] = None,
                passive_wall_friction_deg: Optional[float] = None
                ) -> LayeredProfile:
    """The LayeredProfile of an ``ExcavationGeometry``; with
    ``passive_theory`` "log_spiral" each layer's Kp from
    :func:`passive_by_layer` (GM-242)."""
    if passive_theory != "rankine" or log_spiral_source is not None \
            or passive_wall_friction_deg is not None:
        if Kp is not None:
            raise ValueError("Kp given and a passive_theory: give one")
        Kp = [row["Kp"] for row in passive_by_layer(
            geometry.soil_layers, geometry.excavation_depth,
            passive_theory, log_spiral_source, passive_wall_friction_deg)]
    return LayeredProfile(
        geometry.soil_layers, geometry.excavation_depth,
        surcharge=geometry.surcharge, gwt_retained=geometry.gwt_depth,
        gwt_excavation=getattr(geometry, "gwt_depth_excavation", None),
        Ka=Ka, Kp=Kp)


def apparent_pressure_profile(geometry, stiff_clay_coefficient=None):
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
    ``stiff_clay_coefficient``: the stiff clay ordinate / gamma H (None:
    ``soe.earth_pressure.STIFF_CLAY_COEFFICIENT``, 0.4).
    """
    from soe.earth_pressure import (STIFF_CLAY_COEFFICIENT,
                                    select_apparent_pressure)

    H = float(geometry.excavation_depth)
    q = float(geometry.surcharge or 0.0)
    gwt = geometry.gwt_depth
    ap = dict(select_apparent_pressure(
        geometry.soil_layers, H, q, gwt_depth=gwt,
        stiff_clay_coefficient=(STIFF_CLAY_COEFFICIENT
                                if stiff_clay_coefficient is None
                                else stiff_clay_coefficient)))
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
        elif shape == "ramped_top":
            # soft to medium clay (GM-290): a 0.25H ramp at the top only
            env = np.where(z <= 0.25 * H, p_max * z / (0.25 * H), p_max)
        else:
            env = np.full_like(z, p_max)
        total = env + ps
        if add_water:
            total = total + GAMMA_W * np.maximum(0.0, z - gwt)
        return np.where(inside, total, 0.0)

    return ap, p


__all__ = ["LayeredProfile", "CantileverSolution", "SupportSolution",
           "solve_cantilever", "solve_about_support", "profile_for",
           "apparent_pressure_profile", "PASSIVE_THEORIES",
           "passive_by_layer", "check_passive"]
