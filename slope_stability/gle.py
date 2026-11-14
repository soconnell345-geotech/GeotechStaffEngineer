"""
Rigorous General Limit Equilibrium (GLE / Morgenstern-Price) engine.

Implements the Fredlund-Krahn (1977) GLE formulation with true
interslice-force integration:

- Interslice shear  X = lambda * f(x) * E  at each slice boundary.
- Slice base normal N from vertical equilibrium including interslice
  shear terms (the textbook m_alpha equation).
- Moment-equilibrium FOS (F_m) about the circle centre (circular) or a
  fitted axis point (noncircular), with all moment arms computed by
  explicit cross products so the N*f term is exact for any surface shape.
- Force-equilibrium FOS (F_f) from horizontal equilibrium, with the
  interslice normals E obtained by marching slice-to-slice.
- lambda solved so that F_m(lambda) = F_f(lambda) (bracketed root find).

Special cases recovered exactly:
- Bishop's Simplified  = F_m at lambda = 0 (circular surfaces)
- Janbu's Simplified   = F_f at lambda = 0
- Spencer              = crossing point with f(x) = constant
- Morgenstern-Price    = crossing point with general f(x)

All units SI (or any consistent unit system — the formulation is
dimensionless except for inputs already embedded in the slices).

References
----------
Fredlund, D.G. & Krahn, J. (1977). "Comparison of slope stability methods
    of analysis." Canadian Geotechnical Journal, 14(3), 429-439.
Morgenstern, N.R. & Price, V.E. (1965). Geotechnique, 15(1), 79-93.
Spencer, E. (1967). Geotechnique, 17(1), 11-26.
GEO-SLOPE / Seequent (2022). "Stability Modeling with GeoStudio /
    SLOPE/W", Chapter 2 (GLE formulation) and Chapter 3 (method details).
Duncan, Wright & Brandon (2014). Soil Strength and Slope Stability, Ch. 6-7.

Provenance (VERIFIED vs the ORIGINAL, 2026-07-18 wiki-verification): the
X = lambda*f(x)*E closure (M-P 1965, p. 85 Eq. 18) and the strength-reduction
FOS definition (pp. 82-83 Eq. 5) were verified against the printed paper in the
owner's library; this engine solves the identical statics in the Fredlund-Krahn
discrete-slice form (EQUIVALENT-NOTATION — same assumptions, FOS definition,
and boundary conditions as M-P's differential formulation). Ledger:
module_work/wiki_verification/le_method_originals.md.
"""

import contextlib
import contextvars
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from slope_stability.slices import Slice
from slope_stability.solution import NoSolution

_FOS_MAX = 999.9


# ---------------------------------------------------------------------------
# Friction on a base in tension (GM-140, the GLE solver; GM-344; D29)
# ---------------------------------------------------------------------------

#: What the solver behind Spencer's method, Morgenstern-Price, GLE and
#: Janbu's simplified method does with a slice whose effective normal force
#: N' = N - u l comes out negative (a base in tension).
#: "as_printed" (the default): the equations as printed, on every slice.
#: Spencer (1967, pp. 14 to 15) and Morgenstern and Price (1965, p. 85)
#: write the mobilised shear as (c' l + N' tan(phi')) / F with no rule for a
#: negative N'; EM 1110-2-1902 C-10a (pp. C-33 to C-34) and UFC 3-220-10
#: pp. 361 to 362 ask that such a base be examined, and the result names
#: the slices (`acceptability`, ``base_tension``).
#: "no_friction": a slice whose base is in tension is taken frictionless
#: (tan(phi') = 0 on that slice) in EVERY equation: its normal force (the
#: m_alpha of vertical equilibrium), the march of the interslice forces,
#: horizontal equilibrium and the moment sum. The module's own device, not
#: found in print. Before this step it was the only behaviour and was
#: applied in the horizontal and moment terms only, so the solver's
#: equations were neither the printed ones nor the device's.
BASE_TENSION_RULES = ("as_printed", "no_friction")

_BASE_TENSION: contextvars.ContextVar = contextvars.ContextVar(
    "slope_base_in_tension", default="as_printed")


def base_tension_in_force() -> str:
    """The rule of `BASE_TENSION_RULES` in force in this context."""
    return _BASE_TENSION.get()


def check_base_tension(rule) -> str:
    if rule not in BASE_TENSION_RULES:
        raise ValueError(
            f"base_in_tension must be one of {BASE_TENSION_RULES}, "
            f"not {rule!r}")
    return rule


@contextlib.contextmanager
def base_tension(rule: str):
    """Run a block with ``rule`` (`BASE_TENSION_RULES`) in force for every
    Spencer, Morgenstern-Price, GLE and Janbu solve made inside it."""
    token = _BASE_TENSION.set(check_base_tension(rule))
    try:
        yield
    finally:
        _BASE_TENSION.reset(token)


#: A lambda bracket narrower than this that still holds a sign change of
#: the moment residual without the two factors of safety agreeing is a
#: jump between two branches of solutions, not a crossing (floating-point
#: resolution of lambda, not a setting of the method).
_LAMBDA_CLOSED = 1e-12
#: The march closes (E at the far end is zero) to this fraction of the
#: forces on the mass; a sign change of the closing force across a pole of
#: the slice equations is far above it and is not a state.
_CLOSES = 1e-7


# ---------------------------------------------------------------------------
# Interslice force functions f(x), x normalized to [0, 1] across the surface
# ---------------------------------------------------------------------------

def _f_constant(x):
    """Constant f(x) = 1 (Spencer's assumption)."""
    return 1.0


def _f_half_sine(x):
    """Half-sine f(x) = sin(pi*x) (SLOPE/W default for M-P)."""
    return math.sin(math.pi * x)


def _f_clipped_sine(x, clip=0.5):
    """Half-sine clipped so the ends keep f >= clip (SLOPE/W style).

    f(x) = max(sin(pi*x), clip). Keeps interslice shear active near the
    crest/toe instead of dropping to zero.
    """
    return max(math.sin(math.pi * x), clip)


def _f_trapezoidal(x):
    """Trapezoid: ramp up over first quarter, flat, ramp down last quarter."""
    if x < 0.25:
        return 4.0 * x
    if x > 0.75:
        return 4.0 * (1.0 - x)
    return 1.0


INTERSLICE_FUNCTIONS = {
    "constant": _f_constant,
    "half_sine": _f_half_sine,
    "clipped_sine": _f_clipped_sine,
    "trapezoidal": _f_trapezoidal,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GLEResult:
    """Result of a rigorous GLE solution.

    Attributes
    ----------
    fos : float
        Factor of safety at the F_m = F_f crossing.
    lam : float
        Interslice force scaling lambda (X = lam * f(x) * E).
    fos_moment : float
        Moment-equilibrium FOS at lambda (== fos at convergence).
    fos_force : float
        Force-equilibrium FOS at lambda (== fos at convergence).
    converged : bool
        True if the lambda root find converged.
    f_interslice : str
        Interslice function name used.
    interslice_E : list of float
        Interslice normal force at each slice boundary (n_slices + 1),
        in the original (un-mirrored) slice order, kN/m.
    interslice_X : list of float
        Interslice shear force at each boundary, kN/m.
    boundary_x : list of float
        x-coordinate of each slice boundary (original orientation).
    thrust_elevation : list of float
        Elevation of the line of thrust (point of application of E) at
        each boundary. Equal to the base elevation where E ~ 0.
    base_normal : list of float
        Total normal force N on each slice base, kN/m.
    base_normal_eff : list of float
        Effective normal force N' = N - u*l on each slice base, kN/m.
    shear_mobilized : list of float
        Mobilized shear S_m on each slice base, kN/m.
    bishop_fos : float
        F_m at lambda=0 (Bishop's simplified for circular surfaces).
    janbu_fos : float
        F_f at lambda=0 (Janbu's simplified, uncorrected).
    warnings : list of str
        Numerical warnings (m_alpha clamps, base tension, fallbacks).
    base_tension_rule : str
        The rule of `BASE_TENSION_RULES` the state was solved under.
    frictionless : list of bool
        Per slice: taken frictionless because its base is in tension
        (only under "no_friction").
    jumps : list of float
        Each lambda at which F_m - F_f changed sign by a jump between two
        branches of solutions (not a crossing; the scan went on).
    """
    fos: float = 0.0
    lam: float = 0.0
    fos_moment: float = 0.0
    fos_force: float = 0.0
    converged: bool = False
    f_interslice: str = "half_sine"
    interslice_E: List[float] = field(default_factory=list)
    interslice_X: List[float] = field(default_factory=list)
    boundary_x: List[float] = field(default_factory=list)
    thrust_elevation: List[float] = field(default_factory=list)
    base_normal: List[float] = field(default_factory=list)
    base_normal_eff: List[float] = field(default_factory=list)
    shear_mobilized: List[float] = field(default_factory=list)
    bishop_fos: float = 0.0
    janbu_fos: float = 0.0
    warnings: List[str] = field(default_factory=list)
    base_tension_rule: str = "as_printed"
    frictionless: List[bool] = field(default_factory=list)
    jumps: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal normalized slice representation
# ---------------------------------------------------------------------------

class _NSlice:
    """Slice data in the normalized frame (driving direction positive).

    The GLE marching assumes one consistent sign convention:
    sum(W * sin(alpha)) > 0, i.e. the high side of the slope on the
    right and sliding down-leftward. Geometries failing the other way
    are mirrored in x (FOS and lambda are invariant under mirroring).
    """
    __slots__ = ("x_mid", "x_left", "x_right", "z_base", "z_centroid",
                 "alpha", "width", "length", "W", "u", "c", "tan_phi",
                 "kW", "Fc", "z_crack", "rh", "rv_down", "Ph", "z_pond",
                 "rh_p", "rv_p")

    def __init__(self, s: Slice, mirror: bool):
        sgn = -1.0 if mirror else 1.0
        self.x_mid = sgn * s.x_mid
        self.x_left = sgn * (s.x_right if mirror else s.x_left)
        self.x_right = sgn * (s.x_left if mirror else s.x_right)
        self.z_base = s.z_base
        self.z_centroid = s.z_centroid
        self.alpha = sgn * s.alpha
        self.width = s.width
        self.length = s.base_length
        self.W = s.weight + s.surcharge_force
        self.u = s.pore_pressure
        self.c = s.c
        self.tan_phi = math.tan(math.radians(s.phi))
        self.kW = s.seismic_force          # magnitude, always driving
        self.Fc = s.crack_water_force      # magnitude, always driving
        self.z_crack = s.crack_water_z
        self.Ph = sgn * s.pond_hforce      # signed pond thrust (+x in frame)
        self.z_pond = s.pond_z
        self.rh = 0.0       # reinforcement horizontal force (+ = resisting)
        self.rv_down = 0.0  # reinforcement vertical force (down positive)
        # PASSIVE supports (Method B): the same two components, applied
        # divided by the factor of safety (rh_p / F, rv_p / F).
        self.rh_p = 0.0
        self.rv_p = 0.0


def _normalize(slices: List[Slice]) -> Tuple[List["_NSlice"], bool]:
    """Mirror the problem in x if it slides right-to-left."""
    drive = sum((s.weight + s.surcharge_force) * math.sin(s.alpha)
                for s in slices)
    mirror = drive < 0
    ns = [_NSlice(s, mirror) for s in slices]
    if mirror:
        ns.reverse()
    return ns, mirror


def _fit_axis_point(slices: List["_NSlice"]) -> Tuple[float, float]:
    """Least-squares circle fit (Kasa method) through the base midpoints.

    Used as the moment axis for noncircular surfaces. The exact choice is
    not critical because both moment and force equilibrium are enforced;
    a fitted centre keeps the N*f arms small and the iteration stable.
    """
    pts = [(s.x_mid, s.z_base) for s in slices]
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    syy = sum(p[1] * p[1] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    sxz = sum(p[0] * (p[0] ** 2 + p[1] ** 2) for p in pts)
    syz = sum(p[1] * (p[0] ** 2 + p[1] ** 2) for p in pts)
    sz = sum(p[0] ** 2 + p[1] ** 2 for p in pts)
    # Solve [sxx sxy sx; sxy syy sy; sx sy n] [a b c]^T = [sxz syz sz]
    A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
    B = [sxz, syz, sz]
    try:
        a, b, c = _solve3(A, B)
        xc, yc = a / 2.0, b / 2.0
        r2 = c + xc * xc + yc * yc
        if r2 <= 0:
            raise ValueError
        # Degenerate (nearly straight) fits put the centre absurdly far
        # away; clamp to a generous height above the surface.
        span = max(s.x_mid for s in slices) - min(s.x_mid for s in slices)
        if yc < max(s.z_base for s in slices) or math.sqrt(r2) > 50.0 * max(span, 1.0):
            raise ValueError
        return (xc, yc)
    except (ValueError, ZeroDivisionError):
        # Fallback: perpendicular offset above the chord midpoint
        x0 = 0.5 * (slices[0].x_mid + slices[-1].x_mid)
        z0 = max(s.z_base for s in slices)
        span = max(slices[-1].x_mid - slices[0].x_mid, 1.0)
        return (x0, z0 + 2.0 * span)


def _solve3(A, B):
    """Solve a 3x3 linear system by Cramer's rule."""
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det3(A)
    if abs(d) < 1e-12:
        raise ValueError("singular")
    out = []
    for col in range(3):
        m = [row[:] for row in A]
        for r in range(3):
            m[r][col] = B[r]
        out.append(det3(m) / d)
    return out


#: Which root is reported where the two equations have more than one in
#: the lambda range (the orchestrator's ruling 2, 2026-11-13; OWN-143).
ROOT_CHOICE = (
    "the first crossing of the moment and force factors of safety met "
    "from lambda = 0 upward (steps of 0.1 to the upper bound, 1.5), on "
    "the branch of solutions on which every slice stays above the pole "
    "of its own two equations, F = tan(phi') tan(theta - alpha), each "
    "state found from its neighbour's F. Spencer (1967, p. 15) finds F "
    "where the two curves of F against theta cross; which crossing, "
    "where there are several, is not found in print. Other form: the "
    "lowest of every root in the range. That needs every root found, "
    "which is not built, and on an ordinary circle the roots include "
    "many with slices below their poles (bases and sides in tension) at "
    "very low F")


# ---------------------------------------------------------------------------
# Scalar root finding (GM-60)
# ---------------------------------------------------------------------------

def _refine_root(fun, a, fa, b, fb, max_iter: int = 200) -> Optional[float]:
    """Root of ``fun`` inside the bracket [a, b] (fa * fb <= 0) by the
    Illinois variant of regula falsi; bisection where ``fun`` has no value."""
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c = b
    for k in range(max_iter):
        c = b - fb * (b - a) / (fb - fa)
        if not (min(a, b) < c < max(a, b)) or k % 3 == 2:
            # (every third step a bisection: beside a pole one end's value
            # is very large and false position alone creeps, GM-344)
            c = 0.5 * (a + b)
        fc = fun(c)
        if fc is None:
            c = 0.5 * (a + b)
            fc = fun(c)
            if fc is None:
                return None
        if fc == 0.0:
            return c
        if fc * fb < 0:
            a, fa = b, fb
        else:
            fa *= 0.5
        b, fb = c, fc
        if abs(b - a) < 1e-12 * max(1.0, abs(b)):
            return b
    return c


def _nearest_root(fun, x0: float, lo: float = 0.02, hi: float = 100.0,
                  factor: float = 1.15, accept=None,
                  cuts=None) -> Optional[float]:
    """The root of ``fun`` (None where it has no value) in [lo, hi] nearest
    ``x0`` on a log scale, or None.

    Steps outward from ``x0`` by ``factor`` on both sides, alternately, and
    refines the first sign change met on each side; ``accept(root)`` can
    reject a root (the search on that side then goes on). Stops as soon as
    a root is found closer than anything left unexplored on the other side.

    ``cuts`` are points where ``fun`` has a pole (GM-344): a step that
    would pass one stops just short of it, and the next starts just beyond
    it, so a root beside a pole is not stepped over with it (two sign
    changes in one step show as none).
    """
    if cuts:
        return _nearest_root_cut(fun, x0, lo, hi, factor, accept, cuts)
    x0 = min(max(x0, lo), hi)
    f0 = fun(x0)
    if f0 == 0.0 and (accept is None or accept(x0)):
        return x0
    sides = {+1: [x0, f0, True], -1: [x0, f0, True]}   # last x, f, live
    found = {}
    while any(s[2] for s in sides.values()):
        for d in (+1, -1):
            st = sides[d]
            if not st[2]:
                continue
            xa, fa = st[0], st[1]
            x = xa * factor if d > 0 else xa / factor
            if x > hi * (1 + 1e-12) or x < lo / (1 + 1e-12):
                st[2] = False
                continue
            fx = fun(x)
            if fa is not None and fx is not None and fa * fx <= 0:
                r = _refine_root(fun, xa, fa, x, fx)
                if r is not None and (accept is None or accept(r)):
                    found[d] = r
                    st[2] = False
            st[0], st[1] = x, fx
        if found:
            # a root on one side: stop once the other side has looked as
            # far out (log scale) as that root
            dist = min(abs(math.log(r / x0)) for r in found.values())
            for d, st in sides.items():
                if st[2] and abs(math.log(st[0] / x0)) >= dist:
                    st[2] = False
    if not found:
        return None
    return min(found.values(), key=lambda r: abs(math.log(r / x0)))


def _nearest_root_cut(fun, x0, lo, hi, factor, accept, cuts
                      ) -> Optional[float]:
    """`_nearest_root` with poles of ``fun`` at ``cuts``."""
    # (the march gives no value where a slice's determinant is below 1e-9)
    eps = 1e-6
    x0 = min(max(x0, lo), hi)

    def points(d):
        """The points looked at on side d of x0, nearest first; a pair
        with None between them straddles a pole."""
        seq = []
        x = x0
        mine = sorted((c for c in cuts
                       if (c > x0 if d > 0 else c < x0) and lo < c < hi),
                      reverse=d < 0)
        k = 0
        while True:
            nxt = x * factor if d > 0 else x / factor
            if k < len(mine) and (mine[k] <= nxt if d > 0
                                  else mine[k] >= nxt):
                c = mine[k]
                k += 1
                near = c * (1 - eps) if d > 0 else c * (1 + eps)
                far = c * (1 + eps) if d > 0 else c * (1 - eps)
                if (near > x if d > 0 else near < x):
                    seq.append(near)
                seq.append(None)
                seq.append(far)
                x = far
                continue
            if nxt > hi * (1 + 1e-12) or nxt < lo / (1 + 1e-12):
                break
            seq.append(nxt)
            x = nxt
        return seq

    f0 = fun(x0)
    if f0 == 0.0 and (accept is None or accept(x0)):
        return x0
    # last x, its f, the points still to look at, live
    sides = {d: [x0, f0, points(d), True] for d in (+1, -1)}
    found = {}
    while any(st[3] for st in sides.values()):
        for d in (+1, -1):
            st = sides[d]
            if not st[3]:
                continue
            if not st[2]:
                st[3] = False
                continue
            x = st[2].pop(0)
            if x is None:               # a pole: start again beyond it
                x = st[2].pop(0)
                st[0], st[1] = x, fun(x)
                continue
            xa, fa = st[0], st[1]
            fx = fun(x)
            if fa is not None and fx is not None and fa * fx <= 0:
                r = _refine_root(fun, xa, fa, x, fx)
                if r is not None and (accept is None or accept(r)):
                    found.setdefault(d, r)
            st[0], st[1] = x, fx
        if found:
            dist = min(abs(math.log(r / x0)) for r in found.values())
            for d, st in sides.items():
                if st[3] and (d in found
                              or abs(math.log(st[0] / x0)) >= dist):
                    st[3] = False
    if not found:
        return None
    return min(found.values(), key=lambda r: abs(math.log(r / x0)))


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

class _GLESystem:
    """Holds geometry-derived constants and solves F_m/F_f for a lambda."""

    def __init__(self, ns: List["_NSlice"], axis: Tuple[float, float],
                 f_name: str, tol: float,
                 reinf: Optional[List[Tuple[float, ...]]] = None):
        # reinf: list of (x_n, z, Tx_n, Tz[, passive]) point forces in the
        # normalized frame (Tx_n positive = resisting, Tz up-positive),
        # already assigned to slices as rh / rv_down (active) or rh_p / rv_p
        # (passive) by the caller. A passive force acts divided by F.
        self.ns = ns
        self.x0, self.y0 = axis
        self.tol = tol
        self.warnings: List[str] = []
        # the rule for a base in tension, fixed for this solve
        self.rule = base_tension_in_force()
        self.device = self.rule == "no_friction"
        f_func = INTERSLICE_FUNCTIONS[f_name]

        n = len(ns)
        # Boundary x positions and f(x) values (n+1 boundaries)
        xb = [ns[0].x_left] + [s.x_right for s in ns]
        x_lo, x_hi = xb[0], xb[-1]
        span = (x_hi - x_lo) if x_hi > x_lo else 1.0
        self.xb = xb
        self.fb = [f_func((x - x_lo) / span) for x in xb]

        # Geometric moment arms about the axis point (cross products)
        self.arm_S = []   # shear  (resisting, positive)
        self.arm_N = []   # normal (near zero on a circle (chord bases))
        for s in ns:
            dx = s.x_mid - self.x0
            dz = s.z_base - self.y0
            self.arm_S.append(dx * math.sin(s.alpha) - dz * math.cos(s.alpha))
            self.arm_N.append(dx * math.cos(s.alpha) + dz * math.sin(s.alpha))

        # External driving moment (independent of N, S, F):
        #   weight:  -W*(x-x0)  [driving = clockwise = negative]
        #   seismic: +kW*(z_c-y0) (z_c < y0 -> negative, driving)
        #   crack:   +Fc*(z_cr-y0)
        # Net external moment M_ext is negative; driving magnitude = -M_ext.
        m_ext = 0.0
        for s in ns:
            m_ext -= s.W * (s.x_mid - self.x0)
            if s.kW:
                m_ext += s.kW * (s.z_centroid - self.y0)
            if s.Fc:
                m_ext += s.Fc * (s.z_crack - self.y0)
            if s.Ph:
                # signed horizontal pond thrust: M_z = -(z - y0) * Fx
                m_ext -= s.Ph * (s.z_pond - self.y0)
        # Reinforcement point forces enter the global moment and force
        # balances exactly. ACTIVE (Method A): unfactored, they reduce the
        # driving side. PASSIVE (Method B): the same moment and horizontal
        # force divided by F, kept apart here (m_p, tx_p) and applied in
        # solve_for_lambda.
        self.m_p = 0.0
        self.tx_p = 0.0
        self.point_loads = bool(reinf)
        tx_active = 0.0
        if reinf:
            for t in reinf:
                rx, rz, tx, tz = t[:4]
                passive = len(t) > 4 and bool(t[4])
                m = (rx - self.x0) * tz - (rz - self.y0) * tx
                if passive:
                    self.m_p += m
                    self.tx_p += tx
                else:
                    m_ext += m
                    tx_active += tx
        self.moment_driving = -m_ext

        # Horizontal external driving forces (for F_f denominator);
        # resisting reinforcement components subtract.
        # the size of the forces on the mass, for "zero to rounding"
        self.force_scale = max(
            sum(abs(s.W) + abs(s.kW) + abs(s.Fc) + abs(s.Ph) + abs(s.rh)
                + abs(s.rh_p) + s.c * s.length for s in ns), 1.0)
        # ... and of their moments about a point of the mass itself (the
        # axis can lie far off; a couple does not depend on it)
        self.moment_scale = self.force_scale * max(
            math.hypot(x_hi - x_lo,
                       max(s.z_base for s in ns) - min(s.z_base for s in ns)),
            1.0)
        self.force_driving_ext = (sum(s.kW for s in ns)
                                  + sum(s.Fc for s in ns)
                                  - sum(s.Ph for s in ns)
                                  - tx_active)

    # -- N for one slice ----------------------------------------------------

    def _m_alpha(self, s: "_NSlice", F: float, tp: float) -> float:
        m_alpha = math.cos(s.alpha) + math.sin(s.alpha) * tp / F
        if abs(m_alpha) < 0.05:
            m_alpha = math.copysign(0.05, m_alpha)
            w = "m_alpha clamped (|m_a| < 0.05) on at least one slice"
            if w not in self.warnings:      # (said once; it used to repeat)
                self.warnings.append(w)
        return m_alpha

    def _normal_eff(self, s: "_NSlice", F: float, dV: float
                    ) -> Tuple[float, float]:
        """(N, tan(phi') in force) of one slice from vertical equilibrium.

        N = [W - dV - (c*l - u*l*tan_phi) * sin(a)/F] / m_alpha
        where dV = V_left - V_right (interslice shear difference).

        As printed, on every slice. Under "no_friction"
        (`BASE_TENSION_RULES`) a slice whose N - u l is negative is solved
        again with tan(phi') = 0, and that zero is returned for the
        slice's other equations."""
        tp = s.tan_phi
        base = s.W + s.rv_down - dV
        if s.rv_p:
            base += s.rv_p / F
        num = base - (s.c * s.length - s.u * s.length * tp) \
            * math.sin(s.alpha) / F
        N = num / self._m_alpha(s, F, tp)
        if self.device and tp > 0.0 and N - s.u * s.length < 0.0:
            tp = 0.0
            N = (base - s.c * s.length * math.sin(s.alpha) / F) \
                / self._m_alpha(s, F, 0.0)
        return N, tp

    def _normal(self, s: "_NSlice", F: float, dV: float) -> float:
        """Base normal from vertical slice equilibrium (`_normal_eff`)."""
        return self._normal_eff(s, F, dV)[0]

    # -- the two factor-of-safety equations for a fixed interslice state ---

    def _moment_terms(self, F: float, V: List[float]):
        """(resisting, driving) of moment equilibrium at trial F."""
        resist_m = 0.0
        normal_m_term = 0.0
        for i, s in enumerate(self.ns):
            N, tp = self._normal_eff(s, F, V[i] - V[i + 1])
            resist_m += self.arm_S[i] * (
                s.c * s.length + (N - s.u * s.length) * tp)
            normal_m_term += N * self.arm_N[i]
        den_m = self.moment_driving - normal_m_term
        if self.m_p:
            den_m -= self.m_p / F
        return resist_m, den_m

    def _force_terms(self, F: float, V: List[float]):
        """(resisting, driving, base normals, tan(phi') in force on each
        slice) of horizontal force equilibrium at trial F."""
        resist_f = 0.0
        den_f = self.force_driving_ext
        if self.tx_p:
            den_f -= self.tx_p / F
        N_f = []
        tps = []
        for i, s in enumerate(self.ns):
            N, tp = self._normal_eff(s, F, V[i] - V[i + 1])
            N_f.append(N)
            tps.append(tp)
            resist_f += (s.c * s.length
                         + (N - s.u * s.length) * tp) \
                * math.cos(s.alpha)
            den_f += N * math.sin(s.alpha)
        return resist_f, den_f, N_f, tps

    @staticmethod
    def _root_of(terms, F_near: float) -> Optional[float]:
        """The factor of safety F > 0 solving F = resisting(F) / driving(F)
        with driving(F) > 0, nearest ``F_near``, or None.

        Used when a fixed-point step meets a driving term <= 0 at the
        current estimate. With a large support force (an anchor) the driving
        term is negative at a small trial F and positive at the true one:
        the base normals, and so the horizontal thrust they give, grow
        with F (the c*l*sin(alpha)/F term of the normal-force equation), so
        the step from a low start cannot be taken although the equation has
        a solution (GM-60). A bracketed root of g(F) = F*driving - resisting
        over 0.02 <= F <= 100 finds it."""
        def g(F):
            res, den = terms(F)[:2]
            return F * den - res

        return _nearest_root(g, F_near,
                             accept=lambda F: terms(F)[1] > 1e-9)

    # -- F_m and F_f for given interslice shear state ------------------------

    def solve_for_lambda(self, lam: float,
                         f0: float = 1.3,
                         max_iter: int = 80):
        """Iterate {N, E, V, F_m, F_f} to convergence for fixed lambda.

        Returns (F_m, F_f, E, V, N_list_force) or None on failure.

        With support forces crossing the surface (concentrated loads at the
        slice bases) and lambda != 0, the state is solved by the direct
        march instead (``_solve_marched``): the lagged iteration below swings
        between two states there and its last state is not an equilibrium
        state (GM-60). With support forces it is also the fallback whenever
        the lagged iteration cannot take a step. With or without them, an
        iteration that runs ``max_iter`` times without settling is solved
        by the direct march, and where that finds nothing there is no
        state (GM-140 step 2).
        """
        if self.point_loads and lam != 0.0:
            return self._solve_marched(lam, f0)
        ns = self.ns
        n = len(ns)
        E = [0.0] * (n + 1)
        V = [0.0] * (n + 1)
        Fm = Ff = max(f0, 0.2)

        for it in range(max_iter):
            # --- moment equilibrium with current V state
            resist_m, den_m = self._moment_terms(Fm, V)
            if den_m <= 1e-9:
                # No fixed-point step from this estimate. With support
                # forces, solve the equations by the direct march instead
                # (GM-60); without them, as before, no state.
                return (self._solve_marched(lam, f0) if self.point_loads
                        else None)
            Fm_new = resist_m / den_m

            # --- force equilibrium with current V state
            resist_f, den_f, N_f, tps = self._force_terms(Ff, V)
            if den_f <= 1e-9:
                return (self._solve_marched(lam, f0) if self.point_loads
                        else None)
            Ff_new = resist_f / den_f

            # --- march interslice normals E with force-equilibrium state
            for i, s in enumerate(ns):
                Sm = (s.c * s.length
                      + (N_f[i] - s.u * s.length) * tps[i]) / Ff_new
                E[i + 1] = (E[i] - N_f[i] * math.sin(s.alpha)
                            + Sm * math.cos(s.alpha) - s.kW - s.Fc + s.rh
                            + s.Ph)
                if s.rh_p:
                    E[i + 1] += s.rh_p / Ff_new
            # E at the far boundary closes to ~0 when Ff is exact; do not
            # force it — the ratio form of Ff enforces global closure.

            # --- interslice shear from X = lam * f * E (under-relaxed)
            for j in range(n + 1):
                V[j] = 0.7 * V[j] + 0.3 * (lam * self.fb[j] * E[j])
            # ends carry no interslice force
            V[0] = 0.0
            V[n] = 0.0
            E[0] = 0.0

            if (abs(Fm_new - Fm) < self.tol and abs(Ff_new - Ff) < self.tol
                    and it > 2):
                return (Fm_new, Ff_new, E, V, N_f)
            Fm, Ff = Fm_new, Ff_new

        # The lagged iteration did not settle (with a large point load it
        # can swing between two states, GM-60): solve by the direct march,
        # with or without support forces. Failing that there is no state at
        # this lambda. The last iterate is not an equilibrium state, and a
        # factor of safety is a method's own converged solution or no
        # number (GM-140 step 2; UFC 3-220-10 p. 362); before, it was
        # returned as a state and nothing flagged it.
        return self._solve_marched(lam, f0)

    # -- direct solution for a fixed lambda (GM-60) ---------------------------

    def _march(self, lam: float, F: float):
        """March the slices once at trial F with X = lam * f(x) * E applied
        in each slice's own equations (no lag between E and X).

        For slice i, with E_i and X_i known and X_{i+1} = lam f_{i+1}
        E_{i+1}, vertical equilibrium (the base-normal equation of
        ``_normal``) and horizontal equilibrium (the E march) are two linear
        equations in N_i and E_{i+1}, solved directly. This is the
        Morgenstern-Price recursion (Morgenstern & Price 1965; the closed
        form of Zhu, Lee, Qian & Chen 2005); it needs no under-relaxation,
        so a large point load cannot set the iteration swinging.

        Returns (E, V, N, tan(phi') in force on each slice, the sign of
        each slice's determinant) or None when a slice's equations are
        singular. As printed, on every slice; under
        "no_friction" (`BASE_TENSION_RULES`) a slice whose N - u l comes
        out negative is solved again with tan(phi') = 0 in both of its
        equations.
        """
        ns = self.ns
        n = len(ns)
        E = [0.0] * (n + 1)
        V = [0.0] * (n + 1)
        N_list = []
        tps = []
        side = []           # per slice: below its pole (`_poles`)
        for i, s in enumerate(ns):
            sa, ca = math.sin(s.alpha), math.cos(s.alpha)
            lf = lam * self.fb[i + 1] if i + 1 < n else 0.0
            rv = s.rv_down + (s.rv_p / F if s.rv_p else 0.0)
            rh = s.rh + (s.rh_p / F if s.rh_p else 0.0)
            ext = -s.kW - s.Fc + rh + s.Ph
            tp = s.tan_phi
            N = None
            below = None        # the slice's own (frictional) determinant < 0
            while True:
                K = s.c * s.length - s.u * s.length * tp
                m_alpha = ca + sa * tp / F
                if abs(m_alpha) < 0.05:
                    m_alpha = math.copysign(0.05, m_alpha)
                # vertical: N*m_alpha - lf*E' = W + rv - X_i - K sin(a)/F
                r0 = s.W + rv - V[i] - K * sa / F
                # horizontal: E' = a*N + b0
                a = -sa + tp * ca / F
                b0 = E[i] + K * ca / F + ext
                det = m_alpha - lf * a
                if abs(det) < 1e-9:
                    return None
                if below is None:
                    below = det < 0.0
                N = (r0 + lf * b0) / det
                if (self.device and tp > 0.0
                        and N - s.u * s.length < 0.0):
                    tp = 0.0            # frictionless, in both equations
                    continue
                break
            E[i + 1] = a * N + b0
            V[i + 1] = lf * E[i + 1]
            N_list.append(N)
            tps.append(tp)
            side.append(below)
        V[n] = 0.0
        return E, V, N_list, tps, tuple(side)

    # -- one state at a fixed lambda (GM-344) --------------------------------

    def _poles(self, lam: float) -> List[float]:
        """The factors of safety F > 0 at which a slice's two equations
        are singular at ``lam`` (ascending).

        For slice i the determinant of `_march` is
        (cos a + lf sin a) + tan(phi') (sin a - lf cos a) / F with
        lf = lam f(x) on its upslope side: zero at
        F = tan(phi') tan(theta_i - a), theta_i = atan(lf). The closing
        force changes sign there without being zero, and a root can lie
        close beside it, so the search for a state looks on each side of
        every pole."""
        ns = self.ns
        n = len(ns)
        out = []
        for i, s in enumerate(ns):
            if s.tan_phi <= 0.0:
                continue
            sa, ca = math.sin(s.alpha), math.cos(s.alpha)
            lf = lam * self.fb[i + 1] if i + 1 < n else 0.0
            A = ca + lf * sa
            B = s.tan_phi * (sa - lf * ca)
            if A != 0.0 and -B / A > 0.0:
                out.append(-B / A)
        out.sort()
        return out

    def _state(self, lam: float, F_near: float, side=None):
        """The equilibrium state at ``lam`` whose factor of safety is
        nearest ``F_near``, by the direct march, or None.

        Returns (F_m, F_f, E, V, N, tps, R, side, driven). F_f closes the
        march
        (the interslice force at the far end is zero: horizontal and
        vertical equilibrium of every slice). R is the net moment on the
        mass in THAT state (its own N, E and X), written as resisting
        moment / F_f - driving moment about the axis: with the forces in
        equilibrium it is a couple, the same about any point, and R = 0 is
        moment equilibrium. So a state with R = 0 meets both equations at
        one lambda. F_m = resisting / driving moment of the state (None
        where the driving moment is not positive); F_m = F_f exactly where
        R = 0. ``driven``: the state's net driving moment and net driving
        horizontal force are both positive, as the ratio of resisting to
        driving needs (`first_crossing`).

        One solver for every lambda of the scan, each state found from its
        neighbour's: its F and its ``side`` (for each slice, whether the
        state lies below that slice's pole, `_poles`). The closing force
        is infinite at a pole, so a branch of solutions cannot pass one:
        the state with the neighbour's ``side`` nearest its F is the
        neighbour's own branch, and with ``side`` given no other state is
        returned (None where that branch has no state at ``lam``). Before,
        a scan mixed the lagged iteration and this march, each started
        from a mean of two factors, and passed from one branch to another
        unnoticed (GM-344).

        A sign change of the closing force across a pole (it does not
        close to `_CLOSES` of the forces on the mass) is not a state.
        """
        def residual(F):
            out = self._march(lam, F)
            return None if out is None else out[0][-1]

        def closes(F):
            r = residual(F)
            return r is not None and abs(r) <= _CLOSES * self.force_scale

        def same_side(F):
            return closes(F) and self._march(lam, F)[4] == side

        poles = self._poles(lam)
        x0 = max(F_near, 0.02)
        Ff = _nearest_root(residual, x0, cuts=poles,
                           accept=closes if side is None else same_side)
        if Ff is None:
            return None
        out = self._march(lam, Ff)
        if out is None:
            return None
        E, V, N, tps, side_now = out
        resist_m = 0.0
        normal_m = 0.0
        den_f = self.force_driving_ext
        if self.tx_p:
            den_f -= self.tx_p / Ff
        for i, s in enumerate(self.ns):
            resist_m += self.arm_S[i] * (
                s.c * s.length + (N[i] - s.u * s.length) * tps[i])
            normal_m += N[i] * self.arm_N[i]
            den_f += N[i] * math.sin(s.alpha)
        den_m = self.moment_driving - normal_m
        if self.m_p:
            den_m -= self.m_p / Ff
        R = resist_m / Ff - den_m
        Fm = resist_m / den_m if den_m > 1e-9 else None
        return (Fm, Ff, E, V, N, tps, R, side_now,
                den_m > 1e-9 and den_f > 1e-9)

    # -- the crossing of the two factors of safety (GM-344) ------------------

    def agrees(self, st) -> bool:
        """The state is in moment equilibrium, to rounding (so F_m = F_f
        far inside ``tol``). With a moment axis far from the mass
        F_m - F_f is small at every lambda and cannot tell a root from a
        jump; the net moment can."""
        return (abs(st[6]) <= _CLOSES * self.moment_scale
                and (st[0] is None or abs(st[0] - st[1]) < self.tol))

    def refine(self, lam_a, st_a, lam_b, st_b):
        """The crossing inside a bracket whose net moments differ in
        sign, as (lambda, state), or (lambda of the jump, None).

        Each trial state is found from an end's own F and branch (the end
        the scan came from first), so the bracket closes either on a
        state in moment equilibrium (a crossing) or, in lambda, on the
        end of a branch (a jump: the two ends lie on different branches
        of solutions, and their sign change is not a bracket). Bisection
        and false position by turns."""
        ra, rb = st_a[6], st_b[6]
        for k in range(400):
            if abs(lam_b - lam_a) < _LAMBDA_CLOSED:
                break
            mid = 0.5 * (lam_a + lam_b)
            if k % 2 and rb != ra:
                fp = lam_b - rb * (lam_b - lam_a) / (rb - ra)
                if min(lam_a, lam_b) < fp < max(lam_a, lam_b):
                    mid = fp
            st = self._state(mid, st_a[1], st_a[7])
            if st is None:
                st = self._state(mid, st_b[1], st_b[7])
            if st is None:
                # no state inside the bracket: nothing to close on
                return 0.5 * (lam_a + lam_b), None
            if self.agrees(st):
                return mid, st
            if st[6] * ra > 0:
                lam_a, st_a, ra = mid, st, st[6]
            else:
                lam_b, st_b, rb = mid, st, st[6]
        return 0.5 * (lam_a + lam_b), None

    def first_crossing(self, lam0: float, st0, lo_b: float, hi_b: float):
        """The first crossing met from the state ``st0`` at ``lam0``, on
        its branch of solutions: ((lambda, state) or None, the lambdas of
        the jumps met, the lambdas at which the branch has no state).

        Scans lambda upward from ``lam0`` in steps of 0.1 to ``hi_b``
        (the bound itself is tried too: fifteen steps of 0.1 overshoot 1.5
        by a rounding error). A lambda at which the branch has no state is
        stepped over (GM-60). Lambda below ``lam0`` is not scanned: there
        the interslice shear acts against the sense the sliding gives it,
        and the crossings found there on made-up and benchmark surfaces
        were all states with bases and sides in tension (before, the scan
        went down too, after or before going up; ``lo_b`` is kept in the
        signature and not used).

        A crossing whose state has no net driving moment or no net
        driving force is not taken (``self.undriven`` counts them): a
        factor of safety is resisting over driving, and where the driving
        sum is zero or negative the ratio has no meaning (the rule of
        GM-140 step 2 for every method, `solution.NoSolution` kind
        'no_driving'; the lagged iteration this scan replaces could not
        take a step from such a state)."""
        jumps: List[float] = []
        no_state: List[float] = []
        self.undriven = 0
        if self.agrees(st0):
            if st0[8]:
                return (lam0, st0), jumps, no_state
            self.undriven += 1
        b_tol = 1e-9
        for step in (0.1,):
            lam_try = lam0
            prev_lam, prev = lam0, st0
            for _ in range(40):
                lam_try += step
                if lam_try > hi_b + b_tol or lam_try < lo_b - b_tol:
                    break
                st = self._state(lam_try, prev[1], prev[7])
                if st is None:
                    no_state.append(lam_try)
                    continue
                if self.agrees(st):
                    if st[8]:
                        return (lam_try, st), jumps, no_state
                    self.undriven += 1
                elif prev[6] * st[6] < 0:
                    lam_c, st_c = self.refine(prev_lam, prev, lam_try, st)
                    if st_c is not None and st_c[8]:
                        return (lam_c, st_c), jumps, no_state
                    if st_c is not None:
                        self.undriven += 1
                    else:
                        jumps.append(lam_c)
                prev_lam, prev = lam_try, st
        no_state.sort(key=abs)
        return None, jumps, no_state

    def _solve_marched(self, lam: float, f0: float):
        """(F_m, F_f, E, V, N) for a fixed lambda by the direct march, or
        None when the equations have no solution with F > 0.

        F_f is the F that closes the march (E_n = 0, horizontal
        equilibrium of the whole mass), found by bracketing over F; F_m is
        then the moment-equilibrium F with that interslice state."""
        def residual(F):
            out = self._march(lam, F)
            return None if out is None else out[0][-1]

        Ff = _nearest_root(residual, max(f0, 0.2))
        if Ff is None:
            return None
        out = self._march(lam, Ff)
        if out is None:
            return None
        E, V, N_f = out[:3]

        # moment equilibrium with this interslice state
        Fm = max(f0, 0.2)
        for _ in range(200):
            resist_m, den_m = self._moment_terms(Fm, V)
            if den_m <= 1e-9:
                Fm = self._root_of(lambda F: self._moment_terms(F, V), Fm)
                if Fm is None:
                    return None
                break
            Fm_new = resist_m / den_m
            if abs(Fm_new - Fm) < 0.1 * self.tol:
                Fm = Fm_new
                break
            Fm = Fm_new
        return (Fm, Ff, E, V, N_f)


def _assign_reinforcement(ns, reinf_forces, mirrored):
    """Map reinforcement crossings into the normalized frame and assign
    horizontal/vertical components to the containing slices.

    Each force acts at the base of the slice it crosses (Wright & Duncan
    1991): its horizontal component in that slice's force equilibrium (the
    E march), its vertical component in that slice's vertical equilibrium
    (the base normal, so the shear strength), and its moment exactly. A
    PASSIVE force (``passive=True``, Method B) is applied divided by F.

    Returns the list of (x_n, z, Tx_n, Tz_up, passive) tuples for the
    system-level moment/force balances, or None.
    """
    if not reinf_forces:
        return None
    sgn = -1.0 if mirrored else 1.0
    out = []
    for f in reinf_forces:
        x_n = sgn * f.x
        tx_n = sgn * f.dir_x * f.T
        tz = f.dir_z * f.T  # up-positive
        passive = bool(getattr(f, "passive", False))
        # In the normalized frame the resisting direction is +x; a
        # reinforcement resists by construction, but guard the sign for
        # odd geometries by taking the stabilizing magnitude.
        if tx_n < 0:
            tx_n = abs(tx_n)
        for s in ns:
            if s.x_left <= x_n <= s.x_right:
                if passive:
                    s.rh_p += tx_n
                    s.rv_p += -tz
                else:
                    s.rh += tx_n
                    s.rv_down += -tz
                break
        out.append((x_n, f.z, tx_n, tz, passive))
    return out


def gle_fos(slices: List[Slice],
            slip,
            f_interslice: str = "half_sine",
            tol: float = 1e-5,
            max_iter: int = 80,
            axis_point: Optional[Tuple[float, float]] = None,
            lambda_bounds: Tuple[float, float] = (-1.5, 1.5),
            reinf_forces=None,
            ) -> GLEResult:
    """Rigorous GLE / Morgenstern-Price factor of safety.

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface or PolylineSlipSurface
        Slip surface (moment axis = centre for circular).
    f_interslice : str
        'constant' (Spencer), 'half_sine' (default), 'clipped_sine',
        'trapezoidal'.
    tol : float
        FOS convergence tolerance. Default 1e-5.
    max_iter : int
        Max iterations of the inner {N, E, X, F} loop. Default 80.
    axis_point : (x, y), optional
        Moment axis for noncircular surfaces (e.g. the centre of the
        parent circle of a composite surface). Default: least-squares
        circle fit through the slice base midpoints.
    lambda_bounds : (float, float)
        Range for lambda. Default (-1.5, 1.5). The scan runs from
        lambda = 0 (or the lowest lambda above it that has a state) up
        to the upper bound; the lower bound is not used (GM-344).
    reinf_forces : list of ReinforcementForce, optional
        Resolved reinforcement crossings (see
        slope_stability.reinforcement). Applied as external point forces
        at the base of the slice each crosses: exact moment via cross
        products, horizontal component in that slice's force equilibrium,
        vertical component in its base-normal equation. Active forces
        (Method A) unfactored; passive ones (Method B) divided by F.
        With support forces present a fixed-lambda state is solved by the
        direct march (``_GLESystem._march``), not the lagged iteration.

    Returns
    -------
    GLEResult
    """
    if f_interslice not in INTERSLICE_FUNCTIONS:
        raise ValueError(
            f"Unknown interslice function '{f_interslice}'. "
            f"Choose from {sorted(INTERSLICE_FUNCTIONS)}")
    if len(slices) < 3:
        raise ValueError("GLE requires at least 3 slices")

    ns, mirrored = _normalize(slices)

    is_circular = getattr(slip, "is_circular", True)
    if is_circular:
        axis = ((-slip.xc if mirrored else slip.xc), slip.yc)
    elif axis_point is not None:
        axis = ((-axis_point[0] if mirrored else axis_point[0]),
                axis_point[1])
    else:
        axis = _fit_axis_point(ns)

    reinf_n = _assign_reinforcement(ns, reinf_forces, mirrored)
    sys_ = _GLESystem(ns, axis, f_interslice, tol, reinf=reinf_n)

    result = GLEResult(f_interslice=f_interslice)
    result.warnings = sys_.warnings

    result.base_tension_rule = sys_.rule
    lo_b, hi_b = lambda_bounds

    # Bishop's and Janbu's values (F_m and F_f at lambda = 0), for the
    # record; the scan below does not start from their mean.
    st_l = sys_.solve_for_lambda(0.0, max_iter=max_iter)
    f_start = 1.3
    if st_l is not None:
        result.bishop_fos, result.janbu_fos = st_l[0], st_l[1]
        f_start = st_l[1]

    # A reported factor of safety is a ROOT (GM-344): both equations met
    # to ``tol`` at one lambda in one state. Every state of the scan is
    # `_GLESystem._state`, found from its neighbour's F. A sign change of
    # the moment residual between two states is refined until the two
    # factors agree within ``tol`` (a crossing: that state is the answer,
    # and its own F is reported, not a mean of two) or until the bracket
    # is closed in lambda without that (a jump between two branches of
    # solutions: not a bracket, and the scan goes on). The root reported
    # is the first crossing met from lambda = 0 upward; finding every
    # root is not built (`ROOT_CHOICE`).
    # The scan starts from the lowest lambda above 0 that has a state
    # when lambda = 0 has none, and steps over a lambda with no state
    # (GM-60).
    # The branch followed is the one on which every slice lies above its
    # pole (`_GLESystem._poles`): the side a slice is on for a large F,
    # where its determinant has the sign of cos(alpha - theta). A state
    # with a slice below its pole has that slice's response reversed (the
    # generalised m_alpha negative; Whitman and Bailey 1967, p. 487, on a
    # negative denominator) and is not a start: without this, a surface
    # on which Janbu's simplified method has no solution would start from
    # whichever such state lay nearest and report its nonsense root.
    above = (False,) * len(ns)
    lam0 = 0.0
    st0 = sys_._state(0.0, f_start, above)
    if st0 is None:
        for k in range(1, 16):
            for lam_s in (0.1 * k,):
                if lam_s <= hi_b + 1e-9:
                    st0 = sys_._state(lam_s, f_start, above)
                    if st0 is not None:
                        lam0 = lam_s
                        break
            if st0 is not None:
                break
        if st0 is None:
            result.fos = _FOS_MAX
            result.warnings.append(
                "no equilibrium state at any lambda in range")
            return result
        result.warnings.append(
            f"no equilibrium state at lambda = 0; bracket started from "
            f"lambda = {lam0:+.1f}")

    found, result.jumps, no_state = sys_.first_crossing(
        lam0, st0, lo_b, hi_b)

    if found is None:
        # No crossing in range: no factor of safety (the callers give no
        # number). The starting state is left for the record.
        result.fos = st0[0] if st0[0] is not None else st0[1]
        result.fos_moment = result.fos
        result.fos_force = st0[1]
        result.lam = lam0
        result.converged = False
        w = "no F_m = F_f crossing found in lambda range"
        if no_state:
            w += (f"; the branch of solutions followed from lambda = "
                  f"{lam0:g} has no state at lambda = {no_state[0]:.1f}"
                  + (f" and {len(no_state) - 1} more of the lambdas tried"
                     if len(no_state) > 1 else ""))
        if sys_.undriven:
            w += (f"; {sys_.undriven} crossing(s) met have no net driving "
                  f"moment or force, where the ratio of resisting to "
                  f"driving has no meaning")
        if result.jumps:
            w += ("; F_m - F_f changes sign by a jump between two "
                  "branches of solutions (not a crossing) at lambda = "
                  + ", ".join(f"{j:.4f}" for j in result.jumps))
        result.warnings.append(w)
        return result

    lam_star, st_best = found
    Fm, Ff, E, V, N_f, tps = st_best[:6]
    result.fos = Ff
    result.fos_moment = Fm if Fm is not None else Ff
    result.fos_force = Ff
    result.lam = lam_star
    result.converged = True
    if result.jumps:
        result.warnings.append(
            "F_m - F_f changed sign by a jump between two branches of "
            "solutions (not a crossing) at lambda = "
            + ", ".join(f"{j:.4f}" for j in result.jumps)
            + " before the crossing reported")

    # ---- per-slice / per-boundary reporting (original orientation) --------
    # The state's own forces: nothing is worked out again at a mean F.
    n = len(ns)
    F = result.fos
    base_N, base_Neff, shear_m = [], [], []
    frictionless = []
    for i, s in enumerate(ns):
        N = N_f[i]
        Neff = N - s.u * s.length
        Sm = (s.c * s.length + Neff * tps[i]) / F
        base_N.append(N)
        base_Neff.append(Neff)
        shear_m.append(Sm)
        frictionless.append(tps[i] != s.tan_phi)
        if Neff < 0 and "base tension (N' < 0)" not in result.warnings:
            result.warnings.append("base tension (N' < 0)")
    result.frictionless = (list(reversed(frictionless)) if mirrored
                           else frictionless)

    # Thrust line: moments about each slice base midpoint to locate E
    thrust = [0.0] * (n + 1)
    zb0 = ns[0].z_base
    thrust[0] = zb0
    # E ~ 0 threshold must be RELATIVE to the peak interslice force: near the
    # slip exit E is small-but-nonzero (e.g. 0.01 kN/m vs hundreds mid-slope)
    # and dividing the residual moment by it produced a spurious vertical
    # spike in the line of thrust. Where E is negligible the point of
    # application is undefined and the documented convention (base elevation)
    # applies.
    e_ref = max((abs(e) for e in E), default=0.0)
    e_tol = max(1e-6, 1e-3 * e_ref)
    for i, s in enumerate(ns):
        # Moment balance of slice i about its base midpoint:
        # E_i applied at (zE_i), E_{i+1} at zE_{i+1} (unknown), V at faces.
        # (x_left - x_mid)*V_i - (zE_i - z_base)*E_i
        #   - (x_right - x_mid)*V_{i+1} + (zE_{i+1} - z_base)*E_{i+1}
        #   + seismic + crack = 0
        half_b = 0.5 * s.width
        m = (-half_b * V[i] - (thrust[i] - s.z_base) * E[i]
             - half_b * V[i + 1])
        if s.kW:
            m += s.kW * (s.z_centroid - s.z_base)
        if s.Fc:
            m += s.Fc * (s.z_crack - s.z_base)
        if s.Ph:
            m -= s.Ph * (s.z_pond - s.z_base)
        if abs(E[i + 1]) > e_tol:
            zE = s.z_base - m / E[i + 1]
        else:
            zE = ns[min(i + 1, n - 1)].z_base
        thrust[i + 1] = zE

    if mirrored:
        result.boundary_x = [-x for x in reversed(sys_.xb)]
        result.interslice_E = list(reversed(E))
        result.interslice_X = list(reversed(V))
        result.thrust_elevation = list(reversed(thrust))
        result.base_normal = list(reversed(base_N))
        result.base_normal_eff = list(reversed(base_Neff))
        result.shear_mobilized = list(reversed(shear_m))
    else:
        result.boundary_x = list(sys_.xb)
        result.interslice_E = list(E)
        result.interslice_X = list(V)
        result.thrust_elevation = list(thrust)
        result.base_normal = list(base_N)
        result.base_normal_eff = list(base_Neff)
        result.shear_mobilized = list(shear_m)

    return result


# ---------------------------------------------------------------------------
# Janbu's Simplified method (+ f0 correction factor)
# ---------------------------------------------------------------------------

def _janbu_b1(slices: List[Slice]) -> float:
    """Janbu correction coefficient b1 by soil type along the surface.

    0.69 for c-only soils, 0.31 for phi-only, 0.50 for c-phi soils
    (Janbu 1973; Abramson et al. 2002, Fig. 6.18).
    """
    has_c = any(s.c > 1e-9 for s in slices)
    has_phi = any(s.phi > 1e-9 for s in slices)
    if has_c and has_phi:
        return 0.50
    if has_c:
        return 0.69
    return 0.31


def janbu_f0(slices: List[Slice]) -> float:
    """Janbu correction factor f0 = 1 + b1*[d/L - 1.4*(d/L)^2].

    d = maximum perpendicular distance from the slip surface to the
    entry-exit chord; L = chord length. Accounts for the interslice
    shear forces neglected by the simplified (force-equilibrium-only)
    method; deeper surfaces get a larger correction.
    """
    if len(slices) < 2:
        return 1.0
    # Chord through the surface endpoints (first/last slice bases sit at
    # the ground surface at entry/exit).
    x1, z1 = slices[0].x_left, slices[0].z_base
    x2, z2 = slices[-1].x_right, slices[-1].z_base
    dx, dz = x2 - x1, z2 - z1
    L = math.hypot(dx, dz)
    if L < 1e-9:
        return 1.0
    d_max = 0.0
    for s in slices:
        # perpendicular distance from base midpoint to the chord
        d = abs(dx * (s.z_base - z1) - dz * (s.x_mid - x1)) / L
        d_max = max(d_max, d)
    ratio = d_max / L
    return 1.0 + _janbu_b1(slices) * (ratio - 1.4 * ratio * ratio)


def janbu_fos(slices: List[Slice],
              slip,
              tol: float = 1e-5,
              apply_correction: bool = True,
              reinf_forces=None,
              ) -> Tuple[float, float, float]:
    """Janbu's Simplified method factor of safety.

    Horizontal force equilibrium with zero interslice shear (the GLE
    force-equilibrium FOS at lambda = 0), optionally multiplied by the
    Janbu f0 correction factor. Works for circular and noncircular
    surfaces.

    Parameters
    ----------
    slices : list of Slice
    slip : CircularSlipSurface or PolylineSlipSurface
    tol : float
        Convergence tolerance. Default 1e-5.
    apply_correction : bool
        Reserved for symmetry; both values are always returned.

    Returns
    -------
    (fos_corrected, fos_uncorrected, f0) : tuple of float

    Raises
    ------
    NoSolution
        The force-equilibrium iteration has no state at lambda = 0: there
        is no number (GM-140 step 2).

    References
    ----------
    Janbu (1973); Fredlund & Krahn (1977); Abramson et al. (2002).
    """
    if len(slices) < 3:
        raise ValueError("Janbu requires at least 3 slices")
    ns, mirrored = _normalize(slices)
    is_circular = getattr(slip, "is_circular", True)
    if is_circular:
        axis = ((-slip.xc if mirrored else slip.xc), slip.yc)
    else:
        axis = _fit_axis_point(ns)
    reinf_n = _assign_reinforcement(ns, reinf_forces, mirrored)
    sys_ = _GLESystem(ns, axis, "constant", tol, reinf=reinf_n)
    st = sys_.solve_for_lambda(0.0)
    if st is None or not (0.0 < st[1] < _FOS_MAX):
        # no number (GM-140 step 2): this used to return the 999.9 marker
        # (and, where the iteration did not settle, its last iterate)
        raise NoSolution(
            "horizontal force equilibrium with horizontal interslice "
            "forces has no state: the iteration on F cannot take a step "
            "(no net driving force) or does not settle")
    _, ff, _, _, _ = st
    f0 = janbu_f0(slices)
    return (f0 * ff, ff, f0)
