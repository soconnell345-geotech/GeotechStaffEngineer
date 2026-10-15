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

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from slope_stability.slices import Slice

_FOS_MAX = 999.9


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


# ---------------------------------------------------------------------------
# Scalar root finding (GM-60)
# ---------------------------------------------------------------------------

def _refine_root(fun, a, fa, b, fb, max_iter: int = 100) -> Optional[float]:
    """Root of ``fun`` inside the bracket [a, b] (fa * fb <= 0) by the
    Illinois variant of regula falsi; bisection where ``fun`` has no value."""
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c = b
    for _ in range(max_iter):
        c = b - fb * (b - a) / (fb - fa)
        if not (min(a, b) < c < max(a, b)):
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
                  factor: float = 1.15, accept=None) -> Optional[float]:
    """The root of ``fun`` (None where it has no value) in [lo, hi] nearest
    ``x0`` on a log scale, or None.

    Steps outward from ``x0`` by ``factor`` on both sides, alternately, and
    refines the first sign change met on each side; ``accept(root)`` can
    reject a root (the search on that side then goes on). Stops as soon as
    a root is found closer than anything left unexplored on the other side.
    """
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
        self.arm_N = []   # normal (zero for a true circle about its centre)
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
        self.force_driving_ext = (sum(s.kW for s in ns)
                                  + sum(s.Fc for s in ns)
                                  - sum(s.Ph for s in ns)
                                  - tx_active)

    # -- N for one slice ----------------------------------------------------

    def _normal(self, s: "_NSlice", F: float, dV: float) -> float:
        """Base normal from vertical slice equilibrium.

        N = [W - dV - (c*l - u*l*tan_phi) * sin(a)/F] / m_alpha
        where dV = V_left - V_right (interslice shear difference).
        """
        m_alpha = math.cos(s.alpha) + math.sin(s.alpha) * s.tan_phi / F
        if abs(m_alpha) < 0.05:
            m_alpha = math.copysign(0.05, m_alpha)
            if "m_alpha clamped" not in self.warnings:
                self.warnings.append("m_alpha clamped (|m_a| < 0.05) on at "
                                     "least one slice")
        num = (s.W + s.rv_down - dV
               - (s.c * s.length - s.u * s.length * s.tan_phi)
               * math.sin(s.alpha) / F)
        if s.rv_p:
            num += s.rv_p / F
        return num / m_alpha

    # -- the two factor-of-safety equations for a fixed interslice state ---

    def _moment_terms(self, F: float, V: List[float]):
        """(resisting, driving) of moment equilibrium at trial F."""
        resist_m = 0.0
        normal_m_term = 0.0
        for i, s in enumerate(self.ns):
            N = self._normal(s, F, V[i] - V[i + 1])
            resist_m += self.arm_S[i] * (
                s.c * s.length + max(N - s.u * s.length, 0.0) * s.tan_phi)
            normal_m_term += N * self.arm_N[i]
        den_m = self.moment_driving - normal_m_term
        if self.m_p:
            den_m -= self.m_p / F
        return resist_m, den_m

    def _force_terms(self, F: float, V: List[float]):
        """(resisting, driving, base normals) of horizontal force
        equilibrium at trial F."""
        resist_f = 0.0
        den_f = self.force_driving_ext
        if self.tx_p:
            den_f -= self.tx_p / F
        N_f = []
        for i, s in enumerate(self.ns):
            N = self._normal(s, F, V[i] - V[i + 1])
            N_f.append(N)
            resist_f += (s.c * s.length
                         + max(N - s.u * s.length, 0.0) * s.tan_phi) \
                * math.cos(s.alpha)
            den_f += N * math.sin(s.alpha)
        return resist_f, den_f, N_f

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
        the lagged iteration fails. Without them the iteration is as it was.
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
            resist_f, den_f, N_f = self._force_terms(Ff, V)
            if den_f <= 1e-9:
                return (self._solve_marched(lam, f0) if self.point_loads
                        else None)
            Ff_new = resist_f / den_f

            # --- march interslice normals E with force-equilibrium state
            for i, s in enumerate(ns):
                Sm = (s.c * s.length
                      + max(N_f[i] - s.u * s.length, 0.0) * s.tan_phi) / Ff_new
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
        # can swing between two states, GM-60): with support forces solve by
        # the direct march; failing that, return the last state, flagged by
        # the caller.
        if self.point_loads:
            marched = self._solve_marched(lam, f0)
            if marched is not None:
                return marched
        return (Fm, Ff, E, V, N_f)

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

        Returns (E, V, N) or None when a slice's equations are singular.
        """
        ns = self.ns
        n = len(ns)
        E = [0.0] * (n + 1)
        V = [0.0] * (n + 1)
        N_list = []
        for i, s in enumerate(ns):
            sa, ca = math.sin(s.alpha), math.cos(s.alpha)
            lf = lam * self.fb[i + 1] if i + 1 < n else 0.0
            rv = s.rv_down + (s.rv_p / F if s.rv_p else 0.0)
            rh = s.rh + (s.rh_p / F if s.rh_p else 0.0)
            K = s.c * s.length - s.u * s.length * s.tan_phi
            m_alpha = ca + sa * s.tan_phi / F
            if abs(m_alpha) < 0.05:
                m_alpha = math.copysign(0.05, m_alpha)
            # vertical: N*m_alpha - lf*E' = W + rv - X_i - K sin(a)/F
            r0 = s.W + rv - V[i] - K * sa / F
            ext = -s.kW - s.Fc + rh + s.Ph
            # horizontal: E' = a*N + b0, with N' >= 0 assumed first
            a = -sa + s.tan_phi * ca / F
            b0 = E[i] + K * ca / F + ext
            det = m_alpha - lf * a
            if abs(det) < 1e-9:
                return None
            N = (r0 + lf * b0) / det
            if N - s.u * s.length < 0.0 and s.tan_phi > 0.0:
                # no friction on a base in tension (as the E march does)
                a = -sa
                b0 = E[i] + s.c * s.length * ca / F + ext
                det = m_alpha - lf * a
                if abs(det) < 1e-9:
                    return None
                N = (r0 + lf * b0) / det
            E[i + 1] = a * N + b0
            V[i + 1] = lf * E[i + 1]
            N_list.append(N)
        V[n] = 0.0
        return E, V, N_list

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
        E, V, N_f = out

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
        Bracketing range for lambda. Default (-1.5, 1.5).
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

    # Initial estimate from lambda = 0 (Bishop / Janbu values, free of charge)
    st0 = sys_.solve_for_lambda(0.0)
    lo_b, hi_b = lambda_bounds
    lam0 = 0.0
    # With support forces (concentrated loads at slice bases) the lambda
    # search below is made robust (GM-60); without them it is as it was.
    robust = sys_.point_loads
    if st0 is None and not robust:
        result.fos = _FOS_MAX
        return result
    if st0 is None:
        # No equilibrium state at lambda = 0 (e.g. a large support force
        # leaves no horizontal thrust with horizontal interslice forces).
        # Start the bracket from the nearest lambda that has one instead of
        # giving up (GM-60).
        for k in range(1, 16):
            for lam_s in (0.1 * k, -0.1 * k):
                if lo_b <= lam_s <= hi_b:
                    st0 = sys_.solve_for_lambda(lam_s)
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
    Fm0, Ff0, _, _, _ = st0
    if lam0 == 0.0:
        result.bishop_fos = Fm0
        result.janbu_fos = Ff0
    f_est = Fm0

    def g(lam, f_start):
        st = sys_.solve_for_lambda(lam, f0=f_start)
        if st is None:
            return None
        return st

    # Bracket the root of g(lam) = Fm - Ff starting from lam0. With support
    # forces a lambda with no equilibrium state is stepped over, not taken
    # as the end of the range: with large point loads the states can be
    # broken up (GM-60).
    ga = Fm0 - Ff0
    first = 1.0 if ga >= 0 else -1.0
    conv_tol = max(50 * tol, 1e-3)

    def bisect(bracket):
        """Bisection on a bracket (robust; g is cheap). Returns
        (lambda*, state)."""
        nonlocal f_est
        a, b, ga_, gb_, st_b = bracket
        st_best = st_b
        for _ in range(60):
            mid = 0.5 * (a + b)
            st = g(mid, f_est)
            if st is None:
                break
            gm = st[0] - st[1]
            f_est = 0.5 * (st[0] + st[1])
            st_best = st
            if abs(gm) < tol or (b - a) < 1e-6:
                a = b = mid
                break
            if ga_ * gm <= 0:
                b, gb_ = mid, gm
            else:
                a, ga_ = mid, gm
        return 0.5 * (a + b), st_best

    # Scan from lam0 in the likelier direction first, then the other. With
    # support forces a bracket whose bisection does not close (F_m - F_f
    # jumps instead of crossing, as it can when a large point load gives
    # F_f more than one root) is passed over and the scan goes on (GM-60);
    # the bound itself is tried too (fifteen steps of 0.1 overshoot 1.5 by
    # a rounding error). Without them the first bracket is taken, as before.
    found = None            # (lambda*, state) of a closed crossing
    fallback = None         # the least-bad unclosed one
    b_tol = 1e-9 if robust else 0.0
    for direction in ((first, -first) if ga != 0 else (first,)):
        step = 0.1 * direction
        lam_try = lam0
        prev_lam, prev_g = lam0, ga
        for _ in range(40):
            lam_try += step
            if lam_try > hi_b + b_tol or lam_try < lo_b - b_tol:
                break
            st = g(lam_try, f_est)
            if st is None:
                if robust:
                    continue
                break
            gt = st[0] - st[1]
            f_est = 0.5 * (st[0] + st[1])
            if prev_g * gt <= 0:
                lam_c, st_c = bisect((prev_lam, lam_try, prev_g, gt, st))
                if abs(st_c[0] - st_c[1]) < conv_tol or not robust:
                    found = (lam_c, st_c)
                    break
                if fallback is None or (abs(st_c[0] - st_c[1])
                                        < abs(fallback[1][0]
                                              - fallback[1][1])):
                    fallback = (lam_c, st_c)
            prev_lam, prev_g = lam_try, gt
        if found is not None:
            break

    if found is None and fallback is None:
        # No crossing found in range — report the starting moment solution,
        # flagged as not converged (caller may fall back to legacy).
        result.fos = Fm0
        result.fos_moment = Fm0
        result.fos_force = Ff0
        result.lam = lam0
        result.converged = False
        result.warnings.append("no F_m = F_f crossing found in lambda range")
        return result

    lam_star, st_best = found if found is not None else fallback
    Fm, Ff, E, V, N_f = st_best
    result.fos = 0.5 * (Fm + Ff)
    result.fos_moment = Fm
    result.fos_force = Ff
    result.lam = lam_star
    result.converged = abs(Fm - Ff) < max(50 * tol, 1e-3)
    if not result.converged:
        result.warnings.append(
            f"F_m - F_f residual {abs(Fm - Ff):.2e} at lambda={lam_star:.3f}")

    # ---- per-slice / per-boundary reporting (original orientation) --------
    n = len(ns)
    F = result.fos
    base_N, base_Neff, shear_m = [], [], []
    for i, s in enumerate(ns):
        dV = V[i] - V[i + 1]
        N = sys_._normal(s, F, dV)
        Neff = N - s.u * s.length
        Sm = (s.c * s.length + max(Neff, 0.0) * s.tan_phi) / F
        base_N.append(N)
        base_Neff.append(Neff)
        shear_m.append(Sm)
        if Neff < 0 and "base tension (N' < 0)" not in result.warnings:
            result.warnings.append("base tension (N' < 0)")

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
    if st is None:
        return (_FOS_MAX, _FOS_MAX, 1.0)
    _, ff, _, _, _ = st
    f0 = janbu_f0(slices)
    return (f0 * ff, ff, f0)
