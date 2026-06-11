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
                 "kW", "Fc", "z_crack")

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
# Core solver
# ---------------------------------------------------------------------------

class _GLESystem:
    """Holds geometry-derived constants and solves F_m/F_f for a lambda."""

    def __init__(self, ns: List["_NSlice"], axis: Tuple[float, float],
                 f_name: str, tol: float):
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
        self.moment_driving = -m_ext

        # Horizontal external driving forces (for F_f denominator)
        self.force_driving_ext = sum(s.kW for s in ns) + sum(s.Fc for s in ns)

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
        num = (s.W - dV
               - (s.c * s.length - s.u * s.length * s.tan_phi)
               * math.sin(s.alpha) / F)
        return num / m_alpha

    # -- F_m and F_f for given interslice shear state ------------------------

    def solve_for_lambda(self, lam: float,
                         f0: float = 1.3,
                         max_iter: int = 80):
        """Iterate {N, E, V, F_m, F_f} to convergence for fixed lambda.

        Returns (F_m, F_f, E, V, N_list_force) or None on failure.
        """
        ns = self.ns
        n = len(ns)
        E = [0.0] * (n + 1)
        V = [0.0] * (n + 1)
        Fm = Ff = max(f0, 0.2)

        for it in range(max_iter):
            # --- moment equilibrium with current V state
            resist_m = 0.0
            normal_m_term = 0.0
            for i, s in enumerate(ns):
                dV = V[i] - V[i + 1]
                N = self._normal(s, Fm, dV)
                resist_m += self.arm_S[i] * (
                    s.c * s.length + max(N - s.u * s.length, 0.0) * s.tan_phi)
                normal_m_term += N * self.arm_N[i]
            den_m = self.moment_driving - normal_m_term
            if den_m <= 1e-9:
                return None
            Fm_new = resist_m / den_m

            # --- force equilibrium with current V state
            resist_f = 0.0
            den_f = self.force_driving_ext
            N_f = []
            for i, s in enumerate(ns):
                dV = V[i] - V[i + 1]
                N = self._normal(s, Ff, dV)
                N_f.append(N)
                resist_f += (s.c * s.length
                             + max(N - s.u * s.length, 0.0) * s.tan_phi) \
                    * math.cos(s.alpha)
                den_f += N * math.sin(s.alpha)
            if den_f <= 1e-9:
                return None
            Ff_new = resist_f / den_f

            # --- march interslice normals E with force-equilibrium state
            for i, s in enumerate(ns):
                Sm = (s.c * s.length
                      + max(N_f[i] - s.u * s.length, 0.0) * s.tan_phi) / Ff_new
                E[i + 1] = (E[i] - N_f[i] * math.sin(s.alpha)
                            + Sm * math.cos(s.alpha) - s.kW - s.Fc)
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

        # Did not fully converge — return last state flagged by caller
        return (Fm, Ff, E, V, N_f)


def gle_fos(slices: List[Slice],
            slip,
            f_interslice: str = "half_sine",
            tol: float = 1e-5,
            max_iter: int = 80,
            axis_point: Optional[Tuple[float, float]] = None,
            lambda_bounds: Tuple[float, float] = (-1.5, 1.5),
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

    sys_ = _GLESystem(ns, axis, f_interslice, tol)

    result = GLEResult(f_interslice=f_interslice)
    result.warnings = sys_.warnings

    # Initial estimate from lambda = 0 (Bishop / Janbu values, free of charge)
    st0 = sys_.solve_for_lambda(0.0)
    if st0 is None:
        result.fos = _FOS_MAX
        return result
    Fm0, Ff0, _, _, _ = st0
    result.bishop_fos = Fm0
    result.janbu_fos = Ff0
    f_est = Fm0

    def g(lam, f_start):
        st = sys_.solve_for_lambda(lam, f0=f_start)
        if st is None:
            return None
        return st

    # Bracket the root of g(lam) = Fm - Ff starting from 0
    lo_b, hi_b = lambda_bounds
    lam_a, ga_state = 0.0, st0
    ga = Fm0 - Ff0
    bracket = None
    step = 0.1 if ga >= 0 else -0.1
    lam_try = 0.0
    prev_lam, prev_g, prev_state = lam_a, ga, ga_state
    for _ in range(40):
        lam_try += step
        if lam_try > hi_b or lam_try < lo_b:
            break
        st = g(lam_try, f_est)
        if st is None:
            break
        gt = st[0] - st[1]
        f_est = 0.5 * (st[0] + st[1])
        if prev_g * gt <= 0:
            bracket = (prev_lam, lam_try, prev_g, gt, st)
            break
        prev_lam, prev_g, prev_state = lam_try, gt, st

    if bracket is None and ga != 0:
        # try the other direction
        step = -step
        lam_try = 0.0
        prev_lam, prev_g = 0.0, ga
        for _ in range(40):
            lam_try += step
            if lam_try > hi_b or lam_try < lo_b:
                break
            st = g(lam_try, f_est)
            if st is None:
                break
            gt = st[0] - st[1]
            f_est = 0.5 * (st[0] + st[1])
            if prev_g * gt <= 0:
                bracket = (prev_lam, lam_try, prev_g, gt, st)
                break
            prev_lam, prev_g = lam_try, gt

    if bracket is None:
        # No crossing found in range — report lambda=0 moment solution,
        # flagged as not converged (caller may fall back to legacy).
        result.fos = Fm0
        result.fos_moment = Fm0
        result.fos_force = Ff0
        result.lam = 0.0
        result.converged = False
        result.warnings.append("no F_m = F_f crossing found in lambda range")
        return result

    a, b, ga_, gb_, st_b = bracket
    # Bisection (robust; g is cheap)
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
    lam_star = 0.5 * (a + b)

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
        if abs(E[i + 1]) > 1e-6:
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
