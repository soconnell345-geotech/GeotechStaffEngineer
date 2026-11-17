"""
Limit equilibrium methods for factor of safety computation.

Three methods in order of rigor:
1. Fellenius (OMS) — moment equilibrium only, no iteration
2. Bishop's Simplified — moment + vertical force equilibrium, iterative
3. Spencer — full force + moment equilibrium, two-variable iteration

Spencer supports both circular and noncircular slip surfaces.
Bishop and Fellenius require circular surfaces (moment equilibrium
about circle center).

All units SI: kPa, kN/m, degrees, meters.

References:
    Fellenius (1927)
    Bishop (1955) — Geotechnique, Vol. 5, pp. 7-17
    Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
    Duncan, Wright & Brandon (2014) — Chapters 6-7

Provenance (VERIFIED-PRIMARY, 2026-07-18 wiki-verification): the Bishop
simplified m_alpha/FOS expression (Bishop 1955, p. 10 Eqs. 12/13; the
X_n - X_{n+1} = 0 assumption stated p. 11) and Spencer's Eq. 5 kernel with the
cos(alpha-theta)[1 + (tan phi'/F) tan(alpha-theta)] denominator plus the
F_f = F_m crossing (Spencer 1967, pp. 14-15 Eqs. 5/7/8) were verified
equation-by-equation against the ORIGINAL papers in the owner's library.
Ledger: module_work/wiki_verification/le_method_originals.md.
"""

import contextlib
import contextvars
import math
from typing import List, Tuple, Optional

from slope_stability.slices import Slice
from slope_stability.solution import NoSolution, gle_failure_reason


_FOS_MAX = 999.9  # internal marker; never returned as a factor of safety

_NO_DRIVING = ("the surface has no net driving moment or force (the "
               "driving sum is zero or negative), so the ratio of resisting "
               "to driving has no meaning")
_NO_RESISTANCE = ("the resisting sum is zero or negative at the trial "
                  "factor of safety")


# ---------------------------------------------------------------------------
# A negative frictional term on a slice (GM-140 step 3, D21)
# ---------------------------------------------------------------------------

#: What Bishop's simplified method and the Ordinary Method of Slices do
#: with a slice whose frictional term is negative (u b above W in Bishop's
#: Eq. 13; u l above W cos(alpha) in Eq. 9; Bishop 1955, p. 10).
#: "as_printed" (the default): the term is used as the equation prints it,
#: negative. Bishop (1955) Eq. 13 writes it W(1 - B) = W - u b and Eq. 9
#: W cos(alpha) - u l, with no lower limit; Whitman and Bailey (1967,
#: p. 487) ask that the user be warned when the numerator becomes negative
#: for any slice, and the result names the slices (`acceptability`).
#: "zero": the term is set to zero where negative, as the module did before
#: this step (no printed source).
NEGATIVE_TERM_RULES = ("as_printed", "zero")

_NEGATIVE_TERM: contextvars.ContextVar = contextvars.ContextVar(
    "slope_negative_friction_term", default="as_printed")


def negative_term_in_force() -> str:
    """The rule of `NEGATIVE_TERM_RULES` in force in this context."""
    return _NEGATIVE_TERM.get()


def check_negative_term(rule) -> str:
    if rule not in NEGATIVE_TERM_RULES:
        raise ValueError(
            f"negative_friction_term must be one of {NEGATIVE_TERM_RULES}, "
            f"not {rule!r}")
    return rule


@contextlib.contextmanager
def negative_term(rule: str):
    """Run a block with ``rule`` (`NEGATIVE_TERM_RULES`) in force for every
    Bishop and Ordinary Method of Slices solve made inside it."""
    token = _NEGATIVE_TERM.set(check_negative_term(rule))
    try:
        yield
    finally:
        _NEGATIVE_TERM.reset(token)


def _friction_term(value: float) -> float:
    """A slice's frictional term under the rule in force: as printed, or
    zero where negative."""
    if value < 0.0 and _NEGATIVE_TERM.get() == "zero":
        return 0.0
    return value


# Morgenstern-Price interslice force functions
def _f_constant(x_norm):
    """Constant f(x) = 1 (equivalent to Spencer)."""
    return 1.0


def _f_half_sine(x_norm):
    """Half-sine f(x) = sin(pi * x_norm), x_norm in [0, 1]."""
    return math.sin(math.pi * x_norm)


def _f_trapezoidal(x_norm):
    """Trapezoidal f(x): ramps up in first quarter, flat middle, ramps down last quarter."""
    if x_norm < 0.25:
        return 4.0 * x_norm
    elif x_norm > 0.75:
        return 4.0 * (1.0 - x_norm)
    return 1.0


_MP_FUNCTIONS = {
    "constant": _f_constant,
    "half_sine": _f_half_sine,
    "trapezoidal": _f_trapezoidal,
}


def fellenius_fos(slices: List[Slice],
                  slip,
                  reinf_forces=None) -> float:
    """Ordinary Method of Slices (Fellenius) factor of safety.

    For circular surfaces:
        FOS = sum[c'*dl + (W*cos(alpha) - u*dl)*tan(phi')] /
              sum[W*(x_mid-xc)/R + kh*W*(yc-z_centroid)/R]

    For noncircular surfaces:
        FOS = sum[c'*dl + (W*cos(alpha) - u*dl)*tan(phi')] /
              sum[W*sin(alpha) + kh*W*cos(alpha)]

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface or PolylineSlipSurface
        Slip surface geometry.

    Returns
    -------
    float
        Factor of safety.

    Notes
    -----
    Conservative: typically 10-60% lower than rigorous methods.
    No iteration required (except with passive supports). Used as initial
    guess for Bishop.

    Raises ``NoSolution`` where the surface has no driving moment or force
    (the ratio has no meaning there), or where the iteration with passive
    supports does not settle: there is no number (GM-140 step 2).

    Supports (reinf_forces): where any support crosses the surface the
    method gives no number (``NoSolution``, kind 'with_supports'; GM-140
    step 8). No reference held prints the Ordinary Method of Slices with
    known forces: Duncan, Wright and Brandon (2014, p. 93) give Bishop's
    simplified method with them (Eq. 6.78) and no such equation for this
    method. Until that step the module added the support's normal
    component to the slice's normal force and its moment to the moment
    sum on a circle, and its projection on the base on a polyline: forms
    of its own. The code below the refusal is unreachable with supports
    and is left as it was.
    """
    if reinf_forces:
        from slope_stability.solution import OMS_WITH_SUPPORTS
        raise NoSolution(OMS_WITH_SUPPORTS, kind="with_supports")
    resisting = 0.0
    is_circular = getattr(slip, 'is_circular', True)

    gravity_driving = 0.0
    seismic_driving = 0.0
    crack_water_driving = 0.0

    for s in slices:
        phi_rad = math.radians(s.phi)
        dl = s.base_length
        W = s.weight + s.surcharge_force

        # Resisting: shear strength along base
        # N' = W cos(alpha) - u l as Bishop (1955) Eq. 9, p. 10 prints it:
        # not set to zero where negative unless the "zero" rule is in
        # force (GM-140 step 3, D21; `NEGATIVE_TERM_RULES`).
        N_prime = W * math.cos(s.alpha) - s.pore_pressure * dl
        resisting += s.c * dl + _friction_term(N_prime) * math.tan(phi_rad)

        if is_circular:
            # Circular: moment-arm formulation W*(x_mid - xc)/R
            gravity_driving += W * (s.x_mid - slip.xc) / slip.radius
            if s.pond_hforce != 0:
                # signed moment of the external pond thrust (buttress), in the
                # same clockwise-positive convention as W*(x-xc):
                # M = Fx*(z - yc); combined with gravity BEFORE abs()
                gravity_driving += s.pond_hforce * (s.pond_z - slip.yc) / slip.radius
            if s.seismic_force != 0:
                seismic_driving += s.seismic_force * (slip.yc - s.z_centroid) / slip.radius
            if s.crack_water_force != 0:
                crack_water_driving += s.crack_water_force * (slip.yc - s.crack_water_z) / slip.radius
        else:
            # Noncircular: use W*sin(alpha) directly
            gravity_driving += W * math.sin(s.alpha)
            if s.pond_hforce != 0:
                # +x force drives sliding in -alpha sense: signed -Fx*cos(a)
                gravity_driving += -s.pond_hforce * math.cos(s.alpha)
            if s.seismic_force != 0:
                seismic_driving += s.seismic_force * math.cos(s.alpha)
            if s.crack_water_force != 0:
                crack_water_driving += s.crack_water_force * math.cos(s.alpha)

    # Driving = gravity magnitude + seismic magnitude + crack water.
    # Separated so seismic always increases driving regardless of slope direction.
    driving = abs(gravity_driving) + abs(seismic_driving) + abs(crack_water_driving)

    passive_press = []   # (N' without them, tan phi, pressing) per slice
    if reinf_forces:
        from slope_stability.reinforcement import (
            moment_reduction, horizontal_reduction,
            moment_resistance, horizontal_resistance, slice_support_forces,
        )
        if is_circular:
            # signed moments, positive where they resist the sliding sense
            d_sign = 1.0 if gravity_driving >= 0 else -1.0
            driving -= moment_reduction(reinf_forces, slip.xc, slip.yc,
                                        slip.radius, d_sign=d_sign)
            resisting += moment_resistance(reinf_forces, slip.xc, slip.yc,
                                           slip.radius, d_sign=d_sign)
            # Each support's component pressing the base of the slice it
            # crosses (Fx sin(a) - Fz cos(a)) adds to that slice's normal
            # force: unfactored (active), or / F (passive, below).
            active_press = {}
            for passive in (False, True):
                for i, (fx, fz) in slice_support_forces(
                        reinf_forces, slices, passive=passive,
                        d_sign=d_sign).items():
                    s = slices[i]
                    tan_phi = math.tan(math.radians(s.phi))
                    if tan_phi <= 0:
                        continue
                    press = fx * math.sin(s.alpha) - fz * math.cos(s.alpha)
                    n0 = ((s.weight + s.surcharge_force) * math.cos(s.alpha)
                          - s.pore_pressure * s.base_length)
                    if passive:
                        passive_press.append(
                            (n0 + active_press.get(i, 0.0), tan_phi, press))
                    else:
                        active_press[i] = press
                        resisting += (_friction_term(n0 + press)
                                      - _friction_term(n0)) * tan_phi
        else:
            driving -= horizontal_reduction(reinf_forces, slices)
            resisting += horizontal_resistance(reinf_forces, slices)

    if driving <= 0:
        # no number (GM-140 step 2): this used to return the 999.9 marker
        raise NoSolution(_NO_DRIVING, kind="no_driving")

    if not passive_press:
        return resisting / driving
    # Passive supports press their slice bases with T / F: iterate on F.
    fos = resisting / driving
    if fos <= 0:
        raise NoSolution(_NO_RESISTANCE)
    for _ in range(100):
        extra = sum((_friction_term(n0 + p / fos) - _friction_term(n0))
                    * tp for n0, tp, p in passive_press)
        fos_new = (resisting + extra) / driving
        if fos_new <= 0:
            raise NoSolution(_NO_RESISTANCE)
        if abs(fos_new - fos) < 1e-9:
            return fos_new
        fos = fos_new
    raise NoSolution(
        "the iteration on F with the passive supports' forces divided by F "
        f"did not settle in 100 iterations (last values {fos:.4f})")


def bishop_fos(slices: List[Slice],
               slip,
               tol: float = 1e-4,
               max_iter: int = 50,
               fos_initial: Optional[float] = None,
               reinf_forces=None) -> float:
    """Bishop's Simplified Method factor of safety.

    Requires circular slip surfaces only (moment equilibrium about
    circle center).

    FOS = sum[(c'*b + (W - u*b)*tan(phi')) / m_alpha] /
          sum[W*sin(alpha)]

    where m_alpha = cos(alpha) + sin(alpha)*tan(phi')/FOS.
    The effective-weight term (W - u*b) is used as Bishop (1955) Eq. 13,
    p. 10 prints it, W(1 - B) with no lower limit: where u*b exceeds W
    (artesian or perched pore pressure) the slice's frictional term is
    negative (GM-140 step 3, D21). The result names such slices, as
    Whitman and Bailey (1967, p. 487) ask (`acceptability`). With the
    "zero" rule in force (`NEGATIVE_TERM_RULES`, `negative_term`) the term
    is set to zero where negative, as before this step.

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface
        Circle geometry. Must be circular.
    tol : float
        Convergence tolerance on FOS. Default 1e-4.
    max_iter : int
        Maximum iterations. Default 50.
    fos_initial : float, optional
        Initial FOS guess. If None, uses Fellenius result.

    Returns
    -------
    float
        Factor of safety.

    Raises
    ------
    ValueError
        If slip surface is not circular.
    NoSolution
        The surface has no driving moment, or the iteration is not within
        ``tol`` after ``max_iter`` iterations: there is no number (GM-140
        step 2, D27).

    Notes
    -----
    - Satisfies moment equilibrium and vertical force equilibrium
    - Most commonly used method in practice
    - Typically converges in 5-10 iterations

    Supports (``reinf_forces``) act at the base of the slice they cross
    (Wright & Duncan 1991): the signed moment about the centre (only the
    support's component along the slip surface has one) and the VERTICAL
    component in that slice's vertical equilibrium, W -> W - Fz in the
    numerator, so it changes the base normal force and, where phi > 0, the
    shear strength. ACTIVE supports (Method A) do so unfactored and their
    moment reduces the driving moment; PASSIVE ones (Method B) act as T/F:
    their moment is added to the resisting side and their vertical component
    is divided by F. The horizontal component enters no equation Bishop's
    method satisfies except the moment; a support crossing close to the
    level of the centre therefore counts for little (see
    ``slope_stability.reinforcement``).
    """
    if not getattr(slip, 'is_circular', True):
        raise ValueError(
            "Bishop's method requires a circular slip surface. "
            "Use Spencer's method for noncircular surfaces."
        )

    if fos_initial is None:
        try:
            fos_initial = fellenius_fos(slices, slip)
        except NoSolution:
            fos_initial = 1.5

    # Precompute driving sum (denominator — independent of FOS)
    # Use moment-arm formulation: W*(x_mid - xc)/R for numerical stability
    gravity_driving = 0.0
    seismic_driving = 0.0
    crack_water_driving = 0.0
    for s in slices:
        W = s.weight + s.surcharge_force
        gravity_driving += W * (s.x_mid - slip.xc) / slip.radius
        if s.pond_hforce != 0:
            gravity_driving += s.pond_hforce * (s.pond_z - slip.yc) / slip.radius
        if s.seismic_force != 0:
            seismic_driving += s.seismic_force * (slip.yc - s.z_centroid) / slip.radius
        if s.crack_water_force != 0:
            crack_water_driving += s.crack_water_force * (slip.yc - s.crack_water_z) / slip.radius

    # Separated so seismic always increases driving regardless of slope direction
    driving = abs(gravity_driving) + abs(seismic_driving) + abs(crack_water_driving)

    # Slope-direction sign (SS-4): m_alpha must be evaluated with the base
    # angle measured positive in the sliding direction. For crest-on-the-
    # left slopes (sliding left-to-right) the gravity moment sum is
    # negative and alpha must be mirrored, otherwise the sin(alpha)*tan(phi)
    # term has the wrong sign and Bishop overestimates FOS (found via the
    # Fredlund & Krahn 1977 benchmark).
    d_sign = 1.0 if gravity_driving >= 0 else -1.0

    passive_resistance = 0.0
    fz_a, fz_p = {}, {}
    if reinf_forces:
        from slope_stability.reinforcement import (
            moment_reduction, moment_resistance, slice_support_forces,
        )
        # Signed moments about the centre, positive where they resist.
        driving -= moment_reduction(reinf_forces, slip.xc, slip.yc,
                                    slip.radius, d_sign=d_sign)
        # PASSIVE-convention forces add to the resisting side (Method B),
        # not the driving side. Default (active) = 0.
        passive_resistance = moment_resistance(reinf_forces, slip.xc, slip.yc,
                                               slip.radius, d_sign=d_sign)
        # Vertical components, in the vertical equilibrium of the slice
        # each support crosses (Wright & Duncan 1991).
        fz_a = {i: fz for i, (_fx, fz) in slice_support_forces(
            reinf_forces, slices, passive=False, d_sign=d_sign).items()}
        fz_p = {i: fz for i, (_fx, fz) in slice_support_forces(
            reinf_forces, slices, passive=True, d_sign=d_sign).items()}

    if driving <= 0:
        # no number (GM-140 step 2): this used to return the 999.9 marker
        raise NoSolution(_NO_DRIVING, kind="no_driving")

    fos = fos_initial
    for iteration in range(max_iter):
        resisting = 0.0
        for i, s in enumerate(slices):
            phi_rad = math.radians(s.phi)
            tan_phi = math.tan(phi_rad)
            W = s.weight + s.surcharge_force
            b = s.width
            if i in fz_a:
                W -= fz_a[i]                 # Fz up-positive
            if i in fz_p:
                W -= fz_p[i] / fos

            m_alpha = math.cos(s.alpha) + d_sign * math.sin(s.alpha) * tan_phi / fos
            if abs(m_alpha) < 1e-10:
                m_alpha = 1e-10

            # W - u b as Bishop (1955) Eq. 13 prints it; zero where
            # negative only under the "zero" rule (GM-140 step 3, D21).
            numerator = (s.c * b
                         + _friction_term(W - s.pore_pressure * b) * tan_phi)
            resisting += numerator / m_alpha

        fos_new = (resisting + passive_resistance) / driving
        if abs(fos_new - fos) < tol:
            return fos_new
        fos, fos_prev = fos_new, fos

    # Not converged: no number (GM-140 step 2, D27; UFC 3-220-10 p. 362).
    # Before, the last iterate was returned with a Python warning only, so
    # it could be reported, or win a search, as Bishop's factor of safety.
    raise NoSolution(
        f"the iteration on F (Bishop 1955, Eq. 13) is not within {tol:g} "
        f"after {max_iter} iterations (last two values {fos_prev:.4f} and "
        f"{fos:.4f})")


# ---------------------------------------------------------------------------
# m_alpha on the slice bases (GM-127)
# ---------------------------------------------------------------------------

#: Whitman and Bailey (1967, p. 487, Eq. 5): the user is warned if, for any
#: slice, the denominator becomes negative or small, "for example, if
#: cos(theta) + tan(phi) sin(theta) / F < 0.2 ... the user must examine the
#: result in detail". A warning and an instruction, so a surface below it
#: is analysed, and kept and flagged in a search (GM-140 step 3, OWN-102;
#: `acceptability`). Slide2 reports such a surface as invalid (its error
#: -112; the "Check malpha < 0.2" project setting): leaving it out of a
#: search is the stated option (`search.LOW_M_ALPHA_ACTIONS`).
M_ALPHA_MIN = 0.2

#: Which slices the check looks at. "frictional" (the default): the slices
#: with phi' > 0. On a phi' = 0 slice m_alpha = cos(a) does not depend on F
#: and the slice's shear strength is c l whatever its normal force (in
#: Bishop's method c b / m_alpha = c l), so a small m_alpha there cannot
#: make the FOS spurious, and leaving out a surface for it only raises the
#: minimum: a vertical 5 m cut in undrained clay read 0.933 with every
#: slice checked, against 0.800 and Taylor's 0.851 (GM-127 review).
#: "every_slice": every slice, as Slide2 checks.
M_ALPHA_RULES = ("frictional", "every_slice")


def sliding_sign(slices: List[Slice], slip=None,
                 method: str = "gle") -> float:
    """+1 or -1: the sign that makes a base angle positive in the direction
    of sliding, as each method takes it. Bishop's simplified method on a
    circle takes the sign of the gravity moment about the centre, ponded
    water included (`bishop_fos`); Janbu and the GLE behind Spencer,
    Morgenstern-Price and GLE the sign of sum(W sin(a)) (`gle._normalize`).
    On a circle the two agree unless a pond's thrust turns the moment."""
    if method == "bishop" and getattr(slip, "is_circular", False):
        drive = 0.0
        for s in slices:
            drive += (s.weight + s.surcharge_force) * (s.x_mid - slip.xc)
            if s.pond_hforce != 0:
                drive += s.pond_hforce * (s.pond_z - slip.yc)
        return 1.0 if drive >= 0 else -1.0
    drive = sum((s.weight + s.surcharge_force) * math.sin(s.alpha)
                for s in slices)
    return -1.0 if drive < 0 else 1.0


def m_alpha_values(slices: List[Slice], fos: float, slip=None,
                   method: str = "gle") -> List[float]:
    """m_alpha = cos(a) + sin(a) tan(phi) / F of every slice at the factor
    of safety ``fos``, the base angle a taken positive in the direction of
    sliding (`sliding_sign`).

    m_alpha is the denominator of the base normal force in Bishop's
    simplified method (Bishop 1955, Eq. 12), in Janbu's simplified method
    and in the GLE that gives Spencer, Morgenstern-Price and GLE here
    (Fredlund & Krahn 1977; ``gle._GLESystem._normal``): pass each method's
    own F (for Janbu the uncorrected F of its force equilibrium, before
    f0). The Ordinary Method of Slices has none (its N is W cos(a)).
    """
    sign = sliding_sign(slices, slip, method)
    f = fos if fos and fos > 0 else 1.0
    return [math.cos(s.alpha)
            + sign * math.sin(s.alpha) * math.tan(math.radians(s.phi)) / f
            for s in slices]


def m_alpha_checked(s: Slice, rule: str = "frictional") -> bool:
    """True when the check looks at this slice under ``rule``
    (`M_ALPHA_RULES`): every slice, or only those with phi' > 0."""
    return rule == "every_slice" or s.phi > 1e-9


def m_alpha_min(slices: List[Slice], fos: float, slip=None,
                method: str = "gle", rule: str = "frictional") -> float:
    """The least m_alpha over the slices the check looks at under ``rule``
    (`m_alpha_values`, `m_alpha_checked`); 1.0 when there are none."""
    vals = m_alpha_values(slices, fos, slip, method)
    return min((m for s, m in zip(slices, vals) if m_alpha_checked(s, rule)),
               default=1.0)


# ---------------------------------------------------------------------------
# Rigorous Spencer / Morgenstern-Price (GLE engine wrappers)
# ---------------------------------------------------------------------------

def spencer_fos(slices: List[Slice],
                slip,
                tol: float = 1e-4,
                max_iter: int = 100,
                reinf_forces=None) -> Tuple[float, float]:
    """Spencer's Method factor of safety (rigorous GLE formulation).

    Solves the full Fredlund-Krahn general limit equilibrium with a
    constant interslice force function f(x) = 1 (Spencer's assumption):
    interslice normals E integrated slice-to-slice, shear X = lambda*E,
    and lambda found where the moment- and force-equilibrium factors of
    safety cross. Works for circular and noncircular surfaces.

    Raises ``NoSolution`` where the rigorous solution does not converge:
    there is no number (GM-140 step 2). Before, the old approximate solver
    stood in, without saying so in a search.

    Returns
    -------
    (FOS, theta) : tuple of (float, float)
        Factor of safety and interslice force angle theta = atan(lambda)
        in degrees.

    References
    ----------
    Spencer (1967); Fredlund & Krahn (1977); GeoStudio Stability
    Modeling theory book ch. 2.
    """
    from slope_stability.gle import gle_fos

    res, err = None, None
    try:
        res = gle_fos(slices, slip, f_interslice="constant",
                      tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf_forces)
    except (ZeroDivisionError, OverflowError) as exc:
        err = exc
    if (res is not None and res.converged
            and 0.0 < res.fos < _FOS_MAX):
        return (res.fos, math.degrees(math.atan(res.lam)))
    raise NoSolution(gle_failure_reason(res, err))


def morgenstern_price_fos(slices: List[Slice],
                          slip,
                          f_interslice: str = "half_sine",
                          tol: float = 1e-4,
                          max_iter: int = 100,
                          reinf_forces=None) -> Tuple[float, float]:
    """Morgenstern-Price / GLE factor of safety (rigorous formulation).

    Full Fredlund-Krahn GLE: interslice shear X = lambda * f(x) * E with
    E integrated across slices; lambda solved so the moment- and
    force-equilibrium factors of safety are equal. Supports f(x) in
    {'constant', 'half_sine', 'clipped_sine', 'trapezoidal'}.

    Raises ``NoSolution`` where the rigorous solution does not converge:
    there is no number (GM-140 step 2).

    Returns
    -------
    (FOS, lambda) : tuple of (float, float)

    References
    ----------
    Morgenstern & Price (1965); Fredlund & Krahn (1977).
    """
    from slope_stability.gle import gle_fos

    res, err = None, None
    try:
        res = gle_fos(slices, slip, f_interslice=f_interslice,
                      tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf_forces)
    except (ZeroDivisionError, OverflowError) as exc:
        err = exc
    if (res is not None and res.converged
            and 0.0 < res.fos < _FOS_MAX):
        return (res.fos, res.lam)
    raise NoSolution(gle_failure_reason(res, err))
