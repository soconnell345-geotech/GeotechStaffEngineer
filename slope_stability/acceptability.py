"""
The printed acceptability checks on a solved slip surface (GM-140 step 3).

The rule. Once a method has solved a surface (step 2) and the surface is
one sliding mass (GM-122), every further test is a printed acceptability
check whose finding is reported with the surface: which check, which slice,
what value. Nothing is left out silently, and circles and polylines meet
the same checks. A search keeps a flagged surface by default and says so;
refusing on a check is a stated option.

The checks, each with its printed source:

``low_m_alpha``
    Whitman and Bailey (1967, p. 487, Eq. 5): the user is warned if, for
    any slice, the denominator becomes negative or small, "for example, if
    cos(theta) + tan(phi) sin(theta) / F < 0.2 ... the user must examine
    the result in detail". A warning and an instruction, not a rejection.
    For Spencer, Morgenstern-Price and GLE the value read is the
    denominator of the method's own slice equations at the reported
    state, cos(alpha - theta_i) + sin(alpha - theta_i) tan(phi') / F with
    the slice's side-force inclination theta_i
    (`gle.GLEResult.slice_denominator`); for horizontal side forces it is
    m_alpha, so Bishop's and Janbu's checks are unchanged. Whitman and
    Bailey print the check for m_alpha; reading it with an inclined side
    force is the lead's call (DESIGN.md, GM-344; the other form keeps the
    check on horizontal side forces). EM 1110-2-1902 C-10b (p. C-34): where
    the resultant on a base is nearly parallel to the side force, as on a
    surface exiting too steeply, the forces become very large and the
    factor of safety is not meaningful; pp. C-34 to C-36 print the
    remedies (`C10B_REMEDIES`). Slices with phi' = 0 are
    not looked at by default: there m_alpha = cos(alpha) does not depend on
    F and the strength is c l whatever the normal force. That exemption is
    the module's own reasoning, with no printed source; "every_slice" is
    the option (`methods.M_ALPHA_RULES`).

``bishop_below_oms``
    Duncan (1996, p. 580): calculate the Ordinary Method of Slices value
    for each circle when Bishop's method is used; where Bishop's F is below
    it, Bishop's solution has numerical problems and the Ordinary Method of
    Slices value is the better answer. Made on circles with no supports and
    no seismic coefficient (`solution.bishop_against_oms`).

``negative_frictional_term``
    Whitman and Bailey (1967, p. 487): the user is warned if, for any
    slice, the numerator becomes negative. Bishop (1955) Eq. 13, p. 10
    writes the frictional term as W - u b, and Eq. 9 the Ordinary Method of
    Slices' N' = W cos(alpha) - u l, with no lower limit
    (`methods.NEGATIVE_TERM_RULES`).

``base_tension``, ``side_tension``, ``thrust_outside``
    EM 1110-2-1902 C-10a (pp. C-33 to C-34): calculated normal and
    interslice forces should be examined; tensile forces should not exist
    on the sides or bottom of slices; the line of thrust should be within
    the sliding mass. UFC 3-220-10 pp. 361 to 362. For Spencer,
    Morgenstern-Price and GLE, from the converged state. By default the
    solver's equations are as printed on a base in tension, the friction
    term negative there; ``base_in_tension='no_friction'``
    (`gle.BASE_TENSION_RULES`, D29, GM-344) takes such a slice
    frictionless in every equation. The finding says which was done.

A finding is a dict: ``check`` (a name above), ``flag`` (True: the printed
check is not met), ``source``, ``text`` (plain words) and the located
values (``slices`` or ``sides``, each with its number counted from the
surface's low-x end, its x and the value). ``bishop_against_oms`` is also
reported, unflagged, when the comparison was made and Bishop's value
stands, or was not made (with why).

No constant here is chosen for a case. The 0.2 is Whitman and Bailey's.
`E_NEGLIGIBLE` is the GLE engine's own threshold below which an interslice
force is taken as zero and its point of application as undefined
(`gle.gle_fos`, the thrust line); `ROUNDING` separates a zero from a
negative number in floating point.
"""

import math
from typing import Any, Dict, List, Optional

from slope_stability.methods import (
    M_ALPHA_MIN, m_alpha_checked, m_alpha_values, sliding_sign,
    negative_term_in_force,
)

#: The checks whose finding flags a surface (`surface_findings`).
CHECKS = ("low_m_alpha", "bishop_below_oms", "negative_frictional_term",
          "base_tension", "side_tension", "thrust_outside")

#: Plain names, for messages and tables.
CHECK_TEXT = {
    "low_m_alpha": "m_alpha below 0.2 on a slice",
    "bishop_below_oms": ("Bishop's value below the Ordinary Method of "
                         "Slices value"),
    "negative_frictional_term": "a negative frictional term on a slice",
    "base_tension": "a negative effective normal force on a slice base",
    "side_tension": "a tensile interslice force on a slice side",
    "thrust_outside": "the line of thrust outside the sliding mass",
}

SOURCES = {
    "low_m_alpha": "Whitman and Bailey (1967), p. 487, Eq. 5",
    "bishop_below_oms": "Duncan (1996), p. 580",
    "bishop_against_oms": "Duncan (1996), p. 580",
    "negative_frictional_term": ("Whitman and Bailey (1967), p. 487; "
                                 "Bishop (1955), Eqs. 9 and 13, p. 10"),
    "base_tension": ("EM 1110-2-1902 C-10a, pp. C-33 to C-34; UFC 3-220-10 "
                     "pp. 361 to 362"),
    "side_tension": ("EM 1110-2-1902 C-10a, pp. C-33 to C-34; UFC 3-220-10 "
                     "pp. 361 to 362"),
    "thrust_outside": ("EM 1110-2-1902 C-10a, pp. C-33 to C-34; UFC "
                       "3-220-10 pp. 361 to 362"),
}

#: EM 1110-2-1902 C-10b (pp. C-34 to C-36): the remedies for a steep exit,
#: in the order printed. Named in the warning; not applied to a surface.
C10B_REMEDIES = (
    "EM 1110-2-1902 C-10b (pp. C-34 to C-36) prints four remedies for a "
    "steep exit, in this order: (1) flatten the toe of the surface, to an "
    "exit at alpha = -(45 deg - phi'/2) for a horizontal ground surface "
    "and interslice force (Eq. C-32; Eq. C-33 for an inclined interslice "
    "force), which needs a noncircular surface (the EM: probably the best "
    "approach); (2) change the side-force inclination (the EM names the "
    "Modified Swedish Method and the Morgenstern and Price method; here "
    "Morgenstern-Price with the half-sine side-force function is the "
    "module's form of it); (3) use the Ordinary Method of Slices, which "
    "neglects side forces (the EM: not accurate for effective stress "
    "analyses where pore pressures are high); (4) give the exit zone a "
    "passive strength as cohesion with phi = 0 (Eq. C-35), the only one "
    "usable with Bishop's simplified method. None is applied here: the "
    "surface is analysed as given")

#: The method's name in a finding's text.
_METHOD_NAME = {"spencer": "Spencer's method",
                "morgenstern_price": "Morgenstern-Price",
                "gle": "the GLE"}

#: An interslice force below this fraction of the largest on the surface
#: is taken as zero (the GLE engine's own threshold for the thrust line).
E_NEGLIGIBLE = 1e-3
#: Floating-point rounding, as a fraction of the largest value compared.
ROUNDING = 1e-9

_RIGOROUS = ("spencer", "morgenstern_price", "gle")
#: The methods the checks of EM 1110-2-1902 C-10a (pp. C-33 to C-34:
#: the effective normal force on each base, the interslice forces, the
#: line of thrust) are made for here: the GLE's. Janbu's simplified
#: method and the Ordinary Method of Slices have none.
C10A_METHODS = _RIGOROUS


def _finding(check: str, flag: bool, text: str, **located) -> Dict[str, Any]:
    out = {"check": check, "flag": flag,
           "source": SOURCES.get(check, ""), "text": text}
    out.update(located)
    return out


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def _numbers(items: List[Dict[str, Any]], key: str = "slice") -> str:
    """'3, 4 and 7' from the located items."""
    ns = [str(it[key]) for it in items]
    if len(ns) <= 1:
        return "".join(ns)
    return ", ".join(ns[:-1]) + " and " + ns[-1]


def low_m_alpha_finding(slices, slip, method: str, fos: Optional[float],
                        fos_janbu_uncorrected: Optional[float] = None,
                        rule: str = "frictional", gle=None
                        ) -> Optional[Dict[str, Any]]:
    """The ``low_m_alpha`` finding of a surface solved by ``method`` at its
    own factor of safety (Janbu: the uncorrected F of its force
    equilibrium; Bishop: Bishop's own value, also where the Ordinary
    Method of Slices value was taken), or None. The Ordinary Method of
    Slices has no m_alpha (its base normal force is W cos(alpha)).

    Spencer, Morgenstern-Price and GLE, given their converged state
    (``gle``, a `gle.GLEResult`): the denominator of the slice's own
    equations in that state, cos(alpha - theta_i) + sin(alpha - theta_i)
    tan(phi') / F with the slice's side-force inclination theta_i
    (``gle.slice_denominator``), against the same 0.2; each slice located
    also gives theta_i, the interslice normal force on that side and the
    effective normal force on its base. For horizontal side forces this is
    m_alpha. Without the state, m_alpha (horizontal side forces)."""
    if method == "fellenius" or fos is None or not (0.0 < fos < 900.0):
        return None
    which = ("gle" if method in _RIGOROUS
             else "janbu" if method == "janbu" else "bishop")
    f = (fos_janbu_uncorrected if method == "janbu"
         and fos_janbu_uncorrected else fos)
    side = (method in _RIGOROUS and gle is not None
            and len(getattr(gle, "slice_denominator", None) or [])
            == len(slices))
    vals = (list(gle.slice_denominator) if side
            else m_alpha_values(slices, f, slip, which))
    # A slice the solve took frictionless (base_in_tension='no_friction')
    # has, as a slice with phi' = 0, a denominator cos(alpha - theta) that
    # does not depend on F and a strength c l whatever N': it is exempt as
    # those are (the module's own reasoning, GM-344 fix round).
    fr = list(getattr(gle, "frictionless", None) or []) if side else []
    if len(fr) != len(slices):
        fr = [False] * len(slices)

    def looked_at(i, s):
        return m_alpha_checked(s, rule) and (rule == "every_slice"
                                             or not fr[i])
    sign = sliding_sign(slices, slip, which)
    low = []
    for i, (s, m) in enumerate(zip(slices, vals)):
        if m < M_ALPHA_MIN and looked_at(i, s):
            it = {"slice": i + 1, "x": _r(s.x_mid),
                  "m_alpha": _r(m, 4),
                  "alpha_deg": _r(math.degrees(s.alpha), 1),
                  "phi_deg": _r(s.phi, 2),
                  "against_sliding": bool(sign * s.alpha < 0)}
            if side:
                it.update(theta_deg=_r(gle.side_force_theta_deg[i], 2),
                          E_side_kN_per_m=_r(gle.side_force_E[i], 1),
                          N_eff_kN_per_m=_r(gle.base_normal_eff[i], 1))
            low.append(it)
    if not low:
        return None
    worst = min(low, key=lambda it: it["m_alpha"])
    n_exempt = sum(1 for s in slices if not m_alpha_checked(s, rule))
    n_frictionless = sum(1 for i, s in enumerate(slices)
                         if m_alpha_checked(s, rule) and not looked_at(i, s))
    sense = ("against" if worst["against_sliding"] else "with")
    if side:
        text = (f"The denominator of the slice equations of "
                f"{_METHOD_NAME.get(method, method)}, cos(alpha - theta) + "
                f"sin(alpha - theta) tan(phi') / F with theta the "
                f"inclination of the side force on the slice (m_alpha "
                f"where the side forces are horizontal), is below 0.2 on "
                f"slice {_numbers(low)} of {len(slices)} at F = {f:.3f} "
                f"(least {worst['m_alpha']:.3g} on slice {worst['slice']}, "
                f"x = {worst['x']:.2f}, base inclined "
                f"{abs(worst['alpha_deg']):.0f} deg {sense} the sliding, "
                f"side force inclined {worst['theta_deg']:.1f} deg; the "
                f"forces on it: an interslice normal force of "
                f"{worst['E_side_kN_per_m']:.0f} kN/m on that side and an "
                f"effective normal force of {worst['N_eff_kN_per_m']:.0f} "
                f"kN/m on its base). The result must be examined in detail "
                f"(Whitman and Bailey 1967, p. 487, who print the check "
                f"for m_alpha; here it is read with the slice's own "
                f"side-force inclination, DESIGN.md).")
        if any(it["against_sliding"] for it in low):
            text += (" Where the resultant on a base is nearly parallel to "
                     "the side force, as on a surface that exits too "
                     "steeply, the forces near the toe become very large "
                     "and the factor of safety is not meaningful "
                     "(EM 1110-2-1902 C-10b, p. C-34).")
    else:
        text = (f"m_alpha = cos(alpha) + sin(alpha) tan(phi') / F is below "
                f"0.2 on slice {_numbers(low)} of {len(slices)} at "
                f"F = {f:.3f} (least {worst['m_alpha']:.2f} on slice "
                f"{worst['slice']}, x = {worst['x']:.2f}, base inclined "
                f"{abs(worst['alpha_deg']):.0f} deg {sense} the sliding). "
                f"The result must be examined in detail (Whitman and "
                f"Bailey 1967, p. 487).")
    if rule != "every_slice" and n_exempt:
        text += (f" {n_exempt} slice(s) with phi' = 0 are not looked at: "
                 f"there m_alpha does not depend on F and the strength is "
                 f"c l whatever the normal force (the module's own "
                 f"reasoning, no printed source; m_alpha_rule="
                 f"'every_slice' looks at every slice).")
    if n_frictionless:
        text += (f" {n_frictionless} slice(s) taken frictionless "
                 f"(base_in_tension='no_friction') are not looked at, for "
                 f"the same reason (m_alpha_rule='every_slice' looks at "
                 f"them).")
    # EM C-10b's remedies are for a steep exit: named where a low slice's
    # base is inclined against the sliding (the passive zone). A base
    # nearly vertical where the surface enters is located, no more.
    passive = any(it["against_sliding"] for it in low)
    return _finding("low_m_alpha", True, text, slices=low, F=_r(f, 4),
                    rule=rule,
                    denominator=("side_force_inclination" if side
                                 else "m_alpha"),
                    remedies=(C10B_REMEDIES + ".") if passive else "")


def negative_term_finding(slices, method: str
                          ) -> Optional[Dict[str, Any]]:
    """The ``negative_frictional_term`` finding of Bishop's simplified
    method (u b above W) or the Ordinary Method of Slices (u l above
    W cos(alpha)) on these slices, or None."""
    if method not in ("bishop", "fellenius"):
        return None
    neg = []
    for i, s in enumerate(slices):
        W = s.weight + s.surcharge_force
        if method == "bishop":
            term, uforce = W - s.pore_pressure * s.width, \
                s.pore_pressure * s.width
        else:
            term = W * math.cos(s.alpha) - s.pore_pressure * s.base_length
            uforce = s.pore_pressure * s.base_length
        if term < -ROUNDING * max(abs(W), abs(uforce), 1.0):
            neg.append({"slice": i + 1, "x": _r(s.x_mid),
                        "term_kN_per_m": _r(term),
                        "W_kN_per_m": _r(W),
                        "u_force_kN_per_m": _r(uforce),
                        "phi_deg": _r(s.phi, 2)})
    if not neg:
        return None
    rule = negative_term_in_force()
    what = ("W - u b (Bishop 1955, Eq. 13, p. 10)" if method == "bishop"
            else "N' = W cos(alpha) - u l (Bishop 1955, Eq. 9, p. 10)")
    used = ("It is used as printed, negative"
            if rule == "as_printed" else
            "It is set to zero there (negative_friction_term='zero', the "
            "module's earlier rule, no printed source)")
    return _finding(
        "negative_frictional_term", True,
        f"The frictional term {what} is negative on slice {_numbers(neg)} "
        f"of {len(slices)}: the pore-water force on the base exceeds the "
        f"slice's weight there. {used}. Whitman and Bailey (1967, p. 487) "
        f"ask that the user be warned; check the pore pressures on those "
        f"slices.", slices=neg, rule=rule)


def bishop_oms_findings(sol) -> List[Dict[str, Any]]:
    """The Bishop against Ordinary Method of Slices record of a solved
    Bishop circle (`solution.bishop_against_oms`): flagged where Bishop's
    value was the lower and the Ordinary Method of Slices value was taken;
    unflagged where it stands, or where the comparison was not made."""
    if sol is None or sol.method != "bishop" or sol.fos_bishop is None:
        return []
    if sol.oms_not_compared:
        return [_finding(
            "bishop_against_oms", False,
            f"Bishop's factor of safety was not compared with the Ordinary "
            f"Method of Slices value for this circle: "
            f"{sol.oms_not_compared}.", compared=False,
            FOS_bishop=_r(sol.fos_bishop, 4),
            reason=sol.oms_not_compared)]
    if sol.oms_taken:
        return [_finding(
            "bishop_below_oms", True,
            f"Bishop's simplified method gives {sol.fos_bishop:.3f} on "
            f"this circle, below the Ordinary Method of Slices value of "
            f"{sol.fos_oms:.3f}: Bishop's solution has numerical problems "
            f"there and the Ordinary Method of Slices value is the better "
            f"answer (Duncan 1996, p. 580). The factor of safety reported "
            f"is the Ordinary Method of Slices value.", compared=True,
            FOS_bishop=_r(sol.fos_bishop, 4), FOS_oms=_r(sol.fos_oms, 4),
            taken="oms")]
    return [_finding(
        "bishop_against_oms", False,
        f"Bishop's factor of safety ({sol.fos_bishop:.3f}) is not below "
        f"the Ordinary Method of Slices value for this circle "
        f"({sol.fos_oms:.3f}).", compared=True,
        FOS_bishop=_r(sol.fos_bishop, 4), FOS_oms=_r(sol.fos_oms, 4),
        taken="bishop")]


def thrust_line_unclipped(gle) -> List[tuple]:
    """(x, elevation) of the line of thrust at each slice boundary as the
    GLE state gives it, not clipped to the sliding mass."""
    return list(zip(gle.boundary_x, gle.thrust_elevation))


def rigorous_findings(geom, slip, slices, gle) -> List[Dict[str, Any]]:
    """``base_tension``, ``side_tension`` and ``thrust_outside`` of a
    converged Spencer / Morgenstern-Price / GLE state (EM 1110-2-1902
    C-10a). ``geom`` gives the ground at the slice sides; without it the
    line of thrust is not checked."""
    out: List[Dict[str, Any]] = []
    if gle is None or not gle.base_normal_eff:
        return out
    n = len(slices)

    # Effective normal force on each base
    n_ref = max((abs(v) for v in gle.base_normal), default=0.0)
    tens = [{"slice": i + 1, "x": _r(s.x_mid),
             "N_eff_kN_per_m": _r(gle.base_normal_eff[i]),
             "phi_deg": _r(s.phi, 2)}
            for i, s in enumerate(slices)
            if gle.base_normal_eff[i] < -ROUNDING * max(n_ref, 1.0)]
    rule = getattr(gle, "base_tension_rule", "as_printed")
    if tens:
        done = (
            "The method's equations are used as printed on those bases, "
            "the friction term negative there (base_in_tension="
            "'as_printed', the default; base_in_tension='no_friction' "
            "takes such a slice frictionless in every equation, the "
            "module's own device, not found in print)."
            if rule == "as_printed" else
            "Those slices are taken frictionless in every equation "
            "(base_in_tension='no_friction', the module's own device, not "
            "found in print; 'as_printed' is the default).")
        out.append(_finding(
            "base_tension", True,
            f"The effective normal force on the base is negative (tension) "
            f"on slice {_numbers(tens)} of {n}. Tensile forces should not "
            f"exist on the bottom of slices (EM 1110-2-1902 C-10a, "
            f"pp. C-33 to C-34). {done} EM C-10c (p. C-36) prints "
            f"the remedy for tension at the crest, a tension crack; none "
            f"is added here.", slices=tens, rule=rule))

    # Interslice normal force on each interior side
    E = gle.interslice_E
    e_ref = max((abs(e) for e in E), default=0.0)
    e_tol = max(1e-6, E_NEGLIGIBLE * e_ref)
    sides = [{"side": j, "x": _r(gle.boundary_x[j]),
              "E_kN_per_m": _r(E[j])}
             for j in range(1, n) if E[j] < -e_tol]
    if sides:
        out.append(_finding(
            "side_tension", True,
            f"The interslice normal force is tensile on side "
            f"{_numbers(sides, 'side')} of the {n - 1} between slices "
            f"(side j lies between slices j and j + 1). Tensile forces "
            f"should not exist on the sides of slices (EM 1110-2-1902 "
            f"C-10a, pp. C-33 to C-34).", sides=sides))

    # The line of thrust, not clipped, against the sliding mass
    if geom is not None:
        outside = []
        for j in range(1, n):
            if abs(E[j]) <= e_tol:
                continue            # no force there: no point of application
            bx, tz = gle.boundary_x[j], gle.thrust_elevation[j]
            z_slip = slip.slip_elevation_at(bx)
            z_ground = geom.ground_elevation_at(bx)
            if z_slip is None or z_ground is None:
                continue
            lo, hi = min(z_slip, z_ground), max(z_slip, z_ground)
            eps = ROUNDING * max(abs(hi), abs(lo), 1.0)
            if tz < lo - eps or tz > hi + eps:
                outside.append({
                    "side": j, "x": _r(bx), "z_thrust": _r(tz),
                    "z_slip": _r(z_slip), "z_ground": _r(z_ground),
                    "where": "below the slip surface" if tz < lo
                    else "above the ground"})
        if outside:
            out.append(_finding(
                "thrust_outside", True,
                f"The line of thrust lies outside the sliding mass on side "
                f"{_numbers(outside, 'side')} of the {n - 1} between "
                f"slices. The line of thrust should be within the sliding "
                f"mass (EM 1110-2-1902 C-10a, pp. C-33 to C-34).",
                sides=outside))
    return out


def surface_findings(geom, slip, slices, sol, rule: str = "frictional"
                     ) -> List[Dict[str, Any]]:
    """Every finding of the printed acceptability checks on a surface the
    method solved (``sol``, a `solution.MethodSolution`); [] where it has
    no solution (there is nothing to check: the result says why instead).

    ``rule`` is the m_alpha rule (`methods.M_ALPHA_RULES`)."""
    if sol is None or not sol.solved:
        return []
    out: List[Dict[str, Any]] = []
    method = sol.method
    f_own = sol.fos_bishop if method == "bishop" and sol.fos_bishop else \
        sol.fos
    f = low_m_alpha_finding(slices, slip, method, f_own,
                            sol.fos_uncorrected, rule,
                            gle=sol.gle if method in _RIGOROUS else None)
    if f:
        out.append(f)
    out.extend(bishop_oms_findings(sol))
    f = negative_term_finding(slices, method)
    if f:
        out.append(f)
    if method in _RIGOROUS:
        out.extend(rigorous_findings(geom, slip, slices, sol.gle))
    return out


def flags(findings: List[Dict[str, Any]]) -> List[str]:
    """The names of the checks not met, in `CHECKS` order."""
    got = {f["check"] for f in findings if f.get("flag")}
    return [c for c in CHECKS if c in got]


def flags_text(names) -> str:
    """'m_alpha below 0.2 on a slice; ...' from flag names."""
    return "; ".join(CHECK_TEXT.get(n, n) for n in names)


def findings_warnings(findings: List[Dict[str, Any]]) -> List[str]:
    """The plain statement of each flagged finding, and of a comparison
    that was not made, for a result's ``warnings``."""
    out = []
    for f in findings:
        if f.get("flag"):
            text = f["text"]
            if f["check"] == "low_m_alpha" and f.get("remedies"):
                text += " " + f["remedies"]
            out.append(text)
        elif f["check"] == "bishop_against_oms" and not f.get("compared"):
            out.append(f["text"])
    return out
