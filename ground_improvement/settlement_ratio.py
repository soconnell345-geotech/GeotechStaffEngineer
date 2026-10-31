"""Settlement ratio of aggregate columns by E_c / E_s (BACKLOG GM-220).

GEC-13 (FHWA-NHI-16-027) Vol. I, Fig. 5-21, p. 5-36, "Comparison of elastic
theories and field observations" (Greenwood and Kirsch 1984; Wallays et al.
1983), for infinite grid patterns: the settlement ratio, which p. 5-36
prints as "n" in quotes, "defined as the ratio of settlement without stone
columns to that with stone columns", against the area ratio A/A_c = 1 /
a_s. It is not the stress ratio n of Eq. 5-2, so the module calls it the
settlement ratio (GM-279). Two of the figure's methods work it out from the
ratio of the column modulus to the soil modulus, E_c / E_s:

``weighted_modulus``
    Curve (2), "'Weighted modulus' method", drawn for E_c/E_s = 10 (2a) and
    5 (2b). The curves are settlement ratio = 1 + a_s (E_c/E_s - 1): the
    composite modulus a_s E_c + (1 - a_s) E_s over E_s, which is the
    stress-ratio method with n = E_c/E_s. Traced off the figure, this form
    is within 1 % of curve (2a) up to A/A_c = 5 and 2.2 % beyond, and
    within 0.6 % of curve (2b) over A/A_c 1.5 to 3.5.
``balaam_rigid`` / ``balaam_flexible``
    Curves (5), "Balaam": (5b) E_c/E_s = 10 rigid, (5c) E_c/E_s = 5 rigid
    and (5a) E_c/E_s = 10 flexible. p. 5-37: the Equilibrium Method of
    FHWA (1983) "is roughly equivalent to the Balaam relationships shown in
    Figure 5-21 and represents an average or lower bound estimate suitable
    for preliminary analyses". The curves are printed only as graphs. Their
    values here were read off the rendered page (2026-10-19): each curve
    traced pixel by pixel on the scanned figure, calibrated on its printed
    grid lines, and the reads drawn back over the figure to check they sit
    on the curves. Between the tabulated A/A_c, the settlement ratio is
    interpolated linearly in a_s (within 0.5 % of the traced curves). A
    rigid E_c/E_s between 5 and 10 is interpolated linearly in E_c/E_s
    between (5c) and (5b); the flexible curve is drawn for E_c/E_s = 10
    only. Off the drawn range the curves are not extended.

Curve (4), "PRIEBE phi = 37.5°", is the Priebe curve GEC-13 reads in its
Example 2 (Sec. 4.3.2, Fig. 5-27, p. 5-49) and of which p. 5-37 says "for
preliminary estimates, the Priebe curve may be used to evaluate the upper
bound effectiveness". It takes no moduli: :data:`PRIEBE_CURVE_4` holds it,
read off the page as the Balaam curves were (GM-221), and the tool
``priebe_improvement_factor`` reads it. It is the tool's only basis
(GM-276): GEC-13 prints the curve, not a formula for it. From A/A_c about
4.0 on, the printed curve (4) runs on one line with curve (5a). The table
stops at A/A_c 8.0; the printed curve runs on a little further, to about
8.5 to 8.7 where the lines end, and the tool refuses an A/A_c past 8.0
rather than extend the table.
"""

from __future__ import annotations

import math

METHODS = ("weighted_modulus", "balaam_rigid", "balaam_flexible")

#: GEC-13 Fig. 5-21 curve (5b): Balaam, E_c/E_s = 10, rigid. A/A_c ->
#: settlement ratio.
BALAAM_RIGID_10 = {
    1.75: 4.89, 2.0: 4.35, 2.25: 3.93, 2.5: 3.60, 2.75: 3.35, 3.0: 3.15,
    3.5: 2.84, 4.0: 2.59, 4.5: 2.40, 5.0: 2.24, 5.5: 2.11, 6.0: 2.02,
    6.5: 1.92, 7.0: 1.83, 7.5: 1.74, 8.0: 1.65, 8.5: 1.56, 9.0: 1.46,
}
#: GEC-13 Fig. 5-21 curve (5c): Balaam, E_c/E_s = 5, rigid.
BALAAM_RIGID_5 = {
    1.75: 2.92, 2.0: 2.67, 2.25: 2.47, 2.5: 2.30, 2.75: 2.16, 3.0: 2.04,
    3.5: 1.85, 4.0: 1.71, 4.5: 1.60, 5.0: 1.52, 5.5: 1.48, 6.0: 1.46,
    6.5: 1.43, 7.0: 1.42, 7.5: 1.41, 8.0: 1.38, 8.5: 1.35, 9.0: 1.33,
}
#: GEC-13 Fig. 5-21 curve (5a): Balaam, E_c/E_s = 10, flexible.
BALAAM_FLEXIBLE_10 = {
    1.5: 5.24, 1.75: 4.56, 2.0: 4.04, 2.25: 3.60, 2.5: 3.24, 2.75: 2.97,
    3.0: 2.76, 3.5: 2.47, 4.0: 2.26, 4.5: 2.09, 5.0: 1.96, 5.5: 1.86,
    6.0: 1.79, 6.5: 1.72, 7.0: 1.65, 7.5: 1.57, 8.0: 1.50, 8.5: 1.42,
}

#: GEC-13 Fig. 5-21 curve (4): Priebe, phi = 37.5 deg (GM-221), p. 5-36.
#: A/A_c -> settlement ratio. Traced and drawn back as the Balaam curves;
#: the table is within 1 % of the trace between its nodes, and a read is
#: good to about one line width, +/- 0.03 in the settlement ratio. Checked
#: again on both printed copies
#: (GM-276): on p. 5-36 the table sits on the line within 0.015 (95 % of
#: reads) from A/A_c 2.25 to 8.0 and within 0.04 below A/A_c 2.25, where
#: the curve is steep; Fig. 5-27 (p. 5-49), the same chart printed again
#: for Example 2, agrees to the same order. At A/A_c 3.61 both read 2.54
#: to 2.55. The printed curve runs on past 8.0 (to about 8.5 to 8.7); the
#: table stops at 8.0 and nothing past it is read.
PRIEBE_CURVE_4 = {
    1.6: 6.01, 1.75: 5.55, 2.0: 4.82, 2.25: 4.26, 2.5: 3.80, 2.75: 3.41,
    3.0: 3.11, 3.25: 2.85, 3.5: 2.63, 3.75: 2.45, 4.0: 2.30, 4.25: 2.16,
    4.5: 2.07, 5.0: 1.96, 5.5: 1.86, 6.0: 1.79, 6.5: 1.72, 7.0: 1.65,
    7.5: 1.58, 8.0: 1.49,
}

#: What each method is, in GEC-13's words, for a result's choices.
BASIS = {
    "weighted_modulus": (
        "GEC-13 (FHWA-NHI-16-027) Fig. 5-21, p. 5-36, curve (2), the "
        "'weighted modulus' method, drawn for E_c/E_s = 10 (2a) and 5 (2b): "
        "settlement ratio = 1 + a_s (E_c/E_s - 1), the composite modulus "
        "over E_s; the "
        "stress-ratio method with n = E_c/E_s"),
    "balaam_rigid": (
        "GEC-13 (FHWA-NHI-16-027) Fig. 5-21, p. 5-36, Balaam curves (5b) "
        "E_c/E_s = 10 and (5c) E_c/E_s = 5, rigid, read off the printed "
        "figure; p. 5-37: the FHWA (1983) Equilibrium Method 'is roughly "
        "equivalent to the Balaam relationships shown in Figure 5-21 and "
        "represents an average or lower bound estimate suitable for "
        "preliminary analyses'"),
    "balaam_flexible": (
        "GEC-13 (FHWA-NHI-16-027) Fig. 5-21, p. 5-36, Balaam curve (5a) "
        "E_c/E_s = 10, flexible, read off the printed figure; p. 5-37: the "
        "FHWA (1983) Equilibrium Method 'is roughly equivalent to the "
        "Balaam relationships shown in Figure 5-21 and represents an "
        "average or lower bound estimate suitable for preliminary "
        "analyses'"),
}


def _on_curve(table: dict, as_ratio: float, name: str,
              instead: str = "settlement_method 'weighted_modulus' or "
                             "'stress_ratio'") -> float:
    """The settlement ratio off one tabulated curve at a_s, linear in a_s
    between nodes."""
    xs = sorted(table)
    x = 1.0 / as_ratio
    if not xs[0] - 1e-9 <= x <= xs[-1] + 1e-9:
        raise ValueError(
            f"GEC-13 Fig. 5-21 draws {name} over A/A_c = {xs[0]:g} to "
            f"{xs[-1]:g} (a_s = {1 / xs[-1]:.3f} to {1 / xs[0]:.3f}); "
            f"a_s = {as_ratio:.4f} is A/A_c = {x:.2f}, off the curve. Use "
            f"{instead}")
    x = min(max(x, xs[0]), xs[-1])
    for x0, x1 in zip(xs, xs[1:]):
        if x <= x1:
            a0, a1 = 1.0 / x0, 1.0 / x1
            t = (1.0 / x - a0) / (a1 - a0)
            return table[x0] + t * (table[x1] - table[x0])
    return table[xs[-1]]


def _check(as_ratio: float, modular_ratio: float) -> None:
    if not 0.0 < as_ratio < 1.0:
        raise ValueError(
            f"Area replacement ratio must be in (0, 1), got {as_ratio}")
    if not modular_ratio >= 1.0:
        raise ValueError(
            f"E_c / E_s = {modular_ratio:g}: the settlement ratio by the "
            "moduli (GEC-13 Fig. 5-21) is for a column stiffer than the "
            "soil, E_c / E_s >= 1")


def weighted_modulus_settlement_ratio(as_ratio: float,
                                      modular_ratio: float) -> float:
    """Settlement ratio = 1 + a_s (E_c/E_s - 1), GEC-13 Fig. 5-21 curve
    (2)."""
    _check(as_ratio, modular_ratio)
    return 1.0 + as_ratio * (modular_ratio - 1.0)


def balaam_settlement_ratio(as_ratio: float, modular_ratio: float,
                            footing: str = "rigid") -> float:
    """The settlement ratio off GEC-13 Fig. 5-21's Balaam curves (5).

    ``footing`` 'rigid' takes E_c/E_s from 5 to 10, on curves (5c) and
    (5b) and linear in E_c/E_s between them; 'flexible' takes E_c/E_s =
    10 only, curve (5a). Refused off the drawn E_c/E_s and A/A_c."""
    _check(as_ratio, modular_ratio)
    r = modular_ratio
    if footing == "flexible":
        if not math.isclose(r, 10.0, rel_tol=0.01):
            raise ValueError(
                f"GEC-13 Fig. 5-21 draws Balaam's flexible curve (5a) for "
                f"E_c/E_s = 10 only; E_c/E_s = {r:g}. Use 'balaam_rigid' "
                "(E_c/E_s 5 to 10) or 'weighted_modulus'")
        return _on_curve(BALAAM_FLEXIBLE_10, as_ratio,
                         "Balaam's flexible curve (5a)")
    if footing != "rigid":
        raise ValueError(
            f"footing must be 'rigid' or 'flexible', got {footing!r}")
    if math.isclose(r, 5.0, rel_tol=0.01):
        r = 5.0
    elif math.isclose(r, 10.0, rel_tol=0.01):
        r = 10.0
    if not 5.0 <= r <= 10.0:
        raise ValueError(
            f"GEC-13 Fig. 5-21 draws Balaam's rigid curves for E_c/E_s = 5 "
            f"(5c) and 10 (5b); E_c/E_s = {modular_ratio:g} is outside them. "
            "Use 'weighted_modulus', which takes any E_c/E_s")
    name = "Balaam's rigid curves (5b, 5c)"
    n5 = _on_curve(BALAAM_RIGID_5, as_ratio, name)
    n10 = _on_curve(BALAAM_RIGID_10, as_ratio, name)
    return n5 + (r - 5.0) / 5.0 * (n10 - n5)


def priebe_curve_settlement_ratio(as_ratio: float) -> float:
    """The settlement ratio off GEC-13 Fig. 5-21 curve (4), 'PRIEBE phi =
    37.5 deg' (GM-221), read at A/A_c 1.6 to 8.0; refused outside that,
    never extended."""
    if not 0.0 < as_ratio < 1.0:
        raise ValueError(
            f"Area replacement ratio must be in (0, 1), got {as_ratio}")
    lo, hi = min(PRIEBE_CURVE_4), max(PRIEBE_CURVE_4)
    x = 1.0 / as_ratio
    if not lo - 1e-9 <= x <= hi + 1e-9:
        raise ValueError(
            f"a_s = {as_ratio:.4f} is A/A_c = {x:.2f}. The tool reads GEC-13 "
            f"Fig. 5-21's Priebe curve (4) at A/A_c {lo:g} to {hi:g} (a_s "
            f"{1 / hi:.3f} to {1 / lo:.3f}) and gives no value outside that: "
            "the curve is not extended. The printed curve starts at A/A_c "
            "about 1.6 and runs on past 8.0 to about 8.5 to 8.7, on one line "
            "with Balaam's curve (5a); there, read the printed figure (p. "
            "5-36)")
    return _on_curve(PRIEBE_CURVE_4, as_ratio, "the Priebe curve (4)")


def settlement_ratio(method: str, as_ratio: float,
                     modular_ratio: float) -> float:
    """The settlement ratio by one of :data:`METHODS` (GEC-13 Fig. 5-21)."""
    if method == "weighted_modulus":
        return weighted_modulus_settlement_ratio(as_ratio, modular_ratio)
    if method == "balaam_rigid":
        return balaam_settlement_ratio(as_ratio, modular_ratio, "rigid")
    if method == "balaam_flexible":
        return balaam_settlement_ratio(as_ratio, modular_ratio, "flexible")
    raise ValueError(f"settlement ratio method must be one of {METHODS}, "
                     f"got {method!r}")
