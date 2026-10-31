"""
Beam analysis for braced and cantilever excavation walls.

Braced walls: the design pressure above the excavation is the
apparent-pressure envelope plus the surcharge and water pressures GEC-4 adds
explicitly (``soe.free_earth.apparent_pressure_profile``). Support loads and
span moments by the Caltrans Trenching and Shoring Manual's hinge method
(Sec. 8-4.02, Fig. 8-7) by default, or GEC-4's (Fig. 38) or UFC 3-220-20's
simple spans (Fig. 4-54) (``support_load_method``, GM-290). Embedment, and the
moment in the embedded portion, come from free earth support about the lowest
support (``soe.embedment``).

Cantilever walls use the Caltrans Simplified Method on layered Rankine
pressures with water (``soe.free_earth.solve_cantilever``); the moment and
shear from its FS = 1.0 body (Example 7-3, GM-290).

History (2026-09-15, field feedback N2): the cantilever used to take every
property from the first soil layer and put the passive resultant at H + D/3
about the wall base, so embedment came out about a third of the correct value;
the braced top span was treated as simply supported (pL^2/8 instead of the
cantilever pL^2/2); and surcharge and water were left out of the braced loads.
See DESIGN.md.

All units SI: kPa, kN/m, kN·m/m, meters.

References:
    California Dept. of Transportation, Trenching and Shoring Manual (2025),
        Sec 7-5.02 and 8-4
    FHWA-IF-99-015, GEC-4, Sections 5.2-5.4
    Terzaghi & Peck (1967) Soil Mechanics in Engineering Practice
"""

from typing import List

import numpy as np

from geotech_common.numpy_compat import trapezoid
from soe.earth_pressure import STIFF_CLAY_COEFFICIENT
from soe.geometry import ExcavationGeometry
from soe.free_earth import (
    _grid,
    apparent_pressure_profile,
    profile_for, passive_by_layer, PASSIVE_BASIS,
    PASSIVE_FRICTION_LEFT_OUT,
    solve_cantilever,
)
from soe.results import BracedExcavationResult, CantileverExcavationResult

#: Caltrans T&S Manual 7-1: cantilever sheet pile walls are mainly used for
#: temporary excavations not greater than about 18 ft.
_CANTILEVER_TYPICAL_MAX_H = 5.5


#: The allowable bending stress as a fraction of Fy (GM-290): 0.60, as the
#: Caltrans T&S Manual's examples take it (Example 7-3, p. 7-43: "Fb =
#: 0.60Fy"; Example 8-1, p. 8-19: "42,000 psi x 0.6"). Its Sec. 6-2.02 (p.
#: 6-3) prints no value: it refers steel members to the AISC ASD
#: requirements and says Fb depends on whether the section is compact.
#: GEC-4 Sec. 5.4.1 (p. 81)
#: recommends 0.55 Fy "for permanent walls and temporary walls that are
#: considered critical"; the manual's railroad example compares with 0.66 Fy
#: (p. 9-20). Until GM-290 the module used 0.66 Fy everywhere.
FB_RATIO = 0.60

#: The allowable bending stress's printed alternatives, for the choice.
_FB_BASIS = (
    "Fb = 0.60 Fy, as the Caltrans Trenching and Shoring Manual's examples "
    "take it (Example 7-3, p. 7-43; Example 8-1, p. 8-19); its Sec. 6-2.02 "
    "(p. 6-3) refers steel members to the AISC ASD requirements and says Fb "
    "depends on whether the section is compact")
_FB_OTHERS = (
    (0.55, "GEC-4 Sec. 5.4.1 (p. 81), for permanent walls and temporary "
           "walls that are considered critical (20 % more allowed for "
           "temporary SOE walls at positive moments between anchors)"),
    (0.60, "the Caltrans manual's examples"),
    (0.66, "the Caltrans manual's railroad example (p. 9-20)"),
)


def _check_fb(Fb_ratio: float) -> None:
    if not 0.0 < Fb_ratio <= 1.0:
        raise ValueError(f"Fb_ratio (Fb / Fy) must be in (0, 1], got "
                         f"{Fb_ratio:g}")


def _section_modulus_cm3(moment_kNm: float, Fy_MPa: float,
                         Fb_ratio: float = FB_RATIO) -> float:
    """ASD required Sx (cm³/m): M / (Fb_ratio Fy)."""
    Fb = Fb_ratio * Fy_MPa * 1000.0  # kPa
    return moment_kNm / Fb * 1e6 if Fb > 0 else 0.0


def _fb_choice(Fb_ratio: float, Fy: float, moment) -> dict:
    """The allowable bending stress, as a result's choice (GM-290)."""
    from geotech_common.choices import choice, how_given

    def sx(r):
        return ("" if moment is None else
                f", Sx {_section_modulus_cm3(moment, Fy, r):.0f} cm3/m")
    return choice(
        "allowable bending stress (Fb_ratio)",
        f"Fb = {Fb_ratio:g} Fy" + sx(Fb_ratio),
        how_given(Fb_ratio, FB_RATIO, name="Fb_ratio"), _FB_BASIS,
        [f"Fb = {r:g} Fy{sx(r)}: {why}" for r, why in _FB_OTHERS
         if r != Fb_ratio])


# ============================================================================
# Support loads and span moments for multi-level braced excavations
# ============================================================================

#: How a braced wall's support loads and span moments are taken (GM-290).
#: "hinge" (default): the Caltrans T&S Manual's hinge method (Sec. 8-4.02,
#: Fig. 8-7, p. 8-10; Example 8-2, pp. 8-21 to 8-34): the load above the
#: first support goes to it, and its moment M1 there is carried by the span
#: below (T1L = P3/2 + M1/S, T2U = P3/2 - M1/S; the figure prints S1, which
#: is the span below in Example 8-2, both 10 ft); the other spans between
#: supports are simply supported (P/2 each, for a uniform load; the
#: centroid's share in general); the lowest support's load from below is
#: free earth support's about it (step 3, MR = MD: T4L = P6 + P7 + Pa1 + Pa2
#: - Pp1). One support: the whole wall about it (Sec. 8-4.01 step 5).
#: "hinge_subgrade": GEC-4's hinge method (Fig. 38, p. 79): the same above
#: the lowest support, and the span below it simply supported between it and
#: a subgrade reaction R at the excavation base. "simple_spans": UFC
#: 3-220-20 Fig. 4-54 (p. 257), narrow cuts: "Reactions at the braces are
#: computed assuming simple span between braces" (the load above the first
#: brace all to it) and "Moments in sheeting at the lowest strut are
#: computed for the cantilever span below the lowest strut". Until GM-290
#: the module took simple spans with the span below the lowest support all
#: to it but its moment as a simple span's (none of the three), and reported
#: free earth support's load in a note only.
SUPPORT_LOAD_METHODS = ("hinge", "hinge_subgrade", "simple_spans")

_METHOD_NAME = {
    "hinge": "hinge (support_load_method 'hinge'), the Caltrans T&S "
             "Manual's hinge method, Fig. 8-7 (p. 8-10)",
    "hinge_subgrade": "hinge_subgrade (support_load_method "
                      "'hinge_subgrade'), GEC-4's hinge method, Fig. 38 (p. "
                      "79): the span below the lowest support simply "
                      "supported on it and a subgrade reaction R at the "
                      "excavation base",
    "simple_spans": "simple_spans (support_load_method 'simple_spans'), "
                    "UFC 3-220-20 Fig. 4-54 (p. 257): simple spans between "
                    "supports, the span below the lowest support a "
                    "cantilever on it",
}
_METHOD_BASIS = (
    "The Caltrans Trenching and Shoring Manual's hinge method (Sec. 8-4.02 "
    "and Fig. 8-7, p. 8-10; Example 8-2), which the embedment below the "
    "lowest support follows too: the load above the first support and its "
    "moment M1 go to that support and the span below it (T1L = P/2 + M1/S, "
    "T2U = P/2 - M1/S), the spans between supports are simply supported, "
    "and the lowest support's load from below is free earth support's "
    "(moments about it, MR = MD); one support takes the whole wall about it "
    "(Sec. 8-4.01 step 5)")


def _span_load(p, z_top: float, z_bot: float, breaks) -> tuple:
    """(force kN/m, centroid below z_top m) of pressure p over a span."""
    if z_bot <= z_top:
        return 0.0, 0.0
    segs = _grid(z_top, z_bot, breaks)
    F = float(sum(trapezoid(p(z), z) for z in segs))
    Mt = float(sum(trapezoid(p(z) * (z - z_top), z) for z in segs))
    return F, (Mt / F if F > 0 else 0.5 * (z_bot - z_top))


def _span_diagram(p, z_top: float, z_bot: float, breaks, top_shear: float,
                  top_moment: float = 0.0):
    """Shear and moment along a span whose top carries ``top_shear``
    (0 for the free top of the cantilever span) and ``top_moment``
    (sagging positive: -M1 below the first support under the hinge
    method, GM-290)."""
    z = np.concatenate(_grid(z_top, z_bot, breaks))
    q = p(z)
    V = top_shear - np.concatenate(
        [[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(z))])
    M = top_moment + np.concatenate(
        [[0.0], np.cumsum(0.5 * (V[1:] + V[:-1]) * np.diff(z))])
    return z, V, M


def _supports(p, depths, H: float, breaks, method: str, fe_load) -> dict:
    """Support loads and the span diagrams above the excavation by
    ``method`` (:data:`SUPPORT_LOAD_METHODS`). ``fe_load``: free earth
    support's load on the lowest support (FS = 1; the whole wall's with one
    support), or None without a balance.

    Returns ``upper`` and ``lower`` (each support's load from the span above
    it and from below it; under "hinge" the lowest's lower part is
    ``fe_load``'s, None without one), ``spans`` (one row per span),
    ``z``/``p``/``V``/``M`` (the spans' diagram, in the free earth body's
    sign: push toward the excavation positive), ``M_max``, ``z_max`` and
    ``V_max`` over that diagram, and ``subgrade`` (R, "hinge_subgrade"
    only)."""
    n = len(depths)
    upper, lower = [0.0] * n, [0.0] * n
    rows, dz, dp, dV, dM = [], [], [], [], []
    peak = {"M": 0.0, "z": 0.0, "V": 0.0}
    subgrade = None

    def _r(x):
        return None if x is None else round(x, 2)

    def span(z_top, z_bot, force, centroid, to_upper, to_lower, top_shear,
             top_moment=0.0, draw=True):
        rows.append({"z_top_m": round(z_top, 3), "z_bot_m": round(z_bot, 3),
                     "load_kN_per_m": round(force, 2),
                     "centroid_below_top_m": round(centroid, 3),
                     "to_upper_kN_per_m": _r(to_upper),
                     "to_lower_kN_per_m": _r(to_lower)})
        if not draw:
            return
        z, V, M = _span_diagram(p, z_top, z_bot, breaks, top_shear,
                                top_moment)
        # Recorded in the free earth body's sign (push toward the
        # excavation positive, supports resisting it), so the two diagrams
        # the calc package draws on one set of axes agree: _span_diagram's
        # V and M are the negatives of that.
        dz.extend(map(float, z))
        dp.extend(map(float, p(z)))
        dV.extend(-float(v) for v in V)
        dM.extend(-float(m) for m in M)
        k = int(np.argmax(np.abs(M)))
        if abs(M[k]) > peak["M"]:
            peak["M"], peak["z"] = float(abs(M[k])), float(z[k])
        peak["V"] = max(peak["V"], float(np.max(np.abs(V))))

    # The cantilever above the first support: all its load to that support.
    F0, c0 = _span_load(p, 0.0, depths[0], breaks)
    M1 = F0 * (depths[0] - c0)
    upper[0] = F0
    span(0.0, depths[0], F0, c0, 0.0, F0, 0.0)
    # The spans between supports (to the excavation base for
    # "hinge_subgrade", whose base carries R): simply supported, the first
    # carrying M1 at its top under the hinge methods.
    tops = list(depths) + ([H] if method == "hinge_subgrade" else [])
    for i in range(len(tops) - 1):
        a, b = tops[i], tops[i + 1]
        L = b - a
        F, c = _span_load(p, a, b, breaks)
        m0 = M1 if (i == 0 and method != "simple_spans") else 0.0
        r_top = (F * (L - c) + m0) / L if L > 0 else 0.0
        r_bot = F - r_top
        lower[i] += r_top
        if i + 1 < n:
            upper[i + 1] += r_bot
        else:
            subgrade = r_bot
        span(a, b, F, c, r_top, r_bot, r_top, -m0)
    # Below the lowest support.
    a = depths[-1]
    if method == "simple_spans":
        F, c = _span_load(p, a, H, breaks)
        lower[-1] += F
        span(a, H, F, c, F, 0.0, F, -F * c)
    elif method == "hinge":
        F, c = _span_load(p, a, H, breaks)
        below = None if fe_load is None else (
            fe_load - (F0 if n == 1 else 0.0))
        lower[-1] = below
        span(a, H, F, c, below, None, 0.0, draw=False)
    return {"upper": upper, "lower": lower, "spans": rows,
            "z": dz, "p": dp, "V": dV, "M": dM, "M_max": peak["M"],
            "z_max": peak["z"], "V_max": peak["V"], "subgrade": subgrade}


def _loads_text(s) -> str:
    """A method's support loads in a line, for the choice."""
    return ", ".join("not found" if lo is None else f"{up + lo:.1f}"
                     for up, lo in zip(s["upper"], s["lower"])) + " kN/m" + (
        "" if s["subgrade"] is None else f", R {s['subgrade']:.1f} kN/m")


def _braced_choices(ap: dict, H: float, method: str, loads: dict,
                    coefficient: float) -> list:
    """The support load method and the clay envelope's choices (GM-290)."""
    from geotech_common.choices import MODULE, choice, how_given
    out = [choice(
        "support loads and span moments (support_load_method)",
        f"{method}: support loads {_loads_text(loads[method])}",
        how_given(method, "hinge", name="support_load_method"),
        _METHOD_BASIS,
        [f"{_METHOD_NAME[m]}: support loads {_loads_text(loads[m])}"
         for m in SUPPORT_LOAD_METHODS if m != method])]
    t = ap.get("type")
    N = ap.get("stability_number")
    gamma = ap.get("gamma_avg")
    if t == "stiff_clay" or (t == "soft_clay" and "governed_by" in ap):
        other = 0.2 if coefficient != 0.2 else 0.4
        out.append(choice(
            "stiff clay apparent pressure ordinate (stiff_clay_coefficient)",
            f"{coefficient:g} gamma H = {coefficient * gamma * H:.2f} kPa",
            how_given(coefficient, 0.4, name="stiff_clay_coefficient"),
            "UFC 3-220-20 Fig. 4-53 (p. 256), internally supported walls: "
            "'For N0 <= 4, use sigma_H2 = 0.4 gamma H', and 'When movements "
            "are minimal and the construction period is short, use sigma_H1 "
            "= 0.2 gamma H'; the Caltrans T&S Manual (Eq. 8-3-2, p. 8-5) "
            "leaves the factor between 0.2 and 0.4 to 'best engineering "
            "judgment'",
            [f"{other:g} gamma H = {other * gamma * H:.2f} kPa"]))
    if t == "soft_clay" and "governed_by" in ap:
        cu = ap["cu_avg"]
        ka_tp = 1.0 - 4.0 * cu / (gamma * H)
        alts = [f"Ka gamma H with Ka = 1 - 4 cu/(gamma H) = {ka_tp:.3f} and "
                f"no floor (Terzaghi and Peck; UFC 3-220-20 Fig. 4-52, "
                f"internally supported): {max(ka_tp, 0.0) * gamma * H:.2f} "
                "kPa"]
        if N <= 5.14:
            alts.append(f"Ka = 0.22 for 4 < N <= 5.14 (GEC-4 p. 59; UFC Fig. "
                        f"4-52, externally supported): {0.22 * gamma * H:.2f} "
                        "kPa")
        else:
            alts.append("Henkel's equation for N > 5.14 (GEC-4 Eq. 12, p. 58; "
                        "UFC Fig. 4-52), which needs the depth to a firm "
                        "layer below the base: not offered")
        out.append(choice(
            "soft to medium clay envelope ordinate, 4 < N < 6",
            f"the larger of Ka gamma H (Ka {ap['Ka']:.3f}, at least 0.25) "
            f"and the stiff clay ordinate "
            f"{ap['stiff_clay_ordinate_kPa']:.2f} kPa: "
            f"{ap['max_pressure_kPa']:.2f} kPa (N = {N:.2f})",
            MODULE,
            "The Caltrans T&S Manual Sec. 8-3.02 (p. 8-6): 'For soils with "
            "4 < Ns < 6, use the larger sigma_a from Equations 8-3-2 and "
            "8-3-3', with Ka from Eq. 8-3-4 'not to be less than 0.25'",
            alts))
    return out


def analyze_braced_excavation(geometry: ExcavationGeometry,
                              Fy: float = 345.0,
                              FOS_passive: float = 1.5,
                              embedment_increase: float = 1.0,
                              passive_theory: str = "rankine",
                              log_spiral_source: str = None,
                              passive_wall_friction_deg: float = None,
                              support_load_method: str = "hinge",
                              stiff_clay_coefficient: float =
                              STIFF_CLAY_COEFFICIENT,
                              Fb_ratio: float = FB_RATIO,
                              ) -> BracedExcavationResult:
    """Analyze a multi-level braced excavation.

    Parameters
    ----------
    geometry : ExcavationGeometry
        Complete excavation geometry with soil layers and support levels.
    Fy : float
        Steel yield strength (MPa). Default 345 (Grade 50). Used to
        compute required section modulus.
    FOS_passive : float
        Embedment factor of safety, MR = FS x MD about the lowest support.
        Default 1.5; the Caltrans T&S Manual uses 1.3 for temporary shoring.
    embedment_increase : float
        Multiplier on the free-earth-support embedment. Default 1.0 (the FS
        carries the safety, as in Caltrans Example 8-1).
    passive_theory : str
        The passive coefficient below the excavation (GM-242): "rankine"
        (default, no wall friction; ``soe.free_earth.PASSIVE_BASIS``) or
        "log_spiral" (``soe.free_earth.passive_by_layer``).
    log_spiral_source : str, optional
        With "log_spiral", the printed source of Kp
        (``geotech_common.log_spiral.SOURCES``); None: UFC 3-220-20's,
        Eq. 4-12 for the vertical face and level ground in front.
    passive_wall_friction_deg : float, optional
        With "log_spiral", the wall friction (deg) in every layer below the
        excavation, at most its phi'; None: the source's own in a layer
        without cohesion, none in one with cohesion.
    support_load_method : str
        How the support loads and span moments are taken
        (:data:`SUPPORT_LOAD_METHODS`, GM-290): "hinge" (default, the
        Caltrans T&S Manual's hinge method, Fig. 8-7), "hinge_subgrade"
        (GEC-4 Fig. 38) or "simple_spans" (UFC 3-220-20 Fig. 4-54).
    stiff_clay_coefficient : float
        The stiff clay envelope's ordinate / gamma H, 0.2 to 0.4. Default
        0.4 (UFC 3-220-20 Fig. 4-53; GM-290).
    Fb_ratio : float
        Allowable bending stress / Fy for the required section. Default
        0.60 (:data:`FB_RATIO`; GM-290).

    Returns
    -------
    BracedExcavationResult
        Analysis results including support reactions, moments, and demands.
    """
    geometry.validate()
    H = geometry.excavation_depth
    if support_load_method not in SUPPORT_LOAD_METHODS:
        raise ValueError(f"support_load_method must be one of "
                         f"{list(SUPPORT_LOAD_METHODS)}, got "
                         f"{support_load_method!r}")
    _check_fb(Fb_ratio)
    passive = _with_passive(passive_theory, log_spiral_source,
                            passive_wall_friction_deg)
    passive_by_layer(geometry.soil_layers, H, **passive)  # checked first

    if not geometry.support_levels:
        raise ValueError(
            "Braced excavation requires at least one support level. "
            "Use analyze_cantilever_excavation() for unsupported walls."
        )

    # Step 1: design pressure above the excavation
    ap, p = apparent_pressure_profile(
        geometry, stiff_clay_coefficient=stiff_clay_coefficient)
    depths = [s.depth for s in geometry.support_levels]
    breaks = depths + [H] + ([geometry.gwt_depth]
                             if geometry.gwt_depth is not None else [])

    # Step 2: embedment by free earth support about the lowest support
    from soe.embedment import embedment_detail
    if embedment_increase < 1.0:
        raise ValueError("embedment_increase must be >= 1.0")
    fe = embedment_detail(geometry, FOS_passive=FOS_passive,
                          stiff_clay_coefficient=stiff_clay_coefficient,
                          **passive)
    embedment = None if fe.D is None else fe.D * embedment_increase

    def embedment_with(other):
        D = embedment_detail(geometry, FOS_passive=FOS_passive,
                             stiff_clay_coefficient=stiff_clay_coefficient,
                             **other).D
        return None if D is None else D * embedment_increase

    # Step 3: support loads and span moments
    loads = {m: _supports(p, depths, H, breaks, m, fe.support_load)
             for m in SUPPORT_LOAD_METHODS}
    s = loads[support_load_method]
    max_moment, max_moment_depth = s["M_max"], s["z_max"]
    max_shear = s["V_max"]
    span_moment = max_moment
    if fe.max_moment is None:
        # No FS = 1 body: the moment below the excavation is unknown, so the
        # wall's maximum moment (and the section it sizes) is too.
        max_moment = max_moment_depth = max_shear = None
    else:
        if support_load_method == "hinge":
            # the wall below the lowest support is the free earth body
            max_shear = max(max_shear, fe.max_shear)
        if fe.max_moment > max_moment:
            max_moment, max_moment_depth = fe.max_moment, fe.max_moment_depth
            max_shear = max(max_shear, fe.max_shear)

    low = s["lower"][-1]
    if support_load_method == "hinge":
        method_note = (
            "Support loads by the Caltrans T&S hinge method (8-4.02, Fig. "
            "8-7): the load above the first support and its moment M1 to "
            "that support and the span below it, simple spans between "
            "supports, " + (
                "and the lowest support's load from below by free earth "
                f"support about it (FS = 1): {low:.1f} kN/m."
                if len(depths) > 1 and low is not None else
                "and, with one support, its load by free earth support "
                "about it, the whole wall (8-4.01 step 5)."
                if low is not None else
                "but free earth support found no balance at FS = 1, so the "
                "lowest support's load from below, and its load, are not "
                "found."))
    elif support_load_method == "hinge_subgrade":
        method_note = (
            "Support loads by GEC-4's hinge method (Fig. 38, p. 79): the "
            "load above the first support and its moment M1 to that support "
            "and the span below it, simple spans down to the excavation "
            f"base, which takes the subgrade reaction R = "
            f"{s['subgrade']:.1f} kN/m" + (
                "." if fe.support_load is None else
                f"; free earth support (FS = 1) would give the lowest "
                f"support {fe.support_load:.1f} kN/m from below."))
    else:
        method_note = (
            "Support loads by simple spans between supports (UFC 3-220-20 "
            "Fig. 4-54, p. 257), the span below the lowest support a "
            "cantilever on it" + (
                "" if fe.support_load is None else
                f" (free earth support, FS = 1, would give it "
                f"{fe.support_load:.1f} kN/m from below)") + ". UFC also "
            "takes 80 % of the simple span moments between braces below the "
            "upper span; not applied here. It adds an unbalanced force on "
            "the buried length where the base is unstable; not included.")

    # A no-balance warning goes first, ahead of the method notes.
    notes: List[str] = list(fe.notes) + [
        f"Pressure above the excavation: {ap['type']} apparent envelope "
        f"(max {ap['max_pressure_kPa']:.2f} kPa) + surcharge "
        f"{ap['surcharge_pressure_kPa']:.2f} kPa"
        + (" + water pressure below the water table" if ap["water_pressure_added"]
           else "")
        + " (GEC-4 5.2.4 adds surcharge and water explicitly).",
        method_note,
        f"Embedment: free earth support about the lowest support, MR = FS x MD "
        f"with FS = {FOS_passive:g} (Caltrans T&S 8-4 uses 1.3).",
    ]
    if max_moment is None:
        notes.append(
            f"The spans above the excavation reach {span_moment:.1f} kN·m/m; "
            "the moment below the excavation, and so the wall's maximum "
            "moment and required section, are undefined without a moment "
            "balance.")

    def _r(x, nd):
        return None if x is None else round(x, nd)

    support_reactions = []
    for i, sup in enumerate(geometry.support_levels):
        up, lo = s["upper"][i], s["lower"][i]
        support_reactions.append({
            "depth_m": round(sup.depth, 3),
            "load_kN_per_m": None if lo is None else round(up + lo, 2),
            "from_above_kN_per_m": round(up, 2),
            "from_below_kN_per_m": _r(lo, 2),
            "type": sup.support_type})

    return BracedExcavationResult(
        excavation_depth=H,
        n_support_levels=len(depths),
        apparent_pressure_type=ap["type"],
        max_apparent_pressure_kPa=round(float(ap["max_pressure_kPa"]), 2),
        support_reactions=support_reactions,
        max_moment_kNm_per_m=_r(max_moment, 2),
        max_moment_depth_m=_r(max_moment_depth, 2),
        max_shear_kN_per_m=_r(max_shear, 2),
        required_embedment_m=_r(embedment, 2),
        total_wall_length_m=None if embedment is None else round(H + embedment, 2),
        required_Sx_cm3=(None if max_moment is None
                         else round(_section_modulus_cm3(max_moment, Fy,
                                                         Fb_ratio), 1)),
        embedment_FS=FOS_passive,
        surcharge_pressure_kPa=round(float(ap["surcharge_pressure_kPa"]), 2),
        water_pressure_included=bool(ap["water_pressure_added"]),
        free_earth={
            "embedment_D_m": _r(fe.D, 3),
            "embedment_D_FS1_m": _r(fe.D_prime, 3),
            "lowest_support_load_from_below_kN_per_m": _r(fe.support_load, 2),
            "max_moment_kNm_per_m": _r(fe.max_moment, 2),
            "max_moment_depth_m": _r(fe.max_moment_depth, 2),
            "converged": fe.converged,
        },
        converged=fe.converged,
        notes=notes,
        support_load_method=support_load_method,
        subgrade_reaction_kN_per_m=_r(s["subgrade"], 2),
        Fb_ratio=Fb_ratio,
        apparent_pressure=dict(ap),
        layers_used=_layers_used(profile_for(geometry, **passive),
                                 geometry, passive),
        spans=s["spans"],
        diagram={"z": s["z"], "p": s["p"], "V": s["V"], "M": s["M"]},
        free_earth_diagram=getattr(fe, "diagram", None),
        choices=passive_choices(geometry, passive, embedment, embedment_with)
        + _braced_choices(ap, H, support_load_method, loads,
                          stiff_clay_coefficient)
        + [_fb_choice(Fb_ratio, Fy, max_moment)],
    )


# ============================================================================
# Cantilever excavation analysis
# ============================================================================

def _layers_used(prof, geometry, passive=None) -> list:
    """Each layer of a profile as the pressures used it: top_m, bottom_m,
    unit_weight, friction_angle, cohesion, Ka, Kp; with the log-spiral Kp
    (``passive``, GM-242) also the wall friction on the passive side,
    passive_wall_friction_deg."""
    tops = np.concatenate([[0.0], prof.bottoms[:-1]])
    rows = (passive_by_layer(geometry.soil_layers, geometry.excavation_depth,
                             **passive)
            if passive and passive["passive_theory"] == "log_spiral"
            else None)
    out = [
        {"top_m": round(float(t), 3), "bottom_m": round(float(b), 3),
         "unit_weight": float(prof.gamma[k]),
         "friction_angle": float(geometry.soil_layers[k].friction_angle),
         "cohesion": float(prof.c[k]),
         "Ka": round(float(prof.Ka[k]), 4), "Kp": round(float(prof.Kp[k]), 4)}
        for k, (t, b) in enumerate(zip(tops, prof.bottoms))]
    for row, r in zip(out, rows or ()):
        row["passive_wall_friction_deg"] = round(r["delta"], 4)
    return out


def _pressure_points(prof, FS: float, O: float) -> list:
    """The pressures of a cantilever profile at its break depths down to O
    (just above and just below a depth where a layer changes)."""
    eps = 1e-6
    depths = sorted({0.0, O, *[b for b in prof.breaks() if 0.0 < b < O]})
    rows = []
    for z in depths:
        for zz in ((z,) if z in (0.0, O) else (z - eps, z + eps)):
            act = float(prof.active(zz))
            water = float(prof.u_retained(zz) - prof.u_excavation(zz))
            pas = float(prof.passive(zz)) / FS
            rows.append({
                "z_m": round(z, 3),
                "side": ("" if zz == z else ("above" if zz < z else "below")),
                "sv_retained_kPa": round(float(prof.sv_retained(zz)), 2),
                "active_kPa": round(act, 2),
                "water_net_kPa": round(water, 2),
                "sv_excavation_kPa": round(float(prof.sv_excavation(zz)), 2),
                "passive_over_FS_kPa": round(pas, 2),
                "net_kPa": round(act + water - pas, 2)})
    # one row where nothing changes across the depth
    out = []
    for r in rows:
        if out and out[-1]["z_m"] == r["z_m"] and all(
                out[-1][k] == r[k] for k in r if k not in ("side",)):
            out[-1]["side"] = ""
            continue
        out.append(r)
    return out


def analyze_cantilever_excavation(
    geometry: ExcavationGeometry,
    FOS_passive: float = 1.5,
    Fy: float = 345.0,
    embedment_increase: float = 1.2,
    passive_theory: str = "rankine",
    log_spiral_source: str = None,
    passive_wall_friction_deg: float = None,
    Fb_ratio: float = FB_RATIO,
) -> CantileverExcavationResult:
    """Analyze a cantilever (unbraced) excavation wall.

    Caltrans Trenching and Shoring Manual Simplified Method (Sec 7-5.02) on
    layered Rankine active and passive pressures (or the log-spiral passive
    Kp, ``passive_theory``, GM-242), with water on both sides:
    moments about the rotation point O at depth D0 below the excavation give
    D0, and D = ``embedment_increase`` x D0 (1.2 per AASHTO 3.11.5.6 -- it
    accounts for rotation below O and is not a factor of safety). Passive
    pressure is divided by ``FOS_passive``. The maximum moment and shear come
    from the same method at FS = 1.0 (Example 7-3, p. 7-41: "Calculate
    embedment depth using a factor of safety (FS) equal to 1.0 for the
    purposes of calculating the shear and moments"; GM-290); the moment of
    the factored diagram, as UFC 3-220-20 Fig. 4-41 takes it, is in the
    choices. Each depth uses its own layer's phi and c.

    Parameters
    ----------
    geometry : ExcavationGeometry
        Excavation geometry. support_levels should be empty.
    FOS_passive : float
        Factor of safety on passive resistance. Default 1.5.
    Fy : float
        Steel yield strength (MPa). Default 345.
    embedment_increase : float
        D / D0. Default 1.2; pass 1.0 to carry all safety on FOS_passive.
    passive_theory, log_spiral_source, passive_wall_friction_deg : optional
        The passive coefficient below the excavation, as
        :func:`analyze_braced_excavation` takes them (GM-242).
    Fb_ratio : float
        Allowable bending stress / Fy for the required section. Default
        0.60 (:data:`FB_RATIO`; GM-290).

    Returns
    -------
    CantileverExcavationResult
    """
    geometry.validate()
    _check_fb(Fb_ratio)
    H = geometry.excavation_depth

    if geometry.support_levels:
        raise ValueError(
            "Cantilever analysis requires no support levels. "
            "Use analyze_braced_excavation() for braced walls."
        )

    passive = _with_passive(passive_theory, log_spiral_source,
                            passive_wall_friction_deg)
    prof = profile_for(geometry, **passive)
    sol = solve_cantilever(prof, FS_passive=FOS_passive,
                           embedment_increase=embedment_increase)

    def embedment_with(other):
        return solve_cantilever(profile_for(geometry, **other),
                                FS_passive=FOS_passive,
                                embedment_increase=embedment_increase).D

    choices = passive_choices(geometry, passive,
                              sol.D if sol.converged else None,
                              embedment_with)
    i = int(prof._layer(H + 1e-6))  # the layer the wall is embedded in

    notes = list(sol.notes)
    if H > _CANTILEVER_TYPICAL_MAX_H:
        notes.append(
            f"H = {H:.1f} m: cantilever walls are mainly used for excavations up "
            f"to about {_CANTILEVER_TYPICAL_MAX_H:.1f} m (18 ft, Caltrans T&S "
            "7-1); check deflection.")

    layers_used = _layers_used(prof, geometry, passive)

    if not sol.converged:
        # No D0 within 6H: nothing below is reported (see the first note).
        return CantileverExcavationResult(
            layers_used=layers_used,
            excavation_depth=H, FOS_passive=FOS_passive,
            Ka=round(float(prof.Ka[i]), 4), Kp=round(float(prof.Kp[i]), 4),
            required_embedment_m=None, total_wall_length_m=None,
            max_moment_kNm_per_m=None, max_shear_kN_per_m=None,
            required_Sx_cm3=None, embedment_converged_m=None,
            embedment_increase=embedment_increase, max_moment_depth_m=None,
            net_force_at_O_kN_per_m=None, converged=False, notes=notes,
            choices=choices, Fb_ratio=Fb_ratio)

    choices = choices + [_cantilever_moment_choice(sol, FOS_passive),
                         _fb_choice(Fb_ratio, Fy, sol.max_moment)]

    return CantileverExcavationResult(
        layers_used=layers_used, diagram=sol.diagram,
        pressure_points=_pressure_points(prof, FOS_passive, H + sol.D0),
        excavation_depth=H,
        FOS_passive=FOS_passive,
        Ka=round(float(prof.Ka[i]), 4),
        Kp=round(float(prof.Kp[i]), 4),
        required_embedment_m=round(sol.D, 2),
        total_wall_length_m=round(H + sol.D, 2),
        max_moment_kNm_per_m=round(sol.max_moment, 2),
        max_shear_kN_per_m=round(sol.max_shear, 2),
        required_Sx_cm3=round(_section_modulus_cm3(sol.max_moment, Fy,
                                                   Fb_ratio), 1),
        embedment_converged_m=round(sol.D0, 3),
        moment_body_D0_m=(None if sol.D0_FS1 is None
                          else round(sol.D0_FS1, 3)),
        Fb_ratio=Fb_ratio,
        embedment_increase=embedment_increase,
        max_moment_depth_m=round(sol.max_moment_depth, 2),
        net_force_at_O_kN_per_m=round(sol.net_force_at_O, 2),
        notes=notes,
        choices=choices,
    )


def _cantilever_moment_choice(sol, FS: float) -> dict:
    """Which diagram the cantilever's moment and shear come from (GM-290)."""
    from geotech_common.choices import MODULE, choice
    if sol.D0_FS1 is None:
        chosen = (f"the factored diagram (passive / {FS:g}) to O: Mmax "
                  f"{sol.max_moment:.1f} kN-m/m at z = "
                  f"{sol.max_moment_depth:.2f} m, as no D0 balances the "
                  "moments at FS = 1.0")
        alts = []
    else:
        chosen = (f"the FS = 1.0 diagram to its O (D0 = {sol.D0_FS1:.2f} m): "
                  f"Mmax {sol.max_moment:.1f} kN-m/m at z = "
                  f"{sol.max_moment_depth:.2f} m")
        alts = [f"the factored diagram (passive / {FS:g}) to O, as UFC "
                f"3-220-20 Fig. 4-41 step 5 (p. 242) and its chart, Fig. "
                f"4-42, take it: Mmax {sol.max_moment_factored:.1f} kN-m/m "
                f"at z = {sol.max_moment_factored_depth:.2f} m"]
    return choice(
        "cantilever moment and shear", chosen, MODULE,
        "The Caltrans Trenching and Shoring Manual's Example 7-3 (p. 7-41): "
        "the embedment at FS 1.3, then 'Calculate embedment depth using a "
        "factor of safety (FS) equal to 1.0 for the purposes of calculating "
        "the shear and moments' (Mmax at zero shear, p. 7-42)", alts)


# ============================================================================
# The passive coefficient below the excavation (GM-242)
# ============================================================================
def _with_passive(theory, source, delta):
    """The keyword arguments of a passive theory."""
    return {"passive_theory": theory, "log_spiral_source": source,
            "passive_wall_friction_deg": delta}


def _layers_Kp(rows) -> str:
    """Each layer below the excavation's Kp and wall friction, in a line."""
    return "; ".join(
        f"layer {k}: Kp {r['Kp']:.3g}" + (f" at delta {r['delta']:.3g} deg"
                                         if r["delta"] else "")
        for k, r in enumerate(rows, 1) if r["below"])


#: How each layer's wall friction came, for the choice.
_CAME = {"given": "given, at most phi'",
         "source": "delta/phi' 0.66, taken when left out",
         "cohesion": "none with cohesion, taken when left out",
         "phi_0": "none, phi' 0"}


def passive_choices(geometry: ExcavationGeometry, passive: dict,
                    embedment, embedment_with) -> list:
    """How the analysis took the passive pressure below the excavation
    (GM-242), as a result's ``choices`` (``geotech_common.choices``): the
    theory with the other theory's Kp and embedment as the alternative
    (``embedment_with(passive)`` gives it, None with no balance), and with
    the log-spiral Kp its source and each layer's wall friction.
    ``embedment`` is the analysis's own (m, or None)."""
    from geotech_common import log_spiral as _ls
    from geotech_common.choices import DEFAULT, choice, how_given
    layers, H = geometry.soil_layers, geometry.excavation_depth
    theory = passive["passive_theory"]
    rows = passive_by_layer(layers, H, **passive)
    if theory == "rankine":
        other = _with_passive("log_spiral", None, None)
        other_name = ("log_spiral (passive_theory 'log_spiral'), UFC "
                      "3-220-20 Eq. 4-12, with passive_wall_friction_deg "
                      "left out: delta/phi' 0.66, none with cohesion")
    else:
        other = _with_passive("rankine", None, None)
        other_name = "rankine (passive_theory 'rankine'), no wall friction"

    def depth(D):
        return ("no embedment balances the moments" if D is None
                else f"embedment {D:.2f} m")

    try:
        rows_other = passive_by_layer(layers, H, **other)
        alt = (f"{other_name}: {_layers_Kp(rows_other)}; "
               f"{depth(embedment_with(other))}")
    except ValueError as exc:
        alt = f"{other_name}: none here, {exc}"
    chosen = (f"{theory}: {_layers_Kp(rows)}; {depth(embedment)}")
    out = [choice(
        "passive pressure below the excavation (passive_theory)", chosen,
        how_given(theory, "rankine", name="passive_theory"),
        PASSIVE_BASIS, [alt])]
    if theory == "rankine":
        return out
    source = passive["log_spiral_source"] or _ls.DEFAULT_SOURCE
    below = [(k, layer, r) for k, (layer, r) in
             enumerate(zip(layers, rows), 1) if r["below"]]
    basis = _ls.SOURCE_BASIS[source]
    if source == _ls.DEFAULT_SOURCE:
        basis += (". In front of the wall the face is vertical and the "
                  "ground level, so it is Eq. 4-12 (p. 191). "
                  + _ls.EQ_VS_CHARTS_NOTE)
    if source in (_ls.DEFAULT_SOURCE, "dm7_2_eq_4_12") and any(
            _ls.above_the_charts(layer.friction_angle, r["delta"])
            for _, layer, r in below):
        basis += ". " + _ls.HIGH_FRICTION_NOTE
    # the other sources, at the layer the wall is embedded in
    k, layer, r = next(((k, l, r) for k, l, r in below
                        if r["delta"] > 0), below[0])
    out.append(choice(
        "Log-spiral passive Kp below the excavation (log_spiral_source)",
        source, how_given(source, _ls.DEFAULT_SOURCE,
                          name="log_spiral_source"), basis,
        [f"layer {k}, " + a for a in _ls.other_sources(
            layer.friction_angle, r["delta"],
            used=(source, "dm7_2_eq_4_12"
                  if source == _ls.DEFAULT_SOURCE else source))]))
    given = passive["passive_wall_friction_deg"]
    left_out = any(r["came"] in ("source", "cohesion") for _, _, r in below)
    out.append(choice(
        "Wall friction delta below the excavation "
        "(passive_wall_friction_deg)",
        "; ".join(f"layer {k}: {r['delta']:.3g} deg ({_CAME[r['came']]})"
                  for k, _, r in below),
        DEFAULT if left_out else how_given(
            given, None, name="passive_wall_friction_deg"),
        PASSIVE_FRICTION_LEFT_OUT + ". " + _ls.wall_friction_note(
            source, left_out="")))
    return out
