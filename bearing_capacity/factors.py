"""
Bearing capacity factors and correction factors.

Implements the general bearing capacity equation factors:
    Nc, Nq, Ngamma — bearing capacity factors (functions of phi)
    sc, sq, sg — shape factors
    dc, dq, dg — depth factors
    ic, iq, ig — load inclination factors
    bc, bq, bg — base inclination factors
    gc, gq, gg — ground inclination factors

Terzaghi (1943) has its own Nc, Nq, Ngamma and shape coefficients; see the
TERZAGHI section at the end of this file.

References:
    Vesic, A.S. (1973) — Bearing capacity factors and Ngamma
    Meyerhof, G.G. (1963) — Shape, depth, inclination factors
    Hansen, J.B. (1970) — General bearing capacity theory
    Terzaghi, K. (1943) — Theoretical Soil Mechanics, Ch. VIII
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 5: Eqs. 5-2 to 5-8 (pp. 51-53),
        Tables 5-1, 5-2, 5-4, 5-5 (pp. 54-60), Eq. 5-14 (p. 63)
    EN 1997-1:2004 Annex D.4 (p. 159) — Vesic's load inclination factors
    FHWA Soils & Foundations Reference Manual, Volume II, Chapter 8
    UFC 3-220-20 (2025) Foundations, Ch. 5, Tables 5-2 to 5-8
"""

import math
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════
# BEARING CAPACITY FACTORS  Nc, Nq, Ngamma
# ═══════════════════════════════════════════════════════════════════════

def bearing_capacity_Nq(phi_deg: float) -> float:
    """Bearing capacity factor Nq.

    Nq = exp(pi*tan(phi)) * tan²(45 + phi/2)

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Nq factor.

    References
    ----------
    Reissner (1924); Vesic (1973), FHWA GEC-6 Eq. 5-2 (p. 51).
    """
    phi = math.radians(phi_deg)
    return math.exp(math.pi * math.tan(phi)) * math.tan(math.pi / 4 + phi / 2) ** 2


def bearing_capacity_Nc(phi_deg: float) -> float:
    """Bearing capacity factor Nc.

    Nc = (Nq - 1) * cot(phi)  for phi > 0
    Nc = 5.14               for phi = 0 (Prandtl solution)

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Nc factor.

    References
    ----------
    Prandtl (1921); Vesic (1973), FHWA GEC-6 Eqs. 5-3 and 5-4 (p. 52).
    """
    if phi_deg == 0:
        return 5.14  # Exact Prandtl solution for phi=0
    phi = math.radians(phi_deg)
    Nq = bearing_capacity_Nq(phi_deg)
    return (Nq - 1.0) / math.tan(phi)


def bearing_capacity_Ngamma(phi_deg: float, method: str = "vesic") -> float:
    """Bearing capacity factor Ngamma.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    method : str, optional
        Method for Ngamma computation:
        - "vesic" (default): Ngamma = 2*(Nq+1)*tan(phi) — Vesic (1973)
        - "meyerhof": Ngamma = (Nq-1)*tan(1.4*phi) — Meyerhof (1963)
        - "hansen": Ngamma = 1.5*(Nq-1)*tan(phi) — Hansen (1970)

    Returns
    -------
    float
        Ngamma factor. Returns 0 for phi=0.

    References
    ----------
    Vesic (1973), FHWA GEC-6 Eq. 5-8 (p. 53), Table 5-1 (p. 54).
    Meyerhof (1963), Canadian Geotechnical Journal.
    Hansen (1970), Danish Geotechnical Institute Bulletin 28.
    """
    if phi_deg == 0:
        return 0.0

    phi = math.radians(phi_deg)
    Nq = bearing_capacity_Nq(phi_deg)

    method = method.lower()
    if method == "vesic":
        return 2.0 * (Nq + 1.0) * math.tan(phi)
    elif method == "meyerhof":
        return (Nq - 1.0) * math.tan(1.4 * phi)
    elif method == "hansen":
        return 1.5 * (Nq - 1.0) * math.tan(phi)
    elif method in TERZAGHI_NGAMMA_SOURCES:
        raise ValueError(
            f"Ngamma method '{method}' is a Terzaghi (1943) N-gamma; it goes "
            "with Terzaghi's own Nc and Nq (terzaghi_Ngamma, or "
            "factor_method='terzaghi'), not with this Nq.")
    else:
        raise ValueError(f"Unknown Ngamma method '{method}'. Options: 'vesic', 'meyerhof', 'hansen'")


def all_N_factors(phi_deg: float, ngamma_method: str = "vesic") -> Tuple[float, float, float]:
    """Compute all three bearing capacity factors.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    ngamma_method : str, optional
        Method for Ngamma. Default "vesic".

    Returns
    -------
    Nc, Nq, Ngamma : tuple of float
    """
    Nc = bearing_capacity_Nc(phi_deg)
    Nq = bearing_capacity_Nq(phi_deg)
    Ng = bearing_capacity_Ngamma(phi_deg, method=ngamma_method)
    return Nc, Nq, Ng


# ═══════════════════════════════════════════════════════════════════════
# SHAPE FACTORS  sc, sq, sgamma
# ═══════════════════════════════════════════════════════════════════════

def shape_factors(phi_deg: float, B: float, L: float,
                  method: str = "vesic") -> Tuple[float, float, float]:
    """Shape correction factors for non-strip footings.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    B : float
        Effective footing width (m). Must be <= L.
    L : float
        Effective footing length (m).
    method : str, optional
        "vesic" (default) or "meyerhof".

    Returns
    -------
    sc, sq, sgamma : tuple of float
        Shape factors.

    References
    ----------
    Vesic (1973): FHWA GEC-6 Table 5-2 (p. 56).
        sc = 1 + (B/L)*(Nq/Nc)
        sq = 1 + (B/L)*tan(phi)
        sg = 1 - 0.4*(B/L)
    Meyerhof (1963):
        sc = 1 + 0.2*Kp*(B/L)
        sq = sg = 1 + 0.1*Kp*(B/L) for phi>10; else 1.0
    """
    if L <= 0:
        raise ValueError(f"Footing length L must be positive, got {L}")

    BoverL = B / L

    method = method.lower()
    if method == "vesic":
        Nc = bearing_capacity_Nc(phi_deg)
        Nq = bearing_capacity_Nq(phi_deg)
        phi = math.radians(phi_deg)

        if phi_deg == 0:
            sc = 1.0 + 0.2 * BoverL  # Vesic for phi=0: sc = 1 + 0.2*(B/L)
        else:
            sc = 1.0 + BoverL * (Nq / Nc)
        sq = 1.0 + BoverL * math.tan(phi)
        sg = 1.0 - 0.4 * BoverL
        sg = max(sg, 0.6)  # Minimum per FHWA guidance

    elif method == "meyerhof":
        Kp = math.tan(math.pi / 4 + math.radians(phi_deg) / 2) ** 2
        sc = 1.0 + 0.2 * Kp * BoverL
        if phi_deg > 10:
            sq = 1.0 + 0.1 * Kp * BoverL
            sg = sq
        else:
            sq = 1.0
            sg = 1.0

    else:
        raise ValueError(f"Unknown shape factor method '{method}'. Options: 'vesic', 'meyerhof'")

    return sc, sq, sg


# ═══════════════════════════════════════════════════════════════════════
# DEPTH FACTORS  dc, dq, dgamma
# ═══════════════════════════════════════════════════════════════════════

#: ``depth_method`` values (GM-197): how the depth factors are worked out.
#: "hansen": Brinch Hansen's (1970) depth parameter k in the Vesic or
#: Meyerhof factors; "gec6": GEC-6's d_q alone (Eq. 5-14, Table 5-4);
#: "none": no depth factors.
DEPTH_FACTOR_METHODS = ("hansen", "gec6", "none")

#: Where each depth-factor method is printed.
DEPTH_FACTOR_BASIS = {
    "hansen": ("Brinch Hansen (1970): k = Df/B for Df/B < 1, k = "
               "arctan(Df/B) for Df/B >= 1, UFC 3-220-20 Tables 5-7 and "
               "5-8 note B (pp. 295-296); d_q = 1 + 2 k tan phi (1 - sin "
               "phi)^2, Table 5-8; d_c = d_q - (1 - d_q)/(N_c tan phi), "
               "the correspondence formula of its note C (Brinch Hansen "
               "1961, de Beer 1970), = 1 + 2 k (1 - sin phi)^2 N_q/N_c; "
               "d_gamma = 1. Table 5-8 prints d_c = 1 + k (1 - sin phi)^2 "
               "N_q/N_c, half of that, while its printed s_c and i_c follow "
               "the note C formula exactly; the formula's d_c tends to 1 + "
               "0.39 k as phi goes to 0, Table 5-7's 1 + 0.4 k, the printed "
               "one to 1 + 0.19 k"),
    "gec6": ("FHWA GEC-6 Eq. 5-14 (p. 63): the depth factor on the "
             "surcharge term alone (d_c = d_gamma = 1); d_q of Table 5-4 "
             "(Brinch Hansen 1970, p. 57) for Df/B from 1, which k = "
             "arctan(Df/B) reproduces; d_q = 1.0 for Df/B < 1, as the "
             "worked example on p. B-67 takes it"),
    "none": ("no depth factors: GEC-6 Sec. 5.2.3.3 (p. 57) omits them "
             "unless the cover over the footing is high-quality compacted "
             "granular material sure to stay in place, and Table 5-4's "
             "note takes d_q = 1.0 unless the soil above the bearing level "
             "is as competent as the soil below"),
}


def depth_parameter_k(Df: float, B: float, method: str = "vesic") -> float:
    """Brinch Hansen's (1970) depth parameter k (radians beyond Df/B = 1).

    For the Vesic factors k = Df/B for Df/B < 1 and arctan(Df/B) for
    Df/B >= 1, as UFC 3-220-20 Tables 5-7 and 5-8 note B (pp. 295-296)
    print it; GEC-6 Table 5-4 (p. 57) prints d_q at Df/B = 1 from the
    arctan too (1.20 at 32 deg, not the 1.30 of k = 1). Meyerhof's factors
    take k = Df/B as UFC 3-220-20 Table 5-6 (p. 294) prints it up to
    Df/B = 1 and Hansen's arctan beyond it.
    """
    if B <= 0:
        raise ValueError(f"Footing width B must be positive, got {B}")
    df_b = Df / B
    linear = df_b <= 1.0 if str(method).lower() == "meyerhof" else df_b < 1.0
    return df_b if linear else math.atan(df_b)


def ufc_table_5_8_printed_dc(phi_deg: float, Df: float, B: float) -> float:
    """d_c as UFC 3-220-20 Table 5-8 (p. 296) prints it for phi' > 0:
    1 + k (1 - sin phi)^2 N_q/N_c, half of what its note C correspondence
    formula gives from the same d_q (GM-214). Reported beside the factor
    :func:`depth_factors` uses; not used in the capacity."""
    if phi_deg <= 0:
        raise ValueError("Table 5-8 is for phi' > 0; Table 5-7 gives d_c = "
                         "0.4 k for phi = 0")
    phi = math.radians(phi_deg)
    k = depth_parameter_k(Df, B)
    return 1.0 + k * (1.0 - math.sin(phi)) ** 2 * (
        bearing_capacity_Nq(phi_deg) / bearing_capacity_Nc(phi_deg))


def depth_factors(phi_deg: float, Df: float, B: float,
                  method: str = "vesic",
                  depth_method: str = "hansen") -> Tuple[float, float, float]:
    """Depth correction factors for embedded footings.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    Df : float
        Footing embedment depth (m).
    B : float
        Effective footing width (m).
    method : str, optional
        "vesic" (default) or "meyerhof".
    depth_method : str, optional
        One of :data:`DEPTH_FACTOR_METHODS` (basis in
        :data:`DEPTH_FACTOR_BASIS`): "hansen" (default), "gec6" (with
        ``method="vesic"`` only) or "none".

    Returns
    -------
    dc, dq, dgamma : tuple of float
        Depth factors.

    References
    ----------
    Depth parameter k: :func:`depth_parameter_k`.

    Vesic/Hansen: UFC 3-220-20 Table 5-8 (p. 296); GEC-6 Table 5-4 (p. 57).
        dq = 1 + 2*tan(phi)*(1-sin(phi))² * k
        dc = dq - (1-dq)/(Nc*tan(phi))   for phi > 0 (Table 5-8 note C)
        dc = 1 + 0.4*k                    for phi = 0 (UFC Table 5-7)
        dgamma = 1.0
    Table 5-8's printed d_c is half of its note C formula's
    (:func:`ufc_table_5_8_printed_dc`); the note C form is used (GM-214).

    GEC-6 (depth_method="gec6"): Eq. 5-14 (p. 63) and Table 5-4.
        dq = 1 + 2*tan(phi)*(1-sin(phi))² * arctan(Df/B)  for Df/B >= 1
        dq = 1.0                                           for Df/B < 1
        dc = dgamma = 1.0

    Meyerhof (1963): UFC 3-220-20 Table 5-6 (p. 294).
        dc = 1 + 0.2*sqrt(Kp) * k
        dq = dg = 1 + 0.1*sqrt(Kp) * k   for phi > 10
        dq = dg = 1.0                     for phi <= 10
    """
    if B <= 0:
        raise ValueError(f"Footing width B must be positive, got {B}")

    DfoverB = Df / B
    method = method.lower()
    if method not in ("vesic", "meyerhof"):
        raise ValueError(f"Unknown depth factor method '{method}'. Options: 'vesic', 'meyerhof'")
    depth_method = str(depth_method).strip().lower()
    if depth_method not in DEPTH_FACTOR_METHODS:
        raise ValueError(
            f"Unknown depth_method '{depth_method}'. Options: "
            f"{list(DEPTH_FACTOR_METHODS)}")
    if depth_method == "none":
        return 1.0, 1.0, 1.0
    if depth_method == "gec6":
        if method != "vesic":
            raise ValueError(
                "depth_method 'gec6' is GEC-6's depth factor (Eq. 5-14, "
                "Table 5-4), which goes with its own factors: method "
                "'vesic'")
        if DfoverB < 1.0:
            return 1.0, 1.0, 1.0
        phi = math.radians(phi_deg)
        dq = (1.0 + 2.0 * math.tan(phi) * (1.0 - math.sin(phi)) ** 2
              * math.atan(DfoverB))
        return 1.0, dq, 1.0

    k = depth_parameter_k(Df, B, method)

    if method == "vesic":
        phi = math.radians(phi_deg)

        if phi_deg == 0:
            dc = 1.0 + 0.4 * k
            dq = 1.0
        else:
            dq = 1.0 + 2.0 * math.tan(phi) * (1.0 - math.sin(phi)) ** 2 * k
            Nc = bearing_capacity_Nc(phi_deg)
            dc = dq - (1.0 - dq) / (Nc * math.tan(phi))

        dg = 1.0  # Vesic: dgamma = 1.0 always

    else:  # meyerhof
        Kp = math.tan(math.pi / 4 + math.radians(phi_deg) / 2) ** 2
        dc = 1.0 + 0.2 * math.sqrt(Kp) * k
        if phi_deg > 10:
            dq = 1.0 + 0.1 * math.sqrt(Kp) * k
            dg = dq
        else:
            dq = 1.0
            dg = 1.0

    return dc, dq, dg


# ═══════════════════════════════════════════════════════════════════════
# INCLINATION FACTORS  ic, iq, igamma
# ═══════════════════════════════════════════════════════════════════════

def _meyerhof_inclination_factors(phi_deg: float,
                                  beta_deg: float) -> Tuple[float, float, float]:
    """Angle-based (Meyerhof, 1963) load inclination factors, UFC 3-220-20
    Table 5-6 (p. 294).

        ic = iq = (1 - beta/90)^2
        ig = (1 - beta/phi)^2   (phi > 0), else 1.0 (N_gamma = 0)

    The table takes theta = atan(H/V) < phi' (its note B), so beta >= phi
    is refused (GM-214): i_gamma is 0 at beta = phi and grows again beyond
    it, to 1 at beta = 2 phi; a clamp at 0 had let that through.
    """
    if phi_deg > 0 and beta_deg >= phi_deg:
        raise ValueError(
            f"Meyerhof's load inclination factors are for a load inclined "
            f"less than phi from vertical: UFC 3-220-20 Table 5-6 (p. 294) "
            f"takes theta = atan(H/V) < phi'. Got {beta_deg:g} deg with phi "
            f"= {phi_deg:g} deg. At theta = phi, i_gamma = (1 - theta/phi)^2 "
            "is 0, and beyond it the formula grows again. Check sliding "
            "for this load.")
    ic = (1.0 - beta_deg / 90.0) ** 2
    iq = ic
    if phi_deg > 0:
        ig = (1.0 - beta_deg / phi_deg) ** 2
    else:
        ig = 1.0
    return ic, iq, ig


def vesic_inclination_needs_V(phi_deg: float, c: float) -> bool:
    """Whether Vesic's load inclination factors need the vertical load V.

    They take H/(V + A' c cot phi) (phi > 0) or H/(A' c N_c) (phi = 0). With
    c = 0 and phi > 0 that is H/V = tan(beta): V cancels and the angle alone
    gives the factors. Otherwise V is needed (GM-203).
    """
    return not (phi_deg > 0 and c == 0)


def inclination_factors(phi_deg: float, beta_deg: float, c: float,
                        B: float, L: float, V: float,
                        method: str = "vesic",
                        A: float = None) -> Tuple[float, float, float]:
    """Load inclination factors for inclined loading.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    beta_deg : float
        Inclination angle of load from vertical (degrees).
        0 = purely vertical load. 0 to below 90; Meyerhof's factors below
        phi (UFC 3-220-20 Table 5-6).
    c : float
        Cohesion (kPa).
    B : float
        Effective footing width (m).
    L : float
        Effective footing length (m).
    V : float
        Total vertical load (kN; kN/m for a strip). Vesic's factors need it
        unless c = 0 (:func:`vesic_inclination_needs_V`); without it they
        raise rather than fall back to another method's factors (GM-203).
    method : str, optional
        "vesic" (default) or "meyerhof".
    A : float, optional
        Effective base area A' in the adhesion term (m²; m²/m, i.e. B, for a
        strip). Default B * L.

    Returns
    -------
    ic, iq, igamma : tuple of float
        Inclination factors.

    References
    ----------
    Vesic (1975), as EN 1997-1:2004 Annex D.4 (p. 159) prints it:
        iq = [1 - H/(V + A' c cot phi)]^m, ig = [...]^(m+1),
        ic = iq - (1 - iq)/(Nc tan phi),
        m = mB = (2 + B'/L')/(1 + B'/L') (H along B).
    phi = 0: ic = 1 - m H/(A' c Nc), Vesic (1975). EN 1997-1 does not
        print it: its D.3 (p. 158) gives ic = [1 + sqrt(1 - H/(A' c))]/2,
        as UFC 3-220-20 Table 5-7 (p. 295) does in its additive form.
    GEC-6 Sec. 5.2.3.7 (p. 62) omits load inclination factors where
    sliding is checked separately.
    Meyerhof (1963), UFC 3-220-20 Table 5-6 (p. 294):
        ic = iq = (1 - beta/90)², ig = (1 - beta/phi)², beta < phi
    beta must be 0 or more and below 90 deg: a negative angle gave factors
    above 1 (GM-214).
    """
    if not 0.0 <= beta_deg < 90.0:
        raise ValueError(
            f"load_inclination is the load's angle from vertical, 0 to below "
            f"90 deg; got {beta_deg:g}. The inclination factors take its "
            "size (H/V), not its direction; a negative angle gave factors "
            "above 1.")
    if beta_deg == 0:
        return 1.0, 1.0, 1.0

    beta = math.radians(beta_deg)

    method = method.lower()
    if method == "vesic":
        phi = math.radians(phi_deg)
        # m for H along B (EN 1997-1 Annex D.4): 2 for a strip
        m = (2.0 + B / L) / (1.0 + B / L)
        if not vesic_inclination_needs_V(phi_deg, c):
            # c = 0: H/V = tan(beta), so V cancels
            ratio = 1.0 - math.tan(beta)
        else:
            if V is None or V <= 0:
                raise ValueError(
                    "Vesic's load inclination factors take H/(V + A' c cot "
                    "phi) (EN 1997-1 Annex D.4), or H/(A' c N_c) at phi = "
                    "0, so with cohesion they need "
                    f"the vertical load V; got V = {V}. Give vertical_load "
                    "(kN; kN/m for a strip), or factor_method 'meyerhof' for "
                    "the angle-only factors of UFC 3-220-20 Table 5-6.")
            H = V * math.tan(beta)
            Af = B * L if A is None else A
            if phi_deg == 0:
                Nc = bearing_capacity_Nc(0.0)
                ic = 1.0 - m * H / (Af * c * Nc)
                if ic < 0.0:
                    raise ValueError(
                        f"H = V tan(beta) = {H:.1f} kN exceeds A' c Nc / m = "
                        f"{Af * c * Nc / m:.1f} kN: Vesic's i_c is negative; "
                        "the footing slides before it fails in bearing "
                        "(check sliding).")
                return ic, 1.0, 1.0
            ratio = 1.0 - H / (V + Af * c / math.tan(phi))
        if ratio < 0.0:
            raise ValueError(
                f"The load inclination ({beta_deg} deg) gives H more than "
                "V + A' c cot phi: Vesic's inclination factors are "
                "undefined; the footing slides before it fails in bearing "
                "(check sliding).")
        iq = ratio ** m
        ig = ratio ** (m + 1)
        Nc = bearing_capacity_Nc(phi_deg)
        ic = iq - (1.0 - iq) / (Nc * math.tan(phi))

    elif method == "meyerhof":
        ic, iq, ig = _meyerhof_inclination_factors(phi_deg, beta_deg)

    else:
        raise ValueError(f"Unknown inclination method '{method}'. Options: 'vesic', 'meyerhof'")

    return ic, iq, ig


# ═══════════════════════════════════════════════════════════════════════
# BASE INCLINATION FACTORS  bc, bq, bgamma
# ═══════════════════════════════════════════════════════════════════════

def base_inclination_factors(phi_deg: float,
                             alpha_deg: float) -> Tuple[float, float, float]:
    """Base inclination factors for tilted footings.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    alpha_deg : float
        Base tilt angle from horizontal (degrees).

    Returns
    -------
    bc, bq, bgamma : tuple of float
        Base inclination factors.

    References
    ----------
    Hansen (1970); FHWA GEC-6 Table 5-5 (p. 60).
        bq = bgamma = (1 - alpha*tan(phi))²
        bc = bq - (1-bq)/(Nc*tan(phi))  for phi > 0
        bc = 1 - 2*alpha/(pi+2)          for phi = 0
    alpha in radians for the equations; upward + (GEC-6 Table 5-5).
    Refused where alpha tan(phi) passes 1: b_q is 0 there and grows again
    beyond it (GM-214).
    """
    if alpha_deg == 0:
        return 1.0, 1.0, 1.0

    alpha = math.radians(alpha_deg)
    phi = math.radians(phi_deg)
    if alpha * math.tan(phi) > 1.0:
        raise ValueError(
            f"Base tilt {alpha_deg:g} deg with phi = {phi_deg:g} deg gives "
            "alpha tan(phi) above 1 (alpha in radians): b_q = (1 - alpha "
            "tan phi)^2 is 0 at 1 and grows again beyond it. GEC-6 Sec. "
            "5.2.3.5 (p. 58) limits inclined footings to about 8 to 10 deg.")

    if phi_deg == 0:
        bc = 1.0 - 2.0 * alpha / (math.pi + 2.0)
        bq = 1.0
        bg = 1.0
    else:
        bq = (1.0 - alpha * math.tan(phi)) ** 2
        bg = bq
        Nc = bearing_capacity_Nc(phi_deg)
        bc = bq - (1.0 - bq) / (Nc * math.tan(phi))

    return bc, bq, bg


# ═══════════════════════════════════════════════════════════════════════
# GROUND INCLINATION FACTORS  gc, gq, ggamma
# ═══════════════════════════════════════════════════════════════════════

def ground_inclination_factors(phi_deg: float,
                               omega_deg: float) -> Tuple[float, float, float]:
    """Ground surface inclination factors for footings on slopes.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    omega_deg : float
        Ground surface slope angle from horizontal (degrees).

    Returns
    -------
    gc, gq, ggamma : tuple of float
        Ground inclination factors.

    References
    ----------
    Vesic (1975). (GEC-6 Sec. 5.2.3.4, p. 58, uses the sloping-ground
    factors Ncq and Ngq of its Fig. 5-7 instead.)
        gq = ggamma = (1 - tan(omega))²
        gc = gq - (1-gq)/(Nc*tan(phi))  for phi > 0
        gc = 1 - 2*omega/(pi+2)          for phi = 0
    omega in radians for the equations. Refused below 0 (the factors
    went above 1) and above 45 deg, where g_q = (1 - tan omega)^2 is 0
    and grows again beyond it (GM-214).
    """
    if not 0.0 <= omega_deg <= 45.0:
        raise ValueError(
            f"ground_slope is the slope's angle down from horizontal, 0 to "
            f"45 deg here; got {omega_deg:g}. Vesic's g_q = (1 - tan "
            "omega)^2 is 0 at 45 deg and grows again beyond it; a negative "
            "angle gave factors above 1. For a footing on or near a slope "
            "GEC-6 Sec. 5.2.3.4 (p. 58) uses N_cq and N_gamma-q (Fig. 5-7), "
            "and the slope's own stability is checked.")
    if omega_deg == 0:
        return 1.0, 1.0, 1.0

    omega = math.radians(omega_deg)
    phi = math.radians(phi_deg)

    gq = (1.0 - math.tan(omega)) ** 2
    gg = gq

    if phi_deg == 0:
        gc = 1.0 - 2.0 * omega / (math.pi + 2.0)
    else:
        Nc = bearing_capacity_Nc(phi_deg)
        gc = gq - (1.0 - gq) / (Nc * math.tan(phi))

    return gc, gq, gg


# ═══════════════════════════════════════════════════════════════════════
# TERZAGHI (1943)
# ═══════════════════════════════════════════════════════════════════════
#
#   strip     qu = c Nc       + q Nq + 0.5 gamma B Ngamma
#   square    qu = 1.3 c Nc   + q Nq + 0.4 gamma B Ngamma
#   circular  qu = 1.3 c Nc   + q Nq + 0.3 gamma B Ngamma   (B = diameter)
#
# Nc and Nq are closed form. Terzaghi gave Ngamma only as a chart (values at
# phi = 0, 34 and 48 deg; UFC 3-220-20 Table 5-2 note), through the passive
# coefficient K_p-gamma found graphically, so every Ngamma "for Terzaghi" is
# someone's evaluation of it. Two are offered, each cited:
#
#   "kumbhojkar"  Kumbhojkar, A.S. (1993). "Numerical evaluation of
#                 Terzaghi's N-gamma." J. Geotech. Eng., ASCE, 119(3),
#                 598-607 -- Terzaghi's own mechanism solved numerically.
#                 Values as tabulated (phi = 0-50 deg, 1 deg steps) in Das,
#                 Principles of Foundation Engineering, table of Terzaghi's
#                 bearing capacity factors. DEFAULT.
#   "coduto"      Coduto et al. (2016) approximation
#                 Ngamma = 2 (Nq + 1) tan(phi) / (1 + 0.4 sin 4phi),
#                 the basis of the Terzaghi column of UFC 3-220-20 (2025)
#                 Table 5-2.
#
# They differ by 5 % at phi = 30 (19.13 vs 20.1) and by a factor of ~2 below
# 10 deg, where Ngamma is small anyway.

#: Kumbhojkar (1993) Terzaghi Ngamma at phi = 0, 1, ..., 50 deg (as tabulated
#: in Das, Principles of Foundation Engineering).
_KUMBHOJKAR_NGAMMA = (
    0.00, 0.01, 0.04, 0.06, 0.10, 0.14, 0.20, 0.27, 0.35, 0.44,      # 0-9
    0.56, 0.69, 0.85, 1.04, 1.26, 1.52, 1.82, 2.18, 2.59, 3.07,      # 10-19
    3.64, 4.31, 5.09, 6.00, 7.08, 8.34, 9.84, 11.60, 13.70, 16.18,   # 20-29
    19.13, 22.65, 26.87, 31.94, 38.04, 45.41, 54.36, 65.27, 78.61,   # 30-38
    95.03, 115.31, 140.51, 171.99, 211.56, 261.60, 325.34, 407.11,   # 39-46
    512.84, 650.67, 831.99, 1072.80,                                 # 47-50
)

#: Terzaghi N-gamma sources (``terzaghi_Ngamma(source=...)``) and their
#: citations, as reported in results.
TERZAGHI_NGAMMA_SOURCES: Dict[str, str] = {
    "kumbhojkar": ("Kumbhojkar (1993), numerical evaluation of Terzaghi's "
                   "N-gamma, as tabulated in Das, Principles of Foundation "
                   "Engineering (phi = 0-50 deg; log-linear between whole "
                   "degrees)"),
    "coduto": ("Coduto et al. (2016) approximation "
               "2(Nq+1)tan(phi)/(1+0.4 sin 4phi), as used for UFC 3-220-20 "
               "(2025) Table 5-2"),
}

#: Terzaghi's local-shear reduction: c* = 2/3 c, tan(phi*) = 2/3 tan(phi)
#: (UFC 3-220-20 Table 5-4 prints 0.67).
TERZAGHI_LOCAL_SHEAR = 2.0 / 3.0


def terzaghi_Nq(phi_deg: float) -> float:
    """Terzaghi (1943) Nq = a² / (2 cos²(45° + φ/2)), a = e^((3π/4 − φ/2) tan φ).

    References
    ----------
    Terzaghi (1943); UFC 3-220-20 (2025) Table 5-2.
    """
    if phi_deg == 0:
        return 1.0
    phi = math.radians(phi_deg)
    a = math.exp((0.75 * math.pi - phi / 2.0) * math.tan(phi))
    return a * a / (2.0 * math.cos(math.radians(45.0 + phi_deg / 2.0)) ** 2)


def terzaghi_Nc(phi_deg: float) -> float:
    """Terzaghi (1943) Nc = (Nq − 1) cot φ; 1.5π + 1 = 5.71 at φ = 0 (the
    limit, printed 5.7)."""
    if phi_deg == 0:
        return 1.5 * math.pi + 1.0
    return (terzaghi_Nq(phi_deg) - 1.0) / math.tan(math.radians(phi_deg))


def terzaghi_Ngamma(phi_deg: float, source: str = "kumbhojkar") -> float:
    """Terzaghi's Ngamma by one of ``TERZAGHI_NGAMMA_SOURCES``.

    ``"kumbhojkar"`` reads the 1-degree table (exact at whole degrees,
    log-linear between them, linear below 1 degree) and covers 0-50 deg;
    ``"coduto"`` is the closed-form approximation.
    """
    source = str(source).strip().lower()
    if source not in TERZAGHI_NGAMMA_SOURCES:
        raise ValueError(
            f"Unknown Terzaghi Ngamma source '{source}'. Options: "
            f"{list(TERZAGHI_NGAMMA_SOURCES)}")
    if phi_deg < 0:
        raise ValueError(f"Friction angle must be >= 0, got {phi_deg}")
    if phi_deg == 0:
        return 0.0
    if source == "coduto":
        phi = math.radians(phi_deg)
        return (2.0 * (terzaghi_Nq(phi_deg) + 1.0) * math.tan(phi)
                / (1.0 + 0.4 * math.sin(4.0 * phi)))
    table = _KUMBHOJKAR_NGAMMA
    if phi_deg > len(table) - 1:
        raise ValueError(
            f"The Kumbhojkar (1993) table ends at {len(table) - 1} deg, got "
            f"phi = {phi_deg}; use ngamma_method='coduto' beyond it.")
    i = int(math.floor(phi_deg))
    if i == phi_deg:
        return table[i]
    t = phi_deg - i
    lo, hi = table[i], table[i + 1]
    if lo <= 0.0:
        return lo + t * (hi - lo)
    return lo * (hi / lo) ** t


def terzaghi_shape_factors(shape: str, B: float,
                           L: float) -> Tuple[float, float, float, str]:
    """Terzaghi shape coefficients (sc, sq, sγ) on c·Nc, q·Nq, 0.5·γ·B·Nγ.

    Strip 1, 1, 1; square 1.3, 1, 0.8; circular 1.3, 1, 0.6 (UFC 3-220-20
    Table 5-5: 1.3 c Nc and 0.4 / 0.3 γ B Nγ). Terzaghi gave no rectangle;
    UFC 3-220-20 Sec 5-3.3.1 allows interpolating between the square and
    continuous factors, done here linearly in B/L:
    sc = 1 + 0.3 B/L, sγ = 1 − 0.2 B/L. Returns a note when it interpolates.
    """
    shape = str(shape).strip().lower()
    if shape == "strip":
        return 1.0, 1.0, 1.0, ""
    if shape == "circular":
        return 1.3, 1.0, 0.6, ""
    if shape not in ("square", "rectangular"):
        raise ValueError(f"Unknown footing shape '{shape}'")
    if L <= 0:
        raise ValueError(f"Footing length L must be positive, got {L}")
    r = min(B / L, 1.0)
    if r == 1.0:
        return 1.3, 1.0, 0.8, ""
    note = (f"Terzaghi (1943) gives strip, square and circular footings only; "
            f"for B/L = {r:.3f} the shape factors are interpolated linearly "
            f"between continuous and square (UFC 3-220-20 Sec 5-3.3.1): "
            f"sc = 1 + 0.3 B/L = {1 + 0.3 * r:.3f}, "
            f"s_gamma = 1 - 0.2 B/L = {1 - 0.2 * r:.3f}.")
    return 1.0 + 0.3 * r, 1.0, 1.0 - 0.2 * r, note


def terzaghi_local_shear(c: float, phi_deg: float) -> Tuple[float, float]:
    """Terzaghi's local-shear strengths: c* = 2/3 c, φ* = atan(2/3 tan φ).

    References
    ----------
    Terzaghi (1943); UFC 3-220-20 (2025) Table 5-4 (printed as 0.67).
    """
    k = TERZAGHI_LOCAL_SHEAR
    return (k * c,
            math.degrees(math.atan(k * math.tan(math.radians(phi_deg)))))
