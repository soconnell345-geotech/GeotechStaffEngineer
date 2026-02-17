"""
Bearing capacity factors and correction factors.

Implements the general bearing capacity equation factors:
    Nc, Nq, Ngamma — bearing capacity factors (functions of phi)
    sc, sq, sg — shape factors
    dc, dq, dg — depth factors
    ic, iq, ig — load inclination factors
    bc, bq, bg — base inclination factors
    gc, gq, gg — ground inclination factors

References:
    Vesic, A.S. (1973) — Bearing capacity factors and Ngamma
    Meyerhof, G.G. (1963) — Shape, depth, inclination factors
    Hansen, J.B. (1970) — General bearing capacity theory
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 6, Tables 6-1 through 6-6
    FHWA Soils & Foundations Reference Manual, Volume II, Chapter 8
"""

import math
from typing import Tuple


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
    Reissner (1924); Vesic (1973), FHWA GEC-6 Eq. 6-2.
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
    Prandtl (1921); Vesic (1973), FHWA GEC-6 Eq. 6-1.
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
    Vesic (1973), FHWA GEC-6 Table 6-1.
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
    Vesic (1973): FHWA GEC-6 Table 6-2.
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

def depth_factors(phi_deg: float, Df: float, B: float,
                  method: str = "vesic") -> Tuple[float, float, float]:
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

    Returns
    -------
    dc, dq, dgamma : tuple of float
        Depth factors.

    References
    ----------
    Depth parameter k (Hansen, 1970; FHWA GEC-6 Table 6-3):
        If Df/B <= 1:  k = Df/B
        If Df/B > 1:   k = arctan(Df/B) (radians)
        k is used by both Vesic and Meyerhof methods to bound depth
        factors for deeply embedded footings (Das, 2019).

    Vesic/Hansen: FHWA GEC-6 Table 6-3.
        dq = 1 + 2*tan(phi)*(1-sin(phi))² * k
        dc = dq - (1-dq)/(Nc*tan(phi))   for phi > 0
        dc = 1 + 0.4*k                    for phi = 0
        dgamma = 1.0

    Meyerhof (1963):
        dc = 1 + 0.2*sqrt(Kp) * k
        dq = dg = 1 + 0.1*sqrt(Kp) * k   for phi > 10
        dq = dg = 1.0                     for phi <= 10
    """
    if B <= 0:
        raise ValueError(f"Footing width B must be positive, got {B}")

    DfoverB = Df / B

    # Depth parameter k (Hansen, 1970; FHWA GEC-6 Table 6-3):
    #   k = Df/B           for Df/B <= 1
    #   k = arctan(Df/B)   for Df/B > 1  (radians)
    # Applied to both Vesic and Meyerhof methods to bound depth factors
    # for deeply embedded footings.
    if DfoverB <= 1.0:
        k = DfoverB
    else:
        k = math.atan(DfoverB)  # radians

    method = method.lower()
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

    elif method == "meyerhof":
        Kp = math.tan(math.pi / 4 + math.radians(phi_deg) / 2) ** 2
        dc = 1.0 + 0.2 * math.sqrt(Kp) * k
        if phi_deg > 10:
            dq = 1.0 + 0.1 * math.sqrt(Kp) * k
            dg = dq
        else:
            dq = 1.0
            dg = 1.0

    else:
        raise ValueError(f"Unknown depth factor method '{method}'. Options: 'vesic', 'meyerhof'")

    return dc, dq, dg


# ═══════════════════════════════════════════════════════════════════════
# INCLINATION FACTORS  ic, iq, igamma
# ═══════════════════════════════════════════════════════════════════════

def inclination_factors(phi_deg: float, beta_deg: float, c: float,
                        B: float, L: float, V: float,
                        method: str = "vesic") -> Tuple[float, float, float]:
    """Load inclination factors for inclined loading.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).
    beta_deg : float
        Inclination angle of load from vertical (degrees).
        0 = purely vertical load. Range [0, phi].
    c : float
        Cohesion (kPa).
    B : float
        Effective footing width (m).
    L : float
        Effective footing length (m).
    V : float
        Total vertical load (kN). Used in Vesic formulation.
    method : str, optional
        "vesic" (default) or "meyerhof".

    Returns
    -------
    ic, iq, igamma : tuple of float
        Inclination factors.

    References
    ----------
    Vesic (1975): FHWA GEC-6 Table 6-4.
    Meyerhof (1963): ic = iq = (1 - beta/90)², ig = (1 - beta/phi)²
    """
    if beta_deg == 0:
        return 1.0, 1.0, 1.0

    beta = math.radians(beta_deg)

    method = method.lower()
    if method == "vesic":
        # Vesic (1975) formulation
        # H = V * tan(beta) = horizontal component
        H = V * math.tan(beta)
        Af = B * L  # footing base area
        phi = math.radians(phi_deg)

        # m factor depends on load direction relative to B and L
        # For load parallel to B: mB = (2 + B/L)/(1 + B/L)
        # For load parallel to L: mL = (2 + L/B)/(1 + L/B)
        # Using mB by default (load inclined along B direction)
        m = (2.0 + B / L) / (1.0 + B / L)

        if phi_deg == 0:
            # Special case: phi = 0
            Nc = 5.14
            ic = 1.0 - m * H / (Af * c * Nc)
            iq = 1.0
            ig = 1.0
        else:
            iq = (1.0 - H / (V + Af * c / math.tan(phi))) ** m
            ig = (1.0 - H / (V + Af * c / math.tan(phi))) ** (m + 1)
            Nc = bearing_capacity_Nc(phi_deg)
            Nq = bearing_capacity_Nq(phi_deg)
            ic = iq - (1.0 - iq) / (Nc * math.tan(phi))

    elif method == "meyerhof":
        ic = (1.0 - beta_deg / 90.0) ** 2
        iq = ic
        if phi_deg > 0:
            ig = (1.0 - beta_deg / phi_deg) ** 2
            ig = max(ig, 0.0)
        else:
            ig = 1.0

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
    Hansen (1970); FHWA GEC-6 Table 6-5.
        bq = bgamma = (1 - alpha*tan(phi))²
        bc = bq - (1-bq)/(Nc*tan(phi))  for phi > 0
        bc = 1 - 2*alpha/(pi+2)          for phi = 0
    alpha in radians for the equations.
    """
    if alpha_deg == 0:
        return 1.0, 1.0, 1.0

    alpha = math.radians(alpha_deg)
    phi = math.radians(phi_deg)

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
    Hansen (1970); FHWA GEC-6 Table 6-6.
        gq = ggamma = (1 - tan(omega))²
        gc = gq - (1-gq)/(Nc*tan(phi))  for phi > 0
        gc = 1 - 2*omega/(pi+2)          for phi = 0
    omega in radians for the equations.
    """
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
