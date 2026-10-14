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
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 6, Tables 6-1 through 6-6
    FHWA Soils & Foundations Reference Manual, Volume II, Chapter 8
    UFC 3-220-20 (2025) Foundations, Ch. 5, Tables 5-2 to 5-5
"""

import math
import warnings
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

def _meyerhof_inclination_factors(phi_deg: float,
                                  beta_deg: float) -> Tuple[float, float, float]:
    """Angle-based (Meyerhof, 1963) load inclination factors.

    These depend only on the load inclination angle ``beta`` (and ``phi`` for
    ``ig``), not on the H/V force ratio, so they remain usable when the vertical
    load is unknown.

        ic = iq = (1 - beta/90)^2
        ig = (1 - beta/phi)^2   (phi > 0; clamped >= 0), else 1.0
    """
    ic = (1.0 - beta_deg / 90.0) ** 2
    iq = ic
    if phi_deg > 0:
        ig = max((1.0 - beta_deg / phi_deg) ** 2, 0.0)
    else:
        ig = 1.0
    return ic, iq, ig


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
        # The Vesic (1975) factors are written in terms of the actual horizontal
        # (H) and vertical (V) load components, not the inclination angle alone.
        # With no vertical load supplied we cannot form the H/V ratio, so the
        # inclination would silently vanish (H = V*tan(beta) = 0, giving i = 1).
        # Fall back to the angle-based Meyerhof factors — and warn — so a
        # specified load_inclination still reduces capacity.
        if V <= 0:
            warnings.warn(
                "Vesic inclination factors need a positive vertical_load (V) to "
                f"form the H/V ratio; got V={V}. Using angle-based Meyerhof "
                "inclination factors instead so load_inclination still applies "
                "(pass vertical_load for the full Vesic formulation).",
                stacklevel=2,
            )
            return _meyerhof_inclination_factors(phi_deg, beta_deg)

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
