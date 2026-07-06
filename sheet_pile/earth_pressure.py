"""
Earth pressure calculations for sheet pile wall design.

Implements Rankine and Coulomb active/passive earth pressure coefficients
and lateral pressure distributions for layered soils with water.

All units are SI: kPa, kN/m³, degrees, meters.

References:
    Rankine, W.J.M. (1857) — On the stability of loose earth
    Coulomb, C.A. (1776) — Essai sur une application des regles de maximis
    USACE EM 1110-2-2504, Chapter 3
    Das, B.M., "Principles of Foundation Engineering", Ch 7
"""

import math
import warnings
from typing import Optional


def rankine_Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient.

    Ka = tan²(45° - phi/2)

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).

    Returns
    -------
    float
        Active earth pressure coefficient Ka.

    References
    ----------
    Rankine (1857); USACE EM 1110-2-2504, Eq 3-1.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 - phi_rad / 2) ** 2


def rankine_Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient.

    Kp = tan²(45° + phi/2)

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).

    Returns
    -------
    float
        Passive earth pressure coefficient Kp.

    References
    ----------
    Rankine (1857); USACE EM 1110-2-2504, Eq 3-2.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi_rad / 2) ** 2


def coulomb_Ka(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb active earth pressure coefficient.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float, optional
        Wall friction angle (degrees). Default 0 (smooth wall = Rankine).
    alpha_deg : float, optional
        Wall inclination from horizontal (degrees). Default 90 (vertical).
    beta_deg : float, optional
        Backfill slope angle (degrees). Default 0 (horizontal).

    Returns
    -------
    float
        Active earth pressure coefficient Ka.

    References
    ----------
    Coulomb (1776); USACE EM 1110-2-2504, Eq 3-3.
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    sin_alpha = math.sin(alpha)
    num = math.sin(alpha + phi) ** 2
    term1 = math.sin(alpha) ** 2 * math.sin(alpha - delta)
    term2 = math.sin(phi + delta) * math.sin(phi - beta)
    term3 = math.sin(alpha - delta) * math.sin(alpha + beta)

    if term3 <= 0:
        raise ValueError("Invalid geometry for Coulomb Ka calculation")

    denom = term1 * (1 + math.sqrt(term2 / term3)) ** 2
    return num / denom


def coulomb_Kp(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb passive earth pressure coefficient.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float, optional
        Wall friction angle (degrees). Default 0.
    alpha_deg : float, optional
        Wall inclination from horizontal (degrees). Default 90.
    beta_deg : float, optional
        Surface slope angle (degrees). Default 0.

    Returns
    -------
    float
        Passive earth pressure coefficient Kp.

    References
    ----------
    Coulomb (1776); USACE EM 1110-2-2504, Eq 3-4.

    Notes
    -----
    Coulomb Kp can be unconservative for high delta/phi ratios.
    Use log-spiral method for delta/phi > 0.5.
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    # Passive numerator uses sin^2(alpha - phi) (note the sign: the passive
    # failure geometry flips phi relative to the active case). For a vertical
    # wall (alpha = 90 deg) sin(alpha - phi) = sin(alpha + phi) = cos(phi), so
    # the sign only matters for inclined walls.
    num = math.sin(alpha - phi) ** 2
    term1 = math.sin(alpha) ** 2 * math.sin(alpha + delta)
    term2 = math.sin(phi + delta) * math.sin(phi + beta)
    term3 = math.sin(alpha + delta) * math.sin(alpha + beta)

    if term3 <= 0:
        raise ValueError("Invalid geometry for Coulomb Kp calculation")

    denom = term1 * (1 - math.sqrt(term2 / term3)) ** 2

    if denom <= 0:
        warnings.warn("Coulomb Kp computation failed; using Rankine Kp")
        return rankine_Kp(phi_deg)

    return num / denom


# ----------------------------------------------------------------------------
# Caquot-Kerisel (1948) log-spiral passive coefficient
# ----------------------------------------------------------------------------

# Base passive coefficient Kp at delta = phi (full wall friction), vertical wall,
# horizontal backfill, from the Caquot & Kerisel (1948) log-spiral charts (as
# reproduced in the Caltrans T&S Manual Figure 4-20 and NAVFAC DM-7.2). The
# phi = 30 deg anchor is the Caltrans Fig 4-20 read (6.30) used in the Ex 8-1
# validation; neighbouring phi are the corresponding chart values.
_CK_KP0_PHI = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
_CK_KP0 = [2.20, 3.10, 4.40, 6.30, 9.20, 14.0]

# Wall-friction reduction factor R = Kp(delta) / Kp(delta=phi), from the
# Caquot-Kerisel chart (Caltrans T&S Matrix 4-1 / NAVFAC DM-7.2). Tabulated for
# phi = 30/32/35 deg and delta/phi = 0.40/0.44/0.50; R = 1.0 at delta = phi.
_CK_R_PHI = [30.0, 32.0, 35.0]
_CK_R_RATIO = [0.40, 0.44, 0.50, 1.00]
_CK_R = {
    30.0: {0.40: 0.686, 0.44: 0.710, 0.50: 0.746, 1.00: 1.000},
    32.0: {0.40: 0.653, 0.44: 0.679, 0.50: 0.717, 1.00: 1.000},
    35.0: {0.40: 0.603, 0.44: 0.631, 0.50: 0.674, 1.00: 1.000},
}


def _lininterp(x, xs, ys):
    """Clamped piecewise-linear interpolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def caquot_kerisel_Kp(phi_deg: float, delta_deg: float = None,
                      Kp_initial: float = None) -> float:
    """Caquot-Kerisel (1948) log-spiral passive earth-pressure coefficient.

    Kp' = R * Kp0, where Kp0 is the passive coefficient at delta = phi (read
    from the log-spiral chart) and R = Kp(delta)/Kp(delta=phi) <= 1 reduces it
    for the actual wall-friction ratio delta/phi. For a vertical wall and level
    backfill this is the standard log-spiral passive value, which — unlike
    Coulomb — does NOT over-predict Kp at high delta/phi (Coulomb assumes a
    planar passive wedge; the true surface is a log spiral).

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float, optional
        Wall friction angle (degrees). Default None -> delta = phi (R = 1.0).
    Kp_initial : float, optional
        Chart-read base Kp at delta = phi. If None, taken from the digitized
        Caquot-Kerisel table (phi 15-40 deg; the phi = 30 anchor is the Caltrans
        Fig 4-20 value 6.30).

    Returns
    -------
    float
        Log-spiral passive coefficient Kp'.

    Notes
    -----
    The reduction factor R is tabulated (Caltrans Matrix 4-1 / NAVFAC DM-7.2)
    for phi 30-35 deg and delta/phi 0.40-0.50; delta >= phi gives R = 1.0. For
    delta/phi BELOW the lowest tabulated column (0.40) R is interpolated down to
    the Rankine anchor R(0) = Kp_rankine(phi)/Kp0, so a smooth wall (delta = 0)
    returns exactly the Rankine Kp rather than clamping to 0.686*Kp0 (which would
    over-predict passive resistance and under-predict embedment). For phi = 30
    deg, delta/phi = 0.5, Kp' = 6.30 * 0.746 = 4.70 (Caltrans Ex 8-1).

    References
    ----------
    Caquot & Kerisel (1948); Caltrans T&S Manual Section 4-6 / Fig 4-20 /
    Matrix 4-1; NAVFAC DM-7.2.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    Kp0 = Kp_initial if Kp_initial is not None else _lininterp(phi_deg, _CK_KP0_PHI, _CK_KP0)
    if delta_deg is None or phi_deg <= 0:
        return Kp0
    ratio = min(max(delta_deg / phi_deg, 0.0), 1.0)
    if ratio >= 1.0:
        return Kp0

    def _R_tab(r):
        # Bilinear over the tabulated (phi, delta/phi) grid, clamped in phi.
        phi_c = min(max(phi_deg, _CK_R_PHI[0]), _CK_R_PHI[-1])
        r_row = [_lininterp(r, _CK_R_RATIO, [_CK_R[p][k] for k in _CK_R_RATIO])
                 for p in _CK_R_PHI]
        return _lininterp(phi_c, _CK_R_PHI, r_row)

    lowest = _CK_R_RATIO[0]  # 0.40 -- lowest tabulated delta/phi column
    if ratio >= lowest:
        R = _R_tab(ratio)
    else:
        # Below the lowest tabulated column the table would clamp to R(0.40),
        # over-predicting passive resistance for near-smooth walls. Interpolate R
        # linearly between the Rankine anchor R(0) = Kp_rankine(phi)/Kp0 -- which
        # makes delta = 0 return exactly the Rankine Kp -- and R(0.40).
        R0 = rankine_Kp(phi_deg) / Kp0
        R = R0 + (ratio / lowest) * (_R_tab(lowest) - R0)
    return Kp0 * R


def K0(phi_deg: float) -> float:
    """At-rest earth pressure coefficient (Jaky's formula).

    K0 = 1 - sin(phi)

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).

    Returns
    -------
    float
        At-rest earth pressure coefficient K0.
    """
    return 1.0 - math.sin(math.radians(phi_deg))


def active_pressure(gamma: float, z: float, Ka: float,
                    c: float = 0.0, q_surcharge: float = 0.0) -> float:
    """Active lateral earth pressure at depth z.

    sigma_a = Ka * (gamma * z + q_surcharge) - 2*c*sqrt(Ka)

    Parameters
    ----------
    gamma : float
        Unit weight of soil (kN/m³). Use effective weight below GWT.
    z : float
        Depth below the top of the soil layer (m).
    Ka : float
        Active earth pressure coefficient.
    c : float, optional
        Cohesion (kPa). Default 0.
    q_surcharge : float, optional
        Uniform surcharge at ground surface (kPa). Default 0.

    Returns
    -------
    float
        Active lateral pressure (kPa). Can be negative near surface
        in cohesive soils (tension crack zone).
    """
    return Ka * (gamma * z + q_surcharge) - 2.0 * c * math.sqrt(Ka)


def passive_pressure(gamma: float, z: float, Kp: float,
                     c: float = 0.0) -> float:
    """Passive lateral earth pressure at depth z.

    sigma_p = Kp * gamma * z + 2*c*sqrt(Kp)

    Parameters
    ----------
    gamma : float
        Unit weight of soil (kN/m³). Use effective weight below GWT.
    z : float
        Depth below the excavation level (m).
    Kp : float
        Passive earth pressure coefficient.
    c : float, optional
        Cohesion (kPa). Default 0.

    Returns
    -------
    float
        Passive lateral pressure (kPa).
    """
    return Kp * gamma * z + 2.0 * c * math.sqrt(Kp)


def tension_crack_depth(c: float, gamma: float, Ka: float,
                        q_surcharge: float = 0.0) -> float:
    """Depth of tension crack in cohesive soil behind an active wall.

    The active pressure is zero when Ka*(gamma*z + q) = 2*c*sqrt(Ka),
    i.e. gamma*z + q = 2*c/sqrt(Ka), so
    z_crack = (2*c/sqrt(Ka) - q) / gamma.

    Simplified for q=0: z_crack = 2*c / (gamma * sqrt(Ka))

    Parameters
    ----------
    c : float
        Cohesion (kPa).
    gamma : float
        Unit weight (kN/m³).
    Ka : float
        Active earth pressure coefficient.
    q_surcharge : float, optional
        Surcharge (kPa). Default 0.

    Returns
    -------
    float
        Tension crack depth (m). Returns 0 if no tension crack.
    """
    if c <= 0 or Ka <= 0 or gamma <= 0:
        return 0.0
    z_crack = (2.0 * c / math.sqrt(Ka) - q_surcharge) / gamma
    return max(z_crack, 0.0)
