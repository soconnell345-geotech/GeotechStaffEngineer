"""
Mononobe-Okabe seismic earth pressure theory.

Computes active (KAE) and passive (KPE) earth pressure coefficients
under seismic loading, and the seismic pressure increment.

Sign convention (v5.1, SG-3): the wall batter ``beta_deg`` is measured
from vertical, POSITIVE when the wall back face leans toward (over) the
backfill — equivalent to a Coulomb wall inclination alpha = 90° + beta
in the sin-based Coulomb form (``sheet_pile.earth_pressure``). With this
convention, at kh = kv = 0 KAE reduces exactly to Coulomb Ka and KPE to
Coulomb Kp for any batter beta and backfill slope i. Both coefficients
have been verified against a numerical Coulomb/M-O wedge limit-
equilibrium solution (planar failure surface, tilted body force) to
4+ decimal places.

The active case uses the seismic inertia force directed toward the wall
(increasing the driving thrust); the passive case uses it directed away
from the wall (reducing the available resistance) — both worst-case per
the standard M-O treatment.

Note that planar-wedge passive coefficients (Coulomb and M-O alike)
overpredict resistance for high wall friction (delta/phi > ~0.5); a
log-spiral solution is preferred in that range.

All units are SI: kN, kPa, kN/m³, degrees, meters.

References:
    Mononobe & Matsuo (1929); Okabe (1926)
    AASHTO LRFD Section 11.6.5
    FHWA GEC-3, Chapter 7
    Kramer (1996), Geotechnical Earthquake Engineering, Section 11.6
"""

import math
import warnings
from typing import Dict, Any


def mononobe_okabe_KAE(phi_deg: float, delta_deg: float, kh: float,
                       kv: float = 0.0, beta_deg: float = 0.0,
                       i_deg: float = 0.0) -> float:
    """Mononobe-Okabe active earth pressure coefficient.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float
        Wall-soil interface friction angle (degrees).
    kh : float
        Horizontal seismic coefficient (dimensionless).
    kv : float, optional
        Vertical seismic coefficient (dimensionless). Default 0.
    beta_deg : float, optional
        Wall batter angle from vertical (degrees), positive when the
        wall back face leans toward the backfill (Coulomb alpha =
        90° + beta; see module docstring). Default 0 (vertical wall).
    i_deg : float, optional
        Backfill slope angle (degrees). Default 0 (horizontal).

    Returns
    -------
    float
        Seismic active earth pressure coefficient KAE. At kh = kv = 0
        this reduces exactly to Coulomb Ka for any beta and i.

    References
    ----------
    AASHTO LRFD Eq. 11.6.5.2-1
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    beta = math.radians(beta_deg)
    i = math.radians(i_deg)

    # Seismic inertia angle
    if abs(1.0 - kv) < 1e-10:
        raise ValueError("kv cannot equal 1.0")
    theta = math.atan(kh / (1.0 - kv))

    # Check M-O applicability: phi - theta - i must be >= 0
    if phi - theta - i < 0:
        warnings.warn(
            f"phi-theta-i = {math.degrees(phi - theta - i):.1f}° < 0; "
            f"M-O not applicable. Returning simplified estimate."
        )
        # Fall back to simplified Seed-Whitman
        Ka_static = math.tan(math.pi / 4 - phi / 2) ** 2
        return Ka_static + 0.75 * kh

    cos_theta = math.cos(theta)
    cos_beta = math.cos(beta)

    num = math.cos(phi + beta - theta) ** 2

    sin_pd = math.sin(phi + delta)
    sin_pti = math.sin(phi - theta - i)
    cos_dbt = math.cos(delta + theta - beta)
    cos_ib = math.cos(i + beta)

    if cos_dbt * cos_ib <= 0:
        raise ValueError("Invalid geometry for M-O calculation")

    inner = math.sqrt(sin_pd * sin_pti / (cos_dbt * cos_ib))
    denom = cos_theta * cos_beta ** 2 * cos_dbt * (1.0 + inner) ** 2

    if denom <= 0:
        raise ValueError("M-O denominator <= 0; check input parameters")

    return num / denom


def mononobe_okabe_KPE(phi_deg: float, delta_deg: float, kh: float,
                       kv: float = 0.0, beta_deg: float = 0.0,
                       i_deg: float = 0.0) -> float:
    """Mononobe-Okabe passive earth pressure coefficient (reduced).

    KPE = cos²(φ−θ−β) / [cosθ·cos²β·cos(δ+θ+β)
          ·(1 − √(sin(φ+δ)·sin(φ+i−θ) / (cos(δ+θ+β)·cos(i+β))))²]

    with θ = atan(kh/(1−kv)) and β per the module sign convention
    (positive = back face leaning toward the backfill, Coulomb
    alpha = 90° + β). The seismic inertia force is taken acting away
    from the wall (worst case — minimum passive resistance).

    Prior to v5.1 this function carried a flipped β sign in the
    numerator and a flipped θ sign in the cos(δ+θ+β) terms (SG-3); it
    did not reduce to Coulomb Kp for battered walls at kh = 0, and
    under-predicted vertical-wall seismic KPE (e.g. φ=30°, δ=15°,
    kh=0.2: 3.35 vs the correct 4.13). The corrected form matches a
    numerical M-O wedge limit-equilibrium solution exactly.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float
        Wall-soil interface friction angle (degrees).
    kh : float
        Horizontal seismic coefficient.
    kv : float, optional
        Vertical seismic coefficient. Default 0.
    beta_deg : float, optional
        Wall batter from vertical (degrees), positive when the wall
        back face leans toward the backfill (Coulomb alpha = 90° + beta;
        see module docstring). Default 0 (vertical wall).
    i_deg : float, optional
        Ground slope (degrees). Default 0.

    Returns
    -------
    float
        Seismic passive earth pressure coefficient KPE. At kh = kv = 0
        this reduces exactly to Coulomb Kp for any beta and i.
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    beta = math.radians(beta_deg)
    i = math.radians(i_deg)

    if abs(1.0 - kv) < 1e-10:
        raise ValueError("kv cannot equal 1.0")
    theta = math.atan(kh / (1.0 - kv))

    cos_theta = math.cos(theta)
    cos_beta = math.cos(beta)

    num = math.cos(phi - theta - beta) ** 2

    sin_pd = math.sin(phi + delta)
    sin_pti = math.sin(phi - theta + i)
    cos_dbt = math.cos(delta + theta + beta)
    cos_ib = math.cos(i + beta)

    if cos_dbt * cos_ib <= 0:
        raise ValueError("Invalid geometry for M-O passive calculation")

    inner_sq = sin_pd * sin_pti / (cos_dbt * cos_ib)
    if inner_sq < 0:
        inner_sq = 0.0
    inner = math.sqrt(inner_sq)

    denom = cos_theta * cos_beta ** 2 * cos_dbt * (1.0 - inner) ** 2

    if denom <= 0:
        warnings.warn("M-O passive denominator <= 0; returning Rankine Kp")
        return math.tan(math.pi / 4 + phi / 2) ** 2

    return num / denom


def seismic_earth_pressure(gamma: float, H: float,
                           KAE: float, KA: float) -> Dict[str, float]:
    """Compute seismic earth pressure resultants.

    Parameters
    ----------
    gamma : float
        Unit weight of backfill (kN/m³).
    H : float
        Wall height (m).
    KAE : float
        Seismic active earth pressure coefficient.
    KA : float
        Static active earth pressure coefficient.

    Returns
    -------
    dict
        PAE_total_kN_per_m : Total seismic active force per unit width
        PA_static_kN_per_m : Static active force per unit width
        delta_PAE_kN_per_m : Seismic increment
        height_of_application_m : Height of seismic increment (0.6*H)

    References
    ----------
    Seed & Whitman (1970); FHWA GEC-3
    """
    PAE_total = 0.5 * gamma * H ** 2 * KAE
    PA_static = 0.5 * gamma * H ** 2 * KA
    delta_PAE = PAE_total - PA_static

    return {
        "PAE_total_kN_per_m": round(PAE_total, 2),
        "PA_static_kN_per_m": round(PA_static, 2),
        "delta_PAE_kN_per_m": round(delta_PAE, 2),
        "height_of_application_m": round(0.6 * H, 2),
    }
