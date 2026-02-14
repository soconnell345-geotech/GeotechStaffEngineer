"""
Simplified liquefaction triggering evaluation.

Implements the Seed & Idriss (1971) simplified procedure with
updates from Youd et al. (2001) NCEER/NSF workshop.

All units are SI: kPa, m, g (acceleration).

References:
    Seed & Idriss (1971), ASCE JSMFED
    Youd et al. (2001), ASCE JGGE, Vol 127, No 10
    Boulanger & Idriss (2014), CPT/SPT-based liquefaction triggering
"""

import math
import warnings
from typing import List, Dict, Any, Optional


def stress_reduction_rd(z: float) -> float:
    """Depth-dependent stress reduction factor rd.

    Parameters
    ----------
    z : float
        Depth below ground surface (m).

    Returns
    -------
    float
        Stress reduction factor rd.

    References
    ----------
    Liao & Whitman (1986); Youd et al. (2001) Eq. 4
    """
    if z <= 9.15:
        return 1.0 - 0.00765 * z
    elif z <= 23.0:
        return 1.174 - 0.0267 * z
    else:
        # Extended per Idriss (1999)
        return 0.744 - 0.008 * z  # approximate for z > 23m


def magnitude_scaling_factor(M: float) -> float:
    """Magnitude scaling factor MSF for earthquakes M != 7.5.

    Parameters
    ----------
    M : float
        Earthquake moment magnitude.

    Returns
    -------
    float
        Magnitude scaling factor MSF.

    References
    ----------
    Youd et al. (2001) Eq. 20
    """
    if M <= 0:
        raise ValueError(f"Magnitude must be positive, got {M}")
    return 10 ** 2.24 / (M ** 2.56)


def compute_CSR(amax_g: float, sigma_v: float, sigma_v_eff: float,
                z: float, M: float = 7.5) -> float:
    """Cyclic stress ratio at depth z.

    CSR = 0.65 * (amax/g) * (sigma_v/sigma_v') * rd
    Adjusted for magnitude: CSR_M7.5 = CSR / MSF

    Parameters
    ----------
    amax_g : float
        Peak ground acceleration (fraction of g).
    sigma_v : float
        Total vertical stress at depth z (kPa).
    sigma_v_eff : float
        Effective vertical stress at depth z (kPa).
    z : float
        Depth below ground surface (m).
    M : float, optional
        Earthquake magnitude. Default 7.5.

    Returns
    -------
    float
        Cyclic stress ratio adjusted to M=7.5.
    """
    if sigma_v_eff <= 0:
        return 0.0

    rd = stress_reduction_rd(z)
    CSR = 0.65 * amax_g * (sigma_v / sigma_v_eff) * rd

    if abs(M - 7.5) > 0.01:
        MSF = magnitude_scaling_factor(M)
        CSR = CSR / MSF

    return CSR


def fines_correction(N160: float, FC: float) -> float:
    """Correct (N1)60 for fines content to get (N1)60cs.

    Parameters
    ----------
    N160 : float
        Corrected SPT blow count (N1)60.
    FC : float
        Fines content (% passing #200 sieve).

    Returns
    -------
    float
        Clean-sand equivalent (N1)60cs.

    References
    ----------
    Youd et al. (2001) Eqs. 5-6
    """
    if FC <= 5:
        alpha = 0.0
        beta = 1.0
    elif FC < 35:
        alpha = math.exp(1.76 - 190.0 / FC ** 2)
        beta = 0.99 + FC ** 1.5 / 1000.0
    else:
        alpha = 5.0
        beta = 1.2

    return alpha + beta * N160


def CRR_from_N160cs(N160cs: float) -> float:
    """Cyclic resistance ratio from clean-sand SPT (N1)60cs.

    Uses the Youd et al. (2001) deterministic curve for M=7.5.

    Parameters
    ----------
    N160cs : float
        Clean-sand corrected SPT blow count.

    Returns
    -------
    float
        Cyclic resistance ratio CRR_7.5.
        Returns 2.0 for N160cs >= 30 (too dense to liquefy).

    References
    ----------
    Youd et al. (2001) Eq. 2 (NCEER curve)
    """
    if N160cs >= 30:
        return 2.0  # Too dense to liquefy

    if N160cs < 0:
        N160cs = 0.0

    CRR = (
        1.0 / (34.0 - N160cs)
        + N160cs / 135.0
        + 50.0 / (10.0 * N160cs + 45.0) ** 2
        - 1.0 / 200.0
    )
    return max(CRR, 0.0)


def evaluate_liquefaction(
    layer_depths: List[float],
    layer_N160: List[float],
    layer_FC: List[float],
    layer_gamma: List[float],
    amax_g: float,
    gwt_depth: float,
    M: float = 7.5,
    gamma_w: float = 9.81,
) -> List[Dict[str, Any]]:
    """Evaluate liquefaction potential at multiple depths.

    Parameters
    ----------
    layer_depths : list of float
        Depth to midpoint of each evaluation layer (m).
    layer_N160 : list of float
        Corrected SPT (N1)60 at each depth.
    layer_FC : list of float
        Fines content (%) at each depth.
    layer_gamma : list of float
        Total unit weight (kN/m³) at each depth.
    amax_g : float
        Peak ground acceleration (fraction of g).
    gwt_depth : float
        Depth to groundwater (m).
    M : float, optional
        Earthquake magnitude. Default 7.5.
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default 9.81.

    Returns
    -------
    list of dict
        Per-layer results with keys: depth_m, N160, N160cs, FC_pct,
        sigma_v_kPa, sigma_v_eff_kPa, CSR, CRR, FOS_liq, liquefiable.
    """
    results = []

    for z, N160, FC, gamma in zip(layer_depths, layer_N160, layer_FC, layer_gamma):
        # Total stress
        sigma_v = gamma * z

        # Effective stress
        if z <= gwt_depth:
            sigma_v_eff = sigma_v
        else:
            sigma_v_eff = sigma_v - gamma_w * (z - gwt_depth)

        if sigma_v_eff <= 0:
            sigma_v_eff = 1.0  # prevent division by zero

        # Clean sand correction
        N160cs = fines_correction(N160, FC)

        # CSR and CRR
        CSR = compute_CSR(amax_g, sigma_v, sigma_v_eff, z, M)
        CRR = CRR_from_N160cs(N160cs)

        FOS = CRR / CSR if CSR > 0 else 99.9

        results.append({
            "depth_m": round(z, 2),
            "N160": round(N160, 1),
            "N160cs": round(N160cs, 1),
            "FC_pct": round(FC, 1),
            "sigma_v_kPa": round(sigma_v, 1),
            "sigma_v_eff_kPa": round(sigma_v_eff, 1),
            "CSR": round(CSR, 4),
            "CRR": round(CRR, 4),
            "FOS_liq": round(FOS, 3),
            "liquefiable": FOS < 1.0,
        })

    return results
