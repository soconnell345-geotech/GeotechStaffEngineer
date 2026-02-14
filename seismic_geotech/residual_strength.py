"""
Post-liquefaction residual strength estimation.

Implements:
- Seed & Harder (1990) lower-bound relationship
- Idriss & Boulanger (2008) relationship

All units are SI: kPa.

References:
    Seed & Harder (1990), H. Bolton Seed Memorial Symposium
    Idriss & Boulanger (2008), Earthquake Spectra
"""

import math
import warnings
from typing import Optional


def Sr_seed_harder(N160cs: float) -> float:
    """Residual strength per Seed & Harder (1990) lower bound.

    Piecewise linear approximation of the lower-bound curve.

    Parameters
    ----------
    N160cs : float
        Clean-sand corrected SPT blow count (N1)60cs.

    Returns
    -------
    float
        Residual undrained strength Sr (kPa).

    References
    ----------
    Seed & Harder (1990), Fig. 1 lower bound
    """
    # Piecewise linear approximation of lower-bound curve
    # (N160cs, Sr_kPa) control points
    _points = [
        (0, 0),
        (4, 2.5),
        (8, 5.0),
        (12, 10.0),
        (16, 15.0),
        (20, 25.0),
        (24, 35.0),
    ]

    if N160cs <= 0:
        return 0.0
    if N160cs >= _points[-1][0]:
        return _points[-1][1]

    for i in range(len(_points) - 1):
        x0, y0 = _points[i]
        x1, y1 = _points[i + 1]
        if x0 <= N160cs <= x1:
            frac = (N160cs - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)

    return _points[-1][1]


def Sr_idriss_boulanger(N160cs: float, sigma_v_eff: float) -> float:
    """Residual strength per Idriss & Boulanger (2008).

    Sr/sigma_v' = exp(N160cs/16 + (N160cs/21.2)^3 - 3.0)
    Capped at Sr/sigma_v' = 0.6 (tanphi_residual)

    Parameters
    ----------
    N160cs : float
        Clean-sand corrected SPT blow count.
    sigma_v_eff : float
        Pre-earthquake effective vertical stress (kPa).

    Returns
    -------
    float
        Residual undrained strength Sr (kPa).

    References
    ----------
    Idriss & Boulanger (2008), Eq. 10
    """
    if sigma_v_eff <= 0:
        return 0.0

    ratio = math.exp(
        N160cs / 16.0
        + (N160cs / 21.2) ** 3
        - 3.0
    )
    ratio = min(ratio, 0.6)  # cap at tan(phi) ~ 0.6

    return ratio * sigma_v_eff


def post_liquefaction_strength(N160cs: float,
                               sigma_v_eff: float = None,
                               method: str = "seed_harder") -> float:
    """Compute post-liquefaction residual strength.

    Parameters
    ----------
    N160cs : float
        Clean-sand corrected SPT blow count.
    sigma_v_eff : float, optional
        Effective vertical stress (kPa). Required for Idriss-Boulanger.
    method : str, optional
        "seed_harder" (default) or "idriss_boulanger".

    Returns
    -------
    float
        Residual strength Sr (kPa).
    """
    if method == "seed_harder":
        return Sr_seed_harder(N160cs)
    elif method == "idriss_boulanger":
        if sigma_v_eff is None:
            raise ValueError(
                "sigma_v_eff is required for Idriss-Boulanger method"
            )
        return Sr_idriss_boulanger(N160cs, sigma_v_eff)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'seed_harder' or 'idriss_boulanger'")
