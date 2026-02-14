"""
AASHTO/NEHRP seismic site classification.

Classifies a site as A-F based on:
- Vs30: weighted harmonic mean shear wave velocity in top 30m
- N-bar: weighted average SPT N in top 30m
- su-bar: weighted average undrained shear strength in top 30m

Also provides site coefficients Fpga, Fa, Fv from AASHTO tables.

All units are SI: m/s, kPa, blows/ft.

References:
    AASHTO LRFD Bridge Design Specifications, 9th Ed., Section 3.10.3
    NEHRP Recommended Seismic Provisions (FEMA P-1050)
"""

import warnings
from typing import List, Optional, Dict, Any, Tuple

from seismic_geotech.results import SiteClassResult


def compute_vs30(layer_thicknesses: List[float],
                 layer_vs: List[float]) -> float:
    """Compute Vs30 (weighted harmonic mean of Vs in top 30m).

    Vs30 = 30 / sum(di / Vsi)

    Parameters
    ----------
    layer_thicknesses : list of float
        Thickness of each layer (m).
    layer_vs : list of float
        Shear wave velocity of each layer (m/s).

    Returns
    -------
    float
        Vs30 (m/s).
    """
    total = 0.0
    depth_used = 0.0
    for h, vs in zip(layer_thicknesses, layer_vs):
        if vs <= 0:
            raise ValueError(f"Shear wave velocity must be positive, got {vs}")
        remaining = 30.0 - depth_used
        if remaining <= 0:
            break
        h_used = min(h, remaining)
        total += h_used / vs
        depth_used += h_used

    if depth_used < 30.0:
        warnings.warn(
            f"Profile depth ({depth_used}m) < 30m; Vs30 based on available depth"
        )
    if total <= 0:
        return 0.0
    return depth_used / total


def compute_n_bar(layer_thicknesses: List[float],
                  layer_N: List[float]) -> float:
    """Compute N-bar (weighted average SPT N in top 30m).

    N-bar = 30 / sum(di / Ni)

    Parameters
    ----------
    layer_thicknesses : list of float
        Thickness of each layer (m).
    layer_N : list of float
        SPT N-value for each layer (blows/ft).

    Returns
    -------
    float
        Average N-bar value.
    """
    total = 0.0
    depth_used = 0.0
    for h, N in zip(layer_thicknesses, layer_N):
        if N <= 0:
            raise ValueError(f"SPT N must be positive, got {N}")
        remaining = 30.0 - depth_used
        if remaining <= 0:
            break
        h_used = min(h, remaining)
        total += h_used / N
        depth_used += h_used

    if depth_used < 30.0:
        warnings.warn(
            f"Profile depth ({depth_used}m) < 30m; N-bar based on available depth"
        )
    if total <= 0:
        return 0.0
    return depth_used / total


def compute_su_bar(layer_thicknesses: List[float],
                   layer_su: List[float]) -> float:
    """Compute su-bar (weighted average cu in top 30m of cohesive soil).

    su-bar = sum(di) / sum(di / sui)  (harmonic mean)

    Parameters
    ----------
    layer_thicknesses : list of float
        Thickness of each cohesive layer (m).
    layer_su : list of float
        Undrained shear strength of each layer (kPa).

    Returns
    -------
    float
        Average su-bar (kPa).
    """
    total = 0.0
    depth_used = 0.0
    for h, su in zip(layer_thicknesses, layer_su):
        if su <= 0:
            raise ValueError(f"Undrained strength must be positive, got {su}")
        remaining = 30.0 - depth_used
        if remaining <= 0:
            break
        h_used = min(h, remaining)
        total += h_used / su
        depth_used += h_used

    if total <= 0:
        return 0.0
    return depth_used / total


def classify_site(vs30: float = None,
                  n_bar: float = None,
                  su_bar: float = None) -> str:
    """Classify site per AASHTO/NEHRP.

    Parameters
    ----------
    vs30 : float, optional
        Average shear wave velocity in top 30m (m/s).
    n_bar : float, optional
        Average SPT N in top 30m.
    su_bar : float, optional
        Average undrained shear strength in top 30m (kPa).

    Returns
    -------
    str
        Site class: "A", "B", "C", "D", "E", or "F".

    References
    ----------
    AASHTO LRFD Table 3.10.3.1-1
    """
    # Prefer Vs30, then N-bar, then su-bar
    if vs30 is not None:
        if vs30 > 1500:
            return "A"
        elif vs30 > 760:
            return "B"
        elif vs30 > 360:
            return "C"
        elif vs30 > 180:
            return "D"
        else:
            return "E"

    if n_bar is not None:
        if n_bar > 50:
            return "C"
        elif n_bar >= 15:
            return "D"
        else:
            return "E"

    if su_bar is not None:
        if su_bar > 100:
            return "C"
        elif su_bar >= 50:
            return "D"
        else:
            return "E"

    raise ValueError("At least one of vs30, n_bar, or su_bar must be provided")


def site_coefficients(site_class: str, Ss: float, S1: float) -> SiteClassResult:
    """Compute site coefficients Fpga, Fa, Fv per AASHTO.

    Parameters
    ----------
    site_class : str
        Site class "A" through "E".
    Ss : float
        Spectral acceleration at 0.2s period (g).
    S1 : float
        Spectral acceleration at 1.0s period (g).

    Returns
    -------
    SiteClassResult
        Site classification results with Fpga, Fa, Fv.

    References
    ----------
    AASHTO LRFD Tables 3.10.3.2-1, 3.10.3.2-2, 3.10.3.2-3
    """
    # Fa table: rows = site class, columns = Ss values
    # Ss:     0.25  0.50  0.75  1.00  1.25
    _Fa_table = {
        "A": [0.8, 0.8, 0.8, 0.8, 0.8],
        "B": [1.0, 1.0, 1.0, 1.0, 1.0],
        "C": [1.2, 1.2, 1.1, 1.0, 1.0],
        "D": [1.6, 1.4, 1.2, 1.1, 1.0],
        "E": [2.5, 1.7, 1.2, 0.9, 0.9],
    }
    _Fa_Ss = [0.25, 0.50, 0.75, 1.00, 1.25]

    # Fv table: rows = site class, columns = S1 values
    # S1:     0.10  0.20  0.30  0.40  0.50
    _Fv_table = {
        "A": [0.8, 0.8, 0.8, 0.8, 0.8],
        "B": [1.0, 1.0, 1.0, 1.0, 1.0],
        "C": [1.7, 1.6, 1.5, 1.4, 1.3],
        "D": [2.4, 2.0, 1.8, 1.6, 1.5],
        "E": [3.5, 3.2, 2.8, 2.4, 2.4],
    }
    _Fv_S1 = [0.10, 0.20, 0.30, 0.40, 0.50]

    if site_class.upper() == "F":
        raise ValueError("Site Class F requires site-specific analysis")

    sc = site_class.upper()
    if sc not in _Fa_table:
        raise ValueError(f"Invalid site class '{site_class}'. Use A-E.")

    Fa = _interpolate_table(_Fa_Ss, _Fa_table[sc], Ss)
    Fv = _interpolate_table(_Fv_S1, _Fv_table[sc], S1)
    # Fpga same table as Fa per AASHTO
    Fpga = Fa

    return SiteClassResult(
        site_class=sc,
        Fpga=round(Fpga, 3),
        Fa=round(Fa, 3),
        Fv=round(Fv, 3),
        Ss=Ss,
        S1=S1,
    )


def _interpolate_table(x_vals: List[float], y_vals: List[float],
                       x: float) -> float:
    """Linear interpolation within AASHTO coefficient tables."""
    if x <= x_vals[0]:
        return y_vals[0]
    if x >= x_vals[-1]:
        return y_vals[-1]

    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x <= x_vals[i + 1]:
            frac = (x - x_vals[i]) / (x_vals[i + 1] - x_vals[i])
            return y_vals[i] + frac * (y_vals[i + 1] - y_vals[i])

    return y_vals[-1]
