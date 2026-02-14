"""
Stress distribution methods for settlement analysis.

Computes the vertical stress increase (delta_sigma) at depth z below
a loaded area, using various methods.

All units are SI: kPa, meters.

References:
    Boussinesq, J. (1885) — Point load in elastic half-space
    Westergaard, H.M. (1938) — Alternating rigid/elastic layers
    FHWA Soils & Foundations Reference Manual, Vol II, Ch 8
    USACE EM 1110-1-1904, Chapter 4
"""

import math
from typing import Optional

import numpy as np


def boussinesq_point(Q: float, z: float, r: float) -> float:
    """Vertical stress from a point load (Boussinesq solution).

    Parameters
    ----------
    Q : float
        Point load (kN).
    z : float
        Depth below the point of load application (m).
    r : float
        Horizontal distance from the load (m).

    Returns
    -------
    float
        Vertical stress increase delta_sigma_z (kPa).
    """
    if z <= 0:
        return 0.0
    R = math.sqrt(r**2 + z**2)
    return 3.0 * Q * z**3 / (2.0 * math.pi * R**5)


def boussinesq_rectangular(q: float, B: float, L: float, z: float) -> float:
    """Vertical stress under the corner of a uniformly loaded rectangle.

    Uses the Newmark (1935) integration of the Boussinesq solution.

    Parameters
    ----------
    q : float
        Uniform applied pressure (kPa).
    B : float
        Width of the loaded area (m).
    L : float
        Length of the loaded area (m).
    z : float
        Depth below the loaded area (m).

    Returns
    -------
    float
        Vertical stress increase delta_sigma_z (kPa) at the corner.

    References
    ----------
    Newmark, N.M. (1935), Simplified computation of vertical pressures
    in elastic foundations, Univ. of Illinois Engineering Experiment Station.
    """
    if z <= 0:
        return q
    m = B / z
    n = L / z
    mn = m * n
    m2 = m**2
    n2 = n**2

    term1 = m2 + n2 + 1
    term2 = m2 * n2

    # Influence factor I
    part1 = 2 * mn * math.sqrt(m2 + n2 + 1) / (m2 + n2 + 1 + term2)
    part1 *= (m2 + n2 + 2) / (m2 + n2 + 1)

    # Check if arctan correction needed (when denom < 0)
    denom = m2 + n2 + 1 - term2
    if abs(denom) < 1e-12:
        part2 = math.pi / 2 if mn > 0 else -math.pi / 2
    else:
        part2 = math.atan(2 * mn * math.sqrt(m2 + n2 + 1) / denom)

    if denom < 0:
        part2 += math.pi

    I = (part1 + part2) / (4 * math.pi)
    return q * I


def boussinesq_center_rectangular(q: float, B: float, L: float, z: float) -> float:
    """Vertical stress under the center of a uniformly loaded rectangle.

    Uses superposition: center = 4 × corner of (B/2 × L/2) sub-rectangles.

    Parameters
    ----------
    q : float
        Uniform applied pressure (kPa).
    B : float
        Full width of the loaded area (m).
    L : float
        Full length of the loaded area (m).
    z : float
        Depth below the loaded area (m).

    Returns
    -------
    float
        Vertical stress increase delta_sigma_z (kPa) at center.
    """
    return 4.0 * boussinesq_rectangular(q, B / 2.0, L / 2.0, z)


def approximate_2to1(q: float, B: float, L: float, z: float) -> float:
    """Stress increase using the 2:1 (2V:1H) approximate method.

    The load spreads at a 2V:1H slope, so at depth z the stress acts
    over an area (B+z) × (L+z).

    Parameters
    ----------
    q : float
        Uniform applied pressure (kPa).
    B : float
        Width of the loaded area (m).
    L : float
        Length of the loaded area (m).
    z : float
        Depth below the loaded area (m).

    Returns
    -------
    float
        Vertical stress increase delta_sigma_z (kPa).

    References
    ----------
    FHWA Soils & Foundations Reference Manual, Vol II, Section 8.3.
    """
    if z <= 0:
        return q
    return q * B * L / ((B + z) * (L + z))


def approximate_2to1_strip(q: float, B: float, z: float) -> float:
    """Stress increase using 2:1 method for a strip load.

    Parameters
    ----------
    q : float
        Uniform applied pressure (kPa).
    B : float
        Width of the strip load (m).
    z : float
        Depth below the loaded area (m).

    Returns
    -------
    float
        Vertical stress increase delta_sigma_z (kPa).
    """
    if z <= 0:
        return q
    return q * B / (B + z)


def westergaard_rectangular(q: float, B: float, L: float, z: float) -> float:
    """Vertical stress under the corner of a rectangle — Westergaard solution.

    The Westergaard solution assumes alternating rigid and elastic layers,
    which is more appropriate for varved clays and stratified deposits.

    Parameters
    ----------
    q : float
        Uniform applied pressure (kPa).
    B : float
        Width of the loaded area (m).
    L : float
        Length of the loaded area (m).
    z : float
        Depth below the loaded area (m).

    Returns
    -------
    float
        Vertical stress increase delta_sigma_z (kPa) at the corner.

    References
    ----------
    Westergaard, H.M. (1938), "A Problem of Elasticity Suggested by a
    Problem in Soil Mechanics", Contributions to the Mechanics of Solids,
    Stephen Timoshenko 60th Anniversary Volume, Macmillan, New York.
    """
    if z <= 0:
        return q

    m = B / z
    n = L / z

    # Westergaard influence factor
    a = 1.0 / (2 * math.pi)
    arg = 1 + m**2 + n**2
    I = a * math.atan(m * n / math.sqrt(arg))
    return q * I


def stress_at_depth(q: float, B: float, L: float, z: float,
                    method: str = "2:1",
                    location: str = "center") -> float:
    """Compute stress increase at depth z using the specified method.

    This is the primary interface for stress distribution calculations.

    Parameters
    ----------
    q : float
        Net applied pressure at footing base (kPa).
    B : float
        Footing width (m).
    L : float
        Footing length (m). For strip footings, use a large value.
    z : float
        Depth below the footing base (m).
    method : str, optional
        Stress distribution method:
        - "2:1" (default): 2V:1H approximate method
        - "boussinesq": Boussinesq elastic solution
        - "westergaard": Westergaard layered solution
    location : str, optional
        "center" (default) or "corner".

    Returns
    -------
    float
        Vertical stress increase delta_sigma_z (kPa).
    """
    method = method.lower().replace(" ", "")
    if method == "2:1" or method == "2to1":
        return approximate_2to1(q, B, L, z)
    elif method == "boussinesq":
        if location == "center":
            return boussinesq_center_rectangular(q, B, L, z)
        else:
            return boussinesq_rectangular(q, B, L, z)
    elif method == "westergaard":
        if location == "center":
            return 4.0 * westergaard_rectangular(q, B / 2, L / 2, z)
        else:
            return westergaard_rectangular(q, B, L, z)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Options: '2:1', 'boussinesq', 'westergaard'"
        )
