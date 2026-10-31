"""
Stress distribution methods for settlement analysis.

Computes the vertical stress increase (delta_sigma) at depth z below
a loaded area, using various methods.

All units are SI: kPa, meters.

References:
    UFS 3-220-10 (2026), Table 4-2 (p. 185): Boussinesq point, strip and
        rectangle formulas; Fig. 4-10 (p. 194): Westergaard (nu = 0)
        influence factors below the corner of a rectangle
    FHWA GEC-6 (FHWA-SA-02-054), Eq. 5-17 and Fig. 5-12: the 2:1 method
    FHWA NHI-06-088, Sec. 2.4
"""

import math
from typing import Optional

import numpy as np


def boussinesq_point(Q: float, z: float, r: float) -> float:
    """Vertical stress from a point load (Boussinesq solution),
    sigma_z = 3 Q z^3 / (2 pi R^5), R = sqrt(r^2 + z^2) (UFS 3-220-10
    Table 4-2, p. 185, with r^2 = x^2 + y^2).

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
    """Vertical stress under the corner of a uniformly loaded rectangle
    (Boussinesq), as UFS 3-220-10 Table 4-2 (p. 185) prints it, with
    x = B and y = L:

        sigma_z = (q0 / 2 pi) [atan(x y / (z R3))
                               + (x y z / R3) (1 / R1^2 + 1 / R2^2)]

        R1 = sqrt(y^2 + z^2), R2 = sqrt(x^2 + z^2), R3 = sqrt(x^2 + y^2 + z^2)

    (GEC-6 Fig. 5-11 is Newmark's chart of the same stress.)

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
    """
    if z <= 0:
        return q
    x, y = B, L
    R1 = math.sqrt(y * y + z * z)
    R2 = math.sqrt(x * x + z * z)
    R3 = math.sqrt(x * x + y * y + z * z)
    return q / (2.0 * math.pi) * (
        math.atan(x * y / (z * R3))
        + x * y * z / R3 * (1.0 / R1**2 + 1.0 / R2**2))


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

    Notes
    -----
    At z = 0 the stress equals the applied pressure q. The guard is needed
    because the corner function returns q (not q/4) at z <= 0, so the 4x
    superposition would otherwise yield 4q at exactly z = 0 (SET-4). The
    z -> 0+ limit of the superposition is q, consistent with the guard.
    """
    if z <= 0:
        return q
    return 4.0 * boussinesq_rectangular(q, B / 2.0, L / 2.0, z)


def approximate_2to1(q: float, B: float, L: float, z: float) -> float:
    """Stress increase using the 2:1 (2V:1H) approximate method.

    The load spreads at a 2V:1H slope, so at depth z the stress acts
    over an area (B+z) × (L+z): delta_sigma = q B L / ((B + z)(L + z)).

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
    FHWA GEC-6 Eq. 5-17 (p. 78) and Fig. 5-12; FHWA NHI-06-088 Sec. 2.4.2.
    """
    if z <= 0:
        return q
    return q * B * L / ((B + z) * (L + z))


def approximate_2to1_strip(q: float, B: float, z: float) -> float:
    """Stress increase using 2:1 method for a strip load,
    delta_sigma = q B / (B + z) (GEC-6 Fig. 5-12, used so in GEC-6
    Appendix B Example 4).

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


def boussinesq_center_strip(q: float, B: float, z: float) -> float:
    """Vertical stress below the centre of a uniformly loaded strip of
    width B, UFS 3-220-10 Table 4-2's uniform strip load (p. 185),

        delta_sigma = (q / pi) [alpha + sin alpha cos(alpha + 2 delta)],

    below the centre, where delta = -alpha / 2:
    (q / pi)(alpha + sin alpha), alpha = 2 atan(B / 2z) being the angle the
    strip subtends at the point. It is the limit L -> infinity of
    :func:`boussinesq_center_rectangular`.
    """
    if z <= 0:
        return q
    alpha = 2.0 * math.atan(B / (2.0 * z))
    return q / math.pi * (alpha + math.sin(alpha))


#: UFS 3-220-10 Fig. 4-10 (p. 194): the influence factor I below the corner
#: of a uniformly loaded rectangle by Westergaard theory (nu = 0),
#: sigma_z = I q0 with m = x/z and n = y/z, which the figure says are
#: interchangeable. Digitised from the printed curves (GM-285): row i is the
#: curve m = WESTERGAARD_M[i] read at n = WESTERGAARD_M[i], WESTERGAARD_M[i+1],
#: ... 10. At n = 5, 6 and 8, where the figure prints the curve labels
#: instead of the curves, each curve is read across the gap.
WESTERGAARD_M = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                 1.4, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
_WESTERGAARD_I = (
    (0.0032, 0.0061, 0.0087, 0.0110, 0.0128, 0.0145, 0.0157, 0.0167,
     0.0176, 0.0182, 0.0192, 0.0200, 0.0204, 0.0211, 0.0215, 0.0218,
     0.0220, 0.0222, 0.0223, 0.0223, 0.0223),  # m = 0.1
    (0.0118, 0.0169, 0.0214, 0.0250, 0.0281, 0.0307, 0.0327, 0.0343,
     0.0357, 0.0376, 0.0390, 0.0400, 0.0413, 0.0422, 0.0427, 0.0432,
     0.0433, 0.0437, 0.0437, 0.0438),  # m = 0.2
    (0.0243, 0.0308, 0.0362, 0.0407, 0.0443, 0.0472, 0.0497, 0.0517,
     0.0546, 0.0567, 0.0581, 0.0600, 0.0613, 0.0620, 0.0628, 0.0632,
     0.0635, 0.0636, 0.0637),  # m = 0.3
    (0.0390, 0.0458, 0.0516, 0.0563, 0.0601, 0.0633, 0.0658, 0.0696,
     0.0724, 0.0743, 0.0768, 0.0785, 0.0795, 0.0805, 0.0810, 0.0813,
     0.0816, 0.0816),  # m = 0.4
    (0.0540, 0.0609, 0.0666, 0.0711, 0.0749, 0.0780, 0.0828, 0.0861,
     0.0884, 0.0916, 0.0937, 0.0950, 0.0962, 0.0969, 0.0974, 0.0975,
     0.0976),  # m = 0.5
    (0.0687, 0.0752, 0.0804, 0.0849, 0.0884, 0.0940, 0.0979, 0.1007,
     0.1044, 0.1069, 0.1084, 0.1099, 0.1105, 0.1107, 0.1113, 0.1116),
    # m = 0.6
    (0.0824, 0.0883, 0.0932, 0.0973, 0.1035, 0.1080, 0.1111, 0.1154,
     0.1183, 0.1200, 0.1218, 0.1226, 0.1231, 0.1236, 0.1237),  # m = 0.7
    (0.0947, 0.1003, 0.1047, 0.1115, 0.1165, 0.1201, 0.1248, 0.1280,
     0.1300, 0.1320, 0.1330, 0.1336, 0.1341, 0.1342),  # m = 0.8
    (0.1060, 0.1108, 0.1183, 0.1237, 0.1276, 0.1328, 0.1364, 0.1386,
     0.1409, 0.1419, 0.1425, 0.1431, 0.1434),  # m = 0.9
    (0.1160, 0.1240, 0.1298, 0.1340, 0.1397, 0.1437, 0.1461, 0.1486,
     0.1497, 0.1502, 0.1510, 0.1514),  # m = 1.0
    (0.1329, 0.1394, 0.1442, 0.1508, 0.1554, 0.1582, 0.1611, 0.1625,
     0.1631, 0.1639, 0.1644),  # m = 1.2
    (0.1465, 0.1517, 0.1590, 0.1643, 0.1674, 0.1708, 0.1723, 0.1731,
     0.1742, 0.1747),  # m = 1.4
    (0.1574, 0.1654, 0.1711, 0.1747, 0.1783, 0.1800, 0.1808, 0.1822,
     0.1828),  # m = 1.6
    (0.1741, 0.1806, 0.1848, 0.1893, 0.1915, 0.1930, 0.1944, 0.1947),
    # m = 2.0
    (0.1881, 0.1928, 0.1981, 0.2007, 0.2021, 0.2038, 0.2046),  # m = 2.5
    (0.1979, 0.2038, 0.2068, 0.2085, 0.2105, 0.2113),  # m = 3.0
    (0.2105, 0.2142, 0.2163, 0.2187, 0.2198),  # m = 4.0
    (0.2183, 0.2208, 0.2228, 0.2247),  # m = 5.0
    (0.2235, 0.2262, 0.2280),  # m = 6.0
    (0.2300, 0.2317),  # m = 8.0
    (0.2338,),  # m = 10.0
)
#: The figure's curve m = infinity, read at n = each of WESTERGAARD_M.
_WESTERGAARD_I_INF = (
    0.0224, 0.0438, 0.0637, 0.0819, 0.0978, 0.1120, 0.1242, 0.1346,
    0.1439, 0.1517, 0.1650, 0.1753, 0.1834, 0.1957, 0.2058, 0.2129,
    0.2219, 0.2273, 0.2313, 0.2359, 0.2383)
#: How the module reads UFS Fig. 4-10 between and beyond its curves.
WESTERGAARD_READING = (
    "UFS 3-220-10 Fig. 4-10 (p. 194), Westergaard theory with nu = 0, "
    "digitised; between the printed curves (m and n from 0.1 to 10, and "
    "m = infinity) interpolated linearly in m and n; beyond n = 10 "
    "toward the m = infinity curve in 10/n, with I = 0.25 when both m and n "
    "are infinite (the corner of the loaded area at the surface, where the "
    "module takes sigma_z = q0 under the centre); below 0.1 in proportion "
    "to m or n (no load, no stress, at m = 0)")


def _westergaard_node(v: float):
    """``(i, j, t)``: v between the nodes WESTERGAARD_M + (infinity,) i and
    j, a fraction t of the way to j (v from 0.1 up)."""
    nodes = WESTERGAARD_M
    if v >= nodes[-1]:
        return len(nodes) - 1, len(nodes), 1.0 - nodes[-1] / v
    for i in range(len(nodes) - 1):
        if v <= nodes[i + 1]:
            # linear in v: in log v the reading ran up to 11 % above the
            # printed curves where m and n both lie between 0.1 and 0.3
            return i, i + 1, (v - nodes[i]) / (nodes[i + 1] - nodes[i])
    raise AssertionError("unreachable")


def _westergaard_at(i: int, j: int) -> float:
    """I at the nodes i, j (len(WESTERGAARD_M) is m = infinity)."""
    inf = len(WESTERGAARD_M)
    if i == inf and j == inf:
        return 0.25
    if i == inf or j == inf:
        return _WESTERGAARD_I_INF[min(i, j)]
    a, b = min(i, j), max(i, j)
    return _WESTERGAARD_I[a][b - a]


def westergaard_influence_factor(m: float, n: float) -> float:
    """The influence factor I of UFS 3-220-10 Fig. 4-10 (Westergaard,
    nu = 0) below the corner of a uniformly loaded rectangle, m = x/z and
    n = y/z (interchangeable), as :data:`WESTERGAARD_READING` reads the
    figure; ``math.inf`` for an infinitely long side."""
    if m <= 0 or n <= 0:
        return 0.0
    scale = min(m / 0.1, 1.0) * min(n / 0.1, 1.0)
    m, n = max(m, 0.1), max(n, 0.1)
    i0, i1, s = _westergaard_node(m)
    j0, j1, t = _westergaard_node(n)
    value = ((1 - s) * (1 - t) * _westergaard_at(i0, j0)
             + s * (1 - t) * _westergaard_at(i1, j0)
             + (1 - s) * t * _westergaard_at(i0, j1)
             + s * t * _westergaard_at(i1, j1))
    return scale * value


def westergaard_center_strip(q: float, B: float, z: float) -> float:
    """Vertical stress below the centre of a uniformly loaded strip of
    width B by Westergaard theory (nu = 0): four corner rectangles of
    width B/2 and infinite length, delta_sigma = 4 q I(B / 2z, infinity),
    I from UFS Fig. 4-10's curve m = infinity
    (:func:`westergaard_influence_factor`)."""
    if z <= 0:
        return q
    return 4.0 * q * westergaard_influence_factor(B / (2.0 * z), math.inf)


def westergaard_rectangular(q: float, B: float, L: float, z: float) -> float:
    """Vertical stress under the corner of a rectangle — Westergaard solution.

    The Westergaard solution assumes the elastic half space interspersed
    with thin rigid layers that allow vertical movement but no lateral
    movement; it is used for varved clays (FHWA NHI-06-088 Sec. 2.4).
    sigma_z = I q0, I from UFS 3-220-10 Fig. 4-10 (nu = 0) with m = B/z and
    n = L/z (:func:`westergaard_influence_factor`).

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
    """
    if z <= 0:
        return q
    return q * westergaard_influence_factor(B / z, L / z)


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
        Footing length (m). ``math.inf`` is a strip: each method's limit
        L -> infinity, below the strip's centre (GM-142).
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
    if math.isinf(L):
        if location != "center":
            raise ValueError("an infinitely long strip has no corner; "
                             "location 'center' only")
        if method in ("2:1", "2to1"):
            return approximate_2to1_strip(q, B, z)
        if method == "boussinesq":
            return boussinesq_center_strip(q, B, z)
        if method == "westergaard":
            return westergaard_center_strip(q, B, z)
    if method == "2:1" or method == "2to1":
        return approximate_2to1(q, B, L, z)
    elif method == "boussinesq":
        if location == "center":
            return boussinesq_center_rectangular(q, B, L, z)
        else:
            return boussinesq_rectangular(q, B, L, z)
    elif method == "westergaard":
        if location == "center":
            # z <= 0 guard: the corner function returns q at z <= 0, so the
            # 4x superposition would give 4q at exactly z = 0 (SET-4)
            if z <= 0:
                return q
            return 4.0 * westergaard_rectangular(q, B / 2, L / 2, z)
        else:
            return westergaard_rectangular(q, B, L, z)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Options: '2:1', 'boussinesq', 'westergaard'"
        )


#: Each stress-distribution method as a result's ``choices`` names it, with
#: its source (GM-115).
_STRESS_TEXT = {
    "2:1": ("2:1 (2V:1H spread, the average over the spread area)",
            "GEC-6 Eq. 5-17 and Fig. 5-12; FHWA NHI-06-088 Sec. 2.4.2, "
            "Fig. 2-10"),
    "boussinesq": ("Boussinesq, below the footing centre",
                   "elastic half space; four corner rectangles by UFS "
                   "3-220-10 Table 4-2 (p. 185); GEC-6 Fig. 5-11 is "
                   "Newmark's chart of the same stress"),
    "westergaard": ("Westergaard, below the footing centre",
                    "elastic half space with thin rigid layers, as in varved "
                    "clay (FHWA NHI-06-088 Sec. 2.4); four corner rectangles "
                    "by " + WESTERGAARD_READING),
}


_STRESS_NAME = {"2:1": "2:1", "boussinesq": "Boussinesq",
                "westergaard": "Westergaard"}


def stress_method_choice(method: str = "2:1") -> dict:
    """The stress-increase method as an entry of a result's ``choices``
    (``geotech_common.choices``): ``how`` is "default" for the default
    "2:1", "given" for another."""
    from geotech_common.choices import choice, how_given
    key = method.lower().replace(" ", "")
    key = "2:1" if key == "2to1" else key
    chosen, basis = _STRESS_TEXT.get(key, (method, ""))
    return choice("stress increase with depth", chosen,
                  how_given(key, "2:1", name="stress_method"), basis,
                  [f"{_STRESS_NAME[k]} (stress_method '{k}')"
                   for k in _STRESS_TEXT if k != key])
