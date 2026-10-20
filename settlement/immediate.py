"""
Immediate (elastic) settlement methods.

Computes elastic settlement for foundations on granular soils or
elastic settlements in clay.

Methods:
1. Elastic method (Timoshenko & Goodier) — simple closed-form
2. Schmertmann (1970, 1978) — strain influence factor method

All units are SI: kPa, meters.

References:
    Schmertmann, J.H. (1970) — Static cone to compute static settlement
    Schmertmann, J.H. et al. (1978) — Improved strain influence factor diagrams
    FHWA Soils & Foundations Reference Manual, Vol II, Ch 8
    FHWA GEC-6 (FHWA-IF-02-054), Section 8.3
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


#: GEC-6 Table 5-12 (p. 97): shape and rigidity factors C_d for points on
#: loaded areas at the surface of an elastic half space of infinite depth
#: (after Winterkorn & Fang, 1975), as printed. The points: centre, corner,
#: middle of the short side, middle of the long side, and the average.
GEC6_TABLE_5_12_POINTS = ("center", "corner", "mid_short", "mid_long",
                          "average")
#: Circle, flexible: centre, average, and the edge, printed 0.64 under the
#: corner and both middle-of-side columns. Rigid footings settle the same at
#: every point.
_CD_CIRCLE = {"center": 1.00, "average": 0.85, "edge": 0.64,
              "corner": 0.64, "mid_short": 0.64, "mid_long": 0.64}
_CD_RIGID = {"circular": 0.79, "square": 0.99}
#: Flexible square (L/B = 1) and rectangles by length/width, in the order of
#: GEC6_TABLE_5_12_POINTS.
_CD_RECT_LB = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0, 1000.0, 10000.0)
_CD_RECT = (
    (1.12, 0.56, 0.76, 0.76, 0.95),
    (1.36, 0.67, 0.89, 0.97, 1.15),
    (1.52, 0.76, 0.98, 1.12, 1.30),
    (1.78, 0.88, 1.11, 1.35, 1.52),
    (2.10, 1.05, 1.27, 1.68, 1.83),
    (2.53, 1.26, 1.49, 2.12, 2.25),
    (4.00, 2.00, 2.20, 3.60, 3.70),
    (5.47, 2.75, 2.94, 5.03, 5.15),
    (6.90, 3.50, 3.70, 6.50, 6.60),
)
GEC6_TABLE_5_12_BASIS = (
    "GEC-6 Table 5-12 (p. 97), shape and rigidity factors C_d for points "
    "at the surface of an elastic half space of infinite depth (after "
    "Winterkorn & Fang 1975; the same table is FHWA NHI-06-089 Table 8-13, "
    "p. 8-59), used as Iw in Se = q B (1 - nu^2) Iw / Es (GEC-6 Eq. 5-36, "
    "p. 96)")
#: GM-214: the rigid square as printed, and why it is kept.
GEC6_RIGID_SQUARE_NOTE = (
    "The table prints the rigid square at 0.99, above the flexible "
    "square's average (0.95), where its rigid circle (0.79) is below the "
    "flexible circle's average (0.85). NHI-06-089 Table 8-13 prints the "
    "same 0.99, and no other reference in the app's reference library "
    "(UFC 3-220-20, UFC 3-220-10 / DM 7.1, EN 1997-1 and -2) prints a "
    "rigid square, so 0.99 is used as printed. Give Iw for another "
    "value.")


def gec6_influence_factor(shape: str = "square", rigidity: str = "flexible",
                          point: str = "center", B: float = None,
                          L: float = None):
    """``(Iw, where)``: the influence factor C_d of GEC-6 Table 5-12 for a
    footing's shape, rigidity and point, and the table row and column it
    came from (GM-203: ``elastic_settlement`` took Iw = 1.0 for every
    footing, which matches no printed point).

    ``shape``: "square", "circular" or "rectangular" (needs ``B`` and
    ``L``; between the printed length/width rows the factor is
    interpolated in log(L/B), as Schleicher's closed form for the centre
    runs). ``rigidity``: "flexible" or "rigid" (the table prints a rigid
    circle and square only, the same at every point). ``point``: one of
    :data:`GEC6_TABLE_5_12_POINTS`, or for a circle also "edge" (the
    table prints 0.64 under the corner and both middle-of-side columns).
    A strip has no factor: it grows without limit with L/B.
    """
    shape = str(shape or "square").strip().lower()
    rigidity = str(rigidity or "flexible").strip().lower()
    point = str(point or "center").strip().lower()
    if rigidity not in ("flexible", "rigid"):
        raise ValueError(f"rigidity must be 'flexible' or 'rigid', got "
                         f"'{rigidity}'")
    if shape in ("circle", "circular"):
        shape = "circular"
    elif shape == "strip":
        raise ValueError(
            "A strip has no shape and rigidity factor: GEC-6 Table 5-12's "
            "C_d grows without limit with L/B (6.90 at the centre at L/B = "
            "10,000). Give Iw, or the footing's length with shape "
            "'rectangular'.")
    elif shape not in ("square", "rectangular", "rectangle"):
        raise ValueError(f"shape must be 'square', 'rectangular' or "
                         f"'circular', got '{shape}'")
    lb = 1.0
    if shape in ("rectangular", "rectangle"):
        if L is None or B is None or B <= 0:
            raise ValueError("A rectangular footing needs its length L and "
                             "width B for GEC-6 Table 5-12's factor.")
        lb = max(L, B) / min(L, B)
        if lb > _CD_RECT_LB[-1]:
            raise ValueError(f"L/B = {lb:g} is beyond GEC-6 Table 5-12 "
                             "(10,000); give Iw.")
        shape = "square" if lb == 1.0 else "rectangular"
    elif shape == "square" and L is not None and B is not None \
            and abs(L - B) > 1e-9 * B:
        raise ValueError(f"A square footing has L = B; got L = {L}, B = "
                         f"{B}. Use shape 'rectangular'.")
    if rigidity == "rigid":
        if shape not in _CD_RIGID:
            raise ValueError(
                "GEC-6 Table 5-12 prints a rigid circle (0.79) and a rigid "
                "square (0.99) only, no rigid rectangle: give Iw.")
        return _CD_RIGID[shape], (f"{shape} (rigid), the same at every "
                                  "point")
    if shape == "circular":
        if point not in _CD_CIRCLE:
            raise ValueError(f"A circle's points in GEC-6 Table 5-12 are "
                             f"{list(_CD_CIRCLE)}, got '{point}'")
        return _CD_CIRCLE[point], f"circle (flexible), {point}"
    if point == "edge":
        raise ValueError("'edge' is a circle's point; a square or "
                         "rectangle has corner, mid_short and mid_long.")
    if point not in GEC6_TABLE_5_12_POINTS:
        raise ValueError(f"point must be one of "
                         f"{list(GEC6_TABLE_5_12_POINTS)}, got '{point}'")
    col = GEC6_TABLE_5_12_POINTS.index(point)
    for i, x in enumerate(_CD_RECT_LB):
        if lb == x:
            row = "square" if x == 1.0 else f"rectangle L/B = {x:g}"
            return _CD_RECT[i][col], f"{row} (flexible), {point}"
        if lb < x:
            x0, y0, y1 = _CD_RECT_LB[i - 1], _CD_RECT[i - 1][col], \
                _CD_RECT[i][col]
            t = math.log(lb / x0) / math.log(x / x0)
            return y0 + t * (y1 - y0), (
                f"rectangle L/B = {lb:.3g} (flexible), {point}: between "
                f"the rows L/B = {x0:g} ({y0:.2f}) and {x:g} ({y1:.2f}), "
                "interpolated in log(L/B)")
    raise AssertionError("unreachable")


def elastic_settlement(q: float, B: float, Es: float,
                       nu: float = 0.3,
                       Iw: Optional[float] = None,
                       shape: str = "square",
                       L: Optional[float] = None,
                       rigidity: str = "flexible",
                       point: str = "center") -> float:
    """Elastic (immediate) settlement of a footing on an elastic half space.

    Se = q * B * (1 - nu²) / Es * Iw

    Parameters
    ----------
    q : float
        Net applied pressure (kPa).
    B : float
        Footing width (m).
    Es : float
        Elastic (Young's) modulus of soil (kPa).
    nu : float, optional
        Poisson's ratio. Default 0.3.
    Iw : float, optional
        Influence factor. Not given: GEC-6 Table 5-12's C_d for ``shape``,
        ``rigidity`` and ``point`` (:func:`gec6_influence_factor`), by
        default the centre of a flexible square, 1.12 (GM-203: the default
        was 1.0, which matches no printed point). Rigid: circle 0.79,
        square 0.99 as printed (:data:`GEC6_RIGID_SQUARE_NOTE`).
    shape, L, rigidity, point : optional
        The footing and the point, for the table; not used when Iw is
        given. ``L`` for a rectangle.

    Returns
    -------
    float
        Immediate settlement Se (m).

    References
    ----------
    Timoshenko & Goodier, "Theory of Elasticity".
    FHWA GEC-6 Eq. 5-36 (p. 96) and Table 5-12 (p. 97).
    """
    if Es <= 0:
        raise ValueError(f"Elastic modulus must be positive, got {Es}")
    if Iw is None:
        Iw = gec6_influence_factor(shape, rigidity, point, B=B, L=L)[0]
    return q * B * (1.0 - nu**2) / Es * Iw


def elastic_influence_factor(shape: str = "square",
                             L: float = 1.0, B: float = 1.0) -> float:
    """Influence factor Iw for the elastic settlement of a flexible footing.

    Settlement at the CENTER of a uniformly loaded flexible rectangle on an
    elastic half-space (Schleicher, 1926):

        Iw = (2/pi) * [ m*ln((1+sqrt(1+m^2))/m) + ln(m+sqrt(1+m^2)) ],  m = L/B

    This gives Iw = 1.122 for a square (GEC-6 Table 5-12 prints 1.12) and
    grows with L/B (2.54 at L/B = 10; printed 2.53). A circular footing
    uses Iw = 1.0. For a rigid footing, or another point, see
    :func:`gec6_influence_factor`.

    Parameters
    ----------
    shape : str
        "circular"/"circle" -> 1.0; otherwise rectangular (square when L == B).
    L, B : float
        Footing length and width (m); the ratio L/B (>= 1) drives the factor.

    References
    ----------
    Schleicher (1926); Timoshenko & Goodier, "Theory of Elasticity";
    Bowles, "Foundation Analysis and Design," 5th ed.
    """
    if (shape or "").lower() in ("circular", "circle"):
        return 1.0
    if B <= 0:
        raise ValueError(f"Footing width B must be positive, got {B}")
    m = max(L / B, 1.0)
    return (2.0 / math.pi) * (
        m * math.log((1.0 + math.sqrt(1.0 + m * m)) / m)
        + math.log(m + math.sqrt(1.0 + m * m))
    )


@dataclass
class SchmertmannLayer:
    """A sublayer for Schmertmann's method.

    Parameters
    ----------
    depth_top : float
        Depth from footing base to top of sublayer (m).
    depth_bottom : float
        Depth from footing base to bottom of sublayer (m).
    Es : float
        Elastic modulus of sublayer (kPa). Can be estimated from CPT:
        Es = 2.5*qc (square/circular) or Es = 3.5*qc (strip).
    """
    depth_top: float
    depth_bottom: float
    Es: float

    def __post_init__(self):
        if self.depth_bottom <= self.depth_top:
            raise ValueError(
                f"depth_bottom ({self.depth_bottom}) must be > depth_top ({self.depth_top})"
            )
        if self.Es <= 0:
            raise ValueError(f"Elastic modulus must be positive, got {self.Es}")

    @property
    def thickness(self) -> float:
        return self.depth_bottom - self.depth_top

    @property
    def depth_mid(self) -> float:
        return (self.depth_top + self.depth_bottom) / 2.0


def schmertmann_settlement(q_net: float, q0: float, B: float,
                           layers: List[SchmertmannLayer],
                           footing_shape: str = "square",
                           time_years: float = 0.0,
                           L: Optional[float] = None,
                           gamma_soil: Optional[float] = None) -> float:
    """Schmertmann's improved method (1978) for immediate settlement.

    Se = C1 * C2 * C3 * delta_q * SUM(Iz/Es * dz)

    Parameters
    ----------
    q_net : float
        Net applied pressure at footing base (kPa):
        q_net = q_applied - q_overburden.
    q0 : float
        Overburden pressure at footing base before construction (kPa).
    B : float
        Footing width (m).
    layers : list of SchmertmannLayer
        Soil sublayers below the footing base, extending to at least
        2B (square) or 4B (strip) depth.
    footing_shape : str, optional
        "square" or "circular" (peak Iz at 0.5B, influence to 2B) or
        "strip" (peak Iz at B, influence to 4B). Default "square".
    time_years : float, optional
        Time since load application (years) for creep correction C2.
        Default 0 (no creep).
    L : float, optional
        Footing length (m). If provided and L/B > 10, treat as strip.
    gamma_soil : float, optional
        Effective soil unit weight (kN/m^3) used to evaluate the effective
        overburden at the peak-influence depth for the peak strain-influence
        factor Izp = 0.5 + 0.1*sqrt(delta_q/sigma'_vp). If None, sigma'_vp
        falls back to q0 (the overburden at the footing base) -- an
        approximation.

    Returns
    -------
    float
        Settlement Se (m).

    References
    ----------
    Schmertmann, J.H. et al. (1978), "Improved Strain Influence Factor
    Diagrams", JGED, ASCE, Vol. 104, No. GT8.
    FHWA GEC-6, Section 8.3.2.
    """
    if q_net <= 0:
        return 0.0
    return schmertmann_terms(q_net, q0, B, layers, footing_shape,
                             time_years, L, gamma_soil)["Se"]


def schmertmann_terms(q_net: float, q0: float, B: float,
                      layers: List[SchmertmannLayer],
                      footing_shape: str = "square",
                      time_years: float = 0.0,
                      L: Optional[float] = None,
                      gamma_soil: Optional[float] = None) -> dict:
    """The terms of :func:`schmertmann_settlement` as it works them out
    (it returns this ``Se``): ``strip`` (the plane-strain diagram),
    ``C1``, ``C2``, ``C3``, ``z_peak``, ``z_max``, ``Iz_0``,
    ``sigma_vp`` (kPa), ``Iz_peak``, ``sum_Iz_dz_over_Es`` (m/kPa) and
    ``Se`` (m). Same parameters; ``q_net`` must be positive."""

    # Determine shape behavior
    is_strip = footing_shape.lower() == "strip"
    if L is not None and L / B > 10:
        is_strip = True

    # C1: depth correction (embedment effect)
    if q_net > 0:
        C1 = 1.0 - 0.5 * (q0 / q_net)
        C1 = max(C1, 0.5)  # minimum 0.5
    else:
        C1 = 1.0

    # C2: creep (secondary) correction
    if time_years > 0.1:
        C2 = 1.0 + 0.2 * math.log10(time_years / 0.1)
    else:
        C2 = 1.0

    # Schmertmann's method applies C1 (embedment) and C2 (creep) only. The
    # footing-shape effect is captured by selecting the axisymmetric vs
    # plane-strain strain-influence diagram below, so NO separate shape factor
    # is applied. (The previous "C3 = 1.03 - 0.03*L/B" double-counted shape and
    # has been removed -- SET-2.)
    C3 = 1.0

    # Strain-influence diagram geometry (Schmertmann 1978): axisymmetric
    # (square/circular) peaks at z = B/2 and dies out at 2B; plane-strain
    # (strip) peaks at z = B and dies out at 4B. Iz at the footing base is 0.1
    # (axisymmetric) or 0.2 (plane strain).
    if is_strip:
        z_peak = B
        z_max = 4.0 * B
        Iz_0 = 0.2
    else:
        z_peak = 0.5 * B
        z_max = 2.0 * B
        Iz_0 = 0.1

    # Peak strain influence factor Izp = 0.5 + 0.1*sqrt(delta_q / sigma'_vp),
    # where sigma'_vp is the effective overburden at the PEAK-influence depth
    # (z_peak below the footing base) -- not at the base (SET-1). Use
    # q0 + gamma*z_peak when the unit weight is supplied; otherwise q0.
    sigma_vp = q0 + gamma_soil * z_peak if gamma_soil is not None else q0
    Iz_peak = 0.5 + 0.1 * math.sqrt(q_net / sigma_vp) if sigma_vp > 0 else 0.5
    Iz_peak = min(Iz_peak, 1.0)

    def strain_influence(z: float) -> float:
        """Triangular strain influence factor Iz at depth z below footing."""
        if z <= 0 or z >= z_max:
            return 0.0
        if z <= z_peak:
            return Iz_0 + (Iz_peak - Iz_0) * z / z_peak
        return Iz_peak * (z_max - z) / (z_max - z_peak)

    # Summation: SUM(Iz/Es * dz) over all sublayers. Midpoint x thickness is
    # EXACT on each linear segment of the triangular Iz diagram, so any
    # sublayer straddling the z_peak kink is split there internally — a
    # coarse user layering can no longer clip the peak (Das-sweep ergonomics
    # note 2026-07: 4- vs 3-layer encodings of the same profile differed by
    # ~1 % from the lost kink; now both integrate the diagram exactly).
    total = 0.0
    for layer in layers:
        if layer.depth_top >= z_max:
            continue
        # Clip layer to influence zone
        zt = max(layer.depth_top, 0)
        zb = min(layer.depth_bottom, z_max)
        if zb <= zt:
            continue

        cuts = [zt, z_peak, zb] if zt < z_peak < zb else [zt, zb]
        for za, zc in zip(cuts[:-1], cuts[1:]):
            z_mid = (za + zc) / 2.0
            total += strain_influence(z_mid) / layer.Es * (zc - za)

    Se = C1 * C2 * C3 * q_net * total
    return {"strip": is_strip, "C1": C1, "C2": C2, "C3": C3,
            "z_peak": z_peak, "z_max": z_max, "Iz_0": Iz_0,
            "sigma_vp": sigma_vp, "Iz_peak": Iz_peak,
            "sum_Iz_dz_over_Es": total, "Se": Se}
