"""
Immediate (elastic) settlement methods.

Computes elastic settlement for foundations on granular soils or
elastic settlements in clay.

Methods:
1. Elastic method — Se = q B (1 - nu^2) Iw / Es on an elastic half space
   (the form of GEC-6 Eq. 5-36), Iw from GEC-6 Table 5-12
2. Schmertmann (1970, 1978) — strain influence factor method

All units are SI: kPa, meters (any consistent set works).

References:
    FHWA GEC-6 (FHWA-SA-02-054), Eq. 5-36 and Table 5-12 (pp. 96, 97)
    FHWA NHI-06-089, Sec. 8.5.1.1, Eqs. 8-16 to 8-18 and Fig. 8-21
        (pp. 8-45 to 8-47), Example 8-2 (pp. 8-48 to 8-52)
    UFS 3-220-10 (2026), Eqs. 5-7 to 5-9 and Fig. 5-7 (pp. 232, 233)
    Schmertmann, J.H. et al. (1978) — Improved strain influence factor diagrams
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

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
    interpolated in log(L/B), a module reading: the rows from 10 up are
    spaced by factors of ten). ``rigidity``: "flexible" or "rigid" (the table prints a rigid
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
    FHWA GEC-6 Eq. 5-36 (p. 96) and Table 5-12 (p. 97), the same as
    FHWA NHI-06-089 Eq. 8-19 and Table 8-13 (pp. 8-58, 8-59): both print
    the form for a footing on rock, with the rock mass modulus; here Es is
    the soil's. UFS 3-220-10 Eq. 5-5 (p. 229) prints another elastic form
    for soil, s = q0 B mu0 mu1 / Es with its Fig. 5-6, which the module
    does not offer.
    """
    if Es <= 0:
        raise ValueError(f"Elastic modulus must be positive, got {Es}")
    if Iw is None:
        Iw = gec6_influence_factor(shape, rigidity, point, B=B, L=L)[0]
    return q * B * (1.0 - nu**2) / Es * Iw


def elastic_influence_factor(shape: str = "square",
                             L: float = 1.0, B: float = 1.0) -> float:
    """Influence factor Iw for the elastic settlement at the centre of a
    flexible footing: GEC-6 Table 5-12's C_d (p. 97), 1.00 for a circle,
    1.12 for a square, and a rectangle by L/B between the printed rows,
    interpolated in log(L/B) (:func:`gec6_influence_factor`). GM-285: it
    was Schleicher's closed form, which no reference in the library prints
    (1.122 for a square, 1.53 at L/B = 2 against the printed 1.12 and
    1.52). For a rigid footing, or another point, see
    :func:`gec6_influence_factor`.

    Parameters
    ----------
    shape : str
        "circular"/"circle"; otherwise a rectangle (square when L == B).
    L, B : float
        Footing length and width (m); the ratio of the longer to the
        shorter drives the factor.
    """
    if (shape or "").lower() in ("circular", "circle"):
        return gec6_influence_factor("circular")[0]
    if B <= 0:
        raise ValueError(f"Footing width B must be positive, got {B}")
    if L is None or abs(L - B) <= 1e-9 * B:
        return gec6_influence_factor("square")[0]
    return gec6_influence_factor("rectangular", B=B, L=L)[0]


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
        Elastic modulus of sublayer (kPa). From CPT, Es = 2.5 qc for
        axisymmetric (L = B) and 3.5 qc for plane strain (L/B > 10)
        conditions (UFS 3-220-10 Sec. 5-4.1.2, Fig. 5-7). NHI-06-089 uses
        X E, X = 1.25 (L/B = 1) to 1.75 (L/B >= 10) for a modulus
        correlated with N or qc (Eq. 8-16): give the product here.
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


#: What may be said of the depth of the strain influence diagram that the
#: Schmertmann layers do not cover (``below_profile``, GM-320):
#: ``"settled_by_another_method"``, not granular soil, its settlement
#: worked out by another method (NHI-06-089 App. A.7: a sand over clay,
#: the sum stopped at the bottom of the sand, the clay consolidated);
#: ``"incompressible"``, rock or other ground that does not compress;
#: ``"last_layer_continued"``, the deepest layer's Es down to the bottom of
#: the diagram. The first two count nothing there, on the same diagram.
BELOW_PROFILE = ("settled_by_another_method", "incompressible",
                 "last_layer_continued")

#: Depths closer than this (m) count as the same when the layers are
#: checked against the diagram (GM-320): a numerical tolerance, so that a
#: depth given to the centimetre passes. The strain of 10 mm at the tip of
#: the diagram is of the order of 1e-5 of the total.
COVER_TOL_M = 0.01

#: The printed case of a granular layer over ground settled otherwise.
_APP_A7 = ("NHI-06-089 App. A.7 (pp. A.7-10 to A.7-12) works a footing "
           "(L/B 14.3, plane strain) on sand over clay with the diagram "
           "unchanged and the sum stopped "
           "at the bottom of the sand: 'The immediate settlement will "
           "occur only in the sand layer'; the clay is settled by "
           "consolidation (Step 4, p. A.7-13)")

_BELOW_HELP = {
    "settled_by_another_method": (
        "not granular soil, a clay say, whose settlement is worked out by "
        "another method such as consolidation: nothing counted there, as "
        "NHI-06-089 App. A.7 stops the sum at the bottom of a sand over "
        "clay"),
    "incompressible": ("rock or other ground that does not compress: "
                       "nothing counted there"),
    "last_layer_continued": ("the deepest layer's Es down to the bottom "
                             "of the diagram, when only the depth below "
                             "the deepest layer is not covered"),
}


def _z(x: float) -> str:
    """A depth (m) for a message, to the millimetre."""
    return f"{round(x, 3):g}"


def _depth_up(z: float) -> float:
    """``z`` rounded up to the centimetre, so that a layer bottom given as
    printed covers it (GM-320)."""
    return math.ceil(z * 100.0 - 1e-6) / 100.0


def _ranges_text(ranges) -> str:
    return " and ".join(f"{_z(a)} to {_z(b)} m" for a, b in ranges)


def schmertmann_uncovered(layers: List[SchmertmannLayer],
                          z_max: float) -> List[Tuple[float, float]]:
    """The depth ranges (m below the footing base) from the base to
    ``z_max`` that no layer covers: a first layer that starts below the
    base, gaps between layers, and the depth below the deepest one. A
    range of ``COVER_TOL_M`` or less is not counted (GM-320)."""
    spans = sorted((max(layer.depth_top, 0.0),
                    min(layer.depth_bottom, z_max)) for layer in layers)
    out, z = [], 0.0
    for top, bottom in spans:
        if bottom <= top:     # above the base or below the diagram
            continue
        if top - z > COVER_TOL_M:
            out.append((z, top))
        z = max(z, bottom)
    if z_max - z > COVER_TOL_M:
        out.append((z, z_max))
    return out


def _overlaps(ranges, others) -> List[Tuple[float, float]]:
    """The parts of ``ranges`` that ``others`` overlap by more than
    ``COVER_TOL_M``."""
    out = []
    for a, b in ranges:
        for c, d in others:
            lo, hi = max(a, c), min(b, d)
            if hi - lo > COVER_TOL_M:
                out.append((lo, hi))
    return sorted(out)


def schmertmann_settlement(q_net: float, q0: float, B: float,
                           layers: List[SchmertmannLayer],
                           footing_shape: str = "square",
                           time_years: float = 0.0,
                           L: Optional[float] = None,
                           gamma_soil: Optional[float] = None,
                           below_profile: Optional[str] = None) -> float:
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
        Soil sublayers covering the strain influence diagram from the
        footing base down to its bottom (2B at L/B = 1, 4B at L/B >= 10).
        A depth they do not cover (a first layer that starts below the
        base, a gap, or the depth below the deepest layer; more than
        ``COVER_TOL_M``) is refused unless ``below_profile`` says what
        lies there.
    footing_shape : str, optional
        "square" or "circular" (peak Iz at 0.5B, influence to 2B) or
        "strip" (peak Iz at B, influence to 4B). Default "square".
    time_years : float, optional
        Time since load application (years) for creep correction C2.
        Default 0: C2 = 1, as at t = 0.1 year, immediately after
        construction (NHI-06-089 Sec. 8.5.1.2).
    L : float, optional
        Footing length (m); not given, L = B. The diagram is the
        axisymmetric one at L/B = 1 and the plane-strain one at
        L/B >= 10, linearly interpolated between (NHI-06-089 Sec.
        8.5.1.1, Fig. 8-21a); B is the lesser side.
    gamma_soil : float
        Effective soil unit weight (kN/m^3) below the base, for the
        vertical effective stress at the depth of the peak,
        sigma'_vp = q0 + gamma_soil z_peak, in
        Izp = 0.5 + 0.1 sqrt(delta_q / sigma'_vp) (NHI-06-089 Fig. 8-21b;
        UFS Fig. 5-7). Required (GM-285: without it sigma'_vp was taken as
        q0, printed nowhere, and Izp as 0.5 when q0 = 0).
    below_profile : str, optional
        What lies in the depth of the diagram the layers do not cover
        (GM-320): ``"settled_by_another_method"`` (not granular soil, a
        clay say, whose settlement is worked out by another method such
        as consolidation) or ``"incompressible"`` (rock or other ground
        that does not compress): both count no strain there, on the same
        diagram, as NHI-06-089 App. A.7 (pp. A.7-10 to A.7-12) stops the
        sum at the bottom of a sand over clay; or
        ``"last_layer_continued"`` (the deepest layer's Es down to the
        bottom of the diagram; only when the depth below the deepest
        layer is all that is not covered). Not given, a depth not
        covered is refused, with the depth ranges: nobody described that
        ground. It changes nothing when the layers cover the diagram.

    Returns
    -------
    float
        Settlement Se (m).

    References
    ----------
    FHWA NHI-06-089 Eqs. 8-16 to 8-18 and Fig. 8-21 (pp. 8-45 to 8-47),
    worked in its Example 8-2; UFS 3-220-10 Eqs. 5-7 to 5-9 and Fig. 5-7
    (pp. 232, 233). (GEC-6 gives no Schmertmann method; it refers to
    Munfakh 2000.)
    Schmertmann, J.H. et al. (1978), "Improved Strain Influence Factor
    Diagrams", JGED, ASCE, Vol. 104, No. GT8.
    """
    if q_net <= 0:
        return 0.0
    return schmertmann_terms(q_net, q0, B, layers, footing_shape,
                             time_years, L, gamma_soil,
                             below_profile)["Se"]


def schmertmann_terms(q_net: float, q0: float, B: float,
                      layers: List[SchmertmannLayer],
                      footing_shape: str = "square",
                      time_years: float = 0.0,
                      L: Optional[float] = None,
                      gamma_soil: Optional[float] = None,
                      below_profile: Optional[str] = None,
                      below_param: str = "below_profile",
                      settled_elsewhere: Sequence[Tuple[float, float]] = ()
                      ) -> dict:
    """The terms of :func:`schmertmann_settlement` as it works them out
    (it returns this ``Se``): ``strip`` (the plane-strain diagram),
    ``L_over_B`` (infinite for a strip), ``plane_strain_share`` (0 for the
    axisymmetric diagram, 1 for the plane-strain one, linear between),
    ``C1``, ``C2``, ``C3``, ``z_peak``, ``z_max``, ``Iz_0``,
    ``sigma_vp`` (kPa), ``Iz_peak``, ``sum_Iz_dz_over_Es`` (m/kPa),
    ``Se`` (m), ``profile_bottom`` (m, the bottom of the deepest layer),
    ``uncovered`` (the depth ranges of the diagram no layer covers, m
    below the base; empty when covered) and ``below_profile`` (None when
    the layers cover the diagram; else what was taken in ``uncovered``).
    Same parameters; ``q_net`` must be positive. Two more, for a caller
    with its own names (GM-320): ``below_param``, the name of
    ``below_profile`` in the messages, and ``settled_elsewhere``, depth
    ranges (m below the base) the same call settles by another method
    (the consolidation layers of a combined analysis): where they lie in
    a depth not covered, only ``"settled_by_another_method"`` is taken
    and offered."""
    if below_profile is not None and below_profile not in BELOW_PROFILE:
        raise ValueError(
            f"{below_param} must be one of {list(BELOW_PROFILE)}, "
            f"got {below_profile!r}")
    if not layers:
        raise ValueError(
            "no Schmertmann layers given: give at least one {depth_top, "
            "depth_bottom, Es}, depths in m below the footing base")
    if gamma_soil is None:
        raise ValueError(
            "Schmertmann's peak strain influence factor Izp = 0.5 + 0.1 "
            "sqrt(delta_q / sigma'_vp) needs the vertical effective stress "
            "at the depth of the peak (NHI-06-089 Fig. 8-21b; UFS 3-220-10 "
            "Fig. 5-7): give gamma_soil, the effective unit weight below "
            "the footing base (kN/m3).")

    # The diagram (NHI-06-089 Fig. 8-21a): axisymmetric at L/B = 1,
    # plane strain at L/B >= 10, linearly interpolated between; B is the
    # least lateral dimension.
    if footing_shape.lower() == "strip" or (L is not None and math.isinf(L)):
        lb = math.inf
    else:
        Lx = B if L is None else L
        lb = max(Lx, B) / min(Lx, B)
        B = min(Lx, B)
    share = 1.0 if lb >= 10.0 else (lb - 1.0) / 9.0
    is_strip = share >= 1.0

    # C1: depth correction (embedment effect), NHI-06-089 Eq. 8-17,
    # UFS Eq. 5-8
    if q_net > 0:
        C1 = 1.0 - 0.5 * (q0 / q_net)
        C1 = max(C1, 0.5)  # minimum 0.5
    else:
        C1 = 1.0

    # C2: creep correction, NHI-06-089 Eq. 8-18, UFS Eq. 5-9; t = 0.1 year
    # (C2 = 1) immediately after construction (NHI-06-089 Sec. 8.5.1.2)
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

    # Strain-influence diagram geometry (NHI-06-089 Fig. 8-21a, UFS
    # Fig. 5-7): axisymmetric peaks at z = B/2 and dies out at 2B;
    # plane strain peaks at z = B and dies out at 4B. Iz at the footing
    # base is 0.1 (axisymmetric) or 0.2 (plane strain).
    z_peak = (0.5 + 0.5 * share) * B
    z_max = (2.0 + 2.0 * share) * B
    Iz_0 = 0.1 + 0.1 * share

    # Peak strain influence factor Izp = 0.5 + 0.1*sqrt(delta_q / sigma'_vp),
    # where sigma'_vp is the effective overburden at the PEAK-influence depth
    # (z_peak below the footing base) -- not at the base (SET-1). No cap:
    # the print gives none (GM-285; it was held to 1.0).
    sigma_vp = q0 + gamma_soil * z_peak
    if sigma_vp <= 0:
        raise ValueError(
            "the vertical effective stress at the depth of the peak, "
            f"q0 + gamma_soil z_peak = {sigma_vp:g}, must be positive")
    Iz_peak = 0.5 + 0.1 * math.sqrt(q_net / sigma_vp)

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
    # The layers must cover the diagram from the footing base down
    # (GM-320): the sum of Eq. 8-16 is over "the soil layers within the
    # zone of strain influence" (NHI-06-089 p. 8-45). A depth not covered
    # is refused unless the caller says what lies there; the two values
    # that count nothing there follow NHI-06-089 App. A.7, which stops
    # the sum at the bottom of a sand over clay on the same diagram.
    profile_bottom = max(layer.depth_bottom for layer in layers)
    gaps = schmertmann_uncovered(layers, z_max)
    settled = _overlaps(gaps, settled_elsewhere)
    if gaps and below_profile is None:
        offer = ("settled_by_another_method",) if settled else BELOW_PROFILE
        raise ValueError(
            "the Schmertmann layers do not cover the strain influence "
            "diagram, which runs from the footing base to "
            f"{_z(z_max)} m ({z_max / B:.3g} B, NHI-06-089 Fig. 8-21a): "
            f"{_ranges_text(gaps)} below the footing base "
            + ("is" if len(gaps) == 1 else "are")
            + " not covered, and the strain there would be left out of "
            "the settlement. Give layers from the footing base down to "
            f"{_depth_up(z_max):g} m with no gaps, or say what lies there "
            f"with {below_param}: "
            + (f"the consolidation layers of this call lie at "
               f"{_ranges_text(settled)} and are settled by consolidation, "
               "so " if settled else "")
            + "; ".join(f"'{v}' ({_BELOW_HELP[v]})" for v in offer))
    if gaps and settled and below_profile != "settled_by_another_method":
        raise ValueError(
            f"{below_param} '{below_profile}' "
            + ("calls the ground at " if below_profile == "incompressible"
               else "counts the deepest layer's Es over ")
            + f"{_ranges_text(settled)} below the footing base, where the "
            "same call consolidates its consolidation layers: give "
            f"{below_param} 'settled_by_another_method' ("
            + _BELOW_HELP["settled_by_another_method"] + ")")
    bottom_gap = (gaps and gaps[-1][1] >= z_max
                  and abs(gaps[-1][0] - profile_bottom) <= COVER_TOL_M)
    if gaps and below_profile == "last_layer_continued":
        others = gaps[:-1] if bottom_gap else gaps
        if others:
            raise ValueError(
                f"{below_param} 'last_layer_continued' continues the "
                f"deepest layer below its bottom ({_z(profile_bottom)} m); "
                f"it says nothing of {_ranges_text(others)} below the "
                "footing base, which no layer covers: give layers there, "
                f"or {below_param} 'settled_by_another_method' or "
                "'incompressible'")
    used_below = below_profile if gaps else None
    sum_layers = list(layers)
    if used_below == "last_layer_continued":
        deepest = max(layers, key=lambda layer: layer.depth_bottom)
        sum_layers.append(SchmertmannLayer(
            depth_top=profile_bottom, depth_bottom=z_max, Es=deepest.Es))

    total = 0.0
    for layer in sum_layers:
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
    return {"strip": is_strip, "L_over_B": lb, "plane_strain_share": share,
            "C1": C1, "C2": C2, "C3": C3,
            "z_peak": z_peak, "z_max": z_max, "Iz_0": Iz_0,
            "sigma_vp": sigma_vp, "Iz_peak": Iz_peak,
            "sum_Iz_dz_over_Es": total, "Se": Se,
            "profile_bottom": profile_bottom,
            "uncovered": [list(g) for g in gaps] if used_below else [],
            "below_profile": used_below}


def _below_text(terms: dict) -> str:
    """What was taken in the depth the layers do not cover (GM-320)."""
    where = _ranges_text(terms["uncovered"]) + " below the footing base"
    kind = terms["below_profile"]
    if kind == "last_layer_continued":
        return ("the deepest layer's Es continued over " + where
                + ", to the bottom of the strain influence diagram")
    if kind == "settled_by_another_method":
        return ("the ground at " + where + " as not granular soil whose "
                "settlement is worked out by another method: its strain is "
                "not counted in this immediate settlement, on the same "
                "strain influence diagram")
    return ("the ground at " + where + " as incompressible (rock): no "
            "strain counted there, on the same strain influence diagram")


_BELOW_BASIS = {
    "settled_by_another_method": (
        _APP_A7 + "; UFS 3-220-10 Sec. 5-4.1.2 (p. 231) gives the method "
        "for coarse-grained soils"),
    "incompressible": (
        "ground that does not compress adds no strain; the sum taken over "
        "the layers only, on the same diagram, as NHI-06-089 App. A.7 "
        "(pp. A.7-10 to A.7-12) stops it at the bottom of a sand over "
        "clay; no held page redraws the diagram for a rigid layer"),
    "last_layer_continued": (
        "Eq. 8-16 summed over the whole zone of strain influence "
        "(NHI-06-089 p. 8-45), the deepest layer's Es taken down to its "
        "bottom"),
}


def schmertmann_below_choice(terms: Optional[dict],
                             param: str = "below_profile"
                             ) -> Optional[dict]:
    """The ``choices`` entry for what was taken in the depth of the
    diagram the Schmertmann layers do not cover; None when they cover it
    (GM-320). ``param``: the input's name in the caller's tool."""
    if not terms or not terms.get("below_profile"):
        return None
    from geotech_common.choices import GIVEN, choice
    kind = terms["below_profile"]
    gaps = terms["uncovered"]
    below_only = (len(gaps) == 1 and abs(gaps[0][0] - terms["profile_bottom"])
                  <= COVER_TOL_M)
    others = [b for b in BELOW_PROFILE if b != kind
              and (b != "last_layer_continued" or below_only)]
    return choice(
        "ground the Schmertmann layers do not cover", _below_text(terms),
        GIVEN, f"stated ({param}): " + _BELOW_BASIS[kind],
        [f"{param} '{b}'" for b in others]
        + [f"layers given from the footing base down to "
           f"{_depth_up(terms['z_max']):g} m with no gaps"])


def schmertmann_below_warning(terms: Optional[dict],
                              param: str = "below_profile"
                              ) -> Optional[str]:
    """The warning for the same case; None when the layers cover the
    diagram."""
    if not terms or not terms.get("below_profile"):
        return None
    text = ("The Schmertmann layers do not cover the strain influence "
            f"diagram, which runs to {_z(terms['z_max'])} m below the "
            "footing base: the settlement takes " + _below_text(terms)
            + f", as given ({param}), not as explored.")
    if terms["below_profile"] == "settled_by_another_method":
        text += (" Its settlement comes from its own method "
                 "(consolidation, for a clay), not from this one.")
    return text
