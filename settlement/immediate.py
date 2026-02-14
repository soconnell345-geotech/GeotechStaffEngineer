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


def elastic_settlement(q: float, B: float, Es: float,
                       nu: float = 0.3,
                       Iw: float = 1.0,
                       shape: str = "square") -> float:
    """Elastic (immediate) settlement using Timoshenko & Goodier.

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
        Influence factor (depends on shape, flexibility, depth).
        Default 1.0. For rigid circular: Iw=0.79, rigid square: Iw=0.82.
    shape : str, optional
        Not directly used if Iw is provided. For reference only.

    Returns
    -------
    float
        Immediate settlement Se (m).

    References
    ----------
    Timoshenko & Goodier, "Theory of Elasticity".
    FHWA GEC-6, Eq 8-1.
    """
    if Es <= 0:
        raise ValueError(f"Elastic modulus must be positive, got {Es}")
    return q * B * (1.0 - nu**2) / Es * Iw


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
                           L: Optional[float] = None) -> float:
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

    # C3: shape correction (Terzaghi et al., 1996 addition)
    # C3 = 1.03 - 0.03*(L/B) >= 0.73, but simplify for square vs strip
    if is_strip:
        C3 = 0.73  # for L/B -> infinity
    elif L is not None:
        C3 = 1.03 - 0.03 * (L / B)
        C3 = max(C3, 0.73)
    else:
        C3 = 1.0  # square/circular

    # Strain influence factor Iz
    if is_strip:
        z_peak = B  # peak at depth B below footing
        Iz_peak = 0.5 + 0.1 * math.sqrt(q_net / q0) if q0 > 0 else 0.5
        Iz_peak = min(Iz_peak, 1.0)
        z_max = 4.0 * B  # influence extends to 4B
    else:
        z_peak = 0.5 * B  # peak at depth B/2 below footing
        Iz_peak = 0.5 + 0.1 * math.sqrt(q_net / q0) if q0 > 0 else 0.5
        Iz_peak = min(Iz_peak, 1.0)
        z_max = 2.0 * B  # influence extends to 2B

    def strain_influence(z: float) -> float:
        """Triangular strain influence factor Iz at depth z below footing."""
        if z <= 0 or z >= z_max:
            return 0.0
        if z <= z_peak:
            # Linear increase from Iz=0.1 at z=0 to Iz_peak at z_peak
            # (Schmertmann 1978 starts at Iz=0.1 at z=0 for square,
            #  Iz=0.2 at z=0 for strip)
            Iz_0 = 0.2 if is_strip else 0.1
            return Iz_0 + (Iz_peak - Iz_0) * z / z_peak
        else:
            # Linear decrease from Iz_peak at z_peak to 0 at z_max
            return Iz_peak * (z_max - z) / (z_max - z_peak)

    # Summation: SUM(Iz/Es * dz) over all sublayers
    total = 0.0
    for layer in layers:
        if layer.depth_top >= z_max:
            continue
        # Clip layer to influence zone
        zt = max(layer.depth_top, 0)
        zb = min(layer.depth_bottom, z_max)
        if zb <= zt:
            continue

        z_mid = (zt + zb) / 2.0
        dz = zb - zt
        Iz = strain_influence(z_mid)
        total += Iz / layer.Es * dz

    Se = C1 * C2 * C3 * q_net * total
    return Se
