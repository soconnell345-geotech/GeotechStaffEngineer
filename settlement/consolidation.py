"""
Primary consolidation settlement.

Computes 1-D consolidation settlement using the e-log(p) method
(Cc/Cr compression indices) with layer summation.

All units are SI: kPa, meters.

References:
    Terzaghi, K. (1925) — Theory of 1-D consolidation
    USACE EM 1110-1-1904, Chapter 5
    FHWA Soils & Foundations Reference Manual, Vol II, Ch 8
    FHWA GEC-6 (FHWA-IF-02-054), Section 8.4
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional

from settlement.stress_distribution import stress_at_depth


@dataclass
class ConsolidationLayer:
    """A compressible sublayer for consolidation settlement.

    Parameters
    ----------
    thickness : float
        Layer thickness H (m).
    depth_to_center : float
        Depth from footing base to center of sublayer (m).
    e0 : float
        Initial void ratio.
    Cc : float
        Compression index (slope of virgin compression line on e-log(p) plot).
    Cr : float
        Recompression index (slope of recompression line).
        Typically Cr ≈ 0.1*Cc to 0.2*Cc.
    sigma_v0 : float
        Initial effective vertical stress at center of layer (kPa).
    sigma_p : float, optional
        Preconsolidation pressure (kPa). If None, soil is assumed
        normally consolidated (sigma_p = sigma_v0).
    description : str, optional
        Layer description.
    """
    thickness: float
    depth_to_center: float
    e0: float
    Cc: float
    Cr: float
    sigma_v0: float
    sigma_p: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.e0 <= 0:
            raise ValueError(f"Initial void ratio must be positive, got {self.e0}")
        if self.Cc <= 0:
            raise ValueError(f"Compression index Cc must be positive, got {self.Cc}")
        if self.Cr < 0:
            raise ValueError(f"Recompression index Cr must be non-negative, got {self.Cr}")
        if self.Cr > self.Cc:
            warnings.warn(f"Cr ({self.Cr}) > Cc ({self.Cc}); typically Cr < Cc")
        if self.sigma_v0 <= 0:
            raise ValueError(f"Initial stress must be positive, got {self.sigma_v0}")
        if self.sigma_p is None:
            self.sigma_p = self.sigma_v0  # Normally consolidated

    @property
    def OCR(self) -> float:
        """Overconsolidation ratio = sigma_p / sigma_v0."""
        return self.sigma_p / self.sigma_v0

    @property
    def is_normally_consolidated(self) -> bool:
        """True if OCR ≈ 1 (NC soil)."""
        return abs(self.OCR - 1.0) < 0.05


def consolidation_settlement_layer(layer: ConsolidationLayer,
                                   delta_sigma: float) -> float:
    """Compute consolidation settlement for a single sublayer.

    Three cases:
    1. NC soil (sigma_p ≈ sigma_v0):
       Sc = Cc*H/(1+e0) * log10((sigma_v0 + delta_sigma)/sigma_v0)
    2. OC soil, stays in OC range (sigma_v0 + delta_sigma <= sigma_p):
       Sc = Cr*H/(1+e0) * log10((sigma_v0 + delta_sigma)/sigma_v0)
    3. OC soil, exceeds preconsolidation (sigma_v0 + delta_sigma > sigma_p):
       Sc = Cr*H/(1+e0)*log10(sigma_p/sigma_v0) +
            Cc*H/(1+e0)*log10((sigma_v0+delta_sigma)/sigma_p)

    Parameters
    ----------
    layer : ConsolidationLayer
        The compressible sublayer.
    delta_sigma : float
        Stress increase at the center of the layer (kPa).

    Returns
    -------
    float
        Consolidation settlement of this layer (m).

    References
    ----------
    FHWA GEC-6, Eqs 8-5 through 8-7.
    """
    if delta_sigma <= 0:
        return 0.0

    H = layer.thickness
    e0 = layer.e0
    Cc = layer.Cc
    Cr = layer.Cr
    sigma_v0 = layer.sigma_v0
    sigma_p = layer.sigma_p
    sigma_final = sigma_v0 + delta_sigma

    if layer.is_normally_consolidated:
        # Case 1: Normally consolidated
        Sc = Cc * H / (1.0 + e0) * math.log10(sigma_final / sigma_v0)

    elif sigma_final <= sigma_p:
        # Case 2: Overconsolidated, stays in OC range
        Sc = Cr * H / (1.0 + e0) * math.log10(sigma_final / sigma_v0)

    else:
        # Case 3: Overconsolidated, but stress exceeds preconsolidation
        Sc_oc = Cr * H / (1.0 + e0) * math.log10(sigma_p / sigma_v0)
        Sc_nc = Cc * H / (1.0 + e0) * math.log10(sigma_final / sigma_p)
        Sc = Sc_oc + Sc_nc

    return Sc


def total_consolidation_settlement(
    layers: List[ConsolidationLayer],
    q_net: float,
    B: float,
    L: float,
    stress_method: str = "2:1"
) -> float:
    """Compute total consolidation settlement by layer summation.

    Divides the compressible zone into sublayers, computes the stress
    increase at each sublayer center, and sums the settlements.

    Parameters
    ----------
    layers : list of ConsolidationLayer
        Compressible sublayers. Each must have depth_to_center
        measured from the footing base.
    q_net : float
        Net applied pressure at footing base (kPa).
    B : float
        Footing width (m).
    L : float
        Footing length (m).
    stress_method : str, optional
        Stress distribution method: "2:1", "boussinesq", or "westergaard".
        Default "2:1".

    Returns
    -------
    float
        Total primary consolidation settlement (m).
    """
    total = 0.0
    for layer in layers:
        delta_sigma = stress_at_depth(q_net, B, L, layer.depth_to_center,
                                       method=stress_method)
        Sc = consolidation_settlement_layer(layer, delta_sigma)
        total += Sc
    return total
