"""
Hough (granular / C'-index) settlement method.

Computes settlement of granular (cohesionless) soils below a shallow
foundation using the Hough (1959) bearing-capacity-index method, as
presented in FHWA GEC-6 (Shallow Foundations).

Per granular sublayer the compression is

    dH = (H / C') * log10[(sigma'_vo + delta_sigma) / sigma'_vo]

where
    H            = layer thickness (m),
    C'           = Hough bearing-capacity index (dimensionless), correlated
                   to the corrected SPT blow count N' (GEC-6 Fig 5-19),
    sigma'_vo    = effective overburden at the layer mid-depth (kPa),
    delta_sigma  = vertical stress increase at the layer mid-depth (kPa),
                   computed here with the 2:1 (2V:1H) load-spread method.

The total settlement is the sum of dH over all layers.

This is distinct from the module's Cc/Cr e-log(p) consolidation method:
the index C' is a granular *bearing-capacity index*, NOT Cc/(1+e0). Hough
is the appropriate method for sands/gravels where SPT data is available.

All units are SI: kPa, meters; settlement is reported in meters (with mm
convenience fields on the result).

References:
    Hough, B.K. (1959), "Compressibility as the Basis for Soil Bearing
    Value", JSMFD, ASCE, Vol. 85, No. SM4.
    FHWA GEC-6 (FHWA-SA-02-054), "Shallow Foundations", Section 5 and
    Appendix B, Example 1 (Tables B1-2 / B1-3).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from settlement.stress_distribution import approximate_2to1


@dataclass
class HoughLayer:
    """A granular sublayer for the Hough settlement method.

    Parameters
    ----------
    thickness : float
        Layer thickness H (m).
    depth_to_center : float
        Depth from the footing base to the center (mid-depth) of the
        sublayer (m). Used to evaluate the 2:1 stress increase.
    sigma_v0 : float
        Initial effective vertical (overburden) stress at the center of the
        layer (kPa). Must be > 0.
    C_prime : float
        Hough bearing-capacity index C' (dimensionless), correlated to the
        corrected SPT N' (GEC-6 Fig 5-19). Must be > 0.
    description : str, optional
        Layer description.
    """
    thickness: float
    depth_to_center: float
    sigma_v0: float
    C_prime: float
    description: str = ""

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.depth_to_center < 0:
            raise ValueError(
                f"depth_to_center must be non-negative, got {self.depth_to_center}")
        if self.sigma_v0 <= 0:
            raise ValueError(
                f"Initial effective stress must be positive, got {self.sigma_v0}")
        if self.C_prime <= 0:
            raise ValueError(
                f"Hough index C' must be positive, got {self.C_prime}")


def hough_settlement_layer(layer: HoughLayer, delta_sigma: float) -> float:
    """Compute Hough settlement for a single granular sublayer.

    dH = H/C' * log10[(sigma'_vo + delta_sigma) / sigma'_vo]

    Parameters
    ----------
    layer : HoughLayer
        The granular sublayer.
    delta_sigma : float
        Vertical stress increase at the center of the layer (kPa).

    Returns
    -------
    float
        Settlement of this layer (m). Zero if delta_sigma <= 0.

    References
    ----------
    Hough (1959); FHWA GEC-6, Section 5.
    """
    if delta_sigma <= 0:
        return 0.0
    return (layer.thickness / layer.C_prime
            * math.log10((layer.sigma_v0 + delta_sigma) / layer.sigma_v0))


@dataclass
class HoughResult:
    """Results from a Hough (granular C'-index) settlement analysis.

    Attributes
    ----------
    total : float
        Total Hough settlement (m).
    layers : list of dict
        Per-layer breakdown: depth_m, thickness_m, delta_sigma_kPa,
        sigma_v0_kPa, C_prime, settlement_mm.
    q_net : float
        Net applied pressure used (kPa).
    B, L : float
        Footing width and length used (m).
    stress_method : str
        Stress distribution method (always "2:1" for Hough here).
    """
    total: float = 0.0
    layers: List[Dict[str, Any]] = field(default_factory=list)
    q_net: float = 0.0
    B: float = 0.0
    L: float = 0.0
    stress_method: str = "2:1"

    @property
    def total_mm(self) -> float:
        """Total settlement in mm."""
        return self.total * 1000.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 60,
            "  HOUGH GRANULAR SETTLEMENT (C'-index)",
            "=" * 60,
            "",
            f"  Footing: B = {self.B:.2f} m, L = {self.L:.2f} m, "
            f"q_net = {self.q_net:.1f} kPa",
            f"  Stress distribution: {self.stress_method}",
            "",
            "  Layer Breakdown:",
        ]
        for i, lyr in enumerate(self.layers):
            lines.append(
                f"    Layer {i+1}: dH={lyr['settlement_mm']:.1f} mm, "
                f"delta_sigma={lyr['delta_sigma_kPa']:.1f} kPa, "
                f"sigma'_vo={lyr['sigma_v0_kPa']:.1f} kPa, "
                f"C'={lyr['C_prime']:.0f}"
            )
        lines.extend([
            f"  {'-'*44}",
            f"  TOTAL Hough settlement:   {self.total_mm:8.1f} mm",
            "",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for LLM agent consumption."""
        return {
            "total_settlement_m": round(self.total, 6),
            "total_settlement_mm": round(self.total_mm, 2),
            "method": "hough",
            "stress_method": self.stress_method,
            "layer_breakdown": self.layers,
        }


def hough_settlement(layers: List[HoughLayer], q_net: float, B: float,
                     L: Optional[float] = None) -> HoughResult:
    """Total granular settlement by the Hough (1959) C'-index method.

    Sums the per-layer Hough compression
    dH = H/C'*log10[(sigma'_vo + delta_sigma)/sigma'_vo] over all granular
    sublayers, where the stress increase delta_sigma at each layer mid-depth
    is computed with the 2:1 (2V:1H) load-spread method (GEC-6 Appendix B,
    Example 1).

    Parameters
    ----------
    layers : list of HoughLayer
        Granular sublayers below the footing base. Each carries its thickness,
        mid-depth (depth_to_center, from the footing base), effective
        overburden sigma_v0, and Hough index C'.
    q_net : float
        Net applied pressure at the footing base (kPa):
        q_net = q_applied - q_overburden.
    B : float
        Footing width (m).
    L : float, optional
        Footing length (m). Defaults to B (square footing). For a square
        footing the 2:1 increase reduces to q*B^2/(B+Z)^2; for a rectangle it
        is q*B*L/((B+Z)(L+Z)).

    Returns
    -------
    HoughResult
        Total settlement (m) plus a per-layer breakdown.

    Notes
    -----
    delta_sigma is computed from the 2:1 method (the convention used in the
    GEC-6 worked example); sigma_v0 is supplied per layer (not computed from a
    unit-weight profile here), matching how the published example tabulates the
    effective overburden at each layer mid-depth.

    References
    ----------
    Hough (1959); FHWA GEC-6 (FHWA-SA-02-054), Appendix B, Example 1.
    """
    if B <= 0:
        raise ValueError(f"Footing width B must be positive, got {B}")
    if L is None:
        L = B
    if L <= 0:
        raise ValueError(f"Footing length L must be positive, got {L}")

    result = HoughResult(q_net=q_net, B=B, L=L, stress_method="2:1")
    if q_net <= 0:
        return result

    total = 0.0
    for layer in layers:
        delta_sigma = approximate_2to1(q_net, B, L, layer.depth_to_center)
        dH = hough_settlement_layer(layer, delta_sigma)
        total += dH
        result.layers.append({
            "depth_m": layer.depth_to_center,
            "thickness_m": layer.thickness,
            "delta_sigma_kPa": round(delta_sigma, 2),
            "sigma_v0_kPa": layer.sigma_v0,
            "C_prime": layer.C_prime,
            "settlement_mm": round(dH * 1000.0, 2),
            "description": layer.description,
        })

    result.total = total
    return result
