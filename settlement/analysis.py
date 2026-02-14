"""
Combined settlement analysis.

Combines immediate, primary consolidation, secondary compression,
and time-rate calculations into a single analysis interface.

All units are SI: kPa, meters, years.

References:
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 8
    USACE EM 1110-1-1904
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from settlement.immediate import elastic_settlement, schmertmann_settlement, SchmertmannLayer
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer, total_consolidation_settlement,
)
from settlement.stress_distribution import stress_at_depth
from settlement.time_rate import (
    time_factor, degree_of_consolidation, settlement_at_time,
)
from settlement.secondary import secondary_settlement
from settlement.results import SettlementResult


@dataclass
class SettlementAnalysis:
    """Combined settlement analysis for a shallow foundation.

    Parameters
    ----------
    q_applied : float
        Applied bearing pressure at footing base (kPa).
    q_overburden : float
        Overburden pressure at footing base before construction (kPa).
    B : float
        Footing width (m).
    L : float
        Footing length (m). For strip, use large value.
    footing_shape : str, optional
        "square", "circular", "rectangular", or "strip". Default "square".
    stress_method : str, optional
        Stress distribution: "2:1", "boussinesq", or "westergaard".
        Default "2:1".

    immediate_method : str, optional
        "elastic" or "schmertmann". Default "elastic".
    Es_immediate : float, optional
        Elastic modulus for immediate settlement (kPa). Used by "elastic" method.
    nu : float, optional
        Poisson's ratio for "elastic" method. Default 0.3.
    schmertmann_layers : list of SchmertmannLayer, optional
        Sublayers for Schmertmann method.
    time_years_schmertmann : float, optional
        Time for Schmertmann creep correction (years). Default 0.

    consolidation_layers : list of ConsolidationLayer, optional
        Compressible sublayers for consolidation analysis.

    cv : float, optional
        Coefficient of consolidation (m²/year). For time-rate analysis.
    Hdr : float, optional
        Drainage path length (m). For time-rate analysis.
    drainage : str, optional
        "double" or "single". Default "double".
        If "double", Hdr = total_thickness/2 (computed automatically if Hdr not given).

    C_alpha : float, optional
        Secondary compression index. If None, no secondary settlement.
    e0_secondary : float, optional
        Void ratio for secondary settlement. Default 1.0.
    t_secondary : float, optional
        Time for secondary settlement (years). Default 0.

    Examples
    --------
    >>> analysis = SettlementAnalysis(
    ...     q_applied=150, q_overburden=20, B=3.0, L=3.0,
    ...     immediate_method="elastic", Es_immediate=15000, nu=0.3,
    ...     consolidation_layers=[
    ...         ConsolidationLayer(thickness=2, depth_to_center=1,
    ...                            e0=0.8, Cc=0.3, Cr=0.05, sigma_v0=30)
    ...     ]
    ... )
    >>> result = analysis.compute()
    """
    q_applied: float = 0.0
    q_overburden: float = 0.0
    B: float = 1.0
    L: float = 1.0
    footing_shape: str = "square"
    stress_method: str = "2:1"

    # Immediate settlement parameters
    immediate_method: str = "elastic"
    Es_immediate: Optional[float] = None
    nu: float = 0.3
    schmertmann_layers: Optional[List[SchmertmannLayer]] = None
    time_years_schmertmann: float = 0.0

    # Consolidation parameters
    consolidation_layers: Optional[List[ConsolidationLayer]] = None

    # Time-rate parameters
    cv: Optional[float] = None
    Hdr: Optional[float] = None
    drainage: str = "double"

    # Secondary compression parameters
    C_alpha: Optional[float] = None
    e0_secondary: float = 1.0
    t_secondary: float = 0.0

    def __post_init__(self):
        if self.q_applied < 0:
            raise ValueError(f"Applied pressure must be non-negative, got {self.q_applied}")
        if self.B <= 0:
            raise ValueError(f"Footing width must be positive, got {self.B}")
        if self.L <= 0:
            raise ValueError(f"Footing length must be positive, got {self.L}")

    @property
    def q_net(self) -> float:
        """Net applied pressure (kPa)."""
        return self.q_applied - self.q_overburden

    def compute(self) -> SettlementResult:
        """Run the complete settlement analysis.

        Returns
        -------
        SettlementResult
            Complete results including all settlement components.
        """
        result = SettlementResult(stress_method=self.stress_method)

        # 1. Immediate settlement
        result.immediate = self._compute_immediate()
        result.immediate_method = self.immediate_method

        # 2. Primary consolidation
        if self.consolidation_layers:
            result.consolidation, result.consolidation_layers = \
                self._compute_consolidation()
            result.consolidation_method = "Cc/Cr e-log(p)"

        # 3. Secondary compression
        if self.C_alpha is not None and self.t_secondary > 0:
            result.secondary = self._compute_secondary()

        # Total
        result.total = result.immediate + result.consolidation + result.secondary

        # Time-settlement curve
        if self.cv is not None and self.consolidation_layers:
            result.time_settlement_curve = self._compute_time_curve(
                result.immediate, result.consolidation
            )

        return result

    def _compute_immediate(self) -> float:
        """Compute immediate settlement."""
        q_net = self.q_net
        if q_net <= 0:
            return 0.0

        if self.immediate_method == "elastic":
            if self.Es_immediate is None or self.Es_immediate <= 0:
                return 0.0
            return elastic_settlement(q_net, self.B, self.Es_immediate,
                                       self.nu)
        elif self.immediate_method == "schmertmann":
            if self.schmertmann_layers is None:
                return 0.0
            return schmertmann_settlement(
                q_net, self.q_overburden, self.B,
                self.schmertmann_layers,
                footing_shape=self.footing_shape,
                time_years=self.time_years_schmertmann,
                L=self.L
            )
        else:
            raise ValueError(
                f"Unknown immediate method '{self.immediate_method}'. "
                "Options: 'elastic', 'schmertmann'"
            )

    def _compute_consolidation(self):
        """Compute consolidation settlement with per-layer breakdown."""
        q_net = self.q_net
        if q_net <= 0:
            return 0.0, []

        layer_results = []
        total_Sc = 0.0

        for layer in self.consolidation_layers:
            delta_sigma = stress_at_depth(
                q_net, self.B, self.L,
                layer.depth_to_center,
                method=self.stress_method
            )
            Sc = consolidation_settlement_layer(layer, delta_sigma)
            total_Sc += Sc

            layer_results.append({
                "depth_m": layer.depth_to_center,
                "thickness_m": layer.thickness,
                "delta_sigma_kPa": round(delta_sigma, 2),
                "settlement_mm": round(Sc * 1000, 2),
                "OCR": round(layer.OCR, 2),
                "description": layer.description,
            })

        return total_Sc, layer_results

    def _compute_secondary(self) -> float:
        """Compute secondary compression settlement."""
        if self.consolidation_layers is None:
            return 0.0

        # t1 = time at end of primary consolidation
        # Estimate from time-rate if cv available
        if self.cv is not None:
            Hdr = self._get_Hdr()
            from settlement.time_rate import time_for_consolidation
            t1 = time_for_consolidation(95.0, self.cv, Hdr)
        else:
            t1 = 1.0  # default 1 year

        total_H = sum(layer.thickness for layer in self.consolidation_layers)
        return secondary_settlement(
            self.C_alpha, total_H, t1, t1 + self.t_secondary,
            self.e0_secondary
        )

    def _get_Hdr(self) -> float:
        """Get the drainage path length."""
        if self.Hdr is not None:
            return self.Hdr
        if self.consolidation_layers:
            total_H = sum(layer.thickness for layer in self.consolidation_layers)
            if self.drainage == "double":
                return total_H / 2.0
            return total_H
        return 1.0

    def _compute_time_curve(self, S_imm: float, S_consol: float):
        """Compute settlement vs time curve."""
        Hdr = self._get_Hdr()

        # Generate time points (log-spaced)
        # From 0.01 year to 100 years
        times = np.concatenate([
            [0.0],
            np.logspace(-2, 2, 50)
        ])

        curve = []
        for t in times:
            S_c = settlement_at_time(S_consol, self.cv, Hdr, t) if t > 0 else 0
            S_total = S_imm + S_c
            curve.append((float(t), float(S_total)))

        return curve
