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

from settlement.immediate import (
    elastic_settlement, schmertmann_settlement, SchmertmannLayer,
    elastic_influence_factor,
)
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer, total_consolidation_settlement,
)
from settlement.stress_distribution import stress_at_depth
from settlement.time_rate import (
    time_factor, degree_of_consolidation, settlement_at_time,
    time_for_consolidation,
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

        .. note:: Single-zone drainage assumption (SET-5)
            When ``Hdr`` is not given, the entire consolidating zone
            (all ``consolidation_layers`` combined) is treated as ONE
            homogeneous layer with the stated drainage condition —
            ``Hdr = sum(thickness)/2`` for "double", ``sum(thickness)``
            for "single". Interbedded drainage layers (sand seams) within
            the zone are not detected; supply ``Hdr`` explicitly to model
            them.

    C_alpha : float, optional
        Secondary compression index. If None, no secondary settlement.
    e0_secondary : float, optional
        Void ratio for secondary settlement. Default 1.0.
    t_secondary : float, optional
        Time for secondary settlement (years), measured from the end of
        primary consolidation t1. Default 0.

        .. note:: End-of-primary t1 assumption (SET-5)
            Secondary compression runs from t1 (end of primary) to
            ``t1 + t_secondary``. If ``cv`` is provided, t1 is estimated
            as the time to 95% consolidation (t95). If ``cv`` is None,
            t1 defaults to **1.0 year** — an arbitrary but common
            placeholder. The secondary settlement scales with
            log10(t2/t1), so the choice of t1 matters; provide ``cv``
            (or check t1) for time-sensitive estimates.

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
    gamma_soil: Optional[float] = None  # for Schmertmann Izp sigma'_vp at peak depth
    Iw_immediate: Optional[float] = None  # elastic influence factor override

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
            Hdr = self._get_Hdr()
            result.cv = self.cv
            result.Hdr = Hdr
            result.t50_years = time_for_consolidation(50.0, self.cv, Hdr)
            result.t90_years = time_for_consolidation(90.0, self.cv, Hdr)

        result.choices = self._choices(result)
        result.warnings = self._warnings(result)
        return result

    def _choices(self, result: SettlementResult) -> list:
        """What this analysis chose (GM-115, ``geotech_common.choices``):
        the immediate-settlement method and its influence factor, the
        stress increase under the footing, the drainage path and the start
        of secondary compression."""
        from geotech_common.choices import (
            DEFAULT, GIVEN, MODULE, choice, how_given,
        )
        from settlement.stress_distribution import stress_method_choice
        out = []
        how = how_given(self.immediate_method, "elastic",
                        name="immediate_method")
        if self.q_net > 0 and self.immediate_method == "elastic":
            if self.Es_immediate is None or self.Es_immediate <= 0:
                out.append(choice(
                    "immediate settlement", "not computed: no Es given",
                    MODULE, "", ["the elastic method with Es given (Es)",
                                 "Schmertmann (immediate_method "
                                 "'schmertmann')"]))
            else:
                Iw = self.Iw_immediate
                given_iw = Iw is not None
                if not given_iw:
                    Iw = elastic_influence_factor(self.footing_shape,
                                                  self.L, self.B)
                out.append(choice(
                    "immediate settlement", f"elastic, Iw = {Iw:.3g}"
                    + ("" if given_iw else " (flexible footing, centre)"),
                    how, "Se = q B (1 - nu^2) Iw / Es on an elastic half "
                    "space"
                    + ("" if given_iw else "; Iw is the shape factor Cd at "
                       "the centre of a flexible area (GEC-6 Table 5-12)"),
                    ["Schmertmann (immediate_method 'schmertmann')"]))
        elif self.q_net > 0 and self.immediate_method == "schmertmann":
            out.append(choice(
                "immediate settlement",
                "Schmertmann (1978)" if self.schmertmann_layers else
                "not computed: no schmertmann_layers given", how,
                (f"creep factor C2 for {self.time_years_schmertmann:g} years"
                 if self.time_years_schmertmann > 0 else
                 "no creep factor (time_years_schmertmann = 0)"),
                ["elastic (immediate_method 'elastic')"]))
        if self.consolidation_layers and self.q_net > 0:
            out.append(stress_method_choice(self.stress_method))
        # Hdr is used only with a cv (the time curve and t95).
        if self.cv is not None and self.consolidation_layers:
            Hdr = self._get_Hdr()
            if self.Hdr is not None:
                out.append(choice("drainage path", f"Hdr = {Hdr:g} m",
                                  GIVEN))
            else:
                H = sum(l.thickness for l in self.consolidation_layers)
                out.append(choice(
                    "drainage path",
                    f"Hdr = {Hdr:g} m ({self.drainage} drainage of "
                    f"{H:g} m)", MODULE,
                    "the consolidating layers taken as one layer; a "
                    "module does not look for a drainage layer inside them",
                    ["a drainage path given (Hdr)",
                     "single drainage (drainage 'single')"
                     if self.drainage == "double" else
                     "double drainage (drainage 'double')"]))
        if result.secondary:
            if self.cv is not None:
                Hdr = self._get_Hdr()
                t1 = time_for_consolidation(95.0, self.cv, Hdr)
                out.append(choice(
                    "end of primary consolidation t1",
                    f"{t1:.3g} years (t95 from cv)", MODULE,
                    "secondary settlement from t1 to t1 + t_secondary"))
            else:
                out.append(choice(
                    "end of primary consolidation t1", "1 year (assumed)",
                    DEFAULT, "no cv given; Ss scales with log10(t2/t1)",
                    ["t95 from a given cv (cv)"]))
            out.append(choice(
                "void ratio e0 for secondary compression",
                f"{self.e0_secondary:g}",
                how_given(self.e0_secondary, 1.0, name="e0_secondary"),
                "Ss = C_alpha / (1 + e0) H log10(t2/t1); the consolidation "
                "layers' own e0 is not used", ["e0 given (e0_secondary)"]))
        return out

    def _warnings(self, result: SettlementResult) -> list:
        """What the user must be told about the result (GM-115)."""
        out = []
        if result.secondary and self.cv is None:
            out.append(
                "Secondary settlement starts at an assumed t1 = 1 year (the "
                "end of primary consolidation): no cv was given, and this "
                "placeholder is arbitrary. Ss scales with log10(t2/t1); give "
                "cv to have t1 worked out as t95.")
        return out

    def _compute_immediate(self) -> float:
        """Compute immediate settlement."""
        q_net = self.q_net
        if q_net <= 0:
            return 0.0

        if self.immediate_method == "elastic":
            if self.Es_immediate is None or self.Es_immediate <= 0:
                return 0.0
            Iw = self.Iw_immediate
            if Iw is None:
                Iw = elastic_influence_factor(self.footing_shape, self.L, self.B)
            return elastic_settlement(q_net, self.B, self.Es_immediate,
                                       self.nu, Iw=Iw)
        elif self.immediate_method == "schmertmann":
            if self.schmertmann_layers is None:
                return 0.0
            return schmertmann_settlement(
                q_net, self.q_overburden, self.B,
                self.schmertmann_layers,
                footing_shape=self.footing_shape,
                time_years=self.time_years_schmertmann,
                L=self.L,
                gamma_soil=self.gamma_soil,
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
        """Compute secondary compression settlement.

        Assumptions (SET-5)
        -------------------
        * t1 (end of primary consolidation) is estimated as t95 from the
          time-rate solution when ``cv`` is available; otherwise it falls
          back to a **1.0-year default** — an assumption, not a computed
          value. Because Ss scales with log10(t2/t1), results are
          sensitive to this default when ``cv`` is omitted.
        * The whole consolidating zone is treated as a single layer of
          total thickness sum(layer.thickness) with uniform C_alpha/(1+e0).
        """
        if self.consolidation_layers is None:
            return 0.0

        # t1 = time at end of primary consolidation
        # Estimate from time-rate if cv available
        if self.cv is not None:
            Hdr = self._get_Hdr()
            from settlement.time_rate import time_for_consolidation
            t1 = time_for_consolidation(95.0, self.cv, Hdr)
        else:
            t1 = 1.0  # assumed default (see docstring, SET-5)

        total_H = sum(layer.thickness for layer in self.consolidation_layers)
        return secondary_settlement(
            self.C_alpha, total_H, t1, t1 + self.t_secondary,
            self.e0_secondary
        )

    def _get_Hdr(self) -> float:
        """Get the drainage path length.

        Assumption (SET-5): when ``Hdr`` is not supplied, the entire
        consolidating zone (all layers combined) is treated as one
        homogeneous layer with the stated drainage condition:
        Hdr = total_thickness/2 for "double" drainage, total_thickness
        for "single". Internal drainage layers (e.g. sand seams between
        clay sublayers) are NOT detected — pass ``Hdr`` explicitly to
        model them.
        """
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
