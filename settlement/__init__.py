"""
Settlement Analysis Module

Computes immediate (elastic), primary consolidation, time-rate,
and secondary compression settlement for shallow foundations.

Methods:
- Immediate: Elastic method, Schmertmann (1978)
- Consolidation: Cc/Cr e-log(p) with layer summation
- Time rate: Terzaghi 1-D theory
- Secondary: C_alpha creep
- Stress distribution: 2:1 approximate, Boussinesq, Westergaard

References:
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 8
    USACE EM 1110-1-1904
    FHWA-TS-86-205 (CSETT User's Guide)
"""

from settlement.stress_distribution import (
    stress_at_depth, approximate_2to1, boussinesq_center_rectangular,
)
from settlement.immediate import (
    elastic_settlement, schmertmann_settlement, SchmertmannLayer,
)
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer,
    total_consolidation_settlement,
)
from settlement.time_rate import (
    time_factor, degree_of_consolidation, time_for_consolidation,
    settlement_at_time,
)
from settlement.secondary import secondary_settlement
from settlement.analysis import SettlementAnalysis
from settlement.results import SettlementResult

__all__ = [
    'stress_at_depth', 'approximate_2to1', 'boussinesq_center_rectangular',
    'elastic_settlement', 'schmertmann_settlement', 'SchmertmannLayer',
    'ConsolidationLayer', 'consolidation_settlement_layer',
    'total_consolidation_settlement',
    'time_factor', 'degree_of_consolidation', 'time_for_consolidation',
    'settlement_at_time',
    'secondary_settlement',
    'SettlementAnalysis', 'SettlementResult',
]
