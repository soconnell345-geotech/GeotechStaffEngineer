"""
Support of Excavation (SOE) Module

Designs multi-level braced and cantilever excavation support walls
using Terzaghi-Peck apparent earth pressure envelopes and the
tributary area method.

Supports:
- Multi-level braced excavations (struts, anchors, rakers)
- Cantilever (unbraced) excavation walls
- Sand, soft clay, and stiff clay soil profiles
- HP section, sheet pile, and W section selection
- Embedment depth computation (USACE method)
- Stability checks: basal heave, bottom blowout, piping

References:
    Terzaghi & Peck (1967) Soil Mechanics in Engineering Practice
    FHWA-IF-99-015, GEC-4: Ground Anchors and Anchored Systems
    California Trenching and Shoring Manual (2011)
    USACE EM 1110-2-2504
    AISC Steel Construction Manual, 16th Edition
"""

from soe.geometry import ExcavationGeometry, SOEWallLayer, SupportLevel
from soe.earth_pressure import (
    rankine_Ka,
    rankine_Kp,
    select_apparent_pressure,
)
from soe.beam_analysis import (
    analyze_braced_excavation,
    analyze_cantilever_excavation,
)
from soe.embedment import compute_embedment
from soe.wall_sections import (
    select_hp_section,
    select_sheet_pile,
    select_w_section,
    check_flexural_demand,
)
from soe.results import (
    BracedExcavationResult,
    CantileverExcavationResult,
    StabilityCheckResult,
    AnchorDesignResult,
)
from soe.stability import (
    check_basal_heave_terzaghi,
    check_basal_heave_bjerrum_eide,
    check_bottom_blowout,
    check_piping,
)
from soe.anchor_design import (
    design_ground_anchor,
    compute_unbonded_length,
    compute_bond_length,
    select_tendon,
    list_bond_stress_types,
    get_bond_stress,
)

__all__ = [
    'ExcavationGeometry',
    'SOEWallLayer',
    'SupportLevel',
    'rankine_Ka',
    'rankine_Kp',
    'select_apparent_pressure',
    'analyze_braced_excavation',
    'analyze_cantilever_excavation',
    'compute_embedment',
    'select_hp_section',
    'select_sheet_pile',
    'select_w_section',
    'check_flexural_demand',
    'BracedExcavationResult',
    'CantileverExcavationResult',
    'StabilityCheckResult',
    'check_basal_heave_terzaghi',
    'check_basal_heave_bjerrum_eide',
    'check_bottom_blowout',
    'check_piping',
    'AnchorDesignResult',
    'design_ground_anchor',
    'compute_unbonded_length',
    'compute_bond_length',
    'select_tendon',
    'list_bond_stress_types',
    'get_bond_stress',
]
