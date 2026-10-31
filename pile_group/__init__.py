"""
Pile Group Analysis Module

Analyzes pile groups with rigid caps under combined axial, lateral,
and moment loading. Distributes loads to individual piles and checks
each pile against allowable capacity.

Methods:
- Simplified elastic: Pi = V/n +/- M*xi/SUM(xi^2)
- General 6-DOF: stiffness matrix assembly for battered piles
- Group resistance: the group efficiency factor eta_g (GEC-12 Sec. 7.2.2;
  UFC 3-220-20 Table 6-26), block failure (GEC-12 Eqs. 7-36, 7-37)
- Lateral group effects: p-multipliers (AASHTO, as GEC-12 and UFC
  3-220-20 Table 6-38 print them; Mokwa 1999)
- Group settlement: Meyerhof (1976), the equivalent footing

References:
    CPGA User's Guide (USACE ITL-89-4)
    USACE EM 1110-2-2906
    FHWA GEC-12 (FHWA-NHI-16-009), Vol. I ch. 7
    UFC 3-220-20 (2025), ch. 6
"""

from pile_group.pile_layout import GroupPile, create_rectangular_layout
from pile_group.group_efficiency import (
    group_efficiency_factor, block_failure_resistance, BLOCK_FAILURE_PHI,
    block_failure_capacity, group_plan_dimensions, p_multiplier,
    group_settlement_equivalent_raft, meyerhof_group_settlement,
)
from pile_group.rigid_cap import (
    GroupLoad, PileGroupResult,
    analyze_vertical_group_simple, analyze_group_6dof,
)

__all__ = [
    'GroupPile', 'create_rectangular_layout',
    'group_efficiency_factor', 'block_failure_resistance',
    'BLOCK_FAILURE_PHI',
    'block_failure_capacity', 'group_plan_dimensions', 'p_multiplier',
    'group_settlement_equivalent_raft', 'meyerhof_group_settlement',
    'GroupLoad', 'PileGroupResult',
    'analyze_vertical_group_simple', 'analyze_group_6dof',
]
