"""
Pile Group Analysis Module

Analyzes pile groups with rigid caps under combined axial, lateral,
and moment loading. Distributes loads to individual piles and checks
each pile against allowable capacity.

Methods:
- Simplified elastic: Pi = V/n +/- M*xi/SUM(xi^2)
- General 6-DOF: stiffness matrix assembly for battered piles
- Group efficiency: Converse-Labarre, block failure, p-multipliers

References:
    CPGA User's Guide (USACE ITL-89-4)
    USACE EM 1110-2-2906
    FHWA GEC-12, Chapter 9
"""

from pile_group.pile_layout import GroupPile, create_rectangular_layout
from pile_group.group_efficiency import (
    converse_labarre, block_failure_capacity, p_multiplier,
    group_settlement_equivalent_raft,
)
from pile_group.rigid_cap import (
    GroupLoad, PileGroupResult,
    analyze_vertical_group_simple, analyze_group_6dof,
)

__all__ = [
    'GroupPile', 'create_rectangular_layout',
    'converse_labarre', 'block_failure_capacity', 'p_multiplier',
    'group_settlement_equivalent_raft',
    'GroupLoad', 'PileGroupResult',
    'analyze_vertical_group_simple', 'analyze_group_6dof',
]
