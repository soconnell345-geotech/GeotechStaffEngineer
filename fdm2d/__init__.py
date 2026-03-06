"""
2D Explicit Lagrangian Finite Difference Module for Geotechnical Analysis

FLAC-style explicit solver using numpy only — no global stiffness matrix.

Key differences from fem2d (implicit FEM):
- Element-by-element operations (no global matrix assembly)
- Explicit central-difference time stepping (no Newton-Raphson)
- Quad zones with mixed discretization (4 sub-triangles per zone)
- Dynamic relaxation with local damping for static equilibrium

Capabilities:
- Rectangular quad grid generation
- Mixed discretization (Marti & Cundall 1982) to prevent locking
- Linear elastic and Mohr-Coulomb materials
- Gravity loading, surface pressure loads
- Automatic critical timestep calculation

High-level API:
    analyze_gravity()      — gravity loading of a soil column
    analyze_foundation()   — strip load on elastic half-space

References:
    Cundall (1976) — Explicit finite difference method
    Marti & Cundall (1982) — Mixed discretization procedure
    Itasca (2019) — FLAC 8.1 Theory and Background
"""

from fdm2d.analysis import (
    analyze_gravity,
    analyze_foundation,
)
from fdm2d.results import FDMResult
from fdm2d.materials import (
    elastic_D,
    bulk_shear_moduli,
    wave_speed,
    mc_return_mapping,
)
from fdm2d.grid import (
    generate_quad_grid,
    build_sub_triangles,
    compute_sub_triangle_geometry,
    detect_boundary_gridpoints,
    compute_lumped_mass,
)
from fdm2d.zones import (
    compute_strain_rates,
    apply_mixed_discretization,
    compute_internal_forces,
    compute_gravity_forces,
    compute_surface_pressure,
    zone_averaged_stress,
)
from fdm2d.solver import (
    critical_timestep,
    solve_explicit,
)

__all__ = [
    # High-level API
    'analyze_gravity', 'analyze_foundation',
    # Results
    'FDMResult',
    # Materials
    'elastic_D', 'bulk_shear_moduli', 'wave_speed', 'mc_return_mapping',
    # Grid
    'generate_quad_grid', 'build_sub_triangles',
    'compute_sub_triangle_geometry', 'detect_boundary_gridpoints',
    'compute_lumped_mass',
    # Zones
    'compute_strain_rates', 'apply_mixed_discretization',
    'compute_internal_forces', 'compute_gravity_forces',
    'compute_surface_pressure', 'zone_averaged_stress',
    # Solver
    'critical_timestep', 'solve_explicit',
]
