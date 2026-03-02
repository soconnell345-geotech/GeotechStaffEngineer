"""
2D Plane-Strain Finite Element Module for Geotechnical Analysis

Custom FEM solver using numpy/scipy — no external FEM dependencies.

Capabilities:
- CST (3-node triangle) and Q4 (4-node quad) elements
- Euler-Bernoulli beam elements for structural members
- Linear elastic, Mohr-Coulomb, and Hardening Soil materials
- Delaunay mesh generation with variable density
- Gravity loading, surface loads, prescribed displacements
- Strength Reduction Method (SRM) for slope stability FOS
- Braced excavation analysis with sheet pile walls
- Pore water pressures and effective stress analysis
- Steady-state seepage (Laplace equation, CST flow elements)
- Coupled Biot consolidation (staggered scheme)

High-level API:
    analyze_gravity()        — elastic gravity loading of a soil column
    analyze_foundation()     — strip load on elastic half-space
    analyze_slope_srm()      — slope stability FOS via SRM
    analyze_excavation()     — braced excavation with wall
    analyze_seepage()        — steady-state seepage analysis
    analyze_consolidation()  — coupled Biot consolidation
    analyze_staged()         — staged construction (multi-phase)

References:
    Griffiths & Lane (1999) — Slope stability by finite elements
    Schanz, Vermeer & Bonnier (1999) — Hardening Soil model
    Clausen et al. (2006) — MC return mapping in principal stress space
    de Souza Neto et al. (2008) — Computational Methods for Plasticity
    Biot (1941) — General theory of three-dimensional consolidation
    Smith & Griffiths (2004) — Programming the Finite Element Method
"""

from fem2d.analysis import (
    analyze_gravity,
    analyze_foundation,
    analyze_slope_srm,
    analyze_excavation,
    analyze_seepage,
    analyze_consolidation,
    analyze_staged,
    create_wall_elements,
    ConstructionPhase,
    assign_element_groups,
)
from fem2d.results import (
    FEMResult, BeamForceResult, SeepageResult, ConsolidationResult,
    PhaseResult, StagedConstructionResult,
)
from fem2d.materials import (
    elastic_D, mc_return_mapping, drucker_prager_params,
    hs_return_mapping,
)
from fem2d.elements import (
    cst_stiffness, cst_B, cst_area, cst_body_force, cst_stress,
    q4_stiffness, q4_body_force, q4_stress,
    BeamElement, beam2d_stiffness, beam2d_internal_forces,
)
from fem2d.mesh import (
    generate_rect_mesh, generate_slope_mesh, generate_polygon_mesh,
    detect_boundary_nodes, assign_layers_by_elevation,
    assign_layers_by_polylines,
    points_in_polygon, triangle_quality,
)
from fem2d.assembly import (
    assemble_stiffness, assemble_gravity, assemble_surface_load,
    apply_bcs_penalty, solve_linear, recover_element_stresses,
    nodal_stresses,
    build_rotation_dof_map, beam_element_dofs,
    assemble_beam_stiffness, assemble_beam_gravity,
)
from fem2d.solver import solve_elastic, solve_nonlinear
from fem2d.srm import strength_reduction
from fem2d.porewater import (
    compute_pore_pressures, element_pore_pressures,
    effective_stress_correction, pore_pressure_force,
    cst_permeability_matrix, assemble_flow_system,
    apply_head_bcs, solve_seepage, seepage_velocity,
    cst_coupling_matrix, cst_compressibility_matrix,
    assemble_coupling, assemble_compressibility,
    solve_consolidation,
)

__all__ = [
    # High-level API
    'analyze_gravity', 'analyze_foundation', 'analyze_slope_srm',
    'analyze_excavation', 'analyze_seepage', 'analyze_consolidation',
    'analyze_staged', 'create_wall_elements',
    'ConstructionPhase', 'assign_element_groups',
    # Results
    'FEMResult', 'BeamForceResult', 'SeepageResult', 'ConsolidationResult',
    'PhaseResult', 'StagedConstructionResult',
    # Materials
    'elastic_D', 'mc_return_mapping', 'drucker_prager_params',
    'hs_return_mapping',
    # Elements
    'cst_stiffness', 'cst_B', 'cst_area', 'cst_body_force', 'cst_stress',
    'q4_stiffness', 'q4_body_force', 'q4_stress',
    'BeamElement', 'beam2d_stiffness', 'beam2d_internal_forces',
    # Mesh
    'generate_rect_mesh', 'generate_slope_mesh', 'generate_polygon_mesh',
    'detect_boundary_nodes', 'assign_layers_by_elevation',
    'assign_layers_by_polylines',
    'points_in_polygon', 'triangle_quality',
    # Assembly
    'assemble_stiffness', 'assemble_gravity', 'assemble_surface_load',
    'apply_bcs_penalty', 'solve_linear', 'recover_element_stresses',
    'nodal_stresses',
    'build_rotation_dof_map', 'beam_element_dofs',
    'assemble_beam_stiffness', 'assemble_beam_gravity',
    # Solver
    'solve_elastic', 'solve_nonlinear',
    # SRM
    'strength_reduction',
    # Pore water pressures
    'compute_pore_pressures', 'element_pore_pressures',
    'effective_stress_correction', 'pore_pressure_force',
    # Seepage
    'cst_permeability_matrix', 'assemble_flow_system',
    'apply_head_bcs', 'solve_seepage', 'seepage_velocity',
    # Consolidation
    'cst_coupling_matrix', 'cst_compressibility_matrix',
    'assemble_coupling', 'assemble_compressibility',
    'solve_consolidation',
]
