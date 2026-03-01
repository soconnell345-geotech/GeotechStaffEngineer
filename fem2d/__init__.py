"""
2D Plane-Strain Finite Element Module for Geotechnical Analysis

Custom FEM solver using numpy/scipy — no external FEM dependencies.

Capabilities:
- CST (3-node triangle) and Q4 (4-node quad) elements
- Linear elastic and Mohr-Coulomb elastoplastic materials
- Delaunay mesh generation with variable density
- Gravity loading, surface loads, prescribed displacements
- Strength Reduction Method (SRM) for slope stability FOS

High-level API:
    analyze_gravity()     — elastic gravity loading of a soil column
    analyze_foundation()  — strip load on elastic half-space
    analyze_slope_srm()   — slope stability FOS via SRM

References:
    Griffiths & Lane (1999) — Slope stability by finite elements
    Clausen et al. (2006) — MC return mapping in principal stress space
    de Souza Neto et al. (2008) — Computational Methods for Plasticity
"""

from fem2d.analysis import (
    analyze_gravity,
    analyze_foundation,
    analyze_slope_srm,
)
from fem2d.results import FEMResult
from fem2d.materials import elastic_D, mc_return_mapping, drucker_prager_params
from fem2d.elements import (
    cst_stiffness, cst_B, cst_area, cst_body_force, cst_stress,
    q4_stiffness, q4_body_force, q4_stress,
)
from fem2d.mesh import (
    generate_rect_mesh, generate_slope_mesh, generate_polygon_mesh,
    detect_boundary_nodes, assign_layers_by_elevation,
    points_in_polygon, triangle_quality,
)
from fem2d.assembly import (
    assemble_stiffness, assemble_gravity, assemble_surface_load,
    apply_bcs_penalty, solve_linear, recover_element_stresses,
    nodal_stresses,
)
from fem2d.solver import solve_elastic, solve_nonlinear
from fem2d.srm import strength_reduction

__all__ = [
    # High-level API
    'analyze_gravity', 'analyze_foundation', 'analyze_slope_srm',
    # Results
    'FEMResult',
    # Materials
    'elastic_D', 'mc_return_mapping', 'drucker_prager_params',
    # Elements
    'cst_stiffness', 'cst_B', 'cst_area', 'cst_body_force', 'cst_stress',
    'q4_stiffness', 'q4_body_force', 'q4_stress',
    # Mesh
    'generate_rect_mesh', 'generate_slope_mesh', 'generate_polygon_mesh',
    'detect_boundary_nodes', 'assign_layers_by_elevation',
    'points_in_polygon', 'triangle_quality',
    # Assembly
    'assemble_stiffness', 'assemble_gravity', 'assemble_surface_load',
    'apply_bcs_penalty', 'solve_linear', 'recover_element_stresses',
    'nodal_stresses',
    # Solver
    'solve_elastic', 'solve_nonlinear',
    # SRM
    'strength_reduction',
]
