# FEM2D — 2D Plane-Strain Finite Element Module

## Purpose

General-purpose 2D plane-strain geotechnical FEM solver. Computes
displacement, stress and strain fields for gravity-loaded soil masses.
Implements Strength Reduction Method (SRM) for slope stability FOS.

No new dependencies — uses only numpy and scipy (sparse solver, Delaunay).

## Theory

### Plane Strain Formulation

Out-of-plane strain ε_zz = 0; stress σ_zz ≠ 0.

Elasticity D-matrix (3×3 in-plane):

    D = E / ((1+ν)(1−2ν)) × [[1−ν, ν, 0], [ν, 1−ν, 0], [0, 0, (1−2ν)/2]]

For nonlinear (Mohr-Coulomb), the full 4-component stress vector
[σ_xx, σ_yy, σ_zz, τ_xy] is tracked so that principal stress ordering
accounts for σ_zz.

### Element Types

- **CST (3-node triangle)**: Constant strain, closed-form K_e = t·A·B^T·D·B.
  Simple and robust. Used as the primary element.
- **Q4 (4-node quad)**: Bilinear isoparametric, 2×2 Gauss quadrature.
  Available but not default (CST meshes are easier to generate).

### Constitutive Models

1. **Linear Elastic**: D-matrix above. Used for elastic analysis and
   as the trial predictor in elastoplastic analysis.

2. **Mohr-Coulomb Elastoplastic**: 2D in-plane Mohr circle return mapping.
   Yield criterion: f = q + p·sin(φ) - c·cos(φ) where p = (σ_xx+σ_yy)/2
   and q = Mohr circle radius. Non-associated flow (ψ=0: scale deviatoric,
   ψ>0: stress-space return). Apex return when deviatoric capacity exhausted.

3. **Drucker-Prager**: Smooth cone approximation of MC. Plane-strain
   matching: α = tan(φ)/√(9+12·tan²(φ)), k = 3c/√(9+12·tan²(φ)).

### Mesh Generation

Uses scipy.spatial.Delaunay with:
- Boundary point sampling at controlled spacing
- Interior point seeding (grid-based, with variable density)
- Point-in-polygon filtering (ray casting)
- Centroid-based exterior triangle removal
- Laplacian smoothing (3-5 iterations)
- Soil layer assignment via centroid elevation check

### Boundary Conditions

Standard geotechnical BCs:
- Bottom: fixed (u_x = u_y = 0)
- Sides: rollers (u_x = 0, u_y free)
- Top surface: free

Applied via penalty method (penalty = 1e20) for sparse matrix compatibility.

### Newton-Raphson Solver

For nonlinear problems (MC/DP materials):
- Incremental gravity loading (5-20 steps)
- Full Newton-Raphson iteration (reform K_T each iteration)
- Convergence: ||R|| / ||F_ext|| < tol (default 1e-5)
- Max 100 iterations per load step

### Strength Reduction Method (SRM)

1. Establish gravity equilibrium at SRF = 1.0
2. Bracket failure: increment SRF by 0.1 until non-convergence
3. Bisect [SRF_low, SRF_high] until tolerance (default 0.02)
4. Critical SRF = FOS

Validated against Griffiths & Lane (1999) and Bishop results from
the slope_stability module.

## Sign Conventions

- **Compression positive** in soil mechanics convention
- **y-axis upward**: gravity is body force b = [0, −γ]
- **Counter-clockwise** node ordering for triangles
- **Engineering shear strain**: γ_xy = 2ε_xy

## Module Structure

```
fem2d/
  __init__.py          # Public API
  elements.py          # CST and Q4 element routines
  materials.py         # Elastic D-matrix, Mohr-Coulomb return mapping
  mesh.py              # Delaunay mesh generation, boundary detection
  assembly.py          # Global assembly, BCs, sparse solver
  solver.py            # Linear solve, Newton-Raphson, load stepping
  srm.py               # Strength Reduction Method
  analysis.py          # High-level analyze_*() functions
  results.py           # Result dataclasses with summary()/to_dict()
  DESIGN.md
  tests/
    test_elements.py   # Patch test, element stiffness
    test_materials.py  # Elastic D, MC return mapping
    test_mesh.py       # Meshing, point-in-polygon
    test_solver.py     # Gravity column, strip load
    test_srm.py        # SRM vs Bishop benchmarks
    test_analysis.py   # High-level API tests
```

## References

- Griffiths & Lane (1999), "Slope stability analysis by finite elements", Géotechnique 49(3)
- Clausen, Damkilde & Andersen (2006), "An efficient return algorithm for non-associated plasticity"
- de Souza Neto, Peric & Owen (2008), "Computational Methods for Plasticity"
- Potts & Zdravkovic (1999), "Finite Element Analysis in Geotechnical Engineering"
- Taylor (1937), "Stability of earth slopes"
