# FEM2D — 2D Plane-Strain Finite Element Module

## Purpose

General-purpose 2D plane-strain geotechnical FEM solver. Computes
displacement, stress and strain fields for gravity-loaded soil masses.
Implements Strength Reduction Method (SRM) for slope stability FOS.
Supports structural beam elements for retaining walls and sheet piles.

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
- **2D Euler-Bernoulli beam**: 2-node beam for structural members (walls,
  sheet piles). DOFs: [u, v, θ] per node. Local-to-global rotation transform.
  Axial stiffness EA/L + flexural stiffness 12EI/L³.

### Constitutive Models

1. **Linear Elastic**: D-matrix above. Used for elastic analysis and
   as the trial predictor in elastoplastic analysis.

2. **Mohr-Coulomb Elastoplastic**: 2D in-plane Mohr circle return mapping.
   Yield criterion: f = q + p·sin(φ) - c·cos(φ) where p = (σ_xx+σ_yy)/2
   and q = Mohr circle radius. Non-associated flow (ψ=0: scale deviatoric,
   ψ>0: stress-space return). Apex return when deviatoric capacity exhausted.

3. **Hardening Soil (shear hardening)**: Schanz, Vermeer & Bonnier (1999).
   - Stress-dependent stiffness: E_50 = E_50_ref × ((c·cosφ + σ₃·sinφ) / (c·cosφ + p_ref·sinφ))^m
   - Hyperbolic deviatoric response: tangent E_t = E_50·(1 − R_f·q/q_f)²
   - Unload/reload at E_ur (typically 3× E_50_ref)
   - MC failure envelope as yield limit
   - Parameters: E50_ref, Eur_ref, m, p_ref, R_f, nu, c, phi, psi
   - State tracking: plastic shear strain, loading/unloading detection
   - Material dict: `{'model': 'hs', 'E50_ref': 25000, 'Eur_ref': 75000, ...}`

4. **Drucker-Prager**: Smooth cone approximation of MC. Plane-strain
   matching: α = tan(φ)/√(9+12·tan²(φ)), k = 3c/√(9+12·tan²(φ)).

### Mixed DOF System (Beams)

Soil nodes have 2 DOFs (u_x, u_y). Beam nodes get an additional rotation
DOF (θ_z) numbered starting at 2×n_nodes. This avoids inflating the system
size for soil-only problems.

DOF layout: [u0, v0, u1, v1, ..., u_{n-1}, v_{n-1}, θ_i, θ_j, ...]

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
- Bottom: fixed (u_x = u_y = 0, θ = 0 for beam nodes)
- Sides: rollers (u_x = 0, u_y free)
- Top surface: free

Applied via penalty method (penalty = 1e20) for sparse matrix compatibility.

### Newton-Raphson Solver

For nonlinear problems (MC/HS materials):
- Incremental gravity loading (5-20 steps)
- Full Newton-Raphson iteration (reform K_T each iteration)
- Convergence: ||R|| / ||F_ext|| < tol (default 1e-5)
- Max 100 iterations per load step
- HS state variables tracked per-element (tentative during NR, committed on step convergence)

### Strength Reduction Method (SRM)

1. Establish gravity equilibrium at SRF = 1.0
2. Bracket failure: increment SRF by 0.1 until non-convergence
3. Bisect [SRF_low, SRF_high] until tolerance (default 0.02)
4. Critical SRF = FOS
5. For HS elements: reduce c and φ, pass stiffness params unchanged

Validated against Griffiths & Lane (1999) and Bishop results from
the slope_stability module.

### Pore Water Pressures and Effective Stress

Flexible GWT input:
- **Constant elevation**: float, hydrostatic below GWT
- **Polyline profile**: (M, 2) array of (x, z_gwt), linearly interpolated
- **Per-node prescribed**: (n_nodes,) array for artesian conditions

Sign convention:
- Pore pressure u > 0 below GWT, u = 0 above
- Effective stress: σ' = σ_total - u × m  where m = [1, 1, 0]^T
- In tension-positive convention: subtracting positive u makes compression
  more negative = more confining in effective stress sense

Implementation:
- Pore pressures act as equivalent nodal forces: F_pp = Σ t·A·B^T·m·u_avg
- In NR loop: constitutive model sees effective stress (total - pore pressure)
- SRM: pore pressures unchanged during strength reduction (only c and φ reduced)

### Steady-State Seepage

Solves Laplace equation ∇·(k∇h) = 0 for hydraulic head h using CST elements.

CST flow element:
- Permeability matrix: H_e = k·t·A·G^T·G  where G = [∂N/∂x; ∂N/∂y] (2×3)
- Same shape function derivatives as mechanical B-matrix
- Darcy velocity: v = −k·∇h per element

Boundary conditions:
- Dirichlet: prescribed head h at nodes (penalty method)
- Neumann: prescribed flow rate at nodes (added to RHS)

Post-processing:
- Pore pressures from head: u = γ_w·(h − z), clipped ≥ 0
- Darcy velocity via CST gradient operator

### Coupled Biot Consolidation

Staggered (sequential) scheme for Biot's consolidation equations:
- **Equilibrium**: K·u + Q·p = F_ext
- **Continuity**: Q^T·du/dt + S·dp/dt + H·p = q

Coupling matrix Q: maps pore pressure DOFs to displacement DOFs
- Q_e = t·A·B^T·m·N_avg^T  (6×3 for CST, displacement DOFs × pressure DOFs)

Compressibility matrix S: fluid storage
- S_e = (t·A / (12·n_w)) × [[2,1,1],[1,2,1],[1,1,2]]  (consistent mass type)
- n_w = bulk modulus of water (2.2×10^6 kPa for incompressible)

Staggered algorithm per time step Δt:
1. Displacement: K·u_{n+1} = F_ext − Q·p_n
2. Pressure: (S/Δt + H)·p_{n+1} = S·p_n/Δt − Q^T·(u_{n+1} − u_n)/Δt
3. Apply drainage BCs to pressure system

Implicit backward Euler → unconditionally stable for any Δt.

### Staged Construction

Plaxis-style multi-phase analysis where each phase activates/deactivates
soil element groups, beam elements, and loads. Cumulative state carries
forward between phases.

**Phase data model** (`ConstructionPhase`):
- `name`: descriptive label
- `active_soil_groups`: list of group names to activate
- `active_beam_ids`: beam indices to include (None = no beams)
- `surface_loads`: list of (edges, qx, qy) tractions
- `gwt`: groundwater table (float, polyline, or None)
- `n_steps`: gravity load increments
- `reset_displacements`: zero u at phase start (keeps stress/strain)

**Element group assignment** (`assign_element_groups()`):
- Groups defined by bounding box regions (x_min, x_max, y_min, y_max)
- Elements assigned by centroid location
- Unassigned elements go to `'_default'` group

**Active element filtering**:
- Inactive elements contribute zero stiffness and zero gravity
- Assembly functions (`assemble_stiffness`, `assemble_gravity`, etc.) accept
  `active_elements` parameter — inactive elements are skipped in the loop
- NR iteration skips inactive elements for internal force and tangent
- Beam filtering via `active_beams` parameter on beam assembly functions

**Cumulative state carryover**:
- Each phase starts from previous phase's u, sigma, strain, elem_state
- `solve_nonlinear()` accepts `u_init`, `sigma_init`, `strain_init`, `state_init`
- HS hardening state preserved across phases
- `reset_displacements=True` zeros u but keeps stress/strain (useful for
  construction reference levels)

**Delta gravity loading**:
- Each phase applies full gravity for its active elements from scratch
  (incremental via n_steps), with initial state from previous phase
- Newly activated elements start contributing gravity in their phase
- Deactivated elements are simply skipped (no unloading of previous stress)

**Typical workflow**:
```python
groups = assign_element_groups(nodes, elements, {
    'soil': {'x_min': 0, 'x_max': 20, 'y_min': -10, 'y_max': 0},
    'excavation': {'x_min': 5, 'x_max': 15, 'y_min': -3, 'y_max': 0},
})

phases = [
    ConstructionPhase(name="Initial gravity", active_soil_groups=['soil', 'excavation']),
    ConstructionPhase(name="Excavate", active_soil_groups=['soil'],
                      active_beam_ids=[0, 1, 2]),
]

result = analyze_staged(nodes, elements, material_props, gamma, bc_nodes,
                        element_groups=groups, phases=phases)
```

## Sign Conventions

- **Compression positive** in soil mechanics convention
- **y-axis upward**: gravity is body force b = [0, −γ]
- **Counter-clockwise** node ordering for triangles
- **Engineering shear strain**: γ_xy = 2ε_xy
- **Beam moments**: positive = counter-clockwise (right-hand rule)

## Module Structure

```
fem2d/
  __init__.py          # Public API (58 exports)
  elements.py          # CST, Q4, and beam element routines
  materials.py         # Elastic D, Mohr-Coulomb, Hardening Soil, Drucker-Prager
  mesh.py              # Delaunay mesh generation, boundary detection
  assembly.py          # Global assembly, BCs, beam DOF mapping, sparse solver
  solver.py            # Linear solve, Newton-Raphson (MC/HS/beam/pore pressure)
  srm.py               # Strength Reduction Method (with pore pressure support)
  porewater.py         # Pore pressures, seepage, Biot consolidation
  analysis.py          # High-level API: gravity, foundation, slope SRM,
                       #   excavation, seepage, consolidation, staged construction
  results.py           # FEMResult, BeamForceResult, SeepageResult,
                       #   ConsolidationResult, PhaseResult,
                       #   StagedConstructionResult with summary()/to_dict()
  DESIGN.md
  tests/
    test_fem2d.py             # 89 core module tests
    test_cross_validation.py  # 15 cross-validation tests
    test_hs_beams.py          # 55 HS model + beam element tests
    test_groundwater.py       # ~50 groundwater/seepage/consolidation tests
    test_staged.py            # ~45 staged construction tests
```

## References

- Griffiths & Lane (1999), "Slope stability analysis by finite elements", Géotechnique 49(3)
- Schanz, Vermeer & Bonnier (1999), "The Hardening Soil model", Beyond 2000 in Computational Geotechnics
- Clausen, Damkilde & Andersen (2006), "An efficient return algorithm for non-associated plasticity"
- de Souza Neto, Peric & Owen (2008), "Computational Methods for Plasticity"
- Potts & Zdravkovic (1999), "Finite Element Analysis in Geotechnical Engineering"
- Taylor (1937), "Stability of earth slopes"
- Biot (1941), "General theory of three-dimensional consolidation", J Applied Physics
- Verruijt (1969), "Elastic storage of aquifers", Flow Through Porous Media
- Smith & Griffiths (2004), "Programming the Finite Element Method", 4th edition
