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

- **T6 (6-node quadratic triangle)**: isoparametric, 3-point interior
  Gauss rule (exact for straight-sided elastic stiffness; 6-point
  available via n_gp). DEFAULT for collapse/FOS work — validated within
  a few percent of published benchmarks (see VALIDATION.md). Generated
  from CST meshes by midside insertion (convert_to_t6).
- **CST (3-node triangle)**: Constant strain, closed-form
  K_e = t·A·B^T·D·B. Locks volumetrically in plastic collapse (Prandtl
  Nc overshoots by >75%) — use for elastic/seepage/Biot work only.
- **Q4 (4-node quad)**: Bilinear isoparametric, 2×2 Gauss quadrature.
  Available but not default.
- **2D Euler-Bernoulli beam**: 2-node beam for structural members (walls,
  sheet piles). DOFs: [u, v, θ] per node. Local-to-global rotation transform.
  Axial stiffness EA/L + flexural stiffness 12EI/L³.

### Constitutive Models

1. **Linear Elastic**: D-matrix above. Used for elastic analysis and
   as the trial predictor in elastoplastic analysis.

2. **Mohr-Coulomb Elastoplastic**: 3D principal-stress return mapping
   (de Souza Neto et al. 2008; Clausen et al. 2006): sorted principal
   stresses incl. σ_zz, main-plane return, Koiter two-plane edge
   returns (extension/compression), apex return; non-associated flow
   (ψ ≤ φ); fully vectorized over all Gauss points. The legacy
   in-plane `mc_return_mapping` is retained for API compatibility but
   is no longer used by the solver.

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

**Mesh-consistency study** (`srm_mesh_refinement_study`, `mesh_study.py`): a
convenience that runs `analyze_slope_srm` over a sequence of mesh densities and
returns a `MeshRefinementResult` (per-mesh FOS table + successive changes,
`converged` flag, and a Richardson-extrapolated mesh-independent FOS when the
finest three levels settle monotonically). It only drives the existing SRM at
several densities — no algorithm/default change. Used to demonstrate that the FE
FOS is mesh-converged and consistent with the limit-equilibrium answer (GL99
Ex1 stays banded near the published 1.4; the shared-geometry Bishop cross-check
converges monotonically toward the LE value with refinement). Convergence tables:
`VALIDATION.md` §7.

### Pore Water Pressures and Effective Stress

Flexible GWT input:
- **Constant elevation**: float, hydrostatic below GWT
- **Polyline profile**: (M, 2) array of (x, z_gwt), linearly interpolated
- **Per-node prescribed**: (n_nodes,) array for artesian conditions

Sign convention:
- Pore pressure u > 0 below GWT, u = 0 above
- Effective stress: σ' = σ_total + u × m  where m = [1, 1, 0]^T
- In tension-positive convention: adding positive u makes effective stress
  less compressive (less negative) = less confining pressure = lower strength
- Total from effective: σ_total = σ' − u × m

Implementation:
- Elastic solver: pore pressure forces F_pp = Σ t·A·B^T·m·u_avg added to F_ext
  (buoyant loading, no effective stress conversion needed)
- Nonlinear solver: F_pp added to F_ext alongside gravity, both ramped together
  during incremental loading. Solver works entirely in effective stress —
  constitutive model (MC/HS) sees effective stress, internal forces use
  effective stress. Equilibrium: B^T·σ' = F_gravity + F_pp.
  No total↔effective conversion needed in the NR loop.
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

Staggered (sequential) scheme for Biot's consolidation equations (tension-positive):
- **Equilibrium**: K·u = F_ext + Q·p
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

The staggered split transports a **prescribed** pore field and dissipates it, but
it does not create the **load-induced undrained excess pore pressure** (an applied
total-stress increment splits between effective stress and pore pressure only if the
u–p equations are solved together). For that transient use the monolithic scheme.

#### Monolithic u–p (Taylor–Hood), `solve_consolidation(scheme="monolithic")`

The coupled block system is solved simultaneously with a θ-method in time:

```
| K      -Q  | | u_{n+1} |   | F_ext                                    |
|            | |         | = |                                          |
| Q^T   S+θΔtH| | p_{n+1} |   | Q^T u_n + S p_n − (1−θ)Δt H p_n + drainage |
```

At t = 0 (Δt = 0, no flow) the block solve gives the **instantaneous undrained
response** — the applied load is split into effective stress + excess pore pressure
p0 in one solve. In 1-D (confined column, drained top), the interior p0 matches the
analytical `α·M/(K+4G/3+α²·M)·Δσ`; the field then dissipates as the Terzaghi/Biot
transient with `c = k_mob/S`, `S = 1/M + α²/(K+4G/3)`.

- **LBB / Taylor–Hood.** Equal-order (linear u / linear p) interpolation violates the
  LBB (inf-sup) condition → spurious pressure overshoot at the drained boundary. The
  monolithic path therefore uses a **Taylor–Hood pairing: T6 (quadratic) displacement /
  T3 (linear) pressure**. The CST input mesh is converted to T6 internally
  (`convert_to_t6`); pressure DOFs live on the corner nodes only, so H and S reuse the
  CST corner-skeleton matrices while the coupling `Q_e = ∫ B_u^T m N_p dA` integrates the
  T6 strain operator against the linear corner-pressure shape
  (`assemble_coupling_taylor_hood`). No stabilization term or physics fudge is used.
- **Units for the transient.** The flow rate needs the **mobility** k_mob = k/γ_w
  (m²/(kPa·s)), passed as `k`, and the Biot modulus **M** (kPa) passed as `n_w` — not the
  hydraulic conductivity, which only fixes the drained end-state.
- **θ (`theta`, default 1.0).** θ=1 backward Euler is L-stable/robust; θ=0.5
  Crank–Nicolson is 2nd-order and matches the Terzaghi decay to <1% with a fine step
  schedule (θ must be in [0.5, 1]).
- **Default preserved.** `scheme="staggered"` is the default and the staggered code path
  is byte-for-byte unchanged.
- **`degree_of_consolidation`.** For the monolithic scheme this is the mean
  excess-pore-pressure dissipation `U = 1 − mean|p_final| / mean|p0|` (p0 = the
  instantaneous undrained t=0 field), so it rises from 0 at t=0 toward 1 as the
  excess dissipates. (The staggered scheme has no undrained predictor, so its
  settlement-ratio `U` is identically 1.0 and cannot report the transient — left
  unchanged as the default.)
- **Factorization reuse (monolithic).** The coupled A-block depends only on Δt
  (K/Q/S/H are constant), so a uniform-time-step schedule is LU-factorized once
  (`scipy.sparse.linalg.splu`) and reused across steps; the factorization is
  rebuilt only when Δt changes (`_same_dt`). `splu.solve` is the same SuperLU
  backend as `spsolve`, so results are numerically identical to a per-step
  rebuild (asserted in `test_groundwater.py::TestMonolithicRefactorization`).

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
  analysis.py          # High-level API: gravity, foundation, footing capacity,
                       #   slope SRM, excavation, seepage, consolidation, staged
  mesh_study.py        # SRM mesh-refinement (mesh-consistency) study
  local_fos.py         # pointwise local factor-of-safety (mobilized-strength) map
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
