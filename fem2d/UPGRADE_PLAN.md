# FEM2D Modernization Plan (branch: fem-modern)

Goal: bring fem2d to the standards of modern geotech FEM programs (PLAXIS 2D,
RS2, OptumG2) within numpy/scipy. Headline items: T6 quadratic elements,
proper 3D principal-stress Mohr-Coulomb return mapping, Griffiths-Lane-grade
SRM robustness, NR solver hardening, and a published-benchmark validation
suite (fem2d/VALIDATION.md).

**Successor instructions**: read PROGRESS section at the bottom first, then
`git log --oneline` on this branch. Each phase is independently shippable;
suite must be green at every commit. Tests:
`"C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\.venv\Scripts\python.exe" -m pytest fem2d -q`
(baseline ~254 tests; slow SRM tests are `-m slow`-marked).
NOTE for agent sessions: the Write/Edit tools may be pinned to another
worktree; if so, stage files in %TEMP%\femstage and `cp` them in via Bash.

---

## Phase 0 — Theory and benchmarks (DONE — this document)

### Griffiths & Lane (1999), Geotechnique 49(3) 387-403
[verified against the paper PDF from inside.mines.edu/~vgriffit/slope64/]

Method: plane strain, 8-node quads, reduced 4-pt Gauss integration,
elastic-perfectly-plastic Mohr-Coulomb, **viscoplastic stress redistribution**
(Perzyna 1966), gravity "turn-on" in a single increment, **psi = 0 throughout**
(non-associated, zero volume change at yield). SRF definition:
c_f = c'/F, phi_f = arctan(tan phi'/F). **Failure = non-convergence within an
iteration ceiling of 1000**, accompanied by a dramatic increase in nodal
displacements. Results presented as FOS vs dimensionless displacement
**E' * delta_max / (gamma * H^2)**. Key quotes: FOS insensitive to gravity
increment size for elastic-perfectly-plastic MC (their Fig. 3); elastic
E', nu' have little influence on FOS (nominal E'=1e5 kPa, nu'=0.3).

Published example values (validation targets):

| Case | Geometry | Properties | Published FOS |
|---|---|---|---|
| Ex 1: homogeneous slope, D=1 | 2:1 (26.57 deg), height H; mesh: 1.2H crest margin, 2H slope run | phi'=20 deg, c'/(gamma H)=0.05, psi=0 | **FE 1.4; Bishop & Morgenstern 1.380** |
| Ex 1 curve (Table 2) | same | E' delta_max/(gamma H^2) vs SRF | 0.8:0.379, 1.0:0.381, 1.2:0.422, 1.3:0.453, 1.35:0.544, 1.4:1.476(no conv) |
| Ex 2: foundation layer D=1.5 | same slope + H/2 foundation layer | same | **FE 1.4** (unchanged; toe mechanism; B&M deep circle 1.752 is the wrong mechanism) |
| Ex 3: undrained, thin weak layer, D=2 | 2:1 slope, cu1/(gamma H)=0.25, weak layer cu2 | phi_u=0 | homogeneous (cu2/cu1=1): **1.47 (Taylor)**; mechanism switch at cu2/cu1=0.6; cu2/cu1=0.2: FOS ~0.6 (weak-layer wedge) |
| Ex 4: undrained, weak foundation D=2 | 2:1 slope H, foundation 2H deep total, cu1/(gamma H)=0.25 | phi_u=0, vary cu2/cu1 | cu2/cu1=1: **1.47 (Taylor)**; cu2/cu1>=1.5: flattens to **~2.1** (toe circle; Taylor cu2>>cu1: 2.10); transition at 1.5 |
| Ex 5: slow drawdown | Ex-1 slope, horizontal free surface depth L below crest | phi'=20, c'/(gamma H)=0.05 | L/H<=0 (submerged): **1.85** (Morgenstern 1963); minimum **1.3 @ L/H=0.7**; L/H=1: **1.4** |

Pore pressure treatment (their paper): u = gamma_w x (vertical depth below
free surface), subtracted from total normal stresses at Gauss points; gravity
from *total* unit weight; reservoir = normal surface stress. (fem2d already
does the equivalent in effective-stress form.)

### Strip footing benchmark (weightless c-phi soil)
- Prandtl: q_ult = c * N_c, **N_c = 2 + pi = 5.14** for phi=0 (exact).
  For phi>0: N_q = e^(pi tan phi) tan^2(45+phi/2), N_c = (N_q - 1) cot phi.
- CST overshoots N_c badly (locking, ~20-50% high on coarse meshes);
  T6/Q8 land within a few %. This contrast is a validation centerpiece.
- Drive with prescribed displacement (rigid footing) — more stable near
  collapse than load control; collapse load = sum of footing nodal reactions.

### Mohr-Coulomb return mapping in principal stress space
(de Souza Neto, Peric & Owen 2008 ch. 8; Clausen, Damkilde & Andersen 2006/07)

Tension-positive, principal stresses sorted s1 >= s2 >= s3:

    f = (s1 - s3) + (s1 + s3) sin(phi) - 2 c cos(phi)
    g = (s1 - s3) + (s1 + s3) sin(psi)        (non-associated potential)

Elastic principal-space matrix (isotropic): D_p = lam * ones(3,3) + 2G * I.
Return regions (all vectorized over Gauss points):
1. **Main plane**: dlam = f_tr / (n . D_p m), s = s_tr - dlam D_p m with
   n = [1+sin phi, 0, -(1-sin phi)], m = [1+sin psi, 0, -(1-sin psi)].
   Valid if returned ordering s1 >= s2 >= s3 holds.
2. **Edge returns** (Koiter, two active planes): if s1' < s2' -> extension
   edge (planes f(s1,s3) and f(s2,s3)); if s2' < s3' -> compression edge
   (planes f(s1,s3) and f(s1,s2)). Solve the 2x2 system for (dlam1, dlam2);
   valid if both >= 0 and returned ordering holds.
3. **Apex**: s1=s2=s3 = c cot(phi) = c cos(phi)/sin(phi) (tension-positive).
   Used when edge returns invalid. phi=0 (Tresca prism) has no apex; the
   main-plane/edge returns always succeed there.
Reconstruction: in plane strain the principal directions are the two in-plane
eigendirections (angle theta from sxx,syy,txy) plus the z axis; track the
sort permutation, return, unsort, rebuild [sxx,syy,szz,txy].

Tangent: continuum elastoplastic tangent in principal space
D_ep = D_p - (D_p m)(n^T D_p)/(n^T D_p m) (single plane; analogous 2-plane
formula on edges; ~0 at apex), rotated to xy assuming fixed principal
directions (documented simplification — drops the Clausen "T-matrix" spin
terms; convergence near-linear instead of quadratic, acceptable since SRM
uses the constant-stiffness method anyway). 4-component (xx,yy,zz,xy) stress
is stored per Gauss point; plane-strain tangent for assembly is the
[0,1,3]x[0,1,3] submatrix (eps_zz = 0 identically).

**Why this matters / pre-existing defects found in fem2d**:
- current `mc_return_mapping` works on the in-plane Mohr circle only and the
  solver stores 3-component stress; **sigma_zz is NOT tracked at all**
  (DESIGN.md overstates this). In plane strain sigma_zz can become the major
  or minor principal stress, in which case the in-plane return is wrong
  (yield missed entirely or returned to the wrong surface).
- the current return always hands back the *elastic* D as "tangent", so the
  "full NR" solver is actually an initial-stiffness method that reforms an
  unchanged matrix every iteration (slow AND wasteful: lil rebuild + spsolve
  per iteration).
- assembly uses Python triple loops appending scalars (slow for big meshes).

### Smith & Griffiths, Programming the Finite Element Method
- Their slope programs (6.x): constant-stiffness (initial stiffness /
  viscoplastic) iteration — factorize elastic K **once**, iterate body-load
  redistribution; convergence on relative change in displacements.
- For SRM this is ideal: K_elastic is independent of SRF, so ONE
  factorization (scipy splu) serves the whole SRM bracket+bisection.
- T6 with 3-pt interior Gauss rule is their standard quadratic triangle.
  We default T6 + 3-pt (exact for straight-edge T6 elastic stiffness),
  6-pt available for plasticity accuracy studies.

### PLAXIS practice (Material Models / Reference manuals)
- 15-node and 6-node triangles; MC return in principal stress space with
  apex/corner treatment (same Koiter scheme as above).
- Tolerated error 0.01 (1%) on global residual; arc-length for collapse-load
  control. Safety analysis (phi/c reduction) reduces tan phi and c
  simultaneously (Msf); when phi_red < psi, psi_red = phi_red. We adopt
  **psi_red = min(psi, phi_red)** and document.
- Undrained strength analysis: cu reduced directly (phi=0).

### Quadrature rules for triangles (area coordinates, weights sum to 1)
- 1-pt: centroid, w=1 (exact deg 1) — CST.
- 3-pt interior: (2/3,1/6,1/6) cyclic, w=1/3 each (exact deg 2) — T6 default.
- 6-pt (exact deg 4): Dunavant — a=0.445948490915965, b=0.091576213509771,
  w_a=0.223381589678011, w_b=0.109951743655322 (cyclic x3 each).

### T6 shape functions (area coords L1,L2,L3; corners 0-2, midsides 3-5 with
node 3 between corners 0-1, 4 between 1-2, 5 between 2-0):
N_i = L_i(2L_i - 1) (corners), N_3 = 4 L1 L2, N_4 = 4 L2 L3, N_5 = 4 L3 L1.
dL/dx constant for straight sides: L_i = (a_i + b_i x + c_i y)/(2A).
Consistent gravity for straight-sided T6: corners get 0, midsides get A/3
each (classic result — falls out of N integration automatically).
Quadratic edge (3-node) traction: weights (1/6, 2/3, 1/6) x total edge load.

---

## Architecture decisions

1. **Per-Gauss-point state, 4-component stress** `(n_elem, n_gp, 4)`
   [sxx, syy, szz, txy], for ALL element types (CST n_gp=1). The solver's
   public return remains `(n_elem, 3)` in-plane element averages for
   backward compat; per-GP arrays exposed additionally.
2. **Vectorized solver core**: precompute B (n_elem, n_gp, 3, 2*nen) and
   w*detJ once; strain/stress/return-mapping/internal-force/K-assembly all
   batched numpy (einsum + batched COO with precomputed index arrays).
   The constitutive return is vectorized over all Gauss points at once.
   Python-loop beam assembly stays (beam counts are tiny).
3. **Two solution methods**: `method='tangent'` (NR with continuum tangent,
   reform every `reform_interval` iterations, divergence cutback) and
   `method='elastic'` (constant-stiffness, splu factorized once, high
   iteration ceiling — Griffiths-Lane style). SRM default: 'elastic'.
   Dual convergence: residual ratio AND du ratio.
4. **element_type='t6' default** for new high-level analyses ('cst', 'q4'
   still available). T6 generated by midside insertion on the CST mesh
   (straight edges). Beams couple at corner nodes only (documented).
   Seepage/Biot stay CST (documented).
5. **MC return**: 3D principal-stress return replaces in-plane in the solver
   path; old `mc_return_mapping` kept for API compat. DP kept. HS keeps its
   hyperbolic tangent stiffness; failure handled by the new MC return.
6. **SRM**: failure = non-convergence AND dimensionless-displacement blowup
   check on E_max*delta_max/(gamma H^2) curve; bracket 0.1 steps from below,
   bisect to tol (default 0.01); srf_history gains dimensionless displacement
   + iteration counts; `srm_field='c_phi'` covers HS (reduce c, phi;
   stiffness unchanged); psi_red = min(psi, phi_red).

## Phases

- **Phase 1 — T6 foundations**: quadrature + T6 element fns (elements.py),
  `convert_to_t6` mesh conversion + BC detection for midside nodes (mesh.py),
  assembly support (stiffness/gravity/surface loads/stress recovery),
  elastic pipeline + tests (patch test, bending vs CST, consistent loads).
  SHIP: T6 usable in solve_elastic.
- **Phase 2 — MC 3D return + vectorized GP solver core**: vectorized
  principal MC return + unit tests (incl. a sigma_zz-governed case the old
  model gets wrong); per-GP 4-comp solver core behind solve_nonlinear for
  cst/t6/q4; methods 'tangent'/'elastic'; API stable. Update pinned values
  w/ justification. SHIP: suite green.
- **Phase 3 — SRM robustness**: blowup criterion + adaptive stepping +
  factorization reuse + srf curve reporting + HS coverage + psi policy;
  element_type plumbed through analyze_slope_srm (T6 default).
- **Phase 4 — NR hardening**: reform interval, dual convergence (in ph2
  core), step cutback, optional line search; API stable.
- **Phase 5 — Validation suite**: fem2d/VALIDATION.md + tests/test_validation.py:
  GL99 Ex1 (CST vs T6 vs published 1.4/1.380), Ex3 thin weak layer, Ex4 weak
  foundation (ratio 1.0/2.0), Prandtl Nc=5.14 footing (T6 within few %, CST
  overshoot shown), elastic cross-check vs settlement closed form, Bishop
  cross-check vs slope_stability on shared geometry. Tabulate everything.
- **Phase 6 — Performance**: profile, timings for GL99 Ex1 SRM @ T6
  (target: tens of seconds).
- **Phase 7 — Adapter + docs**: funhouse_agent/adapters/fem2d_adapter.py
  (element_type, srm options, new outputs, allowed_values), DESIGN.md
  update, module_work/slope-fem.md notes.

Descope order if time runs out: 7 minimal; 6 only if profiling shows pain;
Ex5 drawdown validation optional; T6 seepage not planned.

---

## PROGRESS

- [x] Phase 0: theory + benchmarks gathered (GL99 PDF verified), plan written.
- [x] Phase 1: T6 elements + mesh conversion + elastic pipeline + tests (76dc5d3).
- [x] Phase 2: 3D principal MC return (vectorized) + per-GP solver core (36048e4).
- [x] Phase 3: SRM robustness: GL99 failure detection (non-convergence +
      dimensionless-displacement blowup), shared factorization across all SRF
      trials, bracket+bisect to tol 0.01, srf_history/srf_curve/fos_basis
      outputs, srm_field c/phi/c_phi, psi_red=min(psi,phi_red), HS coverage,
      element_type through analyze_slope_srm (T6 default), h_ref for
      GL99-comparable curves, opt-in stall_window (default OFF — it can
      misclassify slow convergence as failure). 19 tests.
- [x] Phase 4: NR hardening — mechanisms landed in the ph2 core
      (reform_interval/modified NR, dual residual+du convergence with
      residual cap, divergence cutback, optional backtracking line
      search); 10 dedicated tests in tests/test_nr_hardening.py verify
      tangent-vs-elastic equilibrium agreement (CST+T6), tangent beats
      constant-stiffness on iterations, cutback grace, dual-criterion
      gating, API stability.
- [x] Phase 5: validation suite + VALIDATION.md — GL99 Ex1 (T6 1.341/
      1.366 vs published 1.4; curve tracks Table 2; Ex2 D-invariance
      reproduced at 48x24), Ex4 undrained (published 1.47 bracketed by
      nu=0.30/0.49 runs — T6 incompressibility limits documented),
      Prandtl footing (T6 Nc 5.10-5.25 vs 5.14; CST locks >9 — the
      centerpiece contrast), elastic closed form (T6 exact), Bishop
      cross-check (+21% @ 40x20, +11% @ 56x24). FIXED while validating:
      analyze_slope_srm x_extend default (was 2x width) starved the
      slope-face mesh and overpredicted FOS up to +91%; now 0.5H.
      9 tests in tests/test_validation.py (slow-marked).
- [x] Phase 6: performance — numbers captured in VALIDATION.md sec 6:
      GL99 Ex1 T6 32x12 full SRM 11 s; one factorization per SRM run;
      suite ~13 min incl. slow benchmarks. No further profiling needed.
- [x] Phase 7: adapter + docs — fem2d_slope_srm adapter exposes
      element_type/srm_field/blowup_factor/srf_range/n_gp with
      allowed_values and clear ValueErrors; returns fos_basis,
      n_srf_trials, srf_curve; defaults aligned (max_iter 1000,
      n_load_steps 2). DESIGN.md element/MC sections updated (T6
      default for collapse, 3D principal MC return);
      module_work/slope-fem.md progress note.

Future work (designed, descoped):
- B-bar / selective reduced integration or 15-node triangles for the
  incompressible (undrained nu->0.5) limit; would close the Ex4 bracket.
- Structured (transfinite) slope mesher to remove toe-run sliver
  triangles at D ~ 1 and restore CST convergence there.
- GL99 Ex3 (thin weak layer) and Ex5 (slow drawdown) validation rows:
  geometry support exists (layer_polylines / gwt); add rows when needed.
- Q8 elements if exact GL99 element parity is ever required.

Phase 3 evidence (kept for the validation phase):
- GL99 Ex1 (D=1 geometry, depth=0.5 m base sliver): T6 FOS 1.344 (32x12) /
  1.369 (48x16) / 1.456 (64x20) vs published FE 1.4, B&M 1.380. tol=0.02.
- Prandtl footing (weightless, phi=0): T6 3-pt Nc~5.10, 6-pt ~5.25 vs exact
  5.14; CST locks completely (carried Nc>9 without collapse). 40x20 mesh.
- D=1.5 foundation domain raises apparent FOS to ~1.87 (GL99 Ex2 says it
  should stay 1.4): suspected mesh-quality artifact of the column-Delaunay
  slope mesher near the toe + coarse vertical resolution; investigate in
  Phase 5 before tabulating.
- CST on thin-base (depth 0.5 m) meshes fails to converge even at SRF 0.5
  (sliver triangles, aspect ~30:1, from fixed ny over thin columns) —
  PRE-EXISTING mesher weakness, documented.
- delta_max is dominated by gravity settlement of the foundation when
  D>1, so dimensionless-displacement curves are only comparable to GL99
  with their D (their delta includes settlement of a D=1 mesh).

NEXT ACTION: Phase 5 — fem2d/VALIDATION.md + tests/test_validation.py
per the evidence above (Ex1 D=1 numbers final; Ex4 undrained shows nu
sensitivity: nu=0.49 locks high, nu=0.3 runs low — report both, designed
path = higher-order/B-bar elements; Bishop cross-check needs x_extend=0).
