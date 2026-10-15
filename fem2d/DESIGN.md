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

### The excavation wall line (GM-70, fixed 2026-10-14)

`create_wall_elements` builds the wall from ONE node column: the column
nearest `x_wall` within `tol`, from the node nearest `y_top` to the node
nearest `y_bottom`. Until 2026-10-14 it took every node within `tol`; on a
T6 mesh the midside nodes of horizontal edges sit at x_wall ± dx/2, inside
the 0.6 dx tolerance `analyze_excavation` passes, so the wall hopped
between three node columns (horizontal beam segments across the excavation
face), and where x = 0 was not a mesh column it straddled two. The y range
used the same horizontal `tol`, so a T6 wall could run up to tol past its
tip. `analyze_excavation` now also moves the node column nearest x = 0 onto
x = 0 and the interior node row nearest the tip onto y = -wall_depth
(before the T6 midside nodes are made), so the wall is where and as long as
asked. The staged path takes its `beam_elements` by node id from the
caller and never searched for them; it is unchanged.

What moved (the test cases that run `analyze_excavation`; M, V in
kN-m/m and kN/m, head deflection u_x in mm, + toward the excavation):

| case | wall nodes x (m) | M_max | V_max | head u_x | strut |
|---|---|---|---|---|---|
| clay, 10 x 6 mesh, no strut | -2.5/0/2.5 -> 0 | 77.8 -> 19.5 | 59.1 -> 11.7 | -7.9 -> -6.8 | |
| same, strut at 1.5 m | -2.5/0/2.5 -> 0 | 80.3 -> 28.3 | 62.2 -> 17.0 | -5.9 -> -5.0 | -67.0 -> -36.6 |
| clay, width 6, wall 8, 10 x 6 | -0.8/1.1 -> 0 | 6.6 -> 2.3 | 6.4 -> 1.7 | -2.7 -> -3.0 | |
| sand, width 8, wall 10, 10 x 6 | -0.8/1.6 -> 0 | 3.5 -> 3.7 | 2.7 -> 2.3 | -1.1 -> -0.4 | |
| clay with water at -1 m, 12 x 8 | -0.58/0.96 -> 0 | 51.8 -> 14.5 | 63.4 -> 9.6 | -3.5 -> -2.7 | |
| two layers on a polyline, 8 x 8 | -1.0/1.5 -> 0 | 58.3 -> 13.5 | 50.2 -> 11.7 | -1.9 -> -1.0 | |
| clay, defaults 30 x 15, T6 | -0.83/0/0.83 -> 0 | 91.6 -> 17.2 | 158.0 -> 28.0 | -8.5 -> -6.1 | |
| clay, defaults 30 x 15, T3 | 0 (wall 10.67 -> 10.0 m) | 21.5 -> 19.5 | 16.2 -> 18.9 | -5.5 -> -5.9 | |

T3 against T6 after the fix (clay, wall 10 m, excavation 5 m): the head
deflection agrees to 3.3 / 1.7 / 1.4 % on 30 x 15 / 45 x 22 / 60 x 30
meshes and the largest moment above the excavation level to 19 / 10 /
0.8 %. The whole-wall maximum does not converge together: on T3 it sits in
a stress concentration at the wall tip (18.9 against 14.2 kN-m/m on
60 x 30) that refinement does not remove; quote the moment above the tip
zone, or use T6.

Found underneath (GM-71, fixed the same day, next section): the excavation
itself was not modelled. The numbers in the table above are those of the
old one-solve model and are superseded.

### Staged excavation (GM-71, 2026-10-14)

Until 2026-10-14 `analyze_excavation` switched OFF the self-weight of the
excavated elements (they kept their stiffness) in a single gravity solve:
no in-situ stress, no unloading. The wall head moved INTO the retained soil
(u_x < 0 in every test case) and the wall forces were not those of an
excavation. It now runs on `analyze_staged`:

1. **Model.** Half of a symmetric excavation in plane strain: the wall on
   x = 0, the retained soil to x = -2 wall_depth, the excavation from the
   wall to its centre line at x = width / 2 (rollers: a line of symmetry);
   base fixed. `width` is the FULL excavation width (before, everything
   right of the wall to x = width + 2 wall_depth was "excavated"). A
   one-sided cut is the limit of a wide excavation; a note says when the
   half-width is less than the Rankine passive zone in front of the wall.
   The mesh is structured, with a node column on the wall and a node row at
   the ground, every excavation level, every strut level (unless within a
   quarter row of another level), the wall tip and the base; elsewhere the
   spacing is about even.
2. **Stage 0, in-situ stresses (K0 procedure).** Every Gauss point gets
   sigma_v' = overburden of the layers above it (by the layer boundaries,
   horizontal or `layer_polylines`, plus any water ponded on the ground)
   less the pore pressure, and sigma_h' = sigma_z' = K0 sigma_v', no shear,
   no displacement (`analyze_staged(initial_stresses=...)`). K0 is the
   layer's own `K0`, else Jaky's 1 - sin phi' for a frictional layer, else
   nu / (1 - nu) for phi = 0 (the ratio one-dimensional elastic loading
   gives). The stage settles any out-of-balance (an element whose centroid
   puts it in another layer than the boundary does); on level layered
   ground it moves nothing (tested: < 1e-9 m). Displacements are reported
   from the end of this stage.
3. **Stages 1..n, one per lift.** A lift's elements are removed; the forces
   they carried are released onto the remaining mesh: a stage's
   out-of-balance load runs from the internal force of the state it starts
   from (now without the removed elements) to the stage's load, over
   `n_steps`. The wall is wished in place at the start of the first lift,
   stress-free; a strut is installed, stress-free, at the start of the
   first lift after the excavation has reached its level, so its force is
   stiffness x the movement after installation (default lifts: one to each
   strut level, then to the final depth; a stop is added at a strut level a
   user's lift would pass). Struts are linear springs (a strut, or an
   anchor's axial stiffness), no preload.
4. **Wall-soil contact** (`wall_contact`). `'smooth'` (default): the soil is
   cut along the wall above its tip (the elements right of the wall get
   their own copies of the column's nodes); in stage 0 stiff ties join the
   two faces, so the ground is whole; from the first lift the wall has its
   own nodes, tied to each face horizontally only and to the soil at its
   tip in both directions; a face tie on the excavation side goes with the
   soil above the excavation level. No wall friction (delta = 0): the
   Rankine active pressure the USS Steel Sheet Piling Design Manual and the
   Caltrans Trenching and Shoring Manual use on the retained side.
   `'bonded'`: the wall shares the soil's nodes (no slip; the only option
   before). A bonded wall hangs the retained soil: on the cantilever
   benchmark its axial force reached 1.5 times the horizontal thrust above
   the excavation level (more than any wall friction angle up to phi'
   allows) and the thrust fell to 0.18 gamma H^2/2 against Ka = 0.271
   (moment 66 against 155 kN-m/m smooth); bonded, the clay cases of the
   table below move the wall head back into the soil (-0.1 to -2.0 mm).
   Wall friction between 0 and phi' needs
   interface elements, which fem2d does not have.
5. **Water** (`excavation_water`, GM-73, 2026-10-15; section "Dewatered
   excavation" below). The table `gwt` is the in-situ table and is held
   outside the excavation. Inside, where the table is above the excavation
   level: `'dewatered'` (the default since 2026-10-15) lowers the water to
   the excavation level at each lift, with steady-state seepage under the
   wall's tip, and the wall carries the net water pressure; `'full'` keeps
   the excavation full of water to the table (the only model before): water
   stands against every free face (`porewater.standing_water_forces`,
   `analyze_staged(water_on_free_faces=True)`), the water pressures on the
   two faces of the wall balance, and a warning says so; `'level'` lowers
   it to `water_level_inside`. Before 2026-10-14 a free face below the table
   carried the pore pressure outward with nothing to balance it.
6. **Convergence.** The stages use the constant-stiffness iteration. The
   active wedge behind a smooth wall is at its limit state and the wall's
   bending controls how far it moves, so a step can take several hundred
   iterations: `max_iter` defaults to 1000 (was 100), and a step that
   reaches it with its displacements standing still (second-half movement
   < 0.1 of the first half) and a residual ratio below 1e-2 or still
   falling is accepted, the criterion the strength reduction uses (listed
   in `notes`). On the benchmark the 20x10 to 45x22 meshes converge this
   way; 60x30 does not (the residual stalls near 1e-2 with the active wedge
   still creeping, 3 mm per 1000 iterations); Newton-Raphson with the
   continuum tangent and line search did no better (step cutbacks). Use
   the default mesh or up to about 45x22, or raise `max_iter` / `n_steps`.
7. **Result** (an `FEMResult`, keys kept): `displacements` /
   `max_displacement_*` since the end of the in-situ stage (meaning
   changed: before, the movement of the one gravity solve);
   `elements` / `stresses` the soil left at the end; `beam_forces`,
   `max_beam_moment_kNm_per_m`, `max_beam_shear_kN_per_m` at the end of the
   last stage (`wall_envelope` holds the largest over all stages);
   `strut_forces` with `force_kN_per_m` = stiffness x movement since
   installation (meaning changed: before, since the start) plus
   `installed_in_stage`; new `excavation_stages` (wall head and largest
   wall u_x, M_max, V_max, strut forces per lift), `notes`, and `staged`
   (the StagedConstructionResult, not serialized). `get_tables()` adds the
   printed "Excavation stages" table; the calc package adds an "Excavation
   Stages" section. Sign: u_x > 0 toward the excavation; a strut force > 0
   holds the wall back.
8. **Elements.** T3 locks in plastic flow: on the benchmark a T3 mesh gave
   a quarter (30x15) to a half (60x30) of the T6 moment, with a thrust below
   the active value; the result warns. The clay cases, where little yields,
   still agree within 5 % (`test_wall_line.py`).

Benchmark: VALIDATION.md section 9 (USS Steel Sheet Piling Design Manual,
cantilevered wall in granular soil, pp. 86-90).

What moved (the cases of the GM-70 table, T6 unless noted; M, V in kN-m/m
and kN/m, head u_x in mm, + toward the excavation; "before" is the GM-70
"after" column, i.e. the one-solve model):

| case | head u_x before -> after | M_max | V_max | strut |
|---|---|---|---|---|
| clay, 10 x 6, no strut | -6.8 -> +6.3 | 19.5 -> 30.1 | 11.7 -> 17.5 | |
| same, strut at 1.5 m | -5.0 -> +0.8 | 28.3 -> 37.1 | 17.0 -> 26.9 | -36.6 -> +50.1 |
| clay, width 6, wall 8, 10 x 6 | -3.0 -> +2.8 | 2.3 -> 2.4 | 1.7 -> 2.4 | |
| sand, width 8, wall 10, 10 x 6 | -0.4 -> +9.8 | 3.7 -> 7.3 | 2.3 -> 5.7 | |
| clay with water at -1 m, 12 x 8 | -2.7 -> +2.4 | 14.5 -> 6.5 | 9.6 -> 6.9 | |
| two layers on a polyline, 8 x 8 | -1.0 -> +12.3 | 13.5 -> 27.8 | 11.7 -> 21.8 | |
| clay, defaults 30 x 15 | -6.1 -> +6.7 | 17.2 -> 27.6 | 28.0 -> 25.4 | |
| same, strut at 1.5 m (k 50,000) | -3.7 -> +0.8 | 20.9 -> 35.7 | 27.0 -> 33.7 | -37.6 -> +52.7 |
| clay, defaults 30 x 15, T3 | -5.9 -> +6.2 | 19.5 -> 26.3 | 18.9 -> 15.5 | |

(clay: c 25 kPa, phi 20, E 30 MPa, gamma 18; it stands 5 m unsupported, so
these walls carry little. The water case is now an excavation kept full of
water, so its wall load falls. The widths are now full excavation widths, so
the domains are narrower than before. With a strut the head barely moves:
the wall bows toward the excavation below the strut, largest u_x 5.0 mm on
the 30 x 15 mesh.)

Since GM-73 (2026-10-15) the water case is `excavation_water='full'` (the
default is now `'dewatered'`), and a water table given as one elevation gets
a node row whatever the water condition; with that row the case gives head
u_x +2.4 mm, M_max 7.0, V_max 5.7 (`'full'`) and +4.1 mm, 13.2, 11.2, net
water pressure 18.5 kPa at the excavation level (`'dewatered'`).

### Dewatered excavation (GM-73, 2026-10-15)

Until 2026-10-15 `analyze_excavation` kept an excavation below the water
table full of water to the table: the water pressures on the wall's two
faces balanced, so the wall carried no net water pressure. A dewatered
excavation, the usual construction case, carries it, with seepage under
the wall's tip, and loads the wall more; the answer was on the unsafe side
for that case. `excavation_water` now chooses:

| value | inside the excavation | basis |
|---|---|---|
| `'dewatered'` (default) | water lowered to the excavation level at each lift, steady seepage under the tip | UFC 3-220-20 (2025) "Groundwater Control", p. 97: excavations below the groundwater table require groundwater control; Caltrans T&S Manual (2025) Example 7-2 (Figure 7-29, pp. 7-33 to 7-34): the shoring loaded with the full hydrostatic water pressure on the retained side down to the excavation line; FHWA GEC 4 (1999) section 5.2.9, pp. 62-63 and EM 1110-2-2504 (1994) para 4-5b, p. 4-6: the pore pressures of seepage beneath the wall from a flow net (or line of creep, fragments, finite elements) |
| `'full'` | kept full to the table, no seepage (the model before) | an excavation dug in the wet, e.g. a cofferdam before its seal course is placed and it is dewatered (Caltrans T&S Manual 2025, section 10-4.01, p. 10-24) |
| `'level'` | lowered to `water_level_inside` (or to the excavation level while the excavation is above it); water stands on the floor and against the wall's face | partial dewatering, e.g. the Caltrans cofferdam sequence (p. 10-24: dewater to a level to place the first bracing) |

The default is `'dewatered'` because the references treat the dewatered
excavation as the design case; the result's notes say which condition was
modelled and why, and give the net water pressure beside the no-seepage
value and GEC 4's simplified one. Neglecting seepage altogether (the
unbalanced hydrostatic pressure over the whole embedment, EM 1110-2-2504
para 4-5a, GEC 4 Eq. 17) is the case of a wall cut off into an impervious
layer: give the layers hydraulic conductivities `k` (every layer or none;
only the ratios matter) with a small one below the tip.

**Pore pressures per lift.** For each lift `porewater.seepage_heads` solves
the steady-state Laplace equation on the active soil (T6 or T3 elements
with their own shape functions; `solve_seepage` takes CST only):

- the soil is cut along the wall above its tip (as for the smooth wall; a
  bonded wall is then tied to both faces in both directions), so the wall
  is impermeable and the two faces have their own pore pressures;
- total head = the table elevation at every retained-side node at or above
  the table (the table held at its level outside; nodes above it carry no
  pore pressure); head = the water level inside on the excavation's floor;
  the far side of the retained soil, the base of the model and the centre
  line are impervious. A node row is put at the table and at the water
  level inside;
- the water inside goes down with the lifts: at each lift it is at the
  excavation level ('dewatered') or at max(water_level_inside, the
  excavation level) ('level'), never above the table; a lift above the
  table has hydrostatic pore pressures from the table.

The heads go to `analyze_staged` as the phase's per-node `gwt`; the change of
pore pressure from the in-situ stage is part of each lift's out-of-balance
load (drained, steady state). Water stands on every free face where the
pore pressure is positive (the floor, under 'level'), except the faces that
line the wall (`analyze_staged(wall_face_nodes=...)`): there the pore
pressure pushes the face against the wall, so the wall carries it. Water
standing in the excavation above its floor ('level', 'full' through this
path) pushes the wall's own nodes (`ConstructionPhase.nodal_forces`,
`_free_water_on_wall`, the same edge rule). With the water inside held at
the table the path gives the 'full' answer to 1e-4 (tested, smooth and
bonded).

**Why the far side is impervious.** With the head held on the retained
ground only, the model's domain (retained soil and base two wall depths
from the wall) gives the flow net of a domain eight times larger within
1 % (tested); holding the head on the far side as well let water in
sideways and raised the net pressure 3 %.

**Results.** Each stage row adds the water inside (depth), the pore
pressure and total head at the wall's tip (datum the ground), the net water
pressure on the wall at the excavation level, the upward hydraulic gradient
in front of the wall (tip to excavation level, along the wall) and the
critical gradient (gamma - gamma_w)/gamma_w of the soil there; a warning
when the gradient reaches it (quick condition, heave and piping: Caltrans
T&S Manual 2025 section 10-4, p. 10-21; UFC 3-220-10 (2026) section
6-5.2.2, p. 312 asks for 1.5 to 2 against heave). `wall_water_pressures`
lists both faces and the net at the end. Tables: "Water on the wall by
stage" (printed) and "Water pressures on the wall" (CSV); the calc package
adds "Water on the Wall by Stage".

**Steady state.** Steady seepage is the drained, long-term state; in a soil
of low permeability the pore pressures during construction lag behind it
(the staged analysis is drained), as the notes say.

Benchmark: VALIDATION.md section 10 (GEC 4 section 5.2.9, a closed-form
seepage solution). Before and after, the default clay case of the table
above with the water table at the ground (30 x 15, T6; "before" = today's
`'full'`, which reproduces the GM-71 numbers exactly):

| case | head u_x (mm) | M_max (kN-m/m) | V_max (kN/m) | strut (kN/m) | tip: u (kPa), head (m) | net water at excavation level (kPa) |
|---|---|---|---|---|---|---|
| cantilever, full -> dewatered | 3.0 -> 11.1 | 12.5 -> 33.7 | 11.7 -> 31.6 | | 98.1, 0.00 -> 75.3, -2.32 | 0 -> 41.9 |
| strut at 1.5 m, full -> dewatered | 0.4 -> 1.9 | 16.2 -> 45.9 | 15.2 -> 47.9 | 23.7 -> 80.8 | 98.1, 0.00 -> 75.3, -2.32 | 0 -> 41.9 |

(width 10 m, so the flow up in front of the wall is confined by the centre
line 5 m away; with no seepage the net pressure would be 49.1 kPa.)

### Mesh Generation

Uses scipy.spatial.Delaunay with:
- Boundary point sampling at controlled spacing
- Interior point seeding (grid-based, with variable density)
- Point-in-polygon filtering (ray casting)
- Centroid-based exterior triangle removal
- Laplacian smoothing (3-5 iterations)
- Soil layer assignment via centroid elevation check

`generate_slope_mesh` (the slope mesher behind `analyze_slope_srm`) places
`ny + 1` nodes in each of `nx + 1` columns from the base to the ground
surface and triangulates them. Delaunay covers the convex hull of the
points, which for any slope includes the notch above the toe and the slope
face; triangles with their centroid outside the soil section are removed
(since 2026-10-14). Before, they were kept as soil: a fan of sliver
triangles (minimum angles 0.1-0.9 deg) bridging the toe ground to the
upper slope face, weighing on it and propping it — 130 m2 of extra soil on
a 1,800 m2 section, 60 m2 on the 242 m2 Griffiths & Lane Ex 1 section —
and every strength-reduction FOS on this mesher was too high (a 10 m 2:1
slope, limit equilibrium 1.887, gave 3.0 = never failed on the default
30x15 mesh). Each column still gets `ny` rows whatever its height, so a
thin base layer (D ~ 1) makes flat elements under the toe run.

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

1. Each trial applies gravity (2 increments) at the reduced strengths
   c/SRF, arctan(tan φ/SRF), from zero stress; the elastic stiffness is
   factorized once and shared by every trial (constant-stiffness iteration,
   Smith & Griffiths)
2. Bracket failure: verify the lower bound, then step the SRF by 0.1 from
   1.0 until a trial fails
3. Bisect [SRF_low, SRF_high] until tolerance (default 0.02)
4. Critical SRF = FOS
5. For HS elements: reduce c and φ, pass stiffness params unchanged

**Failure criterion (2026-10-14).** Griffiths & Lane (1999): failure is
non-convergence within the iteration ceiling "accompanied by a dramatic
increase in the nodal displacements". A trial fails when a load step
reaches `max_iter` (1000) without the residual ratio meeting `tol` (1e-5)
AND its displacements are still moving: the largest nodal displacement
change over the second half of the iterations is at least `settled_ratio`
(0.1) times the change over the first half (a mechanism slides at an about
constant rate per iteration, ratio near 1), or the residual ratio is above
1e-2 and no longer falling. A step that reaches the ceiling with its
displacements standing still is accepted (`srf_history[i]['settled']`);
one whose displacements plainly stand still is accepted before the ceiling
(checked every 50 iterations after half-way, with 10 times the recent rate
of movement as margin). Two things stall the
residual while the displacements stand still: Gauss points near the apex or
edges of the Mohr-Coulomb surface switching return region back and forth
(chatter, residual ~1e-4 with |du| ~1e-9 m), and slow convergence where a
large region yields — under level ground the gravity stress state with
K0 = ν/(1−ν) violates the reduced strength once sin φ_red < 1 − 2ν
(Zheng, Liu & Li 2005), so the whole far field yields and the
constant-stiffness iteration crawls. Before, every such trial counted as
failure: the FOS followed the convergence rate, which falls as the mesh is
refined (10 m 2:1 slope, c' 15, φ' 25°, LE 1.887: 1.80 on 40x20, 1.15 on
60x30; ACADS 1(a): 0.93 on 56x20, 0.80 on 72x24). With the mesher fix the
SRM FOS now varies by at most 1.7 % between the coarsest and finest mesh
tried on every benchmark once the slope is spanned by 4 or more elements
(2.7 % at 3 elements), and sits 0–5 % below the limit-equilibrium value
(VALIDATION.md §7).

**Warnings on the result** (`FEMResult.warnings`, plain language): the
slope spanned by fewer than 4 elements from toe to crest (the coarsest
mesh measured, 3 elements, was +2.7 % against a fine mesh); a soil layer
with no elements, or thinner than its elements (elements take the layer of
their centroid, so a thin weak seam is lost or becomes a jagged band); a
search range that did not bracket failure. `mesh_check=True` also runs a
mesh 1.5x finer each way, reports both (`result.mesh_check`) and warns when
they differ by more than 5 %. Poisson's ratio 0.5 is refused.

**Side boundaries over a deep undrained foundation (GM-63, 2026-10-15).**
The side boundaries are rollers, on which the soil slides without
resistance. Over undrained (phi = 0) soil Taylor's critical circle passes
below the toe, tangent to the firm base, and reaches about 1.5-1.8 times
the depth of that soil below the toe behind the crest and 1.2-1.4 times it
beyond the toe (FHWA NHI-06-088 Figs. 6-14, 6-15, after Taylor 1948). A
model narrower than that fails along the rollers and reads low; the result
warns when the level ground on either side is under 1.5 times that depth,
and asks for twice it. Duncan Ex 1 at D = 3: 0.645 with the default
x_extend (0.6 H added each side) against 0.668 with 4 H each side and
Taylor's 0.679 (VALIDATION.md §11, where limit equilibrium on the same
short profile reads 0.72 because no circle may pass its ends). Undrained
layers should carry nu = 0.49: at nu = 0.35 gravity yields the soil at
depth at every trial, and on the finest mesh tried (100x30) the 1000
iterations read a trial that settles at 4100 as failure (0.627 against
0.662).

**Volumetric locking (GM-06, 2026-10-14).** The T6 / 3-point FOS does not
lock as ν → 0.5: Griffiths & Lane Ex 4 (undrained, D = 2) gives 1.419 at
ν = 0.30, 0.49, 0.495, 0.499 and 0.4999 alike (published 1.47; limit
equilibrium 1.486). The +19–26 % once recorded at ν = 0.49 came from the
soil-filled notch. A B-bar (mean-dilatation, T6/P0) variant was built and
measured: it lowered collapse loads 7–12 % below the standard element and
the published values (Ex 4 1.33; Prandtl Nc 4.52 against 5.14), so it was
not adopted.

Validated against Griffiths & Lane (1999) and Bishop results from
the slope_stability module.

**Mesh-consistency study** (`srm_mesh_refinement_study`, `mesh_study.py`): a
convenience that runs `analyze_slope_srm` over a sequence of mesh densities and
returns a `MeshRefinementResult` (per-mesh FOS table + successive changes,
`converged` flag, and a Richardson-extrapolated mesh-independent FOS when the
finest three levels settle monotonically). It only drives the existing SRM at
several densities — no algorithm/default change. Used to demonstrate that the FE
FOS is mesh-converged and consistent with the limit-equilibrium answer (since
2026-10-14 GL99 Ex1 gives 1.331 on every mesh and the shared-geometry Bishop
cross-check 1.156-1.169 against LE 1.173; the "monotonic convergence toward
LE" seen before was the soil-filled notch thinning out). Convergence tables:
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

`seepage_heads` (GM-73, 2026-10-15) solves the same equation on any element
type (T3, T6, Q4, with the element's own shape functions: H_e = Σ_gp w k
G^T G), on the active part of a mesh, with prescribed heads eliminated
rather than penalised; unprescribed boundaries are impervious, so a cut in
the mesh is an impermeable wall. Tested against the closed-form solution
for a sheet pile in a layer on an impervious base (conformal mapping;
`fem2d/tests/test_excavation_dewatered.py`) and against `solve_seepage` on a
CST mesh. The dewatered excavation uses it (section "Dewatered
excavation").

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
- **`degree_of_consolidation`.** For the monolithic scheme this is the
  area-weighted excess-pore-pressure dissipation `U = 1 − avg p(t) / avg p0`
  (p0 = the instantaneous undrained t=0 field; weights = tributary areas), at
  every reported time, so it rises from 0 at t=0 toward 1 as the excess
  dissipates. (The staggered scheme has no undrained predictor: U = 1.)
- **Construction stages (2026-10-09).** `analyze_consolidation` has two stages,
  both reported in `ConsolidationResult.stages` with this stage's own and the
  cumulative surface settlement and movement extremes: stage 0 `initial` — the
  self-weight of the soil in place, drained, in equilibrium with the
  hydrostatic field (`K u0 = F_gravity + Q p_init`, plus the ponded-water
  traction γw·gwt on the surface when gwt is above it — without that load the
  hydrostatic pore pressure would lift the soil); stage 1 `load` — `load_q`
  plus the self-weight of any soil layer flagged `'fill': True` (top layers
  only, above the water table; monolithic: applied UNDRAINED with the load),
  then consolidation. Linear, so the load stage is solved on its own (the G6
  increment) and cumulative = initial + load. `reset_displacements_after`
  (None/`"initial"` default, or `"start"`) picks what `settlements`,
  `surface_settlement_m_by_time` and `max_settlement_m` report: the load stage
  alone (default, as G6) or cumulative. A `judgment` record accompanies the
  default when it changes the final settlement materially. All unit weights
  zero = a weightless analysis: no initial-stage load (and the staggered
  displacement step then sees only the excess pore pressure). The staggered
  scheme follows the same reference (before 2026-10-09 its settlement
  included self-weight).
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
- `n_steps`: load increments for the phase (honoured since 2026-10-14;
  see "Delta gravity loading")
- `reset_displacements`: report this phase's (and later phases') movement from
  the START of this phase — PLAXIS "reset displacements to zero". Reporting
  only; the solver state is untouched (see "Movements per stage")

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
- Each phase starts from previous phase's u and the per-GAUSS-POINT stress,
  strain and elem_state (`solve_nonlinear(return_gp=True)`; since 2026-10-09 —
  element averages lost the T6 variation and the out-of-plane stress of a
  plastic point, so a stage in which nothing changed still moved ~1.5 % of the
  gravity settlement)
- `solve_nonlinear()` accepts `u_init`, `sigma_init`, `strain_init`, `state_init`
- HS hardening state preserved across phases
- `u` is never altered between phases: the solver's strain is TOTAL (B·u), so
  zeroing u (what `reset_displacements` did before 2026-10-09) only made the
  solver re-converge to the same total displacement

**Delta gravity loading**:
- Each phase applies full gravity for its active elements from scratch,
  with initial state from previous phase. The phase's out-of-balance load
  is ramped over `n_steps`: from the internal force of the state it starts
  from (the load that state is in equilibrium with, on this phase's active
  elements and members) to the phase's full load, `F_start + λ (F_ext −
  F_start)`. Before 2026-10-14 `solve_nonlinear` applied any phase with a
  `u_init` in ONE step whatever `n_steps` said (the first phase too), so a
  plastic stage gave the one-step answer, or none: a strip footing loaded
  to 360 kPa on c' 20 kPa, φ' 25° soil does not converge in one step and
  settles 68 mm in ten (`test_staged.py::TestStageLoadSteps`). An elastic
  phase is unaffected (linear)
- Newly activated elements start contributing gravity in their phase, and are
  placed STRESS-FREE on the deformed mesh: zero stress, strain initialized to
  B·u at activation, fresh HS state (since 2026-10-09; before, a fill's strain
  counted the movement its shared nodes had made before it existed, so the
  fill stage's movement was wrong by K_fill·u_prev)
- Deactivated elements are skipped: the stresses they carried drop out of
  the starting internal force, so the stage's out-of-balance releases them
  onto the remaining mesh (an excavation); nodes left in no active element
  are fixed and excluded from the reported movements
- Beams and springs activated in a phase are placed stress-free too (since
  2026-10-14, GM-61): their force is K (u - u_placed), carried in the solve
  as `K_lin u - lin_offset`. Before, a phase with `active_beam_ids=None`
  reached the solver as "all beams" (the docstring said none), and a beam
  reported in a later phase counted every movement since the start (186
  kN/m axial and 9 kN-m/m in a stage that changed nothing). Rotation DOFs
  that no active beam touches are fixed.
- `springs` (struts, anchors, props to ground; or ties between two nodes
  with `node_b`) switched on per phase with `active_spring_ids`; each phase
  reports `spring_forces` (k x stretch since placement)
- `initial_stresses` (per Gauss point or per element) start the first
  phase from a given stress state with zero displacement: the K0
  procedure `analyze_excavation` uses
- `water_on_free_faces` (default True since 2026-10-14): in a phase with a
  groundwater table, water stands against every free face of the active
  mesh where the pore pressure is positive (`-p n` on the face); ponded
  water on level ground then changes no effective stress (tested). Before,
  such a face carried the pore pressure outward unbalanced. A phase's
  `gwt` may be per-node total heads (e.g. from `seepage_heads`)
- `wall_face_nodes` (GM-73): soil faces lining an impermeable wall get no
  standing water; their pore pressure pushes them against the wall, which
  carries it. `ConstructionPhase.nodal_forces` adds a load no element face
  carries (water in an excavation against the wall)
- `settled_ratio` / `settled_residual_cap` (default off): accept a step
  whose displacements stand still at the iteration ceiling, as the strength
  reduction does; listed per phase in `settled_steps`

**Movements per stage** (owner direction 2026-10-09: "construction phasing
will determine output — output the delta movement per stage"). Each
`PhaseResult` carries three movement bases, each a
`results.movement_summary` dict over the nodes of the stage's active elements
(`settlement_m` = most negative uy, signed; `heave_m` = most positive uy;
`horizontal_m` = ux of largest magnitude, signed +x; `total_m` = max |u|; and
`*_at_xy`), with the matching arrays:

| Basis | Key / array | Meaning |
|---|---|---|
| this stage | `delta_displacement` / `delta_displacements` | u(end of k) − u(end of k−1); a node placed in stage k moves from its placement |
| cumulative | `cumulative_displacement` / `displacements` | from the start of the analysis (also `max_displacement_*`, meaning UNCHANGED) |
| since reference | `displacement_since_reference` / `displacements_since_reference` | u(end of k) − u(end of the latest reset stage before k); stages up to the reset stage are measured from the start |

The reset stage(s): `analyze_staged(reset_displacements_after=...)` — a stage
index, a stage name, `"initial"` (= stage 0) or `"start"` (no reset) — plus
any phase with `reset_displacements=True` (resets at its start). **Default**
(neither given): reset after stage 0, which is taken as the initial gravity/K0
stage, so construction movements exclude the self-weight movement (common FE
practice). Phasing, not a global switch, decides whether self-weight counts:
soil placed during construction is a group activated in a later phase, and its
self-weight movement is in that phase's delta. When the DEFAULT was applied and
it changes the last stage's movement materially (> 5 % and > 0.1 mm against
cumulative), the result carries a `judgment` record (question, both options
with their numbers, which was used and why) for the agent to surface.
`StagedConstructionResult.displacement_reference` names the reset stages and
whether the user or the default chose them.

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
