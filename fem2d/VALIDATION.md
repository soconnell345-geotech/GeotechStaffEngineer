# fem2d Validation Suite (branch: fem-modern)

Published-benchmark validation of the modernized fem2d: T6 quadratic
elements, 3D principal-stress Mohr-Coulomb return mapping (with
sigma_zz), and Griffiths-Lane-grade Strength Reduction Method.

Method for all SRM rows: constant-stiffness (initial stiffness)
iteration — the elastic global stiffness is factorized once and shared
by every SRF trial; residual tolerance 1e-5, iteration ceiling 1000 per
load step, 2 gravity steps, psi = 0. Failure = non-convergence with the
displacements still moving (Griffiths & Lane 1999: "accompanied by a
dramatic increase in the nodal displacements"); a trial that reaches the
ceiling with its displacements standing still is stable (`settled_ratio`
0.1, since 2026-10-14); bisection to the stated tolerance.
Reproduction: `fem2d/tests/test_validation.py` (`-m slow`) runs the
starred (*) rows with documented assertion bands.

---

## 0. 2026-10-14: two defects in every strength-reduction FOS before this date (BACKLOG GM-05, GM-06)

Two defects, found while reproducing a live run (a 10 m 2:1 slope,
c' = 15 kPa, phi' = 25 deg, gamma = 19, 20 m below the toe, E 50 MPa,
nu 0.3: FOS 1.80 on 40x20 but 1.147 on 60x30, `warnings: []`; limit
equilibrium with slope_stability's search: Bishop **1.887**, Spencer
**1.883**):

1. **The slope mesher filled the notch above the slope with soil.**
   `generate_slope_mesh` Delaunay-triangulates columns of nodes and kept
   every triangle of the convex hull, so the region between the toe ground
   and the slope face (air) was meshed with a fan of slivers (minimum
   angles 0.1-0.9 deg) carrying weight, stiffness and strength — a buttress
   propping the face against the toe ground. Extra soil: 130 m2 on that
   slope's 1,800 m2 section; 60 m2 on Griffiths & Lane Ex 1's 242 m2 (25 %);
   50 m2 on the ACADS 1(a) and Bishop cross-check sections; 120 m2 on Ex 4.
   With the buttress the discrete model held equilibrium far above the real
   FOS: Ex 1 32x12 still converged at SRF 1.6 (published 1.4), and the slope
   above returned FOS 3.0 (never failed) on the default 30x15 mesh. Fixed:
   triangles whose centroid lies outside the section are dropped (element
   area now equals the section area to round-off, `test_fem2d.py`).
2. **A stalled residual was read as collapse.** In 1000 constant-stiffness
   iterations the residual can stall near 1e-4 while the displacements
   stand still (|du| ~ 1e-9 m per iteration): Gauss points near the apex or
   edges of the Mohr-Coulomb surface switching return region back and
   forth, and slow convergence where a large region yields — under level
   ground the gravity stress state (K0 = nu/(1-nu) = 0.43 at nu = 0.3)
   violates the reduced strength once sin(phi_red) < 1 - 2 nu (Zheng, Liu &
   Li 2005), so the whole far field yields. Such trials counted as failure,
   the stall comes sooner on finer meshes, and the FOS fell with refinement
   (the slope above: 1.80 → 1.15; ACADS 1(a): 0.934 → 0.80) while the
   buttress pushed coarse meshes up. Fixed: failure needs the displacements
   to keep moving — over the second half of the iterations the largest
   nodal displacement change must be at least 0.1 of the change over the
   first half (a mechanism slides at a near-constant rate per iteration,
   ratio ~1; a stalled residual leaves 0.00-0.02), or the residual must
   still be above 1e-2 and no longer falling. (Slow convergence can leave
   it above 1e-2 after 1000 iterations: Ex 4, nu 0.30, 56x24 at SRF 1.2 —
   1.3e-2 and falling, growth ratio 0.04 — converges given 3,200.)

| Case (reference) | Mesh | FOS before | FOS after | after vs reference |
|---|---|---|---|---|
| 10 m 2:1, c' 15, phi' 25 (LE Bishop 1.887 / Spencer 1.883) | 20x10 | — | 1.934 | +2.5 % |
| | 30x15 (adapter default) | **3.000** (never failed) | 1.909 | +1.2 % |
| | 40x20 | 1.803 | 1.897 | +0.5 % |
| | 60x30 | **1.147** | 1.884 | −0.2 % |
| | 80x40, 6-pt | 1.134 | — | |
| GL99 Ex 1 (published FE 1.4; LE 1.368, B&M 1.380) | 16x6 | — | 1.344 | −4.0 % |
| | 32x12 * | 1.341 | 1.331 | −4.9 % |
| | 48x16 | 1.366 | 1.331 | −4.9 % |
| | 64x20 | 1.456 | 1.331 | −4.9 % |
| ACADS 1(a) (published 1.00; LE 0.985) | 30x15 | — | 0.965 | −3.5 % |
| | 40x16 | — | 0.957 | −4.3 % |
| | 56x20 | 0.934 | 0.949 | −5.1 % |
| | 72x24 | **0.80** | 0.949 | −5.1 % |
| Bishop cross-check §6 (LE 1.173) | 20x10 | — | 1.169 | −0.3 % |
| | 40x20 * | 1.419 | 1.156 | −1.4 % |
| | 56x24 | 1.306 | 1.156 | −1.4 % |
| GL99 Ex 4, cu2/cu1 = 1 (Taylor 1.47; LE 1.486) nu 0.30 | 40x16 | 1.306 | 1.419 | −3.5 % |
| nu 0.30 | 56x24 | 1.244 | 1.406 | −4.4 % |
| nu 0.49 | 40x16 | **1.853** | 1.419 | −3.5 % |
| nu 0.49 | 56x24 | 1.756 | 1.406 | −4.4 % |

"vs reference" is against the published value where there is one (GL99,
ACADS, Taylor), else limit equilibrium. The FOS now varies by at most
1.7 % between the coarsest and finest mesh on each benchmark once the
slope is spanned by 4 or more elements (2.7 % at 3), and lies between
+2.5 % and −5.4 % of our limit-equilibrium search (below it on every
fine mesh, as a non-associated psi = 0 strength reduction should). `analyze_slope_srm` warns in plain
words when the slope is spanned by fewer than 4 elements, when a soil
layer has no elements or is thinner than its elements, and when the
search range does not bracket failure; `mesh_check=True` runs a mesh 1.5x
finer and warns when the two differ by more than 5 %.

## 1. Griffiths & Lane (1999) Example 1 — homogeneous 2:1 slope

H = 10 m, 2:1 slope (26.57 deg), crest margin 1.2H, slope run 2H, toe
run 1.2H; phi' = 20 deg, c'/(gamma H) = 0.05 (c' = 10 kPa, gamma = 20
kN/m3), psi = 0, E' = 1e5 kPa, nu' = 0.3, D = 1 (firm base just below
toe; modeled with a 0.5 m base layer). Published: **FE FOS = 1.4**
(GL99), **Bishop & Morgenstern 1.380**. Bisection tol 0.01.

Since 2026-10-14 (corrected mesher and failure criterion, §0); the
numbers before are in §0 and in git history.

| Element | Mesh | Elements | FOS | vs published 1.4 |
|---|---|---|---|---|
| T6 (3-pt) | 16x6, D=1 | 194 | **1.344** | -4.0% |
| T6 (3-pt) | 24x8, D=1 | 386 | **1.331** | -4.9% |
| T6 (3-pt) * | 32x12, D=1 | 770 | **1.331** | -4.9% |
| T6 (3-pt) | 48x16, D=1 | 1538 | **1.331** | -4.9% |
| T6 (3-pt) | 64x20, D=1 | 2562 | **1.331** | -4.9% |
| T6 (3-pt) | 32x12, D=1.5 | 770 | **1.356** | -3.1% — GL99 Ex2 (FOS unchanged when a foundation layer is added) reproduced; was 1.866 (+33 %, written off as an under-refined deep domain: it was the soil-filled notch) |
| T6 (3-pt) | 48x24, D=1.5 | 2306 | **1.344** | -4.0% |
| CST | 32x12, D=1 | 770 | 1.431 | +2.2% — constant-strain triangles stiffer than T6 (+7.5 %); before 2026-10-14 no result at all (the air slivers kept it from converging even at SRF 0.5) |
| CST | 12x6 small slope (tests) | — | — | locking: do not use CST for collapse (§3) |

SRF vs dimensionless displacement E' delta_max/(gamma H^2), T6 32x12
D=1 (*), against GL99 Table 2:

| SRF | GL99 published | fem2d T6 |
|---|---|---|
| 0.8-1.0 | 0.379-0.381 | 0.421 |
| 1.2 | 0.422 | 0.482 |
| 1.3 | 0.453 | 0.553 |
| 1.325 | — | 0.635 (last stable) |
| 1.35 | 0.544 | no convergence (flowing) |
| 1.4 | 1.476 (no convergence) | no convergence |

Curve shape (flat, then knee just below failure) reproduces the paper;
the ~10% level offset is the 0.5 m base-layer settlement plus T6-vs-Q8
discretization, and the knee comes about 0.05 earlier (FOS 1.331 against
1.4).

## 2. Griffiths & Lane (1999) Example 4 — undrained two-layer, D=2

2:1 slope, H = 10 m, phi_u = 0, slope soil cu1/(gamma H) = 0.25 (cu1 =
50 kPa); foundation to D = 2 (10 m below toe), strength cu2. Published:
cu2/cu1 = 1: **1.47** (Taylor); >= 1.5: plateau **~2.1** (toe circle).
Limit equilibrium (slope_stability, circles, cu2/cu1 = 1): 1.486.
T6 40x16 (1282 elem) unless noted, tol 0.02; 2026-10-14 values (§0),
the earlier ones in brackets.

| cu2/cu1 | Published | T6, nu = 0.49 | T6, nu = 0.30 |
|---|---|---|---|
| 0.6 | (weak foundation, deep wedge) | — | 0.914 [0.883] |
| 1.0 * | **1.47** | **1.419** (-3.5%) [1.853]; 1.406 @ 56x24 (-4.4%) [1.756] | **1.419** (-3.5%) [1.306]; 1.406 @ 56x24 (-4.4%) [1.244] |
| 1.5 | ~2.0-2.1 | — | 2.019 [1.819] |
| 2.0 | **~2.1** (plateau) | — [range-capped at 3.0] | 2.156 (+3%) [2.319] |

No volumetric locking: the FOS is the same at nu = 0.30, 0.49, 0.495,
0.499 and 0.4999 (1.419 at 40x16), as it should be for a perfectly
plastic collapse load. What was recorded here before — nu = 0.49 locking
high by 20-25 % and nu = 0.30 low by 10-15 % "via out-of-plane Tresca
yield", the published value bracketed between them — came from the
soil-filled notch above the slope (high) and from stalled iterations read
as collapse while the deep foundation yields under gravity at nu = 0.30
(low; K0 = 0.43 under level ground exceeds the Tresca limit at depth).
A B-bar (mean-dilatation T6/P0) variant was measured as the planned remedy
and gave 1.33 here (Prandtl Nc 4.52 against 5.14): not adopted
(UPGRADE_PLAN.md). (* the test asserts the two agree and lie within
1.38-1.52.)

## 3. Prandtl bearing capacity — the element-quality centerpiece

Smooth rigid strip footing (B = 2 m), weightless phi = 0 soil, c = 100
kPa, nu = 0.3, domain 10B x 5B, load-control ramp (15 kPa steps),
collapse = last converged level. Exact: q_ult = (2 + pi) c, **Nc =
5.14**.

| Element | Mesh | Nc | Error |
|---|---|---|---|
| CST * | 40x20 (1600 elem) | **>= 9.0, never collapsed** | **> +75% — volumetric locking** |
| T6, 3-pt * | 40x20 (1600 elem) | **5.10-5.25** | **within ~2%** |
| T6, 6-pt | 40x20 | 5.25-5.40 | +2 to +5% |
| T6, 3-pt | 60x30 (3600 elem) | 5.10-5.25 | within ~2% |

The classic CST pathology: constant-strain triangles cannot represent
isochoric plastic flow of the Prandtl mechanism and lock, carrying load
without bound. T6/3-pt resolves the exact mechanism within a few
percent. **Do not use CST for collapse loads or FOS** — it is retained
for elastic, seepage and Biot work.

**Convenience** (`analyze_footing_capacity`, v5.4 E11): a one-call wrapper of
this load-control-to-collapse workflow (build the half-space mesh from footing
width B → ramp the strip pressure → last converged level = q_ult), the FE
analogue of a bearing-capacity calculation (analogous to how
`analyze_slope_srm` wraps the slope workflow). It reports q_ult, the
back-figured Nc, the closed-form factors (`bearing_capacity_factors`, Prandtl–
Reissner + Vesic Nγ), and a bearing FOS vs a working pressure. Validated here:
T6 40x20 gives Nc = **5.12** (−0.4% vs 5.14); CST locks (never brackets
collapse, Nc > 8). Tests: `fem2d/tests/test_footing_capacity.py`.

## 4. Elastic closed-form checks

| Check | Closed form | fem2d | Error |
|---|---|---|---|
| 1D gravity compression w = gamma H^2/(2 M), T6 40x10 * | 0.022286 m | 0.022286 m | **0.00%** |
| same, CST 40x10 | 0.022286 m | 0.022488 m | +0.91% |
| K0 = nu/(1-nu) stress ratio | 0.4286 | within 10% | existing suite (test_cross_validation) |
| Boussinesq stress bulb under strip load | — | within band | existing suite (test_cross_validation) |

## 5. Biot consolidation — undrained transient (V-023, Itasca 1-D column)

Itasca FLAC verification problem: a 20 m saturated column, drained top /
impermeable base, instantaneously loaded with Δσ = 100 kPa. K = 500, G = 200,
M = 4000 MPa, α = 1, mobility k_mob = 1e-10 m²/(Pa·s). Solved with the
**monolithic u–p Taylor–Hood** scheme (`scheme="monolithic"`), 40 elements tall.

| Quantity | Reference | fem2d monolithic | Note |
|---|---|---|---|
| Undrained p0 (interior/base) | 0.839×10⁵ Pa (= α·M/(K+4G/3+α²M)·Δσ) | **83.92 kPa** | exact; matches the value **consistent with the stated M** |
| Undrained p0 (Itasca-reported) | 0.981×10⁵ Pa | — | needs an effective M ≈ 10× (near-incompressible fluid); **tuned to neither** — documented |
| Decay p(z,t)/p0 at base, t̂ = 0.05…1.0 | Terzaghi series (c = k_mob/S = 0.0643) | within **<1%** (θ=0.5) / **≤3.1%** (θ=1) | inside the FLAC <5% envelope |
| Drained end-state settlement | Δσ·H/(K+4G/3) = 2.609 mm | **2.596 mm** (−0.5%) | same value the staggered scheme already gets |

The equal-order (CST) monolithic solve shows the classic LBB pressure overshoot at
the drained boundary (max ≈ 173 vs correct 84); the Taylor–Hood T6/T3 pairing removes
it (residual t=0 boundary-layer max ≈ 119, dissipating immediately) with **no
stabilization term**. The staggered scheme cannot generate this load-induced
undrained pressure at all (it stays ≈ 0), which is why the monolithic option was
added. Tests: `validation_examples/test_published_v023_v024.py`
(`test_v023_monolithic_*`).

**Load increment, head units, per-time U (2026-10-09, live smoke G6 / CON-1).**
Four defects fixed in `solve_consolidation` / `analyze_consolidation`:
(1) the monolithic solve applied self-weight undrained with the load, so p0
carried the self-weight response (386 kPa under 100 kPa on the CON-1 column) —
the increment is now the load only, self-weight and the hydrostatic field
being the initial state (linear superposition); (2) drained boundaries took the
head (m) as a pressure (kPa) — now u = γw·(h − z), clipped ≥ 0, then excess
over the initial field; (3) the staggered flow acted on the TOTAL pressure with
no elevation term and with k (m/s) used as a mobility, so a hydrostatic field
"consolidated" away (≈ +50 % settlement on a 10 m column) — it now diffuses the
excess with H/γw; (4) U came back only at the last time, the first requested
time was taken as the loading instant, and the wrapper stepped straight on the
output times — times now run from loading (t = 0 always reported), the
monolithic wrapper integrates on a geometric sub-step schedule (~25/decade) and
reports U and settlement at every requested time. U is the tributary-area
weighted excess dissipation. Through `analyze_consolidation` with sparse output
times on this section's column (ny = 40, θ = 0.5): U(Tv) within **0.5 %** of
Terzaghi for Tv = 0.05–1.0, isochrones p(z,t)/p0 within **0.04 % of p0** (z ≥ 2 m),
p0 = **83.92 kPa** = M/(M_oed + M)·q; the V-023 numbers above are unchanged
(final U 0.9936 → 0.9941 from the area weighting only). CON-1 itself
(1e3, 1e7, 1e8 s): U = 0.005 / 0.985 / 1.000 (Terzaghi 0.014 / 0.985 / 1.000; the
1e3 s value needs elements thinner than √(c·t) ≈ 0.25 m at the drained
boundary). Tests: `fem2d/tests/test_consolidation_increment.py`.

## 5a. Movements per construction stage (2026-10-09)

Owner direction: "construction phasing will determine output (ie output the
delta movement per stage)" — G6 had fixed the monolithic settlement to exclude
self-weight; now every stage reports its own movement, the cumulative movement
and the movement since a reference stage (`reset_displacements_after`, default
after the initial stage, with a `judgment` record when that default changes the
numbers materially). Checks are against independent solves on the same mesh
(linear, so superposition is exact) and 1-D closed forms. Tests:
`fem2d/tests/test_stage_movements.py`,
`funhouse_agent/tests/test_fem2d_stage_movements_adapter.py`.

| Check | Reference | fem2d | Difference |
|---|---|---|---|
| Strip footing B = 2 m, 100 kPa, T6 20x10, E 20 MPa, ν 0.3, γ 18: gravity stage's own movement | gravity-only solve (33.43 mm max) | same field | 6e-16 m |
| same, footing stage's own movement | load-only solve: settlement **15.90 mm** | same field | 4e-16 m |
| same, cumulative after the footing stage | gravity + load: **49.33 mm** | same field | 5e-16 m |
| 2 m fill group activated on 6 m of soil after its gravity stage: fill stage's own movement | solve under the fill weight alone: **9.36 mm** | same field | 1e-16 m (was off by K_fill·u_prev: activation strain not reset) |
| Plastic T6 (c 1, φ 20°) gravity, then a stage that changes nothing | 0 | **0** | was 0.08 mm (1.5 % of the 5.5 mm gravity settlement) with element-average state |
| CON-1 column (G6), initial stage: buoyant self-weight with 20 m ponded water | γ′H²/(2 M_oed) = 2.137 mm | **2.137 mm** | 0.0 % |
| CON-1 load stage (reported by default) | q H / M_oed = 2.609 mm, U = 1.000 at 1e8 s | 2.609 mm, U 1.000 | unchanged from G6 |
| CON-1 cumulative | initial + load = 4.746 mm | 4.746 mm | — |
| 10 m staggered column, 50 kPa: load stage / initial stage | 37.14 / 30.42 mm | 37.14 / 30.52 mm | 0.0 % / +0.3 % (CST) |
| 2 m dry fill flagged `fill` on 18 m clay (WT at the clay top): initial / load-stage drained | γ′·18²/(2M) = 1.731 mm; (20q + 38γ)/M = 3.501 mm | 1.731 / 3.501 mm | 0.0 % |

Basis changes: the staggered column's `surface_settlement_m_by_time` /
`max_settlement_m` now follow the reference stage (default: load stage alone;
they used to include self-weight); a staged phase's `reset_displacements`
flag now works (reporting only — it used to zero u in the solver, which
re-converged to the same total, so it did nothing). `max_displacement_*` of a
staged phase stays cumulative. Fill flagged in the consolidation column must
lie above the water table (refused otherwise: water standing on the soil in
place before an underwater fill is not modelled). Observation, not changed:
with θ = 0.5 a ±5 kPa Crank–Nicolson ringing persists at the drained boundary
long after U → 1 (max |excess| 4.96 kPa at Tv = 16; θ = 1 gives 0), so
`max_excess_pore_pressure_kPa_by_time` carries it at late times.

**Load steps within a stage (2026-10-14, BACKLOG GM-08).** Until then every
stage — the first (gravity) stage too — was applied in ONE step whatever its
`n_steps`. A stage now ramps its out-of-balance load over `n_steps`, from the
load its starting state is in equilibrium with to the stage's full load.
Checks (`fem2d/tests/test_staged.py::TestStageLoadSteps`, T6 16x8, footing
B = 2 m on c' 20 kPa, φ' 25°, ψ 0, E 20 MPa, ν 0.3, γ 18):

| Check | Reference | fem2d |
|---|---|---|
| 360 kPa footing stage, `n_steps` = 10 | the same load as 10 stages of 36 kPa (incremental by construction) | footing settlement within 0.1 %, every node within 0.1 % of max |u| |
| same stage, `n_steps` = 1 (the old behaviour) | — | no equilibrium within 500 iterations (the stage reports non-convergence); 10 steps settle 68 mm |
| plastic gravity stage, `n_steps` = 4 | `solve_nonlinear` with 4 steps, no initial state | identical to 1e-9 |
| elastic stages, `n_steps` 1 vs 7 | linear | identical to 1e-8 |

The elastic checks in the table above are unchanged (linear stages give the
same answer in any number of steps).

## 6. Cross-check vs slope_stability Bishop (shared geometry)

Shared profile [(0,0),(10,0),(30,10),(50,10)] (2:1, H = 10 m), c' = 10
kPa, phi' = 15 deg, gamma = 18 kN/m3, dry, 5 m foundation.
slope_stability `search_critical_surface(method='bishop')` (read-only
import) vs `analyze_slope_srm` with x_extend = 0:

| Method | FOS (2026-10-14) | before (§0) |
|---|---|---|
| Bishop, critical-circle search | **1.173** | |
| Spencer, critical-circle search | 1.170 | |
| SRM T6 20x10 | 1.169 (−0.3%) | — |
| SRM T6 30x15 | 1.156 (−1.4%) | — |
| SRM T6 40x20 * | **1.156** (−1.4%) | 1.419 (+21%) |
| SRM T6 56x24 | 1.156 (−1.4%) | 1.306 (+11%, "converging toward LE") |
| SRM CST 40x20 | 1.206 (+2.8%) | 1.706 (+45%) |

The "convergence toward LE with refinement" recorded here before was the
soil-filled notch being thinned out by finer meshes. Found while building
this row (2026-06, before the §0 corrections): the old `analyze_slope_srm`
default `x_extend = 2 x domain width` starved the slope face of elements at
fixed nx (FOS 2.24 then; the default is now `max(0.5 H, depth, 5 m)` and
profiles with their own margins should pass `x_extend=0`). Orientation
(crest-left vs crest-right) was verified to make no difference (mirrored
geometry, identical FOS).

## 7. SRM mesh-refinement (mesh-consistency) study — B2e

`srm_mesh_refinement_study(surface_points, soil_layers, meshes, ...)` (in
`fem2d/mesh_study.py`, exported from `fem2d`) runs `analyze_slope_srm` over a
sequence of mesh densities and returns a `MeshRefinementResult` — a per-mesh
FOS table with successive relative changes, a `converged` flag (relative FOS
change between the two finest meshes below `conv_tol`, default 3%), and a
Richardson-extrapolated mesh-independent FOS when the three finest levels are
monotonic and settling (characteristic size `h ∝ 1/√elements`). It only drives
the existing SRM at several densities — no algorithm or default changes. This
makes the fem2d↔LE mesh-consistency check reproducible in one call.

Re-measured 2026-10-14 after the §0 corrections (`srf_tol` 0.02); the
values before are in brackets. "Across" is the number of elements spanning
the slope height (the measure behind the coarse-mesh warning, which fires
below 4).

**Griffiths & Lane (1999) Example 1** (published FE **1.4**; LE 1.368), T6,
D=1, `x_extend=0`:

| Mesh | Elements | Across | FOS | vs 1.4 |
|---|---|---|---|---|
| T6 16x6 | 194 | 4.8 | 1.344 | −4.0% |
| T6 24x8 | 386 | 6.9 | 1.331 | −4.9% |
| T6 32x12 | 770 | 9.9 | 1.331 [1.344] | −4.9% |
| T6 40x16 | 1282 | — | 1.331 [1.381] | −4.9% |
| T6 64x20 | 2562 | — | 1.331 [1.456] | −4.9% |

**Shared-geometry Bishop cross-check** (§6 geometry; LE Bishop **1.173**), T6,
5 m foundation, `x_extend=0`:

| Mesh | Elements | Across | FOS | vs LE 1.173 |
|---|---|---|---|---|
| T6 20x10 | 400 | 5.2 | 1.169 | −0.3% |
| T6 24x12 | 578 | — | 1.169 [1.531] | −0.3% |
| T6 32x16 | 1026 | — | 1.156 [1.456] | −1.4% |
| T6 40x20 | 1600 | 10.3 | 1.156 [1.419] | −1.4% |
| T6 56x24 | 2690 | — | 1.156 [1.306] | −1.4% |

**10 m 2:1 slope, c' 15 kPa, phi' 25 deg** (§0; LE Bishop 1.887), T6, 20 m
foundation, 6 m margins:

| Mesh | Elements | Across | FOS | vs LE |
|---|---|---|---|---|
| T6 20x10 | 404 | 3.0 (warns) | 1.934 | +2.5% |
| T6 30x15 | 904 | 4.6 | 1.909 [3.000] | +1.2% |
| T6 40x20 | 1604 | 6.1 | 1.897 [1.803] | +0.5% |
| T6 60x30 | 3602 | — | 1.884 [1.147] | −0.2% |

**ACADS 1(a)** (published 1.00; LE 0.985), T6, 5 m foundation:

| Mesh | Elements | Across | FOS | vs 1.00 |
|---|---|---|---|---|
| T6 30x15 | 900 | 7.7 | 0.965 | −3.5% |
| T6 40x16 | 1280 | 9.2 | 0.957 | −4.3% |
| T6 56x20 | 2242 | 12.2 | 0.949 [0.934] | −5.1% |
| T6 72x24 | 3458 | — | 0.949 [0.80] | −5.1% |

Every benchmark is now mesh-consistent: the FOS moves by at most 1.7 %
between its coarsest and finest mesh once 4 or more elements span the slope
(2.7 % at 3 elements), against swings of 0.934 → 0.80, 1.80 → 1.15 and
1.531 → 1.419 before. With no trend left, Richardson extrapolation has
nothing to work on (`fos_richardson is None`); `converged` is True. The
old "monotonic convergence toward LE" of the cross-check was the soil-filled
notch being thinned out by refinement. Tests: `fem2d/tests/test_mesh_study.py`
(`-m slow` for the two benchmark runs; the extrapolation math + bookkeeping
are fast unit tests) and `test_srm_robustness.py::TestMeshConsistency`.

**How a trial is judged** (`srf_history`: `settled`, `growth_ratio`,
`final_residual`). Near the FOS the growth ratio — displacement change over
the second half of a load step's iterations against the first half — rises
steeply with the SRF; GL99 Ex1 32x12, every step run to the ceiling: 0.02
at 1.300, 0.05 at 1.325 (stable), 0.12 at 1.338, 0.23 at 1.350, 0.62 at
1.400 (failing). The 0.1 threshold therefore moves the FOS by about the
bisection tolerance. A step accepted before the ceiling (displacements
plainly standing still) records the upper bound it was accepted on.

## 8. Local factor-of-safety field (mobilized-strength map) — F5

`local_fos_field(result, c, phi)` (`fem2d/local_fos.py`, exported; also attached
as `result.local_fos` when `analyze_slope_srm(..., compute_local_fos=True)`)
maps the stress field to a pointwise factor of safety against Mohr-Coulomb shear:
`local_FOS = tau_available / tau_mobilized`, with `tau_mobilized` the Mohr-circle
radius and `tau_available = c' cos phi' - p sin phi'` (p tension-positive). It is
the reciprocal of PLAXIS's relative shear stress. Plotting hook:
`fem2d.plotting.plot_local_fos`; adapter: `fem2d_local_fos`.

**Consistency check** (Griffiths & Lane Ex 1, T6 32x12, evaluated with the
ORIGINAL strengths on the critical-SRF stress field):

| Case | c', phi' | global SRM FOS | min local FOS | min / global | frac < 1 |
|---|---|---|---|---|---|
| critical | 10 kPa, 20 deg | 1.331 | 1.291 | **0.97** | 0% |
| stable   | 25 kPa, 25 deg | 2.244 | 2.071 | **0.92** | 0% |

(Re-measured 2026-10-14 after the §0 corrections; before: 1.344 / 1.303
and 2.156 / 1.994.)

The minimum local FOS tracks the global SRM factor of safety (within ~3–7%) and
no element is inadmissible (`local_FOS >= 1` everywhere) — the slope is evaluated
at its last STABLE SRF. The low-FOS band is the mobilized failure mass: at the
critical SRF the whole potential sliding mass is near `c'/SRF` mobilization
(here `frac < 1.5` ≈ 65% of the section), reddest along the slip zone and
minimum on it, while the low-confinement surface wedge and the deep foundation
sit higher (green). It envelops the Bishop LE critical circle (gallery Exhibit
13), and its minimum is the local analogue of the global FOS — the same
quantity the sharp displacement/plastic-point localization shows geometrically.

Honest notes: (a) the min/global ratio is **exactly 1 for phi'=0** but a few
percent below 1 for phi'>0, because the SRM reduces `tan phi'` (not phi'), so the
un-reduced-to-reduced strength ratio along the band is slightly under the SRF.
(b) For this D=1 mesh the global minimum lands on a highly-stressed element at
the base/toe — a genuine stress concentration at the fixed-boundary corner. It is
**reported, not smoothed away**; the near-zero-shear deep interior is capped
(`cap`, default 10) rather than blown up.

## 9. Staged excavation: cantilever sheet pile wall against the USS conventional method (GM-71, 2026-10-14)

`analyze_excavation` was rebuilt on the staged path (DESIGN.md, "Staged
excavation"). Before, it did not model an excavation: the wall head moved
into the retained soil in every case.

**Source.** USS Steel Sheet Piling Design Manual (United States Steel, July
1984, updated and reprinted by FHWA), Design Examples, "Design of
Cantilevered Sheet Pile Wall - Granular Soil", pp. 86-90: H = 14 ft, low
water at the dredge line on both sides (effective weights, no net water),
gamma = 115 pcf above the dredge line and gamma' = 65 pcf below, phi = 35
deg, Ka = 0.27, Kp = 6.56 (delta/phi = -0.5, Fig. 5A). The conventional
assumed pressure diagram and statics (p. 87) give D = 10.5 ft; D used 13.5
ft ("increased by 20 to 40 %", +28.5 %, p. 88); zero shear y + x = 1.0 + 4.0
ft below the dredge line; M_max = 26,000 ft-lb/ft (p. 89). The test's own
implementation of the method (`fem2d/tests/test_excavation_staged.py`,
`uss_cantilever`) reproduces it: D0 10.43 ft, Z 1.596 ft at D = 10.5 ft
(printed 1.60), y 1.06 ft, x 4.00 ft, M_max 26,334 ft-lb/ft (+1.3 %: the
manual rounds y and x to 1.0 and 4.0 ft).

**What is compared.** The FEM wall is smooth (`wall_contact='smooth'`, no
wall friction on either face), so the same method is applied with the
Rankine coefficients for delta = 0, Ka = tan^2(27.5) = 0.271 (the manual's
0.27) and Kp = tan^2(62.5) = 3.690, and the wall lengthened by the manual's
own rule: D0 = 14.73 ft (4.49 m), D = 1.3 D0 = 5.84 m; zero shear 2.30 m
below the dredge line (6.57 m deep); M_max = 33,804 ft-lb/ft = **150.4
kN-m/m**; at the dredge line Ka gamma H^3 / 6 = 63.4 kN-m/m. FEM: SI, H =
4.267 m, gamma 18.07 / 10.21 kN/m3, E 40 MPa, nu 0.3, c = 0, psi = 0,
K0 = 1 - sin 35 = 0.426, PZ-27 (the section the example selects): EI =
50,300 kN-m2/m (I = 184.2 in4/ft), EA = 3.36e6 kN/m; width 50 m (half-width
25 m, twice the 11.2 m Rankine passive zone); one lift, 10 steps.

| Mesh (T6 unless noted) | M_max (kN-m/m) | at depth (m) | M / 150.4 | M at dredge (kN-m/m) | K = 6M/(gamma H^3) | head u_x (mm) |
|---|---|---|---|---|---|---|
| 20 x 10 | 144.3 | 6.21 | 0.96 | 67.1 | 0.287 | 96.6 |
| 30 x 15 (default) * | **155.1** | 6.46 | **1.03** | 65.5 | **0.280** | 97.9 |
| 40 x 20 | 153.2 | 6.70 | 1.02 | 64.0 | 0.273 | 100.6 |
| 45 x 22 | 152.6 | 6.70 | 1.01 | 63.8 | 0.273 | 99.0 |
| 60 x 30 | not converged (below) | | | | | |
| 30 x 15, E 20 MPa | 159.0 | 6.46 | 1.06 | 64.9 | 0.277 | 131.8 |
| 30 x 15, E 80 MPa | 150.2 | 6.46 | 1.00 | 65.7 | 0.281 | 79.7 |
| 30 x 15, `wall_contact='bonded'` | 65.7 | 5.00 | 0.44 | 41.7 | 0.178 | 26.4 |
| 30 x 15, T3 | 37.5 | 5.73 | 0.25 | 10.4 | 0.045 | 20.5 |

In the 30 x 15 run the soil next to the wall has sigma_x / sigma_y = 0.271
to 0.272 over the retained height (Ka 0.271), the wall's axial force is
zero (smooth), the in-situ stage moves nothing (< 1e-9 m), and the head
moves 97.9 mm toward the excavation (Delta/H = 0.023).

**Why they should agree, and the tolerance.** A cantilever is statically
determinate once the pressures are known. Above the dredge line the
retained side reaches the active state when the wall top has moved Delta/H =
0.001-0.004 in sand (Caltrans Trenching and Shoring Manual, 2025, Chapter 4,
p. 4-4, Table 4-1 "Mobilized Wall Movements", after Clough 1991: medium
dense sand 0.002); here it moved 0.023, so the moment at the dredge line is
fixed by statics at Ka gamma H^3 / 6 (FEM +3 %). Below it the conventional
diagram takes the full net passive (Kp - Ka) down to the zero-shear point;
the FEM, with the same Rankine limit in front of a smooth wall, mobilizes
it near the dredge line and less deeper down, and the retained side below
the dredge line is not everywhere at active, so its zero-shear point and
moment differ by a few percent. Rowe's moment reduction (the manual's p. 32)
is for anchored walls (free earth support) and does not apply. The tests
assert M_max within 10 % of 150.4 kN-m/m at the zero-shear depth +/- 0.5 m,
the mobilized K at the dredge line between Ka and K0 and within 10 % of Ka,
sigma_x/sigma_y next to the wall within 5 % of Ka, and every wall node above
the excavation level moving toward the excavation with the head the most and
more than 0.002 H. The limit-equilibrium moment does not depend on the soil
modulus; the FEM moment varies 6 % over a fourfold range of E.

**The published wall itself.** With the manual's D = 13.5 ft and a smooth
wall the excavation stage does not converge: with delta = 0 on both faces
the same method needs D0 = 14.73 ft, so the wall is short of limiting
equilibrium (the manual's Kp = 6.56 counts wall friction delta = -phi/2 on
the passive side, which a smooth FEM wall does not have).

**Wall-soil contact.** A bonded wall (shared nodes, the only option before)
hangs the retained soil: its axial force reaches 1.5 times the horizontal
thrust above the excavation level (no wall friction angle up to phi' allows
more than tan phi' = 0.70), the thrust falls below active (K = 0.178) and
the moment to 0.44 of the method's. Real sheet piles have delta between 0
and about phi/2 (the manual's Table 4, p. 13: 17 deg for clean sand); that
needs interface elements.

**Convergence.** The 60 x 30 mesh does not converge within 1000
constant-stiffness iterations a step: the residual stalls near 1e-2 with the
active wedge still creeping about 3 mm per 1000 iterations; the tangent
(Newton-Raphson, line search, cutbacks) does no better. 20 x 10 to 45 x 22
converge with 1 to 6 of their 10 steps accepted at the ceiling with their
displacements standing still (residual ratio 1.5e-5 to 1.9e-3, second-half
movement 0.02 to 0.099 of the first half), the strength reduction's
criterion; the moments agree within 2 % from 30 x 15 to 45 x 22.

**Before (the one-solve model) and after, the default clay case** (c 25 kPa,
phi 20, E 30 MPa, gamma 18; width 10, depth 5, wall 10 m, EI 50,000, EA 5e6,
30 x 15 T6): head u_x -6.1 -> +6.7 mm, M_max 17.2 -> 27.6 kN-m/m, V_max
28.0 -> 25.4 kN/m; with a strut at 1.5 m (k 50,000 kN/m/m) head -3.7 -> +0.8
mm (largest wall u_x +5.0 mm below the strut), M_max 20.9 -> 35.7, V_max
27.0 -> 33.7, strut force -37.6 -> +52.7 kN/m (now compression, counted from
its installation after the first lift). Every case: DESIGN.md, "Staged
excavation". Tests: `fem2d/tests/test_excavation_staged.py`.

## 6. Performance

Wall-clock on the dev machine, single process, including mesh build and
ALL SRF trials (bracket + bisection):

2026-10-14, after the §0 corrections (srf_tol 0.02; the laptop shared
with another test run), with the committed code before them where it was
timed the same way:

| Case | Before | After |
|---|---|---|
| GL99 Ex1, T6 32x12, 9 trials | 10.0 s (FOS 1.344) | **12.9 s** (1.331) |
| GL99 Ex1, T6 48x16 | 49 s | 36.6 s |
| GL99 Ex1, T6 64x20 | — | 63.6 s |
| GL99 Ex2, T6 48x24 (D=1.5) | 98 s | 86.7 s |
| 10 m slope (§0), T6 30x15 (adapter default) | 91.0 s (FOS 3.000) | 44.3 s (1.909) |
| 10 m slope, T6 40x20 | 97.2 s (1.803) | 96.1 s (1.897) |
| 10 m slope, T6 60x30 | 42.5 s (1.147, stopped early) | 241.8 s (1.884) |
| Bishop cross-check, T6 40x20 | 69.6 s (1.419) | 46.2 s (1.156) |
| Bishop cross-check, T6 56x24 | 154 s | 83.5 s |
| ACADS 1(a), T6 72x24 | — | 135.3 s |
| Ex4 undrained, T6 40x16 | 44-75 s | 26-46 s |
| Prandtl footing, T6 40x20, 60 load steps | ~20 s | ~20 s (unchanged path) |

Trials near the FOS that reach the ceiling with their displacements
standing still run their 1000 iterations before they are accepted (or
stop early when the displacements plainly stand still, checked every 50
iterations after half-way), so a run whose old answer came from an early
stall is now slower (60x30 above); runs the old code had pushed past the
true FOS are faster.

One splu factorization serves all trials of an SRM run (the elastic
operator does not depend on c/phi); each iteration is a
back-substitution plus a fully vectorized Gauss-point return mapping.
The phase-2 vectorization made the pre-existing test suite ~19x faster;
the full suite (461 tests incl. slow SRM, 2026-10-14) runs in ~14 min.

## Known limitations (documented, by design)

- **Mesher**: `generate_slope_mesh` places fixed-count node columns and
  Delaunay-triangulates (keeping only the triangles inside the soil
  since 2026-10-14, §0); thin base layers (D ~ 1) make flat elements
  under the toe run, because every column gets `ny` rows. T6 tolerates
  them (GL99 Ex1 1.331 at D = 1, Ex2 1.344-1.356 at D = 1.5). Keep the
  base layer >= 10% of H unless reproducing a D=1 benchmark.
- **Coarse meshes**: the result warns when fewer than 4 elements span the
  slope height (3 elements measured +2.7 % against a fine mesh), and when
  a soil layer has no elements or is thinner than its elements (a thin
  weak seam is lost or becomes a jagged band). `mesh_check=True` compares
  with a mesh 1.5x finer. (The +33 % at D = 1.5 / ny = 12 and the +91 %
  of the old x_extend default recorded here before were measured with the
  soil-filled notch in the mesh.)
- **Undrained (nu -> 0.5)**: no volumetric locking in the SRM FOS — GL99
  Ex4 1.419 at nu = 0.30, 0.49, 0.495, 0.499 and 0.4999 (§2). The
  +20-25 % at nu = 0.49 and the -10-15 % at nu = 0.30 recorded before
  came from the soil-filled notch and from stalled iterations read as
  collapse. nu = 0.5 is refused.
- **Cohesionless slope faces (c ~ 0)**: the zero-confinement chatter that
  the residual-only criterion read as failure is now accepted when the
  displacements stand still (§0). What remains is resolution: the
  critical slide is a thin layer parallel to the face (infinite slope,
  FOS = tan phi' / tan beta), so a coarse mesh comes out high — Duncan
  Ex 2 (c' 0.1 kPa, phi' 40 deg, 2:1): 1.81 / 1.74 / 1.71 on 15x8 /
  30x15 / 45x24 against 1.68 (2.475 on 15x8 before). The result warns
  and quotes tan phi' / tan beta when the surface soil has c' <= 0.5 kPa
  and the slope is dry.
- **Over a firm base**: an undrained slope with a firm layer just below
  the toe fails along the base, not on a circle. Compare with a
  non-circular limit-equilibrium search: Duncan Ex 1 (cu 28.7 kPa, 12.2 m
  2:1, base 1.2 m below the toe): SRM 0.883 = Spencer non-circular 0.883;
  circles 0.969 (published circles 0.95-1.08).
- **Staged construction**: `analyze_staged(max_iter=100)` is low for a
  plastic stage under the constant-stiffness iteration (a footing at 260
  kPa on c' 20 kPa, phi' 25 deg converged with 500 and not with 100);
  pass a larger `max_iter` for plastic stages. `analyze_excavation` uses
  1000 and accepts steps whose displacements stand still (§9).
- **Excavation walls** (§9): no interface elements, so the wall is either
  smooth (default, delta = 0) or bonded (shares the soil's nodes and hangs
  the retained soil); struts are linear springs without preload; one
  groundwater table in every stage, an excavation below it kept full of
  water (a dewatered excavation, with seepage under the wall, is not
  modelled); very fine meshes (60 x 30 on the benchmark) may not converge;
  T3 elements lock (a quarter to a half of the T6 moment).
- **psi policy**: psi_red = min(psi, phi_red) during reduction (PLAXIS
  convention); GL99 used psi = 0 throughout.
- **delta_max curves** include foundation settlement — compare to
  published curves only at matching D, passing `h_ref` = slope height.
- **HS in SRM**: c/phi reduced, hyperbolic stiffness unchanged; HS szz
  tracked elastically.
- **Q8** (GL99's element) not implemented; T6+3pt is the
  equivalent-accuracy triangle (Smith & Griffiths).
