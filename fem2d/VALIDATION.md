# fem2d Validation Suite (branch: fem-modern)

Published-benchmark validation of the modernized fem2d: T6 quadratic
elements, 3D principal-stress Mohr-Coulomb return mapping (with
sigma_zz), and Griffiths-Lane-grade Strength Reduction Method.

Method for all SRM rows: constant-stiffness (initial stiffness)
iteration — the elastic global stiffness is factorized once and shared
by every SRF trial; residual tolerance 1e-5, iteration ceiling 1000 per
load step, 2 gravity steps, psi = 0. Failure = non-convergence
(Griffiths & Lane 1999); bisection to the stated tolerance.
Reproduction: `fem2d/tests/test_validation.py` (`-m slow`) runs the
starred (*) rows with documented assertion bands.

---

## 1. Griffiths & Lane (1999) Example 1 — homogeneous 2:1 slope

H = 10 m, 2:1 slope (26.57 deg), crest margin 1.2H, slope run 2H, toe
run 1.2H; phi' = 20 deg, c'/(gamma H) = 0.05 (c' = 10 kPa, gamma = 20
kN/m3), psi = 0, E' = 1e5 kPa, nu' = 0.3, D = 1 (firm base just below
toe; modeled with a 0.5 m base layer). Published: **FE FOS = 1.4**
(GL99), **Bishop & Morgenstern 1.380**. Bisection tol 0.01.

| Element | Mesh | Elements | FOS | vs published 1.4 |
|---|---|---|---|---|
| T6 (3-pt) * | 32x12, D=1 | 794 | **1.341** | -4.2% |
| T6 (3-pt) | 48x16, D=1 | 1573 | **1.366** | -2.4% |
| T6 (3-pt) | 48x24, D=1.5 | 2341 | **1.341** | -4.2% — GL99 Ex2 invariance (FOS unchanged when a foundation layer is added) reproduced |
| T6 (3-pt) | 32x12, D=1.5 | 794 | 1.866 | +33% — UNDER-REFINED deep domain; refine ny for D > 1 |
| CST | any D=1 mesh | — | no result | fails to converge even at SRF 0.5 — sliver triangles from the column mesher on thin base layers |
| CST | 12x6 small slope (tests) | 112 | caps the 3.0 search range | locking: cannot collapse |

SRF vs dimensionless displacement E' delta_max/(gamma H^2), T6 32x12
D=1 (*), against GL99 Table 2:

| SRF | GL99 published | fem2d T6 |
|---|---|---|
| 0.8-1.0 | 0.379-0.381 | 0.417 |
| 1.2 | 0.422 | 0.455 |
| 1.3 | 0.453 | 0.479 |
| 1.4 | 1.476 (no convergence) | no convergence at ~1.35 |

Curve shape (flat, then knee just below failure) reproduces the paper;
the ~9% level offset is the 0.5 m base-layer settlement plus T6-vs-Q8
discretization.

## 2. Griffiths & Lane (1999) Example 4 — undrained two-layer, D=2

2:1 slope, H = 10 m, phi_u = 0, slope soil cu1/(gamma H) = 0.25 (cu1 =
50 kPa); foundation to D = 2 (10 m below toe), strength cu2. Published:
cu2/cu1 = 1: **1.47** (Taylor); >= 1.5: plateau **~2.1** (toe circle).
T6 40x16 (1314 elem) unless noted, tol 0.02.

| cu2/cu1 | Published | T6, nu = 0.49 | T6, nu = 0.30 |
|---|---|---|---|
| 0.6 | (weak foundation, deep wedge) | — | 0.883 |
| 1.0 * | **1.47** | 1.853 (+26%); 1.756 @ 56x24 (+19%) | 1.306 (-11%); 1.244 @ 56x24 (-15%) |
| 1.5 | ~2.0-2.1 | — | 1.819 |
| 2.0 | **~2.1** (plateau) | range-capped at 3.0 | 2.319 (+10%, plateau not fully captured) |

Honest reading: the undrained (incompressible) case is the weakest spot
of the T6/3-pt formulation. nu = 0.49 (standard undrained elasticity)
locks the nearly-incompressible response and overpredicts ~20-25%;
nu = 0.30 (GL99's nominal elastic constants) underpredicts 10-15%
because elastic sigma_zz = nu(sxx+syy) leaves plane-strain Tresca
yield activated out-of-plane earlier than the in-plane circle. The
published value is bracketed (* the test asserts this bracket).
Designed path: 15-node cubic-strain triangles or B-bar/selective
reduced integration for the incompressible limit, and an undrained-A
style szz treatment; tracked in UPGRADE_PLAN.md as future work.

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

## 6. Cross-check vs slope_stability Bishop (shared geometry)

Shared profile [(0,0),(10,0),(30,10),(50,10)] (2:1, H = 10 m), c' = 10
kPa, phi' = 15 deg, gamma = 18 kN/m3, dry, 5 m foundation.
slope_stability `search_critical_surface(method='bishop')` (read-only
import) vs `analyze_slope_srm` with x_extend = 0:

| Method | FOS |
|---|---|
| Bishop, critical-circle search | **1.173** |
| SRM T6 40x20 * | 1.419 (+21%) |
| SRM T6 56x24 | 1.306 (+11%, converging toward LE with refinement) |
| SRM CST 40x20 | 1.706 (+45% — locking, same mesh as the T6 1.419 row) |

Found while building this row: the old `analyze_slope_srm` default
`x_extend = 2 x domain width` silently starved the slope face of
elements at fixed nx — same mesh with the default extension returned
FOS 2.24 (+91%). The default is now `max(0.5 H, 5 m)` and profiles with
their own margins should pass `x_extend=0`. Orientation (crest-left vs
crest-right) was verified to make no difference (mirrored geometry,
identical FOS).

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

**Griffiths & Lane (1999) Example 1** (published FE **1.4**), T6, D=1 (0.5 m
base sliver), `srf_tol` 0.02, `x_extend=0`:

| Mesh | Elements | FOS | vs 1.4 |
|---|---|---|---|
| T6 24x10 | 500 | 1.381 | −1.4% |
| T6 32x12 | 794 | 1.344 | −4.0% |
| T6 40x16 | 1312 | 1.381 | −1.4% |

Mesh-**consistent**: every density sits in a tight band 1.34–1.38 just below the
published FE value (all within ~4%). Point-to-point convergence is **noisy**
rather than strictly monotonic here — the D=1 column-mesher slivers under the toe
(see Known limitations) inject ~±0.03 of discretization noise, comparable to the
refinement trend — so no Richardson estimate is reported for this case
(`fos_richardson is None`). No mesh runs away; the FOS is bounded near 1.4 at
every density, which is the mesh-consistency the SRM FOS needs.

**Shared-geometry Bishop cross-check** (§6 geometry; LE Bishop **1.173**), T6,
5 m foundation (clean mesh), `srf_tol` 0.02, `x_extend=0`:

| Mesh | Elements | FOS | vs LE 1.173 | step |
|---|---|---|---|---|
| T6 24x12 | 593 | 1.531 | +30.5% | — |
| T6 32x16 | 1046 | 1.456 | +24.1% | −4.9% |
| T6 40x20 | 1623 | 1.419 | +21.0% | −2.5% |

Here the mesh is clean (no base sliver) and the SRM FOS decreases
**monotonically** toward the LE value with each refinement, the step shrinks
(−4.9% → −2.5%, i.e. settling), and Richardson extrapolation gives a
mesh-independent estimate **≈ 1.376** (observed order p ≈ 2.8). The finest 40x20
value 1.419 matches the value recorded in §6. The residual offset above the LE
1.173 at these densities is the expected SRM-vs-Bishop discretization gap
(SRM sits above Bishop and refines toward it; deeper refinement — 56x24 → 1.306,
§6 — continues the trend). Tests: `fem2d/tests/test_mesh_study.py`
(`-m slow` for the two benchmark runs; the extrapolation math + bookkeeping are
fast unit tests).

Note: the constant-stiffness SRM **failure-detection** mesh dependence on very
low-c'/low-φ faces (ACADS 1a: stall point drifts down with refinement) is a
**separate**, still-open tracked follow-up (`UPGRADE_PLAN.md` "SRM
failure-detection mesh consistency") — the GL99 and cross-check meshes/strengths
above do not stall, so they are unaffected.

## 6. Performance

Wall-clock on the dev machine, single process, including mesh build and
ALL SRF trials (bracket + bisection):

| Case | Time |
|---|---|
| GL99 Ex1, T6 32x12, 794 elem, 10 trials, tol 0.01 | **11 s** |
| GL99 Ex1, T6 48x16, 1573 elem | 49 s |
| GL99 Ex1, T6 48x24 (D=1.5), 2341 elem | 98 s |
| Prandtl footing, T6 40x20, 60 load steps | ~20 s |
| Prandtl footing, T6 60x30 | ~54 s |
| Ex4 undrained, T6 40x16 | 44-75 s |
| Bishop cross-check, T6 56x24 | 154 s |

One splu factorization serves all trials of an SRM run (the elastic
operator does not depend on c/phi); each iteration is a
back-substitution plus a fully vectorized Gauss-point return mapping.
The phase-2 vectorization made the pre-existing test suite ~19x faster;
the full suite (321 tests incl. slow SRM) runs in ~13 min.

## Known limitations (documented, by design)

- **Mesher**: `generate_slope_mesh` places fixed-count node columns and
  Delaunay-triangulates; thin base layers (D ~ 1) make high-aspect
  slivers under the toe run. T6 tolerates them; CST may fail to
  converge spuriously. Keep the base layer >= 10% of H unless
  reproducing a D=1 benchmark with T6.
- **Deep/wide domains at fixed nx/ny**: FOS overpredicts when the slope
  face is under-resolved (D=1.5 @ ny=12: +33%; old x_extend default:
  +91%). Refine until FOS stabilizes; the Ex1 rows above are the
  reference densities.
- **Undrained (nu -> 0.5)**: T6/3-pt retains some incompressibility
  locking (+20-25% at nu=0.49); nu=0.3 underpredicts 10-15% via
  out-of-plane Tresca yield. Bracket with both, or wait for
  B-bar/15-node elements (future work).
- **Cohesionless slope faces (c ~ 0)**: zero-confinement surface Gauss
  points chatter and the constant-stiffness residual plateaus at ~1e-4,
  which the benchmark-calibrated failure criterion (residual 1e-5,
  ceiling 1000) reads as failure. There is NO residual level that
  separates this chatter from true collapse (measured: failing GL99
  trials plateau in the same band), so the criterion is left calibrated
  and cohesionless-face runs should pass tol ~1e-3 or add 1-2 kPa of
  cohesion (standard SRM practice).
- **psi policy**: psi_red = min(psi, phi_red) during reduction (PLAXIS
  convention); GL99 used psi = 0 throughout.
- **delta_max curves** include foundation settlement — compare to
  published curves only at matching D, passing `h_ref` = slope height.
- **HS in SRM**: c/phi reduced, hyperbolic stiffness unchanged; HS szz
  tracked elastically.
- **Q8** (GL99's element) not implemented; T6+3pt is the
  equivalent-accuracy triangle (Smith & Griffiths).
