# Calculation Modules — In-Depth QC Review

Started 2026-06-06. Scope confirmed with user: **both layers, math first** —
engineering correctness of the core computation, then the `calc_steps.py` /
`calc_package` presentation layer. Emphasis: **engineering correctness + code quality**.

## Context / ground rules
- **v5.0 work is on a separate branch/worktree (`.claude/worktrees/v5.0-deepagents/`) owned
  by another effort — do NOT touch it.** This QC runs against the current checked-out files.
- **Validation is done in funhouse** (the `funhouse_agent` adapters are the real consumer
  surface, exercised live in Databricks). Local pytest suites are green but encode current
  behavior; a passing suite is *not* proof of engineering correctness. Findings note how each
  issue surfaces through an agent caller.
- No version bumps / PyPI publishes. No code edits during the review pass — findings first,
  fixes in a later pass once the user prioritizes.

## Scope (13 core engineering modules with both a math layer and a `calc_steps.py`)
bearing_capacity · settlement · axial_pile · sheet_pile · lateral_pile · pile_group ·
wave_equation · drilled_shaft · seismic_geotech · retaining_walls · ground_improvement ·
slope_stability · downdrag — plus the shared `calc_package/` renderer.

## Severity
**Critical** wrong number a user would trust · **High** materially wrong / silently ignored
input in common use · **Medium** wrong in edge cases or non-conservative approximation ·
**Low** docs/dead-code/clarity.

## Status
| Module | Math QC | calc_steps QC | Tests | Notes |
|--------|:------:|:------:|:------:|-------|
| bearing_capacity | ☑ | ☑ | 66 pass | 2 High, 1 Med, 6 Low |
| settlement | ☑ | ◐ | 39 pass | 3 Med, 2 Low |
| axial_pile | ☑ | ☐ | 55 pass | 1 Med, 4 Low |
| sheet_pile | ☑ | ☐ | 26 pass | 1 Med, 3 Low |
| lateral_pile | ☑ | ☐ | 96 pass | 1 Med, 2 Low — solver is excellent |
| pile_group | ☑ | ☐ | 72 pass | 1 Med, 2 Low |
| wave_equation | ☑ | ☐ | 45 pass | 1 High*, 2 Med (*verify numerically) |
| drilled_shaft | ☑ | ☐ | 48 pass | 1 Low/Med, 4 Low (clean) |
| seismic_geotech | ☑ | ☐ | 71 pass | 2 Low/Med, 1 Low (residual_strength not deep-read) |
| retaining_walls | ☑ | ☐ | 70 pass | 1 Med, 4 Low |
| ground_improvement | ☑ | ☐ | 43 pass | 1 Med, 1 Low (clean) |
| slope_stability | ☑ | ☐ | 237 pass/17 skip | 3 Low — solid module |
| downdrag | ☑ | ☐ | 53 pass | 1 Low (clean) |
| calc_package | ☑ | — | 100 pass/3 skip | 1 Low (renderer clean) |

---

## bearing_capacity

**Verified correct:** Nc/Nq/Nγ for vesic/meyerhof/hansen all match textbook (φ=30°:
Nc=30.1, Nq=18.4, Nγ_vesic=22.4). Shape/depth/base/ground factors match Vesic/Hansen/GEC-6
forms. Clean dataclasses, good input validation, good docstrings. 66 tests pass.

### BC-1 [High] — Two-layer method mislabeled; it is linear interpolation, not Meyerhof & Hanna
`capacity.py::_compute_two_layer` docstring claims "Meyerhof & Hanna (1978) punching shear,"
but the code linearly interpolates qult in H/B between top-layer capacity (qt) and bottom-layer
capacity (qb): H≥B → qt; H<B → `qb+(qt−qb)·H/B` (strong/weak) or `qt+(qb−qt)·H/B` (weak/strong),
each capped. True M&H adds a punching-shear term (`2·ca·H/B + γH²(1+2Df/H)·Ks·tanφ/B` for
strong-over-weak) and is non-linear in H; the linear form generally under-predicts
strong-over-weak. Also: `q1_over_q2 = qb/qt` computed and never used (dead code); the returned
result reuses the **top-layer** factor set (Nc, sc, term_cohesion…) but overwrites only
q_ultimate/q_allowable/q_net, so `term_cohesion+term_overburden+term_selfweight ≠ q_ultimate` →
calc_package term breakdown / `summary()` internally inconsistent for two-layer runs.
Fix: implement real M&H or relabel honestly + repair result-object consistency.

### BC-2 [High] — Vesic load-inclination factors silently vanish when vertical_load=0
Inclination factors use `H = V·tan(β)` with `vertical_load` defaulting to 0, so specifying
`load_inclination` without `vertical_load` → H=0 → ic=iq=ig=1 (inclination ignored, no warning).
An agent passing an inclination angle but omitting V silently gets the un-inclined capacity. The
meyerhof path doesn't use V, so the two `factor_method`s disagree on whether inclination applies.
Fix: require/derive V when `load_inclination≠0`, or warn.

### BC-3 [Medium] — GWT within depth B below the footing not handled in the Nγ term
`soil_profile.gamma_below_footing` returns full **total** unit weight whenever
`gwt_depth ≥ footing_depth`, even when GWT lies between Df and Df+B (inside the shear wedge).
Comment calls this "conservative," but total (not buoyant) γ over-predicts the self-weight term →
un-conservative. Standard practice averages γ over depth B below the base.

### BC-4 [Low/Med] — Two-layer `thickness` semantics ambiguous
`overburden_pressure` treats layer1 as ground surface→footing base (layer1.γ); `_compute_two_layer`
treats `layer1.thickness` as footing base→interface (H). "thickness" ≠ the layer's true thickness;
layer1 implicitly sits both above and below the footing. Works for the common case but is a trap.

### BC-5 [Low] — DESIGN.md / API drift
DESIGN.md + docstrings reference public `analyze_bearing_capacity(B,L,D,…)` / `analysis.compute()`;
`__init__.py` exports only classes (no such function). Reconcile docs with real API + funhouse adapter.

### BC-6 [Low] — `BearingSoilProfile.effective_unit_weight()` appears unused (verify, then prune)

### BC-7 [Low, calc_steps] — dead/misleading locals `phi = f.width` and `phi_rad`
`get_calc_steps` (calc_steps.py:109-111) sets `phi = f.width` and `phi_rad`; neither is ever read
(`phi_deg` is what's used). Harmless numerically but confusing — assigning footing width to a var
named `phi`. Delete both.

### BC-8 [Low, calc_steps] — calc package hard-codes the M&H label (perpetuates BC-1)
calc_steps.py:344 emits `"Two-layer analysis (Meyerhof & Hanna, 1978):"` into the rendered package,
presenting the linear-interpolation result to the engineer as Meyerhof & Hanna. Fix alongside BC-1
(the line 351 "Combined (interpolated)" wording is at odds with the line 344 M&H header).

---

## settlement

**Verified correct:** Boussinesq corner via Newmark integration (I=0.1752 at m=n=1 ✓ vs table);
Westergaard corner (I=0.0833 at m=n=1 ✓ vs tabulated ≈0.084); Boussinesq point load; 2:1 and
2:1-strip; consolidation NC / OC-stays / OC-exceeds three-case e-log-p (all standard); Terzaghi
U–Tv in both branches (U<60% parabolic, U≥60% log) and its inverse; secondary compression
Cα/(1+e0)·H·log10(t2/t1); time factor Tv=cv·t/Hdr². 39 tests pass.

### SET-1 [Medium] — Schmertmann peak influence factor uses overburden at the footing base, not at the peak-Iz depth
`immediate.schmertmann_settlement`: `Iz_peak = 0.5 + 0.1·√(q_net/q0)` with `q0` = effective
overburden **at the footing base** (z=0). Schmertmann (1978) defines Izp with σ'vp = effective
overburden at the **depth of peak Iz** (z=B/2 square, z=B strip). Using the (smaller) base value
inflates √(Δq/σ'vp) → larger Izp → over-predicts immediate settlement. Use σ'v at z_peak.

### SET-2 [Medium] — Schmertmann C3 shape factor is non-canonical and likely double-counts shape
`schmertmann_settlement` multiplies by `C3` (1.0 square; `1.03−0.03·L/B` rectangular; 0.73 strip).
Canonical Schmertmann (1978) uses only C1 (depth) and C2 (creep); the strip vs axisymmetric
difference is already carried by the **Iz diagram + z_max** (strip: peak at B, zero at 4B; square:
peak at B/2, zero at 2B), which the code already switches on `is_strip`. Applying C3=0.73 on top of
the strip diagram double-counts the shape effect. Verify against the cited "Terzaghi et al. 1996"
form or drop C3.

### SET-3 [Medium] — Elastic-method settlement never applies a shape/rigidity influence factor
`analysis._compute_immediate` calls `elastic_settlement(q_net, B, Es, nu)` leaving `Iw=1.0` always.
`footing_shape` is accepted but unused for the elastic method, and `elastic_settlement`'s own
`shape` param is documented "not directly used." So a square/circular/rigid footing all get Iw=1.0
(≈flexible). Select Iw from shape+rigidity (e.g. rigid square 0.82, rigid circle 0.79) or document
that the caller must pass Iw. Surfaces through funhouse: "elastic settlement of a square footing"
silently ignores the shape.

### SET-4 [Low] — Boussinesq/Westergaard **center** returns 4·q at exactly z=0
`stress_at_depth(..., method="boussinesq"/"westergaard", location="center")` at z=0 returns
`4·boussinesq_rectangular(...,0)=4q` because the corner function returns `q` (not q/4) at z≤0. The
z→0⁺ limit is correct (4×0.25q=q); only the exact-z=0 value is wrong. Layer centers are >0 in
practice, so low impact — guard z=0.

### SET-5 [Low] — secondary `t1` default and single-layer Hdr assumptions
`_compute_secondary` falls back to `t1=1.0 yr` (end of primary) when `cv` is None — arbitrary;
and `_get_Hdr` treats the whole consolidation zone as one doubly-drained layer (Hdr=ΣH/2). Both are
reasonable defaults but should be documented as such (results scale with log(t2/t1)).

---

## axial_pile

**Verified correct:** Nordlund unit skin friction form `Kd·CF·σv·sin(δ+ω)/cos(ω)`; Tomlinson α
curves; `end_bearing_cohesive = 9·cu·At` (Nc=9); Meyerhof Nq'/qL chart interpolations (plausible);
open-ended pipe plug-vs-unplug governing-min logic (GEC-12 7.2.1.4); effective-stress profile with
GWT-through-layer handling; pipe/H-pile section properties. 55 tests pass.

### AP-1 [Medium] — `beta_from_phi`: "fellenius" and "burland" branches are byte-identical → `method` is a no-op
`beta_method.beta_from_phi` (lines 53-59): both branches compute `K0=(1−sinφ); beta=K0·tanφ·√OCR`.
So selecting `"fellenius"` vs `"burland"` gives the same number, despite the docstring advertising
`fellenius: (1−sinφ)·tanφ` vs `burland: K0·tanφ·OCR^0.5`. Also the Fellenius β=(1−sinφ)·tanφ is the
NC value — applying `√OCR` to the Fellenius branch is non-standard. Fix: implement the two forms
distinctly (Fellenius without √OCR, or per Table 7-9 β ranges), or collapse to one documented form.
Surfaces in funhouse if an agent picks a method expecting different behavior.

### AP-2 [Low] — dead segmentation code in `capacity.compute`
`n_segments=50`, `dz`, and `depths=np.linspace(...)` (lines 91-93) are computed and never used; the
loop integrates per soil **layer** at the layer-in-pile midpoint. (Midpoint × thickness is exact for
the piecewise-linear σv within a layer, so accuracy is fine — but the abandoned discretization is
misleading; the midpoint rule is slightly off only where GWT splits a layer.) Remove the dead code.

### AP-3 [Low] — delta/φ ratio doc drift + hardcoded clay φ in beta mode
`AxialSoilLayer.delta_phi_ratio` docstring says default 0.8 (steel)/1.0 (concrete-timber), but
`nordlund.delta_from_phi` uses 0.75/0.90. And in beta mode, cohesive layers hardcode `phi=25°`
(capacity.py:118,157,262) for both skin and tip — reasonable default but undocumented/unparameterized.

### AP-4 [Low] — Nordlund chart-fits drop real dependencies (documented simplifications)
`nordlund_Kd` ignores the pile displaced-volume V/V0 family (fixes the displacement-pile curve);
`alpha_t_factor` ignores φ (returns 1.0 for D/b>5, the max → can over-predict tip, but capped by
`qL`); `nordlund_CF` keys only on δ/φ. Acceptable per the docstrings, but flag that results can
deviate from rigorous Nordlund; tie out against a GEC-12 worked example if precision matters.

### AP-5 [Low] — uplift estimate is a flat 0.75·Q_skin
`Q_uplift = 0.75·total_Qs` (rule-of-thumb); in the unplugged-governs branch `total_Qs` already
includes inside friction (line 201 runs before line 210), and pile self-weight isn't added. Document
or refine (tension Ks reduction, +W_pile).

---

## drilled_shaft  (cleanest module so far)

**Verified correct:** α-curve (0.55 for cu/pa≤1.5, decreasing); O'Neill-Reese β=1.5−0.245√z_ft
with correct m→ft conversion + [0.25,1.2] clamp; sand side cap 200 kPa; Horvath-Kenney rock
`fs=0.65√(qu·pa)`; tip sand `qb=57.5·N60` (N60≤50) + 1.27/D large-base reduction; rock-tip RQD
heuristic; **AASHTO LRFD resistance factors all match Table 10.5.5.2.4-1** (side 0.45/0.55/0.55,
tip 0.40/0.50/0.50, uplift 0.35/0.45/0.40); GEC-10 exclusion zones (top 1.5 m, bottom 1·D
clay-only, cased zone). 48 tests pass.

### DS-1 [Low/Med] — cohesive end bearing has no limiting qb and no large-base reduction
`end_bearing.end_bearing_cohesive` returns `Nc·cu·A` uncapped. GEC-10/O'Neill-Reese cap net unit
end bearing in clay at ~4 MPa (≈80 ksf) and reduce qb for base diameters > ~1.9 m. For very stiff
clay (cu ≳ 440 kPa) this over-predicts tip. Add the cap + base-size reduction.

### DS-2 [Low] — Nc linearization differs slightly from O'Neill-Reese
`Nc=min(6+L/D, 9)` reaches 9 at L/D=3; O'Neill-Reese `Nc=6(1+0.2·L/D)≤9` reaches 9 at L/D=2.5.
Minor over-conservatism for 2.5<L/D<3.

### DS-3 [Low] — `end_bearing_cohesionless` large-base reduction uses shaft D, not bell/base diameter
capacity.py:189 passes `shaft.tip_area` (bell area if belled) but `D=shaft.diameter` for the 1.27/D
reduction. If belled, the base (bell) diameter governs the reduction. Pass the base diameter.

### DS-4 [Low] — sand side β / 200 kPa cap applied regardless of N60
O'Neill-Reese β formula and the 200 kPa cap presume N60 ≥ 15; lower N60 sands need a reduced β.
`ShaftSoilLayer.N60` is only consulted for tip bearing, not side. Document or branch on N60.

### DS-5 [Low] — `capacity_vs_depth` mutates the shared `shaft.length` (not reentrant)
Same pattern as axial_pile AP-2: temporarily overwrites `self.shaft.length` in a loop. Works
(restored after), but mutating the shared geometry object is fragile under concurrency.

---

## lateral_pile  (solver is the best-engineered numerics in the set)

**Verified correct (high confidence):** the finite-difference beam-column solver in `solver.py` —
the 5-point stencil for `EI·y''''+Q·y''+Es·y=0` (incl. the destabilizing +Q·y'' P-delta sign),
all four boundary-condition rows using 2 fictitious nodes per end (head shear=Vt, head
moment=Mt / fixed slope=0 / partial M=−Kr·θ; tip M=0 and V=0), the slope/moment/shear
post-processing, and the closed-form **Hetenyi** validation oracle (β=(Es/4EI)^¼ and the A/B
coefficients all algebraically correct). `SoftClayMatlock` fully correct (Np wedge + 9c flow-around,
zr critical depth, y50=2.5·ε50·b, static (y/y50)^⅓ to 8y50, cyclic 0.72pu plateau w/ depth taper).
`SandAPI` correct (`p=A·pu·tanh(kzy/A·pu)`, A=max(0.9,3−0.8z/b)). There is a dedicated 54 KB
`validation.py` benchmark suite. This module is in excellent shape.

### LP-1 [Medium] — `SandReese` implements a bilinear curve, not the documented Reese (1974) three-part curve
`SandReese.get_p` (py_curves.py:760-802) computes `pu` correctly but then returns
`min(k·z·y, A·pu)` — a linear-elastic-perfectly-plastic curve. The class docstring promises a
"three-part curve construction with parabolic transition." The parabolic branch, the `pm=B·pu`
intermediate point, and the final linear ramp to `pu` at `yu=3b/80` are all missing; the computed
`yu` is dead. The ultimate plateau (A·pu) is right but the transition is too stiff vs COM624P, so
deflection/moment in sand via this model deviate. **Workaround exists:** `SandAPI` (tanh) is correct
— prefer it. Fix: implement the full Reese construction or relabel `SandReese` as bilinear.

### LP-2 [Low] — dense O(n³) solve of a pentadiagonal system
`_assemble_and_solve` builds a dense (n+5)² matrix + `np.linalg.solve`. Correct but ~O(n³); a
banded solver (`scipy.linalg.solve_banded`) would cut the cracked-EI loop / `capacity_vs_depth`
cost. Performance only — no correctness impact.

### LP-3 [Low] — no above-ground free (stickup) length
Load Vt/Mt is applied at z=0 = ground surface; a free-standing column length above grade isn't
modeled (depths outside the soil layers just return p=0, but the head is fixed at grade). Fine for
embedded-head problems; document the limitation for pile-bent / column applications.

---

## pile_group

**Verified correct:** Converse-Labarre `Eg=1−θ/(90mn)[n(m−1)+m(n−1)]`, θ=atan(d/s); block-failure
`9·cu·B·L + 2(B+L)·L·cu`; FHWA p-multipliers (0.8/0.4/0.3 at s/D≤3 → 1.0 at s/D=5); equivalent-raft
settlement (raft at ⅔L, 2V:1H spread, Σσ·dz/Es); `analyze_vertical_group_simple`
`P=Vz/n ± My·x/Σx² ± Mx·y/Σy²`. 72 tests pass.

### PG-1 [Medium] — `analyze_group_6dof` assembles an incomplete 6×6 stiffness matrix
The off-diagonal translational coupling for battered piles (`kxy/kxz/kyz`) is computed but only
`kxz`/`kyz` are referenced and **multiplied by 0** (lines 310-311, "simplified"); `kxy` is never
added. Individual pile bending/head-fixity rotational stiffness is omitted (piles modeled as
axial+lateral springs; rotational stiffness comes only from axial eccentricity `kzz·y²`,`kzz·x²`).
**Result:** correct for symmetric, vertical groups under vertical + moment (the retained terms are
the governing ones), but battered piles and lateral-load-induced couplings are not captured. The
`# simplified` comments mark this — but the class advertises general 6-DOF battered-pile analysis.
Either complete the matrix (full direction-cosine transform + pile bending) or scope the docstring.

### PG-2 [Low] — sign conventions in the 6-DOF back-calc
`dx_pile = dx + rz·y`, `dy_pile = dy − rz·x` are flipped vs the right-hand rule (`−rz·y`,`+rz·x`),
and `dz_pile` uses `+ry·x` where strict RH gives `−ry·x`. Self-consistent with the K assembly (so
pile **forces** are right for vertical groups, where lx=ly=0 nullifies the dx/dy terms), but the
reported cap **rotations** may not match a textbook RH convention, and the dx/dy signs would matter
for battered piles. Reconcile to one explicit convention.

### PG-3 [Low/Med] — singular-matrix fallback silently drops lateral load
If `K_group` is singular (e.g. vertical piles with `lateral_stiffness=None→0` and a horizontal
`Vx/Vy`), `analyze_group_6dof` falls back to `analyze_vertical_group_simple`, which **ignores Vx/Vy
entirely** — the horizontal load vanishes with no warning. Warn that lateral load was dropped, or
require a lateral stiffness when Vx/Vy≠0.

---

## wave_equation

**Verified correct:** Courant time-step + cushion stability; symplectic-Euler explicit integration;
cushion loading/unloading with COR² hysteresis and no-tension; pile elastic internal springs;
stress unit conversion (N/m²→kPa); soil resistance sign (opposes motion); skin/toe Rult split;
bearing-graph assembly (`blows/m = 1/set`, interpolation). 45 tests pass.

### WE-1 [High — verify numerically] — `permanent_set` is the max/turning-point toe displacement, not the plastic set
The blow terminates at the first turning point after step 100 (`np.all(vel[1:]<0.01) and vel[0]<0`
fires when pile velocities cross zero at max penetration). `permanent_set = disp[n]` is therefore ≈
the **maximum** toe displacement — the elastic rebound (≈ quake) is **never subtracted**, even though
the code's own comment (time_integration.py:292-293) says "permanent set = final toe displacement
minus elastic rebound … Σ(R_static/k)". `bearing_graph` then does `blows/m = 1/set`, so an inflated
set → **too few blows/m → bearing graph over-predicts capacity at a given blow count (unconservative
for driveability)**. Fix: subtract the elastic recovery (or integrate long enough to capture true
residual) and reconcile with the comment.

### WE-2 [Medium] — Smith static spring is reversible nonlinear-elastic, not elasto-plastic
`SmithSoilModel.static_resistance(d)` returns `Ru·d/quake` (|d|≤quake) else `±Ru` as a pure function
of the **current** displacement — no plastic unloading branch / no memory of max displacement. A true
Smith spring unloads along slope Ru/quake from the max point, leaving a plastic offset (= the
permanent set). As written the spring is fully reversible, so the model has no intrinsic permanent
set — which is why WE-1's set is an artifact of the termination time. Implement the elasto-plastic
load/unload path.

### WE-3 [Medium] — damping ∝ R_ultimate labeled "standard GRLWEAP" (default is ∝ mobilized R_static)
`total_resistance` uses `R_d = J·R_ultimate·v`. The docstring asserts this is "the standard
GRLWEAP/GEC-12 formulation," but GRLWEAP's default Smith damping is `R_d = J·R_static·v` (∝ the
**mobilized** static resistance); `∝ R_ultimate` is the less-common "Smith-viscous" option. The two
differ most early in the blow (small mobilized Rs, high v). Also damping is gated to loading only.
Either switch to ∝ R_static or correct the documentation to name the viscous variant.

---

## sheet_pile

**Verified correct:** Rankine Ka/Kp; Coulomb Ka (reduces exactly to Rankine at δ=β=0, α=90°);
Jaky K0; active/passive pressure incl. cohesion terms; tension-crack depth (code, not its docstring);
free-earth-support anchored solver (moment about anchor → D, horizontal equ. → T); cantilever
simplified method (moment about base); detailed effective-stress + differential-water integration
on both sides. 26 tests pass.

### SP-1 [Medium — inclined walls only] — `coulomb_Kp` numerator uses sin²(α+φ) instead of sin²(α−φ)
The passive Coulomb numerator should be `sin²(α−φ)`; the code uses `sin²(α+φ)` (copied from the
active formula — the δ and β signs *were* correctly flipped to passive, only the φ sign in the
numerator was not). Identical for a vertical wall (α=90° → sin(90±φ)=cosφ), so the default case is
correct; over-/under-predicts Kp for battered walls (α≠90°).

### SP-2 [Low] — `pressure_method="coulomb"` is effectively a no-op (≡ Rankine)
`_compute_Ka_Kp` (cantilever.py:226) calls `coulomb_Ka/Kp(phi)` with default δ=0/α=90/β=0, which
equals Rankine. Wall friction δ, batter α, backfill slope β are never threaded from the analysis
into the coefficient calls, so selecting "coulomb" changes nothing. (Same class of "switch that
doesn't switch" as axial_pile AP-1.) Thread δ/α/β through, or drop the option.

### SP-3 [Low/Med] — cantilever applies FOS on passive AND a 1.2× embedment increase (double safety)
`analyze_cantilever`: `_find_embedment` already divides passive by `FOS_passive` (default 1.5), then
`D_design = D_converged * 1.2`. The classic simplified method uses *either* FOS≈1 + 30–40% depth
increase *or* FOS≈1.5–2 with no increase — not both. As written it's notably conservative; pick one
safety basis and document it.

### SP-4 [Low] — `tension_crack_depth` docstring formula is wrong (code is right)
Docstring shows `z=(2c/√Ka − Ka·q)/(Ka·γ)`; the correct (and implemented) form is
`z=(2c/√Ka − q)/γ`. Fix the docstring.

---

## retaining_walls

**Verified correct:** Rankine sloped-backfill Ka (Das Eq 7.18); resultant active/passive forces +
locations; cantilever sliding (`V·tanδb+caB+Pp`/Pa, δb=⅔φ), overturning about toe, bearing with
middle-third + Meyerhof trapezoidal/triangular pressures; **Coulomb δ IS threaded here** (δ=⅔φ,
backfill slope) — so the method switch actually works (unlike sheet_pile). MSE (GEC-11): Kr/Ka
profile (1.7→1.2), F* (2.0→tanφ), Tmax coherent-gravity, pullout `F*·α·σv·Le·C·Rc`, external
sliding on soil-on-soil φ (no ⅔ reduction). 70 tests pass.

### RW-1 [Medium] — active thrust applied fully horizontal (no inclination decomposition)
`horizontal_force_active` returns `Pa=½KaγH²+KaqH−2c√KaH` and the cantilever checks treat it as a
purely horizontal force. For Coulomb (δ=⅔φ) or sloped backfill the thrust acts at δ/β from
horizontal: the horizontal component is `Pa·cosδ` (so driving is overstated by ~1/cosδ) and the
favorable **vertical** component `Pa·sinδ` (which adds to V → more sliding/overturning resistance and
shifts bearing) is **omitted**. Exact only for Rankine vertical wall + level backfill. Decompose the
thrust per the chosen method.

### RW-2 [Low/Med] — MSE internal active-zone uses the Rankine line for metallic reinforcement
`check_internal_stability`: `La=(H−z)·tan(45−φ/2)`. GEC-11 uses the **bilinear (coherent-gravity)**
failure surface for inextensible/metallic reinforcement (0.3H offset at top), reserving the Rankine
line for extensible/geosynthetic. Le and pullout FOS deviate for steel-strip walls.

### RW-3 [Low/Med] — MSE external bearing uses trapezoidal q_toe, not AASHTO effective width
`check_external_stability` uses `q_toe=W/L·(1+6e/L)`; AASHTO/GEC-11 MSE convention is the Meyerhof
**uniform pressure over the effective width** `σv=W/(L−2e)`. Different (and non-conservative vs the
peak trapezoidal value) — align with AASHTO 11.10.

### RW-4 [Low] — soil-on-heel weight is rectangular (h_stem×heel)
Omits the triangular soil wedge above a sloped backfill and uses stem height (not full retained
height). Minor under-count of stabilizing weight/moment.

### RW-5 [Low] — no standalone earth-pressure-coefficient method exposed (funhouse interface gap)
Per `module_feedback.json`, agents look for a Ka/Kp method on retaining_walls and don't find one
(coefficients are computed internally). Expose a thin `earth_pressure_coefficient` entry for the agent.

---

## ground_improvement  (clean)

**Verified correct:** area-replacement ratio (triangular √3/2·s², square s²) — consistent with
de=1.05s/1.13s; SRF = 1/(1+as(n−1)); composite modulus Es(1+as(n−1)) (correctly noted ≠ Voigt avg);
Barron/Hansbo radial: F(n)=ln(n/s)+(kh/ks)ln(s)−0.75, Ur=1−exp(−8Tr/F), Tr=ch·t/de², Carrillo
combination, inverse-time solver; surcharge preloading (delegates to verified time_rate). 43 pass.

### GI-1 [Medium] — `equivalent_drain_diameter` is off by a factor of 2
Returns `dw=(w+t)/π`. Hansbo's equal-perimeter equivalent diameter is `dw=2(w+t)/π`. For a typical
100×4 mm PVD the standard gives ≈0.066 m — which is exactly the module's hardcoded default `dw=0.066`
everywhere else. So the helper yields **half** the correct dw; an agent that derives dw from
width/thickness and passes it in gets a too-small dw → wrong n=de/dw and F(n). Add the factor of 2.

### GI-2 [Low] — `improved_bearing_capacity` uses simplified (1+as(n−1)), not the full Priebe factor
Documented as the "Priebe improvement factor approach" but applies the same stress-concentration
factor as settlement; Priebe's bearing improvement factor n₀ is a distinct function of as and ν.
Acceptable as a first-order estimate; label it as such.

---

## seismic_geotech

**Verified correct:** liquefaction NCEER (rd Liao-Whitman piecewise, MSF=10^2.24/M^2.56, fines
correction α/β, CRR=1/(34−N)+N/135+50/(10N+45)²−1/200, CSR with MSF) all per Youd et al. 2001;
Mononobe-Okabe KAE/KPE reduce correctly to Rankine static (δ=0→Ka/Kp) and carry the seismic angle
θ=atan(kh/(1−kv)) properly + Seed-Whitman fallback and 0.6H increment height; Vs30/N̄/s̄u harmonic
means; site-class boundaries (1500/760/360/180; N 50/15; su 100/50); AASHTO Fa/Fv tables spot-checked
(D, E). 71 tests pass. (residual_strength.py not deep-read.)

### SG-1 [Low/Med] — liquefaction total stress uses γ(z)·z, not the layered integral Σγᵢhᵢ
`evaluate_liquefaction` sets `sigma_v = gamma*z` using each point's **own** unit weight × full depth
— so for a layered profile with lighter shallow soils it mis-estimates total (and hence effective)
stress, shifting CSR. Integrate the overburden through the overlying layers.

### SG-2 [Low/Med] — `Fpga` is taken as `Fa(Ss)`; it should be interpolated against PGA
`site_coefficients(site_class, Ss, S1)` returns `Fpga = Fa` (the Fa table interpolated at Ss). AASHTO
defines Fpga vs **PGA** (a separate input the function doesn't accept). Either add a PGA argument or
document that Fpga≈Fa(Ss) is an approximation.

### SG-3 [Low] — M-O wall-batter β sign convention is flipped vs the typical AASHTO textbook form
`mononobe_okabe_KAE` uses `cos²(φ+β−θ)`, `cos(δ+θ−β)`, `cos(i+β)` where many texts write
`cos²(φ−β−θ)`, `cos(δ+θ+β)`, `cos(i−β)`. Consistent internal flip → correct & Rankine-reducing for
the default vertical wall (β=0); verify the batter sign matches the documented convention before
trusting battered-wall KAE/KPE.

---

## slope_stability  (solid)

**Verified correct:** **Fellenius (OMS)** — `Σ[c·dl+max(N',0)tanφ]/Σ W·sinα` with the
moment-arm form `W·(x_mid−xc)/R ≡ W·sinα` and effective-normal clamping; **Bishop's Simplified** —
`m_α=cosα+sinα·tanφ/F`, `[c·b+(W−u·b)tanφ]/m_α`, iterated — both exact. Slice mechanics: multi-layer
weight (γ above / γ_sat below GWT), hydrostatic + Ru pore pressure, tension crack (c=φ=0 on crack
face + ½γ_w·z_w² thrust at z_w/3), pseudo-static `kh·W`. Ships a Duncan-benchmark verification suite.

### SS-1 [Low/Med] — Spencer & Morgenstern-Price use an approximate (α−θ) m_α form
`spencer_fos`/`morgenstern_price_fos` shift `m_α=cos(α−θ)+sin(α−θ)tanφ/F` and drive force eq. with
`W(sinα+cosα·tanθ)`, iterating θ/λ until FOS_moment=FOS_force. This is a reasonable engineering
approximation, not the textbook-exact Spencer/M-P interslice-force solution — but it's validated
against Duncan's published examples in the test suite. Note the approximation; don't treat as exact GLE.

### SS-2 [Low] — Bishop numerator not clamped for high pore pressure
`(W − u·b)·tanφ` can go negative when `u·b > W` (artesian / perched water), contributing negative
resistance; Fellenius clamps the effective normal but Bishop does not. Clamp to 0 for robustness.

### SS-3 [Low] — pore pressure = γ_w·(z_gwt − z_base), no seepage/cos²β correction
`_pore_pressure_at_base` takes the piezometric head as the GWT elevation above the base (hydrostatic),
which slightly overestimates u on steeply inclined phreatic surfaces (rigorous: head·cos²β or a
flownet). The Ru option is the alternative. Acceptable; document the assumption.

---

## downdrag  (clean)

**Verified correct:** Fellenius unified **neutral-plane** method — load-from-top vs resistance-from-tip
curves crossing = NP (force equilibrium), with `β·σv'` / `α·cu` unit friction, dragload above NP,
positive shaft below; structural load = Q_dead+dragload while the **geotechnical check excludes
dragload** (cancels at NP, per AASHTO/UFC — correct modern practice); toe `Nt·σv'` / `9·cu`; fill +
GW-drawdown Δσ; **UFC 3-220-20 Eq 6-53** clay settlement (NC/OC/OC→NC) and **Eq 6-54** sand
constrained-modulus elastic settlement `H(1+ν)(1−2ν)/((1−ν)Es)·Δσ`. 53 tests pass.

### DD-1 [Low] — NC tolerance band (same as SET-3)
`_settlement_clay` treats `|σp−σv0|/σv0 < 0.05` as NC (uses C_ec over the full increment), creating a
small step vs slightly-OC soils — consistent with the settlement module; document.

### BC-9 [Low, calc_steps] — displayed equations don't match the φ=0 / two-layer computed values
- `_shape_factor_equation`/`_depth_factor_equation` always print the φ>0 Vesic forms; for φ=0 the
  code actually uses `sc=1+0.2(B/L)`, `dc=1+0.4k`, so the shown equation ≠ the shown value.
- `_depth_factor_equation` (vesic) omits the `d_c` formula entirely.
- Two-layer term-breakdown table (calc_steps.py:316-330) shows **top-layer** term_* against the
  **blended** q_ult, so the % column won't sum to 100 and the "Total q_ult" row contradicts the
  parts (direct consequence of BC-1's result-object reuse).

---

## calc_package renderer  (clean)

**Verified correct:** clean pure-data model (InputItem/CalcStep/CheckItem/FigureData/TableData/
CalcSection); InputItem-batching preprocessor; Jinja2 HTML + LaTeX + PDF dispatch; lazy registry of
all 13 modules; figure→base64 embedding (self-contained HTML). 100 tests pass.

### CP-1 [Low] — `render_html` uses `autoescape=False`
Required so the HTML-entity equations render, but header/text fields (`project_name`, `engineer`,
`company`, layer descriptions) are injected unescaped → HTML-injection vector if any of those carry
untrusted markup. Low risk for a local tool; escape the non-equation text fields, or sanitize headers.

---

## calc_steps presentation sweep (cross-module)

**Method:** counted math recomputation in each `calc_steps.py` (display-only vs recompute) + read the
two recomputing files. **8 of 13** (axial_pile, drilled_shaft, lateral_pile, pile_group,
wave_equation, slope_stability, downdrag, + mostly settlement/seismic/ground_improvement) **only
format result fields** — the safe pattern, displayed value ≡ computed value. The 100-test
calc_package suite confirms all 13 render without error. Findings below are the divergence cases.

### CS-1 [Medium, retaining_walls calc_steps] — displayed Ka hardcoded to plain Rankine, diverges from the analysis
`retaining_walls/calc_steps.py` recomputes `Ka = tan²(45−φ/2)` (lines 281/625/1002/1264) and rebuilds
Pa from it (309-311), but the analysis (`cantilever.py`) uses `rankine_Ka_sloped` for sloped backfill
and `coulomb_Ka(δ=⅔φ)` for the coulomb method. So for **sloped backfill or `pressure_method="coulomb"`**,
the rendered package shows a Ka (and Pa) that **don't match the FOS the analysis reported** — an
internally inconsistent calc package. Read `result`/analysis Ka instead of recomputing.

### CS-2 [Low, sheet_pile calc_steps] — Rankine-only display recomputation
`sheet_pile/calc_steps.py` likewise recomputes Rankine Ka/Kp for display; currently consistent only
because `pressure_method="coulomb"` is a no-op (SP-2). If SP-2 is fixed so coulomb truly differs, this
display will diverge. (The calc_steps z_crack here uses the *correct* `(2c/√Ka−q)/γ`, unlike the
sheet_pile docstring — good.)

### CS-3 [Low] — recommend a value-consistency pass on the recomputing calc_steps
General rule for the layer: calc_steps should **render `result.*` fields**, never re-derive
engineering values (re-derivation is where display drifts from computed). Only retaining_walls and
sheet_pile violate this; the rest are clean.

---

## FIX LOG — 2026-06-08 (branch `calc-qc-fixes`, off origin/master)

All 10 prioritized findings fixed in the `calc-qc-fixes` worktree, each with a regression test;
every touched module suite passes locally (funhouse remains the live validation surface).

| ID | Sev | Module | Fix |
|----|-----|--------|-----|
| BC-2 | High | bearing_capacity | Vesic inclination factors no longer silently vanish at `vertical_load=0`; fall back to angle-based Meyerhof i-factors + `UserWarning`. |
| BC-1 | High | bearing_capacity | `_compute_two_layer` replaced with the 2:1 load-spread method (was linear interp mislabeled "Meyerhof & Hanna"); corrected the backwards weak-over-strong trend; term breakdown scaled to sum to `q_ult`; honest references. |
| WE-1 | High | wave_equation | `permanent_set = max(D_max,toe − Q_toe, 0)` (tracks peak toe penetration, subtracts toe quake). Verified numerically (≈2.5 mm; blow count 52→60 at hard driving). |
| AP-1 | Med | axial_pile | `beta_from_phi` Fellenius (NC, OCR ignored + warn) vs Burland (`(1−sinφ)√OCR·tanφ`) — no longer byte-identical. |
| GI-1 | Med | ground_improvement | `equivalent_drain_diameter = 2(w+t)/π` (Hansbo; was half). |
| SP-1 | Med | sheet_pile | `coulomb_Kp` numerator `sin²(α−φ)` (was wrong sign; only affected inclined walls). |
| SP-2 | Med | sheet_pile | Per-layer `wall_friction_deg` threaded into the Coulomb coefficients; `pressure_method="coulomb"` no longer a Rankine no-op (default δ=0 preserves prior behavior). |
| CS-1 | Med | retaining_walls | calc_steps `_Ka_value` + displayed Ka mirror the analysis (sloped Rankine when the backfill slopes), so the package matches the reported FOS. |
| SET-1 | Med | settlement | Schmertmann `Izp` uses σ'vp at the peak-influence depth via new optional `gamma_soil` (was base overburden `q0`). |
| SET-2 | Med | settlement | Removed the non-canonical `C3 = 1.03−0.03·L/B` shape factor (double-counted shape; shape is in the Iz diagram). |
| SET-3 | Med | settlement | Elastic method applies a shape-based influence factor `Iw` (Schleicher flexible-center closed form), overridable via `Iw_immediate`; was a flat 1.0. |
| LP-1 | Med | lateral_pile | `SandReese.get_p` is now a genuine 3-part curve (linear → 1/3-power softening parabola → plateau at `yu=3b/80`); was bilinear. (Full Reese m–u B-factor charts not reproduced; `SandAPI` remains the smooth equivalent.) |
| PG-1 | Med | pile_group | Bounded fix: assembled the dropped `kxy` coupling; replaced the silent lateral-load fallback with DOF condensation + a clear `ValueError` when lateral/torsion cannot be resisted. Full battered force↔rotation coupling (kxz/kyz) + lateral back-calc documented as a remaining limitation (deferred). |

**Modeling choices made autonomously (flagged):**
- BC-1 → load-spread (DM-7/Bowles) instead of Meyerhof & Hanna punching (M&H needs the K_s punching chart, not reproduced from memory).
- SET-3 → Schleicher flexible-center `Iw` (rigid footings settle ~7–15% less; override available).
- LP-1 → 1/3-power softening parabola anchored at the ultimate point (chart-free simplification of the full Reese construction).
- PG-1 → bounded fix per owner decision; full CPGA-style coupled rewrite deferred (no local validation oracle).

**New / activated follow-ups:**
- **CS-2 now live:** with SP-2 fixed, `sheet_pile/calc_steps.py` (which recomputes Rankine Ka/Kp for display) will diverge from the analysis *only if a user sets `wall_friction_deg>0`*. Default δ=0 keeps them consistent. Small follow-up: have sheet_pile calc_steps consume the method's actual coefficients.
- **PG-1 full coupled 6-DOF** (kxz/kyz force↔rotation, lateral/shear back-calc) — needs a CPGA benchmark to validate.

Test deltas (new regression tests added): bearing_capacity 66→72, wave_equation 45→46,
axial_pile/ground_improvement +1/+1, sheet_pile 26→31, retaining_walls 70→73, settlement 39→44,
lateral_pile TestPYCurves +1, pile_group 72→74.
