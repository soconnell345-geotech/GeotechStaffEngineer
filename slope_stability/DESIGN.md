# slope_stability — Limit Equilibrium Slope Stability

## Purpose
Circular and noncircular slip surface analysis using three methods of slices
(Fellenius, Bishop, Spencer) with critical surface search and entry/exit
range constraints.

## References
- Fellenius (1927) — Ordinary Method of Slices
- Bishop (1955) — Geotechnique, Vol. 5, pp. 7-17
- Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
- Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
- FHWA GEC-7, Lazarte et al. (2003) — Soil nail walls (nails.py, currently disconnected)

## Files
- `geometry.py` — SlopeSoilLayer (elevation-based), SlopeGeometry (polyline surface/GWT)
- `slip_surface.py` — CircularSlipSurface (center + radius), PolylineSlipSurface (polyline points)
- `slices.py` — Slice dataclass, build_slices() with multi-layer weight/pore pressure
- `methods.py` — fellenius_fos(), bishop_fos(), spencer_fos() (circular + noncircular)
- `search.py` — optimize_radius(), grid_search(), search_noncircular(), entry/exit filtering
- `analysis.py` — analyze_slope(), search_critical_surface() orchestrators
- `results.py` — SliceData, SlopeStabilityResult, SearchResult
- `nails.py` — SoilNail, NailContribution (kept but disconnected from analysis pipeline)
- `tests/test_slope_stability.py` — 83 tests (core + entry/exit limits + noncircular search)
- `tests/test_duncan_verification.py` — 19 tests (Duncan §7.6-7.7 verification examples)
- `tests/test_nails.py` — 38 tests + 16 skipped (nail intersection/pullout/capacity)

## Public API
```python
# Circular analysis
analyze_slope(geom, xc, yc, radius, method="bishop", tol=1e-4, ...) -> SlopeStabilityResult

# Noncircular analysis (polyline slip surface)
analyze_slope(geom, slip_surface=PolylineSlipSurface(points), method="spencer", tol=1e-4, ...)

# Critical surface search — circular (grid search)
search_critical_surface(geom, x_range, y_range, nx, ny, method="bishop",
                        surface_type="circular", x_entry_range=None, x_exit_range=None,
                        tol=1e-4, ...) -> SearchResult

# Critical surface search — noncircular (random polyline)
search_critical_surface(geom, surface_type="noncircular",
                        x_entry_range=None, x_exit_range=None,
                        n_trials=500, n_points=5, seed=None,
                        tol=1e-4, ...) -> SearchResult
```

## Slip Surface Types

### Circular (CircularSlipSurface)
- Defined by center (xc, yc) and radius
- Entry/exit found by bisection against ground surface
- Works with all three methods (Fellenius, Bishop, Spencer)
- Spencer reduces to Bishop for circular (theta=0)

### Noncircular (PolylineSlipSurface)
- Defined by polyline points [(x1,z1), (x2,z2), ...]
- x-coordinates must be monotonically increasing
- Entry = first point, exit = last point (or intersection with ground)
- Spencer only (force-marching approach, no moment arm needed)
- `slip_elevation_at(x)` — linear interpolation between points
- `tangent_angle_at(x)` — angle of segment containing x
- Both classes share `is_circular` property for dispatch

## Search Algorithms

### Circular Grid Search
- `grid_search(geom, x_range, y_range, nx, ny, method, ...)` — evaluates nx*ny circle centers
- `optimize_radius(geom, xc, yc, method, ...)` — golden-section radius optimization at each center
- Optional `x_entry_range` / `x_exit_range` — reject surfaces whose entry/exit fall outside bounds
- FOS guard: surfaces with FOS < 0.05 are rejected as numerical artifacts

### Noncircular Random Search
- `search_noncircular(geom, n_trials, n_points, x_entry_range, x_exit_range, ...)` — random polylines
- Generates random polyline surfaces between entry and exit x-ranges
- Intermediate points at random depths below ground surface
- Uses Spencer's method for each trial
- Returns best (minimum FOS) surface found

## Critical Sign Conventions
- **Driving moment**: uses `W*(x_mid - xc)/R` + `abs(driving)`, NOT `W*sin(alpha)`
  directly, because `sin(alpha)` can be negative for L-to-R slopes
- **Seismic moment**: computed separately from gravity driving, then
  `driving = abs(gravity) + abs(seismic)` so seismic always increases
  driving regardless of slope direction (L-to-R vs R-to-L)
- **Spencer m_alpha**: `cos(alpha - theta) + sin(alpha - theta)*tan(phi')/F`
  (reduces to Bishop when theta=0)

## Key Technical Notes
- **Spencer = Bishop for circular surfaces**: sin(alpha) = (x-xc)/R identity
  makes moment and force driving identical, forcing theta=0
- **Spencer noncircular**: uses interslice force marching (no moment arm).
  FOS is the value that makes the last interslice force = 0.
- **For c'=0 (cohesionless)**: circular analysis overestimates FOS vs infinite
  slope solution tan(phi)/tan(beta) — this is a well-known limitation
- **Coordinates**: elevation-based (x, z where z=elevation), NOT depth-based
- **SlopeSoilLayer**: separate from geotech_common/SoilLayer (no adapter yet)
- **Undrained**: phi=0, cu used. Bishop = Fellenius when phi=0 (m_alpha = cos(alpha))
- **FOS convergence guard**: in `_compute_fos` (search.py), FOS < 0.05 is treated
  as non-convergence and returns FOS_MAX (999). Prevents negative/absurd FOS from
  polluting search results.
- **Noncircular degenerate-surface guard (SS-6, v5.3 B2)**: the random / weak-layer
  polyline generators can emit jagged zig-zag surfaces on which the rigorous GLE
  either fails to converge (the legacy Spencer fallback then returns a spurious
  low FOS ~0.05-0.2) or converges to a spurious low value — a search would wrongly
  report that as the critical surface (seen on ACADS 4). `_compute_fos` guards
  noncircular trials three ways: (1) a cheap geometric admissibility check rejects
  sliver / too-short-span / too-few-slice surfaces (`_noncircular_admissible`);
  (2) the rigorous FOS REQUIRES GLE convergence (`_rigorous_noncircular_fos`
  returns None otherwise, instead of falling back to the legacy value); and
  (3) a **low-FOS jaggedness gate** rejects any surface whose FOS is below
  `_LOW_FOS_JAGGED_GATE` (1.5) AND is non-concave / near-vertical (`_is_jagged`) —
  a low FOS on a jagged surface is a solver artifact, not a real critical surface.
  The gate is deliberately scoped to LOW-FOS surfaces so the generators keep
  exploring the many high-FOS jagged trials (which harmlessly lose the search).
  Circular surfaces are untouched. See `tests/test_search_robustness.py`.
  - **Span floor is problem-anchored, not model-wide** (v5.3 review fix): the
    admissibility span floor is anchored to the caller's entry/exit WINDOW extent
    (`_window_span`, half the window centre-to-centre distance) when a search
    supplies one, else to the local slope HEIGHT (0.5·H). The previous floor of
    15% of the FULL model width silently rejected legitimate LOCALIZED failures in
    a wide model (a long tailwater bench or wide right-of-way inflated the floor
    past a real slide length). The min-slice check is unchanged.
  - **Silent rejections are counted** (v5.3 review fix): each noncircular
    generator passes a `reject_stats` tally into `_compute_fos`; the per-reason
    counts (geometry / non-converged / jagged) land on
    `SearchResult.n_rejected_*` (and `to_dict()['n_rejected']`), and a search that
    rejects a MAJORITY of its trials emits a `warnings.warn` so an
    under-resolved / mis-windowed search is not silently reported as a clean
    result.
  - **Convergence-gate moment axis has a robust fallback** (v5.3 review fix):
    `_rigorous_noncircular_fos` first tries gle's internal least-squares axis
    (byte-identical to before) and, only if that fails to converge, retries with
    an explicit pinned `axis_point` (`_noncircular_axis_point`: span midpoint,
    half a span above the crest). The least-squares fit can land absurdly far
    away on a near-straight surface and spuriously fail; the second pass rescues
    such a valid surface without ever discarding a convergence the first pass
    already found (so no existing search result moves). A converged GLE FOS is
    axis-independent.
- **`infinite_slope_fos` (v5.3 B2c)**: closed-form planar/translational FOS for
  the shallow surface-parallel mechanism, `analysis.infinite_slope_fos(slope_angle,
  phi, gamma, c=0, depth=1, water_condition='dry'|'seepage_parallel'|'ru', ...)`
  → `InfiniteSlopeResult`. FOS = [c' + (γz cos²β − u)tanφ']/(γz sinβ cosβ);
  cohesionless dry reduces to tan φ'/tan β (depth-free), seepage-parallel at the
  surface to (γ'/γ)·that. Validated vs Slide2 #79 (1.44) / #81 (1.15). This is
  the exact answer for the c'=0 circular-overestimate limitation noted above.
- **Rapid drawdown (v5.3 B2a)** — `rapid_drawdown.rapid_drawdown_fos(geom, from_el,
  to_el, xc/yc/radius, method='corps_2stage'|'duncan_3stage')` → `RapidDrawdownResult`
  (also `analysis.rapid_drawdown_fos`). Low-permeability layers carry the total-stress
  **R-envelope** (`R_c`, `R_phi`) alongside the effective `c_prime`/`phi`; `R_phi is
  None` = free-draining. The method is wired to the GLE engine by OVERRIDING each
  undrained slice's base strength to a fixed `c = τ_ff, φ = 0` and re-solving. Theory:
  * **Stage 1** — effective-stress ("S") analysis at the FULL pool gives the
    consolidation stresses on each base: σ'_fc = N'/l and τ_fc = S_mob/l (from the
    GLE per-slice effective normal / mobilized shear).
  * **Stage 2** — undrained strength τ_ff. Corps 2-stage: τ_ff = min(R-envelope,
    drained) at σ'_fc (the "combined" envelope, capping R-envelope over-strength at
    low σ'). Duncan 3-stage: linear interpolation with the consolidation stress ratio
    Kc = σ'_1c/σ'_3c between the Kc=1 (R / IC-U, lower) and Kc=Kf (drained, upper)
    envelopes: τ_ff = τ_R + (Kc−1)/(Kf−1)·(τ_drained − τ_R), capped at the drained
    (Kf) bound. Kc per slice is back-figured from σ'_fc, τ_fc and the base angle α
    assuming a vertical major principal consolidation stress; Kf = (1+sinφ')/(1−sinφ').
  * **Stage 3** (3-stage only) — where the post-drawdown DRAINED strength (low-pool
    effective stress) is less than the stage-2 undrained strength, it is substituted
    (Duncan, Wright & Brandon 2014, Ch. 9: third-stage drained strength on the
    drawn-down effective stresses).
  * The final FOS is a GLE/Spencer solve at the DRAWN-DOWN pool (external water load
    removed to the low level; undrained slices carry τ_ff as φ=0 total-stress
    strength). Validated vs Slide2 #95 (2-stage) / #96 (3-stage) — see RESULTS V-037/V-038.
  * **Stage-1 seepage option (`stage1_phreatic_points`)** — the DEFAULT stage-1 uses a
    flat full-pool phreatic (hydrostatic to the reservoir), the conservative
    no-through-seepage bound. A low-permeability embankment under an established full
    pool actually has a steady-state seepage field whose phreatic DECLINES through the
    dam, so the true σ'_fc (and mobilized τ_ff) are higher. Passing the flow-net /
    Casagrande phreatic line as `stage1_phreatic_points` (a `[(x,z),…]` surface that
    starts at the pool level on the upstream face, preserving the external reservoir
    load there) reproduces the steady-seepage condition Slide2/EM 1110-2-1902 use.
    Default `None` = unchanged. For Slide2 #95 this recovers the Corps 2-stage FOS from
    the conservative 1.21 to 1.34 vs the published 1.347 (~0.6%).
  * **Duncan 3-stage for c'=0 soils — stage-3 normal (v5.4 E2).** For #95/#96 the
    default 3-stage lands ~12% below the published 1.443. The V5.3 note blamed the Kc
    interpolation; the E2 re-investigation found the Kc interpolation is sound (it yields
    ~1.45 on its own) and the deficit is in the STAGE-3 drained substitution: the
    drawn-down effective normal σ'_post was estimated with the Fellenius `W·cosα/l − u`
    term, which under-predicts N' (it neglects interslice forces) and so substitutes a
    too-low drained strength on 17/50 slices. The optional
    `stage3_effective_normal='gle'` uses the RIGOROUS GLE effective normal from a
    drawn-down drained solve (the same `base_normal_eff` basis stage 1 uses) — 9
    physically-genuine substitutions instead of 17 — lifting #96 to 1.306 (flat) /
    1.370 (seepage), closing most of the residual (~5% left, within the
    representative-flow-net + LE-N'-at-FOS sensitivity that shows as the +0.6% Corps
    residual). **Default is `'fellenius'` (byte-identical to before);** the refinement
    is opt-in. Only the 3-stage has a stage 3, so the 2-stage is untouched.
- **Rapid-drawdown critical-surface search (v5.4 E1)** — `rapid_drawdown.
  search_rapid_drawdown(geom, from_el, to_el, method=…, surface_type='circular'|
  'entry_exit'|'noncircular'|'noncircular_de', …)` → `RapidDrawdownSearchResult`
  (also `analysis.search_rapid_drawdown`; exported at the package level). It finds the
  MINIMUM-FOS surface with the drawdown strength substituted per trial, for both stage
  methods and both circular/noncircular surfaces. It is a THIN composition: `search.py`
  gained an optional `fos_fn(geom, slip)->FOS` hook on `_compute_fos` and the search
  entry points (`grid_search`/`optimize_radius`/`search_entry_exit`/`search_noncircular`/
  `search_de`), default `None` = byte-identical; the wrapper passes a drawdown-FOS
  closure so the grid/entry-exit/random/DE loops, radius optimisation, entry/exit
  filtering, and noncircular guards are all SHARED — no search internals are duplicated.
  The result carries the ordinary rich `SearchResult` plus the stage-level
  `RapidDrawdownResult` recomputed on the winning surface. **Robustness gate:** the
  drawdown closure REJECTS a trial surface whose stage-1 or stage-3 GLE did not converge,
  whose drawn-down solve retained too few slices, or whose stage-1 (full-pool) FOS is
  implausibly low (<0.5) — a large near-flat circle self-clips to a handful of slices
  whose GLE "converges" to a spurious low FOS (~0.18) that would otherwise win the
  search. Validated vs Slide2 #98 Walter Bouldin search minima (Corps 0.931 / DWW 1.039)
  on the RECOVERED geometry (~10% low, geometry-limited) and by exact wrapper mechanics
  on #95/#96 — see RESULTS V-041. Note: each trial is several LE solves + two geometry
  deep-copies, so a circular grid search is heavy; prefer a modest `nx·ny` or
  `entry_exit` and bounded entry/exit windows (an O(n_surfaces) hotspot, not optimised).
- **Newmark seismic sliding block (v5.3 B2b)** — `newmark.py`, three functions
  (exported + adapter `yield_acceleration` / `newmark_displacement` /
  `newmark_jibson2007`):
  * **`yield_acceleration`** — the yield seismic coefficient ky (Newmark critical
    acceleration ay = ky·g) for a SPECIFIED surface, by bisection on the module's
    pseudo-static FOS (`analyze_slope` with a trial `kh`; the seismic_force = kh·W path
    validated by Loukidis #62/#63 = V-033). FOS(kh) is monotone decreasing, so bisection
    on FOS=1 gives ky directly. If the static FOS ≤ 1, ky=0. The caller's `geom.kh` is
    not mutated (shallow copy).
  * **`newmark_displacement(ky, accel, dt, polarity=…)`** — rigid-block double
    integration (Newmark 1965), no upslope rebound (relative velocity clamped ≥ 0), so
    displacement is monotonic. Two polarity conventions:
      - `polarity="downslope"` (**default**, standard Newmark 1965 / Jibson 2007): the
        SIGNED record is integrated and the block is driven only when the ground
        acceleration exceeds ay in the destabilizing (downslope-positive) direction; the
        reverse polarity decelerates it. Only the destabilizing polarity of a symmetric
        record contributes → about HALF the rectified displacement.
      - `polarity="rectified"`: the ABSOLUTE record is integrated so both polarities drive
        the single downslope block — a conservative, orientation-independent option (use
        when the record's sign relative to the slope's downslope direction is unknown).
    Trapezoidal on the piecewise-linear relative velocity — EXACT for a rectangular pulse
    (`D = ap(ap−ay)T²/(2ay)`), the integrator's validation (identical under both
    polarities for a one-sided pulse).
  * **`newmark_jibson2007(ky, amax)`** — Jibson (2007) Eq. 6 regression,
    `log10 D[cm] = 0.215 + log10[(1−ky/amax)^2.341·(ky/amax)^−1.438]`, σ=0.510 log10,
    a time-history-free cross-check (valid 0 < ky < amax).
  * Slide2 #104 (Tutorial-28 record) is the target problem; its acceleration record and
    geometry are not published, so the integrator is validated analytically and Jibson
    against its published equation — see RESULTS V-039.
- **Stabilizing piles (v5.3 B2d)** — `reinforcement.StabilizingPile` on
  `geom.stabilizing_piles`, resolved by `compute_reinforcement_forces` like the other
  reinforcement (active convention: the resisting force reduces the driving moment) and
  applied where the vertical pile crosses the slip surface. Two ways to set the row's
  resistance:
  * **Specified shear** — `shear_capacity` (kN per pile); the per-metre force is
    `shear_capacity/spacing` (Slide2 #54 / Yamagami 2000). Direction 'horizontal'
    (default) or 'normal' (perpendicular to the slip surface).
  * **Ito & Matsui (1975)** — `ito_matsui=True` with the pile `diameter` (clear spacing
    D2 = spacing − diameter). `ito_matsui_pressure(c,phi,gamma,z,D1,D2)` is the ORIGINAL
    1975 plastic-deformation lateral force per unit depth on one pile (their Eq. 13),
    `p(z) = c·D1·[(A−2√N_phi·tanphi−1)/(N_phi·tanphi) + Fc] − c·(D1·Fc − 2·D2/√N_phi)
    + (gamma·z/N_phi)·(D1·A − D2)` with N_phi=tan²(45+phi/2),
    `A = (D1/D2)^(√N_phi·tanphi+N_phi−1)·exp[(D1−D2)/D2·N_phi·tanphi·tan(pi/8+phi/4)]`,
    `Fc=(2tanphi+2√N_phi+1/√N_phi)/(√N_phi·tanphi+N_phi−1)`. Note the exp argument is
    `tan(pi/8+phi/4)` and the first-term coefficient is `1/(N_phi·tanphi)` (per the
    printed equation; the D1·Fc contributions cancel). The phi=0 cohesive limit
    (Eq. 23) is `p(z)=c·{D1·(3 ln(D1/D2)+(D1−D2)/D2·tan(pi/8)) − 2(D1−D2)}+gamma·z·(D1−D2)`.
    Hand-checked: c=10/phi=20/gamma=18/z=5/D1=2/D2=1.5 → 105.079; c=25/phi=0/gamma=18/z=4/
    D1=2/D2=1 → 146.683. `ito_matsui_lateral_force` integrates p(z) from the pile head to
    the slip surface (closed form, linear in gamma·z); divided by the spacing for the
    per-metre force.
  * Validation: #54 (specified shear) → CONVENTION, RESULTS V-040 (no-pile +1.1%,
    with-pile +2.5%; residual = active-vs-passive support convention + figure-read pile
    location). The Ito-Matsui FORMULA is unit-tested for the #106 spacing trend (its
    cross-section is not in the manual).
  * **Support convention — active/passive (v5.4 E6).** `StabilizingPile.
    support_convention` = `'active'` (**default**) applies the unfactored pile force as
    a REDUCTION of the driving moment (Slide2 "Method A"); `'passive'` adds it to the
    RESISTING side instead (Slide2 "Method B"). Active gives a larger FOS gain for the
    same capacity (it shrinks the denominator), which slightly over-predicts vs a
    passive pile; passive is the more conservative treatment. Only the circular moment
    methods (Fellenius/Bishop) honour the convention — the rigorous GLE (Spencer/M-P)
    applies the pile force as an external equilibrium force either way. On #54 the
    passive option narrows the with-pile Bishop residual from +2.5% (active) to +0.8%
    vs the published 1.193 (VALIDATION.md B10 / RESULTS V-040-E6). Default 'active' is
    byte-identical. Implemented via `reinforcement.moment_resistance` /
    `horizontal_resistance` (passive forces) alongside the existing `*_reduction`
    (active forces), split on `ReinforcementForce.passive`.
- **Tension crack — side + model (v5.4 E4)**: `geom.tension_crack_depth` opens a
  vertical crack at the CREST end of the slip surface; `tension_crack_water_depth`
  fills it (hydrostatic thrust `0.5·γw·z_w²` at `z_w/3` above the crack base,
  treated downstream as an always-driving magnitude). Two selectors, both
  DEFAULT-PRESERVING:
  * `tension_crack_side` = `'entry'` (default, low-x) or `'exit'` (high-x) — the
    crack forms on whichever side the crest is on. The exit-side option removes
    the old need to MIRROR the slope so the crest landed on the entry side (the
    exit-side crack on the un-mirrored slope is machine-precision identical to the
    entry-side crack on the mirror, by symmetry).
  * `tension_crack_model` = `'strength'` (default) keeps the cracked wedge as
    zero-shear-strength DRIVING soil; `'truncation'` REMOVES it from the sliding
    mass (the mass ends at the vertical crack face), matching Slide2/UTEXAS.
    Truncation is LESS conservative (removes the driving wedge). On Slide2 #2 /
    ACADS 1(b) the truncation model reproduces the published water-crack FOS
    (Bishop 1.596/Spencer 1.592/GLE 1.592) to <0.1%, resolving the V-026
    strength-model conservatism (documented there as a CONVENTION). See
    VALIDATION.md B9 / RESULTS V-026.
- **Empty-space slices**: slices where the slip surface is below all soil layers
  are skipped (zero weight, no contribution)
- **Nails disconnected**: `nails.py` is importable but not called from the analysis
  pipeline. Nail fields removed from `SlopeStabilityResult`. Can be re-enabled later.
- **Spencer / Morgenstern-Price are APPROXIMATE (SS-1)**: both use a shifted
  m_alpha = cos(alpha−theta) + sin(alpha−theta)·tanφ/F formulation (with
  theta = const for Spencer, atan(λ·f(x)) for M-P) and a force-equilibrium
  driving term W·(sinα + cosα·tanθ), iterated until FOS_moment = FOS_force.
  This is an engineering approximation, NOT the textbook-exact GLE
  interslice-force recursion. Validated against the Duncan, Wright & Brandon
  examples (below) — treat results as Spencer/M-P-class accuracy, not exact.
- **Bishop numerator clamp (SS-2, v5.1)**: the frictional term uses
  max(W − u·b, 0)·tanφ so artesian/perched pore pressure (u·b > W) cannot
  contribute negative resistance; consistent with the Fellenius
  effective-normal clamp. Spencer/M-P share the (W − u·b) form but are left
  unclamped to preserve the Duncan-validated behavior (noted for follow-up).
- **Pore pressure model (SS-3)**: `_pore_pressure_at_base` uses the full
  hydrostatic head to the GWT surface (no cos²β seepage correction for
  inclined phreatic surfaces) — slightly conservative (overestimates u) on
  steep water tables. The layer `ru` coefficient is the alternative.
- **Pore-pressure GRID / TIN input (v5.4 E3)**: `SlopeGeometry.pore_pressure_points`
  is an optional list of scattered `(x, z, u)` triples (kPa) — a flow-net / TIN
  sampling of the pore-pressure field. When set, the base pore pressure at each
  slice is INTERPOLATED from it by `geometry.build_pore_pressure_interpolator`
  (scipy `LinearNDInterpolator` = linear on the Delaunay triangulation, with a
  `NearestNDInterpolator` fallback outside the convex hull and for degenerate
  point sets < 3 points / collinear; suction clamped to u ≥ 0). It OVERRIDES the
  piezometric-line (`gwt_points`) and per-layer `ru` base pore pressure and is
  wired through the search path (build_slices builds the interpolator once per
  call). The ponded-water buttress + submerged weight still come from
  `gwt_points`, so a reservoir over a flow-net field sets BOTH. Default `None` =
  unchanged. A grid encoding a hydrostatic field reproduces the `gwt_points`
  base pressure (hence FOS) to machine precision; TIN-linear is exact for a
  linear field. Enables Slide2 #10 / ACADS 5 (pore-pressure-grid, previously
  N/A-scope) — see VALIDATION.md B8 / RESULTS V-029. (scipy imported lazily, as
  in `search.search_de`.)
- **Multiple surcharge zones (v5.4 E8)**: `SlopeGeometry.surcharges` is an optional
  list of `(pressure_kPa, x_start, x_end)` zones for problems with several distinct
  loaded areas (a bench load + a crest load, etc.). `surcharge_at(x)` SUMS every zone
  covering x on top of the single `surcharge`/`surcharge_x_range` pair (which is
  unchanged), so slices and the search pick them up with no change (both route through
  `surcharge_at`). Default `None` = single-surcharge behaviour byte-identical. A
  linearly-varying (trapezoidal) load is represented as its mean uniform pressure or as
  several thin zones. Exercised on Slide2 #9 / ACADS 4 (bench + crest loads) — see
  VALIDATION.md B11 / RESULTS V-028.
- **Probabilistic FOS — correlated depth-varying su law (v5.4 F1)**: `probabilistic.py`
  normally varies each scalar parameter (phi/c_prime/cu/gamma/gamma_sat) INDEPENDENTLY
  per targeted layer. That over-counts a strength profile whose uncertainty is a single
  shared su-GRADIENT spread across many thin sub-layers. The additive `linear_su` law
  (a `variables` entry carrying a `'law'` key) treats `su(z) = a + b*(datum_z - z)` as
  ONE correlated `(a, b)` random-variable pair, applied COHERENTLY to every targeted
  layer's cu (each layer's mid/top/bottom elevation, `z_ref`; floored at `su_min`; layer
  forced undrained). FOSM adds the Taylor cross-term
  `Var = (dFa/2)^2 + (dFb/2)^2 + 2*rho_ab*(dFa/2)*(dFb/2)` (a std-0 component, e.g. a
  fixed intercept, drops out); Monte Carlo samples `(a, b)` from a bivariate normal via
  Cholesky (`b = b_m + b_s*zb`, `a = a_m + a_s*(rho*zb + sqrt(1-rho^2)*za)`). The scalar
  path is byte-identical when no law is present (variables split; scalar RNG stream drawn
  first). Also NEW: **`gamma_sat` is now a valid scalar variable** — for a submerged slope
  the operative unit weight is `gamma_sat` (below the water table), and its uncertainty is
  buoyancy-amplified (the buoyant weight `gamma_sat - gamma_w` carries a larger relative
  spread than `gamma_sat` itself), so varying the dry `gamma` alone is inert. Closes the
  Duncan (2000) LASH input-COV FOSM gap — see VALIDATION / RESULTS V-030 and
  `validation_examples/test_published_v030_fosm_slope.py`.

## Duncan Verification Examples (test_duncan_verification.py)

| Example | Description | Published FOS | Method | Tolerance |
|---------|-------------|--------------|--------|-----------|
| 1 | Saturated clay, undrained (cu=600psf, phi=0) | ~1.0 Fellenius, ~1.08 Bishop | Fellenius, Bishop | ±10% |
| 2 | Cohesionless slope (c=0, phi=40, 2H:1V) | ~1.17 | Bishop, Spencer | ±15% |
| 3 | Cohesionless with seismic (kh=0.15) | < Ex 2 | Spencer | directional |
| 4 | Two-layer (sand over clay) | ~1.4 Bishop, ~1.3 Fellenius | Bishop, Fellenius | ±20% |
| 6 | Submerged slope with GWT | varies | Bishop | ±15% |

All dimensions converted from imperial (ft/psf/pcf) to SI (m/kPa/kN/m³).
Example 5 skipped (curved Mohr-Coulomb envelope not supported).
