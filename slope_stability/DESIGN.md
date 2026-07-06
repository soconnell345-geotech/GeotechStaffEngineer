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
  Circular surfaces are untouched. See `tests/test_b2_round1.py`.
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
  * **Known limitation — Duncan 3-stage for c'=0 soils.** When the drained (Kc=Kf)
    envelope falls BELOW the R (Kc=1) envelope at the operative σ'_fc (a c'=0 soil with a
    cohesive R-envelope, e.g. #95/#96), the Kc interpolation under-captures the
    anisotropic-consolidation strength GAIN that lifts the published 3-stage above the
    2-stage; the module's 3-stage tracks the 2-stage (ordering preserved) but lands
    ~12% below the published 3-stage even under the steady-seepage stage-1 that
    validates the 2-stage. Follow-up: a τ_ff-vs-σ'_fc envelope-crossing treatment per
    Duncan-Wright-Wong (1990). Not geometry- or seepage-related.
- **Newmark seismic sliding block (v5.3 B2b)** — `newmark.py`, three functions
  (exported + adapter `yield_acceleration` / `newmark_displacement` /
  `newmark_jibson2007`):
  * **`yield_acceleration`** — the yield seismic coefficient ky (Newmark critical
    acceleration ay = ky·g) for a SPECIFIED surface, by bisection on the module's
    pseudo-static FOS (`analyze_slope` with a trial `kh`; the seismic_force = kh·W path
    validated by Loukidis #62/#63 = V-033). FOS(kh) is monotone decreasing, so bisection
    on FOS=1 gives ky directly. If the static FOS ≤ 1, ky=0. The caller's `geom.kh` is
    not mutated (shallow copy).
  * **`newmark_displacement(ky, accel, dt)`** — rigid-block double integration
    (Newmark 1965). Downslope-only: the block slides only while |a_ground| > ay and the
    relative velocity is positive (clamped ≥ 0, no upslope rebound), so displacement is
    monotonic; the absolute value of the record is used so both polarities of a
    symmetric record drive the single downslope block (conservative, orientation-
    independent). Trapezoidal on the piecewise-linear relative velocity — EXACT for a
    rectangular pulse (`D = ap(ap−ay)T²/(2ay)`), which is the integrator's validation.
  * **`newmark_jibson2007(ky, amax)`** — Jibson (2007) Eq. 6 regression,
    `log10 D[cm] = 0.215 + log10[(1−ky/amax)^2.341·(ky/amax)^−1.438]`, σ=0.510 log10,
    a time-history-free cross-check (valid 0 < ky < amax).
  * Slide2 #104 (Tutorial-28 record) is the target problem; its acceleration record and
    geometry are not published, so the integrator is validated analytically and Jibson
    against its published equation — see RESULTS V-039.
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
