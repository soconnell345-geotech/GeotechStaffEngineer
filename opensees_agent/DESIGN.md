# OpenSees Agent — Design Notes

## Purpose

Provides high-level geotechnical analysis wrappers around OpenSeesPy, the
Python interface to the OpenSees finite element framework.  Each wrapper
builds an entire FE model internally, runs the analysis, and returns a
results dataclass — the LLM agent never touches raw OpenSees commands.

## Architecture

```
opensees_agent/
  opensees_utils.py    -- import guard, fresh_model(), response spectrum
  ground_motions.py    -- built-in earthquake records
  results.py           -- PM4SandDSSResult, SiteResponseResult
  pm4sand_dss.py       -- analyze_pm4sand_dss()
  site_response.py     -- analyze_site_response()   [Tier 2]
```

## Key Design Decisions

### 1. openseespy is Optional
All metadata, result classes, ground motion utilities, and Tier 1 tests
work without openseespy installed.  Only actual FE analyses require it.
The `has_opensees()` function checks availability at runtime.

### 2. Global State Management
OpenSees uses a global C++ model — there is no "model object."
Every analysis wrapper:
- Calls `ops.wipe()` at start via `fresh_model()`
- Wraps everything in `try/finally` with `ops.wipe()` in finally
- Extracts all results before wiping
- Cannot run analyses in parallel (serial only)

### 3. Sign Conventions
- **Stress**: OpenSees uses negative = compression.  All results are
  converted to positive = compression for consistency with geotechnical
  convention.
- **Displacement**: Positive = direction of applied load.
- **Depth**: Positive downward from ground surface.

### 4. Units
All inputs and outputs in SI: meters, kPa, kN, kN-m, seconds, degrees.
OpenSees internally uses the same units (no conversion needed).

## Analysis Methods

### PM4Sand Undrained Cyclic DSS
- Single SSPquadUP element with PM4Sand nDMaterial
- Consolidation under σ'v, then stress-controlled cyclic shear
- Monitors excess pore pressure ratio (ru) for liquefaction triggering
- Reference: Boulanger & Ziotopoulou (2017)
- Key outputs: n_cycles_to_liq, max_ru, stress-strain loops, stress path

> Note: a redundant OpenSees BNWF lateral-pile pathway was retired in the
> Phase 2a consolidation. Lateral pile analysis is provided by the native
> `lateral_pile` module (COM624P-style FD solver with 8 p-y models).

### 1D Effective-Stress Site Response
Follows the OpenSees example "Effective Stress Site Response Analysis of a
Layered Soil Column" (McGann & Arduino; OpenSeesPy docs 14.4.2,
`freeFieldEffective.py`).
- 1D soil column with SSPquadUP elements (4-node u-p formulation)
- Sand: `PressureDependMultiYield02`, arguments in the documented order
  (OpenSees wiki / OpenSeesPy docs). Gr = rho*Vs^2 at refPress = the mean
  effective stress at mid-layer (weight of all layers above, less
  hydrostatic pore pressure), pressDependCoe 0.5, peakShearStra 0.1, 20
  yield surfaces; the calibration constants are the wiki's table of
  suggested parameters, Dr = 40 % column (`PDMY02_DR40`): PTAng 26 (capped
  at phi), contrac1/3 0.067/0.23, dilat1/3 0.06/0.27, e 0.77; contrac2 5,
  dilat2 3, liquefac1/2 1/0, cs1-3 0.9/0.02/0.7, pa 101 (command defaults).
  Layer keys `PTAng`, `e_init`, `n_surf` override.
- Clay: `PressureIndependMultiYield` (phi = 0, total stress,
  near-incompressible bulk modulus capped at 100*Gmax). cohesi =
  sqrt(3)/2 * su: von Mises peaks at 2c/sqrt(3) in simple shear, so the
  column's simple-shear strength is su (wiki: c = sqrt(3)*sigma_m/2;
  checked on one SSPquad element, peak tau = 1.1547c). su is the
  simple-shear (DSS) strength.
- Gravity: element body forces (b1, b2) = (0, -g) — the components of
  gravity, not unit weights (SSPquadUP note 3) — which also gives the pore
  fluid its weight, so pore pressure settles hydrostatic and effective
  stress buoyant. Checked: sigma'v after gravity within 0.05 % of the hand
  calculation on a sand/clay/sand profile.
- Water table: nodes at or above gwt_depth have pp fixed at 0, so the water
  table sits at the deepest node at or above gwt_depth (exact on a node).
- Analysis phases: (1) gravity elastic 10 steps of 500 s, (2) elastoplastic
  40 steps of 500 s (skipped when `linear_elastic=True`), (3) clock set back
  to 0 (`setTime(0.0)`), (4) dynamic Newmark (0.5, 0.25) with Rayleigh
  damping
- Lysmer-Kuhlemeyer dashpot at base: `Viscous` uniaxial material with
  c = rho_bedrock * Vs_bedrock * area (zeroLength element)
- Loading: the motion is a ROCK OUTCROP motion; force = c_dashpot *
  v_outcrop (Joyner & Chen 1975: F = 2 c v_incident, and the outcrop motion
  is twice the incident wave). Velocity by trapezoid, at rest at t = 0;
  record index k is t = k*dt.
- Mesh: layers reversed bottom-to-top, n_elem_per_layer elements per layer,
  left/right node columns with equalDOF periodic BCs
- Rayleigh damping: f1 = Vs_avg/(4*H) (the site frequency), f2 = 5*f1
- Convergence fallback: KrylovNewton → ModifiedNewton → substeps of dt/10
  to the same target time; if all fail, RuntimeError naming the time (a
  failed step is never skipped silently). A failed gravity step is also a
  RuntimeError.
- Time limit: the model runs in a child Python process (`time_limit_s`,
  default 600 s; None = in process). openseespy holds the GIL inside
  `ops.analyze`, so a stuck step would freeze every thread of the caller
  (an app server, a test run); the child is killed at the limit with a
  RuntimeError. The child gets the parent's sys.path; errors come back by
  type (ValueError, ImportError, else RuntimeError).
- `linear_elastic=True`: materials stay in stage 0 (G = rho*Vs^2), a check
  of the column, not a design analysis.
- Output: surface accel time history, response spectra, depth profiles of
  max accel, max shear strain, max pore pressure ratio (ru = 1 -
  sigma'v/sigma'v0; 0 in elements with no pore water) and the initial
  vertical effective stress; `analysis_type`, `site_period_s` (4H/Vs).
- Reference: Lysmer & Kuhlemeyer (1969); Joyner & Chen (1975); Yang,
  Elgamal & Parra (2003); Kramer (1996) Sec. 7.2

#### GM-77 (2026-10-14): what was wrong until then
1. The gravity weight was nodal loads on a `Linear` time series (factor =
   time) run with 500 s steps, so the column carried 10,000 times its
   weight by t = 10,000 s (sigma'v 3.3e6 kPa at 17.5 m in sand).
2. The clock stayed at 10,000 s, past the end of the motion's force
   history, so no motion was applied: 8.5e-7 g at the surface of uniform
   sand from a 0.30 g input; 6.7e-9 g on clay.
3. PDMY02 was called without PTAng, so every later argument sat one slot
   off (noYieldSurf read 0.27 -> "0 surfaces, will use 10", void ratio and
   critical-state constants 0).
4. Sand over clay never finished its first elastoplastic gravity step (an
   endless loop inside OpenSees, holding the GIL: nothing else in the
   process ran). The cause was 1, the 10,000-times weight on a clay of
   su 40 kPa: with the weight right, even the shifted sand arguments
   finish; with the arguments fixed and the weight wrong it still hangs
   (or fails, depending on small changes). Fixed, it runs in about 2 s;
   72 runs over six profiles (sand over clay, soft clay under sand, clay
   over sand, four layers, loose and dense sand), three water tables, 1
   and 3 elements per layer and 0.1 / 0.5 g all finished, the longest in
   4.7 s. The child-process limit covers any that would not.
5. The input force was 2 * c * v_outcrop, twice the outcrop motion.
6. su went in as cohesi, so the clay was 15 % strong in simple shear.
7. A node on the water table was left free, which put the water table one
   element too shallow.
8. Elements with no pore water reported ru up to 0.34 (dry loose sand at
   0.5 g): ru = 1 - sigma'v / sigma'v0 picks up the column's vertical
   stress swings (+/-15-25 % there). ru is now 0 where an element has no
   pore water; below the water table the same swings remain in ru
   (BACKLOG follow-up: take ru from the pore pressure instead).

## Validation (site response, GM-77)

Run in WSL (openseespy 3.8.0, Python 3.12); tests in
`tests/test_opensees_agent.py::TestSiteResponseBenchmarks`.

| Check | Benchmark | Result |
|---|---|---|
| Column with the rock's impedance, no damping, synthetic pulse | Outcrop motion reproduced, F = 1 | amplification 1.0006 (4 and 10 elements) |
| Uniform layer H 20 m, Vs 200, rho 1.9 on rock 760 / 2.4, xi 2 %, steady sine, 10 elements | Kramer (1996) Sec. 7.2 closed form with the Rayleigh xi(f) | 0.5 f1: 1.361 vs 1.376 (-1.1 %); f1 = 2.5 Hz: 4.178 vs 4.169 (+0.2 %); 2 f1: 0.989 vs 0.989; 3 f1: 3.488 vs 3.543 (-1.6 %) |
| Same column, synthetic pulse, linear elastic | pystrata linear elastic (frequency domain, constant 2 %) | PGA 0.749 vs 0.758 g (-1.3 %); Sa 0.1-1.0 s within 3.3 % |
| Sand/clay/sand, water at 5 m | Hand sigma'v | within 0.05 % |
| Clay H 15, Vs 150, su 60, synthetic pulse, nonlinear | pystrata EQL with the same hyperbolic backbone (gamma_ref = su/Gmax, Masing damping + 2 %) | amplification 0.01 g: 2.62 vs 2.88; 0.05 g: 2.79 vs 3.17; 0.10 g: 1.92 vs 2.58; 0.30 g: 0.97 vs 1.40 (linear elastic 2.81) |

Why nonlinear and EQL part as strain grows: EQL applies one
strain-compatible modulus and damping to the whole record and has no shear
strength; the nonlinear model follows the backbone cycle by cycle and its
shear stress is capped at su, so in strong shaking it passes less motion
up. The tests pin: within 20 % at 0.05 g (max strain about 0.1 %),
nonlinear below EQL at 0.30 g.

Not a benchmark, recorded for calibration: dry sand (phi 33, H 20 m,
Vs 200) at 0.30 g amplifies 1.8 in the nonlinear model against 0.94-0.97 in
EQL with Darendeli (PI 0) or Menq curves — the PDMY02 default backbone
(hyperbolic, peakShearStra 0.1) is far stiffer at 0.1-0.3 % strain than
those laboratory curves; at 0.05 g the two are within 12-19 %.

## Ground Motions

Built-in synthetic motions are provided for testing.  Custom motions can
be provided as acceleration arrays.  The `ground_motions.py` module
includes pack/unpack utilities for embedding real records as compressed
base64 numpy arrays.

## Testing Strategy

- **Tier 1** (no openseespy): Input validation, result dataclasses,
  ground motions, utilities, Foundry agent metadata, site response
  validation.
  Since GM-77 also the site response column's OpenSees commands, recorded
  by a fake `ops` (material arguments, body forces, clock reset, input
  force, water table, ru), so the Windows gate sees them.
- **Tier 2** (requires openseespy): Integration tests with actual FE
  analyses and the benchmarks above.  Skipped via `@pytest.mark.skipif`
  when not installed — and on Windows, where Smart App Control blocks the
  openseespy DLL, so run them in WSL or rely on CI (BACKLOG INF-13). Every
  site response run in the tests has a time limit, so a stuck step fails
  instead of hanging CI.
