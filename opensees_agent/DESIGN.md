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
- Water table: nodes at or above gwt_depth have pp fixed at 0, and the
  mesh has a node at gwt_depth (GM-79): a layer it falls inside is meshed
  in two parts, the layer's element count shared in proportion to their
  thicknesses (at least one each; a one-element layer gets two), and
  `choices` gives the split. Until GM-79 the water table sat at the
  deepest node at or above gwt_depth, up to an element too shallow.
- Analysis phases: (1) gravity elastic 10 steps of 500 s, (2) elastoplastic
  40 steps of 500 s (skipped when `linear_elastic=True`), both with every
  element's permeability at 1 m/s, (3) each element's own permeability
  (sand 1e-4, clay 1e-7 m/s) set through element parameters (`parameter
  ... 'element' tag 'hPerm' / 'vPerm'`, `updateParameter`), as the
  effective stress example does (GM-85), (4) clock set back to 0
  (`setTime(0.0)`), (5) dynamic Newmark (0.5, 0.25) with Rayleigh damping
- Lysmer-Kuhlemeyer dashpot at base: `Viscous` uniaxial material with
  c = rho_bedrock * Vs_bedrock * area (zeroLength element)
- Loading: the motion is a ROCK OUTCROP motion; force = c_dashpot *
  v_outcrop (Joyner & Chen 1975: F = 2 c v_incident, and the outcrop motion
  is twice the incident wave). Velocity by trapezoid, at rest at t = 0;
  record index k is t = k*dt.
- Mesh: layers reversed bottom-to-top, n_elem_per_layer elements per layer,
  left/right node columns with equalDOF periodic BCs
- Rayleigh damping: f1 = Vs_avg/(4*H) (the site frequency), f2 = 5*f1
- Dynamic solver (GM-84): Newton at NormDispIncr 1e-3 m, 35 iterations
  (the total stress example's algorithm, the examples' test); a step
  Newton cannot take goes to KrylovNewton, then ModifiedNewton, each at
  NormDispIncr 1e-5 m, 50 iterations, and Newton is restored for the next
  step; a step none of them takes goes in 10 substeps of dt/10 to the
  same target time, each through the same three; if a substep fails all
  three, RuntimeError naming the time (a failed step is never skipped
  silently). A failed gravity step is also a RuntimeError. Why the
  fallbacks get a tighter test: Newton converges quadratically, so once a
  correction is under 1 mm the step is in balance whatever the element
  size; KrylovNewton and modified Newton converge linearly, slowly while
  the soil yields, and the 1 mm test passed them out of balance on small
  elements (defect A below).
- Time limit: the model runs in a child Python process (`time_limit_s`,
  default 600 s; None = in process). openseespy holds the GIL inside
  `ops.analyze`, so a stuck step would freeze every thread of the caller
  (an app server, a test run); the child is killed at the limit with a
  RuntimeError. The child gets the parent's sys.path; errors come back by
  type (ValueError, ImportError, else RuntimeError).
- `linear_elastic=True`: materials stay in stage 0 (G = rho*Vs^2), a check
  of the column, not a design analysis.
- Output: surface accel time history, response spectra, depth profiles of
  max accel, max shear strain, max pore pressure ratio and the initial
  vertical effective stress; `analysis_type`, `site_period_s` (4H/Vs).
  ru (GM-79) is the excess pore pressure over the initial vertical
  effective stress, the OpenSees effective stress example's r_u: at each
  element's centre, (u - u0) / sigma'v0, u the mean of its four nodes'
  pore pressure (the "velocity" of DOF 3, which the example records), u0
  the same after gravity, sigma'v0 the element's vertical effective stress
  then; 0 in elements with no pore water. Not capped at 1: once sand has
  liquefied the pore pressure also carries the column's swings in total
  vertical stress (1.016 in 20 m of loose sand at 0.30 g).
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
   pore water; below the water table the same swings remained in ru
   until GM-79 took ru from the pore pressure.

## Validation (site response, GM-77)

Run in WSL (openseespy 3.8.0, Python 3.12); tests in
`tests/test_opensees_agent.py::TestSiteResponseBenchmarks`.

| Check | Benchmark | Result |
|---|---|---|
| Column with the rock's impedance, no damping, synthetic pulse | Outcrop motion reproduced, F = 1 | amplification 1.0006 (4 and 10 elements) |
| Uniform layer H 20 m, Vs 200, rho 1.9 on rock 760 / 2.4, xi 2 %, steady sine, 10 elements | Kramer (1996) Sec. 7.2 closed form with the Rayleigh xi(f) | 0.5 f1: 1.361 vs 1.376 (-1.1 %); f1 = 2.5 Hz: 4.178 vs 4.169 (+0.2 %); 2 f1: 0.989 vs 0.989; 3 f1: 3.488 vs 3.543 (-1.6 %) |
| Same column, synthetic pulse, linear elastic | pystrata linear elastic (frequency domain, constant 2 %) | PGA 0.749 vs 0.758 g (-1.3 %); Sa 0.1-1.0 s within 3.3 % |
| Sand/clay/sand, water at 5 m | Hand sigma'v | within 0.0003 % (0.04 % with gravity at the shaking permeability, before GM-85) |
| Clay H 15, Vs 150, su 60, synthetic pulse, nonlinear | pystrata EQL with the same hyperbolic backbone (gamma_ref = su/Gmax, Masing damping + 2 %) | amplification 0.01 g: 2.62 vs 2.88; 0.05 g: 2.78 vs 3.17; 0.10 g: 1.91 vs 2.58; 0.30 g: 0.96 vs 1.40 (linear elastic 2.81; before GM-84 2.62 / 2.79 / 1.92 / 0.97) |

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

## Validation against the OpenSees examples (GM-83, 2026-10-15)

Two examples by the authors of the recipe this module adapts, ported to
openseespy as test helpers and run on the same record:

1. **"Site Response Analysis of a Layered Soil Column (Total Stress
   Analysis)"**, Christopher McGann and Pedro Arduino, University of
   Washington (OpenSees wiki,
   https://opensees.berkeley.edu/wiki/index.php/Site_Response_Analysis_of_a_Layered_Soil_Column_(Total_Stress_Analysis);
   `freeFieldSingle.tcl`, whose header also names HyungSuk Shin and Peter
   Mackenzie-Helnwein). 40 m of PressureIndependMultiYield soil (rho 1.7,
   Vs 250, cohesion 95 kPa, Poisson's ratio 0) on rock 760 m/s / 2.4 Mg/m3,
   160 `quad` elements of 0.25 m, 2 % Rayleigh at 0.2 and 20 Hz. Port:
   `tests/opensees_wiki_free_field.py`. It reproduces the wiki's published
   result: surface PGA 0.353 g at 4.43 s (Fig. 2: about 0.35 g near 4.5 s),
   Sa(0.4 s) 1.38 g (Fig. 3: a peak of about 1.4 g near 0.4 s). The Tcl
   does reset the clock (`setTime 0.0`) and its Path series starts at
   t = 0.
2. **"Effective Stress Site Response Analysis of a Layered Soil Column"**,
   OpenSeesPy docs section 14.4.2 (`freeFieldEffective.py`,
   https://openseespydoc.readthedocs.io/en/latest/src/freeFieldEffective.html),
   converted to openseespy by Harsh Mistry (University of Manchester) from
   the wiki example by McGann, Shin, Arduino and Mackenzie-Helnwein. 30 m
   of PressureDependMultiYield02 sand (2 m dry crust, 8 m loose, 20 m
   dense), water 2 m down, a 2 % slope, 60 nine-node `9_4_QuadUP` elements.
   Port: `tests/opensees_docs_free_field_effective.py` (about 60 s a run).
   It reproduces the wiki's published peak r_u: 0.43 / 0.97 / 0.56 at 3 /
   7 / 16 m (the wiki's figure: about 0.42 / 0.97 / 0.55).

The record: the first 20 s of the examples' Gilroy No. 1 east-west motion
(`tests/data/gilroy1_ew_first20s_microg.txt`, 25.8 KB, micro-g; the full
text record is 121 KB; 99.8 % of the Arias intensity; the header says how
it was derived). Tests: `tests/test_site_response_wiki_example.py`,
`tests/test_site_response_effective_example.py` (WSL, openseespy 3.8.0;
the port's numbers are stored, so only the total stress port runs in the
suite). Spectral accelerations are 5 %-damped, in g; the amplification
spectrum (surface / input Sa) is pinned alongside.

**Total stress column.** The module mapped as closely as its arguments
allow: su = 2 x 95 / sqrt(3) (so cohesi is the Tcl's 95), 25 surfaces, 160
elements, the water table at the base (a dry column), damping 0.41 % (the
example's Rayleigh ratio at the site frequency 1.56 Hz). It cannot set
Poisson's ratio (its clay is 0.495, the Tcl's 0), the Rayleigh frequencies
(it uses f1 and 5 f1) or the peak shear strain (0.1 against 0.05); "port,
module's inputs" is the port with those three set as the module sets them.
On this column the Tcl's `quad`, an `SSPquad` and the module's dry
`SSPquadUP` give the same motion to 1e-12.

| Run | PGA | 0.1 s | 0.2 | 0.3 | 0.4 | 0.5 | 0.64 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|
| Input | 0.473 | 1.067 | 0.973 | 1.411 | 1.984 | 1.198 | 0.411 | 0.310 | 0.222 |
| Port, linear elastic | 0.789 | 1.089 | 1.859 | 1.779 | 2.595 | 2.115 | 0.740 | 0.498 | 0.278 |
| Module, linear elastic | 0.811 | 1.109 | 1.896 | 1.788 | 2.596 | 2.118 | 0.741 | 0.499 | 0.278 |
| Port as written | 0.353 | 0.952 | 0.762 | 1.096 | 1.376 | 0.799 | 0.388 | 0.419 | 0.364 |
| Port, module's inputs | 0.433 | 0.940 | 0.748 | 1.071 | 1.362 | 0.778 | 0.429 | 0.468 | 0.388 |
| Module before GM-84 (KrylovNewton, 1e-3 m) | 0.455 | 0.610 | 0.619 | 0.978 | 1.236 | 0.711 | 0.336 | 0.355 | 0.329 |
| Module since GM-84 (Newton) | 0.433 | 0.941 | 0.748 | 1.071 | 1.362 | 0.778 | 0.429 | 0.468 | 0.388 |

Pinned: linear within 3 % (measured -0.1 to +2.0 % from 0.1 to 1.5 s;
PGA +2.7 %, within 5 %: the module's Rayleigh curve is less damped above
about 8 Hz); nonlinear against the port as written within 15 % from 0.3 to
1.5 s (measured -2.6 to +11.7 %; Poisson's ratio alone moves 0.64-1.5 s by
7-12 %), PGA higher by 10 to 35 % (measured +23 %); nonlinear against the
port with the module's inputs, amplification within 3 % and PGA within
5 % (measured within 0.1 % and -0.1 %). Before GM-84 the last was a strict
expected failure (-35 to -9 %, PGA +5 %), and the as-written pins were
25 % and PGA under +50 % because the unconverged steps scattered them
(-9.8 to -15.3 %, PGA +29 to +38 %).

**Effective stress column.** Fifteen 2 m sublayers of four 0.5 m elements
(the module puts the same number of elements in every layer), each Vs set
so rho Vs^2 at the module's reference pressure equals the example's
Gr (p/101)^0.5; water 2 m, rock 700 / 2.5, permeability 1e-4 m/s; damping
0.40 % (the example's ratio at the module's site frequency 1.77 Hz). Not
settable: the slope, the element, bulk moduli (the module's drained nu 0.3),
the fluid's bulk modulus (2.2e6 against 6.88e6 / 5.06e6 kPa), and the dense
layer's contraction and dilation constants (the module uses the Dr = 40 %
ones for every sand).

| Run | PGA | 0.1 s | 0.2 | 0.3 | 0.4 | 0.5 | 0.64 | 1.0 | 1.5 | peak r_u 3 / 7 / 16 m |
|---|---|---|---|---|---|---|---|---|---|---|
| Port as written (2 % slope) | 0.356 | 0.832 | 0.562 | 0.680 | 0.790 | 0.753 | 0.618 | 0.325 | 0.311 | 0.43 / 0.97 / 0.56 |
| Port, module's inputs, level | 0.545 | 0.917 | 0.506 | 0.657 | 0.733 | 0.605 | 0.579 | 0.323 | 0.303 | 0.38 / 0.91 / 0.97 |
| Module before GM-85 | gravity stage fails (defect B) | | | | | | | | | |
| Module, gravity at 1 m/s then 1e-4, KrylovNewton (GM-83 probe) | 0.289 | 0.951 | 0.492 | 0.641 | 0.715 | 0.568 | 0.556 | 0.324 | 0.302 | 0.36 / 0.89 / 0.98 |
| Module since GM-84 and GM-85 | 0.291 | 0.958 | 0.494 | 0.641 | 0.715 | 0.568 | 0.556 | 0.324 | 0.302 | 0.36 / 0.89 / 0.98 |

Pinned (a strict expected failure until GM-85): Sa within 10 % of the
port with the module's inputs (measured -6.2 to +4.5 %; the elements
differ) and peak r_u within 0.1. PGA is not compared: in liquefying sand
it is a dilation spike that depends on the element. Against the port as
written the dense layer is the difference (the module's liquefies: r_u
0.98 at 16 m against 0.56).

"Probe" rows: the module's own model code run with one OpenSees call
changed by intercepting it in a scratch script, before the fix.

**Defect A (fixed, GM-84): the dynamic stage accepted unconverged steps
on small elements.** KrylovNewton with NormDispIncr 1e-3 m (as the
effective stress example has it). 1 mm is 0.4 % strain on a 0.25 m element, and
KrylovNewton (modified Newton on the step's first tangent) makes small
corrections while the soil yields, so the test passes with the residual
still large: high-frequency noise and lost mid-period response. Evidence:
the total stress port (with the module's damping band) with only its
algorithm switched from Newton to KrylovNewton: PGA +56 %, Sa(0.4 s)
-27 %, Sa(1.0 s) -30 %, the share of motion above 15 Hz 0.28 -> 0.65; the
port at 1e-5 is unchanged, and Penalty + KrylovNewton at 1e-5 returns the
converged answer; rounding the input by 5e-7 g moved the module's PGA 6 %
and Sa(0.1 s) 27 %. By element size (the same column): 5, 2.5 and 1 m
within 2 % from 0.1 s up; 0.5 m Sa(0.1 s) -23 %; 0.25 m -10 to -35 %. On
the existing tests' coarse meshes Newton changes results by under 1 %
(clay 0.05 g amplification 2.790 -> 2.785, 0.30 g 0.967 -> 0.959; sand
0.30 g unchanged). Candidate fix: Newton in the dynamic stage (as the
total stress example), KrylovNewton and ModifiedNewton as fallbacks, or
KrylovNewton at 1e-5; either gives the converged answer (about +30 % run
time on the 160-element column). GM-83 reported it; GM-84 applied the
first (below).

**Defect B (fixed, GM-85): the elastoplastic gravity stage failed on finer
sand meshes.**
Failing at the first plastic gravity step: SAND33 (20 m) at 40 elements
(0.5 m); with water 2 m down at 20 elements (1 m); 30 m of loose sand at
15 elements (2 m); two 10 m sands at 10 elements each; the effective
example's column. The module ran gravity at the shaking permeability
(1e-4 m/s); the example runs gravity at 1 m/s and then sets 1e-4 m/s
through element parameters (`parameter ... hPerm / vPerm`, which SSPquadUP
takes: without that update the column stays drained, r_u about 0.2). With
1 or 1e-2 m/s during gravity every case above finishes; 200 iterations
instead of 35 rescue the 20 m sand but not the effective column. Candidate
fix: the example's procedure. GM-85 applied it (below).

## GM-84 and GM-85 (2026-10-15): the fixes, measured

What changed in `site_response.py`:

- **GM-85, gravity:** every SSPquadUP element is built with permeability
  1 m/s; after the gravity stages each element's own value (sand 1e-4,
  clay 1e-7 m/s) is set with `parameter <tag> element <ele> hPerm|vPerm`
  and `updateParameter`, before the clock reset: the effective stress
  example's procedure. Checked that the update sets the quantity the
  element's constructor takes: updating an element built at 1e-4 to 1e-4
  changes the surface motion by 0 (to machine zero), and 1 m/s through
  gravity then 1e-4 gives the 1e-4-throughout motion within 1.3e-3 g
  (PGA 0.1496 against 0.1499 g; sigma'v0 0.2 % lower, now equal to the
  hand value) on 20 m of sand at 4 elements; without the update the
  column shakes drained (max ru 0.22 against 1.0, PGA doubled).
- **GM-84, dynamic stage:** Newton at NormDispIncr 1e-3 m (35
  iterations), then KrylovNewton and modified Newton at 1e-5 m (50), then
  10 substeps through the same three, else an error naming the time.

Why Newton rather than a tolerance scaled to the element size: on the
wiki column KrylovNewton at 1e-5 m gives the converged answer too
(amplification within 0.4 % of the port, PGA -4.3 %, 8.5 s against
Newton's 8.7 s), but the right test would then depend on the element size
and on a norm that mixes metres with kPa of pore pressure and grows with
the number of nodes. Newton converges quadratically, so the examples' 1 mm
test leaves it in balance on 0.25 m elements as on 5 m ones, and it is the
algorithm of the authors' total stress example, which the module now
matches to 0.1 %. Solved to a 100 times tighter test (Newton or
KrylovNewton at 1e-5 m) the wiki column moves by under 0.6 % in
amplification and -4 to -5 % in PGA (one spike).

The fallbacks, forced (Newton never needed one in any run below):

| Wiki column (160 elements) unless named | Steps not taken by the first algorithm | Amplification vs the port, 0.1-1.5 s | PGA vs the port |
|---|---|---|---|
| Newton, 1e-3 m (the module) | 0 of 3,999 | -0.0 to +0.1 % | -0.1 % |
| Newton, 1e-5 m | 0 | +0.0 to +0.6 % | -4.7 % |
| KrylovNewton, 1e-5 m, alone | 0 | +0.0 to +0.4 % | -4.3 % |
| Modified Newton, 1e-5 m, alone | 205 (substeps) | -3.2 to +0.2 % | -20 % |
| Modified Newton, 1e-7 m, 400 iterations, alone | 298 (substeps) | -3.3 to -0.1 % | -21 % |
| Newton, 1 step in 20 forced into substeps | 199 (substeps) | -0.4 to +0.1 % | -6.0 % |
| Effective column: Newton allowed 2 iterations | 328 (all taken by KrylovNewton at 1e-5) | within 0.15 points of all-Newton | the same |

So a step a fallback accepts is in balance; what moves the PGA is the
substeps (dt/10 integrates the peak differently from dt), which modified
Newton needs on about 5 % of this column's steps. Hence modified Newton
last and substeps after it.

Before and after, run in WSL (openseespy 3.8.0, Python 3.12), the model
only (`time_limit_s=None`):

| Column | Before | After | Run time (s), before -> after |
|---|---|---|---|
| Wiki column, 160 elements of 0.25 m, nonlinear | amplification -35 to -9 % vs the port, PGA +5 %, max strain 1.40 % | within 0.1 %, PGA -0.1 %, max strain 0.45 % | 7.57 -> 8.72 |
| Wiki column, linear elastic | PGA 0.811 g | 0.811 g (same to 1e-9) | 6.10 -> 7.13 |
| Effective example's column, 60 elements of 0.5 m | elastoplastic gravity fails | Sa -6.2 to +4.5 % vs the port, r_u 0.36 / 0.89 / 0.98 | fails at 0.17 -> 5.07 |
| 20 m sand at 40 elements, water at the surface | gravity fails | PGA amplification 0.513, max ru 1.0, sigma'v0 within 0.0005 % | fails -> 4.05 |
| 20 m sand at 20 elements, water 2 m down | gravity fails | 0.666, max ru 0.996, within 0.0004 % | fails -> 1.81 |
| 30 m loose sand (Vs 150, phi 30) at 15 elements | gravity fails | 0.417, max ru 1.0 | fails -> 1.36 |
| Two 10 m sands at 10 elements each | gravity fails | 0.452, max ru 1.0 | fails -> 2.15 |
| Uniform sand 20 m, 4 elements, 0.30 g (PGA amplification / max ru) | 0.4995 / 1.00 | 0.4987 / 1.00 | 0.54 -> 0.59 |
| Uniform clay 15 m, 3 elements | 0.9813 | 0.9818 | 0.36 -> 0.38 |
| Sand / clay / sand, 2 elements a layer, water 2 m | 0.5039 | 0.5038 | 0.63 -> 0.67 |
| Rock-impedance column, 4 elements | 1.0006 | 1.0006 | 0.35 -> 0.37 |
| Kramer column, 10 elements, sine at 0.5 / 1 / 2 / 3 f1 | 1.361 / 4.178 / 0.989 / 3.488 | the same to 1e-8 | 0.56-0.59 -> 0.61-0.66 |
| pystrata linear column, 10 elements | PGA 0.7486 g | 0.7486 g | 0.47 -> 0.52 |
| Sand / clay / sand, water 5 m: sigma'v0 against the hand value | 0.036 % | 0.0003 % | 0.45 -> 0.48 |
| Clay 15 m, 6 elements, 0.01 / 0.05 / 0.10 / 0.30 g | 2.618 / 2.790 / 1.919 / 0.967 | 2.617 / 2.785 / 1.914 / 0.959 | 0.41-0.43 -> 0.44-0.46 |
| Sand over clay, water 2 m / dry | 0.5084 / 0.7036 | 0.5082 / 0.7038 | 0.68 / 0.39 -> 0.73 / 0.41 |
| Dry sand 20 m, 0.30 g, 4 / 10 elements | 1.756 / 1.831 | 1.756 / 1.817 | 0.55 / 0.94 -> 0.56 / 0.99 |

Newton costs 2-17 % more time than KrylovNewton, the most on the
160-element column (less than GM-83's +30 % estimate). The `opensees_agent` suite in WSL: 125 passed and 2 expected
failures in 77 s before; 134 passed in 86 s after (the two markers off,
seven tests added).

## GM-80 and GM-79 (2026-10-20): stress-strain curves, solver steps, the fixed inputs

**Curves (GM-80, `curves.py`).** A layer's `curve` picks the backbone of
its multi-yield material: `opensees` (default, unchanged: the material's own
hyperbolic backbone, reaching the strength at 10 % strain), `darendeli`
(Darendeli 2001; `plas_index` 0, `ocr` 1, `freq_hz` 1, `n_cycles` 10 by
default), `menq` (Menq 2003; `uniformity_coeff` and `diam_mean` required)
or `user` (`G_Gmax_curve` `{strain_pct, G_Gmax}`, optional `damping_curve`
`{strain_pct, damping_pct}`). The chosen curve is evaluated at the layer's
mid-depth mean effective stress (the material's reference pressure) and
sent as the user-defined backbone: a negative `noYieldSurf` followed by
(shear strain, secant G/Gmax) pairs at 31 strains from 1e-6 to 10 %. In
the OpenSees source (`PressureDependMultiYield02::setUpSurfaces`,
`PressureIndependMultiYield::setUpSurfaces`) the LAST pair sets the
strength: sand M = sqrt(3) tau_max / p_ref, sin(phi) = 3M / (6 + M); clay
c = sqrt(3)/2 tau_max. So the fit follows the curve until its stress
reaches the layer's strength (phi at the reference pressure, or su) and
puts the last pair there; a curve still below the strength at 10 % gets its
last pair at the strength at 10 %. The layer keeps its phi (the material
prints 32.96 deg for 33; the residual pressure term) or su. On the GM-77
dry sand column Darendeli and Menq (Cu 10) never reach it: the curve is at
42.8 / 49.0 kPa at 6.8 % against a strength of 91.2 kPa, so the last
segment stiffens (tangent 67 to 1,521 kPa for Darendeli) and the curve
below it implies about half the layer's strength. The runs below peak at
0.35 % strain, far from it; the result's `choices` says which way each
layer's backbone meets its strength. Stress must
rise from pair to pair: the material exits the process otherwise, so the
fit and the user-table checks refuse a falling stress before OpenSees sees
it.

The equations of the two published curves are not printed in the
reference library (GEC 5, FHWA-NHI-16-072, Sec. 8.8, p. 8-23, and FEMA
P-2082, p. 515, cite them only); the constants are pystrata's, and the
tests check them against pystrata to machine precision. The materials damp
on Masing's rules and take no damping curve: `layer_curves` reports the
curve's damping beside the Masing damping of the fitted backbone (for
Darendeli PI 30 clay at 1 %: 21 % against 43 %), and `damping` stays the
Rayleigh term.

Measured on the GM-77 dry sand column (20 m, Vs 200 m/s, phi 33, water
below), synthetic pulse at 0.30 g, 10 elements; equivalent-linear is
`pystrata_agent` with the same curve at the same mean effective stress:

| curve | nonlinear | equivalent-linear |
|---|---|---|
| opensees (default) | 1.817 | -- |
| darendeli, PI 0 | 0.637 | 0.939 |
| menq, Cu 10, D50 5 mm | 0.676 | 0.972 |
| menq, Cu 2, D50 0.4 mm | 1.640 | 1.407 |

At 0.05 g: 2.564 / --, 2.337 / 2.225, 2.272 / 2.111, 2.795 / 3.167.

**Solver steps (GM-79).** `solver_steps` counts the dynamic steps by the
algorithm that converged at the full step, the steps that needed a
fallback (`fallback_steps`) and those taken in substeps
(`substepped_steps`, with their times and a note: substeps moved the PGA
spike -6 to -20 % on the wiki column).

**Inputs that were fixed (GM-79).** Per layer: `nu` (default 0.3 sand,
0.495 clay; the wiki total-stress example uses 0.0 for clay), `n_elem`
(default `n_elem_per_layer`), and for sand the PressureDependMultiYield02
constants `contrac1..3`, `dilat1..3`, `liquefac1..2` (default the Dr =
40 % column). Top level: `rayleigh_freqs_hz` (default the site frequency
and five times it). Each is in the result's `choices`, given or default.

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
- **The OpenSees examples** (GM-83): ports of the two McGann & Arduino
  examples as test helpers, the stored record and the ports' numbers. The
  two tests that were strict expected failures for defects A and B pass
  since GM-84 and GM-85 and pin the fixes.
