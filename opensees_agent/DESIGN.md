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
  results.py           -- PM4SandDSSResult, BNWFPileResult, SiteResponseResult
  pm4sand_dss.py       -- analyze_pm4sand_dss()
  bnwf_pile.py         -- analyze_bnwf_pile()       [Tier 2]
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

### BNWF Lateral Pile
- Beam-on-nonlinear-Winkler-foundation (BNWF) approach
- Pile: `elasticBeamColumn` elements (2D beam, ndm=2, ndf=3)
- Lateral springs: `PySimple1` uniaxial material (soilType 1=clay, 2=sand)
- Shaft friction: `TzSimple1` on same zero-length elements
- Tip bearing: `QzSimple1` at pile toe
- Reuses all 7 p-y models from `lateral_pile/py_curves.py` — no duplicated
  p-y formulations.  `_build_py_model()` instantiates the correct class from
  a layer dict, then `_get_py_params()` extracts pult and y50 from the p-y
  curve at each node depth.
- Node numbering: soil nodes (fixed) → spring nodes (free) → pile nodes,
  connected via `zeroLength` + `equalDOF`
- Static analysis with Newton-Raphson, 20 load increments, ModifiedNewton fallback
- Reference: OpenSees BNWF example (openseespydoc.readthedocs.io), API RP2A

### 1D Effective-Stress Site Response
- 1D soil column with SSPquadUP elements (4-node u-p formulation)
- Sand: `PressureDependMultiYield02` (Gmax = rho*Vs^2, conservative
  contraction/dilation defaults, 20 yield surfaces)
- Clay: `PressureIndependMultiYield` (su as cohesion, phi=0 total stress,
  near-incompressible bulk modulus capped at 100*Gmax)
- Lysmer-Kuhlemeyer dashpot at base: `Viscous` uniaxial material with
  c = rho_bedrock * Vs_bedrock * area (zeroLength element)
- Loading: force = 2 * c_dashpot * velocity_input (factor 2 for outcrop
  motion assumption)
- Mesh: layers reversed bottom-to-top, n_elem_per_layer elements per layer,
  left/right node columns with equalDOF periodic BCs
- Surface pore pressure fixed (drained); above-GWT nodes have pp DOF fixed
- Analysis phases: (1) gravity elastic 10 steps, (2) switch to elastoplastic
  10 steps, (3) dynamic Newmark (0.5, 0.25) with Rayleigh damping
- Rayleigh damping: f1 = Vs_avg/(4*H), f2 = 5*f1
- Convergence fallback: KrylovNewton → ModifiedNewton → 10 substeps of dt/10
- Output: surface accel time history, response spectra, depth profiles of
  max accel, max shear strain, and max pore pressure ratio
- Reference: Lysmer & Kuhlemeyer (1969); Yang, Elgamal & Parra (2003)

## Ground Motions

Built-in synthetic motions are provided for testing.  Custom motions can
be provided as acceleration arrays.  The `ground_motions.py` module
includes pack/unpack utilities for embedding real records as compressed
base64 numpy arrays.

## Testing Strategy

- **Tier 1** (no openseespy): Input validation, result dataclasses,
  ground motions, utilities, Foundry agent metadata, BNWF layer parsing,
  site response validation.  ~97 tests.
- **Tier 2** (requires openseespy): Integration tests with actual FE
  analyses.  Skipped via `@pytest.mark.skipif` when not installed.  ~9 tests.
