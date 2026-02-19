# pyStrata Agent — Design Notes

## Purpose

Wraps the pystrata library (arkottke/pystrata, MIT license) for 1D
equivalent-linear site response analysis (SHAKE-type). This is the standard
frequency-domain approach used in everyday practice for moderate-shaking
scenarios.

## Architecture

```
pystrata_agent/
  __init__.py              # exports public API
  pystrata_utils.py        # has_pystrata(), import_pystrata()
  eql_site_response.py     # analyze_eql/linear_site_response()
  results.py               # EQLSiteResponseResult dataclass
  tests/
    test_pystrata_agent.py  # Tier 1 + Tier 2 tests
pystrata_agent_foundry.py  # Foundry wrapper (project root)
```

## Key Design Decisions

1. **pystrata is optional** — same pattern as openseespy. `has_pystrata()`
   runtime check; all Tier 1 tests (validation, result, metadata) run without it.

2. **Units: all SI, no conversion needed** — pystrata uses kN/m³, m/s, kPa, g,
   and decimal strains/damping, which matches our conventions exactly.

3. **Bedrock half-space** — last layer in the list with thickness=0.
   This follows pystrata's convention directly.

4. **Ground motions** — reuses `opensees_agent/ground_motions.py` (lazy import).
   That module has zero openseespy dependency.

5. **stress_mean auto-calculation** — when Darendeli/Menq layers omit
   `stress_mean`, it is computed from profile geometry:
   `sigma_v_mid * (1 + 2*K0) / 3`, K0=0.5, floor at 5 kPa.

6. **Profile auto-discretization** — pystrata subdivides thick layers for
   numerical accuracy. Controlled by `max_freq_hz` (default 25 Hz) and
   `wave_frac` (default 0.2 = λ/5).

## Supported Soil Models

| Model | Class | Required Params | Optional |
|-------|-------|----------------|----------|
| darendeli | DarendeliSoilType | plas_index (%) | ocr, stress_mean |
| menq | MenqSoilType | — | uniformity_coeff, diam_mean, stress_mean |
| linear | SoilType | damping (decimal) | — |
| custom | SoilType | strains, mod_reduc, damping_values | — |

## Comparison: EQL (pystrata) vs Nonlinear (opensees_agent)

| Feature | pystrata_agent (EQL) | opensees_agent (Nonlinear) |
|---------|---------------------|---------------------------|
| Domain | Frequency | Time |
| Iteration | Strain-compatible | Fully nonlinear |
| Pore pressure | No | Yes (effective stress) |
| Speed | Fast (< 1 second) | Slow (minutes) |
| Large strain | Limited (strain_limit) | Unlimited |
| Soil models | G/Gmax + damping curves | PDMY02 (sand), PIMY (clay) |
| Typical use | Moderate shaking, screening | Strong shaking, liquefaction |

## Edge Cases

- **Very soft soil over stiff rock**: Large impedance contrast amplifies
  motion significantly. EQL may not converge if strains exceed ~5%.
  Check `converged` flag in results.

- **High-frequency input motion**: Auto-discretization subdivides layers
  based on wavelength. Very high-frequency content (>25 Hz) may require
  increasing `max_freq_hz`.

- **Linear bedrock**: The half-space is always linear elastic (no
  modulus reduction). Use `soil_model="linear"` with appropriate damping
  (typically 0.5-2% for rock).
