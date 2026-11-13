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
  compat.py                # the soil types and EQL tolerance on pystrata 0.5 or 0.6
  eql_site_response.py     # analyze_eql/linear_site_response()
  results.py               # EQLSiteResponseResult dataclass
  tests/
    test_pystrata_agent.py  # Tier 1 + Tier 2 tests
    test_compat.py          # same curves and answers on 0.5 and 0.6
pystrata_agent_foundry.py  # Foundry wrapper (project root)
```

## pystrata versions (BACKLOG INF-68)

Python 3.12 installs pystrata 0.5.x (with pyrvt < 0.9); Python 3.13+
installs pystrata 0.6.x, which needs 3.13 (pyproject markers). 0.6 renamed
the soil classes (`DarendeliSoilType`, `MenqSoilType` and
`NonlinearProperty` became `factories.darendeli_soil_type`,
`factories.menq_soil_type`, `ModulusReductionCurve` and `DampingCurve`) and
changed the equivalent-linear `tolerance` from percent to decimal.
`compat.py` is the only code that knows which API is installed (it checks
for `site.DarendeliSoilType`). The Darendeli and Menq curves are the same
formulas on both (0.6's factories are a port of 0.5's
`ModifiedHyperbolicSoilType`); the 2026-11-06 check gave the same
amplification, iterations and compatible Vs to 1e-15 on seven
equivalent-linear and two linear cases (`module_work/inf68/CHECK.md`).
0.6 clips interpolated G/Gmax to (0.001, 1) and damping to (0, 0.49); 0.5
did not, which only matters for a custom curve with points outside those
ranges.

## Key Design Decisions

1. **pystrata is optional** — same pattern as openseespy. `has_pystrata()`
   runtime check; all Tier 1 tests (validation, result, metadata) run without it.

2. **Units: all SI, no conversion needed** — pystrata uses kN/m³, m/s, kPa, g,
   and decimal strains/damping, which matches our conventions exactly.

3. **Bedrock half-space** — last layer in the list with thickness=0.
   This follows pystrata's convention directly.

4. **Ground motions** — reuses `opensees_agent/ground_motions.py` (lazy import).
   That module has zero openseespy dependency.

5. **stress_mean auto-calculation** — when a Darendeli or Menq layer
   omits `stress_mean`, it is computed at the layer's mid-depth
   (`eql_site_response.layer_stress`): total stress from the layers'
   unit weights, less hydrostatic pore pressure below `water_table_depth`
   (9.81 kN/m3), then `sigma'm = sigma'v (1 + 2 K0) / 3` (NAVFAC DM7.2
   Eq. 6-22, with `k0` for its 1 - sin(phi')), floor 5 kPa. Defaults:
   NO water table (effective = total stress) and K0 = 0.5; with no water
   table given the result's `warnings` and `choices` say so. Until GM-325
   (2026-11-13) this was the same formula on total stress with K0 fixed
   at 0.5 and no water table input. The result lists each curve layer's
   stresses (`layer_stresses`). A total-stress analysis has no pore
   pressure otherwise: the water table changes only this confining
   stress.
6. **Profile auto-discretization** — pystrata subdivides thick layers for
   numerical accuracy. Controlled by `max_freq_hz` (default 25 Hz) and
   `wave_frac` (default 0.2 = λ/5).

7. **Convergence tolerance is a FRACTION** (GM-78, 2026-10-14) —
   `tolerance=0.01` means the iteration stops once no layer's shear modulus
   or damping changes by more than 1 % between iterations. pystrata 0.5's
   tolerance is in percent, so the wrapper passes it `tolerance * 100`;
   pystrata 0.6's is in decimal and takes the fraction as given
   (`compat.py`, INF-68). Until
   2026-10-14 the fraction went through unchanged (a 0.01 % target), every
   run used all `max_iterations` and was reported not converged.
   `converged` is judged on the SIZE of the largest change
   (`_max_abs_error_percent`): pystrata's `max_error` keeps its sign, so a
   layer that stiffened could read as converged. The test profile now
   converges in 6 iterations at 1 %.

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

## What the tool result carries (GM-325, 2026-11-13)

`to_dict()` returns, beside the scalars: the surface and input spectra at
11 periods (`REPORTED_PERIODS_S`, read off the 500-period spectra on a log
period axis), the largest surface ordinate and its period, and
`depth_profile`, one row per sub-layer of the discretised profile (at most
20; more are merged, keeping the largest strain and acceleration and the
travel-time average Vs). pystrata's profile outputs do not share one depth
array: the strain is at each sub-layer's middle after a leading surface
point, the acceleration at each sub-layer's top, and the two Vs profiles
are in step-plot form (the first value twice, pystrata 0.5.4
`output.InitialVelProfile`). `depth_profile` matches the strain by depth
and reads Vs from entry i + 1; a Vs array not of that form gives no
profile instead of a shifted one. Checked on pystrata 0.5.4 only (0.6
needs Python 3.13).

The input motion is applied as a rock outcrop motion at the top of the
half-space; the tool text and `choices` say so.
