# Seismic Signals Agent — Design Notes

## Purpose

Wraps the eqsig and pyrotd libraries for earthquake signal processing.
Provides response spectra (Nigam-Jennings), intensity measures,
orientation-independent spectra (RotD50/RotD100), and signal processing
(filtering + baseline correction).

## Architecture

```
seismic_signals_agent/
  __init__.py              # exports public API
  signal_utils.py          # has_eqsig(), has_pyrotd(), _G constant
  response_spectrum.py     # analyze_response_spectrum()
  intensity_measures.py    # analyze_intensity_measures()
  rotd_spectrum.py         # analyze_rotd_spectrum()
  signal_processing.py     # analyze_signal_processing()
  results.py               # 4 result dataclasses
  tests/
    test_seismic_signals.py
seismic_signals_agent_foundry.py  # Foundry wrapper (project root)
```

## Key Design Decisions

1. **Two independent optional dependencies** — eqsig and pyrotd are checked
   separately. `has_eqsig()` and `has_pyrotd()` follow the same pattern as
   `has_opensees()`. Tier 1 tests work without either installed.

2. **Unit conversion at the boundary** — eqsig uses m/s² internally; our
   project convention is g. The constant `_G = 9.81` lives in signal_utils.py.
   - Input to eqsig: `accel_g * 9.81`
   - Output from eqsig: `accel_m_s2 / 9.81`
   - pyrotd uses g natively — no conversion needed.

3. **Ground motions** — reuses `opensees_agent/ground_motions.py` (lazy import).
   That module has zero openseespy dependency.

4. **Per-method dependency check** in the Foundry agent — since methods use
   different libraries, the Foundry wrapper checks eqsig or pyrotd only for
   the specific method being called.

5. **RotD without pyrotd (2026-10-09, live smoke G10).** pyrotd 0.6.1
   reads its `__version__` through `pkg_resources`, which setuptools 81+
   no longer ships, so on a fresh environment it is installed but
   `import pyrotd` fails. `pyrotd_import_error()` reports which case it is
   (not installed / installed but failing, with the real exception), and
   `analyze_rotd_spectrum` falls back to `rotd_native.py`, a numpy
   implementation of the same procedure (frequency-domain SDOF response,
   rotation 0-179 deg, linear percentiles, the Stewart et al. 2017 reduced
   rotation set). It matches pyrotd to round-off (test `TestRotDNative`,
   which loads pyrotd with a stub `pkg_resources` as the oracle). The
   result says which engine ran (`engine`, `engine_note`).

6. **Results carry what the tool docs promise (2026-10-09, live smoke G6).**
   `ResponseSpectrumResult.to_dict` returns `Sa_max_g`, `T_peak_s`,
   `periods_s`, `Sa_g`; `RotDSpectrumResult.to_dict` returns `periods_s`,
   `percentiles` and `spectra` (every requested percentile, not just
   0/50/100); `SignalProcessingResult.to_dict` adds `filter_applied` and
   `baseline_corrected`.

## Comparison: eqsig vs Existing Response Spectrum

| Feature | eqsig (this module) | compute_response_spectrum (opensees_utils) |
|---------|--------------------|--------------------------------------------|
| Algorithm | Nigam-Jennings | Newmark-beta average acceleration |
| Accuracy | Exact for piecewise-linear input | Dt-dependent |
| Speed | Fast (vectorized C/Fortran) | Moderate (Python loop) |
| Intensity measures | Yes (Arias, CAV, Ds5-95) | No |
| Dependencies | eqsig | numpy only |
| Use case | Full signal analysis | Quick spectrum when eqsig not installed |

## Intensity Measures

| Measure | Symbol | Units | Description |
|---------|--------|-------|-------------|
| Arias Intensity | Ia | m/s | π/(2g) ∫ a²(t) dt — cumulative energy measure |
| Significant Duration | D5-95 | s | Time between 5% and 95% Arias intensity |
| CAV | CAV | m/s | ∫ |a(t)| dt — Cumulative Absolute Velocity |
| Bracketed Duration | Db | s | Time between first and last exceedance of 0.05g |
| PGA | — | g | Peak ground acceleration |
| PGV | — | m/s | Peak ground velocity |
| PGD | — | m | Peak ground displacement |

## RotD50/RotD100 (Boore 2010)

Rotated spectral acceleration computes the PSA for all rotation angles
(0° to 180°, 1° increment) of two orthogonal components, then takes
percentiles across orientations:

- **RotD0**: minimum over all orientations
- **RotD50**: median over all orientations (orientation-independent)
- **RotD100**: maximum over all orientations

RotD50 is now the standard for NGA-West2 ground motion models (replacing
geometric mean).

## Edge Cases

- **Very short records (<1s)**: Arias intensity and significant duration
  may be unreliable. The wrapper does not restrict this — user judgment.

- **Bandpass filter order**: eqsig uses a 4th-order Butterworth by default.
  Very narrow bands (f_high ≈ f_low) may cause numerical issues.

- **Processing order**: Bandpass is applied before baseline correction
  (standard seismological practice).

- **RotD with identical components**: RotD0 = RotD50 = RotD100 (degenerate
  but valid).

- **Unequal record lengths for RotD**: Shorter record is zero-padded
  to match the longer one.
