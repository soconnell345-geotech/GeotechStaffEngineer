# Seismic Signals Agent — Design Notes

## Purpose

Wraps the eqsig and pyrotd libraries for earthquake signal processing.
Provides response spectra (Nigam-Jennings, the shared
`geotech_common.response_spectrum` since GM-274), intensity measures,
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

## Response spectrum: one code with the OpenSees tool (GM-274, 2026-10-21)

`analyze_response_spectrum` takes Sa from `geotech_common.response_spectrum`
(`pseudo_spectral_accel`), the code `opensees_agent.opensees_utils.
compute_response_spectrum` wraps, so the two tools give the same Sa for the
same record (`TestOneSpectrumCode` in both suites, bit for bit). Every
period by Nigam and Jennings (1969): the record linear between its
samples, each oscillator solved in closed form, in sub-steps of at most
T / 20 below 20 record steps a period, the peak read between points off
the cubic through u and v; within 0.004 % of scipy's lsim (GM-272). PGV
and PGD still come from eqsig (`AccSignal`), as do the intensity measures
and the filters, so the tool still needs eqsig. The result's fields and
units are unchanged; `summary()` names the method.

Until GM-274 the spectrum was eqsig's (`AccSignal.generate_response_spectrum`):
the same step, but the whole record at one step, the larger of the
shortest period / 20 and a quarter of the record's step, the peak read
only at its points, and the PGA where a period was under six of them.
Against lsim it read up to 2.5 % low under ten record steps a period
(-2.5 % at 0.0137 s on the dense sand surface, 0.01 s; -0.32 % at 0.026 s
on the Gilroy input, 0.005 s) and within 0.04 % from ten steps up. So Sa
under ten record steps now rises by up to about 2.5 % (synthetic pulse,
0.01 s: 0.3000 -> 0.3023 g); no test pinned those values.

## RotD: measured against lsim, not changed (GM-274)

`analyze_rotd_spectrum` (pyrotd's procedure, or `rotd_native`, the same to
2e-16 on the records below) computes each response in the frequency
domain. Its RotD100 of one component with the other zero is that
component's spectrum, so it was checked against lsim like the spectrum
above: 27 periods 0.01-4 s, the GM-272 records (20 s each).

| Record (dt) | under ten steps a period | ten steps and up |
|---|---|---|
| dense sand surface, 80 el. (0.01 s) | +0.9 to +60.9 % | -2.3 to +2.8 % |
| dense sand input (0.01 s) | -0.8 to +0.1 % | -0.4 to +4.9 % |
| wiki column surface (0.005 s) | +1.8 to +16.8 % | -8.2 to +14.7 % |
| effective example surface (0.005 s) | +2.0 to +10.1 % | -8.1 to +30.0 % |
| Gilroy input (0.005 s) | -0.1 to +3.2 % | -3.5 to +25.4 % |

Two causes:

- **The record is not padded.** The FFT takes the record as periodic, so
  the oscillator still ringing at its end runs on into its start. With
  80 s of zeros after each record, the procedure is within -0.4 to +1.6 %
  of Nigam and Jennings on the same padded record from ten steps a period
  up (the rest is the peak read only at its points, ten or more a period).
- **Under ten steps it reads the record as band-limited**, sine waves
  through the samples rather than straight lines between them: padded,
  still up to +61 % on the dense sand surface (0.02 s, two steps a period,
  the record's Nyquist frequency), +15 % on the wiki surface and +3.7 % on
  the Gilroy input.

So RotD100 of one component is not, as it stands, the spectrum
`analyze_response_spectrum` gives for it. Which reading of a record
between its samples a short-period spectrum should use is a method choice;
padding is not. Left for the backlog (see `BACKLOG.md`).

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
