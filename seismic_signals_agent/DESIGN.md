# Seismic Signals Agent — Design Notes

## Purpose

Wraps the eqsig library for earthquake signal processing. Provides
response spectra (Nigam-Jennings, the shared
`geotech_common.response_spectrum` since GM-274), intensity measures,
orientation-independent spectra (RotD0/RotD50/RotD100, from the same
time-domain responses since GM-275; pyrotd before), and signal processing
(filtering + baseline correction).

## Architecture

```
seismic_signals_agent/
  __init__.py              # exports public API
  signal_utils.py          # has_eqsig(), has_pyrotd(), _G constant
  response_spectrum.py     # analyze_response_spectrum()
  intensity_measures.py    # analyze_intensity_measures()
  rotd_spectrum.py         # analyze_rotd_spectrum()
  rotd_time_domain.py      # RotD from rotated time-domain responses (GM-275)
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
   `analyze_rotd_spectrum` fell back to `rotd_native.py`, a numpy
   implementation of the same procedure (frequency-domain SDOF response,
   rotation 0-179 deg, linear percentiles, the Stewart et al. 2017 reduced
   rotation set), equal to pyrotd to round-off. Both went in GM-275 (below);
   `has_pyrotd()` and `pyrotd_import_error()` stay for callers that check
   them (the Foundry wrapper).

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

## RotD from time-domain responses (GM-275, 2026-10-21)

**Method.** RotD as Boore (2010) defines it: for each period, each
component's oscillator displacement history, the two combined at every
angle (`u_a cos theta + u_b sin theta`, theta 0-179 deg in 1 deg steps), the
peak at each angle, and RotDnn the nn-th percentile of those peaks over the
angles (linear interpolation; RotD0 the least, RotD50 the median, RotD100
the largest). The histories come from the response spectrum's own code,
`geotech_common.response_spectrum.oscillator_response`: Nigam and Jennings
(1969), the record linear between samples, sub-steps of at most T / 20
below 20 record steps a period, over the record's length. The peak at each
angle is read between points as the spectrum reads it, off the cubic
through the rotated displacement and velocity at both ends of each step.
So RotD100 of a component paired with zeros is that component's
`analyze_response_spectrum` Sa, value for value (test). Boore (2010) is not
in the reference library; the definition is as the module has always
cited it. `engine` is `"nigam_jennings"` and `engine_note` states the
method; the other result fields are unchanged.

Only the steps that can hold an angle's peak are rotated, none that can is
left out: on a step no projection of the vector (u_a, u_b) exceeds the
larger of its lengths at the two ends plus 4/27 of the two end velocities
times the step (the cubic's bound). Each angle's peak starts from the
points of largest vector length; the steps are rotated in blocks, largest
bound first, and an angle stops once its peak reaches the bound of the
steps left. (The fixed level `rotd_native` used, 0.7 of the smaller
component's peak, can leave out the peak of an angle whose peak is below
it: on a record moving mostly along one line with a later, smaller motion
across it, RotD0 0.0005 g at 0.5 s where every step gives 0.29 g; test
`test_no_angle_loses_its_peak`.)

**Before (GM-274's check, one component with the other zero, 27 periods
0.01-4 s, 20 s records):**

| Record (dt) | under ten steps a period | ten steps and up |
|---|---|---|
| dense sand surface, 80 el. (0.01 s) | +0.9 to +60.9 % | -2.3 to +2.8 % |
| dense sand input (0.01 s) | -0.8 to +0.1 % | -0.4 to +4.9 % |
| wiki column surface (0.005 s) | +1.8 to +16.8 % | -8.2 to +14.7 % |
| effective example surface (0.005 s) | +2.0 to +10.1 % | -8.1 to +30.0 % |
| Gilroy input (0.005 s) | -0.1 to +3.2 % | -3.5 to +25.4 % |

Two causes. The record was not padded, so the FFT took it as periodic and
the oscillator still ringing at its end ran on into its start (with 80 s
of zeros, -0.4 to +1.6 % from ten steps up). And below ten steps the FFT
read the record as band-limited, sine waves through the samples rather
than straight lines between them (padded, still up to +61 % on the dense
sand surface at 0.02 s, the record's Nyquist period).

**After, two-component pairs, RotD0/50/100 against scipy's lsim** (the
same records linear between samples, the two responses read at least 100
times a period and 10 times a record step, rotated the same way):

| Pair (dt, length) | T (s) | before (FFT) | after |
|---|---|---|---|
| Gilroy No. 1 EW + the same reversed in time (0.005 s, 20 s) | 0.0137 | +2.1 to +2.2 % | +0.001 to +0.002 % |
| | 0.026 | +2.5 to +3.3 % | 0.000 to +0.003 % |
| | 0.2 | 0.00 % | 0.000 % |
| | 1.0 | -0.5 to +0.3 % | 0.000 % |
| | 3.0 | -11.6 to +29.1 % | 0.000 % |
| synthetic pulse + synthetic long (0.01 s, 40 s) | 0.0137 | 0.0 to +0.1 % | 0.000 % |
| | 0.2 | +0.1 to +0.2 % | 0.000 to +0.001 % |
| | 1.0 | -0.1 to +0.2 % | 0.000 % |
| | 3.0 | +0.4 to +1.1 % | 0.000 % |

(Only Gilroy's EW component is stored, so its second component is the
same record reversed in time: the same frequency content, its peaks
elsewhere.) Against the FFT with the records padded by 60 s of zeros and
read every dt / 8, at 0.2, 0.5 and 1 s, the new RotD is 0.01 to 0.16 %
lower: there the straight lines between samples carry slightly less than
the band-limited record near the period.

**Time.** 100 periods x 180 angles on a 40 s pair: 0.8 s at 0.005 s
(Gilroy pair repeated), 1.1 s at 0.01 s (synthetic pair); the FFT took
0.4 and 0.7 s. The default 200 periods take about twice that. Two
identical components (the motion along one line, so the angles across it
peak near zero and read every step) take about 4 s for 200 periods on a
20 s record.

**The FFT path went rather than stay as an alternative.** Kept honest it
needs padding long enough for the longest period's free vibration to die
out, and below ten record steps it still answers a different question
(the record read as band-limited, up to 61 % apart on an OpenSees surface
record), so a second engine would give two RotD for one record with
nothing to choose between them. `rotd_native.py` is deleted; pyrotd is no
longer imported by any analysis (the `[seismic-signals]` extra still lists
it; whether to drop it is a packaging call).

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
(0° to 179°, 1° increment) of two orthogonal components, then takes
percentiles across orientations (computed as in GM-275 above):

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

- **RotD with identical components**: the motion is along one line at
  45 deg, so RotD0 is 0, RotD50 the component's Sa and RotD100 the square
  root of two times it (degenerate but valid).

- **Unequal record lengths for RotD**: Shorter record is zero-padded
  to match the longer one.
