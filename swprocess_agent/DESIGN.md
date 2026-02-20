# swprocess Agent — Design Notes

## Purpose

Wraps the swprocess library for Multichannel Analysis of Surface Waves
(MASW) — extracting Rayleigh wave dispersion curves from seismic array
recordings, which are then inverted to obtain Vs profiles.

## Architecture

```
swprocess_agent/
  __init__.py           # exports analyze_masw + DispersionResult
  swprocess_utils.py    # has_swprocess(), import helpers
  masw_analysis.py      # analyze_masw()
  results.py            # DispersionResult
  tests/
    test_swprocess_agent.py
  DESIGN.md
swprocess_agent_foundry.py  # Foundry wrapper
```

## MASW Pipeline

1. **Input**: Multi-channel seismograms (traces + offsets + dt)
2. **Wavefield Transform**: Convert time-distance to frequency-velocity
   - PhaseShift (default, most common)
   - FK (frequency-wavenumber)
   - FDBF (frequency-domain beamforming)
3. **Peak Picking**: Extract dispersion curve (peak velocity at each freq)
4. **Output**: Dispersion curve (frequency, phase velocity)

## Geotechnical Applications

- Vs profiling for site classification (non-invasive)
- Foundation design Vs measurements
- Liquefaction assessment (Vs-based methods)
- Input for site response analysis

## Data Format

- traces: list of 1D arrays (one per sensor)
- offsets: source-to-receiver distances (m)
- dt: sampling interval (s)

## Edge Cases

- swprocess `find_peak_power()` returns velocity values (not indices)
- Power grid is large — omit from JSON, keep dispersion curve only
- All traces must have same length
- Minimum 3 channels for meaningful transform
