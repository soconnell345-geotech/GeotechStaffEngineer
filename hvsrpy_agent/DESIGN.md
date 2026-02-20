# hvsrpy Agent — Design Notes

## Purpose

Wraps the hvsrpy library for computing Horizontal-to-Vertical Spectral
Ratios (HVSR) from 3-component microtremor or earthquake recordings.
HVSR is used to identify:

- **Site resonant frequency (f0)** — the fundamental frequency of the soil column
- **Peak amplification (A0)** — the HVSR amplitude at f0
- **Site period (T0 = 1/f0)** — for seismic site classification and design

## Architecture

```
hvsrpy_agent/
  __init__.py              # exports analyze_hvsr, HvsrResult, has_hvsrpy
  hvsrpy_utils.py          # has_hvsrpy(), import_hvsrpy()
  hvsr_analysis.py         # analyze_hvsr() wrapper
  results.py               # HvsrResult dataclass
  tests/
    test_hvsrpy_agent.py
  DESIGN.md
hvsrpy_agent_foundry.py    # Foundry wrapper (project root)
```

## Key Design Decisions

1. **Optional dependency** — hvsrpy (and its dependency obspy) checked
   at runtime via `has_hvsrpy()`. Tier 1 tests work without it.

2. **Array-based input** — accepts numpy arrays directly (ns, ew, vt, dt)
   rather than file paths, making it usable from any data source.

3. **Single function API** — `analyze_hvsr()` wraps the full pipeline:
   preprocess → process → reject → extract → SESAME check.

4. **SESAME (2004) criteria** — automatically evaluates reliability (3)
   and clarity (6) criteria for quality assessment.

5. **obspy pkg_resources fix** — obspy uses deprecated `pkg_resources`;
   requires `setuptools<82` on Python 3.14.

## HVSR Theory

The HVSR method (Nakamura, 1989) estimates site amplification by computing
the ratio of horizontal-to-vertical Fourier amplitude spectra from ambient
vibration recordings:

    HVSR(f) = sqrt(H_ns(f) * H_ew(f)) / V(f)    [geometric mean]

The peak of the HVSR curve identifies the fundamental resonant frequency
of the soil column (f0). The site period T0 = 1/f0 relates to soil
thickness H and average shear wave velocity Vs:

    T0 = 4H / Vs   →   Vs30 ≈ 4 * H * f0  (for bedrock at depth H)

## Processing Pipeline

1. **Create recording** — 3 TimeSeries (ns, ew, vt) → SeismicRecording3C
2. **Preprocess** — detrend, filter, split into time windows
3. **Process** — FFT each window, smooth spectra, compute H/V ratio
4. **Reject** — frequency-domain outlier rejection (optional)
5. **Statistics** — mean/std of f0 and curves (lognormal recommended)
6. **SESAME** — evaluate reliability and clarity criteria

## Smoothing Operators

| Operator | Default Bandwidth | Notes |
|----------|-------------------|-------|
| konno_and_ohmachi | 40 | Log-space, most commonly used |
| parzen | 0.5 Hz | Linear-space |
| savitzky_and_golay | 9 points | Polynomial |

## SESAME (2004) Quality Criteria

**Reliability** (all 3 must pass):
1. f0 > 10 / window_length
2. Significant cycles nc > 200
3. Standard deviation below threshold

**Clarity** (5 of 6 must pass):
1-2. H/V drops to A0/2 on both sides of f0
3. A0 > 2.0
4-6. Statistical stability of f0 and A0

## Edge Cases

- **Short recordings**: Need at least 1 full window; longer records give
  better statistics. Minimum ~10 windows recommended.
- **No clear peak**: f0 still returned but SESAME criteria will fail.
- **Flat sites**: HVSR ≈ 1.0 everywhere, no resonance detected.
- **Numba JIT**: First call triggers compilation; subsequent calls fast.
