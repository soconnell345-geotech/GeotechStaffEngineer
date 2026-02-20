# GSTools Agent — Design Notes

## Purpose

Wraps the GSTools library for geostatistical analysis of soil properties:

- **Kriging** — spatial interpolation of measured soil properties (SPT N,
  Vs, friction angle) from borehole/CPT locations onto a regular grid
- **Variogram** — estimate spatial correlation structure of soil properties
- **Random fields** — generate spatially correlated random fields for
  probabilistic analysis (Monte Carlo, reliability)

## Architecture

```
gstools_agent/
  __init__.py              # exports 3 analyze functions + 3 result classes
  gstools_utils.py         # has_gstools(), import_gstools()
  kriging.py               # analyze_kriging()
  variogram.py             # analyze_variogram()
  random_field.py          # generate_random_field()
  results.py               # KrigingResult, VariogramResult, RandomFieldResult
  tests/
    test_gstools_agent.py
  DESIGN.md
gstools_agent_foundry.py   # Foundry wrapper (project root)
```

## Key Design Decisions

1. **2D only** — soil properties are typically mapped in plan view (x, y).
   Depth variation is handled by separate analyses per depth layer.

2. **Available covariance models**: Gaussian, Exponential, Matern,
   Spherical, Linear, Stable, Rational, Cubic, HyperSpherical.

3. **Auto variogram fitting** — by default, kriging fits a variogram
   model to the data before interpolation.

4. **Grid auto-bounds** — if grid extent not specified, uses data
   bounds with 10% buffer.

## Geotechnical Applications

1. **Site characterization** — interpolate SPT N-values between boreholes
   to create spatial maps of soil stiffness.

2. **Vs30 mapping** — krige Vs measurements from MASW/HVSR tests.

3. **Probabilistic analysis** — generate random fields of friction angle
   or undrained strength for slope stability Monte Carlo.

4. **Data quality** — kriging variance shows where additional data
   would reduce uncertainty most (guide future borings).

## Edge Cases

- **Nugget effect**: Non-zero nugget handles measurement noise.
  Kriging won't pass exactly through data points with nugget > 0.
- **Extrapolation**: Kriging variance increases rapidly outside data
  coverage. Results should be masked beyond data extent.
- **Anisotropy**: Not exposed in this wrapper (GSTools supports it
  but adds complexity). Use isotropic models for simplicity.
