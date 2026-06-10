# bearing_capacity — Shallow Foundation Bearing Capacity

## Purpose
Computes ultimate and allowable bearing capacity for shallow footings using
the general bearing capacity equation with Vesic (1973), Meyerhof (1963), or
Hansen (1970) factor sets, matching CBEAR program output.

## References
- Vesic (1973) — general bearing capacity factors Nc, Nq, Ngamma
- Meyerhof (1963) — shape/depth/inclination factors
- Hansen (1970) — base/ground inclination factors
- FHWA GEC-6 (Shallow Foundations)
- NAVFAC DM-7.01 Ch. 4 / Bowles Sec. 4-7 — two-layer load-spread method

## Files
- `capacity.py` — `BearingCapacityAnalysis` dataclass with `.compute()`
- `factors.py` — Nc, Nq, Ngamma, shape/depth/inclination/base/ground factor functions
- `footing.py` — `Footing` geometry (shape, eccentricity, effective dimensions)
- `soil_profile.py` — `SoilLayer`, `BearingSoilProfile` (1 or 2 layers, GWT)
- `results.py` — `BearingCapacityResult` with summary()/to_dict()
- `concrete_design.py` — structural checks for the footing
- `calc_steps.py` — calc-package presentation layer

## Public API (BC-5: matches `__init__.py` exports)
```python
from bearing_capacity import (
    Footing, SoilLayer, BearingSoilProfile,
    BearingCapacityAnalysis, BearingCapacityResult,
)

footing = Footing(width=2.0, depth=1.5, shape="square")
soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30, unit_weight=18.0))
result = BearingCapacityAnalysis(footing=footing, soil=soil).compute()
```
There is no `analyze_bearing_capacity(B, L, D, ...)` convenience function;
the class-based API above is the whole public surface (the funhouse adapter
wraps it).

## Key Notes
- Two-layer support uses the load-spread (projected-area) method
  (NAVFAC DM-7.01 / Bowles), bounded between the two single-layer capacities
- **Two-layer `thickness` convention (BC-4):** `layer1.thickness` is the
  upper-layer thickness BELOW the footing base (footing base → layer
  interface, the H of the load-spread method) — not the full layer
  thickness from the ground surface. The overburden above the footing base
  is computed with `layer1.unit_weight` (layer 1 implicitly extends from
  the ground surface). See `BearingSoilProfile` docstring.
- GWT handling: overburden uses effective stress above the base; the
  self-weight (Nγ) term averages the effective unit weight over depth B
  below the base, so a GWT inside the bearing wedge is captured (BC-3)
- Vesic inclination factors fall back to angle-based Meyerhof factors with
  a UserWarning when `vertical_load` is not supplied (BC-2)
- `BearingSoilProfile.effective_unit_weight()` was removed (BC-6, unused);
  use `geotech_common.water.effective_unit_weight` or
  `gamma_below_footing()` instead
- calc_steps displays the factor equations actually used, including the
  φ = 0 forms (sc = 1+0.2·B/L, dc = 1+0.4k) and the Vesic d_c formula (BC-9)
- Tests in `tests/test_bearing_capacity.py`
