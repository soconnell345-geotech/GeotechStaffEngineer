# bearing_capacity — Shallow Foundation Bearing Capacity

## Purpose
Computes ultimate and allowable bearing capacity for shallow footings using
the general bearing capacity equation with Vesic (1973), Meyerhof (1963), or
Hansen (1970) factor sets, matching CBEAR program output; or Terzaghi (1943)
with his own factors (`factor_method="terzaghi"`, below).

## References
- Vesic (1973) — general bearing capacity factors Nc, Nq, Ngamma
- Meyerhof (1963) — shape/depth/inclination factors
- Hansen (1970) — base/ground inclination factors
- Terzaghi (1943), *Theoretical Soil Mechanics* — Terzaghi's factors
- Kumbhojkar (1993), J. Geotech. Eng. 119(3) 598-607 — Terzaghi's Nγ evaluated
- UFC 3-220-20 (2025) *Foundations*, Ch. 5, Tables 5-2 to 5-5
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
- Tests in `tests/test_bearing_capacity.py` and `tests/test_terzaghi.py`

## Terzaghi (1943) — `factor_method="terzaghi"` (2026-10-14, BACKLOG GM-30)

    strip     qult = c·Nc       + q·Nq + 0.5·γ·B·Nγ
    square    qult = 1.3·c·Nc   + q·Nq + 0.4·γ·B·Nγ
    circular  qult = 1.3·c·Nc   + q·Nq + 0.3·γ·B·Nγ     (B = diameter)

- **Nc, Nq** closed form: Nq = a²/(2cos²(45°+φ/2)), a = e^((3π/4−φ/2)tanφ);
  Nc = (Nq−1)cotφ, 1.5π+1 = 5.7 at φ = 0. They match Das's table to 0.5 % at
  every degree 0-50 (except its Nc misprint at 18°, see below) and every
  Terzaghi entry of UFC 3-220-20 Table 5-2 to its rounding (four misprints).
- **Nγ** — Terzaghi gave it only as a chart (values at φ = 0, 34, 48°; UFC
  Table 5-2 note), so every "Terzaghi Nγ" is someone's evaluation. Two cited
  options, `ngamma_method`:
  - `"kumbhojkar"` (**default**) — Kumbhojkar (1993), Terzaghi's own
    mechanism solved numerically, as tabulated at 1° steps 0-50° in Das,
    *Principles of Foundation Engineering* (the table of Terzaghi's bearing
    capacity factors). Log-linear between whole degrees; refuses φ > 50°.
    Chosen as default because it is Terzaghi's Nγ rather than a fit to it.
  - `"coduto"` — Coduto et al. (2016), Nγ = 2(Nq+1)tanφ/(1+0.4 sin 4φ), the
    basis of UFC 3-220-20 (2025) Table 5-2's Terzaghi column.
  The two differ by 5 % at φ = 30° (19.13 vs 20.1) and by ~2× below 10°. Every
  Terzaghi result carries the other source's value in `Ngamma_alternatives`
  and a note naming the one used.
- **Provenance.** The UFC table was read from the PDF in the reference library
  (`geotech-references/docs/ufc_3_220_20_2025.pdf`, p. 289) and is pinned at
  every printed row. The Das/Kumbhojkar column was transcribed from the
  table as published in Das and cross-checked row by row (51/51 rows, Nc, Nq
  and Nγ) against an independent reproduction of the same table; it has
  **not** been checked against the book or the 1993 paper in hand —
  wiki-wishlist item (Das's table; Kumbhojkar 1993 Table 1).
- **Shape** (UFC Table 5-5): strip 1/1/1, square 1.3/1/0.8, circular 1.3/1/0.6
  (sγ multiplies 0.5γBNγ, giving 0.4 and 0.3). Rectangles: Terzaghi gave
  none; UFC Sec 5-3.3.1 allows interpolating between square and continuous,
  done linearly in B/L (sc = 1+0.3B/L, sγ = 1−0.2B/L) with a note. An
  eccentric square (B′ ≠ L′) is treated the same way.
- **No other factors** (depth, inclination, base, ground = 1; UFC Sec
  5-3.3.1). Inclined load, ground slope and base tilt **raise** (UFC Table 5-3:
  do not use Terzaghi for them); Df/B > 1 adds a note (Table 5-3: Df/B ≤ 1).
- **Local shear** (`failure_mode="local"`, Terzaghi only): c* = 2/3·c,
  φ* = atan(2/3·tanφ) (UFC Table 5-4 prints 0.67; "probably too conservative
  for sands"). `c_used_kPa` / `phi_used_deg` report the reduced values.
- `ngamma_method` now defaults to `None`, resolved to `"vesic"` (or
  `"kumbhojkar"` for Terzaghi); the Terzaghi sources are refused with the
  Vesic/Meyerhof factor sets and vice versa. Vesic/Meyerhof results are
  unchanged (no new keys in `to_dict()`).
- **Misprints found in the published tables** (pinned in `test_terzaghi.py`):
  Das Nc at 18° printed 15.12 (formula 15.52; neighbours 14.60, 16.56);
  UFC 3-220-20 Table 5-2 Terzaghi Nc 18° = 15 (Das's misprint carried over),
  Nq 27° = 17 (formula 15.90), Nc 44° = 160 (151.95), Nq 49° = 350 (344.6).
- Check value: 2 m square, Df = 1.5 m, φ′ = 25°, c′ = 20 kPa, γ = 16.5 kN/m³ →
  qult = 653.4 + 314.8 + 110.1 = 1,078 kPa (hand calculation from the
  tabulated factors).
