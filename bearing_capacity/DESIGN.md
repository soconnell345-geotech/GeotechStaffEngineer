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
- Vesic load inclination factors (Vesic 1975, as EN 1997-1:2004 Annex D.4
  p. 159 prints them; GEC-6 Sec. 5.2.3.7 omits inclination factors where
  sliding is checked separately) take H/(V + A' c cot φ). With c = 0 that
  is H/V = tan β and V cancels, so the angle alone gives them. With
  cohesion (or φ = 0) an inclined load needs `vertical_load` (kN; kN/m for
  a strip) and the analysis raises without it (GM-203; before, it fell
  back to Meyerhof's angle-only factors with a warning, BC-2). A' is B' per
  metre for a strip and the effective area otherwise (a strip took
  B × 1e6 m², so its inclination factors were 1). H is taken along B
  (m = m_B). The result's `choices` state the factors and what they took.
  For φ = 0 the module keeps Vesic's i_c = 1 − m H/(A' c N_c), which EN
  1997-1 does not print: its D.3 (p. 158) and UFC 3-220-20 Table 5-7
  (p. 295) give i_c = [1 + √(1 − H/(A' c))]/2, listed with its value in
  the choices' alternatives.
- Depth factors (GM-197), `depth_factor_method`: "hansen" (default)
  takes Brinch Hansen's k = Df/B for Df/B < 1 and arctan(Df/B) from 1
  (UFC 3-220-20 Tables 5-7 and 5-8 note B, pp. 295-296; GEC-6 Table 5-4
  prints d_q at Df/B = 1 from the arctan too; until GM-197 the Vesic
  factors took k = 1 there); Meyerhof's factors take k = Df/B up to and at
  Df/B = 1 as UFC Table 5-6 prints it, Hansen's arctan beyond. "gec6"
  (Vesic only) is GEC-6 Eq. 5-14 and Table 5-4: d_q alone, from the
  arctan for Df/B ≥ 1 (all 12 printed cells to their 0.05 rounding),
  1.0 below (the worked example on p. B-67). "none" uses no depth factors
  (GEC-6 Sec. 5.2.3.3 and Table 5-4 note). The result's `choices` state
  the factors and what the other options give.
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
- `ngamma_method` now defaults to `None`, resolved to the factor method's
  own N-gamma (`OWN_NGAMMA`: `"vesic"`, `"meyerhof"` since GM-188, or
  `"kumbhojkar"` for Terzaghi); the Terzaghi sources are refused with the
  Vesic/Meyerhof factor sets and vice versa.
- **Misprints found in the published tables** (pinned in `test_terzaghi.py`):
  Das Nc at 18° printed 15.12 (formula 15.52; neighbours 14.60, 16.56);
  UFC 3-220-20 Table 5-2 Terzaghi Nc 18° = 15 (Das's misprint carried over),
  Nq 27° = 17 (formula 15.90), Nc 44° = 160 (151.95), Nq 49° = 350 (344.6).
- Check value: 2 m square, Df = 1.5 m, φ′ = 25°, c′ = 20 kPa, γ = 16.5 kN/m³ →
  qult = 653.4 + 314.8 + 110.1 = 1,078 kPa (hand calculation from the
  tabulated factors).

## Allowable loads on the footing (2026-10-19, BACKLOG GM-170)

`q_allowable` is the GROSS allowable pressure q_ult / FS (UFC 3-220-20
Sec. 5-2.4 Eq. 5-13). Waves w12 and w13 (F18) multiplied it by the footing
area and called the product "an allowable column load" (1,545 kN on a 2 m
square footing 1 m deep); that product is the total load on the bearing
surface, Q_DL+LL + W_F + W_S (Eq. 5-1, p. 275), the footing's and the
soil's weight in it. The result now also gives:

- `q_allowable_net` = q_ult / FS - q (Eq. 5-14, p. 279), q the overburden
  at the base the bearing equation used;
- `allowable_loads()` (`allowable_load` in `to_dict()`): the structural
  load Q_DL+LL = q_all,net x A' (Eq. 5-3, p. 276: the foundation and the
  soil over it taken at the soil's unit weight) and the gross load
  q_all x A', each named, on the effective area A' = B'L' the analysis
  set (`area_m2`; a strip's width B', per metre run), with
  `results.LOAD_BASIS`. F18: 1,469 kN structural, 1,545 kN gross.
  Under an eccentric load the structural load is q_all x A' - q x A: the
  load bears on A' (Eq. 5-10, p. 277) and the footing's and the soil's
  weight on the whole footing A (`area_full_m2`), so q_all,net x A' would
  overstate it by q (A - A').

GEC-6 Sec. 5.2.2 (p. 53) recommends the allowable bearing capacity on the
gross pressure, the footing and backfill weight counted in the applied
load, and Sec. 6.3.1 (p. 113) notes a footing heavier than the soil it
replaces; the basis quotes both. The calc package prints the two steps
after q_all. No number the result gave before changes.

## Each method with its own N-gamma (2026-10-19, BACKLOG GM-188)

`factor_method="meyerhof"` used to take Vesic's N-gamma, 2(Nq + 1) tan phi,
because `ngamma_method` defaulted to `"vesic"` for both factor sets. Wave
w14 (F18: 2 m square, Df = 1 m, phi = 32 deg, gamma = 19 kN/m3, c = 0)
reported a Meyerhof q_ult of 1,466 kPa; Meyerhof's own N-gamma gives 1,241.
Each method now takes its own N-gamma unless `ngamma_method` names another,
each checked against its printed source:

| factor_method | N-gamma | printed |
|---|---|---|
| vesic | 2 (Nq + 1) tan phi, Caquot and Kerisel (1948) | GEC-6 Eq. 5-8 (p. 53): "currently used by FHWA and AASHTO"; Table 5-1 (p. 54) |
| meyerhof | (Nq - 1) tan(1.4 phi) | UFC 3-220-20 Table 5-6 (p. 294); Table 5-2 (p. 289) tabulates it, 22 at 32 deg; Example B-1.6 (p. 603) 53.27 at 37 deg |
| terzaghi | Kumbhojkar (1993), as tabulated in Das | unchanged; UFC 3-220-20 Table 5-2 (p. 289) prints Coduto et al.'s (2016), `ngamma_method="coduto"` (its note: Terzaghi gave N-gamma at 0, 34 and 48 deg only) |

Hansen's 1.5 (Nq - 1) tan phi stays an option. Another N-gamma than the
method's own is a mixed method, and the result's choices say so.

**Meyerhof shape factors under an inclined load are 1.** UFC 3-220-20
Table 5-6 note A (p. 294): "Do not combine factors for shape and inclined
load"; Sec. 5-3.3.2: "it is safe to assume that sc = sq = sgamma = 1 if load
inclination is present". The Vesic set is unchanged: GEC-6 Sec. 5.2.3.7
(p. 62) finds shape and inclination factors applied together "overly
conservative" and omits the inclination factor, with a sliding check, but
keeps no rule for its shape factors under an inclined load.

**Not changed, stated in the choices:** the depth parameter k. Table 5-6
prints k = Df/B for Meyerhof; the module keeps Hansen's (1970) bound,
k = arctan(Df/B) beyond Df/B = 1, for both factor sets (the smaller depth
factors).

**The result's `choices`** (`geotech_common.choices`, in `to_dict()`, the
summary and the calc package): the bearing capacity method; its N-gamma,
with the other N-gammas as alternatives; the shear mode (general, or
Terzaghi's local shear; UFC 3-220-20 Sec. 5-3.1.3 and Table 5-4, p. 291);
the allowable pressure's basis (gross q_all = q_ult / FS, Eq. 5-13, with the
net beside it, Eq. 5-14; GEC-6 Sec. 5.2.2 p. 53 recommends the gross); and,
for Meyerhof, k beyond Df/B = 1 and the shape factors under an inclined
load.

**Release note:** with `factor_method="meyerhof"` and phi below about
46 deg, where Meyerhof's N-gamma is the smaller, q_ult falls (above 46 deg
it rises). F18 (2 m square, Df = 1 m, phi = 32 deg, gamma = 19 kN/m3):
q_ult 1,465.9 -> 1,241.0 kPa (-15 %); a 2 m strip at phi = 30 deg:
842.4 -> 703.4 kPa (-17 %). With a load inclined 10 deg the shape factors
also drop to 1: F18 894.9 -> 594.9 kPa (-34 %; the N-gamma alone takes
it to 788.6, the shape factors the rest). Pass `ngamma_method="vesic"` for
the old mixed method under a vertical load; under an inclined load no input
restores the old combination of shape and inclination factors (675.1 kPa
with Vesic's N-gamma). Vesic and Terzaghi results do not change.

## Two printed oddities (2026-10-19, BACKLOG GM-214)

**d_c against UFC 3-220-20 Table 5-8 (p. 296).** The table prints
d_c = 1 + k (1 - sin phi')^2 (N_q/N_c). Its note C says the c factors come
from the q factors by the correspondence formula (Brinch Hansen 1961, de
Beer 1970), x_c = x_q - (1 - x_q)/(N_c tan phi'), which from the printed
d_q gives d_c = 1 + 2 k (1 - sin phi')^2 (N_q/N_c): twice the increment
the table prints. The module keeps the note C form (no number changes),
for three printed reasons: the table's own s_c = 1 + (B/L)(N_q/N_c) cos phi'
and i_c = (i_q N_q - 1)/(N_q - 1) are that formula exactly; note C names it
as the source of the c factors; and as phi' goes to 0 it tends to
1 + 2k/5.14 = 1 + 0.39 k, Table 5-7's 1 + 0.4 k, where the printed form
tends to 1 + 0.19 k. The depth choice's basis says so, and its
alternatives give the printed value for the footing
(`factors.ufc_table_5_8_printed_dc`; not an option of the tool). The calc
steps cite note C for d_c. GEC-6 prints no d_c (Table 5-4 is d_q alone).

**Meyerhof's i_gamma and theta < phi'.** `max((1 - beta/phi)^2, 0)` never
clamps: past phi it grows again (beta = 2 phi gives 1). UFC 3-220-20
Table 5-6 note B (p. 294) takes theta = atan(H/V) < phi', so beta >= phi is
refused with that reason (phi = 0: N_gamma = 0 and i_gamma stays 1). Under
Vesic's factors, which stand to 45 deg with c = 0, the Meyerhof
alternative in the choices reads "none" there. Phi 32 deg, 2 m square,
Df 1 m, c 0: q_ult was 199.4 kPa at 32 deg (i_gamma 0), 176.7 at 40 and
496.2 at 64; now refused. Below phi nothing changes (215.2 kPa at 30 deg).

**The other factors that turned back up**, now refused with a plain
message: a load inclination below 0 or from 90 deg (Vesic at -10 deg gave
i_q 1.28 and q_ult 1,555.9 kPa against 1,158.8 vertical); a ground slope
below 0 or above 45 deg, where Vesic's g_q = (1 - tan omega)^2 is 0 and
grows again (621.0 kPa at 60 deg against 0.0 at 45; 1,603.4 at -10); a
base tilt with alpha tan phi above 1 (radians), where b_q is 0 and grows
again (GEC-6 Sec. 5.2.3.5, p. 58, keeps inclined footings to about 8-10
deg). A negative base tilt stays: GEC-6 Table 5-5 takes alpha upward +.
Vesic's load inclination factors already refused H > V + A' c cot phi.

**Release note:** `factor_method="meyerhof"` with a load inclined at phi
or more from vertical is refused (it gave a capacity that rose again with
the angle); so are a negative or 90-degree load inclination, a ground slope
outside 0-45 deg and a base tilt past alpha tan phi = 1. No other result
changes.
