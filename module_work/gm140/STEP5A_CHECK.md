# GM-140 step 5a: pore pressure as printed

Branch `claude/gm-140-step5a-pore-pressure`, from master at aa0458b (steps 1
and 2 merged; `STEP1_CHECK.md` and `STEP2_CHECK.md` are on master; origin's
master has since moved on to c3ad788 without touching `slope_stability/`).
2026-11-13. Brief: `module_work/cloud_briefs/SLOPE_GM140.md`, section
"Step 5a"; account departures D2, D3, D4 and D16's slicing half; owner call
OWN-104.

**The rule.** The pore pressure on a slice base is the printed one for the
kind of water input and the kind of strength the layer has, from one
function that every caller uses: `slope_stability/pore_pressure.py`,
`base_pore_pressure`, called by `slices.build_slices`.

## 1. What the print says

Read in the reference library (geotech-references, `fhwa_soils_foundations`
chapter 6 and `dm7_1` chapter 7):

- **FHWA NHI-06-088 p. 6-14, Eq. 6-16.** "When the water table is sloping,
  use Equation 6-16 to calculate the water pressure on the base of the
  slice: u = hw γw cos² αw", "αw = slope of water table from horizontal",
  "hw = depth from ground water surface to the centroid point on the
  circle".
- **FHWA NHI-06-088 p. 6-14, Step 3.** "locate the vertical sides of the
  slices so that the bottom of any one slice is located entirely in a single
  soil layer or at the intersection of the ground water level and the
  circle. Locate the top boundaries of vertical slices at breaks in the
  slope."
- **FHWA NHI-06-088 Sec. 6.3, pp. 6-7 to 6-8 (Fig. 6-4, Eqs. 6-6 to 6-10).**
  The infinite slope with parallel seepage: the figure's pore-water force on
  a slice base is U = γw b h cos β (so u = γw h cos² β on a base b / cos β
  long), and "the FS for the dry material is reduced by the factor γ'/γsat
  for saturated cohesionless materials" (Eq. 6-10, c' = 0). The library's
  text carries the words and the figure's labels; Eqs. 6-6 to 6-10 are
  images there, so the closed form is taken from the sentence quoted and
  from Eq. 6-5's tan φ / tan β, which the same page names.
- **UFC 3-220-10 Sec. 7-5.1, p. 355.** "The piezometric surface is a
  surface connecting the height or elevation that water would rise in a
  series of standpipe piezometers." Methods of entering water: "(1) Water
  table or phreatic surface – can be corrected to approximate the
  piezometric surface based on the slope of the surface above the specific
  point, (2) Piezometric surface, (3) Finite element seepage analysis,
  (4) Point-by-point entry in x-y coordinates with subsequent interpolation
  between points, and (5) Pore pressure coefficient, ru."

From the account's quotations (the owner's copy; not in a repository):

- **EM 1110-2-1902 2-2c, p. 2-4; Table 2-1, p. 2-2.** Pore pressure is set
  to zero in total-stress analyses. Table 2-1 note 2: total-stress envelopes
  with φ > 0 apply only to partly saturated soils.
- **EM 1110-2-1902 C-7, p. C-29.** ru = u / (γ z), for a uniform slope.

No printed rule for ru in a layered column was found (the account says the
same); see "Calls made".

## 2. The departures, reproduced

`slope_stability/tests/test_pore_pressure_as_printed.py`, 38 test functions
(47 cases), all values made up. Copied onto the base commit when it held 43
cases, 36 failed there (the other 7 are statements that already held, such
as a uniform slope's ru); the tests that use only the old inputs fail on VALUES: ru in a
two-soil column, u on an undrained base under water, the pond's edge
missing from the slice boundaries, a pond thrust under a line at the
ground.

## 3. What changed, by the brief's items

1. **The water line's kind (D2, OWN-104).** New `SlopeGeometry.
   water_line_kind`: `'piezometric'` (default; the full vertical head, never
   corrected) or `'water_table'` (u = hw γw cos² αw). αw is the slope of the
   line's stretch above the base's mid-point (`pore_pressure.
   water_line_slope`); level beyond the line's ends. Where the line is above
   the ground at the slice (ponded water) nothing is corrected. A result
   with a line that slopes below the ground has a `choices` entry "pore
   pressure under the sloping water line": the reading, given or default,
   the premise (seepage parallel to a straight water table), that the
   uncorrected head is the lower factor of safety, and the other reading. A
   level line has no entry.
2. **ru in a layered column (D3).** u = ru × the total overburden stress at
   the base, which is the slice's soil weight per unit width, the column
   step 1 weighs (`column_at`). It was ru × γ(base layer) × the column's
   height. The result says the layered form is the module's reading of EM
   C-7 (`choices`, "pore pressure from ru").
   - **A surcharge is not counted.** EM C-7 prints γ z, the soil's weight.
   - **Ponded water is not counted** either, and cannot matter: ru is used
     only where the water line gives no pressure at the base, and a base
     under a pond is under the line.
3. **Total-stress layers (D4).** No pore pressure on the base of a slice
   whose layer is Mohr-Coulomb `analysis_mode='undrained'`, `'shansep'` or
   `'anisotropic'`, whatever the water input (line, ru, grid). `Slice.
   water_pressure` keeps the water pressure there; the SHANSEP and
   Hoek-Brown stresses still use it (D40, out of scope). Every such layer
   has φ = 0 at its base, so no factor of safety moves: what changes is the
   u column of the slice table and the GLE's "base tension" note. A
   total-stress layer given φ > 0 that lies under water (or was given ru, or
   sits under a pore-pressure grid) draws a warning citing Table 2-1 note 2
   and saying the φ given is not used. The module has never used φ on an
   undrained layer (`shear_strength_params` returns (cu, 0)); that is not
   changed.
4. **The pond's edge (D16).** `pore_pressure.pond_edges` gives the x where
   the water line comes down to the ground; `slices.slice_breaks` adds them,
   so circles and polylines are both cut there. The slices' pond weights now
   add up to the water over the surface at any slice count.
5. **Every method, family and caller**: section 6.

**Found on the way, and fixed: the pond thrust under a line at the ground.**
Pinning the correction to FHWA's infinite slope gave 1.15 where the print
gives 0.67. The cause was not in the pore pressure. The ponded-water code
took the pool's depth at BOTH sides of a slice from the line's elevation at
the slice's middle. That is right under a level pool and wherever the line
is above the ground at both sides. But a sloping line AT or just under the
ground (a water table at the surface of a face, a common input) read as a
pool half a slice's rise deep at the slice's lower side, and its thrust
buttressed the slope: the long plane read OMS 0.860 and Spencer 1.103 where
statics gives 0.501 for the full head. `slices._pond_thrust` now takes the
depth at each side from the line there and loads only the wetted rise;
under a level pool it is the old expression exactly. This sits in the same
lines as D16 (where the pond is), it is unsafe-side, and it blocked the
printed benchmark, so it is fixed here rather than noted. It is not in the
account; the lead may want it recorded as its own departure.

## 4. Before and after on worked examples

Made-up values. Base = aa0458b.

**The printed case (FHWA NHI-06-088 Eq. 6-10, p. 6-8).** β = 26.57° (2H:1V),
φ' = 34°, c' = 0, γsat = 19.5 kN/m³, water table at the surface; a plane
390 m long, 0.5 m deep, 60 slices. Printed: (19.5 − 9.81) / 19.5 × tan 34° /
0.5 = **0.6704**. By hand for one slice: u = 9.81 × 0.5 × 0.8 = 3.924 kPa.

| | OMS | Janbu | Spencer | M-P | Closed form |
|---|---|---|---|---|---|
| base (no option; full head) | 0.8599 | 1.1025 | 1.1032 | none | 0.5007 with u = γw h |
| branch, `'piezometric'` (default) | 0.5007 | 0.5012 | 0.5014 | 0.5009 | 0.5007 |
| branch, `'water_table'` | 0.6704 | 0.6709 | 0.6710 | 0.6706 | **0.6704 printed** |

**A sloping line under a face, a circle.** 2H:1V face 10 m high, c' = 5 kPa,
φ' = 30°, the line rising 1 in 2 under the face (cos² αw = 0.8), circle
(24, 34), R 25.

| | OMS | Bishop | Janbu | Spencer | M-P |
|---|---|---|---|---|---|
| base | 1.2093 | 1.3367 | 1.3189 | 1.3416 | 1.3415 |
| branch, default | 1.2093 | 1.3367 | 1.3189 | 1.3416 | 1.3415 |
| branch, `'water_table'` | 1.2867 | 1.4147 | 1.3936 | 1.4184 | 1.4183 |

One base by hand, 4 m below the line: 9.81 × 4 = 39.24 kPa as a piezometric
line; 39.24 × 0.8 = 31.39 kPa as a water table.

**ru in two soils.** 16 kN/m³ fill over 20 kN/m³ clay with ru = 0.3, the
same circle. One column by hand, 3 m of fill over 2 m of clay: u = 0.3 ×
(48 + 40) = 26.4 kPa (it was 0.3 × 20 × 5 = 30.0).

| | OMS | Bishop | Janbu | Spencer | M-P |
|---|---|---|---|---|---|
| base | 1.2733 | 1.3939 | 1.3825 | 1.3836 | 1.3829 |
| branch | 1.3060 | 1.4281 | 1.4150 | 1.4168 | 1.4161 |

**A partly ponded face** (pool at el. 14 on the same face), Bishop on 20 /
30 / 60 slices: base 1.6370 / 1.6372 / 1.6380, branch 1.6360 / 1.6372 /
1.6380. The 30- and 60-slice layouts already had a boundary at the edge.

## 5. The validation set (account section 11): before, after, printed

`module_work/gm140/step2_section11.py` was run unchanged on both trees: of
its 66 lines only V-031's differ (pinned circle Bishop 1.4205 -> 1.4204, GLE
1.4091 -> 1.4090; Spencer search 1.4174 -> 1.4172), and run times. It recomputes the benchmarks step 2 moved,
so for this step `module_work/gm140/step5a_section11.py` (new) covers the
rest another way: it runs the tests that pin the set (`slope_stability/
tests` and every `validation_examples/test_published_*` that imports the
module) and records every factor of safety those tests compute, on the base
and on the branch. Nothing is re-entered by hand.

**37,043 values recorded on each tree; 116 differ.** 42 of the 116 are the
slice-count sweeps of V-031 (each ±0.0001 to ±0.0004). All the others are
in the table. Every row of section 11 not listed here is unchanged to the
fourth decimal.

With the default (no correction), what can move a value: (a) the pond's
edge as a slice boundary; (b) the pond thrust where a line is at or under
the ground; (c) ru in a layered column; (d) total-stress bases (no row: φ =
0 on all of them).

| Ref | Benchmark | Before | After | Printed | Which part moved it |
|---|---|---|---|---|---|
| B1 | F&K circle, dry: OMS / Bishop / Janbu / Spencer / M-P | 1.9271 / 2.0755 / 2.0182 / 2.0745 / 2.0764 | same | 1.928 / 2.080 / 2.041 / 2.073 / 2.076 | none (no water) |
| B1 | F&K circle, ru = 0.25: OMS / Bishop / Spencer / M-P | 1.6067 / 1.7590 / 1.7633 / 1.7651 | same | 1.607 / 1.766 / 1.761 / 1.764 | none: one soil, ru × overburden = ru γ h |
| B2 | F&K composite, dry: OMS / Spencer / M-P | 1.3132 / 1.3858 / 1.3814 | same | 1.288 / 1.373 / 1.370 | none |
| B2 | F&K composite, ru = 0.25: Spencer / M-P | 1.1381 / 1.1322 | same | 1.118 / 1.118 | none: both soils 120 pcf, so (c) changes nothing |
| B3 | ACADS 1(a) search: Bishop / Spencer / GLE / Janbu | 0.9851 / 0.984 / 0.984 / 0.983 | same | 1.00 | none |
| B8 | V-029 grid against the line; ACADS 5 demonstration | 1.7630, 1.7625; 1.2471 | same | not pinned to print | none: the grid governs the base, as before |
| B9 | V-026 ACADS 1(b), crack: strength 1.4845 / 1.4861 / 1.4816 | same | same | Slide2 1.596 / 1.592 / 1.592 | none (step 4's) |
| B10 | V-040 pile row, active | 1.2232 | same | 1.193 | none |
| B11 | V-028 ACADS 4, a piezometric surface: kept surface GLE; weak-layer search | 1.0095; 0.8301 | same | referee 0.78 | none: a piezometric line is never corrected |
| B12 | V-032 search | 1.3149 | same | 1.366 | none |
| B15 | V-030 Spencer / GLE / Janbu | 1.2181 / 1.2204 / 1.2070 | same | Slide2 1.157 / 1.160 / 1.168 | none |
| B16 | V-031 (Slide2 #57) circle, 50 slices: Bishop / Spencer / GLE | 1.4205 / 1.4247 / 1.4091 | 1.4204 / 1.4246 / 1.4090 | 1.417 / 1.422 | (a): the line meets the ground at the toe |
| B16 | V-031 Spencer search; Bishop search | 1.4174; 1.4163 | 1.4172; 1.4162 | 1.422; 1.417 | (a) |
| B16/17 | V-043 circle, Spencer; refined search; noncircular | 1.2096; 1.1859; 1.2337 | same | 1.209; 1.188 | none |
| B17 | V-042 Bishop entry/exit | 1.3335 | same | 1.334 | none |
| V-044 | Slide2 #70 submerged: Bishop / Spencer / M-P | 1.5974 / 1.5962 / 1.5933 | same | 1.603 / 1.599 / 1.599 | none: the pond covers the whole surface, no edge inside it |
| V-054 | isotropic anchor; toe circle | 2.3786; 2.2222 | same | 2.16 (FLAC 2.18) | none |
| B5 | Duncan Ex. 6 searches (submerged toe) | 1.5419 / 1.7977 / 1.6098 | 1.5418 / 1.7976 / 1.6097 | book not in hand | (a); in the third the main circle now ties the toe circle and is named |
| B7 | V-037 #95 Corps, seepage stage 1 (stage-1 FOS) | 1.3392 (2.3660) | 1.3393 (2.3662) | 1.347 | (a) |
| B7 | V-037 #95 Corps, flat pool | 1.2073 | 1.2074 | 1.347 | (a) |
| B7 | V-038 #96 DWW: flat / seepage | 1.2349 / 1.2733 | 1.2351 / 1.2733 | 1.443 | (a) |
| B7 | V-038 #96 DWW 'gle': flat / seepage | 1.3061 / 1.3699 | 1.3064 / 1.3698 | 1.443 | (a) |
| B7 | V-048 LK on #95/#96: flat / seepage | 1.3566 / 1.4502 | 1.3569 / 1.4502 | 1.443 (DWW) | (a) |
| B7 | V-041 #98 search: Corps / DWW / DWW 'gle' / LK | 0.8362 / 0.9350 / 0.9580 / 0.9739 | 0.8364 / 0.9351 / 0.9580 / 0.9742 | 0.931 / 1.039 / 1.039 / 1.075 | (a) |
| B15 | V-041 #95/#96 given surface, Corps | 1.2058 | 1.2055 | (consistency) | (a) |
| B13 | V-052 Pilarcitos, assumed 2.5:1, search: Corps / LK | 0.7861 / 1.3473 | 0.7866 / 1.3488 | 0.823 / 1.047 | (a); the search ends on a neighbouring circle |
| V-052 | assumed 2:1: Corps / DWW | 0.6610 / 1.0918 | 0.6619 / 1.0983 | 0.823 / 1.043 | (a), same |
| V-052 | assumed 3:1: Corps / LK / DWW | 0.9131 / 1.5429 / 1.5416 | 0.9141 / 1.5426 / 1.5441 | 0.823 / 1.047 / 1.043 | (a), same; **DWW now 0.1 % above LK: strict xfail, below** |
| V-052b | true section: Corps / DWW; automatic windows | 0.7218 / 1.2633; 0.7180 | 0.7223 / 1.2652; 0.7179 | 0.823 / 1.043 | (a) |

Module tests that are not section 11 rows but moved (for the record):
`test_usability_features.TestRu::test_multi_layer_different_ru` 1.4447 ->
1.4592 by (c) (two soils of different weight, ru on both; the test asserts
only a range). The other fixture values that moved did so by 0.0001 to
0.0009, by (a) or (b).

**Away from print: V-052, assumed 3:1 section, DWW against LK.** The test
asserted Corps < DWW < LK on three assumed sections. On the 3:1 section the
two searches now end on different circles: DWW 1.5441 on (41.605, 44.623),
R 33.510, against LK 1.5426 on (40.416, 44.623), R 33.995. Before, both
ended on (40.416, 44.623), R 33.985. On each of the two circles DWW is
still below LK, as it must be (stage 3 only lowers a strength): 1.5412 /
1.5426 on LK's circle, 1.5441 / 1.5461 on DWW's. Likely cause: at that
centre the 20-slice drawdown FOS is not smooth in the radius (LK: 1.5429,
1.5446, 1.5426, 1.5674 at R = 33.985, 33.990, 33.995, 34.000 m; the jump is
where the circle reaches the next boundary), the pond's edge moved the 20
slices of every circle, and each method's one-dimensional radius search
stops at a different lip of that function. That belongs to the search and
the drawdown slicing (steps 7 and 9), not to the pore-pressure rule. The
value is not re-pinned and no tolerance is widened: the test is now one
case per section, and the 3:1 case is a strict xfail with this reason; the
2:1 and 2.5:1 cases still assert the ordering.

No pinned test was re-pinned. No tolerance was changed.

## 6. Where the rule now holds

`slices.build_slices` is the only place a slice base is given a pore
pressure, and it calls `pore_pressure.base_pore_pressure`. Every caller
below takes its slices from `build_slices` (grep: `build_slices(` in
`slope_stability/`, `geo_project/`, `reliability/`, `calc_package/`,
`funhouse_agent/`):

- Every method: `methods.fellenius_fos`, `bishop_fos`; `gle.gle_fos`
  (Spencer, Morgenstern-Price, GLE), `gle.janbu_fos`, through
  `solution.solve_method`. They read `Slice.pore_pressure` and nothing else
  about water in the ground.
- `analysis.analyze_slope`, `compare_methods_table`,
  `search_critical_surface`; `search._compute_fos`, so every family: grid
  circles, entry and exit circles, the refinement walk, circles through the
  toe and above it (`search_supplements.search_toe_circles`), planes
  (`check_infinite_slope`), random, differential-evolution, particle-swarm
  and weak-layer polylines.
- Rapid drawdown (`rapid_drawdown.rapid_drawdown_fos`, `search_rapid_
  drawdown`): stages 1 and 3 build their slices with `build_slices` on the
  pool geometries, so they take the same function. The pools are level. A
  sloping stage-1 seepage line is read as `water_line_kind` says, and the
  result's `choices` names it ("Stage 1 (before drawdown): ..."). Stage 3
  still sets u = 0 on the slices it gives an undrained strength, as before.
- `newmark.yield_acceleration`; `probabilistic.fosm_fos`, `monte_carlo_fos`
  (a varied `ru` or unit weight goes through the same function), so
  `reliability.wrappers.slope_reliability`.
- `geo_project.builders` (not edited): it builds a `SlopeGeometry`, so the
  rule holds with the default reading; the project schema has no input for
  `water_line_kind` (noted, section 10).
- `funhouse_agent/adapters/slope_stability.py`: `water_line_kind` on every
  method that takes the geometry (declared, described, `allowed_values`,
  `applies_when` a water line without a grid, or a stage-1 seepage line for
  the drawdown tools); `gwt_points`' description says what the line is and
  that total-stress bases carry no pore pressure. The yield acceleration,
  FOSM and Monte Carlo results carry no `choices` of their own, so the
  adapter adds the statements to their output.
- `slope_stability/calc_steps.py`: the geometry section of a slope calc
  package prints "Pore Pressure on the Slice Bases" (what, used, basis)
  when there is something to say. The calc-package tools build their
  geometry without `water_line_kind` (they have no crack or reverse-
  curvature options either), so they run the default reading and say so.
- A pore-pressure grid given with a line: unchanged. The grid governs the
  base pressure wherever it is given; the line gives the pond's loads and
  the saturated unit weights below it. The result now says so (`choices`,
  "pore pressure on slice bases").
- Results: `SlopeStabilityResult.choices` and `RapidDrawdownResult.choices`
  (new fields, in `to_dict()` and `summary()` when not empty);
  `SearchResult.choices` gains the entries, said once by the search and
  taken off its critical surface.
- `docs/user_manual/manual.html`: the `gwt_points` row and a new
  `water_line_kind` row in the nine parameter tables (hand-edited; no
  rebuild). `slope_stability/DESIGN.md` ("Pore pressure as printed") and
  `VALIDATION.md` (B-5a).
- `funhouse_agent/tests/method_probes.json`: two probes with a sloping line
  (`analyze_slope#sloping_water_line`, `rapid_drawdown_fos#seepage_line`).

## 7. Calls made (for the owner to revisit)

1. **The names.** `water_line_kind`, with `'piezometric'` and
   `'water_table'`. UFC 3-220-10 Sec. 7-5.1 uses "piezometric surface" and
   "water table or phreatic surface" for the two.
2. **αw at a vertex of the line, and beyond its ends.** At a vertex the
   flatter of the two stretches is taken (the larger u). Beyond the line's
   ends the line is carried on level, as the geometry already does, so αw =
   0 there. FHWA and UFC say "the slope of the water table" / "of the
   surface above the specific point" and nothing about a break.
3. **ru in a layered column** is ru × the soil column's weight per unit
   width (each soil at its own unit weight, saturated below a water line),
   without surcharge or ponded water. EM C-7 defines ru for a uniform slope
   only; the result states that the layered form is the module's reading.
4. **Which layers are total-stress.** Mohr-Coulomb `analysis_mode=
   'undrained'`, `'shansep'` and `'anisotropic'`. A Hoek-Brown layer is
   taken as an effective-stress strength and keeps its pore pressure. A
   drained layer given φ = 0 with c' is still an effective-stress layer by
   its own declaration.
5. **The φ > 0 warning is about the model, not one surface.** It is raised
   when a total-stress layer given φ > 0 holds ground under the water line
   anywhere (or was given ru, or the model has a pore-pressure grid), so a
   search and a single surface say the same thing. It also says the φ is
   not used, which is the module's long-standing behaviour.
6. **The pond's edge is cut as a surcharge's edge is.** FHWA p. 6-14 names
   the water level's intersection with the circle and breaks in the slope;
   the brief reads it as covering the pond's edge, and so does this step.
7. **The pond thrust fix** (section 3) was made although the brief does not
   list it. Under a level pool nothing changes.
8. **Results that had no `choices`.** `SlopeStabilityResult` and
   `RapidDrawdownResult` got the field; the yield acceleration, FOSM and
   Monte Carlo results get the entries in the adapter's output only (their
   library result classes are unchanged).

## 8. Tests

All by pytest's exit code, Python 3.12, the brief's venv (planlens and
geotech-references from PyPI), `-n 4`.

| Run | Result |
|---|---|
| `slope_stability/tests` (all of it) and the 21 `validation_examples/test_published_*` files that import `slope_stability`, on the finished tree | **858 passed, 17 skipped, 4 xfailed, exit 0** |
| the same on the base (aa0458b) | 810 passed, 17 skipped, 3 xfailed, exit 0 |
| the difference | +47 new cases; V-052's ordering test is now three cases, two passing and the 3:1 case a strict xfail (it was one passing test) |
| the 41 `calc_package`, `funhouse_agent`, `reliability`, `fem2d`, `geotech_common`, `report_ingest`, `dxf_import` and `geo_project` test files that name `slope_stability` (`funhouse_agent/tests/test_defaults_used.py` whole), with `webapp/tests/test_page_fingerprints.py` and `webapp/tests/test_public_text.py` | 2,925 passed, 25 skipped, 1 xfailed, **2 failed, exit 1** |
| those 2 on the base | the same 2 fail there, so they are not this step's: `calc_package/tests/test_calc_package.py::TestDm7FigureHelper::test_figure_436_with_query` (a DM7 figure helper refusing its inputs) and `funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]` (`logx`, `logy`) |
| `funhouse_agent/tests/test_defaults_used.py -k slope` | 45 passed, 16 skipped, exit 0 (the skips are that file's own: the slope probes of the "changes its result" check are frozen under GM-140) |
| `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py` | 18 passed, exit 0 |

The page fingerprints pass as recorded: the adapter's parameter text is not
in them, so there is no re-record commit.

The new test file was also run against the base's code: it fails there
(section 2).

## 9. Release note

Slope stability, pore pressure (GM-140 step 5a). A water line (`gwt_points`)
is a piezometric line by default, as before: the full vertical head at each
slice base. New option `water_line_kind='water_table'` applies FHWA
NHI-06-088 Eq. 6-16, u = hw γw cos² αw, for a water table with seepage
parallel to it; it raises the factor of safety under a sloping line (a
made-up example: Bishop 1.337 to 1.415) and reproduces FHWA's infinite
slope with seepage, (γ'/γsat) tan φ' / tan β. Every result with a sloping
line names the reading it used. Answers that move with the default: (1) a
sloping water line drawn AT or just under the ground surface no longer
acts as a pool pushing on the slope, so those factors of safety are LOWER,
in the extreme case of a water table at the surface of a long sand face by
more than half (Spencer 1.10 to 0.50, where statics gives 0.50); (2) `ru`
in a column of soils of different unit weights now acts on the column's own
weight (a made-up two-soil example: Bishop 1.394 to 1.428); (3) the edge of
ponded water is a slice boundary, which moves factors of safety of partly
submerged slopes and rapid-drawdown runs in the third or fourth decimal
(up to 0.6 % where a drawdown search then ends on a neighbouring circle);
(4) slices in undrained (total-stress) layers show u = 0 in the slice
table, with no change in the factor of safety; an undrained layer given
φ > 0 under water now draws a warning. Compare a saved slope report with a
sloping water line at the ground, layered ru or ponded water with this note
in hand.

## 10. Stopped short of, and noted, not fixed

- **EM 1110-2-1902 and its pages** are quoted from the account only (2-2c,
  Table 2-1 and its note 2, C-7); the EM is in no repository.
- **FHWA Eqs. 6-6 to 6-10** are images in the library's text; the closed
  form is pinned from the page's own sentence (the factor γ'/γsat on Eq.
  6-5) and the figure's label U = γw b h cos β.
- **D40** (the stresses SHANSEP and Hoek-Brown strengths are taken at): not
  changed. They still use the water pressure at the base, and leave the
  pond's weight out.
- **How the drawdown stages match their slices** (step 9): not touched.
  `rapid_drawdown.py` gained only the `choices` field and the stage-1
  statement.
- **Step 5b** (line, inclined and varying loads, the vertical seismic
  coefficient, surcharge inertia, the Bishop-with-kh warning): not touched.
- **A total-stress envelope with φ > 0** cannot be entered: an undrained
  layer is (cu, 0). The step warns and says so; building c-φ total-stress
  envelopes for partly saturated soils (EM Table 2-1) is not in the brief.
- **A vertex of the water line is not a slice boundary.** A slice with a
  break of the line inside it takes u, and the pond's weight, from its
  mid-point. FHWA p. 6-14 does not ask for a boundary there. Small.
- **The 3:1 Pilarcitos ordering** (section 5): the radius search on a
  20-slice drawdown objective that is not smooth. Steps 7 and 9.
- **`geo_project`'s schema and the calc-package tools have no
  `water_line_kind` input** (not edited, by the brief); they run the
  default reading, and the calc package prints it.
- **`funhouse_agent/tests/test_defaults_used.py`'s "every default changes
  its result" check skips the slope probes** (frozen under GM-140, its own
  skip reason); the two new probes were run by hand through `call_agent`:
  `water_line_kind` moves Bishop 1.3368 to 1.4142 and the drawdown 1.3549 to
  1.3732.
- **The foundry wrappers** (`foundry/`, retired) build a `SlopeGeometry`
  without the option: the default reading.
- **The local `master` ref in the session's clone was stale** (1d219c8); the
  branch was cut from the checked-out HEAD, aa0458b, which is origin's
  current master with steps 1 and 2. All "before" numbers are from aa0458b.
