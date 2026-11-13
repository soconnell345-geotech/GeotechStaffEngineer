# GM-140 step 1: one material per point

Branch `claude/gm-140-step1-one-material-per-point`, from master f108ae8,
2026-11-13. The specification is `module_work/slope_integral_review/ACCOUNT.md`
(section 2, departure D1; section 12, step 1), agreed as OWN-101
(`module_work/owner_calls_2026-11-12/CALLS.md`).

## 1. What the print says

USACE EM 1110-2-1902 is not in the repository. As the account quotes it:

* Total unit weights for all soils, above and below water: one weight per
  point (EM 1110-2-1902 2-2d, p. 2-5).
* The engineer checks "the weights of slices, shear strength properties, and
  pore water pressures at the bottoms of slices" (EM 4-2d(2), p. 4-2).

No printed source gives a rule for an input in which two layers are described
over the same ground; "the first listed layer holds it" is the module's own
rule (it is the rule the strength lookup already used), stated in the tool
text, the module document and the result.

## 2. The D1 reproduction

`slope_stability/tests/test_one_material_per_point.py`, written first and run
on the unchanged code: 17 of 18 failed (the weights, and the things not built
yet). All values are made up, except V-028's, which are the benchmark's.

**Two-layer model.** Ground (0,0) (10,0) (30,10) (40,10). A fill (20 kN/m3)
over a clay (16 kN/m3); their boundary rises 1 in 5 from (0,-2) to (40,6). The
usual description: the fill has that polyline as its bottom, the clay a flat
top at 6.0 (the boundary's highest point). Slip surface (10,0) (18,-3) (30,-1)
(38,10).

* One slice, x = 24 to 25, at x = 24.5: ground 7.25, boundary 2.90, base
  -1.9167. Fill 4.35 m x 20 = 87.0; clay 4.8167 m x 16 = 77.07; **W = 164.07
  kN/m**. The module read **213.67** (the clay counted from its flat top:
  87.0 + 7.9167 x 16).
* Every slice against the same hand formula, at 12, 30 and 77 slices.
* The mass against its areas: 180.0 m2 in all, 86.638 m2 of clay and 93.362
  m2 of fill, 3,253.4 kN/m.

**V-028 (ACADS 4).** The slice from x = 47.9746 to 48.5348 on the pinned
surface: ground 28 + (12/25)(48.2547 - 43) = 30.5222, base 26.8149, h = 3.7073
m, b = 0.5602 m, W = 18.84 x 3.7073 x 0.5602 = **39.13 kN/m**; the module read
**116.0**. The sliding mass: 3,003 kN/m from its area; the module read 8,768.

## 3. The rule, and what an input means now

**The rule.** At a point (x, z) below the ground the layer is the first one
listed in `soil_layers` whose bottom and top at x contain z.

* `SlopeGeometry.column_at(x, z_low, z_high)` returns a column as stretches,
  each with the one layer that holds it. `layer_at_point(x, z)` is the same
  rule at one point (its code did not change; it was already this rule).
* Read from it: the slice weight (`slices._compute_slice_weight`), and through
  the weight SHANSEP's sigma'v, the Hoek-Brown normal estimate and the seismic
  force; the strength, `ru`, the rapid-drawdown envelope and the stabilizing
  piles' layer, which already came from `layer_at_point` at the base; the
  section drawings (`plotting._draw_section_base`,
  `calc_package.interactive.slope_interactive_figure`), which now fill each
  layer where it holds the ground; the calc package's slice table (it prints
  the slices). The pond and crack terms take nothing from the layers.
* Callers that integrate over a column all go through `build_slices`: limit
  equilibrium, the rapid-drawdown stages, Newmark, the probabilistic runs.
  None has a second weight calculation.

**What an input means.**

* A layer's `top_elevation` / `bottom_elevation` or, new,
  `top_boundary_points` / `bottom_boundary_points` (polylines sorted by x,
  level beyond their ends) say where the layer MAY be.
* A polyline bottom on the upper layer and a flat top on the lower one now
  means the lower layer starts at the boundary, not at the flat top. No
  existing input has to change.
* A layer can also be given the boundary as its top polyline. That is the
  same ground with no overlap to report.

**Overlaps.** The first listed layer wins. `SlopeGeometry.layer_overlaps()`
lists each layer that loses ground: the layers that hold it, the x range, the
tallest shared stretch below the ground, and a sentence. Every result carries
it: `layer_overlaps` on `SlopeStabilityResult`, `SearchResult` and
`RapidDrawdownResult` (in `to_dict()` only when not empty), a `LAYERS:` line
in `summary()`, a key in `compare_methods_table`, and an "Overlapping layers"
table in the calc package. The layer table there shows a sloping top or
bottom as its range.

**Gaps.** Ground between the surface and the bottom of the deepest layer that
no layer covers is refused when the `SlopeGeometry` is built: the message
gives the two elevations, the x where they were measured and the x range.
Ground above the first layer's top is such a gap. No model in the test suites
had one. Below the bottom of the deepest layer is the model's base, as before
(GM-128).

**Lenses.** A layer listed before the layer it sits in holds its own ground,
and the host is weighed around it (tested). This fell out of the rule; no
closed-region input was built.

**How gaps and overlaps are found.** The ground and every top and bottom are
straight between their vertices, so a gap or an overlap opens and closes only
where two of those lines cross. The column is checked at the ends and the
middle of every stretch between such points. A stretch shorter than 1e-6 m is
neither. The scan costs about 65 microseconds for three layers, once per
model; the weight lookup about 1 microsecond more per slice.

## 4. The validation set (account section 11): before, after, printed

**How "after" was established.** The solvers, searches, screens and slicing
were not edited. During one run of `slope_stability/tests`,
`validation_examples`, `dxf_import`, `geo_project` and `calc_package/tests`,
every slice weight was computed by the old rule and the new one side by side
(a recorder outside the repository). They differ in 20 tests:

* 8 in the new test file, and `test_polyline_layers`' two-layer geometry
  (up to 21 % per slice);
* 5 V-028 tests (67 % per slice);
* 6 V-030 tests, by 2e-15 (rounding: its sub-layer elevations do not meet to
  the last bit). No pinned digit moves.

In every other test the slice weights are the same numbers, so each pinned
value below is the same before and after, and its test passes on its
unchanged pin. Only the V-028 tests failed after the change (5), and they were
re-pinned (section 5). I did not recompute each unchanged value one by one.

| Ref | Problem | Before | After | Printed |
|---|---|---|---|---|
| B1 | F&K (1977) homogeneous, dry | OMS 1.926, Bishop 2.075, Janbu 2.017, Spencer 2.074, M-P 2.076 | same | F&K 1.928, 2.080, 2.041, 2.073, 2.076 |
| B1 | Same, ru = 0.25 | OMS 1.606, Bishop 1.758, Spencer 1.763, M-P 1.764 | same | 1.607, 1.766, 1.761, 1.764 |
| B2 | F&K composite surface on a weak layer (flat layers) | OMS 1.313, Spencer 1.386, M-P 1.381; ru 1.138, 1.132 | same | F&K 1.288, 1.373, 1.370; 1.118, 1.118 |
| B3 | ACADS 1(a), search | Bishop 0.986, Spencer/GLE 0.985, Janbu 0.989 | same | 1.00 (Slide2 0.987, 0.986, 0.990) |
| B4 | Duncan (2000) reliability | beta 2.318, pf 1.02 % | same | 2.32, about 1 % |
| B5 | Duncan, Wright & Brandon suite (19 tests) | all pass | same | book not in hand |
| B6 | ACADS 1(a), LE against fem2d SRM | LE 0.985 to 0.989; SRM 1.053 | same | 1.00 |
| B7 | #95 dam, Corps 2-stage (V-037) | 1.34 | same | 1.347 |
| B7 | #96 dam, DWW 3-stage (V-038) | 1.235 default; 1.273; 1.370 "gle" | same | 1.443 |
| B7 | #98, search (V-041) | Corps 0.837, DWW 0.938 | same | 0.931, 1.039 |
| B8 | Pore-pressure grid (V-029) | identical to the line; 1.25 | same | not pinned to print |
| B9/B16 | ACADS 1(b), crack (V-026) | strength 1.485/1.486/1.482; truncation 1.603/1.599/1.600 | same | Slide2 1.596/1.592/1.592 |
| B10 | #54 micro-pile row (V-040) | active 1.223; passive 1.203 | same | 1.193 |
| **B11** | **ACADS 4 (V-028), seam surface** | **Spencer 0.808, GLE 0.802, Janbu 0.819** | **Janbu 1.073; Spencer and GLE no solution (fallback 1.041, 1.010)** | **referee 0.78; Slide2 Spencer 0.760, GLE 0.720, Janbu 0.734** |
| **B11** | **V-028 E8, both loads** | **Spencer 0.806, GLE 0.799** | **Janbu 1.034 (crest zone alone 1.023, bench zone alone 1.085)** | **none for the loads alone** |
| **B15** | **V-028 weak-layer search, 30 / 40 slices** | **0.808 / 0.808** | **1.430 / 1.431** | **referee 0.78** |
| B12 | V-054 steep phi = 0 slope, toe circle | 2.346 (2.222) | same | 2.16 (FLAC 2.18) |
| B12 | #86 without supports | 0.942 | same | 0.942 |
| B12 | Planar family against closed forms | +0.11 to +0.15 % | same | closed form |
| B12 | #86, five tiebacks | Bishop 1.612, Spencer 1.606, GLE 1.606 | same | 1.629, 1.620, 1.622 |
| B12/B16 | #85, one support, active | GLE 1.573, Spencer 1.551, Bishop 1.372 (discard 1.508) | same | GLE 1.575, Spencer 1.884, Bishop 1.531 |
| B12/B16 | #85, passive | Bishop 1.254 (discard 1.316), GLE 1.379 | same | 1.324, 1.378 |
| B12 | V-032 (Slide2 #61) | 1.315 | same | 1.366 (Baker 1.35) |
| B13 | Duncan Ex. 1, D = 2 and 3 | 0.703, 0.676 | same | Taylor 0.706, 0.679 |
| B13 | Pilarcitos section, Corps | 0.7236 | same | 0.823 |
| B14 | Weak layer over rock, 10 m cut | 4.82 default; about 1.05 directed | same | none |
| B15 | V-030 Duncan (2000) surface | Spencer 1.218, GLE 1.220, Janbu 1.207 | same (weights differ by 2e-15) | Slide2 1.157, 1.160, 1.168; Duncan 1.17 |
| B15 | V-043 sand fill, DE | 1.234 (strict xfail) | same | Slide2 1.188, Tandjiria 1.192 |
| B15 | V-041 #95/#96, Corps search | 0.783 | same | infinite slope 0.772 |
| B16 | V-043 circle, Spencer | 1.210 | same | Slide2 1.209, Tandjiria 1.219 |
| B16 | V-031 (Slide2 #57) | 1.421/1.425/1.409 | same | 1.417/1.422 |
| B16 | Vertical cut, Bishop search | 0.842 | same | Taylor 0.851 |
| B17 | V-043 Spencer, refined | 1.1859 | same | Slide2 1.209, 1.188 |
| B17 | V-042 (Hassan & Wolff) | Bishop 1.3334 | same | 1.334 |
| B17 | Duncan Ex. 1, 4 H each side | Bishop 0.7049 | same | Taylor 0.706 |
| B18 | Surface touching the base | 0.7045 to 0.7063 | same | (consistency) |
| V-035/036 | Slide2 #79, #81 | as the test file | same | referee 1.44, 1.15 |
| V-039 | Slide2 #104, Newmark | as the test file | same | FS 1.359, 0.978; Ky 0.139 |
| V-044 | Slide2 #70, submerged | as the test file | same | 1.603, 1.599, 1.599 |
| V-045 | Slide2 #34, FOSM | 3.482 | same | none pinned |
| V-048, V-052 | Lowe-Karafiath; Pilarcitos (#97) | as the test files | same | Corps 0.823, LK 1.047, DWW 1.043 |
| (unit) | `test_polyline_layers` two-layer geometry, Bishop | 2.488 | 2.849 | none |

No flat-layer benchmark moved. The account said only B11 / V-028 has a sloping
boundary; the recorder confirms it.

## 5. V-028 and B11: what I found

**The result.** With each point weighed once, on the pinned seam surface:

* Janbu (corrected) 1.073 (0.819 before).
* Spencer and GLE (half-sine) **do not converge**. They return 1.041 and
  1.010, which are the old approximate solver's fallback values (the account's
  D23 / D30, step 2). The account's "Spencer 1.038" is this fallback (1.038
  with the piezometric line as entered before; 1.041 with Table 9.3).
* The weak-layer search, same settings, returns 1.430 (30 slices) and 1.431
  (40), a converged Spencer value on another surface.
* These are 38 % (Janbu) and 83 % (search) above the referee's 0.78.

**Inputs against the problem text** (`module_work/slope_v53/
slide2_selected_problems.txt`, manual pp. 50-54):

| Input | Problem text | As entered | Finding |
|---|---|---|---|
| Soil 1 | c' 28.5, phi' 20.0, gamma 18.84 (Table 9.1) | the same | agrees |
| Soil 2 (seam) | c' 0, phi' 10.0, gamma 18.84 (Table 9.1) | the same | agrees |
| Piezometric line | (20, 27.75) (43, 27.75) (49, 29.8) (60, 34.0) (66, 35.8) (74, 37.6) (80, 38.4) (84, 38.4) (Table 9.3) | 29.86, 34.06, 37.68 at x = 49, 60, 74 | **differs from the text; corrected to Table 9.3.** Effect +0.3 % |
| Loads | 20 kPa from (23, 27.75) to (43, 27.75); 20 to 40 kPa from (70, 40) to (80, 40) (Table 9.2) | left out of V-028; in E8 as 20 kPa on x 23-43 and a uniform 30 kPa on x 70-80 | the ramp is entered as its mean (the module has uniform loads only, D5); x ranges agree |
| Ground surface | Figure 9.1 only | (20, 28) (43, 28) (68, 40) (84, 40) | **cannot be checked.** Table 9.2 puts the bench load at y = 27.75 and Table 9.3 the water at 27.75 along the bench, which suggests the bench is at 27.75, not 28. Not changed: the text does not print the ground. With the bench at 27.75, Janbu reads 1.047 |
| Seam | Figure 9.1 only | 1 m band, top (20, 19) to (84, 37), bottom (20, 18) to (84, 36) | **cannot be checked** |
| Slip surface | Figure 9.2 to 9.4 only | the module's own search result of 2026-06 | not a published surface |
| Tension crack | "to be ignored" | none | agrees |

**What I did.** Corrected the piezometric line, the one transcription error
the text shows. Changed nothing else and tuned nothing. V-028 and B11 are
marked NOT VALIDATING, with the reason, in the two test files (a record of
the module's values, the weights checked against the area, and a strict
xfail against the referee), `VALIDATION.md` (B11, B19), `INVENTORY.md` and
`RESULTS.md`. The geometry read off the figure is the likely cause of what
remains (a seam 9 m below the bench is hard to square with 0.78), but that is
a guess: it needs Figure 9.1, which is not in the repository.

The E8 load tests moved to Janbu because Spencer does not solve on this
surface now. One thing they showed before I moved them: with the fallback
Spencer value, adding 15 kPa to the crest load raised the "FOS" from 0.9435
to 0.9452. That is the fallback, not the weights (noted below).

## 6. A sloping-boundary benchmark

**None in the repository is suitable.** Searched: the Slide2 problem text
(layer coordinates are in figures only; the printed coordinates are grid
corners and circle centres), the reference library's text (FHWA NHI-06-088
Examples 6-2 and 6-3 have level boundaries; DM 7.1 Fig. 7-4 is a caption; the
Micropile manual's section 6.7 slope has a sloping rock contact but its
geometry is in Fig. 6-24 only; GEC 13 Vol. 2 section 7.3.3.3 has level and
vertical boundaries), `worked_examples.json` and `INVENTORY.md` (the Caltrans
slice table is a single soil).

So the check added is a **consistency test only**, the made-up two-layer case
of section 2: every slice weight against a hand formula at three slice counts,
one slice by hand, the mass against its two areas, the strength and the
weight taking the same layer, and the same answer whether the lower layer's
top is flat or the polyline. Section 11 still has no published check of a
sloping boundary. The candidates worth getting are ACADS 4's own figure and
DM 7.1 Fig. 7-4.

## 7. Functions touched outside `slope_stability`

| File | Function | Change |
|---|---|---|
| `dxf_import/converter.py` | `build_slope_geometry` (only) | The layer named for a boundary has that boundary as its top polyline and the next boundary down as its bottom polyline. Before, its bottom was its own boundary, so the ground under a boundary had no layer, the lower layers were wedges inside the top layer, and the model ended at the lowest boundary. `top_elevation` / `bottom_elevation` are as before. `build_fem_inputs` and `_fem_layer_dict` not touched |
| `geo_project/builders.py` | `to_slope_geometry` (only) | The layer under a polyline bottom gets that polyline as `top_boundary_points`. `to_fem_kwargs`, the schema and `validate` not touched |
| `funhouse_agent/adapters/slope_stability.py` | `build_soil_layer`; the `soil_layers` text in `METHOD_INFO` | passes `top_boundary_points`; the text states the rule |
| `funhouse_agent/adapters/calc_package.py` | the `soil_layers` text of `slope_report_package` | states the rule |
| `funhouse_agent/adapters/dxf_import_adapter.py` | `_slope_params` | writes `top_boundary_points` with the layers it hands back |
| `calc_package/interactive.py` | `slope_interactive_figure` | fills each layer where it holds the ground |

In `slope_stability`: `geometry.py` (`SlopeSoilLayer.top_boundary_points`,
`top_at`; `SlopeGeometry.column_at`, `layer_bands`, `layer_overlaps`,
`_scan_layers`, the gap refusal), `slices.py` (`_compute_slice_weight`; a
polyline top's vertices in `slice_breaks`' crossing search), `results.py`,
`analysis.py`, `rapid_drawdown.py` (the report), `plotting.py`,
`calc_steps.py`. No solver, search, screen, crack, water rule or default
changed.

**Truth in results.** `choices` and `defaults_used` are unchanged and still
true (no default changed; the new input has no default behaviour). The page
fingerprints did not change (`test_page_fingerprints.py` passes without a
re-record: parameter descriptions are not in the fingerprinted text). The
user manual's ten `soil_layers` rows are hand-edited to the tool text.

## 8. Tests

Python 3.12, the brief's venv; judged by pytest's exit code.

**Before any change** (master f108ae8): `slope_stability/tests`,
`validation_examples`, `dxf_import`, `geo_project`, `calc_package/tests` in
one run: 1,456 passed, 20 skipped, 1 xfailed, 0 failed (exit 0).

**After, with the old and new weights recorded side by side:** the same
folders: 1,469 passed, 5 failed (the five V-028 tests, expected), 20 skipped,
1 xfailed. Then V-028 was re-pinned.

**Final run on the branch** (one run, exit 1 for the two failures below):
3,625 passed, 2 failed, 47 skipped, 2 xfailed.

| Folder | Passed | Failed | Skipped | xfailed |
|---|---:|---:|---:|---:|
| `slope_stability/tests` (all 35 files) | 666 | 0 | 17 | 0 |
| `validation_examples` (all files) | 340 | 0 | 0 | 2 |
| `dxf_import` | 116 | 0 | 0 | 0 |
| `geo_project` | 93 | 0 | 0 | 0 |
| `calc_package/tests` | 264 | 0 | 3 | 0 |
| `funhouse_agent/tests`, the 35 files that name slope (with `test_defaults_used.py` whole) | 2,128 | 2 | 27 | 0 |
| `webapp/tests/test_page_fingerprints.py`, `test_public_text.py` | 18 | 0 | 0 | 0 |

* The two failures are not from this work: both fail the same way on
  untouched f108ae8 (section 10, item 11). `test_defaults_used.py -k slope`
  passes.
* The second xfail is new and strict: V-028 against the referee. The first is
  V-043's (GM-117), as before.
* The skips are as before: the removed nails-only API (16), pdflatex (3), one
  circle that misses the slope, and dependency skips in the agent tests.
* New tests: 18 in `test_one_material_per_point.py`, 3 in
  `dxf_import/tests/test_converter.py`, 1 in `geo_project/tests/
  test_builders.py`, 3 in `funhouse_agent/tests/
  test_slope_stability_adapter.py`; V-028's two files went from 6 tests to 8.
* The full suite was not run, as the brief allows.

## 9. Release note

**Slope stability: every model with a sloping layer boundary changes.** A
slice's weight used to count each soil layer from its own flat top, so where a
layer boundary sloped the soil between the boundary and the lower layer's
flat top was weighed twice (three times on a model with three such layers).
Each point is now weighed once, with the layer that also gives its strength.
The factor of safety moves up or down with the geometry: on the ACADS 4
benchmark the same surface goes from 0.81 to 1.07, on a two-layer test slope
from 2.49 to 2.85. Models with level layers are unchanged. Slopes imported
from a DXF with soil boundaries change more: the ground under each boundary
now belongs to the layer named for it (it had no layer), and the model
reaches 2 m below the lowest boundary instead of ending at it. A layer can
take a sloping top (`top_boundary_points`); the result lists any layers that
were given the same ground (`layer_overlaps`); a model that leaves ground
with no layer is refused, with the place. The ACADS 4 benchmark (V-028) no
longer agrees with its published answer and is marked as not validating.

## 10. Noted, not fixed

For the later steps of GM-140 or the lead; none was changed here.

1. **V-028's rigorous methods have no solution on its seam surface**, and
   the fallback value is not monotonic in the load (0.9435 to 0.9452 for 15
   kPa more on the crest). Step 2 (D23 / D30).
2. **A layer boundary's vertices are not slice boundaries** (only its
   crossings with the slip surface are). A slice with a kink of a boundary
   inside it takes its weight from the mid-point column: a second-order
   error, zero where the boundary is straight across the slice. Step 3's
   subject (slicing); it would change a value slightly.
3. **A flat top that another layer holds still puts a slice boundary** where
   the slip surface crosses it. Harmless (one more slice); it is why the same
   ground entered with a flat or a polyline top differs by 1e-7 in F.
4. **The weak-layer search aims at a layer's flat `top_elevation` and
   `bottom_elevation`** (`search.py` about 2019 to 2067). On V-028 that is
   el. 15 to 40 for every layer, not the 1 m seam. Step 7.
5. **The su-with-depth law of the probabilistic runs** takes a layer's
   elevation from the flat values (`probabilistic._apply_su_law`).
6. **`ru`** still multiplies the base layer's unit weight by the column
   height (D3, step 5). The weight in the column is right now, so step 5 can
   take the overburden from `column_at`.
7. **The part of a slice below the model's base weighs nothing** (GM-128, D19).
8. **`dxf_import.build_fem_inputs` and `geo_project.to_fem_kwargs`** build
   FEM layers by their own convention; not looked at here.
9. **`geo_project.validate`** still refuses an explicit lower-layer top that
   is not the boundary's highest point (GEOM004). With the rule that is
   stricter than needed, but nothing depends on it.
10. **V-028's bench elevation** (27.75 by Tables 9.2 and 9.3, 28 as entered)
    and its seam need Figure 9.1.
11. **Two tests fail on untouched master in this environment** and are not
    from this work: `test_calc_package_figure_conventions.py::
    test_large_pile_group_leaves_the_forces_to_the_table` and
    `test_defaults_used.py::...[profile_figure|plot_data]` (logx / logy
    change nothing). Checked by running both on f108ae8.
12. **`layer_at_point` does not look at the ground**: a point above the
    ground still reads a layer. Nothing calls it there.
