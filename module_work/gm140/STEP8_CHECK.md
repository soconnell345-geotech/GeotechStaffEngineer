# GM-140 step 8 (supports as printed): check file

Branch `claude/gm-140-step8-supports`, from master c9a375e (steps 1, 2, 3,
4, 5a and 9 and "Where a circle ends" merged; their CHECK files are on
master). Brief: `module_work/cloud_briefs/SLOPE_GM140.md`, section "Step 8:
supports as printed". Folds GM-82, OWN-59, GM-353 and the account's D34 to
D37.

## The rule

A support enters a solution as a known force, the way the print gives it:
by Method A or Method B, along the reinforcement's own direction, and no
larger than the reinforcement can carry where the slip surface crosses it.

One function, `slope_stability/supports.py::resolve_supports`, now turns a
geometry's supports into the forces that every method takes
(`reinforcement.compute_reinforcement_forces` returns its `forces`).

## What the print says

Duncan, Wright and Brandon (2014), "Soil Strength and Slope Stability",
2nd ed., is the owner's copy and is in no repository: every item from it
below is as the brief gives it, by page and equation. GEC 11 and GEC 7 were
read in the reference library (geotech-references e1bbc5b, v1.7.0).

- **Method A and Method B.** Sec. 8.2, pp. 159 to 160. A: the forces are
  allowable forces, not divided by F; F = soil resisting moment /
  (overturning moment - reinforcement moment) (Eq. 8.1; Eq. 8.2 in
  stresses). B: the forces are ultimate forces, divided by F as the soil
  strength is; F = (soil resisting moment + reinforcement moment) /
  overturning moment (Eqs. 8.3, 8.4). A is preferable (p. 160).
- **Bishop's simplified method with known forces.** Sec. 6.5.3, pp. 91 to
  93, Eq. 6.78 with Eqs. 6.70 and 6.75. Horizontal reinforcement enters
  moment equilibrium only; inclined reinforcement both; the vertical part
  is always included. The method neglects what a horizontal force adds to
  the normal stress on the slip surface (p. 93). No Ordinary Method of
  Slices equation with known forces is printed (p. 93).
- **Direction.** Sec. 8.6, pp. 163 to 164: the reinforcement's original
  orientation (recommended, the more conservative); parallel to the slip
  surface is the other extreme.
- **Capacity along a layer.** Sec. 8.4, pp. 161 to 163: the lesser of
  T_lim and the pullout measured from each end (Fig. 8.2); T_po = 2 gamma z
  alpha F* L_e (Eq. 8.8); T_po = gamma tan(beta) alpha F* L_e^2 beneath the
  slope face (Eq. 8.9); Table 8.3; T_all = T_LTDS / F_R (Eq. 8.10), F_R =
  1.3, 1.5 or 2.0 (Table 8.4).
- **GEC 11** (read in the library): "The tensile capacity of a
  reinforcement layer is taken as the minimum of its allowable pullout
  resistance behind the potential failure surface or its long-term
  allowable design strength" (Sec. 9.1, p. 9-1). Pr = F* alpha sigma'v Le C
  (Eq. 3-2, p. 3-16; C = 2). Le = Tmax FS / (F* alpha sigma'v Rc C)
  (Eq. 9-9, p. 9-14). "Use F.S. = 1.5 for granular soils; use F.S. = 2 for
  cohesive soils; minimum anchorage length, Le, = 3 ft (1 m)" (p. 9-5).
  "Only count reinforcement that extends 3 ft (1 m) beyond the surface to
  account for pullout resistance" (p. 9-14). In the absence of test data
  alpha = 0.8 for geogrids and 0.6 for geotextiles (p. 3-16). Programs that
  take the reinforcement as reducing the driving moment: Tal divided by
  FS_R = 1.3 (Eq. 9-17, p. 9-24).
- **GEC 7** (read in the library): Tmax_i = minimum of (R_Ti, R_Fi, R_POi)
  (Eq. 5.2). The library's text of 5.2.1d says the load "is limited by the
  minimum of pullout resistance RPO and tendon tensile resistance RT, and
  at the facing by the minimum facing resistance RF (Figure 5.7)". The
  form "the facing with the pullout of the length in front of the surface"
  is the brief's reading of Eq. 5.2 and Fig. 5.7; the library holds the
  figure by name only. Listed under "Calls made".
- **The Ordinary Method of Slices with supports.** Looked for once more in
  the library (FHWA NHI-06-088 chapter 6, UFC 3-220-10, GEC 11 chapter 9,
  DM 7.1 chapter 7, the Caltrans trenching manual chapter 10): none prints
  it. GEC 11's Eq. 9-1 / 9-17 is a moment equation for a circle on an
  unreinforced factor of safety found by any method, not a form of the
  Ordinary Method of Slices.

## What was wrong (measured on master c9a375e)

- D34: a geosynthetic carried its full `T_allow` wherever a surface
  crossed it, to the last millimetre of the layer.
- D35: a nail had no facing term.
- D36 / GM-82: a geosynthetic's crossing was the first sign change from
  the surface's low-x end, and its `dir_x` could point away from its
  anchorage; every method hid that by taking the horizontal component's
  magnitude as resisting. A pile's 'normal' direction had no printed basis.
- D37: `reinforcement_force` without `reinforcement_elevation` was left
  out without notice.
- A nail's or an anchor's crossing was found by sampling its length at 200
  points.
- GM-353: V-043's force read 40.88 kN/m at R - 1e-8 m and 45.22 at R and
  at R + 1e-8 m (`step8_section11.py` on master).
- The Ordinary Method of Slices took supports by forms of the module's own.
- On a circle split into arcs (GM-343) a tieback's head was tested against
  the whole circle, so it was counted on an arc it does not cross.
- No result said whether the forces were allowable or ultimate beyond the
  per-force `support_convention`.

## What changed, and why

1. **`slope_stability/supports.py` (new).** `resolve_supports`: exact
   crossings (`line_crossings`, `level_crossings`), the direction
   (`up_slope_tangent`, `force_direction`), the capacity at the crossing
   (`layer_capacity`, the nail's facing term), the slack rule,
   `slice_index_for` (GM-353), `check_legacy_force` (D37),
   `support_statements` (the `choices` entries).
2. **`reinforcement.py`.** `compute_reinforcement_forces` returns
   `resolve_supports(...).forces`. `Geosynthetic` gains `F_star`, `alpha`,
   `pullout_factor`, `pullout_basis`, `development_length`,
   `anchored_end`, `coverage_ratio`, `force_direction`; `Anchor` gains
   `force_direction`; `StabilizingPile.force_direction` is 'horizontal' or
   'along_slip_surface' ('normal' refused); `ReinforcementForce` gains
   `side` and `note`. `support_components` takes the force's own direction
   (no forced sign). `slice_index_at` follows `slice_index_for`.
3. **`nails.py`.** `SoilNail.facing_capacity`, `force_direction`.
4. **`geometry.py`.** `SlopeGeometry.assisting_support` ('slack' default,
   'as_given'); the legacy force without an elevation refused at
   construction.
5. **`gle.py::_assign_reinforcement`.** The force's own direction; the
   slice by `slice_index_for`.
6. **`methods.py::fellenius_fos`.** With supports: `NoSolution`, kind
   `with_supports` (`solution.OMS_WITH_SUPPORTS`). Its old support code is
   unreachable and left in place (smallest diff; the lead may delete it).
7. **`methods.py::bishop_fos`.** Checked against Eq. 6.78 with Eqs. 6.70
   and 6.75, term by term: the signed moment about the centre over R taken
   off the driving sum (M_n / r), `W - F_v` with F_v upward in the slice
   crossed, Method B both divided by F. It matched; no change.
   `test_supports_step8._bishop_by_hand` writes the equation out and
   agrees to 1e-9 for a horizontal and for an inclined support.
8. **Results.** `SlopeStabilityResult.supports_not_counted`; a `note` on
   each entry of `reinforcements`; `choices` entries on every result with
   supports (single surface, search, rapid drawdown and its search);
   warnings for supports not counted, pullout not checked, the facing term
   not checked. The short-arm warning cites p. 93 and says nothing is
   adjusted. The calc package's reinforcement section prints the same.
9. **Adapter** (`funhouse_agent/adapters/slope_stability.py`): the new
   inputs, `assisting_support` (applies where there is a support), the
   legacy force's refusal (its "not applied" declared case is gone), the
   descriptions; a probe `analyze_slope#steep_nail`.
10. **Docs.** `slope_stability/DESIGN.md` ("Supports as printed"),
    `VALIDATION.md` (B24), the user manual rebuilt by its generator.

## Places the rule now holds

Every method reaches supports through `solution.solve_method` with the
forces of `resolve_supports`:

- `analyze_slope` (`analysis.py`), and `compare_methods_table` through it;
- every search family (`search._compute_fos` -> `trial_reinforcement`):
  grid circles, entry and exit circles, toe circles and circles above the
  toe, planes, random, differential evolution, particle swarm and
  weak-layer polylines; the refinement walk and the re-scoring in
  `search_supplements.py`; `_attach_reinforcement` (the governing surface's
  list, with what counts nothing);
- rapid drawdown and its search (`rapid_drawdown.py`, each stage);
- the yield acceleration and Newmark (`newmark.py`);
- FOSM and Monte Carlo (`probabilistic.py`), and `reliability/wrappers.py`
  through them;
- `geo_project`'s limit-equilibrium builder and the calc packages, which
  call the functions above;
- the agent's adapter.

Methods: Bishop (Eq. 6.78), Janbu, Spencer, Morgenstern-Price and GLE (the
force in each equation the method satisfies, as before); the Ordinary
Method of Slices refused.

## Calls made (for the owner; OWN-143)

1. **The Ordinary Method of Slices with supports is refused** where a
   support crosses the surface, on circles and polylines. The lead's call
   (the brief). A search by that method scores only surfaces no support
   crosses and counts the rest (`no_solution`, `with_supports`).
2. **A support on a slice boundary acts on the slice that holds the part
   of the support inside the sliding mass** (a vertical pile: the slice
   toward -x). Reasoned before the number was known: that slice is the
   body the support pulls on. No reference held says which slice. On V-043
   it gives 45.22 kN/m (printed 44 to 45); the other side gives 40.88.
3. **"Assists the sliding" is tested by the force's component along the
   slip surface at the crossing** (down the slope = assists). On a circle
   this is the sign of its moment about the centre. A complete-equilibrium
   method can still gain a little from such a force through the base
   normal: the made-up steep nail reads 2.2129 slack and 2.2138 as given.
   The other test one could state is "the support lowers F", which needs
   two solves per support per trial surface and depends on the method.
4. **A layer with no pullout input carries its full `T_allow` and the
   result says the pullout was not checked**, naming what to give. The
   stricter reading (refuse a layer without F*, alpha and F_R) would refuse
   every existing input with a geosynthetic. Once F* and alpha are given,
   nothing is filled in: `pullout_factor` is required under Method A.
5. **The default pullout form, once F* and alpha are given, is the book's**
   (from each end, total `gamma z`, divided by `pullout_factor` under
   Method A), because the forces are entered by the book's Methods A and
   B. GEC 11's form (`pullout_basis='gec11'`) is the option.
6. **The integral of 2 alpha F* gamma z along the embedded length** stands
   for Eqs. 8.8 and 8.9 on ground that is neither level nor one uniform
   face. It equals each on its own ground (tested to 1e-9 and 1e-12).
7. **Under Method B the pullout is not divided** (ultimate forces, divided
   by F). GEC 11 divides its pullout by 1.5 or 2 in its own rotational
   method whatever the program; under `'gec11'` with a passive layer the
   module does not.
8. **The nail's facing term is R_F plus the pullout of the length in
   front of the surface**, applied only when `facing_capacity` is given;
   left out, the term is not checked and the result says so. See "What
   the print says": the sum is the brief's reading of Fig. 5.7.
9. **A pile's 'normal' direction is refused**; 'along_slip_surface' is
   the option beside 'horizontal'. A removed input value.
10. **A layer is crossed wherever the surface passes through it**, each
    crossing with its own force toward its own anchorage; the toe-side
    crossing of a layer crossed twice then falls under the slack rule.
11. **A crossing where the layer's level is the ground surface** (a layer
    at the toe's elevation met by a toe circle) is not a crossing: the
    layer is not buried there.

## Section 11, before and after

`step2_section11.py`, `step3_section11.py` and `circle_ends_section11.py`
were run unchanged on master (a worktree at c9a375e) and on the branch;
`step8_section11.py` (new) adds the rows with supports and the two printed
benchmarks. Outside the rows below, the three older scripts' output is
identical line for line (run times apart): every benchmark without
supports is untouched.

| Benchmark | Module before | Module after | Printed | What moved it |
|---|---|---|---|---|
| V-043 force to raise the pinned sand circle to 1.35 (Spencer), at R - 1e-8 m / R / R + 1e-8 m | 40.88 / 45.22 / 45.22 kN/m | 45.22 / 45.22 / 45.22 | 44 to 45 (Slide2 Verification #43) | the slice-boundary rule (GM-353) |
| V-040 micro-pile row, Ordinary Method of Slices, active / passive | 1.1946 / 1.1758 | no number | none for this method (Slide2 1.193 is Bishop's) | the method refused with supports |
| V-040, Bishop active / passive | 1.2232 / 1.2034 | 1.2232 / 1.2034 | 1.193 | - |
| V-040, Spencer active / passive | 1.2240 / 1.2039 | 1.2240 / 1.2039 | - | - |
| #86 five tiebacks: Bishop / Spencer / GLE | 1.6131 / 1.6065 / 1.6069 | the same | Slide2 1.629 / 1.620 / 1.622 | - |
| #85 one support, active, searches: Bishop / Spencer / GLE | 1.508 / 1.5505 / 1.5726 | the same | 1.531 / 1.884 / 1.575 | - |
| #85 passive, Bishop search | 1.3157 | 1.3157 | 1.324 | - |
| #85, circles above the toe (not the governing family): Bishop candidate; arcs with no net driving moment | 1.8195; 207 | 1.819; 141 | - | a tieback counts only on the arc it crosses |
| #85, GLE and Spencer candidates of that family | 2.0562; 2.052 | 2.0529; 2.0519 | - | the same |
| Fig. 8.1, Bishop, tangent circles, Method A (new) | 2.218 (computable before: plain layer) | 2.218 | 2.19 (+1.3 %) | - |
| Fig. 8.1, Method B (new) | 1.722 | 1.722 | 1.72 (+0.1 %) | - |
| Fig. 8.4, Spencer, Method A (new) | not available: a layer's force could not fall off toward its end | 1.607 | 1.62 (-0.8 %) | the pullout given by `development_length` |
| Fig. 8.4, Spencer, Method B (new) | not available | 1.448 | 1.46 (-0.8 %) | the same |

The brief expected movement on #85, V-040, V-043 and V-056 / V-057.
Measured: #85's headline values and V-040's Bishop and Spencer values do
not move (their supports cross their critical surfaces once, well inside
the support, with a resisting pull); V-043 moves as above; V-056 and V-057
(`validation_examples/test_published_v056_v057_gec11_slope.py`) are GEC
11's MSE wall examples and do not call this module's supports.

**Fig. 8.1's critical circles.** Method A: centre x 0.90 m, R = yc =
6.116 m (the crest is at 6.096 m), from x 0.031 m (0.06 m above the toe on
the face) to 7.016 m. The minimum lies at the limit before reverse
curvature: tangent circles with their centre below the crest's level are
discarded (step 4), and F is still falling there. The brief's hand check
gives 2.21 on that circle. Method B: centre x 0.50 m, R 7.316 m, from x
0.008 m (0.02 m above the toe) to 7.714 m. Grid: 0.1 m in centre x and in
radius.

**Fig. 8.4 against the validation set's #86.** The figure: six layers from
the bottom of the slope, 4 ft apart, each 20 ft long, 800 lb/ft falling
linearly to zero over the last 4 ft. The pinned #86 test
(`test_search_reinforcement._slide86`): five tiebacks at 4 to 20 ft, the
full 800 lb/ft to within 1 ft of their ends (the anchor's 5 % rule), no
taper, no layer at the base. The layer at the base lies on the rock under
every circle and counts nothing, and the critical circles cross the other
five in their full-force lengths, so both inputs give the same Spencer
value by Method A (1.6065). The #86 test's docstring now says what differs.

## Before and after on one worked example

V-043 (above): 40.88 or 45.22 kN/m by a rounding -> 45.22.

A made-up layer near its end (`test_layer_crossed_near_its_end...`): a
200 kN/m layer at elevation 14 m ending 1.24 m behind the circle, under
6 m of 19 kN/m3 fill, alpha 0.8, F* 0.4, F_R 1.5: before, 200 kN/m; after,
2 x 19 x 6 x 0.8 x 0.4 x 1.24 / 1.5 = 60.2 kN/m (Eq. 8.8 and Eq. 8.10).

## Tests

All by pytest's exit code, in a Python 3.12 venv (`pip install -e
packages/aec-agent-core -e . pytest pytest-xdist`), on the finished tree.
The files outside `slope_stability/tests` were run one file at a time: run
together under `-n`, the run stopped answering near its end twice, with its
workers gone, on master's files as on the branch's (no test failed; the
cause was not looked for).

| What | Result |
|---|---|
| `slope_stability/tests` (all) | 880 passed, 17 skipped, 1 xfailed (exit 0). On master: 857 passed with the same skips and xfail; +22 new in `test_supports_step8.py`, +1 in `test_search_reinforcement.py` |
| `validation_examples/test_published_*` that import slope_stability (21 files) | 118 passed, 4 xfailed (the standing ones: V-028 E8, V-037, V-043's noncircular gap, V-054) |
| `calc_package`, `reliability`, `fem2d`, `geotech_common`, `report_ingest`, `dxf_import`, `geo_project` test files that name slope_stability (22 files) | 1,223 passed |
| `funhouse_agent/tests` files that name slope_stability (17 files) | 1,740 passed, 30 skipped, 1 failed: `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure\|plot_data]` (logx / logy), which fails the same way on master c9a375e in this environment and does not touch this module |
| `funhouse_agent/tests/test_defaults_used.py -k slope` | 53 passed, 20 skipped |
| `webapp/tests/test_page_fingerprints.py` | 13 passed, unchanged: the changed tool text is not in either page's fingerprint, so there is no re-record commit |
| `webapp/tests/test_public_text.py` | 5 passed |
| The manual: `check_manual.py --rebuild --write`, `check_manual.py`, `webapp/tests/test_user_manual_current.py`, with `geotech-references` (e1bbc5b, the pin) first on PYTHONPATH | exit 0, exit 0, 1 passed |

Failing first, as far as it was measured. On master the new test file
does not import (it uses the new functions), so its tests were not run one
by one there. What was measured on master is in "What was wrong" and in
the section 11 table: V-043's two forces, the Ordinary Method of Slices
returning 1.1946 and 1.1758 with a pile, the 207 arcs. The other
departures (a layer near its end carrying its full force, no facing term,
the legacy force accepted) are read from master's code, not run. Fig.
8.1's two values could already be computed on master (the script ran
there) and are pinned as new printed benchmarks; Fig. 8.4's layers could
not be entered.

Tests re-pinned or rewritten, each against print:

- `test_reinforcement.py`: the two closed forms pinned on the Ordinary
  Method of Slices with a support now use Bishop's method (phi = 0, Eq.
  6.78, the same closed form); the 'normal' pile tests became its refusal
  and the 'along_slip_surface' option; the Ordinary Method of Slices'
  refusal is tested on a circle and on a polyline.
- `test_search_reinforcement.py`: the Ordinary Method of Slices taken out
  of "passive below active in every method" and given its own refusal
  test; the comparison test expects it among the methods left out.
- `validation_examples/test_published_v040_e6.py`: the Ordinary Method of
  Slices test became its refusal (the split is asserted on Spencer).
- `validation_examples/test_published_v043_slope_reinforced.py`: 45.22
  kN/m (abs 0.05; printed 44 to 45), and F = 1.35 at R - 1e-8 and
  R + 1e-8 m with that force.
- `funhouse_agent/tests/test_slope_stability_adapter.py`: the comparison
  table's first row.

## Release note

Slope supports follow print (GM-140 step 8). Answers move where: (1) a
support meets the slip surface on a slice boundary, as a layer on a
material boundary does (one force now, where a rounding chose between two:
45.2 against 40.9 kN/m in one validation case); (2) the Ordinary Method of
Slices is asked for with supports crossing the surface: it now gives no
number, with the reason, because no reference prints it with known forces
(use Bishop's, Spencer's, Morgenstern-Price's or Janbu's method); (3) a
circle is split into several arcs and a tieback crosses only one of them;
(4) a support's pull along the slip surface points down the slope (a steep
nail high on the surface, or the toe-side crossing of a layer crossed
twice): it now counts nothing and is named, unless `assisting_support` is
'as_given'; (5) a stabilizing pile was given `force_direction='normal'`:
refused, use 'horizontal' or 'along_slip_surface'; (6) a
`reinforcement_force` was given with no `reinforcement_elevation`: refused
(it was silently left out). New, changing nothing unless used: a
geosynthetic's pullout resistance (`F_star`, `alpha`, `pullout_factor`,
`pullout_basis`, `development_length`, `anchored_end`), a nail's
`facing_capacity`, and a force parallel to the slip surface
(`force_direction`). Every result with supports now states Method A or
Method B and whether the forces entered are allowable or ultimate, and
says when a layer's pullout or a nail's facing term was not checked.

## Stopped short of, and noted, not fixed

- **An anchor's capacity** is still its full `T_allow` once more than 5 %
  of its length lies beyond the surface. The 5 % is the module's own; D34
  to D37 do not name it. Noted, not fixed.
- **The short-arm warning's threshold** (a share of 0.5) is the module's
  own; kept as the brief says, with the page cited.
- **The pullout stress under `'dwb2014'`** is the total `gamma z` as Eqs.
  8.8 and 8.9 print it, whatever the water; `'gec11'` uses the effective
  stress under the water line read as hydrostatic.
- **A nail's bond, bar and facing** carry no factor of their own: they are
  taken in the measure of `support_convention`.
- **The retired `foundry/slope_stability_agent_foundry.py`** still passes
  `reinforcement_elevation` through and was not edited.
- **`fellenius_fos`'s old support code and `horizontal_reduction` /
  `horizontal_resistance`** are unreachable through the solver and were
  left in place.
- **Stabilizing piles' Ito-Matsui force as its own method** (OWN-13): out
  of scope.
- **GM-82's lambda bound of +-1.5** on #85's GLE search: not a support
  rule; left for "The GLE solver" step.
