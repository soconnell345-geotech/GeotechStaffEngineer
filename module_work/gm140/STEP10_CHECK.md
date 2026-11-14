# GM-140 step 10: reporting as printed

Branch `claude/gm-140-step10-reporting`, from master eb4286f (steps 1 to 6, 8 and 9, the GLE step and "where a circle ends" merged; their CHECK files are on master). Built 2026-11-14 by a cloud session. Brief: `module_work/cloud_briefs/SLOPE_GM140.md`, section "Step 10", including "From the steps since". BACKLOG GM-140 and GM-346.

**The rule.** A result and its report show what the print asks an engineer to be shown, and an input that changes nothing is not accepted. **No computed value changes** (section 4).

All thirteen items are built. Nothing was stopped short of, except what section 7 lists as noted.

## 1. What the print says

| Item | Print | Where in the code |
|---|---|---|
| 1 Contours of minimum F over the centres (D46) | FHWA NHI-06-088 Fig. 6-13, p. 6-23; GEC 11 Vol. II p. 9-24 ("a contour plot of lowest safety factors about trial failure circle centroids is recommended"); Duncan, Buchignani and De Wet (1987) Fig. 28 (the contours should enclose the minimum) | `slope_stability/report_views.py` `centre_contours`; `plotting.plot_centre_contours` |
| 2 Ordered list of the ten lowest (D47) | FHWA p. 6-49 ("ordered output and plotting capability for the failure surface of 10 minimum safety factors"); EM 1110-2-1902 4-3b(9)(b), p. 4-8 (critical and other significant surfaces with their F; for circles centre and radius) | `report_views.lowest_surfaces` |
| 3 Slice table first | EM 1110-2-1902 4-2d(2), p. 4-2 (the engineer checks slice weights, strengths and pore pressures at the bottoms of slices) | `calc_steps.get_calc_steps` |
| 11, 8 Method per family | EM 1110-2-1902 4-3b(9)(a), p. 4-8; FHWA p. 6-20 and Table 6-1 (stand-in order); UFC 3-220-10 p. 362 (unsolved surfaces) | `calc_steps._family_methods_table` |
| 12 Supports, OMS, m_alpha | Duncan, Wright and Brandon (2014) pp. 93, 159 to 168 (Method A/B, Eqs. 8.1 to 8.4; no OMS with known forces, p. 93); GEC 11 Vol. II p. 9-24, Eq. 9-17; EM 1110-2-1902 C-10b, pp. C-34 to C-36 | `calc_steps._no_solution_detail`, `_acceptability_section` (statements otherwise already in the result, step 8) |
| 7 Circle in c' = 0 soil | Duncan, Wright and Brandon (2014): where c' = 0 soil forms the face the critical failure tends to a shallow slide parallel to it (the module's existing citation for the infinite-slope family) | `analysis.cohesionless_circle_statements` |

The rule for "the lowest contour closes inside the grid" is the module's reading of the contour plot, not a printed formula: a contour at level L encloses the lowest centre without touching the grid's edge when the lowest value inside the grid is below L and every edge value is above it, so a closed contour exists exactly when the lowest value inside is below the lowest value on the edge (Calls made, 1).

## 2. What changed, and why

1. **Contours (D46).** `grid_search` marks each lattice centre in `grid_fos` with `node` = (ix, iy) and returns its grid as `SearchResult.centre_grid` (new field). `report_views.centre_contours(search)` lays the per-centre minima scored by the method on the grid (a centre a screen alone scored is left out; a centre with no solution is empty), gives the minimum, the lowest edge value, `closed` and a statement naming the edge to widen when it does not close. In `SearchResult.to_dict()["centre_contours"]`, the summary, the report's table "Minimum FOS over the Circle Centres", and a new figure (contours, the grid outlined, the lowest centre and the critical centre marked). Entry/exit and noncircular searches have no centre grid: nothing is drawn or said.
2. **The ten lowest (D47).** `apply_supplements` keeps each family's reported surface in `SearchResult.family_surfaces` (new field; the toe-circle, above-toe and plane families kept only their lowest before as well). `report_views.lowest_surfaces(search)`: the main search's trials scored by the method plus each family's surface, deduplicated by geometry, sorted, ten, each with rank, FOS, family, method and centre and radius or points. In `to_dict()["lowest_surfaces"]`, the summary, the report's table "The 10 Lowest Surfaces" and the new figure (the first in red, 2 to 10 in orange).
3. **Slice table first.** The report's calculation sections start with the slice force table. The package's input echo (Given Inputs, Assumed Inputs, Values Used, Method Choices: the generic calc-package front matter) still comes before every calculation section (Calls made, 6).
4. **Inputs that do nothing (D45).** The two slope package probes leave `FROZEN` (`method_probes.py`; they run in about a minute). What the guard found and what was done: `tol` above 1e-4 changes nothing with Janbu's, Spencer's, Morgenstern-Price or GLE (`solve_method` runs them at min(tol, 1e-4)/10) and the OMS does not iterate, so `slope_report_package` refuses a tol above 1e-4 with any method but Bishop's (`_check_slope_tol`) and the description says what tol does. `FOS_required` moves `is_stable` and the stability check (the probe's FOS is above both values tried: `NOT_REACHED` with a test); `time_budget_s` stops a search it cannot finish in (`NOT_REACHED` for the probe, with a test). An unknown `f_interslice` was already refused (step 2); a test now pins it on the package. The METHOD_INFO `returns` text of `circle_check` now says when it comes back.
5. **Figures at print size (GM-186).** Measured with the layout check at the printed width (5 in): the interslice figure and the report's cross-section were "kept" at their drawn 9.4 to 9.8 in (lettering printing near 4 to 5 pt) because the title and the line-of-thrust key were wider than the page; both now print at 7 pt or more. The keys of the cross-section, slice-force, interslice (lower panel) and contour figures sit below the plot at a fixed distance (`plotting._key_below`), the factor of safety heads the cross-section's key (it was a box over the drawing and the title), the circle centre's label hangs from the centre on the side away from the radius line, and the entry and exit labels sit below the ground outside the arc. The two slope packages join `test_calc_package_figure_layout.py` (no findings on any of their figures).
6. **choices and defaults_used (GM-132's slope part).** `"set"` in place of `[True]` for `compare_methods` (analyze_slope's f_interslice, negative_friction_term, base_in_tension), `research_surface` (monte_carlo_fos's time_budget_s) and `monte_carlo` (the report package's n, research_surface, seed): the code runs on any truthy value. Rapid drawdown's m_alpha check runs by `m_alpha_rule` 'frictional', a name its tools do not declare: an adapter may now name such module defaults (`MODULE_DEFAULTS`), and `dispatch` adds them to the result's `defaults_used`; giving the name is still refused. The defaults guard leaves those names out of what it gives back.
7. **A circle in cohesionless soil.** `analyze_slope` adds a warning where c' = 0 soil forms the slope face within the circle's span: a single circle there overestimates against the infinite-slope mechanism, which `search_critical_surface` scores with planes; tan(phi')/tan(beta) for a dry face is given; the circle's FOS is not changed.
8. and 11. **The method each family was scored with**, a table in the report's search section: family, minimum, method, interslice function, and "the method named" or "stand-in for X: <the result's reason>"; a screen named where one picked the circles; where Janbu's method stands in, f0 applied, not found in print (D28), its value and the uncorrected FOS; below the table method_named, method_used and named_method_lowest where they differ, the unsolved line in the result's words with the lowest screen value (UFC 3-220-10 p. 362), the circle check (scored as a trial and its value, or "Not checked against a circle"), and an OMS search refused where supports cross, nothing standing in.
9. **GM-346 and the arc ends.** Both packages build their geometry with the slope tools' own `_build_geometry`, so a tension crack (step 4's inputs and rules: given, Rankine or minimum-F depth, side, model, water, reverse curvature) reaches them; WE-SLOPE-1 as printed (its 6.096 m crack) now makes a package, with the tool's FOS. `slope_stability_package` takes `arc_x_start` and `arc_x_end`.
10. **The later inputs reach the packages**: `water_line_kind`, `base_in_tension` (passed to the search and to `analyze_slope`), and the supports (`nails`, `anchors`, `geosynthetics`, `stabilizing_piles`, with each item's `support_convention` = Method A or B and `pullout_basis`, and `assisting_support`). Each parameter's spec is the tool's own (one text); `applies_when` as in the tools. The report names the reading of the water line, the base-in-tension rule and Method A or B with the measure of force through the result's own `choices` and warnings (already printed in the package's Method Choices and Warnings sections). No default changes.
12. **Step 8's and the GLE step's statements in the report**: Method A or B and the measure of force, GEC 11 Eq. 9-17 as the result states it, "pullout not checked" in the Warnings table (it is in `result.warnings`, step 8), the short-arm warning (Bishop, step 8) were already printed and are now pinned by tests that read them from a report. New: the OMS row of the method comparison where a support crosses says "left out of the comparison" with its reason (it said "does not converge"); the Printed Acceptability Checks row for low_m_alpha names the denominator read and adds EM 1110-2-1902 C-10b to its source where the finding carries C-10b's remedies.
13. Tests: below.

Places the rule now holds: `search_critical_surface` (every kind of search; centre contours for the grid), `grid_search`, `apply_supplements`, `SearchResult.to_dict` and `summary` (so the agent's search tool), `analyze_slope` (item 7, so every caller that analyses a given circle: compare_methods_table, the packages, rapid drawdown's and yield's own analyses are not touched), `calc_steps` (both packages), `plotting`, the calc-package adapter, `dispatch` (MODULE_DEFAULTS).

## 3. Before and after on a worked example

No value moves; the "before and after" is what is shown.

| Case | Before | After |
|---|---|---|
| WE-SLOPE-1 through `slope_stability_package` (its 6.096 m crack, OMS) | refused: `tension_crack_depth` not a package input | FOS 0.839 = `analyze_slope` 0.8385 (the worked example records 0.8328, an older run; printed 0.87) |
| Made-up 10 m clay slope, Bishop, 5 x 5 grid, x 10 to 25, y 10 to 22 | grid values in `grid_fos` only | lowest contour closes (2.166 inside, 2.354 on the top edge); ten lowest 2.162 to 2.1665 |
| Same, y 10 to 15 | — | does not close: 2.163 on the top edge; "extend x_range / y_range past that edge" |
| Report probe (`slope_report_package`), interslice figure | saved 9.4 in wide, smallest lettering 7.5 pt there (about 4 pt at 5 in) | 5.25 in, 7.5 pt (7.2 pt printed) |

## 4. Section 11, before and after

Script: `module_work/gm140/step4_section11.py`, unchanged (it runs every `validation_examples` file that uses slope_stability and the module's benchmark test files and reports, per test, whether its pinned value holds). Before: master eb4286f; after: this branch.

| Run | Passed | Skipped | Xfailed (strict) | Moved |
|---|---|---|---|---|
| master eb4286f | 391 | 1 | 3 | none |
| this branch | 391 | 1 | 3 | none (the same tests, the same outcomes, line for line) |

The three strict xfails are the same on both trees (V-028's weak-layer search against print, Slide2 #95 on its own printed pool, V-054 on the former critical circle). Every benchmark of the account's section 11 is pinned in those files; none moves, as the brief requires for this step.

## 5. Tests

By pytest's exit code, `-n 4`:

* The brief's set and item 13's files, in one run: `slope_stability/tests` (all), every `validation_examples/test_published_*` file and every `calc_package`, `funhouse_agent`, `reliability`, `fem2d`, `geotech_common`, `report_ingest`, `dxf_import` and `geo_project` test file that names `slope_stability` (this includes `test_defaults_used.py` in full and `test_slope_stability_adapter.py`), plus `funhouse_agent/tests/test_top_level_params.py`, `test_gm179_unused_inputs.py`, `test_calc_package_tool_parity.py`, `test_calc_package_figure_layout.py`, `funhouse_agent/deep/tests/test_wave1_geotech.py` (every describe result within its 16,000-character budget) and `webapp/tests/test_public_text.py`: **4,338 passed, 50 skipped, 4 xfailed, 1 failed**. The one failure, `test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]` (logx, logy change nothing), fails the same way on master eb4286f: not this step's, noted.
* `webapp/tests/test_page_fingerprints.py`: 13 passed. The pages' fingerprints do not move (the changed texts are tool parameter and return descriptions reached through describe_method, not the pages' prompts or tool lists), so no re-recording commit was needed.
* User manual: after the submodule at e1bbc5b (v1.7.0, the pin), `check_manual.py --rebuild --write` (exit 0; four rows of the slope package and search tables rewritten), `check_manual.py` (exit 0), `webapp/tests/test_user_manual_current.py` (1 passed).

New: `slope_stability/tests/test_gm140_step10_reporting.py` (13: the closed and open contours, on synthetic grids and two searches; screen and unsolved centres not contoured; no contours without a grid; the ordered list sorted and complete; family surfaces; the slice table first; the report's tables and figure; the cohesionless statement; the method table on a real stand-in and on the f0 / unsolved / circle check / OMS statements) and `funhouse_agent/tests/test_gm140_step10_package_inputs.py` (10: FOS_required, time_budget_s, tol refused, f_interslice refused; WE-SLOPE-1's crack; the arc ends; water_line_kind, base_in_tension and supports in each package with the report naming them; the OMS row; the m_alpha row). Three probes added (`calc_package|slope_stability_package#crack`, `#supports`, `calc_package|slope_report_package#water`) for the conditional inputs. `test_defaults_used.py` gains a test of MODULE_DEFAULTS.

The page fingerprints are re-recorded in the last commit (tool descriptions changed).

## 6. Calls made

1. **Closed contour** = the lowest value inside the grid is below the lowest value on its edge (section 1). The references ask for the plot and for the minimum to be enclosed; none prints a test.
2. **The ten lowest**: the pool is the main search's own trials plus each other family's lowest surface (those families never stored more), values a screen alone gave are left out, and equal surfaces (same centre and radius to 0.01 m, or same points) are listed once. The list is therefore not the ten lowest trials of every family, and says so in its note.
3. **tol refused above 1e-4 unless the method is Bishop's** (a Bishop search's plane families scored by Spencer still ignore a tol above 1e-4; the circles, which dominate, take it). The other choice was to wire tol into the GLE as given, which would move values at the default.
4. **MODULE_DEFAULTS** in `dispatch` rather than a new rapid-drawdown input: declaring `m_alpha_rule` on `rapid_drawdown_fos` would take its description past the 16,000-character budget (15,823 today) and make a new option for one statement.
5. **The factor of safety in the cross-section's key title** (and the key below the plot), instead of a box on the drawing: any corner of the drawing collided with an entry label or the centre label on one of the probes.
6. **"The slice table first"** read as the first calculation section; the calc package's generic input echo stays at the front of every package.
7. The report package's `base_in_tension` always applies (the report compares every method on the critical surface).

## 7. Noted, not fixed

* The probabilistic annex (FOSM, Monte Carlo) of `slope_report_package` takes the geometry with its crack, water line and supports, but not `base_in_tension` (the probabilistic functions take none): no default moves.
* WE-SLOPE-1's recorded `computed_result` (OMS 0.8328) is older than the tool's present 0.8385 on its own inputs; the worked-example file is not this step's.
* **Out of scope (the audit's row 1), what each would need:** a rapid-drawdown package: the `rapid_drawdown_fos` / `search_rapid_drawdown` result's three stages and the per-slice table of Figs. G-10b and G-11b (`slice_table()`) as calc steps, the water before and after drawn on the section, and the stage factors of safety as a check; a yield-acceleration package: the kh search's trace (kh against FOS to FOS = 1) and the critical surface at ky; FOSM and Monte Carlo packages: the existing probabilistic annex of `calc_steps._probabilistic_section` as the package body, on a given surface, with the variables table. None is built.
* DXF export of the slip surface (GM-18): out of scope.

## Release note

No calculated value changes. A slope search's result now carries the contours of the minimum factor of safety over the circle centres, with whether the lowest contour closes inside the grid (and where to extend it when it does not), and the ten lowest surfaces in order; the slope reports print both, with a new figure, put the slice table first, and say which method scored each family of surfaces and why where another stood in. The slope calc packages now take a tension crack, the arc's two ends, `water_line_kind`, `base_in_tension` and the supports as the slope tools do, so a package can be made for any model the tools take; `slope_report_package` refuses a `tol` above 1e-4 except with Bishop's method (it changed nothing). A single circle through cohesionless soil at the slope face now carries a warning that it overestimates against the infinite-slope mechanism. The slope figures print at 7 pt or more on the page, with their keys below the drawing.
