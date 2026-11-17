# GM-140 step 7a: search coverage as printed (the noncircular search)

Branch `claude/gm-140-step7a-search-starts`, from master 7a5bb26 (steps 1 to 6, 8, 9 and 10, the GLE step and "Where a circle ends" merged; their CHECK files are on master). Built 2026-11-15 by a cloud session. Brief: `module_work/cloud_briefs/SLOPE_GM140.md`, section "Step 7a", with its "Calls made before the build". BACKLOG GM-140, GM-117 and GM-89 (its noncircular item).

**The rule.** A noncircular search finishes the way the print describes one. It runs from several independent starts: the locally critical circles, a wedge along each layer, the search's own best surface and the caller's surface. Each start is moved one point at a time until no shift lowers F and the last shifts are small against the thinnest stratum. The lowest result is reported, and every locally critical surface is listed beside it. **No circular value changes** (section 4: every pinned circular test holds).

All ten items are built. Section 7 lists what stops short of the brief's expectations and why.

## 1. What the print says

| Item | Print | Where in the code |
|---|---|---|
| 1 Refinement | EM 1110-2-1902 C-1g(3), p. C-10, Fig. C-8 (Celestino and Duncan 1981): hold all points but one, shift that point a small distance in two directions, move the surface, repeat until F stops falling and the moves become minimal | `search_refine.refine_polyline` |
| 2 Final increment | EM 4-2e(2)(b), p. 4-7: final shift increments no more than 10 to 25 % of the thickness of the thinnest stratum | `search_refine.thinnest_stratum`, `refinement_stage` |
| 3 Starts | EM 4-2e(1)(b), pp. 4-3 and 4-6; 4-2e(2), p. 4-7 (the initial surface matches the estimated critical surface); C-1g(4), p. C-11; FHWA NHI-06-088 p. 6-22 point 1 | `search_refine.refinement_stage`, `circle_starts_of`; `search._refines_noncircular` |
| 4 Wedges | EM C-1f(2), p. C-6; C-1g(2), pp. C-8 to C-10, Fig. C-7b (45 + phi'_D/2 active, 45 - phi'_D/2 passive, tan phi'_D = tan phi'/F); FHWA pp. 6-43 to 6-44 | `search_refine.wedge_starts`; `search.search_weak_layer_biased` |
| 5 Locally critical surfaces | EM 4-2d(2), p. 4-2 (the output names the most critical surface and the other surfaces analysed); 4-2e(1)(b) | `SearchResult.local_minima`; `search_refine.grid_minima_record` |
| 6 Faces | EM 4-2e(1)(a), p. 4-3 (the search must not jump from one face to another) | `search_refine.end_allowed` |
| 7 Toe | EM 4-2e(2)(a), p. 4-7 (orientation at the toe restricted); EM C-10b, p. C-34, Eq. C-32 (alpha = -(45 - phi'/2)) | `search_refine.passive_exit_limit`, `refine_polyline` |
| 8 Windows | Call 5 (OWN-143 part 42, confirmed by the owner 2026-11-15) | `search._run_refinement` (`windows_given`), `wedge_starts` |
| 9 Shape | Call 6: none | (none) |
| 10 Switch | Call 8; GM-123 (the time budget) | `search._refines_noncircular`; `analysis.search_critical_surface` (`refine`, `refine_fraction`, `seed_surface`) |

**The account's D12 cite corrected.** The account cites EM p. 4-7 for a concave-up shape rule. That page (4-2e(2)) prints only the toe restriction (a) and the increment rule (b). No shape rule was found in the pages read, so none is applied (call 6).

## 2. What changed, and why

1. **The refinement.** `search_refine.refine_polyline` shifts one point at a time, all others held: interior points up and down, end points along the ground. For each point the better of its two shifts is taken where it lowers F by more than the solver's tolerance. Passes repeat until none does. Then the increment is halved, from the starting increment down to the final one, with the last level at the final increment. Every shift is scored by the search's own `_compute_fos`, so step 3's checks, step 6's method rule and a `fos_fn` (the drawdown search) all hold. The starting increment is 0.10 x the ground's relief in the windows; the vertex count is 11 (a start with fewer gets points on its longest segments, so the surface is unchanged). Both are module calls, stated in `choices`.
2. **The final increment** is 0.10 x the least vertical thickness of the thinnest stratum in the windows' span, down to the bottom of the deepest layer. A layer the ground cuts at every x of the span, a surface layer, takes its greatest thickness below the ground. `refine_fraction` (the option, at most 0.25; above that it is refused). The result names the stratum, its thickness, where it is thinnest and the increment (`refinement.final_increment`).
3. **The starts.** `search._refines_noncircular` wraps `search_noncircular`, `search_de`, `search_pso` and `search_weak_layer_biased`. The sampler runs under `1 - _REFINE_SHARE` of the time left, then `refinement_stage` refines each start separately. The starts are: the kind's own best surface; each locally critical circle (`circle_starts_of`: the circular search's critical circle, its separate grid minima and the toe and above-toe families' critical circles), as the polyline of its slice-base points on 10 slices; a wedge per layer; and `seed_surface`, now taken by every kind and by `search_critical_surface` (DE also seeds its population with it). `search_critical_surface` passes the circles of the circular search it already runs for `circle_check`. With `circle_check=False` there are no circle starts, and `refinement.circle_starts` says so. DE's internal seeding search runs with `refine=False`.
4. **A wedge along each layer.** `wedge_starts`: every layer holding ground below the ground surface within the windows' span and the search's depth. The central block runs on the layer's mid-line at each x (`_layer_band_at`), its two ends at 21 positions each over the span. The toe-side wedge rises at 45 - phi'_D/2, the other at 45 + phi'_D/2, with phi' of the material just above the base end (45 degrees where phi' = 0). The passes use F = 1 first, then the F each pass gave, 3 passes at most; the lowest pass is kept. The lowest wedge of each layer is that layer's start. The weak-layer search's random aiming and its weight `1/(c + 50 tan phi)` are removed (`_weighted_choice` with them). `search_weak_layer_biased` is now the wedge stage; with `refine=False` that stage alone is the result. Its `n_trials`, `n_points` and `seed` no longer change anything; the adapter's `applies_when` and the defaults test say so.
5. **Locally critical surfaces.** `SearchResult.local_minima` (new field, in `to_dict` when not empty): each start's minimum, lowest first, with `start`, `F_start`, `FOS`, `points`, `flags` (the printed checks, from the same scoring call), `refined`, `levels`, `n_scored`, `toe_limit_deg` and `eq_c32_deg`. The headline is the lowest of these and the sampler's own best. `SearchResult.refinement` (new field) holds the increments, the wedges (one per layer, with FOS and points), the circle-start note, `unrefined`, `best` and `n_scored`. A circular grid search lists its grid minima in `local_minima`: centre, radius, grid FOS (the screen's where screened), whether the walk started from it and, where not, why (the cap `_REFINE_STARTS`, `_REFINE_WITHIN`, or the time budget). That list is a record only: no circular value changes.
6. **No jumping faces.** An end may not cross any face group's crest from the side it started on. An end that started on a face group may not move onto another face group. An end on a flat may move onto the face whose toe bounds its flat.
7. **The toe (call 4).** A shift is refused where it makes the toe-end segment (the end with the lower ground) steeper than EM Eq. C-32's 45 - phi'/2. phi' is that of the material at the segment's middle. Where the start's own exit is steeper still, the limit is the start's exit. The brief says "nor steeper than the start's own exit". Read as two separate limits, a start already steeper than C-32 could not be refined at all, so the larger of the two is taken (Calls made, 1).
8. **Windows (call 5).** Windows the caller gives bound the ends of every shift and of every wedge. Automatic windows bound only the samplers. Under automatic windows the refinement's trials also do not anchor the module's span screen on the windows: `_noncircular_admissible` then takes half the slope's height, its own fallback (Calls made, 2).
9. **Shape (call 6).** No shape rule. The jagged screen stays only where step 3 kept it.
10. **The switch (call 8).** `refine` (default True) switches this refinement as well as the circular grid's walk. It is in the library (`search_critical_surface` and each noncircular search), in the adapter (`refine`, `refine_fraction`, `seed_surface`, with their `applies_when`), in `choices` (`search_refine.refinement_choices`: the refinement, the final increment, the starting increment and vertices, the starts, the wedges, the toe, the windows and faces, the shape), in `defaults_used`, in the manual (rebuilt) and in a new probe (`#noncircular_refine_fraction`, `method_probes.json`). The refinement is a stage of the time budget. Where the budget stops it, `refinement.unrefined` names the starts left, and `search_ranges.time_budget` records a stop of stage "noncircular refinement" with "starts left unrefined: ..." (step 10's wording, through `note_budget`). The record carries no run time, so a result does not change from run to run.

**Places the rule now holds:** `search_noncircular`, `search_de`, `search_pso`, `search_weak_layer_biased` (and so every noncircular kind of `search_critical_surface`); `search_rapid_drawdown`'s `noncircular` and `noncircular_de` kinds (through their `fos_fn`: refined with the drawdown F, no circle starts, since that search runs no circular search); `compare_methods_table` and `analyze_slope` score given surfaces and do not search; the yield acceleration and Newmark take a given surface; FOSM and Monte Carlo with `research_surface` (`probabilistic.py`, through `search_critical_surface` with the caller's kwargs); `reliability/wrappers.py` (through the probabilistic functions); `geo_project/builders.py`'s limit-equilibrium builder (`search_critical_surface` with the project's `surface_type`); the agent's adapter (`funhouse_agent/adapters/slope_stability.py`, with the new options); and the slope report package (`funhouse_agent/adapters/calc_package.py`, `search_critical_surface` at its default `refine`). The circular search is unchanged except for the new `local_minima` record.

## 3. Before and after on a worked example

ACADS 4 (Slide2 problem #9, V-028), weak-layer search, Spencer, Table 9.2's loads on, windows (20, 50) / (58, 84):

| | before (master 7a5bb26) | after |
|---|---|---|
| weak-layer search, Spencer | 0.830, flagged (700 random trials, seed 3) | 0.697 refined (Table 9.5: 0.707, -1.4 %); wedge stage 0.804 (Table 9.4: 0.760, +5.8 %) |
| headline flags | side_tension, thrust_outside | base_tension, side_tension, thrust_outside (lowest unflagged 1.455) |
| final increment | (none) | 0.1 m (weak, 1.0 m thick) |

## 4. The section 11 table

Scripts: `module_work/gm140/step4_section11.py`, unchanged (every pinned test of the validation set), and `module_work/gm140/step7a_section11.py` (new; this step's noncircular rows). Both were run on a checkout of master 7a5bb26 (before) and on this branch (after). The run times are from one development container; the two trees ran one after the other.

**The pinned validation set (`step4_section11.py`):** master 391 passed / 1 skipped / 3 xfailed; branch 392 passed / 1 skipped / 4 xfailed, no line MOVED (V-028's one strict xfail became one pass and two strict xfails). The only pinned values that move are the noncircular ones below. Every circular benchmark (B1 to B10, B12, B13, B16 to B18 and the circular parts of the others) holds on both trees.

| Benchmark (all Spencer unless named) | before | after | printed | what moved it |
|---|---|---|---|---|
| ACADS 4 weak-layer, Spencer, refined | 0.8301 (flagged) | 0.6971 (flagged) | Table 9.5: 0.707 (-1.4 %) | wedge starts (item 4), refinement (item 1) |
| ACADS 4 weak-layer, Spencer, wedge stage (refine=False) | (no such stage) | 0.8043 | Table 9.4: 0.760 (+5.8 %) | printed wedge inclinations (section 7) |
| ACADS 4 weak-layer, Morgenstern-Price, refined / wedge stage | 0.8158 | 0.6769 / 0.7514 | Table 9.5 GLE 0.683 (-0.9 %) / Table 9.4 GLE 0.720 (+4.4 %) | as above |
| ACADS 4 weak-layer, Janbu corrected, refined / wedge stage | 0.8301 | 0.6894 / 0.7551 | Table 9.5 0.699 (-1.4 %) / Table 9.4 0.734 (+2.9 %) | as above |
| ACADS 4 random, 700 trials, seed 3 | 0.9773 | 0.6971 | (weak-layer result) | the same starts as the weak-layer search: no more than it, as expected |
| V-028 DE, seeds 1 / 2 / 3 | 0.8473 / 0.7849 / 0.7979 | 0.6945 / 0.6971 / 0.6971 | Table 9.5 0.707 | refinement of the DE best and the wedge start |
| V-043 sand, DE seed 1 | 1.1941 (circle polyline; DE's own 1.237) | 1.1831 | Slide2 1.188 (-0.4 %), Tandjiria 1.192 | the critical circle's start refined (item 1, 3); below the circle 1.1859 |
| V-043 sand, random, 2000 trials, seeds 1 to 5 | 1.1941 each (the circle trial); the samples' own 1.34 to 1.46 | 1.1831 each | Slide2 1.188 | spread of the headline 0 (D12: 1.149 to 1.270 in the account, before step 6) |
| ACADS 1(a), random, 200 trials, seed 42 | 0.9844 (circle polyline) | 0.9844 (circle polyline); own minimum 1.0916 -> 0.9862 | 1.00 (Slide2 0.986) | own starts refined; the critical circle on 10 slices does not refine below 0.9862 |
| B14, circle search | 0.8822 | 0.8822 | (no printed answer) | unchanged (circular) |
| B14, random, default windows, seed 1 | 6.3656 | 1.2762 | (no printed answer; circle 0.8822) | the wedge along the weak layer, free of the automatic windows (call 5); short of the circle, section 7 |
| B14, DE directed (18-30 / 30-50), seed 1 | 5.2352 | 2.3549 | (circle 0.8822) | the wedge start; the given windows still anchor the span screen, section 7 |
| Duncan Ex. 1, D = 2, DE seed 1 | 0.6796 | 0.6791 | Taylor 0.706 | a surface along the firm base (7b's composite mechanism): reported, not refused |
| Duncan Ex. 1, D = 2, random seeds 1 / 2 / 3 | 0.7086 each | 0.6824 / 0.6869 / 0.6869 | Taylor 0.706 | refinement along the base (as above) |

Run times (s), before -> after: ACADS 4 weak-layer 9.5 -> 4.3 (refine=False) and 13.5 (refined); Morgenstern-Price 9.6 -> 4.0 / 14.7; Janbu 2.0 -> 1.9 / 3.9; ACADS 4 random 11.9 -> 23.8; V-028 DE 7.0-7.4 -> 17.7-18.1; V-043 DE 5.9 -> 10.2; V-043 random 2000 trials 12.9-13.5 -> 17.3-18.6; ACADS 1(a) random 3.1 -> 3.5; B14 random 10.2 -> 17.6, DE 11.2 -> 16.5; Duncan Ex. 1 DE 5.6 -> 6.5, random 5.6-5.8 -> 6.3-6.6. The adapter's `refine` description carries the ranges.

## 5. Calls made

1. **The toe limit's reading.** The larger of Eq. C-32's angle and the start's own exit. The other form, both as separate limits, would leave a start steeper than C-32 unrefinable.
2. **The span screen under automatic windows.** The refinement's trials anchor the module's span screen (D22, not in print) on the windows only where the caller gave both. Otherwise the screen takes half the slope's height. This follows call 5 (automatic windows bound only the samplers). Other form: the automatic windows' span, which refused B14's wedge and every short start.
3. **Wedge ends under automatic windows.** The wedge stage is a start, not a sampler. Its ends on the ground are bounded only by windows the caller gives (call 5).
4. **Starting increment.** 0.10 x the ground's relief in the windows, halved to the final increment. Where the final increment is the larger (thick strata), there is one level at the final increment.
5. **Vertex count, wedge positions and passes.** 11 vertices, 21 positions per end, 3 passes of F. None is printed. None was chosen to make a case pass: they were set before any benchmark was run and not changed after.
6. **Circle starts on 10 slices**, the polyline of the slice-base points (`circle_as_polyline` with `N_VERTICES - 1` slices). The full 30-slice circle polyline is still a trial through `add_circle_trial`.
7. **The record keeps no run time**, as `budget_choice` does not, so a result repeats.

## 6. Tests

Run by pytest's exit code (`-n 4`):

- `slope_stability/tests` (all), before the fixes: 9 failed / 976 passed / 17 skipped / 1 xfailed. Each failure was a test of a sampler's own mechanics or a value the refinement moves. On the final tree: **999 passed / 17 skipped / 1 xfailed / 0 failed**.
- `slope_stability/tests/test_search_refine.py` (new, 14 tests): passes. The tests were written after the code (this file says so). Each uses the step's new API (`refine`, `local_minima`, `refinement`), so on master they cannot run.
- The `validation_examples/test_published_*` files that import slope_stability and the `calc_package`, `funhouse_agent`, `reliability`, `fem2d`, `geotech_common`, `report_ingest`, `dxf_import` and `geo_project` test files that name slope_stability: 3 failed (V-043, re-pinned against print below) / 2241 passed / 4 xfailed; then V-043's file 11 passed.
- `funhouse_agent/tests/test_defaults_used.py -k slope` (and the probe tables): pass after the weak-layer settings row and the new probe (the probe found a missing import in `seed_surface`'s list form, fixed).
- `webapp/tests/test_page_fingerprints.py`: 13 passed (the pages' fingerprints do not cover these descriptions; nothing to re-record).
- `webapp/tests/test_public_text.py`: 5 passed before each commit.
- `module_work/doc10/check_manual.py --rebuild --write`, then `check_manual.py` and `webapp/tests/test_user_manual_current.py`: exit 0 each, with geotech-references at e1bbc5b (v1.7.0, the pin) first on PYTHONPATH.

Tests changed and why:
- `test_published_v028_e8.py`: the strict xfail against Table 9.4 for the old random search is replaced. The refined search is tested against Table 9.5 at 3 % (passes, -1.4 %). The wedge stage against Table 9.4 at 3 % and the coverage assertion against the declared family are two strict xfails with the reason (section 7).
- `test_published_v043_slope_reinforced.py`: 1.194 -> 1.183, printed 1.188 (-0.4 %). The band test of DE's own minima (1.22 to 1.25, "the generator's gap") becomes "no more than the circle, and repeats".
- `test_gle_root.py`: the search's own minimum 1.0916 -> 0.9862, printed 1.00 (-1.4 %). The headline, 0.9844, is unchanged.
- `test_search_supplements.py::test_profile_end_check_covers_noncircular_surfaces`: 2.252 -> 2.221 (made-up slope, nothing printed; the circular search's profile-end circle is 2.237).
- `refine=False` added where the test is of a sampler's own mechanics (its grid_fos count, its flags, DE's seeding fallback and its 2,326-surface budget, the short-search report): `test_acceptability`, `test_m_alpha` (2), `test_choices`, `test_slope_stability`.
- `test_new_search.py::TestWeakLayerBiased::test_grid_fos_populated`: no random trials now; asserts the wedge record instead.
- `test_ground_crossings.py`: the weak-layer search no longer draws polylines that cross the ground. A `seed_surface` that leaves the ground and comes back is refused and counted instead.
- `test_defaults_used.py`: the weak-layer row lists `refine` as used and `n_trials`/`n_points` as not.

## 7. What stopped short, and noted, not fixed

- **ACADS 4's wedge stage at the printed inclinations is above Table 9.4 and above the declared family.** Spencer 0.804 against 0.760 (+5.8 %) and the family's 0.748; Morgenstern-Price 0.751 against GLE 0.720 (+4.4 %). Janbu, 0.755 against 0.734 (+2.9 %), is inside 3 %. The family's lowest surface has its bench (passive) wedge at 25 degrees. C-1g(2)'s passive inclination on the trial's F is 31 to 33 degrees, and on the family's own base ends (47, 76) the printed angles give 0.79. So the gap is the fixed inclinations, not the positions. Slide2's block search varies the projection angles, and EM Fig. C-7b shows searched inclinations, but the brief's item 4 fixes the angles. A wedge-stage option that varies the inclinations about C-1g(2)'s values (Fig. C-7b) is left for the owner. The refined result is below Table 9.5 in all three methods (-0.9 to -1.4 %). Two strict xfails carry the reason. Nothing was re-pinned or widened.
- **B14 stays above its circle.** Random reads 1.276 and directed DE 2.355, against the circle search's 0.8822. That critical circle (x 26.0 to 30.66) spans 4.66 m. The module's span screen (`_noncircular_admissible`, D22, the module's own, not in print) refuses any polyline shorter than half the slope's height, 5 m (or half the given windows' span), so the circle's start is not scored (`local_minima`: "the start is not scored") and the refinement cannot reach that mechanism. Not changed here (step 3 kept the screen). It changes B14's reported value, so it is a candidate follow-up: the span screen against print (D22).
- **Duncan's Example 1, D = 2.** DE 0.679 and random 0.682 to 0.687 refine to surfaces along the firm base, below Taylor's 0.706. As the brief says, this is 7b's composite mechanism; it is reported with its flags, not refused.
- **Thick strata give a coarse final increment.** With the thinnest stratum 24 m thick (Duncan's Example 1), the printed bound gives 2.4 m and the refinement runs one level. That is within the print ("no more than"); `refine_fraction` makes it finer. Noted, not fixed.
- **The account's D12 cite** of EM p. 4-7 for a concave-up shape: that page prints no shape rule (section 1).
- **Tests written after the code.** The brief asks for failing tests first. The new tests use the step's new API, so on master they could not run. The before column of section 4 is the reproduction of the departures (master's values).
- Out of scope, untouched: 7b (D14), the samplers themselves, a refinement for the entry-and-exit search (GM-89), D11, GM-89's sliver tolerance, varying loads.

## 8. Release note

Every noncircular search (random, differential evolution, particle swarm and weak-layer) now finishes as EM 1110-2-1902 C-1g(3) describes. It refines several starts one point at a time down to a final shift of 10 % of the thinnest stratum: its own best surface, the locally critical circles of the circular check search, a wedge along each layer, and a surface you give (`seed_surface`). It reports the lowest and lists every start's minimum (`local_minima`). Noncircular factors of safety fall, often substantially: ACADS 4's weak-layer search reads 0.697 against Slide2's 0.707 (it read 0.830), and V-043's noncircular search reads 1.183 from every seed against Slide2's 1.188 (1.194). The weak-layer search is now a wedge search along each layer, with no random trials, so `n_trials`, `n_points` and `seed` no longer change it. `refine=False` restores the sampler alone (for the weak-layer search, the wedge stage alone). Run times grow by about 1 to 2 times. Circular searches are unchanged; they now list their separate grid minima.

## 9. Fix round after review

Branch `fix/gm140-step7a-review` (from 2c87e4a), master 6e2b94c merged first (04c5f6e; PL-42's notes and app merge, no slope file). An independent review passed the build with defects (D1 to D10); the orchestrator's rulings (R1 to R4 and the calls) are binding. Where this section differs from sections 2 to 7, this section holds. Every new constant was decided before a benchmark was run with it.

**What changed, and where.**

| Item | What | Where |
|---|---|---|
| D1 | A circle start whose F is None (a walk with no solution, a node valued by the Bishop screen) is no value: kept, sorted last, no crash under a time budget | `search_refine.circle_starts_of` |
| D3, ruling 9(b) | Every start is scored as it stands before it is refined; its minimum is the lower of the two, labelled (`minimum`: "as it stands" or "refined"); a start the budget leaves unrefined keeps its value. A stop of the wedge stage is recorded as a stage of `search_ranges.time_budget` and in the warning, naming the layers and positions not built | `search_refine.refinement_stage`, `wedge_starts`; `search._run_refinement` |
| D5 | `search_rapid_drawdown` passes `refine`, `refine_fraction` and the windows given to its noncircular kinds; its `choices` carry `refinement_choices`; the adapter's drawdown tool takes `refine` and `refine_fraction` (applies_when, defaults_used, the manual's row, the probe `#noncircular_refine_off`) | `rapid_drawdown._drawdown_search`; `funhouse_agent/adapters/slope_stability.py`; `method_probes.json` |
| D6 | The wedge's central block on the mid-line between the layer's bottom and the lesser of its top and the ground; a layer with no wedge says why | `search_refine._wedge_mid`, `wedge_starts` |
| D8 | `refine_fraction` declares `"default": 0.1`; `seed` no longer applies to the weak-layer search | adapter |
| D9 | V-043: the 2,000-trial random headline no more than the circle (or DE's lowest + 0.002), the same seed run twice the same; the bound against DE restored from -0.002 to -1e-9 | `test_published_v043_slope_reinforced.py` |
| D10 | The run-time text re-measured after the rulings (below) | adapter, `refine` |
| Ruling 8 | Inside the refinement, and for every start as it stands, the span floor is the lesser of the module's floor (D22) and the start's own span; the samplers keep step 3's screen | `search.module_span_floor`, `start_span_floor`, `_noncircular_admissible(span_floor=)` |
| Ruling 9(a) | Final increment = min(fraction x thinnest stratum, starting increment / 4), the starting increment never raised; `choices` states it as the module's call (EM 4-2e(2)(b) prints a maximum only) | `refinement_stage`; `refinement_choices` |
| Ruling 9(b) | A circle start as it stands is the polyline of its slice bases on the search's `n_slices`; the refinement keeps the 11-vertex polyline; each circle start reports `F_circle`, `F_start`, `F_refine_start`, `F_refined` | `refinement_stage`; `search.circle_start` |
| Ruling 10 | The best wedge's two ends halved, one at a time, inclinations fixed, from half the spacing down to the final increment; a result of the wedge stage alone says its inclinations are fixed and that it is not a minimum over them | `wedge_starts`; `search.WEDGE_STAGE_ALONE` |
| Ruling 11 | The toe limit max(Eq. C-32, the start's exit), unchanged | (none) |
| Ruling 12 | DESIGN.md "Calls made" for step 7a: the seven calls of section 5 and rulings 8 to 11, each with both forms | `slope_stability/DESIGN.md` |

**The probe.** The reviewer's `probe.py`, run once on the fixed tree (074d569); its only change: the record keeps the fixed code's new keys (`F_circle`, `F_refine_start`, `F_refined`, `minimum`; the wedges' `ends`, `why`). Times are from this laptop: master's and the review branch's from the review's runs (2026-11-14; their output files were written 11 minutes apart, so the two runs overlapped and their times may read high), the fixed tree's from one run alone (2026-11-15).

Check (a): all 18 circular cases identical on master, the review's branch and the fixed tree to 6 decimals, with the same centre, radius and ends.

| Case (Spencer) | master | review | fixed | flags (fixed) | lowest unflagged | time master / review / fixed (s) |
|---|---|---|---|---|---|---|
| V-028 weak layer, wedge stage (refine=False) | 0.8301 | 0.8043 | 0.7759 | base_tension, side_tension, thrust_outside | 1.4552 | 26.8 / 11.7 / 13.2 |
| V-028 weak layer, refined | 0.8301 | 0.6971 | 0.6961 | as above | 1.4552 | 26.8 / 37.6 / 41.1 |
| V-028 random, 700 trials, seed 3 | 0.9773 | 0.6971 | 0.6961 | as above | 1.4552 | 34.2 / 65.9 / 70.5 |
| V-028 DE, seeds 1 / 2 / 3 | 0.8204 / 0.8223 / 0.8202 | 0.6942 / 0.6956 / 0.6971 | 0.6942 / 0.6956 / 0.6961 | as above | 1.4552 | 16.5-19.4 / 48.0-53.8 / 51.3-56.4 |
| V-043 DE, seeds 1 to 3 | 1.1941 | 1.1831 | 1.1801 | none | | 15.9-16.6 / 30.0-30.4 / 35.6-36.7 |
| V-043 random, 2000 trials, seeds 1 to 5 | 1.1941 | 1.1831 | 1.1801 | none | | 37.1-39.6 / 51.1-52.9 / 57.9-60.9 |
| ACADS 1(a) random, 200 trials, seed 42 | 0.9844 | 0.9844 | 0.9844 | side_tension | 0.9859 | 8.7 / 9.6 / 16.0 |
| B14 random, default windows, seed 1 | 6.3656 | 1.2762 | 0.8827 | base_tension, side_tension, thrust_outside | 2.3921 | 27.4 / 47.8 / 53.2 |
| B14 DE, directed (25.5-28 / 29-40), seed 1 | 0.8966 | 0.8846 | 0.8827 | as above | 1.8335 | 24.8 / 27.0 / 38.3 |
| Duncan Ex. 1, D = 2, DE seed 1 | 0.6796 | 0.6791 | 0.6632 | as above | 1.4688 | 15.3 / 18.0 / 35.0 |
| Duncan Ex. 1, D = 2, random seeds 1 / 2 / 3 | 0.7086 | 0.6824 / 0.6869 / 0.6869 | 0.6632 | as above | 1.4688 | 15.7-15.9 / 18.3-18.8 / 35.7-38.9 |

Every fixed value is at or below master's. ACADS 4: the wedge stage 0.7759 against Table 9.4's 0.760 (+2.1 %; its strict xfail removed), still above the declared family's 0.748; refined 0.6961 against Table 9.5's 0.707 (-1.5 %). V-043: 1.1801 from every DE and random seed, below 1.1831 (the review's) and the circle (1.1859); printed 1.188 (-0.7 %). Starts: 131 checked, none above itself as it stands; 8 not scored as they stand (V-043's toe circle in each run: "the start could not be built as a polyline").

Circle starts against their circle (one row per case; the other runs of a case give the same):

| Case | start | F circle | as it stands (`n_slices`) | 11-vertex start | refined | minimum |
|---|---|---|---|---|---|---|
| ACADS 1(a) | critical circle | 0.9838 | 0.9844 | 0.9862 | 0.9859 | 0.9844 as it stands |
| B14 random / DE | critical circle | 0.8822 | 0.8827 | 0.8846 | 0.8846 | 0.8827 as it stands |
| V-043 | critical circle | 1.1859 | 1.1941 | 1.2025 | 1.1801 | 1.1801 refined |
| V-028 | critical circle | 0.9765 | 0.9773 | 0.9842 | 0.7321 | 0.7321 refined |
| Duncan Ex. 1, D = 2 | critical circle | 0.7044 | 0.7086 | 0.7094 | 0.6632 | 0.6632 refined |

The headline above the circle's own F (ruling 9's list), each said in the result with both values ("The noncircular search's lowest surface (FOS 0.8827) is above the critical circle's value by the same method (Spencer, FOS 0.8822 as a circle, 0.8827 as the polyline of its 30 slice bases; circle_check). Every start, that polyline among them, was scored as it stands and refined one point at a time (EM 1110-2-1902 C-1g(3), p. C-10), and none reached below the circle: report the circle's value beside this one."): ACADS 1(a) +0.0006 (0.9844 against 0.9838); B14 random +0.0005 and B14 DE +0.0005 (0.8827 against 0.8822).

Run time (D10): worst 70.5 s (V-028 random); fixed against master, noncircular 1.6 to 2.5 times, noncircular_de 1.5 to 3.2 times, the weak-layer search 1.5 times refined and 0.5 times as the wedge stage. The adapter's `refine` text carries these.

**The review's three failing paths, re-run.**
- V-043, `noncircular_de`, `time_budget_s=6`: no error; 1.1941 (the critical circle's start as it stands); `search_ranges.time_budget` records the stops of differential evolution, the wedge stage (the positions not built along fill and softclay) and the refinement, and `local_minima` lists eight starts "not refined: the time budget (GM-123); its value is the start's as it stands".
- V-028 weak layer, `refine=False`, `time_budget_s=0.5`: the wedge stage's stop is recorded (0 of 63 low-end positions; the circular check spent the budget) and named in the warning, each layer saying why it has no wedge; the wedge-stage statement is given.
- V-041 #98, `search_rapid_drawdown(..., surface_type="noncircular", refine=False)`: 0.9771 in 2.7 s, the sampler alone (`refinement` {"refined": false}, choices "off"); refined: 0.8438 in 26.6 s.

**Other values that moved (made-up slopes, nothing printed; each lower).**
- `test_search_supplements::test_profile_end_check_covers_noncircular_surfaces` (`_toe_geom`): refined, the search reads 2.138 (flagged) on a surface ending at the left end, and no circle past the end is 1 % lower, so the clipped-range statement is given instead; the profile-end check is tested on the sampler alone (2.479; the circle past the end 2.219).
- `test_search_supplements::test_infinite_slope_mirrored_and_noncircular_search` (`_fill_geom`, mirrored): the circular search's own critical circle, a shallow slide on the face (0.942), is a start as it stands under ruling 8 and reads below the planar family (0.944), so the noncircular search governs and the result names the infinite-slope mechanism, as where a circle governs; the sampler alone still leaves the planar family governing.
- Duncan Ex. 1, D = 2: 0.6632 (flagged; lowest unflagged 1.4688), 6 % below Taylor's 0.706: the critical circle's start refined along the firm base, 7b's composite mechanism (section 7), reported with its flags.

**Section 7, after the fix round.** B14 now reaches its circle (0.8827 against 0.8822, said with both values). ACADS 4's wedge stage is within 3 % of Table 9.4 but above the declared family (fixed inclinations; an inclination band is the other form, not built). The coarse final increment on thick strata is closed by ruling 9(a).

**Description sizes** (`describe_method`, the deep agent's 16,000-character cap): `search_critical_surface` 15,953 returned (shortened; its full document 38,771, as on the review's branch); `search_rapid_drawdown` 15,475 returned (shortened; full 20,831, the two new parameters +1,119 characters of METHOD_INFO); `rapid_drawdown_fos` 15,823, whole and unchanged.

**Tests** (one file at a time, no `-n`, by pytest's exit code; this laptop):

| File | Result | Exit |
|---|---|---|
| `slope_stability/tests/test_search_refine.py` | 24 passed | 0 |
| `slope_stability/tests/test_new_search.py` | 19 passed | 0 |
| `slope_stability/tests/test_ground_crossings.py` | 21 passed | 0 |
| `slope_stability/tests/test_search_supplements.py` | 18 passed | 0 |
| `slope_stability/tests/test_gle_root.py` | 33 passed | 0 |
| `slope_stability/tests/test_rapid_drawdown.py` | 17 passed | 0 |
| `slope_stability/tests/test_choices.py` | 7 passed | 0 |
| `validation_examples/test_published_v028_e8.py` | 10 passed, 1 xfailed (the family, strict, reason corrected) | 0 |
| `validation_examples/test_published_v028_slope.py` | 3 passed | 0 |
| `validation_examples/test_published_v043_slope_reinforced.py` | 11 passed | 0 |
| `funhouse_agent/tests/test_defaults_used.py` (in full) | 906 passed, 24 skipped | 0 |
| `funhouse_agent/tests/test_top_level_params.py` | 53 passed | 0 |
| `funhouse_agent/tests/test_gm179_unused_inputs.py` | 20 passed | 0 |
| `funhouse_agent/tests/test_allowed_values.py` | 322 passed | 0 |
| `funhouse_agent/tests/test_probe_warnings.py` | 8 passed | 0 |
| `funhouse_agent/tests/test_render_tables.py` | 17 passed | 0 |
| `funhouse_agent/tests/test_calc_package_figure_layout.py` | 15 passed | 0 |
| `funhouse_agent/tests/test_gm140_step10_package_inputs.py` | 17 passed | 0 |
| `funhouse_agent/tests/test_fem2d_mesh_study_adapter.py` | 6 passed | 0 |
| `funhouse_agent/deep/tests/test_wave1_geotech.py` (in full) | 20 passed | 0 |
| `module_work/doc10/check_manual.py --rebuild --write`, then `check_manual.py` | OK | 0, 0 |
| `webapp/tests/test_user_manual_current.py` | 1 passed | 0 |
| `webapp/tests/test_public_text.py` (before each commit) | 5 passed | 0 |

Tests changed in the fix round: `test_search_refine.py` (D1 on made-up records; D3, the wedge stage's stop; D5, the drawdown's refine=False; D6, a wedge for the upper layer and why for a layer without one; ruling 8, a start shorter than the floor and the B14 pair; ruling 9(a), the quarter rule and a thick stratum halving at least twice; ruling 9(b), tests (i) and (ii); ruling 10, the halved ends and the wedge-stage statement); `test_gle_root.py` (ACADS 1(a): the search's own minimum 0.9862 -> 0.9844, the circle start as it stands; governs False and the sentence with both values); `test_published_v028_e8.py`; `test_published_v043_slope_reinforced.py`; `test_search_supplements.py` (the two made-up cases above); `test_defaults_used.py` (`seed_surface`'s absence is no case: a start the call adds); `method_probes.json` (the drawdown probe `#noncircular_refine_off`, without windows: with every window given the search-ranges choice is written "given" by the module, `search_supplements` about l. 2787, a call outside the refinement, left as it is and noted for the lead).

**The lead's check before the merge (2026-11-15).** V-043's headline surface (DE seed 1 and random seed 1 give the same 11 points) re-solved directly by Spencer's method outside any search (`analyze_slope` and `gle_fos` with the constant function): 30 slices 1.18009 (theta 9.11 degrees, lambda 0.1604), 480 slices 1.17989 (-0.02 %); F by moments equals F by forces and the last interslice force is 0 on both, so it is a root of the method; no flags on either. The merge's terms (the orchestrator's: a root with no flags, no more than 0.3 % move on 480 slices) are met.
