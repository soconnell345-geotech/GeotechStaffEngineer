# GM-140 step 9: rapid drawdown as EM 1110-2-1902 Appendix G prints it

Branch `claude/gm-140-step9-rapid-drawdown`, from master d80890c (step 1 merged; `STEP1_CHECK.md` on master). Built 2026-11-13 by a cloud session. Brief: `module_work/cloud_briefs/SLOPE_GM140.md`, section "Step 9".

**The rule.** Each stage is computed as Appendix G prints it, each stage's factor of safety is reported, and the answer is the factor of safety of the last stage computed.

Departures closed: D38; OWN-17; GM-15; GM-135; D16's drawdown half; GM-130's drawdown item (in part, see "Calls made" 12).

## 1. What the print says

EM 1110-2-1902 (31 Oct 2003), Appendix G, pp. G-1 to G-23, and Table 2-1, p. 2-2. The manual is the owner's copy and is in no repository. Everything below is from the equations and the worked example's values that the brief gives, each with its page; nothing was written from memory.

| Step | Print | Page |
|---|---|---|
| First stage | pore pressures before drawdown; sigma'_c = N'/dl (Eq. G-1), tau_c = S/dl (Eq. G-2) | G-3a, p. G-2 |
| K_c = 1 strength | d = c_R cos(phi_R) cos(phi') / (1 - sin phi_R) (Eq. G-5); psi = arctan[sin(phi_R) cos(phi') / (1 - sin phi_R)] (Eq. G-6); tau_ff = d + sigma'_c tan(psi) (Eq. G-9). Or d and psi as measured | p. G-6; p. G-7; G-3b(1)(a), (b), pp. G-4 to G-5 |
| K_c = K_f strength | tau_ff = c' + sigma'_c tan(phi') (Eq. G-10); K_f by Eq. G-7 (c' = 0) or G-8 | p. G-7 |
| K_c | [sigma'_c + tau_c (sin phi' + 1)/cos phi'] / [sigma'_c + tau_c (sin phi' - 1)/cos phi'] (Eq. G-11; again as G-24), held between 1 and K_f | p. G-7; p. G-17; G-3b(2)(c) |
| Second-stage strength | tau_ff = [(K_f - K_c) tau_ff(K_c = 1) + (K_c - 1) tau_ff(K_c = K_f)] / (K_f - 1) (Eq. G-12) | p. G-8 |
| Second stage | tau_ff as cohesion, phi = 0, no pore pressure on slices that do not drain freely; c', phi', u after drawdown on free-draining ones; water left against the slope an external load | G-3c, p. G-8 |
| Drained check | sigma = N/dl, N from the second stage (G-13); sigma'_d = sigma - u after drawdown (G-14); s_d = c' + sigma'_d tan(phi') (G-15) | G-3d, p. G-8 |
| Third stage | "the same conditions as for the second-stage computations" except c', phi' and u after drawdown on the slices whose drained strength is lower; only if a slice needs it | G-3e, pp. G-8 to G-9 |
| Corps 1970 procedure | the R envelope's c_R and phi_R as entered, composite with the effective-stress envelope; no third stage | G-2, pp. G-1 to G-2 |
| Water | level piezometric lines at the maximum and the lowered pool "for simplicity"; seepage analyses would often be appropriate. Steady seepage before drawdown | G-4b, p. G-9; Table 2-1, p. 2-2 |
| Example | Bishop: 2.20, 1.52, 1.44. Modified Swedish at Spencer's inclinations: 2.23, 1.52, 1.44. Corps 1970, Modified Swedish at 19.4 deg: 3.49, 1.35 | pp. G-15, G-18, G-20; G-20 to G-22; G-9 to G-15 |

## 2. What changed, and why

`slope_stability/rapid_drawdown.py` (the stages and the result class are rewritten; the search keeps its machinery):

1. **K_c = 1 strength** by Eqs. G-5, G-6 and G-9 for `lowe_karafiath` and `duncan_3stage` (`r_envelope_d_psi`). A layer may give d and psi instead: new `SlopeSoilLayer.R_d`, `R_psi` (appended last in the dataclass). `corps_2stage` keeps c_R and phi_R as entered and refuses a layer that gives only d and psi. Before: c_R and phi_R as entered in every procedure, 13 % low on the manual's example.
2. **K_c** by Eq. G-11, held between 1 and K_f (`consolidation_stress_ratio`). The base inclination and the two special branches are gone. Before: 1.68 to 3.00 on the example's slices; print 1.71 on all twelve.
3. **The second stage is solved and kept**: `stage2_fos` and its N per slice. It is the answer of `lowe_karafiath` and `corps_2stage`.
4. **The drained check uses the second stage's N** (`drained_check`, Eqs. G-13 to G-15). `stage3_effective_normal` ('fellenius' and 'gle') is removed; a caller who names it is refused with the rule (`STAGE3_NORMAL_RULE`; the adapter declares it under `refused_parameters`). This settles OWN-17.
5. **The third stage is solved only if a slice needs it**, with c', phi' and the after-drawdown pore pressure on those slices. Before: the drained strength was entered as a cohesion with phi = 0, in one merged solve.
6. **Water, stated**: `stage1_phreatic_points` as before; new `drawdown_phreatic_points`; left out, level pools, named in `water_before`, `water_after` and `choices` as the manual's own assumption.
7. **The result carries each stage**: `stage1_fos`, `stage2_fos`, `stage3_fos`, `answer_stage`, `stage_method`, `no_solution`, `notes`, `choices`, and per slice the columns of Figs. G-10b and G-11b (`slice_table()`, `to_dict()["slices"]`).
8. **One set of slices**: paired by index; a surface whose slices before and after drawdown differ is refused (`_same_slices`). The nearest-mid-point lookup is gone.
9. **A stage with no solution gives no number**: `FOS` is None with the reason. The Ordinary Method of Slices fall-back for the first stage is gone, and so is "the last iterate" of a final stage that did not converge.
10. **The search** (`search_rapid_drawdown`) scores every trial with `rapid_drawdown_fos` itself; a trial with no number is not scored.
11. **The manual's check method** (lead's call): `fixed_inclination_solution(slices, slip, theta_deg, equilibrium)` gives the force-equilibrium factor of safety at a fixed side-force inclination (the GLE engine's direct march at lambda = tan theta; EM C-4's Modified Swedish Method), and the Simplified Bishop Method at theta = 0 by moments. `rapid_drawdown_fos(stage_solver=..., extra_boundaries=...)` are library check arguments. None of this is on the agent's tools or in `method` names.

Other files: `slope_stability/analysis.py` (the wrapper's signature and docstring only), `slope_stability/geometry.py` (two fields), `funhouse_agent/adapters/slope_stability.py` (the two tools' text, the new input, `R_d`/`R_psi`, the refused input, `when_absent`), `docs/user_manual/manual.html` (the matching rows, hand-edited: 2 briefs twice each, `method`, `stage1_phreatic_points`, the removed row replaced by `drawdown_phreatic_points`, the `soil_layers` sentence), `slope_stability/DESIGN.md` and `VALIDATION.md` (the drawdown sections), `validation_examples/RESULTS.md` and `INVENTORY.md`, `funhouse_agent/worked_examples.json` (WE-SLOPE-5's values and notes), `funhouse_agent/geotech_test_suite.json` (RDD-2's key, RDD-3 reworded) and `docs/geotech_eval_v54_additions.md`.

## 3. The manual's worked example, before and after

Section, soil, pools, circle and the twelve slice edges as the brief gives them (EM G-4, Figs. G-5, G-6); converted to SI in the test and back for comparison. Test: `slope_stability/tests/test_rapid_drawdown_em_appendix_g.py`.

| Run | Stage 1 | Stage 2 | Stage 3 | Printed |
|---|---:|---:|---:|---|
| Before, three-stage, Spencer, 50 slices | 2.200 | not reported (Lowe-Karafiath: 1.375) | 1.241 | 2.20 / 1.52 / 1.44 |
| After, three-stage, Spencer, 50 slices | 2.200 | 1.513 | 1.434 | 2.20 / 1.52 / 1.44 (0.0 %, -0.5 %, -0.4 %) |
| After, three-stage, Bishop, the manual's 12 slices | 2.192 | 1.514 | 1.438 | 2.20 / 1.52 / 1.44 (-0.3 %, -0.4 %, -0.1 %) |
| After, Corps 1970, force equilibrium at 19.4 deg, 50 slices | 3.486 | 1.341 | none | 3.49 / 1.35 (-0.1 %, -0.7 %) |
| After, Corps 1970, the same, the manual's 12 slices | 3.455 | 1.348 | none | 3.49 / 1.35 (-1.0 %, -0.2 %) |
| Corps 1970, Spencer, 50 slices (unchanged) | 2.200 | 1.239 | none | printed by another method of slices |

Slice by slice on the manual's twelve slices, Bishop (ksf; N in kips; slice 1 at the crest). Module value / printed value:

| Slice | sigma'_c | K_c | tau_ff (K_c = 1) | tau_ff (K_c = K_f) | tau_ff | N, stage 2 | sigma'_d | s_d | Stage 3 |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 0.33 / 0.32 | 1.72 / 1.71 | 1.49 / 1.49 | 0.19 / 0.19 | 1.02 / 1.02 | -6 / -6 | -0.71 / -0.72 | 0.00 / 0.00 | drained / drained |
| 2 | 1.51 / 1.44 | 1.72 / 1.71 | 1.88 / 1.85 | 0.87 / 0.83 | 1.52 / 1.49 | 57 / 52 | 1.62 / 1.48 | 0.94 / 0.86 | drained / drained |
| 3 | 2.93 / 2.90 | 1.72 / 1.71 | 2.34 / 2.33 | 1.69 / 1.67 | 2.11 / 2.10 | 122 / 120 | 4.67 / 4.61 | 2.70 / 2.66 | undrained / undrained |
| 4 | 3.66 / 3.64 | 1.72 / 1.71 | 2.58 / 2.58 | 2.11 / 2.10 | 2.41 / 2.41 | 149 / 148 | 6.56 / 6.52 | 3.79 / 3.76 | undrained / undrained |
| 5 | 4.09 / 4.05 | 1.72 / 1.71 | 2.73 / 2.71 | 2.36 / 2.34 | 2.59 / 2.58 | 253 / 250 | 7.78 / 7.69 | 4.49 / 4.44 | undrained / undrained |
| 6 | 4.47 / 4.45 | 1.72 / 1.71 | 2.85 / 2.84 | 2.58 / 2.57 | 2.75 / 2.75 | 205 / 204 | 8.15 / 8.13 | 4.71 / 4.69 | undrained / undrained |
| 7 | 4.59 / 4.57 | 1.72 / 1.71 | 2.89 / 2.88 | 2.65 / 2.64 | 2.81 / 2.80 | 209 / 208 | 7.79 / 7.77 | 4.50 / 4.49 | undrained / undrained |
| 8 | 4.56 / 4.54 | 1.72 / 1.71 | 2.88 / 2.87 | 2.63 / 2.62 | 2.79 / 2.78 | 192 / 191 | 7.32 / 7.30 | 4.23 / 4.22 | undrained / undrained |
| 9 | 4.34 / 4.31 | 1.72 / 1.71 | 2.81 / 2.80 | 2.51 / 2.49 | 2.70 / 2.69 | 248 / 246 | 6.64 / 6.61 | 3.84 / 3.81 | undrained / undrained |
| 10 | 3.71 / 3.66 | 1.72 / 1.71 | 2.60 / 2.58 | 2.14 / 2.11 | 2.44 / 2.42 | 238 / 235 | 5.43 / 5.38 | 3.14 / 3.11 | undrained / undrained |
| 11 | 2.64 / 2.60 | 1.72 / 1.71 | 2.25 / 2.23 | 1.53 / 1.50 | 1.99 / 1.97 | 152 / 149 | 3.79 / 3.75 | 2.19 / 2.16 | undrained / undrained |
| 12 | 1.08 / 1.01 | 1.72 / 1.71 | 1.74 / 1.71 | 0.62 / 0.59 | 1.34 / 1.31 | 77 / 73 | 1.68 / 1.61 | 0.97 / 0.93 | drained / drained |

**tau_ff is within 0.03 ksf of the print on every slice, not within its 0.005 rounding as the brief asked.** The largest differences are on slices 2 and 12, the two thin end slices: the printed weights are those of slices with straight bases, and the module weighs down to the arc at the slice's middle, 5 to 7 % more there (sigma'_c 1.51 against 1.44 and 1.08 against 1.01). Nothing was tuned to close it. The equations themselves are pinned to the table's rounding by separate tests that start from the printed sigma'_c, tau_c, K_c and N (`test_kc_is_eq_g11_...`, `test_strengths_are_eqs_g9_g10_g12_...`, `test_drained_check_is_eqs_g13_to_g15_...`).

## 4. Section 11, before and after

Script: `module_work/gm140/step9_section11.py` (new; no script for section 11 was in `module_work/gm140/`). It ran on a checkout of master (before) and on this branch (after).

Rapid-drawdown rows:

| Ref | Benchmark | Module before | Module after | Printed | What moved it |
|---|---|---:|---:|---|---|
| B7 | EM example, three-stage, Spencer, pool 103 | 1.241 (stage 1 2.200) | 1.434 (2.200 / 1.513 / 1.434) | 2.20 / 1.52 / 1.44 | items 1 and 2 (the envelope alone gives 1.444 at stage 2, Eq. G-11 alone 1.375; lead's check), then items 4 and 5 |
| B7 | EM example, Lowe-Karafiath | 1.375 | 1.513 | stage 2: 1.52 | items 1 and 2 |
| B7 | EM example, Corps 1970, Spencer | 1.239 | 1.239 | 3.49 / 1.35 by Modified Swedish at 19.4 deg | does not move (c_R, phi_R as entered) |
| B7 | EM example, Corps 1970, force equilibrium at 19.4 deg | not offered | 3.486 / 1.341 | 3.49 / 1.35 | item 11 (new check method) |
| B7 | #95 (V-037), Corps, Spencer, pool 110 | 1.207 | 1.207 | Slide2 1.347 | does not move; -10.4 %, the method of slices |
| B7 | #95, Corps, force equilibrium at 19.4 deg | not offered | 1.338 (pool 103); 1.304 (pool 110) | 1.347 (EM 1.35) | item 11; -0.7 % and -3.2 % |
| B7 | #96 (V-038), three-stage, pool 110 | 1.235 | 1.427 (2.278 / 1.496 / 1.427) | Slide2 1.443 | items 1, 2, 4, 5; -1.1 % |
| B7 | #96, three-stage, pool 103 | 1.237 | 1.431 (2.194 / 1.510 / 1.431) | 1.443; EM 1.44 | the same; -0.6 % on 1.44 |
| V-048 | #95/#96 dam, Lowe-Karafiath, pool 110 | 1.357 | 1.496 | none for this pool (EM stage 2 1.52 at pool 103: module 1.510) | items 1 and 2 |
| B15 | V-041 #95/#96 section, Corps search | 0.783 (planar; circles 0.895) | 0.783 (planar; circles 0.895) | infinite slope 0.772 | does not move |
| B7 | #98 (V-041), Corps search | 0.836 | 0.836 | 0.931 | does not move; -10.2 % |
| B7 | #98, three-stage search | 0.935 | 0.951 | 1.039 | items 1, 2, 4, 5; -8.5 % |
| V-048 | #98, Lowe-Karafiath search | 0.974 | 0.988 | 1.075 | items 1 and 2; -8.1 % |
| V-052 | Pilarcitos 2.5:1, deep windows, Corps / LK / three-stage | 0.786 / 1.347 / 1.339 | 0.786 / 1.008 / 1.004 | 0.823 / 1.047 / 1.043 | item 2 (K_c); -4.5 %, -3.7 %, -3.7 % |
| B13, V-052b | Pilarcitos true section, deep windows, Corps / LK / three-stage | 0.722 / 1.279 / 1.263 | 0.722 / 0.944 / 0.943 | 0.823 / 1.047 / 1.043 | item 2; -12 %, -9.8 %, -9.6 % |
| V-052b | Pilarcitos true section, automatic windows, Corps / LK / three-stage | 0.718 / 1.075 / 1.075 | 0.718 / 0.929 / 0.929 | 0.823 / 1.047 / 1.043 | item 2: a circle 66 ft long and about 3 m deep now governs; before, a sliver an inch deep at the infinite-slope value |

Readings:

- **GM-15 is cause 2.** The 22 % overshoot of Lowe-Karafiath and the three-stage procedure on Pilarcitos' true section (1.279 and 1.263 against 1.047 and 1.043) was the module's own K_c rule. With Eq. G-11 they read 0.944 and 0.943, about 10 % low, as the Corps procedure already was on the same section (12 % low). So the remaining difference is common to the three procedures on this recovered section, not one procedure's strength rule.
- **#98 stays 8 to 10 % low in every procedure**, as the brief expected. Reported, not tuned.
- **#96 cannot tell a 103 ft pool from a 110 ft pool** by its final value (1.431 and 1.427).

The rows of section 11 that are not rapid drawdown (B1 to B6, B8 to B12, B13's Duncan example, B14 to B18 other than the rows above, V-035/036, V-039, V-044, V-045) do not pass through `rapid_drawdown.py`; the only edits outside it are two appended layer fields and the wrapper's signature. Their pinned tests ran unchanged and pass (section 6), so before and after are the same values the account lists.

## 5. Where the rule now holds

Callers found by grep of the whole repository:

- `slope_stability.rapid_drawdown.rapid_drawdown_fos` and `slope_stability.analysis.rapid_drawdown_fos` (the wrapper passes every argument through).
- `slope_stability.rapid_drawdown.search_rapid_drawdown`: every trial of every family (grid circles, entry and exit circles, random and differential-evolution polylines, toe circles, circles above the toe, planes) is scored by `rapid_drawdown_fos`; the winner's detail is the same function.
- The agent's adapter (`funhouse_agent/adapters/slope_stability.py`): `rapid_drawdown_fos` and `search_rapid_drawdown`.
- `funhouse_agent/worked_examples.json` WE-SLOPE-5 and the eval suite's RDD-1 to RDD-4.
- `docs/gallery/build_gallery.py` exhibit 3 calls the function (see "Noted, not fixed").

No other caller exists: `analyze_slope`, `compare_methods_table`, `search_critical_surface`, the yield acceleration and Newmark, FOSM and Monte Carlo, `reliability/wrappers.py`, `geo_project`'s builder and the calc packages do not call rapid drawdown, and no calc package has a drawdown template.

Every method of slices: the stages are solved by the GLE engine (Spencer, or Morgenstern-Price with another `f_interslice`); Bishop and force equilibrium at a fixed inclination through the check arguments. The Ordinary Method of Slices and Janbu's simplified method are not offered for the stages (they were not before).

## 6. Tests

All by pytest's exit code, venv `/tmp/v` (Python 3.12).

| Run | Result |
|---|---|
| New: `slope_stability/tests/test_rapid_drawdown_em_appendix_g.py` on the code before this step | 22 failed of 22 (failing first) |
| The same on this branch | 23 passed (one test added after the first run) |
| `slope_stability/tests` (all), every `validation_examples/test_published_*` that imports `slope_stability` (22 files), the adapter, eval-suite, worked-example, defaults-used, top-level-parameter and refusal-hint tests, `test_page_fingerprints.py`, `test_public_text.py`, `test_backlog.py` | 1,881 passed, 42 skipped, 2 xfailed (both strict xfails that were on master), 1 failed |
| After the last edit (the first stage's m_alpha): the two drawdown test files, `test_m_alpha`, `test_water`, `test_search_reinforcement`, `test_circle_slicing`, `test_ground_crossings`, `test_search_ranges`, the four drawdown validation files, the adapter, eval-suite, wave-1, worked-example and defaults-used tests, fingerprints, public text | 1,162 passed, 25 skipped, 1 failed |
| `module_work/doc10/check_manual.py` (the manual's name check; no rebuild) | OK |
| Every `funhouse_agent`, `calc_package`, `reliability`, `fem2d`, `geotech_common`, `report_ingest`, `dxf_import`, `geo_project`, `webapp` test file that names `slope_stability` or `rapid_drawdown` (43 files), with `test_page_fingerprints.py` and `test_public_text.py` | 2,947 passed, 26 skipped, 2 failed |

The failures are not this branch's: `calc_package/tests/test_calc_package.py::TestDm7FigureHelper::test_figure_436_with_query` and `funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]` (the one failure of the two later runs) fail the same way on a checkout of master in this environment.

`webapp/tests/test_page_fingerprints.py` passes without a re-record: the page fingerprints do not take in these two tools' descriptions, so there is no fingerprint commit on this branch.

Pins changed, each against print (the assertion's comment carries the old value, the printed value and the difference): V-037, V-038, V-041 (#98 three-stage), V-048 (#98 Lowe-Karafiath; the EM's printed second stage), V-052 and V-052b. Pins removed because nothing printed stands behind them: 1.34 (#95), 1.273, 1.370 and 1.450 (#96) on the unprinted seepage line; the `gle` rows (1.306, 0.959); 1.57 and 2.048 on the Pilarcitos toe circle and 1.006 and 1.043 on a made-up reinforced dam (module's own values; each replaced by the property the test is about).

## 7. Calls made

1. **(Lead's call, from the brief.)** The Corps 1970 procedure keeps Spencer's method and its answer (1.239 on the manual's example; printed 1.35 by the Modified Swedish Method at 19.4 deg). The result names the method and carries the statement in `notes`, not in `warnings`. The printed 3.49 and 1.35 are pinned through `fixed_inclination_solution`, which is not on the agent's tools.
2. **s_d = c' where sigma'_d is negative.** The manual prints s_d = 0.00 for slice 1 (c' = 0). For c' above zero this is a reading; the result's `choices` says so.
3. **A base in tension at the first stage (sigma'_c below zero).** No printed rule. The module takes sigma'_c = 0 there, uses the lower of the two envelope strengths (K_c is not defined), and names the slices in a warning.
4. **Eq. G-11 where its denominator is not above zero.** The denominator is the minor principal consolidation stress times a positive factor, so the ratio is beyond K_f; K_f is returned, which is "held between 1 and K_f" applied.
5. **The removed input is refused, not ignored.** `stage3_effective_normal` with any value raises with the rule, in the library and the adapter. A saved call that names 'fellenius' now gets the rule instead of a number.
6. **The Corps procedure refuses a layer that gives only R_d and R_psi**, since it uses c_R and phi_R as entered; Eqs. G-5 and G-6 were not inverted.
7. **Library check arguments** `stage_solver` and `extra_boundaries` on `rapid_drawdown_fos` (to solve the stages by Bishop or at a fixed inclination, and to cut the manual's own twelve slices). They are not on the adapter, so they got no manual row or probe.
8. **The 1970 example's 3.49 and 1.35 are pinned on the module's usual 50 slices** (3.486, 1.341). On the manual's twelve slices the first stage reads 3.455, 1.0 % low, just outside a 1 % gate; that run is reported in the tables and not pinned.
9. **tau_ff pinned within 0.03 ksf on the full run**, not within the table's rounding (section 3).
10. **The eval suite.** RDD-2's key moves 1.385 to 1.528 (module ground truth, as before). RDD-3 asked for the removed `'gle'` option; it now asks for Lowe and Karafiath's procedure on the same dam (key 1.588). The owner may prefer another question.
11. **`FOS` is None where a stage has no solution** (`RapidDrawdownResult.FOS` is now Optional), following step 2's convention as its brief states it; `stage1_converged` and `stage3_converged` stay as read-only properties for earlier callers.
12. **GM-130's drawdown item, in part.** The first stage's least m_alpha is now computed, reported (`m_alpha_min_stage1`) and, below 0.2 on a given surface, said in a warning. The search's refusal still reads the last stage only: step 3 replaces that refusal (OWN-102), and changing which trials a drawdown search counts here would move #98 and Pilarcitos for a rule about to be rewritten.
13. **A new input's probe.** `drawdown_phreatic_points` is declared with `when_absent` (the level lowered pool); the probe guard (`test_defaults_used.py`) passes with it, and no entry in `method_probes.py` was needed.

## 8. Not settled (stated, not guessed)

- Which method of slices Slide2 used for #95 and #96.
- A base in tension at the first stage: no printed rule (call 3).
- Soils with c' above zero exercise Eq. G-8 with Eq. G-11 only on #98's recovered section.
- The after-drawdown line (`drawdown_phreatic_points`) has no printed benchmark; its tests are properties (a level line equals the default; a higher line lowers the drained strengths).

## 9. Release note

Rapid drawdown now follows EM 1110-2-1902 Appendix G stage by stage, and its answers move. For the Lowe and Karafiath and the three-stage procedures the strength of soil consolidated at K_c = 1 is derived from the R envelope by Eqs. G-5 and G-6 (higher than before), K_c is Eq. G-11 (lower than before on steep, high-phi' fills), the second stage is solved and reported, the drained check uses the second stage's normal force (Eqs. G-13 to G-15), and a third stage is solved only where a slice needs it. On the manual's own example the three-stage answer goes from 1.24 to 1.43 (printed 1.44) and the second stage reads 1.51 (printed 1.52); Slide2 #96 goes from 1.235 to 1.427 (printed 1.443); Pilarcitos' true section by Lowe and Karafiath goes from 1.28 to 0.94 (printed 1.05). The Corps 1970 procedure's values do not change; its result now says that the manual computed its example by another method of slices. A result gives each stage's factor of safety, the stage the answer is from, the water it assumed and a per-slice table; where a stage has no solution there is no number. `stage3_effective_normal` is removed (a call that names it is told the rule); `drawdown_phreatic_points` and a layer's measured `R_d`, `R_psi` are new.

## 10. Noted, not fixed

- `docs/gallery/build_gallery.py` exhibit 3 and its figure in `docs/gallery/` show the values of their day (a seepage line; 1.21 to 1.37); the script still runs, the gallery was not rebuilt.
- WE-SLOPE-5's `n_surfaces_per_search` says 84; the searches now score 276 to 280 surfaces (stale before this step).
- The drawdown search keeps two guards no page prints (fewer than half the slices kept; a first-stage factor of safety below 0.5). D38 does not name them.
- `search_rapid_drawdown` defaults to `corps_2stage` and `rapid_drawdown_fos` to `duncan_3stage` (as before).
- D16's other half (the pond's edge as a slice boundary in ordinary analyses) is step 5a's; in rapid drawdown both pools' crossings are slice edges in every stage.
- The module's slice weight to the arc at the slice's middle against the manual's straight-base areas (0.9 % on the whole mass, 5 to 7 % on thin end slices): slicing, not this step.
- Two tests fail on master in this environment (section 6).

## 11. Stopped short of

Nothing in the section's "What changes" list was left out. Not done because the print is not in hand or the brief puts it out of scope: seepage analysis; the pore-pressure function; a Modified Swedish method for the agent's tools; the manual's Modified Swedish runs at Spencer's inclinations per stage (6.0, 12.2, 13.7 deg: 2.23 / 1.52 / 1.44), which the brief does not ask to pin.
