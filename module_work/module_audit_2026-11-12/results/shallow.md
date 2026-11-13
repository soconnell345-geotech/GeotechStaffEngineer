# shallow

All paths are under C:\dev\GeotechStaffEngineer. "Known?" = grepped BACKLOG.md and BACKLOG_DONE.md for the topic; "new" means no match found.

## bearing_capacity
Size: 8 python files, 5 public classes exported (no analyze_* function, by design), 2 adapter methods. The most carefully kept of the group: every documented parameter reaches the module, the returns agree with to_dict(), refusals and choices are stated, and published examples are pinned (GEC-6 B-1, UFC B-1 and B-3). The gaps are small.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 1 | B | `concrete_design.py` (footing structural checks, ACI 318 development length) is in no `__init__` export, no adapter method and no calc package; only its own test imports it, so the agent cannot reach it. DESIGN.md lists it as a module file with no word that it is unreachable. | bearing_capacity/tests/test_concrete_design.py:6; bearing_capacity/__init__.py:25-31; DESIGN.md:27 | new |
| 2 | 1 | C | DESIGN.md says the module matches "CBEAR program output"; no test or validation file cites CBEAR. The pinned examples are GEC-6 B-1 and UFC B-1/B-3, not CBEAR runs. | bearing_capacity/DESIGN.md:5-6; grep CBEAR in bearing_capacity/tests and validation_examples/test_published_v021_v022.py: no hit | new |
| 3 | 1 | C | The adapter's `returns` for `bearing_capacity_analysis` names only q_ult, q_allowable, factors, choices, notes; `to_dict()` also returns the shape/depth/inclination/base/ground factors (sc..ggamma), the three terms, B_eff/L_eff and q_overburden. Not told, but present. | funhouse_agent/adapters/bearing_capacity.py:191-199 vs bearing_capacity/results.py:317-349 | new |
| 4 | 3 | C | A strip given with `layer1_thickness` but no layer 2 simply ignores the thickness (it is read only in the two-layer load-spread). No warning in the result. Unverified beyond the one read site. | bearing_capacity/capacity.py:739 | new |
| 5 | 3 | C | A two-layer result is bounded by the load-spread method, stated in DESIGN.md; the adapter's description of `layer2_unit_weight` ("triggers 2-layer analysis") does not say layer 2 is taken as infinitely deep or that the result is a bracket. | adapter bearing_capacity.py:175; DESIGN.md:46-47 | new |
| 6 | 5 | C | Output set is complete: calc package (`bearing_capacity_package`), calculation tables (`get_tables`), summary() and to_dict() all present. | bearing_capacity/results.py:212,308,368; calc_package/__init__.py:117 | n/a |
| 7 | 6 | C | Pinned to printed examples: V-021 (GEC-6 Example B-1 Step 7), UFC 3-220-20 Example B-3, Table 5-2 rows; 91 page/table citations across 8 test files. The module's test files cover every public class; no uncovered public function. | validation_examples/test_published_v021_v022.py:1-33; bearing_capacity/tests/test_factor_combination.py | n/a |

Not checked: engineering values; DM-7.01 / Bowles two-layer load-spread against print; `docs/user_manual` text beyond a grep (entry at manual_content.py:186,222 matches the method name).

## settlement
Size: 10 python files, 20 exported names, 5 adapter methods. Immediate (elastic, Schmertmann), consolidation, Hough, time rate and secondary are all coded and pinned to printed examples (NHI-06-089 Ex. 8-2/8-3, UFS figures, GEC-6 B-1 Hough). The gaps are in what the adapter lets the agent reach, and a few silent edges.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 3 | A | `consolidation_settlement` with a `delta_sigmas` list shorter or longer than `layers` silently drops the extra layers (or extra stresses) and returns a smaller total; no length check anywhere. | funhouse_agent/adapters/settlement.py:163 (`zip(layers, delta_sigmas)`); no `len(` check in adapter or settlement/consolidation.py | new |
| 2 | 3 | A | Schmertmann sums only the part of each layer inside the influence depth z_max; layers that stop short of z_max contribute nothing for the missing depth, and nothing is said. A profile covering only 1B of a 2B influence zone gives a settlement that is too small with no warning. | settlement/immediate.py:424-431 | new |
| 3 | 3 | B | A Cr greater than Cc raises a Python `warnings.warn`, which never reaches the agent: the adapter returns no warning field for `consolidation_settlement`, `schmertmann_settlement` or `hough_settlement` (only the combined analysis carries `warnings`). | settlement/consolidation.py:74; adapter settlement.py:133,184,220 | new |
| 4 | 1 | B | Hough (granular, SPT) settlement cannot be combined with consolidation or secondary: `combined_settlement_analysis` offers immediate_method only elastic or schmertmann. A sand-over-clay profile (the GEC-6 B-1 case) needs two separate tool calls added by hand. | adapter settlement.py:375; settlement/__init__.py:9 | new |
| 5 | 5 | B | The calc package covers `combined_settlement_analysis` only. Hough, Schmertmann alone, elastic alone and standalone consolidation have no package route and no calculation tables, although Hough's `HoughResult` has summary() and to_dict(). | funhouse_agent/adapters/calc_package.py:1798-1810; grep hough in settlement/calc_steps.py: no hit | new |
| 6 | 1 | B | Exported time-rate and secondary functions (`time_factor`, `degree_of_consolidation`, `time_for_consolidation`, `settlement_at_time`, `secondary_settlement`, `total_consolidation_settlement`, `stress_at_depth`) are reachable only inside the combined analysis; a user who wants only "t90 for cv = ..., Hdr = ..." or a stress increase at depth has no tool. Also `schmertmann_terms` (C1, C2, C3, z_peak, Iz) exists but the adapter returns only Se. | settlement/__init__.py:42-51; adapter settlement.py:133 | new |
| 7 | 3 | C | Schmertmann result carries no `choices` or `defaults_used`: q_overburden defaults to 0.0 (which sets C1 = 1) and time_years to 0 (no creep, C2 = 1) without being stated. | adapter settlement.py:127-133 | new |
| 8 | 3 | C | Hough: 2:1 stress spread only, not stated in `returns`; no `choices` entry. (METHOD_INFO brief does say 2:1.) | settlement/hough.py:175-183 | new |
| 9 | 3 | C | `consolidation_settlement`: omitted `sigma_p` makes the layer normally consolidated (sigma_p = sigma_v0) with no note in the standalone result; the combined analysis warns about underconsolidated layers. | settlement/consolidation.py:76-77; analysis.py:~410 | new |
| 10 | 2 | C | Secondary compression with no `cv` assumes t1 = 1 year; the result now carries a warning saying so (verified in `_warnings`). | settlement/analysis.py:108-117, 402-409 | fixed (GM-115), code agrees |
| 11 | 4 | C | Strip-footing elastic Iw refused unless Iw given; layered Es for the elastic method is not offered (one Es only). Stated in the adapter. | adapter settlement.py:314-316, 379 | n/a |
| 12 | 6 | C | Pinned: Schmertmann Ex. 8-2 and Table 8-11, consolidation Ex. 8-3, secondary p. 255, t90 p. 7-38, Hough B-1 Ex. 2-4, 2:1 and Westergaard. Every exported function appears in a test (checked by name). | settlement/tests/test_printed_examples.py:30-205; validation_examples/test_published_v021_v022.py:17 | n/a |

Not checked: Westergaard chart read-off against the print; Boussinesq at L/B beyond the table; manual_content.py text for settlement beyond the method name.

## ground_improvement
Size: 9 python files, 11 exported names (6 functions, 5 result classes), 7 adapter methods (adds `priebe_improvement_factor`, which has no export). Rebuilt against GEC-13 in GM-291 and carefully cited. Two real mismatches: the feasibility adapter describes outputs the result does not have, and DESIGN.md's "Public API" is stale.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 1 | A | The adapter tells the agent `feasibility` returns `recommended_methods` and `is_feasible`. `FeasibilityResult.to_dict()` returns `applicable_methods`, `not_applicable`, `recommendations`, `preliminary_sizing`, `soil_description`, `design_problem`; neither named key exists anywhere in the code. An agent that reads the stated keys finds nothing. | funhouse_agent/adapters/ground_improvement.py:354 vs ground_improvement/results.py:545-552 | new |
| 2 | 1 | B | DESIGN.md "Public API" signatures are wrong for four functions: `analyze_aggregate_piers(B, L, q_applied, ...)`, `analyze_wick_drains(cv, ch, drain_spacing, ...)`, `analyze_surcharge_preloading(target_consolidation, ...)`. Real signatures take column_diameter/spacing, spacing/ch/cv/Hdr/time, S_ultimate/surcharge_kPa/cv/Hdr. | ground_improvement/DESIGN.md:26-30 vs aggregate_piers.py:252, wick_drains.py:367, surcharge.py:270 | new |
| 3 | 1 | B | `surcharge_preloading` returns, per the adapter, only `time_to_target_years`; the result also carries settlement at target, ultimate settlement, equivalent sigma_p, wick-drain sub-result and time curve. | adapter ground_improvement.py:323 vs results.py:367-382 | new |
| 4 | 3 | B | `equivalent_sigma_p_kPa` is 0.0 whenever `sigma_v0` is left at its default 0, with no statement; the adapter lists sigma_v0 default 0.0 as "current effective stress". A 0 reads as a computed value. | ground_improvement/surcharge.py:339-342; adapter ground_improvement.py:320 | new |
| 5 | 3 | C | `vibro_compaction` returns `recommended_spacing_m` = 0 when no target relative density is given or the soil is infeasible (a sentinel, documented in the adapter text). `aggregate_piers` q_unreinforced and S_unreinforced 0 = "not computed" is likewise a sentinel in the numbers, though the result then reports `improved_bearing_kPa` and `settlement_improved_mm` as 0 values. | adapter ground_improvement.py:258-259, 334; results.py:111-122, 467-475 | new |
| 6 | 3 | B | In the feasibility screen a clay with no `cu_kPa` is taken as soft enough for aggregate piers ("assume soft enough"); the result lists the method as applicable without saying the strength was assumed. | ground_improvement/feasibility.py:232-235 | new |
| 7 | 4 | B | DESIGN.md says "four ground improvement techniques", and the feasibility screen recommends only piers, wick drains, surcharge and vibro. GEC-13 methods outside it (dynamic compaction, jet grouting, deep soil mixing, column-supported embankments) have no design or screening tool here; the reference library has GEC-13 lookups for some. | ground_improvement/DESIGN.md:3-5; feasibility.py:200,317,383,436 | OWN-21 open (parked scope, names deep mixing and column-supported embankments) |
| 8 | 5 | B | Calculation tables: only wick drain and surcharge results have `get_tables()`; aggregate piers, vibro and feasibility have none, though all three are in the calc package route. | ground_improvement/results.py:255,386 (no other get_tables) | new |
| 9 | 5 | C | `summary()` and `to_dict()` present on all five result classes. | ground_improvement/results.py:73,111,183,233,318,367,442,467,507,545 | n/a |
| 10 | 3 | C | Priebe settlement ratio refused outside A/A_c 1.6-8.0 (stated, not extended). Balaam curves likewise refused off their drawn range. Good edge handling, noted. | adapter ground_improvement.py:267; DESIGN.md:106-134 | n/a |
| 11 | 6 | C | Pinned to GEC-13: wick-drain example (p. 2-38: 299 and 503 days), vibro screens, Fig. 5-21 curves, Example 2 (p. 5-48). Every exported analyze_* function appears in a test. | ground_improvement/tests/test_gec13_print.py:3-60 | n/a |

Not checked: the traced Fig. 5-21 / Fig. 4-34 curves against the print (later stage); adapter-to-calc_package parameter parity.

## pavement_design
Size: 13 python files, 13 exported names, 9 adapter methods. Covers AASHTO 1993 flexible and rigid, ESALs, composite k, performance period, plus a UFC 3-250-01 path and a method comparison. Unit exception (US customary) is documented in `__init__.py`. Scope limits are stated and match the code. Weakest on outputs (no tables) and on a library-version fallback that raises.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 5 | B | No calculation tables for pavement: `pavement_design/calc_steps.py` has no `get_tables` or `CalcTable`, and no test of tables exists, while 13 other modules have them (GM-39). | grep get_tables / CalcTable in pavement_design/calc_steps.py: no hit; pavement_design/tests: only test_pavement_design.py | new |
| 2 | 5 | C | The UFC functions return plain dicts, not result classes: `design_flexible_pavement_ufc`, `design_rigid_pavement_ufc`, `compare_flexible_pavement_methods`, `ufc_mixed_traffic`, `figure_e1_thickness`, `figure_f1_thickness` have no `summary()` or `to_dict()`. | pavement_design/ufc.py:138,300,375,509; results.py has only 3 classes | new |
| 3 | 1 | B | The adapter says `design_traffic_esals` takes triple axles. If the installed geotech_references lacks `aashto_1993.lef`, the fallback raises `NotImplementedError` for triple axles. Pinned refs 1.6.0 has the module, so the path is dormant today; the same import-guard pattern guards composite k and swelling/frost. | pavement_design/traffic.py:80-109; composite_k.py:193; performance.py:33; rigid.py:99 | new |
| 4 | 3 | B | `design_traffic_esals` computes LEFs at an assumed SN = 5.0 (or D = 9 in) and pt = 2.5 when not given; the adapter says "LEFs at the assumed sn/d_in/pt" but the defaults enter the result only via parameters, not in a `choices` or notes line. Truck factors depend on SN/D, so the W18 shifts with the assumption. | adapter pavement_design_adapter.py:481-492; traffic.py:80-90 | new |
| 5 | 3 | B | A refused asphalt modulus below the chart range, and other chart clamps (UFC and AASHTO), are known to be silent or refused in ways the app does not yet handle. | pavement_design/flexible.py:129 | known REF-74, REF-76 open |
| 6 | 4 | B | Scope not covered, as DESIGN.md and `__init__` state: overlays/rehabilitation (AASHTO Part III), rigid joint and reinforcement design, low-volume catalog, UFC 3-260-02 airfield and rigid curves F-2 to F-31. These are stated, not hidden. | pavement_design/__init__.py:30-34 | known OWN-21 (parked) |
| 7 | 1 | C | `plots.py` has 8 plot functions (design charts, layer section, seasonal MR, comparison, environmental loss); none is exported or in an adapter method. They are reached only through the calc package; not checked whether every one is called from it. | pavement_design/plots.py:31-319; __init__.py:46-58 | new |
| 8 | 3 | C | An asphalt layer's drainage coefficient m given on a layer type that Table 2.4 does not apply to is ignored, with a note in the result (verified at the warning site). | pavement_design/flexible.py:181 | n/a |
| 9 | 6 | C | Pinned: AASHTO 1993 worked examples (v055: 3 examples; v074-076; composite k and V-0xx examples). UFC functions, `figure_e1_thickness` and `figure_f1_thickness` appear in the module's own test file only; no published UFC example number found in validation_examples for the UFC path. Unverified: whether test_pavement_design.py cites a UFC Appendix example. | validation_examples/test_published_v055_aashto1993_pavement.py; pavement_design/tests/test_pavement_design.py | new (UFC pinning unverified) |

Not checked: AASHTO charts and tables against the print (later stage); composite-k and Appendix G environmental routines beyond the import guards; calc_steps parity with adapter inputs.

## wind_loads
Size: 0 python files, 0 exports, 0 adapter methods. The folder `wind_loads/` contains only an empty `tests/` subdirectory. The module (ASCE 7-22 wind on freestanding walls and fences, 62 tests) was added in commit 22d4d72 and removed with its adapter in the Phase 1 consolidation (commit 99fde72, 2026-06-09: "remove pyseismosoil_agent, fdm2d, geolysis_agent, wind_loads (+full de-wire)").

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 1 | C | `wind_loads` is not a module any more: no code, no adapter, no registry entry, no CLAUDE.md or manual claim (grepped CLAUDE.md, funhouse_agent, docs/user_manual/manual_content.py, pyproject, README, BACKLOG: no "wind_loads"). Nothing tells the agent it exists, so nothing fails. | find wind_loads: only `wind_loads` and `wind_loads/tests` (no files); git ls-files wind_loads: empty | new |
| 2 | 1 | C | The leftover empty directory is untracked residue and can be deleted; it only misleads a folder listing (the audit brief listed it as a module). | as above | new |
| 3 | 4 | C | If wind on walls or fences is wanted, it would be a new build, not a repair; the removed code is in git history at 22d4d72. Not an incompleteness of a shipped module. | git log -- wind_loads | n/a |

Not checked: whether any worktree under `.claude/worktrees/` still carries wind_loads docs (several do: CONSOLIDATION_PLAN.md, V5.0_TODO.md); those are stale copies, outside the audit.
