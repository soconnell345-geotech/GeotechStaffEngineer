# Calculation modules: what the agent is told against what the code does (stage 1 audit, 2026-11-12)

Read-only audit of GeotechStaffEngineer master (b5579d2). The yardstick is the one the owner set for the reference library (REF-92): the agent can do everything it is told it can, and a result says what it is. All paths are under `C:\dev\GeotechStaffEngineer`. `~` before a line number means the line was found by reading around it and may be off by a few.

Not covered: `geotech-references/` (another session owns it), `packages/aec-document-review/`, `packages/aec-agent-core/aec_agent_core/review_eval/`, planlens.

## 1. How it was done

**Scouts.** Ten Sonnet scouts each read one group of modules: shallow, deep_axial, lateral_drive, walls, slope_numerical, reliability_stats, seismic, site_data, structural_calcpkg and registry. Their checks were (1) what the agent is told against what the code does, (2) gaps the code admits itself (TODO, "not implemented"), (3) silent edges, (4) cases missing from a method the module says it covers, (5) outputs (calc package, tables, figure, `to_dict`), (6) published examples pinned in tests, (7) already in the backlog. An eleventh scout answered two questions on calculation examples (§4, last subsection). The scout files are in `results/` beside this file. They hold every row, including the C rows this audit only counts.

**Re-checks (coordinator, Opus).** I opened the cited lines of every severity-A finding (all 50) and of 42 of the 148 B findings. The brief asked for one B in three (49). Two groups got fewer: structural_calcpkg (2 of 27) and slope_numerical (2 of 8). I grepped `BACKLOG.md` and `BACKLOG_DONE.md` for the topic of every confirmed A finding and of the re-checked B findings. The lead asked three questions, and each one was traced through the code (§2.1). Nothing was run. There was no pytest and the package was never imported, so every claim about behaviour comes from reading the code.

**Scout accuracy.**

| Scout file | A given | A kept | A to B | A to C | A wrong (dropped) | B re-checked | B wrong | B regraded | B backlog status corrected |
|---|---|---|---|---|---|---|---|---|---|
| shallow | 3 | 3 | 0 | 0 | 0 | 6 of 16 | 0 | 1 (to C) | 0 |
| deep_axial | 7 | 6 | 1 | 0 | 0 | 4 of 8 | 0 | 1 (to C) | 0; 1 manual cite wrong |
| lateral_drive | 5 | 2 | 1 | 2 | 0 | 5 of 16 | 0 | 0 | 0 |
| walls | 2 | 2 | 0 | 0 | 0 | 4 of 12 | 0 | 3 (to C) | 0 |
| slope_numerical | 1 | 1 | 0 | 0 | 0 | 2 of 8 | 0 | 0 | 1 (it is in GM-140) |
| reliability_stats | 12 | 7 | 3 | 0 | 2 | 4 of 12 | 0 | 0 | 0 |
| seismic | 4 | 4 | 0 | 0 | 0 | 9 of 27 | 0 | 0 | 0 |
| site_data | 9 | 7 | 0 | 2 | 0 | 5 of 19 | 0 | 0 | 0 |
| structural_calcpkg | 7 | 0 (1 duplicate) | 4 | 2 | 0 | 2 of 27 | 0 | 0 | 0 |
| registry | 0 | 0 | 0 | 0 | 0 | 1 of 3 | 0 | 0 | 0 |
| calc_examples | n/a | | | | | | | | 1 claim wrong: fem2d is not reachable through `render_tables`/`render_figures`, which need a named `*_package` (adapters/calc_package.py:219-237) |
| **all** | **50** | **32** | **9** | **6** | **2** (+1 duplicate) | **42 of 148** | **0** | **5** | **1** |

What the table shows: 47 of the 49 distinct A findings were real (96 %). Only 32 held at A (65 %). The scouts over-graded two kinds. One was a wrong API in a DESIGN.md: no agent or helper reads DESIGN.md (no reference to it anywhere in funhouse_agent or aec_agent_core), so those are now C quick fixes. The other was a described parameter that is unread but does not change a number, now B. Both A findings dropped as wrong (pystra #6, salib #4) said a misspelt top-level parameter is silently ignored. Dispatch refuses it for every method with documented parameters (`_check_parameter_names`, packages/aec-agent-core/aec_agent_core/method_call.py:217). The structural scout was the least reliable: none of its 7 A findings held, and only 2 of its 27 B findings were re-checked. Re-check that group's B rows before acting on them.

## 2. What was found

**Totals after re-check: 32 A, 152 B, 186 C.** Three findings were dropped (2 wrong, 1 duplicate). The scouts matched 75 rows (any grade) to a backlog entry, open or done. None of the 32 confirmed A findings is covered by an open item: pile_group #2 touches GM-28, but only GM-28's benchmark.

Grades: **A** means the agent is told something false, or a result is wrong or incomplete with nothing said, in a way a user would act on. **B** means a real gap a user can see or work around, a missing output, or wording that misleads. **C** means a document, a hygiene issue or a small edge.

Ranked for a deeper look by confirmed A, then B, then how much users lean on the module:

| Module | A | B | C | In backlog | Worst confirmed finding |
|---|---|---|---|---|---|
| pystra_agent | 5 | 3 | 1 | 0 | When SORM fails, the FORM values come back under the SORM names, and `converged` only means beta > 0 (pystra_agent/reliability.py ~246-249, ~139-140) |
| subsurface_characterization | 4 | 6 | 3 | 1 | `compute_trend` fits any trend_type as a straight line and labels it as the type asked; with fewer than 2 points it returns zeros (subsurface_characterization/statistics.py:150-195) |
| pile_group | 3 | 1 | 4 | 2 | `pile_group_simple` accepts per-pile batter and stiffness and ignores them, using piles[0]'s stiffness x n (pile_group/rigid_cap.py:338-352, 383-384) |
| dxf_import | 3 | 4 | 4 | 0 | Missing soil strengths become gamma 18, phi 0, c 0 with nothing said (funhouse_agent/adapters/dxf_import_adapter.py:172-182, 236-247) |
| settlement | 2 | 4 | 6 | 1 | A `delta_sigmas` list of a different length from `layers` drops layers from the total (funhouse_agent/adapters/settlement.py:163) |
| lateral_pile | 2 | 5 | 6 | 5 | Soil that starts below the ground or stops above the toe only raises a Python warning, which never reaches the agent (lateral_pile/soil.py:152-161) |
| axial_pile | 2 | 1 | 4 | 0 | `capacity_vs_depth` drops a failed depth with `except Exception: pass` (axial_pile/capacity.py:2015-2018) |
| liquepy_agent | 2 | 6 | 4 | 1 | The CPT tool is told to return `n_liquefiable` and lower-case keys; neither exists (funhouse_agent/adapters/liquepy_adapter.py:126-131; liquepy_agent/results.py:107-120) |
| reliability | 2 | 2 | 6 | 3 | FOSM and PEM ignore the `lower`/`upper` truncation the agent is told about (funhouse_agent/adapters/reliability_adapter.py:229-234; reliability/fosm.py:91-92) |
| soe | 2 | 3 | 9 | 4 | Braced supports' `spacing`, `angle_deg` and `preload` are accepted and never read (funhouse_agent/adapters/soe.py:99-105; soe/geometry.py:50-62) |
| drilled_shaft | 1 | 2 | 4 | 2 | A shaft longer than the profile runs on with only a Python warning; results have no warnings field (drilled_shaft/capacity.py:981-985) |
| ground_improvement | 1 | 5 | 5 | 1 | `feasibility` is told to return `recommended_methods` and `is_feasible`; it returns neither (funhouse_agent/adapters/ground_improvement.py:354; ground_improvement/results.py:545-553) |
| pystrata_agent | 1 | 5 | 3 | 2 | The Darendeli/Menq mean effective stress is computed from total stress, with K0 0.5 and no water table (pystrata_agent/eql_site_response.py:182-205) |
| seismic_signals_agent | 1 | 4 | 4 | 2 | When the calculation fails, bracketed duration comes back as 0.0 (seismic_signals_agent/intensity_measures.py:88-92) |
| fem2d | 1 | 3 | 7 | 4 | fem2d is registered for calc packages but has no `fem2d_package` method (calc_package/__init__.py:133; funhouse_agent/adapters/calc_package.py:3107-3128) |
| seismic_geotech | 0 | 8 | 6 | 0 | Two total-stress rules for the same `gamma` input: layered (seismic_geotech/liquefaction.py:255-263) against gamma x depth (liquepy_agent/spt_liquefaction.py:167-168) |
| section_props_agent | 0 | 9 | 3 | 2 | `warping=False` returns J = 0.0 as a number (section_props_agent/sections.py:146, 242-245). B rows mostly not re-checked |
| concrete_props_agent | 0 | 9 | 3 | 2 | No strain check (rc_section.py:341-349, not re-checked) |
| pynite_agent | 0 | 8 | 3 | 1 | `rho` is accepted but no self-weight is applied (funhouse_agent/adapters/pynite_adapter.py:21; pynite_agent/frame.py:61) |
| wave_equation | 0 | 7 | 7 | 1 | The pile is always one uniform section (wave_equation/pile_model.py:100-110) |
| retaining_walls | 0 | 6 | 6 | 1 | No global stability check for either wall (retaining_walls/DESIGN.md:417) |
| salib_agent | 0 | 5 | 3 | 1 | The expression compiler allows unbounded powers, unlike the reliability engine's (salib_agent/expression.py:26-31) |
| sheet_pile | 0 | 5 | 6 | 2 | Net active pressure is clamped at zero, with no minimum pressure and no water in the crack; to be checked against the print (sheet_pile/cantilever.py:883, 904; anchored.py:228, 240) |
| slope_stability | 0 | 5 | 7 | 5 | On hold under GM-140 |
| pavement_design | 0 | 5 | 4 | 2 | No calculation tables (no `get_tables` in pavement_design) |
| calc_package | 0 | 5 | 3 | 2 | A misspelt `format` renders HTML (calc_package/__init__.py:395) |
| downdrag | 0 | 4 | 4 | 3 | Calculation tables only on the combined result (funhouse_agent/adapters/calc_package.py:584-587) |
| geo_project | 0 | 4 | 4 | 5 | Known drift (OWN-22) |
| opensees_agent | 0 | 4 | 3 | 2 | PM4Sand DSS runs one CSR only (not re-checked) |
| dxf_export | 0 | 3 | 3 | 0 | Layer names not cleaned (an edge case, no item) |
| gstools_agent | 0 | 3 | 3 | 1 | No anisotropy in kriging or random fields |
| geotech_common | 0 | 2 | 4 | 3 | `engineering_checks.py` and the SPT correlations are unreachable by the agent |
| profile_figure | 0 | 2 | 5 | 4 | Layer keys unchecked (by design) |
| bearing_capacity | 0 | 1 | 6 | 0 | `concrete_design.py` is unreachable (bearing_capacity/__init__.py:25-31) |
| registry (5 checks) | 0 | 3 | 18 | 3 | The model-facing brief says calc_package has "15 modules" (funhouse_agent/adapters/__init__.py:357); the renderer has 17 |
| removed modules (wind_loads, fdm2d, DM7Eqs, hvsrpy, pyseismosoil, swprocess, pygef, ags4, pydiggs, geolysis) | 0 | 0 | 14 | 7 | Empty folders left on disk; stale extras in pyproject.toml |

### 2.1 The lead's three questions

1. **Python warnings never reach the agent: confirmed.** A tool call runs `funhouse_agent/dispatch.py:971 call_agent`, then `run_method` (packages/aec-agent-core/aec_agent_core/method_call.py:591), and the handler runs at `method_call.py:622` (`result = mod.METHOD_REGISTRY[method](parameters)`) with no `warnings.catch_warnings`. The only catches in funhouse_agent and aec_agent_core are in chart evaluation, in tests, and in `aec_agent_core/charts/layout_ink.py:1196`; that last one takes a lock because `catch_warnings` swaps process-wide state. There are about 50 `warnings.warn` calls in about 25 module files, so each of those messages goes to stderr and the agent never sees it. See X1.
2. **Empty and leftover folders: confirmed.** `git ls-files` gives 0 files for each of fdm2d, wind_loads, DM7Eqs, hvsrpy_agent, pyseismosoil_agent, swprocess_agent, pygef_agent, ags4_agent, pydiggs_agent and geolysis_agent, and `find` gives 0 files on disk, not even `__pycache__`. They are empty local directories, not in git. Stale mentions that still count: `pyproject.toml:276, 279` (empty extras `hvsrpy = []` and `swprocess = []`, which put two empty rows in the manual's extras table in docs/user_manual/manual.html) and `HANDOFF.md:589` (says `.claude/agents/seismic-reviewer.md` lists hvsrpy/swprocess; the file no longer does). The other mentions are history that already says the module was removed (CLAUDE.md:1643, 2504; docs/funhouse_agent_guide.md:180, 806; docs/funhouse_agent_expansion_plan.md; README.md:183), so they are fine.
3. **fem2d calc package: confirmed missing.** `calc_package/__init__.py:133` registers fem2d (17 modules in `_IMPORT_MAP`). The agent's METHOD_REGISTRY (funhouse_agent/adapters/calc_package.py:3107-3128) has `html_to_pdf`, `render_figures` and `render_tables` plus 16 packages, and none of them is fem2d. `render_*` cannot stand in for it, because they need a named `*_package` (calc_package.py:219-237). The model is told calc_package has "15 modules" (adapters/__init__.py:357). The manual says "17 analysis modules" and offers "Plotly interactive viewers" (docs/user_manual/manual_content.py:1255, 1838), but `save_interactive_report` has no caller in funhouse_agent or aec_agent_core.

### 2.2 Cross-cutting findings (one fix each, many modules)

- **X1. Python warnings are lost** (see 2.1). They hide four of the 32 A findings (axial_pile #2, drilled_shaft #1, lateral_pile #1 and #2) and several B findings, such as settlement's Cr > Cc warning (settlement/consolidation.py:74). The fix is in one place, around `method_call.py:622`: collect the warnings raised during the call and add them to a dict result under `warnings`. Two cautions. `catch_warnings` is not thread-safe and helpers run at the same time, so install a `warnings.showwarning` hook once and have it write to a context variable that is set around line 622. The default filter also shows a warning only once per code location (`__warningregistry__`), so module warnings need `filterwarnings("always")`, or every repeat after the first call is lost. This is a change to aec-agent-core, so it needs a core release.
- **X2. `allowed_values` is never enforced.** There are 198 declarations in 29 adapter files. Dispatch reads them only to suggest method names (dispatch.py:858), and `_check_parameter_names` (method_call.py:217) checks names, not values. So a value outside the list runs as something else. Two confirmed cases: `compute_trend` with "exponential" fits a line and labels it exponential (subsurface_characterization/statistics.py:158-177, no check at adapters/subsurface_adapter.py:1036-1048), and `format="docx"` renders HTML (calc_package/__init__.py:395, list at adapters/calc_package.py:3173). Before switching on a central check, confirm that every declared list is complete; a list missing a valid value would start refusing good calls.
- **X3. Told returns against `to_dict()`.** Keys the agent is told about are missing in ground_improvement feasibility, liquepy CPT (`n_liquefiable`, and `LPI` against `lpi`) and liquepy field_correlations (the profiles). Keys that exist but are not told about: subsurface plot (`skipped`, `trend_results`), concrete_props (interaction arrays), bearing_capacity (factors). Nothing checks either direction today.
- **X4. Documents that say what the code does not do.** DESIGN.md public-API sections name functions that do not exist in wave_equation, sheet_pile, pile_group, ground_improvement, retaining_walls and soe. Module counts disagree with each other ("15 modules" in the brief; "17 analysis modules" in the manual; CLAUDE.md's 32/30, registry check 5.1). The extras are stale. None of these reaches the agent except the brief, but builders read DESIGN.md.
- **X5. The calc helper and calc packages.** The calc helper (funhouse_agent/deep/agent.py:559 `build_calc_subagent`) is never told about `worked_examples`; only the main prompt mentions it (funhouse_agent/deep/prompt.py:177-188). fem2d, pystrata, seismic_signals, the structural engines and settlement's separate methods (Hough, Schmertmann alone, consolidation alone) have no package. No committed example shows today's package layout (§4, last subsection).

## 3. Kinds of finding

The 32 confirmed A findings, sorted by hand:

| Kind | Confirmed A | Findings |
|---|---|---|
| A failure or a missing input hidden (silent edge, default applied, warning lost) | 16 | settlement #1 and #2; axial_pile #1 and #2; drilled_shaft #1; lateral_pile #1 and #2; reliability #2; pystra #5; seismic_signals #1; subsurface #2, #3 and #4; dxf_import #1, #2 and #3 |
| The agent is told of a parameter, output or method that is not there or does nothing | 10 | ground_improvement #1; pile_group #1 and #3; soe #1; reliability #1; pystra #1 and #3; liquepy #1 and #2; fem2d #1 |
| A result that does not say what it is (mislabelled, wrong basis) | 5 | pystra #2 and #4; soe #2; pystrata #1; subsurface #1 |
| A stated method with a case missing | 1 | pile_group #2 (6-DOF without batter coupling) |

All findings by the scouts' check. B and C counts use the scouts' grades before re-grading. The shift is 9 from A to B, 6 from A to C and 5 from B to C, mostly under checks 1 and 3.

| Scout check | A (scout) | B (scout) | C (scout) |
|---|---|---|---|
| 1. Told against reachable, including documents that say what the code does not do | 16 | 38 | 47 |
| 2. Gaps the code admits itself | 0 | 5 | 5 |
| 3. Silent edges (unread parameter, fall-through option, unstated default, swallowed failure) | 33 | 42 | 52 |
| 4. Cases missing from a method the module says it covers | 0 | 28 | 9 |
| 5. Outputs missing (calc package, tables, figure, to_dict) | 1 | 20 | 18 |
| 6. Tests that pin only the code's own numbers (no published example) | 0 | 12 | 24 |
| 7. Already known | 0 | 0 | 2 |
| Registry checks | 0 | 3 | 18 |

The geotechnical modules were checked against their printed references in GM-281 to GM-298 (2026-10-30/31). What this audit finds in them is interface gaps, not engineering: silent edges and told-but-not-there. The modules never checked against the print are the reliability stack, liquepy, pystrata, opensees, seismic_signals, subsurface, dxf, the structural engines and fem2d. They have proportionally more of both kinds.

## 4. Module by module

Each table lists the A and B findings, plus any finding whose grade changed. C rows are counted only; they are in the scout file named in the heading. "Re-check" says what the coordinator found when opening the lines.

### axial_pile  (scout file `deep_axial.md`)

A 2, B 1, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | `capacity_vs_depth` wraps each trial depth in `except Exception: pass`: a depth that fails for any reason except a judgment refusal is dropped with no row and no message, and if every depth fails the tool returns an empty curve with no error. Drilled shaft's version records each skipped depth and ... | axial_pile/capacity.py:2015-2018; adapter returns it as is, adapter axial_pile.py:274-291 | new | re-checked: confirmed (axial_pile/capacity.py:2015-2018). |
| 2 | A | A pile tip deeper than the soil profile only raises a Python `warnings.warn` ("using full profile depth"); the result carries no warning, so the agent is never told the capacity assumes the last layer continues below the data. | axial_pile/capacity.py:341-346 (the "continues below" assumption stated at :797); no `catch_warnings` in funhouse_agent | new | re-checked: confirmed (capacity.py:341-346). Part of X1 (Python warnings never reach the agent). |
| 3 | B | The adapter tells the agent `capacity_vs_depth` returns rows `{depth, Q_ult, Q_skin, Q_tip}`; the real keys are `depth_m`, `Q_ultimate_kN`, `Q_skin_kN`, `Q_tip_kN`, plus `Q_factored_kN` and `warnings` when present. | funhouse_agent/adapters/axial_pile.py:389 vs axial_pile/capacity.py:2006-2014 | new | not re-checked |

### drilled_shaft  (scout file `deep_axial.md`)

A 1, B 2, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | A shaft longer than the soil profile is run with the last layer taken to go on below the profile, announced only by a Python `warnings.warn`; `DrillShaftResult` has no `warnings` field at all, so the agent never learns the base sits in assumed soil. | drilled_shaft/capacity.py:981-985, :1397-1398; drilled_shaft/results.py (no warnings field; to_dict :123-162) | new | re-checked: confirmed (drilled_shaft/capacity.py:981-985; results.py has no warnings field). Part of X1 (Python warnings never reach the agent). |
| 2 | A to B | `get_resistance_factors` hands the agent three uplift resistance factors (0.35 clay, 0.45 sand, 0.40 rock), and the manual says the module covers LRFD, but the module computes no uplift (tension) capacity anywhere; nothing reads the uplift factors. | drilled_shaft/lrfd.py:24-26; grep "uplift" in drilled_shaft/*.py finds only lrfd.py and a comment at capacity.py:590; adapter drilled_shaft.py:189-190 | GM-284 note | regraded A to B: the three factors are AASHTO's printed values; what is missing is an uplift capacity method to apply them to (lrfd.py:24-26; adapter drilled_shaft.py:189-190). The GM-284 print check already notes 'no uplift run' (module_work/drilled_shaft_print, line 17); not in BACKLOG. |
| 3 | B | `drilled_shaft_package` runs only `drilled_shaft_capacity`; the LRFD factored resistances (`lrfd_capacity`) and the capacity-vs-depth curve are not in the calc package, though the manual lists "LRFD factored resistance" as a covered output. The only factored item in calc_steps is the ... | funhouse_agent/adapters/calc_package.py:579-583; drilled_shaft/calc_steps.py:959-985; docs/user_manual/manual_content.py:606-609 | new | re-checked: confirmed (adapter calc_package.py:579-583). |

### downdrag  (scout file `deep_axial.md`)

A 0, B 4, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 3.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | The calc package covers only `downdrag_analysis`. Post-liquefaction downdrag and the six CGPR #56 methods (Endo, Poulos, Fellenius, PILENEG, rigid block, group reduction) have no calc package, and `PostLiquefactionResult` has no `get_tables`. | funhouse_agent/adapters/calc_package.py:584-587; downdrag/results.py:447 (get_tables only on DowndragResult); downdrag/liquefaction.py:666,750 | new | re-checked: confirmed (calc_package.py:584-587; get_tables only on DowndragResult, results.py:447). |
| 4 | B | The downdrag module has no way to turn off or replace the "last settling layer" test of which layers drag: a layer is marked `settling` by the user and nothing checks that the settling soil lies above the neutral plane or that fill/drawdown loads match the flag. The module documents the settlement ... | downdrag/DESIGN.md:970 (Fill placement: uniform, 1-D assumption) | new | not re-checked |
| 5 | B | GEC-12 Fig. 7-18 for a stiff clay under sand or soft clay is not offered (UFC Fig. 6-14 default, Fig. 7-17 as a warned option). | BACKLOG.md:232 | GM-258 parked | not re-checked |
| 6 | B | The headline neutral plane can sit 11.7 m below where the soil stops settling more than the pile (worked example); the module warns and gives every method side by side. | BACKLOG.md:69 | OWN-105 | re-checked: confirmed; open as OWN-105. |

### pile_group  (scout file `deep_axial.md`)

A 3, B 1, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | The user manual lists "an elastic equivalent-footing settlement (GEC-12 Sec. 7.3.5.3, Eq. 7-55 ...) ... as `influence_depth_m`" and the module docstring lists "Group settlement: Meyerhof (1976), the equivalent footing", but no adapter method reaches `group_settlement_equivalent_raft`; the only ... | docs/user_manual/manual_content.py:682-686 (pile_group methods list); pile_group/group_efficiency.py:644; funhouse_agent/adapters/pile_group.py:758-808 (no call to it; grep for `equivalent_raft`, `influence_depth_m`, `soil_modulus` in adapters and calc_package finds none) | new | re-checked: confirmed; manual cite corrected to docs/user_manual/manual_content.py:592-596; function at pile_group/group_efficiency.py:644, no adapter caller. |
| 2 | A | `pile_group_6dof` is described as "General 6-DOF rigid cap analysis" with batter angles accepted per pile, but the code leaves out the force-rotation coupling that batter introduces (kxz, kyz); the module's own docstring says a coupled analysis is preferred for heavily battered groups. Neither the ... | adapter pile_group.py:843-862 (no mention of limitation); pile_group/rigid_cap.py:422-427, :479-486 | GM-28 open (benchmark only; the missing statement to the agent is not covered) | re-checked: confirmed (rigid_cap.py:421-427, 479-486; adapter pile_group.py:841-862). GM-28 covers a benchmark only. |
| 3 | A | `pile_group_simple` says its `piles[]` entries may carry `batter_x, batter_y`, and top-level `Vx`, `Vy`, `Mz` and stiffnesses are refused, but a batter or stiffness given on an explicit pile is accepted and silently ignored (all piles are taken vertical); cap settlement `dz` is read from pile 1's ... | adapter pile_group.py:81, :832; pile_group/rigid_cap.py:337-345 (forces from x, y only), :371-376 (`piles[0].axial_stiffness`) | new | re-checked: confirmed (adapter pile_group.py:57-62, 81, 832; rigid_cap.py:338-352, 383-384 uses piles[0].axial_stiffness x n). |
| 5 | B | `block_failure_capacity` and `group_settlement_equivalent_raft` are exported with no adapter method; `block_failure_capacity` is a wrapper of the reached `block_failure_resistance`, the raft function is finding 1. | pile_group/__init__.py:26-27; adapter pile_group.py:7-12 | new | not re-checked |
| 4 | B to C | DESIGN.md "Public API" gives `analyze_pile_group(piles, loads, soil_params, ...) -> PileGroupResult`; no such function exists (the exports are `analyze_vertical_group_simple` and `analyze_group_6dof`). | pile_group/DESIGN.md (Public API section); pile_group/__init__.py:44-58; grep `analyze_pile_group` finds nothing in pile_group, funhouse_agent, geotech_common, calc_package | new | regraded B to C. DESIGN.md only: no agent or helper reads DESIGN.md, so it misleads builders, not the agent. Quick fix. |

### lateral_pile  (scout file `lateral_drive.md`)

A 2, B 5, C 6 (C rows are in the scout file), dropped 0, matched to a backlog item 5.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | If the soil layers start below the ground surface or stop above the pile toe, the module only calls `warnings.warn`: the run goes ahead with no soil resistance there and the result and the agent are told nothing. | lateral_pile/soil.py:148-158 | new | re-checked: confirmed (lateral_pile/soil.py:152-161). Part of X1 (Python warnings never reach the agent). |
| 2 | A | The cracked-EI loop for a reinforced-concrete pile reports non-convergence only through `warnings.warn`; `Results` has `ei_iterations` but no "EI converged" flag or message, so a non-converged EI profile is returned as an answer. | lateral_pile/analysis.py:514-518, results.py:183-210 | new | re-checked: confirmed (analysis.py:514-518; results carry only ei_iterations). Part of X1 (Python warnings never reach the agent). |
| 3 | B | `head_condition='partial'` with `rotational_stiffness` is implemented and tested (a rotational spring at the head) but the adapter allows only free and fixed, and `rotational_stiffness` is not in the valid-parameter list, so the agent cannot ask for a partly fixed head. | solver.py:568; analysis.py:352-355; funhouse_agent/adapters/lateral_pile.py:130-135, 250 | new | not re-checked |
| 4 | B | Cracked reinforced-concrete piles (`ReinforcedConcreteSection`, `Pile.from_rc_section`, `from_composite_section`, Branson EI iteration) are exported and used by the module, but no adapter method builds one; the agent gets only the uncracked `composite_section_ei` or a hand-set `moment_of_inertia`. | lateral_pile/__init__.py:14,18-20; pile.py:288,325,465; adapter _build_pile :103-123 | partly OWN-19 (moment-curvature EI); the unreachable RC section is new | not re-checked |
| 5 | B | `SandLiquefied` (Rollins et al. 2005, GEC 9 App. A.6) is a p-y model in the module, in DESIGN.md's list of "8 models", and in validation.py, but is not in `_PY_MODELS`; the agent's list is 7 models. | py_curves.py:1356; adapter :16-20; DESIGN.md:5,37 | GM-287 | re-checked: confirmed (seven models, adapters/lateral_pile.py:16-20); noted under GM-287. |
| 6 | B | `ReinforcedConcreteSection.fy` (steel yield) is a constructor field and appears in the docstring example but nothing reads it; Mcr, Icr and Branson EI use only the gross section, bars, f'c and E. | pile.py:478,499, whole class 465-725 | new | re-checked: confirmed (fy never read in pile.py classes 465-725). |
| 7 | B | The cracked-EI model takes the cracking moment from the gross section with no axial-load term, so `Q` does not enter Mcr or Icr even though `Q` is a solver input. | pile.py:553-557 vs analysis.py:482-499 | new | not re-checked |

### wave_equation  (scout file `lateral_drive.md`)

A 0, B 7, C 7 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 2 | B | With `hammer_name` given, `ram_weight`, `stroke`, `efficiency` and `hammer_type` are accepted by the validator and silently ignored (the named hammer's values win), and the `efficiency` default is declared "applies when hammer_name unset" only in the description. | adapter :33-41, :29, :442-445 | new | not re-checked |
| 3 | B | `Hammer.ram_stiffness` is a documented field ("None: rigid ram") that no code reads; every ram is rigid regardless. | hammer.py:48,57 (no other hit in the package) | new | not re-checked |
| 4 | B | The pile is always one uniform section: `discretize_pile` takes one area and modulus; `PileSegment`/`PileModel` could hold a stepped pile but the adapter cannot build one. A tapered, spliced or composite pile cannot be analysed. | pile_model.py:23,100; adapter :173,271 | new | re-checked: confirmed (pile_model.py:100-110 one uniform section). |
| 5 | B | `BearingGraphResult.capacity_at_blow_count` (the usual design question: capacity at the blow count the contractor will see) is not reachable; the agent can only read the table. When the target is outside the analysed range it silently returns the end value. | bearing_graph.py:62-81; adapter has no such method | new | not re-checked |
| 6 | B | The tool result gives driving stresses but no check against the allowable driving stress (0.9 Fy steel; concrete 0.85 f'c - f_pe and tension limits are only "named, not run"). Only the calc package gives the steel yield the stress needs. | DESIGN.md (driving stress note); calc_steps.py:575-590,746 | new | not re-checked |
| 7 | B | A diesel hammer is run as a rigid ram at rated energy at every resistance, with a stated warning; GEC-12's own Example 6 gives 50 and 103 blows/ft here against 99 and 209. | hammer.py:163-181; DESIGN.md | known: GM-29 open | not re-checked |
| 13 | B | Only GEC-12 Vol 2 Example 6 is pinned (the Vulcan 014, with the ICE 42-S as the diesel counter-example). Examples 1, 3, 4, 5 and 7 are quoted in the docs but have no test; the shaft-distribution tests cite page numbers but assert the module's shares. | tests/test_wave_equation.py:787; tests/test_gm300_embedment.py:189 | new | re-checked: confirmed (only Example 6 pinned: test_wave_equation.py:787, test_gm300_embedment.py:190). |
| 1 | A to C | DESIGN.md's "Public API" names `analyze_bearing_graph(...)` and `analyze_drivability(...)` and a `WaveEqResult` in results.py. None exists; the code exports `generate_bearing_graph`, `drivability_study` and `BlowResult`, and results.py holds only two plot functions. | wave_equation/DESIGN.md:33,37-38; __init__.py:20-31; results.py:18,57 | new | regraded A to C. DESIGN.md only: no agent or helper reads DESIGN.md, so it misleads builders, not the agent. Quick fix. |

### sheet_pile  (scout file `lateral_drive.md`)

A 0, B 5, C 6 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 2 | A to B | Active pressure is clamped at zero wherever cohesion would make it negative (`pa = max(pa, 0)`), in every routine, with no tension crack depth and no water in the crack; `tension_crack_depth` is exported and imported into `cantilever.py` but never called. The clamp is not mentioned in the result or ... | cantilever.py:24,883,904,1085,1097; anchored.py:228,240,298,309,347,357; earth_pressure.py:288 | new | regraded A to B: clamping net active pressure at zero is a common reading; whether a minimum pressure or water in the crack applies must be read from Caltrans / USS as printed (stage 2). The calc package already shows a crack depth (calc_steps.py:321-338). |
| 3 | B | The anchored wall's description says "embedment, anchor force, and moment" but its `returns` lists only embedment, anchor force and choices; `to_dict` also gives the maximum moment, its depth, total wall length, anchor depth. The cantilever `returns` omits total length, FOS and excavation depth. | adapter :256,268; anchored.py:127-138; cantilever.py:628-639 | new | not re-checked |
| 4 | B | Only a single anchor, and only free earth support. Fixed earth support, a second anchor level and Rowe's moment reduction are not offered; the DESIGN note says Rowe's reduction (UFC Fig. 4-36) and the D + 20 % step are "noted, not changed". | anchored.py:1-2; DESIGN.md (Cantilever safety basis, last paragraph) | new | not re-checked |
| 5 | B | No pile section: no required section modulus, section choice, deflection or toe-kick stability. The module returns embedment and moment per metre of wall only. | adapter :252,268 | new | re-checked: confirmed (no section modulus or deflection anywhere in sheet_pile/). |
| 6 | B | Cantilever and anchored wall defaults used with nothing in the result: friction angle 30 deg and cohesion 0 for a layer that gives none; `pressure_method` Rankine; the water level; each is in `defaults_used` through the adapter only. The 30 deg friction default is applied silently if ... | adapter :34-35 | new | not re-checked |
| 1 | A to C | DESIGN.md says the public API is `analyze_sheet_pile(wall_type, excavation_depth, layers, ...) -> SheetPileResult` and that `results.py` holds `SheetPileResult`. There is no such function, class or file; the real entries are `analyze_cantilever` and `analyze_anchored`. | sheet_pile/DESIGN.md:14-19 vs __init__.py:18-34; folder listing | new | regraded A to C. DESIGN.md only: no agent or helper reads DESIGN.md, so it misleads builders, not the agent. Quick fix. |

### 1. Registry against adapter files  (scout file `registry.md`)

A 0, B 0, C 2 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

No A or B finding.


### 2. METHOD_INFO against the dispatch dict  (scout file `registry.md`)

A 0, B 0, C 2 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

No A or B finding.


### 3. Documented parameters against the names the dispatch function reads  (scout file `registry.md`)

A 0, B 0, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

No A or B finding.


### 4. Module folders at the repo root against the registry  (scout file `registry.md`)

A 0, B 1, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 4.3 | B | `foundry/` (26 py files, per-module Foundry wrappers) and `foundry_test_harness/` (7 files) have no registry entry and nothing imports them. The two `from foundry.transforms import Dataset` lines in `webapp/foundry_dataset_store.py:154,355` name the platform's own package: the local `foundry/` has ... | `pyproject.toml:352-354,414` | known INF-06 [Later] | not re-checked |

### 5. Places that tell a person or the model what modules exist  (scout file `registry.md`)

A 0, B 2, C 7 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 5.1 | B | Stated count of analysis modules disagrees everywhere. CLAUDE.md says 32 (line 70, 1652), README 30 (lines 5, 72), `docs/overview.html` 30 (line 292), CLAUDE.md 30 (line 2443). The registry has 36 `ANALYSIS_MODULES`, or 29 once the seven tool-like entries (calc_package, profile_figure, ... | `CLAUDE.md:70`, `README.md:5`, `docs/overview.html:292`, `funhouse_agent/dispatch.py:81` | new | not re-checked |
| 5.2 | B | The registry's own description of calc_package says "15 modules": `adapters/__init__.py:357`. The code has 16 `*_package` modules and 17 package methods (slope_stability has two: `slope_stability_package` and `slope_report_package`): `calc_package.py:3107-3127`. The manual says "17 analysis ... | `adapters/__init__.py:357`, `calc_package.py:3107-3127`, `docs/user_manual/manual_content.py:1255` | new | re-checked: confirmed. |

### reliability  (scout file `reliability_stats.md`)

A 2, B 2, C 6 (C rows are in the scout file), dropped 0, matched to a backlog item 3.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | The adapter tells the agent that `lower`/`upper` truncate normal and lognormal variables, but FOSM and PEM never read the bounds: they evaluate at `mean ± std` of the untruncated variable, while Monte Carlo and FORM sample the truncated distribution. The same input gives engines that disagree, and ... | funhouse_agent/adapters/reliability_adapter.py:229-234; reliability/fosm.py:91-92; reliability/pem.py:136; reliability/variables.py:126-137,234 | new | re-checked: confirmed (reliability_adapter.py:229-234; fosm.py:91-92; pem.py:136). |
| 2 | A | Monte Carlo silently drops every realization where g is NaN or infinite. The result's `n` becomes the number kept, and no dropped count is stored in the result or `to_dict()`, so pf is computed on a smaller sample than the one asked for. | reliability/monte_carlo.py:103-106,141; reliability/results.py:142-213 | new | re-checked: confirmed (monte_carlo.py:103-114, 140-141). |
| 3 | B | With zero failures Monte Carlo reports pf = 0 and a 95% interval of exactly (0, 0). That is a Wald interval, which collapses at the edge, yet the adapter calls it a "95% binomial confidence interval". Zero failures reads as certainty. | reliability/monte_carlo.py:112-116; adapter:318 | new | not re-checked |
| 4 | B | The pre-canned wrappers `bearing_capacity_reliability`, `axial_pile_reliability`, `slope_reliability` and `run_engine` are exported but no adapter method reaches them. The engines take only a `g_expression` string, so the agent cannot run FOSM/PEM/MC/FORM on a calculation module's own FOS. SALib ... | reliability/wrappers.py:31,67,138,224; adapter registry :216-224 | LT-01 open covers "FORM/SORM on any module's limit state"; wrappers not named | not re-checked |

### pystra_agent  (scout file `reliability_stats.md`)

A 5, B 3, C 1 (C rows are in the scout file), dropped 1, matched to a backlog item 0.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | The adapter says every variable takes `mean` and `stdv` ("same as FORM") and lists `uniform` among the distributions. The code requires `a` and `b` for uniform and refuses mean/stdv. A uniform variable written as the agent was told fails, and `a`/`b` appear nowhere in METHOD_INFO. | funhouse_agent/adapters/pystra_adapter.py:71,105; pystra_agent/pystra_utils.py:438-446 | new | re-checked: confirmed (pystra_utils.py:438-446). Fails loudly, but the agent is told the wrong inputs. |
| 2 | A | When SORM returns nothing, `beta_breitung` and `pf_breitung` are filled with the FORM values, under the SORM names, with no flag. A FORM answer is reported as a SORM answer. | pystra_agent/reliability.py:246-249 | new | re-checked: confirmed (pystra_agent/reliability.py ~246-249). |
| 3 | A | `n_function_calls` is computed as iterations × number of variables (an estimate), and the adapter describes it as "Number of limit state evaluations". | pystra_agent/reliability.py:136-137,259; adapter:92,129 | new | re-checked: confirmed (reliability.py ~136-137, ~259; adapter :92, :129). |
| 4 | A | `converged` is not pystra's convergence flag. It is `beta > 0 and pf > 0` (FORM) or both betas positive (SORM). A converged run whose mean point is already in the failure domain (beta < 0) says False; an unconverged run with a positive beta says True. | pystra_agent/reliability.py:139-140,261-265 | new | re-checked: confirmed (reliability.py ~139-140, ~261-265). |
| 5 | A | Alpha, design point (x and u) and curvatures are each read in a `try / except Exception` that returns an empty dict or list. A failed extraction returns `{}` with no warning, and the result still looks complete. | pystra_agent/reliability.py:104-132,228-255 | new | re-checked: confirmed (reliability.py ~104-132, ~228-255). |
| 7 | B | Monte Carlo has no seed input and seeds from numpy's global generator, so a run differs from the last and the adapter does not declare it (no `declare_when_absent`, unlike reliability's monte_carlo). `n_failures` is rebuilt as `round(pf*n)`, and `cov_pf` is 0.0 when pf is 0, so zero failures reads ... | pystra_agent/pystra_utils.py:99-103; pystra_agent/reliability.py:355-362; adapter:149-153 | new | re-checked: confirmed (pystra_utils.py:97-104 seeds from the global generator). |
| 9 | B | No calc package, no figure and no calculation table for pystra results. `reliability` has all three. Result classes have `summary()` and `to_dict()`. | calc_package/__init__.py:116-134; pystra_agent/results.py:56,93,222,272,330,366 | new | not re-checked |
| 10 | B | The 45 tests are mostly structural (dataclass defaults, describe/dispatch). The only numeric check is the FORM-versus-`reliability.form` comparison in reliability/VALIDATION.md section 2. I found no numeric SORM test by grep (only `test_describe_sorm_analysis` and dataclass tests), and no published ... | pystra_agent/tests/test_pystra_agent.py:123-140,354; reliability/VALIDATION.md:25-36 | new | not re-checked |
| 6 | A to drop | The adapter does not call `reject_unknown_params` (the reliability, gstools and one-call salib adapters do). A misspelt top-level name such as `n_sample` is dropped and the default 100,000 runs. | adapter:10-54 (compare reliability_adapter.py:60,73,91,115) | new | dropped: wrong. Dispatch refuses an unknown top-level name for every method with documented parameters (`_check_parameter_names`, packages/aec-agent-core/aec_agent_core/method_call.py:217), so the misspelt name is refused, not ignored. |

### salib_agent  (scout file `reliability_stats.md`)

A 0, B 5, C 3 (C rows are in the scout file), dropped 1, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A to B | `sobol_analyze` takes `n_samples` and the adapter describes it as "Base number of samples used in sobol_sample()", but the body never reads it. The result's `n_samples` is `len(Y)`, the total number of evaluations, and the adapter's returns text calls it "Number of model evaluations". The same name ... | salib_agent/sensitivity.py:36-41,60-100; adapter (sobol_analyze METHOD_INFO) lines ~576-612 | new | regraded A to B: n_samples is unread (salib_agent/sensitivity.py:40-100) but the count used is len(Y), so the indices are right for the samples given; the description is wrong. Quick fix. |
| 2 | A to B | `morris_analyze` takes `n_trajectories` and only copies it into the result. SALib is never told it, and `X` is not checked against it, so a mismatched value is printed as the trajectory count of the run. | salib_agent/sensitivity.py:187-206 | new | regraded A to B: n_trajectories is only copied (sensitivity.py:180-206); same nature as #1. |
| 3 | B | `_validate_model_output` (Y length against sample count) is defined and never called. A Y of the wrong length goes straight to SALib. | salib_agent/sensitivity.py:32-36 (no call in the package) | new | re-checked: confirmed (`_validate_model_output` is never called). |
| 5 | A to B | The adapter says the expression follows "the reliability engines' g_expression rules" and the reliability adapter bounds powers and nesting (`safe_expr`: `_bounded_pow`, `_MAX_POW_BITS`). `salib_agent/expression.py` is a separate compiler: it allows `Pow` and I found no power-size or nesting limit ... | funhouse_agent/adapters/salib_adapter.py:433-434; salib_agent/expression.py:26-31; reliability_adapter.py:11-20 | new | regraded A to B (hardening): expression.py:26-31 allows ast.Pow with no bound and no nesting cap, unlike the reliability safe_expr; the adapter claims the same rules (salib_adapter.py:432-434). |
| 8 | B | No calc package, figure or calculation table for sensitivity results; `summary()` and `to_dict()` exist on both result classes. | calc_package/__init__.py:116-134; salib_agent/results.py:45,63,135,153 | new | not re-checked |
| 4 | A to drop | The four two-step adapter methods do not call `reject_unknown_params` (the one-call methods do), so a misspelt `seed`, `num_levels` or `calc_second_order` is silently ignored. Their `calc_second_order` default is True, against False in the one-call methods, so a sample matrix is 2D+2 per N unless ... | funhouse_agent/adapters/salib_adapter.py:341-420 vs :264; defaults :352,375 vs :288 | new | dropped: wrong. Dispatch refuses an unknown top-level name for every method with documented parameters (`_check_parameter_names`, packages/aec-agent-core/aec_agent_core/method_call.py:217), so the misspelt name is refused, not ignored. |

### gstools_agent  (scout file `reliability_stats.md`)

A 0, B 3, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | `Matern` and `Stable` are offered as covariance models, but their shape parameters (Matern smoothness nu, Stable alpha) are not an input anywhere. I found no `nu` or `alpha` in the four analysis files. The run uses whatever GSTools defaults to (or fits), and the result and `choices` do not report ... | funhouse_agent/adapters/gstools_adapter.py:28-29; gstools_agent/kriging.py:19-22,64-80; variogram.py:114-121; random_field.py:36-48; conditional.py:64-67 | new | not re-checked |
| 2 | B | Anisotropy (`anis`, `angle_deg`) exists only in `conditional_simulation`. `kriging`, `variogram` and `random_field` are isotropic and the variogram is omnidirectional, with nothing saying so. A kriged map of a layered or dipping surface ignores direction. | adapter:569-570; gstools_agent/kriging.py:64-80; variogram.py:114-121; random_field.py:36-48 | new | re-checked: confirmed (no anisotropy in kriging, variogram or random_field). |
| 4 | B | No calc package for kriging or conditional simulation. Figures (plan maps, exceedance curve) and CSV files exist, and result classes have `summary()` and `to_dict()`. | calc_package/__init__.py:116-134; gstools_agent/results.py:88,108,202,219,316,335,487,580 | new | not re-checked |

### geotech_common  (scout file `reliability_stats.md`)

A 0, B 2, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 3.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | `engineering_checks.py` (1,116 lines, ten `check_*` functions) is called by nothing but its own tests and the `__init__` re-export. Its docstring says it returns warnings "the LLM agent can include in its response", but no adapter, module, prompt or calc package calls any of them (repo-wide grep ... | geotech_common/engineering_checks.py:3-5; geotech_common/__init__.py:25-29 | new | not re-checked |
| 2 | B | The checked SPT correlations are not reachable: `spt_to_phi` (7 methods), `spt_to_cu`, `spt_to_relative_density`, `phi_to_Ka/Kp/K0`, `spt_energy_correction` and `spt_overburden_correction` are called by no module or adapter outside `geotech_common`. Only `spt_n1_60` and `stress_dilatancy_bolton` ... | geotech_common/soil_properties.py:68,181,250,304,325,345,390,471; funhouse_agent/adapters/subsurface_adapter.py:1064,1099 | GM-295 done (values checked); reachability not mentioned | re-checked: confirmed: the correlations are used only inside geotech_common.SoilProfile (soil_profile.py:30, 634, 647, 725, 822), and no adapter builds a SoilProfile (subsurface_characterization/site_model.py:357 `to_soil_profile` has no adapter caller). |

### DM7Eqs  (scout file `reliability_stats.md`)

A 0, B 0, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

No A or B finding.


### seismic_geotech  (scout file `seismic.md`)

A 0, B 8, C 6 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | The NCEER liquefaction route never applies K_sigma; CRR is the clean-sand curve at every depth, although Youd 2001 reduces CRR where sigma'v is above 100 kPa (unconservative for deep layers). The result's `choices` says so, but nothing flags which layers it affects. | seismic_geotech/liquefaction.py:187-193, 326-328 | new | not re-checked |
| 2 | B | Zero or negative effective stress is handled two different ways. `evaluate_liquefaction` sets it to 1.0 kPa (huge CSR). `compute_CSR` returns 0 and `csr_crr_check` then reports FOS 99.9 and liquefiable = False. Neither says it happened. | seismic_geotech/liquefaction.py:116-117, 271-272; funhouse_agent/adapters/seismic_geotech.py:233 | new | not re-checked |
| 3 | B | `post_liquefaction_strength` with Seed & Harder returns a flat 35 kPa for any (N1)60cs of 24 or more, and 0 for N of 0 or less. No warning in the result or the adapter. | seismic_geotech/residual_strength.py:49-52; adapters/seismic_geotech.py:196-209 | new | re-checked: confirmed (residual_strength.py:49-52). |
| 4 | B | `site_coefficients` holds the end value beyond the table (Ss above 1.25, S1 above 0.5, PGA above 0.5 or below 0.1) and says nothing in the result. | seismic_geotech/site_class.py:304-309 | new | not re-checked |
| 5 | B | Class F and the soft-clay rule for Class E are not checked. The adapter text says so, but a profile with soft clay or peat still returns an A-E class. | seismic_geotech/site_class.py:178-180, 187-210; adapters/seismic_geotech.py:348 | new | not re-checked |
| 8 | B | The adapter says "Unit weights (kN/m3)" for `gamma` and never tells the agent that each value is taken to apply over the interval ABOVE its depth. The B&I SPT route in liquepy uses gamma times depth instead. The same inputs give different overburden by route (see liquepy #3). | adapters/seismic_geotech.py:375; seismic_geotech/liquefaction.py:261 | new | re-checked: confirmed (adapter seismic_geotech.py:375; liquefaction.py:255-263 integrates the interval above each depth). |
| 11 | B | A calc package exists only for site classification, M-O and `liquefaction_evaluation` (NCEER). Nothing for gmax, modulus_reduction or residual_strength. Those return a bare float or dict with no result class, no `summary()`/`to_dict()`, no tables. | funhouse_agent/adapters/calc_package.py:589-600; seismic_geotech/residual_strength.py:213 | new | not re-checked |
| 13 | B | There is no published liquefaction example behind the module. INVENTORY itself says none was found. Liquefaction tests assert the code's own values, and the CRR/CSR tests are not tied to a page. | validation_examples/INVENTORY.md:1177; seismic_geotech/tests/test_seismic_geotech.py:572-573 | new | not re-checked |

### liquepy_agent  (scout file `seismic.md`)

A 2, B 6, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | The CPT tool is described as returning `n_liquefiable` ("Number of liquefiable points"). The CPT result has no such key, in `to_dict()` or the class. Both CPT tools carry the false claim. | funhouse_agent/adapters/liquepy_adapter.py:130; liquefaction_adapter.py:376; liquepy_agent/results.py:107-120 | new | re-checked: confirmed (liquepy_adapter.py:126-131; liquefaction_adapter.py:376; liquepy_agent/results.py:107-120; key case also differs: LPI/LSN vs lpi/lsn). |
| 2 | A | `field_correlations` is described as returning the Vs, Dr, su/sigma'v and I_c PROFILES. `to_dict()` gives only n_points, gwl, vs_method and Vs min/max/average and Dr min/max. The su ratio, permeability and I_c are never returned, and nor are the profiles. | adapters/liquepy_adapter.py:213-218; liquepy_agent/results.py:397-415 | new | re-checked: confirmed (liquepy_adapter.py ~213-218; results.py:397-415). |
| 3 | B | The B&I SPT route takes total stress as gamma times depth (each point's own gamma over the whole depth). The NCEER route integrates through the layers above. The adapter describes `gamma` the same way for both, so changing `method` changes the overburden as well as the procedure. `choices` states ... | liquepy_agent/spt_liquefaction.py:167-169, 238-242; seismic_geotech/liquefaction.py:261 | new | re-checked: confirmed (spt_liquefaction.py:167-168 sv = gamma[i] * z). Same finding as seismic_geotech #8, one item. |
| 4 | B | CPT `min_fos` is taken over liquefiable points (below the water table, Ic under the limit). When there are none it falls back to the minimum over ALL points. The value then comes from soil the result itself calls non-liquefiable, with no flag. | liquepy_agent/cpt_liquefaction.py:374-379 | new | not re-checked |
| 5 | B | The unified tool accepts `N` and `n_values` as spellings of (N1)60. A raw field N goes into a procedure that wants the corrected value, without a word. | funhouse_agent/adapters/liquefaction_adapter.py:34, 188, 217; declare_aliases 345-346 | new | re-checked: confirmed (liquefaction_adapter.py:193 takes N / n_values as N1,60). |
| 6 | B | The registry brief says "B&I 2014 CPT/SPT triggering, LPI/LSN/LDI". LPI, LSN, LDI, strain and settlement are CPT only. The SPT routes return layer FS only. | funhouse_agent/adapters/__init__.py:401; liquefaction_adapter.py:371 | REF-35 open (SPT strain option only) | brief text confirmed (adapters/__init__.py:399-402); the SPT outputs were not re-opened. |
| 9 | B | There is no calc package, no `get_tables` and no calc_steps for any liquepy result. B&I-2014 is the default for both SPT and CPT, yet the only liquefaction calc package is the NCEER one in `seismic_package`. | grep `get_tables` in liquepy_agent/results.py: 0 hits; adapters/calc_package.py:589-600 | new | not re-checked |
| 10 | B | The SPT test called "published" checks CRR within +/-0.04 of about 0.29 (a rounded number, not a page). INVENTORY has no liquefaction example. CPT results depend on the installed liquepy and are tested against its own output. | liquepy_agent/tests/test_spt_liquefaction.py:119-129; validation_examples/INVENTORY.md:1177 | new | not re-checked |

### pystrata_agent  (scout file `seismic.md`)

A 1, B 5, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | `stress_mean` for Darendeli and Menq layers is documented as "mean effective stress" but is computed from TOTAL stress (sum of unit_wt times thickness), K0 fixed at 0.5. There is no water table input anywhere in the tool. Below the water table the curves are built for too high a confinement. The ... | pystrata_agent/eql_site_response.py:181-205; adapters/pystrata_adapter.py:96 | new | re-checked: confirmed (pystrata_agent/eql_site_response.py:182-205: total stress, K0 0.5, no water table). |
| 2 | B | The module computes the surface and input response spectra (5 %), the surface time history, and strain, acceleration and strain-compatible Vs by depth. `to_dict()` and `METHOD_INFO.returns` carry only scalars (PGA, amplification, iterations, peak strain). The agent cannot get a site-response ... | pystrata_agent/results.py:109-124; eql_site_response.py:482-500; adapters/pystrata_adapter.py:130-141 | new | re-checked: confirmed (results.py:109-124 scalars only). |
| 3 | B | The motion is applied as a rock OUTCROP motion (`location("outcrop", index=-1)`). The pystrata adapter never says so. OpenSees' says "taken as a rock outcrop motion". | pystrata_agent/eql_site_response.py:479-480; adapters/opensees_adapter.py (motion param) | new | not re-checked |
| 4 | B | Only two built-in motions exist, both synthetic. A comment marks real records as "future". No adapter method lists the motions (`list_motions` is not reachable), so the agent learns the names only from description text. | opensees_agent/ground_motions.py:124-137, 177; grep `list_motions` in funhouse_agent/adapters: no hit | new | not re-checked |
| 7 | B | No calc package, no tables, no figure for site response. `plot_*` methods exist but no adapter or package calls them. | grep `get_tables`/`_package` for pystrata: none | new | re-checked: confirmed (no pystrata package in adapters/calc_package.py METHOD_REGISTRY). |
| 8 | B | The tests carry no published example or closed-form check (uniform layer over rock). HANDOFF records a pystrata-vs-OpenSees cross-check (0.749 vs 0.758 g), but no pystrata test cites it. | pystrata_agent/tests/test_pystrata_agent.py (grep: none); HANDOFF.md:685 | new | not re-checked |

### opensees_agent  (scout file `seismic.md`)

A 0, B 4, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | The PM4Sand integration tests cannot fail on a wrong answer. They assert `max_ru >= 0`, array lengths, and the "dense resists better" test passes when both liquefy or neither does. There is no published PM4Sand example. The site-response tests are pinned (wiki and docs examples). | opensees_agent/tests/test_opensees_agent.py:564-595 | new | not re-checked |
| 2 | B | PM4Sand DSS is sold as "for liquefaction triggering", but it returns the cycles to liquefaction at ONE `CSR` (default 0.15). There is no CRR at a given number of cycles and no CSR-N curve. The agent has to loop it. The G0, hpo and Dr inputs have no calibration help from (N1)60. | adapters/opensees_adapter.py:98-100, 183; pm4sand_dss.py (single CSR) | new | not re-checked |
| 3 | B | `max_ru` is clipped to [0, 1] in the DSS result. The site-response tool states its ru is "not capped at 1". The DSS cannot show an exceedance. | opensees_agent/pm4sand_dss.py:383-384; adapters/opensees_adapter.py (max_ru) | new | not re-checked |
| 5 | B | No calc package, tables or figure for either method. | grep `get_tables`/`_package`: none | new | not re-checked |

### seismic_signals_agent  (scout file `seismic.md`)

A 1, B 4, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | `intensity_measures` swallows an `AttributeError`/`TypeError` from `calc_brac` and returns `bracketed_duration_s` = 0.0, which reads as a real value. The bracketed-duration threshold is never stated either. | seismic_signals_agent/intensity_measures.py:90-95; results.py:252 | new | re-checked: confirmed (intensity_measures.py:88-92). |
| 2 | B | `rotd_spectrum` zero-pads the shorter component to the longer one's length without telling the user. Only `n_points` shows. | seismic_signals_agent/rotd_spectrum.py:107-112 | new | re-checked: confirmed (rotd_spectrum.py:107-112). |
| 3 | B | `signal_processing` returns only peak values. The filtered or baseline-corrected record (which the module holds) cannot be taken out or passed to site response. The filter type and order are not stated in the tool text. | seismic_signals_agent/signal_processing.py:101-123; results.py:538-550 | new | not re-checked |
| 5 | B | Only two synthetic built-in motions, and `rotd_spectrum` has no `target_pga_g` while the other three methods do. | opensees_agent/ground_motions.py:177; adapters/seismic_signals_adapter.py:68-83 | new | not re-checked |
| 8 | B | No calc package, tables or figure for any signals method. | grep `get_tables`/`_package`: none | new | re-checked: confirmed (no signals package in the calc_package METHOD_REGISTRY). |

### hvsrpy_agent, pyseismosoil_agent, swprocess_agent  (scout file `seismic.md`)

A 0, B 0, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 3.

No A or B finding.


### bearing_capacity  (scout file `shallow.md`)

A 0, B 1, C 6 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | `concrete_design.py` (footing structural checks, ACI 318 development length) is in no `__init__` export, no adapter method and no calc package; only its own test imports it, so the agent cannot reach it. DESIGN.md lists it as a module file with no word that it is unreachable. | bearing_capacity/tests/test_concrete_design.py:6; bearing_capacity/__init__.py:25-31; DESIGN.md:27 | new | not re-checked |

### settlement  (scout file `shallow.md`)

A 2, B 4, C 6 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | `consolidation_settlement` with a `delta_sigmas` list shorter or longer than `layers` silently drops the extra layers (or extra stresses) and returns a smaller total; no length check anywhere. | funhouse_agent/adapters/settlement.py:163 (`zip(layers, delta_sigmas)`); no `len(` check in adapter or settlement/consolidation.py | new | re-checked: confirmed (adapter settlement.py:163 zips with no length check). |
| 2 | A | Schmertmann sums only the part of each layer inside the influence depth z_max; layers that stop short of z_max contribute nothing for the missing depth, and nothing is said. A profile covering only 1B of a 2B influence zone gives a settlement that is too small with no warning. | settlement/immediate.py:424-431 | new | re-checked: confirmed; cite is settlement/immediate.py:422-435, z_max at :394. |
| 3 | B | A Cr greater than Cc raises a Python `warnings.warn`, which never reaches the agent: the adapter returns no warning field for `consolidation_settlement`, `schmertmann_settlement` or `hough_settlement` (only the combined analysis carries `warnings`). | settlement/consolidation.py:74; adapter settlement.py:133,184,220 | new | mechanism confirmed under X1; consolidation.py:74 not opened. |
| 4 | B | Hough (granular, SPT) settlement cannot be combined with consolidation or secondary: `combined_settlement_analysis` offers immediate_method only elastic or schmertmann. A sand-over-clay profile (the GEC-6 B-1 case) needs two separate tool calls added by hand. | adapter settlement.py:375; settlement/__init__.py:9 | new | re-checked: confirmed (adapter settlement.py:375 offers elastic/schmertmann only). |
| 5 | B | The calc package covers `combined_settlement_analysis` only. Hough, Schmertmann alone, elastic alone and standalone consolidation have no package route and no calculation tables, although Hough's `HoughResult` has summary() and to_dict(). | funhouse_agent/adapters/calc_package.py:1798-1810; grep hough in settlement/calc_steps.py: no hit | new | not re-checked |
| 6 | B | Exported time-rate and secondary functions (`time_factor`, `degree_of_consolidation`, `time_for_consolidation`, `settlement_at_time`, `secondary_settlement`, `total_consolidation_settlement`, `stress_at_depth`) are reachable only inside the combined analysis; a user who wants only "t90 for cv = ... | settlement/__init__.py:42-51; adapter settlement.py:133 | new | not re-checked |

### ground_improvement  (scout file `shallow.md`)

A 1, B 5, C 5 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | The adapter tells the agent `feasibility` returns `recommended_methods` and `is_feasible`. `FeasibilityResult.to_dict()` returns `applicable_methods`, `not_applicable`, `recommendations`, `preliminary_sizing`, `soil_description`, `design_problem`; neither named key exists anywhere in the code. An ... | funhouse_agent/adapters/ground_improvement.py:354 vs ground_improvement/results.py:545-552 | new | re-checked: confirmed (adapter ground_improvement.py:354 vs ground_improvement/results.py:545-553). |
| 3 | B | `surcharge_preloading` returns, per the adapter, only `time_to_target_years`; the result also carries settlement at target, ultimate settlement, equivalent sigma_p, wick-drain sub-result and time curve. | adapter ground_improvement.py:323 vs results.py:367-382 | new | not re-checked |
| 4 | B | `equivalent_sigma_p_kPa` is 0.0 whenever `sigma_v0` is left at its default 0, with no statement; the adapter lists sigma_v0 default 0.0 as "current effective stress". A 0 reads as a computed value. | ground_improvement/surcharge.py:339-342; adapter ground_improvement.py:320 | new | not re-checked |
| 6 | B | In the feasibility screen a clay with no `cu_kPa` is taken as soft enough for aggregate piers ("assume soft enough"); the result lists the method as applicable without saying the strength was assumed. | ground_improvement/feasibility.py:232-235 | new | re-checked: confirmed (feasibility.py:233-234). |
| 7 | B | DESIGN.md says "four ground improvement techniques", and the feasibility screen recommends only piers, wick drains, surcharge and vibro. GEC-13 methods outside it (dynamic compaction, jet grouting, deep soil mixing, column-supported embankments) have no design or screening tool here; the reference ... | ground_improvement/DESIGN.md:3-5; feasibility.py:200,317,383,436 | OWN-21 open (parked scope, names deep mixing and column-supported embankments) | not re-checked |
| 8 | B | Calculation tables: only wick drain and surcharge results have `get_tables()`; aggregate piers, vibro and feasibility have none, though all three are in the calc package route. | ground_improvement/results.py:255,386 (no other get_tables) | new | not re-checked |
| 2 | B to C | DESIGN.md "Public API" signatures are wrong for four functions: `analyze_aggregate_piers(B, L, q_applied, ...)`, `analyze_wick_drains(cv, ch, drain_spacing, ...)`, `analyze_surcharge_preloading(target_consolidation, ...)`. Real signatures take column_diameter/spacing, spacing/ch/cv/Hdr/time, ... | ground_improvement/DESIGN.md:26-30 vs aggregate_piers.py:252, wick_drains.py:367, surcharge.py:270 | new | regraded B to C. DESIGN.md only: no agent or helper reads DESIGN.md, so it misleads builders, not the agent. Quick fix. |

### pavement_design  (scout file `shallow.md`)

A 0, B 5, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | No calculation tables for pavement: `pavement_design/calc_steps.py` has no `get_tables` or `CalcTable`, and no test of tables exists, while 13 other modules have them (GM-39). | grep get_tables / CalcTable in pavement_design/calc_steps.py: no hit; pavement_design/tests: only test_pavement_design.py | new | re-checked: confirmed (no get_tables in pavement_design calc_steps/results). |
| 3 | B | The adapter says `design_traffic_esals` takes triple axles. If the installed geotech_references lacks `aashto_1993.lef`, the fallback raises `NotImplementedError` for triple axles. Pinned refs 1.6.0 has the module, so the path is dormant today; the same import-guard pattern guards composite k and ... | pavement_design/traffic.py:80-109; composite_k.py:193; performance.py:33; rigid.py:99 | new | not re-checked |
| 4 | B | `design_traffic_esals` computes LEFs at an assumed SN = 5.0 (or D = 9 in) and pt = 2.5 when not given; the adapter says "LEFs at the assumed sn/d_in/pt" but the defaults enter the result only via parameters, not in a `choices` or notes line. Truck factors depend on SN/D, so the W18 shifts with the ... | adapter pavement_design_adapter.py:481-492; traffic.py:80-90 | new | not re-checked |
| 5 | B | A refused asphalt modulus below the chart range, and other chart clamps (UFC and AASHTO), are known to be silent or refused in ways the app does not yet handle. | pavement_design/flexible.py:129 | REF-74 | re-checked: confirmed; open as REF-74. |
| 6 | B | Scope not covered, as DESIGN.md and `__init__` state: overlays/rehabilitation (AASHTO Part III), rigid joint and reinforcement design, low-volume catalog, UFC 3-260-02 airfield and rigid curves F-2 to F-31. These are stated, not hidden. | pavement_design/__init__.py:30-34 | known OWN-21 (parked) | not re-checked |

### wind_loads  (scout file `shallow.md`)

A 0, B 0, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

No A or B finding.


### subsurface_characterization  (scout file `site_data.md`)

A 4, B 6, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | `compute_trend` takes any `trend_type`. Anything other than "log_linear" (for example "exponential" or "power") is fitted as a straight line, and the result echoes the name the caller gave, so the label and the fit disagree. | subsurface_characterization/statistics.py:158,177 (else branch); adapter lists allowed_values only in METHOD_INFO, funhouse_agent/adapters/subsurface_adapter.py:1303, and `_run_compute_trend` does not check them (line 1041) | new | re-checked: confirmed (statistics.py:150-195; adapter subsurface_adapter.py:1036-1048 does not check trend_type). Part of X2. |
| 2 | A | With fewer than 2 points (or fewer than 2 positive points in log-linear), `compute_trend` returns a result with slope 0, R-squared 0 and COV 0 and nothing says "not fitted". The same happens in `plot_parameter_vs_depth` with `show_trend`. With exactly 2 points it returns R-squared 1 and a residual ... | statistics.py:150-155, 161-165, 189 (`std_residual` 0 when n <= 2); to_dict has no flag, statistics.py:97-109 | new | re-checked: confirmed (statistics.py:150-195). |
| 3 | A | Log-linear fit silently drops points with depth <= 0 or value <= 0, yet `n_points` reports the full count; the line is also drawn from a clamped 0.01 m when the plot starts at depth 0. | statistics.py:159-168 (mask), 193 (n_points=n), 73-74 (clamp) | new | re-checked: confirmed (statistics.py:150-195). |
| 4 | A | GEF/BRO CPT parse replaces every NaN, null or infinite value (qc, fs, u2, Rf) with 0.0 and does not count or report it. A missing channel value therefore reads as a zero cone resistance, and `load_cpt_to_investigation` turns those zeros into measurements. | subsurface_characterization/formats/gef.py:162-164; subsurface_characterization/csv_loader.py:166-198 | new | re-checked: confirmed (formats/gef.py:160-164). |
| 5 | B | `load_cpt_to_investigation`, `load_site_from_csv` and `compute_grouped_trends` are exported but no adapter method reaches them. In particular a CPT read with `parse_cpt` cannot be added to a site model and plotted (the arrays come back as JSON only). Grouped trends are only reachable through ... | subsurface_characterization/__init__.py exports; funhouse_agent/adapters/subsurface_adapter.py:1117-1136 (registry) has no such method | new | not re-checked |
| 6 | B | `read_ags4`, `validate_ags4`, `validate_diggs_schema`, `validate_diggs_dictionary` do not list `attachment_key` in METHOD_INFO, while the error text they raise tells the caller to use one; `parse_cpt` / `parse_bore` take `file_path` only (required), so a GEF or BRO file uploaded to the chat has no ... | subsurface_adapter.py:38-44, 1322, 1342, 1358, 1376; funhouse_agent/dispatch.py:990-995 | new | not re-checked |
| 7 | B | `plot_parameter_vs_depth` is told to return "plot_type, n_data_points, n_investigations" but `PlotResult.to_dict` also carries `trend_results` and `skipped` (what was left out, such as a trend line not drawn for lack of elevation). METHOD_INFO does not mention `skipped`, so the agent is not told to ... | subsurface_adapter.py:1219-1223; subsurface_characterization/plots_xy.py:264-273, 296-300 | new | re-checked: confirmed (returns at subsurface_adapter.py:1219-1223; `skipped` in plots_xy.py:27-41). |
| 9 | B | `_read_csv` treats a path that cannot be opened as the CSV text itself, so a mistyped file name loads an empty or garbage site with no error. | subsurface_characterization/csv_loader.py:222-230 | new | not re-checked |
| 11 | B | No calc package, no calculation table and no `summary()`-bearing result for the site model itself (`SiteModel` has no summary). Results classes (`PlotResult`, `DiggsParseResult`, `TrendAnalysisResult`, the five format results) all have `summary()` and `to_dict()`. | grep of `_package` in subsurface_characterization and its adapter finds nothing; results.py:57,75,135,153 | new | not re-checked |
| 12 | B | No published example behind any of it. validation_examples has no subsurface entry, and the tests check the module's own output on made-up files (sample_cpt.gef, sample_bore.gef). `read_ags4` and `parse_bore_file` each appear in only 2 test files, `compute_grouped_trends` in 2. | validation_examples/ (no match), subsurface_characterization/tests/ | new | not re-checked |

### pygef_agent / ags4_agent / pydiggs_agent / geolysis_agent (dead folders)  (scout file `site_data.md`)

A 0, B 0, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 3 | A to C | `geolysis_agent`: it was removed on purpose ("no new value vs base + groundhog", `bearing_capacity` + `groundhog_agent` replace it). No registry entry, no code, no CLAUDE.md claim. The agent cannot reach it and nothing is lost; the only trace is the empty folder. | CONSOLIDATION_CHANGES.md:17,32,62-65; funhouse_agent grep for "geolysis" empty | closed by design | regraded A to C: removed on purpose; nothing to fix beyond the stale mentions in X4. |

### dxf_import  (scout file `site_data.md`)

A 3, B 4, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | Text annotations are not converted to metres. Surface, boundaries, water table and nails are converted from the drawing units; TEXT/MTEXT x and y are copied raw. A drawing in mm or ft returns annotation positions in drawing units next to coordinates in metres. | dxf_import/parser.py:150,167,182,209-223 (no `convert_coords`); nails do convert, parser.py:320-334 | new | re-checked: confirmed (dxf_import/parser.py:200-223). |
| 2 | A | Soil strength defaults are applied silently. A soil_properties entry without gamma gets 18.0, without phi 0.0, c_prime 0.0, cu 0.0, "drained", and FEM entries get E 30000, nu 0.3, psi 0, model "mc". DESIGN/converter say these "must always come from the user", and the result does not list which were ... | funhouse_agent/adapters/dxf_import_adapter.py:173-182, 236-247; dxf_import/converter.py:9-11, 47-52, 261-266; adapter METHOD_INFO 299, 314 | new | re-checked: confirmed (dxf_import_adapter.py:172-182, 236-247; contradicts converter.py:9-11). |
| 3 | A | Soil nails get structural defaults silently: bar diameter 25 mm, drill hole 150 mm, fy 420, bond stress 100, spacing 1.5 m when `nail_defaults` is missing a key. The nail length and angle come from the line, but a nail drawn tip-to-head the wrong way round is "fixed" by taking the leftmost point as ... | dxf_import/converter.py:196-220; parser.py:329-342 | new | re-checked: confirmed (converter.py:194-220). |
| 4 | B | Invented depths. A single-layer slope gets a bottom 5 m below the lowest surface point; the lowest layer gets 2 m below its boundary's lowest point. Not reported in the result. | converter.py:117, 169-170 (and FEM twin 328, 370) | new | not re-checked |
| 5 | B | Polyline points from every entity on a layer are merged, de-duplicated and sorted by x. A vertical face, an overhang, a retaining wall or two disjoint polylines on one layer therefore lose their order; no warning. | parser.py:236-296 (sort at 291) | new | not re-checked |
| 6 | B | `units` defaults to "m" and is never compared with the DXF's own $INSUNITS header (`discover_layers` reports `units_hint`, `parse_geometry` ignores it). A drawing in mm run with the default produces a section 1000 times too large with no warning. | dxf_import/parser.py:102-105, 150, 231; discovery.py:96; adapter default, dxf_import_adapter.py:95 | new | re-checked: confirmed (parser.py:63, 102 default 'm'; only discovery.py:96 reads the header units). |
| 7 | B | Quiet skips: a missing annotations layer is skipped with no warning (the surface, boundary, water and nail layers do warn); on the nail layer anything that is not a LINE (a polyline nail) is dropped with no count. | parser.py:202-203, 322-324 | new | not re-checked |

### dxf_export  (scout file `site_data.md`)

A 0, B 3, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 0.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | `units` only sets the DXF $INSUNITS header; no coordinates are scaled. Telling it "ft" for metre coordinates writes a drawing that claims feet. | dxf_export/writer.py:70-73 | new | not re-checked |
| 2 | B | Layer names are `BOUNDARY_<soil name>` with the name used as given. A name with a character DXF layer names do not allow (a slash, colon, quote) is not cleaned or warned of; whether ezdxf raises or writes a bad layer was not tested. After a re-import the soil names are lost anyway (the user must ... | writer.py:96-99 | new | confirmed in code (writer.py:96-99); an edge case, no item (rule of 2026-10-27). |
| 3 | B | A nail line missing a key (`x_head` etc.) raises a bare KeyError; the adapter pre-checks key names (`_check_items`) but not that all four are present. An annotation without x or y is written at (0, 0) silently, and a missing height gets 0.5. | writer.py:117-120, 130-133; funhouse_agent/adapters/dxf_export.py:19-30 | new | not re-checked |

### profile_figure  (scout file `site_data.md`)

A 0, B 2, C 5 (C rows are in the scout file), dropped 0, matched to a backlog item 4.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | The agent is told `plot_data` and `subsurface_profile` live on module `profile_figure`, but the code for `plot_data` is in the core package (`funhouse_agent.charts`, aliased). `profile_figure.render_data_plot` is loaded lazily through `__getattr__`; the local files `data_plot.py` and ... | profile_figure/__init__.py:46-56; profile_figure/data_plot.py:1-11; funhouse_agent/adapters/profile_figure_adapter.py:28-38,527-533 | INF-35 / INF-31 known | not re-checked |
| 2 | B | `layers` and `fill` are left open on purpose (an analysis's own layer dicts draw as they are), so a misspelt layer key ("thickness_m", "desc") is dropped silently; only `surcharge`, `foundation` and `annotations` are key-checked. | profile_figure_adapter.py:54-64, 72-80 (comment lines 54-58) | new | confirmed; by design (adapter comment profile_figure_adapter.py:54-58). |

### geo_project  (scout file `site_data.md`)

A 0, B 4, C 4 (C rows are in the scout file), dropped 0, matched to a backlog item 5.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | The model-setup agent (the whole staged, human-gated LE/FEM setup workflow) is switched off in every host. No app page, launcher or profile turns `enable_setup_agent` on, so the agent cannot reach `geo_project` in the shipped app. | funhouse_agent/deep/agent.py:868, 1390; funhouse_agent/deep/setup_agent.py:15; Grep of webapp/ for `enable_setup_agent` returns nothing | OWN-22 open (Parked) | not re-checked |
| 2 | B | DESIGN.md says the notebook form renderer for the confirmation payload is "designed, not shipped"; the chat fallback is the shipped route. Stated in the document. | geo_project/DESIGN.md (section "Notebook forms") | known (stated) | not re-checked |
| 3 | B | Known drift: no anisotropic su, no pore-pressure points, no stabilizing piles or tension cracks; the LE builder maps only the first surcharge; no confirmed-IR ingest; slope parts follow the slope hold GM-140. | BACKLOG.md:98 (OWN-22, was GM-37) | OWN-22 | re-checked: confirmed; open as OWN-22. |
| 8 | B | Seven test files, but no published example (the module builds inputs for slope_stability and fem2d; its numbers are theirs). Whether `from_pdf_vector` and `from_vision_draft` are tested end to end was not read. | geo_project/tests/ | new | not re-checked |
| 4 | A to C | Probabilistic runs carry a silent edge: if an analysis asks for FOSM or Monte Carlo and no variables are defined, the entry gets an error string inside the result rather than failing the run, so a `project_run` can look complete with the probabilistic part missing. | geo_project/builders.py:301-310 | new | regraded A to C: the error string is stated in the entry (geo_project/builders.py:301-310). |

### slope_stability  (scout file `slope_numerical.md`)

A 0, B 5, C 7 (C rows are in the scout file), dropped 0, matched to a backlog item 5.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | Only one slope calc package route exists for a single circle or search (`slope_stability_package`, `slope_report_package`); rapid drawdown, Newmark / yield acceleration, FOSM and Monte Carlo results have no calc package, so a report of those runs has no package. | funhouse_agent/adapters/calc_package.py:3114-3115 (only two slope entries in the registry, 3111-3127) | new | not re-checked |
| 6 | B | `spencer_fos_legacy` and `morgenstern_price_fos_legacy` are exported in `__all__` and used as the fallback when the rigorous solver fails, but no test or published example names either (0 files under slope_stability/tests or validation_examples). The fallback is what GM-139 says can return a state ... | slope_stability/__init__.py:64; slope_stability/methods.py:445; 0 grep hits for the names in tests | GM-139 open (covers the false FOS; not the missing tests) | not re-checked |
| 7 | B | DESIGN states "Slide2's impenetrable material and composite surfaces ... are not implemented": a firm layer inside the model is crossed by every circle with its own strength. A user with a bedrock or firm-layer surface cannot get a surface that follows its top except by giving a polyline. | slope_stability/DESIGN.md:1366-1369 | GM-140 | re-checked: confirmed; already in GM-140's text (scout marked it new). |
| 8 | B | Methods the code does not have, as DESIGN/backlog state: Janbu rigorous, Sarma, Barton-Bandis, liquefied (residual) strength, curved Mohr-Coulomb envelope (published Example 5 skipped for this). | slope_stability/DESIGN.md:1837; BACKLOG GM-19 | GM-19 parked (deferred to GM-140) | not re-checked |
| 9 | B | Circular analysis of a c'=0 slope overestimates FOS against the infinite-slope value; `analyze_slope` on a given circle still carries this ("a circle scored on its own"), only the search compensates with planar surfaces. The result says nothing about it. | slope_stability/DESIGN.md:1411-1417 | partly GM-14 (done); the single-circle case is stated in DESIGN only | not re-checked |

### fem2d  (scout file `slope_numerical.md`)

A 1, B 3, C 7 (C rows are in the scout file), dropped 0, matched to a backlog item 4.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | `calc_package` registers `fem2d.calc_steps` as a supported module and `fem2d` has `get_tables()` and calculation tables (GM-39 lists "fem2d (FEM, staged, consolidation)" as done), but the agent has no `fem2d_package` method: the registry of package methods (3107-3127) has no fem2d entry. The tables ... | calc_package/__init__.py:133 (registered); funhouse_agent/adapters/calc_package.py:3107-3127 (no fem2d entry); BACKLOG_DONE.md:692 (GM-39 claims fem2d tables) | new | re-checked: confirmed (calc_package/__init__.py:133 registers fem2d; adapters/calc_package.py:3107-3128 has 16 packages, none fem2d; render_tables/render_figures need a named *_package, calc_package.py:219-237). |
| 2 | B | `mc_return_mapping(..., sigma_t=0.0)` documents a tension cutoff ("tension cutoff (kPa), default 0") but never reads it, and neither caller passes it. The in-plane Mohr-Coulomb return has no tension cut-off, so any MC soil can carry tension up to the c/tan(phi) apex, whatever the docstring says. | fem2d/materials.py:60 (declared), :84 (documented); callers fem2d/solver.py:1257-1260 and fem2d/materials.py:309-310 (no sigma_t) | new | re-checked: confirmed (fem2d/materials.py:60-61, 84: sigma_t stored, never used). |
| 4 | B | The staggered consolidation scheme transports a prescribed pore field; it does not create load-induced excess pore pressure. A user who picks the staggered scheme and applies a load gets no undrained pore pressure. DESIGN says "use the monolithic scheme"; whether the adapter default and the ... | fem2d/DESIGN.md:752-756 | new | not re-checked |
| 7 | B | Open feature gaps the backlog itself lists: strut preload and inclined anchors, Hardening Soil layers in the staged excavation untested (K0 and lifts verified with Mohr-Coulomb only), wells lowering water below the excavation level refused, Q8, cohesionless chatter flag, cavity/tunnel analyses. | BACKLOG GM-73, GM-74, GM-75, GM-10 | known (GM-73/74/75 Later, GM-10 Later) | not re-checked |

### fdm2d  (scout file `slope_numerical.md`)

A 0, B 0, C 2 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

No A or B finding.


### pynite_agent  (scout file `structural_calcpkg.md`)

A 0, B 8, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A to B | `frame_analysis` accepts a member key `rho` (density) and passes it to PyNite, but no self-weight load is ever applied, so a model given rho runs with no self-weight and says nothing. The key is not in METHOD_INFO either. | pynite_adapter.py:21; frame.py:384 (add_material), no add_member_self_weight anywhere | new | regraded A to B: no self-weight is applied (pynite_adapter.py:21; pynite_agent/frame.py:61); the agent can add it as a load, but the description should say so. |
| 2 | B | The frame result carries no node displacements. "max_deflection_mm" is the member's local-axis deflection (dy) from PyNite, not sway of a frame, so a portal frame's sideways drift cannot be read. | frame.py:443-485 (only Mz, Fy, axial, dy read); results.py:603-651 | new | not re-checked |
| 1 | B | The module says "2D/3D" frames, but only Mz, Fy, axial and dy are extracted. No Mx torsion, My, Fz, dz and no moment/shear diagram along a member, only envelopes. A 3D model's weak-axis and torsion results are discarded. | frame.py:459-479; pynite_adapter.py:93-106 | new | not re-checked |
| 4 | B | No member end releases (hinges), so a truss or a pinned-end frame cannot be modeled. Supports are fixed/pinned/roller presets or flags only; no spring supports, settlement or temperature load. | frame.py:368-411 (nothing for releases) | GM-36 parked (P-Delta/plates only; releases not named) | not re-checked |
| 3 | B | A support given both `type` and explicit dx..rz flags uses `type` and drops the flags silently. | frame.py:393-403 | new | not re-checked |
| 3 | B | Member distributed-load direction is upper-cased and then checked against a list that contains "Fx","Fy","Fz", so the local-axis forms can never be reached. Member point loads check no direction at all. | frame.py:420-423, 434-438 | new | not re-checked |
| 5 | B | No calc package, no tables, no figure (moment/shear diagram). `summary()` and `to_dict()` exist on all three result classes. | no pynite entry in calc_package/__init__.py:116-134 or adapters/calc_package.py:3107-3130 | new | not re-checked |
| 6 | B | Tests are exact closed forms (wL/2, wL2/8, 5wL4/384EI, 3-moment results). Good, but they cite no published example. Not covered by any test: trapezoidal loads (w1/w2/x1/x2), `auto_stabilize_2d=False`, 3D (z != 0), nodal loads in a frame, roller_x/free, G/nu/J overrides. | pynite_agent/tests/test_pynite_agent.py:32-154 (21 tests) | new | not re-checked |

### section_props_agent  (scout file `structural_calcpkg.md`)

A 0, B 9, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A to B | The adapter tells the agent `mesh_size` is an "FE mesh element area target ... Auto if omitted" and declares a "150 cells" default for `section_properties`. For the five parametric shapes `analyze_section` never reads `mesh_size` at all; there is no mesh. | section_props_adapter.py:54, 82-88; sections.py:433-486 (mesh_size not used in the body) | new | regraded A to B, quick fix: mesh_size is unused for parametric shapes (section_props_adapter.py:54, 82-88; sections.py:191-245). |
| 3 | A to B | `warping=False` returns `j_mm4 = 0.0` (and `to_dict()` prints it as a number) for every shape. A reader sees a torsion constant of zero, not "not computed". | sections.py:483-486, 388; results.py:627 | new | regraded A to B: j_mm4 0.0 when warping=False (sections.py:146, 242-245; results.py:101); should be absent or flagged. |
| 3 | B | RHS corner radius and I-section root radius are clamped to what fits for the geometry (`_rounded_rect`, `_i_section_ring`), but the unclamped value is handed to the torsion formula. Geometry and J then use different radii, and the result does not say the radius was clamped. | sections.py:291, 314 vs 423, 425-429 | new | torsion side confirmed (section_props_agent/torsion.py:72-90 takes r_out unclamped); the geometry clamp not re-opened. |
| 3 | B | Bredt thin-wall J is used for any RHS wall up to t < half the width, with no warning that thick walls are outside the thin-wall formula. The El Darwish-Johnston fillet term is applied for any r/t ratio with no range check. | torsion.py:72-89, 92-111; sections.py:349-350 | new | not re-checked |
| 3 | B | Polygon J uses 150 cells across the longer side. A thin long outline (a plate 1000 x 10) gets about 1.5 cells across the thickness, with no warning. The grid is also silently coarsened when `max_cells` is hit, which breaks the 2x Richardson step. A sharp inside corner is "the least accurate case" ... | torsion.py:129-166, 187-190; sections.py:517-525 | new | not re-checked |
| 3 | B | Dimensions irrelevant to the shape (e.g. `t` for a rectangle, `r` for an RHS, `r_out` for an I-section) are accepted and ignored with no message; only positivity is checked. | sections.py:466-471 | new | not re-checked |
| 4 | B | Only five parametric shapes. No channel, angle, tee, double angle, and no polygon with holes (a single outer ring only), so any other hollow shape is not possible. No composite or multi-material section. | sections.py:266-272, 514 | GM-36 parked (names RC and warping; shapes not named) | not re-checked |
| 5 | B | No calc package, tables or figure (section drawing). `summary()` and `to_dict()` exist. | not in calc_package/__init__.py:116-134 | new | not re-checked |
| 6 | B | Pinned to the retired library's own outputs (oracles.json) plus Roark Table 10.1 case 4 for rectangle J. No test against a published steel-table shape (e.g. a W section from the AISC shapes database) for I, Z, S, J, Cw. `analyze_section` with `mesh_size` and the adapter's `polygon_section` path are ... | section_props_agent/tests/test_native_engine.py:88-97; test_retired_library_oracle.py | new | not re-checked |
| 1 | A to C | The user manual says the module gives "torsion and warping constants" for rectangles, circles, CHS, RHS, I-sections and any polygon. The warping constant exists only for the I-section (and 0 for circles). Rectangle, RHS and polygon return None, and `to_dict()` leaves the key out. METHOD_INFO ... | docs/user_manual/manual_content.py:1041-1056; sections.py:417-430, 526; results.py:632 | GM-36 | regraded A to C: manual wording (manual_content.py:1040-1056); the engineering gap is GM-36. |

### concrete_props_agent  (scout file `structural_calcpkg.md`)

A 0, B 9, C 3 (C rows are in the scout file), dropped 0, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 4 | B | The diagram is the nominal curve. No 0.80 P0 cap for tied columns and no phi are applied, yet DESIGN.md says the two curves "agree everywhere the diagram is usable" because ACI caps Pn at 0.80 P0. The result neither applies nor mentions that cap. | rc_native.py:223-271; DESIGN.md:217-223; results.py:454 (note about phi only) | GM-36 parked (Mn without phi) | not re-checked |
| 4 | B | The capacity at a given axial load exists in the module (`ultimate_capacity(axial_n=)`) but `analyze_rc_rectangle` and the adapter give no way to ask for it. The only route is to read it off the 24-point diagram. | rc_native.py:182-194; rc_section.py:261-275; concrete_props_adapter.py:52-66 | new | not re-checked |
| 4 | B | No strain check: the neutral-axis depth `c` is computed and thrown away (`[0]`), so the result does not say whether the bottom steel yields, whether the section is tension-controlled (eps_t), or flag over-reinforcement; no min/max steel check. | rc_section.py:341-349 | new | not re-checked |
| 4 | B | Interaction diagram is sagging-side only (`sagging=True`); an unsymmetric section's other bending direction is not given. Cracked Ixx and M_cr are sagging only (labelled so). | rc_native.py:244-249; rc_section.py:333-338 | new | not re-checked |
| 3 | B | Geometry checks cover only the bottom bar and the cover. No check on the top-bar diameter against depth, no bar-spacing or fit-in-width check (20 bars of 32 mm in a 300 mm width runs). A negative `n_top` passes the `if n_top and ...` test and gives a negative steel area. | rc_section.py:309-314, 323, 328-330 | new | not re-checked |
| 1 | B | METHOD_INFO "returns" lists 4 keys and does not name the interaction arrays (`interaction_n_kN`, `interaction_m_kNm`), `ixx_gross_mm4`, `d_eff_mm`, or `note`. The agent is not told how to read the diagram. | concrete_props_adapter.py:67-72; results.py:479-480 | new | not re-checked |
| 4 | B | Rectangular only, one layer per face, no T/flanged beam, no circular column. | DESIGN.md:190-194; manual_content.py:1058-1070 | GM-36 parked | not re-checked |
| 5 | B | No calc package, tables or figure (the N-M diagram is not plotted anywhere). `summary()` and `to_dict()` exist. | not in calc_package/__init__.py:116-134 | new | not re-checked |
| 6 | B | One hand-calc anchor (300x550, 3-28, Mn 398.7 vs 398.46) plus the retired library's numbers. No published textbook example number is cited. Tests do cover top steel and interaction. No test of the dropped-point path or a negative `n_top`. | DESIGN.md:203-213; concrete_props_agent/tests/test_concrete_props_agent.py:74-102 | new | not re-checked |
| 3 | A to C | The interaction diagram drops any point whose neutral axis cannot be solved (`except ValueError: continue`), so fewer than `n_interaction_points` come back, with no note. | rc_native.py:243-247 | new | regraded A to C: dropped points thin the curve but do not move a point (rc_native.py:241-247); say how many were dropped. |

### calc_package (the renderer)  (scout file `structural_calcpkg.md`)

A 0, B 5, C 4 (C rows are in the scout file), dropped 1, matched to a backlog item 2.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 3 | A to B | `generate_calc_package(format=...)` treats any value other than "pdf" or "latex" as HTML (`else:  # html`). A typo such as "docx" or "pdf2" returns an HTML string with no error. `keep_tex` and `compiler` are ignored for HTML. | calc_package/__init__.py:389-398 | new | regraded A to B and folded into X2: any format but pdf/latex renders HTML (calc_package/__init__.py:395; allowed_values declared at adapters/calc_package.py:3173, never enforced). |
| 1 | B | `save_interactive_report` (plotly viewer for slope and fem2d results) has no agent route; it is not in the calc_package adapter and not in DESIGN.md, though CLAUDE.md-style docs describe it as a feature. It accepts only SlopeStabilityResult and FEMResult and raises TypeError for every other module. | calc_package/interactive.py:464-506; grep of funhouse_agent: no caller; DESIGN.md has no "interactive" | new | re-checked: confirmed: no caller in funhouse_agent or aec_agent_core; the manual still offers Plotly viewers (manual_content.py:1838). |
| 1 | B | Retaining walls and pavement "hold only scalars" and have no tables; the SOE package is GM-41. Calc packages exist for no structural module (pynite, section_props, concrete_props). The 17-module map has no calc package for any of the 32 analysis modules beyond these 17 (another scout lists which). | DESIGN.md:186-189; __init__.py:116-134 | GM-41 / GM-39 Done note "retaining walls and pavement hold only scalars" | not re-checked |
| 3 | B | A table that fails to build is reduced to a warnings.warn and `render_report["tables_error"]`; in the agent route the package still returns success with no tables section. By design (DESIGN.md says so), but the adapter result should carry the message. Unverified whether the adapter forwards ... | calc_package/__init__.py:233-243 | new | not re-checked |
| 5 | B | Two renderings of one calculation (HTML template and LaTeX renderer) are separate code (template.py 550 lines, latex_renderer.py 1,837). The equations are converted from HTML text to LaTeX by `equation_converter` (unicode_to_latex). No test seen that proves a step with a new unicode symbol ... | calc_package/equation_converter.py:1-211; template.py | new | not re-checked |
| 1 | A to drop | `_IMPORT_MAP` registers `fem2d` as a calc-package module, and DESIGN.md lists fem2d tables, but the agent has no `fem2d_package` method (the dispatch has 17 generators, none for fem2d; `fem2d` is not mentioned in the adapter). So fem2d's steps, figures and tables are reachable only from Python. | calc_package/__init__.py:133; adapters/calc_package.py:3107-3130 (no fem2d); DESIGN.md:183-186 | new | duplicate of slope_numerical fem2d #1 (counted there). |

### retaining_walls  (scout file `walls.md`)

A 0, B 6, C 6 (C rows are in the scout file), dropped 0, matched to a backlog item 1.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | B | The cantilever tool checks external stability only. It gives no stem, toe or heel moment or shear and no reinforcement design, though it auto-sizes the base, stem and toe. No stem or heel force appears anywhere in the module (zero hits for stem_moment or M_stem). | retaining_walls/cantilever.py (whole file); retaining_walls/DESIGN.md:1-12 | new | not re-checked |
| 2 | B | No seismic case in either wall. `retaining_walls/*.py` has no seismic or Mononobe-Okabe code. Validation V-011 (GEC-11 Example E7, MSE under earthquake, CDR 1.58) passes only because the test does the E7 sliding chain by hand around `seismic_geotech`'s M-O coefficient, so no wall function is pinned ... | validation_examples/test_published_v009_v011.py:752-781; validation_examples/INVENTORY.md:156-164 (Target says "optionally retaining_walls seismic external") | new | not re-checked |
| 3 | B | No global or compound stability, and no settlement check, for either wall. The docs only say global stability takes the live load over both zones (GEC-11 Fig. 4-2); nothing computes it. | retaining_walls/DESIGN.md:415-417 | new | re-checked: confirmed (no global stability code; DESIGN.md:417). |
| 4 | B | MSE facing-connection strength is not checked. GEC-11 internal stability includes it, and `mse.py` has no "connection" at all. Internal stability covers rupture and pullout only. | retaining_walls/mse.py:1924 (`check_internal_stability`); no match for "connection" in the file | new | not re-checked |
| 5 | B | The user manual says MSE internal stability (tension and pullout) is checked "by ASD and LRFD". Internal stability is ASD only (allowable Tal, FOS_pullout 1.5, FOS_rupture 1.0 or 1.5). The LRFD path is external stability only, and its "limits" paragraph says so for external only. | docs/user_manual/manual_content.py:767-769; retaining_walls/mse.py:1924-1935, 4147-4175 | new | not re-checked |
| 7 | B | The ASD MSE tool has fixed required factors of safety (sliding 1.5, overturning 2.0, pullout 1.5). `analyze_mse_wall` and the adapter take no `FOS_*_required`, unlike the cantilever tool. A user who needs another requirement gets a wrong pass/fail verdict. | retaining_walls/mse.py:2498-2515 (`FOS_sliding_req=1.5`, `FOS_overturning_req=2.0`); retaining_walls/mse.py:1924-1935; funhouse_agent/adapters/retaining_walls.py:600-660 (nothing passed) | new | not re-checked |
| 6 | B to C | `retaining_walls/DESIGN.md` lists `list_reinforcement() -> list of built-in reinforcement products` as public API. No such function exists in the module. The list lives only in the adapter's `_REINFORCEMENT_DB`. | retaining_walls/DESIGN.md:29; funhouse_agent/adapters/retaining_walls.py:1191-1201 | new | regraded B to C. DESIGN.md only: no agent or helper reads DESIGN.md, so it misleads builders, not the agent. Quick fix. (DESIGN.md:29). |

### soe  (scout file `walls.md`)

A 2, B 3, C 9 (C rows are in the scout file), dropped 0, matched to a backlog item 4.

| # | Grade (scout to final) | Finding | Evidence | Backlog | Re-check |
|---|---|---|---|---|---|
| 1 | A | `braced_excavation` tells the agent supports take `spacing`, `angle_deg` (and the adapter also accepts `preload_kN_per_m`), and builds them into `SupportLevel`. The analysis never reads spacing, angle or preload. Support loads come out per metre of wall, horizontal, with no conversion for incline ... | funhouse_agent/adapters/soe.py:99-105, 382; soe/geometry.py:38-62; soe/beam_analysis.py:513 (the only support attribute besides depth read); no other hit for `.spacing`, `.angle_deg` or `.preload` in soe/*.py outside the standalone FHWA and anchor functions | new | re-checked: confirmed (adapter soe.py:99-105, told at :382; soe/geometry.py:50-62; nothing in soe/ reads .spacing, .angle_deg or .preload). |
| 2 | A | `check_bottom_blowout` documents `gamma_soil` as the SUBMERGED (buoyant) unit weight, but the agent is told only "Soil unit weight (kN/m3)". Given a total unit weight, FOS is overstated by about gamma/(gamma - 9.81), typically 2x, on the unsafe side. | soe/stability.py:452 against funhouse_agent/adapters/soe.py:400 | new | re-checked: confirmed (soe/stability.py:451-452, 469 vs adapter soe.py:400). |
| 5 | B | The wall analyses do not run basal heave, blowout or piping. They are separate tools with their own inputs, so a braced or cantilever result carries no stability verdict. Calc package `soe_package` covers only cantilever, braced and FHWA-anchored walls. There is no package for the stability checks, ... | funhouse_agent/adapters/calc_package.py:809-825 (the three tools); soe/beam_analysis.py (no call into stability) | new | not re-checked |
| 6 | B | Calc package figures exist for cantilever and braced only; the `fhwa_anchored` package returns no figure. | soe/calc_steps.py:668-673 (`if wt == "fhwa_anchored": return []`) | new | not re-checked |
| 7 | B | No seismic, raker or corner bracing, lagging design, or beam-spring (soil-structure) analysis. These are the module's own Future Work list. SupportLevel's docstring accepts "raker" with no raker geometry. | soe/DESIGN.md:561-569; soe/geometry.py:42-50 | GM-23 open | not re-checked |
| 3 | B to C | `soe/DESIGN.md` Scope lists secant/tangent pile and diaphragm walls ("structural demands only"). No code does this: `select_wall_section` offers only HP, sheet pile and W, and no secant or diaphragm function exists. | soe/DESIGN.md:8-9; funhouse_agent/adapters/soe.py:214-225; soe/wall_sections.py:141-216 | GM-23 open covers "ACI 318 secant / diaphragm walls"; the scope claim itself is new | regraded B to C. DESIGN.md only: no agent or helper reads DESIGN.md, so it misleads builders, not the agent. Quick fix. |
| 4 | B to C | `soe/DESIGN.md` Future Work still says stability checks "basal heave, bottom blowout, piping (Phase 2)" are future. They are built and exposed. The module docstring still says "Embedment depth computation (USACE method)", but the code is the Caltrans free-earth hinge and simplified method (GEC-4 ... | soe/DESIGN.md:561-569; soe/__init__.py:13; soe/embedment.py:1-20 | new | regraded B to C. DESIGN.md only: no agent or helper reads DESIGN.md, so it misleads builders, not the agent. Quick fix. (DESIGN.md:561-569). |

### Calculation examples  (scout file `calc_examples.md`)

- The committed examples are out of date. `docs/examples/` holds two calc package demos (slope LE and FEM strength reduction), last changed for a package on 2026-06-11. The top-level `sample_calc_package.pdf` and `sample_pdfs/` (13 modules) date from 2026-02-17. None of them shows today's layout: given against assumed inputs, choices, calculation tables and drawn figures. pavement_design, soe and reliability have no committed example. (Dates re-checked with `git log -1`.)
- The examples that are kept current are `funhouse_agent/worked_examples.json` (28 published examples, 2026-11-05) and `validation_examples/` (36 `test_published_*.py`). Both are inputs and answers, not finished write-ups.
- The calc helper (funhouse_agent/deep/agent.py:559) is shown no finished package. Its prompt does not mention `worked_examples`, which it can reach. Only the main prompt does (funhouse_agent/deep/prompt.py:177-188, re-checked).

## 5. Quick fixes (no engineering judgment; one Sonnet helper, one branch)

Each item is a wording or document change, or a removal of something dead. Where the engineering fix is a separate item, the quick fix only makes the text true until then.

| # | Where | Fix |
|---|---|---|
| Q1 | wave_equation/DESIGN.md:33, 37-38 | The public API names `analyze_bearing_graph`, `analyze_drivability` and `WaveEqResult`; none exists. List the real exports (wave_equation/__init__.py:20-31) |
| Q2 | sheet_pile/DESIGN.md:14-19 | `analyze_sheet_pile` / `SheetPileResult` do not exist; list the exports in sheet_pile/__init__.py:18-34 |
| Q3 | pile_group/DESIGN.md (Public API section) | `analyze_pile_group` does not exist; list pile_group/__init__.py:44-58 |
| Q4 | ground_improvement/DESIGN.md:26-30 | Four signatures are wrong; copy them from aggregate_piers.py:252, wick_drains.py:367 and surcharge.py:270 |
| Q5 | retaining_walls/DESIGN.md:29 | Remove `list_reinforcement()`, which does not exist |
| Q6 | soe/DESIGN.md:8-9, 561-569; soe/__init__.py:13 | The scope lists secant/tangent pile and diaphragm walls the code does not do; Future Work still lists basal heave, blowout and piping, which are built (soe/embedment.py, soe/stability.py) |
| Q7 | funhouse_agent/adapters/__init__.py:357 | The brief says calc_package has "15 modules"; write the number the registry has, or drop the number |
| Q8 | docs/user_manual/manual_content.py:1255; CLAUDE.md module counts (registry check 5.1) | Make the module counts agree with the registry |
| Q9 | pyproject.toml:276, 279 | Remove the empty `hvsrpy = []` and `swprocess = []` extras (they also remove two empty rows from the manual's extras table) |
| Q10 | HANDOFF.md:589 | Says `.claude/agents/seismic-reviewer.md` lists hvsrpy/swprocess; it no longer does |
| Q11 | Local disk only (not in git): fdm2d/, wind_loads/, DM7Eqs/, hvsrpy_agent/, pyseismosoil_agent/, swprocess_agent/, pygef_agent/, ags4_agent/, pydiggs_agent/, geolysis_agent/ | Delete the ten empty folders (each has 0 files) |
| Q12 | funhouse_agent/adapters/ground_improvement.py:354 | `feasibility` returns: name the keys `to_dict()` gives (ground_improvement/results.py:545-553), not `recommended_methods` / `is_feasible` |
| Q13 | salib_agent adapter descriptions of `n_samples` and `n_trajectories` (salib_agent/sensitivity.py:40-100, 180-206) | Say they are recorded only; the count used is the length of Y / X |
| Q14 | funhouse_agent/adapters/section_props_adapter.py:54 | `mesh_size`: say parametric shapes ignore it (sections.py:191-245) |
| Q15 | pystra adapter, distribution text (~71, ~105) and `n_function_calls` (:92, :129) | Uniform takes `a`/`b`, not mean/stdv (pystra_utils.py:438-446); `n_function_calls` is an estimate (iterations x variables) |
| Q16 | funhouse_agent/adapters/soe.py:400 | Blowout `gamma_soil` is the submerged (buoyant) unit weight (soe/stability.py:451-452) |
| Q17 | funhouse_agent/adapters/drilled_shaft.py:189-190 | Say there is no uplift capacity run; the uplift factors are for hand use (GM-284's note) |
| Q18 | funhouse_agent/adapters/subsurface_adapter.py:1219-1223 | Add `trend_results` and `skipped` to the plot returns |
| Q19 | funhouse_agent/adapters/__init__.py:399-402 | liquepy brief: LPI/LSN/LDI are CPT only |
| Q20 | funhouse_agent/adapters/concrete_props_adapter.py:67-72 | Name `interaction_n_kN`, `interaction_m_kNm`, `ixx_gross_mm4`, `d_eff_mm` and `note` in the returns (not re-checked) |
| Q21 | funhouse_agent/adapters/pynite_adapter.py:21 | `rho`: say no self-weight is applied; add it as a load |
| Q22 | docs/user_manual/manual_content.py:1838 | The "Plotly interactive viewers" have no agent route; remove the claim unless a route is added (see the X5 item) |

## 6. Plan for stage 2

Rules carried into every brief: the rule of 2026-10-27 against chasing edge cases (a follow-up is an item only if it changes a saved report's output); made-up test values only; full suites on GitHub Actions (`ci/<name>`), with local pytest only for the files a change touches; never two heavy suites at once; `model` set on every helper; slope_stability stays untouched until GM-140.

| Wave | What | Model | Size | Yardstick |
|---|---|---|---|---|
| 0 | Quick fixes Q1-Q22 | Sonnet builder, Opus reviews the diff | one branch, about 25 edits | each claim in §5 true after the edit; `test_public_text`, `test_docs_currency` |
| 1 | Cross-cutting: X1 (warnings into results, in aec-agent-core, so a core release), X2 (`allowed_values` enforced), and the standing test below (T1) | Opus builder, independent Opus review | three branches, one at a time in method_call.py | T1 and T2 green; every probe in funhouse_agent/tests/method_probes.json (251) replays unchanged except where a refusal is now expected |
| 2a | Piles: profile shorter than the pile refused or stated (axial_pile, drilled_shaft, lateral_pile); `capacity_vs_depth` names failed depths; cracked-EI convergence flag; pile_group's three A findings | Opus | one helper | the A findings in §4 closed; recorded inputs replayed |
| 2b | Shallow and walls: settlement's two A findings; soe supports (read `spacing`, `angle_deg` and `preload`, or refuse them); ground_improvement returns | Opus | one helper | same |
| 2c | Reliability: reliability, pystra and salib (see the item) | Opus | one helper | same, plus FOSM/PEM/MC agreement on a made-up truncated case |
| 2d | Seismic: liquepy returns and one total-stress rule; pystrata effective stress; seismic_signals failures as None with a reason | Opus | one helper | same |
| 2e | Site data: subsurface trends and GEF gaps; dxf_import units and defaults | Opus | one helper | same |
| 3 | Print checks for the modules never checked against their references (table below), with the references read through the library | Opus with the printed pages | one helper per row, in rank order | the manual's method step by step and case by case against the code; the manual's worked examples run through the module and pinned in validation_examples |
| 4 | Calc package gaps: fem2d_package, settlement's separate methods, pystrata, the structural engines; examples regenerated from made-up inputs; the calc helper told about worked_examples | Sonnet builds from the existing package template, Opus reviews | after wave 1 | T1's calc-package clause green |

Waves 2a-2e touch different modules and can run in parallel. Waves 0 and 1 share files in adapters and method_call.py, so run them one after the other.

**Standing tests** (the REF-92 analogue for the calculation modules):

- **T1. Everything the agent is told about is real.** One test walks the registry. For every method with a probe in method_probes.json, it runs the probe and checks four things. (a) Every key in METHOD_INFO `returns` is in the result, which closes X3. (b) Every documented parameter is read: change it to another valid value, and the result must change or the call must be refused. `test_every_default_a_run_lists_changes_its_result` (funhouse_agent/tests/test_defaults_used.py:1030) already does this for defaults; extend it to all documented parameters. It would have caught soe supports, the salib counts, `mesh_size` and pile_group's simple batter. (c) A value outside `allowed_values` is refused (X2). (d) Every module in calc_package's `_IMPORT_MAP` has a `*_package` method or a stated reason. Counts stated in briefs equal the registry's count.
- **T2. No warning is lost.** Run the probes with module warnings recorded: any warning raised during a call must appear in that call's result (X1).
- **T3. DESIGN.md public API names exist.** Parse the code spans in each DESIGN.md "Public API" section and check each name by AST against the package `__init__` (no import). This closes X4 for DESIGN files.
- **T4. No swallowed failures in module code.** An AST scan for `except Exception:` followed by `pass`, `continue`, or a return of `{}`, `[]` or `0.0` in the analysis packages. Today's hits are listed as allowed, each with a reason, and any new one fails the test. It would have caught axial_pile #1, pystra #5, seismic_signals #1 and concrete_props' dropped points.

**References each module cites, and what is left for a print check.** Counts are mentions in DESIGN.md, the package docstring and VALIDATION.md. "Checked" means a GM-281 to GM-298 print check is done.

| Module | Main references cited | Print check | Stage-2 print check (wave 3) |
|---|---|---|---|
| reliability | Duncan (13), Rosenblueth (7), UFC 3-220-20 (7), ETL 1110-2-547 (3) | none | FOSM/PEM/MC steps as printed; truncation; the ETL's worked example |
| pystra_agent, salib_agent, gstools_agent | almost none cited | none | pystra: FORM/SORM against a published example (e.g. a textbook limit state); salib: Ishigami against the published indices; gstools: one kriging example |
| liquepy_agent | Boulanger & Idriss 2014, Youd et al. 2001, Zhang et al., Robertson | seismic_geotech side in GM-293 | B&I 2014 CPT and SPT procedures step by step; LPI/LSN/LDI definitions; the B&I worked example |
| pystrata_agent | Darendeli (7), Menq (5) | none | Darendeli/Menq mean effective stress and curves; one published site-response example |
| opensees_agent | Newmark, PM4Sand, Darendeli, Kramer | none | PM4Sand DSS against the PM4Sand manual's element tests |
| seismic_signals_agent | none cited | none | bracketed duration, Arias intensity and RotD definitions against their papers |
| subsurface_characterization | DIGGS, AGS4, GEF format specs | none | reader behaviour against the format specs (missing values, units) |
| dxf_import, dxf_export | none | none | units against the DXF $INSUNITS table |
| section_props_agent | AISC (2), Roark, Bredt | none | J, Cw, Z and S against the AISC shapes database for a W, HSS and pipe |
| concrete_props_agent | ACI 318 (7) | none | flexure, strain limits and interaction against an ACI textbook example |
| pynite_agent | none cited | none | a textbook frame example |
| fem2d | Griffiths (19), Bishop, Duncan, GEC-4, GEC-6 | GM-308 open | Griffiths & Lane strength reduction examples (after GM-140 for its slope side) |
| sheet_pile | Caltrans (8), UFC 3-220-20 (7), EM 1110-2-2504 (6), USS (3) | GM-290 | minimum active pressure and water in the tension crack as printed (the soe side matched the crack depth in GM-290) |
| drilled_shaft | GEC-10 (35), FHWA-NHI-18-024 | GM-284 | uplift run (GM-284 notes there is none) only if the owner wants it |
| bearing_capacity, settlement, ground_improvement, pavement_design, axial_pile, downdrag, pile_group, lateral_pile, wave_equation, retaining_walls, soe, seismic_geotech, soil correlations | as their DESIGN.md | GM-281 to GM-298 | none in stage 2; interface fixes only (waves 2a-2e) |
| slope_stability | Bishop, Spencer, Slide2, Duncan, Janbu | held | waits on GM-140 |

## 7. Not checked

- **Engineering against the print.** Neither the scouts nor I judged whether a method matches its printed reference; that is wave 3. The sheet_pile crack pressure and the shaft uplift factors were re-graded on that basis and still need the print.
- **Most B and all C findings.** 106 of the 148 scout B findings were not re-checked, nor were any of the C findings. Structural_calcpkg is weakest: 2 of 27 B re-checked, after 0 of 7 of its A findings held at A.
- **Nothing was run.** There was no pytest, no import of the package and no tool call, so behaviour is read from the code. Line numbers marked `~` (pystra, liquepy adapter) are approximate.
- **The warning count.** "About 50 `warnings.warn` in about 25 files" comes from a grep. Which of them matter to a saved report was not judged one by one.
- **Not opened at all:** the webapp, the review harness, prompts other than the main and calc ones, the user manual beyond the lines cited, the Foundry hosts, `geotech-references/`, `packages/aec-document-review/`, `aec_agent_core/review_eval/`, planlens.
- **slope_stability** was read for the record only (hold GM-140). No item is proposed for it.
