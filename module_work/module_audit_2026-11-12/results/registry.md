# Registry cross-check: told versus reachable

Script: `scratchpad/modaudit/registry_check.py` (standard library only, parses and never imports; run with the repo venv). Reference-library adapters are skipped as instructed (34 of the 70 registry entries; the set equals `REFERENCE_MODULES` in `funhouse_agent/dispatch.py:65`). In scope: 36 entries (`ANALYSIS_MODULES`), 194 documented methods, 195 dispatchable.

Overall: the registry is in good shape. No confirmed A finding. The agent's catalog text (`system_prompt.py:8`, `deep/prompt.py:191-205`) is built from the registry at run time, so it cannot drift. The drift is in the human-facing documents and in leftover folders.

## 1. Registry against adapter files

| # | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|
| 1.1 | C | Every one of the 70 registry entries names an adapter file that exists; every `adapters/*.py` file (apart from `__init__.py` and `_reference_common.py`) is in the registry. No gap either way. | `funhouse_agent/adapters/__init__.py:207-` | n/a |
| 1.2 | C | `pynite` is dropped from the registry at import time when PyNiteFEA is missing; the module is then absent for the agent and the catalog (by design, INF-53). Registry count is 70 only where PyNiteFEA is installed. | `adapters/__init__.py:570-592` | known (INF-53 in code comment) |

## 2. METHOD_INFO against the dispatch dict

| # | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|
| 2.1 | C | After handling later assignments, all 36 adapters have the same method names in `METHOD_INFO` and `METHOD_REGISTRY`, both directions. The script's one hit, `downdrag|post_liquefaction_downdrag` "dispatchable, not documented", is a script limit: it is documented by a later assignment. | `adapters/downdrag.py:676`, `adapters/downdrag.py:879` | n/a |
| 2.2 | C | Counts: bearing_capacity 2, settlement 5, slope_stability 12, seismic_geotech 7, liquefaction 1, retaining_walls 5, axial_pile 3, drilled_shaft 4, sheet_pile 2, lateral_pile 3, pile_group 5, ground_improvement 7, wave_equation 4, downdrag 9, soe 9, pavement_design 9, dxf_export 2, calc_package 20, profile_figure 2, worked_examples 2, pystra 3, reliability 7, salib 6, dxf_import 4, pdf_import 9, drawing_ir 5, opensees 2, pystrata 2, liquepy 3, seismic_signals 4, fem2d 11, gstools 4, section_props 2, concrete_props 1, pynite 2, subsurface 17. Total 195. Many modules extend `METHOD_INFO` after the literal (loops, `.update`, `.pop`), so any outside reader that parses the literal alone sees wrong parameter lists (fem2d, axial_pile, downdrag, slope yield_acceleration, profile_figure). | `fem2d_adapter.py:1018-1032`, `axial_pile.py:410`, `slope_stability.py` (`METHOD_INFO["yield_acceleration"]["parameters"].pop("kh")`) | n/a |

## 3. Documented parameters against the names the dispatch function reads

Method: for each method the script follows same-file helper calls, collects every string the code reads from the params dict (subscripts, `.get`, `require_params`), and every string constant in the reachable code, then compares with the documented names. Documentation built by spreads, helper dicts and later mutation is resolved where it is a plain dict.

| # | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|
| 3.1 | C | No confirmed "documented, never read" and no confirmed "read, never documented" in any of the 36 modules. Every script hit I opened was a false positive (list below). | see below | n/a |
| 3.2 | C | Pass-through methods ("passes everything through, not checked"): ground_improvement (6 methods: aggregate_piers, wick_drains, design_drain_spacing, surcharge_preloading, vibro_compaction, feasibility), pavement_design (8 methods: compare_pavement_methods, design_traffic_esals, flexible_pavement_design, flexible_pavement_design_ufc, mixed_traffic_ufc, performance_period, rigid_pavement_design, rigid_pavement_design_ufc), calc_package.slope_report_package. For ground_improvement I checked by hand: the adapter validates against the called function's own signature (`_run`, `ground_improvement.py:20-34`); documented names against the six functions' signatures give no mismatch except `area_replacement_ratio` (converted to spacing, `ground_improvement.py:115-124`), and the one signature name not documented, `gwt_depth_m`, is deliberately refused (`ground_improvement.py:182-184`). The pavement_design and slope_report_package pass-throughs are not checked. | `adapters/ground_improvement.py:20-45`, `:107-129`, `:181-184` | n/a |
| 3.3 | C | 57 of 195 methods never call `reject_unknown_params` in their own reachable code (all calc_package `*_package` and render methods, subsurface, salib, pystra, opensees, pystrata, liquepy, seismic_signals, worked_examples, profile_figure.plot_data, liquefaction_analysis, three small list/lookup methods). Their unknown names are caught at dispatch from `METHOD_INFO`, and the runtime guard in 6.2 covers that. Static check for them rests on item 3.1 only. | `registry_check.py` output list | n/a |

Script hits not confirmed (false positives, count only): 198 table rows after filtering plus 78 more removed by the "name appears after METHOD_INFO" filter (276 in all). Causes I confirmed: documentation added after the literal (fem2d `gamma_w`, `tol`, `nx`... at `fem2d_adapter.py:1018-1032`; axial_pile `_SECTION_INFO` at `axial_pile.py:316,410`; soe `_ANCHOR_INFO` at `soe.py:347,405`; downdrag `ground_ele­vation` at `downdrag.py:917`); aliases (`lateral_pile free_length` -> `stickup`, `lateral_pile.py:152,313`; `wave_equation R_ultimate` -> `R_total`, `wave_equation.py:277-278,386`); a refused-by-design parameter (`pile_group_simple` ignores Vx, Vy, Mz, stiffness: `pile_group.py:81`); nested keys of a model spec (`salib_adapter.py:45-46,135-140`); a shared reader (`output_format` via `figure_output_format`, `subsurface_adapter.py:232`; `attachment_key` is always accepted at dispatch, `test_top_level_params.py:140`); calc_package common fields in `_COMMON_PARAMS` merged at run time (`calc_package.py:1522,3135`).

## 4. Module folders at the repo root against the registry

Folders with `__init__.py` and no registry entry:

| # | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|
| 4.1 | C | `geo_project` (15 py files, staged human-gated model setup) has no registry entry. It is reached by the deep agent's setup tools, not through `call_agent`: 12 imports in `funhouse_agent/deep/setup_tools.py` (lines 40, 41, 401, 413 and others). | `funhouse_agent/deep/setup_tools.py:40` | new |
| 4.2 | C | `geotech_common` has no registry entry by design (shared library); 11 adapters import it (41 import lines under `funhouse_agent/`). | `funhouse_agent/dispatch.py:1055`, `adapters/axial_pile.py:178` | n/a |
| 4.3 | B | `foundry/` (26 py files, per-module Foundry wrappers) and `foundry_test_harness/` (7 files) have no registry entry and nothing imports them. The two `from foundry.transforms import Dataset` lines in `webapp/foundry_dataset_store.py:154,355` name the platform's own package: the local `foundry/` has no `transforms`. Both folders are excluded from the wheel (`pyproject.toml:352,354`) but `foundry_test_harness` still sits in pytest `testpaths` (`pyproject.toml:414`). CLAUDE.md says "46 agents" for these (32 + 14). | `pyproject.toml:352-354,414` | known INF-06 [Later] |
| 4.4 | C | Nine of the twelve folders you expected to be unregistered are not modules: `wind_loads`, `fdm2d`, `geolysis_agent`, `hvsrpy_agent`, `pyseismosoil_agent`, `swprocess_agent`, `pygef_agent`, `ags4_agent`, `pydiggs_agent` hold only an empty `tests/` folder (0 files, no `__init__.py`); `DM7Eqs` holds only empty `geotech/dm7_1`, `geotech/dm7_2`, `tests`. They are leftovers of removed modules (pyproject.toml:276,279 mark hvsrpy and swprocess "REMOVED 2026-09"). No import of any of them under `funhouse_agent/`, `webapp/`, `report_ingest/`. | directory listing under `C:\dev\GeotechStaffEngineer` | new |
| 4.5 | C | Registry names differ from folder names for 12 modules (`concrete_props` -> `concrete_props_agent`, `liquepy` -> `liquepy_agent`, `subsurface` -> `subsurface_characterization`, and so on); `liquefaction` has no folder at all (its adapter calls `seismic_geotech` and `liquepy_agent`). `pdf_import` and `drawing_ir` have no folder either: `pdf_import` is served by `dxf_import` code, `drawing_ir` by planlens. | `registry_check.py` roots table | n/a |

## 5. Places that tell a person or the model what modules exist

| # | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|
| 5.1 | B | Stated count of analysis modules disagrees everywhere. CLAUDE.md says 32 (line 70, 1652), README 30 (lines 5, 72), `docs/overview.html` 30 (line 292), CLAUDE.md 30 (line 2443). The registry has 36 `ANALYSIS_MODULES`, or 29 once the seven tool-like entries (calc_package, profile_figure, worked_examples, dxf_export, dxf_import, pdf_import, drawing_ir) are set aside. No reading gives 30 or 32. | `CLAUDE.md:70`, `README.md:5`, `docs/overview.html:292`, `funhouse_agent/dispatch.py:81` | new |
| 5.2 | B | The registry's own description of calc_package says "15 modules": `adapters/__init__.py:357`. The code has 16 `*_package` modules and 17 package methods (slope_stability has two: `slope_stability_package` and `slope_report_package`): `calc_package.py:3107-3127`. The manual says "17 analysis modules" (`manual_content.py:1255`, `:1838`). Three numbers for one thing; the first one is the text the model sees. | `adapters/__init__.py:357`, `calc_package.py:3107-3127`, `docs/user_manual/manual_content.py:1255` | new (DOC-23 Done checked module bullets, not this count) |
| 5.3 | C | CLAUDE.md reference-layer counts are stale: "16 reference adapters" (lines 2352, 2366), "21 reference modules" (2426), "Reference layer is now 30 modules" (2497), "system_prompt (50 modules)" (2361). The registry has 34 reference entries (32 library adapters plus `reference_db` and `figure_db`); `system_prompt.py:8` counts `len(MODULE_REGISTRY)` = 70 at run time. | `CLAUDE.md:2352,2361,2366,2426,2497`, `dispatch.py:65-79` | new |
| 5.4 | C | README names 28 of the registry's keys; it names none of: pavement_design, worked_examples, pdf_import, drawing_ir, seismic_signals, section_props, concrete_props, pynite. `overview.html` omits 13 (pavement_design, profile_figure, worked_examples, pystra, salib, pdf_import, drawing_ir, liquepy, gstools, section_props, concrete_props, pynite, subsurface). `manual_content.py` omits by exact key: pavement_design, profile_figure, drawing_ir (prose may still describe them). | `registry_check.py` section 5 | new |
| 5.5 | C | CLAUDE.md's module table (rows from line 2200) has no row for `calc_package`, `profile_figure`, `worked_examples` or `liquefaction` as such (all four are named elsewhere in the file). `pdf_import` and `drawing_ir` rows point to planlens as "historical" (`CLAUDE.md:2230-2231`) while the registry still serves 9 and 5 methods through adapters. | `CLAUDE.md:2200-2234` | new |
| 5.6 | C | `docs/overview.html` and README draw `geo_project` as an analysis box ("30 modules ... No cross-imports") though it is not in the registry. | `README.md:63,166`, `docs/overview.html:272` | new |
| 5.7 | C | `deep/prompt.py` and `system_prompt.py` carry no static list of modules; the catalog is generated from the registry (`system_prompt.py:198-205`, `deep/prompt.py:191-205`). No drift possible; the 26 "missing" names the script printed for `prompt.py` are a false alarm. | `funhouse_agent/system_prompt.py:198-205` | n/a |
| 5.8 | C | The dispatch scoping sets (`SEISMIC_`, `FOUNDATIONS_`, `EARTH_RETENTION_`, `SLOPE_FEM_`, `PAVEMENT_`, `STRUCTURAL_` modules and references, `DELIVERABLE_MODULES`) all name only keys that exist in the registry, and `REFERENCE_MODULES` (34 names) equals the set of reference entries. Checked by parsing. | `funhouse_agent/dispatch.py:65,98-250` | n/a |
| 5.9 | C | CLAUDE.md names an `fdm2d` wall-clock guard "added here too" while no `fdm2d` code exists (the folder is empty). Unverified beyond the line. | `CLAUDE.md:2504` | new |

Names in README, overview.html, manual_content.py for `*_agent`, `gec*`, `ufc_*`, `em_*`, `dm7`, `eurocode*` that the registry lacks: none.

## 6. Existing tests that guard 1-5

(Read only, not run.)

| # | Test | What it covers | What it does not cover |
|---|---|---|---|
| 6.1 | `funhouse_agent/tests/test_new_adapters.py` (`assert_method_info_complete`), `test_phase34_adapters.py` (`test_keys_match`, `test_method_count == 11` for fem2d), `test_structural_adapters.py`, `test_reliability_adapter.py` | Check 2 for opensees, pystrata, fem2d, gstools and a few others, by name and required fields. | No test walks every registry entry's `METHOD_INFO` keys against `METHOD_REGISTRY` keys. Hand-listed modules only (and fem2d's count is a hard-coded 11). |
| 6.2 | `funhouse_agent/tests/test_top_level_params.py`: `test_adapter_own_checks_agree_with_method_info` (line 175), `test_method_info_declares_only_what_the_adapter_accepts` (line 220), `test_every_declared_parameter_and_alias_is_accepted` (108), `test_every_alias_names_a_parameter_of_its_method` (124), `test_metadata_keys_are_known_shapes` (132) | Check 3, at run time over `D.MODULE_REGISTRY`: both directions between `METHOD_INFO` and what each handler's own `reject_unknown_params` accepts, with probes from `method_probes.json` (guards GM-150/GM-158); skips methods marked `open_parameters`. Only methods that call `reject_unknown_params`. | The 57 methods with no such call (3.3); methods in `method_probes.EXCLUDED` (file readers, needs-handle, optional backends); parameters read but never passed to the module (a documented name accepted by the check and then dropped); pass-through methods. |
| 6.3 | `funhouse_agent/tests/test_adapter_ergonomics.py` (`TestAllowedValuesCoverage`, line 74-110) | `allowed_values` present and well-formed across the registry. | Nothing about parameters or methods matching. |
| 6.4 | `funhouse_agent/tests/test_catalog_routing_hints.py` (lines 14-47) | Every registry module has a brief under a size budget; named modules keep a routing phrase. | Whether a brief's counts or claims are true (5.2 slips past it). |
| 6.5 | `funhouse_agent/tests/test_calc_package_tool_parity.py`, `test_defaults_used.py` (line 75 walks `calc_package.METHOD_INFO`) | Calc packages follow their tool's parameters; `defaults_used` rules. | Count of packages. |
| 6.6 | `funhouse_agent/tests/test_cited_documents.py` (line 169), `test_reference_adapters.py` | Reference modules through `REFERENCE_MODULES`. | Registry against `REFERENCE_MODULES` equality (checked above by hand, not by test). |
| 6.7 | `webapp/tests/test_docs_currency.py` | Version names in CLAUDE.md state, HANDOFF, install guide only. | No stated module count, method count or module name in any document. |
| 6.8 | None found | Check 1 (adapter file against registry), check 4 (folders against registry), check 5 (counts and names in README, overview.html, manual, CLAUDE.md). | All of it. |

## Not checked

- Reference-library adapters (skipped as instructed).
- Whether a documented parameter that is read reaches the module's own function correctly (that is the per-module scouts' job; this check only compares names).
- pavement_design and slope_report_package pass-through methods against the called functions' signatures.
- Return descriptions against `to_dict()` keys.
- Method counts per document (the "165 methods", "407 dispatch methods", "90 wrapped methods" figures in CLAUDE.md are in other contexts: probe table, Foundry catalog, a capability audit; not compared).
- `docs/overview.html` was checked by key name only; its SVG text for module labels may use display names.
