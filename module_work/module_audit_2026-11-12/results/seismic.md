# seismic

Paths are under C:\dev\GeotechStaffEngineer. Read-only scout pass; nothing was run or imported.

## seismic_geotech
Size: 8 python files plus calc_steps.py (15 with tests), 11 names in `__all__` (the six `dynamic_properties` functions are imported in `__init__` but not listed), 7 adapter methods (site_classification, gmax, modulus_reduction, seismic_earth_pressure, liquefaction_evaluation, residual_strength, csr_crr_check). Site class and Mononobe-Okabe are pinned to printed pages (GEC-12 print, GEC-11 E7, Whitman Fig. 11). The liquefaction and residual-strength edges are loose.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 3 | B | The NCEER liquefaction route never applies K_sigma; CRR is the clean-sand curve at every depth, although Youd 2001 reduces CRR where sigma'v is above 100 kPa (unconservative for deep layers). The result's `choices` says so, but nothing flags which layers it affects. | seismic_geotech/liquefaction.py:187-193, 326-328 | new |
| 2 | 3 | B | Zero or negative effective stress is handled two different ways. `evaluate_liquefaction` sets it to 1.0 kPa (huge CSR). `compute_CSR` returns 0 and `csr_crr_check` then reports FOS 99.9 and liquefiable = False. Neither says it happened. | seismic_geotech/liquefaction.py:116-117, 271-272; funhouse_agent/adapters/seismic_geotech.py:233 | new |
| 3 | 3 | B | `post_liquefaction_strength` with Seed & Harder returns a flat 35 kPa for any (N1)60cs of 24 or more, and 0 for N of 0 or less. No warning in the result or the adapter. | seismic_geotech/residual_strength.py:49-52; adapters/seismic_geotech.py:196-209 | new |
| 4 | 3 | B | `site_coefficients` holds the end value beyond the table (Ss above 1.25, S1 above 0.5, PGA above 0.5 or below 0.1) and says nothing in the result. | seismic_geotech/site_class.py:304-309 | new |
| 5 | 2 | B | Class F and the soft-clay rule for Class E are not checked. The adapter text says so, but a profile with soft clay or peat still returns an A-E class. | seismic_geotech/site_class.py:178-180, 187-210; adapters/seismic_geotech.py:348 | new |
| 6 | 3 | C | When vs30, n_bar and su_bar are all given, vs30 silently decides the class. The result does not name the criterion used. | seismic_geotech/site_class.py:187-210; adapters/seismic_geotech.py:52, 64 | new |
| 7 | 3 | C | CRR is clamped to 2.0 at (N1)60cs of 30 or more, and N below 0 is taken as 0. `csr_crr_check` returns no note of either. The cap appears in `choices` only for `liquefaction_evaluation`. | seismic_geotech/liquefaction.py:181-185; adapters/seismic_geotech.py:232-234 | new |
| 8 | 1 | B | The adapter says "Unit weights (kN/m3)" for `gamma` and never tells the agent that each value is taken to apply over the interval ABOVE its depth. The B&I SPT route in liquepy uses gamma times depth instead. The same inputs give different overburden by route (see liquepy #3). | adapters/seismic_geotech.py:375; seismic_geotech/liquefaction.py:261 | new |
| 9 | 1 | C | DESIGN.md still says that without `pga`, Fpga is approximately Fa(Ss). The code now returns None (GM-293). | seismic_geotech/DESIGN.md:57-60 vs site_class.py:279-283 | new |
| 10 | 1 | C | `METHOD_INFO.returns` omit keys the tools return: `layer_results` and `magnitude` for liquefaction_evaluation; rd, MSF, N160cs and liquefiable for csr_crr_check; `K`, `m`, `n` for gmax and the other extras. | adapters/seismic_geotech.py:380, 405; 192-193, 234 | new |
| 11 | 5 | B | A calc package exists only for site classification, M-O and `liquefaction_evaluation` (NCEER). Nothing for gmax, modulus_reduction or residual_strength. Those return a bare float or dict with no result class, no `summary()`/`to_dict()`, no tables. | funhouse_agent/adapters/calc_package.py:589-600; seismic_geotech/residual_strength.py:213 | new |
| 12 | 5 | C | `get_tables` exists only on LiquefactionResult. SiteClassResult and SeismicEarthPressureResult have none. | seismic_geotech/results.py:289 (the only hit) | new |
| 13 | 6 | B | There is no published liquefaction example behind the module. INVENTORY itself says none was found. Liquefaction tests assert the code's own values, and the CRR/CSR tests are not tied to a page. | validation_examples/INVENTORY.md:1177; seismic_geotech/tests/test_seismic_geotech.py:572-573 | new (REF-35 is only the SPT-strain option) |
| 14 | 1 | C | `__all__` leaves out the six `dynamic_properties` functions that `__init__` imports. The adapter reaches them, so this only misleads a reader of the package. | seismic_geotech/__init__.py:22-37 | new |

Not checked: calc_steps.py (1,189 lines) beyond its function list; whether `funhouse_agent/deep/prompt.py` or `manual_content.py` claim more than the adapter does (the manual only lists module names, manual_content.py:126-130, 200-201); the printed values themselves (a later stage).

## liquepy_agent
Size: 6 python files (11 with tests), 11 names in `__all__`, 3 adapter methods in liquepy_adapter plus the unified `liquefaction_analysis` (liquefaction_adapter.py, 1 method). The CPT route wraps liquepy's own B&I-2014 run. The SPT route is the module's own composition. Two CPT paths and two SPT paths give different outputs, and the descriptions do not match the result classes in two places.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 1 | A | The CPT tool is described as returning `n_liquefiable` ("Number of liquefiable points"). The CPT result has no such key, in `to_dict()` or the class. Both CPT tools carry the false claim. | funhouse_agent/adapters/liquepy_adapter.py:130; liquefaction_adapter.py:376; liquepy_agent/results.py:107-120 | new |
| 2 | 1 | A | `field_correlations` is described as returning the Vs, Dr, su/sigma'v and I_c PROFILES. `to_dict()` gives only n_points, gwl, vs_method and Vs min/max/average and Dr min/max. The su ratio, permeability and I_c are never returned, and nor are the profiles. | adapters/liquepy_adapter.py:213-218; liquepy_agent/results.py:397-415 | new |
| 3 | 3 | B | The B&I SPT route takes total stress as gamma times depth (each point's own gamma over the whole depth). The NCEER route integrates through the layers above. The adapter describes `gamma` the same way for both, so changing `method` changes the overburden as well as the procedure. `choices` states it. | liquepy_agent/spt_liquefaction.py:167-169, 238-242; seismic_geotech/liquefaction.py:261 | new |
| 4 | 3 | B | CPT `min_fos` is taken over liquefiable points (below the water table, Ic under the limit). When there are none it falls back to the minimum over ALL points. The value then comes from soil the result itself calls non-liquefiable, with no flag. | liquepy_agent/cpt_liquefaction.py:374-379 | new |
| 5 | 1 | B | The unified tool accepts `N` and `n_values` as spellings of (N1)60. A raw field N goes into a procedure that wants the corrected value, without a word. | funhouse_agent/adapters/liquefaction_adapter.py:34, 188, 217; declare_aliases 345-346 | new |
| 6 | 1 | B | The registry brief says "B&I 2014 CPT/SPT triggering, LPI/LSN/LDI". LPI, LSN, LDI, strain and settlement are CPT only. The SPT routes return layer FS only. | funhouse_agent/adapters/__init__.py:401; liquefaction_adapter.py:371 | REF-35 open (SPT strain option only) |
| 7 | 3 | C | The SPT route clamps effective stress to 1.0 kPa and clips FC to 0-100 without saying so. | liquepy_agent/spt_liquefaction.py:58, 174 | new |
| 8 | 1 | C | There are two CPT tools with different outputs. The unified one returns `depth_profile` (FS and strain by depth, for downdrag). liquepy's `cpt_liquefaction` does not. `max_settlement_mm` is returned but described nowhere. `p_a` and `gamma_w` are never forwarded on the SPT routes. | adapters/liquepy_adapter.py:35-55 vs liquefaction_adapter.py:176-185; liquepy_agent/results.py:107-118 | new |
| 9 | 5 | B | There is no calc package, no `get_tables` and no calc_steps for any liquepy result. B&I-2014 is the default for both SPT and CPT, yet the only liquefaction calc package is the NCEER one in `seismic_package`. | grep `get_tables` in liquepy_agent/results.py: 0 hits; adapters/calc_package.py:589-600 | new |
| 10 | 6 | B | The SPT test called "published" checks CRR within +/-0.04 of about 0.29 (a rounded number, not a page). INVENTORY has no liquefaction example. CPT results depend on the installed liquepy and are tested against its own output. | liquepy_agent/tests/test_spt_liquefaction.py:119-129; validation_examples/INVENTORY.md:1177 | new |
| 11 | 3 | C | CPT LDI is integrated over the whole profile; `z_max` is never passed. Unverified against the Zhang print. | liquepy_agent/cpt_liquefaction.py:239-247, 372 | new |
| 12 | 6 | C | `bi2014_spt_fines_correction` appears in one test file only. The `depth_profile` banding is well tested (12 tests). | liquepy_agent/tests (grep) | new |

Not checked: B&I equations against the print; liquepy's own internals (not imported); the `choices` text in test_choices.py beyond the headings.

## pystrata_agent
Size: 4 python files plus compat.py and utils (8 with results), 5 names in `__all__`, 2 adapter methods (eql_site_response, linear_site_response). Compat across pystrata 0.5/0.6 is careful. The tool hands back much less than the module computes, and the confining stress ignores the water table.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 3 | A | `stress_mean` for Darendeli and Menq layers is documented as "mean effective stress" but is computed from TOTAL stress (sum of unit_wt times thickness), K0 fixed at 0.5. There is no water table input anywhere in the tool. Below the water table the curves are built for too high a confinement. The adapter says only "default from depth", and the result has no `choices`. | pystrata_agent/eql_site_response.py:181-205; adapters/pystrata_adapter.py:96 | new |
| 2 | 1 | B | The module computes the surface and input response spectra (5 %), the surface time history, and strain, acceleration and strain-compatible Vs by depth. `to_dict()` and `METHOD_INFO.returns` carry only scalars (PGA, amplification, iterations, peak strain). The agent cannot get a site-response spectrum, which is the usual product. OpenSees' tool does return one. | pystrata_agent/results.py:109-124; eql_site_response.py:482-500; adapters/pystrata_adapter.py:130-141 | new |
| 3 | 3 | B | The motion is applied as a rock OUTCROP motion (`location("outcrop", index=-1)`). The pystrata adapter never says so. OpenSees' says "taken as a rock outcrop motion". | pystrata_agent/eql_site_response.py:479-480; adapters/opensees_adapter.py (motion param) | new |
| 4 | 4 | B | Only two built-in motions exist, both synthetic. A comment marks real records as "future". No adapter method lists the motions (`list_motions` is not reachable), so the agent learns the names only from description text. | opensees_agent/ground_motions.py:124-137, 177; grep `list_motions` in funhouse_agent/adapters: no hit | new |
| 5 | 3 | C | If both `motion` and `accel_history` are given, `motion` wins and the history is dropped without a message. The same function serves every motion-taking tool in the group. | opensees_agent/ground_motions.py:216-226 | new |
| 6 | 1 | C | DESIGN.md names `pystrata_agent_foundry.py` at the project root (it is under foundry/). The module imports `opensees_agent.ground_motions`, against the "no cross-module imports" rule. | pystrata_agent/DESIGN.md:23; eql_site_response.py:337, 393 | foundry deletion INF-06 open |
| 7 | 5 | B | No calc package, no tables, no figure for site response. `plot_*` methods exist but no adapter or package calls them. | grep `get_tables`/`_package` for pystrata: none | new |
| 8 | 6 | B | The tests carry no published example or closed-form check (uniform layer over rock). HANDOFF records a pystrata-vs-OpenSees cross-check (0.749 vs 0.758 g), but no pystrata test cites it. | pystrata_agent/tests/test_pystrata_agent.py (grep: none); HANDOFF.md:685 | new |
| 9 | 3 | C | 0.6 clips interpolated G/Gmax to (0.001, 1) and damping to (0, 0.49), and 0.5 does not. This matters only for custom curves outside those ranges, and the result does not say which pystrata ran. | pystrata_agent/compat.py:26-31 | documented, noted |

Not checked: the pystrata library's own SHAKE implementation; DESIGN.md soil-model table past line 80.

## opensees_agent
Size: 5 python files plus curves and utils (14 with tests), 8 names in `__all__`, 2 adapter methods (pm4sand_cyclic_dss, site_response_1d). Site response is the best-documented tool in the group: it is pinned to two OpenSees examples, reports its choices and returns a spectrum. PM4Sand DSS is thin.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 6 | B | The PM4Sand integration tests cannot fail on a wrong answer. They assert `max_ru >= 0`, array lengths, and the "dense resists better" test passes when both liquefy or neither does. There is no published PM4Sand example. The site-response tests are pinned (wiki and docs examples). | opensees_agent/tests/test_opensees_agent.py:564-595 | new |
| 2 | 1 | B | PM4Sand DSS is sold as "for liquefaction triggering", but it returns the cycles to liquefaction at ONE `CSR` (default 0.15). There is no CRR at a given number of cycles and no CSR-N curve. The agent has to loop it. The G0, hpo and Dr inputs have no calibration help from (N1)60. | adapters/opensees_adapter.py:98-100, 183; pm4sand_dss.py (single CSR) | new |
| 3 | 3 | B | `max_ru` is clipped to [0, 1] in the DSS result. The site-response tool states its ru is "not capped at 1". The DSS cannot show an exceedance. | opensees_agent/pm4sand_dss.py:383-384; adapters/opensees_adapter.py (max_ru) | new |
| 4 | 3 | C | The DSS result has no `choices` (the site-response one does), and its 25 secondary parameters take defaults, some of them "-1 = model default", with no echo of what the model chose. DESIGN lists the stress-strain loops and stress path as key outputs, but `to_dict()` returns scalars only. | pm4sand_dss.py (grep `choices`: none); results.py:88-97; DESIGN.md:56 | partly GM-79 (site response only) |
| 5 | 5 | B | No calc package, tables or figure for either method. | grep `get_tables`/`_package`: none | new |
| 6 | 2 | C | GM-79's leftover list is open: ground slope, a strength-corrected curve option, K0/PTAng/e_init/yield surfaces into `choices`. | BACKLOG.md:203 | GM-79 open |
| 7 | 3 | C | `scale_factor` and `target_pga_g` are described as "give one or the other". The code multiplies both. | adapters/opensees_adapter.py (target_pga_g text); site_response.py:523, 551, 984 | new |

Not checked: the 1,805-line site_response.py beyond greps (heavily reviewed per BACKLOG); the PM4Sand parameter ranges.

## seismic_signals_agent
Size: 7 python files (10 with tests), 12 names in `__all__`, 4 adapter methods (response_spectrum, intensity_measures, rotd_spectrum, signal_processing). Numerics are tested; the outputs are thin and a few edges are silent.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 3 | A | `intensity_measures` swallows an `AttributeError`/`TypeError` from `calc_brac` and returns `bracketed_duration_s` = 0.0, which reads as a real value. The bracketed-duration threshold is never stated either. | seismic_signals_agent/intensity_measures.py:90-95; results.py:252 | new |
| 2 | 3 | B | `rotd_spectrum` zero-pads the shorter component to the longer one's length without telling the user. Only `n_points` shows. | seismic_signals_agent/rotd_spectrum.py:107-112 | new |
| 3 | 1 | B | `signal_processing` returns only peak values. The filtered or baseline-corrected record (which the module holds) cannot be taken out or passed to site response. The filter type and order are not stated in the tool text. | seismic_signals_agent/signal_processing.py:101-123; results.py:538-550 | new |
| 4 | 3 | C | If both `motion` and `accel_history` are given, `motion` wins silently. | opensees_agent/ground_motions.py:216-226 | new |
| 5 | 4 | B | Only two synthetic built-in motions, and `rotd_spectrum` has no `target_pga_g` while the other three methods do. | opensees_agent/ground_motions.py:177; adapters/seismic_signals_adapter.py:68-83 | new |
| 6 | 1 | C | `has_pyrotd` and `pyrotd_import_error` are still exported (pyrotd was dropped), and results.py still describes a "pyrotd" engine. The adapter reports only `nigam_jennings`. | seismic_signals_agent/__init__.py:31-47; results.py:341 | INF-06 / INF-47 open |
| 7 | 1 | C | The four modules import `opensees_agent.ground_motions`, so seismic_signals fails without opensees_agent present. | seismic_signals_agent/intensity_measures.py:68; response_spectrum.py:78; rotd_spectrum.py:88; signal_processing.py:78 | new |
| 8 | 5 | B | No calc package, tables or figure for any signals method. | grep `get_tables`/`_package`: none | new |
| 9 | 6 | C | Spectrum numerics are tested in geotech_common/tests/test_response_spectrum.py (not opened). This module's own suite is 87 tests, mostly result-class and shape checks. No published or closed-form oracle seen in test_seismic_signals.py. | seismic_signals_agent/tests/test_seismic_signals.py (grep) | unverified |

Not checked: eqsig internals; the Nigam-Jennings code in geotech_common; BACKLOG GM-274/275 detail.

## hvsrpy_agent, pyseismosoil_agent, swprocess_agent
Size: 0 python files each. Each folder holds only an empty `tests/` directory. No registry entry, no adapter, nothing in `__init__.py` of the registry.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 1 | C | hvsrpy and swprocess were wrapper modules for HVSR and MASW processing, deleted on purpose (commit da5b7f2, CLAUDE.md "geophysics excision"). pyseismosoil_agent was removed in the Phase 1 consolidation (commit 99fde72). The agent cannot reach any of the three by any route. | `git log -- hvsrpy_agent`; `git log --all -- pyseismosoil_agent`; CLAUDE.md:1643 | known (intentional) |
| 2 | 1 | C | Nothing else uses them: no tracked files (`git ls-files` empty), no imports. Leftovers are the three empty working-tree folders; empty extras `hvsrpy = []` and `swprocess = []` in pyproject.toml; and the historical mentions in docs/funhouse_agent_guide.md:180, 806 and in HANDOFF.md:589. pyseismosoil has no extra and no live mention. | pyproject.toml:276, 279; docs/funhouse_agent_guide.md:180 | known; the empty folders and extras are new |
| 3 | 1 | C | `.claude/agents/seismic-reviewer.md` was noted in HANDOFF as listing the deleted modules. The one current mention is a generic MASW/HVSR sentence (line 96) and needs no code. | HANDOFF.md:589; .claude/agents/seismic-reviewer.md:96 | HANDOFF note only, not in BACKLOG |

Not checked: whether the three empty folders are ignored by .gitignore or just untracked (they hold no files, so git ignores them either way).
