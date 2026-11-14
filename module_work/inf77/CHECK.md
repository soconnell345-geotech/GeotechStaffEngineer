# INF-77: standing tests that what the agent is told about a calculation is real

Branch `feat/inf77-standing-tests`, from master d3ae402. Source: BACKLOG INF-77 and the
module audit (module_work/module_audit_2026-11-12/AUDIT.md §6, "Standing tests").

## What each test checks, and where it lives

No probe runs twice: the checks that need a run sit in the probe guards that already run
every probe in `funhouse_agent/tests/method_probes.json`.

| Test | Check | File and test |
|---|---|---|
| T1 (a) | Every key METHOD_INFO `returns` names is in the probe's result, or its text says when it comes back | `test_defaults_used.py::test_the_defaults_a_run_lists_are_the_values_it_used` (on the run it already makes; `_promised_not_returned`) |
| T1 (b) | Every parameter changes the result or is refused: the defaults a run lists (GM-160, as before) and now every parameter the probe gives | `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result` (`_given_changes`, on the same base run) |
| T1 (c) | A value outside `allowed_values` is refused | `test_allowed_values.py::test_every_allowed_values_parameter_refuses_a_made_up_value` (already there, GM-317; unchanged) |
| T1 (d) | Every calc_package `_IMPORT_MAP` module has a `*_package` method or a stated reason | `test_told_is_real.py::test_every_calc_package_module_has_a_package_method` (new) |
| T1 counts | A count of modules, tools or packages in a brief (the module catalog, every method's brief, the agent package's docstring) is one of the registry's counts | `test_told_is_real.py::test_a_count_in_a_brief_is_the_registrys` (new) |
| T2 | Every module warning a call raises is in its result | `test_defaults_used.py::test_the_defaults_a_run_lists_are_the_values_it_used` (already there, GM-316: `MP.lost_warnings`; unchanged) |
| T3 | The names in the code blocks of each DESIGN.md "Public API" section are bound in the package `__init__.py` (AST, no import) | `test_told_is_real.py::test_every_design_public_api_name_exists` (new) |
| T4 | No new `except Exception:` (or bare `except:`) whose whole body is `pass`, `continue` or an empty return in the calculation packages | `test_told_is_real.py::test_no_new_swallowed_failure_in_the_analysis_packages` (new; today's hits in `SWALLOWED`, each with why and a count per function) |

Each new reader has a small test of its own (`test_the_returns_check_reads_conditions_and_pairs`,
`test_the_given_changes_reach_fields_and_lists`, `test_the_count_check_reads_words_and_figures`,
`test_the_api_reader_takes_calls_assignments_and_methods`, `test_the_scan_sees_each_form`), and
`test_the_inf77_tables_name_probes_and_promised_keys` checks the two new tables.

## What already existed, and what was extended

- `test_every_default_a_run_lists_changes_its_result` (GM-160) changed only the defaults a run
  lists. It now also changes each parameter the probe gives, once each, on the base run it
  already makes: a scalar to another value (an integer quantity moves as a number does, since
  + 1 on 150 pci moves nothing a run prints), each number or boolean of the first item of a
  list of dicts (`layers[0].thickness`) or of a dict, and a list of numbers all at once. A
  refusal counts as moved. Cases that do not move go in the same tables as before
  (`MP.LABEL_ONLY`, `MP.NOT_REACHED`), keyed by the probe and the label; the table check now
  accepts a label whose parameter is declared.
- `test_the_defaults_a_run_lists_are_the_values_it_used` (GM-158, with GM-316's warning check)
  now also checks the promised `returns` keys on its first run. Rules, so the check does not
  chase keys that come back only in some cases: the key is in the result; or its text says
  when (a condition word, a case label such as "Cohesive: ...", "For a ...", or "with" a
  declared parameter); or it is a warnings or notes key (there only when there is something to
  say); or it is not a name ("...", "(method numbers)"); or it is in the new
  `MP.RETURNED_SOMETIMES` with why. A key written `a / b` names both. The probes of `MP.FROZEN`
  (slope_stability) are left to their review, as the GM-160 guard does.
- `test_every_allowed_values_parameter_refuses_a_made_up_value` (T1 c) and
  `MP.lost_warnings` (T2) were already there (GM-317, GM-316); not changed.
- `test_module_followup_adapters.py::_assert_documented` checks returns for a few named
  methods; left as it is.
- New: `funhouse_agent/tests/test_told_is_real.py` (T1 d, counts, T3, T4).

## Failures found on master, and what became of each

### T1 (a), promised returns keys

A first scan found 476 named keys missing from probe results; nearly all come back only in
some cases and say so. After the rules above:

- **Fixed text (7 keys, 5 methods):** soe `select_wall_section` named `designation`, the result
  has `name`; `check_basal_heave` named `is_stable`, the result has `passes`; `check_piping`
  named `i_exit`, the result has `demand` (and `parameters.i_exit`); ground_improvement
  `design_drain_spacing` named `spacing_m`, the result has `drain_spacing_m`; seismic_geotech
  `site_classification` `Fa`, `Fv`, `Fpga` come back only with both `Ss` and `S1` (checked), and
  now say so.
- **Allowlisted (`MP.RETURNED_SOMETIMES`, 26 entries, each with when):** reliability_package's
  per-method keys (6: `beta`, `pf`, `beta_normal`, `pf_normal`, `beta_lognormal`,
  `pf_lognormal`), pile_group `group_efficiency`'s block-failure and eta_g keys (6),
  `group_uplift` utilization and adequacy (with `uplift_load`, 2), html_to_pdf (3), the
  placeholder `extra` of fem2d_package and pavement_design_package (2), and one each:
  pile_group_package `group_efficiency_row`, retaining_wall_package `drawing_scale`,
  fem2d_excavation `wall_water_pressures`, liquefaction_analysis `layer_results` (SPT only),
  flexible_pavement_design_ufc `frost_governs`, earth_pressure_coefficient `choices`,
  seismic_earth_pressure `delta_passive_deg`.
- The audit's named cases (liquepy CPT and field_correlations, ground_improvement feasibility)
  no longer fail: GM-322 and GM-324 fixed them.

### T1 (b), every parameter read

The first pass flagged 102 given parameters; 36 were integer quantities moved by + 1 (150 to
151 pci reads the same chart thickness), which the number rule now moves properly. The 96 left
(87 on probes that stay in the sweep) are all by design or out of the probe's reach,
allowlisted with reasons (82 entries; 5 more come from the fem2d tool entries that
fem2d_package mirrors):

- one-layer profiles taken to go on below (layer 0's thickness or bottom; 23), a FEM layer
  that fills the mesh (2) or whose top is cut at the ground (2);
- probes with no load (pynite `continuous_beam` and `frame_analysis`, `pile_group_6dof`; 8);
- build tools that carry the property into their output (dxf_import; 5);
- p-y curves whose points all lie past the initial line, so `k`, `Gmax`, `ks`, `Er` do not show
  (4; checked: k z y is above p at the first point);
- figure-only inputs (5); a label (`render_tables.output_path`);
- the rest one by one, each reason checked against the run or the code: e.g. `surcharge_kPa`
  acts only with `sigma_v0` (code read), drained sliding does not count `c_foundation` (as its
  text says), GEC-4's 4.5 m minimum unbonded length governs over `phi_deg`,
  fem2d_consolidation's `k` under the staggered scheme (see "Noted" 1).
- The audit's named cases (soe supports, the salib counts, `mesh_size`, pile_group's simple
  batter) no longer fail: GM-318 to GM-323 fixed them.

### T1 (d), counts, T3, T4

- T1 (d): passes; fem2d_package exists since GM-328. `NO_PACKAGE_METHOD` is empty.
- Counts: no brief states a count. The agent package's own docstring said "50 modules (~901
  methods)" (the registry has 72 modules); fixed to name the registry, with no number.
- T3: 7 names in 3 DESIGN.md files did not exist: drilled_shaft (`analyze_drilled_shaft`,
  `analyze_capacity_vs_depth`, `get_resistance_factors`), seismic_geotech
  (`mononobe_okabe_active`, `analyze_liquefaction`), settlement (`analyze_settlement`,
  `analyze_consolidation`). Each section now lists the package's real exports. The six the
  audit named were already fixed; soe's DESIGN.md has no Public API section. Prose outside code
  blocks is not read (it says "there is no `analyze_x`").
- T4: 16 hits in 10 functions, listed in `SWALLOWED` with a count per function and why: 8 in
  wave_equation's figure builders, 1 lateral_pile p-y table row, 2 DIGGS path lookups, 2
  temporary-file clean-ups, calc_package's print-width pass (its record says so), gstools'
  optional plotly, dxf_import discovery. The audit's four (axial_pile #1, pystra #5,
  seismic_signals #1, concrete_props' dropped points) are gone from master.

### Not checked here

- T1 (b) covers only what the probes reach (review #4). Of the 2,227 parameters the probed
  methods declare, 814 (in 111 of the methods, about 37 %) are neither given by any probe nor
  listed in a run's `defaults_used`, so no check ever changes them. Example,
  axial_pile_capacity: `extend_last_layer`, `inside_friction_ratio`, `open_pipe_method`,
  `open_pipe_toe_limit`, `pile_area`, `pile_perimeter`, `pile_tip_area`, `pile_weight`,
  `uplift_skin_fraction`. The test's docstring says the same.
- T3 reads only the 12 of 35 packages whose DESIGN.md has a "Public API" section (review
  #13); the other 23 are not checked. Inside a section, `def`, `class` and `module.func`
  lines would be dropped silently (none today); the `read >= 10` guard stops an empty pass.
- gstools' 4 probes cannot run on this machine (Smart App Control blocks gstools' DLLs). A
  static check found every key their `returns` names in the adapter or gstools_agent code;
  the first GitHub run confirms.
- slope_stability is frozen (GM-140): its probes skip the returns check, as they skip GM-160.
  One run of each showed keys missing with no condition in their text (e.g. analyze_slope
  `methods_without_solution`, `lambda_mp`, `slice_data`, `thrust_line`; compare_methods_table
  `warnings`; yield_acceleration `reason`, `ky_bounds`); for the slope build.

## Time

Measured on this machine (a laptop), one process: the given-parameter sweep is 977 extra runs
taking 176 s (all 1,049 took 383 s; the 8 probes in `MP.GIVEN_LEFT_OUT`, each 2 to 5 s a run,
took 207 s of that and are left out by name). T1 (a) adds no run (it reads the run GM-158
already makes). `test_told_is_real.py` takes 3 s. The extended GM-160 test ran in 11 min here
for its 249 probes (defaults and given parameters together).

**GitHub estimate: about 3 to 5 CPU-minutes added a job** (runners at 1 to 1.5 times this
machine's time), under the 10-minute budget.

Local runs: GM-158 and GM-160 over every probe but slope and gstools (all passed; one more
returns case found and listed on the way), `test_told_is_real.py` 7 passed, the manual rebuilt
by its generator (unchanged; `--rebuild` exit 0), `test_user_manual_current.py`,
`test_page_fingerprints.py`, `test_catalog_routing_hints.py`, `test_public_text.py`,
`test_probe_warnings.py`, `test_module_followup_adapters.py`, `test_gm322_323_adapters.py`,
`test_adapter_ergonomics.py` all passed.

## Noted, not fixed

1. **fem2d_consolidation's default scheme makes no consolidation.** The default `staggered`
   scheme is drained at every step: U = 1 and zero excess pore pressure at t = 0 for any `k`
   (checked 1e-10 to 1e-16 m/s). The result's note says so, but the brief reads "Coupled Biot
   consolidation analysis (settlement + pore pressure dissipation)". A default run's saved
   report shows no consolidation. Proposed BACKLOG line (a default, so the owner's call):
   "fem2d_consolidation: the default staggered scheme turns no load into excess pore pressure
   (U = 1 at every time, any k), while the brief promises pore-pressure dissipation; decide
   whether the default becomes 'monolithic' or the brief says a load's consolidation needs it.
   *Src:* module_work/inf77/CHECK.md"
2. **A figure or table row that fails to draw is left out without a note.** wave_equation's
   bearing-graph and drivability figure builders (8 handlers) and lateral_pile's p-y table
   (one row per layer) swallow the failure; the saved package then lacks the figure or row and
   says nothing. Proposed BACKLOG line: "calc packages: a figure (wave_equation, 8 places) or
   a p-y table row (lateral_pile) whose drawing raises is left out silently; say in the
   package what was left out and why. *Src:* module_work/inf77/CHECK.md, T4"
3. Not proposed (no saved report changes): the pynite and pile_group_6dof probes carry no
   load, so the guard cannot check their stiffness and geometry inputs; a load in those probes
   would.

## Calls

About 90 tool calls for the build, the checks and these notes.

## Review round (2026-11-13)

An independent review graded the branch MERGE AFTER FIXES. Master was merged in first
(b5e49ae): no conflicts; the manual and `toc.json` needed no rebuild. What each finding
became:

- #1 (sheet_pile, a layer that stops above the toe), #6 (the two "Noted, not fixed" items),
  #7 (surcharge `equivalent_sigma_p_kPa` 0.0), #8 (fem2d_consolidation brief): for the
  lead to file in BACKLOG.md.
- #2: `group_uplift_utilization` and `group_uplift_adequate` say they need uplift_load and
  axial_capacity_tension; a run given uplift_load alone gets a note that no pass/fail was
  made and what to give (or that no block factor is printed). group_efficiency's block keys
  say they need pile_length and su1 and su2 (or one cu); s_over_b and the row say "with
  soil_type". The allowlist reasons say the same. No number changes.
- #3: a `returns` text excuses its key's absence only with when, if, only, unless or
  without, a leading "For a ...", or "with" a declared parameter (now any case: "With cv:").
  The colon rule and the loose words (where, absent, omitted, otherwise) are gone. One run
  of every probe (a scratch scan with the test's own helpers) found 30 texts that leaned on
  them; each now states its case: the four the review named (drawing_sheet and
  drawing_scale with wall_type 'cantilever_reinforced'; strut_forces with struts; the
  consolidation choices without delta_sigmas; cited_pages when a record cites a page), and
  case labels such as "Monte Carlo:" now "Monte Carlo only:", "Cohesive:" now "Cohesive
  soil only:", parse_cpt `u_2_kPa` "only when the file has a u2 column", the log-spiral
  keys "only for theory 'caquot_kerisel' passive". No value the code computes was removed.
  The gstools keys could not be run here (Smart App Control); `results` (regions is
  required) and the variogram `nugget` always come back by the code.
- #4: coverage stated above ("Not checked here") and in the test's docstring. No new probes.
- #5: `_norm` drops a figure file's `_2` number and a `site_key`'s `_1` copy suffix. Two
  identical runs of `subsurface|load_site`, `subsurface|parse_diggs` and
  `calc_package|render_figures` now agree. Proof in a scratch run: with `friction_angle`
  (then `width`) made dead in render_figures by a wrapper, the sweep's change reports "not
  moved", so the check goes red; nothing in the tree was changed for it.
- #9: `_moved_by` counts a crash (IndexError, TypeError, AttributeError, ZeroDivisionError,
  NameError, AssertionError and the like, or a KeyError naming a bare key) as not moved. A
  ValueError, an adapter's worded error, a module's own error class, a missing file and a
  worded KeyError ("Hammer ... not found") still count as refusals. The fem2d node-index
  changes (fem2d_seepage `elements` and `head_bcs`, fem2d_staged `elements`, and their
  fem2d_package mirrors) are allowlisted: "a doubled node index is out of range and raises
  IndexError, not a worded refusal".
- #10: `_swallowed` treats a tuple holding Exception or BaseException as broad and a lone
  `...` as empty (the review's M1 cases in the self-test). It then found three figure
  handlers written `except (ImportError, Exception): pass` (pile_group and settlement
  `get_figures`); they are listed in SWALLOWED as figures left out.
- #11: a nested allowlist label must name a field some probe of that key gives.
- #12: `_ONE_LAYER` reads "the probe's one layer reaches below the toe (or the wall); a
  thicker one changes nothing".
- #13: noted above ("Not checked here").

Checks: test_told_is_real 7 passed; test_defaults_used `-k` pile_group, subsurface,
calc_package, fem2d, retaining, diggs and the table tests 342 passed, 2 skipped;
test_catalog_routing_hints 14, test_public_text 5, test_user_manual_current 1,
test_page_fingerprints 13 passed; `check_manual.py --rebuild` exit 0. About 55 tool calls.
