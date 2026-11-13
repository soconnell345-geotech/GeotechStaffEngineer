# GM-316 and GM-317: warnings reach the agent; `allowed_values` enforced

Branch `claude/gm-316-317-warnings-allowed-values`. Items: BACKLOG GM-316,
GM-317; module audit 2026-11-12 (`module_work/module_audit_2026-11-12/
AUDIT.md`) §2.1, §2.2 X1 and X2, §6 T1 (c) and T2. No print reference is
involved: both are dispatch-layer changes, and no formula, default or
coefficient changes.

## Piece 1, GM-316: a Python warning raised inside a method reaches the agent

### What changed

- New `packages/aec-agent-core/aec_agent_core/call_warnings.py`, used by
  `run_method` (`aec_agent_core/method_call.py`): the whole call (checks,
  handler, notes) runs inside `call_warnings.collecting()`, and the
  warnings collected are added to the result before working-folder paths
  are hidden. The old body of `run_method` is `_run_checked`, unchanged.
- Two pieces, installed when `aec_agent_core.method_call` is imported
  (and again by a call if something replaced them), both appending to a
  `contextvars.ContextVar` list that `collecting()` sets, so calls
  running at once in threads keep their own (changed by the review fix
  A1, below):
  - a thin wrapper round `warnings.warn`: while a call collects, a
    warning the app's packages raise is recorded for the call BEFORE the
    process's filters and once-per-location registry act, then handed to
    the original `warn` unchanged (its `stacklevel` one deeper for the
    wrapper's frame, so it is attributed, filtered and shown as before);
    outside a call it only hands on;
  - one `warnings.showwarning` hook, for what does not pass through the
    Python `warn` (numpy's floating-point warnings, raised from C; code
    that bound `warn` to its own name first). Outside a call, or for a
    warning the wrapper recorded, it hands on to the hook it replaced,
    unchanged. When a `catch_warnings(record=True)` block (pytest puts one
    round every test) has swapped the hook out, the next call puts a new
    hook on top of whatever is there (the block puts its own back when it
    ends, so a hook installed earlier, e.g. `logging.captureWarnings`, is
    not lost). A hook that hands on to another of these hooks does not
    record twice.

  `catch_warnings` is not used round the call.
- The core names no application: GSE hands in its calculation packages
  through the new hook `method_call.warning_packages`
  (`funhouse_agent/core_hooks.py: WARNING_PACKAGES`, filled in
  `funhouse_agent/__init__.py`; listed in `aec_agent_core/hooks.py`).

### The rule: which warnings reach the agent

A warning raised while a handler runs reaches the agent when

1. the code that issued it, or the code its `stacklevel` attributes it to,
   is in one of the app's calculation packages (GSE: the 34 calculation
   packages pyproject ships, `funhouse_agent.adapters` and
   `geotech_references`; a test checks the list against pyproject), or is
   `aec_agent_core.method_call` (a module's `stacklevel` one too high lands
   on the frame that called the handler); or
2. it is numpy's floating-point RuntimeWarning (`overflow`, `invalid
   value`, `divide by zero encountered in ...`), wherever it is attributed
   (scipy's internals included): it means a NaN or an infinity in a
   calculation;

and never when its category is DeprecationWarning, PendingDeprecationWarning,
FutureWarning, ImportWarning, ResourceWarning, SyntaxWarning, BytesWarning or
EncodingWarning, or when a plotting library issued it (matplotlib,
mpl_toolkits, plotly, PIL, kaleido, seaborn: matplotlib's `warn_external`
attributes its layout warnings to the caller's line, so attribution alone
would let them in). Anything else from a third-party library, attributed to
that library, stays on stderr only.

Improvement on the brief's starting point: the rule looks at the issuing
frame as well as the attributed one, so a scipy warning raised with
`stacklevel=2` against our own call (e.g. a fit that could not estimate its
covariance) does reach the agent (it is about our calculation), while
matplotlib's, attributed the same way, does not.

**Repeats.** Python's default action shows a warning once per location
(`__warningregistry__`). The wrapper records a package's warning on every
call that raises it whatever the registry holds, and leaves the registry
alone: on stderr such a warning is shown once per location for the life of
the process, as before GM-316, and an "always" recorder (`pytest.warns`)
sees every occurrence. For a warning only the hook sees (numpy's), the hook
takes its key `(text, category, lineno)` back out of the registry of the
module it is attributed to, so the next call raising it reaches the hook
again; within one call it goes on to the replaced hook (stderr) once per
location. No filter is added and no other registry is touched, so
third-party warnings keep their once-per-location behaviour.

**What the agent gets.** The message text only (no file path, no line
number); at most 20 a call, then "and N more". A dict result gets them under
`warnings`, after the entries the module already returns there, with no
duplicates (a module that both warns and lists the same text gets it once).
The `warnings` key is moved to the FRONT of the dict when anything is added:
a reference module's result is cut at its tail for size
(`funhouse_agent/deep/tools.py` `_result_cap_for_module` -> `_truncate`;
calc modules are uncapped) and a result of 60,000 characters or more is
shown to the model as a key map whose preview is its first 1,000 characters
(`aec_agent_core/deep/result_map.py`); at the front, the warnings survive
both. An error result (`{"error": ...}`) also carries the warnings raised
before the error. **A result that is not a dict is returned unchanged**:
no `call_agent` method returns one today (every probe returns a dict), and a
library caller of `run_method` that gets a list keeps a list rather than
finding it wrapped.

### Consumers checked

- `report_ingest/recompute.py:1854, 2356` copies a module result's
  `warnings` to the row's `module_warnings`: it now also holds the Python
  warnings the module raised, which is what the name says.
- `funhouse_agent/chart_eval/arms.py:370` prints a result's warnings: same.
- `funhouse_agent/tests/test_defaults_used.py` leaves `warnings` out of its
  comparisons (`_VOLATILE`), so a warning does not make an input look
  moved.
- The calc packages' METHOD_INFO says their `warnings` are "also printed in
  the package": a Python warning raised while a package runs is now added
  to that list but is not printed in the PDF. Noted, not fixed (the
  package's own list is still printed; no probe raises one).
- `funhouse_agent/deep/prompt.py` and the core's prompts say nothing about
  warnings, so no prompt text changed (page fingerprints unchanged).
- The webapp reads no tool result's `warnings`.

### Probes that now carry warnings

None. All 251 probes in `method_probes.json` were run with every warning
recorded (`simplefilter("always")`) before and after the change: they raise
no Python warning at all (they are the smallest happy-path inputs). So the
standing test (T2, `funhouse_agent/tests/test_probe_warnings.py`) also runs
four made-up calls that do raise the module warnings the audit named, each
twice and under both the "always" and the default filter:

| Call (made-up inputs) | Warning the agent now sees |
|---|---|
| axial_pile `axial_pile_capacity`, pile 40 m in a 30 m profile | "Pile tip (head_depth 0.0m + pile_length 40m) exceeds soil profile (30m); using full profile depth" |
| drilled_shaft `drilled_shaft_capacity`, length 20 m in 15 m | "Shaft length (20.0m) exceeds soil profile (15m); using full profile depth" |
| lateral_pile `lateral_pile_analysis`, soil to 10 m, pile 15 m | "Soil layers extend to 10 m but pile length is 15.0 m. No soil resistance will be applied below 10 m." |
| settlement `consolidation_settlement`, Cr 0.3 > Cc 0.05 | "Cr (0.3) > Cc (0.05); typically Cr < Cc" |

Each of these is a module defect the audit lists for wave 2 (a profile
shorter than the pile should be refused or stated, not extended); they are
not fixed here.

### Review fix A1: delivery no longer depends on the process's filters

The review (2026-11-12, finding A1) showed GM-316 could be switched off for
the life of the process. Three blocks on the tool-call path used
`warnings.catch_warnings()` + `simplefilter("ignore")`:
`drilled_shaft/capacity.py` `_alternative` (several times in every
cohesionless `drilled_shaft_capacity` call), `lateral_pile/solver.py`'s
capacity search, and `aec_agent_core/charts/layout_ink.py`'s two (under a
lock of their own). `catch_warnings` swaps process-wide state: when two
blocks interleave across threads (A enters, B enters, A exits, B exits), B
puts back A's copy of the filters, so `('ignore', None, Warning, None, 0)`
stays at the front for good, and from then on no warning reached the hook.
The review's `p2_leak` probe showed it with two threads of drilled-shaft
calls: after the leak, the axial-pile call's "exceeds soil profile" warning
was gone from every result. Third-party libraries use `catch_warnings` too,
so it could happen with none of our blocks.

The fix, in two parts:

1. **Delivery of the packages' own warnings is independent of the filter
   list and the registry**: the `warnings.warn` wrapper above records them
   before any filter acts. `catch_warnings` never swaps `warn`, so the
   wrapper, once installed, stays; it is installed at import of
   `method_call`. The `showwarning` hook stays for warnings raised from C
   (numpy's floating-point ones), which never pass through the Python
   `warn`; those still depend on the filters, so an "ignore" filter left
   in front drops them (and everything on stderr), as it would in any
   Python process. Chosen over the review's two suggestions: a shared lock
   round every filter swap stops only OUR blocks from interleaving (not a
   library's) and still drops a call's warnings while a block is open; a
   context-local `quiet()` would mean the analysis modules importing the
   core, which they must not. The wrapper costs one Python frame per
   `warnings.warn` in the process; outside a call it changes nothing
   (same attribution, same filters, same registry: test
   `test_outside_a_call_a_warning_is_shown_as_before`, and
   `skip_file_prefixes` on 3.12+).
   Tests (`packages/aec-agent-core/tests/test_call_warnings.py`): an
   ignore-everything filter put at the front, 100 calls each still carry
   their warning; the p2_leak interleaving reproduced with two threads
   and events (the blanket filter is left at the front), then 100 calls
   each carry theirs.
2. **The hazard we own is removed** where it was safe, the numbers
   unchanged (every drilled_shaft and lateral_pile probe plus four made-up
   calls, before and after: identical results and warnings):
   - drilled_shaft `_alternative` silenced the warning the trial copy of
     the analysis raises again in `__post_init__` (the shaft longer than
     the profile, already raised by the analysis it was copied from) and
     any numpy notice. Now a `ContextVar` (`_IN_TRIAL`, per thread and per
     context) keeps the copy from raising it, and `np.errstate(all=
     "ignore")` (per thread in numpy 2) quiets numpy's notices in the
     trial.
   - lateral_pile's capacity search silenced the solver's own "Singular
     matrix in solver" warning in its trial solves (a factor of 0
     follows) and any numpy notice. Now the trial calls `_solve_system`,
     which returns None on a singular system without warning (the zeros
     follow as before), under `np.errstate(all="ignore")`;
     `_assemble_and_solve` still warns for a real solve.
   - layout_ink's two blocks are left under their lock: they silence
     matplotlib's layout warnings, which the rule never delivers anyway,
     and they are the only filter swaps left on the path among ours.
     They can still interleave with a library's own block, which part 1
     covers.

   Neither analysis module imports the core or the other. A standing test
   (`funhouse_agent/tests/test_probe_warnings.py::
   test_no_calculation_package_swaps_the_warning_filters`) fails if a
   calculation package's code calls `catch_warnings`, `simplefilter`,
   `filterwarnings` or `resetwarnings`: such a block would no longer hide
   the package's own warnings from the agent, and it swaps the filters for
   every thread.

### Noted, not fixed

- A handler that starts worker threads of its own: a worker started with
  `contextvars.copy_context().run` (the core's vision tools do this) runs
  in the call's context, so its warnings are recorded while the call runs;
  one that finishes after the call has returned loses them to the agent
  (stderr keeps them). A plain `threading.Thread` or pool worker has no
  context of the call, so its warnings stay on stderr only (on a
  free-threaded 3.14 build, whose threads inherit the context, they are
  recorded too). No module does either through a probe today. (Corrected
  after review C7.)
- numpy's floating-point warnings (the hook's path) still depend on the
  filters and, across calls, on taking the registry key out: a warning
  raised OUTSIDE any call earlier at the same line with the same text is
  suppressed by the registry until the filters next change, and an
  "ignore" filter left in front drops them. A package's own
  `warnings.warn` no longer has either limit (before review fixes C6 and
  A1, it had both).

## Piece 2, GM-317: a value outside a parameter's `allowed_values` is refused

### What changed

`_check_parameter_values` in `aec_agent_core/method_call.py`, run right after
the parameter-name check (GM-110), on the same path and in the same voice:

> `axial_pile_capacity: method='Beta' is not one of the values it takes.
> Did you mean 'beta'? Valid values: ['auto', 'beta']. describe_method gives
> each one's meaning.`

The did-you-mean is the same value in another case, else the closest one or
two by spelling (hyphens and spaces read as underscores). Nothing is coerced:
the method gets the value as given or the call is refused.

Where the code already refused a value with guidance of its own, that
guidance is kept: a parameter may declare `refused_values` (value -> reason),
given in place of the did-you-mean. Two do: bearing_capacity
`factor_method="hansen"` ("Hansen's N-gamma is chosen separately with
ngamma_method='hansen'...") and soe `design_ground_anchor` `soil_type`, whose
rows removed by GM-191 say where each went (`soe.anchor_design.
_REMOVED_TYPES`, the same text the module raises).

**Shapes enforced.** Every declared list (346 parameter entries, 220 distinct
lists, in 29 adapter files plus the core's `plot_data`) is on a top-level
parameter; no adapter declares a list inside a dict or a list of dicts, so
nested parameters are not checked. Checked: a string against a list of
strings; each string of a list given for an array parameter (pynite
`support_types`); a boolean or a number against a list of that kind. Left to
the method: None (the method's default: several handlers refuse an explicit
None themselves, dxf_export crashes on one; unchanged), and a value of
another kind than the list's (fem2d `n_gp` converts "3" to 3 itself; a
pile-group `method_6dof` of 0 or 1 is equal to False/True in the code). A
parameter given under an alias (`parameter_aliases`) is checked as the
parameter it names; a method with `parameters_of` (render_figures,
render_tables) is checked against the named package's lists too.

**Case.** Exact, unless the parameter now says `"any_case": True` (the code
strips and lower-cases before comparing) or `"any_case": "separators"`
(the code also ignores spaces, hyphens and underscores). `any_case` was added
only where the code does fold case (list below).

### How the lists were proved complete

Each of the 220 distinct lists was traced from the adapter through to the
module (four read-only reviewers, one per group of adapters, each value
followed to the branch, lookup or validation that acts on it, with spot
probes of in-list, upper-case, padded, hyphenated and made-up values; their
reports are summarised here). 177 lists were complete as declared. The 43
others were corrected:

| Adapter (methods) | Parameter | Correction | Why |
|---|---|---|---|
| bearing_capacity (analysis, factors) | shape, factor_method, ngamma_method (both), failure_mode (both), depth_factor_method, method | any_case | the module lower-cases each (capacity.py:165-214, footing.py:67) |
| settlement elastic_settlement | shape | add circle, rectangle; any_case | immediate.py:95, 225 reads both |
| settlement elastic_settlement | rigidity, point | any_case | lower-cased before lookup |
| settlement schmertmann_settlement | shape | add circular; any_case | the module documents and runs "circular" (axisymmetric, immediate.py:295) |
| settlement consolidation / combined | stress_method | add 2to1; any_case separators | stress_distribution.py:375, 434 lower-cases and drops spaces; "2to1" is its alias |
| settlement combined_settlement_analysis | shape | add circle; any_case | analysis.py:170-179 |
| pavement_design rigid | pavement_type, shoulder_type | any_case | tables strip and lower-case |
| pavement_design rigid | drainage_quality | any_case separators | spaces read as underscores ("Very Poor") |
| liquefaction | method | add the ten spellings `_METHOD_NORMALIZE` reads (built from that table); any_case | liquefaction_adapter.py:41-54, 97 |
| liquefaction | input_type | any_case | `_detect_input_type` strips and upper-cases |
| fem2d fem2d_consolidation | reset_displacements_after | add first, none, "0"; any_case | fem2d/analysis.py:2911-2924 |
| retaining_walls | passive_theory, theory | add logspiral, log-spiral | adapter alias map, retaining_walls.py:771-773 |
| retaining_walls mse_wall | reinforcement_type | add barmat, welded_wire_grid, steel_grid, grid, strip, ribbed_strip | adapter alias map, retaining_walls.py:117-131 |
| soe braced/cantilever | passive_theory | add logspiral, log-spiral | soe.py:136-144 |
| sheet_pile cantilever/anchored | pressure_method | add caquot_kerisel, logspiral | sheet_pile/cantilever.py:29, 820-826 |
| dxf_export (both methods) | units | add meters, millimeters, centimeters, feet, inches; any_case | the export's unit table |
| wave_equation (three methods) | driving_system | any_case separators | driving_systems.py:118-142 ignores case, spaces and hyphens |
| pynite continuous_beam | support_types | any_case (each item) | lower-cased per item |
| profile_figure subsurface_profile | axis | any_case | strip + lower |
| core plot_data | legend_position | any_case | lower-cased |
| calc_package slope_stability_package | method | add janbu, morgenstern_price, gle | analyze_slope runs each as its own method (analysis.py:156-192) |
| calc_package selectors (design_type, analysis_type, wall_type x3, ground-improvement method, engine, method_6dof) | the selector | add each package's `selector_values` spellings; any_case | `_selector_value` (calc_package.py:1259-1267) strips, lower-cases and maps them |

The calc packages copy their tools' specs (`_package_parameters`), so each
tool correction above reaches its package too (settlement_package's shape and
stress_method, pavement_design_package's three, bearing_capacity_package's
five, the passive_theory, pressure_method, reinforcement_type and
driving_system of the wall and wave packages).

**No list named a value the code does not act on**, so none lost a value.
Two lists are wider than one of the tools that reads them and that is left
as is: retaining_wall_package `sliding_cohesion` is the union of the three
wall tools' lists, and the MSE tools still refuse `friction_plus_adhesion`
themselves with a reason; lateral_pile `head_condition` leaves out the
solver's `partial`, rightly, since the adapter cannot pass its rotational
stiffness. `compute_trend`'s list stays ["linear", "log_linear"]; nothing
fits an exponential, so "exponential" is refused rather than added.

### Out-of-list values that ran as something else (now refused)

The two the audit confirmed, and the others the trace found:

- subsurface `compute_trend` `trend_type="exponential"`: fitted a straight
  line and labelled it exponential. Refused.
- calc_package `format`: only "pdf" and "latex" are branched on; "docx",
  "PDF", "tex" and " pdf" all wrote HTML into a file with that extension and
  reported success (" pdf" overwrote a real PDF at the same path). Refused;
  format is compared exactly, as the code compares it.
- axial_pile `method`: anything but "beta" (e.g. "Beta", "nordlund") ran
  "auto".
- retaining_walls and sheet_pile `pressure_method`: anything unknown
  ("Coulomb", "LOG_SPIRAL", "log-spiral" on sheet_pile) ran Rankine.
- fem2d `fem2d_local_fos` `srm_field`: an unknown value reduced no strength
  (global FOS 3.0, the SRF ceiling, against 1.444).
- lateral_pile `head_condition`: an unknown value gave zero deflection and
  zero moment; `pile_type`: ran as a generic circular pile.
- settlement `drainage`: anything but "double" ran single drainage.
- drawing_ir `units` read as metres, `origin` and pdf_import `origin` as
  top-left; dxf_export `units` left the units header out of the file.
- downdrag `detail`: anything but "full" returned the short result.
- slope_stability_package / slope_report_package `method` and
  `surface_type`: an unknown value ran Bishop / the circular grid search.
- liquefaction `input_type`: an unknown value fell back to detecting the
  data; wave_equation `hammer_type` "Diesel" skipped the diesel warning.
- subsurface `color_by`, `output_format`: fell back silently.

### Replay: calls newly refused

None. The value check was run on every call in `method_probes.json` (251
probes and their `before` calls), every `dispatch_calls` entry in
`funhouse_agent/worked_examples.json` (28 examples) and every call in
`docs/user_manual/worked_examples.json` (18): 322 calls, 0 refused. The
validation_examples and the test suites that call the dispatch were run in
full (counts under Tests).

### User manual

`docs/user_manual/manual.html`: only the "allowed:" spans of the parameter
rows whose list changed were edited (35 rows), by matching each row's
parameter name and old list within its method's section. `any_case` is not
printed by the manual builder; no rebuild was run.

### Tests changed because a message or a list changed

- `test_adapter_ergonomics.py`: the retaining-wall `theory` and
  `reinforcement_type` lists now carry the adapter's other spellings.
- `test_liquefaction_router.py`: `method`'s list is the two ids then the
  spellings `_METHOD_NORMALIZE` reads.
- `test_calc_package_tool_parity.py` (soe_package `wall_type="anchored"`),
  `test_mse_level_depths_adapter.py` (`sigma_H_average="mean"`),
  `test_drawing_ir_adapter.py` (`query="frobnicate"`): each was already
  refused by the module; the refusal now comes from dispatch, in its words.

## Tests run (this machine, Python 3.12 venv; pytest's exit code)

- Piece 1, before its push: core `packages/aec-agent-core/tests` (incl. the
  new `test_call_warnings.py`, 15), `test_top_level_params.py`,
  `test_defaults_used.py`, `test_probe_warnings.py` (253 passed, 3 skipped),
  `test_page_fingerprints.py`, `test_public_text.py`: 1443 passed, 2 failed
  (both below, failing on master too), then the M-O test updated. Wider run
  (funhouse_agent/tests, deep tools, report_ingest, the modules whose tests
  read a result's warnings), `-m "not slow"`: 9972 passed, 2 failed (the same
  two).
- Piece 2: funhouse_agent/tests, funhouse_agent/deep/tests, core tests,
  webapp/tests, calc_package, validation_examples, report_ingest/tests and
  the tests of bearing_capacity, settlement, pavement_design, dxf_export,
  sheet_pile, retaining_walls, soe, wave_equation, pynite_agent,
  profile_figure, fem2d (`-m "not slow"`): 14727 passed, 10 failed: 8 were
  the messages and lists above (fixed: `refused_values` for two, test
  expectations for six), 2 fail on master too. Re-run of every touched test
  file plus `test_allowed_values.py` (271 passed, 3 skipped: T1 (c) over
  every probe and the shapes), fingerprints, public text and the core tests:
  1093 passed, 3 skipped.
- Failing on master as well, not touched: `test_defaults_used.py::
  test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
  (logx/logy change nothing in the result) and `test_calc_package_figure_
  conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`.
- Page fingerprints unchanged: no tool description or prompt changed
  (`allowed_values`, `any_case` and `refused_values` live in METHOD_INFO,
  read by describe_method).

## Release note

What a user of the agent will notice. **Warnings:** a calculation that hit
something the module warns about (a pile or shaft longer than the soil
profile, soil layers that stop short of a pile toe, Cr greater than Cc, a
Mononobe-Okabe case with no solution, a numerical overflow or a division by
zero inside a calculation) now says so in its result, under `warnings`, on
every call; before, the message went to the server log and the agent never
saw it, so the agent can now report it to the user. The numbers do not
change. Report ingest's recompute rows (`module_warnings`) carry these too.
**Allowed values:** a tool given a value outside the values it lists for a
parameter (e.g. `trend_type="exponential"`, a calc package `format="docx"`
or `"PDF"`, an axial pile `method="Beta"`, a sheet pile
`pressure_method="Coulomb"`) is now refused with the list and a
did-you-mean, where before many such values silently ran as something else
(a straight-line trend labelled exponential, HTML written into a .docx
file, the Nordlund/alpha method instead of beta, Rankine instead of
Coulomb, a zero-deflection pile head, a local FOS with no strength
reduction). Calls with a listed value are unchanged; spellings the code
already accepted (case where it ignores case, "logspiral", "2to1", the calc
packages' older selector names, the liquefaction method aliases) are now
listed and still accepted. This is an aec-agent-core change (method_call.py,
call_warnings.py, hooks.py, plot_tool.py), so it ships with a core release.

## Noted, not fixed (piece 2)

- `pct_saturation_time`, `axis` (lateral pile), `hammer_type` and several
  others are read only on one path (with `drainage_quality` and no `cd`;
  for an H-pile; for the diesel warning); off that path any listed value is
  ignored. Enforcement does not change that.
- `ngamma_method` at phi = 0 returns before checking the name, so before
  this change a made-up name ran silently; it is now refused at dispatch.
- fem2d `n_gp` converts with `int()`, so 3.9 still runs as 3 (a float is a
  different kind from the list's ints and is left to the method).
- `log_linear` in `compute_trend` fits ln(value) against ln(depth), a power
  law; its description is a matter for wave 2e (subsurface trends).
