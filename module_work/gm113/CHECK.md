# GM-113: the rest of the refusal hints (CHECK)

Branch `claude/gm-113-refusal-hints`, from master `1d39240`. BACKLOG GM-113
names five things; each is below. **No calculated value changes** for a
call that leaves out what it did not set: every such call that ran before
and still runs gives the same numbers (before/after table at the end).
What changes is which calls are refused, what a refusal says, and (after
review) that a `fem2d_local_fos` call that sets `t`, `gamma_w`,
`max_iter`, `tol` or `n_gp` now runs with that value instead of ignoring
it.

## What the print says

Only item 5 touches a method basis, and only to name it in the tool text:

- The depth beta (`beta_method='depth'`) is O'Neill and Reese (1999) as
  FHWA NHI-06-089 prints it, Eqs. 9-41 to 9-43 (beta = 1.5 - 0.135 sqrt(z,
  ft), 0.25 to 1.2); its N60/15 reduction where N60 < 15 is from O'Neill
  and Reese (1999), which NHI-06-089 does not print. It takes depth and
  N60, nothing else. These are the cites the module and its tool text
  already carry (`drilled_shaft/capacity.py`, `DEPTH_BETA_BASIS`; the
  adapter's `beta_method` description).
- (N1)60 enters only the GEC-10 (2018) beta, as phi' = 27.5 + 9.2 log
  (N1)60 (GEC-10 Example 10-1; Eq. A-2), and OCR only as Eq. 10-9's OCR
  (Eqs. 10-8 to 10-13, pp. 10-10 to 10-12), the cites `_gec10_beta`
  already carries. A depth-beta run computes that beta too, as the
  "beta_method 'rational'" alternative in its `choices`
  (`capacity.py`, `self._alternative(beta_method="rational")`), so there
  they are used.

Items 1 to 4 are tool behaviour with no printed basis needed.

## After review (REVIEW_GM-113.md, MERGE AFTER FIXES)

Three findings, all taken:

1. **Drilled shaft: N1_60 and OCR are no longer refused under the depth
   beta.** The refusal rested on "the run does not use them"; the result
   does, in its rational alternative. Measured (made-up two-sand profile,
   depth beta): Q_ult 4,976.3 kN in every case; the rational alternative's
   Q_ult is 3,566 kN with neither, 3,278 kN with OCR 2 on layer 1, and
   3,716 kN with N1_60 18 and no phi. `_check_depth_beta_layers` and its two
   calls are gone (`capacity.py` is master's again), and so is the
   `test_drilled_shaft.py` edit. The `layers` text now says where they
   enter: "under beta_method 'depth' the depth beta uses neither N1_60 nor
   OCR: they enter only the result's beta_method 'rational' alternative
   (choices)". Tests: a depth run with OCR 2 (or N1_60 18 without phi)
   runs, gives the same Q_ult, and a different rational alternative.
2. **fem2d_local_fos passes t, gamma_w, max_iter, tol and n_gp to
   `analyze_slope_srm`** as `fem2d_slope_srm` does, with the same defaults
   (1.0, 9.81, 1000, 1e-5, None; `analyze_slope_srm`'s own), so a call that
   leaves them out is unchanged; `n_gp` is checked to be 3 or 6. Refusing
   them had dropped a true `defaults_used` line (gamma_w 9.81 with a water
   table) and left a local-FOS run no way to raise `max_iter`. The
   METHOD_INFO entries are back and read as `fem2d_slope_srm`'s (its
   hand-written `max_iter` and `n_gp` text; `t`, `gamma_w` and `tol` from
   the shared backfill, as before); `refused_parameters` is gone. The
   `test_defaults_used.py` and `method_probes.py` edits are reverted, so
   gamma_w is listed in `defaults_used` again for a run with a water table.
   Tests: a stubbed `analyze_slope_srm` receives the five given values,
   and the analysis's own defaults when none is given.
3. **`audit_unread.py` puts the checkout it is run from on `sys.path`**,
   so it audits that tree rather than an installed copy. Run as documented
   from the repo root: this branch **0 names**; with master's
   `fem2d_adapter.py` swapped in, the same 5 (`fem2d_local_fos` gamma_w,
   max_iter, n_gp, t, tol); no probe failures.

## What changed and why

### 1. A list item's key given at the top level names its list

`dispatch._item_key_hint`, called from `_check_parameter_names` (GM-110's
check) when it refuses an unknown top-level name. Each adapter that has
list parameters now declares `ITEM_KEYS = {method: {param: keys}}` (built
with the new `adapters.item_keys_for` / `merge_item_keys` from the same key
constants its nested checks use, so the two cannot drift). An unknown
top-level name that is a key of an item of one of the method's lists gets,
after the usual message:

> 'ru' is a key of each item of soil_layers, not a parameter of its own:
> give it in each item (soil_layers[i].ru).

Nothing is moved into the list (a top-level value would stand for every
item, which only the caller can say). Declared for: slope_stability
(`soil_layers`), axial_pile, drilled_shaft, sheet_pile, pavement_design,
lateral_pile (every p-y model's keys), opensees and pystrata (every
material's keys) (`layers`), downdrag (`layers`, `liquefied_layers`), soe
(`layers`, `supports`), settlement (`layers` per method,
`consolidation_layers`, `schmertmann_layers`), fem2d (`soil_layers` per
analysis, `material_props`). Not declared: calc packages (they check through
their tool's names; noted below), and lists whose keys are all top-level
names anyway (pile_group's uplift `layers`).

### 2. Prefix matches

`arg_checks.suggest_params` (used by every unknown-name refusal, top-level
and nested) now, when nothing closer matched, offers the valid names whose
leading WHOLE words are the guess: `parameter` -> `parameter_for_color`.
Only a guess of four letters or more, and only at an underscore boundary,
so `E` is never offered `E_s` and `para` is offered nothing (the existing
tests of those cases still pass).

### 3. The reverse audit: names METHOD_INFO declares that no handler reads

Script: `module_work/gm113/audit_unread.py` (run from the repo root). Pass
1 runs every method probe (`funhouse_agent/tests/method_probes.json`, 244
of them here) with the parameters wrapped in a dict that records every key
looked up, present or not. A handler that copies the dict hides its later
reads, so pass 2 takes each name pass 1 did not see and looks for a read
of it in the handler and the adapter functions it calls, or in the
signature of a function it passes `**params` to. Methods without a probe
get pass 2; reference functions are already refused by their own
signature (GM-110); a calc package's names are checked against its tool's
METHOD_INFO (all are the tool's, its header fields, its own or its
refused names).

Result on master: **5 names, all on one method**; on this branch: 0.

| Method | Name | Finding | What is done (after review) |
|---|---|---|---|
| fem2d_local_fos | gamma_w | declared (a shared backfill table), never passed to `analyze_slope_srm`; the run used 9.81 kN/m3 | passed through, default 9.81 |
| fem2d_local_fos | max_iter | same; the run used 1000 | passed through, default 1000 |
| fem2d_local_fos | tol | same; the run used 1e-5 | passed through, default 1e-5 |
| fem2d_local_fos | t | same; the run used 1 m (plane strain) | passed through, default 1 |
| fem2d_local_fos | n_gp | same; the run used the element's own rule (3 points for T6) | passed through (3 or 6), default None: the element's own rule |

`analyze_slope_srm` takes all five, so the handler now passes them as
`fem2d_slope_srm`'s does, with `analyze_slope_srm`'s own defaults: a call
that leaves them out runs exactly as before. (The branch first refused
them; the review showed that dropped a true `defaults_used` line and left
no way to raise `max_iter` for a local-FOS run.) The "two" GM-110 found
(`excavation_width` / `supports` on the SOE excavations) were already
refused by GM-110 and stay so.

What the audit cannot see: a name the handler reads and then does not use
(item 4 is of that kind; item 5 turned out not to be). The GM-160 guard in
`test_defaults_used.py` covers that for defaulted inputs.

### 4. yield_acceleration refuses kh

The method finds kh itself (bisection on the pseudo-static FOS between 0
and `kh_max`), and the module overwrites the geometry's kh at every trial,
so a given kh changed nothing (0.35129 with kh 0, 0.1 or none below).
`kh` is no longer in the method's parameters; it is in its
`refused_parameters` with the reason, and the handler refuses it too
(through the new `adapters.refuse_given`, for a handler called directly).
A kh of 0 (no seismic coefficient) passes. `analyze_slope` and the other
slope methods keep their kh.

### 5. drilled_shaft: N1_60 and OCR under beta_method='depth' are said, not refused

The branch first refused them (`DrillShaftAnalysis._check_depth_beta_layers`).
The review showed the premise false: the depth beta uses neither, but a
depth run's `choices` carry the "beta_method 'rational'" alternative,
computed with the layers' OCR and N1_60 (numbers under "After review"). As
with GSI and m_i on a rock layer below the base (GM-251: accepted where an
alternative uses them), they are taken. The check is gone; no module code
changes. What changes is the `layers` text of the three drilled-shaft
tools and the calc package: "under beta_method 'depth' the depth beta uses
neither N1_60 nor OCR: they enter only the result's beta_method 'rational'
alternative (choices)", so a caller is told where they go.

## Before and after (made-up inputs; `/tmp/claude-0/sp/ba.py` in the session)

| Call | master | this branch |
|---|---|---|
| yield_acceleration, no kh | ky 0.35129 | ky 0.35129 |
| yield_acceleration, kh 0 | ky 0.35129 | ky 0.35129 |
| yield_acceleration, kh 0.1 | ky 0.35129 (kh ignored) | refused, with the reason |
| drilled_shaft_capacity, depth beta, two sands | Q_ult 4976.3 kN; rational alternative 3,566 kN | same |
| same, OCR 2 on layer 1 | 4976.3 kN; rational alternative 3,278 kN | same |
| same, N1_60 18 and no phi on layer 1 | 4976.3 kN; rational alternative 3,716 kN | same |
| fem2d_local_fos, slope with gwt | FOS 1.206, min local 1.1668 | same |

After review, `fem2d_local_fos` on a second made-up slope (c 25 kPa, phi
32 deg, gwt at 2 m; `gm113fix/lfos_ba.py` in the session scratchpad).
Master ignored the five, so its column is the plain run's:

| Call | master | this branch |
|---|---|---|
| no gamma_w / tol given | FOS 1.719, min local 1.5459; gamma_w 9.81 in defaults_used | same |
| gamma_w 10 | 1.719 / 1.5459 (ignored) | 1.706 / 1.5360 |
| tol 1e-3 | 1.719 / 1.5459 (ignored) | 1.719 / 1.5459 |
| analyze_slope with ru at the top level | refused, no hint | refused, names soil_layers[i].ru |
| plot_plan_view with `parameter` | refused, no did-you-mean | did you mean parameter_for_color |

## Tests (pytest exit codes; Python 3.12 venv)

After review (local Windows venv, `-p no:cacheprovider`):

- `funhouse_agent/tests/test_gm113_refusal_hints.py`,
  `drilled_shaft/tests/test_depth_beta_inputs.py`,
  `drilled_shaft/tests/test_drilled_shaft.py`,
  `funhouse_agent/tests/test_slope_stability_adapter.py` and
  `funhouse_agent/tests/test_local_fos_adapter.py` (not slow),
  `webapp/tests/test_page_fingerprints.py`,
  `webapp/tests/test_public_text.py`: 201 passed, 1 deselected; exit 0.
  Fingerprints did not move, so no re-record commit.
- `funhouse_agent/tests/test_defaults_used.py` (all): 822 passed, 14
  skipped; exit 0.

Before review (cloud session):

- `funhouse_agent/tests` (not slow, without test_defaults_used): 3362
  passed, 9 skipped, **1 failed**:
  `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`,
  which **fails the same way on master** (checked with the change stashed).
- `funhouse_agent/tests/test_defaults_used.py` (all, incl. the method
  probes): 810 passed, 23 skipped, 3 failed before the two allow-list
  updates below; after them, the two GM-113 ones pass. The third,
  `test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
  (logx/logy change nothing), **fails the same way on master**. Final run
  of test_defaults_used + test_gm113_refusal_hints + test_top_level_params
  + drilled_shaft + fingerprints + public text: 1116 passed, 23 skipped,
  1 failed (that plot_data case).
- New: `funhouse_agent/tests/test_gm113_refusal_hints.py` 25 passed;
  `drilled_shaft/tests/test_depth_beta_inputs.py` 6 passed.
- `drilled_shaft`, `slope_stability`, `calc_package` (not slow),
  `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
  1133 passed, 20 skipped, 1 failed before the fix below, 0 after.
- `validation_examples`: 339 passed, 1 xfailed.
- Fingerprints did not move (METHOD_INFO text is not fingerprinted), so no
  re-record commit.

Test updates (after review): `method_probes.py`, `test_defaults_used.py`
and `drilled_shaft/tests/test_drilled_shaft.py` are master's again (the
branch's edits followed from the two refusals now dropped).
`drilled_shaft/tests/test_depth_beta_inputs.py` and sections 3 and 5 of
`test_gm113_refusal_hints.py` now test the pass-through and the depth-beta
alternative instead of refusals.

User manual (`docs/user_manual/manual.html`): only the changed rows,
re-rendered with `build_manual`'s own `extract_catalog`,
`method_param_table` and `methods_overview_table` (not a full rebuild): the
drilled-shaft `layers` description (5 copies, the review's wording), the
`yield_acceleration` table (`kh` out; count 27 -> 26), and the
`fem2d_local_fos` table, back to 20 parameters (its `max_iter` and `n_gp`
rows now read as `fem2d_slope_srm`'s). No worked example in
`gen_worked_examples.py` / `manual_content.py` uses the changed inputs.

## Release note

No calculated value changes for a call that leaves out what it did not
set. Refused now, with the reason: `kh` on `yield_acceleration` (it was
accepted and ignored; the method finds kh itself; a kh of 0 is still
accepted). Running now with the value given: `t`, `gamma_w`, `max_iter`,
`tol` and `n_gp` on `fem2d_local_fos`, which `fem2d_local_fos` accepted
and ignored before (the run used 1 m, 9.81 kN/m3, 1000, 1e-5 and the
element's own Gauss rule, the same defaults it uses now; `n_gp` must be 3
or 6). Still run, now described: a layer's `N1_60` and `OCR` on the
drilled-shaft tools under `beta_method='depth'` (the depth beta uses
neither; they enter the result's rational-beta alternative in `choices`).
A layer key given at the top level is refused with a pointer to the layer
list, and a guess that is the first word(s) of a parameter name gets that
name as a did-you-mean.

## Noted, not fixed

- Under `beta_method='depth'` a cohesionless layer still has to give phi
  (or N1_60) to pass the layer check, though the depth beta uses neither
  (pre-existing); `sigma_v_ref` and `sigma_p_correlation`, like N1_60 and
  OCR, enter only the rational alternative there.
- Calc packages do not carry `ITEM_KEYS`, so a layer key at a package's top
  level gets the plain refusal without the list hint.
- The two pre-existing master failures above (pile-group figure,
  plot_data logx/logy on this matplotlib).
