# GM-113: the rest of the refusal hints (CHECK)

Branch `claude/gm-113-refusal-hints`, from master `1d39240`. BACKLOG GM-113
names five things; each is below. **No calculated value changes**: every
call that ran before and still runs gives the same numbers (before/after
table at the end). What changes is which calls are refused, and what a
refusal says.

## What the print says

Only item 5 touches a method basis, and only to name it in the refusal:

- The depth beta (`beta_method='depth'`) is O'Neill and Reese (1999) as
  FHWA NHI-06-089 prints it, Eqs. 9-41 to 9-43 (beta = 1.5 - 0.135 sqrt(z,
  ft), 0.25 to 1.2, times N60/15 where N60 < 15): it takes depth and N60,
  nothing else. These are the cites the module and its tool text already
  carry (`drilled_shaft/capacity.py`, `DEPTH_BETA_BASIS`; the adapter's
  `beta_method` description).
- (N1)60 enters only the GEC-10 (2018) beta, as phi' = 27.5 + 9.2 log
  (N1)60 (GEC-10 Example 10-1; Eq. A-2), and OCR only as Eq. 10-9's OCR
  (Eqs. 10-8 to 10-13, pp. 10-10 to 10-12), the cites `_gec10_beta`
  already carries.

Items 1 to 4 are tool behaviour with no printed basis needed.

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

| Method | Name | Finding | What I did |
|---|---|---|---|
| fem2d_local_fos | gamma_w | declared (a shared backfill table), never passed to `analyze_slope_srm`; the run used 9.81 kN/m3 | stopped advertising; refused with the reason; 9.81 accepted |
| fem2d_local_fos | max_iter | same; the run used 1000 | same; 1000 accepted |
| fem2d_local_fos | tol | same; the run used 1e-5 | same; 1e-5 accepted |
| fem2d_local_fos | t | same; the run used 1 m (plane strain) | same; 1 accepted |
| fem2d_local_fos | n_gp | same; the run used the element's own rule (3 points for T6) | same; 3 accepted |

None of the five is computed with under another name by the handler, so
each is refused (`fem2d_adapter.LOCAL_FOS_REFUSED`, as the method's
`refused_parameters`; the handler refuses them too when called directly,
through the new `adapters.refuse_given`). The reason says
`fem2d_slope_srm` takes them. Wiring them through to `analyze_slope_srm`
(which does take them) would be new behaviour, so it is left: noted below.
The "two" GM-110 found (`excavation_width` / `supports` on the SOE
excavations) were already refused by GM-110 and stay so.

What the audit cannot see: a name the handler reads and then does not use
(items 4 and 5 are of that kind). The GM-160 guard in
`test_defaults_used.py` covers that for defaulted inputs.

### 4. yield_acceleration refuses kh

The method finds kh itself (bisection on the pseudo-static FOS between 0
and `kh_max`), and the module overwrites the geometry's kh at every trial,
so a given kh changed nothing (0.35129 with kh 0, 0.1 or none below).
`kh` is no longer in the method's parameters; it is in its
`refused_parameters` with the reason, and the handler refuses it too. A kh
of 0 (no seismic coefficient) passes. `analyze_slope` and the other slope
methods keep their kh.

### 5. drilled_shaft refuses N1_60 and OCR under beta_method='depth'

The module's own pattern for an input the run would not use is a refusal
(GM-179, GM-249, the GSI/m_i check `_check_hoek_brown_layers`), so this is
a refusal, not a warning: `DrillShaftAnalysis._check_depth_beta_layers`,
run by `compute()` and `capacity_vs_depth()` (so `drilled_shaft_capacity`,
`capacity_vs_depth`, `lrfd_capacity` and the calc package all refuse),
naming the layer by depth, both cites, and what to do. It is NOT run by the
alternatives a rational run computes (`_compute`), so a rational run's
"beta_method 'depth'" alternative is unchanged (tested). The layer's
`N160` alias is refused as N1_60. The adapter's layer description says so.

## Before and after (made-up inputs; `/tmp/claude-0/sp/ba.py` in the session)

| Call | master | this branch |
|---|---|---|
| yield_acceleration, no kh | ky 0.35129 | ky 0.35129 |
| yield_acceleration, kh 0 | ky 0.35129 | ky 0.35129 |
| yield_acceleration, kh 0.1 | ky 0.35129 (kh ignored) | refused, with the reason |
| drilled_shaft_capacity, depth beta, two sands | Q_ult 4976.3 kN | 4976.3 kN |
| same, OCR 2 on layer 1 | 4976.3 kN (ignored) | refused |
| same, N1_60 18 on layer 1 | 4976.3 kN (ignored) | refused |
| fem2d_local_fos, slope with gwt | FOS 1.206, min local 1.1668 | same |
| same, gamma_w 10 | 1.206 / 1.1668 (ignored) | refused |
| same, tol 1e-3 | 1.206 / 1.1668 (ignored) | refused |
| analyze_slope with ru at the top level | refused, no hint | refused, names soil_layers[i].ru |
| plot_plan_view with `parameter` | refused, no did-you-mean | did you mean parameter_for_color |

## Tests (pytest exit codes; Python 3.12 venv)

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

Test updates: `method_probes.py` loses the `fem2d_local_fos` entries for
`t`, `max_iter` and `tol` (names it no longer declares);
`test_defaults_used.py`'s local_fos case now says gamma_w is NOT listed;
`drilled_shaft/tests/test_drilled_shaft.py::test_high_level_beta_rational_vs_depth`
ran the depth beta on a layer with N1_60 21; its depth run now leaves N1_60
out (the rational run keeps it), same assertions.

User manual: hand-edited rows only (`docs/user_manual/manual.html`): the
drilled-shaft `layers` description (5 copies), the `fem2d_local_fos`
parameter table (five rows out; count 20 -> 15) and the
`yield_acceleration` table (`kh` out; count 27 -> 26). No worked example
in `gen_worked_examples.py` / `manual_content.py` uses the refused inputs.

## Release note

No calculated value changes. Three inputs that were accepted and silently
ignored are now refused with the reason: `kh` on `yield_acceleration`
(the method finds kh; 0 still accepted), a layer's `N1_60` or `OCR` on the
drilled-shaft tools under `beta_method='depth'` (that beta uses neither;
give phi for a sand layer, or use the default rational beta), and
`gamma_w`, `max_iter`, `tol`, `t` and `n_gp` on `fem2d_local_fos` (it ran
with the module's own values; those values are still accepted). A layer
key given at the top level is refused with a pointer to the layer list,
and a guess that is the first word(s) of a parameter name gets that name
as a did-you-mean.

## Noted, not fixed

- `fem2d_local_fos` could pass `gamma_w`, `max_iter`, `tol`, `t` and `n_gp`
  through to `analyze_slope_srm`, which takes them, as `fem2d_slope_srm`
  does; that is new behaviour, so it is refused instead. It would change a
  saved output only for a call that sets them, which today is refused.
- Under `beta_method='depth'` a cohesionless layer still has to give phi
  (or N1_60, now refused) to pass the layer check, though the depth beta
  uses neither; and `sigma_v_ref` / `sigma_p_correlation` are as unused as
  N1_60 / OCR there and are not refused (GM-113 names only the two).
- Calc packages do not carry `ITEM_KEYS`, so a layer key at a package's top
  level gets the plain refusal without the list hint.
- The two pre-existing master failures above (pile-group figure,
  plot_data logx/logy on this matplotlib).
