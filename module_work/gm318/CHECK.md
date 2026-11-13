# GM-318 CHECK: piles longer than the soil profile, and failures that vanish

Branch `claude/gm-318-piles-short-profile`, from master 2feea3d. Modules:
`axial_pile`, `drilled_shaft`, `lateral_pile`. Brief:
`module_work/cloud_briefs/AUDIT_STAGE2.md`, section GM-318.

## What the print says

| Manual | Where | What it prints |
|---|---|---|
| GEC-12 (FHWA-NHI-16-009) Vol I | Sec. 4.2.3.2, pp. 86-87 | "Minimum exploration depths should extend 20 feet below expected pile toe elevation, or at least two times the pile group width. For piles on rock, at least 10 feet of rock coring should be performed." |
| GEC-12 Vol I | Sec. 7.2.1.3.1, Step 7c | the toe's phi from "the zone from the pile toe to 3 diameters below the pile toe" |
| GEC-10 (FHWA-NHI-18-024) | Eq. 10-14, p. 10-15 | sand base: N60 "the average value between the base and two diameters beneath the base" |
| GEC-10 | Eq. 10-19, p. 10-19 | clay base: su "over a depth of 2B below the base" |
| GEC 9, COM624P | searched the held text | nothing on p-y layers that do not reach the toe or the ground surface |

Each method takes its toe or base resistance from the ground at and below
the tip, and GEC-12 has that ground explored. No held manual prints carrying
the last explored layer on below the data, for axial or lateral loading. So
by the brief's rule the extension is not a default.

## Findings taken

| Audit row | Confirmed on master 2feea3d | What I did |
|---|---|---|
| axial_pile #2 (A): toe below the profile, Python warning only (`capacity.py:341-346`) | Yes. The run gave no side resistance below the profile and put the toe in the last layer. | **Refused** by default, naming the depth; `extend_last_layer` runs it and the result says so (warnings, choices). |
| axial_pile #1 (A): `capacity_vs_depth` drops failed depths (`capacity.py:2015-2018`) | Yes. | **Fixed**: each refused length is named with its reason (`skipped`; the tool's `skipped_depths` and `warning`); all refused is a refusal. |
| axial_pile #3 (B, same function): the tool's returns text gave the wrong row keys | Yes. | **Fixed**: the real keys. |
| drilled_shaft #1 (A): shaft longer than the profile, Python warning only; no `warnings` field (`capacity.py:981-985`) | Yes. | **Refused** by default; `extend_last_layer`; `DrillShaftResult.warnings` added. |
| lateral_pile #1 (A): layers start below the ground or stop above the toe, Python warning only (`soil.py:152-161`) | Yes. The run gave p = 0 there. | **Refused**, both cases. `extend_last_layer` carries the last layer to the toe and says so. Layers that start below the ground have no override. |
| lateral_pile #2 (A): cracked-EI loop's non-convergence only a Python warning (`analysis.py:514-518`) | Yes. | **Fixed**: `ei_converged`, `ei_iterations`, `ei_last_change`, `ei_tolerance` in the result, and a warning in `warnings` when it did not converge. |
| drilled_shaft #2 (uplift wording) | not opened | **Not done**: the brief leaves it to DOC-24. |
| drilled_shaft #3 (B), lateral_pile #3, #4, #5 (B) | not opened | **Not done**: not in the functions this task changed (see "Noted, not fixed"). |

## What changed

- **`axial_pile/capacity.py`**: `AxialPileAnalysis.extend_last_layer`
  (default False). A toe (`head_depth + pile_length`) below the profile
  raises a `ValueError` that names both depths and says the toe would bear
  in soil nobody explored. With `extend_last_layer=True` the last layer's
  thickness is extended to the toe (side resistance and toe in it); the
  result's first warning and a choice "soil below the profile" say so.
  `capacity_vs_depth(skipped=...)` names refused lengths
  (`skipped_depths_note`), and its default deepest length is the profile's
  depth below the pile head (it was the whole profile depth, which a buried
  head would now put below the profile).
- **`drilled_shaft/capacity.py`, `results.py`**: the same refusal and input
  on `DrillShaftAnalysis`; `DrillShaftResult.warnings` (in `to_dict()` and
  `summary()` when not empty). `capacity_vs_depth` already listed refused
  lengths; a length past the profile is now one of them.
- **`lateral_pile/soil.py`, `analysis.py`, `results.py`**:
  `validate_layers` refuses layers that start below the ground surface or
  stop above the toe; `LateralPileAnalysis(..., extend_last_layer=True)`
  extends the last layer's bottom to the toe. `Results.warnings`,
  `ei_converged`, `ei_last_change`, `ei_tolerance`.
- **Adapters** (`funhouse_agent/adapters/axial_pile.py`,
  `drilled_shaft.py`, `lateral_pile.py`): `extend_last_layer` on
  `axial_pile_capacity`, `capacity_vs_depth` (both modules),
  `drilled_shaft_capacity`, `lrfd_capacity` and `lateral_pile_analysis`,
  and through them on the three calc packages. Descriptions say what the
  default does. `applies_when` False, so it is never listed in
  `defaults_used` (its default refuses the only case that reads it).
- **Calc packages** (`*/calc_steps.py`,
  `funhouse_agent/adapters/calc_package.py`): a first section "Soil Below
  the Profile (Assumed)" when the last layer was carried on; the tool's
  result carries `warnings`; the lateral package's EI step prints the
  convergence record.
- **Docs**: each module's `DESIGN.md`; `docs/user_manual/manual.html`: the
  parameter tables and parameter counts of the changed methods only (17
  lines; taken from an in-memory rebuild, one unrelated differing line left
  as it was).
- **Fingerprints**: `python -m webapp.tests.page_builds record` wrote the
  same file; `test_page_fingerprints.py` passes with no re-record.

## Before and after (made-up values)

0.324 m closed pipe pile; sand 0-4 m (18 kN/m3, 30 deg), sand 4-10 m
(19 kN/m3, 34 deg); 1.0 m shaft in clay 0-5 m (cu 60 kPa) over clay 5-12 m
(cu 90 kPa); 0.6 m pipe pile, 15 m, Matlock soft clay given 0-10 m, Vt 80 kN.

| Run | Before (master) | After, default | After, `extend_last_layer=true` |
|---|---|---|---|
| axial pile 12 m, profile to 10 m | Q_ult 685.9 kN (Q_s 402.6, Q_t 283.3), no warning: the 10 m pile's answer | refused: "profile ends at 10 m ... toe is at 12 m" | 875.7 kN (Q_s 592.4, Q_t 283.3), 1 warning; equals the profile given to 12 m |
| drilled shaft 15 m, profile to 12 m | 1987.6 kN (Q_s 1359.9, Q_b 627.7), no warning | refused | 2404.4 kN (Q_s 1776.7, Q_b 627.7), 1 warning |
| lateral pile 15 m, layers to 10 m | y_top 5.86 mm, M_max 108.0 kN-m, no warning (p = 0 below 10 m) | refused | y_top 5.86 mm, M_max 108.0 kN-m, 1 warning (the pile is long; the last 5 m hardly matter here) |
| axial `capacity_vs_depth`, 6 to 14 m in 5 | rows at 12 and 14 m both 685.9 kN | rows 6, 8, 10 m; `skipped_depths` 12 and 14 m with the reason | |
| axial pile 10 m, profile to 10 m | 685.9 kN | 685.9 kN (unchanged) | |

## Every probe, worked example and validation example that relied on the silent extension

None found. After the default became a refusal I ran the three modules'
suites, `calc_package`, `funhouse_agent` (the method probes, the worked
examples run by execution, the calc-package tests), `validation_examples`,
`foundry_test_harness`, `pile_group`, `downdrag`, `wave_equation`,
`geotech_common` and `reliability`: nothing was refused. I also read the
inputs of `docs/user_manual/gen_worked_examples.py`, `generate_all_pdfs.py`
and `docs/gallery/build_gallery.py`: each profile covers its pile.

## Calls made (for the owner to revisit)

1. **The default is a refusal** for a pile toe or shaft base below the
   profile (axial_pile, drilled_shaft) and for p-y layers that stop above
   the toe (lateral_pile). `extend_last_layer=true` is the one input that
   runs it. The other form, kept behaviour with a warning, was the old
   default.
2. **What the extension is.** With `extend_last_layer` the last layer
   really goes on to the toe: side resistance (or p-y resistance) in it as
   well as the toe. The old silent basis was different (axial and drilled:
   no side resistance below the profile, toe in the last layer; lateral:
   p = 0 below the layers) and no input reproduces it. A profile with a
   weak or zero-strength layer written in does.
3. **Layers that start below the ground surface (lateral) are refused with
   no override.** `stickup` is the input for a free length; for scour the
   message says to measure depths from the new surface. The other form
   would be an input that keeps p = 0 above the first layer.
4. **A toe exactly at the profile's bottom is accepted** with no warning,
   though GEC-12 would have the exploration go 20 ft deeper. Refusing it
   would refuse most profiles as they are usually typed. Drilled shaft's
   base choices already say "the mean over what the profile gives" for the
   2B zone; axial `toe_phi_zone` "average_3b" (an option) still takes the
   last layer through its 3b zone, as its docstring says.
5. **A cracked-EI run that did not converge still returns its numbers**,
   flagged (`ei_converged` False, a warning), as the p-y iteration does
   with `converged` False.

## Tests

Venv per COMMON_MODULES (Python 3.12, planlens 0.15.1), `pytest -n 4`.

| Run | Result |
|---|---|
| `axial_pile drilled_shaft lateral_pile calc_package funhouse_agent validation_examples foundry_test_harness pile_group downdrag` (full change) | 7171 passed, 37 skipped, 1 xfailed, 2 failed (exit code 1; the 2 below) |
| the same plus `wave_equation geotech_common reliability` (after the first commit) | 7545 passed, 37 skipped, 1 xfailed, 2 failed |
| new: `axial_pile/tests/test_gm318.py` (10), `drilled_shaft/tests/test_gm318.py` (6), `lateral_pile/tests/test_gm318.py` (7), `funhouse_agent/tests/test_gm318_packages.py` (3) | 26 passed |
| `webapp/tests/test_page_fingerprints.py test_public_text.py test_docs_currency.py` | 24 passed |
| `module_work/doc10/check_manual.py` | OK |

The 2 failures are on master too in this venv and are not from this change:
`test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
and `test_defaults_used.py::...[profile_figure|plot_data]` (logx/logy). I
ran both on master 2feea3d with my changes stashed: both fail the same way.

## Release note

A driven pile, drilled shaft or laterally loaded pile whose toe is below
the soil layers given is now refused, with the depth the layers do not
cover; before, it ran with only a Python warning that the agent never saw
(axial and drilled: no side resistance below the profile and the toe in the
last layer; lateral: no soil resistance below the layers). Set
`extend_last_layer` to take the last layer on to the toe; the result and
the calc package then say so, and the capacity is that of a profile given
to the toe (higher than the old silent answer by the side resistance below
the profile). Lateral p-y layers must start at the ground surface.
`capacity_vs_depth` for driven piles lists the trial lengths it could not
run instead of leaving them out. A reinforced-concrete pile's
cracked-stiffness iteration reports whether it converged. No value changes
for a pile inside its profile.

## Noted, not fixed

- drilled_shaft #3 (the calc package has no LRFD or capacity-vs-depth
  content), lateral_pile #3 (`head_condition='partial'` not on the tool),
  #4 (no adapter method builds a cracked RC pile, so the new EI record
  reaches the agent only through the library today), #5 (`SandLiquefied`
  not on the tool): B findings outside the functions changed.
- `drilled_shaft/shaft.py:106`: a casing deeper than the shaft raises only
  a Python warning (seen in the test log). Left to the branch that delivers
  Python warnings.
- `capacity_vs_depth` rows carry `depth_m` as a numpy float (both modules);
  it serialises as a number.
- `docs/user_manual/manual.html` line 1594 differs from a rebuild on master
  (asterisks read as emphasis in a soe description); not this task's row.
- `axial_pile_capacity`'s returns text did not list `warnings`, though the
  result has carried them since before this task: listed in the review
  round (C3).

## Review round (branch `fix/gm318-review`, 2026-11-13)

The independent review (MERGE AFTER FIXES: A 1, B 2, C 8) and the merge of
master (the dispatch fix, GM-316) were taken on `fix/gm318-review`, from
6a8b64c (the cloud head 018d991 with master merged in). Made-up values
throughout.

### What each finding became

| # | Finding | What it became | Commit |
|---|---|---|---|
| M1 | After the merge, master's `funhouse_agent/tests/test_probe_warnings.py` used the removed "exceeds soil profile" warning (4 tests failed) | The cases now raise warnings the modules still raise on allowed runs: drilled_shaft a casing deeper than the shaft (`shaft.py`), lateral_pile a Matlock clay with c above 100 kPa (`py_curves.py`). **axial_pile's case is out**: it raises no Python warning on an allowed tool run (its chart notes go to the result's `warnings`; the capacity-against-length note is passed back as `skipped`). The package test uses the drilled shaft package. New test: a returned warning (extend_last_layer) and a raised one are each said once in the result and the package (the returned one in its own section, not again under "Warnings Raised by the Calculation"). No module was given a warning to feed the test. | d67cc69 |
| A1 | Drilled `capacity_vs_depth` with `extend_last_layer` said nothing of the lengths past the profile | Each row carries its run's `warnings`; the tool's result has one `warnings` entry naming the lengths past the profile and the choice "soil below the profile" (base at the deepest such length); returns text and the package table's note say it. Each trial length now runs on the layers as given, so it equals a run of that length alone (see Calls made). | b41e7ec |
| B1 | Extended drilled run: the base choice and `base_steps` said the profile ends at the carried-on depth | `su_profile_ends_m` / `N60_profile_ends_m` give the depth the layers given reach; new `su_last_layer_carried_to_m` / `N60_last_layer_carried_to_m` give the base; the words read "as the profile ends at 12 m (its last layer carried on to the base at 15 m, extend_last_layer)". A run not extended reads as before. Tests for a clay and a sand base. | a42e088 |
| B2 | The lateral refusal of layers that start below the ground offered only the scour model | One sentence added to the message, the tool's `extend_last_layer` text, DESIGN.md and the manual's three rows of that parameter: stickup or depths from the new surface take the soil as removed, as GEC 9 (FHWA-HIF-18-031) Sec. 6.7.5, p. 91 takes the scour zone ("the entire soil depth within the scour zone should be considered to be removed by scour"; page opened in the PDF, printed p. 91, Sec. 6.7.5 starting p. 90); soil that stays in place (its weight kept) but is to resist nothing is different, and is given as a top layer of negligible strength from the ground surface. Arithmetic unchanged. Call 3 above stays the owner's to revisit. | 48c35a0 |
| C1 | A library caller's `profile_depth_given` deeper than the profile brought back the silent basis | Refused in `_cover_the_toe` and `_cover_the_base`. | f701ae3 |
| C2 | The axial package's capacity-against-length table did not pass `skipped`; all refused raised | It passes `skipped` and names refused lengths in the table note; with every length refused the table is left out and a warning (printed in the package and returned) says why. | 46e59c9 |
| C3 | Pile packages' and `axial_pile_capacity`'s returns texts did not list `warnings` | Listed in all four. | fe4f925 |
| C4 | `max_ei_iterations=0` reported a change of nan | Refused below 1. | 024d9e3 |
| C5 | The "toe friction angle" choice did not say the 3b zone's part below the profile is assumed | Added, for example "10.00-10.97 m is below the profile, which ends at 10 m: the last layer carried on there, assumed". | eb6768c |
| C6 | Soil table rows and the lateral p-y choice showed the carried-on layer as given | The last layer is marked "(given to 10 m, assumed below)" in the axial and lateral printed soil tables, the three packages' soil-layer CSVs and the lateral p-y choice. | 881a40d |
| C7 | Each skipped length repeated the whole refusal | "below the profile, which ends at 10 m" (with a buried head, "toe at 11 m, below the profile, ..."), axial and drilled curves. The warning for the review's 6-14 m case is now 171 characters. | 1f85a0a |
| C8 | `_IN_TRIAL` in `drilled_shaft/capacity.py` had no reader after the merge | Removed with its import; `_alternative`'s docstring reworded. | 6455e10 |

### Before and after (the review's made-up inputs, through `call_agent`)

| Case | Before this round (018d991 / 6a8b64c) | After |
|---|---|---|
| A1: drilled `capacity_vs_depth`, shaft 15 m, clay to 12 m, extended, 6 m in 4 | rows 1153.9, 1570.7, 1987.6, 2404.4 kN; result keys `capacity_vs_depth`, `defaults_used` only | same four numbers; the 15 m row has `warnings`; top `warnings` names "(15 m)"; choice "last layer taken to go on from 12 m to the base at 15 m" |
| B1 clay base, 15 m in 12 m, extended | Q_ult 2404.4 kN; "as the profile ends at 15 m"; `su_profile_ends_m` 15.0 | 2404.4 kN; "as the profile ends at 12 m (its last layer carried on to the base at 15 m, extend_last_layer)"; 12.0 and `su_last_layer_carried_to_m` 15.0 |
| B1 sand base, 14 m in 12 m, extended | Q_ult 3659.4 kN; "as the profile ends at 14 m"; `N60_profile_ends_m` 14.0 | 3659.4 kN; "ends at 12 m (its last layer carried on to the base at 14 m, ...)"; 12.0 and 14.0 |
| C7: axial `capacity_vs_depth` 6-14 m in 5 | two skipped lengths, each with the full refusal | same two lengths, reason "below the profile, which ends at 10 m" |
| C5: axial 10 m pile in 10 m, `average_3b` | 685.9 kN; zone listed with no word on the profile | 685.9 kN; "10.00-10.97 m is below the profile ... assumed" |
| C6: lateral 15 m pile, layers to 10 m, extended | 5.86 mm, 108.0 kN-m; "p-y curves 0-15 m" | 5.86 mm, 108.0 kN-m; "p-y curves 0-15 m (given to 10 m, assumed below)" |

**Covered cases did not move.** The review's replay (`p2_replay.py`: 41
calls, every `method_probes.json` entry of the three modules and three
pile packages, the `dispatch_calls` of WE-PILE-1, WE-SHAFT-1 and WE-SF-2,
the manual's three worked examples) was run on 6a8b64c before this round
and on the head after it: every call is the same; the only differences are
the packages' output folder names (two scratch folders).

### Calls made in this round (for the owner to revisit)

1. **A drilled capacity-against-length curve runs each length on the
   layers as given.** Before, with `extend_last_layer`, every trial ran on
   the profile carried on to the full shaft's base, so a shorter trial's 2B
   zone below its base could reach into the carried-on layer with nothing
   said. Now each row equals the run of that length alone. No covered case
   changes (with the base inside the profile nothing is carried on).
2. **All lengths refused in the axial package's table:** the table is left
   out and a Python warning says why; it reaches the tool's `warnings` and
   the package's "Warnings Raised by the Calculation" (GM-316). There is no
   table left to carry the note.
3. **C6 and C7 were done in both modules where the same text arises**
   (drilled skipped reasons; the drilled and lateral CSV soil tables), so
   the two curves and three packages read alike.
4. **The negligible-strength layer is named, not sized.** B2's sentence
   says what to give, not a strength value: no held page prints one.

### Tests (venv Python 3.14, one file at a time, `-q -p no:cacheprovider`)

| File | Result | Exit |
|---|---|---|
| `axial_pile/tests/test_gm318.py` | 14 passed | 0 |
| `drilled_shaft/tests/test_gm318.py` | 11 passed | 0 |
| `lateral_pile/tests/test_gm318.py` | 8 passed | 0 |
| `funhouse_agent/tests/test_gm318_packages.py` | 3 passed | 0 |
| `funhouse_agent/tests/test_probe_warnings.py` | 8 passed | 0 |
| `funhouse_agent/tests/test_adapter_ergonomics.py` | 170 passed | 0 |
| `funhouse_agent/tests/test_defaults_used.py -k "axial_pile or drilled_shaft or lateral_pile"` | 105 passed | 0 |
| `webapp/tests/test_page_fingerprints.py` | 13 passed (no re-record needed) | 0 |
| `webapp/tests/test_public_text.py` | 5 passed | 0 |
| `module_work/doc10/check_manual.py` | OK | 0 |
| `module_work/doc10/check_manual.py --rebuild` | differs only at line 1594 (the older soe line) | 1 |

### Release note (lines to add)

A drilled shaft capacity-against-length curve run with `extend_last_layer`
now says which lengths stand on the last layer carried on below the
profile: on each such row, once at the top of the result, and in a choice;
each length is run on the layers as given. In such a run the base
resistance choice names the depth the layers given reach (it gave the
carried-on depth). A pile toe's averaged friction angle, the packages' soil
tables and the lateral p-y choices mark the part of the ground that is
carried on below the profile. A trial length below the profile is skipped
with a short reason. A lateral run with layers that start below the ground
is refused, not reshaped; the message now says that stickup takes the soil
as removed (scour) and how to give soil that stays in place but resists
nothing. No value changes for a pile inside its profile.

### Follow-ups (each would change a saved report's output)

- **drilled shaft package with `casing_depth`:** the capacity-against-length
  table re-runs the shaft at each trial length, and every trial shorter
  than the casing raises "Casing depth (X m) exceeds shaft length (Y m)"
  (`drilled_shaft/shaft.py:106`). Since GM-316 each is a separate warning
  in the tool's result and in the package's "Warnings Raised by the
  Calculation": a 15 m shaft with a 5 m casing (allowed, ordinary) prints
  three, at trial lengths 3.0, 3.89 and 4.79 m (the last two unrounded,
  for example "3.8947368421052633m"). Found while writing M1's test; not
  changed here.
