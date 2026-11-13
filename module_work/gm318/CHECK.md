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
- `axial_pile_capacity`'s returns text does not list `warnings`, though the
  result has carried them since before this task.
