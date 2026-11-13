# GM-320 and GM-321: settlement and support of excavation

Branch `claude/gm-320-321-settlement-soe`, from master 2feea3d. Audit rows:
`module_work/module_audit_2026-11-12/AUDIT.md` section 4, settlement rows 1
and 2 and soe row 1. All made-up values below.

## Findings taken

| Audit row | Confirmed on master? | What was done |
|---|---|---|
| settlement 1 (A): `consolidation_settlement` zips `layers` with `delta_sigmas`, no length check | Yes (`funhouse_agent/adapters/settlement.py`, the `zip(layers, delta_sigmas)` loop) | **Refused.** A list of another length (or not a list) raises, naming both counts. |
| settlement 2 (A): Schmertmann adds nothing below layers that stop above z_max, and says nothing | Yes (`settlement/immediate.py`, the clip `zb = min(layer.depth_bottom, z_max)`) | **Refused by default**, naming the depth not covered; one explicit input says what lies below, and the result states it (`choices`, `warnings`, the terms). |
| settlement 3 (B): `warnings.warn` (Cr > Cc) never reaches the agent | Not opened | Not done: the warnings task (`claude/gm-316-317-warnings-allowed-values`) owns it. |
| settlement 4, 5, 6 (B): Hough not in the combined analysis; no package for the separate methods; time-rate functions without a tool | Not opened | Not done: other functions, out of this section's scope. |
| soe 1 (A): supports' `spacing`, `angle_deg`, `preload_kN_per_m` accepted and never read | Yes (nothing in `soe/` read them; `SupportLevel.spacing` defaulted to 3.0) | **Spacing and angle used** (GEC-4 Eqs. 19 to 21); **preload refused** with a message (no held page prints how to use it). |
| soe 2 (A): blow-out unit weight wording | n/a | Left to DOC-24, as the brief says. |
| soe 5, 6, 7 (B) | Not opened | Not done: other functions. |

## What the print says

**Schmertmann.** FHWA NHI-06-089 Sec. 8.5.1 (pp. 8-45 to 8-51): Eq. 8-16
sums over "n = number of soil layers within the zone of strain influence
(strain influence diagram)" (p. 8-45). Example 8-2, step 2 (p. 8-49) sets
the maximum depth of influence (2Bf at Lf/Bf = 1, 4Bf above 10,
interpolated) and step 5 (p. 8-50) divides "the influence depth of 16 ft"
into layers down to its bottom. UFS 3-220-10 Sec. 5-3 (Eq. 5-7, Fig. 5-7):
Iz "decreases to zero at a depth of 2B for square footings and 4B for
continuous footings". Neither prints what to do when the explored profile
stops above that depth, nor a diagram redrawn for a shallow rigid layer
(searched the held text of NHI-06-089 ch. 8, UFS 3-220-10 ch. 5 and GEC-6
ch. 5 for rigid, incompressible, rock and depth-of-influence wording near
the method). So: no printed rule, and the brief's default applies.

**Support loads.** GEC-4 (FHWA-IF-99-015) Sec. 5.3.3, "Calculation of
Ground Anchor Loads from Apparent Earth Pressure Diagrams":
- Eq. 19 (p. 66): Th = Thi s, Thi the horizontal anchor load per unit width
  of wall (Figs. 33, 34), s the horizontal spacing between anchors;
- Eq. 20 (p. 67): T = Th / cos theta, the anchor design load, theta the
  inclination below the horizontal;
- Eq. 21 (p. 67): Tv = T sin theta, the vertical component.
GEC-4 App. A Design Example 1 uses them (TH1 168 kN/m, 2.5 m, 15 deg:
DL1 435 kN), the numbers `soe.fhwa_apparent_pressure` is already pinned to.
GEC-4 prints these for ground anchors; an inclined strut or raker is the
same statics.

**Caltrans Trenching and Shoring Manual.** The held text
(`geotech_references/california_trenching/text/`, searched for strut
spacing, load per strut, raker inclination and preload) prints no spacing
or inclination rule for struts that I could find; its held chapters are
summaries without page marks for this. The cites are therefore GEC-4 only.
The brief asked for both manuals: this half is **not found in the held
text**, not confirmed absent from the manual.

**Preload.** GEC-4 Sec. 7.5 "Anchor Lock-off Load" (pp. 154-155): the
lock-off load is "generally 75 to 100 percent of the design load from
apparent envelopes". That selects a lock-off load from the design load; it
does not change the design load. UFC 3-220-20 ch. 4 leaves to "the
judgement of the design engineer" which apparent pressure diagram to use
"when the struts are preloaded". No held page prints how a preload enters
the support loads of this analysis, so it is no longer accepted.

## What changed

**settlement** (commit 3c64d5f)
- `settlement/immediate.py`: `schmertmann_settlement` and
  `schmertmann_terms` take `below_profile` (None, `"last_layer_continued"`
  or `"incompressible"`). Layers whose deepest bottom is above z_max are
  refused unless it is given. The terms gain `profile_bottom` and
  `below_profile` (None when the layers reach z_max). New
  `schmertmann_below_choice` and `schmertmann_below_warning`.
- `settlement/analysis.py`: `SettlementAnalysis.schmertmann_below_profile`;
  the choice goes in `choices`, the note in `warnings`.
- `funhouse_agent/adapters/settlement.py`: `schmertmann_settlement` takes
  `below_profile` and returns `influence_depth_m`, `layers_bottom_m` and,
  when the layers stop short, `below_profile`, `choices`, `warnings`;
  `combined_settlement_analysis` (and so `settlement_package`) takes
  `schmertmann_below_profile`; `consolidation_settlement` refuses a
  `delta_sigmas` of another length. Descriptions and the absence
  declarations say so.
- `settlement/DESIGN.md`: a section for this.

**soe** (commit 080faf2)
- `soe/geometry.py`: `SupportLevel.spacing` is optional with no default
  (was 3.0, never read); `angle_deg` must be 0 to under 90, `spacing`
  positive; `preload_kN_per_m` removed.
- `soe/beam_analysis.py`: `support_own_loads`; each entry of
  `support_reactions` gains `spacing_m`, `angle_deg`,
  `axial_load_kN_per_m`, `horizontal_load_kN_per_support`,
  `axial_load_kN_per_support`, `vertical_load_kN_per_support` (the last
  three None without a spacing); a note and a `choices` entry say what the
  loads are and what was given.
- `soe/results.py` (summary line), `soe/calc_steps.py` (a "design load,
  one support" step in the package when a spacing is given).
- `funhouse_agent/adapters/soe.py`: descriptions; a preload key on a
  support is refused with the reason.
- `foundry/soe_agent_foundry.py` (retired route, still tested): no longer
  passes the removed field or the 3.0 default.
- `soe/DESIGN.md`: a section for this.

**Manual** (commit ade2168): the rows of `docs/user_manual/manual.html` for
`schmertmann_settlement` (layers, new `below_profile`),
`consolidation_settlement` (`delta_sigmas`), `combined_settlement_analysis`
and `settlement_package` (`schmertmann_layers`, new
`schmertmann_below_profile`), `braced_excavation` and `soe_package`
(`supports`). No rebuild.

**Fingerprints:** `webapp/tests/test_page_fingerprints.py` passes unchanged;
nothing to re-record.

## Before and after

Schmertmann, square footing B = 2 m (diagram to 4 m), q_net 120 kPa,
q_overburden 20 kPa, gamma_soil 18 kN/m3, one layer 0 to 2 m, Es 12 000 kPa:

| | Before | After |
|---|---|---|
| as given | 8.74 mm, nothing said | refused: "the 2 m from 2 to 4 m is not covered" |
| `below_profile="incompressible"` | (no such input) | 8.74 mm, stated in `choices` and `warnings` |
| `below_profile="last_layer_continued"` | (no such input) | 12.88 mm, stated |
| layers given to 4 m (second layer Es 12 000) | 12.88 mm | 12.88 mm |

Consolidation, three 2 m clay layers (e0 0.9, Cc 0.3, Cr 0.05; sigma_v0 50,
70, 90 kPa): `delta_sigmas` [40, 30, 20] gives 157.05 mm before and after;
[40, 30] gave 129.53 mm (third layer dropped, nothing said) and is now
refused.

Braced wall, H 7 m, sand (18.5 kN/m3, phi 33), surcharge default; supports
at 2 m (spacing 3 m) and 5 m (anchor, spacing 2.4 m, 20 deg):

| Support | Before | After |
|---|---|---|
| 1 | `load_kN_per_m` 115.66 | `load_kN_per_m` 115.66; per support 346.97 kN horizontal and axial |
| 2 | `load_kN_per_m` 67.03 | `load_kN_per_m` 67.03; `axial_load_kN_per_m` 71.33; per support 160.87 kN horizontal, 171.19 kN axial, 58.55 kN vertical |

Moment, shear, embedment and section: unchanged.

## Calls made (for the owner to revisit)

1. **Schmertmann layers that stop short are refused by default.** The
   options are `last_layer_continued` and `incompressible`; neither is the
   default because either would be a guess about ground nobody described.
2. **`incompressible` keeps the same diagram** and stops the sum at the
   bottom of the layers (Eq. 8-16 with no strain below). No held page
   prints a diagram redrawn for a shallow rigid layer, so none is offered.
3. **`spacing` has no default.** It was 3.0 m in the dataclass and the
   adapter. A per-support load on an unstated 3 m would be a silent basis;
   without a spacing the per-support values are null and the result says so.
4. **GEC-4's anchor equations are applied to any support type** (strut,
   anchor, raker), as statics. `support_type` stays a label.
5. **`angle_deg` is limited to 0 up to under 90.** A negative value (a
   support inclined upward) is refused rather than read by its sign.
6. **The vertical component is reported, not applied**: the wall analysis
   has no axial check, and adding one is not this item.
7. **Preload is refused, not ignored**, on the tool, and the field is gone
   from `SupportLevel`.
8. **Gaps** between Schmertmann layers, or a first layer that starts below
   the footing base, are not checked (see "Noted").

## Tests

Venv per COMMON_MODULES (Python 3.12). All by pytest's exit code.

| Run | Result |
|---|---|
| `settlement soe funhouse_agent/tests/test_gm320_321_tools.py funhouse_agent/tests/test_nested_keys.py webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py webapp/tests/test_docs_currency.py` | 589 passed, 0 failed |
| `calc_package validation_examples foundry_test_harness funhouse_agent/tests/test_calc_package_tool_parity.py test_soe_no_moment_balance.py test_adapter_ergonomics.py` | 1079 passed, 3 skipped (pdflatex), 1 xfailed, 0 failed |
| `funhouse_agent/tests calc_package validation_examples foundry_test_harness` with `-k "settlement or soe or braced or schmertmann or excavation or defaults_used or probe or method_info or fingerprint or public_text or nested or ergonom or parity"`, before the absence declarations | 1416 passed, 24 skipped, 2 failed |
| the same guard after the declarations: `funhouse_agent/tests/test_defaults_used.py -k "absence or settlement or soe or plot_data"` | 70 passed, 1 failed |
| `funhouse_agent/tests -k worked_example` | 60 passed, 1 skipped |

The one remaining failure,
`test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`,
fails the same way on master without this branch (checked with the changes
stashed); it is not from this work and was not looked into.

New tests: `settlement/tests/test_gm320_short_profile.py` (7),
`soe/tests/test_gm321_support_loads.py` (10),
`funhouse_agent/tests/test_gm320_321_tools.py` (15, with a check that every
support key the tool promises is returned); `test_nested_keys.py` now
checks the preload refusal. No probe, worked example or validation example
relied on a short Schmertmann profile, an unequal `delta_sigmas` or a
preload: none needed new inputs.

## Release note

Settlement: Schmertmann layers that stop above the bottom of the strain
influence diagram (2B below the base for a square footing, 4B for a strip)
are now refused, where before the missing depth was silently counted as
not settling; say what lies below with `below_profile`
(`schmertmann_below_profile` in the combined analysis and the calc
package). A `delta_sigmas` list that is not one value per layer is refused.
Support of excavation: each braced support now reports its own load from
its spacing and inclination (GEC-4 Eqs. 19 to 21); `spacing` no longer
defaults to 3 m, and a support's preload is refused. No number that was
returned before changes: the settlement of a full profile, and a wall's
loads per metre, moment, shear and embedment, are the same.

## Noted, not fixed

- A gap between Schmertmann layers, or layers starting below the footing
  base, still adds nothing for the gap with nothing said. Same kind of
  shortfall, not the item as written.
- `settlement_package` prints `choices` (so the below-profile statement is
  in the package) but none of the analysis's `warnings`; true of its other
  warnings before this branch.
- `docs/user_manual/worked_examples.json` holds a saved
  `soe.braced_excavation` result with struts at 4 m spacing; a fresh run
  now adds 586.66 and 408.23 kN per strut, a note and a choice. Not
  regenerated (no manual rebuild); the manual page prints that list only as
  "[2 entries]".
- `foundry/soe_agent_foundry.py` drops a preload silently where the app's
  tool refuses it; its description text was not touched (retired route).
- The Caltrans manual's own strut-spacing wording was not found in the held
  text (above).
