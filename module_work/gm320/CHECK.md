# GM-320 and GM-321: settlement and support of excavation

Branch `claude/gm-320-321-settlement-soe`, from master 2feea3d; the review's
fixes on `fix/gm320-review` (from its head 1c8befd). Audit rows:
`module_work/module_audit_2026-11-12/AUDIT.md` section 4, settlement rows 1
and 2 and soe row 1. All made-up values below, except where a printed
example is named.

## Findings taken

| Audit row | Confirmed on master? | What was done |
|---|---|---|
| settlement 1 (A): `consolidation_settlement` zips `layers` with `delta_sigmas`, no length check | Yes (`funhouse_agent/adapters/settlement.py`, the `zip(layers, delta_sigmas)` loop) | **Refused.** A list of another length, not a list, or with an entry that is not a number, is refused, naming the counts or the entry. |
| settlement 2 (A): Schmertmann adds nothing below layers that stop above z_max, and says nothing | Yes (`settlement/immediate.py`, the clip `zb = min(layer.depth_bottom, z_max)`) | **Refused by default** when the layers do not cover the diagram from the footing base down (short bottom, gap, late start), naming the depth ranges; one explicit input says what lies there, and the result states it (`choices`, `warnings`, the ranges). |
| settlement 3 (B): `warnings.warn` (Cr > Cc) never reaches the agent | Not opened | Not done: the warnings task (`claude/gm-316-317-warnings-allowed-values`) owns it. |
| settlement 4, 5, 6 (B): Hough not in the combined analysis; no package for the separate methods; time-rate functions without a tool | Not opened | Not done: other functions, out of this section's scope. |
| soe 1 (A): supports' `spacing`, `angle_deg`, `preload_kN_per_m` accepted and never read | Yes (nothing in `soe/` read them; `SupportLevel.spacing` defaulted to 3.0) | **Spacing and angle used** (GEC-4 Eqs. 19 to 21; Caltrans Example 8-1); **preload refused** with a message (no held page prints how to use it). |
| soe 2 (A): blow-out unit weight wording | n/a | Left to DOC-24, as the brief says. |
| soe 5, 6, 7 (B) | Not opened | Not done: other functions. |

## The review's findings and what each became

Review: `rev-gm320`, 2026-11-12, of head 1c8befd (A 2, B 7, C 11). Numbers
in this table are the review's (F) and the lead's (A1-B7) ids.

| # | Lead | Grade | Finding | What it became |
|---|---|---|---|---|
| 1 | A1 | A | The print does give the case: NHI-06-089 App. A.7 stops the sum at the bottom of a sand over clay; the branch refused it, and the combined result called a clay "incompressible" while consolidating it. | **Fixed.** New value `settled_by_another_method` (App. A.7 cited); `incompressible` now means rock; both give the same sum. The refusal for ground nobody described stays. In `SettlementAnalysis` (so `combined_settlement_analysis` and `settlement_package`), where a consolidation layer lies in a depth not covered, the refusal offers only `settled_by_another_method`, and `incompressible` or `last_layer_continued` there is refused. "No printed rule" taken out of the docstring, the code comment, the choice basis, both descriptions, both absence texts, DESIGN.md, the test docstring and this file. Tests: App. A.7's arrangement with made-up numbers, by hand (6.51 mm), in the module and through the tool. |
| 2 | A2 | A | The refusal's own depth was refused again (5.33333 m). | **Fixed.** `COVER_TOL_M` = 10 mm (a numerical tolerance, in DESIGN.md); the message prints the depth rounded up to the centimetre (5.34 m). Test: B 2 m, L 8 m, the message's depth fed back, 5.33 m accepted, 5.3133 m refused. |
| 3 | B2 | B | The combined analysis and the package named `below_profile`, their parameter is `schmertmann_below_profile`. | **Fixed.** `schmertmann_terms(below_param=...)`; the refusal, choice and warning name the caller's parameter. Test on both tools. |
| 4 | B1 | B | Gaps and a first layer below the base still silent (4.6 mm, 6.47 mm, 0.0 mm). | **Fixed.** `schmertmann_uncovered`: every depth from the base to z_max not covered is refused, naming the ranges; when stated, the terms (`uncovered`), choice, warning and the tool (`depths_not_covered_m`) list them. `last_layer_continued` speaks only for the depth below the deepest layer and is refused for a gap. Overlaps: not in the ruling, see Follow-ups. |
| 5 | B4 | B | "UFS 3-220-10 Sec. 5-3" is wrong. | **Fixed.** The library's PDF is UFS 3-220-10 (16 January 2026); the Schmertmann Method is Sec. 5-4.1.2 (p. 231); Sec. 5-3 is "Settlement Calculations". Corrected in all five places. |
| 6 | B7 | B | The package's sublayer table left out the continued layer its S_e includes. | **Fixed** (a few lines in `settlement/calc_steps.py`): a row marked "continued" and the below-profile note under the table. |
| 7 | B5 | B | `angle_deg: null`, `"15"`, `spacing: "3"` raised a raw TypeError. | **Fixed.** Null angle = horizontal, as before the branch; a string or bool spacing or angle is refused naming the key and the support. |
| 8 | B3 | B | The vertical component reported and not applied was said in DESIGN.md only. | **Fixed.** The description (supports and returns) and a note in the result when a support is inclined: the wall and its toe must carry it, this tool does not check that (Caltrans Sec. 11-3.04, p. 11-12, read; GEC-4 App. A p. A-13, read). |
| 9 | B6 | B | The Caltrans manual does print spacing and inclination. | **Fixed.** Example 8-1, Eq. 8-4-21 (p. 8-15) and Eq. 8-4-22 (p. 8-16) read on the PDF and cited beside GEC-4 in the basis, description and DESIGN.md; "not found" corrected here and in DESIGN.md; a made-up hand check (50 kN/m, 3 m, 15 deg: 150, 155.29, 40.19 kN) pins the code's form. |
| 10 | | C | `below_profile` listed in `defaults_used` when the layers cover the diagram. | Not done (needs a condition in the dispatch's defaults_used, more than a line). Its text now reads "nothing assumed where the layers do not reach", true either way. |
| 11 | | C | Empty layers with `last_layer_continued`: raw `max()` error. | **Fixed:** empty layers refused first, by name. |
| 12 | | C | A `delta_sigmas` dict, or a null inside the list: raw TypeError. | **Fixed:** must be a list or tuple; the entry that is not a number is named. |
| 13 | | C | `preload_kN_per_m: 0` refused. | Not done: the owner's call; no output changes. |
| 14 | | C | `spacing: true` taken as 1 m; 89.9 deg accepted. | Bool **refused** (with 7). The angle: no printed upper range, left. |
| 15 | | C | The choice's alternative repeated what was chosen. | **Fixed:** alternative "no spacing: loads per metre of wall only" when a spacing is given. |
| 16 | | C | The braced package's inputs table lists support depths only. | Follow-up (changes the package's inputs table). |
| 17 | | C | The retired route cannot pass `below_profile`. | Not done (retired route); for the lead's retired-route item. |
| 18 | | C | `SoilProfile.to_soe_input` keeps the 3.0 default and raises before it returns. | Not done; for the lead to file as a backlog line (it cannot run, so no output changes). |
| 19 | | C | Small truths in wording. | **Fixed:** the GEC-4 Sec. 7.5 words are now the print's (below); the soe test file names the test that uses GEC-4's printed numbers. The unused `schmertmann_below_profile` with the elastic method: left (no output change). |
| 20 | | C | UFC 3-220-20 p. 261 as a third cite. | Not done (optional). |

## What the print says

**Schmertmann.** FHWA NHI-06-089 Sec. 8.5.1 (pp. 8-45 to 8-51): Eq. 8-16
sums over "n = number of soil layers within the zone of strain influence
(strain influence diagram)" (p. 8-45). Example 8-2, step 2 (p. 8-49) sets
the maximum depth of influence (2Bf at Lf/Bf = 1, 4Bf above 10,
interpolated) and step 5 (p. 8-50) divides "the influence depth of 16 ft"
into layers down to its bottom. UFS 3-220-10 Sec. 5-4.1.2 (p. 231; Eq.
5-7, Fig. 5-7): "a common approach for the calculation of settlement for
coarse-grained soils"; Iz "decreases to zero at a depth of 2B for square
footings and 4B for continuous footings".

**The printed case of layers that stop above the bottom of the diagram.**
NHI-06-089 App. A.7 (the held text has no appendices; read on the PDF):
a 7 ft pier footing, L/B 14.3, on 11 ft of sand over clay. The diagram is
drawn to DI = 4Bf = 28 ft (p. A.7-10) and the print says "The immediate
settlement will occur only in the sand layer. Therefore, the portion of
the strain influence diagram in the clay layer will not be considered in
the computation of the immediate settlement" (p. A.7-10); the sum runs
over the sand's sublayers only (p. A.7-12, Si = 0.43 in), and Step 4
(p. A.7-13) settles the clay by consolidation. That is
`settled_by_another_method`. Neither section prints a diagram redrawn for
a shallow rigid layer, or a rule for ground nobody described.

**Support loads.** GEC-4 (FHWA-IF-99-015) Sec. 5.3.3, "Calculation of
Ground Anchor Loads from Apparent Earth Pressure Diagrams":
- Eq. 19 (p. 66): Th = Thi s, Thi the horizontal anchor load per unit width
  of wall (Figs. 33, 34), s the horizontal spacing between anchors;
- Eq. 20 (p. 67): T = Th / cos theta, the anchor design load, theta the
  inclination below the horizontal;
- Eq. 21 (p. 67): Tv = T sin theta, the vertical component.
GEC-4 App. A Design Example 1 uses them (TH1 168 kN/m, 2.5 m, 15 deg:
DL1 435 kN, p. A-7), the numbers `soe.fhwa_apparent_pressure` is already
pinned to; p. A-13 adds 435 kN x sin 15 = 113 kN to the soldier beam's
axial load.

**Caltrans Trenching and Shoring Manual** (July 2025; read on the PDF, the
held text is a summary). Example 8-1, a single ground anchor sheet pile
wall "with the ground anchors spaced at every 10 feet" (p. 8-11):
Eq. 8-4-21 (p. 8-15) multiplies the load per foot of wall by the 10 ft
spacing to get T_H = 143.87 kips, and Eq. 8-4-22 (p. 8-16) gives
T = 143.87 / cos 15 deg = 148.95 kips: the same two steps as GEC-4 Eqs. 19
and 20, and the code's form. Sec. 11-3.04 (p. 11-12): "Ground anchors are
generally inclined; therefore, the vertical component of the ground anchor
force must be resisted by the vertical member through skin friction on
the embedded length of the piling in contact with the soil and by end
bearing." Both manuals print these for ground anchors; an inclined strut
or raker is the same statics.

**Preload.** GEC-4 Sec. 7.5 "Anchor Lock-off Load" (p. 154): the lock-off
load "generally ranges between 75 and 100 percent of the anchor design
load, where the anchor design load is evaluated based on apparent earth
pressure envelopes". That selects a lock-off load from the design load; it
does not change the design load. UFC 3-220-20 (p. 254): "It is up to the
judgement of the design engineer ... particularly when the struts are
preloaded". No held page prints how a preload enters the support loads of
this analysis, so it is no longer accepted.

## What changed

**settlement** (3c64d5f, then the review's be76e55 and 6f6477d)
- `settlement/immediate.py`: `schmertmann_settlement` and
  `schmertmann_terms` take `below_profile`: `"settled_by_another_method"`
  (not granular soil, its settlement worked out by another method: App.
  A.7), `"incompressible"` (rock), both counting nothing in the depth not
  covered on the same diagram, or `"last_layer_continued"` (the deepest
  layer's Es to the bottom of the diagram; only for the depth below the
  deepest layer). `schmertmann_uncovered` finds every depth from the base
  to z_max no layer covers (more than `COVER_TOL_M` = 10 mm); any is
  refused unless stated. `schmertmann_terms` also takes `below_param` (the
  caller's name for the input, in every message) and `settled_elsewhere`
  (depth ranges the same call settles otherwise: only
  `settled_by_another_method` there). The terms gain `profile_bottom`,
  `uncovered` and `below_profile`. New `schmertmann_below_choice` and
  `schmertmann_below_warning` (both take `param`). Empty layers refused.
- `settlement/analysis.py`: `SettlementAnalysis.schmertmann_below_profile`;
  passes its consolidation layers' depth ranges as `settled_elsewhere`;
  the choice goes in `choices`, the note in `warnings`, both naming
  `schmertmann_below_profile`.
- `settlement/calc_steps.py`: the "Schmertmann Sublayers" table shows a
  continued layer as a row and the note under the table.
- `funhouse_agent/adapters/settlement.py`: `schmertmann_settlement` takes
  `below_profile` and returns `influence_depth_m`, `layers_bottom_m` and,
  when the layers do not cover the diagram, `below_profile`,
  `depths_not_covered_m`, `choices`, `warnings`;
  `combined_settlement_analysis` (and so `settlement_package`) takes
  `schmertmann_below_profile`; `consolidation_settlement` refuses a
  `delta_sigmas` of another length or with an entry not a number.
  Descriptions and the absence declarations say so.
- `settlement/DESIGN.md`: a section for this.

**soe** (080faf2, then the review's cd5f2b2)
- `soe/geometry.py`: `SupportLevel.spacing` is optional with no default
  (was 3.0, never read); `angle_deg` must be 0 to under 90 (None: 0),
  `spacing` positive, both numbers; `preload_kN_per_m` removed.
- `soe/beam_analysis.py`: `support_own_loads`; each entry of
  `support_reactions` gains `spacing_m`, `angle_deg`,
  `axial_load_kN_per_m`, `horizontal_load_kN_per_support`,
  `axial_load_kN_per_support`, `vertical_load_kN_per_support` (the last
  three None without a spacing); a note and a `choices` entry say what the
  loads are and what was given (GEC-4 and Caltrans Example 8-1); when a
  support is inclined the note says the vertical component is not applied
  or checked (Caltrans Sec. 11-3.04, GEC-4 p. A-13).
- `soe/results.py` (summary line), `soe/calc_steps.py` (a "design load,
  one support" step in the package when a spacing is given).
- `funhouse_agent/adapters/soe.py`: descriptions; a preload key on a
  support is refused with the reason; a null `angle_deg` is horizontal.
- `foundry/soe_agent_foundry.py` (retired route, still tested): no longer
  passes the removed field or the 3.0 default.
- `soe/DESIGN.md`: a section for this.

**Manual** (ade2168, then a719b80): the rows of
`docs/user_manual/manual.html` for `schmertmann_settlement` (layers,
`below_profile` with its three values), `consolidation_settlement`
(`delta_sigmas`), `combined_settlement_analysis` and `settlement_package`
(`schmertmann_layers`, `schmertmann_below_profile`), `braced_excavation`
and `soe_package` (`supports`), edited from the live tool text; a script
check finds every parameter description of these six tools in the page.
No rebuild.

**Fingerprints:** nothing to re-record. The page fingerprints hash the app
tools' descriptions, and the method catalog in them carries each method's
`brief` only (`funhouse_agent/dispatch.py`); no brief changed. The branch's
own run of `webapp/tests/test_page_fingerprints.py` passed after the same
kind of change. Not run in the review round (the suite runs on GitHub).

## Before and after

"Base" is master 2feea3d, "head" the cloud session's 1c8befd, "now" this
branch after the review's fixes.

Schmertmann, square footing B = 2 m (diagram to 4 m), q_net 120 kPa,
q_overburden 20 kPa, gamma_soil 18 kN/m3, Es 12 000 kPa:

| Layers | Base | Head | Now |
|---|---|---|---|
| 0 to 2 m, as given | 8.74 mm, nothing said | refused | refused, "2 to 4 m ... not covered", three values offered |
| 0 to 2 m, `settled_by_another_method` | (no input) | (no such value) | 8.74 mm, stated, App. A.7 cited |
| 0 to 2 m, `incompressible` | (no input) | 8.74 mm, "incompressible" | 8.74 mm, stated as rock |
| 0 to 2 m, `last_layer_continued` | (no input) | 12.88 mm | 12.88 mm; the package's table now shows the continued row |
| 0 to 1 and 3 to 4 m | 4.60 mm, nothing said | 4.60 mm, nothing said | refused, "1 to 3 m ... not covered"; stated: 4.60 mm, the range listed |
| 1.5 to 4 m | 6.47 mm, nothing said | 6.47 mm, nothing said | refused, "0 to 1.5 m"; stated: 6.47 mm |
| 5 to 6 m | 0.0 mm, nothing said | 0.0 mm, nothing said | refused, "0 to 4 m" |
| 0 to 4 m | 12.88 mm | 12.88 mm | 12.88 mm |

B 2 m, L 8 m (diagram to 5.3333 m), one layer from 0 m: head refused a
bottom at 5.33333 m, the depth its own message printed (3.3e-6 m "not
covered"), and at 5.33 m (3.3 mm); now its message prints 5.34 m, and
5.34 m, 5.3333 m and 5.33 m (within 10 mm) all give 17.07 mm.

Sand over consolidating clay in the combined analysis (sand 0 to 2 m as
above, clay 2 to 4 m, e0 0.9, Cc 0.3, Cr 0.05, sigma_v0 70 kPa): head with
`incompressible` returned 8.74 + 33.24 mm and said the clay was
"incompressible"; now `incompressible` is refused there and
`settled_by_another_method` returns 8.74 mm immediate, 33.24 mm
consolidation, 41.98 mm total, stated as not granular soil settled by
consolidation.

NHI-06-089 App. A.7 itself (scratch probe, not a repo test): base 10.75 mm
(0.423 in; the print 0.43 in), head refused, now 10.75 mm with
`settled_by_another_method`, the range 3.35 to 8.53 m listed.

Consolidation, three 2 m clay layers (e0 0.9, Cc 0.3, Cr 0.05; sigma_v0 50,
70, 90 kPa): `delta_sigmas` [40, 30, 20] gives 157.05 mm before and after;
[40, 30] gave 129.53 mm (third layer dropped, nothing said) and is now
refused.

Braced wall, H 7 m, sand (18.5 kN/m3, phi 33), surcharge default; supports
at 2 m (spacing 3 m) and 5 m (anchor, spacing 2.4 m, 20 deg):

| Support | Base | Now |
|---|---|---|
| 1 | `load_kN_per_m` 115.66 | `load_kN_per_m` 115.66; per support 346.97 kN horizontal and axial |
| 2 | `load_kN_per_m` 67.03 | `load_kN_per_m` 67.03; `axial_load_kN_per_m` 71.33; per support 160.87 kN horizontal, 171.19 kN axial, 58.55 kN vertical, and a note that the wall must carry the vertical and this tool does not check it |

Moment, shear and embedment: unchanged. `angle_deg: null`: base ran
(horizontal), head raised a TypeError, now runs as horizontal;
`angle_deg: "15"`: now "supports[0]: a support's angle_deg must be a
number".

## Calls made (for the owner to revisit)

1. **A depth of the strain influence diagram that no Schmertmann layer
   covers is refused by default**: below the deepest layer, a gap, or
   above the first layer. Nothing is assumed about ground nobody
   described.
2. **Three stated values.** `settled_by_another_method` is the printed
   case (NHI-06-089 App. A.7: sand over clay, the sum stopped at the
   bottom of the sand, the clay consolidated); `incompressible` is rock;
   both give the same sum, on the same diagram, and differ in what the
   result says. `last_layer_continued` speaks only for the depth below the
   deepest layer. No held page prints a diagram redrawn for a shallow
   rigid layer, so none is offered.
3. **With consolidation layers in the depth not covered** (the combined
   analysis and its package), only `settled_by_another_method` is offered
   and taken. The lead ruled out `incompressible` there; I also refuse
   `last_layer_continued` there, since it would count the clay's strain
   with the sand's Es and consolidate it too. A consolidation layer only
   below the diagram does not restrict the choice.
4. **10 mm tolerance** on the cover check, a numerical one; the refusal
   prints the depth rounded up to the centimetre.
5. **`spacing` has no default.** It was 3.0 m in the dataclass and the
   adapter. A per-support load on an unstated 3 m would be a silent basis;
   without a spacing the per-support values are null and the result says so.
6. **GEC-4's anchor equations (and Caltrans Example 8-1's same steps) are
   applied to any support type** (strut, anchor, raker), as statics.
   `support_type` stays a label.
7. **`angle_deg` is 0 up to under 90**; null is 0, a negative value (a
   support inclined upward) is refused rather than read by its sign, a
   value that is not a number is refused.
8. **The vertical component is reported, not applied**: the wall analysis
   has no axial check, and adding one is not this item. The description
   and the result now say so, with the print's words that it must be
   carried.
9. **Preload is refused, not ignored**, on the tool, and the field is gone
   from `SupportLevel`.

## Tests

Cloud session, venv per COMMON_MODULES (Python 3.12). All by pytest's exit
code.

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

Review round (2026-11-12), on this machine: the main checkout's venv
(Python 3.14), the worktree on `PYTHONPATH` (import checked:
`settlement/immediate.py` from the worktree), one file at a time, by exit
code:

| File | Result |
|---|---|
| `settlement/tests/test_gm320_short_profile.py` | 19 passed, exit 0 |
| `soe/tests/test_gm321_support_loads.py` | 15 passed, exit 0 |
| `funhouse_agent/tests/test_gm320_321_tools.py` | 20 passed, exit 0 |
| `funhouse_agent/tests/test_nested_keys.py` | 67 passed, exit 0 |
| `funhouse_agent/tests/test_adapter_ergonomics.py` | 170 passed, exit 0 |
| `webapp/tests/test_public_text.py` | 5 passed, exit 0 |

Not run here (they run on GitHub): the settlement and soe suites as
folders, the calc package tests (the settlement package's sublayer table
changed: checked by a probe of `get_calc_steps`), `test_defaults_used.py`
(the absence texts changed wording only), `test_page_fingerprints.py`
(see Fingerprints above).

New tests: `settlement/tests/test_gm320_short_profile.py` (19),
`soe/tests/test_gm321_support_loads.py` (15),
`funhouse_agent/tests/test_gm320_321_tools.py` (20, with a check that
every support key the tool promises is returned); `test_nested_keys.py`
checks the preload refusal. No probe, worked example or validation example
relied on a Schmertmann profile that does not cover the diagram, an
unequal `delta_sigmas` or a preload (the one Schmertmann probe's layers run
0 to 9.76 m against a 4.88 m diagram, with no gap): none needed new inputs.

## Release note

Settlement: Schmertmann layers must now cover the strain influence diagram
from the footing base down to its bottom (2B for a square footing, 4B for
a strip). A depth they leave out (below the deepest layer, a gap, or above
the first) is refused with its ranges, where before it was silently
counted as not settling. Say what lies there with `below_profile`
(`schmertmann_below_profile` in the combined analysis and the calc
package): `settled_by_another_method` for non-granular soil settled by
another method, as NHI-06-089 App. A.7 does for sand over clay;
`incompressible` for rock; or `last_layer_continued`. When consolidation
layers lie there, only `settled_by_another_method` is accepted. Depths
within 10 mm count as covered. A `delta_sigmas` list that is not one
number per layer is refused. Support of excavation: each braced support
now reports its own load from its spacing and inclination (GEC-4 Eqs. 19
to 21; Caltrans Example 8-1); `spacing` no longer defaults to 3 m; an
inclined support's vertical component is reported with a note that the
wall must carry it and this tool does not check it; a support's preload is
refused. No number that was returned before changes: the settlement of a
profile that covers the diagram, and a wall's loads per metre, moment,
shear and embedment, are the same.

## Follow-ups (would change a saved report's output)

- **Overlapping Schmertmann layers still count twice** (review finding 4,
  last sentence: layers 0 to 2 m and 1 to 4 m, Es 12 000 kPa, give
  18.06 mm where one layer 0 to 4 m gives 12.88 mm). Refusing an overlap,
  naming the depths, would change any saved run that had one. Not in the
  lead's ruling for this round.
- **The braced package's inputs table** lists each support's depth only;
  spacing and inclination are in the echo, the choices row and the step.
  Adding them changes the package's inputs table (review C16).
- `docs/user_manual/worked_examples.json` holds a saved
  `soe.braced_excavation` result with struts at 4 m spacing; a fresh run
  now adds 586.66 and 408.23 kN per strut, a note and a choice. Not
  regenerated (no manual rebuild); the manual page prints that list only as
  "[2 entries]".

## Noted, not fixed

- `foundry/soe_agent_foundry.py` drops a preload silently where the app's
  tool refuses it, and its description text still says "spacing (m,
  default 3.0)"; `foundry/settlement_agent_foundry.py` cannot pass
  `below_profile`, so a profile that does not cover the diagram is always
  an error there (retired route; review C17).
- `SoilProfile.to_soe_input` (`geotech_common/soil_profile.py`) still
  defaults `support_spacing` to 3.0, but raises AttributeError before it
  returns and nothing calls it (review C18): a backlog line for the lead.
- `defaults_used` lists `below_profile` when the layers cover the diagram
  (review C10); its text is true either way.
